"""
07_model_holt_winters.py — Holt-Winters / ETS model training with rolling-origin CV.

For each region × treatment:
  1. Smooth COVID Apr-Jun 2020 outliers (linear interpolation)
  2. Test all 4 ETS configurations by AICc
  3. Rolling-origin CV (6 folds, 12-month horizon)
  4. Prediction intervals via bootstrap residuals (500 samples)
  5. Save results to fact_model_cv_results

Usage:
    python scripts/07_model_holt_winters.py [--config ...] [--region Q71]

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.data_helpers import get_covid_era
from scripts.utils.db_connect import get_engine, load_config, upsert_dataframe
from scripts.utils.metrics import compute_all_metrics

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FORECAST_HORIZON = 12
N_FOLDS          = 6
CV_INITIAL       = 36
N_BOOTSTRAP      = 500
COVID_SMOOTH     = ["2020-04-01", "2020-05-01", "2020-06-01"]


def load_series(engine, region_code: str, treatment_code: str = "C_999", part_type: str = "Part_2") -> pd.DataFrame:
    from sqlalchemy import text
    sql = text("""
        SELECT period_date, waiting_list_size
        FROM nhs.v_rtt_regional_monthly
        WHERE region_code = :rc AND treatment_code = :tc AND part_type = :pt
        ORDER BY period_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"rc": region_code, "tc": treatment_code, "pt": part_type},
                         parse_dates=["period_date"])
    df = df.dropna(subset=["waiting_list_size"]).set_index("period_date")
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp("M")
    return df


def smooth_covid_outliers(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Linearly interpolate the 3 months of maximum COVID disruption
    (April, May, June 2020) where NHS suspended elective care.

    Returns (smoothed_series, original_series).
    Only modifies those 3 months — all other values are unchanged.
    """
    smoothed = series.copy()
    for m in COVID_SMOOTH:
        ts = pd.Timestamp(m)
        if ts in smoothed.index:
            smoothed.loc[ts] = np.nan
    smoothed = smoothed.interpolate(method="linear")
    return smoothed, series.copy()


def fit_ets(series: pd.Series, trend: str = "add", seasonal: str = "add") -> Optional[object]:
    """Fit one ETS model configuration. Returns fitted model or None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=12,
                damped_trend=True,
            )
            return m.fit(optimized=True, use_brute=False)
    except Exception:
        return None


def select_best_ets(series: pd.Series) -> Dict:
    """
    Test 4 ETS configurations, select best by AICc.
    Returns dict with keys: trend, seasonal, fitted, aicc.
    """
    configs = [
        ("add", "add"),
        ("add", "mul"),
        ("mul", "add"),
        ("mul", "mul"),
    ]
    best = None
    best_aicc = float("inf")

    for trend, seasonal in configs:
        fitted = fit_ets(series, trend, seasonal)
        if fitted is None:
            continue
        try:
            aicc = fitted.aicc if hasattr(fitted, "aicc") else fitted.aic
            if aicc < best_aicc:
                best_aicc = aicc
                best = {"trend": trend, "seasonal": seasonal, "fitted": fitted, "aicc": aicc}
        except Exception:
            continue

    return best or {"trend": "add", "seasonal": "add", "fitted": fit_ets(series, "add", "add"), "aicc": np.nan}


def bootstrap_intervals(fitted_model, residuals: np.ndarray, horizon: int,
                        n_samples: int = N_BOOTSTRAP) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate prediction intervals by bootstrapping residuals.
    Returns (lower_80, upper_80, lower_95, upper_95) each of length horizon.
    """
    point_forecast = fitted_model.forecast(horizon)
    bootstrap_forecasts = np.zeros((n_samples, horizon))

    for i in range(n_samples):
        noise = np.random.choice(residuals, size=horizon, replace=True)
        bootstrap_forecasts[i] = point_forecast.values + noise

    lower_80 = np.percentile(bootstrap_forecasts, 10, axis=0)
    upper_80 = np.percentile(bootstrap_forecasts, 90, axis=0)
    lower_95 = np.percentile(bootstrap_forecasts,  2.5, axis=0)
    upper_95 = np.percentile(bootstrap_forecasts, 97.5, axis=0)
    return lower_80, upper_80, lower_95, upper_95


def rolling_cv(series: pd.Series, best_config: Dict,
               n_folds: int = N_FOLDS, horizon: int = FORECAST_HORIZON,
               initial: int = CV_INITIAL) -> List[Dict]:
    results   = []
    n         = len(series)
    fold_size = max(1, (n - initial) // n_folds)
    trend     = best_config["trend"]
    seasonal  = best_config["seasonal"]

    for fold in range(n_folds):
        train_end = initial + fold * fold_size
        test_end  = min(train_end + horizon, n)
        if test_end <= train_end or train_end >= n:
            break

        train = series.iloc[:train_end]
        test  = series.iloc[train_end:test_end]
        actual_h = len(test)
        if actual_h == 0:
            break

        try:
            fitted = fit_ets(train, trend, seasonal)
            if fitted is None:
                continue

            fc    = fitted.forecast(actual_h)
            resid = fitted.resid.values
            l80, u80, l95, u95 = bootstrap_intervals(fitted, resid, actual_h)

            metrics = compute_all_metrics(test.values, fc.values, l80, u80, l95, u95)
            test_era = get_covid_era(test.index[0].date())

            results.append({
                "fold_number":      fold + 1,
                "train_start_date": str(train.index[0].date()),
                "train_end_date":   str(train.index[-1].date()),
                "test_start_date":  str(test.index[0].date()),
                "test_end_date":    str(test.index[-1].date()),
                "horizon_months":   actual_h,
                "covid_era":        test_era,
                **{k: (round(v, 6) if v is not None else None) for k, v in metrics.items()},
            })
        except Exception as exc:
            logger.debug("HW CV fold %d failed: %s", fold + 1, exc)

    return results


def get_dim_ids(engine, region_code, treatment_code, part_type, model_name):
    from sqlalchemy import text
    with engine.connect() as conn:
        rid = conn.execute(text("SELECT region_id FROM nhs.dim_region WHERE region_code=:c"), {"c": region_code}).scalar()
        tid = conn.execute(text("SELECT treatment_id FROM nhs.dim_treatment WHERE function_code=:c"), {"c": treatment_code}).scalar()
        pid = conn.execute(text("SELECT part_id FROM nhs.dim_rtt_part WHERE part_type=:c"), {"c": part_type}).scalar()
        mid = conn.execute(text("SELECT model_id FROM nhs.dim_model WHERE model_name=:c"), {"c": model_name}).scalar()
    return rid, tid, pid, mid


def save_cv(cv_results, engine, rid, tid, pid, mid, model_params):
    rows = [
        {
            "region_id": rid, "treatment_id": tid, "part_id": pid, "model_id": mid,
            "fold_number": r["fold_number"],
            "train_start_date": r["train_start_date"], "train_end_date": r["train_end_date"],
            "test_start_date":  r["test_start_date"],  "test_end_date":  r["test_end_date"],
            "horizon_months": r["horizon_months"], "covid_era": r.get("covid_era"),
            "mae": r.get("mae"), "rmse": r.get("rmse"),
            "mape": r.get("mape"), "mase": r.get("mase"),
            "coverage_80": r.get("coverage_80"), "coverage_95": r.get("coverage_95"),
            "model_params": json.dumps(model_params),
        }
        for r in cv_results
    ]
    if rows:
        upsert_dataframe(rows, "fact_model_cv_results", engine)


def main(config_path: str = "config/config.yaml", region_filter: Optional[str] = None) -> None:
    logger.info("=" * 60)
    logger.info("Step 7 — Holt-Winters / ETS Modelling")
    logger.info("=" * 60)

    cfg    = load_config(config_path)
    engine = get_engine(config_path)
    regions    = list(cfg["regions"]["modern_codes"].keys())
    treatments = [cfg["treatments"]["primary_code"]]
    part_type  = cfg["treatments"]["primary_part"]

    if region_filter:
        regions = [region_filter]

    for region_code in regions:
        for treatment_code in treatments:
            logger.info("Processing HW: region=%s treatment=%s", region_code, treatment_code)
            try:
                df = load_series(engine, region_code, treatment_code, part_type)
                if len(df) < CV_INITIAL + FORECAST_HORIZON:
                    logger.warning("  Insufficient data — skipping")
                    continue

                raw_series = df["waiting_list_size"].astype(float)
                smoothed, original = smooth_covid_outliers(raw_series)
                logger.info("  COVID smoothing applied to %d months", len(COVID_SMOOTH))

                # Select best ETS on smoothed series
                best = select_best_ets(smoothed)
                logger.info("  Best ETS: trend=%s seasonal=%s AICc=%.2f",
                            best["trend"], best["seasonal"], best.get("aicc", float("nan")))

                model_name = f"HoltWinters_{best['seasonal'].capitalize()}itive" if best["seasonal"] == "add" \
                    else "HoltWinters_Multiplicative"

                rid, tid, pid, mid = get_dim_ids(engine, region_code, treatment_code, part_type, model_name)
                if not all(x is not None for x in [rid, tid, pid, mid]):
                    # Try additive as fallback
                    model_name = "HoltWinters_Additive"
                    rid, tid, pid, mid = get_dim_ids(engine, region_code, treatment_code, part_type, model_name)

                if not all(x is not None for x in [rid, tid, pid, mid]):
                    logger.warning("  Cannot find dim IDs — skipping")
                    continue

                cv_results = rolling_cv(smoothed, best,
                                        n_folds=cfg["modelling"]["cv_folds"],
                                        horizon=cfg["modelling"]["forecast_horizon"],
                                        initial=cfg["modelling"]["cv_initial_months"])

                params = {"trend": best["trend"], "seasonal": best["seasonal"],
                          "covid_smoothing": COVID_SMOOTH, "n_bootstrap": N_BOOTSTRAP}
                save_cv(cv_results, engine, rid, tid, pid, mid, params)

                if cv_results:
                    mean_mae = np.mean([r["mae"] for r in cv_results if r.get("mae")])
                    logger.info("  HW CV complete: %d folds, mean MAE=%.0f", len(cv_results), mean_mae)

                # Save params
                params_dir = Path(cfg["paths"]["model_params"])
                params_dir.mkdir(parents=True, exist_ok=True)
                with open(params_dir / f"hw_{region_code}_{treatment_code}.json", "w") as f:
                    json.dump(params, f)

            except Exception as exc:
                logger.error("Failed region=%s: %s", region_code, exc, exc_info=True)

    logger.info("=" * 60)
    logger.info("Holt-Winters modelling complete.")
    logger.info("Next step: python scripts/08_model_prophet.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()
    main(config_path=args.config, region_filter=args.region)
