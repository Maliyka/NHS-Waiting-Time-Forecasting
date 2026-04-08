"""
08_model_prophet.py — Facebook Prophet model training with rolling-origin CV.

For each region × treatment:
  1. Load series + new_periods_count regressor
  2. Build Prophet with UK holidays + fixed COVID changepoints
  3. Rolling-origin CV (manual, monthly)
  4. Save results to fact_model_cv_results

Usage:
    python scripts/08_model_prophet.py [--config ...] [--region Q71]

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Suppress Prophet / Stan verbose logging before import
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.data_helpers import get_covid_era
from scripts.utils.db_connect import get_engine, load_config, upsert_dataframe
from scripts.utils.metrics import compute_all_metrics

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    logger.error("Prophet is not installed. Run: pip install prophet")
    PROPHET_AVAILABLE = False

FORECAST_HORIZON = 12
N_FOLDS          = 6
CV_INITIAL       = 36

# Fixed changepoints reflecting known NHS structural shifts
CHANGEPOINTS = ["2020-03-01", "2021-07-01", "2022-04-01", "2022-10-01"]


def load_series(engine, region_code: str, treatment_code: str = "C_999", part_type: str = "Part_2") -> pd.DataFrame:
    from sqlalchemy import text
    sql = text("""
        SELECT r.period_date, r.waiting_list_size, r.covid_dummy,
               COALESCE(np.new_periods_count, 0) AS new_periods_count
        FROM nhs.v_rtt_regional_monthly r
        LEFT JOIN nhs.v_new_periods_regional np
            ON np.period_date  = r.period_date
           AND np.region_code  = r.region_code
           AND np.treatment_code = r.treatment_code
        WHERE r.region_code    = :rc
          AND r.treatment_code = :tc
          AND r.part_type      = :pt
        ORDER BY r.period_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"rc": region_code, "tc": treatment_code, "pt": part_type},
                         parse_dates=["period_date"])
    df = df.dropna(subset=["waiting_list_size"])
    return df


def build_gb_holidays() -> pd.DataFrame:
    """Build UK bank holiday DataFrame in Prophet format (2019–2026)."""
    import holidays as hols
    rows = []
    for year in range(2019, 2027):
        uk_hols = hols.country_holidays("GB", years=year)
        for h_date, h_name in uk_hols.items():
            rows.append({"ds": pd.Timestamp(h_date), "holiday": h_name[:50]})
    # COVID lockdown as a special event
    rows.append({"ds": pd.Timestamp("2020-03-23"), "holiday": "COVID Lockdown Start"})
    rows.append({"ds": pd.Timestamp("2021-01-05"), "holiday": "COVID Lockdown 3"})
    df = pd.DataFrame(rows).drop_duplicates(subset=["ds", "holiday"])
    return df


def to_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert project DataFrame to Prophet format (ds, y columns)."""
    return pd.DataFrame({
        "ds": pd.to_datetime(df["period_date"]),
        "y":  df["waiting_list_size"].astype(float),
        "new_periods_count": df["new_periods_count"].astype(float),
    })


def build_model(holidays_df: pd.DataFrame) -> "Prophet":
    """Build a configured Prophet model instance."""
    m = Prophet(
        changepoints=CHANGEPOINTS,
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=10.0,
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=500,
    )
    m.add_regressor("new_periods_count", standardize=True)
    return m


def rolling_cv_prophet(df_prophet: pd.DataFrame, holidays_df: pd.DataFrame,
                        n_folds: int = N_FOLDS, horizon: int = FORECAST_HORIZON,
                        initial: int = CV_INITIAL) -> List[Dict]:
    results   = []
    n         = len(df_prophet)
    fold_size = max(1, (n - initial) // n_folds)

    for fold in range(n_folds):
        train_end = initial + fold * fold_size
        test_end  = min(train_end + horizon, n)
        if test_end <= train_end or train_end >= n:
            break

        train = df_prophet.iloc[:train_end].copy()
        test  = df_prophet.iloc[train_end:test_end].copy()
        actual_h = len(test)
        if actual_h == 0:
            break

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logging.getLogger("prophet").setLevel(logging.ERROR)
                logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

                m = build_model(holidays_df)
                m.fit(train[["ds", "y", "new_periods_count"]])

                # Future regressors: use last 3-month mean for new_periods_count
                future_regressor = float(train["new_periods_count"].tail(3).mean())
                future_df = test[["ds"]].copy()
                future_df["new_periods_count"] = future_regressor

                forecast = m.predict(future_df)

            pred    = forecast["yhat"].values
            lower_80 = forecast["yhat_lower"].values   # Prophet 80% by default
            upper_80 = forecast["yhat_upper"].values
            # 95% CI: scale out from point forecast
            half_80  = (upper_80 - lower_80) / 2
            lower_95 = pred - half_80 * 1.65
            upper_95 = pred + half_80 * 1.65
            actual   = test["y"].values

            metrics = compute_all_metrics(actual, pred, lower_80, upper_80, lower_95, upper_95)
            test_era = get_covid_era(pd.Timestamp(test["ds"].iloc[0]).date())

            results.append({
                "fold_number":      fold + 1,
                "train_start_date": str(pd.Timestamp(train["ds"].iloc[0]).date()),
                "train_end_date":   str(pd.Timestamp(train["ds"].iloc[-1]).date()),
                "test_start_date":  str(pd.Timestamp(test["ds"].iloc[0]).date()),
                "test_end_date":    str(pd.Timestamp(test["ds"].iloc[-1]).date()),
                "horizon_months":   actual_h,
                "covid_era":        test_era,
                **{k: (round(v, 6) if v is not None else None) for k, v in metrics.items()},
            })
        except Exception as exc:
            logger.debug("Prophet CV fold %d failed: %s", fold + 1, exc)

    return results


def get_dim_ids(engine, region_code, treatment_code, part_type, model_name):
    from sqlalchemy import text
    with engine.connect() as conn:
        rid = conn.execute(text("SELECT region_id FROM nhs.dim_region WHERE region_code=:c"), {"c": region_code}).scalar()
        tid = conn.execute(text("SELECT treatment_id FROM nhs.dim_treatment WHERE function_code=:c"), {"c": treatment_code}).scalar()
        pid = conn.execute(text("SELECT part_id FROM nhs.dim_rtt_part WHERE part_type=:c"), {"c": part_type}).scalar()
        mid = conn.execute(text("SELECT model_id FROM nhs.dim_model WHERE model_name=:c"), {"c": model_name}).scalar()
    return rid, tid, pid, mid


def save_cv(cv_results, engine, rid, tid, pid, mid, params):
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
            "model_params": json.dumps(params),
        }
        for r in cv_results
    ]
    if rows:
        upsert_dataframe(rows, "fact_model_cv_results", engine)


def main(config_path: str = "config/config.yaml", region_filter: Optional[str] = None) -> None:
    if not PROPHET_AVAILABLE:
        logger.error("Prophet not installed — aborting")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Step 8 — Prophet Modelling")
    logger.info("=" * 60)

    cfg    = load_config(config_path)
    engine = get_engine(config_path)
    regions    = list(cfg["regions"]["modern_codes"].keys())
    treatments = [cfg["treatments"]["primary_code"]]
    part_type  = cfg["treatments"]["primary_part"]

    if region_filter:
        regions = [region_filter]

    holidays_df = build_gb_holidays()
    logger.info("UK holidays loaded: %d entries", len(holidays_df))

    for region_code in regions:
        for treatment_code in treatments:
            logger.info("Processing Prophet: region=%s treatment=%s", region_code, treatment_code)
            try:
                df = load_series(engine, region_code, treatment_code, part_type)
                if len(df) < CV_INITIAL + FORECAST_HORIZON:
                    logger.warning("  Insufficient data (%d rows) — skipping", len(df))
                    continue

                df_prophet = to_prophet_df(df)
                rid, tid, pid, mid = get_dim_ids(engine, region_code, treatment_code, part_type, "Prophet")
                if not all(x is not None for x in [rid, tid, pid, mid]):
                    logger.warning("  Dim IDs not found — skipping")
                    continue

                cv_results = rolling_cv_prophet(df_prophet, holidays_df,
                                                n_folds=cfg["modelling"]["cv_folds"],
                                                horizon=cfg["modelling"]["forecast_horizon"],
                                                initial=cfg["modelling"]["cv_initial_months"])

                params = {
                    "changepoints": CHANGEPOINTS,
                    "changepoint_prior_scale": 0.3,
                    "seasonality_prior_scale": 10.0,
                    "yearly_seasonality": True,
                    "regressor": "new_periods_count",
                }
                save_cv(cv_results, engine, rid, tid, pid, mid, params)

                if cv_results:
                    mean_mae = np.mean([r["mae"] for r in cv_results if r.get("mae")])
                    logger.info("  Prophet CV complete: %d folds, mean MAE=%.0f", len(cv_results), mean_mae)

                # Save params
                params_dir = Path(cfg["paths"]["model_params"])
                params_dir.mkdir(parents=True, exist_ok=True)
                with open(params_dir / f"prophet_{region_code}_{treatment_code}.json", "w") as f:
                    json.dump(params, f)

            except Exception as exc:
                logger.error("Failed region=%s: %s", region_code, exc, exc_info=True)

    logger.info("=" * 60)
    logger.info("Prophet modelling complete.")
    logger.info("Next step: python scripts/09_model_evaluation.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()
    main(config_path=args.config, region_filter=args.region)
