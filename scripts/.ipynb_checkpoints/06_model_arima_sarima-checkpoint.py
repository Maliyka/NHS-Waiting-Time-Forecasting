"""
06_model_arima_sarima.py — ARIMA / SARIMA model training with rolling-origin CV.

For each region × treatment combination:
  1. Load time series from v_rtt_regional_monthly
  2. Check stationarity (ADF + KPSS)
  3. Grid search for best ARIMA(p,d,q) by AICc
  4. Test seasonal orders SARIMA(p,d,q)(P,D,Q,12)
  5. Rolling-origin cross-validation (6 folds, 12-month horizon)
  6. Save CV results to fact_model_cv_results

Usage:
    python scripts/06_model_arima_sarima.py [--config ...] [--region Q71]

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import itertools
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.data_helpers import get_covid_era
from scripts.utils.db_connect import get_engine, load_config, upsert_dataframe
from scripts.utils.metrics import compute_all_metrics

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FORECAST_HORIZON = 12
N_FOLDS          = 6
CV_INITIAL       = 36   # minimum training months


def load_series(engine, region_code: str, treatment_code: str = "C_999", part_type: str = "Part_2") -> pd.DataFrame:
    from sqlalchemy import text
    sql = text("""
        SELECT period_date, waiting_list_size, covid_dummy, is_covid_period
        FROM nhs.v_rtt_regional_monthly
        WHERE region_code = :rc AND treatment_code = :tc AND part_type = :pt
        ORDER BY period_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"rc": region_code, "tc": treatment_code, "pt": part_type},
                         parse_dates=["period_date"])
    df = df.dropna(subset=["waiting_list_size"])
    df = df.set_index("period_date")
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp("M")
    return df


def check_stationarity(series: pd.Series) -> Dict:
    result = {}
    try:
        adf = adfuller(series.dropna(), autolag="AIC")
        result["adf_stat"]    = float(adf[0])
        result["adf_pvalue"]  = float(adf[1])
        result["adf_stationary"] = adf[1] < 0.05
    except Exception:
        result["adf_stationary"] = False
    try:
        k = kpss(series.dropna(), regression="c", nlags="auto")
        result["kpss_pvalue"] = float(k[1])
        result["kpss_stationary"] = k[1] > 0.05
    except Exception:
        result["kpss_stationary"] = True
    result["is_stationary"] = result.get("adf_stationary", False) and result.get("kpss_stationary", True)
    return result


def compute_aicc(model_result, n: int, k: int) -> float:
    try:
        aic = model_result.aic
        return float(aic + (2 * k * (k + 1)) / max(n - k - 1, 1))
    except Exception:
        return float("inf")


def find_best_arima(series: pd.Series, exog: Optional[pd.Series],
                    max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[tuple, float]:
    best_order = (1, 1, 1)
    best_aicc  = float("inf")
    n = len(series)
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p + d + q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ex = exog.values.reshape(-1, 1) if exog is not None else None
                m  = SARIMAX(series, exog=ex, order=(p, d, q),
                             trend="n", enforce_stationarity=False, enforce_invertibility=False)
                r  = m.fit(disp=False, maxiter=100)
            aicc = compute_aicc(r, n, p + d + q + (1 if exog is not None else 0))
            if aicc < best_aicc:
                best_aicc  = aicc
                best_order = (p, d, q)
        except Exception:
            continue
    return best_order, best_aicc


def find_best_sarima(series: pd.Series, exog: Optional[pd.Series],
                     arima_order: tuple) -> Tuple[tuple, float]:
    best_s_order = (1, 1, 1, 12)
    best_aicc    = float("inf")
    n            = len(series)
    for P, D, Q in itertools.product([0, 1], [0, 1], [0, 1]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ex = exog.values.reshape(-1, 1) if exog is not None else None
                m  = SARIMAX(series, exog=ex, order=arima_order,
                             seasonal_order=(P, D, Q, 12),
                             enforce_stationarity=False, enforce_invertibility=False)
                r  = m.fit(disp=False, maxiter=100)
            k    = sum(arima_order) + P + D + Q + (1 if exog is not None else 0)
            aicc = compute_aicc(r, n, k)
            if aicc < best_aicc:
                best_aicc    = aicc
                best_s_order = (P, D, Q, 12)
        except Exception:
            continue
    return best_s_order, best_aicc


def rolling_cv(series: pd.Series, exog: Optional[pd.Series],
               order: tuple, seasonal_order: tuple,
               n_folds: int = N_FOLDS, horizon: int = FORECAST_HORIZON,
               initial: int = CV_INITIAL) -> List[Dict]:
    results = []
    n        = len(series)
    fold_size = max(1, (n - initial) // n_folds)

    for fold in range(n_folds):
        train_end = initial + fold * fold_size
        test_end  = min(train_end + horizon, n)
        if test_end <= train_end or train_end >= n:
            break

        train_series = series.iloc[:train_end]
        test_series  = series.iloc[train_end:test_end]
        train_exog   = exog.iloc[:train_end]   if exog is not None else None
        test_exog    = exog.iloc[train_end:test_end] if exog is not None else None

        if len(test_series) == 0:
            break

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ex = train_exog.values.reshape(-1, 1) if train_exog is not None else None
                m  = SARIMAX(train_series, exog=ex,
                             order=order, seasonal_order=seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False)
                fitted = m.fit(disp=False, maxiter=150)

            fex     = test_exog.values.reshape(-1, 1) if test_exog is not None else None
            fc      = fitted.get_forecast(steps=len(test_series), exog=fex)
            pred    = fc.predicted_mean
            ci      = fc.conf_int(alpha=0.20)  # 80% interval
            ci95    = fc.conf_int(alpha=0.05)  # 95% interval
            actual  = test_series.values

            metrics = compute_all_metrics(
                actual, pred.values,
                ci.iloc[:, 0].values, ci.iloc[:, 1].values,
                ci95.iloc[:, 0].values, ci95.iloc[:, 1].values,
            )

            train_start_date = train_series.index[0].date()
            test_era         = get_covid_era(test_series.index[0].date())

            results.append({
                "fold_number":      fold + 1,
                "train_start_date": str(train_start_date),
                "train_end_date":   str(train_series.index[-1].date()),
                "test_start_date":  str(test_series.index[0].date()),
                "test_end_date":    str(test_series.index[-1].date()),
                "horizon_months":   len(test_series),
                "covid_era":        test_era,
                **{k: (round(v, 6) if v is not None else None) for k, v in metrics.items()},
            })
        except Exception as exc:
            logger.debug("ARIMA CV fold %d failed: %s", fold + 1, exc)

    return results


def save_cv_to_db(cv_results: List[Dict], engine, region_id: int, treatment_id: int,
                  part_id: int, model_id: int, model_params: dict) -> None:
    rows = []
    for r in cv_results:
        rows.append({
            "region_id":       region_id,
            "treatment_id":    treatment_id,
            "part_id":         part_id,
            "model_id":        model_id,
            "fold_number":     r["fold_number"],
            "train_start_date":r["train_start_date"],
            "train_end_date":  r["train_end_date"],
            "test_start_date": r["test_start_date"],
            "test_end_date":   r["test_end_date"],
            "horizon_months":  r["horizon_months"],
            "covid_era":       r.get("covid_era"),
            "mae":             r.get("mae"),
            "rmse":            r.get("rmse"),
            "mape":            r.get("mape"),
            "mase":            r.get("mase"),
            "coverage_80":     r.get("coverage_80"),
            "coverage_95":     r.get("coverage_95"),
            "model_params":    json.dumps(model_params),
        })
    if rows:
        upsert_dataframe(rows, "fact_model_cv_results", engine)


def get_dim_ids(engine, region_code: str, treatment_code: str, part_type: str, model_name: str) -> Tuple:
    from sqlalchemy import text
    with engine.connect() as conn:
        rid = conn.execute(text("SELECT region_id FROM nhs.dim_region WHERE region_code=:c"), {"c": region_code}).scalar()
        tid = conn.execute(text("SELECT treatment_id FROM nhs.dim_treatment WHERE function_code=:c"), {"c": treatment_code}).scalar()
        pid = conn.execute(text("SELECT part_id FROM nhs.dim_rtt_part WHERE part_type=:c"), {"c": part_type}).scalar()
        mid = conn.execute(text("SELECT model_id FROM nhs.dim_model WHERE model_name=:c"), {"c": model_name}).scalar()
    return rid, tid, pid, mid


def main(config_path: str = "config/config.yaml", region_filter: Optional[str] = None) -> None:
    logger.info("=" * 60)
    logger.info("Step 6 — ARIMA / SARIMA Modelling")
    logger.info("=" * 60)

    cfg    = load_config(config_path)
    engine = get_engine(config_path)

    regions    = list(cfg["regions"]["modern_codes"].keys())
    treatments = [cfg["treatments"]["primary_code"]]
    part_type  = cfg["treatments"]["primary_part"]

    if region_filter:
        regions = [r for r in regions if r == region_filter]

    for region_code in regions:
        for treatment_code in treatments:
            logger.info("Processing ARIMA: region=%s treatment=%s", region_code, treatment_code)
            try:
                df = load_series(engine, region_code, treatment_code, part_type)
                if len(df) < CV_INITIAL + FORECAST_HORIZON:
                    logger.warning("  Insufficient data (%d rows) — skipping", len(df))
                    continue

                series = df["waiting_list_size"].astype(float)
                exog   = df["covid_dummy"].astype(float)

                stat = check_stationarity(series)
                logger.info("  Stationarity: ADF p=%.4f, is_stationary=%s",
                            stat.get("adf_pvalue", 1), stat.get("is_stationary"))

                # Grid search ARIMA
                logger.info("  Grid search ARIMA orders...")
                arima_order, arima_aicc = find_best_arima(series, exog,
                    max_p=cfg["modelling"]["arima_max_p"],
                    max_d=cfg["modelling"]["arima_max_d"],
                    max_q=cfg["modelling"]["arima_max_q"])
                logger.info("  Best ARIMA order: %s  AICc=%.2f", arima_order, arima_aicc)

                # ARIMA CV
                arima_model_name = "ARIMA"
                rid, tid, pid, arima_mid = get_dim_ids(engine, region_code, treatment_code, part_type, arima_model_name)
                if all(x is not None for x in [rid, tid, pid, arima_mid]):
                    cv_r = rolling_cv(series, exog, arima_order, (0, 0, 0, 0),
                                      n_folds=cfg["modelling"]["cv_folds"],
                                      horizon=cfg["modelling"]["forecast_horizon"],
                                      initial=cfg["modelling"]["cv_initial_months"])
                    params = {"order": arima_order, "seasonal_order": (0,0,0,0), "exog": "covid_dummy"}
                    save_cv_to_db(cv_r, engine, rid, tid, pid, arima_mid, params)
                    if cv_r:
                        mean_mae = np.mean([r["mae"] for r in cv_r if r.get("mae")])
                        logger.info("  ARIMA CV complete: %d folds, mean MAE=%.0f", len(cv_r), mean_mae)

                # SARIMA order search + CV
                logger.info("  Grid search SARIMA seasonal orders...")
                s_order, sarima_aicc = find_best_sarima(series, exog, arima_order)
                logger.info("  Best SARIMA seasonal: %s  AICc=%.2f", s_order, sarima_aicc)

                sarima_model_name = "SARIMA"
                rid, tid, pid, sarima_mid = get_dim_ids(engine, region_code, treatment_code, part_type, sarima_model_name)
                if all(x is not None for x in [rid, tid, pid, sarima_mid]):
                    cv_s = rolling_cv(series, exog, arima_order, s_order,
                                      n_folds=cfg["modelling"]["cv_folds"],
                                      horizon=cfg["modelling"]["forecast_horizon"],
                                      initial=cfg["modelling"]["cv_initial_months"])
                    params_s = {"order": arima_order, "seasonal_order": s_order, "exog": "covid_dummy"}
                    save_cv_to_db(cv_s, engine, rid, tid, pid, sarima_mid, params_s)
                    if cv_s:
                        mean_mae_s = np.mean([r["mae"] for r in cv_s if r.get("mae")])
                        logger.info("  SARIMA CV complete: %d folds, mean MAE=%.0f", len(cv_s), mean_mae_s)

                # Save model params JSON
                params_dir = Path(cfg["paths"]["model_params"])
                params_dir.mkdir(parents=True, exist_ok=True)
                with open(params_dir / f"arima_{region_code}_{treatment_code}.json", "w") as f:
                    json.dump({"arima_order": arima_order, "sarima_seasonal": s_order}, f)

            except Exception as exc:
                logger.error("Failed region=%s treatment=%s: %s", region_code, treatment_code, exc, exc_info=True)

    logger.info("=" * 60)
    logger.info("ARIMA/SARIMA modelling complete.")
    logger.info("Next step: python scripts/07_model_holt_winters.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--region",  default=None)
    args = parser.parse_args()
    main(config_path=args.config, region_filter=args.region)
