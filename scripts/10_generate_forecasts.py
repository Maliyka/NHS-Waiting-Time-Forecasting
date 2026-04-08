"""
10_generate_forecasts.py — Generate final 12-month forecasts for all regions.

For each region × best model:
  1. Re-fit on full history
  2. Generate base scenario (12-month ahead)
  3. Generate demand_plus10 (+10% new_periods regressor)
  4. Generate demand_minus10 (-10%)
  5. Store in fact_forecast_outputs
  6. Export combined CSV and fan-chart PNGs

Usage:
    python scripts/10_generate_forecasts.py [--config ...] [--horizon 12]

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.data_helpers import get_covid_era
from scripts.utils.db_connect import get_engine, load_config, upsert_dataframe

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

COVID_START   = pd.Timestamp("2020-04-01")
COVID_END     = pd.Timestamp("2022-04-01")
TARGET_18WK   = 92.0


def load_best_models(engine) -> pd.DataFrame:
    """Return the best model (lowest mean MAE) per region for C_999 Part_2."""
    sql = """
        SELECT region_code, model_name, mean_mae, mean_rmse
        FROM nhs.v_model_leaderboard
        WHERE treatment_code = 'C_999' AND mae_rank = 1
    """
    try:
        return pd.read_sql(sql, engine)
    except Exception:
        # Fallback: return SARIMA for all regions
        cfg = load_config()
        rows = [{"region_code": r, "model_name": "SARIMA", "mean_mae": None, "mean_rmse": None}
                for r in ["Q71","Y60","Y61","Y62","Y63","Y58","Y59"]]
        return pd.DataFrame(rows)


def load_series(engine, region_code: str) -> pd.DataFrame:
    from sqlalchemy import text
    sql = text("""
        SELECT r.period_date, r.waiting_list_size, r.covid_dummy,
               COALESCE(np.new_periods_count, 0) AS new_periods_count
        FROM nhs.v_rtt_regional_monthly r
        LEFT JOIN nhs.v_new_periods_regional np
            ON np.period_date   = r.period_date
           AND np.region_code   = r.region_code
           AND np.treatment_code = r.treatment_code
        WHERE r.region_code    = :rc
          AND r.treatment_code = 'C_999'
          AND r.part_type      = 'Part_2'
        ORDER BY r.period_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"rc": region_code}, parse_dates=["period_date"])
    return df.dropna(subset=["waiting_list_size"])


def get_dim_ids(engine, region_code: str, treatment_code: str,
                part_type: str, model_name: str) -> tuple:
    from sqlalchemy import text
    with engine.connect() as conn:
        rid = conn.execute(text("SELECT region_id FROM nhs.dim_region WHERE region_code=:c"), {"c": region_code}).scalar()
        tid = conn.execute(text("SELECT treatment_id FROM nhs.dim_treatment WHERE function_code=:c"), {"c": treatment_code}).scalar()
        pid = conn.execute(text("SELECT part_id FROM nhs.dim_rtt_part WHERE part_type=:c"), {"c": part_type}).scalar()
        mid = conn.execute(text("SELECT model_id FROM nhs.dim_model WHERE model_name=:c"), {"c": model_name}).scalar()
    return rid, tid, pid, mid


def forecast_sarima(df: pd.DataFrame, horizon: int, scenario: str) -> Optional[pd.DataFrame]:
    """Fit SARIMA on full history and forecast `horizon` months ahead."""
    import itertools
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = df["waiting_list_size"].astype(float)
    exog   = df["covid_dummy"].astype(float)
    idx    = pd.DatetimeIndex(pd.to_datetime(df["period_date"])).to_period("M").to_timestamp("M")
    series.index = idx
    exog.index   = idx

    # Use a sensible default order — could be replaced by stored params from script 06
    order   = (1, 1, 1)
    s_order = (1, 1, 1, 12)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(series, exog=exog.values.reshape(-1,1),
                        order=order, seasonal_order=s_order,
                        enforce_stationarity=False, enforce_invertibility=False)
            fitted = m.fit(disp=False, maxiter=150)

        future_exog = np.zeros((horizon, 1))  # Post-COVID → dummy=0
        fc   = fitted.get_forecast(steps=horizon, exog=future_exog)
        pred = fc.predicted_mean
        ci80 = fc.conf_int(alpha=0.20)
        ci95 = fc.conf_int(alpha=0.05)

        last_date = series.index[-1]
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(horizon)]

        return pd.DataFrame({
            "forecast_period_date": [d.date() for d in future_dates],
            "predicted_value":      pred.values.round(2),
            "lower_80":             ci80.iloc[:, 0].values.round(2),
            "upper_80":             ci80.iloc[:, 1].values.round(2),
            "lower_95":             ci95.iloc[:, 0].values.round(2),
            "upper_95":             ci95.iloc[:, 1].values.round(2),
            "scenario":             scenario,
        })
    except Exception as exc:
        logger.warning("SARIMA forecast failed: %s", exc)
        return None


def forecast_prophet(df: pd.DataFrame, horizon: int, scenario: str,
                     regressor_scale: float = 1.0) -> Optional[pd.DataFrame]:
    """Fit Prophet on full history and forecast `horizon` months ahead."""
    try:
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        from prophet import Prophet
        import holidays as hols

        ph_df = pd.DataFrame({
            "ds": pd.to_datetime(df["period_date"]),
            "y":  df["waiting_list_size"].astype(float),
            "new_periods_count": df["new_periods_count"].astype(float),
        })

        hol_rows = []
        for yr in range(2019, 2027):
            for d, n in hols.country_holidays("GB", years=yr).items():
                hol_rows.append({"ds": pd.Timestamp(d), "holiday": n[:50]})
        holidays_df = pd.DataFrame(hol_rows).drop_duplicates(subset=["ds","holiday"])

        m = Prophet(
            changepoints=["2020-03-01","2021-07-01","2022-04-01","2022-10-01"],
            changepoint_prior_scale=0.3, seasonality_prior_scale=10.0,
            holidays=holidays_df, yearly_seasonality=True,
            weekly_seasonality=False, daily_seasonality=False, uncertainty_samples=500,
        )
        m.add_regressor("new_periods_count", standardize=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(ph_df)

        future = m.make_future_dataframe(periods=horizon, freq="MS")
        future_regressor = float(ph_df["new_periods_count"].tail(3).mean()) * regressor_scale
        future["new_periods_count"] = future_regressor
        forecast = m.predict(future).tail(horizon)

        half_80 = (forecast["yhat_upper"].values - forecast["yhat_lower"].values) / 2
        lower_95 = forecast["yhat"].values - half_80 * 1.65
        upper_95 = forecast["yhat"].values + half_80 * 1.65

        return pd.DataFrame({
            "forecast_period_date": [d.date() for d in pd.to_datetime(forecast["ds"])],
            "predicted_value":      forecast["yhat"].values.round(2),
            "lower_80":             forecast["yhat_lower"].values.round(2),
            "upper_80":             forecast["yhat_upper"].values.round(2),
            "lower_95":             lower_95.round(2),
            "upper_95":             upper_95.round(2),
            "scenario":             scenario,
        })
    except Exception as exc:
        logger.warning("Prophet forecast failed: %s", exc)
        return None


def forecast_holt_winters(df: pd.DataFrame, horizon: int, scenario: str) -> Optional[pd.DataFrame]:
    """Fit Holt-Winters on full history and forecast `horizon` months ahead."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    series = df["waiting_list_size"].astype(float)
    idx    = pd.DatetimeIndex(pd.to_datetime(df["period_date"])).to_period("M").to_timestamp("M")
    series.index = idx

    # Smooth COVID months
    for m in ["2020-04-01","2020-05-01","2020-06-01"]:
        ts = pd.Timestamp(m)
        if ts in series.index:
            series.loc[ts] = np.nan
    series = series.interpolate(method="linear")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12, damped_trend=True)
            fitted = model.fit(optimized=True)

        fc    = fitted.forecast(horizon)
        resid = fitted.resid.values
        bootstrap = np.zeros((500, horizon))
        for i in range(500):
            bootstrap[i] = fc.values + np.random.choice(resid, size=horizon, replace=True)

        last_date    = series.index[-1]
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(horizon)]

        return pd.DataFrame({
            "forecast_period_date": [d.date() for d in future_dates],
            "predicted_value":      fc.values.round(2),
            "lower_80":             np.percentile(bootstrap, 10, axis=0).round(2),
            "upper_80":             np.percentile(bootstrap, 90, axis=0).round(2),
            "lower_95":             np.percentile(bootstrap, 2.5, axis=0).round(2),
            "upper_95":             np.percentile(bootstrap, 97.5, axis=0).round(2),
            "scenario":             scenario,
        })
    except Exception as exc:
        logger.warning("Holt-Winters forecast failed: %s", exc)
        return None


def run_forecast(model_name: str, df: pd.DataFrame, horizon: int,
                 scenario: str, regressor_scale: float = 1.0) -> Optional[pd.DataFrame]:
    if "ARIMA" in model_name or "SARIMA" in model_name:
        return forecast_sarima(df, horizon, scenario)
    elif "Prophet" in model_name:
        return forecast_prophet(df, horizon, scenario, regressor_scale)
    else:
        return forecast_holt_winters(df, horizon, scenario)


def save_fan_chart(df_history: pd.DataFrame, fc_df: pd.DataFrame,
                   region_code: str, model_name: str, out: Path):
    """Save a fan chart showing historical data + 12-month forecast with PI bands."""
    hist = df_history.copy()
    hist["period_date"] = pd.to_datetime(hist["period_date"])
    hist = hist.sort_values("period_date")

    fc = fc_df[fc_df["scenario"] == "base"].copy()
    fc["forecast_period_date"] = pd.to_datetime(fc["forecast_period_date"])
    fc = fc.sort_values("forecast_period_date")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(hist["period_date"], hist["waiting_list_size"] / 1e6,
            color="#1f77b4", linewidth=2, label="Historical")
    ax.plot(fc["forecast_period_date"], fc["predicted_value"] / 1e6,
            color="#E74C3C", linewidth=2, linestyle="--", label="Forecast (base)")
    ax.fill_between(fc["forecast_period_date"],
                    fc["lower_80"] / 1e6, fc["upper_80"] / 1e6,
                    alpha=0.25, color="#E74C3C", label="80% PI")
    ax.fill_between(fc["forecast_period_date"],
                    fc["lower_95"] / 1e6, fc["upper_95"] / 1e6,
                    alpha=0.12, color="#E74C3C", label="95% PI")
    ax.axvspan(COVID_START, COVID_END, alpha=0.12, color="orange", label="COVID period")
    ax.set_title(f"NHS Waiting List Forecast — {region_code} ({model_name})\nPart_2 Incomplete Pathways, Total (C_999)", fontsize=12)
    ax.set_xlabel("Date"); ax.set_ylabel("Patients (millions)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); fig.tight_layout()

    fname = out / f"{region_code}_forecast_fan_chart.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fan chart saved: %s", fname.name)


def main(config_path: str = "config/config.yaml", horizon: int = 12) -> None:
    logger.info("=" * 60)
    logger.info("Step 10 — Generate Forecasts")
    logger.info("=" * 60)

    cfg    = load_config(config_path)
    engine = get_engine(config_path)
    fc_out = Path(cfg["paths"]["forecasts"])
    pl_out = Path(cfg["paths"]["forecast_plots"])
    fc_out.mkdir(parents=True, exist_ok=True)
    pl_out.mkdir(parents=True, exist_ok=True)

    best_models = load_best_models(engine)
    if best_models.empty:
        logger.warning("No model leaderboard data — using SARIMA for all regions")
        regions = list(cfg["regions"]["modern_codes"].keys())
        best_models = pd.DataFrame([
            {"region_code": r, "model_name": "SARIMA"} for r in regions
        ])

    all_forecasts = []
    treatment_code = "C_999"
    part_type      = "Part_2"
    scenarios      = [("base", 1.0), ("demand_plus10", 1.1), ("demand_minus10", 0.9)]

    for _, bm in best_models.iterrows():
        region_code = bm["region_code"]
        model_name  = bm["model_name"]
        logger.info("Forecasting %s with %s", region_code, model_name)

        try:
            df  = load_series(engine, region_code)
            rid, tid, pid, mid = get_dim_ids(engine, region_code, treatment_code, part_type, model_name)
            if not all(x is not None for x in [rid, tid, pid, mid]):
                logger.warning("  Dim IDs missing for %s — skipping", region_code)
                continue

            training_end = str(df["period_date"].max())
            fc_frames    = []

            for scenario_name, scale in scenarios:
                fc_df = run_forecast(model_name, df, horizon, scenario_name, scale)
                if fc_df is None:
                    continue
                fc_df["region_id"]         = rid
                fc_df["treatment_id"]      = tid
                fc_df["part_id"]           = pid
                fc_df["model_id"]          = mid
                fc_df["horizon_months"]    = horizon
                fc_df["training_end_date"] = training_end
                fc_df["model_params"]      = json.dumps({"model": model_name})
                fc_frames.append(fc_df)

                if scenario_name == "base":
                    all_forecasts.append(
                        fc_df.assign(region_code=region_code, model_name=model_name)
                    )

            if fc_frames:
                db_rows = []
                for frame in fc_frames:
                    for _, row in frame.iterrows():
                        db_rows.append({
                            "region_id":            row["region_id"],
                            "treatment_id":         row["treatment_id"],
                            "part_id":              row["part_id"],
                            "model_id":             row["model_id"],
                            "forecast_period_date": str(row["forecast_period_date"]),
                            "horizon_months":       int(row["horizon_months"]),
                            "scenario":             row["scenario"],
                            "predicted_value":      row["predicted_value"],
                            "lower_80":             row["lower_80"],
                            "upper_80":             row["upper_80"],
                            "lower_95":             row["lower_95"],
                            "upper_95":             row["upper_95"],
                            "training_end_date":    str(row["training_end_date"]),
                            "model_params":         row["model_params"],
                        })
                upsert_dataframe(db_rows, "fact_forecast_outputs", engine)

                # Fan chart (base scenario only)
                base_fc = fc_frames[0][fc_frames[0]["scenario"] == "base"]
                save_fan_chart(df, base_fc, region_code, model_name, pl_out)
                logger.info("  %s complete — %d forecast rows", region_code, len(db_rows))

        except Exception as exc:
            logger.error("Failed %s: %s", region_code, exc, exc_info=True)

    # Export combined CSV
    if all_forecasts:
        combined = pd.concat(all_forecasts, ignore_index=True)
        today    = date.today().isoformat()
        out_path = fc_out / f"forecasts_all_regions_{today}.csv"
        combined.to_csv(out_path, index=False)
        logger.info("Combined forecasts CSV saved to %s", out_path)

    logger.info("=" * 60)
    logger.info("Forecast generation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    main(config_path=args.config, horizon=args.horizon)
