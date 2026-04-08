"""
04_clean_harmonise.py — Data quality checks, outlier detection, and
COVID disruption handling.

Produces data/processed/data_quality_report.csv with all identified issues.

Usage:
    python scripts/04_clean_harmonise.py [--config ...]

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.db_connect import get_engine, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(config_path: str = "config/config.yaml") -> None:
    logger.info("=" * 60)
    logger.info("Step 4 — Data Cleaning and Harmonisation")
    logger.info("=" * 60)

    cfg    = load_config(config_path)
    engine = get_engine(config_path)
    issues = []

    # ── 1. Missing months check ───────────────────────────────────────────────
    logger.info("1. Checking for missing months in fact_rtt_monthly...")
    sql = """
        SELECT c.period_date, c.year_month,
               COUNT(f.fact_id) AS row_count
        FROM nhs.dim_calendar c
        LEFT JOIN nhs.fact_rtt_monthly f ON f.calendar_id = c.calendar_id
        WHERE c.period_date >= '2019-03-01'
          AND c.period_date <= CURRENT_DATE
        GROUP BY c.period_date, c.year_month
        ORDER BY c.period_date
    """
    calendar_df = pd.read_sql(sql, engine)
    missing_months = calendar_df[calendar_df["row_count"] == 0]
    if len(missing_months) > 0:
        logger.warning("Missing months with no RTT data: %d", len(missing_months))
        for _, r in missing_months.iterrows():
            issues.append({
                "check": "missing_month",
                "period_date": str(r["period_date"]),
                "region": "ALL",
                "detail": f"No rows in fact_rtt_monthly for {r['year_month']}",
                "severity": "HIGH",
            })
    else:
        logger.info("  ✓  No missing months found")

    # ── 2. Regional total verification ───────────────────────────────────────
    logger.info("2. Verifying regional totals per month...")
    sql = """
        SELECT c.year_month,
               SUM(CASE WHEN r.is_modern_region = TRUE THEN f.total_all ELSE 0 END) AS region_sum,
               SUM(CASE WHEN t.function_code = 'C_999' AND p.part_type = 'Part_2'
                        AND r.is_modern_region = FALSE THEN f.total_all ELSE 0 END) AS sub_sum
        FROM nhs.fact_rtt_monthly f
        JOIN nhs.dim_calendar    c ON f.calendar_id  = c.calendar_id
        JOIN nhs.dim_region      r ON f.region_id    = r.region_id
        JOIN nhs.dim_treatment   t ON f.treatment_id = t.treatment_id
        JOIN nhs.dim_rtt_part    p ON f.part_id      = p.part_id
        WHERE t.function_code = 'C_999'
          AND p.part_type = 'Part_2'
        GROUP BY c.year_month
        ORDER BY c.year_month
    """
    try:
        totals_df = pd.read_sql(sql, engine)
        if len(totals_df) > 0:
            logger.info("  ✓  Regional totals query complete — %d months", len(totals_df))
        else:
            logger.info("  (No data yet — run after ingestion)")
    except Exception as exc:
        logger.warning("Could not run regional total check: %s", exc)

    # ── 3. Outlier detection ──────────────────────────────────────────────────
    logger.info("3. Detecting statistical outliers (>3 SD from 12-month rolling mean)...")
    sql = """
        SELECT c.period_date, r.modern_region_code AS region_code,
               SUM(f.total_all) AS waiting_list_size
        FROM nhs.fact_rtt_monthly f
        JOIN nhs.dim_calendar  c ON f.calendar_id  = c.calendar_id
        JOIN nhs.dim_region    r ON f.region_id    = r.region_id
        JOIN nhs.dim_treatment t ON f.treatment_id = t.treatment_id
        JOIN nhs.dim_rtt_part  p ON f.part_id      = p.part_id
        WHERE t.function_code = 'C_999'
          AND p.part_type = 'Part_2'
          AND r.modern_region_code IS NOT NULL
        GROUP BY c.period_date, r.modern_region_code
        ORDER BY c.period_date, r.modern_region_code
    """
    try:
        ts_df = pd.read_sql(sql, engine, parse_dates=["period_date"])
        if len(ts_df) > 0:
            outlier_count = 0
            for region, grp in ts_df.groupby("region_code"):
                grp = grp.sort_values("period_date")
                vals = grp["waiting_list_size"].astype(float)
                rolling_mean = vals.rolling(12, min_periods=6).mean()
                rolling_std  = vals.rolling(12, min_periods=6).std()
                z_scores     = (vals - rolling_mean) / rolling_std.replace(0, np.nan)
                outliers     = grp[z_scores.abs() > 3]
                for _, row in outliers.iterrows():
                    issues.append({
                        "check":       "statistical_outlier",
                        "period_date": str(row["period_date"].date()),
                        "region":      region,
                        "detail":      f"Waiting list size {row['waiting_list_size']:,.0f} is >3 SD from 12-month mean",
                        "severity":    "MEDIUM",
                    })
                    outlier_count += 1
            logger.info("  Found %d statistical outliers", outlier_count)
        else:
            logger.info("  (No data yet — run after ingestion)")
    except Exception as exc:
        logger.warning("Outlier detection error: %s", exc)

    # ── 4. COVID disruption flag ──────────────────────────────────────────────
    logger.info("4. Flagging COVID disruption months (Apr-Jun 2020)...")
    covid_months = ["2020-04-01", "2020-05-01", "2020-06-01"]
    for m in covid_months:
        issues.append({
            "check":       "covid_disruption",
            "period_date": m,
            "region":      "ALL",
            "detail":      "NHS suspended elective care — zero/very low referrals expected. Not imputed for ARIMA (uses covid_dummy=1). Smoothed for Holt-Winters.",
            "severity":    "INFO",
        })
    logger.info("  ✓  COVID disruption months flagged")

    # ── 5. Row count sanity check ─────────────────────────────────────────────
    logger.info("5. Checking row counts per loaded month...")
    sql = """
        SELECT c.year_month, COUNT(f.fact_id) AS rows
        FROM nhs.fact_rtt_monthly f
        JOIN nhs.dim_calendar c ON f.calendar_id = c.calendar_id
        GROUP BY c.year_month
        ORDER BY c.year_month
    """
    try:
        counts_df = pd.read_sql(sql, engine)
        if len(counts_df) > 0:
            median_rows = counts_df["rows"].median()
            low_count   = counts_df[counts_df["rows"] < median_rows * 0.7]
            for _, r in low_count.iterrows():
                issues.append({
                    "check":       "low_row_count",
                    "period_date": r["year_month"],
                    "region":      "ALL",
                    "detail":      f"Only {r['rows']} rows loaded (median={median_rows:.0f}). Possible partial load or NHS reporting change.",
                    "severity":    "HIGH",
                })
            logger.info(
                "  Row counts: median=%d, min=%d, max=%d",
                int(median_rows), counts_df["rows"].min(), counts_df["rows"].max()
            )
    except Exception as exc:
        logger.warning("Row count check error: %s", exc)

    # ── Write quality report ──────────────────────────────────────────────────
    report_path = Path(cfg["paths"]["processed"]) / "data_quality_report.csv"
    report_df   = pd.DataFrame(issues)
    if len(report_df) > 0:
        report_df.to_csv(report_path, index=False)
        high   = len(report_df[report_df["severity"] == "HIGH"])
        medium = len(report_df[report_df["severity"] == "MEDIUM"])
        info   = len(report_df[report_df["severity"] == "INFO"])
        logger.info("Data quality report: HIGH=%d MEDIUM=%d INFO=%d", high, medium, info)
    else:
        logger.info("No issues found — writing empty report")
        pd.DataFrame(columns=["check","period_date","region","detail","severity"]).to_csv(report_path, index=False)

    logger.info("Report saved to %s", report_path)
    logger.info("=" * 60)
    logger.info("Cleaning complete. Review %s before modelling.", report_path)
    logger.info("Next step: python scripts/05_eda.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data quality checks and harmonisation")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
