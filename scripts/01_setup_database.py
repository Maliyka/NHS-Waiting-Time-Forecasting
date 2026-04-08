"""
01_setup_database.py — Build the full PostgreSQL star schema and seed all
dimension tables.

Executes the 5 SQL files in order, then populates:
  - dim_calendar  (every month 2019-01 to 2026-12)
  - dim_region    (all 20 NHS England region codes)
  - dim_treatment (all 20 treatment function codes)
  - dim_rtt_part  (all 5 RTT pathway types)
  - dim_model     (ARIMA, SARIMA, Holt-Winters ×2, Prophet)

Usage:
    python scripts/01_setup_database.py [--config config/config.yaml]

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.db_connect import (
    execute_sql_file,
    get_engine,
    get_row_count,
    load_config,
    upsert_dataframe,
)
from scripts.utils.data_helpers import (
    MODERN_REGION_NAMES,
    REGION_CONSOLIDATION,
    TREATMENT_CODES,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── SQL files to execute in order ─────────────────────────────────────────────

SQL_FILES = [
    "sql/01_create_schema.sql",
    "sql/02_create_dimensions.sql",
    "sql/03_create_facts.sql",
    "sql/04_create_indexes.sql",
    "sql/05_create_views.sql",
]


# ── Seed data definitions ─────────────────────────────────────────────────────

REGION_SEED = [
    # (region_code, region_name, region_short, modern_region_code, modern_region_name, is_modern)
    ("Q71", "NHS England London",                               "London",          "Q71", "NHS England London",                   True),
    ("Y60", "NHS England North East and Yorkshire",             "North East",      "Y60", "NHS England North East and Yorkshire",  True),
    ("Y61", "NHS England Midlands",                             "Midlands",        "Y61", "NHS England Midlands",                  True),
    ("Y62", "NHS England East of England",                      "East of England", "Y62", "NHS England East of England",           True),
    ("Y63", "NHS England North West",                           "North West",      "Y63", "NHS England North West",                True),
    ("Y58", "NHS England South East",                           "South East",      "Y58", "NHS England South East",                True),
    ("Y59", "NHS England South West",                           "South West",      "Y59", "NHS England South West",                True),
    ("Q72", "NHS England North (Yorkshire and Humber)",         "Yorks & Humber",  "Y60", "NHS England North East and Yorkshire",  False),
    ("Q74", "NHS England North (Cumbria and North East)",       "Cumbria NE",      "Y60", "NHS England North East and Yorkshire",  False),
    ("Q75", "NHS England North (Cheshire and Merseyside)",      "Cheshire",        "Y63", "NHS England North West",               False),
    ("Q83", "NHS England North (Greater Manchester)",           "Gtr Manchester",  "Y63", "NHS England North West",               False),
    ("Q84", "NHS England North (Lancashire and S Cumbria)",     "Lancs",           "Y63", "NHS England North West",               False),
    ("Q76", "NHS England Midlands and East (North Midlands)",   "N Midlands",      "Y61", "NHS England Midlands",                 False),
    ("Q77", "NHS England Midlands and East (West Midlands)",    "W Midlands",      "Y61", "NHS England Midlands",                 False),
    ("Q78", "NHS England Midlands and East (Central Midlands)", "C Midlands",      "Y61", "NHS England Midlands",                 False),
    ("Q79", "NHS England Midlands and East (East)",             "East",            "Y62", "NHS England East of England",          False),
    ("Q85", "NHS England South West (South West South)",        "SW South",        "Y59", "NHS England South West",               False),
    ("Q86", "NHS England South West (South West North)",        "SW North",        "Y59", "NHS England South West",               False),
    ("Q87", "NHS England South East (Hampshire, IoW, Thames Valley)", "Hants IoW", "Y58", "NHS England South East",              False),
    ("Q88", "NHS England South East (Kent, Surrey and Sussex)", "Kent Surrey",     "Y58", "NHS England South East",              False),
]

TREATMENT_SEED = [
    # (function_code, function_name, specialty_group, is_total)
    ("C_100", "General Surgery",        "Surgical", False),
    ("C_101", "Urology",                "Surgical", False),
    ("C_110", "Trauma & Orthopaedics",  "Surgical", False),
    ("C_120", "Ear Nose & Throat",      "Surgical", False),
    ("C_130", "Ophthalmology",          "Surgical", False),
    ("C_140", "Oral Surgery",           "Surgical", False),
    ("C_150", "Neurosurgery",           "Surgical", False),
    ("C_160", "Plastic Surgery",        "Surgical", False),
    ("C_170", "Cardiothoracic Surgery", "Surgical", False),
    ("C_300", "General Medicine",       "Medical",  False),
    ("C_301", "Gastroenterology",       "Medical",  False),
    ("C_320", "Cardiology",             "Medical",  False),
    ("C_330", "Dermatology",            "Medical",  False),
    ("C_340", "Thoracic Medicine",      "Medical",  False),
    ("C_400", "Neurology",              "Medical",  False),
    ("C_410", "Rheumatology",           "Medical",  False),
    ("C_430", "Geriatric Medicine",     "Medical",  False),
    ("C_502", "Gynaecology",            "Medical",  False),
    ("C_999", "Total",                  "Total",    True),
    ("X01",   "Other",                  "Other",    False),
]

RTT_PART_SEED = [
    ("Part_1A", "Completed Pathways For Admitted Patients",     False),
    ("Part_1B", "Completed Pathways For Non-Admitted Patients", False),
    ("Part_2",  "Incomplete Pathways",                          True),
    ("Part_2A", "Incomplete Pathways Adjusted",                 False),
    ("Part_3",  "New Periods Starting",                         False),
]

MODEL_SEED = [
    ("ARIMA",                     "Statistical", "AutoRegressive Integrated Moving Average with COVID dummy exogenous regressor"),
    ("SARIMA",                    "Statistical", "Seasonal ARIMA (p,d,q)(P,D,Q,12) — statsmodels SARIMAX"),
    ("HoltWinters_Additive",      "Statistical", "Holt-Winters ETS — additive trend and additive seasonality"),
    ("HoltWinters_Multiplicative","Statistical", "Holt-Winters ETS — additive trend and multiplicative seasonality"),
    ("Prophet",                   "Bayesian",    "Facebook Prophet with fixed COVID changepoints and UK bank holidays"),
]


# ── Calendar generation ───────────────────────────────────────────────────────

def generate_calendar_rows() -> list:
    """Generate one dict per month from 2019-01-01 to 2026-12-01."""
    import calendar

    MONTH_NAMES = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    COVID_START = date(2020, 4, 1)
    COVID_END   = date(2022, 4, 1)

    rows = []
    for year in range(2019, 2027):
        for month in range(1, 13):
            d = date(year, month, 1)
            is_covid   = COVID_START <= d < COVID_END
            is_pre     = d < COVID_START
            is_post    = d >= COVID_END
            covid_dum  = 1 if is_covid else 0
            quarter    = (month - 1) // 3 + 1

            # Approximate bank holiday count
            if month in (4,):   bh = 2   # Good Friday + Easter Monday
            elif month in (5,): bh = 2   # Two May BHs
            elif month in (1, 8): bh = 1
            elif month == 12:   bh = 2
            else:               bh = 0

            rows.append({
                "period_date":        d.isoformat(),
                "year":               year,
                "month":              month,
                "quarter":            quarter,
                "month_name":         MONTH_NAMES[month - 1],
                "year_month":         f"{year}-{month:02d}",
                "is_pre_covid":       is_pre,
                "is_covid_period":    is_covid,
                "is_post_covid":      is_post,
                "covid_dummy":        covid_dum,
                "bank_holiday_count": bh,
            })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str = "config/config.yaml") -> None:
    logger.info("=" * 60)
    logger.info("Step 1 — Setup Database")
    logger.info("=" * 60)

    engine = get_engine(config_path)
    logger.info("Database engine connected")

    # ── Execute SQL files ──────────────────────────────────────────────────────
    logger.info("Executing SQL schema files...")
    for sql_file in SQL_FILES:
        if not Path(sql_file).exists():
            logger.error("SQL file missing: %s — aborting", sql_file)
            sys.exit(1)
        execute_sql_file(engine, sql_file)
        logger.info("  ✓  %s", sql_file)

    # ── Seed dim_calendar ──────────────────────────────────────────────────────
    logger.info("Seeding dim_calendar (2019-01 to 2026-12)...")
    calendar_rows = generate_calendar_rows()
    upsert_dataframe(calendar_rows, "dim_calendar", engine)
    count = get_row_count(engine, "dim_calendar")
    logger.info("  ✓  dim_calendar: %d rows", count)

    # ── Seed dim_region ────────────────────────────────────────────────────────
    logger.info("Seeding dim_region...")
    region_rows = [
        {
            "region_code":        r[0],
            "region_name":        r[1],
            "region_short":       r[2],
            "modern_region_code": r[3],
            "modern_region_name": r[4],
            "is_modern_region":   r[5],
        }
        for r in REGION_SEED
    ]
    upsert_dataframe(region_rows, "dim_region", engine)
    count = get_row_count(engine, "dim_region")
    logger.info("  ✓  dim_region: %d rows", count)

    # ── Seed dim_treatment ─────────────────────────────────────────────────────
    logger.info("Seeding dim_treatment...")
    treatment_rows = [
        {
            "function_code":   t[0],
            "function_name":   t[1],
            "specialty_group": t[2],
            "is_total":        t[3],
        }
        for t in TREATMENT_SEED
    ]
    upsert_dataframe(treatment_rows, "dim_treatment", engine)
    count = get_row_count(engine, "dim_treatment")
    logger.info("  ✓  dim_treatment: %d rows", count)

    # ── Seed dim_rtt_part ──────────────────────────────────────────────────────
    logger.info("Seeding dim_rtt_part...")
    part_rows = [
        {"part_type": p[0], "part_description": p[1], "is_primary": p[2]}
        for p in RTT_PART_SEED
    ]
    upsert_dataframe(part_rows, "dim_rtt_part", engine)
    count = get_row_count(engine, "dim_rtt_part")
    logger.info("  ✓  dim_rtt_part: %d rows", count)

    # ── Seed dim_model ─────────────────────────────────────────────────────────
    logger.info("Seeding dim_model...")
    model_rows = [
        {"model_name": m[0], "model_type": m[1], "description": m[2]}
        for m in MODEL_SEED
    ]
    upsert_dataframe(model_rows, "dim_model", engine)
    count = get_row_count(engine, "dim_model")
    logger.info("  ✓  dim_model: %d rows", count)

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Database setup complete.")
    logger.info("  dim_calendar  : %d rows", get_row_count(engine, "dim_calendar"))
    logger.info("  dim_region    : %d rows", get_row_count(engine, "dim_region"))
    logger.info("  dim_treatment : %d rows", get_row_count(engine, "dim_treatment"))
    logger.info("  dim_rtt_part  : %d rows", get_row_count(engine, "dim_rtt_part"))
    logger.info("  dim_model     : %d rows", get_row_count(engine, "dim_model"))
    logger.info("Next step: python scripts/02_ingest_rtt_csv.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup NHS project database schema")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
