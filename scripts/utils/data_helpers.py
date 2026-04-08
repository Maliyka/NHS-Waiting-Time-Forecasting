"""
data_helpers.py — Data parsing, cleaning, and column mapping utilities.

Used by all ingestion and modelling scripts in the NHS Forecasting project.
Francis Kwesi Acquah | B01821156 | UWS
"""

import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

# ── Month name lookup (covers both full and 3-letter abbreviations) ────────────

MONTH_NAME_TO_NUM: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# ── Region consolidation map (2019 sub-regional → modern 7-region) ─────────────

REGION_CONSOLIDATION: Dict[str, str] = {
    # North East and Yorkshire
    "Q72": "Y60", "Q74": "Y60",
    # North West
    "Q75": "Y63", "Q83": "Y63", "Q84": "Y63",
    # Midlands
    "Q76": "Y61", "Q77": "Y61", "Q78": "Y61",
    # East of England
    "Q79": "Y62",
    # South West
    "Q85": "Y59", "Q86": "Y59",
    # South East
    "Q87": "Y58", "Q88": "Y58",
    # London (unchanged)
    "Q71": "Q71",
    # Modern codes map to themselves
    "Y60": "Y60", "Y61": "Y61", "Y62": "Y62",
    "Y63": "Y63", "Y58": "Y58", "Y59": "Y59",
}

MODERN_REGION_NAMES: Dict[str, str] = {
    "Q71": "NHS England London",
    "Y60": "NHS England North East and Yorkshire",
    "Y61": "NHS England Midlands",
    "Y62": "NHS England East of England",
    "Y63": "NHS England North West",
    "Y58": "NHS England South East",
    "Y59": "NHS England South West",
}

# ── Treatment function codes ───────────────────────────────────────────────────

TREATMENT_CODES: Dict[str, str] = {
    "C_100": "General Surgery",
    "C_101": "Urology",
    "C_110": "Trauma & Orthopaedics",
    "C_120": "Ear Nose & Throat",
    "C_130": "Ophthalmology",
    "C_140": "Oral Surgery",
    "C_150": "Neurosurgery",
    "C_160": "Plastic Surgery",
    "C_170": "Cardiothoracic Surgery",
    "C_300": "General Medicine",
    "C_301": "Gastroenterology",
    "C_320": "Cardiology",
    "C_330": "Dermatology",
    "C_340": "Thoracic Medicine",
    "C_400": "Neurology",
    "C_410": "Rheumatology",
    "C_430": "Geriatric Medicine",
    "C_502": "Gynaecology",
    "C_999": "Total",
    "X01":   "Other",
}

# XLS specialty column names → treatment function codes
XLS_SPECIALTY_TO_CODE: Dict[str, str] = {
    "General Surgery":         "C_100",
    "Urology":                 "C_101",
    "Trauma & Orthopaedics":   "C_110",
    "Trauma and Orthopaedics": "C_110",
    "Ear Nose & Throat":       "C_120",
    "Ear, Nose & Throat":      "C_120",
    "Ophthalmology":           "C_130",
    "Oral Surgery":            "C_140",
    "Neurosurgery":            "C_150",
    "Plastic Surgery":         "C_160",
    "Cardiothoracic Surgery":  "C_170",
    "General Medicine":        "C_300",
    "Gastroenterology":        "C_301",
    "Cardiology":              "C_320",
    "Dermatology":             "C_330",
    "Thoracic Medicine":       "C_340",
    "Neurology":               "C_400",
    "Rheumatology":            "C_410",
    "Geriatric Medicine":      "C_430",
    "Gynaecology":             "C_502",
    "Other":                   "X01",
    "Total":                   "C_999",
}


# ── Column name mappings ───────────────────────────────────────────────────────

def get_raw_week_columns() -> List[str]:
    """Return the 53 weekly bucket column names as they appear in the RTT CSV."""
    cols = []
    for i in range(52):
        lo = str(i).zfill(2)
        hi = str(i + 1).zfill(2)
        cols.append(f"Gt {lo} To {hi} Weeks SUM 1")
    cols.append("Gt 52 Weeks SUM 1")
    return cols


def get_db_week_columns() -> List[str]:
    """Return the 53 weekly bucket column names as stored in fact_rtt_monthly."""
    cols = []
    for i in range(52):
        lo = str(i).zfill(2)
        hi = str(i + 1).zfill(2)
        cols.append(f"wk_{lo}_{hi}")
    cols.append("wk_52_plus")
    return cols


def map_raw_to_db_columns() -> Dict[str, str]:
    """Return dict mapping raw CSV column name → database column name for all 53 buckets."""
    raw = get_raw_week_columns()
    db  = get_db_week_columns()
    return dict(zip(raw, db))


# ── Period string parsing ──────────────────────────────────────────────────────

def parse_period_to_date(period_str: str) -> Optional[date]:
    """
    Parse the RTT CSV 'Period' field to a date object (first of month).

    Handles formats like:
        'RTT-MARCH-2019'  → date(2019, 3, 1)
        'RTT-JAN-2023'    → date(2023, 1, 1)

    Args:
        period_str: Value from the 'Period' CSV column.

    Returns:
        date object set to the 1st of the reported month, or None on failure.
    """
    if not period_str or not isinstance(period_str, str):
        return None

    # Strip surrounding quotes and whitespace
    period_str = period_str.strip().strip('"').strip("'")

    # Expected format: RTT-MONTHNAME-YEAR
    parts = period_str.upper().split("-")
    if len(parts) != 3 or parts[0] != "RTT":
        return None

    month_str = parts[1].lower()
    year_str  = parts[2]

    month_num = MONTH_NAME_TO_NUM.get(month_str)
    if month_num is None:
        return None

    try:
        year = int(year_str)
        return date(year, month_num, 1)
    except ValueError:
        return None


def parse_filename_to_date(filename: str) -> Optional[date]:
    """
    Parse a New Periods XLS filename to a date object (first of month).

    Handles patterns like:
        'New-Periods-Commissioner-Mar19-revised-XLS-553K.xls' → date(2019, 3, 1)
        'New-Periods-Commissioner-Dec25-revised-XLS.xls'      → date(2025, 12, 1)
        'RTT-MARCH-2019-full-extract.csv'                      → date(2019, 3, 1)

    Args:
        filename: Bare filename or full path string.

    Returns:
        date object set to the 1st of the month, or None on failure.
    """
    filename = str(filename)
    basename = filename.split("/")[-1].split("\\")[-1]

    # Pattern 1: XLS style — Mon## (e.g. Mar19, Dec25)
    m = re.search(r"([A-Za-z]{3})(\d{2})(?:[^0-9]|$)", basename)
    if m:
        mon_str  = m.group(1).lower()
        yr_short = int(m.group(2))
        year     = 2000 + yr_short if yr_short <= 50 else 1900 + yr_short
        month    = MONTH_NAME_TO_NUM.get(mon_str)
        if month:
            return date(year, month, 1)

    # Pattern 2: RTT CSV style — RTT-MONTH-YEAR
    m = re.search(r"RTT-([A-Za-z]+)-(\d{4})", basename, re.IGNORECASE)
    if m:
        mon_str = m.group(1).lower()
        year    = int(m.group(2))
        month   = MONTH_NAME_TO_NUM.get(mon_str)
        if month:
            return date(year, month, 1)

    return None


# ── Value cleaning ─────────────────────────────────────────────────────────────

SUPPRESSED_VALUES = frozenset(["*", "-", ".", "..", "n/a", "na", "nan", "", "none"])


def clean_numeric_value(val: Union[str, int, float, None]) -> Optional[int]:
    """
    Convert a raw CSV/XLS cell value to an integer.

    Returns None for:
      - NHS suppression marker '*'
      - Empty strings, dashes, dots
      - NaN / None
      - Values that cannot be coerced to integer

    IMPORTANT: Never converts '*' to 0 — suppressed values are genuinely
    unknown small numbers, not zeros.

    Args:
        val: Raw cell value from CSV or XLS.

    Returns:
        Integer value, or None if the value is missing/suppressed.
    """
    if val is None:
        return None

    s = str(val).strip().lower().replace(",", "")

    if s in SUPPRESSED_VALUES:
        return None

    # Handle float strings like "12345.0"
    try:
        return int(float(s))
    except (ValueError, OverflowError):
        return None


def clean_string_value(val: Union[str, None]) -> Optional[str]:
    """Strip whitespace and surrounding quotes from a string value."""
    if val is None:
        return None
    return str(val).strip().strip('"').strip("'").strip() or None


# ── 18-week metric computation ─────────────────────────────────────────────────

def compute_18wk_metrics(
    row: Dict[str, Optional[int]]
) -> Dict[str, Optional[Union[int, float]]]:
    """
    Compute the NHS 18-week RTT performance metrics from a row dict.

    Patients within 18 weeks = sum of wk_00_01 through wk_17_18 (18 buckets).
    Patients over 18 weeks   = sum of wk_18_19 through wk_52_plus (35 buckets).
    pct_within_18wks         = patients_within / total_all * 100.

    NULL weekly buckets (suppressed values) are treated as 0 for summation.
    If total_all is 0 or None, pct_within_18wks is None.

    Args:
        row: Dict with keys wk_00_01, wk_01_02, ..., wk_52_plus, total_all.

    Returns:
        Dict with keys patients_within_18wks, patients_over_18wks, pct_within_18wks.
    """
    db_cols = get_db_week_columns()
    within_cols = db_cols[:18]   # wk_00_01 .. wk_17_18
    over_cols   = db_cols[18:]   # wk_18_19 .. wk_52_plus

    within = sum(row.get(c) or 0 for c in within_cols)
    over   = sum(row.get(c) or 0 for c in over_cols)

    total_all = row.get("total_all")
    if total_all and total_all > 0:
        pct = round(within / total_all * 100, 4)
    else:
        pct = None

    return {
        "patients_within_18wks": within,
        "patients_over_18wks":   over,
        "pct_within_18wks":      pct,
    }


# ── Region consolidation ───────────────────────────────────────────────────────

def consolidate_region(region_code: str) -> Tuple[str, str]:
    """
    Map any NHS England region code (2019 or modern) to its modern equivalent.

    Args:
        region_code: Raw region code from the RTT CSV Provider Parent Org Code column.

    Returns:
        Tuple of (modern_region_code, modern_region_name).
        Returns (region_code, '') if code is not in the consolidation map
        (e.g. individual provider codes — these should be filtered upstream).
    """
    code = str(region_code).strip().upper()
    modern_code = REGION_CONSOLIDATION.get(code, code)
    modern_name = MODERN_REGION_NAMES.get(modern_code, "")
    return modern_code, modern_name


# ── COVID era labelling ────────────────────────────────────────────────────────

COVID_START = date(2020, 4, 1)
COVID_END   = date(2022, 4, 1)


def get_covid_era(period_date: date) -> str:
    """
    Return the COVID era label for a given period date.

    Returns:
        'pre_covid'  — before April 2020
        'covid'      — April 2020 to March 2022 inclusive
        'post_covid' — April 2022 onwards
    """
    if period_date < COVID_START:
        return "pre_covid"
    elif period_date < COVID_END:
        return "covid"
    else:
        return "post_covid"


def get_covid_dummy(period_date: date) -> int:
    """Return 1 if the period falls in the COVID era, else 0."""
    return 1 if COVID_START <= period_date < COVID_END else 0
