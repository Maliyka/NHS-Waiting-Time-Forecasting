"""
03_ingest_new_periods_xls.py

Actual XLS column layout (LONG format):
  Region Code | CCG Code | CCG Name | Treatment Function Code |
  Treatment Function | Number of new RTT clock starts during the month

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import xlrd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.data_helpers import clean_numeric_value, consolidate_region, parse_filename_to_date
from scripts.utils.db_connect import get_engine, load_config, upsert_dataframe

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TREATMENT_CODE_MAP = {
    "100": "C_100", "101": "C_101", "110": "C_110", "120": "C_120",
    "130": "C_130", "140": "C_140", "150": "C_150", "160": "C_160",
    "170": "C_170", "300": "C_300", "301": "C_301", "320": "C_320",
    "330": "C_330", "340": "C_340", "400": "C_400", "410": "C_410",
    "430": "C_430", "502": "C_502", "999": "C_999",
    "c_100": "C_100", "c_101": "C_101", "c_110": "C_110", "c_120": "C_120",
    "c_130": "C_130", "c_140": "C_140", "c_150": "C_150", "c_160": "C_160",
    "c_170": "C_170", "c_300": "C_300", "c_301": "C_301", "c_320": "C_320",
    "c_330": "C_330", "c_340": "C_340", "c_400": "C_400", "c_410": "C_410",
    "c_430": "C_430", "c_502": "C_502", "c_999": "C_999", "x01": "X01",
}
NAME_TO_CODE = {
    "general surgery": "C_100", "urology": "C_101",
    "trauma & orthopaedics": "C_110", "trauma and orthopaedics": "C_110",
    "ear, nose & throat": "C_120", "ear nose & throat": "C_120",
    "ent": "C_120", "ophthalmology": "C_130", "oral surgery": "C_140",
    "neurosurgery": "C_150", "plastic surgery": "C_160",
    "cardiothoracic surgery": "C_170", "general medicine": "C_300",
    "gastroenterology": "C_301", "cardiology": "C_320",
    "dermatology": "C_330", "thoracic medicine": "C_340",
    "neurology": "C_400", "rheumatology": "C_410",
    "geriatric medicine": "C_430", "gynaecology": "C_502",
    "total": "C_999", "other": "X01",
}


def resolve_treatment_code(code_raw: str, name_raw: str) -> Optional[str]:
    code = str(code_raw).strip().lower().replace(" ", "")
    name = str(name_raw).strip().lower()
    if code in TREATMENT_CODE_MAP:
        return TREATMENT_CODE_MAP[code]
    for k, v in NAME_TO_CODE.items():
        if k in name or name in k:
            return v
    return None


def cell_str(sheet, r: int, c: int) -> str:
    try:
        v = sheet.cell(r, c).value
        return "" if v is None else str(v).strip()
    except IndexError:
        return ""


def find_data_sheet(wb):
    for name in wb.sheet_names():
        if any(k in name.lower() for k in ["commissioner", "data", "national"]):
            return wb.sheet_by_name(name)
    return wb.sheet_by_index(0) if wb.nsheets > 0 else None


def load_dim_caches(engine) -> Dict:
    from sqlalchemy import text
    caches = {"calendar": {}, "region": {}, "treatment": {}}
    with engine.connect() as conn:
        for r in conn.execute(text("SELECT calendar_id, period_date::TEXT FROM nhs.dim_calendar")).fetchall():
            caches["calendar"][r[1]] = r[0]
        for r in conn.execute(text("SELECT region_id, region_code FROM nhs.dim_region")).fetchall():
            caches["region"][r[1]] = r[0]
        for r in conn.execute(text("SELECT treatment_id, function_code FROM nhs.dim_treatment")).fetchall():
            caches["treatment"][r[1]] = r[0]
    return caches


def parse_xls(filepath: Path, calendar_id: int, caches: Dict) -> List[dict]:
    try:
        wb = xlrd.open_workbook(str(filepath))
    except Exception as e:
        logger.error("Cannot open %s: %s", filepath.name, e)
        return []

    sheet = find_data_sheet(wb)
    if sheet is None:
        return []

    # Scan for header row
    hdr_row = None
    col_region = col_ccg_code = col_ccg_name = col_tf_code = col_tf_name = col_count = None

    for r in range(min(25, sheet.nrows)):
        row = [cell_str(sheet, r, c).lower() for c in range(sheet.ncols)]
        if any("region code" in v or "ccg code" in v or "treatment function code" in v
               or "number of new" in v for v in row):
            hdr_row = r
            for c, v in enumerate(row):
                if "region code" in v and col_region is None:           col_region  = c
                elif "ccg code" in v and col_ccg_code is None:          col_ccg_code= c
                elif "ccg name" in v and col_ccg_name is None:          col_ccg_name= c
                elif "treatment function code" in v and col_tf_code is None: col_tf_code= c
                elif "treatment function" in v and col_tf_name is None: col_tf_name = c
                elif "number of new" in v or "clock starts" in v:       col_count   = c
            break

    if hdr_row is None or col_count is None:
        logger.warning("No usable header found in %s", filepath.name)
        return []

    records = []
    current_region = None

    for r in range(hdr_row + 1, sheet.nrows):
        # Track region context
        if col_region is not None:
            rc = cell_str(sheet, r, col_region).upper()
            if rc and rc not in ("", "NAN", "REGION CODE"):
                current_region = rc

        # Get commissioner code
        org_code = ""
        if col_ccg_code is not None:
            org_code = cell_str(sheet, r, col_ccg_code).upper()
        if not org_code or org_code in ("", "CCG CODE", "NAN"):
            org_code = current_region or ""
        if not org_code:
            continue

        # Get treatment
        tf_code_raw = cell_str(sheet, r, col_tf_code) if col_tf_code is not None else ""
        tf_name_raw = cell_str(sheet, r, col_tf_name) if col_tf_name is not None else ""
        if not tf_code_raw and not tf_name_raw:
            continue

        treat_code = resolve_treatment_code(tf_code_raw, tf_name_raw)
        if treat_code is None:
            continue

        treatment_id = caches["treatment"].get(treat_code)
        if treatment_id is None:
            continue

        # Get count
        count = clean_numeric_value(cell_str(sheet, r, col_count))

        # Region lookup
        modern_code, _ = consolidate_region(org_code)
        region_id = caches["region"].get(modern_code) or caches["region"].get(org_code)
        if region_id is None and current_region:
            mc2, _ = consolidate_region(current_region)
            region_id = caches["region"].get(mc2) or caches["region"].get(current_region)
        if region_id is None:
            continue

        ccg_name = cell_str(sheet, r, col_ccg_name) if col_ccg_name is not None else ""
        records.append({
            "calendar_id":           calendar_id,
            "region_id":             region_id,
            "treatment_id":          treatment_id,
            "commissioner_org_code": org_code,
            "commissioner_org_name": ccg_name,
            "new_periods_count":     count,
            "source_file":           filepath.name,
        })

    return records


def main(config_path: str = "config/config.yaml", dry_run: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("Step 3 — Ingest New Periods XLS (long format)")
    logger.info("=" * 60)

    cfg    = load_config(config_path)
    engine = get_engine(config_path)
    caches = load_dim_caches(engine)

    raw_dir = Path(cfg["paths"]["raw_new_periods"])
    xls_files = sorted([f for f in raw_dir.glob("*.xls") if f.name != ".gitkeep"])
    logger.info("Found %d XLS files", len(xls_files))

    total = 0
    iterator = tqdm(xls_files, desc="XLS", unit="file") if HAS_TQDM else xls_files

    for fp in iterator:
        period_date = parse_filename_to_date(fp.name)
        if period_date is None:
            logger.warning("Cannot parse date from %s", fp.name)
            continue
        cal_id = caches["calendar"].get(period_date.isoformat())
        if cal_id is None:
            logger.warning("Period %s not in dim_calendar", period_date)
            continue
        try:
            recs = parse_xls(fp, cal_id, caches)
            if recs and not dry_run:
                upsert_dataframe(recs, "fact_new_periods_monthly", engine)
            total += len(recs)
            logger.info("  %s → %d records", fp.name, len(recs))
        except Exception as e:
            logger.error("Error: %s — %s", fp.name, e, exc_info=True)

    logger.info("Total records: %d", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run)
