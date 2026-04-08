"""
02_ingest_rtt_csv.py — Ingest all RTT waiting time CSV files into fact_rtt_monthly.

Scans data/raw/rtt_csv/ for .csv and .zip files, processes each file in chunks
of 5,000 rows, maps to dimension IDs, and upserts into the fact table.

Expected files: RTT-{MONTH}-{YEAR}-full-extract.csv (one per month, 2019–2025)
Expected rows:  ~202,000 per file, ~15M total

Usage:
    python scripts/02_ingest_rtt_csv.py [--config ...] [--file single_file.csv] [--dry-run]

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.data_helpers import (
    REGION_CONSOLIDATION,
    TREATMENT_CODES,
    clean_numeric_value,
    clean_string_value,
    compute_18wk_metrics,
    consolidate_region,
    get_db_week_columns,
    get_raw_week_columns,
    map_raw_to_db_columns,
    parse_period_to_date,
)
from scripts.utils.db_connect import get_dim_id, get_engine, load_config, upsert_dataframe

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_COL_MAP   = map_raw_to_db_columns()
DB_WEEK_COLS  = get_db_week_columns()
CHUNK_SIZE    = 5_000

# Known RTT Part Type → description mapping
RTT_PART_MAP = {
    "Part_1A": "Completed Pathways For Admitted Patients",
    "Part_1B": "Completed Pathways For Non-Admitted Patients",
    "Part_2":  "Incomplete Pathways",
    "Part_2A": "Incomplete Pathways Adjusted",
    "Part_3":  "New Periods Starting",
}


# ── Dimension ID caches (loaded once, reused) ─────────────────────────────────

def load_dim_caches(engine) -> Dict[str, Dict[str, int]]:
    """Load all dimension IDs into memory to avoid per-row DB lookups."""
    from sqlalchemy import text

    caches: Dict[str, Dict[str, int]] = {
        "calendar":  {},   # period_date_str → calendar_id
        "region":    {},   # region_code     → region_id
        "treatment": {},   # function_code   → treatment_id
        "rtt_part":  {},   # part_type       → part_id
    }

    with engine.connect() as conn:
        # Calendar
        rows = conn.execute(
            text("SELECT calendar_id, period_date::TEXT FROM nhs.dim_calendar")
        ).fetchall()
        for r in rows:
            caches["calendar"][r[1]] = r[0]

        # Region — map all known codes
        rows = conn.execute(
            text("SELECT region_id, region_code FROM nhs.dim_region")
        ).fetchall()
        for r in rows:
            caches["region"][r[1]] = r[0]

        # Treatment
        rows = conn.execute(
            text("SELECT treatment_id, function_code FROM nhs.dim_treatment")
        ).fetchall()
        for r in rows:
            caches["treatment"][r[1]] = r[0]

        # RTT Part
        rows = conn.execute(
            text("SELECT part_id, part_type FROM nhs.dim_rtt_part")
        ).fetchall()
        for r in rows:
            caches["rtt_part"][r[1]] = r[0]

    logger.info(
        "Dim caches loaded: %d calendar, %d region, %d treatment, %d rtt_part",
        len(caches["calendar"]), len(caches["region"]),
        len(caches["treatment"]), len(caches["rtt_part"]),
    )
    return caches


def get_or_insert_region(
    engine, caches: Dict, region_code: str, region_name: str
) -> Optional[int]:
    """Look up region_id, inserting a new row if the code is unseen."""
    modern_code, modern_name = consolidate_region(region_code)

    # Try modern code first, then raw code
    if modern_code in caches["region"]:
        return caches["region"][modern_code]
    if region_code in caches["region"]:
        return caches["region"][region_code]

    # Insert new region
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    "INSERT INTO nhs.dim_region "
                    "(region_code, region_name, modern_region_code, modern_region_name, is_modern_region) "
                    "VALUES (:rc, :rn, :mc, :mn, :im) "
                    "ON CONFLICT (region_code) DO UPDATE SET region_name = EXCLUDED.region_name "
                    "RETURNING region_id"
                ),
                {
                    "rc": region_code,
                    "rn": region_name or region_code,
                    "mc": modern_code,
                    "mn": modern_name,
                    "im": modern_code == region_code,
                },
            )
            row = result.fetchone()
            if row:
                rid = row[0]
                caches["region"][region_code] = rid
                caches["region"][modern_code] = rid
                logger.debug("Inserted new region: %s → id %d", region_code, rid)
                return rid
    except Exception as exc:
        logger.warning("Could not insert region %s: %s", region_code, exc)
    return None


# ── File discovery ────────────────────────────────────────────────────────────

def find_csv_files(data_dir: Path) -> List[Path]:
    """Find all .csv and .zip files in the RTT raw data directory."""
    files = []
    for pattern in ("*.csv", "*.CSV", "*.zip", "*.ZIP"):
        files.extend(data_dir.glob(pattern))
    files = sorted(set(files))
    # Exclude placeholder
    files = [f for f in files if f.name != ".gitkeep"]
    logger.info("Found %d files in %s", len(files), data_dir)
    return files


def extract_zip(zip_path: Path, extract_to: Path) -> List[Path]:
    """Extract a zip file and return list of extracted CSV paths."""
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                zf.extract(name, extract_to)
                extracted.append(extract_to / name)
    return extracted


# ── Single-file processing ────────────────────────────────────────────────────

def process_file(
    filepath: Path,
    engine,
    caches: Dict,
    source_name: str,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Process one RTT CSV file and upsert into fact_rtt_monthly.

    Returns:
        Dict with keys: rows_read, rows_inserted, rows_skipped, errors.
    """
    stats = {"rows_read": 0, "rows_inserted": 0, "rows_skipped": 0, "errors": 0}

    # ── Detect period date from first row ──────────────────────────────────────
    try:
        header_df = pd.read_csv(
            filepath, nrows=1, dtype=str, encoding="utf-8", encoding_errors="replace"
        )
    except Exception as exc:
        logger.error("Cannot read %s: %s", filepath.name, exc)
        stats["errors"] += 1
        return stats

    first_period = header_df.iloc[0].get("Period", None)
    if first_period is None:
        # Try stripping quotes from column names
        header_df.columns = [c.strip().strip('"') for c in header_df.columns]
        first_period = header_df.iloc[0].get("Period", None)

    if first_period is None:
        logger.error("Cannot find 'Period' column in %s", filepath.name)
        stats["errors"] += 1
        return stats

    period_date = parse_period_to_date(str(first_period))
    if period_date is None:
        logger.error("Cannot parse period '%s' from %s", first_period, filepath.name)
        stats["errors"] += 1
        return stats

    calendar_id = caches["calendar"].get(period_date.isoformat())
    if calendar_id is None:
        logger.warning(
            "Period %s not found in dim_calendar — skipping file %s",
            period_date, filepath.name
        )
        stats["errors"] += 1
        return stats

    logger.info("Processing %s → period %s", filepath.name, period_date)

    # ── Chunked reading ────────────────────────────────────────────────────────
    try:
        reader = pd.read_csv(
            filepath,
            dtype=str,
            chunksize=CHUNK_SIZE,
            encoding="utf-8",
            encoding_errors="replace",
        )
    except Exception as exc:
        logger.error("Cannot open CSV %s: %s", filepath.name, exc)
        stats["errors"] += 1
        return stats

    for chunk_df in reader:
        # Normalise column names
        chunk_df.columns = [c.strip().strip('"') for c in chunk_df.columns]

        batch_records: List[dict] = []

        for _, row in chunk_df.iterrows():
            stats["rows_read"] += 1
            try:
                record = _build_record(row, calendar_id, caches, engine, source_name)
                if record is None:
                    stats["rows_skipped"] += 1
                    continue
                batch_records.append(record)
            except Exception as exc:
                logger.debug("Row error in %s: %s", filepath.name, exc)
                stats["errors"] += 1

        if batch_records and not dry_run:
            inserted = upsert_dataframe(batch_records, "fact_rtt_monthly", engine)
            stats["rows_inserted"] += inserted
        elif batch_records:
            stats["rows_inserted"] += len(batch_records)  # dry-run count

    return stats


def _build_record(
    row: pd.Series,
    calendar_id: int,
    caches: Dict,
    engine,
    source_name: str,
) -> Optional[dict]:
    """Build a single fact_rtt_monthly record dict from a CSV row."""

    # ── Treatment function ─────────────────────────────────────────────────────
    func_code = clean_string_value(row.get("Treatment Function Code"))
    if func_code not in TREATMENT_CODES:
        return None  # Skip unknown specialty codes
    treatment_id = caches["treatment"].get(func_code)
    if treatment_id is None:
        return None

    # ── RTT Part Type ─────────────────────────────────────────────────────────
    part_type = clean_string_value(row.get("RTT Part Type"))
    if not part_type or part_type not in RTT_PART_MAP:
        return None
    part_id = caches["rtt_part"].get(part_type)
    if part_id is None:
        return None

    # ── Region ────────────────────────────────────────────────────────────────
    provider_parent_code = clean_string_value(row.get("Provider Parent Org Code"))
    provider_parent_name = clean_string_value(row.get("Provider Parent Name"))
    region_id = get_or_insert_region(
        engine, caches, provider_parent_code or "", provider_parent_name or ""
    )
    if region_id is None:
        return None

    # ── Weekly bucket columns ─────────────────────────────────────────────────
    weekly: Dict[str, Optional[int]] = {}
    for raw_col, db_col in RAW_COL_MAP.items():
        weekly[db_col] = clean_numeric_value(row.get(raw_col))

    # ── Totals ────────────────────────────────────────────────────────────────
    total_patients         = clean_numeric_value(row.get("Total"))
    patients_unknown_start = clean_numeric_value(row.get("Patients with unknown clock start date"))
    total_all              = clean_numeric_value(row.get("Total All"))

    # Use total_all for 18-week metric denominator; fall back to total_patients
    metrics_input = {**weekly, "total_all": total_all or total_patients or 0}
    wk_metrics    = compute_18wk_metrics(metrics_input)

    # ── Assemble record ───────────────────────────────────────────────────────
    record: Dict = {
        "calendar_id":             calendar_id,
        "region_id":               region_id,
        "treatment_id":            treatment_id,
        "part_id":                 part_id,
        "provider_org_code":       clean_string_value(row.get("Provider Org Code")),
        "provider_org_name":       clean_string_value(row.get("Provider Org Name")),
        "commissioner_org_code":   clean_string_value(row.get("Commissioner Org Code")),
        "commissioner_org_name":   clean_string_value(row.get("Commissioner Org Name")),
        "total_patients":          total_patients,
        "patients_unknown_start":  patients_unknown_start,
        "total_all":               total_all,
        "patients_within_18wks":   wk_metrics["patients_within_18wks"],
        "patients_over_18wks":     wk_metrics["patients_over_18wks"],
        "pct_within_18wks":        wk_metrics["pct_within_18wks"],
        "source_file":             source_name,
    }
    record.update(weekly)
    return record


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str = "config/config.yaml",
         data_dir: Optional[str] = None,
         single_file: Optional[str] = None,
         dry_run: bool = False) -> None:

    logger.info("=" * 60)
    logger.info("Step 2 — Ingest RTT CSV files")
    if dry_run:
        logger.info("DRY RUN — no data will be written")
    logger.info("=" * 60)

    cfg    = load_config(config_path)
    engine = get_engine(config_path)
    caches = load_dim_caches(engine)

    raw_dir = Path(data_dir or cfg["paths"]["raw_rtt"])
    if not raw_dir.exists():
        logger.error("Raw RTT directory not found: %s", raw_dir.resolve())
        sys.exit(1)

    # Collect files to process
    if single_file:
        files = [Path(single_file)]
    else:
        all_files = find_csv_files(raw_dir)
        # Extract any ZIPs to temp
        import tempfile
        tmp_dir   = Path(tempfile.mkdtemp())
        csv_files = []
        for f in all_files:
            if f.suffix.lower() == ".zip":
                extracted = extract_zip(f, tmp_dir)
                csv_files.extend(extracted)
                logger.info("Extracted %s → %d CSV(s)", f.name, len(extracted))
            else:
                csv_files.append(f)
        files = csv_files

    if not files:
        logger.warning("No CSV files found in %s", raw_dir)
        return

    # Process all files
    totals = {"rows_read": 0, "rows_inserted": 0, "rows_skipped": 0, "errors": 0}

    for filepath in tqdm(files, desc="RTT files", unit="file"):
        try:
            stats = process_file(filepath, engine, caches, filepath.name, dry_run)
            for k in totals:
                totals[k] += stats[k]
            logger.info(
                "  %s: read=%d inserted=%d skipped=%d errors=%d",
                filepath.name, stats["rows_read"], stats["rows_inserted"],
                stats["rows_skipped"], stats["errors"],
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", filepath.name, exc, exc_info=True)
            totals["errors"] += 1

    # Summary
    logger.info("=" * 60)
    logger.info("Ingestion complete.")
    logger.info("  Total rows read:     %d", totals["rows_read"])
    logger.info("  Total rows inserted: %d", totals["rows_inserted"])
    logger.info("  Total rows skipped:  %d", totals["rows_skipped"])
    logger.info("  Total errors:        %d", totals["errors"])

    # Write summary CSV
    import csv
    summary_path = Path(cfg["paths"]["processed"]) / "ingestion_log_rtt.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(totals.keys()))
        writer.writeheader()
        writer.writerow(totals)
    logger.info("Ingestion summary written to %s", summary_path)
    logger.info("Next step: python scripts/03_ingest_new_periods_xls.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest RTT CSV files into PostgreSQL")
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--data-dir", default=None, help="Override raw_rtt path")
    parser.add_argument("--file",     default=None, help="Process a single file")
    parser.add_argument("--dry-run",  action="store_true", help="Parse but do not write to DB")
    args = parser.parse_args()
    main(
        config_path=args.config,
        data_dir=args.data_dir,
        single_file=args.file,
        dry_run=args.dry_run,
    )
