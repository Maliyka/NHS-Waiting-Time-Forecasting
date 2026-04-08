"""
db_connect.py — Database connection utilities.

Provides SQLAlchemy engine creation, raw psycopg2 helpers, and batch upsert
for the NHS Waiting Time Forecasting project.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import psycopg2
import psycopg2.extras
import yaml
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and return the project YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}. "
            "Run from the project root directory."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.debug("Config loaded from %s", path)
    return cfg


# ── Engine factory ─────────────────────────────────────────────────────────────

def get_engine(config_path: str = "config/config.yaml") -> Engine:
    """
    Create and return a SQLAlchemy engine connected to the NHS PostgreSQL database.

    The engine sets search_path=nhs on every connection so all queries
    resolve to the correct schema without needing 'nhs.' prefixes.

    Args:
        config_path: Path to config.yaml relative to project root.

    Returns:
        SQLAlchemy Engine with connection pooling configured.
    """
    cfg = load_config(config_path)
    db  = cfg["database"]

    # Support environment variable overrides
    host     = os.environ.get("DB_HOST",     db["host"])
    port     = int(os.environ.get("DB_PORT", db["port"]))
    name     = os.environ.get("DB_NAME",     db["name"])
    user     = os.environ.get("DB_USER",     db["user"])
    password = os.environ.get("DB_PASSWORD", db["password"])
    schema   = os.environ.get("DB_SCHEMA",   db.get("schema", "nhs"))

    url = (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{host}:{port}/{name}"
    )

    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,          # reconnect if connection dropped
        connect_args={
            "options": f"-csearch_path={schema},public",
            "connect_timeout": 10,
        },
        echo=False,
    )

    # Verify connectivity on first checkout
    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_conn, _conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute(f"SET search_path TO {schema}, public")
        cursor.close()

    logger.info(
        "Database engine created: %s@%s:%s/%s (schema=%s)",
        user, host, port, name, schema
    )
    return engine


# ── Context managers ───────────────────────────────────────────────────────────

@contextmanager
def get_connection(engine: Engine) -> Generator:
    """
    Context manager yielding a raw psycopg2 connection from the engine pool.

    Usage:
        with get_connection(engine) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    """
    raw_conn = engine.raw_connection()
    try:
        yield raw_conn
        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise
    finally:
        raw_conn.close()


@contextmanager
def get_sa_connection(engine: Engine) -> Generator:
    """Context manager for a SQLAlchemy connection (for pd.read_sql etc.)."""
    with engine.connect() as conn:
        yield conn


# ── SQL file execution ─────────────────────────────────────────────────────────

def execute_sql_file(engine: Engine, filepath: str) -> None:
    """
    Read a .sql file and execute it against the database.

    Args:
        engine:   SQLAlchemy engine.
        filepath: Path to the .sql file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path.resolve()}")

    sql = path.read_text(encoding="utf-8")
    logger.info("Executing SQL file: %s", path.name)

    with engine.begin() as conn:
        # Execute as a single block to support multi-statement files
        conn.execute(text(sql))

    logger.info("SQL file executed successfully: %s", path.name)


# ── Schema introspection helpers ───────────────────────────────────────────────

def table_exists(engine: Engine, table_name: str, schema: str = "nhs") -> bool:
    """Return True if the table exists in the given schema."""
    sql = text(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = :schema AND table_name = :table"
    )
    with engine.connect() as conn:
        result = conn.execute(sql, {"schema": schema, "table": table_name})
        return result.scalar() > 0


def get_row_count(engine: Engine, table_name: str, schema: str = "nhs") -> int:
    """Return the row count of a table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{table_name}"))
        return result.scalar()


def get_dim_id(
    engine: Engine,
    table: str,
    code_col: str,
    code_val: str,
    schema: str = "nhs",
) -> Optional[int]:
    """
    Look up a surrogate key from a dimension table by its natural key.

    Example:
        region_id = get_dim_id(engine, 'dim_region', 'region_code', 'Q71')

    Returns:
        Integer surrogate key, or None if not found.
    """
    id_col = table.replace("dim_", "") + "_id"
    sql    = text(
        f"SELECT {id_col} FROM {schema}.{table} WHERE {code_col} = :val LIMIT 1"
    )
    with engine.connect() as conn:
        result = conn.execute(sql, {"val": code_val})
        row = result.fetchone()
    return row[0] if row else None


# ── Batch upsert ───────────────────────────────────────────────────────────────

def upsert_dataframe(
    records: List[Dict[str, Any]],
    table_name: str,
    engine: Engine,
    schema: str = "nhs",
    batch_size: int = 2000,
) -> int:
    """
    Batch-insert a list of dicts into a table using INSERT ... ON CONFLICT DO NOTHING.

    This is idempotent — re-running the same data will not create duplicates.
    The UNIQUE constraint on the table defines what counts as a duplicate.

    Args:
        records:    List of dicts, each dict is one row. All dicts must have
                    the same keys (corresponding to column names).
        table_name: Target table name (without schema prefix).
        engine:     SQLAlchemy engine.
        schema:     PostgreSQL schema name.
        batch_size: Number of rows per INSERT batch.

    Returns:
        Total number of rows processed.
    """
    if not records:
        logger.warning("upsert_dataframe called with empty records list")
        return 0

    cols        = list(records[0].keys())
    col_str     = ", ".join(cols)
    placeholder = ", ".join([f"%({c})s" for c in cols])
    sql         = (
        f"INSERT INTO {schema}.{table_name} ({col_str}) "
        f"VALUES ({placeholder}) "
        f"ON CONFLICT DO NOTHING"
    )

    total = 0
    with get_connection(engine) as conn:
        with conn.cursor() as cur:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                psycopg2.extras.execute_batch(cur, sql, batch, page_size=batch_size)
                total += len(batch)
                logger.debug(
                    "Upserted batch %d-%d into %s.%s",
                    i, i + len(batch), schema, table_name
                )

    logger.info("upsert_dataframe: %d rows processed into %s.%s", total, schema, table_name)
    return total


def bulk_copy_from_csv(
    csv_path: str,
    table_name: str,
    engine: Engine,
    columns: List[str],
    schema: str = "nhs",
    delimiter: str = ",",
    null_string: str = "",
) -> None:
    """
    Ultra-fast bulk load using PostgreSQL COPY FROM (fastest possible insert method).

    Args:
        csv_path:   Absolute path to the CSV file.
        table_name: Target table name.
        engine:     SQLAlchemy engine.
        columns:    List of column names in CSV order.
        schema:     Schema name.
        delimiter:  CSV delimiter character.
        null_string:String that represents NULL in the CSV.
    """
    col_str = ", ".join(columns)
    sql     = (
        f"COPY {schema}.{table_name} ({col_str}) "
        f"FROM STDIN WITH (FORMAT CSV, DELIMITER '{delimiter}', "
        f"NULL '{null_string}', HEADER FALSE)"
    )
    with get_connection(engine) as conn:
        with conn.cursor() as cur:
            with open(csv_path, "r", encoding="utf-8") as f:
                cur.copy_expert(sql, f)
    logger.info("COPY FROM completed: %s → %s.%s", csv_path, schema, table_name)
