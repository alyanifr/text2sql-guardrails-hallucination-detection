import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

CSV_PATH = Path("data/ai_job_market_insights.csv")
DB_PATH = Path("job_insights.db")
TABLE_NAME = "job_insights"

def setup_database(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH, table_name: str = TABLE_NAME, echo: bool = False) -> Engine:
    """Load the CSV into sqlite and return a read-only engine.
    
    Table is built with a short-lived writable connection that is
    closed before returning. the engine handed back opens the database
    with mode=ro, so even if a malicious query slips past the string-level
    guardrails, sqlite will refuse to write.
    """
    csv_path, db_path = Path(csv_path), Path(db_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), csv_path)

    # Writabel engine, used only to build table, then disposed
    writer = create_engine(f"sqlite+pysqlite:///{db_path}")
    try:
        with writer.begin() as conn:
            df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)
    finally:
        writer.dispose()
    logger.info("Table '%s' craeted in %s", table_name, db_path)

    # Read-only engine for agents
    reader = create_engine(f"sqlite+pysqlite://file:{db_path.resolve()}?mode=ro&uri=true", echo=echo)

    # Sanity checks: Table is readable and writes are rejected
    with reader.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM '{table_name}'")).scalar_one()
        logger.info("Verified %d rows readable via read-only engine", count)
        try:
            conn.execute(text(f"DELETE FROM '{table_name}'"))
        except Exception:
            pass        # expected: attempt to write a read-only database
        else:
            reader.dispose()
            raise RuntimeError("Read-only engine accepted a write; refusing to continue.")

    return reader

    