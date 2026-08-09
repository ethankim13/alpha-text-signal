"""
Thin wrapper around sqlite3. No ORM — at this scale (a few thousand rows,
one user) plain SQL is easier to reason about and easier to explain in an
interview than dragging in SQLAlchemy for no reason.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import config


def get_connection() -> sqlite3.Connection:
    """Open a connection to the project DB, creating the parent folder if needed."""
    db_path = Path(config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Run schema.sql against the DB. Safe to call repeatedly (CREATE TABLE IF NOT EXISTS)."""
    schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
    schema_sql = schema_path.read_text()
    conn = get_connection()
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def upsert_company(ticker: str, cik: str, name: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO companies (ticker, cik, name)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET cik = excluded.cik, name = excluded.name;
            """,
            (ticker, cik, name),
        )
        conn.commit()
    finally:
        conn.close()


def filing_exists(accession_number: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM filings WHERE accession_number = ?;", (accession_number,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def insert_filing(
    accession_number: str,
    ticker: str,
    cik: str,
    form_type: str,
    filing_date: str,
    report_date: str,
    primary_document: str,
    document_url: str,
) -> None:
    """Idempotent insert — if this filing is already in the DB (by accession
    number), do nothing. This is what makes re-running the pipeline safe:
    you only ever fetch and store NEW filings, not everything again."""
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO filings (
                accession_number, ticker, cik, form_type, filing_date,
                report_date, primary_document, document_url, raw_text, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?);
            """,
            (
                accession_number,
                ticker,
                cik,
                form_type,
                filing_date,
                report_date,
                primary_document,
                document_url,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def insert_prices(ticker: str, rows: list[tuple]) -> None:
    """rows: list of (date, open, high, low, close, adj_close, volume) tuples."""
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [(ticker, *row) for row in rows],
        )
        conn.commit()
    finally:
        conn.close()


def summary() -> dict:
    """Quick counts, used by the CLI scripts to report what's in the DB."""
    conn = get_connection()
    try:
        companies = conn.execute("SELECT COUNT(*) FROM companies;").fetchone()[0]
        filings = conn.execute("SELECT COUNT(*) FROM filings;").fetchone()[0]
        price_rows = conn.execute("SELECT COUNT(*) FROM prices;").fetchone()[0]
        return {"companies": companies, "filings": filings, "price_rows": price_rows}
    finally:
        conn.close()
