"""
Only module that talks to 'data/alpha_signal.db'
Any piece of code that wants to read or write the database goes through
a function in this file.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import config

import json


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

# Phase 1b

# Bridge between the extracted text in Python and it being saved in the database
def insert_filing_section(accession_number: str, section_type: str, section_text: str | None) -> None:
    '''INSERT OR REPLACE ensures that old text gets replaced, not duplicated.
    Improves re-running extraction and makes it safer.'''
    char_count = len(section_text) if section_text else 0
    extracted_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO filing_sections (
                accession_number, section_type, section_text, char_count, extracted_at
            ) VALUES (?, ?, ?, ?, ?);
            """,
            (accession_number, section_type, section_text, char_count, extracted_at),
        )
        conn.commit()
    finally:
        conn.close()

# Indicates what still needs extracting from the filings table
# What is in 'filings' but doesn't have a row in 'filing_sections'
def get_filings_needing_extraction() -> list[dict]:
    '''Returns filings missing at least one section extraction. Counts how
    many filing_sections rows exist per filing, and returns any filing under
    that full count.
    LEFT JOIN used to keep every row from filings.'''
    conn = get_connection()
    
    try:
        rows = conn.execute(
            """
            SELECT f.accession_number, f.document_url
            FROM filings f
            LEFT JOIN filing_sections fs ON f.accession_number = fs.accession_number
            GROUP BY f.accession_number, f.document_url
            HAVING COUNT(fs.section_type) < ?;
            """,
            (len(config.FILING_SECTIONS),),
        ).fetchall()
        return [{"accession_number": r[0], "document_url": r[1]} for r in rows] # converts each raw tuple into a dict
    finally:
        conn.close()


# Phase 2: Embeddings

def insert_filing_embedding(accession_number: str, section_type: str, embedding: list[float]) -> None:
    """Takes the pieces computed in Python and writes them into the database as one clean row.
    """
    embedding_json = json.dumps(embedding)  # list of floats -> a single storable string
    created_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO filing_embeddings (
                accession_number, section_type, embedding, model_name, created_at
            ) VALUES (?, ?, ?, ?, ?);
            """,
            (accession_number, section_type, embedding_json, config.EMBEDDING_MODEL, created_at),
        )
        conn.commit()  # writes are staged until commit — without this, nothing actually saves
    finally:
        conn.close()

def get_sections_needing_embedding() -> list[dict]:
    """Looks at every section extracted, identifies the ones that haven't been embedded yet"""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT fs.accession_number, fs.section_type, fs.section_text
            FROM filing_sections fs
            LEFT JOIN filing_embeddings fe
                ON fs.accession_number = fe.accession_number
               AND fs.section_type = fe.section_type
            WHERE fe.accession_number IS NULL
              AND fs.char_count >= ?;
            """,
            (config.MIN_CHARS_FOR_EMBEDDING,),
        ).fetchall()
        return [
            {"accession_number": r[0], "section_type": r[1], "section_text": r[2]}
            for r in rows
        ]
    finally:
        conn.close()
        