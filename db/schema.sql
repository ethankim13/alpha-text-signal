-- One row per company we're tracking.
CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    cik TEXT NOT NULL,          -- zero-padded to 10 digits, e.g. 0000320193
    name TEXT
);

-- One row per filing (10-Q metadata now; raw_text gets filled in Phase 1b).
CREATE TABLE IF NOT EXISTS filings (
    accession_number TEXT PRIMARY KEY,   -- SEC's unique ID for this filing, e.g. 0000320193-24-000123
    ticker TEXT NOT NULL,
    cik TEXT NOT NULL,
    form_type TEXT NOT NULL,             -- "10-Q" or "10-K"
    filing_date TEXT NOT NULL,           -- date it was filed with the SEC (YYYY-MM-DD)
    report_date TEXT,                    -- the period the filing actually covers (YYYY-MM-DD)
    primary_document TEXT NOT NULL,      -- filename of the main document in the filing package
    document_url TEXT NOT NULL,          -- full URL to that document
    raw_text TEXT,                       -- filled in Phase 1b once we extract filing text
    fetched_at TEXT NOT NULL,            -- when our script pulled this row (for incremental updates)
    FOREIGN KEY (ticker) REFERENCES companies (ticker)
);

-- Daily price bars, used to build the forward-return target in Phase 3.
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker) REFERENCES companies (ticker)
);


-- One row per (filing, section) pair. Populated in Phase 1b, consumed starting in Phase 2 for embeddings.
-- Where the results of the parser sit
CREATE TABLE IF NOT EXISTS filing_sections (
    accession_number TEXT NOT NULL,
    section_type TEXT NOT NULL,      -- 'risk_factors' or 'mda'
    section_text TEXT,
    char_count INTEGER,
    extracted_at TEXT NOT NULL,
    PRIMARY KEY (accession_number, section_type),
    FOREIGN KEY (accession_number) REFERENCES filings (accession_number)
);