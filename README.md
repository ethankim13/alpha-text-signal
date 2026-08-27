# Alpha Text Signal — Phase 0 + Phase 1 (metadata + prices)

Testing whether quarter-over-quarter language shifts in bank 10-Q filings carry any predictive information about short-term stock moves (price change over 5 days) built with proper time-aware evaluation, not a single train/test split.

I use SEC's structured JSON API for filing metadata, then fetch and parse the raw HTML documents directly since SEC doesn't expose section-level content as structured data.

## What's built so far

- `config.py` — every scope decision (tickers, filing type, target
  variable) in one place. **Read this file first.**
- `db/schema.sql` — the database structure: `companies`, `filings`, `prices`
- `src/edgar_client.py` — resolves tickers to SEC CIKs, pulls 10-Q filing
  metadata (dates, accession numbers, document URLs)
- `src/price_fetcher.py` — pulls daily price history via yfinance
- `src/db_utils.py` — all database reads/writes, re-runnable without
  duplicating data
- `scripts/01_build_dataset.py` — runs the whole thing end to end
- 'alpha_signal.db'
  - SQLite Database created by my own pipeline
1. config.py sets DB_PATH = "data/alpha_signal.db"
2. db_utils.init_db() creates it using db/schema.sql, where `companies`, `filings`, `prices` tables are defined
3. edgar_client.py pulls filing metadata from SEC EDGAR, and price_fetcher.py pulls prices from Yahoo Finance, and both get written into here.
  - Basically the accumulated output of my own script

**What this does NOT do yet:** download the actual text of each filing,
extract the Risk Factors section, generate embeddings, or train anything.
That's Phase 1b and Phase 2 — next session. Right now the goal is: prove
the pipeline works and get real filings + price data sitting in your
database.

## Setup

```bash
cd alpha-text-signal
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then open `config.py` and replace the placeholder email in
`SEC_USER_AGENT` with your real name/email. The SEC requires this on every
request — it's how they'd contact you if a script misbehaved. Requests
without a real one get blocked.

## Run it

```bash
python scripts/01_build_dataset.py
```

First run will take a few minutes (16 tickers × ~12 filings each, plus
~3 years of daily prices, with a small delay between SEC requests to stay
polite to their API). Re-running it is fast and safe — already-fetched
filings are skipped, not re-downloaded.

## Check what landed in the database

```bash
sqlite3 data/alpha_signal.db "SELECT ticker, form_type, filing_date FROM filings ORDER BY ticker, filing_date DESC LIMIT 20;"
sqlite3 data/alpha_signal.db "SELECT ticker, COUNT(*) FROM prices GROUP BY ticker;"
```

If `sqlite3` isn't installed as a CLI tool, open the `.db` file with
[DB Browser for SQLite](https://sqlitebrowser.org/) (free, GUI) instead.

## Why this scope (read before changing tickers/sector)

16 tickers across 8 sectors (tech, retail, healthcare, energy,
industrials, consumer staples, airlines, telecom) — 2 companies each. This
tests whether a text-shift signal generalizes across very different
businesses, rather than working only because of one industry's specific
boilerplate.

The signal itself (embedding distance / sentiment delta between a
company's own consecutive filings) is computed *within* a company, so
mixing industries doesn't break that. But it does mean **sector needs to
become a control feature once we start modeling in Phase 3** — a tech
company's normal quarter-to-quarter drift isn't the same baseline as an
airline's. This is flagged in `config.py` too so it doesn't get lost.

If you swap tickers, keep one thing in mind regardless of sector: a
company whose Risk Factors boilerplate barely changes quarter to quarter
won't give the NLP piece anything to detect.

## What's next (Phase 1b, next session)

1. Download the actual filing document from each `document_url` already
   in your `filings` table
2. Parse out the Risk Factors / MD&A section specifically (HTML parsing —
   the annoying but necessary part)
3. Store the extracted text back into `filings.raw_text`

Once that's done, Phase 2 is embeddings + similarity scoring between a
company's consecutive filings, which is where this starts becoming an
actual signal instead of a data pipeline.
