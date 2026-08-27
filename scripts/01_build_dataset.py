"""
  1. Initializes the SQLite DB (serverless relational database) from db/schema.sql (safe to re-run)
  2. Resolves each ticker in config.py to its SEC CIK (Central Index Key)
  3. Pulls filing metadata (10-Qs) for each ticker and stores it
  4. Pulls daily price history for each ticker and stores it

This does not download the actual filing text or build any features. 
This script's job is to prove the pipeline works end to
end and get real data sitting in your database.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

# Let this script find config.py and src/ when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import config
from src import db_utils, edgar_client, price_fetcher


def main():
    print("Initializing database...")
    db_utils.init_db()

    print("Fetching SEC ticker -> CIK map (cached after first run)...")
    cik_map = edgar_client.get_cik_map()

    price_start = (date.today() - timedelta(days=config.LOOKBACK_QUARTERS * 95)).isoformat()

    missing_tickers = []
    for ticker in tqdm(config.TICKERS, desc="Tickers"):
        if ticker not in cik_map:
            missing_tickers.append(ticker)
            continue

        cik10 = cik_map[ticker]["cik10"]
        name = cik_map[ticker]["name"]
        db_utils.upsert_company(ticker, cik10, name)

        # --- Filings ---
        filings = edgar_client.get_filings_for_ticker(
            ticker, cik10, config.FILING_TYPE, config.LOOKBACK_QUARTERS
        )
        for f in filings:
            db_utils.insert_filing(
                accession_number=f["accessionNumber"], # unique tracking code for each filing
                ticker=ticker,
                cik=cik10,
                form_type=f["form"],
                filing_date=f["filingDate"],
                report_date=f["reportDate"],
                primary_document=f["primaryDocument"],
                document_url=f["document_url"],
            )

        # --- Prices ---
        price_rows = price_fetcher.fetch_price_history(ticker, start=price_start)
        if price_rows:
            db_utils.insert_prices(ticker, price_rows)

    if missing_tickers:
        print(f"\nWARNING: couldn't resolve CIK for: {missing_tickers} — check spelling in config.py")

    print("\nDone. Database contents:")
    for k, v in db_utils.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
