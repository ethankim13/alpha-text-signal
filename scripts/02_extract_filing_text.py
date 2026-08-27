"""
What it does, in order:
1. Initializes the DB (picks up the filing_sections table if not already there)
2. Finds filings still missing one or both section extractions
3. For each: downloads/caches the raw HTML, strips it to plain text,
extracts risk_factors and mda, writes both to filing_sections
4. Prints a summary at the end
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import config
from src import db_utils, filing_fetcher, filing_parser


def main():
    print("Initializing database...")
    db_utils.init_db()

    filings = db_utils.get_filings_needing_extraction()
    print(f"Found {len(filings)} filings needing extraction.\n")

    extracted_counts = {section: 0 for section in config.FILING_SECTIONS}
    failed_counts = {section: 0 for section in config.FILING_SECTIONS}
    fetch_failures = []

    for filing in tqdm(filings, desc="Filings"):
        accession_number = filing["accession_number"]
        document_url = filing["document_url"]

        try:
            html_path = filing_fetcher.fetch_and_cache_filing(accession_number, document_url)
            raw_html = html_path.read_text()
        except Exception as e:
            # One bad filing (network error, bad URL, etc.) shouldn't kill
            # the whole batch — record it and move on.
            fetch_failures.append((accession_number, str(e)))
            continue

        plain_text = filing_parser.strip_html(raw_html)

        for section_type in config.FILING_SECTIONS:
            section_text = filing_parser.extract_section(plain_text, section_type)
            db_utils.insert_filing_section(accession_number, section_type, section_text)

            if section_text:
                extracted_counts[section_type] += 1
            else:
                failed_counts[section_type] += 1

    print("\nDone.")
    print(f"Filings processed: {len(filings) - len(fetch_failures)}/{len(filings)}")
    for section_type in config.FILING_SECTIONS:
        print(f"  {section_type}: {extracted_counts[section_type]} extracted, {failed_counts[section_type]} empty/not found")

    if fetch_failures:
        print(f"\n{len(fetch_failures)} filings failed to fetch:")
        for accession_number, error in fetch_failures:
            print(f"  {accession_number}: {error}")


if __name__ == "__main__":
    main()