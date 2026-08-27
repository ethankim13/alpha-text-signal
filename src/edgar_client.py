"""
Everything needed to go from a stock ticker --> SEC CIK --> list of filings
with metadata. This module does not download the actual filing document
text.

Two SEC endpoints, per SEC's own docs (sec.gov/search-filings/edgar-application-programming-interfaces):
  1. https://www.sec.gov/files/company_tickers.json
     -> maps ticker to CIK (CIK returned WITHOUT zero-padding)
  2. https://data.sec.gov/submissions/CIK{cik10}.json
     -> that company's full filing history, in a "columnar" JSON format
        (parallel arrays, not a list of row-objects — see _zip_filings below)

SEC requires a descriptive User-Agent header on every request (set in
config.py) and asks for no more than ~10 requests/second. We sleep
between requests to stay well under that.

EDGAR: name of SEC's public filing database (Electronic Data Gathering, Analysis, and Retrieval)
client: the code responsible for talking to EDGAR

"""
import json # used to convert between JSON text and Python dictionaries/lists
import time # used later in time.sleep() to pause execution between requests
from pathlib import Path

import requests # makes HTTP calls

import config

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_URL_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_no_padding}/{accession_no_dashes}/{primary_document}"

_HEADERS = {"User-Agent": config.SEC_USER_AGENT}  # SEC requires name/email for every HTTP request
_TICKER_CACHE_PATH = Path(__file__).parent.parent / "data" / "company_tickers.json" 


def _get(url: str) -> dict: # expects string input, returns dictionary
    """GET with the required SEC headers, a small delay, and a clear error
    if the User-Agent hasn't been customized (SEC will 403 a generic one)."""
    if "replace-with-your-email" in config.SEC_USER_AGENT:
        raise RuntimeError(
            "Set a real name/email in config.SEC_USER_AGENT before hitting the SEC API. "
            "They ask for this so they can contact you if a script misbehaves — "
            "requests without it get rate-limited or blocked."
        )
    resp = requests.get(url, headers=_HEADERS, timeout=15) # HTTP GET request, raise error if no response <15 seconds 
    resp.raise_for_status() # raises exception if there is an error
    time.sleep(config.SEC_REQUEST_DELAY_SECONDS) # 0.2 second pause after every request, makes it polite
    return resp.json()


def get_cik_map(force_refresh: bool = False) -> dict:
    """Returns {ticker: {"cik10": "0000320193", "name": "Apple Inc."}, ...}.
    Caches the ~10,000-entry SEC file locally so you're not re-downloading
    13MB+ every run."""
    if _TICKER_CACHE_PATH.exists() and not force_refresh:
        raw = json.loads(_TICKER_CACHE_PATH.read_text())
    else:
        raw = _get(TICKERS_URL)
        _TICKER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TICKER_CACHE_PATH.write_text(json.dumps(raw))

    cik_map = {}
    for entry in raw.values(): 
        ticker = entry["ticker"]
        cik_map[ticker] = {
            "cik10": str(entry["cik_str"]).zfill(10),
            "name": entry["title"],
        }
    return cik_map


def _zip_filings_recent(recent: dict) -> list[dict]:
    """The submissions API returns filings.recent as parallel arrays:
    {"form": [...], "filingDate": [...], "accessionNumber": [...], ...}
    where index 0 across every array describes the same filing. This zips
    them back into one dict per filing, which is what every downstream
    piece of code actually wants to work with."""
    n = len(recent.get("accessionNumber", []))
    rows = []
    for i in range(n):
        rows.append(
            {
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": recent["filingDate"][i],
                "reportDate": recent["reportDate"][i],
                "form": recent["form"][i],
                "primaryDocument": recent["primaryDocument"][i],
            }
        )
    return rows


def get_filings_for_ticker(ticker: str, cik10: str, form_type: str, limit: int) -> list[dict]:
    """Returns up to `limit` most recent filings of `form_type` for this
    ticker, newest first, each with a ready-to-download document_url."""
    data = _get(SUBMISSIONS_URL_TMPL.format(cik10=cik10))
    all_filings = _zip_filings_recent(data["filings"]["recent"])

    # NOTE: filings.recent only holds the most recent ~1000 filings (or 1
    # year, whichever is more). For our 16 mid-cap banks and 12 quarters of
    # 10-Qs that's plenty — but if you ever need older history, the SEC
    # response includes a "files" list pointing to additional paginated
    # JSON files. Flagging this now so it doesn't surprise you later.
    matching = [f for f in all_filings if f["form"] == form_type]
    matching.sort(key=lambda f: f["filingDate"], reverse=True)
    matching = matching[:limit]

    cik_no_padding = str(int(cik10))  # archive URLs use CIK without leading zeros
    for f in matching:
        accession_no_dashes = f["accessionNumber"].replace("-", "")
        f["document_url"] = ARCHIVES_URL_TMPL.format(
            cik_no_padding=cik_no_padding,
            accession_no_dashes=accession_no_dashes,
            primary_document=f["primaryDocument"],
        )
    return matching
