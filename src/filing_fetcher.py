'''
Fetches the actual 10-Q document from SEC onto my local disk
- Once per filing, meaning everything downstream works off local
files rather than hitting the network repeatedly.
- Reads from 'filings' table that has the document_url for all filings
- Writes files to 'data/raw_filings/'
- Sibling to 'edgar-client.py' (both talk to SEC)

Difference from 'edgar_client.py' is the target.
Target of this: get filing documents, not JSON metadata 
- Need to parse before it is useful
'''
import time # used later in time.sleep() to pause execution between requests
from pathlib import Path

import requests
import config

_HEADERS = {"User-Agent": config.SEC_USER_AGENT}

# Build file path
def get_cached_path(accession_number: str) -> Path:
    return Path("data/raw_filings") / f"{accession_number}.html"

def fetch_and_cache_filing(accession_number: str, document_url: str) -> Path:
    cache_path = get_cached_path(accession_number)

    if cache_path.is_file():
        return cache_path
    resp = requests.get(document_url, headers = _HEADERS, timeout = 15) # HTTP GET request, raise error if no response <15 seconds
    resp.raise_for_status() # raise an exception on a bad HTTP status

    cache_path.parent.mkdir(parents=True, exist_ok=True) # ensure data/raw_filings/ exists
    cache_path.write_text(resp.text) # for caching, don't have to re-download from scratch every time

    time.sleep(config.SEC_REQUEST_DELAY_SECONDS) # only reached after a real network call

    return cache_path
        
