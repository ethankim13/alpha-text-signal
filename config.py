# Cross-industry, 2 large/mid-cap names per sector, 8 sectors. This is
# done to test whether a text-shift signal generalizes across very 
# different businesses, not just one industry's boilerplate.

# IMPORTANT consequence for later: the signal itself (embedding distance /
# sentiment delta between a company's own consecutive filings) is computed
# WITHIN a company, so mixing industries doesn't break that step. But once
# we pool everything together for modeling in Phase 3, sector needs to be
# a control feature (e.g. one-hot SIC code or sector dummy) -- a tech
# company's "normal" quarter-to-quarter risk-factor drift is not the same
# baseline as an airline's. Flagging now so it doesn't get missed later.
TICKERS = [
    "MSFT", "ADBE",  # Technology
    "TGT", "NKE",    # Retail
    "PFE", "ABBV",   # Healthcare/pharma
    "XOM", "COP",    # Energy
    "CAT", "HON",    # Industrials
    "KO", "PG",      # Consumer staples
    "DAL", "UAL",    # Airlines
    "VZ", "T",       # Telecom
]

# Filing scope
FILING_TYPE = "10-Q"              # quarterly reports, more data points over time than 10-K 
LOOKBACK_QUARTERS = 12            # ~3 years of history per ticker
FILING_SECTIONS = ["risk_factors", "mda"]   # which section we'll extract text from, NOW ADDED MD&A in Phase 1b

# Target variable
FORWARD_RETURN_DAYS = 5           # predicting 5-trading-day forward return after each filing date

# SEC EDGAR access: SEC requires a descriptive User-Agent with a real contact
SEC_USER_AGENT = "Ethan Kim (ethankim131@gmail.com)"
SEC_REQUEST_DELAY_SECONDS = 0.2   # stay comfortably under SEC's ~10 req/sec guidance

# Storage
DB_PATH = "data/alpha_signal.db"



# Phase 2: Embeddings

# I am using the embedding model all-MiniLM-L6-v2 because it runs faster on CPU.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # this model's output vector size — must match schema
MAX_TOKENS = 512 # hard limit for this model architecture

MIN_CHARS_FOR_EMBEDDING = 50


# Phase 3b: Modeling

# Sector previously existed as a comment above TICKERS list in this file
# A dictionary allows it to become real, queryable data
TICKER_SECTORS = {
    "MSFT": "Technology", "ADBE": "Technology",
    "TGT": "Retail", "NKE": "Retail",
    "PFE": "Healthcare", "ABBV": "Healthcare",
    "XOM": "Energy", "COP": "Energy",
    "CAT": "Industrials", "HON": "Industrials",
    "KO": "Consumer Staples", "PG": "Consumer Staples",
    "DAL": "Airlines", "UAL": "Airlines",
    "VZ": "Telecom", "T": "Telecom",
}