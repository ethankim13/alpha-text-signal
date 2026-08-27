# Alpha Text Signal
 
Testing whether quarter-over-quarter language drift in 10-Q filings — specifically the Risk Factors (Item 1A) and MD&A (Item 2) sections — carries any predictive information about short-term stock returns (5-day forward return following each filing).
 
This is a research project, not a product: the deliverable is a finding (does the signal have predictive edge or not), evaluated with a time-aware, walk-forward backtest rather than a single train/test split. A null result is still a legitimate outcome here, as long as the methodology is sound.
 
**Scope:** 16 tickers across 8 sectors (tech, retail, healthcare, energy, industrials, consumer staples, airlines, telecom), ~12 quarters of 10-Q history each — roughly 150-200 filing-events once both filing text and 5-day forward returns are available for a given filing date.
 
## Data sourcing
 
Filing metadata comes from SEC EDGAR's structured JSON API (ticker → CIK resolution, then per-company filing history). The actual filing text does not come from that API — SEC doesn't expose section-level content as structured data, so I fetch the raw HTML document SEC has on file for each filing and parse it directly. That second part is closer to document parsing/light scraping than API use, and I treat it that way rather than overstating it as "using an API" end to end. Price data comes from Yahoo Finance via `yfinance`.
 
## What's built
 
**Phase 0 / 1a — metadata + price pipeline**
- `config.py` — single source of truth for every scope decision: tickers, filing type, lookback window, target variable definition, sections to extract
- `db/schema.sql` — SQLite schema: `companies`, `filings`, `prices`, `filing_sections`
- `src/edgar_client.py` — resolves tickers to SEC CIKs, pulls 10-Q filing metadata (dates, accession numbers, document URLs)
- `src/price_fetcher.py` — pulls daily price history via `yfinance`
- `src/db_utils.py` — all database reads/writes; every insert is idempotent (`INSERT OR IGNORE` / `INSERT OR REPLACE` / `ON CONFLICT`), so the whole pipeline is safe to re-run without duplicating data
- `scripts/01_build_dataset.py` — orchestrates the above end to end
**Phase 1b — filing text extraction**
- `src/filing_fetcher.py` — downloads each filing's raw HTML from its `document_url` and caches it locally, so re-running the pipeline never re-hits SEC's servers for a filing already on disk
- `src/filing_parser.py` — strips HTML down to plain text (BeautifulSoup) and extracts the Risk Factors and MD&A sections specifically, using boundary-pattern matching that accounts for Table-of-Contents false positives (taking the last heading match in the document rather than the first)
- `scripts/02_extract_filing_text.py` — orchestrates fetch + parse + store for every filing, incrementally (only processes filings missing an extraction, so it's re-runnable and resilient to the occasional SEC timeout)
Extracted text is stored in a dedicated `filing_sections` table (one row per filing × section type), not bolted onto the `filings` table — this keeps Risk Factors and MD&A as independently queryable/analyzable text, which matters for Phase 3 (see below).
 
**Not yet built:** embeddings, drift scoring, or any modeling. That's Phase 2 and Phase 3.
 
## A finding worth flagging now, from the extraction step itself
 
Risk Factors extraction produced a clearly bimodal result: roughly 90% of filings had a substantial Risk Factors section (60K-70K+ characters), while the rest came back at just a couple of characters — a page number left over from a Table-of-Contents reference, nothing else. This lines up with an actual SEC rule: a 10-Q is only required to restate Risk Factors if there's been a *material change* since the company's last 10-K. Most quarters, most companies, simply don't restate it.
 
That's a real methodological consideration for the project, not a parsing bug: it means Risk Factors is a much sparser signal than MD&A, which every filing discusses substantively every quarter regardless. I'm not filtering this out yet — I'm keeping `char_count` on every extracted section so Phase 3 can treat "no material change" as its own informative case (or exclude it from the Risk Factors analysis specifically) rather than silently diluting the signal.
 
## Setup
 
```bash
cd alpha-text-signal
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
Open `config.py` and replace the placeholder email in `SEC_USER_AGENT` with a real name/email — SEC requires this on every request so they can contact you if a script misbehaves; requests without one get rate-limited or blocked.
 
## Running it
 
```bash
python scripts/01_build_dataset.py         # metadata + prices
python scripts/02_extract_filing_text.py   # filing text extraction
```
 
Both are safe to re-run — already-processed filings/sections are skipped, not redone.
 
## Checking what's in the database
 
```bash
sqlite3 data/alpha_signal.db "SELECT ticker, form_type, filing_date FROM filings ORDER BY ticker, filing_date DESC LIMIT 20;"
sqlite3 data/alpha_signal.db "SELECT section_type, COUNT(*), AVG(char_count) FROM filing_sections GROUP BY section_type;"
```
 
(If `sqlite3` isn't installed as a CLI tool, [DB Browser for SQLite](https://sqlitebrowser.org/) is a free GUI alternative — never open the `.db` file directly in a text editor, since it's a binary format and editing it as text will corrupt it.)
 
Note: `data/alpha_signal.db` and `data/raw_filings/` are gitignored, not committed — both are fully reproducible by re-running the scripts above, so they're treated as derived artifacts rather than source files.
 
## Why this scope
 
Cross-industry, 2 companies per sector across 8 sectors, rather than one sector deep — this tests whether a language-drift signal generalizes across very different businesses, rather than only working because of one industry's specific boilerplate. The signal itself is computed *within* a company (comparing its own filing to its own prior filing), so mixing industries doesn't break that step — but it does mean sector needs to be a control feature once modeling starts in Phase 3, since a tech company's normal quarter-to-quarter drift isn't the same baseline as an airline's.
 
## Recently Completed
 
**Phase 2 — embeddings.** Generate a vector embedding for each filing section using a pretrained transformer (`all-MiniLM-L6-v2`, via Hugging Face's `AutoTokenizer`/`AutoModel` rather than a black-box `sentence-transformers` call, with a manually implemented mean-pooling forward pass), chunked to fit the model's 512-token limit since most sections run well beyond that.

- Embedding: a way of turning data like words, sentences, or images into a list of numbers (a vector) so computers can understand and compare their meanings

- Used inference rather than training because the pretrained encoder was used as a feature extractor. It was better than fine-tuning a transformer on just over 150 examples would likely overfit.

## What's next

**Phase 3 — drift scoring and modeling.** Cosine similarity between a company's consecutive filing embeddings becomes the core drift feature. Modeling stays classical given the sample size (~150-200 filing-events) — regularized linear models (Ridge/Lasso/Elastic Net) and Random Forest, evaluated with a walk-forward backtest rather than a single split. Sector enters as a control feature per the note above.