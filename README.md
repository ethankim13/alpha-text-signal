# Alpha Text Signal

Testing whether quarter-over-quarter language drift in 10-Q filings, specifically the Risk Factors (Item 1A) and MD&A (Item 2) sections, carries any predictive information about short-term stock returns (5-day forward return following each filing).

This is a research project, not a product. The deliverable is a finding: does the signal have predictive edge or not, evaluated with a time-aware, walk-forward backtest rather than a single train/test split. A null result is a legitimate outcome here, as long as the methodology is sound, and that is in fact what I found.

**Scope:** 16 tickers across 8 sectors (tech, retail, healthcare, energy, industrials, consumer staples, airlines, telecom), roughly 3 years of 10-Q history per ticker, resulting in 161 filings with both a computed drift score and a measurable 5-day forward return.

## Results

I ran a walk-forward backtest (expanding window, 5 folds, an initial training window covering the earliest 50% of filings by date) across four models: Ridge, Lasso, Elastic Net, and Random Forest. Every model used drift_score_mda and drift_score_risk_factors as numeric features, plus one-hot encoded sector, to predict forward_return.

| Model | R2 | Correlation | N (out-of-sample predictions) |
|---|---|---|---|
| Ridge | -0.1962 | -0.2544 | 81 |
| Lasso | -0.0966 | -0.2389 | 81 |
| Elastic Net | -0.0966 | -0.2389 | 81 |
| Random Forest | -0.3250 | -0.3186 | 81 |

All four models produced negative R2 on out-of-sample predictions, meaning each performed worse than simply guessing the average forward return for every filing. Correlation between predicted and actual returns was also negative and modest in magnitude across the board.

**This is a null result.** At this sample size, quarter-over-quarter language drift in Risk Factors and MD&A, using this feature set and this model family, does not show a reliable predictive relationship with 5-day forward stock returns.

A few honest caveats worth stating rather than glossing over:

- 81 out-of-sample predictions is a small evaluation set. A negative correlation of this size is well within the range that noise alone could produce.
- Mean-pooling chunk embeddings into a single vector per section likely smooths over the specific sentences most responsible for real language change, diluting exactly the signal this project is testing for.
- A 5-day forward return is a narrow, noisy target. If a filing-driven price effect exists, it may show up on a different horizon, or it may be swamped by unrelated market news within that window.
- Sector, encoded as 8 one-hot columns, adds real dimensionality relative to roughly 150 rows, which likely hurts the linear models more than it helps.

I am reporting this result as is, rather than tuning parameters until something looks better. A negative or near-zero result from a sound methodology is a legitimate outcome for this kind of question, and it is a more honest thing to bring into an interview than an inflated one.

### Possible next steps (not built)

- Predict direction (up or down) instead of the raw return, since classification may be more robust than regression at this sample size.
- Filter or weight by section length or boilerplate status, instead of mean imputing missing Risk Factors drift.
- Test a longer forward return window (10 or 20 days) to check whether the horizon, not the feature itself, is the limiting factor.
- Expand the ticker universe in a future iteration to increase sample size, which is the single biggest constraint on this analysis.

## Data sourcing

Filing metadata comes from SEC EDGAR's structured JSON API (ticker to CIK resolution, then per company filing history). The actual filing text does not come from that API. SEC does not expose section level content as structured data, so I fetch the raw HTML document SEC has on file for each filing and parse it directly. That second part is closer to document parsing and light scraping than API use, and I treat it that way rather than overstating it as using an API end to end. Price data comes from Yahoo Finance via `yfinance`.

## What's built

**Phase 0 / 1a, metadata and price pipeline**
- `config.py`, single source of truth for every scope decision: tickers, sectors, filing type, lookback window, target variable definition, sections to extract
- `db/schema.sql`, SQLite schema: `companies`, `filings`, `prices`, `filing_sections`, `filing_embeddings`, `drift_scores`
- `src/edgar_client.py`, resolves tickers to SEC CIKs, pulls 10-Q filing metadata (dates, accession numbers, document URLs)
- `src/price_fetcher.py`, pulls daily price history via `yfinance`
- `src/db_utils.py`, all database reads and writes; every insert is idempotent (`INSERT OR IGNORE` / `INSERT OR REPLACE` / `ON CONFLICT`), so the whole pipeline is safe to rerun without duplicating data
- `scripts/01_build_dataset.py`, orchestrates the above end to end

**Phase 1b, filing text extraction**
- `src/filing_fetcher.py`, downloads each filing's raw HTML from its `document_url` and caches it locally, so rerunning the pipeline never rehits SEC's servers for a filing already on disk
- `src/filing_parser.py`, strips HTML down to plain text (BeautifulSoup) and extracts the Risk Factors and MD&A sections specifically, using boundary pattern matching that accounts for Table of Contents false positives (taking the last heading match in the document rather than the first)
- `scripts/02_extract_filing_text.py`, orchestrates fetch, parse, and store for every filing incrementally

A real finding surfaced here: Risk Factors extraction was clearly bimodal. Roughly 90% of filings had a substantial Risk Factors section, while the rest returned only a couple of characters, a leftover page number from a Table of Contents reference. This matches an actual SEC rule: a 10-Q only needs to restate Risk Factors if there has been a material change since the last 10-K. Most quarters, most companies, do not restate it. I kept `char_count` on every extracted section so later phases could treat this as an informative case rather than silently diluting the signal.

**Phase 2, embeddings**
- `src/embedder.py`, generates a 384-dimension vector embedding for each filing section using `all-MiniLM-L6-v2`, loaded directly via Hugging Face's `AutoTokenizer` and `AutoModel` rather than a black-box `sentence-transformers` call, with a manually implemented mean-pooling forward pass
- Each section is chunked into 510-token windows before embedding, since most sections run well past the model's 512-token limit, and the resulting chunk vectors are averaged into one vector per section
- `scripts/03_generate_embeddings.py`, orchestrates embedding generation for every section that has real content

I used inference only, not fine-tuning, since the pretrained encoder is being used purely as a feature extractor. Fine-tuning a transformer on roughly 150 examples would almost certainly overfit. The actual learning in this project happens downstream, in the regularized models used for the backtest.

**Phase 3, drift scoring and modeling**
- `src/drift_scorer.py`, computes cosine similarity between a company's consecutive filing embeddings, same section type, and converts it to a drift score (1 minus similarity, so higher means more language change)
- `scripts/04_compute_drift.py`, orchestrates drift scoring for every ticker and section type, skipping each ticker's first filing since there is no prior filing to compare it to
- `src/dataset_builder.py`, joins drift scores to each filing's 5-day forward return (pulled from `prices`, aligned to the first trading day on or after the filing date) and to sector, producing one row per filing
- `scripts/05_build_dataset.py`, builds and saves the final modeling dataset to `data/model_dataset.csv`
- `src/modeling.py`, defines the feature preparation, the four models being compared, and the walk-forward backtest logic
- `scripts/06_train_and_backtest.py`, runs the backtest and prints results per model

## Setup

```bash
cd alpha-text-signal
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Open `config.py` and replace the placeholder email in `SEC_USER_AGENT` with a real name and email. SEC requires this on every request so they can contact you if a script misbehaves. Requests without one get rate limited or blocked.

## Running it

```bash
python scripts/01_build_dataset.py          # metadata and prices
python scripts/02_extract_filing_text.py    # filing text extraction
python scripts/03_generate_embeddings.py    # embeddings
python scripts/04_compute_drift.py          # drift scores
python scripts/05_build_dataset.py          # final modeling dataset
python scripts/06_train_and_backtest.py     # walk-forward backtest
```

All are safe to rerun. Already processed filings, sections, embeddings, and drift scores are skipped rather than redone.

## Checking what's in the database

```bash
sqlite3 data/alpha_signal.db "SELECT ticker, form_type, filing_date FROM filings ORDER BY ticker, filing_date DESC LIMIT 20;"
sqlite3 data/alpha_signal.db "SELECT section_type, COUNT(*), AVG(char_count) FROM filing_sections GROUP BY section_type;"
sqlite3 data/alpha_signal.db "SELECT section_type, COUNT(*), AVG(drift_score) FROM drift_scores GROUP BY section_type;"
```

If `sqlite3` is not installed as a CLI tool, [DB Browser for SQLite](https://sqlitebrowser.org/) is a free GUI alternative. I never open the `.db` file directly in a text editor, since it is a binary format and editing it as text will corrupt it.

Note: `data/alpha_signal.db` and `data/raw_filings/` are gitignored, not committed. Both are fully reproducible by rerunning the scripts above, so I treat them as derived artifacts rather than source files. `data/model_dataset.csv` is small and is committed, since it is the actual research dataset behind the results above.

## Why this scope

Cross-industry, 2 companies per sector across 8 sectors, rather than one sector deep. This tests whether a language drift signal generalizes across very different businesses, rather than only working because of one industry's specific boilerplate. The signal itself is computed within a company, comparing its own filing to its own prior filing, so mixing industries does not break that step. It does mean sector needed to enter modeling as a control feature, since a tech company's normal quarter-to-quarter drift is not the same baseline as an airline's.