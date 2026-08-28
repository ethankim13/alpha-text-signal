import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # must come before src imports

import config
from src import db_utils, drift_scorer


def main():
    for ticker in config.TICKERS:
        for section_type in config.FILING_SECTIONS:
            embeddings = db_utils.get_embeddings_for_ticker(ticker, section_type) 

            for i in range(1, len(embeddings)):
                prior = embeddings[i - 1]
                current = embeddings[i]
                sim, drift = drift_scorer.compute_drift(prior["embedding"], current["embedding"])
                db_utils.insert_drift_score(
                    ticker, current["accession_number"], section_type,
                    current["filing_date"], prior["accession_number"], sim, drift,
                )

    print("Done computing drift scores.")


if __name__ == "__main__":
    main()