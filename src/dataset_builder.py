"""
Joins drift_scores (Phase 3a) to prices (Phase 0/1a) to produce one row
per filing: drift scores for both sections, sector, and the 5-day forward
return that Phase 3b's models will actually try to predict.

Uses pandas here rather than raw SQL for the forward-return lookup, since
"find the Nth trading day after this date" is naturally row-position logic
(.iloc-style), not something SQL expresses cleanly.
"""
import pandas as pd

import config
from src.db_utils import get_connection


def get_forward_return(filing_date: str, prices_df: pd.DataFrame) -> float | None:
    """
    prices_df: one ticker's rows from `prices`, already sorted oldest-to-newest.

    A filing_date won't always land on an actual trading day (weekends,
    holidays), so "day 0" is defined as the first trading day ON OR AFTER
    filing_date, not filing_date itself.
    """
    # Keep only trading days from the filing date onward, and reset the
    # index so day 0 is always row 0 of what's left.
    future_prices = prices_df[prices_df["date"] >= filing_date].reset_index(drop=True)

    if future_prices.empty:
        return None  # filing_date is past the end of the price history we have

    day_n = config.FORWARD_RETURN_DAYS
    if len(future_prices) <= day_n:
        # Not enough trading days after this filing yet to measure a forward
        # return — happens for filings close to "today" in the dataset.
        return None

    close_start = future_prices.loc[0, "close"]
    close_end = future_prices.loc[day_n, "close"]
    return (close_end - close_start) / close_start


def build_feature_matrix() -> pd.DataFrame:
    """Assembles the full modeling dataset: one row per filing, with both
    section drift scores, sector, and forward_return as columns."""
    conn = get_connection()
    try:
        drift_df = pd.read_sql_query(
            "SELECT ticker, accession_number, section_type, filing_date, drift_score FROM drift_scores;",
            conn,
        )
        prices_df = pd.read_sql_query("SELECT ticker, date, close FROM prices;", conn)
    finally:
        conn.close()

    # drift_scores currently has up to 2 rows per filing (one per
    # section_type). pivot_table reshapes that into 1 row per filing, with
    # section_type's two possible values ("mda", "risk_factors") becoming
    # their own columns — this is the "long format to wide format" step.
    pivoted = drift_df.pivot_table(
        index=["ticker", "accession_number", "filing_date"],
        columns="section_type",
        values="drift_score",
    ).reset_index()
    pivoted.columns.name = None  # pivot_table labels the column axis; not needed here
    pivoted = pivoted.rename(
        columns={"mda": "drift_score_mda", "risk_factors": "drift_score_risk_factors"}
    )

    # Guard against the (unlikely but possible) case where one section
    # type never appears at all in drift_scores — keeps downstream code
    # from breaking on a missing column.
    for col in ["drift_score_mda", "drift_score_risk_factors"]:
        if col not in pivoted.columns:
            pivoted[col] = pd.NA

    prices_df = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    rows = []
    dropped_no_return = 0

    for _, filing in pivoted.iterrows():
        ticker = filing["ticker"]
        ticker_prices = prices_df[prices_df["ticker"] == ticker].reset_index(drop=True)

        forward_return = get_forward_return(filing["filing_date"], ticker_prices)
        if forward_return is None:
            dropped_no_return += 1
            continue  # can't train on a filing with no measurable outcome yet

        rows.append(
            {
                "ticker": ticker,
                "accession_number": filing["accession_number"],
                "filing_date": filing["filing_date"],
                "sector": config.TICKER_SECTORS.get(ticker),
                "drift_score_mda": filing["drift_score_mda"],
                "drift_score_risk_factors": filing["drift_score_risk_factors"],
                "forward_return": forward_return,
            }
        )

    result_df = pd.DataFrame(rows)

    # risk_factors drift is legitimately missing for a real reason (the
    # boilerplate/no-restatement case from Phase 1b) — reported here, but
    # NOT dropped, since MDA drift alone is still a usable row. Phase 3b
    # decides how to handle the missing column (e.g. a separate model, or
    # an indicator feature), rather than losing the row outright here.
    missing_rf = result_df["drift_score_risk_factors"].isna().sum()

    print(f"Filings with at least one drift score: {len(pivoted)}")
    print(f"Dropped (no forward return available yet): {dropped_no_return}")
    print(f"Missing risk_factors drift (boilerplate quarter, kept as NaN): {missing_rf}")
    print(f"Final dataset rows: {len(result_df)}")

    return result_df