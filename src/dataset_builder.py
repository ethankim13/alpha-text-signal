"""
Joins drift_scores (Phase 3a) to prices (Phase 0/1a) to produce one row
per filing: drift scores for both sections, sector, the 5-day forward
return, and the date that return becomes known (target_end_date).
"""
import pandas as pd

import config
from src.db_utils import get_connection


def get_forward_return(filing_date: str, prices_df: pd.DataFrame) -> tuple[float, str] | None:
    """
    prices_df: one ticker's rows from `prices`, sorted oldest to newest.
    Returns (forward_return, target_end_date), or None if no valid return exists.

    Day 0 is the first trading day ON OR AFTER filing_date.
    """
    if prices_df.empty:
        return None

    # Filing is older than our price history: any "return" would be fake.
    if filing_date < prices_df["date"].min():
        return None

    # Keep trading days from the filing date onward; row 0 is day 0.
    future_prices = prices_df[prices_df["date"] >= filing_date].reset_index(drop=True)

    day_n = config.FORWARD_RETURN_DAYS
    if len(future_prices) <= day_n:
        return None  # not enough trading days after the filing yet

    start_price = future_prices.loc[0, "adj_close"]
    end_price = future_prices.loc[day_n, "adj_close"]

    if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
        return None

    forward_return = (end_price - start_price) / start_price
    target_end_date = future_prices.loc[day_n, "date"]  # the day the outcome is known

    return float(forward_return), target_end_date


def build_feature_matrix() -> pd.DataFrame:
    """One row per filing: both drift scores, sector, forward_return, target_end_date."""
    conn = get_connection()
    try:
        drift_df = pd.read_sql_query(
            "SELECT ticker, accession_number, section_type, filing_date, drift_score FROM drift_scores;",
            conn,
        )
        prices_df = pd.read_sql_query("SELECT ticker, date, adj_close FROM prices;", conn)
    finally:
        conn.close()

    # Long format to wide format: one row per filing, one column per section.
    pivoted = drift_df.pivot_table(
        index=["ticker", "accession_number", "filing_date"],
        columns="section_type",
        values="drift_score",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(
        columns={"mda": "drift_score_mda", "risk_factors": "drift_score_risk_factors"}
    )

    for col in ["drift_score_mda", "drift_score_risk_factors"]:
        if col not in pivoted.columns:
            pivoted[col] = pd.NA

    prices_df = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    rows = []
    dropped_no_return = 0

    for _, filing in pivoted.iterrows():
        ticker = filing["ticker"]
        ticker_prices = prices_df[prices_df["ticker"] == ticker].reset_index(drop=True)

        result = get_forward_return(filing["filing_date"], ticker_prices)
        if result is None:
            dropped_no_return += 1
            continue

        forward_return, target_end_date = result

        rows.append(
            {
                "ticker": ticker,
                "accession_number": filing["accession_number"],
                "filing_date": filing["filing_date"],
                "target_end_date": target_end_date,
                "sector": config.TICKER_SECTORS.get(ticker),
                "drift_score_mda": filing["drift_score_mda"],
                "drift_score_risk_factors": filing["drift_score_risk_factors"],
                "forward_return": forward_return,
            }
        )

    result_df = pd.DataFrame(rows)
    missing_rf = result_df["drift_score_risk_factors"].isna().sum()

    print(f"Filings with at least one drift score: {len(pivoted)}")
    print(f"Dropped (no valid forward return): {dropped_no_return}")
    print(f"Missing risk_factors drift (kept as NaN): {missing_rf}")
    print(f"Final dataset rows: {len(result_df)}")

    return result_df