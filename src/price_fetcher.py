"""
Pulls daily price history for a ticker via yfinance. Kept deliberately
separate from edgar_client.py — prices and filings are different data
sources with different failure modes, no reason to tangle them together.
"""
import yfinance as yf


def fetch_price_history(ticker: str, start: str, end: str | None = None) -> list[tuple]:
    """Returns a list of (date, open, high, low, close, adj_close, volume)
    tuples, ready to hand to db_utils.insert_prices."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return []

    # yfinance sometimes returns a MultiIndex column header (ticker, field)
    # even for a single ticker — flatten it so column access is predictable.
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rows = []
    for date, row in df.iterrows():
        rows.append(
            (
                date.strftime("%Y-%m-%d"),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(row["Adj Close"]),
                int(row["Volume"]),
            )
        )
    return rows
