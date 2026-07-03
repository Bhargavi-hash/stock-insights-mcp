"""Computes forward-looking outperformance labels for the `features` table.

Label definition (architecture.md 3.3): did the ticker outperform a
benchmark (e.g. SPY) over the next N trading days? This is more honest than
a raw price/direction label, since it doesn't reward the model for
market-wide moves it had nothing to do with.
"""

import pandas as pd

from data.storage import get_connection, init_db

DEFAULT_BENCHMARK = "SPY"
DEFAULT_HORIZON = 10


def load_close_prices(conn, ticker: str) -> pd.Series:
    """Load a ticker's close price series, indexed by date (later fetches
    win on duplicate dates)."""
    df = pd.read_sql_query(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date, id",
        conn,
        params=(ticker,),
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()
    return df["close"]


def compute_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Pct change from each date's close to the close `horizon` trading days
    later. The last `horizon` dates have no future data yet, so they're NaN."""
    return close.shift(-horizon) / close - 1


def compute_labels(
    ticker: str,
    benchmark: str = DEFAULT_BENCHMARK,
    horizon: int = DEFAULT_HORIZON,
    db_path=None,
) -> pd.DataFrame:
    """Return a DataFrame with columns ticker, date, ticker_return,
    benchmark_return, label — label is 1 if `ticker`'s forward return over
    `horizon` trading days beat `benchmark`'s, else 0. Dates without enough
    future price history, or without a matching benchmark date, are
    dropped."""
    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        ticker_close = load_close_prices(conn, ticker)
        benchmark_close = load_close_prices(conn, benchmark)
    finally:
        conn.close()

    if ticker_close.empty:
        raise ValueError(f"No price data for '{ticker}' — run collect_prices first")
    if benchmark_close.empty:
        raise ValueError(
            f"No price data for benchmark '{benchmark}' — run collect_prices first"
        )

    ticker_fwd = compute_forward_return(ticker_close, horizon)
    benchmark_fwd = compute_forward_return(benchmark_close, horizon)

    df = pd.DataFrame(
        {"ticker_return": ticker_fwd, "benchmark_return": benchmark_fwd}
    ).dropna()
    df["label"] = (df["ticker_return"] > df["benchmark_return"]).astype(int)
    df = df.reset_index().rename(columns={"index": "date"})
    df.insert(0, "ticker", ticker)
    return df


def label_features(
    ticker: str,
    benchmark: str = DEFAULT_BENCHMARK,
    horizon: int = DEFAULT_HORIZON,
    db_path=None,
) -> int:
    """Compute labels for `ticker` and write them into the `label` column of
    the `features` table. Returns the number of rows updated."""
    labels = compute_labels(ticker, benchmark=benchmark, horizon=horizon, db_path=db_path)

    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        updated = 0
        for _, row in labels.iterrows():
            cur = conn.execute(
                "UPDATE features SET label = ? WHERE ticker = ? AND date = ?",
                (int(row["label"]), ticker, row["date"].strftime("%Y-%m-%d")),
            )
            updated += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return updated


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    count = label_features(ticker)
    print(f"Labeled {count} rows for {ticker}")
