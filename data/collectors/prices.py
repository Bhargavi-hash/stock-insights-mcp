"""Fetches OHLCV price data via yfinance and writes it to the `prices` table."""

from datetime import datetime, timezone

import yfinance as yf

from data.storage import get_connection, init_db, insert_prices


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d"):
    """Pull OHLCV history for a ticker from yfinance. Returns a pandas DataFrame
    indexed by date, with columns Open/High/Low/Close/Volume."""
    history = yf.Ticker(ticker).history(period=period, interval=interval)
    if history.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'")
    return history


def to_rows(ticker: str, history) -> list[dict]:
    """Convert a yfinance history DataFrame into row dicts for `insert_prices`."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for date, row in history.iterrows():
        rows.append(
            {
                "ticker": ticker,
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "fetched_at": fetched_at,
            }
        )
    return rows


def collect_prices(ticker: str, period: str = "1y", interval: str = "1d", db_path=None) -> int:
    """Fetch OHLCV for `ticker` and store it in SQLite. Returns rows written."""
    history = fetch_ohlcv(ticker, period=period, interval=interval)
    rows = to_rows(ticker, history)

    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        return insert_prices(conn, rows)
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    count = collect_prices(ticker)
    print(f"Wrote {count} rows for {ticker}")
