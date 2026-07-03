"""get_price_history tool: recent OHLCV rows for a ticker from the local
`prices` table. Deterministic, read-only — returns data, not prose."""

import pandas as pd

from data.storage import get_connection, init_db


def get_price_history(ticker: str, days: int = 30, db_path=None) -> dict:
    """Return the most recent `days` OHLCV rows for `ticker`, oldest first."""
    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM prices "
            "WHERE ticker = ? ORDER BY date, id",
            conn,
            params=(ticker,),
        )
    finally:
        conn.close()

    if df.empty:
        return {"ticker": ticker, "rows": []}

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date").tail(days)

    rows = [
        {
            "date": r["date"].strftime("%Y-%m-%d"),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": int(r["volume"]),
        }
        for _, r in df.iterrows()
    ]
    return {"ticker": ticker, "rows": rows}
