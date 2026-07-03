"""get_technical_signals tool: latest technical feature values for a ticker
from the `features` table."""

from data.storage import get_connection, init_db


def get_technical_signals(ticker: str, db_path=None) -> dict:
    """Return the most recent sma_20, sma_50, rsi_14, volatility_20 for
    `ticker`. Values are None if no features have been built yet."""
    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        row = conn.execute(
            "SELECT date, sma_20, sma_50, rsi_14, volatility_20 FROM features "
            "WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {
            "ticker": ticker,
            "date": None,
            "sma_20": None,
            "sma_50": None,
            "rsi_14": None,
            "volatility_20": None,
        }

    return {
        "ticker": ticker,
        "date": row["date"],
        "sma_20": row["sma_20"],
        "sma_50": row["sma_50"],
        "rsi_14": row["rsi_14"],
        "volatility_20": row["volatility_20"],
    }
