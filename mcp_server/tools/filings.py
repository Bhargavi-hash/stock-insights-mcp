"""get_filing_summary tool: latest fundamental figures for a ticker, from
the `features` table (populated from SEC EDGAR companyfacts by the feature
layer — see features/build_features.py)."""

from data.storage import get_connection, init_db


def get_filing_summary(ticker: str, db_path=None) -> dict:
    """Return the most recent known eps, revenue, revenue_growth, and
    debt_to_equity for `ticker` — i.e. the latest feature row that actually
    has a fundamental value, not necessarily the latest price date."""
    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        row = conn.execute(
            "SELECT date, eps, revenue, revenue_growth, debt_to_equity FROM features "
            "WHERE ticker = ? AND eps IS NOT NULL "
            "ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {
            "ticker": ticker,
            "date": None,
            "eps": None,
            "revenue": None,
            "revenue_growth": None,
            "debt_to_equity": None,
        }

    return {
        "ticker": ticker,
        "date": row["date"],
        "eps": row["eps"],
        "revenue": row["revenue"],
        "revenue_growth": row["revenue_growth"],
        "debt_to_equity": row["debt_to_equity"],
    }
