"""get_recent_news tool: recent headlines with per-headline VADER sentiment
scores. The raw `news` table stores headlines only — sentiment is scored
here on the fly (same scorer the feature layer uses), no API calls needed."""

from datetime import datetime, timedelta, timezone

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data.storage import get_connection, init_db

_analyzer = SentimentIntensityAnalyzer()


def get_recent_news(ticker: str, days: int = 7, db_path=None) -> dict:
    """Return headlines for `ticker` published in the last `days` days, each
    with a VADER compound sentiment score in [-1, 1], newest first."""
    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        df = pd.read_sql_query(
            "SELECT headline, source, url, published_at FROM news WHERE ticker = ?",
            conn,
            params=(ticker,),
        )
    finally:
        conn.close()

    if df.empty:
        return {"ticker": ticker, "headlines": []}

    df["published_at_parsed"] = pd.to_datetime(
        df["published_at"], utc=True, errors="coerce"
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = df[df["published_at_parsed"] >= cutoff].sort_values(
        "published_at_parsed", ascending=False
    )

    headlines = [
        {
            "headline": r["headline"],
            "source": r["source"],
            "url": r["url"],
            "published_at": r["published_at"],
            "sentiment_score": _analyzer.polarity_scores(r["headline"])["compound"],
        }
        for _, r in df.iterrows()
    ]
    return {"ticker": ticker, "headlines": headlines}
