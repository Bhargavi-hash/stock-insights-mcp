"""Fetches recent headlines via NewsAPI and writes them to the `news` table."""

import os
from datetime import datetime, timedelta, timezone

import requests

from data.storage import get_connection, init_db, insert_news

NEWS_API_URL = "https://newsapi.org/v2/everything"


def fetch_headlines(ticker: str, days: int = 7, page_size: int = 50) -> list[dict]:
    """Pull recent headlines mentioning `ticker` from NewsAPI. Returns the raw
    list of article dicts as returned by the API."""
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY environment variable is not set")

    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    response = requests.get(
        NEWS_API_URL,
        params={
            "q": ticker,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "from": since,
        },
        headers={"X-Api-Key": api_key},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {payload.get('message', payload)}")
    return payload.get("articles", [])


def to_rows(ticker: str, articles: list[dict]) -> list[dict]:
    """Convert raw NewsAPI article dicts into row dicts for `insert_news`."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for article in articles:
        title = article.get("title")
        if not title:
            continue
        rows.append(
            {
                "ticker": ticker,
                "headline": title,
                "source": (article.get("source") or {}).get("name"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "fetched_at": fetched_at,
            }
        )
    return rows


def collect_news(ticker: str, days: int = 7, db_path=None) -> int:
    """Fetch recent headlines for `ticker` and store them in SQLite. Returns
    rows written."""
    articles = fetch_headlines(ticker, days=days)
    rows = to_rows(ticker, articles)

    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        return insert_news(conn, rows)
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    count = collect_news(ticker)
    print(f"Wrote {count} rows for {ticker}")

# 6bb90032562647c39b576b94726a2eec