from unittest.mock import Mock, patch

from data.collectors.news import collect_news
from data.collectors.prices import collect_prices
from data.storage import get_connection, init_db


def test_collect_prices_writes_rows_to_sqlite(tmp_path):
    db_path = tmp_path / "test.db"

    rows_written = collect_prices("AAPL", period="5d", db_path=db_path)

    assert rows_written > 0

    conn = get_connection(db_path)
    init_db(conn)
    stored = conn.execute(
        "SELECT ticker, date, open, high, low, close, volume, fetched_at FROM prices"
    ).fetchall()
    conn.close()

    assert len(stored) == rows_written
    row = stored[0]
    assert row["ticker"] == "AAPL"
    assert row["open"] > 0
    assert row["volume"] >= 0
    assert row["fetched_at"] is not None


def test_collect_news_writes_rows_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "test-key")
    db_path = tmp_path / "test.db"

    fake_response = Mock()
    fake_response.raise_for_status = Mock()
    fake_response.json.return_value = {
        "status": "ok",
        "articles": [
            {
                "title": "Apple unveils new product",
                "source": {"name": "Reuters"},
                "url": "https://example.com/article1",
                "publishedAt": "2026-07-01T12:00:00Z",
            },
            {
                # Articles without a title are dropped by known-removed listings.
                "title": None,
                "source": {"name": "Reuters"},
                "url": "https://example.com/removed",
                "publishedAt": "2026-07-01T13:00:00Z",
            },
        ],
    }

    with patch("data.collectors.news.requests.get", return_value=fake_response) as mock_get:
        rows_written = collect_news("AAPL", db_path=db_path)
        assert mock_get.call_args.kwargs["headers"] == {"X-Api-Key": "test-key"}

    assert rows_written == 1

    conn = get_connection(db_path)
    init_db(conn)
    stored = conn.execute(
        "SELECT ticker, headline, source, url, published_at, fetched_at FROM news"
    ).fetchall()
    conn.close()

    assert len(stored) == 1
    row = stored[0]
    assert row["ticker"] == "AAPL"
    assert row["headline"] == "Apple unveils new product"
    assert row["source"] == "Reuters"
    assert row["fetched_at"] is not None


def test_collect_news_raises_without_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    try:
        collect_news("AAPL", db_path=tmp_path / "test.db")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "NEWSAPI_KEY" in str(e)
