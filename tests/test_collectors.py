import json
from unittest.mock import Mock, patch

from data.collectors.filings import collect_filings
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


def test_collect_filings_writes_rows_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "test-agent test@example.com")
    db_path = tmp_path / "test.db"

    ticker_lookup_response = Mock()
    ticker_lookup_response.raise_for_status = Mock()
    ticker_lookup_response.json.return_value = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    }

    submissions_response = Mock()
    submissions_response.raise_for_status = Mock()
    submissions_response.json.return_value = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-Q"],
                "filingDate": ["2026-05-01", "2026-04-15", "2026-02-01"],
                "accessionNumber": [
                    "0000320193-26-000001",
                    "0000320193-26-000002",
                    "0000320193-26-000003",
                ],
                "primaryDocument": ["aapl-10k.htm", "aapl-8k.htm", "aapl-10q.htm"],
            }
        }
    }

    with patch(
        "data.collectors.filings.requests.get",
        side_effect=[ticker_lookup_response, submissions_response],
    ) as mock_get:
        rows_written = collect_filings("AAPL", db_path=db_path)
        for call in mock_get.call_args_list:
            assert call.kwargs["headers"]["User-Agent"] == "test-agent test@example.com"

    # 8-K is filtered out — only 10-K and 10-Q are collected by default.
    assert rows_written == 2

    conn = get_connection(db_path)
    init_db(conn)
    stored = conn.execute(
        "SELECT ticker, filing_type, filing_date, data, fetched_at FROM filings"
    ).fetchall()
    conn.close()

    assert len(stored) == 2
    filing_types = {row["filing_type"] for row in stored}
    assert filing_types == {"10-K", "10-Q"}
    parsed = json.loads(stored[0]["data"])
    assert parsed["accession_number"].startswith("0000320193")
    assert stored[0]["fetched_at"] is not None


def test_collect_filings_raises_without_user_agent(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)

    try:
        collect_filings("AAPL", db_path=tmp_path / "test.db")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "SEC_EDGAR_USER_AGENT" in str(e)
