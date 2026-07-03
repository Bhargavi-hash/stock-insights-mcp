from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd

from data.storage import get_connection, init_db, insert_news, insert_prices
from features.build_features import FEATURE_COLUMNS, collect_features


def _seed_prices(conn, ticker: str, n_days: int = 60):
    dates = pd.bdate_range(end="2026-07-01", periods=n_days)
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "ticker": ticker,
            "date": d.strftime("%Y-%m-%d"),
            "open": 100 + i,
            "high": 101 + i,
            "low": 99 + i,
            "close": 100 + i,
            "volume": 1_000_000,
            "fetched_at": fetched_at,
        }
        for i, d in enumerate(dates)
    ]
    insert_prices(conn, rows)
    return dates


def _seed_news(conn, ticker: str, dates):
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "ticker": ticker,
            "headline": "Company beats earnings expectations",
            "source": "Reuters",
            "url": "https://example.com/1",
            "published_at": dates[-1].strftime("%Y-%m-%dT12:00:00Z"),
            "fetched_at": fetched_at,
        },
        {
            "ticker": ticker,
            "headline": "Company faces regulatory investigation",
            "source": "Reuters",
            "url": "https://example.com/2",
            "published_at": dates[-2].strftime("%Y-%m-%dT12:00:00Z"),
            "fetched_at": fetched_at,
        },
    ]
    insert_news(conn, rows)


def _mock_company_facts_responses():
    ticker_lookup_response = Mock()
    ticker_lookup_response.raise_for_status = Mock()
    ticker_lookup_response.json.return_value = {
        "0": {"cik_str": 320193, "ticker": "TEST", "title": "Test Corp"},
    }

    company_facts_response = Mock()
    company_facts_response.raise_for_status = Mock()
    company_facts_response.json.return_value = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareBasic": {
                    "units": {
                        "USD/shares": [
                            {"end": "2026-03-31", "val": 1.5, "form": "10-Q"},
                            {"end": "2026-06-30", "val": 1.8, "form": "10-Q"},
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {"end": "2026-03-31", "val": 1_000_000, "form": "10-Q"},
                            {"end": "2026-06-30", "val": 1_200_000, "form": "10-Q"},
                        ]
                    }
                },
                "Liabilities": {
                    "units": {
                        "USD": [{"end": "2026-06-30", "val": 500_000, "form": "10-Q"}]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [{"end": "2026-06-30", "val": 250_000, "form": "10-Q"}]
                    }
                },
            }
        }
    }
    return [ticker_lookup_response, company_facts_response]


def test_collect_features_writes_rows_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "test-agent test@example.com")
    db_path = tmp_path / "test.db"

    conn = get_connection(db_path)
    init_db(conn)
    dates = _seed_prices(conn, "TEST")
    _seed_news(conn, "TEST", dates)
    conn.close()

    with patch("requests.get", side_effect=_mock_company_facts_responses()):
        rows_written = collect_features("TEST", db_path=db_path)

    assert rows_written == len(dates)

    conn = get_connection(db_path)
    stored = conn.execute(
        "SELECT * FROM features WHERE ticker = 'TEST' ORDER BY date"
    ).fetchall()
    conn.close()

    assert len(stored) == len(dates)

    stored_columns = set(stored[0].keys())
    for col in FEATURE_COLUMNS:
        assert col in stored_columns

    last_row = stored[-1]
    assert last_row["sma_20"] is not None
    assert last_row["sma_50"] is not None
    assert last_row["rsi_14"] is not None
    assert last_row["volatility_20"] is not None
    assert last_row["sentiment_score"] is not None
    assert last_row["eps"] == 1.8
    assert last_row["debt_to_equity"] == 2.0
    assert last_row["label"] is None

    # Early rows lack enough history for a 50-day SMA.
    assert stored[0]["sma_50"] is None
