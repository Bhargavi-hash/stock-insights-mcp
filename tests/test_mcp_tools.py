from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import mcp_server.tools.prediction as prediction_module
from data.storage import get_connection, init_db, insert_features, insert_news, insert_prices
from features.build_features import build_technical_features
from mcp_server.tools.filings import get_filing_summary
from mcp_server.tools.news import get_recent_news
from mcp_server.tools.prediction import get_model_prediction
from mcp_server.tools.prices import get_price_history
from mcp_server.tools.technicals import get_technical_signals
from model.train import FEATURE_COLUMNS, train_model


def test_get_price_history_returns_recent_rows(tmp_path):
    db_path = tmp_path / "test.db"
    dates = pd.bdate_range(end="2026-07-01", periods=10)
    fetched_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection(db_path)
    init_db(conn)
    insert_prices(
        conn,
        [
            {
                "ticker": "AAPL",
                "date": d.strftime("%Y-%m-%d"),
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100 + i,
                "volume": 1_000_000 + i,
                "fetched_at": fetched_at,
            }
            for i, d in enumerate(dates)
        ],
    )
    conn.close()

    result = get_price_history("AAPL", days=5, db_path=db_path)

    assert result["ticker"] == "AAPL"
    assert len(result["rows"]) == 5
    # Oldest first, and it's the most recent 5 of the 10 seeded days.
    assert [r["date"] for r in result["rows"]] == [d.strftime("%Y-%m-%d") for d in dates[-5:]]
    row = result["rows"][-1]
    assert set(row.keys()) == {"date", "open", "high", "low", "close", "volume"}
    assert row["close"] == 109.0
    assert isinstance(row["volume"], int)


def test_get_price_history_unknown_ticker_returns_empty(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    conn.close()

    result = get_price_history("NOPE", db_path=db_path)
    assert result == {"ticker": "NOPE", "rows": []}


def test_get_technical_signals_returns_latest_row(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    insert_features(
        conn,
        [
            {
                "ticker": "AAPL",
                "date": "2026-06-30",
                "sma_20": 100.0,
                "sma_50": 95.0,
                "rsi_14": 50.0,
                "volatility_20": 0.02,
                "sentiment_score": None,
                "news_count": 0,
                "eps": None,
                "revenue": None,
                "revenue_growth": None,
                "debt_to_equity": None,
                "label": None,
            },
            {
                "ticker": "AAPL",
                "date": "2026-07-01",
                "sma_20": 101.0,
                "sma_50": 96.0,
                "rsi_14": 55.0,
                "volatility_20": 0.021,
                "sentiment_score": None,
                "news_count": 0,
                "eps": None,
                "revenue": None,
                "revenue_growth": None,
                "debt_to_equity": None,
                "label": None,
            },
        ],
    )
    conn.close()

    result = get_technical_signals("AAPL", db_path=db_path)

    assert result == {
        "ticker": "AAPL",
        "date": "2026-07-01",
        "sma_20": 101.0,
        "sma_50": 96.0,
        "rsi_14": 55.0,
        "volatility_20": 0.021,
    }


def test_get_recent_news_filters_by_days_and_scores_sentiment(tmp_path):
    db_path = tmp_path / "test.db"
    now = datetime.now(timezone.utc)
    fetched_at = now.isoformat()

    conn = get_connection(db_path)
    init_db(conn)
    insert_news(
        conn,
        [
            {
                "ticker": "AAPL",
                "headline": "Apple shares soar on excellent record-breaking earnings, delighting investors",
                "source": "Reuters",
                "url": "https://example.com/recent",
                "published_at": (now - timedelta(days=1)).isoformat(),
                "fetched_at": fetched_at,
            },
            {
                "ticker": "AAPL",
                "headline": "Apple faces regulatory investigation",
                "source": "Reuters",
                "url": "https://example.com/old",
                "published_at": (now - timedelta(days=20)).isoformat(),
                "fetched_at": fetched_at,
            },
        ],
    )
    conn.close()

    result = get_recent_news("AAPL", days=7, db_path=db_path)

    assert result["ticker"] == "AAPL"
    assert len(result["headlines"]) == 1
    headline = result["headlines"][0]
    assert headline["headline"] == "Apple shares soar on excellent record-breaking earnings, delighting investors"
    assert set(headline.keys()) == {
        "headline",
        "source",
        "url",
        "published_at",
        "sentiment_score",
    }
    assert -1.0 <= headline["sentiment_score"] <= 1.0
    assert headline["sentiment_score"] > 0  # positive headline


def test_get_filing_summary_skips_rows_without_fundamentals(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    insert_features(
        conn,
        [
            {
                "ticker": "AAPL",
                "date": "2026-05-01",
                "sma_20": None,
                "sma_50": None,
                "rsi_14": None,
                "volatility_20": None,
                "sentiment_score": None,
                "news_count": None,
                "eps": 1.5,
                "revenue": 1_000_000.0,
                "revenue_growth": 0.05,
                "debt_to_equity": 1.2,
                "label": None,
            },
            {
                # Latest date, but fundamentals haven't refreshed yet.
                "ticker": "AAPL",
                "date": "2026-07-01",
                "sma_20": 100.0,
                "sma_50": 95.0,
                "rsi_14": 50.0,
                "volatility_20": 0.02,
                "sentiment_score": None,
                "news_count": None,
                "eps": None,
                "revenue": None,
                "revenue_growth": None,
                "debt_to_equity": None,
                "label": None,
            },
        ],
    )
    conn.close()

    result = get_filing_summary("AAPL", db_path=db_path)

    # Falls back to the latest row that actually has fundamentals.
    assert result == {
        "ticker": "AAPL",
        "date": "2026-05-01",
        "eps": 1.5,
        "revenue": 1_000_000.0,
        "revenue_growth": 0.05,
        "debt_to_equity": 1.2,
    }


def _seed_prices(conn, ticker: str, dates, seed: int):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(loc=0.1, scale=1.0, size=len(dates)))
    fetched_at = datetime.now(timezone.utc).isoformat()
    insert_prices(
        conn,
        [
            {
                "ticker": ticker,
                "date": d.strftime("%Y-%m-%d"),
                "open": c,
                "high": c + 1,
                "low": c - 1,
                "close": c,
                "volume": 1_000_000,
                "fetched_at": fetched_at,
            }
            for d, c in zip(dates, close)
        ],
    )
    return pd.DataFrame({"date": dates, "close": close})


def _seed_features(conn, ticker: str, prices: pd.DataFrame):
    technical = build_technical_features(prices)
    insert_features(
        conn,
        [
            {
                "ticker": ticker,
                "date": r["date"].strftime("%Y-%m-%d"),
                "sma_20": None if pd.isna(r["sma_20"]) else r["sma_20"],
                "sma_50": None if pd.isna(r["sma_50"]) else r["sma_50"],
                "rsi_14": None if pd.isna(r["rsi_14"]) else r["rsi_14"],
                "volatility_20": None if pd.isna(r["volatility_20"]) else r["volatility_20"],
                "sentiment_score": 0.1,
                "news_count": 2,
                "eps": 1.5,
                "revenue": 1_000_000.0,
                "revenue_growth": 0.05,
                "debt_to_equity": 1.2,
                "label": None,
            }
            for _, r in technical.iterrows()
        ],
    )


def test_get_model_prediction_returns_score_and_shap_contributions(tmp_path):
    db_path = tmp_path / "test.db"
    dates = pd.bdate_range(end="2026-07-01", periods=150)

    conn = get_connection(db_path)
    init_db(conn)
    aapl_prices = _seed_prices(conn, "AAPL", dates, seed=1)
    _seed_prices(conn, "SPY", dates, seed=2)
    _seed_features(conn, "AAPL", aapl_prices)
    conn.close()

    train_model("AAPL", db_path=db_path)
    # The tool caches the model/explainer at module scope for server-run
    # efficiency; force a reload so this test sees what was just trained.
    prediction_module._model = None
    prediction_module._feature_columns = None
    prediction_module._explainer = None

    result = get_model_prediction("AAPL", db_path=db_path)

    assert result["ticker"] == "AAPL"
    assert "error" not in result
    assert 0.0 <= result["score"] <= 1.0
    assert result["predicted_label"] in (0, 1)

    contributions = result["feature_contributions"]
    assert {c["feature"] for c in contributions} == set(FEATURE_COLUMNS)
    assert len(contributions) == len(FEATURE_COLUMNS)
    for c in contributions:
        assert set(c.keys()) == {"feature", "value", "shap_value"}
        assert isinstance(c["shap_value"], float)

    # Sorted by contribution magnitude, largest first.
    magnitudes = [abs(c["shap_value"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_get_model_prediction_missing_ticker_returns_error(tmp_path):
    db_path = tmp_path / "test.db"
    dates = pd.bdate_range(end="2026-07-01", periods=150)

    conn = get_connection(db_path)
    init_db(conn)
    aapl_prices = _seed_prices(conn, "AAPL", dates, seed=1)
    _seed_prices(conn, "SPY", dates, seed=2)
    _seed_features(conn, "AAPL", aapl_prices)
    conn.close()

    train_model("AAPL", db_path=db_path)
    prediction_module._model = None
    prediction_module._feature_columns = None
    prediction_module._explainer = None

    result = get_model_prediction("NOPE", db_path=db_path)

    assert result["ticker"] == "NOPE"
    assert "error" in result
