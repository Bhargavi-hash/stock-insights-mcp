"""SQLite interface and schema definitions for the data collection layer.

Raw data is never mutated in place — corrections are inserted as new rows,
each stamped with the time it was fetched (`fetched_at`).
"""

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "stock_insights.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices (ticker, date);

CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT,
    published_at TEXT,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_published ON news (ticker, published_at);

CREATE TABLE IF NOT EXISTS filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    filing_type TEXT,
    filing_date TEXT,
    data TEXT,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_filings_ticker_date ON filings (ticker, filing_date);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    sma_20 REAL,
    sma_50 REAL,
    rsi_14 REAL,
    volatility_20 REAL,
    sentiment_score REAL,
    news_count INTEGER,
    eps REAL,
    revenue REAL,
    revenue_growth REAL,
    debt_to_equity REAL,
    label INTEGER,
    UNIQUE(ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_features_ticker_date ON features (ticker, date);
"""


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def insert_prices(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert raw OHLCV rows. Each dict needs: ticker, date, open, high, low,
    close, volume, fetched_at. Returns the number of rows inserted."""
    conn.executemany(
        """
        INSERT INTO prices (ticker, date, open, high, low, close, volume, fetched_at)
        VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :fetched_at)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def insert_news(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert raw news rows. Each dict needs: ticker, headline, source, url,
    published_at, fetched_at. Returns the number of rows inserted."""
    conn.executemany(
        """
        INSERT INTO news (ticker, headline, source, url, published_at, fetched_at)
        VALUES (:ticker, :headline, :source, :url, :published_at, :fetched_at)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def insert_filings(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert raw filing rows. Each dict needs: ticker, filing_type,
    filing_date, data, fetched_at. Returns the number of rows inserted."""
    conn.executemany(
        """
        INSERT INTO filings (ticker, filing_type, filing_date, data, fetched_at)
        VALUES (:ticker, :filing_type, :filing_date, :data, :fetched_at)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def insert_features(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Upsert computed feature rows, one per (ticker, date). Unlike the raw
    tables, features are derived and idempotently recomputed, so re-inserting
    a (ticker, date) replaces the row instead of appending. Each dict needs:
    ticker, date, sma_20, sma_50, rsi_14, volatility_20, sentiment_score,
    news_count, eps, revenue, revenue_growth, debt_to_equity, label."""
    conn.executemany(
        """
        INSERT INTO features
            (ticker, date, sma_20, sma_50, rsi_14, volatility_20,
             sentiment_score, news_count, eps, revenue, revenue_growth,
             debt_to_equity, label)
        VALUES
            (:ticker, :date, :sma_20, :sma_50, :rsi_14, :volatility_20,
             :sentiment_score, :news_count, :eps, :revenue, :revenue_growth,
             :debt_to_equity, :label)
        ON CONFLICT(ticker, date) DO UPDATE SET
            sma_20 = excluded.sma_20,
            sma_50 = excluded.sma_50,
            rsi_14 = excluded.rsi_14,
            volatility_20 = excluded.volatility_20,
            sentiment_score = excluded.sentiment_score,
            news_count = excluded.news_count,
            eps = excluded.eps,
            revenue = excluded.revenue,
            revenue_growth = excluded.revenue_growth,
            debt_to_equity = excluded.debt_to_equity,
            label = excluded.label
        """,
        rows,
    )
    conn.commit()
    return len(rows)
