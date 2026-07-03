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
