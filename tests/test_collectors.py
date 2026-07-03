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
