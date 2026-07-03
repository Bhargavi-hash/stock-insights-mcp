from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data.storage import get_connection, init_db, insert_features, insert_prices
from features.build_features import build_technical_features
from model.evaluate import evaluate_model
from model.train import chronological_split, train_model


def test_chronological_split_no_date_leakage():
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    df = pd.DataFrame({"date": dates, "value": range(100)})
    # Shuffle input rows to prove the split doesn't depend on incoming order.
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train, test = chronological_split(shuffled, train_frac=0.7)

    assert len(train) == 70
    assert len(test) == 30
    assert train["date"].max() < test["date"].min()
    assert set(train["date"]).isdisjoint(set(test["date"]))
    assert set(train["date"]) | set(test["date"]) == set(dates)


def _seed_prices(conn, ticker: str, dates, seed: int):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(loc=0.1, scale=1.0, size=len(dates)))
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [
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
    ]
    insert_prices(conn, rows)
    return pd.DataFrame({"date": dates, "close": close})


def _seed_features(conn, ticker: str, prices: pd.DataFrame):
    technical = build_technical_features(prices)
    rows = []
    for _, r in technical.iterrows():
        rows.append(
            {
                "ticker": ticker,
                "date": r["date"].strftime("%Y-%m-%d"),
                "sma_20": None if pd.isna(r["sma_20"]) else r["sma_20"],
                "sma_50": None if pd.isna(r["sma_50"]) else r["sma_50"],
                "rsi_14": None if pd.isna(r["rsi_14"]) else r["rsi_14"],
                "volatility_20": None if pd.isna(r["volatility_20"]) else r["volatility_20"],
                "sentiment_score": 0.0,
                "news_count": 0,
                "eps": 1.5,
                "revenue": 1_000_000,
                "revenue_growth": 0.05,
                "debt_to_equity": 1.2,
                "label": None,
            }
        )
    insert_features(conn, rows)


def test_train_and_evaluate_end_to_end_no_leakage(tmp_path):
    db_path = tmp_path / "test.db"
    dates = pd.bdate_range(end="2026-07-01", periods=150)

    conn = get_connection(db_path)
    init_db(conn)
    test_prices = _seed_prices(conn, "TEST", dates, seed=1)
    _seed_prices(conn, "SPY", dates, seed=2)
    _seed_features(conn, "TEST", test_prices)
    conn.close()

    model, train_df, test_df = train_model("TEST", db_path=db_path)

    # Chronological, non-overlapping split.
    assert train_df["date"].max() < test_df["date"].min()
    assert set(train_df["date"]).isdisjoint(set(test_df["date"]))
    assert len(train_df) > 0 and len(test_df) > 0

    report = evaluate_model("TEST", db_path=db_path)

    assert report["n_test"] == len(test_df)
    for key in ("model", "baseline_always_outperform", "baseline_buy_and_hold_benchmark"):
        metrics = report[key]
        for metric_name in ("accuracy", "precision", "recall"):
            assert 0.0 <= metrics[metric_name] <= 1.0

    # "always predict outperform" trivially has perfect recall.
    assert report["baseline_always_outperform"]["recall"] == 1.0
    # "buy-and-hold benchmark" never predicts outperform, so recall is 0.
    assert report["baseline_buy_and_hold_benchmark"]["recall"] == 0.0


def test_train_and_evaluate_multi_ticker_no_cross_ticker_leakage(tmp_path):
    db_path = tmp_path / "test.db"
    dates = pd.bdate_range(end="2026-07-01", periods=150)

    conn = get_connection(db_path)
    init_db(conn)
    prices_a = _seed_prices(conn, "TICKA", dates, seed=1)
    prices_b = _seed_prices(conn, "TICKB", dates, seed=3)
    _seed_prices(conn, "SPY", dates, seed=2)
    _seed_features(conn, "TICKA", prices_a)
    _seed_features(conn, "TICKB", prices_b)
    conn.close()

    model, train_df, test_df = train_model(["TICKA", "TICKB"], db_path=db_path)

    # Both tickers actually made it into the combined basket.
    assert set(train_df["ticker"]) == {"TICKA", "TICKB"}
    assert set(test_df["ticker"]) == {"TICKA", "TICKB"}

    # Single global date cutoff — no ticker's rows leak across it.
    assert train_df["date"].max() < test_df["date"].min()
    assert set(train_df["date"]).isdisjoint(set(test_df["date"]))

    # Explicitly: neither ticker's test dates precede the other ticker's
    # train dates (the failure mode of a per-ticker fractional split).
    for ticker in ("TICKA", "TICKB"):
        other = "TICKB" if ticker == "TICKA" else "TICKA"
        own_test_min = test_df.loc[test_df["ticker"] == ticker, "date"].min()
        other_train_max = train_df.loc[train_df["ticker"] == other, "date"].max()
        assert other_train_max < own_test_min

    report = evaluate_model(["TICKA", "TICKB"], db_path=db_path)

    assert set(report["per_ticker_accuracy"].keys()) == {"TICKA", "TICKB"}
    per_ticker_n = sum(m["n_test"] for m in report["per_ticker_accuracy"].values())
    assert per_ticker_n == report["n_test"]
    for m in report["per_ticker_accuracy"].values():
        assert 0.0 <= m["accuracy"] <= 1.0
