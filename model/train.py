"""Trains an XGBoost binary classifier on the `features` table.

Train/test split is strictly chronological (architecture.md 3.3) — the
earliest `train_frac` of dates go to train, the rest to test. Never a
random split: a random split would let technical indicators computed near
a test-period date (e.g. a 50-day SMA) leak information from overlapping
history into training.
"""

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from data.storage import get_connection, init_db
from model.labeling import DEFAULT_BENCHMARK, DEFAULT_HORIZON, label_features

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.json"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"

FEATURE_COLUMNS = [
    "sma_20",
    "sma_50",
    "rsi_14",
    "volatility_20",
    "sentiment_score",
    "news_count",
    "eps",
    "revenue",
    "revenue_growth",
    "debt_to_equity",
]


def load_labeled_features(ticker: str, db_path=None) -> pd.DataFrame:
    """Load feature rows for `ticker` that already have a label.

    sqlite3 + pandas infers a column's dtype from the values actually
    fetched, so a feature column that happens to be all-NULL in the result
    set (e.g. `sma_50` before 50 days of price history exist) comes back as
    `object` instead of `float64` — and XGBoost rejects `object` columns.
    Explicitly coerce every feature column to numeric, and `label` to a
    nullable integer so a real label of 0 stays distinguishable from
    missing/NA.
    """
    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        df = pd.read_sql_query(
            "SELECT * FROM features WHERE ticker = ? AND label IS NOT NULL ORDER BY date",
            conn,
            params=(ticker,),
        )
    finally:
        conn.close()

    df["date"] = pd.to_datetime(df["date"])
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

    nan_feature_rows = int(df[FEATURE_COLUMNS].isna().any(axis=1).sum())
    labeled_rows = int(df["label"].notna().sum())
    print(
        f"load_labeled_features({ticker}): {len(df)} rows loaded, "
        f"{labeled_rows} with a non-null label, "
        f"{nan_feature_rows} with at least one NaN feature"
    )

    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7):
    """Split `df` (must have a `date` column, may contain multiple tickers
    per date) into train/test using a single global date cutoff — the
    earliest `train_frac` of *unique dates* go to train, the rest to test.
    Splitting on a date cutoff rather than a row-count fraction guarantees
    every row in train has a date strictly earlier than every row in test,
    even with several tickers sharing the same calendar dates — no ticker's
    future rows can end up in another ticker's training set."""
    df = df.sort_values("date").reset_index(drop=True)
    unique_dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    split_idx = int(len(unique_dates) * train_frac)

    if split_idx <= 0:
        return df.iloc[0:0].copy(), df.copy()
    if split_idx >= len(unique_dates):
        return df.copy(), df.iloc[0:0].copy()

    split_date = unique_dates.iloc[split_idx]
    train_df = df[df["date"] < split_date].reset_index(drop=True)
    test_df = df[df["date"] >= split_date].reset_index(drop=True)
    return train_df, test_df


def train_model(
    tickers,
    benchmark: str = DEFAULT_BENCHMARK,
    horizon: int = DEFAULT_HORIZON,
    train_frac: float = 0.7,
    db_path=None,
):
    """Label and load features for each ticker in `tickers` (a single
    ticker string or a list), concatenate them into one basket, and split
    chronologically by date across the whole basket — not per ticker — so
    the same global cutoff date applies to every ticker. Trains one model
    across the combined training rows and saves it plus the feature column
    list to `model/artifacts/`. Returns (model, train_df, test_df)."""
    if isinstance(tickers, str):
        tickers = [tickers]

    frames = []
    for ticker in tickers:
        label_features(ticker, benchmark=benchmark, horizon=horizon, db_path=db_path)
        frames.append(load_labeled_features(ticker, db_path=db_path))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df.empty:
        raise ValueError(
            f"No labeled feature rows for {tickers} — "
            "make sure prices/features/labels have been collected"
        )

    train_df, test_df = chronological_split(df, train_frac=train_frac)
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Not enough labeled rows for {tickers} to form a non-empty "
            f"train/test split (got {len(df)} rows)"
        )

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    FEATURE_COLUMNS_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2))

    return model, train_df, test_df


if __name__ == "__main__":
    import sys

    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL"]
    _, train_df, test_df = train_model(tickers)
    print(f"Trained model for {tickers}: {len(train_df)} train rows, {len(test_df)} test rows")
    print(f"Saved to {MODEL_PATH}")
