"""Evaluates the trained model on the chronological test split against two
baselines (architecture.md 3.3):

- "always predict outperform": a trivial classifier that always predicts 1.
- "buy-and-hold benchmark": since the label is defined as "did the ticker
  beat the benchmark", holding the benchmark itself never beats itself —
  so this baseline is equivalent to always predicting 0.

Accuracy alone is misleading if outperform/underperform classes are
imbalanced, so precision and recall are reported for every candidate.
"""

import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

from model.labeling import DEFAULT_BENCHMARK, DEFAULT_HORIZON, compute_labels
from model.train import (
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    chronological_split,
    load_labeled_features,
)


def load_model():
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model, feature_columns


def _classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(
    tickers,
    benchmark: str = DEFAULT_BENCHMARK,
    horizon: int = DEFAULT_HORIZON,
    train_frac: float = 0.7,
    db_path=None,
) -> dict:
    """Score the saved model against the chronological test split for the
    combined `tickers` basket (a single ticker string or a list) — the same
    global date cutoff `train_model` used, applied across every ticker.
    Returns a report dict with model metrics, both baselines, realized
    average returns over the test period, and a per-ticker test-accuracy
    breakdown."""
    if isinstance(tickers, str):
        tickers = [tickers]

    model, feature_columns = load_model()

    feature_frames = [load_labeled_features(t, db_path=db_path) for t in tickers]
    df = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    _, test_df = chronological_split(df, train_frac=train_frac)

    X_test, y_test = test_df[feature_columns], test_df["label"]
    y_pred = pd.Series(model.predict(X_test), index=test_df.index)

    always_outperform = pd.Series(1, index=y_test.index)
    buy_and_hold_benchmark = pd.Series(0, index=y_test.index)

    return_frames = [
        compute_labels(t, benchmark=benchmark, horizon=horizon, db_path=db_path)
        for t in tickers
    ]
    returns = pd.concat(return_frames, ignore_index=True) if return_frames else pd.DataFrame()
    test_returns = returns.merge(test_df[["ticker", "date"]], on=["ticker", "date"], how="inner")

    per_ticker_accuracy = {}
    for ticker in tickers:
        mask = test_df["ticker"] == ticker
        if mask.sum() == 0:
            continue
        per_ticker_accuracy[ticker] = {
            "n_test": int(mask.sum()),
            "accuracy": accuracy_score(y_test[mask], y_pred[mask]),
        }

    return {
        "tickers": tickers,
        "n_test": len(y_test),
        "model": _classification_metrics(y_test, y_pred),
        "baseline_always_outperform": _classification_metrics(y_test, always_outperform),
        "baseline_buy_and_hold_benchmark": _classification_metrics(
            y_test, buy_and_hold_benchmark
        ),
        "avg_ticker_return_test_period": test_returns["ticker_return"].mean(),
        "avg_benchmark_return_test_period": test_returns["benchmark_return"].mean(),
        "per_ticker_accuracy": per_ticker_accuracy,
    }


def _print_report(tickers, report: dict) -> None:
    print(f"Evaluation for {tickers} — {report['n_test']} test rows")
    for name in ("model", "baseline_always_outperform", "baseline_buy_and_hold_benchmark"):
        m = report[name]
        print(
            f"  {name:35s} accuracy={m['accuracy']:.3f} "
            f"precision={m['precision']:.3f} recall={m['recall']:.3f}"
        )
    print(f"  avg ticker return (test period):    {report['avg_ticker_return_test_period']:.4f}")
    print(f"  avg benchmark return (test period):  {report['avg_benchmark_return_test_period']:.4f}")
    print("  per-ticker test accuracy:")
    for ticker, m in report["per_ticker_accuracy"].items():
        print(f"    {ticker:8s} n={m['n_test']:4d} accuracy={m['accuracy']:.3f}")


if __name__ == "__main__":
    import sys

    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL"]
    _print_report(tickers, evaluate_model(tickers))
