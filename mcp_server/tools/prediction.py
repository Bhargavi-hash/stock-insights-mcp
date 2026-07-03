"""get_model_prediction tool: loads the saved XGBoost model, scores a
ticker's latest feature row, and explains the score with SHAP feature
contributions (shap.TreeExplainer works natively with XGBoost trees).
"""

import json

import pandas as pd
import shap
import xgboost as xgb

from data.storage import get_connection, init_db
from model.train import FEATURE_COLUMNS_PATH, MODEL_PATH

_model = None
_feature_columns = None
_explainer = None


def _load_model():
    """Lazily load and cache the model, feature columns, and SHAP explainer
    — the model file doesn't change between tool calls in a single server
    run, so there's no reason to re-load it from disk every time."""
    global _model, _feature_columns, _explainer
    if _model is None:
        _feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
        _model = xgb.XGBClassifier()
        _model.load_model(str(MODEL_PATH))
        _explainer = shap.TreeExplainer(_model)
    return _model, _feature_columns, _explainer


def get_model_prediction(ticker: str, db_path=None) -> dict:
    """Return the model's outperform-probability score for `ticker`'s
    latest feature row, plus the SHAP contribution of each feature to that
    score, sorted by magnitude (largest impact first)."""
    if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        return {
            "ticker": ticker,
            "error": f"No trained model found at {MODEL_PATH} — run model/train.py first",
        }

    model, feature_columns, explainer = _load_model()

    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        df = pd.read_sql_query(
            "SELECT * FROM features WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            conn,
            params=(ticker,),
        )
    finally:
        conn.close()

    if df.empty:
        return {
            "ticker": ticker,
            "error": f"No feature rows for '{ticker}' — run features/build_features.py first",
        }

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    X = df[feature_columns]

    score = float(model.predict_proba(X)[0][1])
    predicted_label = int(score >= 0.5)

    shap_values = explainer.shap_values(X)[0]
    contributions = [
        {
            "feature": col,
            "value": None if pd.isna(X.iloc[0][col]) else float(X.iloc[0][col]),
            "shap_value": float(value),
        }
        for col, value in zip(feature_columns, shap_values)
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

    return {
        "ticker": ticker,
        "date": df.iloc[0]["date"],
        "score": score,
        "predicted_label": predicted_label,
        "feature_contributions": contributions,
    }
