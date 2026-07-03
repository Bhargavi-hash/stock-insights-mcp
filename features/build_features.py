"""Builds the single (ticker, date) feature table from raw prices, news, and
SEC EDGAR fundamentals, per architecture.md Section 3.2.

Technical features come from the local `prices` table. Sentiment features
come from the local `news` table, scored with VADER (no API calls needed).
Fundamental features are pulled live from SEC EDGAR's companyfacts API
rather than the `filings` table, since `filings` currently stores filing
metadata only, not parsed financial figures.

`label` is left NULL here — it gets filled in by the model layer, which
knows the prediction horizon and benchmark to compare against.
"""

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data.collectors.filings import _get_user_agent, _lookup_cik
from data.storage import get_connection, init_db, insert_features

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# SEC XBRL tags vary by filer; try the most common ones in order.
EPS_TAGS = ["EarningsPerShareBasic", "EarningsPerShareDiluted"]
REVENUE_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]
LIABILITIES_TAGS = ["Liabilities"]
EQUITY_TAGS = ["StockholdersEquity"]

FEATURE_COLUMNS = [
    "ticker",
    "date",
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
    "label",
]

_sentiment_analyzer = SentimentIntensityAnalyzer()


def load_prices(conn, ticker: str) -> pd.DataFrame:
    """Load raw OHLCV rows for `ticker`, one row per date (later fetches win
    on duplicate dates)."""
    df = pd.read_sql_query(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date, id",
        conn,
        params=(ticker,),
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(subset="date", keep="last").reset_index(drop=True)


def load_news(conn, ticker: str) -> pd.DataFrame:
    """Load raw headlines for `ticker`."""
    return pd.read_sql_query(
        "SELECT headline, published_at FROM news WHERE ticker = ?",
        conn,
        params=(ticker,),
    )


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def build_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """20d/50d moving averages, RSI(14), and 20d rolling volatility."""
    df = prices.sort_values("date").copy()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["rsi_14"] = _rsi(df["close"], period=14)
    df["volatility_20"] = df["close"].pct_change().rolling(window=20).std()
    return df[["date", "sma_20", "sma_50", "rsi_14", "volatility_20"]]


def build_sentiment_features(news: pd.DataFrame) -> pd.DataFrame:
    """VADER compound sentiment, averaged per calendar day."""
    if news.empty:
        return pd.DataFrame(columns=["date", "sentiment_score", "news_count"])

    df = news.copy()
    df["date"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"])
    df["sentiment"] = df["headline"].apply(
        lambda h: _sentiment_analyzer.polarity_scores(h)["compound"]
    )
    daily = (
        df.groupby("date")
        .agg(sentiment_score=("sentiment", "mean"), news_count=("sentiment", "count"))
        .reset_index()
    )
    return daily


def fetch_company_facts(ticker: str) -> dict:
    """Pull the full companyfacts payload for `ticker` from SEC EDGAR."""
    user_agent = _get_user_agent()
    cik = _lookup_cik(ticker, user_agent)
    response = requests.get(
        COMPANY_FACTS_URL.format(cik=cik), headers={"User-Agent": user_agent}, timeout=10
    )
    response.raise_for_status()
    return response.json()


def _extract_concept_series(facts: dict, tags: list[str]) -> pd.DataFrame:
    """Return a (end, val) DataFrame for the first XBRL tag in `tags` that
    has 10-K/10-Q data, deduped to one value per period end."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        concept = us_gaap.get(tag)
        if not concept:
            continue
        for unit_values in concept.get("units", {}).values():
            records = [
                {"end": v["end"], "val": v["val"]}
                for v in unit_values
                if v.get("form") in ("10-K", "10-Q") and "end" in v and "val" in v
            ]
            if records:
                df = pd.DataFrame(records)
                df["end"] = pd.to_datetime(df["end"])
                return df.drop_duplicates(subset="end", keep="last").sort_values("end")
    return pd.DataFrame(columns=["end", "val"])


def build_fundamental_features(facts: dict) -> pd.DataFrame:
    """EPS, revenue + YoY revenue growth, and debt/equity, one row per
    filing period end. Values are later forward-filled onto daily rows."""
    eps = _extract_concept_series(facts, EPS_TAGS).rename(columns={"val": "eps"})
    revenue = _extract_concept_series(facts, REVENUE_TAGS).rename(columns={"val": "revenue"})
    liabilities = _extract_concept_series(facts, LIABILITIES_TAGS).rename(
        columns={"val": "liabilities"}
    )
    equity = _extract_concept_series(facts, EQUITY_TAGS).rename(columns={"val": "equity"})

    revenue = revenue.sort_values("end")
    revenue["revenue_growth"] = revenue["revenue"].pct_change()

    merged = eps[["end", "eps"]] if not eps.empty else pd.DataFrame(columns=["end", "eps"])
    for other in (
        revenue[["end", "revenue", "revenue_growth"]],
        liabilities[["end", "liabilities"]],
        equity[["end", "equity"]],
    ):
        merged = merged.merge(other, on="end", how="outer")

    merged = merged.sort_values("end").reset_index(drop=True)
    merged["debt_to_equity"] = (
        merged["liabilities"] / merged["equity"] if not merged.empty else pd.NA
    )
    merged = merged.rename(columns={"end": "date"})
    for col in ("eps", "revenue", "revenue_growth", "debt_to_equity"):
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged[["date", "eps", "revenue", "revenue_growth", "debt_to_equity"]]


def build_features(ticker: str, db_path=None) -> pd.DataFrame:
    """Assemble the (ticker, date) feature table for `ticker`. Requires
    prices to already be collected; news and fundamentals are optional and
    contribute NaN columns if unavailable."""
    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        prices = load_prices(conn, ticker)
        news = load_news(conn, ticker)
    finally:
        conn.close()

    if prices.empty:
        raise ValueError(
            f"No price data found for ticker '{ticker}' — run collect_prices first"
        )

    features = build_technical_features(prices)
    features.insert(0, "ticker", ticker)

    sentiment = build_sentiment_features(news)
    features = features.merge(sentiment, on="date", how="left")

    facts = fetch_company_facts(ticker)
    fundamentals = build_fundamental_features(facts)
    if not fundamentals.empty:
        features = pd.merge_asof(
            features.sort_values("date"),
            fundamentals.sort_values("date"),
            on="date",
            direction="backward",
        )
    else:
        for col in ("eps", "revenue", "revenue_growth", "debt_to_equity"):
            features[col] = pd.NA

    features["label"] = pd.NA
    return features[FEATURE_COLUMNS].sort_values("date").reset_index(drop=True)


def collect_features(ticker: str, db_path=None) -> int:
    """Build features for `ticker` and upsert them into the `features`
    table. Returns rows written."""
    features = build_features(ticker, db_path=db_path)

    rows = features.copy()
    rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
    rows = rows.astype(object).where(pd.notnull(rows), None)
    row_dicts = rows.to_dict(orient="records")

    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        return insert_features(conn, row_dicts)
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    count = collect_features(ticker)
    print(f"Wrote {count} rows for {ticker}")
