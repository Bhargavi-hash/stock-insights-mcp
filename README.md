# Stock Insights MCP

An end-to-end learning project: collect stock market data, engineer features, train
an ML model to score whether a stock looks favorable, and expose it all to Claude via
an MCP (Model Context Protocol) server so Claude can explain the *why* behind the score.

See [`architecture.md`](./architecture.md) for the full system design, key decisions,
and non-goals.

**This is not financial advice.** The model produces a directional research signal
based on technical, sentiment, and fundamental features — not a price prediction, and
not a recommendation to trade.

---

## What this project does

```
Data Collection  -->  Feature Building  -->  ML Model  -->  MCP Server  -->  Claude
(prices, news,        (technical,            (XGBoost      (5 tools          (explains
 filings)              sentiment,             classifier)   exposing data     the score)
                        fundamental)                         + prediction)
```

- **Data collection** — pulls OHLCV prices (yfinance), news headlines (NewsAPI), and
  SEC filings (EDGAR) for a list of tickers, storing raw data in SQLite.
- **Feature building** — turns raw data into technical indicators (SMA, RSI,
  volatility), sentiment scores (vaderSentiment), and fundamentals (EPS, revenue
  growth, debt/equity from SEC's companyfacts API).
- **Model** — an XGBoost classifier predicting whether a stock will outperform a
  benchmark (SPY) over the next N trading days, trained with a strict chronological
  train/test split to avoid lookahead bias.
- **MCP server** — exposes read-only tools (price history, technical signals, news,
  filing summary, model prediction with SHAP explainability) that Claude can call to
  gather evidence and write a grounded explanation.

---

## Prerequisites

- Python 3.10+
- A virtual environment (recommended — see setup below)
- Free API keys:
  - [NewsAPI](https://newsapi.org) (or your chosen news source) for headlines
  - No key needed for yfinance or SEC EDGAR, but EDGAR requires a descriptive
    `User-Agent` (your name + a real contact email — see below)
- [Claude Desktop](https://claude.ai/download) if you want to connect the MCP server
  to Claude directly (Ubuntu 22.04+ / Debian 12+, x86_64 or arm64)

---

## Setup

```bash
# Clone / enter the project
cd stock-insights-mcp

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment variables

```bash
export NEWSAPI_KEY="your_newsapi_key_here"
export SEC_EDGAR_USER_AGENT="Your Name your.email@example.com"
```

Add these to your shell profile (`~/.bashrc`) or a `.env` file (gitignored) so you
don't have to re-export them every session.

**Note on `SEC_EDGAR_USER_AGENT`:** this isn't an API key — SEC just requires a real
name and working email in the User-Agent header on every request. Use one you
actually check.

---

## Running the pipeline

All commands below assume your venv is activated (`source venv/bin/activate`).

### 1. Collect raw data for a ticker

```bash
python3 -c "
from data.collectors.prices import collect_prices
from data.collectors.news import collect_news
from data.collectors.filings import collect_filings

collect_prices('AAPL')
collect_prices('SPY')   # benchmark — required for labeling
collect_news('AAPL')
collect_filings('AAPL')
"
```

Repeat for each ticker you want in your dataset. Data lands in `stock_insights.db`
(SQLite) in the project root.

### 2. Build features

```bash
python3 -c "
from features.build_features import collect_features
collect_features('AAPL')
"
```

Writes a row per `(ticker, date)` into the `features` table — technical indicators,
sentiment score, and forward-filled fundamentals.

### 3. Train the model

```bash
python3 -c "
from model.train import train_model
print(train_model('AAPL'))
"
```

Trains an XGBoost classifier on a chronological train/test split and saves the model
to `model/artifacts/model.json` plus the expected feature columns to
`model/artifacts/feature_columns.json`.

### 4. Evaluate

```bash
python3 -c "
from model.evaluate import evaluate_model
print(evaluate_model('AAPL'))
"
```

Reports the model's precision/recall/accuracy against two baselines
("always predict outperform" and "buy-and-hold benchmark") on the held-out,
most-recent chronological test window.

### 5. Run the MCP server

```bash
python3 mcp_server/server.py
```

Exposes five tools: `get_price_history`, `get_technical_signals`, `get_recent_news`,
`get_filing_summary`, and `get_model_prediction` (includes SHAP feature
contributions).

---

## Connecting to Claude Desktop

1. Install Claude Desktop (Ubuntu/Debian):
   ```bash
   sudo curl -fsSLo /usr/share/keyrings/claude-desktop-archive-keyring.asc \
     https://downloads.claude.ai/claude-desktop/key.asc
   echo "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/claude-desktop-archive-keyring.asc] https://downloads.claude.ai/claude-desktop/apt/stable stable main" \
     | sudo tee /etc/apt/sources.list.d/claude-desktop.list
   sudo apt update && sudo apt install claude-desktop
   ```

2. Open (or create) `~/.config/Claude/claude_desktop_config.json` and add an entry
   pointing at this project's venv Python and server script:

   ```json
   {
     "mcpServers": {
       "stock-insights": {
         "command": "/absolute/path/to/stock-insights-mcp/venv/bin/python3",
         "args": ["/absolute/path/to/stock-insights-mcp/mcp_server/server.py"]
       }
     }
   }
   ```

3. Restart Claude Desktop. Your five tools should now be available for Claude to call.

4. Try asking: *"Should I consider buying AAPL based on the current data?"* — Claude
   should call `get_model_prediction`, `get_recent_news`, and related tools, then
   write an explanation grounded in the evidence it retrieved.

---

## Project structure

```
stock-insights-mcp/
├── architecture.md          # full system design, decisions, non-goals
├── README.md                # this file
├── requirements.txt
├── stock_insights.db        # SQLite data (gitignored)
├── data/
│   ├── storage.py           # SQLite schema + insert helpers
│   └── collectors/
│       ├── prices.py        # yfinance OHLCV collector
│       ├── news.py          # NewsAPI headline collector
│       └── filings.py       # SEC EDGAR filings collector
├── features/
│   └── build_features.py    # technical + sentiment + fundamental features
├── model/
│   ├── labeling.py          # outperformance-vs-benchmark label logic
│   ├── train.py             # chronological train/test split + XGBoost training
│   ├── evaluate.py          # metrics + baseline comparison
│   └── artifacts/           # saved model + feature column list
├── mcp_server/
│   ├── server.py            # MCP entry point
│   └── tools/                # one file per tool
├── tests/
└── notebooks/                # exploration only, not production logic
```

---

## Known limitations

- Trained on a small basket of tickers with a few hundred rows each — not enough
  data for the model's predictions to be considered reliable signal.
- Fundamentals are sparse (quarterly filings forward-filled onto daily rows), so
  `revenue_growth`/`eps`/`debt_to_equity` carry less resolution than
  price-derived features.
- No live/streaming updates — data must be refreshed manually by re-running the
  collectors.
- Sentiment scoring uses a general-purpose lexicon (vaderSentiment), not a
  finance-tuned model — headline sentiment may not always reflect market-relevant
  tone accurately.

See `architecture.md` Section 6 for the full list of non-goals and constraints.

---

## License / disclaimer

This is a personal learning project, not investment advice. Predictions and
explanations produced by this system should not be used as the sole basis for any
financial decision.