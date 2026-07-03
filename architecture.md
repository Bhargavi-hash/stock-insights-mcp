# Stock Insight MCP — Architecture

## 1. Goal

Build an end-to-end system that scores whether a stock looks favorable to buy based on
technical, fundamental, and news-sentiment signals, and exposes that score — along with
the evidence behind it — to Claude via an MCP server so Claude can produce a
human-readable explanation.

**This project explicitly does NOT:**
- Execute trades or connect to a brokerage
- Predict exact future prices
- Claim to beat the market or provide financial advice
- Run continuously / in real time (batch/on-demand only, for now)

**Primary learning goals** (keep visible — this shapes scope decisions):
- Practice a real ML pipeline: data collection → features → model → evaluation
- Learn proper time-series train/test discipline (avoid lookahead bias)
- Learn MCP server design: tools as single-purpose, deterministic functions
- Learn to work with Claude Code / Cursor on a multi-layer project using an
  architecture doc as the source of truth

---

## 2. High-level data flow

```
 ┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────────────────┐
 │ Data Collection  │ --> │ Feature Building  │ --> │ ML Model    │ --> │ MCP Server        │
 │ (prices, news,   │     │ (technical, senti-│     │ (XGBoost    │     │ (tools exposing   │
 │  filings)        │     │  ment, fundamental)│     │  classifier)│     │  data + score)     │
 └─────────────────┘     └──────────────────┘     └─────────────┘     └──────────────────┘
                                                                                │
                                                                                v
                                                                        ┌──────────────┐
                                                                        │ Claude (client)│
                                                                        │ orchestrates   │
                                                                        │ tool calls +   │
                                                                        │ writes the      │
                                                                        │ explanation     │
                                                                        └──────────────┘
```

Design principle: **the ML model produces a number. The LLM explains the number.**
Never ask one component to do both — it makes debugging and evaluation much harder.

---

## 3. Components

### 3.1 Data Collection Layer (`data/collectors/`)

Responsible for pulling raw data and storing it, unmodified, with a timestamp.

| Source | Method | Notes |
|---|---|---|
| Price/volume (OHLCV) | `yfinance` API | Not scraped — exchanges rate-limit/block scrapers |
| News headlines | News API (e.g. Finnhub/NewsAPI) or RSS crawler | Crawler only for sources without an API |
| SEC filings | SEC EDGAR public API | Free, structured, no scraping needed |

Output: raw rows written to SQLite tables `prices`, `news`, `filings`, each with a
`fetched_at` timestamp. Raw data is never mutated in place — corrections are new rows.

### 3.2 Feature Layer (`features/build_features.py`)

Converts raw data into a single feature table, one row per `(ticker, date)`.

- **Technical**: moving averages, RSI, volatility, volume trend (via `pandas`/`ta`)
- **Sentiment**: score each news item (-1 to 1), aggregate per day/ticker
- **Fundamental**: P/E, revenue growth, debt/equity, pulled from filings

**Data contract**: output is a DataFrame / table `features` with columns:
`ticker, date, <technical_cols>, <sentiment_cols>, <fundamental_cols>, label`
`label` is only populated for historical rows used in training (see 3.3).

### 3.3 Model Layer (`model/`)

- **Model**: gradient boosting (XGBoost or LightGBM) — tabular data, small dataset,
  interpretable via feature importance / SHAP. Not a neural net (dataset too small,
  not worth the complexity for a learning project).
- **Target/label**: binary — "did this stock outperform its sector benchmark over the
  next N trading days?" (N configurable, default 10). Outperformance vs. a benchmark
  is more honest than raw direction, and avoids rewarding the model for market-wide moves.
- **Train/test split**: strictly chronological. Train on data up to date `T`, validate
  on `T` to `T+k`, test on the most recent unseen window. **Never** a random split —
  random splits leak future information into training via overlapping technical
  indicators.
- **Evaluation**: compare against a naive baseline (e.g., "always predict outperform"
  and "buy-and-hold sector ETF"). Report precision/recall, not just accuracy — classes
  are likely imbalanced.
- **Artifacts**: trained model saved to `model/artifacts/model.json` +
  `feature_columns.json` (so the MCP layer knows the expected input shape).

### 3.4 MCP Server (`mcp_server/`)

Exposes read-only tools. Each tool does one deterministic thing — no tool should
itself call Claude or make judgment calls; that's the client's job.

| Tool | Input | Output |
|---|---|---|
| `get_price_history` | ticker, range | OHLCV rows |
| `get_technical_signals` | ticker | latest technical feature values |
| `get_recent_news` | ticker, days | headlines + sentiment scores |
| `get_filing_summary` | ticker | latest fundamental figures |
| `get_model_prediction` | ticker | score (0-1), label, top feature contributions (SHAP) |

Claude (the client) calls these tools, gathers the score and supporting evidence, and
writes the final explanation. The MCP server itself never generates prose.

---

## 4. Repo structure

```
stock-insight-mcp/
├── architecture.md
├── data/
│   ├── collectors/
│   │   ├── prices.py
│   │   ├── news.py
│   │   └── filings.py
│   └── storage.py          # SQLite interface, schema definitions
├── features/
│   └── build_features.py
├── model/
│   ├── train.py
│   ├── evaluate.py         # backtesting + baseline comparison
│   └── artifacts/
├── mcp_server/
│   ├── server.py
│   └── tools/
│       ├── prices.py
│       ├── technicals.py
│       ├── news.py
│       ├── filings.py
│       └── prediction.py
├── tests/
│   ├── test_collectors.py
│   ├── test_features.py
│   └── test_model_eval.py  # checks for lookahead bias / leakage
└── notebooks/               # exploration only — never imported by production code
```

---

## 5. Key decisions and why

- **SQLite, not Postgres** — single-user, local, learning project. No concurrent
  writers. Swap later only if there's an actual reason.
- **XGBoost, not a neural net** — small tabular dataset; boosting is easier to debug,
  faster to train, and gives interpretable feature importances out of the box.
- **Outperformance-vs-benchmark label, not raw price/direction** — raw price
  regression overstates precision; raw direction ignores market-wide moves.
- **MCP tools are read-only and single-purpose** — keeps the boundary between
  "compute" (deterministic, testable) and "reason" (Claude's job) clean.
- **Crawler used only where no API exists** — prefer official APIs (yfinance, SEC
  EDGAR) for reliability and to avoid ToS/scraping issues.

---

## 6. Non-goals / constraints

- No trade execution, no brokerage integration, no real-money involvement
- No real-time/streaming pipeline — batch/on-demand refresh only
- No guarantee of predictive accuracy — this is a directional research signal, not
  investment advice
- Respect API rate limits; cache aggressively rather than re-fetching

---

## 7. Open questions (fill in as decided)

- [ ] Which tickers/universe to start with (single stock vs. small basket)?
- [ ] Prediction horizon N (days) — default 10, confirm after first backtest
- [ ] Sentiment scoring: prompt-based LLM scoring vs. a small fine-tuned classifier?
- [ ] Refresh cadence — daily batch job, or manual trigger only?
- [ ] How to version model artifacts as features evolve (schema drift)?

---

## 8. How to use this doc with Claude Code / Cursor

- Point the agent at this file first: *"Read architecture.md before making any
  changes."*
- Build one layer at a time (collection → features → model → MCP), verifying each
  with tests before moving to the next.
- When the agent proposes a design change (e.g., swapping SQLite for Postgres),
  check it against Section 5 first — if it contradicts a documented decision, either
  push back or update this doc deliberately, not implicitly.
- Update the "Open questions" section as decisions get made, so the doc stays the
  source of truth rather than going stale.

  