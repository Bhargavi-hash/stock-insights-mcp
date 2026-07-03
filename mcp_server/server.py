"""MCP server entry point (architecture.md 3.4).

Exposes read-only, single-purpose tools over the local SQLite data + the
trained model. Each tool returns structured JSON — it never generates
prose. Claude (the MCP client) calls these tools, gathers the score and
supporting evidence, and writes the explanation; that separation is the
whole point (architecture.md Section 2: "the ML model produces a number,
the LLM explains the number").

Run locally (stdio transport, the default Claude Desktop expects):

    cd stock-insight-mcp
    source .venv/bin/activate
    python -m mcp_server.server

Or with the MCP CLI's inspector, for interactive debugging:

    mcp dev mcp_server/server.py

To connect it to Claude Desktop, add an entry to
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\\Claude\\claude_desktop_config.json` (Windows):

    {
      "mcpServers": {
        "stock-insights": {
          "command": "/absolute/path/to/stock-insight-mcp/.venv/bin/python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "/absolute/path/to/stock-insight-mcp"
        }
      }
    }

Then restart Claude Desktop. Data must already be collected/featurized/
trained (see data/collectors/, features/build_features.py, model/train.py)
before the tools have anything to return — the server itself never fetches
from external APIs.
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from mcp_server.tools.filings import get_filing_summary as _get_filing_summary
from mcp_server.tools.news import get_recent_news as _get_recent_news
from mcp_server.tools.prediction import get_model_prediction as _get_model_prediction
from mcp_server.tools.prices import get_price_history as _get_price_history
from mcp_server.tools.technicals import get_technical_signals as _get_technical_signals

mcp = FastMCP(
    name="stock-insights",
    instructions=(
        "Tools for researching a stock's technicals, recent news sentiment, "
        "fundamentals, and a model-derived outperformance score. Every tool "
        "returns structured data only, never prose — synthesize the "
        "explanation yourself from the returned evidence."
    ),
)

# All tools here are read-only queries against local data — none of them
# mutate state or call external services, so they share the same hints.
_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)


@mcp.tool(name="get_price_history", annotations=_READ_ONLY)
def get_price_history(ticker: str, days: int = 30) -> dict:
    """Recent OHLCV rows for `ticker`, oldest first."""
    return _get_price_history(ticker, days=days)


@mcp.tool(name="get_technical_signals", annotations=_READ_ONLY)
def get_technical_signals(ticker: str) -> dict:
    """Latest technical indicator values for `ticker`: sma_20, sma_50,
    rsi_14, volatility_20."""
    return _get_technical_signals(ticker)


@mcp.tool(name="get_recent_news", annotations=_READ_ONLY)
def get_recent_news(ticker: str, days: int = 7) -> dict:
    """Recent headlines for `ticker` with per-headline sentiment scores."""
    return _get_recent_news(ticker, days=days)


@mcp.tool(name="get_filing_summary", annotations=_READ_ONLY)
def get_filing_summary(ticker: str) -> dict:
    """Latest fundamental figures for `ticker`: eps, revenue,
    revenue_growth, debt_to_equity."""
    return _get_filing_summary(ticker)


@mcp.tool(name="get_model_prediction", annotations=_READ_ONLY)
def get_model_prediction(ticker: str) -> dict:
    """Model outperformance score (0-1) for `ticker`'s latest feature row,
    with SHAP values showing which features drove the score."""
    return _get_model_prediction(ticker)


if __name__ == "__main__":
    mcp.run(transport="stdio")
