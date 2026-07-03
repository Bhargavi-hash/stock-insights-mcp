"""Fetches recent 10-K/10-Q filings via SEC EDGAR and writes them to the
`filings` table.

SEC EDGAR requires a descriptive User-Agent on every request (name + contact
email) — see https://www.sec.gov/os/webmaster-faq#developers.
"""

import json
import os
from datetime import datetime, timezone

import requests

from data.storage import get_connection, init_db, insert_filings

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

DEFAULT_FORM_TYPES = ("10-K", "10-Q")


def _get_user_agent() -> str:
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT")
    if not user_agent:
        raise RuntimeError("SEC_EDGAR_USER_AGENT environment variable is not set")
    return user_agent


def _lookup_cik(ticker: str, user_agent: str) -> str:
    """Resolve a ticker to its zero-padded 10-digit CIK via SEC's ticker map."""
    response = requests.get(TICKER_CIK_URL, headers={"User-Agent": user_agent}, timeout=10)
    response.raise_for_status()
    mapping = response.json()

    ticker_upper = ticker.upper()
    for entry in mapping.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"No CIK found for ticker '{ticker}'")


def fetch_filings(
    ticker: str, form_types=DEFAULT_FORM_TYPES, limit: int = 10
) -> list[dict]:
    """Pull the most recent filings of `form_types` for `ticker` from SEC
    EDGAR. Returns a list of dicts with form, filing_date, accession_number,
    primary_document, and cik."""
    user_agent = _get_user_agent()
    cik = _lookup_cik(ticker, user_agent)

    response = requests.get(
        SUBMISSIONS_URL.format(cik=cik), headers={"User-Agent": user_agent}, timeout=10
    )
    response.raise_for_status()
    recent = response.json().get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form not in form_types:
            continue
        filings.append(
            {
                "form": form,
                "filing_date": filing_dates[i],
                "accession_number": accession_numbers[i],
                "primary_document": primary_documents[i],
                "cik": cik,
            }
        )
        if len(filings) >= limit:
            break

    return filings


def to_rows(ticker: str, filings: list[dict]) -> list[dict]:
    """Convert raw SEC filing dicts into row dicts for `insert_filings`."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for filing in filings:
        rows.append(
            {
                "ticker": ticker,
                "filing_type": filing["form"],
                "filing_date": filing["filing_date"],
                "data": json.dumps(filing),
                "fetched_at": fetched_at,
            }
        )
    return rows


def collect_filings(
    ticker: str, form_types=DEFAULT_FORM_TYPES, limit: int = 10, db_path=None
) -> int:
    """Fetch recent filings for `ticker` and store them in SQLite. Returns
    rows written."""
    filings = fetch_filings(ticker, form_types=form_types, limit=limit)
    rows = to_rows(ticker, filings)

    conn = get_connection(db_path) if db_path else get_connection()
    try:
        init_db(conn)
        return insert_filings(conn, rows)
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    count = collect_filings(ticker)
    print(f"Wrote {count} rows for {ticker}")
