"""S&P 500 constituents + small optional core list.

Ticker symbols use **Yahoo Finance** conventions (hyphen, e.g. ``BRK-B``).

Bundled CSV ``sp500_constituents.csv`` is generated from Wikipedia’s S&P list
(Symbol + GICS Sector). Refresh with::

    python -m pipeline.refresh_sp500
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Literal

import pandas as pd
import urllib.request

SP500_CONSTITUENTS_CSV = Path(__file__).resolve().parent / "sp500_constituents.csv"

# Legacy 54-name liquid sleeve (offline-friendly).
CORE_UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ADBE", "CRM",
    "INTC", "CSCO", "ORCL", "AMD", "QCOM", "NFLX", "COST", "PEP", "KO", "WMT",
    "HD", "LOW", "MCD", "SBUX", "NKE", "DIS", "V", "MA", "JPM", "BAC",
    "WFC", "GS", "MS", "C", "BRK-B", "UNH", "JNJ", "PFE", "MRK", "LLY",
    "ABT", "TMO", "DHR", "XOM", "CVX", "COP", "SLB", "CAT", "BA", "DE",
    "GE", "MMM", "HON", "UPS",
    # User-added core names (Yahoo symbols)
    "HOOD", "SOFI", "POET", "SMR", "PWR", "PLTR",
)

CORE_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMZN": "Tech", "GOOGL": "Tech",
    "META": "Tech", "TSLA": "Tech", "AVGO": "Tech", "ADBE": "Tech", "CRM": "Tech",
    "INTC": "Tech", "CSCO": "Tech", "ORCL": "Tech", "AMD": "Tech", "QCOM": "Tech",
    "NFLX": "Tech",
    "COST": "Cons", "PEP": "Cons", "KO": "Cons", "WMT": "Cons", "HD": "Cons",
    "LOW": "Cons", "MCD": "Cons", "SBUX": "Cons", "NKE": "Cons", "DIS": "Cons",
    "V": "Fin", "MA": "Fin", "JPM": "Fin", "BAC": "Fin", "WFC": "Fin",
    "GS": "Fin", "MS": "Fin", "C": "Fin", "BRK-B": "Fin",
    "UNH": "Health", "JNJ": "Health", "PFE": "Health", "MRK": "Health", "LLY": "Health",
    "ABT": "Health", "TMO": "Health", "DHR": "Health",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "CAT": "Indus", "BA": "Indus", "DE": "Indus", "GE": "Indus", "MMM": "Indus",
    "HON": "Indus", "UPS": "Indus",
    "HOOD": "Fin", "SOFI": "Fin",
    "POET": "Tech", "PLTR": "Tech",
    "SMR": "Energy", "PWR": "Indus",
}

INDEX_SECTOR_LABELS: dict[str, str] = {"SPY": "Index", "QQQ": "Index"}

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_UA_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; edu-research/1.0; +https://github.com)"}


def read_sp500_from_csv(path: Path | None = None) -> tuple[tuple[str, ...], dict[str, str]]:
    path = path or SP500_CONSTITUENTS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Missing S&P CSV: {path}")
    df = pd.read_csv(path)
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    tickers = tuple(sorted(df["Symbol"].unique()))
    sector_col = "GICS Sector" if "GICS Sector" in df.columns else df.columns[-1]
    sectors = dict(zip(df["Symbol"], df[sector_col].astype(str)))
    return tickers, sectors


def download_sp500_to_csv(save_path: Path | None = None) -> tuple[tuple[str, ...], dict[str, str]]:
    """Fetch current list from Wikipedia; write CSV; return universe + sectors."""
    save_path = save_path or SP500_CONSTITUENTS_CSV
    req = urllib.request.Request(_WIKI_URL, headers=_UA_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", "replace")
    df = pd.read_html(StringIO(html))[0]
    df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    out = df[["Symbol", "Security", "GICS Sector"]].copy()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    sectors = dict(zip(out["Symbol"], out["GICS Sector"].astype(str)))
    tickers = tuple(sorted(out["Symbol"].unique()))
    return tickers, sectors


def load_universe(
    mode: Literal["sp500", "core"],
    *,
    refresh_sp500_csv: bool = False,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return (tickers tuple, ticker->sector mapping) excluding indices."""
    if mode == "core":
        return CORE_UNIVERSE, dict(CORE_SECTOR_MAP)
    if refresh_sp500_csv:
        return download_sp500_to_csv()
    return read_sp500_from_csv()


def default_universe_bundle(mode: Literal["sp500", "core"] = "sp500",
                            ) -> tuple[tuple[str, ...], dict[str, str]]:
    """Used by ``Config``: ``sp500`` reads bundled CSV; ``core`` is the 54-name list."""
    tickers, sec = load_universe(mode)
    merged = dict(sec)
    merged.update({k: v for k, v in INDEX_SECTOR_LABELS.items()})
    return tickers, merged
