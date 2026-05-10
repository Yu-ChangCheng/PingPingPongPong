"""Daily OHLCV downloader with disk cache."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def _extract_ticker_frame(raw: pd.DataFrame, tkr: str) -> pd.DataFrame | None:
    """Normalize yfinance output for one symbol (batch MultiIndex vs single flat columns)."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            sub = raw[tkr].copy()
        else:
            sub = raw.copy()
    except (KeyError, TypeError):
        return None
    sub = sub.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume",
    }).dropna(subset=["close"])
    if sub.empty:
        return None
    sub["ticker"] = tkr
    sub = sub.reset_index().rename(columns={"Date": "date"})
    return sub[["date", "ticker", "open", "high", "low",
                "close", "adj_close", "volume"]]


def download_prices(tickers: Iterable[str], start: str, end: str | None,
                    cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Download daily OHLCV for `tickers`. Returns a long-format frame:
    columns = [date, ticker, open, high, low, close, adj_close, volume].

    `refresh=True` forces re-download (used by the daily run to pick up new bars).
    Cached file is keyed by start date + ticker count, so adding/removing tickers
    invalidates the cache automatically.
    """
    tickers = list(tickers)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"prices_{start}_{end or 'now'}_{len(tickers)}.pkl"
    if cache_file.exists() and not refresh:
        df = pd.read_pickle(cache_file)
        if set(df["ticker"].unique()) >= set(tickers):
            return df[df["ticker"].isin(tickers)].reset_index(drop=True)

    # threads=False: parallel batch hits SQLite cache "database is locked" on CI runners.
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=False, progress=False, group_by="ticker",
        threads=False,
    )

    rows: list[pd.DataFrame] = []
    ok: set[str] = set()
    for tkr in tickers:
        got = _extract_ticker_frame(raw, tkr)
        if got is not None and not got.empty:
            rows.append(got)
            ok.add(tkr)

    # Retry missing tickers one-by-one (handles intermittent failures / locks).
    for tkr in tickers:
        if tkr in ok:
            continue
        try:
            one = yf.download(
                [tkr], start=start, end=end,
                auto_adjust=False, progress=False, group_by="ticker",
                threads=False,
            )
        except Exception:
            continue
        got = _extract_ticker_frame(one, tkr)
        if got is not None and not got.empty:
            rows.append(got)
            ok.add(tkr)

    if not rows:
        raise RuntimeError("yfinance returned no data — check internet/tickers.")
    df = pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"])
    df.to_pickle(cache_file)
    return df
