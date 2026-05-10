"""Daily OHLCV downloader with disk cache."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


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

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=False, progress=False, group_by="ticker",
                      threads=True)

    rows = []
    for tkr in tickers:
        try:
            sub = raw[tkr].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except KeyError:
            continue
        sub = sub.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "adj_close", "Volume": "volume",
        }).dropna(subset=["close"])
        sub["ticker"] = tkr
        sub = sub.reset_index().rename(columns={"Date": "date"})
        rows.append(sub[["date", "ticker", "open", "high", "low",
                         "close", "adj_close", "volume"]])

    if not rows:
        raise RuntimeError("yfinance returned no data — check internet/tickers.")
    df = pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"])
    df.to_pickle(cache_file)
    return df
