"""Hot-stock detector — find names with big recent moves to add to the universe.

Why this matters: the core universe is curated and **survivor-biased**. The hot
detector is a *forward-looking* mechanism: every day we look across a much
larger watchlist and surface names that just started behaving differently
(volume spikes, range expansion, momentum break-outs). They are added to the
prediction set if they have enough history for the model to score them.

Hotness score (each component is z-scored within today's cross section, then summed):
  + 5-day return                    (recent momentum)
  + 20-day return                   (medium-term momentum)
  + volume z-score vs 60d average   (something is happening)
  + range z-score (high-low/close)  (volatility expansion)
  + distance from 252d high (closer = hotter)

The list of "watchlist" tickers can be arbitrarily large; it's only used to
score, not to train. Names that pass a hotness threshold are added to the
panel and ranked alongside the core universe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# A reasonably broad watchlist of liquid US names. Tweak freely.
DEFAULT_WATCHLIST: tuple[str, ...] = (
    "PLTR", "COIN", "SHOP", "SQ", "SNOW", "PANW", "CRWD", "ZS", "DDOG", "NET",
    "MDB", "ABNB", "UBER", "LYFT", "RIVN", "LCID", "F", "GM", "PYPL", "SOFI",
    "HOOD", "BIDU", "PDD", "JD", "BABA", "SE", "MELI", "DASH", "ROKU", "ZM",
    "DOCU", "TWLO", "OKTA", "FSLR", "ENPH", "PLUG", "RUN", "RIOT", "MSTR", "MARA",
    "BBBY", "AMC", "GME", "SPCE", "NIO", "XPEV", "LI", "TDOC", "PINS", "SNAP",
    "ETSY", "EBAY", "WBD", "PARA", "T", "VZ", "CHTR", "TMUS", "GLW", "MU",
    "SMCI", "ARM", "TSM", "ASML", "INTU", "NOW", "WDAY", "TEAM", "FTNT", "CDNS",
    "SNPS", "ADSK", "ROP", "KEYS", "ANET", "CIEN", "JNPR", "AKAM", "DELL", "HPQ",
    "HPE", "STX", "WDC", "TXN", "MCHP", "ON", "ADI", "NXPI", "MPWR", "LRCX",
    "AMAT", "KLAC", "ORLY", "AZO", "CMG", "DPZ", "TJX", "ROST", "ULTA", "BBY",
)


@dataclass
class HotConfig:
    short_window: int = 5
    medium_window: int = 20
    vol_window: int = 60
    high_window: int = 252
    min_history_days: int = 252
    hotness_threshold: float = 1.0   # z-sum cut-off
    max_to_add: int = 15             # cap how many names we add per run


def score_hot_stocks(prices: pd.DataFrame,
                     hot_cfg: HotConfig | None = None) -> pd.DataFrame:
    """Compute the hotness score for the latest available date in `prices`.
    Returns a DataFrame: ticker, score, plus the underlying components.
    """
    if hot_cfg is None:
        hot_cfg = HotConfig()
    if prices.empty:
        return pd.DataFrame()

    latest_date = prices["date"].max()
    rows = []
    for tkr, g in prices.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < hot_cfg.min_history_days:
            continue
        adj = g["adj_close"].astype(float)
        ret = adj.pct_change()

        ret_short = adj.iloc[-1] / adj.iloc[-1 - hot_cfg.short_window] - 1
        ret_med   = adj.iloc[-1] / adj.iloc[-1 - hot_cfg.medium_window] - 1
        vol_today = float(g["volume"].iloc[-1])
        vol_avg   = float(g["volume"].iloc[-hot_cfg.vol_window:].mean())
        vol_std   = float(g["volume"].iloc[-hot_cfg.vol_window:].std() or 1)
        vol_z = (vol_today - vol_avg) / vol_std if vol_std > 0 else 0.0
        rng = (g["high"] - g["low"]) / g["close"].astype(float)
        rng_today = float(rng.iloc[-1])
        rng_avg = float(rng.iloc[-hot_cfg.vol_window:].mean())
        rng_std = float(rng.iloc[-hot_cfg.vol_window:].std() or 1)
        rng_z = (rng_today - rng_avg) / rng_std if rng_std > 0 else 0.0
        high_252 = float(adj.iloc[-hot_cfg.high_window:].max())
        dist_high = float(adj.iloc[-1]) / high_252 - 1  # ~0 means at high

        rows.append({
            "date": latest_date, "ticker": tkr,
            "ret_short": ret_short, "ret_med": ret_med,
            "vol_z": vol_z, "rng_z": rng_z, "dist_high": dist_high,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Cross-sectional z-score each component, then sum.
    def _z(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std() + 1e-12)

    df["score"] = (
        _z(df["ret_short"])
        + _z(df["ret_med"])
        + _z(df["vol_z"])
        + _z(df["rng_z"])
        + (-_z(-df["dist_high"]))   # closer to high → higher score
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def select_hot_additions(prices: pd.DataFrame,
                         core_universe: tuple[str, ...],
                         hot_cfg: HotConfig | None = None) -> list[str]:
    """Pick up to `max_to_add` hot-stock tickers NOT already in the core universe."""
    if hot_cfg is None:
        hot_cfg = HotConfig()
    scored = score_hot_stocks(prices, hot_cfg)
    if scored.empty:
        return []
    novel = scored[~scored["ticker"].isin(core_universe)]
    novel = novel[novel["score"] >= hot_cfg.hotness_threshold]
    return novel.head(hot_cfg.max_to_add)["ticker"].tolist()
