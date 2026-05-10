"""Daily feature engineering with cross-sectional rank normalization."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=FutureWarning,
)


FEATURE_COLS: list[str] = [
    "ret_1d", "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d", "mom_756d",
    "chmom", "maxret_21d", "minret_21d", "ind_mom",
    "log_dolvol_21d", "turn_21d", "std_turn_21d", "baspread_21d",
    "ill_21d", "zerotrade_21d", "log_size",
    "retvol_21d", "retvol_63d", "beta_252d", "betasq_252d", "idiovol_63d",
]


def _lagged_window_return(s: pd.Series, total: int, skip: int) -> pd.Series:
    """Cumulative log-return from t-total to t-skip (skip = days excluded at end)."""
    log_ret = np.log1p(s)
    return log_ret.rolling(total - skip, min_periods=total - skip).sum().shift(skip)


def build_panel(prices: pd.DataFrame, sectors: dict[str, str],
                benchmark: str = "SPY") -> pd.DataFrame:
    """Per-ticker feature engineering. Returns a long panel (date, ticker, features...)."""
    need = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    miss = need - set(prices.columns)
    if miss:
        raise ValueError(f"prices missing columns {sorted(miss)} — got {list(prices.columns)}")
    if benchmark not in prices["ticker"].values:
        raise ValueError(
            f"benchmark {benchmark!r} not in price panel — need SPY (and peers) downloaded.")

    df = prices.copy()
    df["adj_close"] = df["adj_close"].astype(float)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()
    df["dolvol"] = df["close"].astype(float) * df["volume"].astype(float)
    df["log_dolvol"] = np.log1p(df["dolvol"])
    df["log_size"] = np.log(df["close"].astype(float))
    df["hl_spread"] = (df["high"] - df["low"]) / ((df["high"] + df["low"]) / 2.0)

    def per_ticker(d: pd.DataFrame) -> pd.DataFrame:
        r = d["ret_1d"]
        d["mom_5d"]   = _lagged_window_return(r, 5, 0)
        d["mom_21d"]  = _lagged_window_return(r, 21, 5)
        d["mom_63d"]  = _lagged_window_return(r, 63, 5)
        d["mom_126d"] = _lagged_window_return(r, 126, 21)
        d["mom_252d"] = _lagged_window_return(r, 252, 21)
        d["mom_756d"] = _lagged_window_return(r, 756, 21)
        d["chmom"]    = d["mom_126d"] - d["mom_252d"]
        d["maxret_21d"] = r.rolling(21, min_periods=15).max().shift(1)
        d["minret_21d"] = r.rolling(21, min_periods=15).min().shift(1)

        vol = d["volume"].astype(float)
        avg_vol_252 = vol.rolling(252, min_periods=60).mean().shift(1)
        d["turn_21d"]       = (vol.rolling(21, min_periods=15).mean() / avg_vol_252).shift(1)
        d["std_turn_21d"]   = (vol.rolling(21, min_periods=15).std() / avg_vol_252).shift(1)
        d["log_dolvol_21d"] = d["log_dolvol"].rolling(21, min_periods=15).mean().shift(1)
        d["baspread_21d"]   = d["hl_spread"].rolling(21, min_periods=15).mean().shift(1)
        ill_daily = (r.abs() / d["dolvol"].replace(0, np.nan)) * 1e9
        d["ill_21d"]        = ill_daily.rolling(21, min_periods=15).mean().shift(1)
        d["zerotrade_21d"]  = (vol == 0).astype(float).rolling(21, min_periods=15).sum().shift(1)
        d["log_size"]       = d["log_size"].shift(1)
        d["retvol_21d"]     = r.rolling(21, min_periods=15).std().shift(1)
        d["retvol_63d"]     = r.rolling(63, min_periods=40).std().shift(1)
        return d

    # Concat per group — avoids pandas GroupBy.apply dropping `ticker` on some versions.
    chunks: list[pd.DataFrame] = []
    for _, g in df.groupby("ticker", sort=False):
        chunks.append(per_ticker(g.copy()))
    df = pd.concat(chunks, ignore_index=True)

    bench = (df[df["ticker"] == benchmark][["date", "ret_1d"]]
             .rename(columns={"ret_1d": "ret_bench"}))
    df = df.merge(bench, on="date", how="left")

    def beta_block(d: pd.DataFrame) -> pd.DataFrame:
        x, y = d["ret_bench"], d["ret_1d"]
        cov = x.rolling(252, min_periods=120).cov(y)
        var = x.rolling(252, min_periods=120).var()
        beta = (cov / var).shift(1)
        d["beta_252d"]   = beta
        d["betasq_252d"] = beta ** 2
        resid = y - beta * x
        d["idiovol_63d"] = resid.rolling(63, min_periods=40).std().shift(1)
        return d

    chunks_b: list[pd.DataFrame] = []
    for _, g in df.groupby("ticker", sort=False):
        chunks_b.append(beta_block(g.copy()))
    df = pd.concat(chunks_b, ignore_index=True)

    df["sector"] = df["ticker"].map(sectors).fillna("Other")
    df["ind_mom"] = (df.groupby(["date", "sector"])["mom_21d"]
                       .transform(lambda s: s.mean()))
    df["ind_mom"] = df["ind_mom"] - df.groupby("ticker")["ret_1d"].transform(
        lambda r: r.rolling(21, min_periods=15).mean().shift(1))

    df["ret_fwd_1d"] = df.groupby("ticker")["ret_1d"].shift(-1)
    df["ret_bench_fwd_1d"] = df.groupby("ticker")["ret_bench"].shift(-1)
    df["xret_fwd_1d"] = df["ret_fwd_1d"] - df["ret_bench_fwd_1d"]
    return df


def cross_section_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cross-sectionally rank-normalize each `col` to [-1, 1] by date."""
    out = df.copy()
    grp = out.groupby("date")
    for c in cols:
        ranks = grp[c].rank(method="average", pct=True)
        out[c] = (ranks * 2 - 1).astype("float32")
    return out
