"""Backtest helpers — long-short top/bottom-N + summary stats."""
from __future__ import annotations

import numpy as np
import pandas as pd


def perf_stats(daily_pnl: pd.Series, name: str = "") -> dict:
    """Annualised performance summary for a daily-PnL series (in returns space)."""
    pnl = daily_pnl.dropna()
    if len(pnl) == 0:
        return {"Strategy": name, "CAGR": "n/a", "Vol": "n/a",
                "Sharpe": "n/a", "MaxDD": "n/a", "Hit%": "n/a", "Days": 0}
    cum = (1 + pnl).cumprod()
    yrs = max(len(pnl) / 252, 1e-6)
    cagr = cum.iloc[-1] ** (1 / yrs) - 1
    vol = pnl.std() * np.sqrt(252)
    sharpe = (pnl.mean() * 252) / (pnl.std() * np.sqrt(252) + 1e-12)
    dd = (cum / cum.cummax() - 1).min()
    hit = (pnl > 0).mean()
    return {
        "Strategy": name,
        "CAGR":   f"{cagr:+.2%}",
        "Vol":    f"{vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "MaxDD":  f"{dd:.2%}",
        "Hit%":   f"{hit:.2%}",
        "Days":   int(len(pnl)),
    }


def cross_sectional_backtest(preds_df: pd.DataFrame,
                             panel: pd.DataFrame,
                             long_n: int = 5, short_n: int = 5,
                             cost_bps: float = 1.0) -> pd.DataFrame:
    """For each date: long top-N, short bottom-N. Daily rebalance.
    Costs are charged on every name turned over (1bp = 0.0001 of capital)."""
    df = preds_df.merge(
        panel[["date", "ticker", "ret_fwd_1d", "ret_bench_fwd_1d"]],
        on=["date", "ticker"], how="left",
    ).dropna(subset=["y_pred", "ret_fwd_1d"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "y_pred"])

    rows = []
    prev_long, prev_short = set(), set()
    for d, g in df.groupby("date"):
        g = g.dropna(subset=["y_pred"]).sort_values("y_pred")
        if len(g) < long_n + short_n:
            continue
        short_set = g.head(short_n)
        long_set  = g.tail(long_n)

        ret_long  = long_set["ret_fwd_1d"].mean()
        ret_short = short_set["ret_fwd_1d"].mean()
        ret_ls    = ret_long - ret_short

        new_long  = set(long_set["ticker"])
        new_short = set(short_set["ticker"])
        turn = (len(new_long.symmetric_difference(prev_long))
              + len(new_short.symmetric_difference(prev_short)))
        cost = turn * cost_bps / 1e4
        prev_long, prev_short = new_long, new_short

        rows.append({
            "date": d,
            "ret_long_short": ret_ls - cost,
            "ret_long_short_gross": ret_ls,
            "ret_long_only": ret_long - cost / 2,
            "longs":  ",".join(sorted(new_long)),
            "shorts": ",".join(sorted(new_short)),
            "spread_pred": long_set["y_pred"].mean() - short_set["y_pred"].mean(),
        })

    bt = pd.DataFrame(rows).set_index("date").sort_index()
    return bt
