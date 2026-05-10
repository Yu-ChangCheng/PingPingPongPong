"""Explicit entry/exit strategies on top of the RF predictions.

Each strategy answers four questions per (date, ticker):
  1. Is the position open today?
  2. If not, do we ENTER today (buy at next-day open / close)?
  3. If yes, do we EXIT today?  (because of: rotation, stop-loss, take-profit,
     time-stop, or signal flip)
  4. What weight do we use?

We model trades at daily resolution: position changes are applied at the
next bar's price. Returns are then `position_yesterday * ret_today`. This is
the standard "decision today, fill tomorrow" convention used in event-time backtests.

Strategies implemented:
  - daily_rotation       (simple baseline) — long top-N, short bottom-N every day
  - threshold_rotation   — only enter when |pred| > entry_thresh, exit when crosses opposite of exit_thresh
  - hold_for_k_days      — once entered, hold for K days regardless of signal
  - stop_loss_overlay    — wraps another strategy and exits on cumulative loss

All return a DataFrame indexed by date with `ret` (daily PnL) and detail columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    name: str = "daily_rotation"
    long_n: int = 5
    short_n: int = 5
    entry_thresh: float = 0.0   # only used by threshold_rotation, in pred-xret units
    exit_thresh: float = 0.0
    hold_days: int = 1          # 1 = pure daily rotation
    stop_loss_pct: float | None = None     # e.g. -0.05 = exit if -5% on the trade
    take_profit_pct: float | None = None
    cost_bps: float = 1.0       # one-way cost per name turned over
    cooldown_days: int = 0      # bars to wait after a stop-loss exit


def _pivot_predictions(preds_df: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pred_wide, ret_wide) with index=date, columns=ticker."""
    df = preds_df.merge(panel[["date", "ticker", "ret_fwd_1d"]],
                        on=["date", "ticker"], how="left").dropna(subset=["y_pred", "ret_fwd_1d"])
    df["date"] = pd.to_datetime(df["date"])
    pred_w = df.pivot(index="date", columns="ticker", values="y_pred")
    ret_w  = df.pivot(index="date", columns="ticker", values="ret_fwd_1d")
    return pred_w.sort_index(), ret_w.sort_index()


def _ranks_to_targets(pred_row: pd.Series, long_n: int, short_n: int) -> pd.Series:
    """Convert a row of predictions into target weights {+1/long_n, -1/short_n, 0}."""
    valid = pred_row.dropna()
    if len(valid) < long_n + short_n:
        return pd.Series(0.0, index=pred_row.index)
    sorted_ = valid.sort_values()
    target = pd.Series(0.0, index=pred_row.index)
    short_set = sorted_.head(short_n).index
    long_set  = sorted_.tail(long_n).index
    target.loc[long_set]  = 1.0 / long_n
    target.loc[short_set] = -1.0 / short_n
    return target


def run_strategy(preds_df: pd.DataFrame, panel: pd.DataFrame,
                 cfg: StrategyConfig) -> pd.DataFrame:
    """Backtest one strategy. Returns a per-day DataFrame:
        date, ret (daily PnL after costs), gross_ret, cost, n_long, n_short,
        n_entries, n_exits, exit_reason_*  (one column per reason)."""
    pred_w, ret_w = _pivot_predictions(preds_df, panel)
    tickers = pred_w.columns
    dates = pred_w.index

    pos = pd.Series(0.0, index=tickers)            # today's position weights
    days_held = pd.Series(0, index=tickers)
    cumulative = pd.Series(0.0, index=tickers)     # PnL accumulated within current trade
    cooldown = pd.Series(0, index=tickers)

    rows = []
    for d in dates:
        pred_row = pred_w.loc[d]
        ret_row  = ret_w.loc[d]

        # --- 1) realise yesterday's position on today's return ---
        gross = float((pos * ret_row.fillna(0)).sum())
        # update intra-trade cumulative PnL only for open names
        cumulative = cumulative + (pos * ret_row.fillna(0)).where(pos.abs() > 0, 0)

        # --- 2) decide exits ---
        exit_mask = pd.Series(False, index=tickers)
        exit_reason = pd.Series("", index=tickers, dtype=object)

        # Stop-loss
        if cfg.stop_loss_pct is not None:
            sl = (cumulative <= cfg.stop_loss_pct) & (pos.abs() > 0)
            exit_mask |= sl; exit_reason[sl] = "stop_loss"
        # Take-profit
        if cfg.take_profit_pct is not None:
            tp = (cumulative >= cfg.take_profit_pct) & (pos.abs() > 0)
            exit_mask |= tp; exit_reason[tp] = "take_profit"
        # Time-stop (max hold days)
        if cfg.hold_days and cfg.hold_days > 1:
            ts = (days_held >= cfg.hold_days) & (pos.abs() > 0)
            exit_mask |= ts; exit_reason[ts] = "time_stop"

        # --- 3) decide target weights for today ---
        if cfg.name in ("daily_rotation", "threshold_rotation"):
            target = _ranks_to_targets(pred_row, cfg.long_n, cfg.short_n)
            if cfg.name == "threshold_rotation":
                # Only keep names whose prediction magnitude crosses the threshold.
                strong = pred_row.abs() >= cfg.entry_thresh
                target = target.where(strong, 0.0)
            # If we are within hold_days, don't rotate this name.
            if cfg.hold_days and cfg.hold_days > 1:
                holding = (pos.abs() > 0) & (days_held < cfg.hold_days)
                target = target.where(~holding, pos)  # keep current pos
        elif cfg.name == "hold_for_k_days":
            # Enter top/bottom-N only if no current position; hold K days.
            cand = _ranks_to_targets(pred_row, cfg.long_n, cfg.short_n)
            entering = (pos.abs() == 0) & (cand.abs() > 0) & (cooldown == 0)
            keep = (pos.abs() > 0) & (days_held < cfg.hold_days)
            target = pd.Series(0.0, index=tickers)
            target.where(~entering, cand, inplace=True)
            target.where(~keep, pos, inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {cfg.name}")

        # Apply forced exits.
        target = target.where(~exit_mask, 0.0)

        # Cooldown counter ticks down for unfilled names.
        cooldown = (cooldown - 1).clip(lower=0)
        cooldown[exit_mask & (exit_reason == "stop_loss")] = cfg.cooldown_days

        # --- 4) costs from delta (turnover) ---
        delta = (target - pos).abs()
        n_traded = (delta > 1e-12).sum()
        cost = float(delta.sum()) * cfg.cost_bps / 1e4

        net_ret = gross - cost

        # --- 5) update bookkeeping for next day ---
        opened = (pos.abs() == 0) & (target.abs() > 0)
        closed = (pos.abs() > 0)  & (target.abs() == 0)
        days_held = days_held + 1
        days_held[opened] = 0
        days_held[closed] = 0
        cumulative[closed] = 0.0
        cumulative[opened] = 0.0
        pos = target

        rows.append({
            "date": d,
            "gross_ret": gross,
            "cost": cost,
            "ret": net_ret,
            "n_long":  int((pos > 0).sum()),
            "n_short": int((pos < 0).sum()),
            "n_entries": int(opened.sum()),
            "n_exits":   int(closed.sum() + exit_mask.sum()),
            "n_stop_loss":   int(((exit_reason == "stop_loss")).sum()),
            "n_take_profit": int(((exit_reason == "take_profit")).sum()),
            "n_time_stop":   int(((exit_reason == "time_stop")).sum()),
        })

    out = pd.DataFrame(rows).set_index("date").sort_index()
    out.attrs["strategy_name"] = cfg.name
    return out


# Convenience presets — match the docs in `STRATEGY_GUIDE` below.
PRESET_STRATEGIES: dict[str, StrategyConfig] = {
    "daily_rotation":      StrategyConfig(name="daily_rotation"),
    "threshold_rotation":  StrategyConfig(name="threshold_rotation",
                                          entry_thresh=0.0005),
    "hold_for_5":          StrategyConfig(name="hold_for_k_days", hold_days=5),
    "stop_loss_3pct":      StrategyConfig(name="daily_rotation",
                                          stop_loss_pct=-0.03,
                                          cooldown_days=3),
}


STRATEGY_GUIDE = """
ENTER / EXIT RULES — quick reference

  daily_rotation
    Enter:  every day; position = +1/N for top-N predicted, -1/N for bottom-N
    Exit:   any name that drops out of the top/bottom basket the next day
    Pros:   fully captures fresh signal, simple
    Cons:   high turnover, costs matter

  threshold_rotation
    Enter:  same as daily_rotation BUT only names with |pred_xret| >= entry_thresh
    Exit:   when name leaves the top/bottom basket OR |pred_xret| < entry_thresh
    Pros:   skips noisy days, lower turnover
    Cons:   may miss small-but-correct signals

  hold_for_k_days
    Enter:  enter top/bottom-N positions only if you have no current position
    Exit:   hold each trade for exactly K days, then re-evaluate
    Pros:   ~K times less trading, easier in retail accounts
    Cons:   slower to react to signal flips

  stop_loss_overlay (any base + stop_loss_pct)
    Enter:  same as base strategy
    Exit:   base exit, OR cumulative trade PnL <= stop_loss_pct (e.g. -3%)
            then enforce cooldown_days before re-entering same name
    Pros:   bounds tail-risk per name
    Cons:   can cut winners during regular volatility
"""
