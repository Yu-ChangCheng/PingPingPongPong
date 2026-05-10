"""Forward-looking prediction tracker (simulated fills).

Each daily run does two things:
  1. Append today's predictions (for the *next* trading day) to `predictions.csv`,
     marked with `actual_ret = NaN`.
  2. For any prediction whose `for_date` has now occurred, fill in the actual
     realised return from yfinance and compute realised PnL.

`tracker.csv` is a clean, ready-to-display view: one row per (for_date, strategy)
with cumulative PnL.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_COLUMNS = [
    "as_of",          # The date the model was run.
    "for_date",       # The trading day the prediction is *for*.
    "ticker",
    "side",           # 'long', 'short', or 'flat'.
    "rank",
    "pred_xret",      # Predicted next-day excess return.
    "actual_ret",     # Realised next-day total return (filled in next day).
    "actual_xret",    # Realised next-day excess return vs benchmark.
    "model_version",
]


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTIONS_COLUMNS)


def load_tracker(docs_data_dir: Path) -> pd.DataFrame:
    """Load the historical predictions CSV (creates an empty one if missing)."""
    docs_data_dir.mkdir(parents=True, exist_ok=True)
    path = docs_data_dir / "predictions.csv"
    if not path.exists():
        return _empty_predictions()
    df = pd.read_csv(path, parse_dates=["as_of", "for_date"])
    return df


def _next_business_day(d: pd.Timestamp,
                       trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the next trading day strictly after `d` from the supplied index."""
    nxt = trading_days[trading_days > d]
    if len(nxt) == 0:
        return d + pd.Timedelta(days=1)
    return nxt[0]


def update_tracker(predictions_today: pd.DataFrame,
                   panel: pd.DataFrame,
                   docs_data_dir: Path,
                   long_n: int = 5, short_n: int = 5,
                   model_version: str = "rf-v1") -> pd.DataFrame:
    """Append today's predictions to predictions.csv and back-fill realised returns.

    `predictions_today` columns: [date, ticker, y_pred] for ALL stocks in the
    universe on the latest date.
    """
    docs_data_dir.mkdir(parents=True, exist_ok=True)
    pred_path = docs_data_dir / "predictions.csv"
    history = load_tracker(docs_data_dir)

    today = pd.Timestamp(predictions_today["date"].max()).normalize()
    trading_days = pd.DatetimeIndex(sorted(panel["date"].unique())).normalize()
    for_date = _next_business_day(today, trading_days)

    # Side assignment: top-long_n -> long, bottom-short_n -> short, else flat.
    g = (predictions_today.copy()
         .sort_values("y_pred", ascending=False)
         .reset_index(drop=True))
    g["rank"] = np.arange(1, len(g) + 1)
    g["side"] = "flat"
    g.loc[g.index[:long_n], "side"] = "long"
    g.loc[g.index[-short_n:], "side"] = "short"
    new_rows = pd.DataFrame({
        "as_of":         today,
        "for_date":      for_date,
        "ticker":        g["ticker"],
        "side":          g["side"],
        "rank":          g["rank"],
        "pred_xret":     g["y_pred"],
        "actual_ret":    np.nan,
        "actual_xret":   np.nan,
        "model_version": model_version,
    })

    # Avoid duplicating predictions if rerun the same day for the same for_date.
    if not history.empty:
        history = history[~((history["as_of"] == today) &
                            (history["for_date"] == for_date))]
        history = pd.concat([history, new_rows], ignore_index=True)
    else:
        history = new_rows.copy()

    # Backfill actual returns for any rows whose `for_date` is now in the panel
    # AND for which we still don't have an actual_ret recorded.
    actuals = (panel[["date", "ticker", "ret_1d", "ret_bench"]]
               .copy().rename(columns={"date": "for_date",
                                       "ret_1d": "_ret",
                                       "ret_bench": "_ret_bench"}))
    actuals["for_date"] = pd.to_datetime(actuals["for_date"]).dt.normalize()
    history["for_date"] = pd.to_datetime(history["for_date"]).dt.normalize()

    merged = history.merge(actuals, on=["for_date", "ticker"], how="left")
    fill_mask = merged["actual_ret"].isna() & merged["_ret"].notna()
    merged.loc[fill_mask, "actual_ret"]  = merged.loc[fill_mask, "_ret"]
    merged.loc[fill_mask, "actual_xret"] = (merged.loc[fill_mask, "_ret"]
                                            - merged.loc[fill_mask, "_ret_bench"])
    history = merged.drop(columns=["_ret", "_ret_bench"])

    history = history.sort_values(["for_date", "rank"]).reset_index(drop=True)
    history.to_csv(pred_path, index=False)

    # Build the daily PnL summary that the website reads.
    summary = build_daily_summary(history)
    (docs_data_dir / "tracker.csv").write_text(summary.to_csv(index=False))
    return history


def build_daily_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Per `for_date`: realised PnL of the long-short and long-only sleeves."""
    if history.empty:
        return pd.DataFrame(columns=[
            "for_date", "n_long", "n_short",
            "ret_long_only", "ret_long_short",
            "cum_long_only", "cum_long_short",
        ])
    h = history.dropna(subset=["actual_ret"]).copy()
    if h.empty:
        return pd.DataFrame(columns=[
            "for_date", "n_long", "n_short",
            "ret_long_only", "ret_long_short",
            "cum_long_only", "cum_long_short",
        ])
    rows = []
    for d, g in h.groupby("for_date"):
        longs  = g[g["side"] == "long"]
        shorts = g[g["side"] == "short"]
        ret_long_only  = longs["actual_ret"].mean()  if len(longs) else 0.0
        ret_short_only = shorts["actual_ret"].mean() if len(shorts) else 0.0
        ret_ls = ret_long_only - ret_short_only
        rows.append({
            "for_date":       d,
            "n_long":         len(longs),
            "n_short":        len(shorts),
            "ret_long_only":  ret_long_only,
            "ret_long_short": ret_ls,
        })
    out = pd.DataFrame(rows).sort_values("for_date")
    out["cum_long_only"]  = (1 + out["ret_long_only"]).cumprod()
    out["cum_long_short"] = (1 + out["ret_long_short"]).cumprod()
    return out
