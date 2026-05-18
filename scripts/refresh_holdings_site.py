"""Regenerate holdings.csv + docs/index.html without re-running the full RF pipeline.

Use after code fixes or when live_portfolio.json has positions missing from the
last daily artifact commit. Requires existing docs/data/* from a prior run_daily.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import build_panel, download_prices
from pipeline.config import DEFAULT_CONFIG
from pipeline.portfolio import (Order, PortfolioConfig, PortfolioState,
                                 compute_stats, make_holdings_view,
                                 rotate_at_close)
from pipeline.report import build_site
from scripts.run_daily import (_append_prices, _live_holding_tickers,
                                detect_hot_additions)


def _panel_with_live_prices(cfg, refresh: bool = False) -> pd.DataFrame:
    prices = download_prices(cfg.all_tickers, cfg.start, cfg.end,
                             cfg.cache_dir, refresh=refresh)
    _, hot_scored = detect_hot_additions(cfg)
    held = _live_holding_tickers(cfg)
    prices = _append_prices(prices, held, cfg, refresh=refresh)
    return build_panel(prices, cfg.sectors, cfg.benchmark), hot_scored


def main(refresh: bool = False) -> None:
    cfg = DEFAULT_CONFIG
    data = cfg.docs_data_dir
    print("Building price panel (core + open live holdings)...")
    panel, hot_scored = _panel_with_live_prices(cfg, refresh=refresh)
    if hot_scored.empty and (data / "hot_stocks.csv").exists():
        hot_scored = pd.read_csv(data / "hot_stocks.csv")

    live_path = data / "live_portfolio.json"
    with live_path.open(encoding="utf-8") as fh:
        saved = json.load(fh)
    live_state = PortfolioState(
        cash=saved["cash"],
        holdings={k: int(v) for k, v in saved["holdings"].items()},
        entry_prices={k: float(v) for k, v in saved["entry_prices"].items()},
        entry_dates={k: str(v) for k, v in saved.get("entry_dates", {}).items()},
    )
    pf_cfg = PortfolioConfig(starting_capital=5000.0, n_long=cfg.long_n)

    pred = pd.read_csv(data / "predictions.csv", parse_dates=["as_of", "for_date"])
    last_as_of = pred["as_of"].max()
    sub = pred[pred["as_of"] == last_as_of].copy()
    latest = pd.DataFrame({
        "date": pd.to_datetime(sub["as_of"]).dt.normalize(),
        "ticker": sub["ticker"],
        "y_pred": sub["pred_xret"],
    })
    today_iso = pd.Timestamp(latest["date"].max()).strftime("%Y-%m-%d")
    post_close_state, today_orders = rotate_at_close(
        live_state, latest, panel, pf_cfg, today_iso)
    pd.DataFrame([o.to_dict() for o in today_orders]).to_csv(
        data / "orders_today.csv", index=False)

    holdings_df = make_holdings_view(post_close_state, panel, pf_cfg)
    holdings_df.to_csv(data / "holdings.csv", index=False)
    print(f"  -> {data / 'holdings.csv'} ({len(holdings_df)} rows, max {cfg.long_n})")

    saved["cash"] = post_close_state.cash
    saved["holdings"] = post_close_state.holdings
    saved["entry_prices"] = post_close_state.entry_prices
    saved["entry_dates"] = post_close_state.entry_dates

    last_px = panel.sort_values("date").groupby("ticker")["adj_close"].last()
    mtm = sum(
        qty * float(last_px.get(t, float("nan")))
        for t, qty in post_close_state.holdings.items()
    )
    equity = float(post_close_state.cash) + (mtm if pd.notna(mtm) else 0.0)
    hist_path = data / "live_portfolio_history.csv"
    if hist_path.exists():
        live_history = pd.read_csv(hist_path, parse_dates=["date"])
        if not live_history.empty:
            live_history.loc[live_history.index[-1], "equity"] = equity
            live_history.to_csv(hist_path, index=False)
            print(f"  -> {hist_path} (latest equity ${equity:,.2f})")
    else:
        live_history = pd.DataFrame()

    history = pred
    tracker = pd.read_csv(data / "tracker.csv", parse_dates=["for_date"])
    fold_metrics = pd.read_csv(data / "fold_metrics.csv")
    feat_imp = pd.read_csv(data / "feature_importance.csv", index_col=0)["importance"]
    pf_history = pd.read_csv(data / "portfolio_history.csv", parse_dates=["date"],
                             index_col="date")
    pf_trades = pd.read_csv(data / "portfolio_trades.csv", parse_dates=["date"])
    pf_stats = compute_stats(pf_history, pf_trades, pf_cfg.starting_capital)
    orders = today_orders
    with (data / "daily_views.json").open(encoding="utf-8") as fh:
        daily_views_by_mode = json.load(fh)
    active_settle_mode = "margin" if "margin" in daily_views_by_mode else next(iter(daily_views_by_mode))
    daily_views = daily_views_by_mode.get(active_settle_mode, {})
    hot_additions = (
        hot_scored.head(cfg.hot_max_to_add)["ticker"].tolist()
        if not hot_scored.empty else []
    )

    run_at = datetime.now(timezone.utc)
    out = build_site(
        latest, tracker, history, fold_metrics, feat_imp, cfg,
        run_at=run_at,
        hot_scored=hot_scored if not hot_scored.empty else None,
        hot_additions=hot_additions,
        portfolio_history=pf_history,
        portfolio_trades=pf_trades,
        portfolio_stats=pf_stats,
        orders=orders,
        holdings_df=holdings_df,
        live_history=live_history,
        daily_views=daily_views,
        daily_views_by_mode=daily_views_by_mode,
        active_settle_mode=active_settle_mode,
        panel=panel,
        starting_capital=pf_cfg.starting_capital,
        live_cash=float(post_close_state.cash),
    )
    saved["last_run"] = run_at.isoformat()
    with live_path.open("w", encoding="utf-8") as fh:
        json.dump(saved, fh, indent=2)
    print(f"  -> {live_path} ({len(post_close_state.holdings)} positions)")
    print(f"  -> {out}")
    print("Done.")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
