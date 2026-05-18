"""Daily entry point.

Pipeline order:
  1. Download daily OHLCV (cached) for the configured universe.
  2. Detect *hot stocks* from the broader watchlist and download their history.
  3. Build the panel + cross-sectional ranks (rank-normalized features).
  4. Walk-forward Random Forest training (diagnostics).
  5. Fit a fresh RF on ALL history -> today's predictions for tomorrow.
  6. Run the chosen entry/exit strategy and update the simulated-trading tracker.
  7. Render docs/index.html for GitHub Pages.

Run locally:
    python scripts/run_daily.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import (FEATURE_COLS, build_panel,
                      cross_section_rank, download_prices)
from pipeline.config import DEFAULT_CONFIG, DEFAULT_UNIVERSE_MODE
from pipeline.hotstocks import (DEFAULT_WATCHLIST, HotConfig,
                                  score_hot_stocks, select_hot_additions)
from pipeline.model import fit_full_model, predict_latest, walk_forward_rf
from pipeline.tracker import update_tracker
from pipeline.portfolio import (PortfolioConfig, PortfolioState,
                                 _finite_money, simulate_portfolio,
                                 compute_stats, make_holdings_view,
                                 rotate_at_close, build_daily_views)
from pipeline.report import build_site


def detect_hot_additions(cfg) -> tuple[list[str], pd.DataFrame]:
    """Score the broader watchlist and return new tickers to add to the universe."""
    if not cfg.enable_hot_stocks:
        return [], pd.DataFrame()
    watchlist = list(cfg.hot_watchlist) or list(DEFAULT_WATCHLIST)
    watchlist = [t for t in watchlist if t not in cfg.all_tickers]
    if not watchlist:
        return [], pd.DataFrame()

    print(f"  scanning {len(watchlist)} watchlist tickers for hot movers...")
    try:
        hot_prices = download_prices(watchlist, cfg.start, cfg.end,
                                     cfg.cache_dir, refresh=False)
    except Exception as e:
        print(f"  hot stock scan skipped: {e}")
        return [], pd.DataFrame()

    hot_cfg = HotConfig(hotness_threshold=cfg.hot_min_score,
                        max_to_add=cfg.hot_max_to_add)
    scored = score_hot_stocks(hot_prices, hot_cfg)
    additions = select_hot_additions(hot_prices, cfg.all_tickers, hot_cfg)
    return additions, scored


def _live_holding_tickers(cfg) -> list[str]:
    """Open positions from persisted live sim state (if any)."""
    path = cfg.docs_data_dir / "live_portfolio.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        saved = json.load(fh)
    return [t for t, sh in saved.get("holdings", {}).items() if int(sh) > 0]


def _append_prices(prices: pd.DataFrame, tickers: list[str], cfg,
                   *, refresh: bool) -> pd.DataFrame:
    """Merge OHLCV for tickers not already in `prices` (e.g. prior hot adds)."""
    need = [t for t in tickers if t not in set(prices["ticker"].unique())]
    if not need:
        return prices
    extra = download_prices(need, cfg.start, cfg.end, cfg.cache_dir, refresh=refresh)
    out = (pd.concat([prices, extra], ignore_index=True)
           .drop_duplicates(["date", "ticker"])
           .sort_values(["date", "ticker"]))
    for tkr in need:
        cfg.sectors.setdefault(tkr, "Other")
    return out


def main(refresh: bool = True) -> None:
    # Optional: freeze history to a specific calendar day (yfinance `end` is
    # exclusive, so PIPELINE_END=2026-05-14 → last bar is 2026-05-13 US session).
    _end = os.environ.get("PIPELINE_END", "").strip()
    cfg = replace(DEFAULT_CONFIG, end=_end) if _end else DEFAULT_CONFIG
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting daily run")
    print(f"  universe = {len(cfg.universe)} stocks + {len(cfg.indices)} indices "
          f"(UNIVERSE={DEFAULT_UNIVERSE_MODE}; set UNIVERSE=core for legacy 54-stock sleeve)")
    if _end:
        print(f"  PIPELINE_END={_end} (yfinance end is exclusive; predictions/orders "
              f"use the last session on or before that date)")

    print("Step 1/6: download core prices...")
    prices = download_prices(cfg.all_tickers, cfg.start, cfg.end,
                             cfg.cache_dir, refresh=refresh)
    last_close_date = pd.Timestamp(prices["date"].max()).normalize()
    print(f"  loaded {len(prices):,} rows | {prices['date'].min().date()} -> "
          f"{last_close_date.date()}")

    # We only trust the latest bar if the US cash session has actually finished
    # for that day (~20:00 UTC summer / 21:00 UTC winter). If the workflow runs
    # before close, yfinance may return a stale-or-partial last row; warn so
    # the orders can be skipped instead of placed on bad data.
    now_utc = pd.Timestamp.now(tz="UTC")
    today_utc_date = now_utc.normalize().tz_localize(None)
    if last_close_date < today_utc_date:
        weekday = today_utc_date.dayofweek  # 0=Mon..6=Sun
        if weekday < 5 and now_utc.hour >= 21:
            print(f"  WARNING: latest close in yfinance is {last_close_date.date()} "
                  f"but today is {today_utc_date.date()} and the cash session "
                  f"should have closed. Orders below will be based on the "
                  f"prior session's close.")
        else:
            print(f"  note: latest close in yfinance is {last_close_date.date()} "
                  f"(most recent completed US session).")

    print("Step 2/6: scan for hot stocks...")
    hot_additions, hot_scored = detect_hot_additions(cfg)
    if hot_additions:
        print(f"  adding hot movers: {hot_additions}")
        hot_prices_full = download_prices(hot_additions, cfg.start, cfg.end,
                                          cfg.cache_dir, refresh=False)
        prices = pd.concat([prices, hot_prices_full], ignore_index=True)
        prices = prices.drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"])
        # Tag hot additions with a generic sector so feature engineering still works.
        for tkr in hot_additions:
            cfg.sectors.setdefault(tkr, "Other")
    else:
        print("  no hot stocks above threshold today")

    held = _live_holding_tickers(cfg)
    if held:
        before = set(prices["ticker"].unique())
        prices = _append_prices(prices, held, cfg, refresh=refresh)
        added = sorted(set(prices["ticker"].unique()) - before)
        if added:
            print(f"  ensured price history for open live holdings: {added}")

    if not hot_scored.empty:
        cfg.docs_data_dir.mkdir(parents=True, exist_ok=True)
        hot_scored.head(20).to_csv(cfg.docs_data_dir / "hot_stocks.csv",
                                   index=False)

    extended_universe = tuple(sorted(set(cfg.universe + tuple(hot_additions))))

    print("Step 3/6: build features...")
    panel = build_panel(prices, cfg.sectors, cfg.benchmark)

    clean = panel.dropna(subset=FEATURE_COLS + ["ret_fwd_1d", "xret_fwd_1d"]).copy()
    ranked = cross_section_rank(clean, FEATURE_COLS)
    stock_ranked = ranked[ranked["ticker"].isin(extended_universe)].copy()
    history_for_pred = panel[panel["ticker"].isin(extended_universe)
                             ].dropna(subset=FEATURE_COLS).copy()
    ranked_for_pred = cross_section_rank(history_for_pred, FEATURE_COLS)
    print(f"  panel = {ranked.shape[0]:,} rows | universe with hot adds = "
          f"{len(extended_universe)} stocks")

    print("Step 4/6: walk-forward RF (diagnostics)...")
    preds_oos, folds = walk_forward_rf(stock_ranked, FEATURE_COLS,
                                       target_col="xret_fwd_1d",
                                       cfg=cfg, verbose=True)

    fold_metrics = pd.DataFrame([{
        "fold": i, "test_start": f.test_start, "test_end": f.test_end,
        "n_train": f.n_train, "n_test": f.n_test, "test_r2": f.test_r2,
    } for i, f in enumerate(folds)])
    cfg.docs_data_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(cfg.docs_data_dir / "fold_metrics.csv", index=False)

    feat_imp = (pd.concat([f.feat_imp.rename(i) for i, f in enumerate(folds)], axis=1)
                  .mean(axis=1).sort_values(ascending=True))
    feat_imp.to_csv(cfg.docs_data_dir / "feature_importance.csv", header=["importance"])

    print("Step 5/6: fit full-history model + predict tomorrow...")
    model = fit_full_model(stock_ranked, FEATURE_COLS, "xret_fwd_1d", cfg)
    latest = predict_latest(model, ranked_for_pred, FEATURE_COLS)
    print(f"  predicted for {latest['date'].iloc[0].date()} -> "
          f"top: {latest.head(3)['ticker'].tolist()} | "
          f"bottom: {latest.tail(3)['ticker'].tolist()}")

    print("Step 6/7: update tracker...")
    history = update_tracker(latest, panel, cfg.docs_data_dir,
                             long_n=cfg.long_n, short_n=cfg.short_n)
    tracker_path = cfg.docs_data_dir / "tracker.csv"
    tracker = (pd.read_csv(tracker_path, parse_dates=["for_date"])
               if tracker_path.exists() else pd.DataFrame())

    print("Step 7/7: simulate $5k portfolio + generate today's orders...")
    # Cash settlement: SETTLE_DAYS controls the *live* portfolio's mode (the
    # one that gets persisted across daily runs). Backtest replay is rendered
    # in BOTH modes regardless of this env var, so the dashboard can toggle
    # between T+0 (margin) and T+1 (US Reg-T cash account, the rule since
    # May 2024) without re-running anything.
    live_settle_days = int(os.environ.get("SETTLE_DAYS", "0") or 0)
    active_settle_mode = "cash" if live_settle_days >= 1 else "margin"
    if live_settle_days > 0:
        print(f"  live mode: T+{live_settle_days} cash account "
              f"(sell proceeds unsettled for {live_settle_days} trading day(s))")
    else:
        print("  live mode: T+0 margin (sell proceeds usable same day; "
              "set SETTLE_DAYS=1 to flip the live portfolio to a cash account)")

    pf_cfg = PortfolioConfig(starting_capital=5000.0, n_long=cfg.long_n,
                              n_short=0, cost_bps=0.0,   # manual execution: $0 commission
                              stop_loss_pct=0.08, time_stop_days=10,
                              settle_days=live_settle_days)
    # IMPORTANT: simulate using OOS predictions ONLY (these are walk-forward,
    # so each prediction was made strictly before the day it was traded on).
    # Today's prediction is then used to compute the *next* set of orders.
    preds_for_sim = preds_oos[["date", "ticker", "y_pred"]]

    # Active mode: drives stats, equity chart, live portfolio fills.
    pf_history, pf_trades, pf_state = simulate_portfolio(
        preds_for_sim, panel, pf_cfg)
    pf_stats = compute_stats(pf_history, pf_trades, pf_cfg.starting_capital)

    pf_history.to_csv(cfg.docs_data_dir / "portfolio_history.csv")
    pf_trades.to_csv(cfg.docs_data_dir / "portfolio_trades.csv", index=False)

    print("  building per-day backtest views in BOTH modes (margin + cash) ...")
    daily_views_by_mode: dict = {}
    for mode_name, sd in (("margin", 0), ("cash", 1)):
        if sd == live_settle_days:
            mh, mt = pf_history, pf_trades   # reuse the active sim
        else:
            cfg_mode = PortfolioConfig(starting_capital=pf_cfg.starting_capital,
                                        n_long=pf_cfg.n_long, n_short=pf_cfg.n_short,
                                        cost_bps=pf_cfg.cost_bps,
                                        min_trade_dollars=pf_cfg.min_trade_dollars,
                                        rebalance_threshold=pf_cfg.rebalance_threshold,
                                        stop_loss_pct=pf_cfg.stop_loss_pct,
                                        time_stop_days=pf_cfg.time_stop_days,
                                        settle_days=sd)
            mh, mt, _ = simulate_portfolio(preds_for_sim, panel, cfg_mode)
        daily_views_by_mode[mode_name] = build_daily_views(
            mh, mt, preds_for_sim, panel,
            n_top=cfg.long_n, stop_loss_pct=pf_cfg.stop_loss_pct)
        print(f"    {mode_name}: {len(daily_views_by_mode[mode_name])} dates")

    daily_views = daily_views_by_mode[active_settle_mode]
    import json as _json
    with (cfg.docs_data_dir / "daily_views.json").open("w", encoding="utf-8") as fh:
        _json.dump(daily_views_by_mode, fh, separators=(",", ":"))
    print(f"  active mode for live tab = {active_settle_mode} "
          f"({len(daily_views)} dates); both modes available in the toggle")

    # ---- Live $5K simulated portfolio: persisted state across daily runs ----
    live_state_path = cfg.docs_data_dir / "live_portfolio.json"
    live_history_path = cfg.docs_data_dir / "live_portfolio_history.csv"
    start_raw = os.environ.get("LIVE_PORTFOLIO_START", "").strip()
    if not start_raw:
        start_raw = (getattr(cfg, "live_portfolio_start", None) or "").strip()
    live_start = pd.Timestamp(start_raw).normalize() if start_raw else None
    pred_date = pd.Timestamp(latest["date"].max()).normalize()
    pre_live_start = bool(live_start is not None and pred_date < live_start)

    if pre_live_start:
        live_state = PortfolioState(cash=pf_cfg.starting_capital)
        print(f"  live sim: frozen at ${pf_cfg.starting_capital:,.0f} cash until "
              f"{live_start.date()} (prediction date {pred_date.date()} is before start)")
    elif live_state_path.exists():
        import json
        with live_state_path.open("r", encoding="utf-8") as fh:
            saved = json.load(fh)
        live_state = PortfolioState(
            cash=saved["cash"],
            holdings={k: int(v) for k, v in saved["holdings"].items()},
            entry_prices={k: float(v) for k, v in saved["entry_prices"].items()},
            entry_dates={k: str(v) for k, v in saved.get("entry_dates", {}).items()},
        )
    else:
        live_state = PortfolioState(cash=pf_cfg.starting_capital)

    today_iso = pd.Timestamp(latest["date"].max()).strftime("%Y-%m-%d")
    # Close-only rotation: always <= n_long names, fills at today's close.
    post_close_state = live_state
    today_orders: list = []
    if not pre_live_start:
        post_close_state, today_orders = rotate_at_close(
            live_state, latest, panel, pf_cfg, today_iso)
    holdings_df = make_holdings_view(post_close_state, panel, pf_cfg)
    pd.DataFrame([o.to_dict() for o in today_orders]).to_csv(
        cfg.docs_data_dir / "orders_today.csv", index=False)
    holdings_df.to_csv(cfg.docs_data_dir / "holdings.csv", index=False)

    import json
    p_last = panel.sort_values("date").groupby("ticker")["adj_close"].last()

    if pre_live_start:
        # Keep JSON + history clean until the first on-or-after start date.
        new_state = PortfolioState(cash=pf_cfg.starting_capital)
        new_equity = float(pf_cfg.starting_capital)
        live_history = pd.DataFrame(
            columns=["date", "equity", "cash", "n_positions", "n_orders"])
        cfg.docs_data_dir.mkdir(parents=True, exist_ok=True)
        with live_state_path.open("w", encoding="utf-8") as fh:
            json.dump({"cash": new_state.cash,
                       "holdings": {},
                       "entry_prices": {},
                       "entry_dates": {},
                       "live_portfolio_start": str(live_start.date()),
                       "last_run": datetime.now(timezone.utc).isoformat()}, fh, indent=2)
        live_history.to_csv(live_history_path, index=False)
    else:
        new_state = post_close_state
        new_holdings_value = sum(
            qty * float(p_last.loc[t])
            for t, qty in new_state.holdings.items() if t in p_last.index)
        new_equity = _finite_money(new_state.cash) + _finite_money(new_holdings_value, 0.0)

        with live_state_path.open("w", encoding="utf-8") as fh:
            json.dump({"cash": new_state.cash,
                       "holdings": new_state.holdings,
                       "entry_prices": new_state.entry_prices,
                       "entry_dates": new_state.entry_dates,
                       "live_portfolio_start": str(live_start.date()) if live_start else None,
                       "last_run": datetime.now(timezone.utc).isoformat()}, fh, indent=2)

        today_date = pd.Timestamp(latest["date"].max()).normalize()
        new_row = pd.DataFrame([{"date": today_date, "equity": new_equity,
                                 "cash": new_state.cash,
                                 "n_positions": len(new_state.holdings),
                                 "n_orders": len(today_orders)}])
        if live_history_path.exists():
            prior = pd.read_csv(live_history_path, parse_dates=["date"])
            prior = prior[prior["date"] < today_date]
            live_history = pd.concat([prior, new_row], ignore_index=True)
        else:
            live_history = new_row
        live_history.to_csv(live_history_path, index=False)

    print(f"  backtest equity = ${pf_stats['final_equity']:,.2f} ({pf_stats['total_return']*100:+.2f}%)")
    print(f"  live $5k portfolio equity = ${new_equity:,.2f} | orders today = {len(today_orders)}")

    run_at_utc = datetime.now(timezone.utc)
    live_cash_ui = (float(post_close_state.cash) if not pre_live_start
                    else float(pf_cfg.starting_capital))
    out_html = build_site(latest, tracker, history, fold_metrics, feat_imp, cfg,
                          run_at=run_at_utc,
                          hot_scored=hot_scored, hot_additions=hot_additions,
                          portfolio_history=pf_history,
                          portfolio_trades=pf_trades,
                          portfolio_stats=pf_stats,
                          orders=today_orders,
                          holdings_df=holdings_df,
                          live_history=live_history,
                          daily_views=daily_views,
                          daily_views_by_mode=daily_views_by_mode,
                          active_settle_mode=active_settle_mode,
                          panel=panel,
                          starting_capital=pf_cfg.starting_capital,
                          live_cash=live_cash_ui)
    print(f"  -> {out_html}")

    # Instagram-friendly share card (PNG + optional JPEG) + optional SMTP email.
    try:
        from pipeline.share_image import (maybe_email_share_card,
                                           render_tomorrow_predictions_card)
        card_png = cfg.docs_data_dir / "tomorrow_predictions_ig.png"
        _, card_jpg = render_tomorrow_predictions_card(
            latest, panel, cfg.long_n, card_png, run_at=run_at_utc)
        extra = f" + {card_jpg.name}" if card_jpg else ""
        print(f"  share card (IG square) -> {card_png}{extra}")
        as_of = pd.Timestamp(latest["date"].max()).strftime("%Y-%m-%d")
        attachments = [card_png]
        if card_jpg:
            attachments.append(card_jpg)
        maybe_email_share_card(
            attachments,
            subject=f"RF daily basket — closes {as_of}",
            body=(
                f"Basket as of last panel date {as_of}.\n"
                f"Generated {run_at_utc.strftime('%Y-%m-%d %H:%M UTC')}.\n\n"
                "Attach to LINE / Instagram manually, or open from docs/data/.\n"
                "Educational only — not investment advice."
            ),
        )
    except Exception as e:
        print(f"  share card / email skipped: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
