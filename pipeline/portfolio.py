"""$5000 simulated portfolio (backtest + order helper).

Every daily run:
  1. Replays the historical walk-forward predictions against actual prices to
     produce a full equity curve from the strategy's start date until today.
  2. Uses today's prediction set to compute concrete orders: BUY/SELL N shares
     of TICKER at LIMIT PRICE = $X (the user places these tomorrow at open).
  3. Computes the full set of performance stats the dashboard cares about
     (hit rate, win rate, profit factor, Sharpe, drawdown, ...).

The simulator is purely deterministic given (predictions, prices), so re-running
it produces identical state \u2013 no separate "portfolio.pkl" needs to be persisted.

Conventions
-----------
Decision is made at close of day `t` using prediction `y_pred(t)` (which itself
is built from features available through close `t`). The trade is filled at
close of day `t`. PnL is then captured between close `t` and close `t+1` via
the next iteration's mark-to-market. No look-ahead.

Cash settlement
---------------
``PortfolioConfig.settle_days`` controls when SELL proceeds become reusable
for new BUYs:

* ``settle_days = 0`` (default) \u2014 margin-account / no settlement constraint.
  A same-day rotation works: sell A, buy B at the same close.
* ``settle_days = 1`` \u2014 US Reg-T **cash account** (post-May-2024 T+1).
  A sell on day ``t`` lands in *unsettled* funds and only becomes spendable at
  close ``t+1``. Same-day BUYs on rotation days are funded from settled cash
  only; if there isn't enough, the buy is scaled down or skipped (the next
  daily run will retry once cash settles). Equity always includes unsettled
  funds, so total NAV is unchanged \u2014 only the *deployable* cash differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    starting_capital: float = 5000.0
    n_long: int = 5
    n_short: int = 0          # 0 = long-only (default for $5000 retail)
    cost_bps: float = 0.0     # 0 = manual execution at brokers with no commission
    min_trade_dollars: float = 25.0   # skip dust trades
    rebalance_threshold: float = 0.20  # don't rebalance if drift < 20%
    # Risk overlays (used as decision rules surfaced on the page; you choose to
    # honor them or just rotate daily).
    stop_loss_pct: float = 0.08         # close any position down 8% from entry
    time_stop_days: int = 10            # close any position held >10 trading days
    # Cash settlement model. 0 = margin account (sell proceeds usable same-day).
    # 1 = US Reg-T cash account: sell proceeds settle on the NEXT trading day
    # (T+1, the rule that took effect May 2024). Same-day rotation BUYs are then
    # constrained by the *settled* cash balance.
    settle_days: int = 0


@dataclass
class Order:
    action: Literal["BUY", "SELL"]
    ticker: str
    shares: int
    limit_price: float
    notional: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "action": self.action, "ticker": self.ticker,
            "shares": self.shares,
            "limit_price": round(self.limit_price, 2),
            "notional": round(self.notional, 2),
            "rationale": self.rationale,
        }


@dataclass
class PortfolioState:
    cash: float
    holdings: dict[str, int] = field(default_factory=dict)
    entry_prices: dict[str, float] = field(default_factory=dict)
    entry_dates: dict[str, str] = field(default_factory=dict)   # ISO date string per ticker
    # SELL proceeds that have not yet settled. List of ``(settle_date_iso, amount)``.
    # Released into ``cash`` at the start of each trading day whose date is
    # >= settle_date_iso. Empty when settle_days == 0.
    unsettled: list = field(default_factory=list)

    @property
    def unsettled_total(self) -> float:
        return float(sum(amt for _, amt in self.unsettled))


def _equal_weight_targets(today_preds: pd.Series, n_long: int, n_short: int
                          ) -> dict[str, float]:
    """Return ticker -> target weight in [-1, 1]. Sum of |weights| <= 1.0."""
    valid = today_preds.dropna()
    if len(valid) < n_long + n_short:
        return {}
    sorted_ = valid.sort_values()
    targets: dict[str, float] = {}
    if n_long > 0:
        long_set = sorted_.tail(n_long).index
        per = 1.0 / (n_long + n_short) if n_short > 0 else 1.0 / n_long
        for t in long_set:
            targets[t] = per
    if n_short > 0:
        short_set = sorted_.head(n_short).index
        per = 1.0 / (n_long + n_short)
        for t in short_set:
            targets[t] = -per
    return targets


def simulate_portfolio(preds_df: pd.DataFrame,
                       panel: pd.DataFrame,
                       cfg: PortfolioConfig) -> tuple[pd.DataFrame, pd.DataFrame, PortfolioState]:
    """Replay daily predictions against close prices.

    Returns
    -------
    history : DataFrame indexed by date, columns:
        equity, cash, n_positions, daily_ret, pnl, gross_long, gross_short, turnover
    trades : DataFrame, one row per executed buy/sell (date, ticker, action, shares, price, ...)
    final_state : PortfolioState as of the last simulated day
    """
    df = preds_df[["date", "ticker", "y_pred"]].dropna().copy()
    df["date"] = pd.to_datetime(df["date"])

    # Close-price grid (date x ticker) using adj_close so dividends/splits are handled.
    p = (panel[["date", "ticker", "adj_close"]].copy()
           .assign(date=lambda d: pd.to_datetime(d["date"]))
           .pivot(index="date", columns="ticker", values="adj_close")
           .sort_index())

    # Pivot predictions to (date x ticker).
    preds = df.pivot(index="date", columns="ticker", values="y_pred").sort_index()
    # Restrict to dates where we have BOTH predictions and prices.
    common_dates = preds.index.intersection(p.index)
    preds = preds.loc[common_dates]

    state = PortfolioState(cash=cfg.starting_capital)
    history: list[dict] = []
    trades: list[dict] = []
    prev_equity = cfg.starting_capital

    # Map each trading date -> the settle date that a sell *today* resolves to.
    # We use trading-day arithmetic (not calendar days) so weekends/holidays
    # don't create phantom settlement gaps.
    date_index = list(common_dates)
    date_pos = {d: i for i, d in enumerate(date_index)}
    settle_d = max(int(cfg.settle_days), 0)

    def _settle_date_for(d):
        if settle_d <= 0:
            return None
        i = date_pos[d] + settle_d
        return date_index[i] if i < len(date_index) else None

    for d in common_dates:
        # 1) Release any unsettled proceeds whose settle date is on/before today.
        if state.unsettled:
            still: list[tuple] = []
            released = 0.0
            for sd, amt in state.unsettled:
                if sd <= d:
                    released += amt
                else:
                    still.append((sd, amt))
            if released:
                state.cash += released
            state.unsettled = still

        # 2) Mark-to-market BEFORE trading using today's close.
        prices_today = p.loc[d]
        mtm = sum(state.holdings.get(t, 0) * prices_today.get(t, np.nan)
                  for t in state.holdings)
        if np.isnan(mtm):
            mtm = sum(state.holdings.get(t, 0) * prices_today.get(t, prices_today.dropna().mean())
                      for t in state.holdings)
        # Equity (NAV) includes unsettled funds; deployable cash does not.
        equity = state.cash + state.unsettled_total + (mtm or 0)

        # 3) Compute target weights/shares from today's predictions.
        target_w = _equal_weight_targets(preds.loc[d], cfg.n_long, cfg.n_short)
        target_shares: dict[str, int] = {}
        for tkr, w in target_w.items():
            px = float(prices_today.get(tkr, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            target_shares[tkr] = int(np.floor(equity * w / px))

        # 4) Build sell + buy lists; execute SELLs first, then BUYs.
        all_tickers = set(target_shares.keys()) | set(state.holdings.keys())
        sells: list[tuple[str, int, float]] = []   # (ticker, shares_to_sell, px)
        buys:  list[tuple[str, int, float]] = []   # (ticker, shares_to_buy, px)
        for tkr in sorted(all_tickers):
            cur = state.holdings.get(tkr, 0)
            tgt = target_shares.get(tkr, 0)
            if cur == tgt:
                continue
            px = float(prices_today.get(tkr, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            delta = tgt - cur
            notional = abs(delta) * px
            if notional < cfg.min_trade_dollars and tgt != 0:
                continue
            if delta < 0:
                sells.append((tkr, -delta, px))
            else:
                buys.append((tkr, delta, px))

        turnover = 0.0
        skipped_unsettled = 0.0

        # 4a) SELLs: proceeds either go straight to cash (settle_days==0) or
        # to the unsettled queue tagged with the future settle date.
        for tkr, sh, px in sells:
            notional = sh * px
            cost = notional * cfg.cost_bps / 1e4
            net = notional - cost
            sd = _settle_date_for(d)
            if sd is None:
                state.cash += net
            else:
                state.unsettled.append((sd, net))
            cur = state.holdings.get(tkr, 0)
            new_cur = cur - sh
            if new_cur <= 0:
                state.holdings.pop(tkr, None)
                state.entry_prices.pop(tkr, None)
                state.entry_dates.pop(tkr, None)
            else:
                state.holdings[tkr] = new_cur
            trades.append({"date": d, "ticker": tkr, "action": "SELL",
                           "shares": int(sh), "price": px,
                           "notional": notional, "cost": cost})
            turnover += notional

        # 4b) BUYs: cap by available SETTLED cash (state.cash). When cash is
        # short, scale the BUY down to what we can afford; below dust, skip.
        for tkr, want_sh, px in buys:
            cost_each = px * cfg.cost_bps / 1e4
            unit_cost = px + cost_each
            if unit_cost <= 0:
                continue
            affordable = int(state.cash // unit_cost)
            shares = min(want_sh, max(affordable, 0))
            if shares <= 0:
                # Couldn't afford any of this BUY today (likely T+1 cash drag).
                skipped_unsettled += want_sh * px
                continue
            notional = shares * px
            cost = notional * cfg.cost_bps / 1e4
            if notional < cfg.min_trade_dollars:
                skipped_unsettled += notional
                continue
            state.cash -= notional + cost
            cur = state.holdings.get(tkr, 0)
            new_cur = cur + shares
            state.entry_prices[tkr] = (
                px if cur == 0
                else (state.entry_prices.get(tkr, px) * cur + px * shares) / max(new_cur, 1)
            )
            if cur == 0:
                state.entry_dates[tkr] = pd.Timestamp(d).strftime("%Y-%m-%d")
            state.holdings[tkr] = new_cur
            trades.append({"date": d, "ticker": tkr, "action": "BUY",
                           "shares": int(shares), "price": px,
                           "notional": notional, "cost": cost})
            turnover += notional

        # 5) Re-mark equity after trades.
        mtm_post = sum(state.holdings.get(t, 0) * prices_today.get(t, np.nan)
                       for t in state.holdings)
        equity = state.cash + state.unsettled_total + (mtm_post or 0)
        daily_ret = (equity / prev_equity - 1) if prev_equity else 0.0

        gross_long  = sum(state.holdings.get(t, 0) * prices_today.get(t, 0)
                          for t, w in target_w.items() if w > 0)
        gross_short = sum(state.holdings.get(t, 0) * prices_today.get(t, 0)
                          for t, w in target_w.items() if w < 0)

        history.append({
            "date": d, "equity": equity, "cash": state.cash,
            "unsettled": state.unsettled_total,
            "n_positions": len(state.holdings),
            "daily_ret": daily_ret,
            "pnl": equity - prev_equity,
            "gross_long":  gross_long, "gross_short": gross_short,
            "turnover":    turnover,
            "skipped_unsettled": skipped_unsettled,
        })
        prev_equity = equity

    history_df = pd.DataFrame(history).set_index("date").sort_index()
    trades_df  = pd.DataFrame(trades).sort_values("date") if trades else pd.DataFrame(
        columns=["date", "ticker", "action", "shares", "price", "notional", "cost"]
    )
    return history_df, trades_df, state


def compute_stats(history: pd.DataFrame, trades: pd.DataFrame,
                  starting_capital: float) -> dict:
    """All the numbers the dashboard shows."""
    if history.empty:
        return {"days": 0}
    equity = history["equity"]
    daily_ret = history["daily_ret"].fillna(0)
    cum = equity / starting_capital

    days = len(daily_ret)
    final_equity = float(equity.iloc[-1])
    total_return = final_equity / starting_capital - 1
    cagr = (cum.iloc[-1]) ** (252 / max(days, 1)) - 1 if days > 0 else 0.0
    vol  = float(daily_ret.std() * np.sqrt(252))
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252) + 1e-12) if days > 1 else 0.0
    drawdown = (cum / cum.cummax() - 1)
    max_dd = float(drawdown.min())
    win_rate = float((daily_ret > 0).mean())

    pos = daily_ret[daily_ret > 0]
    neg = daily_ret[daily_ret < 0]
    avg_win = float(pos.mean()) if len(pos) > 0 else 0.0
    avg_loss = float(neg.mean()) if len(neg) > 0 else 0.0
    profit_factor = (pos.sum() / abs(neg.sum())) if abs(neg.sum()) > 1e-12 else float("inf")

    n_trades = int(len(trades))
    today_pnl = float(history["pnl"].iloc[-1])
    today_ret = float(history["daily_ret"].iloc[-1])
    best_day  = float(daily_ret.max()) if days > 0 else 0.0
    worst_day = float(daily_ret.min()) if days > 0 else 0.0

    # Realised round-trip win rate per CLOSED trade (BUY then SELL on same ticker).
    closed_trades_win = float("nan")
    if not trades.empty and {"date", "ticker", "action", "price", "shares"}.issubset(trades.columns):
        wins, total = 0, 0
        for tkr, g in trades.groupby("ticker"):
            g = g.sort_values("date").reset_index(drop=True)
            buy_qty, buy_cost = 0, 0.0
            for _, row in g.iterrows():
                if row["action"] == "BUY":
                    buy_qty += int(row["shares"])
                    buy_cost += float(row["shares"]) * float(row["price"])
                elif row["action"] == "SELL" and buy_qty > 0:
                    sell_qty = min(int(row["shares"]), buy_qty)
                    avg_buy = buy_cost / buy_qty if buy_qty else 0
                    pnl = (float(row["price"]) - avg_buy) * sell_qty
                    total += 1
                    if pnl > 0:
                        wins += 1
                    buy_cost = buy_cost * (buy_qty - sell_qty) / buy_qty if buy_qty else 0
                    buy_qty -= sell_qty
        closed_trades_win = wins / total if total else float("nan")

    return {
        "days": days,
        "final_equity": final_equity,
        "starting_capital": starting_capital,
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "trade_win_rate": closed_trades_win,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "today_pnl": today_pnl,
        "today_ret": today_ret,
        "best_day": best_day,
        "worst_day": worst_day,
        "n_trades": n_trades,
    }


def realized_pnl_by_ticker(trades: pd.DataFrame) -> pd.Series:
    """Cumulative *realized* P/L per ticker from the trade log (long-only sim).

    Uses the same average-cost inventory math as :func:`simulate_portfolio`:
    on each SELL, PnL = (sell price - average buy price) \u00d7 shares sold.
    Open lots still held at the end of the backtest are **not** included
    (unrealized).
    """
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    need = {"date", "ticker", "action", "price", "shares"}
    if not need.issubset(trades.columns):
        return pd.Series(dtype=float)
    t = trades.sort_values("date").reset_index(drop=True)
    qty: dict[str, int] = {}
    cost: dict[str, float] = {}
    realized: dict[str, float] = {}

    for _, row in t.iterrows():
        tk = str(row["ticker"])
        sh = int(row["shares"])
        px = float(row["price"])
        if row["action"] == "BUY":
            qty[tk] = qty.get(tk, 0) + sh
            cost[tk] = cost.get(tk, 0.0) + sh * px
        else:  # SELL
            q0 = qty.get(tk, 0)
            if q0 <= 0:
                continue
            sell_sh = min(sh, q0)
            avg = cost[tk] / q0 if q0 else 0.0
            pnl = (px - avg) * sell_sh
            realized[tk] = realized.get(tk, 0.0) + pnl
            cost[tk] = avg * (q0 - sell_sh)
            qty[tk] = q0 - sell_sh
            if qty[tk] <= 0:
                del qty[tk]
                del cost[tk]

    if not realized:
        return pd.Series(dtype=float)
    return pd.Series(realized, dtype=float).sort_values(ascending=False)


def compute_hit_rate(history_predictions: pd.DataFrame) -> float:
    """% of long predictions whose actual_xret was positive
    + % of short predictions whose actual_xret was negative."""
    h = history_predictions.dropna(subset=["actual_xret"]).copy()
    h = h[h["side"].isin(["long", "short"])]
    if h.empty:
        return float("nan")
    h["correct"] = ((h["side"] == "long")  & (h["actual_xret"] > 0)) | \
                    ((h["side"] == "short") & (h["actual_xret"] < 0))
    return float(h["correct"].mean())


def make_today_orders(latest_preds: pd.DataFrame,
                      panel: pd.DataFrame,
                      current_state: PortfolioState,
                      cfg: PortfolioConfig) -> list[Order]:
    """What to BUY / SELL tomorrow morning to align with today's predictions.

    `current_state` is the portfolio AFTER the last simulated day. Today's
    predictions were computed at *today's* close; we now compute the deltas
    needed to reach the new target basket.
    """
    p = panel[["date", "ticker", "adj_close"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    last_date = p["date"].max()
    today_prices = p[p["date"] == last_date].set_index("ticker")["adj_close"]

    # Using current_state.equity = cash + holdings * latest close.
    holdings_value = sum(current_state.holdings.get(t, 0) * today_prices.get(t, np.nan)
                         for t in current_state.holdings)
    if np.isnan(holdings_value):
        holdings_value = 0.0
    equity = current_state.cash + holdings_value

    today_pred = (latest_preds.set_index("ticker")["y_pred"]
                  if "y_pred" in latest_preds.columns else
                  latest_preds.set_index("ticker").iloc[:, 0])
    target_w = _equal_weight_targets(today_pred, cfg.n_long, cfg.n_short)

    target_shares: dict[str, int] = {}
    for tkr, w in target_w.items():
        px = float(today_prices.get(tkr, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        target_shares[tkr] = int(np.floor(equity * w / px))

    orders: list[Order] = []
    all_tkrs = set(target_shares.keys()) | set(current_state.holdings.keys())
    for tkr in sorted(all_tkrs):
        cur = current_state.holdings.get(tkr, 0)
        tgt = target_shares.get(tkr, 0)
        if cur == tgt:
            continue
        px = float(today_prices.get(tkr, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        delta = tgt - cur
        notional = abs(delta) * px
        if notional < cfg.min_trade_dollars and tgt != 0:
            continue
        if delta > 0:
            rationale = ("OPEN long position (entered top-N basket)"
                         if cur == 0 else "INCREASE long position")
            orders.append(Order("BUY", tkr, int(delta), px, notional, rationale))
        else:
            rationale = ("CLOSE position (left top-N basket)" if tgt == 0
                         else "REDUCE position")
            orders.append(Order("SELL", tkr, int(-delta), px, notional, rationale))
    return orders


def build_daily_views(history: pd.DataFrame,
                       trades: pd.DataFrame,
                       preds_df: pd.DataFrame,
                       panel: pd.DataFrame,
                       n_top: int = 5,
                       max_days: int | None = None,
                       stop_loss_pct: float = 0.08) -> dict:
    """Build a per-day "scrub through the backtest" data structure.

    For each trading day in `history`, returns:
      - `equity`, `daily_pnl`, `daily_ret`
      - `recs`: top-`n_top` predicted tickers (the simulator's basket that day).
        Action tags align with fills: BUY / ADD / TRIM / HOLD for names staying
        in the basket (see `simulate_portfolio` equal-weight rounding), plus SELL
        for rotations out.
      - `fills`: trades the simulator executed that day, with the actual fill
                price, the next trading day's close, the realised 1-day return
                and $ PnL contribution
      - `top_realized`: top 10 tickers by **cumulative realized** P/L through
        that close (avg-cost basis on every sell up to and including that day)

    The intended UX: a date dropdown / prev-next stepper in the dashboard. JSON
    output is rounded to keep the embedded payload small.
    """
    if history.empty:
        return {}
    cols_needed = ["date", "ticker", "adj_close"]
    for c in ("open", "high", "low", "close"):
        if c in panel.columns:
            cols_needed.append(c)
    p = panel[cols_needed].copy()
    p["date"] = pd.to_datetime(p["date"])
    # Adjusted close for PnL math (consistent with simulator).
    prices = (p.pivot(index="date", columns="ticker", values="adj_close")
                .sort_index())
    # Raw OHLC for display only (so users see the actual session bar).
    raw_ohlc = {}
    for c in ("open", "high", "low", "close"):
        if c in p.columns:
            raw_ohlc[c] = (p.pivot(index="date", columns="ticker", values=c)
                             .sort_index())

    pr = preds_df.copy()
    pr["date"] = pd.to_datetime(pr["date"])
    tr = trades.copy() if not trades.empty else pd.DataFrame(
        columns=["date", "ticker", "action", "shares", "price", "notional"])
    if not tr.empty:
        tr["date"] = pd.to_datetime(tr["date"])
        tr = tr.sort_values("date").reset_index(drop=True)

    dates = list(pd.to_datetime(history.index))
    if max_days and len(dates) > max_days:
        step = max(len(dates) // max_days, 1)
        dates = dates[::step]
        if dates[-1] != pd.to_datetime(history.index[-1]):
            dates.append(pd.to_datetime(history.index[-1]))

    # Replay trades to track running holdings -> snapshot at end of each `d`.
    holdings_state: dict[str, dict] = {}    # ticker -> {shares, cost (cumulative $ basis)}
    realized_cum: dict[str, float] = {}     # ticker -> cumulative realized P/L on sells so far
    trade_iter = iter(tr.itertuples()) if not tr.empty else iter([])
    next_trade = next(trade_iter, None)

    def _bar(d, tk):
        """Return (open, high, low, close) for ticker `tk` on date `d`, raw."""
        out = {}
        for k, frame in raw_ohlc.items():
            if d in frame.index and tk in frame.columns:
                v = frame.loc[d, tk]
                out[k] = float(v) if pd.notna(v) else None
            else:
                out[k] = None
        return out

    views: dict[str, dict] = {}
    for d in dates:
        d_str = d.strftime("%Y-%m-%d")

        # Snapshot pre-trade holdings (i.e., what we held going INTO `d`).
        # This is used to tag each recommendation with BUY (new entry),
        # HOLD (already owned, model still likes it), or SELL (we own it but
        # model rotated it out -> exit at close).
        pre_trade_tickers = set(holdings_state.keys())
        pre_trade_state = {tk: dict(h) for tk, h in holdings_state.items()}

        # Apply every trade dated on or before this `d` to the running state.
        while next_trade is not None and next_trade.date <= d:
            tk = next_trade.ticker
            sh = int(next_trade.shares)
            px = float(next_trade.price)
            if next_trade.action == "BUY":
                h = holdings_state.setdefault(tk, {"shares": 0, "cost": 0.0})
                h["shares"] += sh
                h["cost"]   += sh * px
            else:                              # SELL
                if tk in holdings_state:
                    h = holdings_state[tk]
                    sell_sh = min(sh, h["shares"])
                    avg = (h["cost"] / h["shares"]) if h["shares"] else 0.0
                    realized_cum[tk] = realized_cum.get(tk, 0.0) + (px - avg) * sell_sh
                    h["shares"] -= sell_sh
                    h["cost"] = avg * h["shares"] if h["shares"] > 0 else 0.0
                    if h["shares"] <= 0:
                        del holdings_state[tk]
            next_trade = next(trade_iter, None)

        # Snapshot of holdings AT END OF day d.
        holdings_snapshot = []
        for tk, h in holdings_state.items():
            cur_px = (float(prices.loc[d, tk])
                      if (d in prices.index and tk in prices.columns
                          and pd.notna(prices.loc[d, tk]))
                      else None)
            shares = int(h["shares"])
            avg_cost = (h["cost"] / shares) if shares > 0 else 0.0
            mv = (shares * cur_px) if cur_px is not None else None
            upnl = ((cur_px - avg_cost) * shares) if cur_px is not None else None
            upct = ((cur_px / avg_cost) - 1) if cur_px is not None and avg_cost > 0 else None
            holdings_snapshot.append({
                "ticker": tk,
                "shares": shares,
                "entry": round(avg_cost, 2),
                "current": round(cur_px, 2) if cur_px is not None else None,
                "mv": round(mv, 2) if mv is not None else None,
                "upnl": round(upnl, 2) if upnl is not None else None,
                "upct": round(upct, 4) if upct is not None else None,
            })
        holdings_snapshot.sort(key=lambda r: r["mv"] or 0, reverse=True)

        # --- That day's simulated trades (source of truth for rebalance deltas) ---
        day_trades = tr[tr["date"] == d] if not tr.empty else tr
        net_share_delta: dict[str, int] = {}
        buy_notional = 0.0
        sell_notional = 0.0
        if not day_trades.empty:
            buy_notional = float(
                day_trades.loc[day_trades["action"] == "BUY", "notional"].sum())
            sell_notional = float(
                day_trades.loc[day_trades["action"] == "SELL", "notional"].sum())
            for _, row in day_trades.iterrows():
                tk = str(row["ticker"])
                sh = int(row["shares"])
                if row["action"] == "BUY":
                    net_share_delta[tk] = net_share_delta.get(tk, 0) + sh
                else:
                    net_share_delta[tk] = net_share_delta.get(tk, 0) - sh

        # Read POST-trade accounting from simulator history...
        h_idx = history.index[history.index == d][0]
        eq_val = float(history.loc[h_idx, "equity"])
        pnl_val = float(history.loc[h_idx, "pnl"]) \
                    if "pnl" in history.columns else 0.0
        ret_val = float(history.loc[h_idx, "daily_ret"]) \
                    if "daily_ret" in history.columns else 0.0
        cash_val = float(history.loc[h_idx, "cash"]) \
                    if "cash" in history.columns else 0.0
        unsettled_val = float(history.loc[h_idx, "unsettled"]) \
                    if "unsettled" in history.columns else 0.0
        skipped_val = float(history.loc[h_idx, "skipped_unsettled"]) \
                    if "skipped_unsettled" in history.columns else 0.0
        # ...and reconstruct PRE-trade cash so equal-weight targets match simulate_portfolio().
        cash_pre = cash_val + buy_notional - sell_notional
        mtm_pre = 0.0
        for tk, h in pre_trade_state.items():
            if d in prices.index and tk in prices.columns:
                px_ = prices.loc[d, tk]
                if pd.notna(px_):
                    mtm_pre += int(h["shares"]) * float(px_)
        equity_pre = cash_pre + mtm_pre
        slot_dollar = (equity_pre / n_top) if n_top > 0 else 0.0

        dp = pr[pr["date"] == d].sort_values("y_pred", ascending=False)
        recs: list[dict] = []
        basket_tickers: set[str] = set()
        for i, row in enumerate(dp.head(n_top).itertuples()):
            tk = row.ticker
            basket_tickers.add(tk)
            close_px = (float(prices.loc[d, tk])
                        if (d in prices.index and tk in prices.columns
                            and pd.notna(prices.loc[d, tk]))
                        else None)
            limit_sell = (round(close_px * (1 + float(row.y_pred)), 2)
                          if close_px is not None else None)
            stop_loss = (round(close_px * (1 - stop_loss_pct), 2)
                         if close_px is not None else None)

            pre_sh = int(pre_trade_state.get(tk, {}).get("shares", 0))
            delta = int(net_share_delta.get(tk, 0))
            post_sh = pre_sh + delta

            entry_px = None
            if pre_sh > 0:
                ph = pre_trade_state[tk]
                entry_px = (ph["cost"] / ph["shares"]) if ph["shares"] > 0 else None

            trade_shares = abs(delta)
            if pre_sh == 0:
                action = "BUY"
                if entry_px is None:
                    entry_px = close_px
            elif delta > 0:
                action = "ADD"
            elif delta < 0:
                action = "TRIM"
            else:
                action = "HOLD"

            # Fallback if predictions JSON and trades diverge (shouldn't happen).
            if action == "BUY" and pre_sh == 0 and delta == 0 and close_px:
                post_sh = int(np.floor(slot_dollar / close_px))
                trade_shares = post_sh
                delta = post_sh

            dollars_qty = ((post_sh * close_px)
                           if close_px is not None and post_sh else None)

            bar = _bar(d, tk)
            recs.append({
                "action": action,
                "rank": i + 1,
                "ticker": tk,
                "pred_xret": round(float(row.y_pred), 4),
                "close": round(close_px, 2) if close_px is not None else None,
                "open":  round(bar["open"],  2) if bar.get("open")  is not None else None,
                "high":  round(bar["high"],  2) if bar.get("high")  is not None else None,
                "low":   round(bar["low"],   2) if bar.get("low")   is not None else None,
                "raw_close": round(bar["close"], 2) if bar.get("close") is not None else None,
                "entry": round(entry_px, 2) if entry_px is not None else None,
                "shares": post_sh,
                "trade_shares": trade_shares,
                "trade_delta": delta,
                "dollars": round(dollars_qty, 2) if dollars_qty is not None else None,
                "limit_sell": limit_sell,
                "stop_loss": stop_loss,
            })

        # SELL recommendations: anything we held going into today that's NOT in
        # today's new basket -> exit at next session close to fund the rotation.
        for tk in sorted(pre_trade_tickers - basket_tickers):
            close_px = (float(prices.loc[d, tk])
                        if (d in prices.index and tk in prices.columns
                            and pd.notna(prices.loc[d, tk]))
                        else None)
            ph = pre_trade_state[tk]
            entry_px = (ph["cost"] / ph["shares"]) if ph["shares"] > 0 else None
            shares_qty = int(ph["shares"])
            dollars_qty = ((shares_qty * close_px)
                           if close_px is not None else None)
            upct = ((close_px / entry_px) - 1
                    if (close_px is not None and entry_px and entry_px > 0)
                    else None)
            recs.append({
                "action": "SELL",
                "rank": None,
                "ticker": tk,
                "pred_xret": None,
                "close": round(close_px, 2) if close_px is not None else None,
                "entry": round(entry_px, 2) if entry_px is not None else None,
                "shares": 0,
                "trade_shares": shares_qty,
                "trade_delta": -shares_qty,
                "dollars": round(dollars_qty, 2) if dollars_qty is not None else None,
                "limit_sell": None,
                "stop_loss": None,
                "upct": round(upct, 4) if upct is not None else None,
            })

        idx_pos = prices.index.searchsorted(d, side="right")
        next_day = prices.index[idx_pos] if idx_pos < len(prices.index) else None

        fills = []
        for _, row in day_trades.iterrows():
            tk = row["ticker"]
            fill_px = float(row["price"])
            today_bar = _bar(d, tk)
            next_bar = _bar(next_day, tk) if next_day is not None else {
                "open": None, "high": None, "low": None, "close": None}
            next_close = None
            if next_day is not None and tk in prices.columns:
                v = prices.loc[next_day, tk]
                if pd.notna(v):
                    next_close = float(v)
            shares = int(row["shares"])
            sign = 1 if row["action"] == "BUY" else -1
            if next_close is not None and fill_px > 0:
                pct = (next_close / fill_px - 1) * sign
                pnl = (next_close - fill_px) * shares * sign
            else:
                pct = None
                pnl = None
            fills.append({
                "action": row["action"],
                "ticker": tk,
                "shares": shares,
                "fill_price": round(fill_px, 2),
                "today_open":  round(today_bar["open"],  2) if today_bar.get("open")  is not None else None,
                "today_high":  round(today_bar["high"],  2) if today_bar.get("high")  is not None else None,
                "today_low":   round(today_bar["low"],   2) if today_bar.get("low")   is not None else None,
                "today_close": round(today_bar["close"], 2) if today_bar.get("close") is not None else None,
                "next_open":   round(next_bar["open"],  2) if next_bar.get("open")  is not None else None,
                "next_high":   round(next_bar["high"],  2) if next_bar.get("high")  is not None else None,
                "next_low":    round(next_bar["low"],   2) if next_bar.get("low")   is not None else None,
                "next_close": round(next_close, 2) if next_close is not None else None,
                "next_pct": round(pct, 4) if pct is not None else None,
                "next_pnl": round(pnl, 2) if pnl is not None else None,
                "filled": True,
            })

        top_realized: list[dict] = []
        if realized_cum:
            items = sorted(realized_cum.items(), key=lambda x: -x[1])[:10]
            top_realized = [{"ticker": t, "realized": round(float(v), 2)}
                            for t, v in items]

        views[d_str] = {
            "date": d_str,
            "equity": round(eq_val, 2),
            "cash": round(cash_val, 2),
            "unsettled": round(unsettled_val, 2),
            "skipped_unsettled": round(skipped_val, 2),
            "daily_pnl": round(pnl_val, 2),
            "daily_ret": round(ret_val, 4),
            "holdings": holdings_snapshot,
            "recs": recs,
            "fills": fills,
            "top_realized": top_realized,
        }
    return views


def make_holdings_view(state: PortfolioState,
                       panel: pd.DataFrame,
                       cfg: PortfolioConfig | None = None) -> pd.DataFrame:
    """Tidy table of current holdings for the dashboard.

    Adds explicit decision-rule columns (stop_loss_price, days_held, time_stop_left)
    so the user can see exactly when to bail on a position.
    """
    cfg = cfg or PortfolioConfig()
    if not state.holdings:
        return pd.DataFrame(columns=["ticker", "shares", "entry_price",
                                      "current_price", "market_value",
                                      "unrealized_pnl", "unrealized_pct",
                                      "stop_loss_price", "days_held",
                                      "time_stop_left", "decision"])
    p = panel[["date", "ticker", "adj_close"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    last_date = p["date"].max()
    today_prices = p[p["date"] == last_date].set_index("ticker")["adj_close"]

    rows = []
    for tkr, shares in state.holdings.items():
        px = float(today_prices.get(tkr, np.nan))
        ep = float(state.entry_prices.get(tkr, np.nan))
        ed_str = state.entry_dates.get(tkr)
        ed = pd.Timestamp(ed_str) if ed_str else last_date
        mv = shares * px
        upnl = (px - ep) * shares if np.isfinite(ep) and np.isfinite(px) else np.nan
        upct = (px / ep - 1) if np.isfinite(ep) and ep > 0 else np.nan

        stop_px = ep * (1 - cfg.stop_loss_pct) if np.isfinite(ep) else np.nan
        # Trading days held: count business days between entry and today.
        days_held = int(pd.bdate_range(ed, last_date).size) - 1 if ed_str else 0
        time_stop_left = max(cfg.time_stop_days - days_held, 0)

        if np.isfinite(upct) and upct <= -cfg.stop_loss_pct:
            decision = "STOP-LOSS HIT \u2192 SELL"
        elif time_stop_left == 0:
            decision = "TIME-STOP \u2192 SELL tomorrow"
        else:
            decision = f"HOLD (rotate when basket changes)"
        rows.append({"ticker": tkr, "shares": shares,
                      "entry_price": ep, "current_price": px,
                      "market_value": mv,
                      "unrealized_pnl": upnl, "unrealized_pct": upct,
                      "stop_loss_price": stop_px,
                      "days_held": days_held,
                      "time_stop_left": time_stop_left,
                      "decision": decision})
    return pd.DataFrame(rows).sort_values("market_value", ascending=False)
