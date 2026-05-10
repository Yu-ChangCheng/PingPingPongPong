"""Builds the GitHub Pages static site (docs/index.html).

Self-contained: uses Plotly via CDN. Single HTML file + CSV data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Config


def _fig_div(fig: go.Figure, div_id: str, height: int | None = None) -> str:
    if height is not None:
        fig.update_layout(height=height)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displaylogo": False})


def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def _fmt_pnl(x: float) -> str:
    """Signed money for PnL ('+$1,234.56' / '-$789.01')."""
    if not np.isfinite(x):
        return "n/a"
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):,.2f}"


def _fmt_pct(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x*100:+.{digits}f}%"


def _fmt_num(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


# ---------- charts -----------------------------------------------------------

def _today_predictions_table(latest: pd.DataFrame, panel: pd.DataFrame,
                              long_n: int, short_n: int,
                              starting_capital: float,
                              stop_loss_pct: float = 0.08) -> str:
    """Rich HTML table: top-N (BUY) and bottom-N (AVOID) with full execution
    plan (entry price, shares, take-profit limit, stop-loss, default exit)."""
    last_date = pd.to_datetime(panel["date"].max())
    last_prices = (panel[pd.to_datetime(panel["date"]) == last_date]
                    .set_index("ticker")["adj_close"])
    df = latest.copy()
    df["close"] = df["ticker"].map(last_prices).astype(float)
    df = df.sort_values("y_pred", ascending=False).reset_index(drop=True)

    per_long = starting_capital / long_n if long_n else 0
    long_rows = df.head(long_n).copy()
    long_rows["alloc"] = per_long
    long_rows["shares"] = (per_long / long_rows["close"]).fillna(0)
    long_rows["limit_sell"] = long_rows["close"] * (1 + long_rows["y_pred"])
    long_rows["stop_loss"]  = long_rows["close"] * (1 - stop_loss_pct)

    short_rows = df.tail(short_n).iloc[::-1].copy() if short_n > 0 else pd.DataFrame()

    stop_pct_ui = int(round(stop_loss_pct * 100))
    head = (
        "<tr>"
        '<th title="BUY basket vs AVOID">Side</th>'
        "<th>#</th><th>Ticker</th>"
        '<th title="Predicted next-day excess vs benchmark">Pred xret</th>'
        '<th title="Last adj. close: buy-limit anchor + share sizing">Close</th>'
        f'<th title="Shares \u2248 portfolio\u00f7{long_n}\u00f7close (~{starting_capital:,.0f} fresh)">Sh</th>'
        '<th title="Optional take-profit (simulator does not use this)">Lim. sell</th>'
        f'<th title="Reference stop \u2212{stop_pct_ui}% vs close; holdings use your fill">Stop</th>'
        "<th>Exit</th></tr>"
    )
    rows = []
    for i, r in enumerate(long_rows.itertuples(), start=1):
        close_str  = f"${r.close:,.2f}" if np.isfinite(r.close) else "n/a"
        shares_str = f"{int(r.shares)}" if np.isfinite(r.shares) else "n/a"
        ls_str = (f"<span style='color:#0b6e4f'>${r.limit_sell:,.2f}</span>"
                  if np.isfinite(r.limit_sell) else "n/a")
        sl_str = (f"<span style='color:#b34030'>${r.stop_loss:,.2f}</span>"
                  if np.isfinite(r.stop_loss) else "n/a")
        rows.append(
            f"<tr class='winner'>"
            f"<td><span class='tag tag-buy'>BUY</span></td>"
            f"<td>#{i}</td>"
            f"<td><b>{r.ticker}</b></td>"
            f"<td>{r.y_pred*100:+.2f}%</td>"
            f"<td>{close_str}</td>"
            f"<td>{shares_str}</td>"
            f"<td>{ls_str}</td>"
            f"<td>{sl_str}</td>"
            f"<td style='color:#666'>next session close</td>"
            f"</tr>"
        )
    for i, r in enumerate(short_rows.itertuples(), start=1):
        close_str = f"${r.close:,.2f}" if np.isfinite(r.close) else "n/a"
        rows.append(
            f"<tr class='loser'>"
            f"<td><span class='tag tag-sell'>AVOID</span></td>"
            f"<td>#{i}</td>"
            f"<td><b>{r.ticker}</b></td>"
            f"<td>{r.y_pred*100:+.2f}%</td>"
            f"<td>{close_str}</td>"
            f"<td>\u2014</td>"
            f"<td>\u2014</td>"
            f"<td>\u2014</td>"
            f"<td style='color:#666'>do not buy</td>"
            f"</tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _build_today_chart(latest: pd.DataFrame, long_n: int, short_n: int) -> go.Figure:
    g = latest.sort_values("y_pred").reset_index(drop=True)
    n = len(g)
    colors = []
    for i in range(n):
        if i >= n - long_n:
            colors.append("#0b6e4f")
        elif short_n > 0 and i < short_n:
            colors.append("#b34030")
        else:
            colors.append("rgba(150,150,150,0.55)")
    fig = go.Figure(go.Bar(
        x=g["ticker"], y=g["y_pred"] * 100, marker_color=colors,
        text=[f"{v*100:+.2f}%" for v in g["y_pred"]],
        textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(
        title=f"Tomorrow's predictions \u2014 green = BUY top {long_n}"
              + (f", red = SHORT bottom {short_n}" if short_n > 0 else ""),
        template="plotly_white", height=380, bargap=0.25,
        yaxis_title="Predicted excess return (%)",
        xaxis=dict(tickangle=-60),
        margin=dict(l=60, r=20, t=70, b=70),
    )
    return fig


def _build_equity_chart(history: pd.DataFrame, starting_capital: float) -> go.Figure:
    if history.empty:
        return _empty_chart("No portfolio history yet.")
    eq = history["equity"]
    cum = eq / starting_capital
    drawdown = cum / cum.cummax() - 1

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       row_heights=[0.72, 0.28], vertical_spacing=0.05,
                       subplot_titles=("Portfolio equity ($)", "Drawdown (%)"))
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines",
                              line=dict(color="#0b6e4f", width=2.4),
                              fill="tozeroy", fillcolor="rgba(11,110,79,0.10)",
                              name="Equity", showlegend=False),
                  row=1, col=1)
    fig.add_hline(y=starting_capital, line=dict(color="#bbb", dash="dot"),
                  row=1, col=1, annotation_text=f"start: ${starting_capital:,.0f}",
                  annotation_position="left")
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100,
                              mode="lines", line=dict(color="#b34030", width=1.5),
                              fill="tozeroy", fillcolor="rgba(179,64,48,0.18)",
                              showlegend=False, name="Drawdown"),
                  row=2, col=1)
    fig.update_layout(template="plotly_white", height=460,
                      margin=dict(l=60, r=20, t=60, b=40))
    fig.update_yaxes(title="$",  row=1, col=1)
    fig.update_yaxes(title="%",  row=2, col=1)
    return fig


def _build_live_equity_chart(live: pd.DataFrame, starting_capital: float) -> go.Figure:
    if live is None or live.empty:
        return _empty_chart("Live tracker starts after the first daily run \u2014 "
                              "$5,000 baseline shown above.")
    eq = pd.Series(live["equity"].values,
                    index=pd.to_datetime(live["date"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines+markers",
                              line=dict(color="#0b6e4f", width=2.6),
                              marker=dict(size=6, color="#0b6e4f"),
                              fill="tozeroy", fillcolor="rgba(11,110,79,0.10)",
                              name="$5K live"))
    fig.add_hline(y=starting_capital, line=dict(color="#bbb", dash="dot"),
                  annotation_text=f"start: ${starting_capital:,.0f}",
                  annotation_position="left")
    fig.update_layout(title="Live $5,000 simulated portfolio \u2014 grows day-by-day",
                      template="plotly_white", height=320,
                      yaxis_title="Equity ($)",
                      margin=dict(l=60, r=20, t=60, b=30))
    return fig


def _build_animated_backtest(history: pd.DataFrame,
                              trades: pd.DataFrame,
                              starting_capital: float,
                              n_frames: int = 120) -> go.Figure:
    """Animated equity curve being built across the backtest, with every entry
    and exit drawn as a triangle on the curve.

    DESIGN: trade markers and a faint full curve are drawn ONCE (static layer);
    animation only updates two thin traces (the revealed equity slice + a
    moving "current" dot) and a vertical date cursor. This keeps the HTML
    payload small (a few MB) regardless of trade count.
    """
    if history.empty:
        return _empty_chart("No backtest history to animate.")

    eq = history["equity"].copy()
    eq.index = pd.to_datetime(eq.index)

    if not trades.empty:
        t = trades.copy()
        t["date"] = pd.to_datetime(t["date"])
        t["eq"] = t["date"].map(lambda d: float(eq.asof(d))
                                  if pd.notna(eq.asof(d)) else np.nan)
        t = t.dropna(subset=["eq"])
    else:
        t = pd.DataFrame(columns=["date", "ticker", "action", "shares",
                                    "price", "eq"])

    buys  = t[t["action"] == "BUY"]
    sells = t[t["action"] == "SELL"]

    # Sample ~n_frames evenly-spaced frame dates from the equity index.
    if len(eq) > n_frames:
        idx = np.linspace(0, len(eq) - 1, n_frames).astype(int)
        frame_dates = eq.index[idx]
    else:
        frame_dates = eq.index

    def _hover(df: pd.DataFrame) -> list[str]:
        return [f"<b>{r.ticker}</b><br>{r.action} {int(r.shares)} sh @ ${r.price:,.2f}"
                f"<br>{pd.Timestamp(r.date).strftime('%Y-%m-%d')}"
                f"<br>equity at trade: ${r.eq:,.0f}"
                for r in df.itertuples()]

    fig = go.Figure(
        data=[
            # 0 \u2014 faint full equity curve (static, ghost)
            go.Scatter(x=eq.index, y=eq.values, mode="lines",
                        line=dict(color="rgba(11,110,79,0.18)", width=1),
                        showlegend=False, hoverinfo="skip", name="ghost"),
            # 1 \u2014 revealed equity slice (animated)
            go.Scatter(x=[eq.index[0]], y=[eq.iloc[0]],
                        mode="lines", name="Equity",
                        line=dict(color="#0b6e4f", width=2.6),
                        fill="tozeroy", fillcolor="rgba(11,110,79,0.10)",
                        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"),
            # 2 \u2014 BUY markers (static, all visible from the start)
            go.Scatter(x=buys["date"], y=buys["eq"], mode="markers",
                        name="Entry (BUY)",
                        marker=dict(symbol="triangle-up", size=8,
                                    color="rgba(11,110,79,0.55)",
                                    line=dict(color="#0b6e4f", width=0.6)),
                        text=_hover(buys),
                        hovertemplate="%{text}<extra></extra>"),
            # 3 \u2014 SELL markers (static)
            go.Scatter(x=sells["date"], y=sells["eq"], mode="markers",
                        name="Exit (SELL)",
                        marker=dict(symbol="triangle-down", size=8,
                                    color="rgba(179,64,48,0.55)",
                                    line=dict(color="#b34030", width=0.6)),
                        text=_hover(sells),
                        hovertemplate="%{text}<extra></extra>"),
            # 4 \u2014 cursor dot (animated)
            go.Scatter(x=[eq.index[0]], y=[eq.iloc[0]], mode="markers",
                        marker=dict(size=14, color="#0b6e4f",
                                    line=dict(color="white", width=2)),
                        showlegend=False, hoverinfo="skip"),
        ],
        frames=[
            go.Frame(
                data=[
                    go.Scatter(x=eq.loc[:d].index, y=eq.loc[:d].values),
                    go.Scatter(x=[d], y=[float(eq.asof(d))]),
                ],
                traces=[1, 4],
                name=pd.Timestamp(d).strftime("%Y-%m-%d"),
                layout=go.Layout(
                    title=(f"Backtest equity \u2014 as of "
                           f"{pd.Timestamp(d).strftime('%Y-%m-%d')} &middot; "
                           f"${float(eq.asof(d)):,.0f}"),
                    shapes=[dict(type="line", x0=d, x1=d, yref="paper",
                                  y0=0, y1=1,
                                  line=dict(color="rgba(0,0,0,0.30)",
                                            dash="dot", width=1))],
                ),
            )
            for d in frame_dates
        ],
    )

    fig.update_layout(
        title=(f"Backtest equity \u2014 watch it grow from ${starting_capital:,.0f}"),
        template="plotly_white", height=540,
        margin=dict(l=60, r=20, t=120, b=90),
        xaxis=dict(title="", range=[eq.index.min(), eq.index.max()]),
        yaxis=dict(title="Equity ($)",
                    range=[starting_capital * 0.85, eq.max() * 1.05]),
        legend=dict(orientation="h", y=1.10, x=0),
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "x": 0.01, "xanchor": "left", "y": 1.20, "yanchor": "top",
            "pad": {"t": 0, "r": 6},
            "buttons": [
                {"label": "\u25B6 Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 80, "redraw": True},
                                   "fromcurrent": True,
                                   "transition": {"duration": 0}}]},
                {"label": "\u2759\u2759 Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0, "y": -0.14, "x": 0.0,
            "len": 1.0, "pad": {"t": 30, "b": 0},
            "currentvalue": {"prefix": "Date: ", "font": {"size": 13}},
            "steps": [
                {"args": [[pd.Timestamp(d).strftime("%Y-%m-%d")],
                            {"frame": {"duration": 0, "redraw": True},
                             "mode": "immediate", "transition": {"duration": 0}}],
                 "label": pd.Timestamp(d).strftime("%Y-%m"), "method": "animate"}
                for d in frame_dates
            ],
        }],
    )
    fig.add_hline(y=starting_capital, line=dict(color="#bbb", dash="dot"),
                  annotation_text=f"start ${starting_capital:,.0f}",
                  annotation_position="left")
    return fig


def _build_ticker_timeline(trades: pd.DataFrame) -> go.Figure:
    """Per-ticker buy/sell timeline so you can see who was held when."""
    if trades.empty:
        return _empty_chart("No trades yet.")
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"])
    tickers = sorted(t["ticker"].unique())
    fig = go.Figure()
    for tkr in tickers:
        b = t[(t["ticker"] == tkr) & (t["action"] == "BUY")]
        s = t[(t["ticker"] == tkr) & (t["action"] == "SELL")]
        if not b.empty:
            fig.add_trace(go.Scatter(
                x=b["date"], y=[tkr] * len(b),
                mode="markers", name=f"{tkr} BUY", showlegend=False,
                marker=dict(symbol="triangle-up", size=9,
                            color="rgba(11,110,79,0.7)",
                            line=dict(color="#0b6e4f", width=0.5)),
                hovertemplate=f"<b>{tkr}</b> BUY<br>%{{x|%Y-%m-%d}}<extra></extra>",
            ))
        if not s.empty:
            fig.add_trace(go.Scatter(
                x=s["date"], y=[tkr] * len(s),
                mode="markers", name=f"{tkr} SELL", showlegend=False,
                marker=dict(symbol="triangle-down", size=9,
                            color="rgba(179,64,48,0.7)",
                            line=dict(color="#b34030", width=0.5)),
                hovertemplate=f"<b>{tkr}</b> SELL<br>%{{x|%Y-%m-%d}}<extra></extra>",
            ))
    fig.update_layout(
        title="Per-ticker entries & exits over the backtest",
        template="plotly_white",
        height=max(280, 22 * len(tickers) + 100),
        margin=dict(l=80, r=20, t=60, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _build_backtest_equity(history: pd.DataFrame,
                            starting_capital: float) -> go.Figure:
    """Plain equity curve for the Backtesting tab. The JS layer adds a vertical
    cursor + a marker dot via Plotly.relayout when the user picks a date."""
    if history.empty:
        return _empty_chart("No backtest history yet.")
    eq = history["equity"].copy()
    eq.index = pd.to_datetime(eq.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines", name="Equity",
        line=dict(color="#0b6e4f", width=2.4),
        fill="tozeroy", fillcolor="rgba(11,110,79,0.10)",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ))
    # Cursor placeholder (updated by JS)
    fig.add_trace(go.Scatter(
        x=[eq.index[-1]], y=[eq.iloc[-1]],
        mode="markers", name="Selected",
        marker=dict(size=14, color="#0b6e4f",
                    line=dict(color="white", width=2)),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_hline(y=starting_capital, line=dict(color="#bbb", dash="dot"),
                  annotation_text=f"start ${starting_capital:,.0f}",
                  annotation_position="left")
    fig.update_layout(
        title="Backtest equity \u2014 click below to scrub through any day",
        template="plotly_white", height=380,
        margin=dict(l=60, r=20, t=70, b=40),
        yaxis_title="Equity ($)",
    )
    return fig


def _build_pnl_bars(history: pd.DataFrame) -> go.Figure:
    if history.empty:
        return _empty_chart("No daily PnL yet.")
    pnl = history["pnl"].fillna(0)
    colors = ["#0b6e4f" if v >= 0 else "#b34030" for v in pnl]
    fig = go.Figure(go.Bar(x=pnl.index, y=pnl.values, marker_color=colors,
                            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>"))
    fig.update_layout(title="Daily PnL ($)", template="plotly_white",
                      height=260, margin=dict(l=60, r=20, t=60, b=30),
                      yaxis_title="PnL ($)")
    return fig


def _top_realized_bar_figure(rows: list[dict], title: str) -> go.Figure:
    """Horizontal bar chart from ``[{"ticker": str, "realized": float}, ...]``."""
    if not rows:
        return _empty_chart(
            "No realized sells yet on this date \u2014 step forward on the calendar."
        )
    tickers = [str(r["ticker"]) for r in rows]
    vals = [float(r["realized"]) for r in rows]
    pairs = sorted(zip(vals, tickers), key=lambda x: x[0])
    vals = [p[0] for p in pairs]
    tickers = [p[1] for p in pairs]
    colors = ["#0b6e4f" if float(v) >= 0 else "#b34030" for v in vals]
    texts = []
    for v in vals:
        sign = "+" if v >= 0 else ""
        texts.append(f"{sign}${abs(float(v)):,.0f}")
    fig = go.Figure(go.Bar(
        x=vals,
        y=tickers,
        orientation="h",
        marker_color=colors,
        text=texts,
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>realized P&amp;L: "
            "$%{x:,.2f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title or "Top tickers by realized P&amp;L (cumulative)",
        template="plotly_white",
        height=max(320, 42 * len(rows) + 120),
        margin=dict(l=72, r=80, t=60, b=40),
        xaxis_title="Realized P&amp;L ($) \u2014 sum on all sells (avg-cost basis)",
        showlegend=False,
    )
    return fig


def _build_hit_chart(history_predictions: pd.DataFrame) -> go.Figure:
    h = history_predictions.dropna(subset=["actual_xret"]).copy()
    h = h[h["side"].isin(["long", "short"])]
    if h.empty:
        return _empty_chart("No realised predictions yet \u2014 hit rate appears tomorrow.")
    h["correct"] = ((h["side"] == "long") & (h["actual_xret"] > 0)) | \
                    ((h["side"] == "short") & (h["actual_xret"] < 0))
    daily = h.groupby("for_date")["correct"].mean()
    rolling = daily.rolling(21, min_periods=5).mean()
    fig = go.Figure()
    fig.add_bar(x=daily.index, y=daily.values * 100,
                marker_color="rgba(150,150,150,0.55)",
                name="daily hit rate", showlegend=False)
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values * 100,
                              mode="lines", line=dict(color="#b5651d", width=3),
                              name="21d rolling", showlegend=True))
    fig.add_hline(y=50, line=dict(color="#888", dash="dot"))
    fig.update_layout(title="Hit rate \u2014 % of long/short calls that were directionally correct",
                      template="plotly_white", height=300,
                      yaxis=dict(title="%", range=[0, 100]),
                      margin=dict(l=60, r=20, t=60, b=30))
    return fig


def _build_hot_chart(hot_scored: pd.DataFrame, top: int = 12) -> go.Figure:
    if hot_scored.empty:
        return _empty_chart("No hot stocks scored on this run.")
    s = hot_scored.head(top).sort_values("score")
    fig = go.Figure(go.Bar(
        x=s["score"], y=s["ticker"], orientation="h",
        marker_color="#b5651d",
        text=[f"{v:+.2f}" for v in s["score"]], textposition="outside",
        customdata=np.column_stack([s["ret_short"], s["ret_med"], s["vol_z"]]),
        hovertemplate=("<b>%{y}</b><br>hotness: %{x:+.2f}<br>"
                        "5d ret: %{customdata[0]:+.1%}<br>"
                        "20d ret: %{customdata[1]:+.1%}<br>"
                        "vol z: %{customdata[2]:+.1f}<extra></extra>"),
    ))
    fig.update_layout(title=f"Hot stocks today \u2014 top {top}",
                      template="plotly_white", height=380,
                      xaxis_title="Hotness score (z-sum)",
                      margin=dict(l=80, r=40, t=60, b=30))
    return fig


def _build_importance_chart(imp: pd.Series) -> go.Figure:
    s = imp.sort_values()
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h", marker_color="#3a7ca5",
        text=[f"{v:.1%}" for v in s.values], textposition="outside",
    ))
    fig.update_layout(title="Average feature importance",
                      template="plotly_white", height=520,
                      xaxis_title="Importance",
                      margin=dict(l=140, r=40, t=60, b=30))
    return fig


def _empty_chart(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5,
                        xref="paper", yref="paper", showarrow=False,
                        font=dict(size=16, color="#666"))
    fig.update_layout(template="plotly_white", height=260,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig


# ---------- tables -----------------------------------------------------------

def _kpi_card(label: str, value: str, sub: str = "", color: str = "#0b6e4f") -> str:
    return f"""<div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def _orders_table(orders: list) -> str:
    if not orders:
        return ('<p style="color:#666;font-style:italic;margin:4px 0">'
                'No orders today \u2014 portfolio already aligned with target basket.</p>')
    head = ("<tr><th>Action</th><th>Ticker</th><th>Shares</th>"
            "<th>Limit price</th><th>Notional</th><th>Why</th></tr>")
    rows = []
    for o in orders:
        d = o.to_dict() if hasattr(o, "to_dict") else o
        cls = "buy" if d["action"] == "BUY" else "sell"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td><b>{d['action']}</b></td>"
            f"<td>{d['ticker']}</td>"
            f"<td>{d['shares']}</td>"
            f"<td>${d['limit_price']:,.2f}</td>"
            f"<td>${d['notional']:,.2f}</td>"
            f"<td>{d['rationale']}</td>"
            f"</tr>"
        )
    return f"<table class='orders'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _holdings_table(holdings_df: pd.DataFrame) -> str:
    if holdings_df.empty:
        return ('<p style="color:#666;font-style:italic;margin:4px 0">'
                'No open positions yet \u2014 first set of orders below will '
                'create them.</p>')
    head = ("<tr><th>Ticker</th><th>Shares</th><th>Entry</th>"
            "<th>Last close</th><th>Market value</th>"
            "<th>Unrealized PnL</th><th>Stop-loss</th>"
            "<th>Days held</th><th>Decision</th></tr>")
    rows = []
    for _, r in holdings_df.iterrows():
        upnl = r["unrealized_pnl"]
        cls = "winner" if (np.isfinite(upnl) and upnl >= 0) else "loser"
        if np.isfinite(upnl):
            color = "#0b6e4f" if upnl >= 0 else "#b34030"
            sign  = "+" if upnl >= 0 else "-"
            upct  = r["unrealized_pct"]
            psign = "+" if upct >= 0 else "-"
            upnl_str = (
                f"<span style='color:{color};font-weight:600'>"
                f"{sign}${abs(upnl):,.2f} "
                f"({psign}{abs(upct)*100:.2f}%)"
                f"</span>"
            )
        else:
            upnl_str = "n/a"
        decision = r.get("decision", "HOLD")
        decision_html = (f"<span style='color:#b34030;font-weight:600'>{decision}</span>"
                          if "SELL" in decision else
                          f"<span style='color:#0b6e4f'>{decision}</span>")
        days_held = int(r.get("days_held", 0)) if np.isfinite(r.get("days_held", 0)) else 0
        time_left = int(r.get("time_stop_left", 0)) if np.isfinite(r.get("time_stop_left", 0)) else 0
        rows.append(
            f"<tr class='{cls}'>"
            f"<td><b>{r['ticker']}</b></td>"
            f"<td>{int(r['shares'])}</td>"
            f"<td>${r['entry_price']:,.2f}</td>"
            f"<td>${r['current_price']:,.2f}</td>"
            f"<td>${r['market_value']:,.2f}</td>"
            f"<td>{upnl_str}</td>"
            f"<td>${r['stop_loss_price']:,.2f}</td>"
            f"<td>{days_held} (stop in {time_left})</td>"
            f"<td>{decision_html}</td>"
            f"</tr>"
        )
    return f"<table class='holdings'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _recent_predictions_table(history: pd.DataFrame, n: int = 20) -> str:
    recent = history.dropna(subset=["actual_ret"]).copy()
    if recent.empty:
        return '<p><i>No realised predictions yet.</i></p>'
    recent["pred_pct"]   = recent["pred_xret"].map(lambda v: f"{v*100:+.2f}%")
    recent["actual_pct"] = recent["actual_xret"].map(lambda v: f"{v*100:+.2f}%")
    recent["correct"] = (((recent["side"] == "long")  & (recent["actual_xret"] > 0))
                          | ((recent["side"] == "short") & (recent["actual_xret"] < 0)))
    recent = (recent[recent["side"].isin(["long", "short"])]
              .sort_values("for_date", ascending=False).head(n))
    head = ("<tr><th>For date</th><th>Side</th><th>Ticker</th>"
            "<th>Pred</th><th>Actual</th><th>Hit?</th></tr>")
    rows = []
    for _, r in recent.iterrows():
        cls = "winner" if r["correct"] else "loser"
        check = "&#10004;" if r["correct"] else "&#10008;"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{r['for_date'].strftime('%Y-%m-%d')}</td>"
            f"<td>{r['side']}</td>"
            f"<td>{r['ticker']}</td>"
            f"<td>{r['pred_pct']}</td>"
            f"<td>{r['actual_pct']}</td>"
            f"<td>{check}</td>"
            f"</tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


# ---------- main entry point -------------------------------------------------

CSS = """
:root {
  --fg: #1a1a1a; --bg: #f6f6f3; --card: #ffffff;
  --accent: #0b6e4f; --danger: #b34030; --muted: #6a6a6a;
  --buy-bg: rgba(11,110,79,0.10); --sell-bg: rgba(179,64,48,0.10);
  --winner: rgba(11,110,79,0.10); --loser: rgba(179,64,48,0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--fg); background: var(--bg); line-height: 1.5;
}
.container { max-width: 1240px; margin: 0 auto; }
h1 { font-size: 28px; margin: 0 0 4px 0; }
h2 { font-size: 19px; margin: 0 0 14px 0; padding-bottom: 8px;
     border-bottom: 1px solid #e0e0dc; }
h3 { font-size: 15px; margin: 0 0 8px 0; color: var(--muted); font-weight: 600; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
.card {
  background: var(--card); border-radius: 10px; padding: 22px 26px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 18px;
}
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.grid-6 { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }
@media (max-width: 980px) { .grid-3 { grid-template-columns: 1fr; }
                            .grid-2 { grid-template-columns: 1fr; }
                            .grid-6 { grid-template-columns: 1fr 1fr 1fr; }
                            .grid-4 { grid-template-columns: 1fr 1fr; } }
.kpi {
  background: #fafaf6; border-radius: 8px; padding: 12px 14px;
  border: 1px solid #ececeb;
}
.kpi-label { font-size: 11px; color: var(--muted); text-transform: uppercase;
             letter-spacing: 0.06em; }
.kpi-value { font-size: 22px; font-weight: 600; line-height: 1.2;
             margin: 4px 0 2px 0; }
.kpi-sub   { font-size: 12px; color: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #ededeb; }
th     { background: #f0f0eb; font-weight: 600; }
table.orders tr.buy   { background: var(--buy-bg); }
table.orders tr.sell  { background: var(--sell-bg); }
table.holdings tr.winner, tr.winner { background: var(--winner); }
table.holdings tr.loser,  tr.loser  { background: var(--loser); }
.disclaimer { font-size: 12px; color: var(--muted); margin-top: 24px;
              padding: 12px; background: #f0f0eb; border-radius: 6px; }
/* Short on-page copy helpers */
.lead { color: #444; font-size: 14px; line-height: 1.45; margin: 0 0 12px 0; max-width: 68ch; }
.hint { color: var(--muted); font-size: 12px; line-height: 1.45; margin: 0 0 10px 0; max-width: 70ch; }
.callout { font-size: 13px; color: #333; line-height: 1.5; margin: 0 0 12px 0; padding: 10px 14px;
           border-radius: 8px; background: #fafaf6; border-left: 3px solid var(--accent); }
.callout ul { margin: 6px 0 0 18px; padding: 0; }
.callout li { margin-bottom: 6px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 12px;
       font-size: 11px; font-weight: 600; }
.tag-buy { background: var(--buy-bg); color: var(--accent); }
.tag-sell { background: var(--sell-bg); color: var(--danger); }

/* Tab navigation */
.tabs { display: flex; gap: 4px; margin: 0 0 22px 0;
         border-bottom: 2px solid #e0e0dc; }
.tab-btn { padding: 12px 22px; background: transparent; border: none;
            cursor: pointer; font-weight: 600; font-size: 14px;
            color: #888; border-bottom: 3px solid transparent;
            margin-bottom: -2px; transition: color 0.15s; font-family: inherit; }
.tab-btn:hover { color: var(--fg); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-content { display: none; }
.tab-content.active { display: block; animation: fadein 0.18s ease-in; }
@keyframes fadein { from { opacity: 0; } to { opacity: 1; } }

/* Backtest day-walker controls */
.bt-controls { display: flex; align-items: center; gap: 10px;
                margin-bottom: 16px; flex-wrap: wrap; }
.bt-controls select { font-size: 14px; padding: 7px 10px;
                       border: 1px solid #ccc; border-radius: 6px;
                       font-family: inherit; min-width: 160px; }
.bt-controls button { font-size: 14px; padding: 7px 14px;
                       border: 1px solid #ccc; background: #fff;
                       border-radius: 6px; cursor: pointer;
                       font-family: inherit; }
.bt-controls button:hover { background: #f0f0eb; }
.bt-controls button:disabled { opacity: 0.4; cursor: not-allowed; }
.bt-summary { font-size: 14px; color: var(--muted); margin-left: 8px; }
.bt-summary b { color: var(--fg); }
.bt-mode-toggle { display: inline-flex; align-items: center; gap: 0;
                   padding: 0 4px 0 10px; border-left: 1px solid #ddd;
                   margin-left: 4px; }
.bt-mode-toggle .mode-btn { padding: 6px 10px; font-size: 12px;
                             border: 1px solid #ccc; background: #fff;
                             cursor: pointer; font-family: inherit; }
.bt-mode-toggle .mode-btn:first-of-type { border-radius: 6px 0 0 6px;
                                            border-right: none; }
.bt-mode-toggle .mode-btn:last-of-type  { border-radius: 0 6px 6px 0; }
.bt-mode-toggle .mode-btn.active { background: var(--accent);
                                     color: #fff; border-color: var(--accent); }
.bt-mode-toggle .mode-btn:hover:not(.active) { background: #f0f0eb; }
"""


def build_site(latest_predictions: pd.DataFrame,
               tracker: pd.DataFrame,
               history: pd.DataFrame,
               fold_metrics: pd.DataFrame,
               feat_imp: pd.Series,
               cfg: Config,
               run_at: datetime | None = None,
               hot_scored: pd.DataFrame | None = None,
               hot_additions: list[str] | None = None,
               portfolio_history: pd.DataFrame | None = None,
               portfolio_trades: pd.DataFrame | None = None,
               portfolio_stats: dict | None = None,
               orders: list | None = None,
               holdings_df: pd.DataFrame | None = None,
               live_history: pd.DataFrame | None = None,
               daily_views: dict | None = None,
               daily_views_by_mode: dict | None = None,
               active_settle_mode: str = "margin",
               panel: pd.DataFrame | None = None,
               starting_capital: float = 5000.0) -> Path:
    """Render docs/index.html."""
    cfg.docs_dir.mkdir(parents=True, exist_ok=True)
    cfg.docs_data_dir.mkdir(parents=True, exist_ok=True)

    if run_at is None:
        run_at = datetime.now(timezone.utc)
    today = pd.Timestamp(latest_predictions["date"].max()).normalize()
    long_n, short_n = cfg.long_n, cfg.short_n

    # ---- KPI strip ----
    portfolio_stats = portfolio_stats or {}
    pf_history = portfolio_history if portfolio_history is not None else pd.DataFrame()
    live = live_history if live_history is not None else pd.DataFrame()
    final_eq = portfolio_stats.get("final_equity", starting_capital)
    total_ret = portfolio_stats.get("total_return", 0.0)
    today_pnl = portfolio_stats.get("today_pnl", 0.0)
    today_ret = portfolio_stats.get("today_ret", 0.0)
    sharpe = portfolio_stats.get("sharpe", 0.0)
    max_dd = portfolio_stats.get("max_dd", 0.0)
    win_rate = portfolio_stats.get("win_rate", 0.0)
    profit_factor = portfolio_stats.get("profit_factor", 0.0)
    days = portfolio_stats.get("days", 0)
    n_trades = portfolio_stats.get("n_trades", 0)
    trade_win_rate = portfolio_stats.get("trade_win_rate", float("nan"))
    avg_win = portfolio_stats.get("avg_win", 0.0)
    avg_loss = portfolio_stats.get("avg_loss", 0.0)

    # Live $5K KPIs
    if not live.empty:
        live_equity = float(live["equity"].iloc[-1])
        live_days = int(len(live))
        live_total_ret = live_equity / starting_capital - 1
        if live_days >= 2:
            live_daily = live["equity"].pct_change().dropna()
            live_sharpe = (live_daily.mean() * 252) / (live_daily.std() * np.sqrt(252) + 1e-12)
            live_win = float((live_daily > 0).mean())
            live_today_pnl = float(live["equity"].iloc[-1] - live["equity"].iloc[-2])
            live_today_ret = float(live_daily.iloc[-1])
        else:
            live_sharpe = 0.0; live_win = 0.0
            live_today_pnl = live_equity - starting_capital; live_today_ret = live_total_ret
    else:
        live_equity = starting_capital; live_days = 0; live_total_ret = 0.0
        live_sharpe = 0.0; live_win = 0.0; live_today_pnl = 0.0; live_today_ret = 0.0

    # Hit rate (from forward tracker)
    h = history.dropna(subset=["actual_xret"]) if not history.empty else pd.DataFrame()
    if not h.empty and "side" in h.columns:
        h = h[h["side"].isin(["long", "short"])].copy()
        if not h.empty:
            h["correct"] = ((h["side"] == "long") & (h["actual_xret"] > 0)) | \
                            ((h["side"] == "short") & (h["actual_xret"] < 0))
            hit_rate_live = float(h["correct"].mean())
            hit_count = int(h["correct"].sum()); hit_total = int(len(h))
        else:
            hit_rate_live = float("nan"); hit_count = 0; hit_total = 0
    else:
        hit_rate_live = float("nan"); hit_count = 0; hit_total = 0

    eq_color = "#0b6e4f" if live_equity >= starting_capital else "#b34030"
    pnl_color = "#0b6e4f" if live_today_pnl >= 0 else "#b34030"
    ret_color = "#0b6e4f" if live_total_ret >= 0 else "#b34030"

    kpi_strip = "".join([
        _kpi_card("Live portfolio equity", _fmt_money(live_equity),
                  f"started: {_fmt_money(starting_capital)} ({live_days} days live)",
                  color=eq_color),
        _kpi_card("Live total return", _fmt_pct(live_total_ret),
                  "from $5K simulated portfolio", color=ret_color),
        _kpi_card("Today's PnL", _fmt_pnl(live_today_pnl),
                  _fmt_pct(live_today_ret), color=pnl_color),
        _kpi_card("Live Sharpe", _fmt_num(live_sharpe, 2),
                  f"win rate (days +): {_fmt_pct(live_win, 1)}", color="#222"),
        _kpi_card("Hit rate (predictions)",
                  _fmt_pct(hit_rate_live, 1) if np.isfinite(hit_rate_live) else "n/a",
                  f"{hit_count}/{hit_total} correct directions", color="#222"),
        _kpi_card("Open positions", str(int(holdings_df.shape[0]) if holdings_df is not None else 0),
                  f"target: {cfg.long_n} longs", color="#222"),
    ])

    # Backtest stats sub-strip (smaller)
    bt_color = "#0b6e4f" if final_eq >= starting_capital else "#b34030"
    backtest_strip = "".join([
        _kpi_card("Backtest equity", _fmt_money(final_eq),
                  f"after {days} sim days", color=bt_color),
        _kpi_card("Backtest return", _fmt_pct(total_ret),
                  f"sim Sharpe: {_fmt_num(sharpe, 2)}", color=bt_color),
        _kpi_card("Backtest max DD", _fmt_pct(max_dd),
                  f"profit factor: {_fmt_num(profit_factor, 2) if np.isfinite(profit_factor) else '\u221e'}",
                  color="#b34030"),
        _kpi_card("Backtest win rate", _fmt_pct(win_rate, 1),
                  f"trades: {n_trades}", color="#222"),
    ])

    # ---- Hit rate (forward tracker) ----
    hit_fig = _build_hit_chart(history)

    # ---- Charts ----
    pf_trades = portfolio_trades if portfolio_trades is not None else pd.DataFrame()
    today_fig = _build_today_chart(latest_predictions, long_n, short_n)
    eq_fig    = _build_equity_chart(pf_history, starting_capital)
    pnl_fig   = _build_pnl_bars(pf_history)
    # Backtest can be served in BOTH settlement modes simultaneously; the
    # dashboard exposes a toggle that swaps datasets in-place. Fall back to
    # the legacy single-dict argument when callers haven't been updated yet.
    if daily_views_by_mode:
        if active_settle_mode not in daily_views_by_mode:
            active_settle_mode = next(iter(daily_views_by_mode))
        daily_views = daily_views_by_mode.get(active_settle_mode) or daily_views
    else:
        daily_views_by_mode = {active_settle_mode: daily_views or {}}

    dv_dates = sorted(daily_views.keys()) if daily_views else []
    last_view_d = dv_dates[-1] if dv_dates else None
    init_top_rows = []
    if last_view_d and daily_views and daily_views.get(last_view_d):
        init_top_rows = daily_views[last_view_d].get("top_realized") or []
    init_top_title = (
        f"Top tickers by realized P&amp;L (cumulative through {last_view_d})"
        if last_view_d else "Top tickers by realized P&amp;L"
    )
    ticker_pnl_fig = _top_realized_bar_figure(init_top_rows, init_top_title)
    live_fig  = _build_live_equity_chart(live, starting_capital)
    hot_fig   = _build_hot_chart(hot_scored if hot_scored is not None else pd.DataFrame())
    fold_fig  = _build_importance_chart(feat_imp)
    bt_eq_fig = _build_backtest_equity(pf_history, starting_capital)
    stop_loss_pct = (portfolio_stats.get("stop_loss_pct", 0.08)
                      if portfolio_stats else 0.08)
    if panel is not None and not panel.empty:
        preds_table_html = _today_predictions_table(
            latest_predictions, panel, long_n, short_n, starting_capital,
            stop_loss_pct=stop_loss_pct)
    else:
        preds_table_html = "<p><i>Universe panel unavailable.</i></p>"

    hot_added_str = (", ".join(hot_additions) if hot_additions
                      else "<i>no new hot adds today</i>")

    orders_html = _orders_table(orders or [])
    holdings_html = _holdings_table(holdings_df if holdings_df is not None else pd.DataFrame())
    recent_html = _recent_predictions_table(history, n=20)

    long_n_str = f"{long_n}"
    capital_str = _fmt_money(starting_capital)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RF Daily Stock Signal \u2014 ${final_eq:,.0f}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<div class="container">

  <h1>Daily Random Forest Stock Signal</h1>
  <div class="meta">
    Updated <b>{run_at.strftime('%Y-%m-%d %H:%M UTC')}</b> &middot;
    based on close prices through <b>{today.strftime('%Y-%m-%d')}</b> &middot;
    targeting <b>next trading day</b> &middot;
    {len(cfg.universe)} core stocks + {len(cfg.indices)} indices &middot;
    starting capital <b>{capital_str}</b> &middot;
    live settlement <b>{("T+1 cash" if active_settle_mode == "cash" else "T+0 margin")}</b>
    <span style="color:#888">· backtest toggle below · rerun with <code>SETTLE_DAYS=1</code> for T+1 live</span>
  </div>

  <div class="tabs">
    <button class="tab-btn active" data-tab="live">Live $5K dashboard</button>
    <button class="tab-btn" data-tab="backtest">Day-by-day backtesting</button>
  </div>

  <div class="tab-content active" id="tab-live">

  <div class="card">
    <h2>Live $5K simulated portfolio \u2014 dashboard</h2>
    <div class="grid-6">{kpi_strip}</div>
  </div>

  <div class="card">
    <h2>Today's orders \u2014 place at next open</h2>
    <p class="lead">
      Rotate into today&apos;s basket at the <strong>next</strong> session open.
      <strong>Limit price</strong> = same adj.-close anchor as the table below (~<strong>${starting_capital / max(cfg.long_n, 1):,.0f}</strong> per fresh BUY slot, equal-weight).
    </p>
    <p class="hint">Playbook lists chase cushion and fallbacks; bump <code>cost_bps</code> if you model commissions.</p>
    {orders_html}
  </div>

  <div class="card">
    <h2>Execution playbook \u2014 quick rules</h2>
    <ol class="hint" style="margin:0 0 0 18px;padding:0;line-height:1.6">
      <li><strong>BUYs:</strong> open market, or day-limit at anchor up to ~+0.5% chase.</li>
      <li><strong>Still open ~11 AM ET?</strong> Market to finish, <em>or</em> skip and rotate tomorrow.
        Skip if gap &gt; +1.5% above anchor.</li>
      <li><strong>SELLs:</strong> exit rotated names same day (MOC or limit ~−0.25%).</li>
      <li><strong>Stop</strong> (holdings/decision column): −8% from <em>your</em> fill \u2192 sell next open.</li>
      <li><strong>Time-stop:</strong> same ticker &gt; 10 sessions \u2192 flatten; rebuy next pass if still top.</li>
      <li><strong>Otherwise</strong> overnight holds only; rotate with the basket, not intraday guesses.</li>
    </ol>
  </div>

  <div class="card">
    <h2>Current holdings \u2014 with explicit decision rules</h2>
    {holdings_html}
  </div>

  <div class="card">
    <h2>Live equity curve \u2014 honest forward record</h2>
    {_fig_div(live_fig, "live_chart")}
  </div>

  <div class="card">
    <h2>Hit rate \u2014 are predictions directionally correct?</h2>
    <p class="hint">
      Long hit = next-day excess &gt; 0; short hit = excess &lt; 0 (~coin flip \u2248 50%). Prefer the trend line over one day.
    </p>
    {_fig_div(hit_fig, "hit_chart")}
  </div>

  <div class="card">
    <h2>Tomorrow's predictions \u2014 basket &amp; prices</h2>
    <p class="lead">
      Ranked RF on {len(cfg.universe)} names (daily cross-section features \u2192 short-horizon excess vs EW peers; walk-forward before production).
      As of <b>{today.strftime('%Y-%m-%d')}</b>: top-<b>{long_n_str}</b> BUY (live is long-only), bottom-<b>{short_n}</b> AVOID,
      equal-weight on <b>{capital_str}</b> using closes below.
    </p>
    <div class="callout">
      <strong>Columns</strong>
      <ul>
        <li><strong>Close</strong> — adj. last session from this run (not streaming); sizing + limit-buy anchor.</li>
        <li><strong>Lim. sell</strong> — optional <span style="white-space:nowrap">close \u00d7 (1 + pred)</span>; simulator ignores it.</li>
        <li><strong>Stop</strong> — reference −{int(round(stop_loss_pct * 100))}% vs anchor; <strong>Current holdings</strong> uses your fill.</li>
        <li><strong>Exit</strong> — ~one session; rotate at next close if neither limit nor stop hits.</li>
        <li><strong>Today&apos;s orders</strong> — same anchor in Limit price.</li>
      </ul>
    </div>
    {preds_table_html}
    <div style="margin-top:14px">{_fig_div(today_fig, "today_chart")}</div>
  </div>

  <div class="card">
    <h2>Hot movers \u2014 dynamic universe additions</h2>
    <p class="lead"><b>Added today:</b> {hot_added_str}.</p>
    <p class="hint">Watchlist movers (volume/momentum/range) can join the ranked universe mid-run.</p>
    {_fig_div(hot_fig, "hot_chart")}
  </div>

  <div class="card">
    <h2>Recent realised predictions (last 20)</h2>
    {recent_html}
  </div>

  <div class="card">
    <h2>Model diagnostics</h2>
    {_fig_div(fold_fig, "imp_chart")}
  </div>

  </div> <!-- /tab-live -->

  <div class="tab-content" id="tab-backtest">

  <div class="card">
    <h2>Backtesting \u2014 walk through any day</h2>
    <p class="lead">Pick a session: equity cursor, holdings, prior basket vs realized close, that day&apos;s basket, and fills (+ PnL to next close). Step with \u25C0/\u25B6 or Play.</p>

    <div class="bt-controls">
      <button id="bt-play" title="Auto-advance through every trading day">\u25B6 Play</button>
      <button id="bt-prev" title="Previous trading day">\u25C0</button>
      <select id="bt-day-select" aria-label="Pick date"></select>
      <button id="bt-next" title="Next trading day">\u25B6</button>
      <select id="bt-speed" aria-label="Playback speed" title="Playback speed">
        <option value="200">0.5\u00d7</option>
        <option value="100" selected>1\u00d7</option>
        <option value="50">2\u00d7</option>
        <option value="20">5\u00d7</option>
        <option value="5">20\u00d7</option>
      </select>
      <button id="bt-jump-best" title="Jump to single best PnL day">Best day</button>
      <button id="bt-jump-worst" title="Jump to single worst PnL day">Worst day</button>
      <span class="bt-mode-toggle" role="group"
            aria-label="Cash-settlement mode used for the backtest replay"
            title="Switch between margin/no-settlement (T+0) and US Reg-T cash account (T+1). Re-runs the visual replay client-side; both backtests are precomputed.">
        <span style="font-size:12px;color:#666;margin-right:6px">Settlement:</span>
        <button id="bt-mode-margin" type="button" class="mode-btn active"
                data-mode="margin">T+0 Margin</button>
        <button id="bt-mode-cash"   type="button" class="mode-btn"
                data-mode="cash">T+1 Cash</button>
      </span>
      <span id="bt-summary" class="bt-summary"></span>
    </div>
    <p id="bt-mode-help" class="hint" style="margin:6px 0 14px 0">
      <strong>T+0 Margin</strong> — sell proceeds recycle same day.
      <strong>T+1 Cash</strong> — proceeds spendable next session (Reg&nbsp;T). Toggle = precomputed replay; live KPIs follow <code>SETTLE_DAYS</code> from the last run.
    </p>

    {_fig_div(bt_eq_fig, "bt_chart", height=380)}

    <div style="margin-top:18px">
      <h3 style="margin-bottom:8px">
        Holdings at end of <span id="bt-rec-date" style="color:var(--accent)">\u2014</span>
        <span id="bt-hold-count" style="font-size:13px;color:#666;font-weight:400"></span>
      </h3>
      <p class="hint" style="margin:0 0 8px 0">Positions rolling into the next session vs that close (avg cost).</p>
      <table id="bt-hold-table"><thead><tr>
        <th>Ticker</th><th>Shares</th><th>Avg entry</th><th>Last close</th>
        <th>Market value</th><th>Unrealized PnL</th>
      </tr></thead><tbody></tbody></table>
    </div>

    <div style="margin-top:22px">
      <h3 style="margin-bottom:8px">
        Recommendation made on the <b>previous</b> session
        (<span id="bt-prev-date" style="color:var(--accent)">\u2014</span>)
        \u2014 what those picks did by <span id="bt-prev-target-date" style="color:var(--accent)">\u2014</span>
      </h3>
      <p class="hint" style="margin:0 0 8px 0">Prior session&apos;s basket through the selected day&apos;s close. Empty on the first replay date.</p>
      <div style="overflow-x:auto"><table id="bt-prev-recs-table"><thead><tr>
        <th>Action</th><th>#</th><th>Ticker</th><th>Pred xret</th>
        <th>Prev close</th><th>Realized close</th>
        <th>Realized 1d %</th><th>$ on 1 share</th>
      </tr></thead><tbody></tbody></table></div>
    </div>

    <div style="margin-top:22px">
      <h3 style="margin-bottom:8px">
        Recommendation made on <span id="bt-rec-date2" style="color:var(--accent)">\u2014</span>
      </h3>
      <p class="hint" style="margin:0 0 8px 0;line-height:1.5">
        Simulator basket at that close.
        <span style="color:#0b6e4f;font-weight:600">BUY</span> ·
        <span style="color:#0b6e4f;font-weight:600">ADD</span>/<span style="color:#b38030;font-weight:600">TRIM</span> rebalance toward <code>1/{long_n_str}</code> equity ·
        <span style="color:#666;font-weight:600">HOLD</span> sized ·
        <span style="color:#b34030;font-weight:600">SELL</span> rotates out (next close).
      </p>
      <p class="hint" style="margin:0 0 8px 0;line-height:1.45;font-style:italic">
        ADD/TRIM comes from <code>floor(equity/{long_n_str}/price)</code> rounding and drift.
        Overnight rerank means HOLD today can read SELL tomorrow.
      </p>
      <div style="overflow-x:auto"><table id="bt-recs-table"><thead><tr>
        <th>Action</th><th>#</th><th>Ticker</th><th>Pred xret</th>
        <th>Price</th><th>Shares</th><th>Limit sell</th><th>Stop-loss</th><th>Exit</th>
      </tr></thead><tbody></tbody></table></div>
    </div>

    <div style="margin-top:22px">
      <h3 style="margin-bottom:8px">
        Trades executed on <span id="bt-fills-trade-date" style="color:var(--accent)">\u2014</span>
        &middot; 1-day PnL through <span id="bt-fills-next-date" style="color:var(--accent)">\u2014</span>
      </h3>
      <p class="hint" style="margin:0 0 8px 0">Close fills; OHLC for context; <strong>Next close</strong> drives 1d % / $PnL.
        Sell PnL = edge from exiting vs riding the bar.</p>
      <div style="overflow-x:auto"><table id="bt-fills-table"><thead><tr>
        <th>Action</th><th>Ticker</th><th>Sh</th><th>Fill</th>
        <th>Open</th><th>High</th><th>Low</th><th>Close</th>
        <th>Next close</th><th>1d %</th><th>$ PnL</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </div>

  <div class="card">
    <h2>Top 10 tickers \u2014 cumulative realized P&amp;L</h2>
    <p class="hint">
      Top ten by cumulative realized P&amp;L through the selected replay date (closed lots only).
    </p>
    {_fig_div(ticker_pnl_fig, "ticker_pnl_chart")}
  </div>

  <div class="card">
    <h2>Backtest replay summary</h2>
    <p class="hint">Long-run replay from walk-forward predictions (from 2019).</p>
    <div class="grid-4">{backtest_strip}</div>
    {_fig_div(eq_fig, "eq_chart")}
    {_fig_div(pnl_fig, "pnl_chart")}
  </div>

  </div> <!-- /tab-backtest -->

  __DV_INJECT__

  <div class="disclaimer">
    <b>Educational only \u2014 not investment advice.</b>
  </div>

</div>
</body>
</html>
"""
    import json as _json
    dvbm_payload = _json.dumps(daily_views_by_mode or {}, separators=(",", ":"))
    dv_payload = _json.dumps(daily_views or {}, separators=(",", ":"))
    inject = (f'<script>'
              f'window.__DAILY_VIEWS_BY_MODE__ = {dvbm_payload};'
              f'window.__DAILY_VIEWS__ = {dv_payload};'
              f'window.__ACTIVE_SETTLE_MODE__ = {_json.dumps(active_settle_mode)};'
              f'window.__STARTING_CAPITAL__ = {starting_capital};'
              f'</script>'
              + _BACKTEST_JS)
    html = html.replace("__DV_INJECT__", inject)

    out = cfg.docs_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


_BACKTEST_JS = r"""
<script>
(function() {
  const DV_BY_MODE = window.__DAILY_VIEWS_BY_MODE__ || {};
  const FALLBACK_DV = window.__DAILY_VIEWS__ || {};
  const MODES = Object.keys(DV_BY_MODE);
  const ACTIVE = window.__ACTIVE_SETTLE_MODE__ || (MODES[0] || 'margin');
  let mode = MODES.includes(ACTIVE) ? ACTIVE
            : (MODES.includes('margin') ? 'margin'
            : (MODES[0] || 'margin'));
  let DV = (DV_BY_MODE[mode]) || FALLBACK_DV;
  let DATES = Object.keys(DV).sort();
  if (!DATES.length) return;
  const STARTING = window.__STARTING_CAPITAL__ || 5000;
  let cur = DATES[DATES.length - 1];
  let playTimer = null;
  /** When true (only during playback), equity chart uses a sliding x/y zoom; otherwise full autorange. */
  let playbackZoom = false;

  const $ = (id) => document.getElementById(id);
  const sel = $('bt-day-select');
  function populateDateOptions() {
    sel.innerHTML = '';
    DATES.forEach(d => {
      const o = document.createElement('option');
      o.value = d; o.textContent = d; sel.appendChild(o);
    });
  }
  populateDateOptions();

  function fmtMoney(x) {
    if (x === null || x === undefined || isNaN(x)) return 'n/a';
    const s = x < 0 ? '-' : '';
    return s + '$' + Math.abs(x).toLocaleString(undefined,
      {minimumFractionDigits: 2, maximumFractionDigits: 2});
  }
  function fmtPct(x, d) {
    if (x === null || x === undefined || isNaN(x)) return 'n/a';
    const sign = x >= 0 ? '+' : '';
    return sign + (x * 100).toFixed(d || 2) + '%';
  }
  // Signed and color-coded $ PnL ("+$1,234.56" green / "-$789.01" red).
  function fmtPnl(x, bold) {
    if (x === null || x === undefined || isNaN(x)) return 'n/a';
    const color = x >= 0 ? '#0b6e4f' : '#b34030';
    const sign  = x >= 0 ? '+' : '-';
    const num   = Math.abs(x).toLocaleString(undefined,
      {minimumFractionDigits: 2, maximumFractionDigits: 2});
    const w = bold === false ? '' : 'font-weight:600;';
    return `<span style="color:${color};${w}">${sign}$${num}</span>`;
  }
  // Signed and color-coded percentage ("+1.23%" green / "-0.45%" red).
  function fmtPctColored(x, d) {
    if (x === null || x === undefined || isNaN(x)) return 'n/a';
    const color = x >= 0 ? '#0b6e4f' : '#b34030';
    const sign  = x >= 0 ? '+' : '-';
    return `<span style="color:${color};font-weight:600">${sign}${(Math.abs(x)*100).toFixed(d || 2)}%</span>`;
  }

  function renderRecs(recs) {
    const tb = $('bt-recs-table').querySelector('tbody');
    if (!recs || !recs.length) {
      tb.innerHTML = '<tr><td colspan=9 style="color:#888;font-style:italic">No predictions for this day.</td></tr>';
      return;
    }
    tb.innerHTML = recs.map(r => {
      const action = r.action || 'BUY';
      const isSell = action === 'SELL';
      const isHold = action === 'HOLD';
      const isAdd  = action === 'ADD';
      const isTrim = action === 'TRIM';
      const rowCls = isSell ? 'loser' : 'winner';
      let actBadge;
      if (action === 'BUY')   actBadge = '<span class="tag tag-buy">BUY</span>';
      else if (isAdd)       actBadge = '<span class="tag tag-buy">ADD</span>';
      else if (isTrim)      actBadge = '<span style="background:#fde8cf;color:#8a4a00;font-size:11px;font-weight:600;padding:2px 6px;border-radius:4px">TRIM</span>';
      else if (isHold)      actBadge = '<span style="color:#666;font-weight:600">HOLD</span>';
      else                  actBadge = '<span class="tag tag-sell">SELL</span>';

      const rankCell = r.rank == null ? '\u2014' : '#' + r.rank;
      const predCell = r.pred_xret == null ? '<span style="color:#888">rotated out</span>'
                                            : fmtPct(r.pred_xret, 2);
      const priceCell = r.close == null ? 'n/a' : '$' + r.close.toFixed(2);

      const ts = r.trade_shares;
      const pos = r.shares;
      let sharesCell;
      if (isSell) {
        const n = ts != null ? ts : (r.shares || 0);
        sharesCell = n ? `<span style="color:#888;font-size:11px">sell </span><b>${n}</b> <span style="color:#888;font-size:11px">(\u2192 0)</span>` : '\u2014';
      } else if (action === 'BUY') {
        const n = ts != null ? ts : pos;
        sharesCell = n ? `<span style="color:#888;font-size:11px">buy </span><b>${n}</b> <span style="color:#888;font-size:11px">(${(r.dollars != null ? Math.round(r.dollars) : '')})</span>` : '\u2014';
      } else if (isAdd) {
        const n = ts != null ? ts : 0;
        sharesCell =
          `<span style="color:#0b6e4f;font-size:11px">+${n}</span> `
          + `<span style="color:#888;font-size:11px">\u2192</span> <b>${pos}</b> total`;
      } else if (isTrim) {
        const n = ts != null ? ts : 0;
        sharesCell =
          `<span style="color:#b34030;font-size:11px">\u2212${n}</span> `
          + `<span style="color:#888;font-size:11px">\u2192</span> <b>${pos}</b> total`;
      } else if (isHold) {
        sharesCell = pos ?
          `<span style="color:#888;font-size:11px">no trade \u2014 </span><b>${pos}</b> sh` : '\u2014';
      } else {
        sharesCell = '\u2014';
      }

      const limitCell = isSell
        ? '<span style="color:#888">\u2014</span>'
        : (r.limit_sell == null ? 'n/a'
            : '<span style="color:#0b6e4f">$' + r.limit_sell.toFixed(2) + '</span>');
      const stopCell  = isSell
        ? '<span style="color:#888">\u2014</span>'
        : (r.stop_loss == null ? 'n/a'
            : '<span style="color:#b34030">$' + r.stop_loss.toFixed(2) + '</span>');
      const exitCell = isSell
        ? '<b style="color:#b34030">SELL at next close</b>'
        : '<span style="color:#666">next session close</span>';
      return `<tr class="${rowCls}">
        <td>${actBadge}</td>
        <td>${rankCell}</td>
        <td><b>${r.ticker}</b></td>
        <td>${predCell}</td>
        <td>${priceCell}</td>
        <td>${sharesCell}</td>
        <td>${limitCell}</td>
        <td>${stopCell}</td>
        <td>${exitCell}</td>
      </tr>`;
    }).join('');
  }

  function renderHoldings(holds) {
    const tb = $('bt-hold-table').querySelector('tbody');
    const cnt = $('bt-hold-count');
    if (!holds || !holds.length) {
      cnt.textContent = '';
      tb.innerHTML = '<tr><td colspan=6 style="color:#888;font-style:italic">No open positions \u2014 portfolio is fully in cash.</td></tr>';
      return;
    }
    let totalMv = 0, totalPnl = 0;
    holds.forEach(h => { totalMv += h.mv || 0; totalPnl += h.upnl || 0; });
    cnt.innerHTML = ` \u00b7 ${holds.length} position${holds.length===1?'':'s'} \u00b7 total MV ${fmtMoney(totalMv)} \u00b7 unrealized PnL ${fmtPnl(totalPnl)}`;
    tb.innerHTML = holds.map(h => {
      const upnl = h.upnl;
      const cls = upnl == null ? '' : (upnl >= 0 ? 'winner' : 'loser');
      const upnlStr = upnl == null ? 'n/a'
        : `${fmtPnl(upnl)} (${fmtPctColored(h.upct, 2)})`;
      return `<tr class="${cls}">
        <td><b>${h.ticker}</b></td>
        <td>${h.shares}</td>
        <td>$${h.entry.toFixed(2)}</td>
        <td>${h.current == null ? 'n/a' : '$' + h.current.toFixed(2)}</td>
        <td>${h.mv == null ? 'n/a' : fmtMoney(h.mv)}</td>
        <td>${upnlStr}</td>
      </tr>`;
    }).join('');
  }

  // Yesterday's basket -> what actually happened by selected day's close.
  function renderPrevRecs(prevDate, todayDate) {
    const tb = $('bt-prev-recs-table').querySelector('tbody');
    const lblPrev   = $('bt-prev-date');
    const lblTarget = $('bt-prev-target-date');
    if (lblPrev)   lblPrev.textContent = prevDate || '\u2014';
    if (lblTarget) lblTarget.textContent = todayDate || '\u2014';
    if (!prevDate || !DV[prevDate]) {
      tb.innerHTML = '<tr><td colspan=8 style="color:#888;font-style:italic">No previous session in the replay window \u2014 step forward to see yesterday\'s recs.</td></tr>';
      return;
    }
    const prevRecs = (DV[prevDate].recs || [])
      .filter(r => r.action !== 'SELL');
    if (!prevRecs.length) {
      tb.innerHTML = '<tr><td colspan=8 style="color:#888;font-style:italic">No basket recommendations on the previous session.</td></tr>';
      return;
    }
    // Build a ticker -> close lookup for the SELECTED day (not prev), so we
    // can show the realized 1-day move. recs/holdings/fills all carry close,
    // so combine them for the broadest coverage.
    const today = DV[todayDate] || {};
    const closeByTk = {};
    (today.recs || []).forEach(r => {
      const px = r.close != null ? r.close : (r.raw_close != null ? r.raw_close : null);
      if (px != null) closeByTk[r.ticker] = px;
    });
    (today.holdings || []).forEach(h => {
      if (h.current != null && closeByTk[h.ticker] == null)
        closeByTk[h.ticker] = h.current;
    });
    (today.fills || []).forEach(f => {
      if (f.today_close != null && closeByTk[f.ticker] == null)
        closeByTk[f.ticker] = f.today_close;
    });
    tb.innerHTML = prevRecs.map(r => {
      const action = r.action || 'BUY';
      const isAdd  = action === 'ADD';
      const isTrim = action === 'TRIM';
      const isHold = action === 'HOLD';
      let actBadge;
      if (action === 'BUY')      actBadge = '<span class="tag tag-buy">BUY</span>';
      else if (isAdd)          actBadge = '<span class="tag tag-buy">ADD</span>';
      else if (isTrim)         actBadge = '<span style="background:#fde8cf;color:#8a4a00;font-size:11px;font-weight:600;padding:2px 6px;border-radius:4px">TRIM</span>';
      else if (isHold)         actBadge = '<span style="color:#666;font-weight:600">HOLD</span>';
      else                     actBadge = `<span class="tag">${action}</span>`;
      const rankCell = r.rank == null ? '\u2014' : '#' + r.rank;
      const predCell = r.pred_xret == null ? '<span style="color:#888">\u2014</span>'
                                            : fmtPct(r.pred_xret, 2);
      const prevPx = r.close != null ? r.close
                    : (r.raw_close != null ? r.raw_close : null);
      const realPx = closeByTk[r.ticker] != null ? closeByTk[r.ticker] : null;
      const ret    = (prevPx != null && realPx != null && prevPx > 0)
                       ? (realPx / prevPx - 1) : null;
      const pnlPerSh = (prevPx != null && realPx != null) ? (realPx - prevPx) : null;
      const cls = ret == null ? '' : (ret >= 0 ? 'winner' : 'loser');
      return `<tr class="${cls}">
        <td>${actBadge}</td>
        <td>${rankCell}</td>
        <td><b>${r.ticker}</b></td>
        <td>${predCell}</td>
        <td>${prevPx == null ? 'n/a' : '$' + prevPx.toFixed(2)}</td>
        <td>${realPx == null ? '<span style="color:#888">n/a</span>' : '$' + realPx.toFixed(2)}</td>
        <td>${ret == null ? 'n/a' : fmtPctColored(ret, 2)}</td>
        <td>${pnlPerSh == null ? 'n/a' : fmtPnl(pnlPerSh)}</td>
      </tr>`;
    }).join('');
  }

  function renderFills(fills) {
    const tb = $('bt-fills-table').querySelector('tbody');
    if (!fills || !fills.length) {
      tb.innerHTML = '<tr><td colspan=11 style="color:#888;font-style:italic">No trades that day \u2014 portfolio held steady (basket unchanged).</td></tr>';
      return;
    }
    const px = (v) => v == null ? '<span style="color:#888">n/a</span>' : '$' + v.toFixed(2);
    tb.innerHTML = fills.map(f => {
      const pnl = f.next_pnl;
      const cls = pnl == null ? '' : (pnl >= 0 ? 'winner' : 'loser');
      return `<tr class="${cls}">
        <td><b style="color:${f.action==='BUY'?'#0b6e4f':'#b34030'}">${f.action}</b></td>
        <td>${f.ticker}</td>
        <td>${f.shares}</td>
        <td>$${f.fill_price.toFixed(2)}</td>
        <td>${px(f.today_open)}</td>
        <td>${px(f.today_high)}</td>
        <td>${px(f.today_low)}</td>
        <td>${px(f.today_close)}</td>
        <td>${px(f.next_close)}</td>
        <td>${fmtPctColored(f.next_pct, 2)}</td>
        <td>${fmtPnl(pnl)}</td>
      </tr>`;
    }).join('');
  }

  // Equity-chart panning window (trading days). The cursor sits ~3/4 across
  // the visible window, leaving room for the future to scroll in during Play.
  const WINDOW_TOTAL = 220;
  const WINDOW_PAST_FRAC = 0.78;

  let baseShapes = null;
  function updateTopRealizedChart(d) {
    const el = $('ticker_pnl_chart');
    if (!el || !window.Plotly) return;
    const v = DV[d];
    const rows = (v && v.top_realized) ? v.top_realized : [];
    const title = 'Top tickers by realized P/L (cumulative through ' + d + ')';
    if (!rows.length) {
      Plotly.react('ticker_pnl_chart', [{
        type: 'scatter', x: [0], y: [0], mode: 'markers',
        marker: {size: 0, opacity: 0}, showlegend: false, hoverinfo: 'skip',
      }], {
        title: {text: title, font: {size: 14}},
        template: 'plotly_white', height: 260,
        annotations: [{
          text: 'No realized sells yet on this date \u2014 step forward on the calendar.',
          xref: 'paper', yref: 'paper', x: 0.5, y: 0.5, showarrow: false,
          font: {size: 14, color: '#666'}
        }],
        xaxis: {visible: false, zeroline: false},
        yaxis: {visible: false, zeroline: false},
        margin: {l: 20, r: 20, t: 50, b: 20}
      }, {displaylogo: false});
      return;
    }
    const vals = rows.map(r => r.realized);
    const tickers = rows.map(r => r.ticker);
    const idx = vals.map((_, i) => i).sort((a, b) => vals[a] - vals[b]);
    const x = idx.map(i => vals[i]);
    const y = idx.map(i => tickers[i]);
    const colors = x.map(val => (val >= 0 ? '#0b6e4f' : '#b34030'));
    const texts = x.map(val => {
      const sign = val >= 0 ? '+' : '-';
      return sign + '$' + Math.abs(val).toLocaleString(undefined,
        {minimumFractionDigits: 0, maximumFractionDigits: 0});
    });
    Plotly.react('ticker_pnl_chart', [{
      type: 'bar', orientation: 'h', x: x, y: y, marker: {color: colors},
      text: texts, textposition: 'outside', cliponaxis: false,
      hovertemplate: '<b>%{y}</b><br>realized: $%{x:,.2f}<extra></extra>',
    }], {
      title: {text: title, font: {size: 14}},
      template: 'plotly_white',
      height: Math.max(320, 42 * rows.length + 120),
      margin: {l: 72, r: 80, t: 60, b: 40},
      xaxis: {title: {text: 'Realized P/L ($) \u2014 sum on all sells (avg-cost basis)'}},
      showlegend: false,
    }, {displaylogo: false});
  }

  function relayoutChart(dStr, equity) {
    const chart = $('bt_chart');
    if (!chart) return;
    if (baseShapes === null) {
      baseShapes = (chart.layout.shapes || []).slice();   // capture original (start-line)
    }
    const cursor = {
      type: 'line', x0: dStr, x1: dStr,
      yref: 'paper', y0: 0, y1: 1,
      line: {color: 'rgba(11,110,79,0.7)', dash: 'dot', width: 2}
    };

    const i = DATES.indexOf(dStr);
    const layoutPatch = {
      shapes: baseShapes.concat([cursor]),
      'title.text': `Backtest equity \u2014 ${dStr} \u00b7 ${fmtMoney(equity)}`
    };

    if (playbackZoom && i >= 0 && DATES.length > 0) {
      // Sliding window during Play only: pans with the cursor and rescales Y
      // to local extrema.
      const total  = Math.min(WINDOW_TOTAL, DATES.length);
      const past   = Math.floor(total * WINDOW_PAST_FRAC);
      const future = total - past;
      let lo = i - past;
      let hi = i + future + 1;
      if (lo < 0) { hi -= lo; lo = 0; }
      if (hi > DATES.length) { lo -= (hi - DATES.length); hi = DATES.length; }
      lo = Math.max(0, lo);
      let ymin = Infinity, ymax = -Infinity;
      for (let j = lo; j < hi; j++) {
        const eq = DV[DATES[j]] && DV[DATES[j]].equity;
        if (eq != null && isFinite(eq)) {
          if (eq < ymin) ymin = eq;
          if (eq > ymax) ymax = eq;
        }
      }
      if (isFinite(ymin) && isFinite(ymax)) {
        if (STARTING < ymin && (ymin - STARTING) < (ymax - ymin) * 0.25) ymin = STARTING;
        if (STARTING > ymax && (STARTING - ymax) < (ymax - ymin) * 0.25) ymax = STARTING;
        const pad = Math.max((ymax - ymin) * 0.10, 1);
        layoutPatch['xaxis.range'] = [DATES[lo], DATES[hi - 1]];
        layoutPatch['yaxis.range'] = [ymin - pad, ymax + pad];
        layoutPatch['xaxis.autorange'] = false;
        layoutPatch['yaxis.autorange'] = false;
      }
    } else {
      // Default: full history visible (opening the tab / browsing / Pause).
      layoutPatch['xaxis.autorange'] = true;
      layoutPatch['yaxis.autorange'] = true;
    }
    Plotly.relayout(chart, layoutPatch);
    Plotly.restyle(chart, {x: [[dStr]], y: [[equity]]}, [1]);
  }

  function selectDate(d) {
    if (!DV[d]) return;
    cur = d; sel.value = d;
    const v = DV[d];
    $('bt-rec-date').textContent = d;
    $('bt-rec-date2').textContent = d;
    const totalRet = v.equity / STARTING - 1;
    let unsettledStr = '';
    if (v.unsettled && v.unsettled > 0.005) {
      unsettledStr =
        ` &middot; <span title="Sale proceeds awaiting T+1 settlement \u2014 part of NAV but not yet usable for new BUYs.">unsettled <b>${fmtMoney(v.unsettled)}</b></span>`;
    }
    let skippedStr = '';
    if (v.skipped_unsettled && v.skipped_unsettled > 0.5) {
      skippedStr =
        ` &middot; <span style="color:#b38030" title="Notional that the basket wanted to BUY today but couldn't fund because cash hadn\u2019t settled yet. Will retry next session.">cash-blocked ${fmtMoney(v.skipped_unsettled)}</span>`;
    }
    $('bt-summary').innerHTML =
      `equity <b>${fmtMoney(v.equity)}</b> &middot; ` +
      `cash <b>${fmtMoney(v.cash)}</b>${unsettledStr} &middot; ` +
      `total ${fmtPctColored(totalRet, 1)} &middot; ` +
      `day PnL ${fmtPnl(v.daily_pnl)} ` +
      `(${fmtPctColored(v.daily_ret, 2)})` +
      skippedStr;
    renderHoldings(v.holdings);
    renderRecs(v.recs);
    renderFills(v.fills);
    const idx = DATES.indexOf(d);
    const prevD = idx > 0 ? DATES[idx - 1] : null;
    const nextD = idx >= 0 && idx < DATES.length - 1 ? DATES[idx + 1] : null;
    const fillsTradeLbl = $('bt-fills-trade-date');
    const fillsNextLbl  = $('bt-fills-next-date');
    if (fillsTradeLbl) fillsTradeLbl.textContent = d;
    if (fillsNextLbl)  fillsNextLbl.textContent  = nextD || '\u2014';
    renderPrevRecs(prevD, d);
    relayoutChart(d, v.equity);
    updateTopRealizedChart(d);
    $('bt-prev').disabled = (idx <= 0);
    $('bt-next').disabled = (idx >= DATES.length - 1);
  }

  function setMode(newMode) {
    if (!DV_BY_MODE[newMode]) return;
    if (newMode === mode) return;
    mode = newMode;
    DV = DV_BY_MODE[mode];
    DATES = Object.keys(DV).sort();
    if (!DATES.length) return;
    populateDateOptions();
    // Keep the same date if available, else jump to the latest.
    if (!DATES.includes(cur)) cur = DATES[DATES.length - 1];
    sel.value = cur;
    // Highlight the active toggle button.
    document.querySelectorAll('.bt-mode-toggle .mode-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
    // baseShapes anchors the *initial* equity-chart shapes; reset so the
    // cursor is recalculated fresh against the current data.
    baseShapes = null;
    selectDate(cur);
  }

  function stopPlay() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    playbackZoom = false;
    $('bt-play').textContent = '\u25B6 Play';
    // Restore full-axis view at the paused date.
    if (DV[cur]) relayoutChart(cur, DV[cur].equity);
  }
  function startPlay() {
    const speed = parseInt($('bt-speed').value, 10) || 100;
    if (DATES.indexOf(cur) >= DATES.length - 1) selectDate(DATES[0]);
    playbackZoom = true;
    playTimer = setInterval(() => {
      const i = DATES.indexOf(cur);
      if (i >= DATES.length - 1) { stopPlay(); return; }
      selectDate(DATES[i + 1]);
    }, speed);
    $('bt-play').textContent = '\u2759\u2759 Pause';
    // Immediately apply zoom on the current day (don't wait first tick).
    if (DV[cur]) relayoutChart(cur, DV[cur].equity);
  }

  sel.addEventListener('change', e => { stopPlay(); selectDate(e.target.value); });
  $('bt-prev').addEventListener('click', () => {
    stopPlay();
    const i = DATES.indexOf(cur);
    if (i > 0) selectDate(DATES[i - 1]);
  });
  $('bt-next').addEventListener('click', () => {
    stopPlay();
    const i = DATES.indexOf(cur);
    if (i < DATES.length - 1) selectDate(DATES[i + 1]);
  });
  $('bt-play').addEventListener('click', () => {
    if (playTimer) stopPlay(); else startPlay();
  });
  $('bt-speed').addEventListener('change', () => {
    if (playTimer) { stopPlay(); startPlay(); }
  });
  // Settlement mode toggle (T+0 vs T+1) -- swaps the precomputed dataset.
  document.querySelectorAll('.bt-mode-toggle .mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
    btn.addEventListener('click', () => {
      stopPlay();
      setMode(btn.dataset.mode);
    });
  });

  $('bt-jump-best').addEventListener('click', () => {
    stopPlay();
    let best = DATES[0], bestV = -Infinity;
    DATES.forEach(d => { if (DV[d].daily_pnl > bestV) { best = d; bestV = DV[d].daily_pnl; } });
    selectDate(best);
  });
  $('bt-jump-worst').addEventListener('click', () => {
    stopPlay();
    let worst = DATES[0], worstV = Infinity;
    DATES.forEach(d => { if (DV[d].daily_pnl < worstV) { worst = d; worstV = DV[d].daily_pnl; } });
    selectDate(worst);
  });

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
      // Plotly charts inside hidden tabs need a resize hint when revealed.
      if (btn.dataset.tab === 'backtest') {
        setTimeout(() => {
          ['bt_chart', 'ticker_pnl_chart', 'eq_chart', 'pnl_chart'].forEach(id => {
            const c = document.getElementById(id);
            if (c) Plotly.Plots.resize(c);
          });
        }, 50);
      }
    });
  });

  // Click chart to jump to date.
  const btChart = $('bt_chart');
  if (btChart) {
    btChart.on('plotly_click', function(data) {
      stopPlay();
      if (!data || !data.points || !data.points.length) return;
      const x = data.points[0].x;
      const d = (typeof x === 'string') ? x.slice(0, 10)
                  : new Date(x).toISOString().slice(0, 10);
      // Snap to nearest available date if exact not present.
      if (DV[d]) { selectDate(d); return; }
      const target = new Date(d).getTime();
      let best = DATES[0], bestDiff = Math.abs(new Date(DATES[0]).getTime() - target);
      for (const dd of DATES) {
        const diff = Math.abs(new Date(dd).getTime() - target);
        if (diff < bestDiff) { best = dd; bestDiff = diff; }
      }
      selectDate(best);
    });
  }

  selectDate(cur);
})();
</script>
"""
