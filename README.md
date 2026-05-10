# RF daily stock signals — panel model + static dashboard

Train a **Random Forest** on a **daily cross-section** of US stocks, score every name for **next-day excess return**, and ship a **self-contained report** under `docs/` (works with **GitHub Pages**).

**Educational only — not investment advice.**

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python scripts/run_daily.py
```

Then open **`docs/index.html`** in a browser (double-click on Windows, or `start docs\index.html`).

**First run** downloads years of daily prices (cached under `data_cache/`, gitignored). Later runs are faster.

---

## What you get

| Output | Purpose |
|--------|---------|
| **`docs/index.html`** | Live dashboard + **Day-by-day backtesting** tab (equity, tables, charts). |
| **`docs/data/predictions.csv`** | Latest scores per ticker. |
| **`docs/data/orders_today.csv`** | Suggested **BUY/SELL** rows for the **next** session (see below). |
| **`docs/data/holdings.csv`** | Simulated positions + risk hints for the live tracker. |
| **`docs/data/portfolio_history.csv`** | Backtest equity path (from walk-forward predictions). |
| **`docs/data/portfolio_trades.csv`** | Every simulated fill in the backtest. |
| **`docs/data/daily_views.json`** | Precomputed per-day replay (margin + T+1 cash). |
| **`docs/data/tracker.csv`** | Prediction vs realised returns as dates mature. |

Enable **GitHub Pages** with source = **`/docs`** on the `main` branch. Your site URL will be:

`https://<your-username>.github.io/<your-repo-name>/`

The included **Actions** workflow runs `python scripts/run_daily.py` on a schedule and can commit updated `docs/` for you.

---

## How the “trading” rules work (simulator + orders)

The code assumes: **decision at today’s close** using features and prices through that close; the **simulator fills at that same close** (no look-ahead into the next bar’s return for the decision). The **HTML playbook** describes how *you* might execute at the **next open** — that part is human workflow, not additional simulation logic.

### Target portfolio (backtest + order math)

- **Long-only** by default: **`long_n = 5`** names get **equal weight** (~20% of equity each when fully invested).  
- Each day the model ranks the universe by predicted **next-day excess return**; the **top 5** are the target basket.  
- **Target shares** ≈ `floor(equity × weight / price)` using that day’s **adjusted close**.

### Buy, sell, hold, rebalance

| Idea | What happens |
|------|----------------|
| **New name enters top 5** | Simulator **BUYs** shares to reach the target weight. |
| **Name drops out of top 5** | Simulator **SELLs** the position (full exit). |
| **Still in top 5, size drifts** | Small gaps (rounding, PnL, partial cash) show as **ADD** (buy more) or **TRIM** (sell some) in the **backtest replay table** so each line stays near **1/5** of equity. |
| **Already at target** | **HOLD** — no trade row for that name that day. |
| **Dust trades** | Trades smaller than **`min_trade_dollars`** (~$25) are skipped unless you must **fully exit** (target weight 0). |

**`orders_today.csv`** is simpler: only **BUY** and **SELL** deltas vs your **persisted live state** (`live_portfolio.json`) so you can line up with a broker.

### Cash vs margin (settlement)

| Mode | How to run | Effect |
|------|------------|--------|
| **T+0 (margin-style)** | Default; `SETTLE_DAYS` unset or `0` | Sell proceeds can fund **same-day** buys in the sim. |
| **T+1 cash (Reg T)** | `SETTLE_DAYS=1` | Sale proceeds sit in **unsettled** until the **next** trading day; buys can be **delayed or scaled** if settled cash is tight. |

The **backtest tab** can flip **T+0 / T+1** in the browser (both paths are precomputed). The **live** banner follows the mode used in the **last** run (`SETTLE_DAYS`).

**Examples:**

```powershell
# Windows PowerShell — cash account behaviour for the live sim
$env:SETTLE_DAYS = "1"
python scripts/run_daily.py
```

```bash
export SETTLE_DAYS=1
python scripts/run_daily.py
```

### Live portfolio start date

Until the model’s prediction date is **on or after** this day, the live state stays **all cash** (backtest still runs).

- Config: `live_portfolio_start` in **`pipeline/config.py`**, or  
- Env: **`LIVE_PORTFOLIO_START=YYYY-MM-DD`**

Example:

```powershell
$env:LIVE_PORTFOLIO_START = "2026-05-12"
python scripts/run_daily.py
```

State is stored in **`docs/data/live_portfolio.json`** (and history in **`live_portfolio_history.csv`**).

---

## How backtesting works

1. **Walk-forward RF** trains on older years, validates, steps forward (**`refit_every_days`** ≈ one year). That produces **out-of-sample** predictions for historical dates — **`preds_oos`** — used only where each prediction was truly known **before** the trade date.  
2. **`simulate_portfolio`** replays those **OOS** predictions day by day against **adjusted closes**: same targeting and settlement rules as above.  
3. The **`docs/index.html`** backtest pane reads **`daily_views.json`**: pick any date to see holdings, recommendations, fills, and PnL to the **next** close.

So the equity curve is **not** “fit on full history then pretend we knew”; it’s **frozen walk-forward simulated performance** plus today’s fresh full-history fit only for **tomorrow’s** ranks and orders.

Limit sell overlays in the **table** are **reference only** — the simulator rotation is driven by **next-day rankings / exits**, not by those limit prices.

---

## Universe and optional “hot” names

| Piece | Details |
|-------|---------|
| **Core list** | Tickers from **`pipeline/sp500_constituents.csv`** plus index ETFs (**SPY**, **QQQ**) by default. |
| **Smaller / faster sleeve** | Set **`UNIVERSE=core`** (54-name legacy bundle) for smoke tests or CI. |
| **Refresh S&P names** | `python scripts/refresh_sp500.py` |
| **Hot watchlist** | If **`enable_hot_stocks`** is on, a momentum/volume screen can add up to **`hot_max_to_add`** extra tickers **for that run only** (see **`pipeline/hotstocks.py`**). |

---

## Configuration reference

Edit **`pipeline/config.py`** (or override with environment variables where noted):

- **`long_n` / `short_n`** — basket size; default is **5 long**, **0 short** (long-only).  
- **`cost_bps_per_side`** — model / panel cost; the **$5K live sim** in `run_daily.py` uses **`0` bps** (treat as manual, zero-commission execution).  
- **`enable_hot_stocks`**, **`hot_max_to_add`**, **`hot_min_score`** — hot-list behaviour.  
- **`live_portfolio_start`** — when the persisted live book starts trading.  

---

## Project layout

```
scripts/run_daily.py       # Main entry: download → features → model → portfolio → site
pipeline/config.py         # Defaults and knobs
pipeline/features.py       # Signals + cross-sectional ranks
pipeline/model.py          # Walk-forward + full-history fit
pipeline/portfolio.py      # Simulation, orders, daily_views
pipeline/report.py         # HTML report
docs/                      # GitHub Pages output (generated)
```

---

## GitHub contributors

**Insights → Contributors** uses **commit author email**. Set your identity before committing:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

Scheduled doc updates from Actions use the **`github-actions[bot]`** identity — that’s normal.
