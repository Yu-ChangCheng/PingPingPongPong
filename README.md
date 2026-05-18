# PingPingPongPong — RF daily stock signal & GitHub Pages dashboard

Daily **Random Forest** signals (Gu / Kelly / Xiu style), a **$5K paper portfolio** tracker,
and a static site published from **`docs/`** — same engine as the parent project, trimmed to
what you need to run locally and on **GitHub Actions**.

```
.
├── pipeline/                 Python package (data, features, model, portfolio, report)
├── scripts/run_daily.py      single entry point
├── .github/workflows/daily.yml
├── docs/                     GitHub Pages root (index.html + data/ produced by the script)
└── requirements.txt
```

---

## Quick start (local)

```bash
cd PingPingPongPong
pip install -r requirements.txt
python scripts/run_daily.py
# Open docs/index.html
```

First run downloads history into `data_cache/` (gitignored).

---

## Deploy on GitHub

### 1. Create the repo on GitHub

Create an empty repository named **`PingPingPongPong`** (no README/license if you will push this tree).

### 2. Push this folder

```bash
cd C:\Users\baby_\Downloads\PingPingPongPong
git init
git add .
git commit -m "Initial PingPingPongPong pipeline"
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/PingPingPongPong.git
git push -u origin main
```

### 3. Enable GitHub Pages

**Settings → Pages** → Deploy from branch **`main`** / folder **`/docs`** → Save.

Site: `https://<YOUR_USER>.github.io/PingPingPongPong/` (or your custom domain if you add one).

### 4. Allow Actions to commit

**Settings → Actions → General → Workflow permissions** → **Read and write** → Save.

### 5. Daily automation

`.github/workflows/daily.yml` runs **4:05 PM America/New_York, Mon–Fri** (five minutes after the cash close; UTC trigger times shift with DST). Manual runs and pushes are unchanged. It runs `python scripts/run_daily.py`, commits changes under `docs/`, and Pages updates automatically. You then have the rest of the US after-hours window (open through 8 PM ET on a margin account) to place the orders near the close.

Manual run: **Actions → Daily prediction → Run workflow**.

Optional repo variable **`LIVE_PORTFOLIO_START`** (ISO date) overrides `pipeline/config.py` → `live_portfolio_start` for when simulated live fills begin.

---

## Live $5K start date

`pipeline/config.py` → **`live_portfolio_start`** (default `2026-05-12`). Before that prediction date, cash stays **$5,000** with no holdings; orders are still generated for planning.

---

## Customising

Universe, model hyperparameters, `long_n` / `short_n`, schedule (`cron` in UTC): see `pipeline/config.py` and `.github/workflows/daily.yml`.

---

## Survivorship bias

The default universe is a **current** large-cap list — historical backtest is optimistic. Forward tracker and live paper strip are the honest record. See the original project README for upgrade paths (point-in-time universe, etc.).

---

## Disclaimer

Educational only — not investment advice.
