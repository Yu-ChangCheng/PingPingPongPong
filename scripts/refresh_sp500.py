"""Refresh bundled ``pipeline/sp500_constituents.csv`` from Wikipedia."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.universe import SP500_CONSTITUENTS_CSV, download_sp500_to_csv


def main() -> None:
    tickers, _ = download_sp500_to_csv(SP500_CONSTITUENTS_CSV)
    print(f"Wrote {SP500_CONSTITUENTS_CSV} ({len(tickers)} tickers)")


if __name__ == "__main__":
    main()
