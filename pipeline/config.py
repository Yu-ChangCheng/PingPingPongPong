"""Configuration: universe, hyper-params, paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .universe import default_universe_bundle

# Resolve once at import so `Config()` matches one bundle. Override with
# `UNIVERSE=core` for the legacy 54-stock sleeve (faster CI / smoke tests).
_um = os.environ.get("UNIVERSE", "sp500").strip().lower()
if _um not in ("sp500", "core"):
    _um = "sp500"
DEFAULT_UNIVERSE_MODE: Literal["sp500", "core"] = _um  # type: ignore[assignment]
_DEFAULT_UNIVERSE, _DEFAULT_SECTORS = default_universe_bundle(
    "core" if _um == "core" else "sp500")

DEFAULT_INDICES: tuple[str, ...] = ("SPY", "QQQ")


@dataclass
class Config:
    start: str = "2010-01-01"
    end: str | None = None

    universe: tuple[str, ...] = field(default_factory=lambda: tuple(_DEFAULT_UNIVERSE))
    indices: tuple[str, ...] = DEFAULT_INDICES
    sectors: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_SECTORS))
    benchmark: str = "SPY"

    train_years: int = 5
    val_years: int = 1
    test_years: int = 1
    refit_every_days: int = 252

    n_estimators: int = 400
    max_depth: int = 4
    min_samples_leaf: int = 50
    max_features: str = "sqrt"

    long_n: int = 5
    short_n: int = 5
    cost_bps_per_side: float = 1.0

    # Live $5K simulated state: do not apply fills or mutate `live_portfolio.json`
    # until the model's latest prediction date is >= this day (ISO).
    # Override anytime with env `LIVE_PORTFOLIO_START=YYYY-MM-DD`.
    live_portfolio_start: str | None = "2026-05-12"

    enable_hot_stocks: bool = True
    hot_watchlist: tuple[str, ...] = ()   # empty -> use DEFAULT_WATCHLIST
    hot_max_to_add: int = 10
    hot_min_score: float = 1.0

    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def cache_dir(self) -> Path:
        return self.repo_root / "data_cache"

    @property
    def docs_dir(self) -> Path:
        return self.repo_root / "docs"

    @property
    def docs_data_dir(self) -> Path:
        return self.docs_dir / "data"

    @property
    def all_tickers(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.universe + self.indices)))


DEFAULT_CONFIG = Config()
