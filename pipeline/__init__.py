"""Daily cross-sectional Random Forest pipeline (prices → features → model → report)."""
from .config import Config, DEFAULT_CONFIG, DEFAULT_UNIVERSE_MODE
from .data import download_prices
from .features import build_panel, FEATURE_COLS, cross_section_rank
from .model import walk_forward_rf, FoldResult
from .backtest import cross_sectional_backtest, perf_stats
from .tracker import update_tracker, load_tracker
from .report import build_site
from .hotstocks import (DEFAULT_WATCHLIST, HotConfig, score_hot_stocks,
                         select_hot_additions)
from .strategy import (StrategyConfig, run_strategy, PRESET_STRATEGIES,
                        STRATEGY_GUIDE)
from .portfolio import (PortfolioConfig, PortfolioState, Order,
                          simulate_portfolio, compute_stats, compute_hit_rate,
                          realized_pnl_by_ticker,
                          make_today_orders, make_holdings_view,
                          build_daily_views)

__all__ = [
    "Config", "DEFAULT_CONFIG", "DEFAULT_UNIVERSE_MODE",
    "download_prices",
    "build_panel", "FEATURE_COLS", "cross_section_rank",
    "walk_forward_rf", "FoldResult",
    "cross_sectional_backtest", "perf_stats",
    "update_tracker", "load_tracker",
    "build_site",
    "DEFAULT_WATCHLIST", "HotConfig", "score_hot_stocks", "select_hot_additions",
    "StrategyConfig", "run_strategy", "PRESET_STRATEGIES", "STRATEGY_GUIDE",
    "PortfolioConfig", "PortfolioState", "Order",
    "simulate_portfolio", "compute_stats", "compute_hit_rate",
    "realized_pnl_by_ticker",
    "make_today_orders", "make_holdings_view", "build_daily_views",
]
