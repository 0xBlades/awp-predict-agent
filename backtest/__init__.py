"""
AWP Backtest Engine
===================
Automated backtesting + optimization for AWP Predict agent.

Usage:
    from backtest import run_backtest, random_search, walk_forward
    from backtest.data_fetcher import load_or_fetch_all
    from backtest.simulator import BacktestParams
    from backtest.metrics import calculate_metrics, objective_score
    from backtest.optimizer import random_search, walk_forward, save_results
"""

from .simulator import BacktestParams, run_backtest, Trade
from .metrics import calculate_metrics, objective_score, format_metrics
from .optimizer import random_search, walk_forward, grid_search, save_results, load_results
from .data_fetcher import load_or_fetch_all, fetch_multi_symbol
from .indicators import compute_indicators

__all__ = [
    "BacktestParams", "run_backtest", "Trade",
    "calculate_metrics", "objective_score", "format_metrics",
    "random_search", "walk_forward", "grid_search",
    "save_results", "load_results",
    "load_or_fetch_all", "fetch_multi_symbol",
    "compute_indicators",
]
