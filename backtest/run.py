#!/usr/bin/env python3
"""
AWP Backtest Runner
===================
Main entry point for running backtests and optimization.

Usage:
    python3 backtest/run.py                    # Quick backtest with defaults
    python3 backtest/run.py --optimize         # Random search optimization
    python3 backtest/run.py --walk-forward     # Walk-forward optimization
    python3 backtest/run.py --grid             # Grid search (slow)
    python3 backtest/run.py --candles 10000    # More data
    python3 backtest/run.py --trials 500       # More optimization trials
"""

import sys
import os
import argparse
import time
import json

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.data_fetcher import load_or_fetch_all
from backtest.simulator import BacktestParams, run_backtest
from backtest.metrics import calculate_metrics, objective_score, format_metrics
from backtest.optimizer import random_search, walk_forward, grid_search, save_results

# Default config
DEFAULT_CANDLES = 5000
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_INTERVAL = "15m"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_quick_backtest(candles: int = DEFAULT_CANDLES,
                       symbols: list = None,
                       params: BacktestParams = None) -> dict:
    """Run a quick backtest with current parameters."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    if params is None:
        params = BacktestParams()

    print(f"\n{'='*50}")
    print(f"QUICK BACKTEST")
    print(f"{'='*50}")
    print(f"  Candles:  {candles}")
    print(f"  Symbols:  {symbols}")
    print(f"  Params:   TP={params.tp_pct*100:.1f}% SL={params.sl_pct*100:.1f}% "
          f"MaxHold={params.max_hold} Threshold={params.long_threshold}")

    # Fetch data
    t0 = time.time()
    print(f"\nFetching data...")
    data = load_or_fetch_all(symbols, DEFAULT_INTERVAL, candles)
    btc_data = data.get("BTCUSDT", list(data.values())[0])
    fetch_time = time.time() - t0
    print(f"Data fetched in {fetch_time:.1f}s")

    # Run backtest
    t0 = time.time()
    print(f"\nRunning backtest...")
    result = run_backtest(data, btc_data, params, lookback=50)
    bt_time = time.time() - t0
    print(f"Backtest completed in {bt_time:.1f}s")

    # Calculate metrics
    m = calculate_metrics(result["trades"], result["equity_curve"])

    # Print results
    print(f"\n{format_metrics(m)}")

    return result, m


def run_optimization(mode: str = "random",
                     candles: int = DEFAULT_CANDLES,
                     symbols: list = None,
                     trials: int = 200) -> dict:
    """Run optimization."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print(f"\n{'='*50}")
    print(f"OPTIMIZATION ({mode.upper()})")
    print(f"{'='*50}")

    # Fetch data
    print(f"\nFetching data...")
    data = load_or_fetch_all(symbols, DEFAULT_INTERVAL, candles)
    btc_data = data.get("BTCUSDT", list(data.values())[0])

    # Run optimizer
    t0 = time.time()
    if mode == "grid":
        best_params, best_metrics, all_results = grid_search(data, btc_data, max_combos=trials)
    elif mode == "walk-forward":
        wf_result = walk_forward(data, btc_data, n_splits=3, optimize_trials=trials)
        best_params = wf_result["avg_params"]
        best_metrics = {"oos_winrate": wf_result["oos_winrate"],
                        "oos_pnl": wf_result["oos_pnl"]}
        all_results = wf_result["splits"]
    else:  # random
        best_params, best_metrics, all_results = random_search(data, btc_data, n_trials=trials)

    opt_time = time.time() - t0

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, f"opt_{mode}_{int(time.time())}.json")
    save_results(best_params, best_metrics, filepath)

    # Print summary
    print(f"\nOptimization completed in {opt_time:.1f}s")
    print(f"Best params: {best_params}")
    print(f"Best score:  {objective_score(best_metrics) if isinstance(best_metrics, dict) and 'winrate' in best_metrics else 'N/A'}")

    # Run final backtest with best params
    print(f"\n{'='*50}")
    print(f"FINAL BACKTEST WITH BEST PARAMS")
    print(f"{'='*50}")

    final_params = BacktestParams(**best_params) if isinstance(best_params, dict) else BacktestParams()
    final_result = run_backtest(data, btc_data, final_params, lookback=50)
    final_metrics = calculate_metrics(final_result["trades"], final_result["equity_curve"])

    print(f"\n{format_metrics(final_metrics)}")

    return {
        "best_params": best_params,
        "best_metrics": final_metrics,
        "all_results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="AWP Backtest Runner")
    parser.add_argument("--optimize", action="store_true", help="Run random search optimization")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward optimization")
    parser.add_argument("--grid", action="store_true", help="Run grid search optimization")
    parser.add_argument("--candles", type=int, default=DEFAULT_CANDLES, help="Number of candles")
    parser.add_argument("--trials", type=int, default=200, help="Optimization trials")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Symbols to test")

    args = parser.parse_args()

    if args.walk_forward:
        run_optimization("walk-forward", args.candles, args.symbols, args.trials)
    elif args.grid:
        run_optimization("grid", args.candles, args.symbols, args.trials)
    elif args.optimize:
        run_optimization("random", args.candles, args.symbols, args.trials)
    else:
        run_quick_backtest(args.candles, args.symbols)


if __name__ == "__main__":
    main()
