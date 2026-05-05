"""
AWP Backtest Engine — Optimizer
Random Search + Walk-Forward Optimization.
"""

import random
import json
import os
import time
import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple
from .simulator import BacktestParams, run_backtest
from .metrics import calculate_metrics, objective_score, format_metrics


# --- Parameter Space ---
PARAM_GRID = {
    "trend_bias": [0.6, 0.7, 0.8, 0.9, 1.0],
    "mean_bias": [1.0, 1.1, 1.2, 1.3, 1.5],
    "long_threshold": [0.15, 0.20, 0.25, 0.30, 0.35],
    "short_threshold": [-0.15, -0.20, -0.25, -0.30, -0.35],
    "tp_pct": [0.003, 0.004, 0.005, 0.006, 0.008],
    "sl_pct": [0.003, 0.004, 0.005, 0.006],
    "max_hold": [4, 6, 8, 10, 12],
}


def sample_random(grid: dict) -> dict:
    """Sample one random combination from parameter grid."""
    return {k: random.choice(v) for k, v in grid.items()}


def grid_search(data: dict, btc_data: pd.DataFrame,
                param_grid: dict = None,
                lookback: int = 50,
                max_combos: int = 500) -> Tuple[dict, dict, list]:
    """
    Grid search optimization.
    Returns (best_params, best_metrics, all_results)
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(product(*values))

    # Cap at max_combos
    if len(all_combos) > max_combos:
        random.shuffle(all_combos)
        all_combos = all_combos[:max_combos]

    print(f"Grid Search: {len(all_combos)} combinations")

    best_score = -999
    best_params = None
    best_metrics = None
    all_results = []

    for idx, combo in enumerate(all_combos):
        param_dict = dict(zip(keys, combo))
        params = BacktestParams(**param_dict)

        try:
            result = run_backtest(data, btc_data, params, lookback)
            m = calculate_metrics(result["trades"], result["equity_curve"])
            score = objective_score(m)

            all_results.append({
                "params": param_dict,
                "score": score,
                "metrics": m,
            })

            if score > best_score:
                best_score = score
                best_params = param_dict
                best_metrics = m

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(all_combos)}] Best score: {best_score:.4f}")

        except Exception as e:
            pass  # Skip failed combos

    return best_params, best_metrics, all_results


def random_search(data: dict, btc_data: pd.DataFrame,
                  param_grid: dict = None,
                  n_trials: int = 200,
                  lookback: int = 50) -> Tuple[dict, dict, list]:
    """
    Random search optimization.
    Faster than grid search, often finds comparable results.
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    print(f"Random Search: {n_trials} trials")

    best_score = -999
    best_params = None
    best_metrics = None
    all_results = []

    for trial in range(n_trials):
        param_dict = sample_random(param_grid)
        params = BacktestParams(**param_dict)

        try:
            result = run_backtest(data, btc_data, params, lookback)
            m = calculate_metrics(result["trades"], result["equity_curve"])
            score = objective_score(m)

            all_results.append({
                "params": param_dict,
                "score": score,
                "metrics": m,
            })

            if score > best_score:
                best_score = score
                best_params = param_dict
                best_metrics = m
                print(f"  Trial {trial+1}: NEW BEST score={score:.4f} "
                      f"WR={m['winrate']*100:.0f}% PF={m['profit_factor']:.2f}")

            if (trial + 1) % 50 == 0:
                print(f"  [{trial+1}/{n_trials}] Best score: {best_score:.4f}")

        except Exception as e:
            pass

    return best_params, best_metrics, all_results


def walk_forward(data: dict, btc_data: pd.DataFrame,
                 n_splits: int = 3,
                 train_ratio: float = 0.7,
                 optimize_trials: int = 100,
                 lookback: int = 50) -> dict:
    """
    Walk-Forward Optimization (pro-level anti-overfit).

    Split data into n_splits windows.
    For each: optimize on train -> validate on test.
    Returns overall out-of-sample performance.
    """
    print(f"\nWalk-Forward Optimization: {n_splits} splits, {train_ratio*100:.0f}% train")

    # Find total length
    total_len = len(btc_data)
    window_size = total_len // n_splits

    oos_results = []
    all_best_params = []

    for split in range(n_splits):
        start = split * window_size
        end = min(start + window_size, total_len)

        if end - start < 200:  # Minimum data needed
            continue

        split_btc = btc_data.iloc[start:end].reset_index(drop=True)

        # Split into altcoin data
        split_data = {}
        for sym, df in data.items():
            # Find matching timestamps
            timestamps = split_btc["timestamp"]
            first_t = timestamps.iloc[0]
            last_t = timestamps.iloc[-1]
            mask = (df["timestamp"] >= first_t) & (df["timestamp"] <= last_t)
            split_data[sym] = df[mask].reset_index(drop=True)

        # Train/test split
        train_end = int(len(split_btc) * train_ratio)

        train_btc = split_btc.iloc[:train_end].reset_index(drop=True)
        test_btc = split_btc.iloc[train_end:].reset_index(drop=True)

        train_data = {}
        test_data = {}
        for sym in data:
            train_data[sym] = split_data[sym].iloc[:train_end].reset_index(drop=True)
            test_data[sym] = split_data[sym].iloc[train_end:].reset_index(drop=True)

        print(f"\n--- Split {split+1}/{n_splits} ---")
        print(f"  Train: {len(train_btc)} candles | Test: {len(test_btc)} candles")

        # Optimize on train
        best_params, train_metrics, _ = random_search(
            train_data, train_btc, n_trials=optimize_trials, lookback=lookback
        )

        if best_params is None:
            print(f"  No valid params found, skipping...")
            continue

        # Validate on test (out-of-sample)
        params = BacktestParams(**best_params)
        test_result = run_backtest(test_data, test_btc, params, lookback)
        test_metrics = calculate_metrics(test_result["trades"], test_result["equity_curve"])

        # Check for overfitting
        train_score = objective_score(train_metrics)
        test_score = objective_score(test_metrics)
        degradation = (train_score - test_score) / abs(train_score) * 100 if train_score != 0 else 0

        print(f"  Train score: {train_score:.4f} | Test score: {test_score:.4f} | Degradation: {degradation:.1f}%")

        oos_results.append({
            "split": split + 1,
            "best_params": best_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_score": train_score,
            "test_score": test_score,
            "degradation_pct": degradation,
        })
        all_best_params.append(best_params)

    # Aggregate out-of-sample results
    all_test_trades = []
    for r in oos_results:
        params = BacktestParams(**r["best_params"])
        test_result = run_backtest(test_data, test_btc, params, lookback)
        all_test_trades.extend(test_result["trades"])

    # Overall OOS metrics
    if all_test_trades:
        # Reconstruct equity from all test trades
        total_pnl = sum(t.pnl_chips for t in all_test_trades)
        wins = len([t for t in all_test_trades if t.pnl_pct > 0])
        oos_winrate = wins / len(all_test_trades) if all_test_trades else 0
    else:
        total_pnl = 0
        oos_winrate = 0

    # Average best params across splits
    avg_params = {}
    if all_best_params:
        for key in all_best_params[0]:
            vals = [p[key] for p in all_best_params]
            if isinstance(vals[0], float):
                avg_params[key] = round(np.mean(vals), 4)
            else:
                avg_params[key] = vals[0]  # Use first split for int params

    print(f"\n{'='*50}")
    print(f"WALK-FORWARD SUMMARY")
    print(f"{'='*50}")
    print(f"  OOS Total Trades:  {len(all_test_trades)}")
    print(f"  OOS Win Rate:      {oos_winrate*100:.1f}%")
    print(f"  OOS Net PnL:       ${total_pnl:,.2f}")
    print(f"  Avg Best Params:   {avg_params}")
    print(f"  Avg Degradation:   {np.mean([r['degradation_pct'] for r in oos_results]):.1f}%")
    print(f"{'='*50}")

    return {
        "splits": oos_results,
        "avg_params": avg_params,
        "oos_trades": len(all_test_trades),
        "oos_winrate": oos_winrate,
        "oos_pnl": total_pnl,
    }


def save_results(best_params: dict, metrics: dict, filepath: str):
    """Save optimization results to JSON."""
    output = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "best_params": best_params,
        "metrics": metrics,
        "objective_score": objective_score(metrics),
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> dict:
    """Load optimization results from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)
