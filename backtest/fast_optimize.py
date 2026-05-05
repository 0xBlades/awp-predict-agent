#!/usr/bin/env python3
"""
Fast Backtest Optimizer
Pre-computes indicators + index mapping for maximum speed.
"""

import sys, os, time, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.data_fetcher import load_or_fetch_all
from backtest.indicators import (compute_indicators, get_btc_trend, get_range_score,
    get_volatility, get_trend_signal, get_mean_signal)
from backtest.simulator import BacktestParams, Trade, simulate_trade_exit, compute_signal_weights, apply_fees
from backtest.metrics import calculate_metrics, objective_score, format_metrics

# --- PARAM GRID ---
PARAM_GRID = {
    "trend_bias": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "mean_bias": [0.8, 1.0, 1.2, 1.5],
    "long_threshold": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    "short_threshold": [-0.10, -0.15, -0.20, -0.25, -0.30, -0.35],
    "tp_pct": [0.003, 0.004, 0.005, 0.006, 0.008, 0.010],
    "sl_pct": [0.003, 0.004, 0.005, 0.006],
    "max_hold": [4, 6, 8, 10, 12, 15],
}


class FastBacktester:
    """Pre-computed backtest engine for optimization speed."""

    def __init__(self, data_dict, n_candles=5000):
        self.lookback = 50
        self.btc = compute_indicators(data_dict["BTCUSDT"])
        self.eth = compute_indicators(data_dict.get("ETHUSDT", data_dict.get("BTCUSDT")))

        self.n = min(len(self.btc), len(self.eth))

        # Pre-compute index mapping
        btc_times = self.btc["timestamp"].values
        eth_times = self.eth["timestamp"].values
        self.eth_idx_map = np.searchsorted(eth_times, btc_times, side="left")
        self.eth_idx_map = np.clip(self.eth_idx_map, 0, len(eth_times) - 1)

        # Pre-compute features for all candles
        self.precomputed = np.zeros((self.n, 5))
        for i in range(self.n):
            self.precomputed[i, 0] = get_btc_trend(self.btc, i)
            self.precomputed[i, 1] = get_range_score(self.btc, i)
            self.precomputed[i, 2] = get_volatility(self.btc, i)
            alt_idx = self.eth_idx_map[i]
            if alt_idx >= self.lookback:
                self.precomputed[i, 3] = get_trend_signal(self.eth, alt_idx)
                self.precomputed[i, 4] = get_mean_signal(self.eth, alt_idx)

        # Pre-compute ETH close prices
        self.eth_close = self.eth["close"].values
        self.eth_high = self.eth["high"].values
        self.eth_low = self.eth["low"].values
        self.eth_timestamps = self.eth["timestamp"].values

        print(f"Pre-computed {self.n} candles")

    def run(self, params):
        """Fast backtest run."""
        balance = 10000.0
        open_trades = []
        trades = []

        for i in range(self.lookback, self.n):
            btc_trend = self.precomputed[i, 0]
            range_s = self.precomputed[i, 1]
            vol_s = self.precomputed[i, 2]

            trend_raw = (0.5 * (btc_trend + 1) / 2 + 0.3 * range_s + 0.2 * vol_s)
            mean_raw = (0.5 * range_s + 0.3 * (1 - (btc_trend + 1) / 2) + 0.2 * vol_s)
            w_trend, w_mean = compute_signal_weights(trend_raw, mean_raw, params)

            # Manage open trades
            closed = []
            for j, trade in enumerate(open_trades):
                alt_idx = self.eth_idx_map[i]
                # Inline trade exit check for speed
                close = self.eth_close[alt_idx]
                high = self.eth_high[alt_idx]
                low = self.eth_low[alt_idx]

                trade.holding_periods += 1
                pnl = (close - trade.entry_price) / trade.entry_price if trade.direction == "LONG" else (trade.entry_price - close) / trade.entry_price

                if trade.direction == "LONG":
                    sl_price = trade.entry_price * (1 - params.sl_pct - params.slippage_pct)
                    tp_price = trade.entry_price * (1 + params.tp_pct)

                    if low <= sl_price:
                        trade.exit_price = sl_price
                        trade.exit_reason = "SL"
                        trade.pnl_pct = apply_fees(((sl_price - trade.entry_price) / trade.entry_price), params)
                        trade.pnl_chips = trade.tickets * trade.pnl_pct
                        balance += trade.pnl_chips
                        trades.append(trade)
                        closed.append(j)
                        continue
                    if high >= tp_price:
                        trade.exit_price = tp_price
                        trade.exit_reason = "TP"
                        trade.pnl_pct = apply_fees(((tp_price - trade.entry_price) / trade.entry_price), params)
                        trade.pnl_chips = trade.tickets * trade.pnl_pct
                        balance += trade.pnl_chips
                        trades.append(trade)
                        closed.append(j)
                        continue
                else:
                    sl_price = trade.entry_price * (1 + params.sl_pct + params.slippage_pct)
                    tp_price = trade.entry_price * (1 - params.tp_pct)

                    if high >= sl_price:
                        trade.exit_price = sl_price
                        trade.exit_reason = "SL"
                        trade.pnl_pct = apply_fees(((trade.entry_price - sl_price) / trade.entry_price), params)
                        trade.pnl_chips = trade.tickets * trade.pnl_pct
                        balance += trade.pnl_chips
                        trades.append(trade)
                        closed.append(j)
                        continue
                    if low <= tp_price:
                        trade.exit_price = tp_price
                        trade.exit_reason = "TP"
                        trade.pnl_pct = apply_fees(((trade.entry_price - tp_price) / trade.entry_price), params)
                        trade.pnl_chips = trade.tickets * trade.pnl_pct
                        balance += trade.pnl_chips
                        trades.append(trade)
                        closed.append(j)
                        continue

                if trade.holding_periods >= params.max_hold:
                    trade.exit_price = close
                    trade.exit_reason = "MAX_HOLD"
                    trade.pnl_pct = apply_fees(pnl, params)
                    trade.pnl_chips = trade.tickets * trade.pnl_pct
                    balance += trade.pnl_chips
                    trades.append(trade)
                    closed.append(j)

            for j in sorted(closed, reverse=True):
                open_trades.pop(j)

            # Signal
            if any(t.symbol == "ETHUSDT" for t in open_trades):
                continue

            alt_idx = self.eth_idx_map[i]
            if alt_idx < self.lookback:
                continue

            trend_sig = self.precomputed[i, 3]
            mean_sig = self.precomputed[i, 4]
            final = trend_sig * w_trend + mean_sig * w_mean

            if final > params.long_threshold:
                entry = self.eth_close[alt_idx]
                trade = Trade(
                    symbol="ETHUSDT", direction="LONG",
                    entry_price=float(entry),
                    entry_time=self.eth_timestamps[alt_idx],
                    entry_idx=int(alt_idx),
                    tp=float(entry * (1 + params.tp_pct)),
                    sl=float(entry * (1 - params.sl_pct)),
                    tickets=1000
                )
                open_trades.append(trade)

        # Close remaining
        for trade in open_trades:
            last_price = float(self.eth_close[-1])
            trade.exit_price = last_price
            trade.exit_reason = "BACKTEST_END"
            trade.pnl_pct = apply_fees(((last_price - trade.entry_price) / trade.entry_price), params)
            trade.pnl_chips = trade.tickets * trade.pnl_pct
            balance += trade.pnl_chips
            trades.append(trade)

        return trades, balance


def sample_random(grid):
    return {k: random.choice(v) for k, v in grid.items()}


def run_optimization(bt, n_trials=500):
    """Random search optimization."""
    best_score = -999
    best_params = None
    best_metrics = None
    history = []

    t_start = time.time()

    for trial in range(n_trials):
        param_dict = sample_random(PARAM_GRID)
        params = BacktestParams(**param_dict)

        trades, balance = bt.run(params)
        m = calculate_metrics(trades, [{"timestamp": 0, "balance": 10000, "equity": 10000, "unrealized": 0, "open_trades": 0}] * len(trades))
        # Override equity-related metrics
        m["final_balance"] = round(balance, 2)
        m["net_profit"] = round(balance - 10000, 2)
        m["total_return_pct"] = round((balance - 10000) / 10000 * 100, 2)

        score = objective_score(m)

        history.append({"params": param_dict, "score": score, "metrics": m})

        if score > best_score:
            best_score = score
            best_params = param_dict
            best_metrics = m
            print(f"  Trial {trial+1}: NEW BEST score={score:.4f} "
                  f"WR={m['winrate']*100:.0f}% PF={m['profit_factor']:.2f} "
                  f"Trades={m['total_trades']} Net=${m['net_profit']:.0f}")

        if (trial + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (trial + 1) / elapsed
            eta = (n_trials - trial - 1) / rate
            print(f"  [{trial+1}/{n_trials}] Best={best_score:.4f} | "
                  f"Rate={rate:.0f}/s | ETA={eta:.0f}s")

    total_time = time.time() - t_start
    print(f"\nOptimization done in {total_time:.1f}s ({n_trials/total_time:.0f} trials/s)")

    return best_params, best_metrics, history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()

    print(f"Fetching {args.candles} candles...")
    data = load_or_fetch_all(["BTCUSDT", "ETHUSDT"], "15m", args.candles)

    print(f"\nBuilding fast backtester...")
    bt = FastBacktester(data, args.candles)

    print(f"\nRunning {args.trials} optimization trials...")
    best_params, best_metrics, history = run_optimization(bt, args.trials)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"opt_fast_{int(time.time())}.json")
    with open(filepath, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "candles": args.candles,
            "trials": args.trials,
            "best_params": best_params,
            "best_score": objective_score(best_metrics) if best_metrics else 0,
            "metrics": best_metrics,
        }, f, indent=2)

    # Final backtest with best params
    print(f"\n{'='*50}")
    print(f"FINAL BACKTEST WITH BEST PARAMS")
    print(f"{'='*50}")
    params = BacktestParams(**best_params)
    trades, balance = bt.run(params)
    m = calculate_metrics(trades, [{"timestamp": 0, "balance": 10000, "equity": 10000}] * len(trades))
    m["final_balance"] = round(balance, 2)
    m["net_profit"] = round(balance - 10000, 2)
    m["total_return_pct"] = round((balance - 10000) / 10000 * 100, 2)

    print(format_metrics(m))
    print(f"\nBest params: {json.dumps(best_params, indent=2)}")

    # Show top 5 results
    history.sort(key=lambda x: x["score"], reverse=True)
    print(f"\nTop 5 parameter sets:")
    for i, h in enumerate(history[:5]):
        print(f"  #{i+1} score={h['score']:.4f} WR={h['metrics']['winrate']*100:.0f}% "
              f"PF={h['metrics']['profit_factor']:.2f} Net=${h['metrics']['net_profit']:.0f} "
              f"| {h['params']}")


if __name__ == "__main__":
    main()
