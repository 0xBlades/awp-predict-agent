"""
AWP Backtest Engine — Strategy Simulator
Core backtest loop + realistic trade execution.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Trade:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: pd.Timestamp
    entry_idx: int
    tp: float
    sl: float
    tickets: int = 1000
    exit_price: float = 0.0
    exit_time: Optional[pd.Timestamp] = None
    exit_idx: int = 0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_chips: float = 0.0
    holding_periods: int = 0


@dataclass
class BacktestParams:
    """Parameters to optimize."""
    # Weight engine
    trend_bias: float = 0.8
    mean_bias: float = 1.2

    # Signal thresholds
    long_threshold: float = 0.25
    short_threshold: float = -0.25

    # Trade management
    tp_pct: float = 0.006  # 0.6%
    sl_pct: float = 0.004  # 0.4%
    max_hold: int = 8      # candles

    # Fees & slippage
    fee_pct: float = 0.0004   # 0.04% per side
    slippage_pct: float = 0.0002  # 0.02%

    # Risk
    risk_per_trade: float = 0.02  # 2% of balance
    initial_balance: float = 10000.0

    # Ticket sizing
    ticket_default: int = 1000
    grade_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "A": 1.15, "B": 1.0, "C": 0.85
    })


def calculate_trade_pnl(trade: Trade, current_price: float) -> float:
    """Calculate PnL as percentage."""
    if trade.direction == "LONG":
        return (current_price - trade.entry_price) / trade.entry_price
    else:
        return (trade.entry_price - current_price) / trade.entry_price


def simulate_trade_exit(trade: Trade, candle: pd.Series, params: BacktestParams,
                        current_idx: int) -> bool:
    """
    Check if trade should be closed.
    Returns True if closed, False if still open.
    """
    close = candle["close"]
    high = candle["high"]
    low = candle["low"]

    pnl = calculate_trade_pnl(trade, close)
    trade.holding_periods += 1

    # Check TP (using high/low for realism)
    if trade.direction == "LONG":
        tp_price = trade.entry_price * (1 + params.tp_pct)
        sl_price = trade.entry_price * (1 - params.slippage_pct)
        sl_price_actual = trade.entry_price * (1 - params.sl_pct - params.slippage_pct)

        # SL hit first check (worst case within candle)
        if low <= sl_price_actual:
            trade.exit_price = sl_price_actual
            trade.exit_reason = "SL"
            trade.exit_time = candle["timestamp"]
            trade.exit_idx = current_idx
            trade.pnl_pct = calculate_trade_pnl(trade, trade.exit_price)
            trade.pnl_chips = trade.tickets * trade.pnl_pct
            return True

        # TP hit
        if high >= tp_price:
            trade.exit_price = tp_price
            trade.exit_reason = "TP"
            trade.exit_time = candle["timestamp"]
            trade.exit_idx = current_idx
            trade.pnl_pct = calculate_trade_pnl(trade, trade.exit_price)
            trade.pnl_chips = trade.tickets * trade.pnl_pct
            return True

    else:  # SHORT
        tp_price = trade.entry_price * (1 - params.tp_pct)
        sl_price_actual = trade.entry_price * (1 + params.sl_pct + params.slippage_pct)

        # SL hit
        if high >= sl_price_actual:
            trade.exit_price = sl_price_actual
            trade.exit_reason = "SL"
            trade.exit_time = candle["timestamp"]
            trade.exit_idx = current_idx
            trade.pnl_pct = calculate_trade_pnl(trade, trade.exit_price)
            trade.pnl_chips = trade.tickets * trade.pnl_pct
            return True

        # TP hit
        if low <= tp_price:
            trade.exit_price = tp_price
            trade.exit_reason = "TP"
            trade.exit_time = candle["timestamp"]
            trade.exit_idx = current_idx
            trade.pnl_pct = calculate_trade_pnl(trade, trade.exit_price)
            trade.pnl_chips = trade.tickets * trade.pnl_pct
            return True

    # Max hold
    if trade.holding_periods >= params.max_hold:
        trade.exit_price = close
        trade.exit_reason = "MAX_HOLD"
        trade.exit_time = candle["timestamp"]
        trade.exit_idx = current_idx
        trade.pnl_pct = calculate_trade_pnl(trade, close)
        trade.pnl_chips = trade.tickets * trade.pnl_pct
        return True

    return False


def compute_signal_weights(trend_score: float, mean_score: float,
                           params: BacktestParams) -> tuple:
    """Compute normalized signal weights with bias."""
    trend_adj = trend_score * params.trend_bias
    mean_adj = mean_score * params.mean_bias
    total = trend_adj + mean_adj
    if total == 0:
        return 0.5, 0.5
    return trend_adj / total, mean_adj / total


def apply_fees(pnl_pct: float, params: BacktestParams) -> float:
    """Subtract entry + exit fees + slippage."""
    total_cost = 2 * (params.fee_pct + params.slippage_pct)
    return pnl_pct - total_cost


def run_backtest(data: dict, btc_data: pd.DataFrame,
                 params: BacktestParams,
                 lookback: int = 50) -> dict:
    """
    Core backtest loop.

    Args:
        data: dict of {symbol: DataFrame} for altcoins
        btc_data: BTC DataFrame (for trend/breadth)
        params: BacktestParams
        lookback: warmup period

    Returns:
        dict with trades, equity_curve, metrics
    """
    from .indicators import (
        compute_indicators, get_btc_trend, get_range_score,
        get_volatility, get_trend_signal, get_mean_signal
    )

    # Compute indicators on BTC
    btc = compute_indicators(btc_data)

    # Compute indicators on each altcoin
    alt_data = {}
    for sym, df in data.items():
        alt_data[sym] = compute_indicators(df)

    # Find common timestamps
    common_times = btc["timestamp"]
    n = len(common_times)
    max_len = min(n, min(len(alt_data[s]["timestamp"]) for s in alt_data))

    # Balance tracking
    balance = params.initial_balance
    equity_curve = []
    trades: List[Trade] = []
    open_trades: List[Trade] = []

    # Candle-by-candle simulation
    for i in range(lookback, max_len):
        current_time = btc["timestamp"].iloc[i]
        current_candle = btc.iloc[i]

        # --- Feature Extraction ---
        btc_trend = get_btc_trend(btc, i)
        range_score = get_range_score(btc, i)
        vol_score = get_volatility(btc, i)

        # --- Weight Engine ---
        trend_score_raw = (
            0.5 * (btc_trend + 1) / 2 +   # Normalize to 0-1
            0.3 * range_score +
            0.2 * vol_score
        )
        mean_score_raw = (
            0.5 * range_score +
            0.3 * (1 - (btc_trend + 1) / 2) +
            0.2 * vol_score
        )

        w_trend, w_mean = compute_signal_weights(trend_score_raw, mean_score_raw, params)

        # --- Manage Open Trades ---
        closed_indices = []
        for j, trade in enumerate(open_trades):
            sym = trade.symbol
            # Find matching candle in alt data (use closest timestamp)
            alt_df = alt_data[sym]
            time_diffs = (alt_df["timestamp"] - current_time).abs()
            closest_idx = time_diffs.idxmin()
            alt_candle = alt_df.loc[closest_idx]

            if simulate_trade_exit(trade, alt_candle, params, closest_idx):
                # Apply fees
                trade.pnl_pct = apply_fees(trade.pnl_pct, params)
                trade.pnl_chips = trade.tickets * trade.pnl_pct
                balance += trade.pnl_chips
                trades.append(trade)  # Save closed trade
                closed_indices.append(j)

        # Remove closed trades (reverse order)
        for j in sorted(closed_indices, reverse=True):
            open_trades.pop(j)

        # --- Signal for each altcoin ---
        for sym, alt_df in alt_data.items():
            # Skip if already have position in this symbol
            if any(t.symbol == sym for t in open_trades):
                continue

            # Find matching candle (use closest timestamp)
            time_diffs = (alt_df["timestamp"] - current_time).abs()
            closest_idx = time_diffs.idxmin()
            alt_idx = closest_idx
            alt_candle = alt_df.loc[alt_idx]

            # Skip if too early
            if alt_idx < lookback:
                continue

            # Compute signals
            trend_sig = get_trend_signal(alt_df, alt_idx)
            mean_sig = get_mean_signal(alt_df, alt_idx)
            final_score = (trend_sig * w_trend) + (mean_sig * w_mean)

            # Decision
            direction = None
            if final_score > params.long_threshold:
                direction = "LONG"
            elif final_score < params.short_threshold:
                direction = "SHORT"

            if direction:
                entry_price = alt_candle["close"]  # Enter at close

                # Grade tickets based on signal quality
                confidence = abs(final_score)
                if confidence > 0.5:
                    grade_mult = params.grade_multipliers["A"]
                elif confidence > 0.3:
                    grade_mult = params.grade_multipliers["B"]
                else:
                    grade_mult = params.grade_multipliers["C"]

                tickets = int(params.ticket_default * grade_mult)

                # Calculate TP/SL prices
                if direction == "LONG":
                    tp_price = entry_price * (1 + params.tp_pct)
                    sl_price = entry_price * (1 - params.sl_pct)
                else:
                    tp_price = entry_price * (1 - params.tp_pct)
                    sl_price = entry_price * (1 + params.sl_pct)

                trade = Trade(
                    symbol=sym,
                    direction=direction,
                    entry_price=entry_price,
                    entry_time=current_time,
                    entry_idx=alt_idx,
                    tp=tp_price,
                    sl=sl_price,
                    tickets=tickets
                )
                open_trades.append(trade)

        # Record equity
        unrealized = 0
        for t in open_trades:
            sym = t.symbol
            alt_df = alt_data[sym]
            time_diffs = (alt_df["timestamp"] - current_time).abs()
            closest_idx = time_diffs.idxmin()
            price = alt_df.loc[closest_idx, "close"]
            unrealized += calculate_trade_pnl(t, price) * t.tickets

        equity_curve.append({
            "timestamp": current_time,
            "balance": balance,
            "unrealized": unrealized,
            "equity": balance + unrealized,
            "open_trades": len(open_trades),
        })

    # Close any remaining open trades at last price
    for trade in open_trades:
        sym = trade.symbol
        alt_df = alt_data[sym]
        if len(alt_df) > 0:
            last_price = alt_df["close"].iloc[-1]
            trade.exit_price = last_price
            trade.exit_reason = "BACKTEST_END"
            trade.exit_time = alt_df["timestamp"].iloc[-1]
            trade.pnl_pct = apply_fees(calculate_trade_pnl(trade, last_price), params)
            trade.pnl_chips = trade.tickets * trade.pnl_pct
            balance += trade.pnl_chips
            trades.append(trade)

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_balance": balance,
        "params": params,
    }
