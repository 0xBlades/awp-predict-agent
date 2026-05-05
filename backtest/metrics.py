"""
AWP Backtest Engine — Performance Metrics
Comprehensive metrics for strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import List
from .simulator import Trade


def calculate_metrics(trades: List[Trade], equity_curve: list,
                      initial_balance: float = 10000.0) -> dict:
    """
    Calculate all performance metrics.
    Returns dict with comprehensive stats.
    """
    if not trades or not equity_curve:
        return _empty_metrics()

    # Basic stats
    total_trades = len(trades)
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    winrate = len(wins) / total_trades if total_trades > 0 else 0

    # PnL
    gross_profit = sum(t.pnl_chips for t in wins)
    gross_loss = abs(sum(t.pnl_chips for t in losses))
    net_profit = gross_profit - gross_loss

    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve
    eq = pd.DataFrame(equity_curve)
    equity_values = eq["equity"].values

    # Returns (percentage change)
    returns = np.diff(equity_values) / equity_values[:-1]
    returns = returns[~np.isnan(returns)]

    # Sharpe Ratio (annualized, assuming 15m candles = 35040/year)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(35040)
    else:
        sharpe = 0

    # Sortino Ratio (downside deviation only)
    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(35040)
    else:
        sortino = 0

    # Max Drawdown
    peak = np.maximum.accumulate(equity_values)
    drawdown = (peak - equity_values) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Max Drawdown Duration (in candles)
    in_drawdown = False
    dd_start = 0
    max_dd_duration = 0
    for idx in range(len(equity_values)):
        if equity_values[idx] < peak[idx]:
            if not in_drawdown:
                dd_start = idx
                in_drawdown = True
        else:
            if in_drawdown:
                duration = idx - dd_start
                max_dd_duration = max(max_dd_duration, duration)
                in_drawdown = False

    # Calmar Ratio (annualized return / max drawdown)
    total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
    n_candles = len(equity_values)
    annualized_return = total_return * (35040 / n_candles) if n_candles > 0 else 0
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else float("inf")

    # Trade duration
    avg_hold = np.mean([t.holding_periods for t in trades]) if trades else 0

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Consecutive wins/losses
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for t in trades:
        if t.pnl_pct > 0:
            if streak > 0:
                streak += 1
            else:
                streak = 1
            max_win_streak = max(max_win_streak, streak)
        else:
            if streak < 0:
                streak -= 1
            else:
                streak = -1
            max_loss_streak = max(max_loss_streak, abs(streak))

    return {
        # Core
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(winrate, 4),

        # PnL
        "net_profit": round(net_profit, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "avg_rr": round(avg_rr, 2),

        # Risk
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_drawdown, 4),
        "max_dd_duration": max_dd_duration,
        "calmar": round(calmar, 3),

        # Trade stats
        "avg_hold": round(avg_hold, 1),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "exit_reasons": exit_reasons,

        # Equity
        "initial_balance": initial_balance,
        "final_balance": round(equity_values[-1], 2),
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
    }


def _empty_metrics() -> dict:
    """Return empty metrics when no trades."""
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "winrate": 0,
        "net_profit": 0,
        "gross_profit": 0,
        "gross_loss": 0,
        "profit_factor": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "avg_rr": 0,
        "sharpe": 0,
        "sortino": 0,
        "max_drawdown": 0,
        "max_dd_duration": 0,
        "calmar": 0,
        "avg_hold": 0,
        "max_win_streak": 0,
        "max_loss_streak": 0,
        "exit_reasons": {},
        "initial_balance": 10000,
        "final_balance": 10000,
        "total_return_pct": 0,
        "annualized_return_pct": 0,
    }


def objective_score(metrics: dict) -> float:
    """
    Objective function for optimization.
    Balance growth + risk management.
    Penalizes: low winrate, high drawdown, low sharpe
    Rewards: high profit factor, positive sharpe
    """
    if metrics["total_trades"] < 10:
        return -999  # Too few trades

    score = (
        metrics["profit_factor"] * 2.0 +        # Reward consistent profit
        metrics["sharpe"] * 0.5 +                # Risk-adjusted return
        metrics["winrate"] * 3.0 +               # Win rate importance
        metrics["avg_rr"] * 0.3 +                # Risk-reward
        - metrics["max_drawdown"] * 10.0 +       # Penalize drawdown heavily
        min(metrics["total_trades"] / 50, 1.0) * 0.5  # Prefer more trades (up to 50)
    )

    return round(score, 4)


def format_metrics(metrics: dict) -> str:
    """Format metrics as readable string."""
    lines = [
        "=" * 50,
        "BACKTEST RESULTS",
        "=" * 50,
        f"Total Trades:    {metrics['total_trades']}",
        f"Win/Loss:        {metrics['wins']}W / {metrics['losses']}L",
        f"Win Rate:        {metrics['winrate']*100:.1f}%",
        f"Net Profit:      ${metrics['net_profit']:,.2f}",
        f"Profit Factor:   {metrics['profit_factor']:.2f}",
        f"Avg Win:         {metrics['avg_win']*100:.2f}%",
        f"Avg Loss:        {metrics['avg_loss']*100:.2f}%",
        f"Avg R:R:         {metrics['avg_rr']:.2f}",
        "-" * 50,
        f"Sharpe Ratio:    {metrics['sharpe']:.3f}",
        f"Sortino Ratio:   {metrics['sortino']:.3f}",
        f"Max Drawdown:    {metrics['max_drawdown']*100:.1f}%",
        f"Calmar Ratio:    {metrics['calmar']:.3f}",
        "-" * 50,
        f"Avg Hold:        {metrics['avg_hold']:.1f} candles",
        f"Win Streak:      {metrics['max_win_streak']}",
        f"Loss Streak:     {metrics['max_loss_streak']}",
        f"Exit Reasons:    {metrics['exit_reasons']}",
        "=" * 50,
        f"Balance:         ${metrics['initial_balance']:,.2f} -> ${metrics['final_balance']:,.2f}",
        f"Total Return:    {metrics['total_return_pct']:.2f}%",
        f"Objective Score: {objective_score(metrics):.4f}",
        "=" * 50,
    ]
    return "\n".join(lines)
