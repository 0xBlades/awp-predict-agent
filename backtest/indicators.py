"""
AWP Backtest Engine — Technical Indicators
All indicators used by the signal engine, computed on DataFrame.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 20) -> pd.Series:
    """Rolling VWAP."""
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).rolling(period).sum()
    cum_vol = volume.rolling(period).sum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands: middle, upper, lower."""
    mid = sma(close, period)
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return mid, upper, lower


def ema_spread(close: pd.Series, fast: int = 9, slow: int = 21) -> pd.Series:
    """EMA spread as percentage."""
    return (ema(close, fast) - ema(close, slow)).abs() / close * 100


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators needed by the signal engine.
    Expects DataFrame with: open, high, low, close, volume
    Returns DataFrame with all indicator columns added.
    """
    df = df.copy()

    # EMAs
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)

    # EMA spread
    df["ema_spread"] = ema_spread(df["close"], 9, 21)

    # RSI
    df["rsi"] = rsi(df["close"], 14)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])

    # ATR
    df["atr"] = atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = df["atr"] / df["close"] * 100

    # VWAP
    df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])
    df["vwap_dist"] = (df["close"] - df["vwap"]).abs() / df["close"] * 100

    # Bollinger Bands
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(df["close"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100

    # Volume
    df["vol_sma20"] = sma(df["volume"], 20)
    df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)

    # Previous candles for multi-timeframe aggregation
    df["prev_close"] = df["close"].shift(1)

    # Trend strength (price vs EMA50)
    df["trend_strength"] = (df["close"] - df["ema50"]) / df["ema50"] * 100

    return df


def compute_htf_indicators(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute higher timeframe indicators for MTF analysis.
    Aggregate 5m data to 15m, 1h, 4h.
    """
    result = {}

    # 15m aggregation (already 15m in our case, but can aggregate from 5m)
    if len(df_5m.columns) > 0:
        agg_15m = df_5m.resample("15min", on="timestamp").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()

        agg_1h = df_5m.resample("1h", on="timestamp").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()

        for label, tf_df in [("15m", agg_15m), ("1h", agg_1h)]:
            result[label] = compute_indicators(tf_df)

    return result


# --- Feature extraction for backtest ---

def get_btc_trend(btc_df: pd.DataFrame, idx: int) -> float:
    """BTC trend score: 1 = bullish, -1 = bearish, 0 = neutral."""
    if idx < 50:
        return 0
    close = btc_df["close"].iloc[idx]
    ema21 = btc_df["ema21"].iloc[idx]
    ema50 = btc_df["ema50"].iloc[idx]

    if close > ema21 > ema50:
        return 1.0
    elif close < ema21 < ema50:
        return -1.0
    elif close > ema50:
        return 0.5
    elif close < ema50:
        return -0.5
    return 0


def get_range_score(df: pd.DataFrame, idx: int) -> float:
    """
    Range score: 1 = ranging (low EMA spread, low BB width),
    0 = trending (high spread, wide BB)
    """
    if idx < 50:
        return 0.5
    spread = df["ema_spread"].iloc[idx]
    bb_width = df["bb_width"].iloc[idx]

    # Normalize: low spread = high range score
    spread_score = max(0, 1 - spread / 0.5)  # 0.5% spread = 0 score
    bb_score = max(0, 1 - bb_width / 4.0)    # 4% width = 0 score

    return (spread_score + bb_score) / 2


def get_volatility(df: pd.DataFrame, idx: int) -> float:
    """Volatility score: 0-1, higher = more volatile."""
    if idx < 50:
        return 0.5
    atr_pct = df["atr_pct"].iloc[idx]
    # Normalize: 0.5% atr = low vol, 1.5% = high vol
    return min(1.0, max(0, (atr_pct - 0.2) / 1.3))


def get_trend_signal(df: pd.DataFrame, idx: int) -> float:
    """
    Trend signal score: +1 = strong long, -1 = strong short, 0 = neutral.
    Uses EMA alignment + MACD + RSI + volume.
    """
    if idx < 50:
        return 0

    score = 0

    # EMA alignment
    close = df["close"].iloc[idx]
    ema9 = df["ema9"].iloc[idx]
    ema21 = df["ema21"].iloc[idx]
    ema50 = df["ema50"].iloc[idx]

    if close > ema9 > ema21 > ema50:
        score += 0.3
    elif close < ema9 < ema21 < ema50:
        score -= 0.3
    elif close > ema50:
        score += 0.1
    elif close < ema50:
        score -= 0.1

    # MACD
    macd_hist = df["macd_hist"].iloc[idx]
    macd_hist_prev = df["macd_hist"].iloc[idx - 1] if idx > 0 else 0
    if macd_hist > 0 and macd_hist > macd_hist_prev:
        score += 0.25
    elif macd_hist < 0 and macd_hist < macd_hist_prev:
        score -= 0.25
    elif macd_hist > 0:
        score += 0.1
    elif macd_hist < 0:
        score -= 0.1

    # RSI momentum
    rsi_val = df["rsi"].iloc[idx]
    if 50 < rsi_val < 70:
        score += 0.15
    elif 30 < rsi_val < 50:
        score -= 0.15
    elif rsi_val >= 70:
        score += 0.05  # Overbought but still bullish
    elif rsi_val <= 30:
        score -= 0.05

    # Volume confirmation
    vol_ratio = df["vol_ratio"].iloc[idx] if pd.notna(df["vol_ratio"].iloc[idx]) else 1
    if vol_ratio > 1.2:
        score *= 1.2
    elif vol_ratio < 0.5:
        score *= 0.7

    return max(-1, min(1, score))


def get_mean_signal(df: pd.DataFrame, idx: int) -> float:
    """
    Mean reversion signal: +1 = oversold bounce, -1 = overbought short.
    Uses RSI extremes + VWAP distance + BB position.
    """
    if idx < 50:
        return 0

    score = 0

    rsi_val = df["rsi"].iloc[idx]
    close = df["close"].iloc[idx]

    # RSI extremes
    if rsi_val < 30:
        score += 0.35
    elif rsi_val < 40:
        score += 0.15
    elif rsi_val > 70:
        score -= 0.35
    elif rsi_val > 60:
        score -= 0.15

    # VWAP distance (price far from VWAP = reversion opportunity)
    vwap_dist = df["vwap_dist"].iloc[idx] if pd.notna(df["vwap_dist"].iloc[idx]) else 0
    if close < df["vwap"].iloc[idx]:
        score += min(0.3, vwap_dist / 2)  # Below VWAP = bounce opportunity
    else:
        score -= min(0.3, vwap_dist / 2)

    # Bollinger Band position
    bb_lower = df["bb_lower"].iloc[idx]
    bb_upper = df["bb_upper"].iloc[idx]
    bb_mid = df["bb_mid"].iloc[idx]
    if close < bb_lower:
        score += 0.2
    elif close > bb_upper:
        score -= 0.2

    return max(-1, min(1, score))
