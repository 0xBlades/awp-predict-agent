"""
AWP Backtest Engine — Data Fetcher
Fetch kline data from Binance API for backtesting.
"""

import requests
import pandas as pd
import time
import os
import json

BINANCE_BASE = "https://api.binance.com/api/v3/klines"

# Symbols we trade
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def fetch_klines(symbol: str, interval: str = "15m", limit: int = 1000,
                 end_time: int = None, cache: bool = True) -> pd.DataFrame:
    """
    Fetch kline data from Binance.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{limit}_{end_time}.parquet")

    # Use cache if fresh (< 1 hour old)
    if cache and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 3600:
            return pd.read_parquet(cache_file)

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if end_time:
        params["endTime"] = end_time

    resp = requests.get(BINANCE_BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore"
    ])

    # Convert types
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if cache:
        df.to_parquet(cache_file, index=False)

    return df


def fetch_multi_symbol(symbol: str, interval: str = "15m",
                       total_candles: int = 5000) -> pd.DataFrame:
    """
    Fetch more than 1000 candles by paginating backwards.
    Binance max per request = 1000, so we loop.
    """
    all_data = []
    remaining = total_candles
    end_time = None

    while remaining > 0:
        batch = min(remaining, 1000)
        df = fetch_klines(symbol, interval, batch, end_time=end_time, cache=True)
        if df.empty:
            break

        all_data.append(df)
        remaining -= len(df)
        # Go further back in time: use oldest timestamp - 1ms
        end_time = int(df["timestamp"].iloc[0].timestamp() * 1000) - 1

        if len(df) < batch:
            break  # No more data
        time.sleep(0.1)  # Rate limit

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return result


def fetch_multi_timeframe(symbol: str, timeframes: list = None) -> dict:
    """
    Fetch multiple timeframes for the same symbol.
    Returns dict of {timeframe: DataFrame}
    """
    if timeframes is None:
        timeframes = ["5m", "15m", "1h"]

    result = {}
    for tf in timeframes:
        result[tf] = fetch_multi_symbol(symbol, tf, total_candles=5000)
        time.sleep(0.2)

    return result


def load_or_fetch_all(symbols: list = None, interval: str = "15m",
                      total_candles: int = 5000) -> dict:
    """
    Load or fetch data for all symbols.
    Returns dict of {symbol: DataFrame}
    """
    if symbols is None:
        symbols = SYMBOLS

    data = {}
    for sym in symbols:
        print(f"  Fetching {sym} {interval} ({total_candles} candles)...")
        data[sym] = fetch_multi_symbol(sym, interval, total_candles)
        print(f"  -> {len(data[sym])} candles loaded")

    return data


if __name__ == "__main__":
    # Quick test
    data = load_or_fetch_all(["BTCUSDT"], "15m", 2000)
    btc = data["BTCUSDT"]
    print(f"\nBTC 15m: {len(btc)} candles")
    print(f"Range: {btc['timestamp'].iloc[0]} to {btc['timestamp'].iloc[-1]}")
    print(btc.tail(3))
