import ccxt
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import os
import config

def get_market_breadth():
    try:
        coins = ['BTC', 'ETH', 'SOL', 'BNB']
        bullish_count = 0
        exchange = ccxt.binance()
        
        for coin in coins:
            symbol = f"{coin}/USDT"
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=21)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            sma20 = df['close'].rolling(window=20).mean().iloc[-1]
            curr_price = df['close'].iloc[-1]
            if curr_price > sma20:
                bullish_count += 1
        
        return (bullish_count / len(coins)) * 100
    except Exception as e:
        return None

def get_volatility_status():
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=15)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        atr14 = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1]
        
        history = []
        if os.path.exists(config.ATR_HISTORY_FILE):
            import json
            with open(config.ATR_HISTORY_FILE, "r") as f:
                history = json.load(f)
        
        history.append({"ts": datetime.now().isoformat(), "val": atr14})
        history = history[-2880:]
        import json
        with open(config.ATR_HISTORY_FILE, "w") as f:
            json.dump(history, f)
        
        if len(history) < 10:
            return "NORMAL", atr14
            
        avg_atr = sum(h['val'] for h in history) / len(history)
        status = "EKSTREM" if atr14 > (avg_atr * 1.5) else "NORMAL"
        return status, atr14
    except Exception as e:
        return "NORMAL", None

def fetch_technical_data(token):
    try:
        exchange = ccxt.binance()
        symbol = f"{token}/USDT"
        ohlcv_15 = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        df_15 = pd.DataFrame(ohlcv_15, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df_15['sma20'] = ta.sma(df_15['close'], length=20)
        df_15['ema50'] = ta.ema(df_15['close'], length=50)
        df_15['rsi'] = ta.rsi(df_15['close'], length=14)
        bb = ta.bbands(df_15['close'], length=20, std=2)
        bbu_col = [c for c in bb.columns if 'BBU' in c][0]
        bbl_col = [c for c in bb.columns if 'BBL' in c][0]
        df_15 = pd.concat([df_15, bb], axis=1)
        df_15['atr'] = ta.atr(df_15['high'], df_15['low'], df_15['close'], length=14)
        
        df_1h['sma20'] = ta.sma(df_1h['close'], length=20)
        df_1h['ema50'] = ta.ema(df_1h['close'], length=50)
        df_1h['rsi'] = ta.rsi(df_1h['close'], length=14)
        
        curr_15 = df_15.iloc[-1]
        curr_1h = df_1h.iloc[-1]
        
        ts_lines = [
            f"Asset: {token}",
            f"15m - Price: {curr_15['close']:.4f} | RSI: {curr_15['rsi']:.2f} | SMA20: {curr_15['sma20']:.4f} | EMA50: {curr_15['ema50']:.4f} | ATR: {curr_15['atr']:.4f}",
            f"15m - BB Upper: {curr_15[bbu_col]:.4f} | BB Lower: {curr_15[bbl_col]:.4f}",
            f"1H - Price: {curr_1h['close']:.4f} | RSI: {curr_1h['rsi']:.2f} | SMA20: {curr_1h['sma20']:.4f} | EMA50: {curr_1h['ema50']:.4f}"
        ]
        tech_summary = "\n".join(ts_lines)
        
        klines_list = []
        for i in range(-5, 0):
            row = df_15.iloc[i]
            klines_list.append(f"T-{abs(i)*15}m: O:{row['open']:.4f} H:{row['high']:.4f} L:{row['low']:.4f} C:{row['close']:.4f} V:{row['volume']:.0f}")
        klines_text = "\n".join(klines_list)
        
        indicators = {
            "rsi_15": float(curr_15['rsi']),
            "sma20_15": float(curr_15['sma20']),
            "ema50_15": float(curr_15['ema50']),
            "rsi_1h": float(curr_1h['rsi']),
            "sma20_1h": float(curr_1h['sma20']),
            "ema50_1h": float(curr_1h['ema50']),
            "atr_15": float(curr_15['atr'])
        }
        return tech_summary, klines_text, indicators
    except Exception as e:
        return None, None, None
