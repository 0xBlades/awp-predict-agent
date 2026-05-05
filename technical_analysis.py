import ccxt
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import os
import config
import urllib.request
import json

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

def get_btc_trend():
    """Get BTC 1H trend direction for regime confirmation."""
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=60)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ema50 = ta.ema(df['close'], length=50).iloc[-1]
        sma20 = ta.sma(df['close'], length=20).iloc[-1]
        curr_price = df['close'].iloc[-1]
        
        # Also check recent momentum (last 5 candles)
        recent_closes = df['close'].iloc[-5:]
        momentum = "UP" if recent_closes.iloc[-1] > recent_closes.iloc[0] else "DOWN"
        
        if curr_price > ema50 and curr_price > sma20:
            return "BULLISH"
        elif curr_price < ema50 and curr_price < sma20:
            return "BEARISH"
        else:
            return "NEUTRAL"
    except Exception as e:
        return "UNKNOWN"

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

def get_funding_rate(token="BTC"):
    """Fetch latest funding rate from Binance Futures (real-time sentiment)."""
    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={token}USDT&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if data:
                rate = float(data[0]["fundingRate"])
                return {
                    "rate": rate,
                    "rate_pct": round(rate * 100, 4),
                    "signal": "BULLISH" if rate > 0 else "BEARISH" if rate < 0 else "NEUTRAL"
                }
    except Exception as e:
        pass
    return None

def get_long_short_ratio(token="BTC"):
    """Fetch top traders Long/Short ratio from Binance Futures."""
    try:
        url = f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={token}USDT&period=15m&limit=2"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if len(data) >= 1:
                latest = data[0]
                long_ratio = float(latest["longAccount"])
                short_ratio = float(latest["shortAccount"])
                ratio = float(latest["longShortRatio"])
                # Previous for trend
                prev_ratio = float(data[1]["longShortRatio"]) if len(data) >= 2 else ratio
                change = ratio - prev_ratio
                return {
                    "long_ratio": round(long_ratio * 100, 2),
                    "short_ratio": round(short_ratio * 100, 2),
                    "ratio": round(ratio, 4),
                    "change": round(change, 4),
                    "signal": "BULLISH" if ratio > 1.0 else "BEARISH" if ratio < 1.0 else "NEUTRAL"
                }
    except Exception as e:
        pass
    return None

def get_taker_ratio(token="BTC"):
    """Fetch Taker Buy/Sell Volume Ratio from Binance Futures (buying/selling pressure)."""
    try:
        url = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={token}USDT&period=15m&limit=2"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if len(data) >= 1:
                latest = data[0]
                ratio = float(latest["buySellRatio"])
                buy_vol = float(latest["buyVol"])
                sell_vol = float(latest["sellVol"])
                # Previous for trend
                prev_ratio = float(data[1]["buySellRatio"]) if len(data) >= 2 else ratio
                change = ratio - prev_ratio
                return {
                    "ratio": round(ratio, 4),
                    "buy_vol": round(buy_vol, 2),
                    "sell_vol": round(sell_vol, 2),
                    "change": round(change, 4),
                    "signal": "BULLISH" if ratio > 1.0 else "BEARISH" if ratio < 1.0 else "NEUTRAL"
                }
    except Exception as e:
        pass
    return None



def get_market_quality(token):
    """
    NO TRADE ZONE Filter — detects chop/consolidation/low-volatility.
    Returns dict with quality metrics and a boolean no_trade_zone flag.
    
    Metrics:
    - atr_percentile: 0-100 (how current ATR ranks vs 50-period avg, low = chop)
    - ema_spread_pct: EMA9-EMA21 distance as % of price (narrow = chop)
    - vwap_distance_pct: price distance from VWAP as % (close = consolidation)
    - rsi_neutral: True if RSI is 45-55 (no directional bias)
    - quality_score: 0-100 composite (higher = better trade quality)
    - no_trade_zone: True if market is in chop/consolidation
    - reason: human-readable reason for skip (if applicable)
    """
    try:
        import numpy as np
        exchange = ccxt.binance()
        symbol = f"{token}/USDT"
        
        # Fetch 15M data (50 candles for ATR history comparison)
        ohlcv_15 = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        df_15 = pd.DataFrame(ohlcv_15, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate indicators
        df_15['atr'] = ta.atr(df_15['high'], df_15['low'], df_15['close'], length=14)
        df_15['ema9'] = ta.ema(df_15['close'], length=9)
        df_15['ema21'] = ta.ema(df_15['close'], length=21)
        df_15['rsi'] = ta.rsi(df_15['close'], length=14)
        df_15['vwap'] = (df_15['close'] * df_15['volume']).cumsum() / df_15['volume'].cumsum()
        
        curr = df_15.iloc[-1]
        price = curr['close']
        
        # 1. ATR Percentile (0-100)
        atr_series = df_15['atr'].dropna()
        if len(atr_series) >= 20:
            current_atr = atr_series.iloc[-1]
            atr_avg_50 = atr_series.iloc[-50:].mean() if len(atr_series) >= 50 else atr_series.mean()
            atr_percentile = min(100, max(0, (current_atr / atr_avg_50) * 50)) if atr_avg_50 > 0 else 50
        else:
            atr_percentile = 50
            current_atr = 0
        
        # 2. EMA Spread % (narrow = chop)
        ema9 = curr['ema9']
        ema21 = curr['ema21']
        if pd.notna(ema9) and pd.notna(ema21) and price > 0:
            ema_spread_pct = abs(ema9 - ema21) / price * 100
        else:
            ema_spread_pct = 0
        
        # 3. VWAP Distance % (close = consolidation)
        vwap = curr['vwap']
        if pd.notna(vwap) and vwap > 0:
            vwap_distance_pct = abs(price - vwap) / vwap * 100
        else:
            vwap_distance_pct = 0
        
        # 4. RSI Neutral zone
        rsi = curr['rsi']
        rsi_neutral = pd.notna(rsi) and 45 <= rsi <= 55
        
        # 5. BB Width (squeeze detection)
        bb = ta.bbands(df_15['close'], length=20, std=2)
        df_15 = pd.concat([df_15, bb], axis=1)
        bbu_col = [c for c in bb.columns if 'BBU' in c][0]
        bbl_col = [c for c in bb.columns if 'BBL' in c][0]
        bbm_col = [c for c in bb.columns if 'BBM' in c][0]
        curr = df_15.iloc[-1]  # Re-fetch curr after concat
        bb_width = (curr[bbu_col] - curr[bbl_col]) / curr[bbm_col] if curr[bbm_col] > 0 else 0
        bb_width_avg = ((df_15[bbu_col] - df_15[bbl_col]) / df_15[bbm_col]).iloc[-20:].mean()
        bb_squeeze = bb_width < (bb_width_avg * 0.7) if pd.notna(bb_width_avg) and bb_width_avg > 0 else False
        
        # --- Composite Quality Score (0-100) ---
        score = 100
        
        # ATR penalty: low ATR = chop (stricter)
        if atr_percentile < 25:
            score -= 50  # Very low volatility — likely chop
        elif atr_percentile < 40:
            score -= 30  # Below average — risky
        elif atr_percentile < 55:
            score -= 10  # Slightly below — caution

        # EMA spread penalty: narrow = no momentum (stricter)
        if ema_spread_pct < 0.05:
            score -= 35  # Extremely narrow — no direction
        elif ema_spread_pct < 0.10:
            score -= 25  # Narrow — weak momentum
        elif ema_spread_pct < 0.20:
            score -= 10  # Moderate

        # VWAP distance penalty: price stuck at VWAP
        if vwap_distance_pct < 0.03:
            score -= 25  # Price glued to VWAP — indecision
        elif vwap_distance_pct < 0.08:
            score -= 10  # Close to VWAP

        # RSI neutral penalty
        if rsi_neutral:
            score -= 15

        # BB squeeze penalty
        if bb_squeeze:
            score -= 20
        
        # Volume penalty (low volume = unreliable moves)
        vol_avg = df_15['volume'].iloc[-20:].mean()
        vol_ratio = df_15['volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        if vol_ratio < 0.5:
            score -= 15
        elif vol_ratio < 0.7:
            score -= 5
        
        score = max(0, min(100, score))
        
        # --- TRADE MODE DETERMINATION (3-state) ---
        # TREND: quality >= threshold → normal trend-following strategy
        # LOW_CONFIDENCE: quality < threshold but above dead zone → mean-reversion scalp only
        # NO_TRADE: quality < dead zone → untradeable (data error, extreme chop, etc.)
        dead_zone = getattr(config, 'DEAD_ZONE', 15)
        threshold = getattr(config, 'QUALITY_THRESHOLD', 50)
        
        no_trade = False
        reason = ""
        reasons = []
        
        if atr_percentile < 30:
            reasons.append(f"low ATR ({atr_percentile:.0f}th pctl)")
        if ema_spread_pct < 0.15:
            reasons.append(f"EMA squeeze ({ema_spread_pct:.3f}%)")
        if vwap_distance_pct < 0.05:
            reasons.append(f"price at VWAP ({vwap_distance_pct:.3f}%)")
        if bb_squeeze:
            reasons.append("BB squeeze")
        
        if score < dead_zone:
            trade_mode = "NO_TRADE"
            no_trade = True
            reason = "DEAD ZONE: " + ", ".join(reasons) if reasons else "DEAD ZONE: untradeable market"
        elif score < threshold:
            trade_mode = "LOW_CONFIDENCE"
            no_trade = False  # Allow trading but with restrictions
            reason = "LOW CONFIDENCE: " + ", ".join(reasons) if reasons else "LOW CONFIDENCE: sideways/chop detected"
        else:
            trade_mode = "TREND"
            no_trade = False
            reason = ""
        
        return {
            "atr_percentile": round(atr_percentile, 1),
            "atr_current": round(float(current_atr), 6) if current_atr else 0,
            "ema_spread_pct": round(ema_spread_pct, 4),
            "vwap_distance_pct": round(vwap_distance_pct, 4),
            "rsi_neutral": rsi_neutral,
            "bb_squeeze": bb_squeeze,
            "bb_width": round(float(bb_width), 4) if pd.notna(bb_width) else 0,
            "volume_ratio": round(vol_ratio, 2),
            "quality_score": score,
            "trade_mode": trade_mode,
            "no_trade_zone": no_trade,
            "reason": reason,
        }
    except Exception as e:
        return {
            "atr_percentile": 50,
            "ema_spread_pct": 0.2,
            "vwap_distance_pct": 0.1,
            "rsi_neutral": False,
            "bb_squeeze": False,
            "bb_width": 0,
            "volume_ratio": 1.0,
            "quality_score": 50,
            "trade_mode": "TREND",
            "no_trade_zone": False,
            "reason": f"Quality check error: {e}",
        }
def get_trading_signal(token):
    """
    Semi-Quant Adaptive Signal Engine.
    Computes trend_score vs mean_score, applies M15 bias, normalizes weights,
    and generates a final_score for entry decision.
    
    Returns dict with:
    - final_score: -1.0 to +1.0 (positive = LONG, negative = SHORT)
    - signal: "LONG" / "SHORT" / "NO_TRADE"
    - w_trend, w_mean: normalized weights
    - trend_score, mean_score: raw scores
    - btc_trend: +1 / -1 / 0
    - tp_pct, sl_pct: recommended TP/SL
    - kill_switch: bool (True = DO NOT TRADE)
    - kill_reason: str
    - asset_type: "BTC" / "ALT"
    """
    try:
        import numpy as np
        exchange = ccxt.binance()
        symbol = f"{token}/USDT"
        
        is_btc = token.upper() == "BTC"
        
        # --- 1. Fetch HTF data (H1 for btc_trend) ---
        ohlcv_1h = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_1h['ema50'] = ta.ema(df_1h['close'], length=50)
        df_1h['ema200'] = ta.ema(df_1h['close'], length=200)
        curr_1h = df_1h.iloc[-1]
        
        # btc_trend: +1 (bullish) / -1 (bearish) / 0 (neutral)
        if pd.notna(curr_1h['ema50']) and pd.notna(curr_1h['ema200']):
            btc_trend = 1 if curr_1h['ema50'] > curr_1h['ema200'] else -1
        else:
            btc_trend = 0
        
        # --- 2. Fetch M15 data for signals ---
        ohlcv_15 = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        df_15 = pd.DataFrame(ohlcv_15, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Indicators
        df_15['ema9'] = ta.ema(df_15['close'], length=9)
        df_15['ema21'] = ta.ema(df_15['close'], length=21)
        df_15['rsi'] = ta.rsi(df_15['close'], length=14)
        df_15['atr'] = ta.atr(df_15['high'], df_15['low'], df_15['close'], length=14)
        df_15['vwap'] = (df_15['close'] * df_15['volume']).cumsum() / df_15['volume'].cumsum()
        
        bb = ta.bbands(df_15['close'], length=20, std=2)
        bbu_col = [c for c in bb.columns if 'BBU' in c][0]
        bbl_col = [c for c in bb.columns if 'BBL' in c][0]
        bbm_col = [c for c in bb.columns if 'BBM' in c][0]
        df_15 = pd.concat([df_15, bb], axis=1)
        curr = df_15.iloc[-1]
        price = curr['close']
        
        # --- 3. Feature Calculation ---
        # breadth: market breadth (0-1)
        breadth_val = get_market_breadth()
        breadth = (breadth_val / 100.0) if breadth_val is not None else 0.5
        
        # range_score: ATR compression / BB width → high = sideways
        bb_width = (curr[bbu_col] - curr[bbl_col]) / curr[bbm_col] if curr[bbm_col] > 0 else 0
        bb_width_avg = ((df_15[bbu_col] - df_15[bbl_col]) / df_15[bbm_col]).iloc[-20:].mean()
        if pd.notna(bb_width_avg) and bb_width_avg > 0:
            # Low BB width ratio = compression = sideways
            range_score = 1.0 - min(1.0, bb_width / bb_width_avg)
        else:
            range_score = 0.5
        
        # volatility: ATR normalized (vs 20-period avg)
        atr_series = df_15['atr'].dropna()
        if len(atr_series) >= 20:
            atr_current = atr_series.iloc[-1]
            atr_avg = atr_series.iloc[-20:].mean()
            volatility = min(1.0, atr_current / atr_avg) if atr_avg > 0 else 0.5
        else:
            volatility = 0.5
            atr_current = 0
            atr_avg = 0
        
        # --- KILL SWITCH ---
        kill_switch = False
        kill_reason = ""
        if atr_avg > 0 and atr_current > 0:
            atr_ratio = atr_current / atr_avg
            if atr_ratio < config.KILL_ATR_MIN_RATIO:
                kill_switch = True
                kill_reason = f"ATR too low ({atr_ratio:.2f}x avg) — dead market"
            elif atr_ratio > config.KILL_ATR_MAX_RATIO:
                kill_switch = True
                kill_reason = f"ATR spike ({atr_ratio:.2f}x avg) — news/chaos"
        
        # --- 4. Weight Engine ---
        btc_trend_norm = (btc_trend + 1) / 2.0  # -1→0, 0→0.5, +1→1
        
        trend_score = 0.5 * btc_trend_norm + 0.3 * breadth + 0.2 * (1 - range_score)
        mean_score = 0.5 * range_score + 0.3 * (1 - breadth) + 0.2 * volatility
        
        # M15 Bias (WAJIB — scalping bias ke mean reversion)
        mean_score *= config.SIGNAL_MEAN_BIAS  # 1.2
        trend_score *= config.SIGNAL_TREND_BIAS  # 0.8
        
        # Asset-specific bonus
        if is_btc:
            trend_score += config.SIGNAL_BTC_TREND_BONUS  # +0.1
        else:
            mean_score += config.SIGNAL_ALT_MEAN_BONUS  # +0.1
        
        # Normalize
        total = trend_score + mean_score
        if total > 0:
            w_trend = trend_score / total
            w_mean = mean_score / total
        else:
            w_trend = 0.5
            w_mean = 0.5
        
        # --- 5. Strategy Signals ---
        # Trend Signal: +1 (LONG) / -1 (SHORT) / 0 (no signal)
        above_vwap = price > curr['vwap'] if pd.notna(curr['vwap']) else False
        ema_bullish = pd.notna(curr['ema9']) and pd.notna(curr['ema21']) and curr['ema9'] > curr['ema21']
        ema_bearish = pd.notna(curr['ema9']) and pd.notna(curr['ema21']) and curr['ema9'] < curr['ema21']
        
        if above_vwap and ema_bullish:
            trend_signal = 1.0
        elif not above_vwap and ema_bearish:
            trend_signal = -1.0
        else:
            trend_signal = 0.0
        
        # Mean Reversion Signal: +1 (LONG) / -1 (SHORT) / 0 (no signal)
        bb_lower = curr[bbl_col] if pd.notna(curr[bbl_col]) else 0
        bb_upper = curr[bbu_col] if pd.notna(curr[bbu_col]) else 0
        rsi = curr['rsi'] if pd.notna(curr['rsi']) else 50
        
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        near_bb_lower = price <= bb_lower * 1.01 if bb_lower > 0 else False  # within 1% of lower band
        near_bb_upper = price >= bb_upper * 0.99 if bb_upper > 0 else False  # within 1% of upper band
        
        if near_bb_lower and rsi_oversold:
            mean_signal = 1.0
        elif near_bb_upper and rsi_overbought:
            mean_signal = -1.0
        else:
            mean_signal = 0.0
        
        # --- 6. Final Score ---
        final_score = (trend_signal * w_trend) + (mean_signal * w_mean)
        
        # --- 7. Entry Decision ---
        threshold = config.SIGNAL_ENTRY_THRESHOLD
        if final_score > threshold:
            signal = "LONG"
        elif final_score < -threshold:
            signal = "SHORT"
        else:
            signal = "NO_TRADE"
        
        # --- 8. TP/SL (Scalping Mode) ---
        # Adaptive TP/SL based on volatility
        vol_factor = max(0.5, min(1.5, volatility))
        tp_pct = config.TP_MIN_PCT + (config.TP_MAX_PCT - config.TP_MIN_PCT) * vol_factor
        sl_pct = config.SL_MIN_PCT + (config.SL_MAX_PCT - config.SL_MIN_PCT) * vol_factor
        
        return {
            "final_score": round(final_score, 4),
            "signal": signal,
            "w_trend": round(w_trend, 4),
            "w_mean": round(w_mean, 4),
            "trend_score": round(trend_score, 4),
            "mean_score": round(mean_score, 4),
            "trend_signal": trend_signal,
            "mean_signal": mean_signal,
            "btc_trend": btc_trend,
            "breadth": round(breadth, 4),
            "range_score": round(range_score, 4),
            "volatility": round(volatility, 4),
            "rsi": round(float(rsi), 2),
            "price": round(float(price), 4),
            "tp_pct": round(tp_pct, 4),
            "sl_pct": round(sl_pct, 4),
            "kill_switch": kill_switch,
            "kill_reason": kill_reason,
            "asset_type": "BTC" if is_btc else "ALT",
            "atr_ratio": round(atr_current / atr_avg, 2) if atr_avg > 0 else 1.0,
        }
    except Exception as e:
        return {
            "final_score": 0.0,
            "signal": "NO_TRADE",
            "w_trend": 0.5, "w_mean": 0.5,
            "trend_score": 0.5, "mean_score": 0.5,
            "trend_signal": 0, "mean_signal": 0,
            "btc_trend": 0, "breadth": 0.5, "range_score": 0.5, "volatility": 0.5,
            "rsi": 50, "price": 0, "tp_pct": 0.006, "sl_pct": 0.004,
            "kill_switch": True, "kill_reason": f"Signal engine error: {e}",
            "asset_type": "ALT", "atr_ratio": 1.0,
        }


def fetch_technical_data(token):
    try:
        exchange = ccxt.binance()
        symbol = f"{token}/USDT"
        ohlcv_5m = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=100)
        ohlcv_15 = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        df_5m = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_15 = pd.DataFrame(ohlcv_15, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # === 5M Indicators (Primary Momentum) ===
        # VWAP (Volume Weighted Average Price) - 5M for cross-timeframe confirmation
        df_5m['vwap'] = (df_5m['close'] * df_5m['volume']).cumsum() / df_5m['volume'].cumsum()
        # EMA 9 & 21 for 5M momentum crossover
        df_5m['ema9'] = ta.ema(df_5m['close'], length=9)
        df_5m['ema21'] = ta.ema(df_5m['close'], length=21)
        df_5m['rsi'] = ta.rsi(df_5m['close'], length=14)
        df_5m['atr'] = ta.atr(df_5m['high'], df_5m['low'], df_5m['close'], length=14)
        macd_5m = ta.macd(df_5m['close'], fast=12, slow=26, signal=9)
        df_5m = pd.concat([df_5m, macd_5m], axis=1)
        macd_col = [c for c in df_5m.columns if 'MACD' in c and 'signal' not in c.lower() and 'hist' not in c.lower()][0]
        macd_signal_col = [c for c in df_5m.columns if 'MACDs' in c][0]
        macd_hist_col = [c for c in df_5m.columns if 'MACDh' in c][0]
        stoch_5m = ta.stoch(df_5m['high'], df_5m['low'], df_5m['close'])
        df_5m = pd.concat([df_5m, stoch_5m], axis=1)
        stoch_k_col = [c for c in df_5m.columns if 'STOCHk' in c][0]
        stoch_d_col = [c for c in df_5m.columns if 'STOCHd' in c][0]

        # === 15M Indicators (Structure) ===
        # PAKET A: Scalping Indicators
        df_15['ema9'] = ta.ema(df_15['close'], length=9)
        df_15['ema21'] = ta.ema(df_15['close'], length=21)
        df_15['rsi'] = ta.rsi(df_15['close'], length=14)
        bb = ta.bbands(df_15['close'], length=20, std=2)
        bbu_col = [c for c in bb.columns if 'BBU' in c][0]
        bbl_col = [c for c in bb.columns if 'BBL' in c][0]
        bbm_col = [c for c in bb.columns if 'BBM' in c][0]
        df_15 = pd.concat([df_15, bb], axis=1)
        df_15['atr'] = ta.atr(df_15['high'], df_15['low'], df_15['close'], length=14)
        # BB Width for volatility
        df_15['bb_width'] = (df_15[bbu_col] - df_15[bbl_col]) / df_15[bbm_col]

        # VWAP (Volume Weighted Average Price)
        # Simple VWAP approximation for 15m timeframe
        df_15['vwap'] = (df_15['close'] * df_15['volume']).cumsum() / df_15['volume'].cumsum()

        # === 1H Indicators (Trend Filter) ===
        df_1h['ema50'] = ta.ema(df_1h['close'], length=50)
        df_1h['rsi'] = ta.rsi(df_1h['close'], length=14)

        curr_5m = df_5m.iloc[-1]
        curr_15 = df_15.iloc[-1]
        prev_15 = df_15.iloc[-2]
        curr_1h = df_1h.iloc[-1]

        # Volume analysis (5m vs 20-period avg)
        vol_avg_5m = df_5m['volume'].rolling(20).mean().iloc[-1]
        vol_ratio = curr_5m['volume'] / vol_avg_5m if vol_avg_5m > 0 else 1.0
        vol_spike = vol_ratio > 1.5

        # 5M momentum direction (last 3 candles)
        last3_close = [df_5m.iloc[i]['close'] for i in range(-3, 0)]
        momentum_5m = "BULLISH" if last3_close[-1] > last3_close[0] else "BEARISH" if last3_close[-1] < last3_close[0] else "FLAT"

        # 1H trend direction
        trend_1h = "BULLISH" if curr_1h['close'] > curr_1h['ema50'] else "BEARISH"

        # PAKET A Signal Scoring (15M)
        paket_a_long = (
            (1 if curr_15['close'] > curr_15['vwap'] else 0) +
            (1 if curr_15['ema9'] > curr_15['ema21'] else 0) +
            (1 if curr_15['rsi'] > 50 else 0)
        )
        paket_a_short = (
            (1 if curr_15['close'] < curr_15['vwap'] else 0) +
            (1 if curr_15['ema9'] < curr_15['ema21'] else 0) +
            (1 if curr_15['rsi'] < 50 else 0)
        )

        ts_lines = [
            f"Asset: {token}",
            f"--- 5M (Primary Momentum) ---",
            f"Price: {curr_5m['close']:.4f} | RSI: {curr_5m['rsi']:.2f} | MACD: {curr_5m[macd_col]:.4f} | Signal: {curr_5m[macd_signal_col]:.4f} | Hist: {curr_5m[macd_hist_col]:.4f}",
            f"Stoch K: {curr_5m[stoch_k_col]:.2f} | D: {curr_5m[stoch_d_col]:.2f} | ATR: {curr_5m['atr']:.4f} | VWAP: {curr_5m['vwap']:.4f} | EMA9: {curr_5m['ema9']:.4f} | EMA21: {curr_5m['ema21']:.4f}",
            f"Volume: {curr_5m['volume']:.0f} | Avg(20): {vol_avg_5m:.0f} | Ratio: {vol_ratio:.2f}x {'⚠️ SPIKE' if vol_spike else ''}",
            f"Momentum(3 candle): {momentum_5m}",
            f"--- 15M (Structure) ---",
            f"Price: {curr_15['close']:.4f} | RSI: {curr_15['rsi']:.2f} | EMA9: {curr_15['ema9']:.4f} | EMA21: {curr_15['ema21']:.4f} | ATR: {curr_15['atr']:.4f}",
            f"BB Upper: {curr_15[bbu_col]:.4f} | BB Mid: {curr_15[bbm_col]:.4f} | BB Lower: {curr_15[bbl_col]:.4f} | BB Width: {curr_15['bb_width']:.4f}",
            f"VWAP: {curr_15['vwap']:.4f}",
            f"PAKET A Signal: LONG={paket_a_long}/3 | SHORT={paket_a_short}/3",
            f"--- 1H (Trend Filter) ---",
            f"Price: {curr_1h['close']:.4f} | RSI: {curr_1h['rsi']:.2f} | EMA50: {curr_1h['ema50']:.4f}",
            f"Trend: {trend_1h}",
        ]

        # Real-time sentiment (Binance Futures)
        funding = get_funding_rate(token)
        ls_ratio = get_long_short_ratio(token)
        taker = get_taker_ratio(token)

        has_sentiment = funding or ls_ratio or taker
        if has_sentiment:
            ts_lines.append(f"--- SENTIMENT (Real-time) ---")
            if funding:
                fr_arrow = "↑" if funding['rate'] > 0 else "↓" if funding['rate'] < 0 else "→"
                ts_lines.append(f"Funding Rate: {funding['rate_pct']:+.4f}% {fr_arrow} ({funding['signal']})")
            if ls_ratio:
                ls_arrow = "↑" if ls_ratio['change'] > 0 else "↓" if ls_ratio['change'] < 0 else "→"
                ts_lines.append(f"Long/Short Ratio: {ls_ratio['ratio']:.4f} {ls_arrow} (Long: {ls_ratio['long_ratio']}% | Short: {ls_ratio['short_ratio']}%) [{ls_ratio['signal']}]")
            if taker:
                tk_arrow = "↑" if taker['change'] > 0 else "↓" if taker['change'] < 0 else "→"
                ts_lines.append(f"Taker Buy/Sell: {taker['ratio']:.4f} {tk_arrow} (Buy: {taker['buy_vol']:.1f} BTC | Sell: {taker['sell_vol']:.1f} BTC) [{taker['signal']}]" )

        tech_summary = "\n".join(ts_lines)

        # Klines: 5M last 6 candles (30 min window)
        klines_list = []
        for i in range(-6, 0):
            row = df_5m.iloc[i]
            klines_list.append(f"T-{abs(i)*5}m: O:{row['open']:.4f} H:{row['high']:.4f} L:{row['low']:.4f} C:{row['close']:.4f} V:{row['volume']:.0f}")
        klines_text = "\n".join(klines_list)

        indicators = {
            # 5M primary (Paket A cross-timeframe)
            "close_5m": float(curr_5m['close']),
            "rsi_5m": float(curr_5m['rsi']),
            "macd_5m": float(curr_5m[macd_col]),
            "macd_signal_5m": float(curr_5m[macd_signal_col]),
            "macd_hist_5m": float(curr_5m[macd_hist_col]),
            "stoch_k_5m": float(curr_5m[stoch_k_col]),
            "stoch_d_5m": float(curr_5m[stoch_d_col]),
            "atr_5m": float(curr_5m['atr']),
            "vwap_5m": float(curr_5m['vwap']),
            "ema9_5m": float(curr_5m['ema9']),
            "ema21_5m": float(curr_5m['ema21']),
            "vol_ratio": float(vol_ratio),
            "vol_spike": vol_spike,
            "momentum_5m": momentum_5m,
            # 15M structure (Paket A primary)
            "close_15m": float(curr_15['close']),
            "rsi_15": float(curr_15['rsi']),
            "ema9_15": float(curr_15['ema9']),
            "ema21_15": float(curr_15['ema21']),
            "atr_15": float(curr_15['atr']),
            "bb_upper": float(curr_15[bbu_col]),
            "bb_mid": float(curr_15[bbm_col]),
            "bb_lower": float(curr_15[bbl_col]),
            "bb_width": float(curr_15['bb_width']),
            "vwap": float(curr_15['vwap']),
            # 1H trend
            "rsi_1h": float(curr_1h['rsi']),
            "ema50_1h": float(curr_1h['ema50']),
            "trend_1h": trend_1h,
            # Paket A signal scoring
            "paket_a_long": (
                (1 if curr_15['close'] > curr_15['vwap'] else 0) +
                (1 if curr_15['ema9'] > curr_15['ema21'] else 0) +
                (1 if curr_15['rsi'] > 50 else 0)
            ),
            "paket_a_short": (
                (1 if curr_15['close'] < curr_15['vwap'] else 0) +
                (1 if curr_15['ema9'] < curr_15['ema21'] else 0) +
                (1 if curr_15['rsi'] < 50 else 0)
            ),
            # Sentiment (Binance Futures)
            "funding_rate": float(funding["rate"]) if funding else None,
            "funding_signal": funding["signal"] if funding else None,
            "ls_ratio": float(ls_ratio["ratio"]) if ls_ratio else None,
            "ls_signal": ls_ratio["signal"] if ls_ratio else None,
            "taker_ratio": float(taker["ratio"]) if taker else None,
            "taker_signal": taker["signal"] if taker else None,
        }
        return tech_summary, klines_text, indicators
    except Exception as e:
        return None, None, None
