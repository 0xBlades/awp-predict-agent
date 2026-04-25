print("STARTING SCRIPT", flush=True)
import subprocess
import json
import time
import os
import ccxt
import pandas as pd
import pandas_ta as ta
import re
import threading
import concurrent.futures
from openai import OpenAI
from datetime import datetime, timedelta

# --- CONFIGURATION ---
AGENT_HOME = os.environ.get("AGENT_HOME", os.path.expanduser("~/.awp-predict-fresh"))
WALLET_HOME = os.environ.get("WALLET_HOME", AGENT_HOME)
LOG_FILE = os.path.join(AGENT_HOME, "predict_daemon.log")
MEMORY_FILE = os.path.join(AGENT_HOME, "memory_bank.json")
LESSONS_FILE = os.path.join(AGENT_HOME, "global_lessons.txt")
ATR_HISTORY_FILE = os.path.join(AGENT_HOME, "atr_history.json")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "sk-or-unknown")
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
MODEL_NAME = "google/gemma-4-31b-it"

# Lock for shared file access
file_lock = threading.Lock()

def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)

def call_cli(command):
    if command is None:
        log("CLI Error: Command passed to call_cli is None")
        return None
        
    env = os.environ.copy()
    env["HOME"] = WALLET_HOME
    env["WALLET_HOME"] = WALLET_HOME
    try:
        if isinstance(command, list):
            cmd_str = " ".join([str(x) for x in command])
            result = subprocess.run(command, shell=False, capture_output=True, text=True, env=env, timeout=60)
        else:
            cmd_str = str(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env, timeout=60)
            
        if result.returncode != 0:
            log(f"CLI Error executing [{cmd_str}]: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        log(f"Execution Error executing [{cmd_str if 'cmd_str' in locals() else 'unknown'}]: {str(e)}")
        return None

# --- MARKET REGIME SYSTEM ---
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
        
        breadth = (bullish_count / len(coins)) * 100
        return breadth
    except Exception as e:
        log(f"Breadth Error: {e}")
        return None

def get_volatility_status():
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=15)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        atr14 = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1]
        
        with file_lock:
            if os.path.exists(ATR_HISTORY_FILE):
                with open(ATR_HISTORY_FILE, "r") as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append({"ts": datetime.now().isoformat(), "val": atr14})
            history = history[-2880:]
            with open(ATR_HISTORY_FILE, "w") as f:
                json.dump(history, f)
        
        if len(history) < 10:
            return "NORMAL", atr14
            
        avg_atr = sum(h['val'] for h in history) / len(history)
        status = "EKSTREM" if atr14 > (avg_atr * 1.5) else "NORMAL"
        return status, atr14
    except Exception as e:
        log(f"Volatility Error: {e}")
        return "NORMAL", None

def determine_market_regime():
    log("Checking Market Regime...")
    breadth = get_market_breadth()
    if breadth is None: return {"action": "ANALYZE", "regime": "UNKNOWN"}
    
    vol_status, current_atr = get_volatility_status()
    
    if breadth >= 55:
        regime = "BULL"
    elif breadth < 30:
        regime = "BEAR"
    else:
        regime = "UNCERTAIN"
    
    log(f"Market Breadth: {breadth:.1f}% | Regime: {regime} | Volatility: {vol_status}")
    
    if regime == "UNCERTAIN":
        return {"action": "HOLD", "regime": regime, "reason": f"Breadth {breadth:.1f}% is Uncertain (Sideways)."}
    if vol_status == "EKSTREM":
        return {"action": "HOLD", "regime": regime, "reason": "Volatility is Extreme. Risk too high."}
    
    return {"action": "ANALYZE", "regime": regime, "breadth": breadth, "vol": vol_status}

# --- MEMORY BANK (RAG) SYSTEM ---
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        log(f"Save Memory Error: {e}")

def load_global_lessons():
    if os.path.exists(LESSONS_FILE):
        try:
            with open(LESSONS_FILE, "r") as f:
                return f.read().strip()
        except:
            return "No global lessons available."
    return "No global lessons available."

def sync_memory():
    log("Syncing memory bank with AWP history...")
    history_out = call_cli("predict-agent history")
    if not history_out:
        return
    try:
        start_idx = history_out.find('{')
        end_idx = history_out.rfind('}') + 1
        data = json.loads(history_out[start_idx:end_idx])
        predictions = data.get("data", {}).get("predictions", [])
        with file_lock:
            memory = load_memory()
            updated = False
            for p in predictions:
                p_id = p.get("id")
                payout = p.get("payout_chips")
                status = p.get("order_status")
                for entry in memory:
                    if entry.get("id") == p_id and entry.get("outcome") == "pending":
                        if status == "filled":
                            outcome = "win" if (payout and float(payout) > 0) else "loss"
                            entry["outcome"] = outcome
                            updated = True
            if updated:
                save_memory(memory)
                log("Memory bank outcomes updated successfully.")
    except Exception as e:
        log(f"Memory Sync Error: {e}")

def get_relevant_memories(token, current_indicators):
    with file_lock:
        memory = load_memory()
    
    # Filter for completed predictions of the same asset
    past_cases = [m for m in memory if m.get("token") == token and m.get("outcome") != "pending"]
    
    if not past_cases:
        return "No past relevant cases found in memory for this asset."

    # Similarity search: Find cases with most similar indicators
    compare_keys = ["rsi", "ema_diff", "stoch_k", "adx"]
    scored_cases = []

    for m in past_cases:
        past_ind = m.get("indicators", {})
        distance = 0
        count = 0
        for k in compare_keys:
            curr_val = current_indicators.get(k)
            past_val = past_ind.get(k)
            if curr_val is not None and past_val is not None:
                distance += abs(float(curr_val) - float(past_val))
                count += 1
        
        final_score = distance / count if count > 0 else float('inf')
        scored_cases.append((final_score, m))

    # Sort by smallest distance (most similar)
    scored_cases.sort(key=lambda x: x[0])
    
    # Take top 5 most similar
    similar_cases = [case for score, case in scored_cases[:5]]
    
    mem_strings = []
    for m in similar_cases:
        ind = m.get("indicators", {})
        res = "WIN" if m.get("outcome") == "win" else "LOSS"
        mem_strings.append(
            f"- Prediction: {m.get('prediction', 'N/A').upper()}, Outcome: {res}, Indicators: "
            f"EMA_Diff: {ind.get('ema_diff', 'N/A')}, RSI: {ind.get('rsi', 'N/A')}, "
            f"Vol_Ratio: {ind.get('vol_ratio', 'N/A')}x"
        )
    
    return "\n".join(mem_strings) if mem_strings else "No similar patterns found in memory.\n"

def check_panic_mode():
    try:
        history_out = call_cli("predict-agent history")
        if not history_out:
            return False
        start_idx = history_out.find('{')
        end_idx = history_out.rfind('}') + 1
        data = json.loads(history_out[start_idx:end_idx])
        predictions = data.get("data", {}).get("predictions", [])
        if len(predictions) < 3:
            return False
        filled_predictions = [p for p in predictions if p.get("order_status") == "filled"]
        if len(filled_predictions) < 3:
            return False
        last_three = filled_predictions[:3]
        is_loss_streak = all(p.get("payout_chips") == "0" or p.get("payout_chips") is None for p in last_three)
        if is_loss_streak:
            last_loss_time_str = last_three[0].get("created_at")
            if last_loss_time_str:
                last_loss_time = datetime.fromisoformat(last_loss_time_str.replace("Z", "+00:00"))
                if datetime.now().astimezone() < last_loss_time + timedelta(minutes=45):
                    log("🚨 PANIC MODE: 3 consecutive losses detected. Cooling down for 45 mins.")
                    return True
    except Exception as e:
        log(f"Panic Mode Check Error: {e}")
    return False

def fetch_technical_data(token):
    try:
        symbol_map = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "BNB": "BNB/USDT"}
        symbol = symbol_map.get(token.upper(), f"{token.upper()}/USDT")
        exchange = ccxt.binance()
        
        # --- 15m Data ---
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        ema9_val = df['ema9'].iloc[-1]
        ema21_val = df['ema21'].iloc[-1]
        ema50_val = df['ema50'].iloc[-1]
        curr_price = df['close'].iloc[-1]
        
        # Volume Spike (2x avg of last 5)
        curr_vol = df['volume'].iloc[-1]
        avg_vol_5 = df['volume'].tail(6).iloc[:-1].mean()
        vol_spike = curr_vol >= (2 * avg_vol_5)
        
        # Standard RSI 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Stochastic RSI
        rsi_series = 100 - (100 / (1 + rs))
        rsi_min = rsi_series.rolling(window=14).min()
        rsi_max = rsi_series.rolling(window=14).max()
        stoch_k = (rsi_series - rsi_min) / (rsi_max - rsi_min)
        stoch_k_val = stoch_k.iloc[-1] * 100 if not pd.isna(stoch_k.iloc[-1]) else None
        
        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        bb_upper = (sma20 + 2 * std20).iloc[-1]
        bb_mid = sma20.iloc[-1]
        bb_lower = (sma20 - 2 * std20).iloc[-1]
        
        # DMI & ADX (Using pandas_ta)
        dmi = df.ta.adx(length=14)
        adx_val = dmi['ADX_14'].iloc[-1] if dmi is not None else None
        plus_di = dmi['DMP_14'].iloc[-1] if dmi is not None else None
        minus_di = dmi['DMN_14'].iloc[-1] if dmi is not None else None
        
        # ATR
        atr_val = df.ta.atr(length=14).iloc[-1] if df.ta.atr(length=14) is not None else None
        
        # --- 1H Anchor Trend (MTF) ---
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=60)
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ema50_1h = df_1h['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        price_1h = df_1h['close'].iloc[-1]
        trend_1h = "UP" if price_1h > ema50_1h else "DOWN"
        
        # 1H DMI for MTF Filtering
        dmi_1h = df_1h.ta.adx(length=14)
        plus_di_1h = dmi_1h['DMP_14'].iloc[-1] if dmi_1h is not None else None
        minus_di_1h = dmi_1h['DMN_14'].iloc[-1] if dmi_1h is not None else None
        trend_dmi_1h = "UP" if plus_di_1h > minus_di_1h else "DOWN"
        
        # 1H RSI for Momentum check
        delta_1h = df_1h['close'].diff()
        gain_1h = (delta_1h.where(delta_1h > 0, 0)).rolling(window=14).mean()
        loss_1h = (-delta_1h.where(delta_1h < 0, 0)).rolling(window=14).mean()
        rsi_1h = 100 - (100 / (1 + (gain_1h / loss_1h))).iloc[-1]
        
        data_summary = (
            f"Price: {curr_price:.2f} | MTF Trend (1H): {trend_1h} | DMI 1H: {trend_dmi_1h}\n"
            f"EMA 9: {ema9_val:.2f}, EMA 21: {ema21_val:.2f}, EMA 50: {ema50_val:.2f}\n"
            f"RSI: {rsi_val:.2f}, Stoch RSI: {stoch_k_val:.2f}, ADX: {adx_val:.2f}\n"
            f"DI+: {plus_di:.2f}, DI-: {minus_di:.2f}, ATR: {atr_val:.2f}\n"
            f"Vol Spike: {'YES' if vol_spike else 'NO'} | Vol Ratio: {curr_vol/df['volume'].rolling(20).mean().iloc[-1]:.2f}x\n"
            f"BB: Upper={bb_upper:.2f}, Mid={bb_mid:.2f}, Lower={bb_lower:.2f}"
        )
        last_5 = df.tail(5)[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False)
        indicators = {
            "ema_diff": ema9_val - ema21_val,
            "rsi": rsi_val,
            "stoch_k": stoch_k_val,
            "vol_spike": int(vol_spike),
            "trend_1h": trend_1h,
            "price_vs_ema50": curr_price - ema50_val,
            "adx": adx_val,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "atr": atr_val,
            "trend_dmi_1h": trend_dmi_1h,
            "rsi_1h": rsi_1h
        }
        return data_summary, last_5, indicators
    except Exception as e:
        log(f"CCXT Data Error for {token}: {e}")
        return None, None, None
    except Exception as e:
        log(f"CCXT Data Error for {token}: {e}")
        return None, None, None


def get_prediction_and_reasoning(market_id, context_data, challenge_prompt, tech_summary, klines_text, indicators, memories, regime):
    token = market_id.split('-')[0].upper()
    global_lessons = load_global_lessons()
    
    system_prompt = f"""You are a Quantitative Trading Analyst expert for AWP Predict WorkNet. 
CURRENT MARKET REGIME: {regime}. 
You MUST use a strictly data-driven approach. Avoid subjective terms.

STRATEGY GUIDELINES (15m Crypto):
1. MTF Filter: If MTF Trend (1H) is UP, prioritize UP. If DOWN, prioritize DOWN. Never fight the 1H trend.
2. Momentum + Retest: Look for Volume Spikes and RSI (60-75 for LONG). Do not chase breakouts; prefer entry on a retest of the breakout level.
3. EMA 50 Pullback: Price touching/approaching EMA 50 with RSI returning to 50 (neutral) is a high-probability entry signal.
4. Volume: A Volume Spike (YES) indicates strong institutional interest.

REGIME GUIDELINES:
- If BULL: Focus on finding UP signals, be less afraid of minor corrections.
- If BEAR: Be extremely cautious. ONLY seek DOWN signals. NEVER predict UP in a BEAR regime.
- If UNKNOWN: Use standard neutral quantitative analysis.

You MUST output your response in the following format: 
DIRECTION: [UP or DOWN]
REASONING: [Your quantitative analysis here]
Challenge: [Your numeric answer to the challenge]

CRITICAL REQUIREMENTS:
1. Reasoning MUST be at least 300 characters.
2. Formatting: Divide reasoning into these exact sections:
   1. Analisis Trend (MTF): (1H Trend vs 15m EMA 9/21/50 position)
   2. Analisis Momentum: (RSI, Stoch RSI, and Volume Spike detection)
   3. Analisis Volatilitas: (Bollinger Band positions and Vol Ratio)
   4. Evaluasi Strategy: (Check for Retest or EMA 50 Pullback patterns)
   5. Evaluasi Memory: (How you adjusted based on past similar cases provided)
   6. Kesimpulan Kuantitatif: (Final probability calculation and direction)
3. Use the provided technical indicators as the absolute source of truth.
4. MUST learn from the 'Past Similar Scenarios' and 'Global Lessons' to avoid repeating mistakes.
5. The 'Challenge: <number>' line MUST be the very last line of your response. DO NOT output any CLI flags (like --challenge) or commands in your response."""

    user_prompt = (
        f"Market: {market_id}\n"
        f"Context: {context_data}\n"
        f"--- GLOBAL STRATEGY GUIDELINES (Lessons Learned) ---\n{global_lessons}\n\n"
        f"--- TECHNICAL INDICATORS (15m + 1H MTF) ---\n{tech_summary}\n\n"
        f"--- PAST SIMILAR SCENARIOS (Memory Bank) ---\n{memories}\n\n"
        f"--- RAW KLINE DATA (Last 5) ---\n{klines_text}\n\n"
        f"Challenge Prompt: {challenge_prompt}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=200000
        )
        content = response.choices[0].message.content
        dir_match = re.search(r"DIRECTION:\s*(UP|DOWN)", content, re.IGNORECASE)
        direction = dir_match.group(1).lower() if dir_match else "up"
        
        reasoning = content.split("REASONING:", 1)[1].strip() if "REASONING:" in content else content
        challenge_match = re.search(r"Challenge:\s*(\d+)", content, re.IGNORECASE)
        challenge_answer = challenge_match.group(1) if challenge_match else None
        
        return direction, reasoning, challenge_answer
    except Exception as e:
        log(f"LLM Error: {e}")
        return None, None, None




def process_single_market(market, context_out, regime):
    m_id = market['id']
    token = m_id.split('-')[0].upper()
    log(f"Processing: {m_id}")
    
    challenge_out = call_cli(f"predict-agent challenge --market {m_id}")
    if not challenge_out: 
        log(f"Challenge failed for {m_id}")
        return False
    try:
        c_start, c_end = challenge_out.find('{'), challenge_out.rfind('}') + 1
        c_data = json.loads(challenge_out[c_start:c_end])
        nonce = c_data.get("data", {}).get("nonce")
        prompt = c_data.get("data", {}).get("challenge")
        if nonce is None:
            log(f"Nonce is missing for {m_id}, skipping submission.")
            return False
    except:
        log(f"Challenge Parse Error for {m_id}")
        return False
    
    tech_summary, klines_text, indicators = fetch_technical_data(token)
    if not tech_summary: 
        log(f"Technical data missing for {token}")
        return False
    
    memories = get_relevant_memories(token, indicators)
    direction, reasoning, challenge_answer = get_prediction_and_reasoning(m_id, context_out, prompt, tech_summary, klines_text, indicators, memories, regime)
    if not direction or not reasoning or not challenge_answer: 
        log(f"LLM failed to provide prediction or challenge answer for {m_id}")
        return False

    with file_lock:
        memory = load_memory()
        memory.append({
            "id": None,
            "token": token,
            "indicators": indicators,
            "prediction": direction,
            "outcome": "pending",
            "timestamp": datetime.now().isoformat()
        })
        save_memory(memory)

    submit_cmd = [
        "predict-agent", "submit", 
        "--market", m_id, 
        "--prediction", direction, 
        "--tickets", "1000", 
        "--reasoning", re.sub(r'--challenge\s*\d+', '', reasoning).replace('\n', ' ').replace('"', "'").strip(), 
        "--challenge-nonce", nonce
    ]
    submit_out = call_cli(submit_cmd)
    if not submit_out:
        log(f"Submit failed for {m_id}: No output from CLI")
        return False
        
    try:
        # Extract JSON from output
        start_idx = submit_out.find('{')
        end_idx = submit_out.rfind('}') + 1
        if start_idx == -1:
            log(f"Submit failed for {m_id}: No JSON in output")
            return False
            
        res_data = json.loads(submit_out[start_idx:end_idx])
        if res_data.get("ok") is True:
            with file_lock:
                memory = load_memory()
                try:
                    id_match = re.search(r'"id":\s*(\d+)', submit_out)
                    if id_match and memory:
                        for entry in reversed(memory):
                            if entry.get("token") == token and entry.get("outcome") == "pending":
                                entry["id"] = int(id_match.group(1))
                                break
                        save_memory(memory)
                except:
                    pass
            log(f"Successfully submitted {m_id} ({direction})!")
            return True
        else:
            log(f"Submit failed for {m_id}: {res_data.get('user_message', 'Unknown error')}")
            return False
    except Exception as e:
        log(f"Submit JSON Parse Error for {m_id}: {e}. Raw output: {submit_out}")
        return False


def run_cycle():
    sync_memory()
    
    regime_info = determine_market_regime()
    if regime_info["action"] == "HOLD":
        log(f"⚠️ REGIME HOLD: {regime_info['reason']}")
        return
    
    current_regime = regime_info["regime"]
    log(f"Regime is {current_regime}. Proceeding to analysis...")

    log("Starting prediction cycle...")
    max_retries = 12
    retry_delay = 10
    markets = []
    context_out = None
    for attempt in range(max_retries):
        context_out = call_cli("predict-agent context")
        if not context_out:
            time.sleep(retry_delay)
            continue
        try:
            start_idx = context_out.find('{')
            end_idx = context_out.rfind('}') + 1
            data = json.loads(context_out[start_idx:end_idx])
            markets = data.get("data", {}).get("markets", [])
            if markets: break
            time.sleep(retry_delay)
        except:
            time.sleep(retry_delay)

    if not markets: 
        log(f"No submittable markets found. Raw output: {context_out}")
        return

    try:
        submitted_count = 0
        for market in markets:
            if submitted_count >= 3:
                break
            
            # We process sequentially here to easily track submitted_count
            # or we could use a shared counter with the executor.
            # For simplicity and reliability, sequential is fine for 3-10 markets.
            result = process_single_market(market, context_out, current_regime)
            if result:
                submitted_count += 1
    except Exception as e:
        log(f"Cycle Error: {e}")

if __name__ == "__main__":
    log(f"Predict Daemon started with {MODEL_NAME} + Parallel Execution + Regime Filter + RAG Memory Bank using MAIN wallet.")
    while True:
        run_cycle()
        next_boundary = ((int(time.time()) // 900) + 1) * 900
        time.sleep(next_boundary - int(time.time()))