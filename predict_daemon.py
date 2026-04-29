import subprocess
import time
import os
import sys
import threading
import concurrent.futures
from datetime import datetime
import json

# Add current directory to path for modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import technical_analysis as ta_mod
import memory_manager as mem_mod
import llm_pipeline as llm_mod

file_lock = threading.Lock()

def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg, flush=True)

def call_cli(command):
    if command is None: return None
    env = os.environ.copy()
    env["WALLET_HOME"] = config.WALLET_HOME
    try:
        if isinstance(command, list):
            result = subprocess.run(command, shell=False, capture_output=True, text=True, env=env, timeout=60)
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env, timeout=60)
        if result.returncode != 0:
            log(f"CLI Error: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        log(f"Execution Error: {str(e)}")
        return None

def determine_market_regime():
    log("Checking Market Regime...")
    breadth = ta_mod.get_market_breadth()
    if breadth is None: return {"action": "ANALYZE", "regime": "UNKNOWN"}
    
    vol_status, current_atr = ta_mod.get_volatility_status()
    
    if breadth >= 55:
        regime = "BULL"
    elif breadth < 30:
        regime = "BEAR"
    else:
        regime = "UNCERTAIN"
    
    log(f"Market Breadth: {breadth:.1f}% | Regime: {regime} | Volatility: {vol_status}")
    
    return {"action": "ANALYZE", "regime": regime, "breadth": breadth, "vol": vol_status}

def process_single_market(market, context_out, regime):
    m_id = market['id']
    token = m_id.split('-')[0].upper()
    log(f"Processing: {m_id}")
    
    challenge_out = call_cli(f"predict-agent challenge --market {m_id}")
    if not challenge_out: return False
    try:
        c_start, c_end = challenge_out.find('{'), challenge_out.rfind('}') + 1
        c_data = json.loads(challenge_out[c_start:c_end])
        nonce = c_data.get("data", {}).get("nonce")
        prompt = c_data.get("data", {}).get("challenge")
        if nonce is None: return False
    except:
        return False
    
    tech_summary, klines_text, indicators = ta_mod.fetch_technical_data(token)
    if not tech_summary: return False
    
    memories = mem_mod.get_relevant_memories(token, indicators)
    global_lessons = mem_mod.load_global_lessons()
    
    # --- DUAL LLM PIPELINE ---
    direction, reasoning, challenge_answer, raw_analysis = llm_mod.get_analysis(
        m_id, context_out, prompt, tech_summary, klines_text, indicators, memories, regime, global_lessons
    )
    log(f"Challenge Prompt: {prompt} | AI Answer: {challenge_answer}")
    if not direction or not reasoning or not challenge_answer:
        log(f"Layer 1 Analysis failed for {m_id}: {raw_analysis}")
        return False
    
    decision, audit, raw_audit = llm_mod.get_validation(m_id, direction, reasoning, tech_summary)
    log(f"[Final Decision] {m_id} -> {decision} | Audit: {audit}")
    
    if decision != "APPROVE":
        return False
    
    with file_lock:
        memory = mem_mod.load_memory()
        memory.append({
            "id": None,
            "token": token,
            "indicators": indicators,
            "prediction": direction,
            "outcome": "pending",
            "timestamp": datetime.now().isoformat()
        })
        mem_mod.save_memory(memory)
    
    submit_cmd = ["predict-agent", "submit", "--market", m_id, "--prediction", direction, "--tickets", "1000", "--reasoning", reasoning, "--challenge-nonce", nonce]
    res = call_cli(submit_cmd)
    if res:
        log(f"Submission result for {m_id}: {res[:200]}...")
    else:
        log(f"Submission failed for {m_id}")
    return True if res else False

def main():
    log("AWP Predict Daemon Started (Modular Version with Auto-Learning)")
    retry_start_time = None
    last_synthesis_time = 0
    
    while True:
        try:
            # 1. Sync Memory
            history = call_cli("predict-agent history")
            sync_res = mem_mod.sync_memory(history)
            if sync_res: log(f"Memory synced: {sync_res} entries")
            
            # 2. Auto-Learning Cycle (Every 12 hours)
            current_time = time.time()
            if current_time - last_synthesis_time > 43200: # 12 hours
                log("Initiating Auto-Learning synthesis...")
                losses = mem_mod.get_losses()
                if losses:
                    new_lessons = llm_mod.synthesize_lessons(losses)
                    mem_mod.save_global_lessons(new_lessons)
                    log(f"New lessons synthesized and saved: {new_lessons[:100]}...")
                else:
                    log("No losses found to synthesize.")
                last_synthesis_time = current_time

            # 3. Determine Regime
            regime_info = determine_market_regime()
            regime = regime_info['regime']
            
            if regime_info['action'] == "ANALYZE":
                # 4. Fetch Markets
                markets_out = call_cli("predict-agent context")
                if markets_out:
                    try:
                        m_start, m_end = markets_out.find('{'), markets_out.rfind('}') + 1
                        m_data = json.loads(markets_out[m_start:m_end])
                        
                        # DEBUG LOGGING
                        agent_data = m_data.get("data", {}).get("agent", {})
                        rem = agent_data.get("timeslot", {}).get("submissions_remaining", "N/A")
                        log(f"Market Context Check -> OK: {m_data.get('ok')}, Rem: {rem}")
                        
                        if m_data.get("ok") and agent_data.get("timeslot", {}).get("submissions_remaining", 0) > 0:
                            markets = m_data.get("data", {}).get("markets", [])
                            
                            if markets:
                                log(f"Found {len(markets)} submittable markets")
                                context_out = f"Agent: {agent_data.get('address', 'unknown')}"
                                
                                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                                    futures = [executor.submit(process_single_market, m, context_out, regime) for m in markets]
                                    concurrent.futures.wait(futures)
                                retry_start_time = None # Reset retry after successful attempt
                            else:
                                # --- RETRY LOGIC FOR EMPTY MARKETS ---
                                if retry_start_time is None:
                                    retry_start_time = time.time()
                                
                                elapsed = time.time() - retry_start_time
                                if elapsed < 300: # Retry for 5 minutes
                                    log(f"No submittable markets found. Retrying... ({int(elapsed)}s / 300s)")
                                    time.sleep(30)
                                    continue # Skip boundary sleep and try again
                                else:
                                    log("Retry limit reached for this round. Sleeping until next boundary.")
                                    retry_start_time = None
                        else:
                            retry_start_time = None
                    except Exception as e:
                        log(f"Market Processing Error: {e}")
                        retry_start_time = None
            else:
                log(f"Holding predictions: {regime_info['reason']}")
                retry_start_time = None
                
        except Exception as e:
            log(f"Main Loop Error: {e}")
            retry_start_time = None
            
        next_boundary = ((int(time.time()) // 900) + 1) * 900
        sleep_time = next_boundary - int(time.time())
        log(f"Sleeping for {sleep_time}s until next cycle...")
        time.sleep(max(0, sleep_time))

if __name__ == "__main__":
    main()
