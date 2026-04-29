import json
import os
from datetime import datetime
import config

def load_memory():
    if os.path.exists(config.MEMORY_FILE):
        try:
            with open(config.MEMORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_memory(memory):
    try:
        with open(config.MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        return f"Save Memory Error: {e}"
    return None

def save_global_lessons(text):
    try:
        with open(config.LESSONS_FILE, "w") as f:
            f.write(text)
    except Exception as e:
        return f"Save Lessons Error: {e}"
    return None

def load_global_lessons():
    if os.path.exists(config.LESSONS_FILE):
        try:
            with open(config.LESSONS_FILE, "r") as f:
                return f.read().strip()
        except:
            return "No global lessons available."
    return "No global lessons available."

def get_losses():
    memory = load_memory()
    return [m for m in memory if m.get("outcome") == "loss"]

def sync_memory(history_out):
    if not history_out:
        return None
    try:
        start_idx = history_out.find('{')
        end_idx = history_out.rfind('}') + 1
        data = json.loads(history_out[start_idx:end_idx])
        predictions = data.get("data", {}).get("predictions", [])
        
        memory = load_memory()
        updated = False
        existing_ids = {m.get("id"): i for i, m in enumerate(memory) if m.get("id")}
        
        for p in predictions:
            p_id = p.get("id")
            if not p_id: continue
            
            payout = p.get("payout_chips")
            status = p.get("order_status")
            direction = p.get("direction", "unknown")
            token = p.get("market_id", "").split('-')[0].upper()
            
            outcome = "pending"
            if status == "filled":
                outcome = "win" if (payout and float(payout) > 0) else "loss"
            
            if p_id in existing_ids:
                idx = existing_ids[p_id]
                if memory[idx].get("outcome") != outcome:
                    memory[idx]["outcome"] = outcome
                    updated = True
            else:
                memory.insert(0, {
                    "id": p_id,
                    "token": token,
                    "prediction": direction,
                    "outcome": outcome,
                    "indicators": {},
                    "timestamp": p.get("created_at", datetime.now().isoformat())
                })
                updated = True
        
        if len(memory) > 500:
            memory = memory[:500]
            updated = True
            
        if updated:
            save_memory(memory)
            return len(memory)
        return None
    except Exception as e:
        return f"Memory Sync Error: {e}"

def get_relevant_memories(token, current_indicators):
    try:
        memory = load_memory()
        token_memories = [m for m in memory if m.get('token') == token]
        if not token_memories:
            return "No relevant memories for this asset."
            
        relevant = []
        curr_rsi = current_indicators.get('rsi_15', 50)
        
        for m in token_memories:
            if m.get('outcome') == 'pending': continue
            m_rsi = m.get('indicators', {}).get('rsi_15', 50)
            # Slightly broader match for RSI, and prioritizing losses for the agent to avoid
            if abs(m_rsi - curr_rsi) < 15:
                outcome = "WIN" if m.get('outcome') == 'win' else "LOSS"
                relevant.append(f"Past {m.get('timestamp')}: RSI {m_rsi:.2f} -> Pred: {m.get('prediction')} -> Outcome: {outcome}")
                
        if not relevant:
            return "No highly similar past scenarios found."
            
        return "\n".join(relevant[-5:])
    except Exception as e:
        return f"Memory Retrieval Error: {e}"
