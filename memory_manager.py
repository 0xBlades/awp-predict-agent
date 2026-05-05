import json
import os
from datetime import datetime, timedelta
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


def clean_memory_bank():
    """Remove entries with empty/insufficient indicators — hallucination source."""
    memory = load_memory()
    original_count = len(memory)
    cleaned = []
    for m in memory:
        indicators = m.get("indicators", {})
        # Count non-None/non-empty values
        valid = sum(1 for v in indicators.values() if v is not None and v != "")
        # Keep if: has 3+ valid indicators OR is a local prediction
        if valid >= 3 or m.get("id", "").startswith("local_"):
            cleaned.append(m)
    if len(cleaned) < original_count:
        save_memory(cleaned)
        return original_count - len(cleaned)
    return 0


def cleanup_stale_pending():
    """Remove pending entries older than PENDING_MAX_AGE_HOURS."""
    memory = load_memory()
    cutoff = datetime.now() - timedelta(hours=config.PENDING_MAX_AGE_HOURS)
    cleaned = []
    removed = 0
    for m in memory:
        if m.get("outcome") == "pending":
            try:
                ts = datetime.fromisoformat(m.get("timestamp", ""))
                if ts < cutoff:
                    removed += 1
                    continue
            except:
                removed += 1
                continue
        cleaned.append(m)
    if removed > 0:
        save_memory(cleaned)
    return removed


def sync_memory(history_out):
    if not history_out:
        return None
    try:
        start_idx = history_out.find('{')
        end_idx = history_out.rfind('}') + 1
        # Skip entries without indicators to prevent garbage data
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
                # Skip API-synced entries without indicators (garbage data)
                # Only local predictions from save_prediction() have full indicators
                pass
        
        # Clean entries with empty indicators (hallucination source)
        before_clean = len(memory)
        memory = [m for m in memory if not m.get("id", "").startswith("local_") or 
                  sum(1 for v in m.get("indicators", {}).values() if v is not None and v != "") >= 3]
        if len(memory) < before_clean:
            updated = True

        if len(memory) > config.MEMORY_MAX_ENTRIES:
            memory = memory[:config.MEMORY_MAX_ENTRIES]
            updated = True
            
        if updated:
            save_memory(memory)
            return len(memory)
        return None
    except Exception as e:
        return f"Memory Sync Error: {e}"


def save_prediction(token, prediction, indicators, regime, outcome="pending"):
    """Save a new prediction with full indicators and regime context."""
    memory = load_memory()
    entry = {
        "id": f"local_{int(datetime.now().timestamp())}",
        "token": token,
        "prediction": prediction,
        "outcome": outcome,
        "indicators": {
            "rsi_5m": indicators.get("rsi_5m"),
            "rsi_15": indicators.get("rsi_15"),
            "rsi_1h": indicators.get("rsi_1h"),
            "macd_hist_5m": indicators.get("macd_hist_5m"),
            "vol_ratio": indicators.get("vol_ratio"),
            "vol_spike": indicators.get("vol_spike"),
            "momentum_5m": indicators.get("momentum_5m"),
            "trend_1h": indicators.get("trend_1h"),
            "stoch_k_5m": indicators.get("stoch_k_5m"),
        },
        "regime": regime,
        "timestamp": datetime.now().isoformat()
    }
    memory.insert(0, entry)
    if len(memory) > config.MEMORY_MAX_ENTRIES:
        memory = memory[:config.MEMORY_MAX_ENTRIES]
    save_memory(memory)
    return entry


def update_prediction_outcome(entry_id, outcome):
    """Update outcome for a local prediction entry."""
    memory = load_memory()
    for m in memory:
        if m.get("id") == entry_id:
            m["outcome"] = outcome
            save_memory(memory)
            return True
    return False


# ============================================================
# IMPROVED MEMORY RETRIEVAL (Multi-Factor Matching)
# ============================================================

def get_relevant_memories(token, current_indicators, regime=None):
    """
    Multi-factor memory retrieval:
    1. Same asset (mandatory)
    2. Similar RSI (±15 on 15m)
    3. Same regime (BULL/BEAR)
    4. Similar volume condition (spike/normal)
    5. Same trend direction (1H)
    
    Scores each memory and returns top 5 most relevant.
    """
    try:
        memory = load_memory()
        # Quality filter: skip entries without indicators (hallucination source)
        token_memories = [m for m in memory if m.get('token') == token 
                         and m.get('outcome') != 'pending'
                         and m.get('indicators') and len(m.get('indicators', {})) >= 5]
        
        if not token_memories:
            return "No relevant memories for this asset."
        
        curr_rsi = current_indicators.get('rsi_15', 50)
        curr_vol_spike = current_indicators.get('vol_spike', False)
        curr_trend = current_indicators.get('trend_1h', 'UNKNOWN')
        
        scored = []
        for m in token_memories:
            score = 0
            m_ind = m.get('indicators', {})
            m_rsi = m_ind.get('rsi_15', 50)
            m_vol_spike = m_ind.get('vol_spike', False)
            m_trend = m_ind.get('trend_1h', 'UNKNOWN')
            m_regime = m.get('regime', 'UNKNOWN')
            
            # Factor 1: RSI similarity (0-30 points)
            rsi_diff = abs(m_rsi - curr_rsi)
            if rsi_diff < 5:
                score += 30
            elif rsi_diff < 10:
                score += 20
            elif rsi_diff < 15:
                score += 10
            
            # Factor 2: Regime match (0-25 points)
            if regime and m_regime == regime:
                score += 25
            
            # Factor 3: Volume condition match (0-20 points)
            if m_vol_spike == curr_vol_spike:
                score += 20
            
            # Factor 4: Trend alignment (0-15 points)
            if m_trend == curr_trend:
                score += 15
            
            # Factor 5: Recency bonus (0-10 points)
            try:
                ts = datetime.fromisoformat(m.get('timestamp', ''))
                hours_ago = (datetime.now() - ts).total_seconds() / 3600
                if hours_ago < 6:
                    score += 10
                elif hours_ago < 12:
                    score += 5
            except:
                pass
            
            outcome = "WIN" if m.get('outcome') == 'win' else "LOSS"
            scored.append({
                "score": score,
                "text": f"[{outcome}] RSI:{m_rsi:.0f} | Regime:{m_regime} | Vol:{'SPIKE' if m_vol_spike else 'normal'} | Trend:{m_trend} | Pred:{m.get('prediction')} | {m.get('timestamp', '')[:16]}"
            })
        
        # Sort by score descending, return top 5
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:5]
        
        if not top:
            return "No highly similar past scenarios found."
        
        result = "PAST SCENARIOS (scored by relevance):\n"
        for i, item in enumerate(top, 1):
            result += f"{i}. {item['text']}\n"
        
        return result
    except Exception as e:
        return f"Memory Retrieval Error: {e}"


def get_few_shot_examples(token, current_indicators, regime, n=3):
    """
    Get winning examples for few-shot prompting.
    Returns n winning trades with similar conditions.
    """
    try:
        memory = load_memory()
        # Quality filter: skip entries without indicators
        wins = [m for m in memory if m.get('token') == token 
                and m.get('outcome') == 'win'
                and m.get('indicators') and len(m.get('indicators', {})) >= 5]
        
        if not wins:
            return ""
        
        curr_rsi = current_indicators.get('rsi_15', 50)
        
        # Score wins by relevance
        scored = []
        for m in wins:
            score = 0
            m_ind = m.get('indicators', {})
            m_rsi = m_ind.get('rsi_15', 50)
            m_regime = m.get('regime', 'UNKNOWN')
            
            # RSI similarity
            rsi_diff = abs(m_rsi - curr_rsi)
            if rsi_diff < 10:
                score += 30
            elif rsi_diff < 20:
                score += 15
            
            # Regime match
            if regime and m_regime == regime:
                score += 25
            
            # Outcome is already a win, that's the main filter
            
            scored.append({
                "score": score,
                "rsi": m_rsi,
                "regime": m_regime,
                "prediction": m.get('prediction', 'unknown'),
            })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:n]
        
        if not top:
            return ""
        
        examples = "SUCCESSFUL PAST TRADES (similar conditions):\n"
        for ex in top:
            examples += f"- {token}: RSI {ex['rsi']:.0f}, {ex['regime']} regime → {ex['prediction'].upper()} → WIN\n"
        
        return examples
    except Exception as e:
        return ""


def get_market_context(token, current_indicators, regime):
    """
    Generate dynamic market context for adaptive prompting.
    Returns conditions that affect prompt strategy.
    """
    rsi_15 = current_indicators.get('rsi_15', 50)
    rsi_5m = current_indicators.get('rsi_5m', 50)
    vol_ratio = current_indicators.get('vol_ratio', 1.0)
    vol_spike = current_indicators.get('vol_spike', False)
    momentum = current_indicators.get('momentum_5m', 'FLAT')
    trend_1h = current_indicators.get('trend_1h', 'UNKNOWN')
    stoch_k = current_indicators.get('stoch_k_5m', 50)
    
    conditions = []
    
    # Volatility assessment
    if vol_spike:
        conditions.append("HIGH_VOLATILITY")
    
    # Consolidation detection
    if 45 <= rsi_15 <= 55 and momentum == 'FLAT':
        conditions.append("CONSOLIDATION")
    
    # Overbought/Oversold
    if rsi_15 > 75:
        conditions.append("OVERBOUGHT")
    elif rsi_15 < 25:
        conditions.append("OVERSOLD")
    
    # Trend alignment (6-state regime)
    bull_regimes = ("STRONG_BULL", "WEAK_BULL", "TRANSITION_BULL")
    bear_regimes = ("STRONG_BEAR", "WEAK_BEAR")
    if regime in bull_regimes and trend_1h == "BULLISH":
        conditions.append("TREND_ALIGNED")
    elif regime in bear_regimes and trend_1h == "BEARISH":
        conditions.append("TREND_ALIGNED")
    elif regime != "CHOPPY" and trend_1h != "UNKNOWN":
        if (regime in bull_regimes and trend_1h == "BEARISH") or (regime in bear_regimes and trend_1h == "BULLISH"):
            conditions.append("TREND_CONFLICT")
    
    # Volume analysis
    if vol_ratio < 0.5:
        conditions.append("LOW_VOLUME")
    elif vol_ratio > 2.0:
        conditions.append("VOLUME_SURGE")
    
    # Stochastic extreme
    if stoch_k > 80:
        conditions.append("STOCH_OVERBOUGHT")
    elif stoch_k < 20:
        conditions.append("STOCH_OVERSOLD")
    
    return conditions


# --- WIN RATE TRACKING ---

def get_winrate():
    """Calculate current win rate from memory bank."""
    memory = load_memory()
    wins = len([m for m in memory if m.get("outcome") == "win"])
    losses = len([m for m in memory if m.get("outcome") == "loss"])
    total = wins + losses
    if total == 0:
        return {"wins": 0, "losses": 0, "total": 0, "rate": 0.0}
    return {
        "wins": wins,
        "losses": losses,
        "total": total,
        "rate": round(wins / total * 100, 1)
    }

def log_winrate_snapshot():
    """Save a timestamped win rate snapshot for trend tracking."""
    wr = get_winrate()
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        **wr
    }
    
    history = []
    if os.path.exists(config.WINRATE_FILE):
        try:
            with open(config.WINRATE_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
    
    history.append(snapshot)
    # Keep last 100 snapshots
    history = history[-100:]
    
    try:
        with open(config.WINRATE_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except:
        pass
    
    return wr


# --- MARKET HEATMAP ---

def load_heatmap():
    if os.path.exists(config.HEATMAP_FILE):
        try:
            with open(config.HEATMAP_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_heatmap(heatmap):
    try:
        with open(config.HEATMAP_FILE, "w") as f:
            json.dump(heatmap, f, indent=2)
    except:
        pass

def update_heatmap(token, outcome):
    """Track win/loss per asset for heatmap ranking."""
    heatmap = load_heatmap()
    if token not in heatmap:
        heatmap[token] = {"wins": 0, "losses": 0, "total": 0}
    
    heatmap[token]["total"] += 1
    if outcome == "win":
        heatmap[token]["wins"] += 1
    elif outcome == "loss":
        heatmap[token]["losses"] += 1
    
    save_heatmap(heatmap)
    return heatmap

def get_top_markets(n=3):
    """Return top N markets by win rate (min 5 trades)."""
    heatmap = load_heatmap()
    ranked = []
    for token, stats in heatmap.items():
        total = stats.get("wins", 0) + stats.get("losses", 0)
        if total >= 3:  # Min 3 trades to qualify
            rate = stats.get("wins", 0) / total
            ranked.append({"token": token, "rate": rate, "total": total})
    
    ranked.sort(key=lambda x: x["rate"], reverse=True)
    return ranked[:n]


# ============================================================
# LOSS STREAK PROTECTION
# ============================================================

LOSS_STREAK_THRESHOLD = 3  # Skip asset after N consecutive losses
LOSS_STREAK_COOLDOWN_HOURS = 1  # Cooldown period in hours

def get_loss_streak(token):
    """Get current loss streak for an asset."""
    memory = load_memory()
    token_trades = [m for m in memory if m.get('token') == token and m.get('outcome') in ('win', 'loss')]
    if not token_trades:
        return 0
    
    # Count consecutive losses from most recent
    streak = 0
    for trade in token_trades:
        if trade.get('outcome') == 'loss':
            streak += 1
        else:
            break  # Stop at first win
    return streak

def is_asset_on_cooldown(token):
    """Check if asset is in cooldown period after loss streak."""
    memory = load_memory()
    token_losses = [m for m in memory if m.get('token') == token and m.get('outcome') == 'loss']
    
    if len(token_losses) < LOSS_STREAK_THRESHOLD:
        return False
    
    # Check if the last LOSS_STREAK_THRESHOLD losses happened within cooldown period
    recent_losses = token_losses[:LOSS_STREAK_THRESHOLD]
    try:
        latest_loss_time = datetime.fromisoformat(recent_losses[0].get('timestamp', ''))
        oldest_loss_in_streak = datetime.fromisoformat(recent_losses[-1].get('timestamp', ''))
        
        # All losses in streak should be within cooldown period
        if (datetime.now() - latest_loss_time).total_seconds() < LOSS_STREAK_COOLDOWN_HOURS * 3600:
            return True
    except:
        pass
    
    return False

def get_cooldown_remaining(token):
    """Get remaining cooldown time in minutes for an asset."""
    memory = load_memory()
    token_losses = [m for m in memory if m.get('token') == token and m.get('outcome') == 'loss']
    
    if len(token_losses) < LOSS_STREAK_THRESHOLD:
        return 0
    
    try:
        latest_loss_time = datetime.fromisoformat(token_losses[0].get('timestamp', ''))
        cooldown_end = latest_loss_time + timedelta(hours=LOSS_STREAK_COOLDOWN_HOURS)
        remaining = (cooldown_end - datetime.now()).total_seconds() / 60
        return max(0, remaining)
    except:
        return 0


# ============================================================
# ADAPTIVE TICKET SIZING
# ============================================================

def get_adaptive_tickets(token, base_tickets, confidence):
    """
    Calculate adaptive ticket size based on:
    1. Asset win rate (from heatmap)
    2. Confidence level (high/low)
    3. Loss streak (reduce if on streak)
    
    Returns adjusted ticket count.
    """
    heatmap = load_heatmap()
    asset_stats = heatmap.get(token, {})
    wins = asset_stats.get("wins", 0)
    losses_count = asset_stats.get("losses", 0)
    total = wins + losses_count
    
    # Start with base tickets
    tickets = base_tickets
    
    # Factor 1: Win rate adjustment
    if total >= 5:  # Only adjust if enough data
        win_rate = wins / total
        if win_rate >= 0.6:  # Hot asset (>60% win rate)
            tickets = int(tickets * 1.25)  # +25%
        elif win_rate >= 0.5:  # Decent asset
            tickets = int(tickets * 1.1)  # +10%
        elif win_rate < 0.35:  # Cold asset (<35% win rate)
            tickets = int(tickets * 0.7)  # -30%
        elif win_rate < 0.45:  # Below average
            tickets = int(tickets * 0.85)  # -15%
    
    # Factor 2: Confidence adjustment
    if confidence == "high":
        tickets = int(tickets * 1.15)  # +15% for high confidence
    else:
        tickets = int(tickets * 0.9)  # -10% for low confidence
    
    # Factor 3: Loss streak adjustment
    streak = get_loss_streak(token)
    if streak >= 3:
        tickets = int(tickets * 0.5)  # -50% on loss streak
    elif streak >= 2:
        tickets = int(tickets * 0.75)  # -25% after 2 losses
    
    # Enforce minimum and maximum
    tickets = max(100, min(tickets, 2000))  # Min 100, Max 2000
    
    return tickets


# ============================================================
# MARKET PRIORITIZATION (Heatmap-Based Selection)
# ============================================================

def prioritize_markets(markets):
    """
    Prioritize markets based on historical win rate.
    Markets with higher win rates get processed first.
    Returns sorted list of markets.
    """
    heatmap = load_heatmap()
    
    def market_priority(market):
        token = market['id'].split('-')[0].upper()
        stats = heatmap.get(token, {})
        wins = stats.get("wins", 0)
        losses_count = stats.get("losses", 0)
        total = wins + losses_count
        
        if total < 3:
            return 0.5  # Neutral priority for new assets
        
        return wins / total  # Win rate as priority score
    
    # Sort by priority (highest win rate first)
    return sorted(markets, key=market_priority, reverse=True)
