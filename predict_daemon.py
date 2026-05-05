import subprocess
import re
import time
import os
import sys
import threading
import concurrent.futures
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import technical_analysis as ta_mod
import memory_manager as mem_mod
import llm_pipeline as llm_mod
import ml_engine as ml_mod

# Global ML engine instance
ml_engine = None

file_lock = threading.Lock()
submission_counter_lock = threading.Lock()
successful_submissions = [0]
low_conf_submissions = [0]

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
    if breadth is None: return {"action": "ANALYZE", "regime": "CHOPPY"}
    
    vol_status, current_atr = ta_mod.get_volatility_status()
    
    # Get BTC trend for confirmation
    btc_trend = ta_mod.get_btc_trend()
    
    # --- 6-State Market Regime (State Machine) ---
    # Priority: btc_trend first, then breadth refines the sub-state
    if btc_trend == "BULLISH":
        if breadth >= 55:
            regime = "STRONG_BULL"
        elif breadth >= 30:
            regime = "TRANSITION_BULL"
        else:
            regime = "WEAK_BULL"  # BTC strong, altcoins bleeding
    elif btc_trend == "BEARISH":
        if breadth <= 45:
            regime = "STRONG_BEAR"
        else:
            regime = "WEAK_BEAR"  # BTC bearish but altcoins holding
    else:
        # BTC NEUTRAL — breadth decides
        if breadth < 20:
            regime = "WEAK_BEAR"
        elif breadth > 50:
            regime = "TRANSITION_BULL"
        else:
            regime = "CHOPPY"
    
    log(f"Market Breadth: {breadth:.1f}% | BTC Trend: {btc_trend} | Regime: {regime} | Volatility: {vol_status}")
    
    return {"action": "ANALYZE", "regime": regime, "breadth": breadth, "vol": vol_status}

def process_single_market(market, context_out, regime, max_submissions, strategic_plan=None):
    m_id = market['id']
    token = m_id.split('-')[0].upper()
    log(f"Processing: {m_id}")
    try:
        return _process_single_market_inner(market, context_out, regime, max_submissions, strategic_plan)
    except Exception as e:
        log(f"[FATAL] Exception in process_single_market({m_id}): {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
        return False

def _process_single_market_inner(market, context_out, regime, max_submissions, strategic_plan=None):
    m_id = market['id']
    token = m_id.split('-')[0].upper()
    
    with submission_counter_lock:
        if successful_submissions[0] >= max_submissions:
            log(f"Skipping {m_id}: submission limit ({max_submissions}) already reached")
            return False
    
    # --- LOSS STREAK PROTECTION ---
    if mem_mod.is_asset_on_cooldown(token):
        cooldown_min = mem_mod.get_cooldown_remaining(token)
        log(f"Skipping {token}: on cooldown ({cooldown_min:.0f}min remaining after loss streak)")
        return False
    
    challenge_out = call_cli(f"predict-agent challenge --market {m_id}")
    if not challenge_out:
        log(f"[SKIP] {m_id}: challenge CLI returned empty")
        return False
    try:
        c_start, c_end = challenge_out.find('{'), challenge_out.rfind('}') + 1
        c_data = json.loads(challenge_out[c_start:c_end])
        nonce = c_data.get("data", {}).get("nonce")
        prompt = c_data.get("data", {}).get("challenge")
        if nonce is None:
            log(f"[SKIP] {m_id}: nonce is None")
            return False
    except Exception as e:
        log(f"[SKIP] {m_id}: challenge parse error: {e}")
        return False
    
    tech_summary, klines_text, indicators = ta_mod.fetch_technical_data(token)
    if not tech_summary:
        log(f"[SKIP] {m_id}: tech_summary empty")
        return False
    
    # --- TRADING SIGNAL ENGINE (Semi-Quant Adaptive) ---
    signal_data = ta_mod.get_trading_signal(token)
    log(f"[Signal] {m_id} -> score={signal_data['final_score']:.4f}, signal={signal_data['signal']}, w_trend={signal_data['w_trend']:.2f}, w_mean={signal_data['w_mean']:.2f}, BTC_trend={signal_data['btc_trend']}, kill={signal_data['kill_switch']}")
    
    # Kill switch: ATR too small or spike
    if signal_data['kill_switch']:
        log(f"[SKIP] {m_id}: KILL SWITCH — {signal_data['kill_reason']}")
        return False
    
    # Signal gate: NO_TRADE from signal engine → skip
    if signal_data['signal'] == "NO_TRADE":
        log(f"[SKIP] {m_id}: Signal engine says NO_TRADE (score={signal_data['final_score']:.4f}, threshold={config.SIGNAL_ENTRY_THRESHOLD})")
        return False
    
    # --- PHASE 1: MARKET QUALITY FILTER (3-state trade mode) ---
    quality = ta_mod.get_market_quality(token)
    trade_mode = quality.get("trade_mode", "TREND")
    log(f"[Quality] {m_id} -> score={quality.get('quality_score', '?')}, mode={trade_mode}, ATR%={quality.get('atr_percentile', '?')}, EMA_spread={quality.get('ema_spread_pct', '?')}%, VWAP_dist={quality.get('vwap_distance_pct', '?')}%")
    
    if quality.get("no_trade_zone", False):
        log(f"[SKIP] {m_id}: {trade_mode} — {quality.get('reason', 'untradeable')}")
        return False
    
    # --- LOW_CONFIDENCE MODE: Submission cap check ---
    if trade_mode == "LOW_CONFIDENCE":
        with submission_counter_lock:
            if low_conf_submissions[0] >= config.LOW_CONF_MAX_SUBMISSIONS:
                log(f"[SKIP] {m_id}: LOW_CONFIDENCE submission limit ({config.LOW_CONF_MAX_SUBMISSIONS}) reached")
                return False
        log(f"[LOW-CONF] {m_id}: Entering LOW CONFIDENCE MODE — mean-reversion scalp only, tickets reduced")
    
    # --- XGBOOST ML PRE-SIGNAL FILTER ---
    global ml_engine
    ml_action = "NORMAL"
    ml_bias = None
    ml_result = None
    
    if ml_engine and ml_engine.model_loaded:
        # Build indicator dict for ML feature extraction
        ml_indicators = {
            'rsi': indicators.get('rsi', 50) if indicators else 50,
            'macd_hist_pct': indicators.get('macd_hist_pct', 0) if indicators else 0,
            'ema_spread_pct': quality.get('ema_spread_pct', 0.2),
            'bb_position': quality.get('bb_position', 0.5),
            'bb_width': quality.get('bb_width', 0.03),
            'bb_width_ratio': quality.get('bb_width_ratio', 1.0),
            'atr_pct': quality.get('atr_percentile', 50) / 100,
            'atr_ratio': signal_data.get('atr_ratio', 1.0),
            'vol_ratio': indicators.get('vol_ratio', 1.0) if indicators else 1.0,
            'vwap_dist_pct': quality.get('vwap_distance_pct', 0.1),
            'above_vwap': 1.0 if quality.get('vwap_distance_pct', 0.1) > 0 else 0.0,
            'body_pct': indicators.get('body_pct', 50) if indicators else 50,
            'is_green': 1.0 if indicators and indicators.get('change_pct', 0) > 0 else 0.0,
            'upper_wick': indicators.get('upper_wick', 0.25) if indicators else 0.25,
            'lower_wick': indicators.get('lower_wick', 0.25) if indicators else 0.25,
            'close_vs_ema9': indicators.get('close_vs_ema9', 0) if indicators else 0,
            'close_vs_ema21': indicators.get('close_vs_ema21', 0) if indicators else 0,
            'close_vs_ema50': indicators.get('close_vs_ema50', 0) if indicators else 0,
            'roc_5': indicators.get('roc_5', 0) if indicators else 0,
            'roc_10': indicators.get('roc_10', 0) if indicators else 0,
            'change_pct': indicators.get('change_pct', 0) if indicators else 0,
        }
        ml_result = ml_engine.predict(token, ml_indicators, regime, signal_data, trade_mode)
        ml_action = ml_result['ml_action']
        
        if ml_action == "SKIP_LLM":
            # HIGH ML confidence — skip LLM entirely, use ML signal directly
            ml_direction = ml_result['ml_signal']  # "UP" or "DOWN"
            ml_conf = ml_result['ml_confidence']
            ml_prob = ml_result['ml_prob']
            
            log(f"[ML-SKIP] {m_id}: XGBoost P(UP)={ml_prob:.3f}, signal={ml_direction}, "
                f"conf={ml_conf} — SKIPPING LLM PIPELINE (token savings)")
            
            # Construct minimal reasoning
            reasoning = (f"ML Pre-Signal: {ml_direction} (P={ml_prob:.3f}, {ml_conf} confidence). "
                        f"Regime: {regime}, Signal score: {signal_data['final_score']:.4f}, "
                        f"Trade mode: {trade_mode}")
            
            # Use signal engine TP/SL
            tp_pct = signal_data.get('tp_pct', 0.008)
            sl_pct = signal_data.get('sl_pct', 0.004)
            rr = tp_pct / sl_pct if sl_pct > 0 else 2.0
            
            # Challenge answer (required for submission)
            challenge_answer = f"Based on ML analysis, predicting {ml_direction}"
            
            # Ticket sizing — ML high conf gets 80% of normal
            tickets = int(config.TICKET_HIGH_CONF * 0.8)
            if trade_mode == "LOW_CONFIDENCE":
                tickets = int(tickets * config.LOW_CONF_TICKET_MULT)
            
            # Save prediction
            try:
                with file_lock:
                    entry = mem_mod.save_prediction(token, ml_direction, indicators, regime, outcome="pending")
                    entry_id = entry.get("id") if entry else None
            except Exception as e:
                log(f"[ML-SKIP] save_prediction FAILED: {e}")
                return False
            
            # Submit
            try:
                submit_cmd = ["predict-agent", "submit", "--market", m_id, 
                            "--prediction", ml_direction.lower(), "--tickets", str(tickets),
                            "--reasoning", reasoning, "--challenge-nonce", nonce]
                log(f"[ML-SKIP] Submitting {m_id}: prediction={ml_direction}, tickets={tickets}")
                res = call_cli(submit_cmd)
                if res:
                    log(f"[ML-SKIP] Submission OK: {res[:200]}")
                    with submission_counter_lock:
                        successful_submissions[0] += 1
                    return True
                else:
                    log(f"[ML-SKIP] Submission FAILED for {m_id}")
                    if entry_id:
                        with file_lock:
                            memory = mem_mod.load_memory()
                            memory = [m for m in memory if m.get("id") != entry_id]
                            mem_mod.save_memory(memory)
                    return False
            except Exception as e:
                log(f"[ML-SKIP] Submit exception: {e}")
                return False
        
        elif ml_action == "BIAS_LLM":
            # MEDIUM ML confidence — pass bias to LLM
            ml_bias = f"ML Pre-Signal bias: {ml_result['ml_signal']} (P={ml_result['ml_prob']:.3f})"
            log(f"[ML-BIAS] {m_id}: Passing ML bias to LLM — {ml_bias}")
        else:
            log(f"[ML-NORMAL] {m_id}: ML uncertain (P(UP)={ml_result['ml_prob']:.3f}) — proceeding with full LLM")
    
    # --- ENHANCED MEMORY RETRIEVAL (Multi-Factor) ---
    memories = mem_mod.get_relevant_memories(token, indicators, regime)
    
    # --- FEW-SHOT EXAMPLES from winning trades ---
    few_shot = mem_mod.get_few_shot_examples(token, indicators, regime, n=3)
    
    # --- DYNAMIC MARKET CONTEXT ---
    market_context = mem_mod.get_market_context(token, indicators, regime)
    
    global_lessons = mem_mod.load_global_lessons()
    
    # --- DUAL LLM PIPELINE (Upgraded: Structure + Trigger) ---
    analysis = llm_mod.get_analysis(
        m_id, context_out, prompt, tech_summary, klines_text, indicators, memories, regime, global_lessons,
        few_shot=few_shot, market_context=market_context, strategic_plan=strategic_plan, trade_mode=trade_mode,
        signal_data=signal_data, ml_bias=ml_bias
    )
    
    # Handle new dict return format
    if isinstance(analysis, dict):
        direction = analysis.get("direction")
        reasoning = analysis.get("reasoning")
        challenge_answer = analysis.get("challenge_answer")
        raw_analysis = analysis.get("raw_content")
        entry_zone = analysis.get("entry_zone", "N/A")
        invalidation = analysis.get("invalidation", "N/A")
        expected_rr = analysis.get("expected_rr", 0.0)
        confidence_score = analysis.get("confidence_score", 5)
        setup_grade = analysis.get("setup_grade", "B")
    else:
        # Fallback for old tuple format
        direction, reasoning, challenge_answer, raw_analysis = analysis
        entry_zone, invalidation, expected_rr, confidence_score, setup_grade = "N/A", "N/A", 0.0, 5, "B"
    
    log(f"Challenge: {prompt} | Answer: {challenge_answer} | Entry: {entry_zone} | Inv: {invalidation} | RR: {expected_rr} | Conf: {confidence_score}/10 | Grade: {setup_grade} | Mode: {trade_mode}")
    
    if not direction or not reasoning or not challenge_answer:
        log(f"Layer 1 Analysis failed for {m_id}: {raw_analysis}")
        return False
    
    # --- PHASE 4: RR MINIMUM GATE (adaptive by trade mode) ---
    rr_min = config.LOW_CONF_RR_MIN if trade_mode == "LOW_CONFIDENCE" else 1.5
    if expected_rr > 0 and expected_rr < rr_min:
        log(f"Skipping {m_id}: RR too low ({expected_rr} < {rr_min}) — mode={trade_mode}")
        return False
    
    # --- L1 SELF-VALIDATION (before external L2) ---
    self_val_decision, self_val_reason = llm_mod.self_validate(
        m_id, direction, reasoning, indicators, strategic_plan
    )
    log(f"[Self-Validate] {m_id} -> {self_val_decision}: {self_val_reason}")
    
    if self_val_decision == "FAIL":
        log(f"Skipping {m_id}: L1 self-validation FAILED — {self_val_reason}")
        return False
    
    # --- L2 VALIDATION (external validator) ---
    decision, audit, raw_audit, confidence_l2 = llm_mod.get_validation(m_id, direction, reasoning, tech_summary, regime=regime, trade_mode=trade_mode, signal_data=signal_data)
    log(f"[Final Decision] {m_id} -> {decision} ({confidence_l2}) | Audit: {audit}")
    
    if decision != "APPROVE":
        return False
    
    # --- PHASE 4: WEIGHTED CONFIDENCE GATING ---
    indicators_with_dir = dict(indicators) if indicators else {}
    indicators_with_dir["direction"] = direction
    final_confidence, composite_score = llm_mod.weighted_confidence(
        confidence_score, confidence_l2, quality.get("quality_score", 50),
        expected_rr, indicators_with_dir
    )
    log(f"[Confidence] {m_id} -> L1={confidence_score}/10, L2={confidence_l2}, Quality={quality.get('quality_score', 50)}, RR={expected_rr} => composite={composite_score}% => {final_confidence}")
    
    if final_confidence == "low":
        log(f"Skipping {m_id}: weighted confidence LOW (composite={composite_score}%) — preventing low-quality trade")
        return False
    
    # --- EXPECTED VALUE (EV) GATE ---
    # EV = (win_probability * potential_reward) - ((1-win_probability) * potential_risk)
    # Estimate win probability from composite score
    est_wr = composite_score / 100.0  # composite maps to estimated WR
    if expected_rr > 0:
        # Risk 1 unit, reward = RR units
        ev = (est_wr * expected_rr) - ((1 - est_wr) * 1.0)
        log(f"[EV Gate] {m_id} -> est_WR={est_wr:.1%}, RR={expected_rr}, EV={ev:.3f}")
        if ev < 0:
            log(f"Skipping {m_id}: negative EV ({ev:.3f}) — trade has negative expected value")
            return False
    else:
        ev = 0

    # --- DYNAMIC TICKET SIZING (Setup Grade Based) ---
    # A+ = max tickets, A = high, B = normal, C = skip (shouldn't reach here)
    grade_multipliers = {"A+": 1.3, "A": 1.15, "B": 1.0, "C": 0.6}
    grade_multiplier = grade_multipliers.get(setup_grade, 1.0)
    
    base_tickets = config.TICKET_HIGH_CONF if final_confidence == "high" else config.TICKET_LOW_CONF
    tickets = int(base_tickets * grade_multiplier)
    
    # --- LOW_CONFIDENCE MODE: Additional ticket reduction ---
    if trade_mode == "LOW_CONFIDENCE":
        tickets = int(tickets * config.LOW_CONF_TICKET_MULT)
        log(f"[LOW-CONF] {m_id}: Tickets reduced by {config.LOW_CONF_TICKET_MULT}x for LOW_CONFIDENCE mode")
    
    # --- HTF FILTER: Counter-trend size reduction ---
    btc_trend = signal_data.get('btc_trend', 0)
    if btc_trend != 0:
        is_counter_trend = (btc_trend == 1 and direction == "DOWN") or (btc_trend == -1 and direction == "UP")
        if is_counter_trend:
            tickets = int(tickets * config.SIGNAL_COUNTER_TREND_SIZE)
            log(f"[HTF-FILTER] {m_id}: Counter-trend trade ({direction} vs BTC {'UP' if btc_trend == 1 else 'DOWN'}) — size reduced to {config.SIGNAL_COUNTER_TREND_SIZE}x")
    
    # --- SIGNAL CONFIDENCE: Size based on final_score ---
    abs_score = abs(signal_data.get('final_score', 0))
    if abs_score < config.SIGNAL_LOW_CONF_THRESHOLD:
        tickets = int(tickets * 0.5)
        log(f"[SIGNAL-CONF] {m_id}: Low signal confidence (|{signal_data.get('final_score', 0):.4f}| < {config.SIGNAL_LOW_CONF_THRESHOLD}) — 0.5x size")
    
    tickets = mem_mod.get_adaptive_tickets(token, tickets, final_confidence)
    log(f"Grade tickets for {token}: {tickets} (base={base_tickets}, grade={setup_grade}, mult={grade_multiplier}, conf={final_confidence}, mode={trade_mode})")
    
    # --- BTC RESTRICTION: Only submit when confidence HIGH ---
    if token == "BTC" and final_confidence != "high":
        log(f"Skipping BTC submission: confidence {final_confidence} (requires HIGH)")
        return False
    
    # --- SAVE PREDICTION WITH FULL INDICATORS + REGIME ---
    try:
        with file_lock:
            entry = mem_mod.save_prediction(token, direction, indicators, regime, outcome="pending")
            entry_id = entry.get("id") if entry else None
    except Exception as e:
        log(f"[DEBUG] save_prediction FAILED for {m_id}: {e}")
        return False
    
    # Truncate reasoning but preserve Challenge: line at end
    try:
        max_reasoning = 1500
        challenge_match = re.search(r"Challenge:\s*\S+", reasoning)
        challenge_line = challenge_match.group(0) if challenge_match else ""
        if len(reasoning) > max_reasoning:
            base = re.sub(r"\n*Challenge:\s*\S+.*", "", reasoning).strip()
            base = base[:max_reasoning - len(challenge_line) - 4] + "..."
            reasoning = f"{base}\n\n{challenge_line}" if challenge_line else base
        submit_cmd = ["predict-agent", "submit", "--market", m_id, "--prediction", direction.lower(), "--tickets", str(tickets), "--reasoning", reasoning, "--challenge-nonce", nonce]
        log(f"[DEBUG] Submitting {m_id}: prediction={direction}, tickets={tickets}")
        log(f"[DEBUG] Reasoning preview: {reasoning[:150]}...")
        res = call_cli(submit_cmd)
        log(f"[DEBUG] Submit result for {m_id}: {str(res)[:200] if res else 'None'}")
        if res:
            log(f"Submission result for {m_id}: {res[:200]}...")
            with submission_counter_lock:
                successful_submissions[0] += 1
                if trade_mode == "LOW_CONFIDENCE":
                    low_conf_submissions[0] += 1
        else:
            log(f"Submission failed for {m_id}")
            if entry_id:
                with file_lock:
                    memory = mem_mod.load_memory()
                    memory = [m for m in memory if m.get("id") != entry_id]
                    mem_mod.save_memory(memory)
    except Exception as e:
        log(f"[DEBUG] Submission EXCEPTION for {m_id}: {e}")
        return False
    return True if res else False

def generate_daily_report():
    """Generate and log a daily performance report."""
    wr = mem_mod.get_winrate()
    top_markets = mem_mod.get_top_markets(3)
    
    report = f"\n{'='*50}\n"
    report += f"DAILY REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    report += f"{'='*50}\n"
    report += f"Win Rate: {wr['rate']}% ({wr['wins']}W / {wr['losses']}L / {wr['total']} total)\n"
    
    if top_markets:
        report += f"\nTop Markets by Win Rate:\n"
        for m in top_markets:
            report += f"  {m['token']}: {m['rate']*100:.1f}% ({m['total']} trades)\n"
    
    heatmap = mem_mod.load_heatmap()
    if heatmap:
        report += f"\nAsset Performance:\n"
        for token, stats in sorted(heatmap.items()):
            total = stats.get("wins", 0) + stats.get("losses", 0)
            if total > 0:
                rate = stats.get("wins", 0) / total * 100
                report += f"  {token}: {rate:.0f}% ({stats.get('wins',0)}W/{stats.get('losses',0)}L)\n"
    
    report += f"{'='*50}\n"
    log(report)
    
    # Save report to file
    report_file = os.path.join(config.AGENT_HOME, "daily_reports.txt")
    try:
        with open(report_file, "a") as f:
            f.write(report + "\n")
    except:
        pass
    
    return report

def main():
    global ml_engine
    log("AWP Predict Daemon Started (Modular Version with Auto-Learning + Enhancements v4 — XGBoost ML)")
    
    # Initialize ML Pre-Signal Engine
    try:
        ml_engine = ml_mod.MLPreSignal()
        status = ml_engine.get_status()
        log(f"ML Engine: loaded={status['model_loaded']}, accuracy={status['cv_accuracy']}, "
            f"trained={status['trained_at']}, age={status['model_age_hours']}h")
    except Exception as e:
        log(f"ML Engine FAILED to initialize: {e} — proceeding without ML")
        ml_engine = None
    
    retry_start_time = None
    last_synthesis_time = 0
    last_report_time = 0
    last_cleanup_time = 0
    
    # --- Startup: Clean memory bank of corrupted entries ---
    cleaned = mem_mod.clean_memory_bank()
    if cleaned > 0:
        log(f"Memory bank cleaned: {cleaned} entries with empty indicators removed")
    
    while True:
        try:
            # 1. Sync Memory
            history = call_cli("predict-agent history")
            sync_res = mem_mod.sync_memory(history)
            if sync_res: log(f"Memory synced: {sync_res} entries")
            
            # 2. Cleanup stale pending entries (every 6 hours)
            current_time = time.time()
            if current_time - last_cleanup_time > 21600:
                removed = mem_mod.cleanup_stale_pending()
                if removed > 0:
                    log(f"Cleaned up {removed} stale pending entries")
                last_cleanup_time = current_time

            # 3. Auto-Learning Cycle (Every 12 hours)
            if current_time - last_synthesis_time > 43200:
                log("Initiating Auto-Learning synthesis...")
                losses = mem_mod.get_losses()
                if losses:
                    new_lessons = llm_mod.synthesize_lessons(losses)
                    mem_mod.save_global_lessons(new_lessons)
                    log(f"New lessons synthesized and saved: {new_lessons[:100]}...")
                else:
                    log("No losses found to synthesize.")
                last_synthesis_time = current_time

            # 4. Daily Report (Every 24 hours)
            if current_time - last_report_time > 86400:
                log("Generating daily report...")
                wr = mem_mod.log_winrate_snapshot()
                report = generate_daily_report()
                last_report_time = current_time

            # 5. ML Auto-Retrain (Every 6 hours)
            if ml_engine and current_time - getattr(ml_engine, 'last_train_time', 0) > 21600:
                try:
                    retrain_result = ml_engine.auto_retrain()
                    if retrain_result.get("status") == "trained":
                        log(f"ML model retrained: accuracy={retrain_result.get('metrics', {}).get('cv_accuracy', 'N/A')}")
                except Exception as e:
                    log(f"ML auto-retrain FAILED: {e}")

            # 5. Determine Regime
            regime_info = determine_market_regime()
            regime = regime_info['regime']
            
            if regime_info['action'] == "ANALYZE":
                # 6. Fetch Markets
                markets_out = call_cli("predict-agent context")
                if markets_out:
                    try:
                        m_start, m_end = markets_out.find('{'), markets_out.rfind('}') + 1
                        m_data = json.loads(markets_out[m_start:m_end])
                        
                        agent_data = m_data.get("data", {}).get("agent", {})
                        rem = agent_data.get("timeslot", {}).get("submissions_remaining", "N/A")
                        log(f"Market Context Check -> OK: {m_data.get('ok')}, Rem: {rem}")
                        
                        if m_data.get("ok") and agent_data.get("timeslot", {}).get("submissions_remaining", 0) > 0:
                            markets = m_data.get("data", {}).get("markets", [])
                            rem = agent_data.get("timeslot", {}).get("submissions_remaining", 3)
                            
                            if markets:
                                # --- HEATMAP-BASED MARKET PRIORITIZATION ---
                                markets = mem_mod.prioritize_markets(markets)
                                markets = markets[:rem]
                                log(f"Found {len(markets)} submittable markets (slots: {rem})")
                                context_out = f"Agent: {agent_data.get('address', 'unknown')}"
                                
                                # --- L2 STRATEGIC PLANNER (once per cycle) ---
                                breadth = regime_info.get('breadth', 50)
                                vol_status = regime_info.get('vol', 'NORMAL')
                                btc_trend = ta_mod.get_btc_trend()  # extra call for plan context
                                heatmap = mem_mod.load_heatmap()
                                
                                strategic_plan = llm_mod.get_strategic_plan(
                                    regime=regime,
                                    breadth=breadth,
                                    vol_status=vol_status,
                                    btc_trend=btc_trend,
                                    heatmap=heatmap,
                                    available_markets=markets,
                                    submissions_remaining=rem
                                )
                                
                                if strategic_plan:
                                    log(f"[STRATEGIC-PLAN] Regime: {strategic_plan.get('regime_bias')}, Direction: {strategic_plan.get('overall_direction')}, Assets: {len(strategic_plan.get('asset_rankings', []))}")
                                    # Save plan to file for debugging
                                    plan_path = os.path.join(config.AGENT_HOME, "logs", "strategic_plan.json")
                                    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
                                    with open(plan_path, "w") as f:
                                        json.dump(strategic_plan, f, indent=2)
                                else:
                                    log("[STRATEGIC-PLAN] Failed to generate plan — proceeding with default regime-based approach")
                                
                                with submission_counter_lock:
                                    successful_submissions[0] = 0
                                    low_conf_submissions[0] = 0
                                
                                # --- Sequential processing (avoid signature clash) ---
                                for m in markets:
                                    process_single_market(m, context_out, regime, rem, strategic_plan=strategic_plan)
                                retry_start_time = None
                            else:
                                if retry_start_time is None:
                                    retry_start_time = time.time()
                                
                                elapsed = time.time() - retry_start_time
                                if elapsed < 300:
                                    log(f"No submittable markets found. Retrying... ({int(elapsed)}s / 300s)")
                                    time.sleep(30)
                                    continue
                                else:
                                    log("Retry limit reached for this round. Sleeping until next boundary.")
                                    retry_start_time = None
                        else:
                            retry_start_time = None
                    except Exception as e:
                        log(f"Market Processing Error: {e}")
                        retry_start_time = None
            else:
                log(f"Holding predictions: {regime_info.get('reason', 'regime not ANALYZE')}")
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
