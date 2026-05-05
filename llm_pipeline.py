from openai import OpenAI
import re
import time
import threading
import json
import os
import config

# --- Client (Swiftrouter only) ---
sr_client = OpenAI(api_key=config.SWIFTROUTER_API_KEY, base_url=config.SWIFTROUTER_BASE_URL)

# --- Circuit Breaker State ---
_cb_state = {
    "failures": 0,
    "last_fail": 0,
    "disabled_until": 0,
}
_cb_lock = threading.Lock()


def _is_sr_available():
    """Check if Swiftrouter is available (not in cooldown)."""
    with _cb_lock:
        if _cb_state["disabled_until"] > time.time():
            return False
        return True


def _record_sr_failure():
    """Record a Swiftrouter failure. Disable after threshold."""
    with _cb_lock:
        _cb_state["failures"] += 1
        _cb_state["last_fail"] = time.time()
        if _cb_state["failures"] >= config.CB_FAILURE_THRESHOLD:
            _cb_state["disabled_until"] = time.time() + config.CB_COOLDOWN_SECONDS
            _cb_state["failures"] = 0


def _record_sr_success():
    """Reset Swiftrouter failure counter on success."""
    with _cb_lock:
        _cb_state["failures"] = 0
        _cb_state["disabled_until"] = 0


def _call_swiftrouter(model, messages, temperature=0.2, max_tokens=2000, timeout=30):
    """
    Call Swiftrouter with circuit breaker and 1 retry on 5xx.
    Returns (content, error_str) tuple.
    """
    if not model or not model.strip():
        return None, "Empty model name"
    if not _is_sr_available():
        return None, "Swiftrouter in cooldown (circuit breaker)"

    for attempt in range(2):
        try:
            response = sr_client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens, timeout=timeout
            )
            msg = response.choices[0].message
            content = msg.content
            # Reasoning models may put output in reasoning field
            if not content or len(content.strip()) < 10:
                reasoning = getattr(msg, 'reasoning', None)
                if reasoning and len(reasoning.strip()) > 10:
                    content = reasoning
            if content and len(content.strip()) > 10:
                _record_sr_success()
                return content, None
            if attempt == 0:
                time.sleep(3)
                continue
        except Exception as e:
            err_str = str(e)
            if "502" in err_str or "500" in err_str or "503" in err_str:
                _record_sr_failure()
                if attempt == 0:
                    time.sleep(5)
                    continue
            else:
                _record_sr_failure()
                break

    return None, f"Swiftrouter failed for model={model}"


def _call_with_fallback(model, messages, temperature=0.2, max_tokens=2000, timeout=30):
    """Legacy wrapper — calls Swiftrouter only."""
    return _call_swiftrouter(model, messages, temperature, max_tokens, timeout)


def _extract_challenge_answer(content, challenge_prompt):
    """Extract numeric challenge answer from LLM response. Retry parsing if needed."""
    if not content:
        return None

    # Try standard format: Challenge: <number>
    match = re.search(r"Challenge:\s*(\d[\d,\.]*)", content, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Try: The answer is <number>
    match = re.search(r"(?:answer|result|total)\s*(?:is|:)\s*(\d[\d,\.]*)", content, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Last resort: find the last standalone number in the content
    numbers = re.findall(r"\b(\d[\d,\.]*)\b", content)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def solve_challenge(challenge_prompt):
    """Solve challenge math problem using Minimax M2.5 via Swiftrouter.
    Returns the numeric answer as a string, or None on failure."""
    import logging
    _log = logging.getLogger("predict")
    if not challenge_prompt:
        return None
    
    try:
        messages = [
            {"role": "system", "content": "Reply with ONLY the numeric answer. No explanation."},
            {"role": "user", "content": challenge_prompt}
        ]
        response = sr_client.chat.completions.create(
            model="deepseek-v3.2-exp",
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            timeout=15
        )
        content = response.choices[0].message.content.strip()
        # Extract the last number from the response (likely the final answer)
        num_match = re.findall(r'[\d,]+\.?\d*', content)
        if num_match:
            answer = num_match[-1].replace(",", "")
            _log.info(f"[CHALLENGE-SOLVER] Minimax M2.5 answer: {answer} (raw: {content[:80]})")
            return answer
        _log.info(f"[CHALLENGE-SOLVER] No number found in response: {content[:100]}")
    except Exception as e:
        _log.info(f"[CHALLENGE-SOLVER] Exception: {e}")
    
    return None


def _build_dynamic_conditions(market_context):
    """
    Build dynamic prompt additions based on market conditions.
    Returns extra instructions to inject into the system prompt.
    """
    if not market_context:
        return ""
    
    additions = []
    
    if "CONSOLIDATION" in market_context:
        additions.append("""⚠️ CONSOLIDATION DETECTED:
- Price is range-bound, no clear momentum
- FOCUS on BB squeeze and Stochastic extremes
- Prefer regime direction (BULL→UP, BEAR→DOWN) unless strong reversal signal
- Volume confirmation is CRITICAL in consolidation""")
    
    if "HIGH_VOLATILITY" in market_context:
        additions.append("""⚠️ HIGH VOLATILITY:
- Wider stops, expect larger swings
- Reduce confidence unless multiple confirmations align
- Volume spike must be >2x average to be reliable""")
    
    if "OVERBOUGHT" in market_context:
        additions.append("""⚠️ OVERBOUGHT (RSI > 75):
- In BULL regime: maintain UP but reduce confidence
- In BEAR regime: strong reversal signal, predict UP if RSI > 80""")
    
    if "OVERSOLD" in market_context:
        additions.append("""⚠️ OVERSOLD (RSI < 25):
- In BEAR regime: maintain DOWN but reduce confidence
- In BULL regime: strong reversal signal, predict DOWN if RSI < 20""")
    
    if "TREND_CONFLICT" in market_context:
        additions.append("""⚠️ TREND CONFLICT:
- 1H trend opposes regime direction
- Require STRONG technical confirmation to follow regime
- If no strong signal, follow 1H trend direction instead""")
    
    if "LOW_VOLUME" in market_context:
        additions.append("""ℹ️ LOW VOLUME:
- Moves are less reliable
- Follow regime direction (low volume = drift with trend)
- Do NOT predict reversals on low volume""")
    
    if "VOLUME_SURGE" in market_context:
        additions.append("""🔥 VOLUME SURGE:
- Strong confirmation signal
- If aligned with regime → HIGH confidence
- If opposing regime → potential reversal, use with caution""")
    
    if "STOCH_OVERBOUGHT" in market_context:
        additions.append("ℹ️ Stochastic overbought (K > 80) — momentum may slow")
    
    if "STOCH_OVERSOLD" in market_context:
        additions.append("ℹ️ Stochastic oversold (K < 20) — momentum may reverse")
    
    if "TREND_ALIGNED" in market_context:
        additions.append("✅ TREND ALIGNED — 1H trend supports regime direction. Higher confidence.")
    
    if not additions:
        return ""
    
    return "\n\nDYNAMIC MARKET CONDITIONS:\n" + "\n".join(additions)


def get_analysis(market_id, context_data, challenge_prompt, tech_summary, klines_text, indicators, memories, regime, global_lessons, few_shot="", market_context=None, strategic_plan=None, trade_mode="TREND", signal_data=None, ml_bias=None):
    # Determine regime-forced direction
    if regime == "STRONG_BULL":
        forced_direction = "UP"
        regime_instruction = """The regime is STRONG_BULL (BTC bullish + market breadth healthy).
Default direction: UP for ALL assets (BTC + ALT).
Do NOT predict HOLD — always choose UP or DOWN."""
    elif regime == "WEAK_BULL":
        forced_direction = "UP"
        regime_instruction = """The regime is WEAK_BULL (BTC bullish but altcoins bleeding, breadth <30%).
For BTC: Predict UP with confidence. BTC is leading the rally.
For ALT: Predict UP only if technicals strongly support it (PAKET A LONG 2/3 minimum).
If ALT technicals are weak/mixed → predict DOWN (altcoins are bleeding despite BTC strength).
Do NOT predict HOLD — always choose UP or DOWN."""
    elif regime == "TRANSITION_BULL":
        forced_direction = "UP"
        regime_instruction = """The regime is TRANSITION_BULL (BTC bullish, breadth improving 30-55%).
Default direction: UP. Market is recovering but not fully confirmed.
For BTC: Predict UP.
For ALT: Predict UP if technicals are constructive. Be selective — only top-strength ALTs.
Do NOT predict HOLD — always choose UP or DOWN."""
    elif regime == "STRONG_BEAR":
        forced_direction = "DOWN"
        regime_instruction = """The regime is STRONG_BEAR (BTC bearish + market breadth weak).
Default direction: DOWN for ALL assets.
Do NOT predict HOLD — always choose UP or DOWN."""
    elif regime == "WEAK_BEAR":
        forced_direction = "DOWN"
        regime_instruction = """The regime is WEAK_BEAR (altcoins weak, BTC holding or neutral).
For ALT: Predict DOWN — altcoins are bleeding.
For BTC: Predict DOWN if 1H trend supports it, otherwise follow technicals.
Do NOT predict HOLD — always choose UP or DOWN."""
    elif regime == "CHOPPY":
        forced_direction = None
        regime_instruction = """The regime is CHOPPY (no clear trend, range-bound market).
Follow 15m price action and volume. Mean-reversion bias.
Predict based on technicals: PAKET A scoring + momentum.
Do NOT predict HOLD — always choose UP or DOWN."""
    else:
        forced_direction = None
        regime_instruction = "The regime is undefined. Follow 15m price action and volume. Do NOT predict HOLD."

    # Build dynamic conditions from market context
    dynamic_conditions = _build_dynamic_conditions(market_context)

    system_prompt = f"""You are a Senior Quantitative Trading Analyst for AWP Predict WorkNet.
CURRENT MARKET REGIME: {regime}.

{regime_instruction}

MULTI-TIMEFRAME STRATEGY (15m Prediction Window — STRUCTURE + TRIGGER):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CONTEXT (Structure: 1H + 15M):
  Identify the dominant market structure BEFORE looking for entries.
  - 1H Trend: Price vs EMA50 (bullish/bearish bias)
  - 15M Structure: Higher highs/higher lows (uptrend) OR lower highs/lower lows (downtrend)
  - VWAP position: Institutional flow direction (price above/below VWAP)
  - EMA9/EMA21 alignment: Momentum direction (not just crossover — check spacing)
  
  CRITICAL: Do NOT just count indicators. Assess the QUALITY of the setup:
  - Wide EMA spread = strong momentum = good
  - Narrow EMA spread = weak/no momentum = BAD (skip or reduce confidence)
  - Price near VWAP = indecision = BAD for trend trades

STEP 2 — TRIGGER (Entry: 15M + 5M):
  Wait for a specific entry trigger before predicting:
  - Pullback to EMA9 in uptrend (bounce = long trigger)
  - Rejection from EMA9 in downtrend (fail = short trigger)
  - Break of micro-structure (15M swing high/low)
  - VWAP reclaim (cross above VWAP in uptrend = confirmation)
  - Volume spike (>1.5x avg) confirming direction
  
  DO NOT predict based on "price is above VWAP so it's bullish."
  Instead: "Price pulled back to EMA9, bounced with volume = long trigger."

STEP 3 — RISK ASSESSMENT:
  - Set ENTRY ZONE: the price range where the trade is valid
  - Set INVALIDATION: the price level that invalidates the setup
  - Estimate RR: risk (entry to invalidation) vs reward (entry to next structure)
  - Only take trades with RR >= 1.5

REGIME-SPECIFIC SCORING WEIGHTS:
  STRONG_BULL: Trend-following. VWAP+EMA alignment = 40%, Trigger quality = 30%, Volume = 20%, RSI = 10%
  WEAK_BULL: Selective. For BTC: follow trend. For ALTs: require STRONG trigger + volume.
  TRANSITION_BULL: Cautious long. Only top-strength setups with clear trigger.
  STRONG_BEAR: Trend-following SHORT. VWAP+EMA alignment = 40%, Trigger quality = 30%, Volume = 20%, RSI = 10%
  WEAK_BEAR: Selective short. For ALTs: follow weakness. For BTC: follow technicals.
  CHOPPY: Mean-reversion only. BB + RSI extremes = key signals. VWAP distance = entry.

QUALITY GATE (NON-NEGOTIABLE):
  Before predicting, verify:
  1. EMA spread is not too narrow (< 0.1% of price = NO TRADE)
  2. Price is not stuck at VWAP (< 0.1% distance = NO TRADE)
  3. ATR is not at multi-period low (low volatility = NO TRADE)
  
  If quality gate fails → still predict, but flag confidence as LOW and explain why.

TREND MODE vs CHOP MODE (CRITICAL - choose strategy based on market state):
================================================================
IF regime is STRONG_BULL / STRONG_BEAR / TRANSITION_BULL / WEAK_BULL / WEAK_BEAR:
  -> TREND MODE: Use VWAP + EMA alignment + volume. Focus on continuation.
    Entry: Wait for pullback to EMA9 in trend direction, then enter.
    Do NOT chase breakouts. Pullback = permission to enter.

IF regime is CHOPPY or Quality Score < 60:
  -> CHOP MODE: Do NOT use trend strategy!
    Options: (a) Mean-reversion at BB extremes + RSI extremes, or (b) NO TRADE.
    If you force trend strategy in chop mode -> WR will tank.
    Only predict if BB + RSI give clear mean-reversion signal.

WARNING: Mixing trend strategy in chop market = guaranteed losses.
================================================================

LOW CONFIDENCE MODE (ACTIVE — market quality is poor):
================================================================
TRADE MODE: {trade_mode}

IF trade_mode == "LOW_CONFIDENCE":
  ⚠️ This is a SCALP-ONLY zone. Strategy is completely different from TREND mode.
  
  STRATEGY: MEAN REVERSION at extremes ONLY.
  - LONG: Price near BB Lower + RSI < 30 (oversold) + rejection wick on 5M → predict UP
  - SHORT: Price near BB Upper + RSI > 70 (overbought) + rejection wick on 5M → predict DOWN
  
  ❌ BANNED in LOW_CONFIDENCE mode:
  - Breakout trades
  - Trend continuation
  - Momentum chasing
  - Trading without BB/RSI extreme confirmation
  
  ✅ REQUIREMENTS for entry:
  - RSI must be < 30 or > 70 (NO neutral zone trades)
  - Price must be near BB band (within 10% of BB width from band)
  - Volume confirmation preferred but not mandatory for scalp
  - Quick TP mindset: target next BB band or VWAP (not extended moves)
  
  CONFIDENCE RULES:
  - Max confidence score: 6/10 (never high conviction in chop)
  - Max setup grade: B (no A/A+ in LOW_CONFIDENCE)
  - RR minimum: 1.0 (lower than TREND mode's 1.5)
  
  If NO BB/RSI extreme exists → predict direction with LOW confidence and explain.
  This mode is about extracting small edges from range-bound markets, NOT chasing big moves.
================================================================

{dynamic_conditions}

OUTPUT FORMAT (STRICT — follow exactly):
DIRECTION: [UP or DOWN]
ENTRY_ZONE: [price range, e.g. "83200-83400" or "83250" for exact level]
INVALIDATION: [price level that invalidates the trade]
EXPECTED_RR: [number, e.g. "2.1" — risk:reward ratio]
CONFIDENCE_SCORE: [1-10 integer]
SETUP_GRADE: [A+ / A / B / C]
REASONING: [Fluid market analysis, 300-2000 chars. Mention: (1) Structure assessment, (2) Entry trigger, (3) Risk/reward, (4) Why this setup. Be specific and actionable.]

SETUP GRADING RULES:
- A+ (Best): Clear trend + pullback to EMA9/VWAP + volume spike + RR >= 2.0 + regime aligned
- A (Strong): Clear trend + pullback trigger + RR >= 1.5 + regime aligned
- B (Decent): Trend exists but trigger is weak OR RR is 1.0-1.5 OR slight regime conflict
- C (Skip): Choppy/no trend, no clear trigger, RR < 1.0, or forced by regime only

CRITICAL:
1. Entry zone and invalidation MUST be specific price levels (not vague).
2. Expected RR must be >= 1.5 to proceed (ideally >= 2.0).
3. Confidence score: 1-4 = skip, 5-6 = low confidence, 7-8 = normal, 9-10 = high conviction.
4. Grade C setups → predict but flag as LOW confidence. Grade A+ → flag as HIGH.
5. Reasoning >= 300 characters. Mention asset name, structure, trigger, and RR.
6. You MUST follow the regime-specific rules above.
7. Do NOT solve any math challenges — focus ONLY on market analysis."""

    user_prompt = (
        f"Market: {market_id}\n"
        f"Context: {context_data}\n"
        f"--- GLOBAL STRATEGY GUIDELINES (Lessons Learned) ---\n"
        f"{global_lessons}\n\n"
        f"--- MULTI-TIMEFRAME INDICATORS (5M + 15M + 1H) ---\n"
        f"{tech_summary}\n\n"
        f"--- PAST SIMILAR SCENARIOS (Memory Bank) ---\n"
        f"{memories}\n\n"
    )
    
    # Inject few-shot examples if available
    if few_shot:
        user_prompt += f"--- {few_shot}\n\n"
    
    user_prompt += (
        f"--- RAW KLINE DATA (5M Last 6 candles — 30min window) ---\n"
        f"{klines_text}"
    )

    # Inject strategic plan context if available
    if strategic_plan:
        plan_direction = strategic_plan.get("overall_direction", "N/A")
        plan_bias = strategic_plan.get("regime_bias", "N/A")
        focus = ", ".join(strategic_plan.get("focus_assets", [])[:3])
        user_prompt += f"\n\n--- STRATEGIC PLAN ---\nOverall Direction: {plan_direction} | Regime Bias: {plan_bias} | Focus: {focus}"
    
    # Inject signal engine context if available
    if signal_data:
        sig = signal_data
        user_prompt += (
            f"\n\n--- QUANTITATIVE SIGNAL ENGINE ---"
            f"\nFinal Score: {sig.get('final_score', 0):.4f} (positive=LONG, negative=SHORT)"
            f"\nSignal: {sig.get('signal', 'N/A')} | Weights: trend={sig.get('w_trend', 0.5):.2f}, mean={sig.get('w_mean', 0.5):.2f}"
            f"\nBTC HTF Trend: {'BULLISH' if sig.get('btc_trend') == 1 else 'BEARISH' if sig.get('btc_trend') == -1 else 'NEUTRAL'}"
            f"\nMarket Breadth: {sig.get('breadth', 0.5)*100:.0f}% | Range Score: {sig.get('range_score', 0.5):.2f} | Volatility: {sig.get('volatility', 0.5):.2f}"
            f"\nRecommended Direction: {sig.get('signal', 'N/A')} | TP: {sig.get('tp_pct', 0.006)*100:.2f}% | SL: {sig.get('sl_pct', 0.004)*100:.2f}%"
            f"\n⚠️ The signal engine suggests {sig.get('signal', 'N/A')}. Use this as STRONG guidance for your analysis."
        )
    
    # Inject ML pre-signal bias if available
    if ml_bias:
        user_prompt += f"\n\n--- ML PRE-SIGNAL (XGBoost) ---\n{ml_bias}\nNote: This is a data-driven signal. Consider it as additional confirmation or contrarian indicator."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    content, error = _call_with_fallback(
        model=config.ANALYSIS_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
        timeout=45
    )

    if error or not content:
        return None, None, None, error or "LLM returned empty content" 

    # --- Parse response (Upgraded: Direction + Entry + Invalidation + RR + Confidence) ---
    dir_match = re.search(r"DIRECTION:\s*(UP|DOWN)", content, re.IGNORECASE)
    if dir_match:
        direction = dir_match.group(1).upper()
    else:
        if "DOWN" in content.upper() and "UP" not in content.upper():
            direction = "DOWN"
        elif "UP" in content.upper() and "DOWN" not in content.upper():
            direction = "UP"
        else:
            direction = forced_direction or "DOWN"

    # Extract new structured fields
    entry_match = re.search(r"ENTRY_ZONE:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
    entry_zone = entry_match.group(1).strip() if entry_match else "N/A"

    invalid_match = re.search(r"INVALIDATION:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
    invalidation = invalid_match.group(1).strip() if invalid_match else "N/A"

    rr_match = re.search(r"EXPECTED_RR:\s*([\d.]+)", content, re.IGNORECASE)
    expected_rr = float(rr_match.group(1)) if rr_match else 0.0

    conf_match = re.search(r"CONFIDENCE_SCORE:\s*(\d+)", content, re.IGNORECASE)
    confidence_score = int(conf_match.group(1)) if conf_match else 5

    grade_match = re.search(r"SETUP_GRADE:\s*(A\+|A|B|C)", content, re.IGNORECASE)
    setup_grade = grade_match.group(1).upper() if grade_match else "B"
    # Normalize A+ 
    if setup_grade == "A+" or setup_grade == "APLUS":
        setup_grade = "A+"
    
    # --- LOW_CONFIDENCE MODE: Enforce caps ---
    if trade_mode == "LOW_CONFIDENCE":
        # Cap confidence at 6/10 in choppy markets (never high conviction)
        if confidence_score > 6:
            confidence_score = 6
        # Cap setup grade at B (no A/A+ in low-quality markets)
        if setup_grade in ("A+", "A"):
            setup_grade = "B"

    # Enforce regime direction (override LLM if it violated)
    if forced_direction and direction != forced_direction:
        # Check if exception conditions are met
        rsi_val = indicators.get("rsi_15", 50)
        trend_1h = indicators.get("trend_1h", "UNKNOWN")
        macd_hist = indicators.get("macd_hist_5m", 0)
        vol_spike = indicators.get("vol_spike", False)
        vol_ratio = indicators.get("vol_ratio", 1.0)
        vwap_15 = indicators.get("vwap", 0)
        ema9_15 = indicators.get("ema9_15", 0)
        ema21_15 = indicators.get("ema21_15", 0)
        price = indicators.get("close_15m", 0) or indicators.get("close_5m", 0)
        
        allow_override = False
        
        if regime in ("STRONG_BULL", "WEAK_BULL", "TRANSITION_BULL") and direction == "DOWN":
            # Allow override if 1H trend is BEARISH + technical confirmation
            if trend_1h == "BEARISH":
                conditions_met = sum([
                    rsi_val > 70,
                    price < vwap_15 if vwap_15 and price else False,
                    ema9_15 < ema21_15 if ema9_15 and ema21_15 else False,
                    macd_hist < 0,
                    vol_spike and vol_ratio > 1.5
                ])
                if conditions_met >= 2:
                    allow_override = True
        
        elif regime in ("STRONG_BEAR", "WEAK_BEAR") and direction == "UP":
            # Allow override if 1H trend is BULLISH + technical confirmation
            if trend_1h == "BULLISH":
                conditions_met = sum([
                    rsi_val < 30,
                    price > vwap_15 if vwap_15 and price else False,
                    ema9_15 > ema21_15 if ema9_15 and ema21_15 else False,
                    macd_hist > 0,
                    vol_spike and vol_ratio > 1.5
                ])
                if conditions_met >= 2:
                    allow_override = True
        
        if not allow_override:
            direction = forced_direction

    # Extract challenge answer with improved parsing
    challenge_answer = _extract_challenge_answer(content, challenge_prompt)

    # If challenge answer is None, retry with explicit prompt
    if not challenge_answer:
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": f"What is the numeric answer to this challenge? Reply with ONLY the number:\n{challenge_prompt}"}
        ]
        retry_content, _ = _call_with_fallback(
            model=config.ANALYSIS_MODEL,
            messages=retry_messages,
            temperature=0.0,
            max_tokens=50,
            timeout=15
        )
        if retry_content:
            challenge_answer = _extract_challenge_answer(retry_content, challenge_prompt)

    # OVERRIDE: Use dedicated challenge solver (Gemini Flash) for higher accuracy
    # LLMs often make arithmetic errors when solving math alongside market analysis
    solved = solve_challenge(challenge_prompt)
    if solved:
        challenge_answer = solved

    reasoning = content.split("REASONING:", 1)[1].strip() if "REASONING:" in content else content
    reasoning = re.sub(r"--challenge\s+\d+", "", reasoning)
    reasoning = re.sub(r"--challenge-nonce\s+[a-zA-Z0-9_]+", "", reasoning)

    if challenge_answer:
        reasoning = re.sub(r"ANSWER:\s*.*$", "", reasoning, flags=re.MULTILINE).strip()
        reasoning = re.sub(r"Challenge:\s*.*$", "", reasoning, flags=re.MULTILINE).strip()
        reasoning = f"{reasoning}\n\nChallenge: {challenge_answer}"

    # --- REASONING TRUNCATION (Kimi K2.5 etc.) ---
    # Preserve Challenge line, truncate base reasoning to 1500 chars max
    max_base = 1500
    challenge_match = re.search(r"Challenge:\s*\S+", reasoning)
    challenge_line = challenge_match.group(0) if challenge_match else ""
    if len(reasoning) > max_base + len(challenge_line) + 4:
        base = re.sub(r"\n*Challenge:\s*\S+.*", "", reasoning).strip()
        base = base[:max_base] + "..."
        reasoning = f"{base}\n\n{challenge_line}" if challenge_line else base

    # Package structured analysis result
    analysis_result = {
        "direction": direction,
        "reasoning": reasoning,
        "challenge_answer": challenge_answer,
        "raw_content": content,
        "entry_zone": entry_zone,
        "invalidation": invalidation,
        "expected_rr": expected_rr,
        "confidence_score": confidence_score,
        "setup_grade": setup_grade,
    }

    return analysis_result


def get_validation(market_id, direction, reasoning, tech_summary, regime="CHOPPY", trade_mode="TREND", signal_data=None):
    # --- DISCIPLINE CHECK: No Speculation ---
    if not tech_summary or len(tech_summary) < 50 or ("Indicators" not in tech_summary and "RSI" not in tech_summary):
        return "HOLD", "Rejected: lack of technical indicators (speculation)", "None", "low"

    system_prompt = """You are the Chief Risk Officer (CRO) and Lead Validator for AWP Predict.
Your job is to audit the analysis provided by the junior analyst.

VALIDATION CRITERIA:
1. Structure Quality: Is there a real setup (EMA alignment, VWAP position, trend structure)?
2. Trigger Validity: Is there a specific entry trigger (pullback, breakout, rejection)?
3. Risk/Reward: Is the expected RR >= 1.5? Is entry zone and invalidation reasonable?
4. Regime Alignment: Follow the regime-specific rules below.
5. Multi-Timeframe Consistency: 1H + 15M + 5M alignment.

REGIME-SPECIFIC DECISION RULES:

STRONG_BULL:
- BTC -> APPROVE (UP). Always.
- ALT -> APPROVE (UP) if technicals constructive. Reject only on clear bearish signal.

WEAK_BULL (BTC bullish, altcoins bleeding):
- BTC -> APPROVE (UP). BTC is the leader.
- ALT -> APPROVE (UP) ONLY if structure is strong (EMA spread wide, VWAP above, clear trigger).
  If ALT technicals weak/mixed -> APPROVE (DOWN) — altcoins are bleeding.

TRANSITION_BULL:
- BTC -> APPROVE (UP).
- ALT -> APPROVE (UP) if top-strength structure + clear trigger. Selective.

STRONG_BEAR:
- ALL assets -> APPROVE (DOWN).

WEAK_BEAR:
- ALT -> APPROVE (DOWN).
- BTC -> APPROVE (DOWN) if 1H trend supports. Otherwise follow technicals.

CHOPPY:
- Mean-reversion only. APPROVE if BB/RSI extremes + clear reversal trigger.
- Only HOLD if reasoning is pure speculation with zero technical basis.

LOW_CONFIDENCE MODE (trade_mode == "LOW_CONFIDENCE"):
- STRICTLY mean-reversion only. NO breakout, NO trend continuation, NO momentum chasing.
- ONLY approve if: RSI < 30 (LONG) or RSI > 70 (SHORT) + price near BB band.
- If reasoning mentions "pullback to EMA", "trend continuation", or "momentum" → HOLD.
- Quick scalp expected: target VWAP or opposite BB band, not extended moves.
- Max confidence: LOW. Always flag LOW confidence in LOW_CONFIDENCE mode.

UNIVERSAL RULES:
- NEVER HOLD for "lack of conviction" if regime rules are followed.
- NEVER HOLD for "UNCERTAIN regime" — all regimes now have clear rules.
- Only HOLD for: factual errors, zero technical basis, or clear regime violation.
- If expected RR < 1.0, recommend HOLD regardless of direction.
- If EMA spread is very narrow (< 0.1% of price), flag LOW confidence.

OUTPUT FORMAT:
DECISION: [APPROVE or HOLD]
CONFIDENCE: [HIGH or LOW]
AUDIT: [1-2 sentence explanation focusing on setup quality and RR]"""

    user_prompt = (
        f"Market: {market_id}\n"
        f"Current Regime: {regime}\n"
        f"Trade Mode: {trade_mode}\n"
        f"Predicted Direction: {direction}\n"
        f"Technical Summary:\n{tech_summary}\n\n"
        f"Junior Analyst Reasoning:\n{reasoning}"
    )
    
    # Inject signal engine context for L2
    if signal_data:
        sig = signal_data
        user_prompt += (
            f"\n\n--- QUANT SIGNAL: score={sig.get('final_score', 0):.4f}, signal={sig.get('signal', 'N/A')}, "
            f"BTC={'BULL' if sig.get('btc_trend') == 1 else 'BEAR' if sig.get('btc_trend') == -1 else 'NEU'}, "
            f"breadth={sig.get('breadth', 0.5)*100:.0f}% ---"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Single validator call via Swiftrouter (no fallback)
    content, error = _call_swiftrouter(
        model=config.VALIDATOR_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=500,
        timeout=30
    )
    if content and len(content.strip()) > 5:
        return _parse_validation(content)

    return "HOLD", f"Validation Error: {error}", "None", "low"


def _parse_validation(content):
    """Parse validation response with confidence extraction."""
    dec_match = re.search(r"DECISION:\s*(APPROVE|HOLD)", content, re.IGNORECASE)
    decision = dec_match.group(1).upper() if dec_match else "HOLD"

    conf_match = re.search(r"CONFIDENCE:\s*(HIGH|LOW)", content, re.IGNORECASE)
    confidence = conf_match.group(1).lower() if conf_match else "low"

    audit = content.split("AUDIT:", 1)[1].strip() if "AUDIT:" in content else content
    return decision, audit, content, confidence


def synthesize_lessons(losses):
    if not losses:
        return "No losses to analyze."

    loss_text = ""
    for l in losses[-20:]:
        loss_text += f"Asset: {l.get('token')} | Pred: {l.get('prediction')} | Indicators: {l.get('indicators')} | Time: {l.get('timestamp')}\n"

    system_prompt = "You are a Quant Trading Auditor. Identify common failure patterns from losses. Provide concise rules (1-3 bullets) in format 'If [Condition] then [Action]'. No filler."
    user_prompt = f"Analyze these losses and synthesize a lesson:\n\n{loss_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    content, error = _call_with_fallback(
        model=config.ANALYSIS_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=500,
        timeout=30
    )

    if content:
        return content.strip()
    return f"Synthesis Error: {error}"


def get_strategic_plan(regime, breadth, vol_status, btc_trend, heatmap, available_markets, submissions_remaining):
    """L2 Strategic Planner — generate a meta-plan across all available markets before individual analysis."""
    import logging
    _log = logging.getLogger("predict")

    market_list = ""
    for m in (available_markets or [])[:10]:
        m_id = m.get("id", "unknown")
        market_list += f"- {m_id}\n"

    heatmap_text = ""
    if heatmap:
        for token, stats in list(heatmap.items())[:10]:
            total = stats.get("wins", 0) + stats.get("losses", 0)
            if total > 0:
                rate = stats.get("wins", 0) / total * 100
                heatmap_text += f"  {token}: {rate:.0f}% ({stats.get('wins',0)}W/{stats.get('losses',0)}L)\n"

    system_prompt = """You are the Chief Strategy Officer for AWP Predict. Generate a strategic plan for the current prediction cycle.
Return a JSON object with these fields:
- regime_bias: One of "STRONG_BULL", "WEAK_BULL", "TRANSITION_BULL", "STRONG_BEAR", "WEAK_BEAR", "CHOPPY"
- overall_direction: "UP" or "DOWN" (default direction for this regime)
- asset_rankings: array of {token, bias, confidence} objects ranked by opportunity
- risk_notes: 1-2 sentence risk summary
- focus_assets: top 3 tokens to prioritize

REGIME DIRECTION RULES:
- STRONG_BULL -> overall_direction: "UP" (all assets)
- WEAK_BULL -> overall_direction: "UP" for BTC, "DOWN" for weak ALTs
- TRANSITION_BULL -> overall_direction: "UP" (selective ALTs)
- STRONG_BEAR -> overall_direction: "DOWN" (all assets)
- WEAK_BEAR -> overall_direction: "DOWN" for ALTs, follow BTC technicals
- CHOPPY -> overall_direction: follow best technical setup (mean-reversion)

Reply with ONLY valid JSON. No markdown, no explanation."""

    user_prompt = f"""Regime: {regime} | Breadth: {breadth:.1f}% | BTC Trend: {btc_trend} | Volatility: {vol_status}
Submissions Remaining: {submissions_remaining}

Available Markets:
{market_list}

Asset Heatmap:
{heatmap_text if heatmap_text else "No heatmap data"}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    content, error = _call_with_fallback(
        model=config.ANALYSIS_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
        timeout=30
    )

    if not content:
        _log.warning(f"[STRATEGIC-PLAN] Failed: {error}")
        return None

    try:
        import json
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group(0))
            return plan
    except Exception as e:
        _log.warning(f"[STRATEGIC-PLAN] Parse error: {e}")

    return None


# ============================================================
# PHASE 4: RR Calculation + Weighted Confidence
# ============================================================

def calculate_rr(entry_zone, invalidation, indicators=None):
    """
    Calculate Risk:Reward ratio from entry zone and invalidation level.
    Uses current price as proxy for target if no explicit target given.
    Returns (rr_ratio, risk_pct, reward_pct) tuple.
    """
    try:
        entry_parts = re.findall(r'[\d.]+', str(entry_zone))
        if not entry_parts:
            return 0.0, 0.0, 0.0
        entry_prices = [float(p) for p in entry_parts]
        entry_mid = sum(entry_prices) / len(entry_prices)

        inv_parts = re.findall(r'[\d.]+', str(invalidation))
        if not inv_parts:
            return 0.0, 0.0, 0.0
        inv_price = float(inv_parts[0])

        risk = abs(entry_mid - inv_price)
        if risk == 0 or entry_mid == 0:
            return 0.0, 0.0, 0.0
        risk_pct = risk / entry_mid * 100

        if indicators:
            bb_upper = indicators.get("bb_upper", 0)
            bb_lower = indicators.get("bb_lower", 0)
            vwap = indicators.get("vwap", 0)

            if indicators.get("direction", "UP") == "UP":
                targets = [t for t in [bb_upper, vwap] if t > entry_mid]
                reward = min(targets) - entry_mid if targets else risk * 2
            else:
                targets = [t for t in [bb_lower, vwap] if t < entry_mid]
                reward = entry_mid - max(targets) if targets else risk * 2
        else:
            reward = risk * 2

        reward_pct = reward / entry_mid * 100 if entry_mid > 0 else 0
        rr = reward / risk if risk > 0 else 0

        return round(rr, 2), round(risk_pct, 3), round(reward_pct, 3)
    except Exception:
        return 0.0, 0.0, 0.0


def weighted_confidence(confidence_score, confidence_l2, quality_score, expected_rr, indicators):
    """
    Calculate weighted confidence from multiple factors:
    - L1 confidence score (1-10)
    - L2 confidence (HIGH/LOW)
    - Market quality score (0-100)
    - Expected RR
    - Technical alignment
    
    Returns ("high"|"low", composite_score)
    """
    l1_norm = min(1.0, max(0.0, confidence_score / 10.0))
    l2_norm = 1.0 if confidence_l2 == "high" else 0.3
    quality_norm = quality_score / 100.0
    rr_norm = min(1.0, expected_rr / 3.0)

    tech_bonus = 0.0
    if indicators:
        trend_1h = indicators.get("trend_1h", "UNKNOWN")
        momentum = indicators.get("momentum_5m", "FLAT")
        direction = indicators.get("direction", "UP")

        if trend_1h != "UNKNOWN" and momentum != "FLAT":
            if (direction == "UP" and trend_1h == "BULLISH" and momentum == "BULLISH") or \
               (direction == "DOWN" and trend_1h == "BEARISH" and momentum == "BEARISH"):
                tech_bonus = 0.15

    composite = (
        l1_norm * 0.25 +
        l2_norm * 0.25 +
        quality_norm * 0.20 +
        rr_norm * 0.15 +
        tech_bonus +
        0.15
    )
    composite = min(1.0, max(0.0, composite))

    final_confidence = "high" if composite >= 0.55 else "low"

    return final_confidence, round(composite * 100, 1)


def self_validate(market_id, direction, reasoning, indicators, strategic_plan=None):
    """L1 Self-Validation — quick sanity check before sending to external L2 validator."""
    if direction not in ("UP", "DOWN"):
        return "FAIL", "Invalid direction"

    paket_a_long = indicators.get("paket_a_long", 0) if indicators else 0
    paket_a_short = indicators.get("paket_a_short", 0) if indicators else 0

    if direction == "UP" and paket_a_short >= 3:
        return "FAIL", "Paket A signal strongly bearish (3/3 SHORT) — conflict"
    if direction == "DOWN" and paket_a_long >= 3:
        return "FAIL", "Paket A signal strongly bullish (3/3 LONG) — conflict"

    if strategic_plan:
        plan_direction = strategic_plan.get("overall_direction")
        if plan_direction and plan_direction != direction:
            rsi_15 = indicators.get("rsi_15", 50) if indicators else 50
            if rsi_15 > 70 or rsi_15 < 30:
                return "PASS", "Deviating from plan (extreme RSI)"
            return "FAIL", f"Plan suggests {plan_direction} but got {direction}"

    return "PASS", "Self-validation passed"
