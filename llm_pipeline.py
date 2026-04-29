from openai import OpenAI
import re
import config

analysis_client = OpenAI(api_key=config.SWIFTROUTER_API_KEY, base_url=config.SWIFTROUTER_BASE_URL)
validator_client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)

def get_analysis(market_id, context_data, challenge_prompt, tech_summary, klines_text, indicators, memories, regime, global_lessons):
    system_prompt = f"""You are a Senior Quantitative Trading Analyst for AWP Predict WorkNet. 
CURRENT MARKET REGIME: {regime}. 
Your goal is high-precision prediction. Avoid generic templates and robotic language.

STRATEGY GUIDELINES (15m Crypto):
1. Trend Alignment: Prioritize the 1H Trend (MTF). However, do not blindly follow it. If the 15m chart shows a clear exhaustion (RSI > 80 or < 20) or a structural break, prioritize the reversal.
2. Momentum: Use RSI (60-75 for LONG, 25-40 for SHORT). Extreme values (>80 or <20) suggest an imminent pullback.
3. Volume: Volume Spikes confirm the move. A move without volume is likely a fake-out.
4. Structure: Look for EMA 50 pullbacks in trends, or Upper/Lower BB bounces in ranges.

REGIME GUIDELINES (STRICT DISCIPLINE MODE):
- In a BULL regime, the ONLY permitted direction is UP.
- In a BEAR regime, the ONLY permitted direction is DOWN.
- THE ONLY EXCEPTION (Extreme Reversal):
    - BULL -> DOWN: ONLY if (RSI > 80 AND Price > EMA 50 AND Volume Spike is detected).
    - BEAR -> UP: ONLY if (RSI < 20 AND Price < EMA 50 AND Volume Spike is detected).
- If the conditions for an extreme reversal are not met, you MUST predict the regime direction regardless of other minor indicators.
- If UNKNOWN: Strictly follow the 15m Price Action and Volume.

OUTPUT FORMAT:
DIRECTION: [UP or DOWN]
REASONING: [Write a fluid, professional market analysis. Do NOT use a numbered list or a fixed template. Integrate trend, momentum, and volatility into a cohesive narrative. Ensure the analysis is original and avoids repetitive phrases like 'The 1H trend is UP'.]
Challenge: [Your numeric answer to the challenge]

CRITICAL REQUIREMENTS:
1. Reasoning MUST be at least 300 characters.
2. You MUST explicitly mention the asset name and the predicted direction in the narrative.
3. Use provided indicators as the source of truth, but interpret them like a human trader, not a bot.
4. The 'Challenge: <number>' line MUST be the very last line. No other text after it."""
    
    user_prompt = (
        f"Market: {market_id}\n"
        f"Context: {context_data}\n"
        f"--- GLOBAL STRATEGY GUIDELINES (Lessons Learned) ---\n"
        f"{global_lessons}\n\n"
        f"--- TECHNICAL INDICATORS (15m + 1H MTF) ---\n"
        f"{tech_summary}\n\n"
        f"--- PAST SIMILAR SCENARIOS (Memory Bank) ---\n"
        f"{memories}\n\n"
        f"--- RAW KLINE DATA (Last 5) ---\n"
        f"{klines_text}\n\n"
        f"Challenge Prompt: {challenge_prompt}"
    )
    
    try:
        response = analysis_client.chat.completions.create(
            model=config.ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        content = response.choices[0].message.content
        if content is None:
            return None, None, None, "LLM returned empty content"
            
        dir_match = re.search(r"DIRECTION:\s*(UP|DOWN)", content, re.IGNORECASE)
        if dir_match:
            direction = dir_match.group(1).lower()
        else:
            if "DOWN" in content.upper() and "UP" not in content.upper():
                direction = "down"
            elif "UP" in content.upper() and "DOWN" not in content.upper():
                direction = "up"
            else:
                direction = "up"
        
        challenge_match = re.search(r"Challenge:\s*(\d+)", content, re.IGNORECASE)
        challenge_answer = challenge_match.group(1) if challenge_match else None
        
        reasoning = content.split("REASONING:", 1)[1].strip() if "REASONING:" in content else content
        reasoning = re.sub(r"--challenge\s+\d+", "", reasoning)
        reasoning = re.sub(r"--challenge-nonce\s+[a-zA-Z0-9_]+", "", reasoning)
        
        if challenge_answer:
            reasoning = re.sub(r"Challenge:\s*.*$", "", reasoning, flags=re.MULTILINE).strip()
            reasoning = f"{reasoning}\n\nChallenge: {challenge_answer}"
        
        return direction, reasoning, challenge_answer, content
    except Exception as e:
        return None, None, None, str(e)

def get_validation(market_id, direction, reasoning, tech_summary):
    system_prompt = """You are the Chief Risk Officer (CRO) and Lead Validator for AWP Predict. 
Your job is to audit the analysis provided by the junior analyst. 
You should be balanced and objective. Your goal is to ensure predictions are logically sound and align with the regime while allowing for a reasonable volume of submissions.

VALIDATION CRITERIA:
1. Regime Adherence: Does the prediction follow the strict regime (BULL -> UP, BEAR -> DOWN)? 
   - If it's a reversal, is the RSI extreme (<20 or >80) and is there a volume spike?
   - If it's a reversal but criteria aren't met -> HOLD.
2. Logical Consistency: Does the reasoning generally support the direction? Does it mention relevant data points from the technical summary?
3. Originality: Does the reasoning sound completely robotic or like a generic template? If it's a blatant copy-paste without asset-specific context -> HOLD.
4. Technical Alignment: Is the prediction broadly consistent with the provided SMA20/EMA50 and RSI levels?

CRITICAL RULES:
- If the analysis is logically consistent and adheres to the regime, you MUST return 'DECISION: APPROVE'.
- Only return 'DECISION: HOLD' if there is a critical factual error, a blatant regime violation, or a logical contradiction.
- If your audit concludes the analysis is correct, do NOT return 'HOLD'.

OUTPUT FORMAT:
DECISION: [APPROVE or HOLD]
AUDIT: [A brief, sharp explanation of why you approved or held this prediction. 1-2 sentences.]"""
    
    user_prompt = (
        f"Market: {market_id}\n"
        f"Predicted Direction: {direction}\n"
        f"Technical Summary:\n{tech_summary}\n\n"
        f"Junior Analyst Reasoning:\n{reasoning}"
    )
    
    try:
        response = validator_client.chat.completions.create(
            model=config.VALIDATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        content = response.choices[0].message.content
        if content is None:
            with open("/home/ubuntu/.hermes/skills/predict-agent/validator_debug.log", "a") as f:
                f.write(f"--- FAILED PROMPT ---\n{user_prompt}\n--- END ---\n")
            return "HOLD", "Validation Error: LLM returned empty content", "None"
            
        dec_match = re.search(r"DECISION:\s*(APPROVE|HOLD)", content, re.IGNORECASE)
        decision = dec_match.group(1).upper() if dec_match else "HOLD"
        audit = content.split("AUDIT:", 1)[1].strip() if "AUDIT:" in content else content
        return decision, audit, content
    except Exception as e:
        return "HOLD", f"Validation Error: {e}", str(e)

def synthesize_lessons(losses):
    if not losses:
        return "No losses to analyze."
    
    loss_text = ""
    for l in losses[-20:]: # Analyze last 20 losses
        loss_text += f"Asset: {l.get('token')} | Pred: {l.get('prediction')} | Indicators: {l.get('indicators')} | Time: {l.get('timestamp')}\n"
    
    system_prompt = "You are a Quant Trading Auditor. Your goal is to identify common failure patterns from a list of losses. Provide concise, rule-based lessons (1-3 bullet points) in the format 'If [Condition] then [Avoid Action]'. No conversational filler, just the rules."
    user_prompt = f"Analyze these recent losses and synthesize a lesson:\n\n{loss_text}"
    
    try:
        response = analysis_client.chat.completions.create(
            model=config.ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Synthesis Error: {e}"
