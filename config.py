import os

# --- PATHS ---
AGENT_HOME = os.environ.get("AGENT_HOME", os.path.expanduser("/home/ubuntu/.awp-predict-2"))
WALLET_HOME = os.environ.get("WALLET_HOME", AGENT_HOME)
LOG_FILE = os.path.join(AGENT_HOME, "predict_daemon.log")
MEMORY_FILE = os.path.join(AGENT_HOME, "memory_bank.json")
LESSONS_FILE = os.path.join(AGENT_HOME, "global_lessons.txt")
ATR_HISTORY_FILE = os.path.join(AGENT_HOME, "atr_history.json")
HEATMAP_FILE = os.path.join(AGENT_HOME, "market_heatmap.json")
WINRATE_FILE = os.path.join(AGENT_HOME, "winrate_log.json")

# --- API KEYS ---
SWIFTROUTER_API_KEY = os.environ.get("SWIFTROUTER_API_KEY", "")

# --- LLM MODELS (All via Swiftrouter, no fallback) ---
ANALYSIS_MODEL = os.environ.get("L1_MODEL", "gpt-5.4")
VALIDATOR_MODEL = os.environ.get("L2_MODEL", "deepseek-v3.2-exp")
STRATEGIC_PLANNER_MODEL = os.environ.get("STRATEGIC_PLANNER_MODEL", "gpt-5.5")

# --- BASE URL ---
SWIFTROUTER_BASE_URL = os.environ.get("SWIFTROUTER_BASE_URL", "https://api.swiftrouter.com/v1")

# --- TICKET SIZING ---
TICKET_DEFAULT = int(os.environ.get("TICKET_DEFAULT", "1000"))
TICKET_LOW_CONF = int(os.environ.get("TICKET_LOW_CONF", "500"))
TICKET_HIGH_CONF = int(os.environ.get("TICKET_HIGH_CONF", "1000"))

# --- MEMORY CLEANUP ---
PENDING_MAX_AGE_HOURS = int(os.environ.get("PENDING_MAX_AGE_HOURS", "24"))
MEMORY_MAX_ENTRIES = int(os.environ.get("MEMORY_MAX_ENTRIES", "500"))

# --- CIRCUIT BREAKER ---
CB_FAILURE_THRESHOLD = int(os.environ.get("CB_FAILURE_THRESHOLD", "3"))
CB_COOLDOWN_SECONDS = int(os.environ.get("CB_COOLDOWN_SECONDS", "300"))

# --- LOW CONFIDENCE MODE (Sideways/Chop Strategy) ---
QUALITY_THRESHOLD = int(os.environ.get("QUALITY_THRESHOLD", "50"))
DEAD_ZONE = int(os.environ.get("DEAD_ZONE", "15"))
LOW_CONF_TICKET_MULT = float(os.environ.get("LOW_CONF_TICKET_MULT", "0.4"))
LOW_CONF_RR_MIN = float(os.environ.get("LOW_CONF_RR_MIN", "1.0"))
LOW_CONF_MAX_SUBMISSIONS = int(os.environ.get("LOW_CONF_MAX_SUBMISSIONS", "1"))
LOW_CONF_RSI_EXTREME = float(os.environ.get("LOW_CONF_RSI_EXTREME", "30"))

# --- CHALLENGE ---
CHALLENGE_MAX_RETRIES = int(os.environ.get("CHALLENGE_MAX_RETRIES", "2"))

# --- TRADING SIGNAL ENGINE (Semi-Quant Adaptive) ---
SIGNAL_ENTRY_THRESHOLD = float(os.environ.get("SIGNAL_ENTRY_THRESHOLD", "0.2"))
SIGNAL_LOW_CONF_THRESHOLD = float(os.environ.get("SIGNAL_LOW_CONF_THRESHOLD", "0.4"))
SIGNAL_MEAN_BIAS = float(os.environ.get("SIGNAL_MEAN_BIAS", "1.8"))
SIGNAL_TREND_BIAS = float(os.environ.get("SIGNAL_TREND_BIAS", "0.7"))
SIGNAL_BTC_TREND_BONUS = float(os.environ.get("SIGNAL_BTC_TREND_BONUS", "0.1"))
SIGNAL_ALT_MEAN_BONUS = float(os.environ.get("SIGNAL_ALT_MEAN_BONUS", "0.1"))
SIGNAL_COUNTER_TREND_SIZE = float(os.environ.get("SIGNAL_COUNTER_TREND_SIZE", "0.3"))

# --- TP/SL (Scalping Mode) ---
TP_MIN_PCT = float(os.environ.get("TP_MIN_PCT", "0.008"))
TP_MAX_PCT = float(os.environ.get("TP_MAX_PCT", "0.010"))
SL_MIN_PCT = float(os.environ.get("SL_MIN_PCT", "0.004"))
SL_MAX_PCT = float(os.environ.get("SL_MAX_PCT", "0.005"))
TRAILING_ACTIVATION_PCT = float(os.environ.get("TRAILING_ACTIVATION_PCT", "0.005"))
MAX_HOLD_CANDLES = int(os.environ.get("MAX_HOLD_CANDLES", "12"))
STAGNANT_CLOSE_CANDLES = int(os.environ.get("STAGNANT_CLOSE_CANDLES", "4"))

# --- KILL SWITCH ---
KILL_ATR_MIN_RATIO = float(os.environ.get("KILL_ATR_MIN_RATIO", "0.15"))
KILL_ATR_MAX_RATIO = float(os.environ.get("KILL_ATR_MAX_RATIO", "3.0"))
