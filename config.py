import os

# --- PATHS ---
AGENT_HOME = os.environ.get("AGENT_HOME", os.path.expanduser("~/.awp-predict-fresh"))
WALLET_HOME = os.environ.get("WALLET_HOME", AGENT_HOME)
LOG_FILE = os.path.join(AGENT_HOME, "predict_daemon.log")
MEMORY_FILE = os.path.join(AGENT_HOME, "memory_bank.json")
LESSONS_FILE = os.path.join(AGENT_HOME, "global_lessons.txt")
ATR_HISTORY_FILE = os.path.join(AGENT_HOME, "atr_history.json")

# --- API KEYS ---
SWIFTROUTER_API_KEY = os.environ.get("SWIFTROUTER_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# --- LLM MODELS ---
ANALYSIS_MODEL = "minimax-m2.7"
VALIDATOR_MODEL = "google/gemma-4-31b-it"
SWIFTROUTER_BASE_URL = "https://api.swiftrouter.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
