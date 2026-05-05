"""
AWP Predict — XGBoost Pre-Signal Engine
Fast ML prediction layer that runs BEFORE the LLM pipeline.
If XGBoost is confident (>70%), it can skip the LLM entirely (saves tokens).
If moderately confident (50-70%), it provides a bias to the LLM.

Features: Same indicators used by the weight engine (RSI, MACD, EMA, BB, ATR, etc.)
Label: Next candle direction (UP=1, DOWN=0)
Inference: <1ms per prediction
"""

import os
import json
import time
import numpy as np
import pandas as pd

# Lazy imports for heavy modules
_xgb = None
_joblib = None
_sklearn = None

def _load_xgb():
    global _xgb, _joblib, _sklearn
    if _xgb is None:
        import xgboost as xgb
        import joblib as jb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, classification_report
        _xgb = xgb
        _joblib = jb
        _sklearn = {
            'tscv': TimeSeriesSplit,
            'accuracy': accuracy_score,
            'report': classification_report
        }
    return _xgb, _joblib, _sklearn

import config

# --- Constants ---
MODEL_DIR = os.path.join(config.AGENT_HOME, "ml_models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_pre_signal.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")
RETRAIN_INTERVAL = 3600 * 6  # retrain every 6 hours
MIN_SAMPLES = 500  # minimum candles needed for training


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [ML] {msg}", flush=True)


# ============================================================
# FEATURE EXTRACTION (from M15 candle data)
# ============================================================

def compute_features(df_15: pd.DataFrame, btc_trend: int = 0, breadth: float = 0.5) -> pd.DataFrame:
    """
    Compute ML features from M15 candle DataFrame.
    Same indicators as get_trading_signal() but structured for batch computation.
    
    Expects df_15 with columns: open, high, low, close, volume
    Returns DataFrame with feature columns added.
    """
    df = df_15.copy()
    
    # --- EMA ---
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # --- EMA Spread % ---
    df['ema_spread_pct'] = (df['ema9'] - df['ema21']).abs() / df['close'] * 100
    
    # --- RSI (14) ---
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # --- MACD (12, 26, 9) ---
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    df['macd_hist_pct'] = df['macd_hist'] / df['close'] * 100  # normalized
    
    # --- ATR (14) ---
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # --- ATR Ratio (current vs 20-period avg) ---
    df['atr_avg20'] = df['atr'].rolling(20).mean()
    df['atr_ratio'] = df['atr'] / df['atr_avg20'].replace(0, np.nan)
    
    # --- Bollinger Bands (20, 2) ---
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    # BB Position: 0 = at lower, 1 = at upper
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range.replace(0, np.nan)
    
    # --- BB Width Ratio (current vs 20-period avg) ---
    df['bb_width_avg'] = df['bb_width'].rolling(20).mean()
    df['bb_width_ratio'] = df['bb_width'] / df['bb_width_avg'].replace(0, np.nan)
    
    # --- VWAP (rolling 20) ---
    tp = (df['high'] + df['low'] + df['close']) / 3
    cum_tp_vol = (tp * df['volume']).rolling(20).sum()
    cum_vol = df['volume'].rolling(20).sum()
    df['vwap'] = cum_tp_vol / cum_vol.replace(0, np.nan)
    df['vwap_dist_pct'] = (df['close'] - df['vwap']).abs() / df['close'] * 100
    df['above_vwap'] = (df['close'] > df['vwap']).astype(float)
    
    # --- Volume Ratio ---
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)
    
    # --- Candle Features ---
    df['body_pct'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, np.nan) * 100
    df['is_green'] = (df['close'] > df['open']).astype(float)
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, np.nan)
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # --- Price Position ---
    df['close_vs_ema9'] = (df['close'] - df['ema9']) / df['close'] * 100
    df['close_vs_ema21'] = (df['close'] - df['ema21']) / df['close'] * 100
    df['close_vs_ema50'] = (df['close'] - df['ema50']) / df['close'] * 100
    
    # --- Momentum (rate of change) ---
    df['roc_5'] = df['close'].pct_change(5) * 100
    df['roc_10'] = df['close'].pct_change(10) * 100
    
    # --- Candle change % ---
    df['change_pct'] = df['close'].pct_change() * 100
    
    # --- Static features (passed as constants) ---
    df['btc_trend'] = btc_trend  # +1, 0, -1
    df['breadth'] = breadth  # 0-1
    
    return df


FEATURE_COLS = [
    'rsi', 'macd_hist_pct', 'ema_spread_pct',
    'bb_position', 'bb_width', 'bb_width_ratio',
    'atr_pct', 'atr_ratio',
    'vol_ratio', 'vwap_dist_pct', 'above_vwap',
    'close_vs_ema9', 'close_vs_ema21', 'close_vs_ema50',
    'body_pct', 'is_green', 'upper_wick', 'lower_wick',
    'roc_5', 'roc_10', 'change_pct',
    'btc_trend', 'breadth'
]


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    Prepare features and labels for XGBoost training.
    Label: 1 if next candle close > current close, else 0.
    
    Returns (X, y, valid_mask) — valid_mask indicates which rows have complete features.
    """
    # Label: next candle goes UP
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop rows with NaN features
    valid_mask = df[FEATURE_COLS].notna().all(axis=1) & df['label'].notna()
    X = df.loc[valid_mask, FEATURE_COLS].values
    y = df.loc[valid_mask, 'label'].values
    
    return X, y, valid_mask


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(candles: int = 5000, verbose: bool = True) -> dict:
    """
    Train XGBoost model on historical M15 data from Binance.
    Uses TimeSeriesSplit for proper temporal validation.
    
    Returns training metrics dict.
    """
    xgb, joblib_mod, sklearn_mod = _load_xgb()
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    log(f"Training XGBoost model with {candles} candles...")
    
    # Fetch historical data from Binance
    import ccxt
    exchange = ccxt.binance()
    
    all_data = []
    for token in ['BTC', 'ETH', 'SOL']:
        symbol = f"{token}/USDT"
        try:
            limit = min(candles, 1000)  # Binance max per request
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Get BTC trend and breadth
            btc_trend = 0
            try:
                btc_ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)
                btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                btc_df['ema50'] = btc_df['close'].ewm(span=50, adjust=False).mean()
                btc_df['ema200'] = btc_df['close'].ewm(span=200, adjust=False).mean()
                curr = btc_df.iloc[-1]
                if pd.notna(curr['ema50']) and pd.notna(curr['ema200']):
                    btc_trend = 1 if curr['ema50'] > curr['ema200'] else -1
            except:
                pass
            
            # Compute features
            df_feat = compute_features(df, btc_trend=btc_trend, breadth=0.5)
            df_feat['token'] = token
            all_data.append(df_feat)
            log(f"  {token}: {len(df)} candles loaded")
            
            time.sleep(0.5)  # rate limit
        except Exception as e:
            log(f"  {token}: FAILED — {e}")
    
    if not all_data:
        log("ERROR: No data loaded for training")
        return {"error": "no data"}
    
    # Combine all tokens
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Prepare features and labels
    X, y, valid_mask = prepare_training_data(df_all)
    
    if len(X) < MIN_SAMPLES:
        log(f"ERROR: Not enough samples ({len(X)} < {MIN_SAMPLES})")
        return {"error": f"insufficient data: {len(X)}"}
    
    log(f"Training data: {len(X)} samples, {X.shape[1]} features")
    log(f"Label distribution: UP={sum(y)}, DOWN={len(y)-sum(y)} ({sum(y)/len(y)*100:.1f}% UP)")
    
    # TimeSeriesSplit (5 folds)
    tscv = sklearn_mod['tscv'](n_splits=5)
    fold_metrics = []
    
    best_model = None
    best_acc = 0
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            verbosity=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        acc = sklearn_mod['accuracy'](y_val, y_pred)
        fold_metrics.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
        
        if verbose:
            log(f"  Fold {fold+1}: accuracy={acc:.4f}")
    
    # Final metrics
    avg_acc = np.mean(fold_metrics)
    std_acc = np.std(fold_metrics)
    
    log(f"Cross-validation: avg_acc={avg_acc:.4f} ± {std_acc:.4f}")
    log(f"Best fold accuracy: {best_acc:.4f}")
    
    # Feature importance
    importance = dict(zip(FEATURE_COLS, best_model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    log("Top 10 features:")
    for name, imp in top_features:
        log(f"  {name}: {imp:.4f}")
    
    # Save model
    joblib_mod.dump(best_model, MODEL_PATH)
    
    # Save feature names
    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(FEATURE_COLS, f)
    
    # Save metrics
    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_tokens": len(all_data),
        "cv_accuracy": round(float(avg_acc), 4),
        "cv_std": round(float(std_acc), 4),
        "best_accuracy": round(float(best_acc), 4),
        "fold_accuracies": [round(float(m), 4) for m in fold_metrics],
        "top_features": {k: round(float(v), 4) for k, v in top_features},
        "label_distribution": {
            "up": int(sum(y)),
            "down": int(len(y) - sum(y)),
            "up_pct": round(float(sum(y) / len(y) * 100), 1)
        }
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    log(f"Model saved to {MODEL_PATH}")
    return metrics


# ============================================================
# MODEL INFERENCE
# ============================================================

class MLPreSignal:
    """XGBoost pre-signal filter for AWP Predict."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.last_train_time = 0
        self._load_model()
    
    def _load_model(self):
        """Load trained model from disk."""
        try:
            if os.path.exists(MODEL_PATH):
                import joblib as jb
                self.model = jb.load(MODEL_PATH)
                self.model_loaded = True
                log(f"Model loaded from {MODEL_PATH}")
            else:
                log("No trained model found — ML pre-signal DISABLED")
        except Exception as e:
            log(f"Failed to load model: {e}")
    
    def predict(self, token: str, indicators: dict, regime: str, 
                signal_data: dict, trade_mode: str = "TREND") -> dict:
        """
        Generate ML pre-signal prediction.
        
        Args:
            token: Asset symbol (BTC, ETH, SOL)
            indicators: dict from get_market_quality() or fetch_technical_data()
            regime: Market regime string
            signal_data: dict from get_trading_signal()
            trade_mode: TREND / LOW_CONFIDENCE / NO_TRADE
        
        Returns:
            dict with:
            - ml_prob: probability of UP (0.0-1.0)
            - ml_signal: "UP" / "DOWN" / "UNCERTAIN"
            - ml_confidence: "high" / "medium" / "low"
            - ml_action: "SKIP_LLM" / "BIAS_LLM" / "NORMAL"
            - ml_features: dict of extracted features
            - model_loaded: bool
        """
        if not self.model_loaded:
            return {
                "ml_prob": 0.5,
                "ml_signal": "UNCERTAIN",
                "ml_confidence": "none",
                "ml_action": "NORMAL",
                "ml_features": {},
                "model_loaded": False
            }
        
        try:
            # Extract features from indicators dict
            features = self._extract_features(token, indicators, signal_data)
            
            # Convert to numpy array
            X = np.array([[features.get(col, 0.0) for col in FEATURE_COLS]])
            
            # Check for NaN
            if np.isnan(X).any():
                log(f"[{token}] NaN in features, falling back to NORMAL")
                return {
                    "ml_prob": 0.5,
                    "ml_signal": "UNCERTAIN",
                    "ml_confidence": "none",
                    "ml_action": "NORMAL",
                    "ml_features": features,
                    "model_loaded": True
                }
            
            # Predict probability
            prob = self.model.predict_proba(X)[0]
            up_prob = float(prob[1])  # P(UP)
            down_prob = float(prob[0])  # P(DOWN)
            
            # Determine signal
            confidence_threshold_high = 0.70
            confidence_threshold_low = 0.55
            
            if up_prob >= confidence_threshold_high:
                ml_signal = "UP"
                ml_confidence = "high"
                ml_action = "SKIP_LLM"
            elif down_prob >= confidence_threshold_high:
                ml_signal = "DOWN"
                ml_confidence = "high"
                ml_action = "SKIP_LLM"
            elif up_prob >= confidence_threshold_low:
                ml_signal = "UP"
                ml_confidence = "medium"
                ml_action = "BIAS_LLM"
            elif down_prob >= confidence_threshold_low:
                ml_signal = "DOWN"
                ml_confidence = "medium"
                ml_action = "BIAS_LLM"
            else:
                ml_signal = "UNCERTAIN"
                ml_confidence = "low"
                ml_action = "NORMAL"
            
            log(f"[{token}] ML: P(UP)={up_prob:.3f}, P(DOWN)={down_prob:.3f}, "
                f"signal={ml_signal}, conf={ml_confidence}, action={ml_action}")
            
            return {
                "ml_prob": up_prob,
                "ml_signal": ml_signal,
                "ml_confidence": ml_confidence,
                "ml_action": ml_action,
                "ml_features": features,
                "model_loaded": True
            }
            
        except Exception as e:
            log(f"[{token}] ML prediction FAILED: {e}")
            return {
                "ml_prob": 0.5,
                "ml_signal": "UNCERTAIN",
                "ml_confidence": "none",
                "ml_action": "NORMAL",
                "ml_features": {},
                "model_loaded": True
            }
    
    def _extract_features(self, token: str, indicators: dict, signal_data: dict) -> dict:
        """Extract ML features from indicator dicts."""
        features = {}
        
        # From indicators (get_market_quality / fetch_technical_data)
        features['rsi'] = indicators.get('rsi', 50.0)
        features['macd_hist_pct'] = indicators.get('macd_hist_pct', 0.0)
        features['ema_spread_pct'] = indicators.get('ema_spread_pct', 0.2)
        features['bb_position'] = indicators.get('bb_position', 0.5)
        features['bb_width'] = indicators.get('bb_width', 0.02)
        features['bb_width_ratio'] = indicators.get('bb_width_ratio', 1.0)
        features['atr_pct'] = indicators.get('atr_pct', 0.5)
        features['atr_ratio'] = indicators.get('atr_ratio', 1.0)
        features['vol_ratio'] = indicators.get('vol_ratio', 1.0)
        features['vwap_dist_pct'] = indicators.get('vwap_dist_pct', 0.1)
        features['above_vwap'] = indicators.get('above_vwap', 0.5)
        features['body_pct'] = indicators.get('body_pct', 50.0)
        features['is_green'] = indicators.get('is_green', 0.5)
        features['upper_wick'] = indicators.get('upper_wick', 0.25)
        features['lower_wick'] = indicators.get('lower_wick', 0.25)
        features['close_vs_ema9'] = indicators.get('close_vs_ema9', 0.0)
        features['close_vs_ema21'] = indicators.get('close_vs_ema21', 0.0)
        features['close_vs_ema50'] = indicators.get('close_vs_ema50', 0.0)
        features['roc_5'] = indicators.get('roc_5', 0.0)
        features['roc_10'] = indicators.get('roc_10', 0.0)
        features['change_pct'] = indicators.get('change_pct', 0.0)
        
        # From signal_data
        features['btc_trend'] = signal_data.get('btc_trend', 0)
        features['breadth'] = signal_data.get('breadth', 0.5)
        
        return features
    
    def auto_retrain(self, force: bool = False) -> dict:
        """Check if retrain is needed and do it."""
        if not force and (time.time() - self.last_train_time) < RETRAIN_INTERVAL:
            return {"status": "skip", "reason": "not due yet"}
        
        # Check if model exists and is fresh enough
        if os.path.exists(MODEL_PATH):
            model_age = time.time() - os.path.getmtime(MODEL_PATH)
            if not force and model_age < RETRAIN_INTERVAL:
                return {"status": "skip", "reason": f"model is {model_age/3600:.1f}h old"}
        
        log("Auto-retraining XGBoost model...")
        metrics = train_model(candles=5000, verbose=True)
        self._load_model()  # reload
        self.last_train_time = time.time()
        return {"status": "trained", "metrics": metrics}
    
    def get_status(self) -> dict:
        """Get current ML engine status."""
        model_exists = os.path.exists(MODEL_PATH)
        model_age = 0
        if model_exists:
            model_age = time.time() - os.path.getmtime(MODEL_PATH)
        
        metrics = {}
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH) as f:
                    metrics = json.load(f)
            except:
                pass
        
        return {
            "model_loaded": self.model_loaded,
            "model_exists": model_exists,
            "model_age_hours": round(model_age / 3600, 1),
            "cv_accuracy": metrics.get("cv_accuracy", "N/A"),
            "trained_at": metrics.get("trained_at", "N/A"),
            "n_samples": metrics.get("n_samples", "N/A"),
        }


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        candles = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        metrics = train_model(candles=candles)
        print(json.dumps(metrics, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        engine = MLPreSignal()
        status = engine.get_status()
        print(json.dumps(status, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        # Test prediction with dummy data
        engine = MLPreSignal()
        dummy_indicators = {
            'rsi': 45.0, 'macd_hist_pct': -0.01, 'ema_spread_pct': 0.15,
            'bb_position': 0.4, 'bb_width': 0.03, 'bb_width_ratio': 0.9,
            'atr_pct': 0.5, 'atr_ratio': 1.1, 'vol_ratio': 1.2,
            'vwap_dist_pct': 0.08, 'above_vwap': 1.0,
            'body_pct': 60.0, 'is_green': 1.0, 'upper_wick': 0.2, 'lower_wick': 0.2,
            'close_vs_ema9': 0.1, 'close_vs_ema21': 0.2, 'close_vs_ema50': 0.5,
            'roc_5': 0.3, 'roc_10': 0.5, 'change_pct': 0.1
        }
        dummy_signal = {'btc_trend': 1, 'breadth': 0.55}
        result = engine.predict("BTC", dummy_indicators, "TRANSITION_BULL", dummy_signal)
        print(json.dumps(result, indent=2))
    else:
        print("Usage:")
        print("  python3 ml_engine.py train [candles]  — Train XGBoost model")
        print("  python3 ml_engine.py status          — Show model status")
        print("  python3 ml_engine.py predict          — Test prediction")
