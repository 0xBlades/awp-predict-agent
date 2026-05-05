# 🧠 AWP Predict Agent

> **Autonomous AI agent for crypto price prediction on the AWP (Agent Work Protocol) Predict WorkNet.**
> Combines multi-layer LLM analysis, XGBoost ML pre-signals, quantitative signal engine, and backtesting — all running 24/7 as a systemd service.

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AWP Predict Daemon                        │
│                     (15-min cycle)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │ Regime   │──▶│ Signal   │──▶│ Quality  │                │
│  │ Detector │   │ Engine   │   │ Filter   │                │
│  └──────────┘   └──────────┘   └──────────┘                │
│       │              │              │                       │
│       ▼              ▼              ▼                       │
│  ┌──────────────────────────────────────┐                  │
│  │     XGBoost ML Pre-Signal Filter     │  ◀── NEW (v4)   │
│  │  P(UP) > 70% → SKIP LLM (save $)   │                  │
│  │  P(UP) 55-70% → BIAS to LLM        │                  │
│  │  P(UP) < 55% → Full LLM pipeline   │                  │
│  └──────────────────────────────────────┘                  │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │ L1 LLM   │──▶│ Self-    │──▶│ L2 LLM   │                │
│  │ Analyst  │   │ Validate │   │ Validator │                │
│  └──────────┘   └──────────┘   └──────────┘                │
│       │                             │                       │
│       ▼                             ▼                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │ Weighted │──▶│ EV Gate  │──▶│ Submit   │                │
│  │ Confidence│  │          │   │ to AWP   │                │
│  └──────────┘   └──────────┘   └──────────┘                │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │     Memory Bank + Auto-Learning      │                  │
│  │  Loss streak protection              │                  │
│  │  Few-shot examples from wins         │                  │
│  │  Global lessons synthesis            │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Features

### 1. 🤖 Multi-Layer LLM Pipeline

| Layer | Model | Role |
|-------|-------|------|
| **L1 Analyst** | `deepseek-v3.2-exp` | Initial market analysis, entry/exit zones, RR estimation |
| **Self-Validate** | `deepseek-v3.2-exp` | L1 checks its own analysis for logical errors |
| **L2 Validator** | `deepseek-v3.2-exp` | Independent auditor — final APPROVE/HOLD decision |
| **Strategic Planner** | `gpt-5.5` | Regime-level strategy (once per cycle, not per market) |

**Token savings:** L2 validator and self-validation prevent low-quality trades before submission.

### 2. 🧮 XGBoost ML Pre-Signal (v4 — NEW)

Before hitting the LLM pipeline, a lightweight ML model runs first:

- **23 features:** RSI, MACD, EMA spread, Bollinger Band position/width, ATR ratio, VWAP distance, volume ratio, candle patterns, BTC trend, market breadth
- **3 decision modes:**
  - `SKIP_LLM` (confidence > 70%) — ML submits directly, **saves 5-10K tokens per trade**
  - `BIAS_LLM` (confidence 55-70%) — ML provides bias context to LLM
  - `NORMAL` (confidence < 55%) — Full LLM pipeline
- **Auto-retrain** every 6 hours with fresh Binance data
- **Cross-validation accuracy:** 56.3% (vs 50% random baseline)

### 3. 📈 Semi-Quantitative Signal Engine

Weight-based scoring system that runs before LLM:

| Component | Weight | Description |
|-----------|--------|-------------|
| `trend_score` | BTC trend + breadth + BB compression | Trend-following signal |
| `mean_score` | BB range + inverse breadth + volatility | Mean-reversion signal |
| **M15 Bias** | `mean × 1.8`, `trend × 0.7` | Scalping-friendly bias |

**Output:** `final_score` (-1.0 to +1.0), `signal` (LONG/SHORT/NO_TRADE), TP/SL recommendations

### 4. 🎯 6-State Market Regime

| Regime | Condition | Behavior |
|--------|-----------|----------|
| `STRONG_BULL` | BTC bullish + breadth ≥ 55% | Aggressive long, all assets |
| `WEAK_BULL` | BTC bullish + breadth < 30% | BTC long, altcoins selective |
| `TRANSITION_BULL` | BTC bullish + breadth 30-55% | Cautious long, top altcoins only |
| `STRONG_BEAR` | BTC bearish + breadth ≤ 45% | Aggressive short |
| `WEAK_BEAR` | BTC bearish + altcoins weak | Selective short |
| `CHOPPY` | BTC neutral, mid-range | Mean-reversion only |

### 5. 🔒 Multi-Gate Quality System

Every prediction must pass **all** gates:

1. **Kill Switch** — ATR too low (dead market) or too high (news/chaos) → skip
2. **Signal Gate** — `NO_TRADE` from signal engine → skip
3. **Quality Filter** — 3-state mode: `TREND` / `LOW_CONFIDENCE` / `NO_TRADE`
4. **RR Minimum** — TREND: 1.5, LOW_CONFIDENCE: 1.0
5. **Self-Validation** — L1 checks its own logic
6. **L2 Validation** — Independent auditor approves/rejects
7. **Weighted Confidence** — Composite score from L1 + L2 + quality + RR
8. **EV Gate** — Expected Value must be positive

### 6. 💾 Memory Bank & Auto-Learning

- **Few-shot examples** from winning trades (similar regime + indicators)
- **Global lessons** synthesized from loss patterns (every 12 hours)
- **Loss streak protection** — asset cooldown after consecutive losses
- **Heatmap tracking** — per-asset win rate for market prioritization

### 7. 📊 Backtest Engine

Full backtesting and optimization suite:

```bash
# Quick backtest (3000 candles, 200 trials)
python3 -m backtest.run --candles 3000 --trials 200

# Full optimization (5000 candles, 2000 trials — ~33 seconds)
python3 -m backtest.fast_optimize --candles 5000 --trials 2000

# Walk-forward validation (3 splits)
python3 -m backtest.run --walk-forward --candles 5000
```

**Optimized parameters (applied to live):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TREND_BIAS` | 0.7 | Trend weight multiplier |
| `MEAN_BIAS` | 1.8 | Mean-reversion weight multiplier |
| `ENTRY_THRESHOLD` | 0.2 | Minimum signal score for entry |
| `TP_MIN_PCT` | 0.8% | Take-profit minimum |
| `TP_MAX_PCT` | 1.0% | Take-profit maximum |
| `SL_MIN_PCT` | 0.4% | Stop-loss minimum |
| `SL_MAX_PCT` | 0.5% | Stop-loss maximum |
| `MAX_HOLD` | 12 candles | Maximum hold time (3 hours) |

---

## 📤 Output

### Per-Trade Output

Each prediction submission includes:

```json
{
  "market": "btc-15m-20260505-0715",
  "direction": "UP",
  "tickets": 500,
  "reasoning": "ML Pre-Signal: UP (P=0.740, high confidence)...",
  "challenge_nonce": "abc123...",
  "entry_zone": "80900-80950",
  "invalidation": "80750",
  "expected_rr": 1.8,
  "confidence": "high",
  "setup_grade": "A",
  "trade_mode": "TREND",
  "regime": "STRONG_BULL",
  "ml_signal": "UP",
  "ml_probability": 0.740
}
```

### Logging Output

Every cycle produces structured logs:

```
[2026-05-05 15:15:52] [STRATEGIC-PLAN] Regime: STRONG_BULL, Direction: UP
[2026-05-05 15:15:59] [Signal] btc-15m... -> score=0.4716, signal=LONG
[2026-05-05 15:16:00] [Quality] btc-15m... -> score=45, mode=LOW_CONFIDENCE
[2026-05-05 15:16:00] [ML] [BTC] ML: P(UP)=0.497, signal=UNCERTAIN, action=NORMAL
[2026-05-05 15:16:11] Challenge: ... | Answer: 92 | Conf: 6/10 | Grade: B
[2026-05-05 15:16:13] [Final Decision] btc-15m... -> HOLD (low)
```

### Daily Report

```
==================================================
DAILY REPORT — 2026-05-05 15:00
==================================================
Win Rate: 52.3% (23W / 21L / 44 total)

Top Markets by Win Rate:
  BTC: 58.3% (12 trades)
  ETH: 50.0% (18 trades)
  SOL: 48.0% (14 trades)

Asset Performance:
  BTC: 58% (12W/8L)
  ETH: 50% (9W/9L)
  SOL: 48% (7W/8L)
==================================================
```

---

## ⚙️ Configuration

All config in `.env`:

```bash
# Models (via Swiftrouter)
L1_MODEL=deepseek-v3.2-exp
L2_MODEL=deepseek-v3.2-exp
STRATEGIC_PLANNER_MODEL=gpt-5.5
SWIFTROUTER_API_KEY=sk-...

# AWP
WALLET_HOME=/home/ubuntu/.awp-predict-2/wallets
```

Signal constants in `config.py` (auto-optimized from backtest):

```python
SIGNAL_ENTRY_THRESHOLD = 0.2
SIGNAL_MEAN_BIAS = 1.8
SIGNAL_TREND_BIAS = 0.7
TP_MIN_PCT = 0.008
TP_MAX_PCT = 0.01
SL_MIN_PCT = 0.004
SL_MAX_PCT = 0.005
MAX_HOLD_CANDLES = 12
```

---

## 🏗️ Architecture

```
awp-predict-2/
├── predict_daemon.py      # Main daemon (15-min cycle, systemd)
├── config.py              # Signal constants + config
├── technical_analysis.py  # Indicators + signal engine + quality filter
├── llm_pipeline.py        # L1/L2 LLM analysis + validation
├── memory_manager.py      # Memory bank CRUD + heatmap + few-shot
├── ml_engine.py           # XGBoost pre-signal (NEW)
├── daily_report.py        # Daily performance report
├── backtest/
│   ├── fast_optimize.py   # Vectorized optimizer (0.12s/trial)
│   ├── simulator.py       # Strategy simulator
│   ├── indicators.py      # Technical indicators
│   ├── metrics.py         # Performance metrics
│   ├── optimizer.py       # Random search + walk-forward
│   ├── data_fetcher.py    # Binance kline fetcher
│   └── run.py             # CLI entry point
├── ml_models/             # XGBoost model artifacts
├── .env                   # Secrets + model config
└── memory_bank.json       # Trade memory (auto-managed)
```

---

## 🔧 Deployment

```bash
# Systemd service
sudo systemctl start awp-predict-2
sudo systemctl status awp-predict-2

# View logs
journalctl -u awp-predict-2 -f

# Retrain ML model
cd ~/.awp-predict-2 && venv/bin/python3 ml_engine.py train 5000

# Check ML status
venv/bin/python3 ml_engine.py status

# Run backtest
venv/bin/python3 -m backtest.fast_optimize --candles 5000 --trials 2000
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Assets** | BTC, ETH, SOL (M15 timeframe) |
| **Cycle** | Every 15 minutes |
| **Max submissions** | 3 per cycle |
| **ML accuracy** | 56.3% (CV) |
| **Optimized R:R** | 1.13 (TP 0.8-1.0%, SL 0.4-0.5%) |
| **Token cost** | ~5-10K per trade (full LLM) |
| **ML savings** | 100% when ML confident (>70%) |

---

## 📝 License

Private — AWP Predict WorkNet Agent
