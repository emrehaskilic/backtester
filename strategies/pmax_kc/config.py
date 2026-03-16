"""PMax+KC Strategy Configuration — parameter bounds, defaults, constants."""

import os

SCALPER_BOT_PATH = r"C:\Users\emrehaskilic\Desktop\Scalper Bot"

# PMax parameter bounds
PARAM_BOUNDS = {
    "atr_period":      {"min": 5,   "max": 30,  "step": 1,    "type": "int"},
    "atr_multiplier":  {"min": 1.0, "max": 5.0, "step": 0.25, "type": "float"},
    "ma_length":       {"min": 5,   "max": 50,  "step": 1,    "type": "int"},
    "ma_type":         {"choices": ["EMA", "SMA", "WMA", "ZLEMA", "ALMA", "TEMA", "HULL"]},
    "source":          {"choices": ["hl2", "hlc3", "ohlc4", "close"]},
}

# Keltner channel bounds
KC_BOUNDS = {
    "kc_length":      {"min": 10, "max": 50, "step": 1,   "type": "int"},
    "kc_multiplier":  {"min": 0.5, "max": 3.0, "step": 0.1, "type": "float"},
    "kc_atr_period":  {"min": 5,  "max": 30, "step": 1,   "type": "int"},
}

# Filter bounds
FILTER_BOUNDS = {
    "ema_filter_period": {"min": 50,  "max": 300, "step": 1, "type": "int"},
    "rsi_overbought":    {"min": 60,  "max": 80,  "step": 1, "type": "int"},
}

# Risk bounds
RISK_BOUNDS = {
    "max_dca_steps":    {"min": 1,   "max": 5,    "step": 1,    "type": "int"},
    "tp_close_percent": {"min": 0.10, "max": 0.50, "step": 0.05, "type": "float"},
}

# Default PMax parameters
DEFAULT_PARAMS = {
    "atr_period": 10,
    "atr_multiplier": 3.0,
    "ma_length": 10,
    "ma_type": "EMA",
    "source": "hl2",
    "kc_length": 20,
    "kc_multiplier": 1.5,
    "kc_atr_period": 10,
    "ema_filter_period": 144,
    "rsi_overbought": 65,
    "max_dca_steps": 2,
    "tp_close_percent": 0.20,
}

# Optimization settings
TRAIN_RATIO = 0.7

# Trading constants
INITIAL_BALANCE = 10000.0
LEVERAGE = 25
MARGIN_PER_TRADE = 1000.0
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
