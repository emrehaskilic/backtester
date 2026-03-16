"""Swinginess Strategy Configuration — parameter bounds, defaults."""

# Default DFS weights
DEFAULT_DFS_WEIGHTS = {
    "w_delta": 0.22, "w_cvd": 0.18, "w_logp": 0.12,
    "w_obi_w": 0.14, "w_obi_d": 0.12, "w_sweep": 0.08,
    "w_burst": 0.08, "w_oi": 0.06,
}

# Default TRS parameters
DEFAULT_TRS = {
    "trs_confirm_ticks": 90,
    "trs_bullish_zone": 0.65,
    "trs_bearish_zone": 0.35,
    "trs_agreement": 0.40,
}

# Default risk parameters
DEFAULT_RISK = {
    "stop_loss_pct": 1.5,
    "trailing_activation_pct": 0.8,
    "trailing_distance_pct": 0.5,
    "exit_score_hard": 0.85,
    "exit_score_soft": 0.70,
}

# Default filter parameters
DEFAULT_FILTERS = {
    "min_prints_per_sec": 1.0,
    "entry_cooldown_sec": 300,
    "rolling_window_sec": 3600,
    "time_flat_sec": 14400,
    "margin_per_trade": 100,
}

# Trading constants
INITIAL_BALANCE = 1000.0
LEVERAGE = 50
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005

# Optimization settings
TRAIN_RATIO = 0.7
N_TRIALS = 500
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
