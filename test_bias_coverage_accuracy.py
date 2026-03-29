"""
Bias Engine: %100 Coverage + ~%53 Accuracy Reproduction

Uses the full pipeline:
1. Bias engine (state mining, robustness, fallback, calibration, regime, sweep)
2. Combined scoring (MR + RSI + CVD + agreement)
3. Walk-forward validation

Tests both default and TPE-optimized parameters.
"""
import time
import numpy as np
import pandas as pd
import rust_engine

# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION: 5m → 1H aggregation
# ═══════════════════════════════════════════════════════════════
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
print(f"5m data: {len(df):,} bars")

PERIOD = 12
n_1h = len(df) // PERIOD

ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

ts_1h = np.zeros(n_1h, dtype=np.uint64)
o_1h = np.zeros(n_1h); h_1h = np.zeros(n_1h); l_1h = np.zeros(n_1h); c_1h = np.zeros(n_1h)
bv_1h = np.zeros(n_1h); sv_1h = np.zeros(n_1h); oi_1h = np.zeros(n_1h)

for i in range(n_1h):
    s, e = i * PERIOD, i * PERIOD + PERIOD
    ts_1h[i] = ts[s]; o_1h[i] = o[s]; h_1h[i] = h[s:e].max()
    l_1h[i] = l[s:e].min(); c_1h[i] = c[e - 1]
    bv_1h[i] = bv[s:e].sum(); sv_1h[i] = sv[s:e].sum(); oi_1h[i] = oi[e - 1]

print(f"1H data: {n_1h:,} bars ({n_1h/24/365:.1f} years)")
print(f"Date: {pd.to_datetime(ts_1h[0], unit='ms')} → {pd.to_datetime(ts_1h[-1], unit='ms')}")

# ═══════════════════════════════════════════════════════════════
# STEP 1: Bias Engine (state mining pipeline)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STEP 1: Bias Engine Pipeline")
print(f"{'='*60}")

t0 = time.time()
r = rust_engine.bias_engine_compute_bias(
    ts_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h
)
print(f"Done in {time.time()-t0:.1f}s")

fb = np.array(r['final_bias'])
conf = np.array(r['confidence'])
md = np.array(r['matched_depth'])
direction = np.array(r['direction'])

print(f"  Validated states: {r['n_validated']}")
print(f"  Coverage: {r['coverage']['coverage_pct']:.1f}%")
print(f"  Direction accuracy: {r['accuracy']['direction_accuracy']:.4f}")
print(f"  Strong accuracy: {r['accuracy']['strong_signal_accuracy']:.4f} (n={r['accuracy']['n_strong_bars']})")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Combined Scoring (Python implementation)
# Replicates Rust scoring.rs logic with optimal params
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STEP 2: Combined Scoring (MR + RSI + CVD + Agreement)")
print(f"{'='*60}")

def ema(data, span):
    k = 2.0 / (span + 1)
    out = np.zeros_like(data, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

def compute_rsi(data, period=14):
    rsi = np.full(len(data), 50.0)
    gains = np.zeros(len(data))
    losses = np.zeros(len(data))
    for i in range(1, len(data)):
        d = data[i] - data[i-1]
        if d > 0: gains[i] = d
        else: losses[i] = -d
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    for i in range(period, len(data)):
        if avg_loss[i] > 1e-15:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - 100 / (1 + rs)
    return rsi

def cvd_zscore(buy_vol, sell_vol, window=24):
    cvd_bar = buy_vol - sell_vol
    cvd_ema = ema(cvd_bar, window)
    std_arr = np.zeros(len(buy_vol))
    for i in range(window, len(buy_vol)):
        diff = cvd_bar[i-window:i] - cvd_ema[i-window:i]
        std_arr[i] = np.sqrt(np.mean(diff**2))
    z = np.zeros(len(buy_vol))
    for i in range(window, len(buy_vol)):
        if std_arr[i] > 1e-15:
            z[i] = (cvd_bar[i] - cvd_ema[i]) / std_arr[i]
    return z

# Outcomes: K-bar lookahead
K = 2  # TPE optimal: K=2
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
outcomes[-K:] = -1
valid = outcomes >= 0

def compute_combined_score(params):
    """Compute combined scoring with given parameters."""
    mr_span1 = params['mr_ema_span1']
    mr_span2 = params['mr_ema_span2']
    rsi_period = params['rsi_period']
    rsi_threshold = params['rsi_threshold']

    # EMA distances → MR signals
    ema1 = ema(c_1h, mr_span1)
    ema2 = ema(c_1h, mr_span2)
    mr1_sign = np.where(c_1h > ema1, -1.0, 1.0)
    mr2_sign = np.where(c_1h > ema2, -1.0, 1.0)

    # RSI
    rsi = compute_rsi(c_1h, rsi_period)

    # CVD z-score
    cvd_z = cvd_zscore(bv_1h, sv_1h, 24)

    # Build combined score
    score = np.zeros(n_1h)

    # 1. Bias engine
    score += fb * params['w_bias']

    # 2. MR primary
    score += mr1_sign * params['w_mr1']

    # 3. MR secondary (when agrees)
    agree_mr = (mr1_sign != 0) & (mr2_sign != 0) & (mr1_sign == mr2_sign)
    score += np.where(agree_mr, mr1_sign * params['w_mr2'], 0)

    # 4. RSI confirmation
    rsi_oversold = rsi < (50 - rsi_threshold)
    rsi_overbought = rsi > (50 + rsi_threshold)
    rsi_confirm = ((rsi_oversold & (mr1_sign > 0)) | (rsi_overbought & (mr1_sign < 0)))
    score += np.where(rsi_confirm, mr1_sign * params['w_rsi'], 0)

    # 5. Bias + MR agreement
    bias_mr_bull = (fb > 0) & (mr1_sign > 0)
    bias_mr_bear = (fb < 0) & (mr1_sign < 0)
    score += np.where(bias_mr_bull, params['w_agree'], 0)
    score -= np.where(bias_mr_bear, params['w_agree'], 0)

    # 6. CVD
    score += np.where(cvd_z > 0.5, params['w_cvd'], np.where(cvd_z < -0.5, -params['w_cvd'], 0))

    # 7. Strong bias override
    strong_bias = np.abs(fb) >= params['bias_override_threshold']
    score = np.where(strong_bias, fb * params['w_bias'] * params['bias_override_mult'], score)

    return score

def evaluate_score(score, name, k_horizon):
    """Evaluate a scoring signal."""
    out = np.zeros(n_1h, dtype=np.int8)
    for i in range(n_1h - k_horizon):
        out[i] = 1 if c_1h[i + k_horizon] > c_1h[i] else 0
    out[-k_horizon:] = -1
    v = out >= 0

    direction = np.sign(score)
    non_neutral = (direction != 0) & v
    coverage = non_neutral.sum() / v.sum() * 100

    if non_neutral.sum() > 0:
        actual = out[non_neutral] == 1
        pred = score[non_neutral] > 0
        accuracy = (pred == actual).sum() / non_neutral.sum()
    else:
        accuracy = 0.5

    changes = np.sum(direction[1:] != direction[:-1])
    avg_hold = n_1h / max(changes, 1)

    print(f"  {name}")
    print(f"    Coverage:  {coverage:.1f}%")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    N bars:    {non_neutral.sum():,}")
    print(f"    Avg hold:  {avg_hold:.1f}H")
    print(f"    Changes:   {changes:,}")
    return accuracy, coverage

# --- TEST 1: Default parameters ---
print(f"\n--- Default Parameters ---")
default_params = {
    'mr_ema_span1': 20, 'mr_ema_span2': 120,
    'rsi_period': 28, 'rsi_threshold': 20.0,
    'w_bias': 0.3, 'w_mr1': 0.5, 'w_mr2': 0.5,
    'w_rsi': 0.7, 'w_agree': 0.3, 'w_cvd': 0.5,
    'bias_override_threshold': 0.07, 'bias_override_mult': 3.0,
}
score_default = compute_combined_score(default_params)
evaluate_score(score_default, "Default params (K=2)", K)

# --- TEST 2: TPE Optimal parameters (from memory) ---
print(f"\n--- TPE Optimal Parameters (2000 trial) ---")
tpe_params = {
    'mr_ema_span1': 32, 'mr_ema_span2': 56,
    'rsi_period': 18, 'rsi_threshold': 15.0,
    'w_bias': 0.0, 'w_mr1': 0.9, 'w_mr2': 0.6,
    'w_rsi': 1.0, 'w_agree': 1.4, 'w_cvd': 1.0,
    'bias_override_threshold': 0.04, 'bias_override_mult': 3.0,
}
score_tpe = compute_combined_score(tpe_params)
evaluate_score(score_tpe, "TPE optimal (K=2)", K)

# --- TEST 3: K=12 (original 12H lookahead) ---
print(f"\n--- K=12 (12H lookahead) ---")
evaluate_score(score_tpe, "TPE optimal (K=12)", 12)

# ═══════════════════════════════════════════════════════════════
# STEP 3: Walk-Forward Validation (Rust)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STEP 3: Walk-Forward Validation (Rust)")
print(f"{'='*60}")

# TPE optimal Group A
group_a = np.array([
    26,   # cvd_micro_window
    96,   # cvd_macro_window
    24,   # vol_micro_window
    240,  # vol_macro_window
    10,   # imbalance_ema_span
    528,  # atr_pct_window
    480,  # oi_change_window
    1000, # quant_window
    7,    # quantile_count
    2,    # k_horizon
    180,  # min_sample_size
    0.045,# min_edge
    15,   # prior_strength
    0.05, # fdr_alpha
    2,    # temporal_min_segments
    0,    # temporal_max_reversals
    0.55, # min_noise_stability
    90,   # ensemble_min_n
], dtype=np.float64)

# TPE optimal Group B (25 values including BTC params)
group_b = np.array([
    32,   # mr_ema_span1
    56,   # mr_ema_span2
    18,   # rsi_period
    15.0, # rsi_threshold
    0.0,  # w_bias
    0.9,  # w_mr1
    0.6,  # w_mr2
    1.0,  # w_rsi
    1.4,  # w_agree
    1.0,  # w_cvd
    0.04, # bias_override_threshold
    3.0,  # bias_override_mult
    0.25, # sweep_scale
    0.50, # sweep_aligned_weight
    0.60, # sweep_conflict_mult
    264,  # regime_dir_lookback
    1.75, # trending_threshold
    0.95, # high_vol_threshold
    48,   # regime_shift_lookback
    0.80, # regime_shift_penalty
    24,   # btc_mom_window (unused, no BTC data)
    0.0,  # w_btc_mom
    12,   # btc_lead_window
    0.0,  # w_btc_lead
    0.0,  # w_btc_cvd
], dtype=np.float64)

t0 = time.time()
wf = rust_engine.bias_engine_scoring_wf(
    c_1h, h_1h, l_1h, bv_1h, sv_1h, oi_1h,
    group_a, group_b,
    None, None, None  # No BTC data
)
elapsed = time.time() - t0

print(f"  Done in {elapsed:.1f}s")
print(f"  WF chunks:         {wf['n_chunks']}")
print(f"  Overall accuracy:  {wf['overall_accuracy']:.4f} ({wf['overall_accuracy']*100:.2f}%)")
print(f"  Accuracy std:      {wf['accuracy_std']:.4f}")
print(f"  Total bars:        {wf['total_bars']:,}")
print(f"  Chunk accuracies:  {wf['chunk_accuracies']}")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Bias Strength Analysis
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STEP 4: Signal Strength Analysis")
print(f"{'='*60}")

score = score_tpe
direction = np.sign(score)

print(f"\n  Score distribution:")
for thresh in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    mask = (np.abs(score) > thresh) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = score[mask] > 0
        acc = (pred == actual).sum() / mask.sum()
        cov = mask.sum() / valid.sum() * 100
        print(f"    |score| > {thresh:.1f}: {mask.sum():>6,} bars ({cov:>5.1f}%), acc={acc:.4f}")

print(f"\n  By depth (from bias engine):")
for depth in range(4):
    mask = (md == depth) & valid
    if mask.sum() > 100:
        actual = outcomes[mask] == 1
        pred = score[mask] > 0
        acc = (pred == actual).sum() / mask.sum()
        print(f"    Depth {depth}: {mask.sum():>6,} bars, acc={acc:.4f}")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
non_neutral = (direction != 0) & valid
coverage = non_neutral.sum() / valid.sum() * 100
actual = outcomes[non_neutral] == 1
pred = score[non_neutral] > 0
accuracy = (pred == actual).sum() / non_neutral.sum()

print(f"  Coverage:          {coverage:.1f}%")
print(f"  Accuracy (K=2):    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  WF Accuracy:       {wf['overall_accuracy']:.4f} ({wf['overall_accuracy']*100:.2f}%)")
print(f"  Bias engine acc:   {r['accuracy']['direction_accuracy']:.4f}")
print(f"  Validated states:  {r['n_validated']}")
print(f"  Non-neutral bars:  {non_neutral.sum():,} / {valid.sum():,}")
