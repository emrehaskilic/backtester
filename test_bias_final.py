"""
Bias Engine: %100 Coverage + ~%53 Accuracy — Final Test

Reproduces the Session 5-6 result:
- Bias engine state mining → validated states
- Combined scoring: MR + RSI + CVD + bias agreement
- 100% coverage (always has a signal)
- ~53% direction accuracy (walk-forward validated)
"""
import time
import numpy as np
import pandas as pd
import rust_engine

# ═══════════════════════════════════════════════════════════════
# DATA: 5m → 1H
# ═══════════════════════════════════════════════════════════════
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
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

# ═══════════════════════════════════════════════════════════════
# BIAS ENGINE
# ═══════════════════════════════════════════════════════════════
print(f"\nRunning bias engine...")
t0 = time.time()
r = rust_engine.bias_engine_compute_bias(
    ts_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h
)
print(f"Done in {time.time()-t0:.1f}s")

fb = np.array(r['final_bias'])
print(f"Validated states: {r['n_validated']}")

# ═══════════════════════════════════════════════════════════════
# COMBINED SCORING — Fixed version with 100% coverage
# ═══════════════════════════════════════════════════════════════

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

# TPE optimal parameters
MR_SPAN1, MR_SPAN2 = 32, 56
RSI_PERIOD, RSI_THRESHOLD = 18, 15.0
W_BIAS, W_MR1, W_MR2, W_RSI = 0.0, 0.9, 0.6, 1.0
W_AGREE, W_CVD = 1.4, 1.0
BIAS_OVERRIDE_THRESH, BIAS_OVERRIDE_MULT = 0.04, 3.0

# Compute indicators
ema1 = ema(c_1h, MR_SPAN1)
ema2 = ema(c_1h, MR_SPAN2)
mr1_sign = np.where(c_1h > ema1, -1.0, 1.0)  # mean reversion
mr2_sign = np.where(c_1h > ema2, -1.0, 1.0)
rsi = compute_rsi(c_1h, RSI_PERIOD)

# CVD z-score
cvd_bar = bv_1h - sv_1h
cvd_ema = ema(cvd_bar, 24)
cvd_std = np.zeros(n_1h)
for i in range(24, n_1h):
    diff = cvd_bar[i-24:i] - cvd_ema[i-24:i]
    cvd_std[i] = np.sqrt(np.mean(diff**2))
cvd_z = np.zeros(n_1h)
for i in range(24, n_1h):
    if cvd_std[i] > 1e-15:
        cvd_z[i] = (cvd_bar[i] - cvd_ema[i]) / cvd_std[i]

# Build combined score
score = np.zeros(n_1h)

# 1. Bias engine (w_bias=0.0 in TPE optimal, so no direct contribution)
score += fb * W_BIAS

# 2. MR primary — ALWAYS non-zero → guarantees 100% coverage
score += mr1_sign * W_MR1

# 3. MR secondary (when agrees)
agree_mr = (mr1_sign == mr2_sign)
score += np.where(agree_mr, mr1_sign * W_MR2, 0)

# 4. RSI confirmation
rsi_oversold = rsi < (50 - RSI_THRESHOLD)
rsi_overbought = rsi > (50 + RSI_THRESHOLD)
rsi_confirm = ((rsi_oversold & (mr1_sign > 0)) | (rsi_overbought & (mr1_sign < 0)))
score += np.where(rsi_confirm, mr1_sign * W_RSI, 0)

# 5. Bias + MR agreement — THE KEY INTERACTION (w=1.4, strongest)
bias_dir = np.sign(fb)
bias_mr_agree = (bias_dir == mr1_sign) & (bias_dir != 0)
score += np.where(bias_mr_agree & (mr1_sign > 0), W_AGREE, 0)
score -= np.where(bias_mr_agree & (mr1_sign < 0), W_AGREE, 0)

# 6. CVD
score += np.where(cvd_z > 0.5, W_CVD, np.where(cvd_z < -0.5, -W_CVD, 0))

# 7. Strong bias override — ONLY when w_bias > 0
#    With w_bias=0.0, override would zero-out the score → skip it
if W_BIAS > 0:
    strong_bias = np.abs(fb) >= BIAS_OVERRIDE_THRESH
    score = np.where(strong_bias, fb * W_BIAS * BIAS_OVERRIDE_MULT, score)

direction = np.sign(score)

# ═══════════════════════════════════════════════════════════════
# EVALUATE ACROSS MULTIPLE K VALUES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS: Combined Scoring (MR+RSI+CVD+Agreement)")
print(f"{'='*60}")

for K in [2, 6, 12]:
    outcomes = np.zeros(n_1h, dtype=np.int8)
    for i in range(n_1h - K):
        outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
    outcomes[-K:] = -1
    valid = outcomes >= 0

    non_neutral = (direction != 0) & valid
    coverage = non_neutral.sum() / valid.sum() * 100

    actual = outcomes[non_neutral] == 1
    pred = score[non_neutral] > 0
    accuracy = (pred == actual).sum() / non_neutral.sum()

    print(f"\n  K={K} ({K}H lookahead):")
    print(f"    Coverage:  {coverage:.1f}%")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    N bars:    {non_neutral.sum():,}")

# ═══════════════════════════════════════════════════════════════
# BIAS ENGINE STANDALONE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  BIAS ENGINE STANDALONE (state mining only)")
print(f"{'='*60}")

for K in [2, 6, 12]:
    outcomes = np.zeros(n_1h, dtype=np.int8)
    for i in range(n_1h - K):
        outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
    outcomes[-K:] = -1
    valid = outcomes >= 0

    non_neutral = (np.array(r['direction']) != 0) & valid
    coverage = non_neutral.sum() / valid.sum() * 100
    actual = outcomes[non_neutral] == 1
    pred = fb[non_neutral] > 0
    accuracy = (pred == actual).sum() / non_neutral.sum()

    print(f"\n  K={K} ({K}H lookahead):")
    print(f"    Coverage:  {coverage:.1f}%")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD (Rust)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  WALK-FORWARD VALIDATION")
print(f"{'='*60}")

# TPE optimal params
group_a = np.array([26, 96, 24, 240, 10, 528, 480, 1000, 7, 2,
                    180, 0.045, 15, 0.05, 2, 0, 0.55, 90], dtype=np.float64)
group_b = np.array([32, 56, 18, 15.0, 0.0, 0.9, 0.6, 1.0, 1.4, 1.0,
                    0.04, 3.0, 0.25, 0.50, 0.60, 264, 1.75, 0.95, 48, 0.80,
                    24, 0.0, 12, 0.0, 0.0], dtype=np.float64)

t0 = time.time()
wf = rust_engine.bias_engine_scoring_wf(
    c_1h, h_1h, l_1h, bv_1h, sv_1h, oi_1h,
    group_a, group_b, None, None, None
)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  WF chunks:       {wf['n_chunks']}")
print(f"  WF accuracy:     {wf['overall_accuracy']:.4f} ({wf['overall_accuracy']*100:.2f}%)")
print(f"  WF acc std:      {wf['accuracy_std']:.4f}")
print(f"  Total bars:      {wf['total_bars']:,}")
print(f"  Chunk accs:      {wf['chunk_accuracies']}")

# Default params for comparison
group_a_def = np.array([12, 288, 12, 288, 12, 288, 288, 2016, 5, 12,
                        100, 0.02, 30, 0.05, 3, 1, 0.80, 50], dtype=np.float64)
group_b_def = np.array([20, 120, 28, 20.0, 0.3, 0.5, 0.5, 0.7, 0.3, 0.5,
                        0.07, 3.0, 0.15, 0.30, 0.70, 72, 1.5, 0.90, 48, 0.50,
                        24, 0.5, 12, 0.5, 0.3], dtype=np.float64)

wf_def = rust_engine.bias_engine_scoring_wf(
    c_1h, h_1h, l_1h, bv_1h, sv_1h, oi_1h,
    group_a_def, group_b_def, None, None, None
)
print(f"\n  Default params WF:")
print(f"  WF accuracy:     {wf_def['overall_accuracy']:.4f} ({wf_def['overall_accuracy']*100:.2f}%)")
print(f"  WF acc std:      {wf_def['accuracy_std']:.4f}")
print(f"  Chunk accs:      {wf_def['chunk_accuracies']}")

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
K_final = 2
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K_final):
    outcomes[i] = 1 if c_1h[i + K_final] > c_1h[i] else 0
outcomes[-K_final:] = -1
valid = outcomes >= 0
non_neutral = (direction != 0) & valid
actual = outcomes[non_neutral] == 1
pred = score[non_neutral] > 0
final_acc = (pred == actual).sum() / non_neutral.sum()
final_cov = non_neutral.sum() / valid.sum() * 100

print(f"\n{'='*60}")
print(f"  FINAL SUMMARY")
print(f"{'='*60}")
print(f"  Bias Engine (standalone, K=12): {r['accuracy']['direction_accuracy']:.4f} acc, {r['coverage']['coverage_pct']}% cov")
print(f"  Combined Scoring (K=2):         {final_acc:.4f} acc, {final_cov:.1f}% cov")
print(f"  Walk-Forward (TPE optimal):     {wf['overall_accuracy']:.4f} acc")
print(f"  Walk-Forward (default):         {wf_def['overall_accuracy']:.4f} acc")
print(f"  Validated states:               {r['n_validated']}")
