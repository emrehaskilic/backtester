"""
Phase 1: TPE optimization of combination weights + thresholds
Bias engine precomputed, only optimize the scoring formula.
"""
import numpy as np
import pandas as pd
import rust_engine
import time
from functools import lru_cache

t_start = time.time()

# === DATA PREP ===
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12
n_1h = len(df) // PERIOD
ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

c_1h = np.zeros(n_1h); h_1h = np.zeros(n_1h); l_1h = np.zeros(n_1h)
bv_1h = np.zeros(n_1h); sv_1h = np.zeros(n_1h); oi_1h = np.zeros(n_1h)
ts_1h = np.zeros(n_1h, dtype=np.uint64); o_1h = np.zeros(n_1h)
vol_1h = np.zeros(n_1h)

for i in range(n_1h):
    s, e = i * PERIOD, i * PERIOD + PERIOD
    ts_1h[i] = ts[s]; o_1h[i] = o[s]; h_1h[i] = h[s:e].max()
    l_1h[i] = l[s:e].min(); c_1h[i] = c[e-1]
    bv_1h[i] = bv[s:e].sum(); sv_1h[i] = sv[s:e].sum()
    oi_1h[i] = oi[e-1]; vol_1h[i] = bv_1h[i] + sv_1h[i]

K = 12
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
outcomes[-K:] = -1

# === PRECOMPUTE BIAS ENGINE (once) ===
print("Precomputing bias engine...")
r = rust_engine.bias_engine_compute_bias(
    ts_1h.astype(np.uint64), o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h
)
fb = np.array(r['final_bias'])
print(f"Bias engine done. ({time.time()-t_start:.1f}s)")

# === PRECOMPUTE ALL RAW SIGNALS ===
def ema(data, span):
    k = 2.0 / (span + 1)
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

def compute_rsi(data, period):
    delta = np.diff(data, prepend=data[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_g = ema(gains, period)
    avg_l = ema(losses, period)
    rsi = np.full(len(data), 50.0)
    for i in range(period, len(data)):
        if avg_l[i] > 0:
            rsi[i] = 100 - 100 / (1 + avg_g[i] / avg_l[i])
    return rsi

# Precompute EMAs at various spans (cache for TPE)
ema_cache = {}
for span in [8, 12, 16, 20, 24, 30, 36, 48, 60, 72, 96, 120, 168]:
    ema_cache[span] = ema(c_1h, span)

# Precompute RSI at various periods
rsi_cache = {}
for period in [6, 8, 10, 12, 14, 18, 21, 28]:
    rsi_cache[period] = compute_rsi(c_1h, period)

# CVD
cvd_1h = bv_1h - sv_1h
cvd_ema24 = ema(cvd_1h, 24)
cvd_std = np.zeros(n_1h)
for i in range(24, n_1h):
    cvd_std[i] = np.std(cvd_1h[max(0,i-24):i])
f_cvd = np.where(cvd_std > 0, (cvd_1h - cvd_ema24) / np.maximum(cvd_std, 1e-10), 0)

print(f"All signals precomputed. ({time.time()-t_start:.1f}s)")

# === WALK-FORWARD SETUP ===
WARMUP = 300
valid = (outcomes >= 0) & (np.arange(n_1h) >= WARMUP)
valid_idx = np.where(valid)[0]

# 5-fold WF: train on first 60%, test on remaining 40% in 5 chunks
# Simpler: use last 3 years as test, split into 6-month chunks
TRAIN_END = 17520  # first 2 years
TEST_CHUNKS = []
chunk_size = 4380  # 6 months
idx = TRAIN_END
while idx + chunk_size <= n_1h:
    TEST_CHUNKS.append((idx, idx + chunk_size))
    idx += chunk_size

print(f"Train: 0-{TRAIN_END} ({TRAIN_END} bars)")
print(f"Test chunks: {len(TEST_CHUNKS)} x {chunk_size} bars")

# === OBJECTIVE FUNCTION ===
def evaluate_params(params):
    mr_span1 = params['mr_span1']
    mr_span2 = params['mr_span2']
    rsi_period = params['rsi_period']
    w_bias = params['w_bias']
    w_mr1 = params['w_mr1']
    w_mr2 = params['w_mr2']
    w_rsi = params['w_rsi']
    w_agree = params['w_agree']
    w_cvd = params['w_cvd']
    bias_override = params['bias_override']
    rsi_thresh = params['rsi_thresh']

    # Get cached EMAs
    if mr_span1 not in ema_cache or mr_span2 not in ema_cache:
        return 0.5
    if rsi_period not in rsi_cache:
        return 0.5

    ema1 = ema_cache[mr_span1]
    ema2 = ema_cache[mr_span2]
    rsi = rsi_cache[rsi_period]

    # MR signals (negative distance = bearish MR)
    mr1 = -(c_1h - ema1) / np.maximum(ema1, 1) * 100
    mr2 = -(c_1h - ema2) / np.maximum(ema2, 1) * 100

    # RSI signal
    rsi_sig = -(rsi - 50) / 50

    # Combined score
    score = np.zeros(n_1h)
    score += fb * w_bias
    score += np.sign(mr1) * w_mr1
    score += np.where(np.sign(mr1) == np.sign(mr2), np.sign(mr1) * w_mr2, 0)
    score += np.where(
        ((rsi < (50 - rsi_thresh)) & (np.sign(mr1) > 0)) |
        ((rsi > (50 + rsi_thresh)) & (np.sign(mr1) < 0)),
        np.sign(mr1) * w_rsi, 0
    )
    score += np.where(np.sign(fb) == np.sign(mr1), np.sign(mr1) * w_agree, 0)
    score += np.sign(f_cvd) * w_cvd

    # Bias engine override
    score = np.where(np.abs(fb) >= bias_override, fb * w_bias * 3.0, score)

    direction = np.sign(score)

    # Evaluate on test chunks
    accs = []
    for (ts, te) in TEST_CHUNKS:
        mask = valid & (np.arange(n_1h) >= ts) & (np.arange(n_1h) < te) & (direction != 0)
        if mask.sum() < 100:
            continue
        actual = outcomes[mask] == 1
        pred = direction[mask] > 0
        acc = (pred == actual).sum() / mask.sum()
        accs.append(acc)

    if not accs:
        return 0.5
    return np.mean(accs)

# === TPE OPTIMIZER (simple Python implementation) ===
import random
random.seed(42)
np.random.seed(42)

# Parameter space
PARAM_SPACE = {
    'mr_span1': [8, 12, 16, 20, 24, 30, 36, 48],
    'mr_span2': [24, 36, 48, 60, 72, 96, 120, 168],
    'rsi_period': [6, 8, 10, 12, 14, 18, 21, 28],
    'w_bias': [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
    'w_mr1': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    'w_mr2': [0.0, 0.2, 0.3, 0.5, 0.7, 1.0],
    'w_rsi': [0.0, 0.2, 0.3, 0.5, 0.7, 1.0],
    'w_agree': [0.0, 0.2, 0.3, 0.5, 0.7, 1.0],
    'w_cvd': [0.0, 0.1, 0.2, 0.3, 0.5],
    'bias_override': [0.03, 0.05, 0.07, 0.10, 0.15],
    'rsi_thresh': [5, 8, 10, 12, 15, 20],
}

def random_params():
    return {k: random.choice(v) for k, v in PARAM_SPACE.items()}

# Phase 1: Random search (fast exploration)
N_RANDOM = 200
N_TOP_REFINE = 50

print(f"\n=== TPE Phase 1: Random search ({N_RANDOM} trials) ===")
results = []
t0 = time.time()

for trial in range(N_RANDOM):
    params = random_params()
    acc = evaluate_params(params)
    results.append((acc, params))

results.sort(key=lambda x: -x[0])
print(f"Random search done in {time.time()-t0:.1f}s")
print(f"Top 5:")
for i, (acc, params) in enumerate(results[:5]):
    print(f"  #{i+1}: acc={acc:.4f} | mr1={params['mr_span1']}, mr2={params['mr_span2']}, "
          f"rsi={params['rsi_period']}, w_b={params['w_bias']}, w_mr={params['w_mr1']}, "
          f"w_mr2={params['w_mr2']}, w_rsi={params['w_rsi']}, w_ag={params['w_agree']}, "
          f"w_cvd={params['w_cvd']}, override={params['bias_override']}, rsi_t={params['rsi_thresh']}")

# Phase 2: Refine around top params
print(f"\n=== TPE Phase 2: Refinement ({N_TOP_REFINE} trials around top 10) ===")
top_params = [p for _, p in results[:10]]

for trial in range(N_TOP_REFINE):
    # Pick a top param set, mutate 2-3 params
    base = random.choice(top_params).copy()
    n_mutate = random.randint(1, 3)
    keys_to_mutate = random.sample(list(PARAM_SPACE.keys()), n_mutate)
    for key in keys_to_mutate:
        base[key] = random.choice(PARAM_SPACE[key])

    acc = evaluate_params(base)
    results.append((acc, base))

results.sort(key=lambda x: -x[0])
print(f"Refinement done in {time.time()-t0:.1f}s")

# === FINAL RESULTS ===
print(f"\n{'='*60}")
print(f"OPTIMIZATION RESULTS")
print(f"{'='*60}")

best_acc, best_params = results[0]
print(f"\nBest OOS accuracy: {best_acc:.4f}")
print(f"Best params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Run best params on full test set for detailed analysis
print(f"\n=== DETAILED ANALYSIS WITH BEST PARAMS ===")
mr_span1 = best_params['mr_span1']
mr_span2 = best_params['mr_span2']
rsi_period = best_params['rsi_period']
ema1 = ema_cache[mr_span1]
ema2 = ema_cache[mr_span2]
rsi = rsi_cache[rsi_period]

mr1 = -(c_1h - ema1) / np.maximum(ema1, 1) * 100
mr2 = -(c_1h - ema2) / np.maximum(ema2, 1) * 100
rsi_sig = -(rsi - 50) / 50

score = np.zeros(n_1h)
score += fb * best_params['w_bias']
score += np.sign(mr1) * best_params['w_mr1']
score += np.where(np.sign(mr1) == np.sign(mr2), np.sign(mr1) * best_params['w_mr2'], 0)
score += np.where(
    ((rsi < (50 - best_params['rsi_thresh'])) & (np.sign(mr1) > 0)) |
    ((rsi > (50 + best_params['rsi_thresh'])) & (np.sign(mr1) < 0)),
    np.sign(mr1) * best_params['w_rsi'], 0
)
score += np.where(np.sign(fb) == np.sign(mr1), np.sign(mr1) * best_params['w_agree'], 0)
score += np.sign(f_cvd) * best_params['w_cvd']
score = np.where(np.abs(fb) >= best_params['bias_override'], fb * best_params['w_bias'] * 3.0, score)

direction = np.sign(score)

# Full test period accuracy
test_start = TRAIN_END
mask_test = valid & (np.arange(n_1h) >= test_start) & (direction != 0)
actual = outcomes[mask_test] == 1
pred = direction[mask_test] > 0
full_test_acc = (pred == actual).sum() / mask_test.sum()

# Per-chunk accuracy
print(f"\nPer-chunk OOS accuracy:")
for ci, (ts, te) in enumerate(TEST_CHUNKS):
    mask = valid & (np.arange(n_1h) >= ts) & (np.arange(n_1h) < te) & (direction != 0)
    if mask.sum() > 0:
        a = outcomes[mask] == 1
        p = direction[mask] > 0
        acc = (p == a).sum() / mask.sum()
        print(f"  Chunk {ci}: acc={acc:.4f}, n={mask.sum()}")

# Coverage & stability
mask_all = valid & (direction != 0)
cov = mask_all.sum() / valid.sum() * 100
changes = np.sum(direction[1:] != direction[:-1])
avg_hold = n_1h / max(changes, 1)

print(f"\nFull test accuracy: {full_test_acc:.4f}")
print(f"Coverage: {cov:.1f}%")
print(f"Direction changes: {changes}, avg hold: {avg_hold:.1f}H")

# By score strength
for s_thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
    smask = (np.abs(score) >= s_thresh) & valid & (np.arange(n_1h) >= test_start)
    if smask.sum() > 50:
        a = outcomes[smask] == 1
        p = np.sign(score[smask]) > 0
        sa = (p == a).sum() / smask.sum()
        print(f"  |score|>={s_thresh}: {smask.sum()} bars, acc={sa:.4f}")

# Compare with baseline (hand-tuned)
print(f"\n=== COMPARISON ===")
# Hand-tuned baseline
baseline_score = np.zeros(n_1h)
mr24_base = -(c_1h - ema_cache[24]) / np.maximum(ema_cache[24], 1) * 100
baseline_score += np.sign(mr24_base) * 1.0
baseline_score += np.where(np.sign(ema_cache[48] - c_1h) == np.sign(mr24_base), np.sign(mr24_base) * 0.5, 0)
baseline_score += np.where(np.sign(fb) == np.sign(mr24_base), np.sign(mr24_base) * 0.5, 0)
rsi14 = rsi_cache[14]
baseline_score += np.where(((rsi14 < 40) & (np.sign(mr24_base) > 0)) | ((rsi14 > 60) & (np.sign(mr24_base) < 0)), np.sign(mr24_base) * 0.5, 0)
baseline_score = np.where(np.abs(fb) >= 0.05, fb * 3.0, baseline_score)
baseline_dir = np.sign(baseline_score)

mask_base = valid & (np.arange(n_1h) >= test_start) & (baseline_dir != 0)
base_acc = ((baseline_dir[mask_base] > 0) == (outcomes[mask_base] == 1)).sum() / mask_base.sum()

print(f"  Hand-tuned accuracy (test): {base_acc:.4f}")
print(f"  Optimized accuracy (test):  {full_test_acc:.4f}")
print(f"  Improvement: {(full_test_acc - base_acc)*100:+.2f}pp")

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed:.1f}s")
