"""
Strong signal carry-forward: use strong signals as direction anchors,
carry direction between them.
"""
import numpy as np
import pandas as pd
import rust_engine

df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12
n_1h = len(df) // PERIOD
ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

ts_1h = np.zeros(n_1h, dtype=np.uint64)
o_1h = np.zeros(n_1h, dtype=np.float64)
h_1h = np.zeros(n_1h, dtype=np.float64)
l_1h = np.zeros(n_1h, dtype=np.float64)
c_1h = np.zeros(n_1h, dtype=np.float64)
bv_1h = np.zeros(n_1h, dtype=np.float64)
sv_1h = np.zeros(n_1h, dtype=np.float64)
oi_1h = np.zeros(n_1h, dtype=np.float64)

for i in range(n_1h):
    s, e = i * PERIOD, i * PERIOD + PERIOD
    ts_1h[i] = ts[s]; o_1h[i] = o[s]; h_1h[i] = h[s:e].max()
    l_1h[i] = l[s:e].min(); c_1h[i] = c[e-1]
    bv_1h[i] = bv[s:e].sum(); sv_1h[i] = sv[s:e].sum(); oi_1h[i] = oi[e-1]

print(f"1H: {n_1h} bars, running bias engine...")
r = rust_engine.bias_engine_compute_bias(ts_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h)

fb = np.array(r['final_bias'])

# Outcomes (K=12 = 12H)
K = 12
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
outcomes[-K:] = -1
valid = outcomes >= 0

# === CARRY-FORWARD STRATEGY ===
# Test multiple strong thresholds
for strong_thresh in [0.03, 0.04, 0.05, 0.07, 0.10]:
    carried_dir = np.zeros(n_1h, dtype=np.float64)
    current_dir = 0.0
    bars_since_strong = 0

    for i in range(n_1h):
        if abs(fb[i]) >= strong_thresh:
            current_dir = 1.0 if fb[i] > 0 else -1.0
            bars_since_strong = 0
        else:
            bars_since_strong += 1
        carried_dir[i] = current_dir

    # Accuracy
    mask = (carried_dir != 0) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = carried_dir[mask] > 0
        acc = (pred == actual).sum() / mask.sum()
        coverage = mask.sum() / valid.sum() * 100

        # Avg hold time between direction changes
        changes = np.sum(carried_dir[1:] != carried_dir[:-1])
        avg_hold = n_1h / max(changes, 1)

        print(f"  thresh={strong_thresh:.2f}: coverage={coverage:.1f}%, acc={acc:.4f}, "
              f"bars={mask.sum()}, changes={changes}, avg_hold={avg_hold:.1f}H")

# === CARRY-FORWARD + TIME DECAY ===
print(f"\n=== WITH TIME DECAY (halve confidence after N hours) ===")
for strong_thresh, max_carry in [(0.05, 12), (0.05, 24), (0.05, 48), (0.05, 72), (0.05, 168)]:
    carried_dir = np.zeros(n_1h, dtype=np.float64)
    current_dir = 0.0
    bars_since = 0

    for i in range(n_1h):
        if abs(fb[i]) >= strong_thresh:
            current_dir = 1.0 if fb[i] > 0 else -1.0
            bars_since = 0
        else:
            bars_since += 1
            if bars_since > max_carry:
                current_dir = 0.0  # expired
        carried_dir[i] = current_dir

    mask = (carried_dir != 0) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = carried_dir[mask] > 0
        acc = (pred == actual).sum() / mask.sum()
        coverage = mask.sum() / valid.sum() * 100
        print(f"  thresh={strong_thresh}, max_carry={max_carry}H: coverage={coverage:.1f}%, acc={acc:.4f}")

# === BEST COMBO: carry + use raw bias when available ===
print(f"\n=== HYBRID: strong carry + raw bias boost ===")
strong_thresh = 0.05
carried_dir = np.zeros(n_1h, dtype=np.float64)
hybrid_conf = np.zeros(n_1h, dtype=np.float64)  # confidence
current_dir = 0.0
bars_since = 0

for i in range(n_1h):
    if abs(fb[i]) >= strong_thresh:
        current_dir = 1.0 if fb[i] > 0 else -1.0
        bars_since = 0
        hybrid_conf[i] = abs(fb[i])
    else:
        bars_since += 1
        # If raw bias agrees with carry direction, boost confidence
        if current_dir != 0 and np.sign(fb[i]) == current_dir:
            hybrid_conf[i] = abs(fb[i]) + 0.02  # small boost for agreement
        else:
            hybrid_conf[i] = max(0, 0.05 - bars_since * 0.001)  # decay

        # If strong opposite signal from raw bias, consider flipping
        if abs(fb[i]) >= 0.03 and np.sign(fb[i]) != current_dir and current_dir != 0:
            # Weaker opposite signal after long carry → flip
            if bars_since > 6:
                current_dir = 1.0 if fb[i] > 0 else -1.0
                bars_since = 0

    carried_dir[i] = current_dir

mask = (carried_dir != 0) & valid
actual = outcomes[mask] == 1
pred = carried_dir[mask] > 0
acc = (pred == actual).sum() / mask.sum()
coverage = mask.sum() / valid.sum() * 100
changes = np.sum(carried_dir[1:] != carried_dir[:-1])
avg_hold = n_1h / max(changes, 1)

print(f"  Coverage: {coverage:.1f}%")
print(f"  Accuracy: {acc:.4f}")
print(f"  Direction changes: {changes}")
print(f"  Avg hold: {avg_hold:.1f}H")

# Accuracy by confidence buckets
for conf_min in [0.01, 0.03, 0.05, 0.08]:
    cmask = mask & (hybrid_conf > conf_min)
    if cmask.sum() > 0:
        a = outcomes[cmask] == 1
        p = carried_dir[cmask] > 0
        ca = (p == a).sum() / cmask.sum()
        print(f"  conf>{conf_min}: {cmask.sum()} bars, acc={ca:.4f}")
