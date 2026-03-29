"""Deeper analysis of bias engine output"""
import numpy as np
import pandas as pd
import rust_engine

df = pd.read_parquet("data/ETHUSDT_5m_cvd_oi_11mo.parquet")
timestamps = df["open_time"].values.astype(np.uint64)
o = df["open"].values.astype(np.float64)
h = df["high"].values.astype(np.float64)
l = df["low"].values.astype(np.float64)
c = df["close"].values.astype(np.float64)
bv = df["buy_vol"].values.astype(np.float64)
sv = df["sell_vol"].values.astype(np.float64)
oi = df["open_interest"].values.astype(np.float64)

print("Running compute_bias...")
r = rust_engine.bias_engine_compute_bias(timestamps, o, h, l, c, bv, sv, oi)

fb = np.array(r['final_bias'])
conf = np.array(r['confidence'])
sb = np.array(r['state_bias'])
swb = np.array(r['sweep_bias'])
md = np.array(r['matched_depth'])
d = np.array(r['direction'])

# Compute outcomes (K=12 lookahead)
outcomes = np.zeros(len(c), dtype=np.int8)
for i in range(len(c) - 12):
    outcomes[i] = 1 if c[i + 12] > c[i] else 0
outcomes[-12:] = -1  # sentinel

valid = outcomes >= 0

print(f"\n=== NON-NEUTRAL ACCURACY ===")
non_neutral = (d != 0) & valid
actual_bull = outcomes[non_neutral] == 1
pred_bull = fb[non_neutral] > 0
nn_correct = (pred_bull == actual_bull).sum()
nn_total = non_neutral.sum()
print(f"Non-neutral bars: {nn_total} ({nn_total/valid.sum()*100:.1f}%)")
print(f"Non-neutral accuracy: {nn_correct/nn_total:.4f}")

print(f"\n=== ACCURACY BY BIAS STRENGTH ===")
for threshold in [0.01, 0.03, 0.05, 0.10, 0.15]:
    mask = (np.abs(fb) > threshold) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = fb[mask] > 0
        acc = (pred == actual).sum() / mask.sum()
        print(f"  |bias| > {threshold:.2f}: {mask.sum():6d} bars, accuracy={acc:.4f}")

print(f"\n=== ACCURACY BY MATCHED DEPTH ===")
for depth in range(4):
    mask = (md == depth) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = fb[mask] > 0
        acc = (pred == actual).sum() / mask.sum()
        print(f"  Depth {depth}: {mask.sum():6d} bars, accuracy={acc:.4f}")

print(f"\n=== STATE BIAS vs SWEEP BIAS ===")
print(f"  state_bias: mean={sb.mean():.4f}, std={sb.std():.4f}")
print(f"  sweep_bias: mean={swb.mean():.4f}, std={swb.std():.4f}")
print(f"  sweep non-zero: {(np.abs(swb) > 0.001).sum()} ({(np.abs(swb) > 0.001).sum()/len(swb)*100:.1f}%)")

# State bias accuracy
sb_mask = (np.abs(sb) > 0.03) & valid
if sb_mask.sum() > 0:
    actual = outcomes[sb_mask] == 1
    pred = sb[sb_mask] > 0
    acc = (pred == actual).sum() / sb_mask.sum()
    print(f"\n  State bias |>0.03| accuracy: {acc:.4f} ({sb_mask.sum()} bars)")

# Sweep bias accuracy
sw_mask = (np.abs(swb) > 0.01) & valid
if sw_mask.sum() > 0:
    actual = outcomes[sw_mask] == 1
    pred = swb[sw_mask] > 0
    acc = (pred == actual).sum() / sw_mask.sum()
    print(f"  Sweep bias |>0.01| accuracy: {acc:.4f} ({sw_mask.sum()} bars)")

print(f"\n=== CONFIDENCE ANALYSIS ===")
for conf_thresh in [0.3, 0.5, 0.7]:
    mask = (conf > conf_thresh) & (np.abs(fb) > 0.03) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = fb[mask] > 0
        acc = (pred == actual).sum() / mask.sum()
        print(f"  conf>{conf_thresh:.1f} & |bias|>0.03: {mask.sum():6d} bars, accuracy={acc:.4f}")

print(f"\n=== BIAS DISTRIBUTION ===")
for b in np.arange(-0.25, 0.26, 0.05):
    mask = (fb >= b) & (fb < b + 0.05)
    print(f"  [{b:+.2f}, {b+0.05:+.2f}): {mask.sum():6d} bars ({mask.sum()/len(fb)*100:.1f}%)")
