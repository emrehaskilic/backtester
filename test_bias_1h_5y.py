"""
Bias Engine on 1H bars — 5 YEAR data (43K bars)
"""
import time
import numpy as np
import pandas as pd
import rust_engine

df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
print(f"5m data: {len(df)} bars")

# Aggregate 5m -> 1H
PERIOD = 12
n_1h = len(df) // PERIOD

ts = df["open_time"].values
o = df["open"].values
h = df["high"].values
l = df["low"].values
c = df["close"].values
bv = df["buy_vol"].values
sv = df["sell_vol"].values
oi = df["open_interest"].values

ts_1h = np.zeros(n_1h, dtype=np.uint64)
o_1h = np.zeros(n_1h, dtype=np.float64)
h_1h = np.zeros(n_1h, dtype=np.float64)
l_1h = np.zeros(n_1h, dtype=np.float64)
c_1h = np.zeros(n_1h, dtype=np.float64)
bv_1h = np.zeros(n_1h, dtype=np.float64)
sv_1h = np.zeros(n_1h, dtype=np.float64)
oi_1h = np.zeros(n_1h, dtype=np.float64)

for i in range(n_1h):
    s = i * PERIOD
    e = s + PERIOD
    ts_1h[i] = ts[s]
    o_1h[i] = o[s]
    h_1h[i] = h[s:e].max()
    l_1h[i] = l[s:e].min()
    c_1h[i] = c[e - 1]
    bv_1h[i] = bv[s:e].sum()
    sv_1h[i] = sv[s:e].sum()
    oi_1h[i] = oi[e - 1]

print(f"1H data: {n_1h} bars ({n_1h/24/365:.1f} years)")
print(f"Date range: {pd.to_datetime(ts_1h[0], unit='ms')} to {pd.to_datetime(ts_1h[-1], unit='ms')}")

# Run bias engine
print(f"\nRunning bias engine on 1H (5 year)...")
t0 = time.time()
r = rust_engine.bias_engine_compute_bias(
    ts_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

print(f"\nn_bars: {r['n_bars']}")
print(f"n_validated: {r['n_validated']}")
print(f"baseline_bull_rate: {r['baseline_bull_rate']}")

cov = r['coverage']
print(f"\nCoverage: {cov['coverage_pct']}%")
print(f"  Depth-3: {cov['depth3_pct']}%")
print(f"  Depth-2: {cov['depth2_pct']}%")
print(f"  Depth-1: {cov['depth1_pct']}%")
print(f"  Fallback: {cov['fallback_pct']}%")

acc = r['accuracy']
print(f"\nDirection accuracy: {acc['direction_accuracy']:.4f}")
print(f"Strong accuracy: {acc['strong_signal_accuracy']:.4f}")
print(f"N strong: {acc['n_strong_bars']}")

reg = r['regime']
print(f"\nRegime: trending={reg['trending_pct']}%, MR={reg['mean_reverting_pct']}%, HV={reg['high_vol_pct']}%")

fb = np.array(r['final_bias'])
conf = np.array(r['confidence'])
md = np.array(r['matched_depth'])
d = np.array(r['direction'])

# Outcomes (K=12 = 12H lookahead)
K = 12
outcomes = np.zeros(len(c_1h), dtype=np.int8)
for i in range(len(c_1h) - K):
    outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
outcomes[-K:] = -1
valid = outcomes >= 0

print(f"\n=== ACCURACY BY BIAS STRENGTH ===")
for threshold in [0.01, 0.03, 0.05, 0.10, 0.15]:
    mask = (np.abs(fb) > threshold) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = fb[mask] > 0
        acc_val = (pred == actual).sum() / mask.sum()
        print(f"  |bias| > {threshold:.2f}: {mask.sum():6d} bars, accuracy={acc_val:.4f}")

print(f"\n=== ACCURACY BY DEPTH ===")
for depth in range(4):
    mask = (md == depth) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = fb[mask] > 0
        acc_val = (pred == actual).sum() / mask.sum()
        print(f"  Depth {depth}: {mask.sum():6d} bars ({mask.sum()/valid.sum()*100:.1f}%), accuracy={acc_val:.4f}")

n_bull = (d == 1).sum()
n_neutral = (d == 0).sum()
n_bear = (d == -1).sum()
print(f"\nDirection: Bull={n_bull} ({n_bull/len(d)*100:.1f}%), "
      f"Neutral={n_neutral} ({n_neutral/len(d)*100:.1f}%), "
      f"Bear={n_bear} ({n_bear/len(d)*100:.1f}%)")

changes = np.sum(d[1:] != d[:-1])
print(f"Direction changes: {changes} ({changes/len(d)*100:.1f}%)")
print(f"Avg hold: {len(d)/max(changes,1):.1f}H")

# Non-neutral accuracy
non_neutral = (d != 0) & valid
if non_neutral.sum() > 0:
    actual = outcomes[non_neutral] == 1
    pred = fb[non_neutral] > 0
    acc_val = (pred == actual).sum() / non_neutral.sum()
    print(f"\nNon-neutral bars: {non_neutral.sum()} ({non_neutral.sum()/valid.sum()*100:.1f}%)")
    print(f"Non-neutral accuracy: {acc_val:.4f}")
