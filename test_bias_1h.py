"""
Quick analysis: Bias Engine on 1H bars
Aggregate 5m -> 1H, then run bias engine directly on 1H data.
Windows auto-scale: micro=12H, macro=288H, quantile=2016H (84 days)
"""
import time
import numpy as np
import pandas as pd
import rust_engine

# Load 5m data
df = pd.read_parquet("data/ETHUSDT_5m_cvd_oi_11mo.parquet")
print(f"5m data: {len(df)} bars")

# Aggregate 5m -> 1H (12 bars per 1H candle)
PERIOD = 12  # 5m bars per 1H
n_1h = len(df) // PERIOD

ts_5m = df["open_time"].values
open_5m = df["open"].values
high_5m = df["high"].values
low_5m = df["low"].values
close_5m = df["close"].values
buy_vol_5m = df["buy_vol"].values
sell_vol_5m = df["sell_vol"].values
oi_5m = df["open_interest"].values

# Build 1H arrays
ts_1h = np.zeros(n_1h, dtype=np.uint64)
open_1h = np.zeros(n_1h, dtype=np.float64)
high_1h = np.zeros(n_1h, dtype=np.float64)
low_1h = np.zeros(n_1h, dtype=np.float64)
close_1h = np.zeros(n_1h, dtype=np.float64)
buy_vol_1h = np.zeros(n_1h, dtype=np.float64)
sell_vol_1h = np.zeros(n_1h, dtype=np.float64)
oi_1h = np.zeros(n_1h, dtype=np.float64)

for i in range(n_1h):
    s = i * PERIOD
    e = s + PERIOD
    ts_1h[i] = ts_5m[s]
    open_1h[i] = open_5m[s]
    high_1h[i] = high_5m[s:e].max()
    low_1h[i] = low_5m[s:e].min()
    close_1h[i] = close_5m[e - 1]
    buy_vol_1h[i] = buy_vol_5m[s:e].sum()
    sell_vol_1h[i] = sell_vol_5m[s:e].sum()
    oi_1h[i] = oi_5m[e - 1]  # last OI value

print(f"1H data: {n_1h} bars")
print(f"Date range: {pd.to_datetime(ts_1h[0], unit='ms')} to {pd.to_datetime(ts_1h[-1], unit='ms')}")

# Run bias engine on 1H data
print(f"\nRunning bias engine on 1H bars...")
t0 = time.time()
r = rust_engine.bias_engine_compute_bias(
    ts_1h, open_1h, high_1h, low_1h, close_1h, buy_vol_1h, sell_vol_1h, oi_1h
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

cal = r['calibration']
print(f"\nBrier uncal={cal['brier_uncalibrated']:.4f}, cal={cal['brier_calibrated']:.4f}")

# Per-bar analysis
fb = np.array(r['final_bias'])
conf = np.array(r['confidence'])
md = np.array(r['matched_depth'])
d = np.array(r['direction'])

print(f"\nfinal_bias: mean={fb.mean():.4f}, std={fb.std():.4f}, min={fb.min():.4f}, max={fb.max():.4f}")
print(f"confidence: mean={conf.mean():.4f}")

# Compute 1H outcomes (K=12 means 12H lookahead now)
outcomes = np.zeros(len(close_1h), dtype=np.int8)
K = 12
for i in range(len(close_1h) - K):
    outcomes[i] = 1 if close_1h[i + K] > close_1h[i] else 0
outcomes[-K:] = -1
valid = outcomes >= 0

# Accuracy by bias strength
print(f"\n=== ACCURACY BY BIAS STRENGTH ===")
for threshold in [0.01, 0.03, 0.05, 0.10, 0.15]:
    mask = (np.abs(fb) > threshold) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = fb[mask] > 0
        acc_val = (pred == actual).sum() / mask.sum()
        print(f"  |bias| > {threshold:.2f}: {mask.sum():5d} bars, accuracy={acc_val:.4f}")

# Accuracy by depth
print(f"\n=== ACCURACY BY DEPTH ===")
for depth in range(4):
    mask = (md == depth) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = fb[mask] > 0
        acc_val = (pred == actual).sum() / mask.sum()
        print(f"  Depth {depth}: {mask.sum():5d} bars, accuracy={acc_val:.4f}")

# Direction distribution
n_bull = (d == 1).sum()
n_neutral = (d == 0).sum()
n_bear = (d == -1).sum()
print(f"\nDirection: Bull={n_bull} ({n_bull/len(d)*100:.1f}%), "
      f"Neutral={n_neutral} ({n_neutral/len(d)*100:.1f}%), "
      f"Bear={n_bear} ({n_bear/len(d)*100:.1f}%)")

# Bias stability: how often does direction change per bar?
changes = np.sum(d[1:] != d[:-1])
print(f"\nDirection changes: {changes} ({changes/len(d)*100:.1f}% of bars)")
print(f"Avg bars between change: {len(d)/max(changes,1):.1f} bars = {len(d)/max(changes,1):.1f}H")
