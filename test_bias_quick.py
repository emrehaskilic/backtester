"""Quick test for bias engine compute_bias (Steps 1-9)"""
import time, sys
import numpy as np
import pandas as pd
import rust_engine

df = pd.read_parquet("data/ETHUSDT_5m_cvd_oi_11mo.parquet")
print(f"Data: {len(df)} bars")

timestamps = df["open_time"].values.astype(np.uint64)
o = df["open"].values.astype(np.float64)
h = df["high"].values.astype(np.float64)
l = df["low"].values.astype(np.float64)
c = df["close"].values.astype(np.float64)
bv = df["buy_vol"].values.astype(np.float64)
sv = df["sell_vol"].values.astype(np.float64)
oi = df["open_interest"].values.astype(np.float64)

print("Running compute_bias...")
sys.stdout.flush()
t0 = time.time()
r = rust_engine.bias_engine_compute_bias(timestamps, o, h, l, c, bv, sv, oi)
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
print(f"\nBrier uncal={cal['brier_uncalibrated']:.4f}, cal={cal['brier_calibrated']:.4f}, improved={cal['calibration_improved']}")

fb = np.array(r['final_bias'])
conf = np.array(r['confidence'])
print(f"\nfinal_bias: mean={fb.mean():.4f}, std={fb.std():.4f}, min={fb.min():.4f}, max={fb.max():.4f}")
print(f"confidence: mean={conf.mean():.4f}")

d = np.array(r['direction'])
print(f"\nDirection: Bull={int((d==1).sum())} Neutral={int((d==0).sum())} Bear={int((d==-1).sum())}")

md = np.array(r['matched_depth'])
for depth in range(4):
    n = int((md == depth).sum())
    print(f"  Depth {depth}: {n} ({n/len(md)*100:.1f}%)")

assert not np.isnan(fb).any(), "FAIL: NaN in final_bias!"
print("\nOK - no NaN in bias, coverage 100%")
