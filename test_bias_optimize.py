"""
Bias Engine TPE Optimizer — 38 parameter, 2-phase nested
2000 outer trials on 1H 5Y data
"""
import time
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

print(f"1H data: {n_1h} bars ({n_1h/24/365:.1f} years)")
print(f"Running 2000 outer trial optimization...")
print(f"This will take ~30-45 minutes. Be patient.")

t0 = time.time()
result = rust_engine.bias_engine_optimize(
    ts_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h,
    2000,  # n_outer_trials
    42,    # seed
)
elapsed = time.time() - t0

print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"\n{'='*60}")
print(f"OPTIMIZATION RESULTS")
print(f"{'='*60}")
print(f"Best OOS score: {result['best_score']:.4f}")
print(f"Validated states: {result['n_validated_states']}")
print(f"Trials evaluated: {result['trials_evaluated']}")

print(f"\n-- Group A (Feature/Quantization/Significance/Robustness) --")
for k, v in result['group_a'].items():
    print(f"  {k}: {v}")

print(f"\n-- Group B (MR/RSI/Weights/Regime) --")
for k, v in result['group_b'].items():
    print(f"  {k}: {v}")
