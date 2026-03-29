"""1H 5Y — Where are patterns lost? Full funnel analysis."""
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

print(f"1H: {n_1h} bars, 5 years")

# Run full analysis (Steps 1-4) to see funnel
r = rust_engine.bias_engine_full(ts_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h)

print(f"\n=== SIGNIFICANCE FUNNEL ===")
f = r['significance_funnel']
for k, v in f.items():
    print(f"  {k}: {v}")

print(f"\n=== ROBUSTNESS ===")
rb = r['robustness_summary']
for k, v in rb.items():
    print(f"  {k}: {v}")

print(f"\nValidated: {r['n_validated']}")
print(f"  D1: {r['val_depth1']}, D2: {r['val_depth2']}, D3: {r['val_depth3']}")
print(f"Significant: {r['n_significant']}")
print(f"  D1: {r['sig_depth1']}, D2: {r['sig_depth2']}, D3: {r['sig_depth3']}")

print(f"\n=== VALIDATED STATES ===")
for s in r['validated_states'][:30]:
    print(f"  {s['state']:<45} d={s['depth']} N={s['n_total']:>5} bias={s['bias']:>+.4f} perm_p={s['perm_p_value']:.4f} noise={s['noise_stability']:.2f}")

print(f"\n=== REJECTED (top 20) ===")
for s in r['rejected_states_top20']:
    reasons = s.get('rejection_reasons', [])
    print(f"  {s['state']:<45} d={s['depth']} N={s['n_total']:>5} bias={s['bias']:>+.4f} {', '.join(reasons)}")

# Bull vs bear
bull = [s for s in r['validated_states'] if s['bias'] > 0]
bear = [s for s in r['validated_states'] if s['bias'] < 0]
print(f"\nBull states: {len(bull)}, Bear states: {len(bear)}")
