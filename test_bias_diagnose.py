"""
Session 4 — Root Cause Analysis
Why does in-sample state bias (%53.3) not carry to OOS (%50.3)?

Hypotheses:
1. Depth-3 states are too specific (overfit)
2. State edge changes direction over time
3. Baseline fallback dominates and washes out edge
4. Calibration is breaking things in some folds
"""
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

# Run full analysis (Steps 1-4) to get raw state data
print("Running full analysis (Steps 1-4)...")
r = rust_engine.bias_engine_full(timestamps, o, h, l, c, bv, sv, oi)

print(f"\n=== VALIDATED STATES ANALYSIS ===")
print(f"Total validated: {r['n_validated']}")
print(f"  Depth-1: {r['val_depth1']}")
print(f"  Depth-2: {r['val_depth2']}")
print(f"  Depth-3: {r['val_depth3']}")

# Analyze validated states
vs = r['validated_states']
print(f"\n=== TOP VALIDATED STATES ===")
print(f"{'State':<45} {'Depth':>5} {'N':>6} {'Bias':>7} {'Prob':>7} {'Perm_p':>7} {'Noise':>6} {'Temp':>5}")
for s in vs[:20]:
    print(f"{s['state']:<45} {s['depth']:>5} {s['n_total']:>6} {s['bias']:>+7.4f} {s['smoothed_prob']:>7.4f} {s['perm_p_value']:>7.4f} {s['noise_stability']:>6.4f} {str(s['temporal_consistent']):>5}")

# Bull vs Bear
bull_states = [s for s in vs if s['bias'] > 0]
bear_states = [s for s in vs if s['bias'] < 0]
print(f"\nBull states: {len(bull_states)}, Bear states: {len(bear_states)}")
if bull_states:
    avg_bull = np.mean([s['bias'] for s in bull_states])
    print(f"Avg bull bias: {avg_bull:.4f}")
if bear_states:
    avg_bear = np.mean([s['bias'] for s in bear_states])
    print(f"Avg bear bias: {avg_bear:.4f}")

# Temporal segment analysis - are edges stable over time?
print(f"\n=== TEMPORAL STABILITY ===")
print(f"{'State':<45} {'Seg1':>6} {'Seg2':>6} {'Seg3':>6} {'Seg4':>6} {'Seg5':>6} {'Consistent':>10}")
for s in vs[:20]:
    segs = s['segment_edges']
    consistent = all(e >= 0 for e in segs) or all(e <= 0 for e in segs)
    seg_str = " ".join(f"{e:>+6.3f}" for e in segs)
    print(f"{s['state']:<45} {seg_str} {'YES' if consistent else 'NO':>10}")

# Check if depth-1 and depth-2 states that are significant but NOT validated
print(f"\n=== REJECTED STATES (Top 20) ===")
rejected = r['rejected_states_top20']
print(f"{'State':<45} {'Depth':>5} {'N':>6} {'Bias':>7} {'Reasons'}")
for s in rejected:
    reasons = s.get('rejection_reasons', ['unknown'])
    print(f"{s['state']:<45} {s['depth']:>5} {s['n_total']:>6} {s['bias']:>+7.4f} {', '.join(reasons)}")

# Significance funnel
print(f"\n=== SIGNIFICANCE FUNNEL ===")
funnel = r['significance_funnel']
for k, v in funnel.items():
    print(f"  {k}: {v}")

# Robustness pass rates
print(f"\n=== ROBUSTNESS PASS RATES ===")
rob = r['robustness_summary']
for k, v in rob.items():
    print(f"  {k}: {v}")

# Question: How many depth-1/2 states pass significance but fail robustness?
print(f"\n=== DEPTH BREAKDOWN OF SIGNIFICANT STATES ===")
print(f"  Significant total: {r['n_significant']}")
print(f"    Depth-1: {r['sig_depth1']}")
print(f"    Depth-2: {r['sig_depth2']}")
print(f"    Depth-3: {r['sig_depth3']}")

print(f"\n  Validated total: {r['n_validated']}")
print(f"    Depth-1: {r['val_depth1']}")
print(f"    Depth-2: {r['val_depth2']}")
print(f"    Depth-3: {r['val_depth3']}")

# Temporal analysis: split data into 5 segments, check overall bull rate
n = len(c)
seg_size = n // 5
print(f"\n=== MARKET REGIME BY TIME SEGMENT ===")
for seg in range(5):
    start = seg * seg_size
    end = min(start + seg_size, n - 12)
    outcomes = [(c[i+12] > c[i]) for i in range(start, end)]
    bull_rate = sum(outcomes) / len(outcomes)
    price_change = (c[end] - c[start]) / c[start] * 100
    print(f"  Seg {seg}: bars [{start}-{end}], bull_rate={bull_rate:.4f}, price_change={price_change:+.1f}%")
