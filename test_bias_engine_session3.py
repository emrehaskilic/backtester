"""
Bias Engine Session 3 — Test Script (Steps 5-9 + Step 12)

Tests:
1. compute_bias — Full Steps 1-9 pipeline (per-bar bias series)
2. walkforward — Step 12 walk-forward evaluation
"""

import time
import numpy as np
import pandas as pd
import rust_engine

# ── Load data ──
print("=" * 70)
print("BIAS ENGINE — SESSION 3 TEST")
print("=" * 70)

df = pd.read_parquet("data/ETHUSDT_5m_cvd_oi_11mo.parquet")
print(f"\nData loaded: {len(df)} bars, {df.columns.tolist()}")

timestamps = df["open_time"].values.astype(np.uint64)
open_ = df["open"].values.astype(np.float64)
high = df["high"].values.astype(np.float64)
low = df["low"].values.astype(np.float64)
close = df["close"].values.astype(np.float64)
buy_vol = df["buy_vol"].values.astype(np.float64)
sell_vol = df["sell_vol"].values.astype(np.float64)
oi = df["open_interest"].values.astype(np.float64)

# ══════════════════════════════════════════════════════════════════
# TEST 1: Full Bias Series (Steps 1-9)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: compute_bias (Steps 1-9)")
print("=" * 70)

t0 = time.time()
result = rust_engine.bias_engine_compute_bias(
    timestamps, open_, high, low, close, buy_vol, sell_vol, oi
)
elapsed = time.time() - t0
print(f"Elapsed: {elapsed:.1f}s")

print(f"\n-- Summary --")
print(f"  n_bars: {result['n_bars']}")
print(f"  n_validated states: {result['n_validated']}")
print(f"  baseline_bull_rate: {result['baseline_bull_rate']}")

print(f"\n-- Coverage --")
cov = result['coverage']
print(f"  Total coverage: {cov['coverage_pct']}%")
print(f"  Depth-3: {cov['depth3_pct']}%")
print(f"  Depth-2: {cov['depth2_pct']}%")
print(f"  Depth-1: {cov['depth1_pct']}%")
print(f"  Fallback: {cov['fallback_pct']}%")

print(f"\n-- Accuracy (in-sample) --")
acc = result['accuracy']
print(f"  Direction accuracy: {acc['direction_accuracy']:.4f}")
print(f"  Strong signal accuracy (|bias|>0.15): {acc['strong_signal_accuracy']:.4f}")
print(f"  N strong bars: {acc['n_strong_bars']}")

print(f"\n-- Regime Distribution --")
reg = result['regime']
print(f"  Trending: {reg['trending_pct']}%")
print(f"  Mean-reverting: {reg['mean_reverting_pct']}%")
print(f"  High volatility: {reg['high_vol_pct']}%")

print(f"\n-- Calibration --")
cal = result['calibration']
print(f"  Brier (uncalibrated): {cal['brier_uncalibrated']:.4f}")
print(f"  Brier (calibrated): {cal['brier_calibrated']:.4f}")
print(f"  Calibration improved: {cal['calibration_improved']}")

# Analyze per-bar arrays
final_bias = np.array(result['final_bias'])
confidence = np.array(result['confidence'])
state_bias = np.array(result['state_bias'])
sweep_bias = np.array(result['sweep_bias'])
matched_depth = np.array(result['matched_depth'])
direction = np.array(result['direction'])

print(f"\n-- Per-bar Bias Statistics --")
print(f"  final_bias: mean={final_bias.mean():.4f}, std={final_bias.std():.4f}, "
      f"min={final_bias.min():.4f}, max={final_bias.max():.4f}")
print(f"  confidence: mean={confidence.mean():.4f}, std={confidence.std():.4f}")
print(f"  state_bias: mean={state_bias.mean():.4f}, std={state_bias.std():.4f}")
print(f"  sweep_bias: mean={sweep_bias.mean():.4f}, std={sweep_bias.std():.4f}")

# Direction distribution
n_bull = (direction == 1).sum()
n_neutral = (direction == 0).sum()
n_bear = (direction == -1).sum()
print(f"\n  Direction: Bull={n_bull} ({n_bull/len(direction)*100:.1f}%), "
      f"Neutral={n_neutral} ({n_neutral/len(direction)*100:.1f}%), "
      f"Bear={n_bear} ({n_bear/len(direction)*100:.1f}%)")

# Matched depth distribution
for d in range(4):
    n = (matched_depth == d).sum()
    print(f"  Depth {d}: {n} bars ({n/len(matched_depth)*100:.1f}%)")

# Verify coverage = 100%
assert not np.isnan(final_bias).any(), "FAIL: NaN in final_bias!"
assert not np.isinf(final_bias).any(), "FAIL: Inf in final_bias!"
print(f"\n  OK Coverage verified: 100% (no NaN, no Inf)")

# ══════════════════════════════════════════════════════════════════
# TEST 2: Walk-Forward Evaluation (Step 12)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: Walk-Forward Evaluation (Step 12)")
print("=" * 70)

t0 = time.time()
wf = rust_engine.bias_engine_walkforward(
    timestamps, open_, high, low, close, buy_vol, sell_vol, oi
)
elapsed = time.time() - t0
print(f"Elapsed: {elapsed:.1f}s")

print(f"\n-- Walk-Forward Summary --")
print(f"  N folds: {wf['n_folds']}")
print(f"  Overall accuracy: {wf['overall_accuracy']:.4f}")
print(f"  Overall strong accuracy: {wf['overall_strong_accuracy']:.4f}")
print(f"  Accuracy std: {wf['accuracy_std']:.4f}")
print(f"  Total test bars: {wf['total_test_bars']}")
print(f"  WF-validated states: {wf['n_wf_validated_states']}")

print(f"\n-- Per-Fold Results --")
for fold in wf['folds']:
    print(f"  Fold {fold['fold_idx']}: "
          f"acc={fold['test_accuracy']:.4f}, "
          f"strong_acc={fold['test_strong_accuracy']:.4f}, "
          f"n_bars={fold['test_n_bars']}, "
          f"n_strong={fold['test_n_strong']}, "
          f"n_validated={fold['n_validated']}, "
          f"|bias|={fold['test_mean_abs_bias']:.4f}, "
          f"brier_cal={fold['brier_calibrated']:.4f}")

if wf['n_wf_validated_states'] > 0:
    print(f"\n-- WF-Validated States (first 20) --")
    for s in wf['wf_validated_states'][:20]:
        print(f"  {s}")

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SESSION 3 — RESULTS SUMMARY")
print("=" * 70)

print(f"""
Steps 1-9 (Batch Bias Series):
  - {result['n_validated']} validated states
  - Direction accuracy: {acc['direction_accuracy']:.4f} (target: >0.53)
  - Strong accuracy: {acc['strong_signal_accuracy']:.4f} (target: >0.57)
  - Coverage: {cov['coverage_pct']}% (target: 100%)
  - Fallback: {cov['fallback_pct']}%
  - Calibration: {'improved' if cal['calibration_improved'] else 'no improvement'}

Step 12 (Walk-Forward):
  - {wf['n_folds']} folds
  - Overall OOS accuracy: {wf['overall_accuracy']:.4f} (target: >0.53)
  - Accuracy std: {wf['accuracy_std']:.4f} (target: <0.05)
  - WF-validated states: {wf['n_wf_validated_states']} (target: >=50)
""")
