"""Walk-forward test for bias engine"""
import time
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

print("Running walk-forward...")
t0 = time.time()
wf = rust_engine.bias_engine_walkforward(timestamps, o, h, l, c, bv, sv, oi)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

print(f"\n-- Walk-Forward Summary --")
print(f"  N folds: {wf['n_folds']}")
print(f"  Overall OOS accuracy: {wf['overall_accuracy']:.4f}")
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
    print(f"\n-- WF-Validated States (first 30) --")
    for s in wf['wf_validated_states'][:30]:
        print(f"  {s}")
