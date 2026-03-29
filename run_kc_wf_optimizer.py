"""
KC Walk-Forward Optimizer
35-fold, her hafta bağımsız OOS test
500 trial/fold TPE optimization
"""

import numpy as np
import pandas as pd
import rust_engine
import time

# ── Load data ──
print("Loading 5m data...")
df5 = pd.read_parquet("data/ETHUSDT_5m_cvd_oi_11mo.parquet")
print(f"  5m bars: {len(df5):,}")

print("Loading 3m data...")
df3 = pd.read_parquet("data/ETHUSDT_3m_vol_11mo.parquet")
print(f"  3m bars: {len(df3):,}")

# ── Prepare arrays ──
ts5 = np.ascontiguousarray(df5["open_time"].values.astype(np.uint64))
closes_5m = np.ascontiguousarray(df5["close"].values, dtype=np.float64)
highs_5m = np.ascontiguousarray(df5["high"].values, dtype=np.float64)
lows_5m = np.ascontiguousarray(df5["low"].values, dtype=np.float64)
buy_vol_5m = np.ascontiguousarray(df5["buy_vol"].values, dtype=np.float64)
sell_vol_5m = np.ascontiguousarray(df5["sell_vol"].values, dtype=np.float64)
oi_5m = np.ascontiguousarray(df5["open_interest"].values, dtype=np.float64)

ts3 = np.ascontiguousarray(df3["open_time"].values.astype(np.uint64))
closes_3m = np.ascontiguousarray(df3["close"].values, dtype=np.float64)
highs_3m = np.ascontiguousarray(df3["high"].values, dtype=np.float64)
lows_3m = np.ascontiguousarray(df3["low"].values, dtype=np.float64)

# ── Run optimizer ──
print("\n" + "=" * 70)
print("  KC WALK-FORWARD OPTIMIZER")
print("  35-fold | 500 trial/fold TPE | 4 params")
print("  kc_length[10-50] kc_mult[1.0-4.0] kc_atr[7-28] max_dca[0-10]")
print("=" * 70)
print()

t0 = time.time()
r = rust_engine.run_kc_wf_optimization_py(
    closes_5m, highs_5m, lows_5m,
    buy_vol_5m, sell_vol_5m, oi_5m, ts5,
    closes_3m, highs_3m, lows_3m, ts3,
    seed=42,
)
elapsed = time.time() - t0

# ── Results ──
print(f"\nCompleted in {elapsed:.1f}s")
print()
print("=" * 70)
print(f"{'AGGREGATE OOS RESULTS':^70}")
print("=" * 70)
print(f"  Total Folds:         {r['total_folds']}")
print(f"  Positive Folds:      {r['positive_folds']}/{r['total_folds']}")
print(f"  Negative Folds:      {r['negative_folds']}/{r['total_folds']}")
print(f"  Avg OOS PnL%:        {r['avg_oos_pnl_pct']:.2f}%")
print(f"  Median OOS PnL%:     {r['median_oos_pnl_pct']:.2f}%")
print(f"  Best Fold:           {r['best_fold_pct']:.2f}%")
print(f"  Worst Fold:          {r['worst_fold_pct']:.2f}%")
print(f"  Total OOS PnL:       {r['total_oos_pnl']:.2f} USDT")
print(f"  Max Consec Neg:      {r['max_consec_neg']}")

# Fold details
folds = r["folds"]
print(f"\n{'FOLD DETAILS':^70}")
print(f"  {'Fold':>5} {'KC_L':>5} {'KC_M':>5} {'ATR':>4} {'DCA':>4} {'TrScr':>7} {'TrW':>4} {'OOS%':>8} {'OOS$':>8} {'Trd':>4} {'Win':>4} {'DD%':>6}")
print(f"  {'-'*5} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*7} {'-'*4} {'-'*8} {'-'*8} {'-'*4} {'-'*4} {'-'*6}")

for f in folds:
    marker = "+" if f["test_pnl"] > 0 else "-" if f["test_pnl"] < 0 else " "
    print(f"  {f['fold_idx']+1:>4}{marker} {f['kc_length']:>5} {f['kc_mult']:>5.1f} {f['kc_atr_period']:>4} {f['max_dca']:>4} {f['train_score']:>7.3f} {f['train_weeks']:>4} {f['test_pnl_pct']:>7.2f}% {f['test_pnl']:>8.2f} {f['test_trades']:>4} {f['test_wins']:>4} {f['test_max_dd']:>5.1f}%")

# Summary stats
oos_pcts = [f["test_pnl_pct"] for f in folds]
print(f"\n{'DISTRIBUTION':^70}")
print(f"  Mean:    {np.mean(oos_pcts):.2f}%")
print(f"  Median:  {np.median(oos_pcts):.2f}%")
print(f"  Std:     {np.std(oos_pcts):.2f}%")
print(f"  Min:     {np.min(oos_pcts):.2f}%")
print(f"  Max:     {np.max(oos_pcts):.2f}%")
print(f"  >0%:     {sum(1 for p in oos_pcts if p > 0)}/{len(oos_pcts)} ({sum(1 for p in oos_pcts if p > 0)/len(oos_pcts)*100:.0f}%)")

# Cumulative PnL
cum = np.cumsum([f["test_pnl"] for f in folds])
print(f"\n  Cumulative OOS PnL:  {cum[-1]:.2f} USDT")

# KC param frequency analysis
print(f"\n{'KC PARAM FREKANSI':^70}")
kc_lengths = [f["kc_length"] for f in folds]
kc_mults = [f["kc_mult"] for f in folds]
kc_atrs = [f["kc_atr_period"] for f in folds]
max_dcas = [f["max_dca"] for f in folds]

print(f"  KC Length:  mean={np.mean(kc_lengths):.1f}  median={np.median(kc_lengths):.0f}  mode={max(set(kc_lengths), key=kc_lengths.count)}")
print(f"  KC Mult:    mean={np.mean(kc_mults):.2f}  median={np.median(kc_mults):.1f}  mode={max(set(kc_mults), key=kc_mults.count)}")
print(f"  KC ATR:     mean={np.mean(kc_atrs):.1f}  median={np.median(kc_atrs):.0f}  mode={max(set(kc_atrs), key=kc_atrs.count)}")
print(f"  Max DCA:    mean={np.mean(max_dcas):.1f}  median={np.median(max_dcas):.0f}  mode={max(set(max_dcas), key=max_dcas.count)}")

print(f"\n{'='*70}")
