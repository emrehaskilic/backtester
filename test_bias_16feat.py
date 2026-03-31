"""
Test 16 features + 7 quintiles vs baseline.
Focus on bias engine accuracy at 100% coverage.
"""
import time
import numpy as np
import pandas as pd
import rust_engine

# Load data
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12
n_1h = len(df) // PERIOD

ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

ts_1h = np.zeros(n_1h, dtype=np.uint64)
o_1h = np.zeros(n_1h); h_1h = np.zeros(n_1h); l_1h = np.zeros(n_1h); c_1h = np.zeros(n_1h)
bv_1h = np.zeros(n_1h); sv_1h = np.zeros(n_1h); oi_1h = np.zeros(n_1h)

for i in range(n_1h):
    s, e = i * PERIOD, i * PERIOD + PERIOD
    ts_1h[i] = ts[s]; o_1h[i] = o[s]; h_1h[i] = h[s:e].max()
    l_1h[i] = l[s:e].min(); c_1h[i] = c[e - 1]
    bv_1h[i] = bv[s:e].sum(); sv_1h[i] = sv[s:e].sum(); oi_1h[i] = oi[e - 1]

print(f"1H data: {n_1h:,} bars")

# ═══════════════════════════════════════════════════════════════
# TEST 1: Default pipeline (16 features, 5 quintiles — via compute_bias_series)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TEST 1: Bias Engine (16 features, default 5 quintiles)")
print(f"{'='*60}")

t0 = time.time()
r = rust_engine.bias_engine_compute_bias(
    ts_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h
)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Validated states:    {r['n_validated']}")
print(f"  Direction accuracy:  {r['accuracy']['direction_accuracy']:.4f}")
print(f"  Strong accuracy:     {r['accuracy']['strong_signal_accuracy']:.4f} (n={r['accuracy']['n_strong_bars']})")
print(f"  Coverage:            {r['coverage']['coverage_pct']:.1f}%")
print(f"  Depth distribution:  D3={r['coverage']['depth3_pct']:.1f}% D2={r['coverage']['depth2_pct']:.1f}% D1={r['coverage']['depth1_pct']:.1f}% Fallback={r['coverage']['fallback_pct']:.1f}%")

# ═══════════════════════════════════════════════════════════════
# TEST 2: Walk-forward with 7 quintiles + all 16 features
# Various min_sample_size values
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TEST 2: Walk-Forward with different min_sample_size")
print(f"{'='*60}")

# Group B (scoring params — bias only, w_bias=1 for pure bias test)
group_b = np.array([
    20, 120,    # mr_ema_span1, mr_ema_span2
    28, 20.0,   # rsi_period, rsi_threshold
    1.0,        # w_bias (full bias weight for pure test)
    0.0, 0.0,   # w_mr1, w_mr2 (no MR)
    0.0,        # w_rsi (no RSI)
    0.0, 0.0,   # w_agree, w_cvd
    0.15, 3.0,  # bias_override_threshold, bias_override_mult
    0.15, 0.30, 0.70,  # sweep
    72, 1.5, 0.90, 48, 0.50,  # regime
    24, 0.0, 12, 0.0, 0.0,  # btc (unused)
], dtype=np.float64)

for q_count in [5, 7]:
    for min_samples in [100, 200, 500]:
        print(f"\n  --- Q={q_count}, min_samples={min_samples} ---")
        group_a = np.array([
            12, 288, 12, 288, 12, 288, 288,  # feature windows
            2016,       # quant_window
            q_count,    # quantile_count
            12,         # k_horizon
            min_samples,# min_sample_size
            0.02,       # min_edge
            30.0,       # prior_strength
            0.05,       # fdr_alpha
            3,          # temporal_min_segments
            1,          # temporal_max_reversals
            0.80,       # min_noise_stability
            50,         # ensemble_min_n
            48, 24, 24, 48, 48, 24,  # new feature windows
            4, 24,      # mtf windows
        ], dtype=np.float64)

        t0 = time.time()
        wf = rust_engine.bias_engine_scoring_wf(
            c_1h, h_1h, l_1h, bv_1h, sv_1h, oi_1h,
            group_a, group_b,
            None, None, None
        )
        elapsed = time.time() - t0
        print(f"    Time:     {elapsed:.1f}s")
        print(f"    WF acc:   {wf['overall_accuracy']:.4f} ({wf['overall_accuracy']*100:.2f}%)")
        print(f"    Chunks:   {wf['chunk_accuracies']}")
