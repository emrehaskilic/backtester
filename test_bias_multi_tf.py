"""
Test bias engine on 4H and 8H timeframes.
Compare OOS accuracy across 1H / 4H / 8H.
"""
import time
import numpy as np
import pandas as pd
import rust_engine

# Load 5m data
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

def aggregate(period):
    n = len(df) // period
    ts_agg = np.zeros(n, dtype=np.uint64)
    o_agg = np.zeros(n); h_agg = np.zeros(n); l_agg = np.zeros(n); c_agg = np.zeros(n)
    bv_agg = np.zeros(n); sv_agg = np.zeros(n); oi_agg = np.zeros(n)
    for i in range(n):
        s, e = i * period, i * period + period
        ts_agg[i] = ts[s]; o_agg[i] = o[s]; h_agg[i] = h[s:e].max()
        l_agg[i] = l[s:e].min(); c_agg[i] = c[e - 1]
        bv_agg[i] = bv[s:e].sum(); sv_agg[i] = sv[s:e].sum(); oi_agg[i] = oi[e - 1]
    return n, ts_agg, o_agg, h_agg, l_agg, c_agg, bv_agg, sv_agg, oi_agg

# Group B: pure bias scoring (w_bias=1, no MR/RSI)
group_b = np.array([
    20, 120, 28, 20.0,   # mr/rsi (unused)
    1.0,                  # w_bias
    0.0, 0.0, 0.0,       # w_mr1, w_mr2, w_rsi
    0.0, 0.0,             # w_agree, w_cvd
    0.15, 3.0,            # bias_override
    0.15, 0.30, 0.70,    # sweep
    72, 1.5, 0.90, 48, 0.50,  # regime
    24, 0.0, 12, 0.0, 0.0,    # btc (unused)
], dtype=np.float64)

configs = [
    ("1H",  12),
    ("4H",  48),
    ("8H",  96),
]

for name, period in configs:
    n, ts_agg, o_agg, h_agg, l_agg, c_agg, bv_agg, sv_agg, oi_agg = aggregate(period)
    print(f"\n{'='*60}")
    print(f"  {name} — {n:,} bars ({n/365.25/(24//(period//12)):.1f} years)")
    print(f"{'='*60}")

    # Adjust windows proportionally for the timeframe
    # 1H defaults: cvd_micro=12, cvd_macro=288, quant=2016
    # Scale factor: how many TF bars per 1H bar
    scale = period // 12  # 1 for 1H, 4 for 4H, 8 for 8H

    # For higher TF, use smaller windows (in bars) since each bar covers more time
    cvd_micro = max(6, 12 // scale)
    cvd_macro = max(24, 288 // scale)
    vol_micro = max(6, 12 // scale)
    vol_macro = max(24, 288 // scale)
    imb_ema = max(4, 12 // scale)
    atr_pct = max(24, 288 // scale)
    oi_change = max(24, 288 // scale)
    quant_w = max(500, 2016 // scale)
    vwap_w = max(12, 48 // scale)
    mom_w = max(6, 24 // scale)
    wick_w = max(6, 24 // scale)
    div_w = max(12, 48 // scale)
    oi_vol_w = max(12, 48 // scale)
    autocorr_w = max(6, 24 // scale)
    mtf_4h_w = max(2, 4 // scale)
    mtf_daily_w = max(6, 24 // scale)

    # K horizon: 12 bars lookahead (12H for 1H, 48H for 4H, 96H for 8H)
    # Keep same time horizon: ~12H lookahead
    k_horizon = max(1, 12 // scale)

    for q_count in [5, 7]:
        for min_samples in [50, 100, 200]:
            group_a = np.array([
                cvd_micro, cvd_macro, vol_micro, vol_macro,
                imb_ema, atr_pct, oi_change,
                quant_w,
                q_count,
                k_horizon,
                min_samples,   # min_sample_size
                0.02,          # min_edge
                30.0,          # prior_strength
                0.05,          # fdr_alpha
                3,             # temporal_min_segments
                1,             # temporal_max_reversals
                0.80,          # min_noise_stability
                50,            # ensemble_min_n
                vwap_w, mom_w, wick_w, div_w, oi_vol_w, autocorr_w,
                mtf_4h_w, mtf_daily_w,
            ], dtype=np.float64)

            t0 = time.time()
            wf = rust_engine.bias_engine_scoring_wf(
                c_agg, h_agg, l_agg, bv_agg, sv_agg, oi_agg,
                group_a, group_b,
                None, None, None
            )
            elapsed = time.time() - t0
            print(f"  Q={q_count} min={min_samples:>3d}: WF={wf['overall_accuracy']:.4f} ({wf['overall_accuracy']*100:.2f}%)  chunks={wf['n_chunks']}  {elapsed:.1f}s  {wf['chunk_accuracies']}")

    # Also run in-sample analysis
    print(f"\n  In-sample analysis:")
    t0 = time.time()
    r = rust_engine.bias_engine_compute_bias(
        ts_agg, o_agg, h_agg, l_agg, c_agg, bv_agg, sv_agg, oi_agg
    )
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"    Validated states: {r['n_validated']}")
    print(f"    Direction acc:    {r['accuracy']['direction_accuracy']:.4f}")
    print(f"    Strong acc:       {r['accuracy']['strong_signal_accuracy']:.4f} (n={r['accuracy']['n_strong_bars']})")
    print(f"    Depths: D3={r['coverage']['depth3_pct']:.1f}% D2={r['coverage']['depth2_pct']:.1f}% D1={r['coverage']['depth1_pct']:.1f}% FB={r['coverage']['fallback_pct']:.1f}%")
