"""BTC Correlation — Optimizer with BTC data (2000 trial)

Loads ETH + BTC 5m data, aggregates to 1H, aligns timestamps,
runs bias_engine_optimize with BTC arrays.
"""
import time, sys, numpy as np, pandas as pd

# ── Load data ──
print("Loading ETH + BTC data...", flush=True)
eth = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
btc = pd.read_parquet("data/BTCUSDT_5m_5y_perp.parquet")

print(f"  ETH: {len(eth)} bars, BTC: {len(btc)} bars")

# Align by open_time (inner join)
merged = pd.merge(eth, btc, on="open_time", suffixes=("_eth", "_btc"), how="inner")
print(f"  Aligned: {len(merged)} bars")

# ── Aggregate to 1H ──
PERIOD = 12
n_1h = len(merged) // PERIOD

def agg_1h(df, prefix=""):
    s = prefix
    n = n_1h
    ts = np.zeros(n, dtype=np.uint64)
    o = np.zeros(n); h = np.zeros(n); l = np.zeros(n); c = np.zeros(n)
    bv = np.zeros(n); sv = np.zeros(n); oi_arr = np.zeros(n)

    ot = df["open_time"].values
    op = df[f"open{s}"].values
    hi = df[f"high{s}"].values
    lo = df[f"low{s}"].values
    cl = df[f"close{s}"].values
    bvol = df[f"buy_vol{s}"].values
    svol = df[f"sell_vol{s}"].values
    oiv = df[f"open_interest{s}"].values

    for i in range(n):
        start, end = i * PERIOD, i * PERIOD + PERIOD
        ts[i] = ot[start]
        o[i] = op[start]
        h[i] = hi[start:end].max()
        l[i] = lo[start:end].min()
        c[i] = cl[end - 1]
        bv[i] = bvol[start:end].sum()
        sv[i] = svol[start:end].sum()
        oi_arr[i] = oiv[end - 1]

    return ts, o, h, l, c, bv, sv, oi_arr

print(f"Aggregating to 1H ({n_1h} bars)...", flush=True)
ts_1h, o_eth, h_eth, l_eth, c_eth, bv_eth, sv_eth, oi_eth = agg_1h(merged, "_eth")
_, _, _, _, c_btc, bv_btc, sv_btc, _ = agg_1h(merged, "_btc")

# Handle NaN in OI
oi_eth = np.nan_to_num(oi_eth, nan=0.0)

print(f"1H data ready: {n_1h} bars, ~{n_1h/8760:.1f} years", flush=True)
print(f"  ETH close range: {c_eth.min():.2f} — {c_eth.max():.2f}")
print(f"  BTC close range: {c_btc.min():.2f} — {c_btc.max():.2f}")

# ── Run optimizer ──
import rust_engine

N_TRIALS = 2000
SEED = 42

print(f"\nStarting {N_TRIALS}-trial optimizer with BTC data...", flush=True)
t0 = time.time()
try:
    r = rust_engine.bias_engine_optimize(
        ts_1h, o_eth, h_eth, l_eth, c_eth, bv_eth, sv_eth, oi_eth,
        N_TRIALS, SEED,
        btc_close=c_btc, btc_buy_vol=bv_btc, btc_sell_vol=sv_btc,
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Best OOS score: {r['best_score']:.4f}")
    print(f"Validated states: {r['n_validated_states']}")
    print(f"Trials: {r['trials_evaluated']}")

    print(f"\n── Group A ──")
    for k, v in r['group_a'].items():
        print(f"  {k}: {v}")

    print(f"\n── Group B ──")
    for k, v in r['group_b'].items():
        print(f"  {k}: {v}")

    # Highlight BTC params
    print(f"\n── BTC Parameters ──")
    b = r['group_b']
    print(f"  btc_mom_window: {b['btc_mom_window']}")
    print(f"  w_btc_mom: {b['w_btc_mom']}")
    print(f"  btc_lead_window: {b['btc_lead_window']}")
    print(f"  w_btc_lead: {b['w_btc_lead']}")
    print(f"  w_btc_cvd: {b['w_btc_cvd']}")

    # Compare: if all BTC weights are 0, BTC didn't help
    btc_total_weight = b['w_btc_mom'] + b['w_btc_lead'] + b['w_btc_cvd']
    if btc_total_weight < 0.01:
        print(f"\n⚠ BTC weights are ~0 — BTC data didn't improve scoring")
    else:
        print(f"\n✓ BTC total weight: {btc_total_weight:.2f} — BTC is contributing!")

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback; traceback.print_exc()
