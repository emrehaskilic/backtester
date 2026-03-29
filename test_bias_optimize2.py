"""Optimizer — 2000 trial, with stderr progress"""
import time, sys, numpy as np, pandas as pd, rust_engine

df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12; n_1h = len(df) // PERIOD
ts = df["open_time"].values; o,h,l,c = df["open"].values,df["high"].values,df["low"].values,df["close"].values
bv,sv,oi = df["buy_vol"].values,df["sell_vol"].values,df["open_interest"].values
ts_1h=np.zeros(n_1h,dtype=np.uint64);o_1h=np.zeros(n_1h);h_1h=np.zeros(n_1h);l_1h=np.zeros(n_1h)
c_1h=np.zeros(n_1h);bv_1h=np.zeros(n_1h);sv_1h=np.zeros(n_1h);oi_1h=np.zeros(n_1h)
for i in range(n_1h):
    s,e=i*PERIOD,i*PERIOD+PERIOD
    ts_1h[i]=ts[s];o_1h[i]=o[s];h_1h[i]=h[s:e].max();l_1h[i]=l[s:e].min();c_1h[i]=c[e-1]
    bv_1h[i]=bv[s:e].sum();sv_1h[i]=sv[s:e].sum();oi_1h[i]=oi[e-1]

print(f"1H: {n_1h} bars. Starting 2000 trial optimizer...", flush=True)
t0=time.time()
try:
    r=rust_engine.bias_engine_optimize(ts_1h,o_1h,h_1h,l_1h,c_1h,bv_1h,sv_1h,oi_1h,2000,42)
    elapsed=time.time()-t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Best OOS score: {r['best_score']:.4f}")
    print(f"Validated states: {r['n_validated_states']}")
    print(f"Trials: {r['trials_evaluated']}")
    print(f"\n-- Group A --")
    for k,v in r['group_a'].items(): print(f"  {k}: {v}")
    print(f"\n-- Group B --")
    for k,v in r['group_b'].items(): print(f"  {k}: {v}")
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback; traceback.print_exc()
