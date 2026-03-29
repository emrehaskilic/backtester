"""Spot vs Perp karsilastirmasi — 4H ve 1D, 5 yillik veri"""
import numpy as np, pandas as pd, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import rust_engine

def load(path):
    df = pd.read_parquet(path)
    df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
    sa = np.ascontiguousarray
    return (
        sa(df['close'].values, dtype=np.float64),
        sa(df['high'].values, dtype=np.float64),
        sa(df['low'].values, dtype=np.float64),
        sa(df['buy_vol'].values, dtype=np.float64),
        sa(df['sell_vol'].values, dtype=np.float64),
        sa(df['open_interest'].fillna(0).values, dtype=np.float64),
        df,
    )

perp = load('data/ETHUSDT_5m_5y.parquet')
spot = load('data/ETHUSDT_5m_5y_spot.parquet')

print("SPOT vs PERP KARSILASTIRMASI — ETHUSDT 5 YIL")
print("=" * 120)
print(f"Perp: {len(perp[6]):,} bar | {perp[6]['dt'].iloc[0].strftime('%Y-%m-%d')} - {perp[6]['dt'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"Spot: {len(spot[6]):,} bar | {spot[6]['dt'].iloc[0].strftime('%Y-%m-%d')} - {spot[6]['dt'].iloc[-1].strftime('%Y-%m-%d')}")
print()

key_features = ["CVD_zscore_micro", "Imbalance_smooth", "Vol_zscore_micro", "CVD_zscore_macro", "ATR_percentile"]

for tf_name, bp in [("4H", 48), ("1D", 288)]:
    print()
    print(f"{'='*120}")
    print(f"  TIMEFRAME: {tf_name} (bars_per_candle={bp})")
    print(f"{'='*120}")

    offsets = []
    if tf_name == "4H":
        offsets = [(0, "ilk 5dk"), (11, "1 saat"), (23, "2 saat"), (47, "kapanisi")]
    else:
        offsets = [(0, "ilk 5dk"), (11, "1 saat"), (47, "4 saat"), (95, "8 saat"), (191, "16 saat"), (287, "kapanisi")]

    for offset, olabel in offsets:
        print(f"\n  --- Olcum zamani: {olabel} (offset={offset}) ---")

        rp = rust_engine.run_candle_miner_py(*perp[:6], offset, bp)
        rs = rust_engine.run_candle_miner_py(*spot[:6], offset, bp)

        print(f"  {'':40s} | {'PERP':^35s} | {'SPOT':^35s}")
        print(f"  {'':40s} | {'Base H':>8s} {'Base L':>8s} {'Ev H':>6s} {'Ev L':>6s} | {'Base H':>8s} {'Base L':>8s} {'Ev H':>6s} {'Ev L':>6s}")
        print(f"  {'':40s} | {rp['high_base_cont_rate']:>7.1f}% {rp['low_base_cont_rate']:>7.1f}% {rp['high_total']:>5d}  {rp['low_total']:>5d}  | {rs['high_base_cont_rate']:>7.1f}% {rs['low_base_cont_rate']:>7.1f}% {rs['high_total']:>5d}  {rs['low_total']:>5d} ")

        # Quantile karsilastirma
        print()
        print(f"  {'Feature':22s} {'Sweep':>5s} | {'PERP Q1':>8s} {'Q3':>6s} {'Q5':>6s} {'Sprd':>6s} | {'SPOT Q1':>8s} {'Q3':>6s} {'Q5':>6s} {'Sprd':>6s}")
        print(f"  {'-'*95}")

        for sweep_type in ["high", "low"]:
            for fname in key_features:
                pq = sorted([q for q in rp['quantile_rows'] if q['sweep_type']==sweep_type and q['feature']==fname], key=lambda x: x['quantile'])
                sq = sorted([q for q in rs['quantile_rows'] if q['sweep_type']==sweep_type and q['feature']==fname], key=lambda x: x['quantile'])
                if len(pq)!=5 or len(sq)!=5:
                    continue
                pv = [pq[i]['cont_rate'] for i in range(5)]
                sv2 = [sq[i]['cont_rate'] for i in range(5)]
                ps = max(pv)-min(pv)
                ss = max(sv2)-min(sv2)
                st = sweep_type[0].upper()
                print(f"  {fname:22s}     {st} | {pv[0]:>7.1f}% {pv[2]:>5.1f}% {pv[4]:>5.1f}% {ps:>5.1f} | {sv2[0]:>7.1f}% {sv2[2]:>5.1f}% {sv2[4]:>5.1f}% {ss:>5.1f}")

        # Top patterns
        print()
        for cat_label, key in [("High Continuation", "high_cont_patterns"), ("Low Continuation", "low_cont_patterns"),
                                ("High Reversal", "high_rev_patterns"), ("Low Reversal", "low_rev_patterns")]:
            pp = rp[key]
            sp = rs[key]
            p_best = f"{pp[0]['target_rate']:.1f}% N={pp[0]['n']}" if pp else "yok"
            s_best = f"{sp[0]['target_rate']:.1f}% N={sp[0]['n']}" if sp else "yok"
            print(f"  {cat_label:22s} | PERP: {p_best:20s} | SPOT: {s_best:20s}")

    print()
