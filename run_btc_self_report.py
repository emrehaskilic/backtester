"""BTC kendi sweep analizi — Spot ve Perp, 4H ve 1D"""
import numpy as np, pandas as pd, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import rust_engine

def load(path):
    df = pd.read_parquet(path)
    sa = np.ascontiguousarray
    return {
        'c': sa(df['close'].values, dtype=np.float64),
        'h': sa(df['high'].values, dtype=np.float64),
        'l': sa(df['low'].values, dtype=np.float64),
        'bv': sa(df['buy_vol'].values, dtype=np.float64),
        'sv': sa(df['sell_vol'].values, dtype=np.float64),
        'oi': sa(df['open_interest'].fillna(0).values, dtype=np.float64),
        'n': len(df),
    }

btc_perp = load('data/BTCUSDT_5m_5y_perp.parquet')
btc_spot = load('data/BTCUSDT_5m_5y_spot.parquet')

print("BTC SWEEP CANDLE ANALIZI — PERP vs SPOT")
print("=" * 120)
print(f"BTC Perp: {btc_perp['n']:,} bars | BTC Spot: {btc_spot['n']:,} bars")
print(f"Periyot: 2021-03-25 → 2026-03-25 (5 yil)")
print()

KEY_FEATURES = ["CVD_zscore_micro", "Imbalance_smooth", "Vol_zscore_micro", "CVD_zscore_macro", "ATR_percentile"]

datasets = [
    ("BTC Perp", btc_perp),
    ("BTC Spot", btc_spot),
]

for tf_name, bp in [("4H", 48), ("1D", 288)]:
    print()
    print("#" * 120)
    print(f"#  TIMEFRAME: {tf_name}")
    print("#" * 120)

    if tf_name == "4H":
        offsets = [(0, "ilk 5 dakika"), (11, "ilk 1 saat"), (23, "ilk 2 saat"), (47, "mum kapanisi")]
    else:
        offsets = [(0, "ilk 5 dakika"), (11, "ilk 1 saat"), (47, "ilk 4 saat"), (95, "ilk 8 saat"), (191, "ilk 16 saat"), (287, "mum kapanisi")]

    for offset, olabel in offsets:
        print(f"\n  {'='*110}")
        print(f"  OLCUM ZAMANI: {olabel} (offset={offset})")
        print(f"  {'='*110}")

        results = []
        for dlabel, data in datasets:
            r = rust_engine.run_candle_miner_py(
                data['c'], data['h'], data['l'],
                data['bv'], data['sv'], data['oi'],
                offset, bp,
            )
            results.append((dlabel, r))

        # Base rates
        print(f"\n  BASE RATES:")
        print(f"  {'Source':>15s} | {'High Base%':>10s} | {'Low Base%':>10s} | {'H Events':>8s} | {'L Events':>8s}")
        print(f"  {'-'*60}")
        for label, r in results:
            print(f"  {label:>15s} | {r['high_base_cont_rate']:>9.1f}% | {r['low_base_cont_rate']:>9.1f}% | {r['high_total']:>8d} | {r['low_total']:>8d}")

        # Quantile
        print(f"\n  QUANTILE ANALIZI:")
        for st in ["high", "low"]:
            print(f"\n    {st.upper()} SWEEP:")
            for fname in KEY_FEATURES:
                print(f"    {fname}:")
                print(f"    {'Source':>15s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s} | {'Spread':>7s}")
                print(f"    {'-'*75}")
                for label, r in results:
                    qrows = sorted([q for q in r['quantile_rows'] if q['sweep_type']==st and q['feature']==fname],
                                  key=lambda x: x['quantile'])
                    if len(qrows) != 5:
                        continue
                    vals = [qrows[i]['cont_rate'] for i in range(5)]
                    spread = max(vals) - min(vals)
                    print(f"    {label:>15s} | {vals[0]:>6.1f}% | {vals[1]:>6.1f}% | {vals[2]:>6.1f}% | {vals[3]:>6.1f}% | {vals[4]:>6.1f}% | {spread:>6.1f}")
                print()

        # Patterns
        print(f"\n  PATTERN SONUCLARI:")
        for cat_label, key in [("High Continuation", "high_cont_patterns"),
                                ("High Reversal", "high_rev_patterns"),
                                ("Low Continuation", "low_cont_patterns"),
                                ("Low Reversal", "low_rev_patterns")]:
            print(f"\n    {cat_label}:")
            print(f"    {'Source':>15s} | {'Rate%':>7s} | {'N':>5s} | {'WF':>7s} | {'Cons%':>6s} | Pattern")
            print(f"    {'-'*90}")
            for label, r in results:
                pats = r[key]
                if not pats:
                    print(f"    {label:>15s} | {'---':>7s} | {'---':>5s} | {'---':>7s} | {'---':>6s} | Pattern yok")
                else:
                    for i, p in enumerate(pats[:3]):
                        wf = f"{p['wf_positive']}/{p['wf_total']}" if p['wf_total'] > 0 else "N/A"
                        src = label if i == 0 else ""
                        print(f"    {src:>15s} | {p['target_rate']:>6.1f}% | {p['n']:>5d} | {wf:>7s} | {p['wf_consistency']:>5.1f}% | {p['conditions']}")

# Ozet
print()
print()
print("#" * 120)
print("#  OZET TABLO")
print("#" * 120)

for tf_name, bp in [("4H", 48), ("1D", 288)]:
    print(f"\n  TIMEFRAME: {tf_name}")

    if tf_name == "4H":
        offsets = [(0, "ilk 5dk"), (11, "1 saat"), (23, "2 saat"), (47, "kapanisi")]
    else:
        offsets = [(0, "ilk 5dk"), (47, "4 saat"), (95, "8 saat"), (287, "kapanisi")]

    for offset, olabel in offsets:
        print(f"\n  Olcum: {olabel}")
        print(f"  {'Source':>15s} | {'HC%':>7s} {'N':>5s} | {'HR%':>7s} {'N':>5s} | {'LC%':>7s} {'N':>5s} | {'LR%':>7s} {'N':>5s}")
        print(f"  {'-'*80}")

        for dlabel, data in datasets:
            r = rust_engine.run_candle_miner_py(
                data['c'], data['h'], data['l'],
                data['bv'], data['sv'], data['oi'],
                offset, bp,
            )
            def best(key):
                pats = r[key]
                if not pats: return 0.0, 0
                return pats[0]['target_rate'], pats[0]['n']

            hc_r, hc_n = best('high_cont_patterns')
            hr_r, hr_n = best('high_rev_patterns')
            lc_r, lc_n = best('low_cont_patterns')
            lr_r, lr_n = best('low_rev_patterns')
            print(f"  {dlabel:>15s} | {hc_r:>6.1f}% {hc_n:>4d} | {hr_r:>6.1f}% {hr_n:>4d} | {lc_r:>6.1f}% {lc_n:>4d} | {lr_r:>6.1f}% {lr_n:>4d}")

print()
print("Rapor tamamlandi.")
