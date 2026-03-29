"""BTC Cross-Asset Analiz — ETH sweep'lerinde BTC feature'lari ile tahmin
4H ve 1D, Continuation + Reversal, 5 yillik veri
BTC Perp + BTC Spot feature kaynaklari karsilastirilir.
"""
import numpy as np, pandas as pd, sys, io, time, json
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

# Load all datasets
print("Loading data...")
eth_perp = load('data/ETHUSDT_5m_5y.parquet')
eth_spot = load('data/ETHUSDT_5m_5y_spot.parquet')
btc_perp = load('data/BTCUSDT_5m_5y_perp.parquet')
btc_spot = load('data/BTCUSDT_5m_5y_spot.parquet')
print(f"  ETH Perp: {eth_perp['n']:,} bars")
print(f"  ETH Spot: {eth_spot['n']:,} bars")
print(f"  BTC Perp: {btc_perp['n']:,} bars")
print(f"  BTC Spot: {btc_spot['n']:,} bars")

# Align lengths
min_len = min(eth_perp['n'], eth_spot['n'], btc_perp['n'], btc_spot['n'])
print(f"  Aligned length: {min_len:,} bars")
for d in [eth_perp, eth_spot, btc_perp, btc_spot]:
    for k in ['c','h','l','bv','sv','oi']:
        d[k] = d[k][:min_len]
print()

FEATURE_NAMES = ["CVD_zscore_micro", "CVD_zscore_macro", "OI_change",
                 "Vol_zscore_micro", "Vol_zscore_macro", "Imbalance_smooth", "ATR_percentile"]

KEY_FEATURES = ["CVD_zscore_micro", "Imbalance_smooth", "Vol_zscore_micro", "CVD_zscore_macro", "ATR_percentile"]

def run_miner(candle_data, feat_data, offset, bp):
    """Run cross-asset miner: candle from candle_data, features from feat_data"""
    return rust_engine.run_candle_miner_cross_py(
        candle_data['c'], candle_data['h'], candle_data['l'],
        feat_data['c'], feat_data['h'], feat_data['l'],
        feat_data['bv'], feat_data['sv'], feat_data['oi'],
        offset, bp,
    )

def run_self_miner(data, offset, bp):
    """Same-asset miner (baseline)"""
    return rust_engine.run_candle_miner_py(
        data['c'], data['h'], data['l'],
        data['bv'], data['sv'], data['oi'],
        offset, bp,
    )

def print_quantile_table(results, key_features, sweep_types=["high","low"]):
    for st in sweep_types:
        print(f"\n    {st.upper()} SWEEP:")
        print(f"    {'Source':>20s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s} | {'Spread':>7s}")
        print(f"    {'-'*75}")
        for fname in key_features:
            for label, r in results:
                qrows = sorted([q for q in r['quantile_rows'] if q['sweep_type']==st and q['feature']==fname],
                              key=lambda x: x['quantile'])
                if len(qrows) != 5:
                    continue
                vals = [qrows[i]['cont_rate'] for i in range(5)]
                spread = max(vals) - min(vals)
                print(f"    {label:>20s} | {vals[0]:>6.1f}% | {vals[1]:>6.1f}% | {vals[2]:>6.1f}% | {vals[3]:>6.1f}% | {vals[4]:>6.1f}% | {spread:>6.1f}")
            print()

def print_pattern_table(results):
    for cat_label, key in [("High Continuation", "high_cont_patterns"),
                            ("High Reversal", "high_rev_patterns"),
                            ("Low Continuation", "low_cont_patterns"),
                            ("Low Reversal", "low_rev_patterns")]:
        print(f"\n    {cat_label}:")
        print(f"    {'Source':>20s} | {'Rate%':>7s} | {'N':>5s} | {'WF':>7s} | {'Cons%':>6s} | Top Pattern")
        print(f"    {'-'*90}")
        for label, r in results:
            pats = r[key]
            if not pats:
                print(f"    {label:>20s} | {'---':>7s} | {'---':>5s} | {'---':>7s} | {'---':>6s} | Pattern yok")
            else:
                p = pats[0]
                wf = f"{p['wf_positive']}/{p['wf_total']}" if p['wf_total'] > 0 else "N/A"
                print(f"    {label:>20s} | {p['target_rate']:>6.1f}% | {p['n']:>5d} | {wf:>7s} | {p['wf_consistency']:>5.1f}% | {p['conditions']}")
                # Print top 3 if more
                for p2 in pats[1:3]:
                    wf2 = f"{p2['wf_positive']}/{p2['wf_total']}" if p2['wf_total'] > 0 else "N/A"
                    print(f"    {'':>20s} | {p2['target_rate']:>6.1f}% | {p2['n']:>5d} | {wf2:>7s} | {p2['wf_consistency']:>5.1f}% | {p2['conditions']}")

# ══════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════

print("=" * 120)
print("  BTC CROSS-ASSET ANALIZ RAPORU")
print("  ETH sweep mumlarinda BTC feature'lari ile continuation/reversal tahmini")
print("  5 yillik veri (2021-03-25 → 2026-03-25)")
print("=" * 120)
print()

# Configurations to test
configs = [
    # (candle_label, candle_data, feat_label, feat_data)
    ("ETH Perp kendi", eth_perp, "ETH Perp", eth_perp),      # baseline
    ("ETH Perp + BTC Perp", eth_perp, "BTC Perp", btc_perp), # cross
    ("ETH Perp + BTC Spot", eth_perp, "BTC Spot", btc_spot),  # cross
    ("ETH Spot kendi", eth_spot, "ETH Spot", eth_spot),        # baseline
    ("ETH Spot + BTC Perp", eth_spot, "BTC Perp", btc_perp),  # cross
    ("ETH Spot + BTC Spot", eth_spot, "BTC Spot", btc_spot),   # cross
]

for tf_name, bp in [("4H", 48), ("1D", 288)]:
    print()
    print("#" * 120)
    print(f"#  TIMEFRAME: {tf_name}")
    print("#" * 120)

    if tf_name == "4H":
        offsets = [(0, "ilk 5 dakika"), (11, "ilk 1 saat"), (23, "ilk 2 saat"), (47, "mum kapanisi")]
    else:
        offsets = [(0, "ilk 5 dakika"), (11, "ilk 1 saat"), (47, "ilk 4 saat"), (95, "ilk 8 saat"), (287, "mum kapanisi")]

    for offset, olabel in offsets:
        print()
        print(f"  {'='*110}")
        print(f"  OLCUM ZAMANI: {olabel} (offset={offset})")
        print(f"  {'='*110}")

        results = []
        for config_label, candle_data, feat_label, feat_data in configs:
            t0 = time.time()
            if candle_data is feat_data:
                r = run_self_miner(candle_data, offset, bp)
            else:
                r = run_miner(candle_data, feat_data, offset, bp)
            elapsed = time.time() - t0
            results.append((config_label, r))

        # Base rates
        print(f"\n  BASE RATES:")
        print(f"  {'Source':>25s} | {'High Base%':>10s} | {'Low Base%':>10s} | {'H Events':>8s} | {'L Events':>8s}")
        print(f"  {'-'*70}")
        for label, r in results:
            print(f"  {label:>25s} | {r['high_base_cont_rate']:>9.1f}% | {r['low_base_cont_rate']:>9.1f}% | {r['high_total']:>8d} | {r['low_total']:>8d}")

        # Quantile comparison
        print(f"\n  QUANTILE ANALIZI:")
        for fname in KEY_FEATURES:
            print(f"\n  Feature: {fname}")
            print_quantile_table(results, [fname])

        # Pattern comparison
        print(f"\n  PATTERN SONUCLARI:")
        print_pattern_table(results)

# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print()
print()
print("#" * 120)
print("#  OZET TABLO — En iyi pattern rate by source and offset")
print("#" * 120)

for tf_name, bp in [("4H", 48), ("1D", 288)]:
    print(f"\n  TIMEFRAME: {tf_name}")

    if tf_name == "4H":
        offsets = [(0, "ilk 5dk"), (11, "1 saat"), (47, "kapanisi")]
    else:
        offsets = [(0, "ilk 5dk"), (47, "4 saat"), (287, "kapanisi")]

    for offset, olabel in offsets:
        print(f"\n  Olcum: {olabel}")
        print(f"  {'Source':>25s} | {'HC%':>7s} {'N':>5s} | {'HR%':>7s} {'N':>5s} | {'LC%':>7s} {'N':>5s} | {'LR%':>7s} {'N':>5s}")
        print(f"  {'-'*85}")

        for config_label, candle_data, feat_label, feat_data in configs:
            if candle_data is feat_data:
                r = run_self_miner(candle_data, offset, bp)
            else:
                r = run_miner(candle_data, feat_data, offset, bp)

            def best(key):
                pats = r[key]
                if not pats:
                    return 0.0, 0
                return pats[0]['target_rate'], pats[0]['n']

            hc_r, hc_n = best('high_cont_patterns')
            hr_r, hr_n = best('high_rev_patterns')
            lc_r, lc_n = best('low_cont_patterns')
            lr_r, lr_n = best('low_rev_patterns')
            print(f"  {config_label:>25s} | {hc_r:>6.1f}% {hc_n:>4d} | {hr_r:>6.1f}% {hr_n:>4d} | {lc_r:>6.1f}% {lc_n:>4d} | {lr_r:>6.1f}% {lr_n:>4d}")

print()
print("Rapor tamamlandi.")
