"""1H Sweep Candle — Feature offset karsilastirmasi
offset 0 = ilk 5dk, 1 = 10dk, 2 = 15dk, 11 = mum kapanisi (referans)
"""
import numpy as np, pandas as pd, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import rust_engine

df = pd.read_parquet('data/ETHUSDT_5m_cvd_oi_11mo.parquet')
df['dt'] = pd.to_datetime(df['open_time'], unit='ms')

sa = np.ascontiguousarray
c = sa(df['close'].values, dtype=np.float64)
h = sa(df['high'].values, dtype=np.float64)
l = sa(df['low'].values, dtype=np.float64)
bv = sa(df['buy_vol'].values, dtype=np.float64)
sv = sa(df['sell_vol'].values, dtype=np.float64)
oi = sa(df['open_interest'].fillna(0).values, dtype=np.float64)

offsets = [
    (0, "5dk (ilk bar)"),
    (1, "10dk (2. bar)"),
    (2, "15dk (3. bar)"),
    (11, "60dk (mum kapanisi — referans)"),
]

print("1H SWEEP CANDLE — FEATURE OFFSET KARSILASTIRMASI")
print("=" * 110)
print(f"Data: {len(df):,} bar | {df['dt'].iloc[0].strftime('%Y-%m-%d')} - {df['dt'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"Soru: t mumunun kacinci dakikasinda feature alinirsa, mum sonucu en iyi tahmin edilir?")
print()

all_results = {}

for offset, label in offsets:
    t0 = time.time()
    r2 = rust_engine.run_candle_miner_py(c, h, l, bv, sv, oi, offset)
    elapsed = time.time() - t0
    all_results[offset] = (r2, label, elapsed)

# ══════════════════════════════════════════════════════════════
# Quantile karsilastirmasi
# ══════════════════════════════════════════════════════════════
print()
print("QUANTILE MONOTONLUK KARSILASTIRMASI")
print("=" * 110)

key_features = ["CVD_zscore_micro", "Imbalance_smooth", "Vol_zscore_micro"]

for sweep_type in ["high", "low"]:
    print(f"\n  {sweep_type.upper()} SWEEP:")
    print(f"    {'Feature':22s} {'Offset':>12s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s} | {'Spread':>8s}")
    print(f"    {'-'*90}")

    for fname in key_features:
        for offset, label in offsets:
            r2 = all_results[offset][0]
            qrows = [qr for qr in r2['quantile_rows'] if qr['sweep_type'] == sweep_type and qr['feature'] == fname]
            if len(qrows) != 5:
                continue
            rows = sorted(qrows, key=lambda x: x['quantile'])
            vals = [rows[q]['cont_rate'] for q in range(5)]
            spread = max(vals) - min(vals)
            vals_str = [f"{v:.1f}%" for v in vals]
            short_label = label.split("(")[0].strip()
            print(f"    {fname:22s} {short_label:>12s} | {vals_str[0]:>7s} | {vals_str[1]:>7s} | {vals_str[2]:>7s} | {vals_str[3]:>7s} | {vals_str[4]:>7s} | {spread:>7.1f}pp")
        print()

# ══════════════════════════════════════════════════════════════
# Top pattern karsilastirmasi
# ══════════════════════════════════════════════════════════════
print()
print("TOP PATTERN KARSILASTIRMASI")
print("=" * 110)

for offset, label in offsets:
    r2 = all_results[offset][0]
    elapsed = all_results[offset][2]
    print(f"\n  OFFSET: {label} ({elapsed:.2f}s)")
    print(f"  Base rates — High: {r2['high_base_cont_rate']:.1f}% | Low: {r2['low_base_cont_rate']:.1f}%")

    for cat_label, key in [("HIGH CONT", "high_cont_patterns"), ("HIGH REV", "high_rev_patterns"),
                            ("LOW CONT", "low_cont_patterns"), ("LOW REV", "low_rev_patterns")]:
        patterns = r2[key]
        if not patterns:
            print(f"    {cat_label}: Pattern yok")
            continue
        # Top 5
        print(f"    {cat_label} (top 5):")
        print(f"      {'#':>2s} {'Rate%':>7s} {'N':>5s} {'WF':>5s} {'Cons%':>6s} | Conditions")
        for i, p in enumerate(patterns[:5]):
            wf = f"{p['wf_positive']}/{p['wf_total']}" if p['wf_total'] > 0 else "N/A"
            print(f"      {i+1:2d} {p['target_rate']:>7.1f} {p['n']:>5d} {wf:>5s} {p['wf_consistency']:>6.1f} | {p['conditions']}")
    print()

# ══════════════════════════════════════════════════════════════
# Ozet tablosu
# ══════════════════════════════════════════════════════════════
print()
print("OZET: EN IYI PATTERN RATE BY OFFSET")
print("=" * 110)
print(f"  {'Offset':>16s} | {'HC best%':>9s} {'HC N':>6s} | {'HR best%':>9s} {'HR N':>6s} | {'LC best%':>9s} {'LC N':>6s} | {'LR best%':>9s} {'LR N':>6s}")
print(f"  {'-'*95}")

for offset, label in offsets:
    r2 = all_results[offset][0]
    short = label.split("(")[0].strip()

    def best(key):
        pats = r2[key]
        if not pats:
            return 0.0, 0
        p = pats[0]
        return p['target_rate'], p['n']

    hc_r, hc_n = best('high_cont_patterns')
    hr_r, hr_n = best('high_rev_patterns')
    lc_r, lc_n = best('low_cont_patterns')
    lr_r, lr_n = best('low_rev_patterns')
    print(f"  {short:>16s} | {hc_r:>8.1f}% {hc_n:>5d} | {hr_r:>8.1f}% {hr_n:>5d} | {lc_r:>8.1f}% {lc_n:>5d} | {lr_r:>8.1f}% {lr_n:>5d}")
