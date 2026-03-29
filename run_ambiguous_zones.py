"""Belirsiz Bolgeler — Cont ve Rev arasinda fark olmayan feature kombinasyonlari
Soru: Hangi kosullarda sonuc tamamen belirsiz? (cont rate ~ base rate)
"""
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
    }

eth_perp = load('data/ETHUSDT_5m_5y.parquet')
btc_perp = load('data/BTCUSDT_5m_5y_perp.parquet')

FEATURE_NAMES = ["CVD_zscore_micro", "CVD_zscore_macro", "OI_change",
                 "Vol_zscore_micro", "Vol_zscore_macro", "Imbalance_smooth", "ATR_percentile"]

print("BELIRSIZ BOLGE ANALIZI")
print("ETH Perp kendi feature'lari + BTC Perp feature'lari ile")
print("Mumun kapanisindaki feature'lar kullaniliyor (t mumunun sonu)")
print("=" * 120)
print()

for asset_label, tf_name, bp in [("ETH Perp", "4H", 48), ("ETH Perp", "1D", 288),
                                   ("BTC Perp", "4H", 48), ("BTC Perp", "1D", 288)]:
    data = eth_perp if "ETH" in asset_label else btc_perp

    # Mum kapanisindaki feature ile analiz (siniflandirma dogrulugu icin)
    r = rust_engine.run_candle_miner_py(
        data['c'], data['h'], data['l'],
        data['bv'], data['sv'], data['oi'],
        bp - 1, bp,  # son bar = mum kapanisi
    )

    print(f"\n{'='*120}")
    print(f"  {asset_label} — {tf_name}")
    print(f"  Base rates: High cont={r['high_base_cont_rate']:.1f}% | Low cont={r['low_base_cont_rate']:.1f}%")
    print(f"  Events: High={r['high_total']} | Low={r['low_total']}")
    print(f"{'='*120}")

    for sweep_type in ["high", "low"]:
        base = r['high_base_cont_rate'] if sweep_type == "high" else r['low_base_cont_rate']
        print(f"\n  {sweep_type.upper()} SWEEP (base cont rate = {base:.1f}%):")

        # Tek feature quantile analizi — hangi quantile'lar base rate'e en yakin?
        print(f"\n    TEK FEATURE — Base rate'e yakinlik:")
        print(f"    {'Feature':22s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s} | {'En yakin':>10s} | {'Fark':>6s}")
        print(f"    {'-'*95}")

        for fname in FEATURE_NAMES:
            qrows = sorted([q for q in r['quantile_rows'] if q['sweep_type']==sweep_type and q['feature']==fname],
                          key=lambda x: x['quantile'])
            if len(qrows) != 5:
                continue
            vals = [qrows[i]['cont_rate'] for i in range(5)]
            # En yakin quantile
            diffs = [abs(v - base) for v in vals]
            closest_q = diffs.index(min(diffs))
            closest_val = vals[closest_q]
            closest_diff = min(diffs)
            vals_str = [f"{v:.1f}%" for v in vals]
            # Base'e yakin olanlari * ile isaretle
            marks = []
            for v in vals:
                if abs(v - base) < 3.0:
                    marks.append("~")
                elif v > base + 5:
                    marks.append("+")
                elif v < base - 5:
                    marks.append("-")
                else:
                    marks.append(" ")
            print(f"    {fname:22s} | {vals_str[0]:>6s}{marks[0]} | {vals_str[1]:>6s}{marks[1]} | {vals_str[2]:>6s}{marks[2]} | {vals_str[3]:>6s}{marks[3]} | {vals_str[4]:>6s}{marks[4]} | Q{closest_q+1}={closest_val:.1f}% | {closest_diff:>5.1f}")

        # Cift feature — belirsiz bolgeler
        # Tum 2-feature quantile kombinasyonlarini tara, base rate'e en yakin olanlari bul
        print(f"\n    CIFT FEATURE — Tam belirsiz bolgeler (cont rate = base rate ± 3pp):")
        print(f"    {'Kosul':50s} | {'Cont%':>7s} | {'N':>5s} | {'Base':>6s} | {'Fark':>6s}")
        print(f"    {'-'*85}")

        # Get all events for this sweep type
        all_qrows = [q for q in r['quantile_rows'] if q['sweep_type'] == sweep_type]

        # Simdi cont ve rev pattern'lerin DISINDA kalan bolgeleri bul
        # Cont patterns: yuksek cont rate
        # Rev patterns: dusuk cont rate (yuksek rev rate)
        # Belirsiz: cont rate ~ base rate

        cont_pats = r['high_cont_patterns'] if sweep_type == "high" else r['low_cont_patterns']
        rev_pats = r['high_rev_patterns'] if sweep_type == "high" else r['low_rev_patterns']

        # Yuksek cont bolgeleri
        if cont_pats:
            print(f"\n    GUCLU CONTINUATION bolgeleri:")
            for p in cont_pats[:5]:
                wf = f"{p['wf_positive']}/{p['wf_total']}" if p['wf_total'] > 0 else "N/A"
                print(f"      {p['conditions']:50s} | cont={p['target_rate']:.1f}% N={p['n']} WF={wf}")

        # Yuksek rev bolgeleri
        if rev_pats:
            print(f"\n    GUCLU REVERSAL bolgeleri:")
            for p in rev_pats[:5]:
                wf = f"{p['wf_positive']}/{p['wf_total']}" if p['wf_total'] > 0 else "N/A"
                rev_rate = 100.0 - p['target_rate']  # target_rate = reversal rate
                print(f"      {p['conditions']:50s} | rev={p['target_rate']:.1f}% N={p['n']} WF={wf}")

        # Belirsiz bolgeleri quantile bazinda tara
        # Q3 (orta) genelde base rate'e yakin
        print(f"\n    BELIRSIZ BOLGELER (cont rate = {base:.0f}% ± 5pp, yani edge yok):")
        ambig_features = []
        for fname in FEATURE_NAMES:
            qrows = sorted([q for q in all_qrows if q['feature']==fname], key=lambda x: x['quantile'])
            if len(qrows) != 5:
                continue
            for i, q in enumerate(qrows):
                if abs(q['cont_rate'] - base) < 5.0:
                    ambig_features.append((fname, i, q['cont_rate'], q['n']))

        if ambig_features:
            print(f"    {'Feature':22s} {'Quantile':>8s} | {'Cont%':>7s} | {'N':>5s} | Yorum")
            print(f"    {'-'*75}")
            for fname, qi, rate, n in ambig_features:
                diff = rate - base
                yorum = "tam base rate" if abs(diff) < 1.0 else f"base + {diff:+.1f}pp"
                print(f"    {fname:22s}    Q{qi+1}    | {rate:>6.1f}% | {n:>5d} | {yorum}")
        print()

    # Feature'lar arasindaki korelasyon — hangi feature'lar birlikte "belirsiz"?
    print(f"\n  SONUC: {asset_label} {tf_name}")
    print(f"  —————————————————————————————————————")

    # Cont rate'ler icin spread hesapla
    for sweep_type in ["high", "low"]:
        base = r['high_base_cont_rate'] if sweep_type == "high" else r['low_base_cont_rate']
        print(f"\n    {sweep_type.upper()} SWEEP feature etkinligi (mum kapanisinda):")
        print(f"    {'Feature':22s} | {'Spread':>7s} | {'Min Q':>7s} | {'Max Q':>7s} | Durum")
        print(f"    {'-'*75}")
        for fname in FEATURE_NAMES:
            qrows = sorted([q for q in r['quantile_rows'] if q['sweep_type']==sweep_type and q['feature']==fname],
                          key=lambda x: x['quantile'])
            if len(qrows) != 5: continue
            vals = [qrows[i]['cont_rate'] for i in range(5)]
            spread = max(vals) - min(vals)
            durum = "GUCLU" if spread > 20 else ("ORTA" if spread > 10 else "ZAYIF/BELIRSIZ")
            print(f"    {fname:22s} | {spread:>6.1f} | {min(vals):>6.1f}% | {max(vals):>6.1f}% | {durum}")

print()
print("=" * 120)
print("GENEL YORUM:")
print("  - 'ZAYIF/BELIRSIZ' feature'lar: Bu feature'in quantile'ina bakmaksizin sonuc rastgele.")
print("  - 'GUCLU' feature'lar: Extreme quantile'larda guclu cont veya rev sinyali var.")
print("  - Belirsiz bolge = GUCLU feature'larin ORTA quantile'lari (Q2-Q4).")
print("  - Strateji: Sadece GUCLU feature'larin Q1 veya Q5'inde islem yap, Q2-Q4'te pas gec.")
print("=" * 120)
