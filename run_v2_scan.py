"""V2 Feature Scan — 28 feature ile 4H sweep analizi, 5 yillik ETH Perp"""
import numpy as np, pandas as pd, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import rust_engine

df = pd.read_parquet('data/ETHUSDT_5m_5y.parquet')
sa = np.ascontiguousarray
c = sa(df['close'].values, dtype=np.float64)
h = sa(df['high'].values, dtype=np.float64)
l = sa(df['low'].values, dtype=np.float64)
bv = sa(df['buy_vol'].values, dtype=np.float64)
sv = sa(df['sell_vol'].values, dtype=np.float64)

print("V2 FEATURE SCAN — 28 Feature, 4H ETH Perp, 5 Yil")
print("=" * 120)
print(f"Data: {len(df):,} bars")
print()

for tf_name, bp in [("4H", 48)]:
    for offset, olabel in [(0, "ilk 5dk"), (11, "ilk 1h"), (47, "mum kapanisi")]:
        print(f"\n{'#'*120}")
        print(f"  {tf_name} — {olabel} (offset={offset})")
        print(f"{'#'*120}")

        t0 = time.time()
        r = rust_engine.run_sweep_miner_v2_py(c, h, l, bv, sv, offset, bp)
        elapsed = time.time() - t0

        print(f"  Sure: {elapsed:.1f}s")
        print(f"  Base: High={r['high_base_cont_rate']:.1f}% Low={r['low_base_cont_rate']:.1f}%")
        print(f"  Events: High={r['high_total']} Low={r['low_total']}")

        # Quantile — spread bazli siralama
        print(f"\n  FEATURE ETKINLIGI (spread'e gore siralanmis):")
        for st in ["high", "low"]:
            print(f"\n    {st.upper()} SWEEP:")
            feat_data = {}
            for qr in r['quantile_rows']:
                if qr['sweep_type'] != st: continue
                fname = qr['feature']
                if fname not in feat_data: feat_data[fname] = {}
                feat_data[fname][qr['quantile']] = (qr['cont_rate'], qr['n'])

            rows = []
            for fname, qs in feat_data.items():
                if len(qs) != 5: continue
                vals = [qs[q][0] for q in range(5)]
                spread = max(vals) - min(vals)
                rows.append((fname, vals, spread))

            rows.sort(key=lambda x: -x[2])

            print(f"    {'Feature':>25s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s} | {'Spread':>7s}")
            print(f"    {'-'*85}")
            for fname, vals, spread in rows:
                vs = [f"{v:.1f}%" for v in vals]
                marker = " ***" if spread > 15 else (" **" if spread > 10 else (" *" if spread > 5 else ""))
                print(f"    {fname:>25s} | {vs[0]:>7s} | {vs[1]:>7s} | {vs[2]:>7s} | {vs[3]:>7s} | {vs[4]:>7s} | {spread:>6.1f}{marker}")

        # Patterns
        print(f"\n  TOP PATTERNS:")
        for cat, key in [("HIGH CONT", "high_cont_patterns"), ("HIGH REV", "high_rev_patterns"),
                          ("LOW CONT", "low_cont_patterns"), ("LOW REV", "low_rev_patterns")]:
            pats = r[key]
            print(f"\n    {cat} ({len(pats)} found):")
            if pats:
                print(f"    {'#':>3s} {'Rate%':>7s} {'N':>5s} {'WF':>7s} {'Cons%':>6s} | Conditions")
                print(f"    {'-'*90}")
                for i, p in enumerate(pats[:10]):
                    wf = f"{p['wf_positive']}/{p['wf_total']}" if p['wf_total'] > 0 else "N/A"
                    print(f"    {i+1:>3d} {p['target_rate']:>6.1f}% {p['n']:>5d} {wf:>7s} {p['wf_consistency']:>5.1f}% | {p['conditions']}")
            else:
                print(f"    Pattern yok")

print()
