"""Sweep Miner — 5m native data, full pipeline."""
import numpy as np, pandas as pd, sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import rust_engine

# 5m CVD+OI data (native)
df = pd.read_parquet('data/ETHUSDT_5m_cvd_oi_11mo.parquet')
df['dt'] = pd.to_datetime(df['open_time'], unit='ms')

total = len(df)
sa = np.ascontiguousarray
c = sa(df['close'].values, dtype=np.float64)
h = sa(df['high'].values, dtype=np.float64)
l = sa(df['low'].values, dtype=np.float64)
bv = sa(df['buy_vol'].values, dtype=np.float64)
sv = sa(df['sell_vol'].values, dtype=np.float64)
oi = sa(df['open_interest'].fillna(0).values, dtype=np.float64)

print("SWEEP MINER — Parametrik Sweep Pattern Discovery v3")
print("=" * 100)
print(f"Data: {total:,} bar (5m native) | {df['dt'].iloc[0].strftime('%Y-%m-%d')} - {df['dt'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"Features: 7 (saf order flow + istatistik)")
print(f"Grid: simetrik (tp==sl), path-dependent triple barrier")
print(f"Validation: Walk-forward 3m train / 1m test, 8 pencere")
print()

t0 = time.time()
r = rust_engine.run_sweep_mining_py(c, h, l, bv, sv, oi)
elapsed = time.time() - t0

print(f"Toplam sure: {elapsed:.1f}s")
print(f"Toplam sweep eventi: {r['total_sweep_events']}")
print(f"Grid testleri: {r['total_grid_tests']:,}")
print()

for sr in r['sweep_results']:
    print("=" * 100)
    print(f"  {sr['sweep_type_name']} | Toplam event: {sr['total_events']}")
    print("=" * 100)

    bg = sr['best_grid']
    print(f"\n  OPTIMAL PARAMETRELER:")
    print(f"    mult: {bg['mult']:.1f} | timeout: {bg['timeout_bars']} bar ({bg['timeout_bars']*5/60:.0f}h)")
    print(f"    N: {bg['n_total']} | Cont: {bg['n_continuation']} ({bg['continuation_rate']:.1f}%) | Rev: {bg['n_reversal']} ({bg['reversal_rate']:.1f}%) | Timeout: {bg['n_timeout']} ({bg['timeout_rate']:.1f}%)")
    print(f"    Ayrisma gucu: {bg['separation']:.1f}%")

    print(f"\n  BASE RATE: Continuation {sr['base_continuation_rate']:.1f}% | Reversal {sr['base_reversal_rate']:.1f}%")

    print(f"\n  FEATURE KARSILASTIRMA (Continuation vs Reversal):")
    print(f"    {'Feature':22s} | {'Cont Med':>10s} | {'Rev Med':>10s} | {'p-value':>10s} | {'Sig':>4s}")
    print(f"    {'-'*65}")
    for fc in sr['feature_comparisons']:
        sig = "***" if fc['significant'] else ""
        print(f"    {fc['feature']:22s} | {fc['cont_median']:>+10.4f} | {fc['rev_median']:>+10.4f} | {fc['p_value']:>10.6f} | {sig:>4s}")

    print(f"\n  QUANTILE ANALIZI (Continuation rate per quantile):")
    print(f"    {'Feature':22s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s}")
    print(f"    {'-'*65}")
    for fi in range(7):
        rows = [qr for qr in sr['quantile_analysis'] if qr['feature_idx'] == fi]
        if len(rows) == 5:
            vals = [f"{rows[q]['continuation_rate']:.1f}%" for q in range(5)]
            fname = rows[0]['feature_name']
            print(f"    {fname:22s} | {vals[0]:>7s} | {vals[1]:>7s} | {vals[2]:>7s} | {vals[3]:>7s} | {vals[4]:>7s}")

    patterns = sr['top_patterns']
    if patterns:
        print(f"\n  TOP {len(patterns)} PATTERN (FDR + Walk-Forward filtered):")
        print(f"    {'#':>2s} {'ContR%':>7s} {'N':>5s} {'p-val':>10s} {'WF':>5s} {'Cons%':>6s} | Conditions")
        print(f"    {'-'*80}")
        for i, p in enumerate(patterns):
            wf = f"{p['wf_positive']}/{p['wf_total']}" if p['wf_total'] > 0 else "N/A"
            print(f"    {i+1:2d} {p['continuation_rate']:>7.1f} {p['n']:>5d} {p['p_value']:>10.6f} {wf:>5s} {p['wf_consistency']:>6.1f} | {p['conditions']}")
    else:
        print(f"\n  Pattern bulunamadi (FDR veya walk-forward filtresi gecemedi)")

    print()

# Save
out = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'elapsed': round(elapsed, 1),
    'total_events': r['total_sweep_events'],
    'total_grid': r['total_grid_tests'],
    'results': [],
}
for sr in r['sweep_results']:
    out['results'].append({
        'type': sr['sweep_type_name'],
        'events': sr['total_events'],
        'best_grid': dict(sr['best_grid']),
        'base_cont': sr['base_continuation_rate'],
        'base_rev': sr['base_reversal_rate'],
        'features': [dict(fc) for fc in sr['feature_comparisons']],
        'patterns': [dict(p) for p in sr['top_patterns']],
    })

with open('results/sweep_miner_results.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"Kaydedildi: results/sweep_miner_results.json")
