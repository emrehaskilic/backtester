"""15M Sweep Candle Analysis + Miner — 5m native data."""
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

print("15M SWEEP CANDLE ANALYSIS + MINER")
print("=" * 100)
print(f"Data: {total:,} bar (5m native) | {df['dt'].iloc[0].strftime('%Y-%m-%d')} - {df['dt'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"15M candle sayisi: ~{total // 3:,}")
print(f"Features: 7 (saf order flow + istatistik)")
print()

# ══════════════════════════════════════════════════════════════
# PART 1: Candle Analysis (continuation/reversal classification + aftermath)
# ══════════════════════════════════════════════════════════════
print("PART 1: CANDLE ANALYSIS (15M)")
print("=" * 100)

t0 = time.time()
r1 = rust_engine.run_candle_analysis_15m_py(c, h, l, bv, sv, oi)
elapsed1 = time.time() - t0

print(f"Sure: {elapsed1:.2f}s")
print(f"Toplam 15M mum: {r1['total_15m_candles']}")
print()

total_events = r1['high_cont'] + r1['high_rev'] + r1['high_ambig'] + r1['low_cont'] + r1['low_rev'] + r1['low_ambig'] + r1['inside_bar']
print(f"  SINYAL DAGILIMI:")
print(f"    High Continuation (LONG) : {r1['high_cont']:>5d} ({r1['high_cont']/total_events*100:.1f}%)")
print(f"    High Reversal (SHORT)    : {r1['high_rev']:>5d} ({r1['high_rev']/total_events*100:.1f}%)")
print(f"    High Ambiguous (pas)     : {r1['high_ambig']:>5d} ({r1['high_ambig']/total_events*100:.1f}%)")
print(f"    Low Continuation (SHORT) : {r1['low_cont']:>5d} ({r1['low_cont']/total_events*100:.1f}%)")
print(f"    Low Reversal (LONG)      : {r1['low_rev']:>5d} ({r1['low_rev']/total_events*100:.1f}%)")
print(f"    Low Ambiguous (pas)      : {r1['low_ambig']:>5d} ({r1['low_ambig']/total_events*100:.1f}%)")
print(f"    Inside Bar (pas)         : {r1['inside_bar']:>5d} ({r1['inside_bar']/total_events*100:.1f}%)")
print()

# High sweep base rates
high_total = r1['high_cont'] + r1['high_rev'] + r1['high_ambig']
low_total = r1['low_cont'] + r1['low_rev'] + r1['low_ambig']
if high_total > 0:
    print(f"  HIGH SWEEP: Cont {r1['high_cont']/high_total*100:.1f}% | Rev {r1['high_rev']/high_total*100:.1f}% | Ambig {r1['high_ambig']/high_total*100:.1f}%  (N={high_total})")
if low_total > 0:
    print(f"  LOW  SWEEP: Cont {r1['low_cont']/low_total*100:.1f}% | Rev {r1['low_rev']/low_total*100:.1f}% | Ambig {r1['low_ambig']/low_total*100:.1f}%  (N={low_total})")
print()

# Aftermath
for label, key in [("HIGH CONTINUATION (LONG)", "high_cont_aftermath"),
                    ("HIGH REVERSAL (SHORT)", "high_rev_aftermath"),
                    ("LOW CONTINUATION (SHORT)", "low_cont_aftermath"),
                    ("LOW REVERSAL (LONG)", "low_rev_aftermath")]:
    afts = r1[key]
    if not afts:
        continue
    print(f"  AFTERMATH: {label}")
    print(f"    {'Horizon':>8s} {'Time':>8s} {'AvgRet%':>8s} {'MedRet%':>8s} {'WinRate':>8s} {'MFE%':>8s} {'MAE%':>8s} {'N':>6s}")
    print(f"    {'-'*65}")
    for a in afts:
        time_str = f"{a['horizon']*15}m"
        if a['horizon']*15 >= 60:
            time_str = f"{a['horizon']*15/60:.0f}h"
        print(f"    {a['horizon']:>8d} {time_str:>8s} {a['avg_return']:>+8.3f} {a['median_return']:>+8.3f} {a['win_rate']:>7.1f}% {a['max_favorable']:>+8.3f} {a['max_adverse']:>+8.3f} {a['sample']:>6d}")
    print()

# Feature comparisons
print(f"  FEATURE KARSILASTIRMA (Continuation vs Reversal):")
for sweep_type in ["high", "low"]:
    fcs = [fc for fc in r1['feature_comparisons'] if fc['sweep_type'] == sweep_type]
    if not fcs:
        continue
    print(f"\n    {sweep_type.upper()} SWEEP:")
    print(f"    {'Feature':22s} | {'Cont Med':>10s} | {'Rev Med':>10s} | {'Ambig Med':>10s} | {'p-value':>10s} | {'Sig':>4s}")
    print(f"    {'-'*80}")
    for fc in fcs:
        sig = "***" if fc['significant'] else ""
        print(f"    {fc['feature']:22s} | {fc['cont_median']:>+10.4f} | {fc['rev_median']:>+10.4f} | {fc['ambig_median']:>+10.4f} | {fc['p_value']:>10.6f} | {sig:>4s}")

print()
print()

# ══════════════════════════════════════════════════════════════
# PART 2: Candle Miner (quantile pattern discovery + walk-forward)
# ══════════════════════════════════════════════════════════════
print("PART 2: CANDLE MINER (15M)")
print("=" * 100)

t0 = time.time()
r2 = rust_engine.run_candle_miner_15m_py(c, h, l, bv, sv, oi)
elapsed2 = time.time() - t0

print(f"Sure: {elapsed2:.2f}s")
print(f"High sweep events: {r2['high_total']} | Base cont rate: {r2['high_base_cont_rate']:.1f}%")
print(f"Low  sweep events: {r2['low_total']} | Base cont rate: {r2['low_base_cont_rate']:.1f}%")
print()

# Quantile analysis
print(f"  QUANTILE ANALIZI (Continuation rate per quantile):")
for sweep_type in ["high", "low"]:
    qrows = [qr for qr in r2['quantile_rows'] if qr['sweep_type'] == sweep_type]
    if not qrows:
        continue
    features = sorted(set(qr['feature'] for qr in qrows))
    print(f"\n    {sweep_type.upper()} SWEEP:")
    print(f"    {'Feature':22s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s}")
    print(f"    {'-'*65}")
    for fname in features:
        rows = sorted([qr for qr in qrows if qr['feature'] == fname], key=lambda x: x['quantile'])
        if len(rows) == 5:
            vals = [f"{rows[q]['cont_rate']:.1f}%" for q in range(5)]
            print(f"    {fname:22s} | {vals[0]:>7s} | {vals[1]:>7s} | {vals[2]:>7s} | {vals[3]:>7s} | {vals[4]:>7s}")

print()

# Patterns
for label, key in [("HIGH CONT", "high_cont_patterns"), ("HIGH REV", "high_rev_patterns"),
                    ("LOW CONT", "low_cont_patterns"), ("LOW REV", "low_rev_patterns")]:
    patterns = r2[key]
    print(f"  {label} PATTERNS ({len(patterns)} found):")
    if patterns:
        print(f"    {'#':>2s} {'Rate%':>7s} {'N':>5s} {'p-val':>10s} {'WF':>5s} {'Cons%':>6s} | Conditions")
        print(f"    {'-'*80}")
        for i, p in enumerate(patterns):
            wf = f"{p['wf_positive']}/{p['wf_total']}" if p['wf_total'] > 0 else "N/A"
            print(f"    {i+1:2d} {p['target_rate']:>7.1f} {p['n']:>5d} {p['p_value']:>10.6f} {wf:>5s} {p['wf_consistency']:>6.1f} | {p['conditions']}")
    else:
        print(f"    Pattern bulunamadi (FDR veya walk-forward filtresi gecemedi)")
    print()

# Save results
out = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'timeframe': '15m',
    'elapsed_analysis': round(elapsed1, 2),
    'elapsed_miner': round(elapsed2, 2),
    'analysis': {
        'total_15m_candles': r1['total_15m_candles'],
        'high_cont': r1['high_cont'], 'high_rev': r1['high_rev'], 'high_ambig': r1['high_ambig'],
        'low_cont': r1['low_cont'], 'low_rev': r1['low_rev'], 'low_ambig': r1['low_ambig'],
        'inside_bar': r1['inside_bar'],
        'feature_comparisons': [dict(fc) for fc in r1['feature_comparisons']],
        'high_cont_aftermath': [dict(a) for a in r1['high_cont_aftermath']],
        'high_rev_aftermath': [dict(a) for a in r1['high_rev_aftermath']],
        'low_cont_aftermath': [dict(a) for a in r1['low_cont_aftermath']],
        'low_rev_aftermath': [dict(a) for a in r1['low_rev_aftermath']],
    },
    'miner': {
        'high_base_cont_rate': r2['high_base_cont_rate'],
        'low_base_cont_rate': r2['low_base_cont_rate'],
        'high_total': r2['high_total'],
        'low_total': r2['low_total'],
        'high_cont_patterns': [dict(p) for p in r2['high_cont_patterns']],
        'high_rev_patterns': [dict(p) for p in r2['high_rev_patterns']],
        'low_cont_patterns': [dict(p) for p in r2['low_cont_patterns']],
        'low_rev_patterns': [dict(p) for p in r2['low_rev_patterns']],
    },
}

with open('results/15m_sweep_candle_results.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"Kaydedildi: results/15m_sweep_candle_results.json")
