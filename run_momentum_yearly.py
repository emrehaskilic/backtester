"""Momentum lookback yillik tutarlilik kontrolu
3h ROC (en iyi lookback) her yil ayni mi calisiyor?
"""
import numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_parquet('data/ETHUSDT_5m_5y.parquet')
df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
c = df['close'].values.astype(np.float64)
h = df['high'].values.astype(np.float64)
l = df['low'].values.astype(np.float64)
n = len(df)

BP = 48

candles = []
for i in range(n // BP):
    s = i * BP
    e = s + BP
    candles.append({
        'start': s, 'open': c[s],
        'high': np.max(h[s:e]), 'low': np.min(l[s:e]),
        'close': c[e-1],
        'dt': df['dt'].iloc[s],
        'year': df['dt'].iloc[s].year,
    })

events = []
for i in range(1, len(candles)):
    prev = candles[i-1]
    curr = candles[i]
    swept_high = curr['high'] > prev['high']
    swept_low = curr['low'] < prev['low']
    is_green = curr['close'] > curr['open']
    is_red = curr['close'] < curr['open']

    if swept_high and not swept_low:
        sig = "high_cont" if curr['close'] > prev['high'] else ("high_rev" if is_red else None)
    elif swept_low and not swept_high:
        sig = "low_cont" if curr['close'] < prev['low'] else ("low_rev" if is_green else None)
    elif swept_high and swept_low:
        if curr['close'] > prev['high']: sig = "high_cont"
        elif curr['close'] < prev['low']: sig = "low_cont"
        elif is_red: sig = "high_rev"
        elif is_green: sig = "low_rev"
        else: sig = None
    else:
        sig = None

    if sig is None: continue

    bar = curr['start']
    ev = {'idx': i, 'bar': bar, 'signal': sig, 'year': curr['year'], 'dt': curr['dt']}

    # ROC hesapla — test edilen lookback'ler
    for lb in [12, 24, 36, 48, 96]:
        if bar >= lb and c[bar - lb] > 0:
            roc = (c[bar] - c[bar - lb]) / c[bar - lb] * 100
            if sig.startswith('low'):
                roc = -roc
            ev[f'roc_{lb}'] = roc
        else:
            ev[f'roc_{lb}'] = np.nan

    events.append(ev)

edf = pd.DataFrame(events)

print("MOMENTUM YILLIK TUTARLILIK KONTROLU")
print("4H ETH Perp | Mum acilisi (ilk 5dk)")
print("=" * 120)

lookbacks = [(12, "1h"), (24, "2h"), (36, "3h"), (48, "4h"), (96, "8h")]
years = sorted(edf['year'].unique())

for sweep_type in ["high", "low"]:
    mask = edf['signal'].str.startswith(sweep_type) & ~edf['signal'].str.contains('ambig')
    sub = edf[mask].copy()
    is_cont = sub['signal'].str.contains('cont')
    base_all = is_cont.mean() * 100

    print(f"\n{'#'*120}")
    print(f"  {sweep_type.upper()} SWEEP (toplam base cont = {base_all:.1f}%)")
    print(f"{'#'*120}")

    # Her lookback icin yillik Q1 ve Q5
    for lb, lbl in lookbacks:
        col = f'roc_{lb}'
        print(f"\n  ROC {lbl} (lookback={lb} bar):")
        print(f"  {'Yil':>6s} | {'N':>5s} | {'Base':>6s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s} | {'Spread':>7s} | {'Q1 tutarli?':>12s}")
        print(f"  {'-'*100}")

        # Once tum veri
        valid_all = sub.dropna(subset=[col]).copy()
        thresholds_all = np.percentile(valid_all[col], [20, 40, 60, 80])

        for period in ['TOPLAM'] + [str(y) for y in years]:
            if period == 'TOPLAM':
                valid = valid_all.copy()
                is_c = is_cont[valid.index]
            else:
                y = int(period)
                valid = valid_all[valid_all['year'] == y].copy()
                is_c = is_cont[valid.index]

            if len(valid) < 30: continue

            base = is_c.mean() * 100

            # Tum verinin threshold'larini kullan (sabit threshold)
            valid_q = valid.copy()
            valid_q['q'] = 1
            for t_idx, t_val in enumerate(thresholds_all):
                valid_q.loc[valid_q[col] > t_val, 'q'] = t_idx + 2

            vals = []
            for q in range(1, 6):
                grp = valid_q[valid_q['q'] == q]
                if len(grp) < 5:
                    vals.append(float('nan'))
                else:
                    cr = is_c[grp.index].mean() * 100
                    vals.append(cr)

            spread = max(v for v in vals if not np.isnan(v)) - min(v for v in vals if not np.isnan(v)) if any(not np.isnan(v) for v in vals) else 0

            vs = [f"{v:.1f}%" if not np.isnan(v) else "  N/A" for v in vals]
            q1_ok = "OK" if (not np.isnan(vals[0]) and vals[0] > base + 10) else ("ZAYIF" if (not np.isnan(vals[0]) and vals[0] > base) else "TERS")
            n_count = len(valid)

            marker = " ***" if period == 'TOPLAM' else ""
            print(f"  {period:>6s} | {n_count:>5d} | {base:>5.1f}% | {vs[0]:>7s} | {vs[1]:>7s} | {vs[2]:>7s} | {vs[3]:>7s} | {vs[4]:>7s} | {spread:>6.1f} | {q1_ok:>12s}{marker}")

# Q1 tutarliligi ozeti
print(f"\n\n{'='*120}")
print("OZET: ROC 3h Q1 (en iyi lookback) YILLIK TUTARLILIK")
print("=" * 120)

col = 'roc_36'
for sweep_type in ["high", "low"]:
    mask = edf['signal'].str.startswith(sweep_type) & ~edf['signal'].str.contains('ambig')
    sub = edf[mask].copy()
    is_cont = sub['signal'].str.contains('cont')

    valid_all = sub.dropna(subset=[col]).copy()
    thresholds = np.percentile(valid_all[col], [20, 40, 60, 80])

    print(f"\n  {sweep_type.upper()} SWEEP — ROC 3h:")
    print(f"  {'Yil':>6s} | {'Toplam N':>8s} | {'Q1 N':>5s} | {'Q1 Cont%':>8s} | {'Q5 N':>5s} | {'Q5 Cont%':>8s} | {'Q1-Q5':>7s}")
    print(f"  {'-'*60}")

    for y in years:
        ydata = valid_all[valid_all['year'] == y].copy()
        if len(ydata) < 20: continue

        ydata['q'] = 1
        for t_idx, t_val in enumerate(thresholds):
            ydata.loc[ydata[col] > t_val, 'q'] = t_idx + 2

        q1 = ydata[ydata['q'] == 1]
        q5 = ydata[ydata['q'] == 5]

        q1_cr = is_cont[q1.index].mean() * 100 if len(q1) > 5 else float('nan')
        q5_cr = is_cont[q5.index].mean() * 100 if len(q5) > 5 else float('nan')
        diff = q1_cr - q5_cr if not (np.isnan(q1_cr) or np.isnan(q5_cr)) else float('nan')

        q1_s = f"{q1_cr:.1f}%" if not np.isnan(q1_cr) else "N/A"
        q5_s = f"{q5_cr:.1f}%" if not np.isnan(q5_cr) else "N/A"
        d_s = f"{diff:+.1f}" if not np.isnan(diff) else "N/A"

        print(f"  {y:>6d} | {len(ydata):>8d} | {len(q1):>5d} | {q1_s:>8s} | {len(q5):>5d} | {q5_s:>8s} | {d_s:>7s}")

print()
