"""Momentum Lookback Taramasi
Ayni formul: (close_now - close_N_bar_once) / close_N_bar_once * 100
Farkli N degerleri ile hangi lookback en iyi sinyal veriyor?
4H ETH Perp sweep, mum acilisindaki (ilk 5dk) feature
"""
import numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_parquet('data/ETHUSDT_5m_5y.parquet')
c = df['close'].values.astype(np.float64)
h = df['high'].values.astype(np.float64)
l = df['low'].values.astype(np.float64)
n = len(df)

BP = 48  # 4H

# Mumlar
candles = []
for i in range(n // BP):
    s = i * BP
    e = s + BP
    candles.append({
        'start': s, 'open': c[s],
        'high': np.max(h[s:e]), 'low': np.min(l[s:e]),
        'close': c[e-1],
    })

# Sweep event'leri
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
    events.append({'idx': i, 'bar': curr['start'], 'signal': sig})

edf = pd.DataFrame(events)

# Farkli lookback'ler (5m bar cinsinden)
lookbacks = [
    (3, "15dk"),
    (6, "30dk"),
    (12, "1h"),
    (24, "2h"),
    (36, "3h"),
    (48, "4h"),
    (72, "6h"),
    (96, "8h"),
    (144, "12h"),
    (192, "16h"),
    (288, "1d"),
    (576, "2d"),
    (864, "3d"),
    (2016, "1w"),
]

print("MOMENTUM LOOKBACK TARAMASI")
print(f"4H ETH Perp | {len(events)} sweep event | 5 yil")
print(f"Olcum zamani: mum acilisi (ilk 5dk, offset=0)")
print(f"Formul: ROC = (close_now - close_N_once) / close_N_once * 100")
print("=" * 120)

# Her lookback icin ROC hesapla
for lb, lb_label in lookbacks:
    col = f'roc_{lb}'
    for ev in events:
        bar = ev['bar']
        if bar >= lb and c[bar - lb] > 0:
            roc = (c[bar] - c[bar - lb]) / c[bar - lb] * 100
            # High sweep icin: negatif ROC = karsi yonden geliyor
            if ev['signal'].startswith('low'):
                roc = -roc  # align: pozitif = sweep yonunde
            ev[col] = roc
        else:
            ev[col] = np.nan

edf = pd.DataFrame(events)

# Quantile analizi her lookback icin
print(f"\nTEK LOOKBACK QUANTILE ANALIZI:")
print(f"{'Lookback':>10s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s} | {'Spread':>7s} | {'Q1 yorum':>30s}")
print(f"{'-'*105}")

best_spreads = []

for sweep_type in ["high", "low"]:
    mask = edf['signal'].str.startswith(sweep_type) & ~edf['signal'].str.contains('ambig')
    sub = edf[mask].copy()
    is_cont = sub['signal'].str.contains('cont')
    base = is_cont.mean() * 100

    print(f"\n  {sweep_type.upper()} SWEEP (base cont = {base:.1f}%):")
    print(f"  {'Lookback':>10s} | {'Q1':>7s} | {'Q2':>7s} | {'Q3':>7s} | {'Q4':>7s} | {'Q5':>7s} | {'Spread':>7s} | Q1 anlam")
    print(f"  {'-'*100}")

    for lb, lb_label in lookbacks:
        col = f'roc_{lb}'
        valid = sub.dropna(subset=[col])
        if len(valid) < 100: continue

        try:
            valid['q'] = pd.qcut(valid[col], 5, labels=[1,2,3,4,5], duplicates='drop')
        except:
            continue

        vals = []
        for q in range(1, 6):
            grp = valid[valid['q'] == q]
            cr = is_cont[grp.index].mean() * 100
            vals.append(cr)

        spread = max(vals) - min(vals)
        best_spreads.append((sweep_type, lb, lb_label, spread, vals))

        vs = [f"{v:.1f}%" for v in vals]
        q1_meaning = f"son {lb_label} TERS yone gitmis" if vals[0] > base else f"son {lb_label} AYNI yone gitmis"
        marker = " ***" if spread > 25 else (" **" if spread > 15 else (" *" if spread > 10 else ""))
        print(f"  {lb_label:>10s} | {vs[0]:>7s} | {vs[1]:>7s} | {vs[2]:>7s} | {vs[3]:>7s} | {vs[4]:>7s} | {spread:>6.1f}{marker} | {q1_meaning}")

# CIFT LOOKBACK kombinasyonlari
print(f"\n\n{'='*120}")
print(f"CIFT LOOKBACK KOMBINASYONLARI (3x3 grid)")
print(f"En iyi tekil lookback'lerin cift kombinasyonu")
print(f"{'='*120}")

# Top 5 lookback per sweep type
for sweep_type in ["high", "low"]:
    type_spreads = [(lb, lbl, sp, vals) for st, lb, lbl, sp, vals in best_spreads if st == sweep_type]
    type_spreads.sort(key=lambda x: -x[3])
    top5 = type_spreads[:6]

    mask = edf['signal'].str.startswith(sweep_type) & ~edf['signal'].str.contains('ambig')
    sub = edf[mask].copy()
    is_cont = sub['signal'].str.contains('cont')
    base = is_cont.mean() * 100

    print(f"\n  {sweep_type.upper()} SWEEP (base={base:.1f}%):")

    # Her cift icin 3x3 grid
    for i in range(len(top5)):
        for j in range(i+1, len(top5)):
            lb1, lbl1, sp1, _ = top5[i]
            lb2, lbl2, sp2, _ = top5[j]

            col1 = f'roc_{lb1}'
            col2 = f'roc_{lb2}'

            valid = sub.dropna(subset=[col1, col2])
            if len(valid) < 200: continue

            try:
                valid['q1'] = pd.qcut(valid[col1], 3, labels=['Ters','Notr','Ayni'], duplicates='drop')
                valid['q2'] = pd.qcut(valid[col2], 3, labels=['Ters','Notr','Ayni'], duplicates='drop')
            except:
                continue

            # Has signal?
            has_extreme = False
            cells = []
            for q1v in ['Ters','Notr','Ayni']:
                row = []
                for q2v in ['Ters','Notr','Ayni']:
                    m = (valid['q1']==q1v) & (valid['q2']==q2v)
                    grp = valid[m]
                    if len(grp) < 20:
                        row.append(("---", 0))
                    else:
                        cr = is_cont[grp.index].mean() * 100
                        row.append((f"{cr:.0f}%", len(grp)))
                        if cr > base + 20 or cr < base - 15:
                            has_extreme = True
                cells.append((q1v, row))

            if not has_extreme: continue

            print(f"\n    ROC {lbl1} × ROC {lbl2}:")
            print(f"    {'ROC '+lbl1:>12s} | {'ROC '+lbl2+' Ters':>13s} | {'Notr':>13s} | {'Ayni':>13s}")
            print(f"    {'-'*58}")
            for q1v, row in cells:
                cs = [f"{r[0]:>6s} N={r[1]:<4d}" if r[1] > 0 else f"{'---':>6s}      " for r in row]
                print(f"    {q1v:>12s} | {cs[0]:>13s} | {cs[1]:>13s} | {cs[2]:>13s}")
            print(f"    {'Base':>12s} | {base:.0f}%")

print()
