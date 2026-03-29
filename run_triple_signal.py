"""Uclu Sinyal Birlestirme
1. ROC 3h (pre-sweep momentum)
2. Pre-CVD 4h (sweep oncesi CVD tukenmisligi)
3. Ilk 1h fiyat (erken teyit)

Hepsini 4H sweep event'inde birlestirip 3D grid olustur.
"""
import numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_parquet('data/ETHUSDT_5m_5y.parquet')
c = df['close'].values.astype(np.float64)
h = df['high'].values.astype(np.float64)
l = df['low'].values.astype(np.float64)
bv = df['buy_vol'].values.astype(np.float64)
sv = df['sell_vol'].values.astype(np.float64)
n = len(df)

delta = bv - sv
cvd = np.cumsum(delta)

BP = 48
EARLY_BARS = 12  # ilk 1h

candles = []
for i in range(n // BP):
    s = i * BP
    e = s + BP
    candles.append({
        'start': s, 'open': c[s],
        'high': np.max(h[s:e]), 'low': np.min(l[s:e]),
        'close': c[e-1],
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

    # Sinyal 1: ROC 3h (36 bar oncesi fiyat degisimi)
    roc_3h = np.nan
    if bar >= 36 and c[bar - 36] > 0:
        roc_3h = (c[bar] - c[bar - 36]) / c[bar - 36] * 100
        if sig.startswith('low'): roc_3h = -roc_3h

    # Sinyal 2: Pre-CVD 4h (48 bar oncesi CVD degisimi)
    pre_cvd = np.nan
    if bar >= 48:
        pre_cvd = cvd[bar] - cvd[bar - 48]
        if sig.startswith('low'): pre_cvd = -pre_cvd

    # Sinyal 3: Ilk 1h fiyat (12 bar sonrasi)
    early_price = np.nan
    if bar + EARLY_BARS < n:
        early_price = (c[bar + EARLY_BARS] - c[bar]) / c[bar] * 100
        if sig.startswith('low'): early_price = -early_price

    events.append({
        'bar': bar, 'signal': sig,
        'roc_3h': roc_3h, 'pre_cvd': pre_cvd, 'early_price': early_price,
    })

edf = pd.DataFrame(events)

print("UCLU SINYAL BIRLESTIRME — 4H ETH Perp 5 Yil")
print("Sinyal 1: ROC 3h (momentum)")
print("Sinyal 2: Pre-CVD 4h (tukenmislik)")
print("Sinyal 3: Ilk 1h fiyat (erken teyit)")
print("=" * 120)
print(f"Toplam sweep event: {len(edf)}")
print()

# Tum sweep'leri birlestir (high + low aligned)
valid = edf.dropna(subset=['roc_3h', 'pre_cvd', 'early_price']).copy()
is_cont = valid['signal'].str.contains('cont')
base = is_cont.mean() * 100
print(f"Valid event: {len(valid)} | Base cont: {base:.1f}%")
print()

# Quintile thresholds
roc_th = np.percentile(valid['roc_3h'], [20, 40, 60, 80])
cvd_th = np.percentile(valid['pre_cvd'], [20, 40, 60, 80])
early_th = np.percentile(valid['early_price'], [20, 40, 60, 80])

def get_q5(val, th):
    q = 1
    for t in th:
        if val > t: q += 1
    return q

valid['roc_q'] = valid['roc_3h'].apply(lambda x: get_q5(x, roc_th))
valid['cvd_q'] = valid['pre_cvd'].apply(lambda x: get_q5(x, cvd_th))
valid['early_q'] = valid['early_price'].apply(lambda x: get_q5(x, early_th))

# ══════════════════════════════════════════════════════════════
# Ikili kombinasyonlar — hangisi en iyi ayristirici?
# ══════════════════════════════════════════════════════════════

print("IKILI KOMBINASYONLAR — 5x5 grid cont rate")
print("=" * 120)

combos = [
    ('roc_q', 'cvd_q', 'ROC 3h', 'Pre-CVD 4h'),
    ('roc_q', 'early_q', 'ROC 3h', 'Ilk 1h fiyat'),
    ('cvd_q', 'early_q', 'Pre-CVD 4h', 'Ilk 1h fiyat'),
]

for col1, col2, name1, name2 in combos:
    print(f"\n  {name1} × {name2}:")
    print(f"  {'':>12s} | {name2+' Q1':>10s} | {'Q2':>10s} | {'Q3':>10s} | {'Q4':>10s} | {'Q5':>10s}")
    print(f"  {'-'*72}")

    for q1 in range(1, 6):
        cells = []
        for q2 in range(1, 6):
            mask = (valid[col1] == q1) & (valid[col2] == q2)
            grp = valid[mask]
            if len(grp) < 15:
                cells.append(f"{'':>10s}")
            else:
                cr = is_cont[grp.index].mean() * 100
                marker = "***" if cr > base + 15 else ("!!!" if cr < base - 15 else "   ")
                cells.append(f"{cr:>5.0f}% {len(grp):>3d}{marker}")
        q1_label = f"{name1} Q{q1}"
        print(f"  {q1_label:>12s} | {cells[0]:>10s} | {cells[1]:>10s} | {cells[2]:>10s} | {cells[3]:>10s} | {cells[4]:>10s}")
    print(f"  {'Base':>12s} | {base:.0f}%")

# ══════════════════════════════════════════════════════════════
# Uclu kombinasyon — 3x3x3 grid (tercile)
# ══════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print("UCLU KOMBINASYON — ROC 3h × Pre-CVD 4h × Ilk 1h fiyat")
print("Her eksen: Ters (Q1-Q2) / Notr (Q3) / Ayni (Q4-Q5)")
print("=" * 120)

def tercile(q):
    if q <= 2: return "Ters"
    elif q == 3: return "Notr"
    else: return "Ayni"

valid['roc_t'] = valid['roc_q'].apply(tercile)
valid['cvd_t'] = valid['cvd_q'].apply(tercile)
valid['early_t'] = valid['early_q'].apply(tercile)

for roc_label in ["Ters", "Notr", "Ayni"]:
    roc_mask = valid['roc_t'] == roc_label
    roc_n = roc_mask.sum()
    roc_meaning = "momentum KARSI yonden" if roc_label == "Ters" else ("momentum NOTR" if roc_label == "Notr" else "momentum AYNI yonde")

    print(f"\n  ROC 3h = {roc_label} ({roc_meaning}, N={roc_n}):")
    print(f"  {'':>15s} | {'Ilk 1h Ters':>13s} | {'Ilk 1h Notr':>13s} | {'Ilk 1h Ayni':>13s}")
    print(f"  {'-'*60}")

    for cvd_label in ["Ters", "Notr", "Ayni"]:
        cells = []
        for early_label in ["Ters", "Notr", "Ayni"]:
            mask = roc_mask & (valid['cvd_t'] == cvd_label) & (valid['early_t'] == early_label)
            grp = valid[mask]
            if len(grp) < 15:
                cells.append(f"{'N<15':>13s}")
            else:
                cr = is_cont[grp.index].mean() * 100
                marker = " ***" if cr > 80 else (" !!!" if cr < 45 else "    ")
                cells.append(f"{cr:>5.0f}% N={len(grp):<3d}{marker}")
        cvd_meaning = "CVD " + cvd_label
        print(f"  {cvd_meaning:>15s} | {cells[0]:>13s} | {cells[1]:>13s} | {cells[2]:>13s}")
    print(f"  {'Base':>15s} | {base:.0f}%")

# ══════════════════════════════════════════════════════════════
# Bolge siniflandirmasi
# ══════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print("BOLGE SINIFLANDIRMASI")
print("=" * 120)

# Her hucreyi sinifla
results = []
for roc_label in ["Ters", "Notr", "Ayni"]:
    for cvd_label in ["Ters", "Notr", "Ayni"]:
        for early_label in ["Ters", "Notr", "Ayni"]:
            mask = (valid['roc_t'] == roc_label) & (valid['cvd_t'] == cvd_label) & (valid['early_t'] == early_label)
            grp = valid[mask]
            if len(grp) < 10: continue
            cr = is_cont[grp.index].mean() * 100

            if cr >= 80:
                zone = "GUCLU CONT"
            elif cr >= 70:
                zone = "CONT"
            elif cr <= 40:
                zone = "GUCLU REV"
            elif cr <= 50:
                zone = "REV"
            else:
                zone = "BELIRSIZ"

            results.append({
                'roc': roc_label, 'cvd': cvd_label, 'early': early_label,
                'cont_rate': cr, 'n': len(grp), 'zone': zone,
            })

rdf = pd.DataFrame(results)

# Zone ozeti
print(f"\n  {'Zone':>12s} | {'Hucre sayisi':>12s} | {'Toplam N':>10s} | {'Ort Cont%':>10s} | {'Tum eventlerin %':>15s}")
print(f"  {'-'*70}")

total_events = len(valid)
for zone in ["GUCLU CONT", "CONT", "BELIRSIZ", "REV", "GUCLU REV"]:
    z = rdf[rdf['zone'] == zone]
    if len(z) == 0: continue
    total_n = z['n'].sum()
    avg_cr = (z['cont_rate'] * z['n']).sum() / total_n if total_n > 0 else 0
    print(f"  {zone:>12s} | {len(z):>12d} | {total_n:>10d} | {avg_cr:>9.1f}% | {total_n/total_events*100:>14.1f}%")

# Detayli liste
print(f"\n  GUCLU CONT bolgesi (cont >= 80%):")
print(f"  {'ROC':>6s} | {'CVD':>6s} | {'Early':>6s} | {'Cont%':>7s} | {'N':>5s}")
print(f"  {'-'*40}")
for _, r in rdf[rdf['zone'] == 'GUCLU CONT'].sort_values('cont_rate', ascending=False).iterrows():
    print(f"  {r['roc']:>6s} | {r['cvd']:>6s} | {r['early']:>6s} | {r['cont_rate']:>6.1f}% | {r['n']:>5d}")

print(f"\n  CONT bolgesi (cont 70-80%):")
print(f"  {'ROC':>6s} | {'CVD':>6s} | {'Early':>6s} | {'Cont%':>7s} | {'N':>5s}")
print(f"  {'-'*40}")
for _, r in rdf[rdf['zone'] == 'CONT'].sort_values('cont_rate', ascending=False).iterrows():
    print(f"  {r['roc']:>6s} | {r['cvd']:>6s} | {r['early']:>6s} | {r['cont_rate']:>6.1f}% | {r['n']:>5d}")

print(f"\n  REV + GUCLU REV bolgesi (cont <= 50%):")
print(f"  {'ROC':>6s} | {'CVD':>6s} | {'Early':>6s} | {'Cont%':>7s} | {'Rev%':>7s} | {'N':>5s}")
print(f"  {'-'*50}")
for _, r in rdf[rdf['zone'].isin(['REV', 'GUCLU REV'])].sort_values('cont_rate').iterrows():
    print(f"  {r['roc']:>6s} | {r['cvd']:>6s} | {r['early']:>6s} | {r['cont_rate']:>6.1f}% | {100-r['cont_rate']:>6.1f}% | {r['n']:>5d}")

print(f"\n  BELIRSIZ bolgesi (cont 50-70%):")
print(f"  {'ROC':>6s} | {'CVD':>6s} | {'Early':>6s} | {'Cont%':>7s} | {'N':>5s}")
print(f"  {'-'*40}")
for _, r in rdf[rdf['zone'] == 'BELIRSIZ'].sort_values('cont_rate', ascending=False).iterrows():
    print(f"  {r['roc']:>6s} | {r['cvd']:>6s} | {r['early']:>6s} | {r['cont_rate']:>6.1f}% | {r['n']:>5d}")

print()
