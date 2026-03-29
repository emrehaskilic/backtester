"""Bagimsiz Sinyal Oylama Sistemi
Her sinyal kendi basina CONT / REV / BELIRSIZ diyor.
3 sinyalin oylari birlestirilip sonuc belirleniyor.
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
EARLY_BARS = 12

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

    roc_3h = np.nan
    if bar >= 36 and c[bar - 36] > 0:
        roc_3h = (c[bar] - c[bar - 36]) / c[bar - 36] * 100
        if sig.startswith('low'): roc_3h = -roc_3h

    pre_cvd = np.nan
    if bar >= 48:
        pre_cvd = cvd[bar] - cvd[bar - 48]
        if sig.startswith('low'): pre_cvd = -pre_cvd

    early_price = np.nan
    if bar + EARLY_BARS < n:
        early_price = (c[bar + EARLY_BARS] - c[bar]) / c[bar] * 100
        if sig.startswith('low'): early_price = -early_price

    events.append({
        'bar': bar, 'signal': sig,
        'roc_3h': roc_3h, 'pre_cvd': pre_cvd, 'early_price': early_price,
    })

edf = pd.DataFrame(events)
valid = edf.dropna(subset=['roc_3h', 'pre_cvd', 'early_price']).copy()
is_cont = valid['signal'].str.contains('cont')
base = is_cont.mean() * 100

# Quintile thresholds
roc_th = np.percentile(valid['roc_3h'], [20, 40, 60, 80])
cvd_th = np.percentile(valid['pre_cvd'], [20, 40, 60, 80])
early_th = np.percentile(valid['early_price'], [20, 40, 60, 80])

def get_q5(val, th):
    q = 1
    for t in th:
        if val > t: q += 1
    return q

valid = valid.copy()
valid['roc_q'] = valid['roc_3h'].apply(lambda x: get_q5(x, roc_th))
valid['cvd_q'] = valid['pre_cvd'].apply(lambda x: get_q5(x, cvd_th))
valid['early_q'] = valid['early_price'].apply(lambda x: get_q5(x, early_th))

print("BAGIMSIZ SINYAL OYLAMA SISTEMI")
print("=" * 120)
print(f"Toplam event: {len(valid)} | Base cont: {base:.1f}%")
print()

# ══════════════════════════════════════════════════════════════
# Her sinyalin bagimsiz karar kurali
# ══════════════════════════════════════════════════════════════

# ROC 3h: Q1-Q2 = CONT (ters yonden geliyor), Q4-Q5 = REV (ayni yonde tukenmis), Q3 = BELIRSIZ
# Pre-CVD: Q1-Q2 = CONT (CVD taze), Q4-Q5 = REV (CVD tukenmis), Q3 = BELIRSIZ
# Early price: Q4-Q5 = CONT (fiyat teyit), Q1-Q2 = REV (fiyat ret), Q3 = BELIRSIZ

def roc_vote(q):
    if q <= 2: return "CONT"
    elif q >= 4: return "REV"
    else: return "?"

def cvd_vote(q):
    if q <= 2: return "CONT"
    elif q >= 4: return "REV"
    else: return "?"

def early_vote(q):
    if q >= 4: return "CONT"
    elif q <= 2: return "REV"
    else: return "?"

valid['vote_roc'] = valid['roc_q'].apply(roc_vote)
valid['vote_cvd'] = valid['cvd_q'].apply(cvd_vote)
valid['vote_early'] = valid['early_q'].apply(early_vote)

# Tek sinyal dogruluk
print("TEK SINYAL DOGRULUK:")
print(f"  {'Sinyal':>15s} | {'Karar':>6s} | {'Cont%':>7s} | {'N':>6s} | {'Yorum':>20s}")
print(f"  {'-'*65}")

for name, col in [("ROC 3h", "vote_roc"), ("Pre-CVD 4h", "vote_cvd"), ("Ilk 1h fiyat", "vote_early")]:
    for vote in ["CONT", "REV", "?"]:
        mask = valid[col] == vote
        grp = valid[mask]
        cr = is_cont[grp.index].mean() * 100
        label = vote if vote != "?" else "BELIRSIZ"
        print(f"  {name:>15s} | {label:>6s} | {cr:>6.1f}% | {len(grp):>6d} | {len(grp)/len(valid)*100:.1f}% of events")

print()

# ══════════════════════════════════════════════════════════════
# Oylama
# ══════════════════════════════════════════════════════════════

def count_votes(row):
    votes = [row['vote_roc'], row['vote_cvd'], row['vote_early']]
    cont_votes = votes.count("CONT")
    rev_votes = votes.count("REV")
    unk_votes = votes.count("?")
    return cont_votes, rev_votes, unk_votes

valid['cont_votes'] = valid.apply(lambda r: count_votes(r)[0], axis=1)
valid['rev_votes'] = valid.apply(lambda r: count_votes(r)[1], axis=1)
valid['unk_votes'] = valid.apply(lambda r: count_votes(r)[2], axis=1)

# Oylama sonucu
def final_decision(row):
    cv, rv, uv = row['cont_votes'], row['rev_votes'], row['unk_votes']
    if cv == 3: return "3C"          # 3 CONT
    elif cv == 2 and rv == 0: return "2C"   # 2 CONT, 1 ?
    elif cv == 2 and rv == 1: return "2C1R" # 2 CONT, 1 REV
    elif rv == 3: return "3R"          # 3 REV
    elif rv == 2 and cv == 0: return "2R"   # 2 REV, 1 ?
    elif rv == 2 and cv == 1: return "2R1C" # 2 REV, 1 CONT
    elif cv == 1 and rv == 1: return "1C1R" # 1 CONT, 1 REV, 1 ?
    elif uv == 3: return "3?"          # 3 BELIRSIZ
    elif cv == 1 and uv == 2: return "1C2?" # 1 CONT, 2 ?
    elif rv == 1 and uv == 2: return "1R2?" # 1 REV, 2 ?
    else: return "OTHER"

valid['decision'] = valid.apply(final_decision, axis=1)

print("OYLAMA SONUCLARI:")
print(f"  {'Oylama':>8s} | {'Anlam':>30s} | {'Cont%':>7s} | {'N':>6s} | {'%':>6s}")
print(f"  {'-'*75}")

# Sirala: en yuksek cont'tan en dusuge
decisions_order = ["3C", "2C", "2C1R", "1C2?", "1C1R", "3?", "1R2?", "2R1C", "2R", "3R"]
decision_labels = {
    "3C": "3 sinyal CONT",
    "2C": "2 CONT + 1 belirsiz",
    "2C1R": "2 CONT + 1 REV",
    "1C2?": "1 CONT + 2 belirsiz",
    "1C1R": "1 CONT + 1 REV + 1 belirsiz",
    "3?": "3 belirsiz",
    "1R2?": "1 REV + 2 belirsiz",
    "2R1C": "2 REV + 1 CONT",
    "2R": "2 REV + 1 belirsiz",
    "3R": "3 sinyal REV",
}

for dec in decisions_order:
    mask = valid['decision'] == dec
    grp = valid[mask]
    if len(grp) == 0: continue
    cr = is_cont[grp.index].mean() * 100
    pct = len(grp) / len(valid) * 100
    label = decision_labels.get(dec, dec)
    print(f"  {dec:>8s} | {label:>30s} | {cr:>6.1f}% | {len(grp):>6d} | {pct:>5.1f}%")

# ══════════════════════════════════════════════════════════════
# Bolge siniflandirmasi
# ══════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print("BOLGE SINIFLANDIRMASI")
print("=" * 120)

def classify_zone(dec, cr):
    if dec in ["3C"]: return "GUCLU CONT"
    elif dec in ["2C", "2C1R"] and cr > 70: return "CONT"
    elif dec in ["3R"]: return "GUCLU REV"
    elif dec in ["2R", "2R1C"] and cr < 50: return "REV"
    else: return "BELIRSIZ"

zone_data = []
for dec in decisions_order:
    mask = valid['decision'] == dec
    grp = valid[mask]
    if len(grp) == 0: continue
    cr = is_cont[grp.index].mean() * 100
    zone = classify_zone(dec, cr)
    zone_data.append({'decision': dec, 'cont_rate': cr, 'n': len(grp), 'zone': zone})

zdf = pd.DataFrame(zone_data)

print(f"\n  {'Bolge':>12s} | {'Oylamalar':>30s} | {'Toplam N':>10s} | {'Ort Cont%':>10s} | {'Eventlerin %':>12s}")
print(f"  {'-'*85}")

for zone in ["GUCLU CONT", "CONT", "BELIRSIZ", "REV", "GUCLU REV"]:
    z = zdf[zdf['zone'] == zone]
    if len(z) == 0: continue
    total_n = z['n'].sum()
    avg_cr = (z['cont_rate'] * z['n']).sum() / total_n
    decs = ", ".join(z['decision'].values)
    print(f"  {zone:>12s} | {decs:>30s} | {total_n:>10d} | {avg_cr:>9.1f}% | {total_n/len(valid)*100:>11.1f}%")

# Genel ozet
cont_zone = zdf[zdf['zone'].isin(['GUCLU CONT', 'CONT'])]
rev_zone = zdf[zdf['zone'].isin(['GUCLU REV', 'REV'])]
amb_zone = zdf[zdf['zone'] == 'BELIRSIZ']

cont_n = cont_zone['n'].sum() if len(cont_zone) > 0 else 0
rev_n = rev_zone['n'].sum() if len(rev_zone) > 0 else 0
amb_n = amb_zone['n'].sum() if len(amb_zone) > 0 else 0

print(f"\n  OZET:")
print(f"  Sinyal verilen (CONT + REV): {cont_n + rev_n} ({(cont_n + rev_n)/len(valid)*100:.1f}%)")
print(f"  Belirsiz (pas):              {amb_n} ({amb_n/len(valid)*100:.1f}%)")
print(f"  Yillik sinyal sayisi:        ~{(cont_n + rev_n) / 5:.0f}")
print(f"  Aylik sinyal sayisi:         ~{(cont_n + rev_n) / 5 / 12:.0f}")
print()
