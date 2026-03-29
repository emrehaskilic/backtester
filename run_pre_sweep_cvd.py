"""Pre-Sweep CVD Analizi
Sweep oncesindeki CVD davranisi, sweep sonucunu (cont/rev) tahmin eder mi?

Senaryolar:
1. Fiyat yukseliyor + CVD de yukseliyor → sweep basarili mi? (confirmation)
2. Fiyat yukseliyor + CVD dusuyor → sweep basarisiz mi? (divergence)
3. CVD trendi sweep oncesi ne kadar suredir devam ediyor?
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

print("PRE-SWEEP CVD ANALIZI — ETH Perp 5 Yil")
print("=" * 120)
print()

# CVD hesapla (kumulatif)
delta = bv - sv  # her 5m bar'daki net taker delta
cvd = np.cumsum(delta)

# 4H mumlari olustur
BP = 48
n_candles = n // BP

candles = []
for i in range(n_candles):
    s = i * BP
    e = s + BP
    candles.append({
        'start': s,
        'open': c[s],
        'high': np.max(h[s:e]),
        'low': np.min(l[s:e]),
        'close': c[e-1],
    })

# Sweep event'leri siniflandir
events = []
for i in range(1, len(candles)):
    prev = candles[i-1]
    curr = candles[i]

    swept_high = curr['high'] > prev['high']
    swept_low = curr['low'] < prev['low']
    is_green = curr['close'] > curr['open']
    is_red = curr['close'] < curr['open']

    if swept_high and not swept_low:
        if curr['close'] > prev['high']:
            sig = "high_cont"
        elif is_red:
            sig = "high_rev"
        else:
            sig = "high_ambig"
    elif swept_low and not swept_high:
        if curr['close'] < prev['low']:
            sig = "low_cont"
        elif is_green:
            sig = "low_rev"
        else:
            sig = "low_ambig"
    elif swept_high and swept_low:
        if curr['close'] > prev['high']:
            sig = "high_cont"
        elif curr['close'] < prev['low']:
            sig = "low_cont"
        elif is_red:
            sig = "high_rev"
        elif is_green:
            sig = "low_rev"
        else:
            continue
    else:
        continue

    events.append({
        'candle_idx': i,
        'bar_start': curr['start'],
        'signal': sig,
        'prev_high': prev['high'],
        'prev_low': prev['low'],
        'curr_close': curr['close'],
    })

print(f"Toplam 4H mum: {len(candles)}")
print(f"Toplam sweep event: {len(events)}")
print()

# ══════════════════════════════════════════════════════════════
# Pre-sweep metrikleri hesapla
# ══════════════════════════════════════════════════════════════

lookbacks = [12, 24, 48, 96]  # 1h, 2h, 4h, 8h (5m bar sayisi)
# Not: 48 bar = 1 adet 4H mum oncesi, 96 = 2 adet 4H mum oncesi

for ev in events:
    bar = ev['bar_start']

    for lb in lookbacks:
        if bar < lb:
            ev[f'cvd_change_{lb}'] = np.nan
            ev[f'price_change_{lb}'] = np.nan
            ev[f'cvd_price_corr_{lb}'] = np.nan
            ev[f'divergence_{lb}'] = np.nan
            continue

        # CVD degisimi (sweep oncesi lb bar)
        cvd_before = cvd[bar - lb:bar]
        cvd_change = cvd_before[-1] - cvd_before[0]
        ev[f'cvd_change_{lb}'] = cvd_change

        # Fiyat degisimi
        price_before = c[bar - lb:bar]
        price_change = (price_before[-1] - price_before[0]) / price_before[0] * 100
        ev[f'price_change_{lb}'] = price_change

        # CVD-Price korelasyonu
        if len(cvd_before) > 2:
            cvd_norm = (cvd_before - cvd_before[0]) / (np.std(cvd_before) + 1e-10)
            price_norm = (price_before - price_before[0]) / (np.std(price_before) + 1e-10)
            corr = np.corrcoef(cvd_norm, price_norm)[0, 1]
            ev[f'cvd_price_corr_{lb}'] = corr
        else:
            ev[f'cvd_price_corr_{lb}'] = np.nan

        # Divergence skoru: fiyat yukari + CVD asagi = negatif divergence
        # Pozitif = confirmation, negatif = divergence
        if price_change > 0:
            # Fiyat yukseliyor
            div = 1 if cvd_change > 0 else -1  # CVD de yukseliyorsa confirm, degilse div
        elif price_change < 0:
            # Fiyat dusuyor
            div = 1 if cvd_change < 0 else -1  # CVD de dusuyorsa confirm, degilse div
        else:
            div = 0
        ev[f'divergence_{lb}'] = div

edf = pd.DataFrame(events)

# ══════════════════════════════════════════════════════════════
# Analiz: Divergence vs Confirmation → Sweep sonucu
# ══════════════════════════════════════════════════════════════

print("=" * 120)
print("ANALIZ 1: CVD-FIYAT DIVERGENCE/CONFIRMATION → SWEEP SONUCU")
print("Divergence = fiyat bir yone gidiyor ama CVD ters yone (zayiflama sinyali)")
print("Confirmation = fiyat ve CVD ayni yone gidiyor (guc teyidi)")
print("=" * 120)

for lb in lookbacks:
    lb_time = f"{lb*5/60:.0f}h" if lb*5 >= 60 else f"{lb*5}dk"
    print(f"\n  Lookback: {lb} bar ({lb_time} oncesi)")
    print(f"  {'Sweep tipi':>15s} | {'Durum':>15s} | {'Cont%':>7s} | {'Rev%':>7s} | {'N':>5s} | {'Yorum':>20s}")
    print(f"  {'-'*80}")

    for sweep_type in ["high", "low"]:
        # Cont + Rev olanlari filtrele
        mask_type = edf['signal'].str.startswith(sweep_type) & ~edf['signal'].str.contains('ambig')
        sub = edf[mask_type].copy()

        is_cont = sub['signal'].str.contains('cont')
        div_col = f'divergence_{lb}'

        for div_val, div_label in [(1, "Confirmation"), (-1, "Divergence"), (0, "Notr")]:
            mask_div = sub[div_col] == div_val
            group = sub[mask_div]
            if len(group) == 0:
                continue
            cont_count = is_cont[mask_div].sum()
            rev_count = len(group) - cont_count
            cont_rate = cont_count / len(group) * 100
            rev_rate = rev_count / len(group) * 100
            yorum = ""
            base = is_cont.sum() / len(sub) * 100 if len(sub) > 0 else 0
            diff = cont_rate - base
            if abs(diff) > 5:
                yorum = f"base'den {diff:+.1f}pp"
            else:
                yorum = "~base rate"

            print(f"  {sweep_type:>15s} | {div_label:>15s} | {cont_rate:>6.1f}% | {rev_rate:>6.1f}% | {len(group):>5d} | {yorum:>20s}")

# ══════════════════════════════════════════════════════════════
# Analiz 2: CVD-Price korelasyon quantile'lari
# ══════════════════════════════════════════════════════════════

print()
print("=" * 120)
print("ANALIZ 2: CVD-FIYAT KORELASYON → SWEEP SONUCU")
print("Korelasyon: +1 = tam ayni yon, -1 = tam ters yon, 0 = bagimsiz")
print("=" * 120)

for lb in lookbacks:
    lb_time = f"{lb*5/60:.0f}h" if lb*5 >= 60 else f"{lb*5}dk"
    corr_col = f'cvd_price_corr_{lb}'

    print(f"\n  Lookback: {lb} bar ({lb_time})")

    for sweep_type in ["high", "low"]:
        mask_type = edf['signal'].str.startswith(sweep_type) & ~edf['signal'].str.contains('ambig')
        sub = edf[mask_type].dropna(subset=[corr_col]).copy()
        if len(sub) < 50:
            continue

        is_cont = sub['signal'].str.contains('cont')

        # Quantile'lara bol
        sub['corr_q'] = pd.qcut(sub[corr_col], 5, labels=[1,2,3,4,5], duplicates='drop')

        print(f"\n    {sweep_type.upper()} SWEEP:")
        print(f"    {'Quantile':>8s} | {'Corr range':>20s} | {'Cont%':>7s} | {'Rev%':>7s} | {'N':>5s}")
        print(f"    {'-'*65}")

        for q in range(1, 6):
            mask_q = sub['corr_q'] == q
            group = sub[mask_q]
            if len(group) == 0:
                continue
            cont = is_cont[mask_q].sum()
            cont_rate = cont / len(group) * 100
            rev_rate = 100 - cont_rate
            corr_min = group[corr_col].min()
            corr_max = group[corr_col].max()
            print(f"    Q{q:>7d} | [{corr_min:>+.3f}, {corr_max:>+.3f}] | {cont_rate:>6.1f}% | {rev_rate:>6.1f}% | {len(group):>5d}")

# ══════════════════════════════════════════════════════════════
# Analiz 3: CVD degisim buyuklugu → Sweep sonucu
# ══════════════════════════════════════════════════════════════

print()
print("=" * 120)
print("ANALIZ 3: PRE-SWEEP CVD DEGISIM BUYUKLUGU → SWEEP SONUCU")
print("CVD cok yukseldiyse sweep oncesi, sweep basarili mi basarisiz mi?")
print("=" * 120)

for lb in lookbacks:
    lb_time = f"{lb*5/60:.0f}h" if lb*5 >= 60 else f"{lb*5}dk"
    cvd_col = f'cvd_change_{lb}'

    print(f"\n  Lookback: {lb} bar ({lb_time})")

    for sweep_type in ["high", "low"]:
        mask_type = edf['signal'].str.startswith(sweep_type) & ~edf['signal'].str.contains('ambig')
        sub = edf[mask_type].dropna(subset=[cvd_col]).copy()
        if len(sub) < 50:
            continue

        is_cont = sub['signal'].str.contains('cont')

        # High sweep icin CVD pozitif = ayni yon, negatif = divergence
        # Low sweep icin CVD negatif = ayni yon, pozitif = divergence
        if sweep_type == "high":
            sub['cvd_aligned'] = sub[cvd_col]  # pozitif = confirmation
        else:
            sub['cvd_aligned'] = -sub[cvd_col]  # negatif CVD = low yonunde confirmation

        sub['cvd_q'] = pd.qcut(sub['cvd_aligned'], 5, labels=[1,2,3,4,5], duplicates='drop')

        base_cont = is_cont.sum() / len(sub) * 100

        print(f"\n    {sweep_type.upper()} SWEEP (base cont={base_cont:.1f}%):")
        print(f"    {'Quantile':>8s} | {'CVD aligned':>20s} | {'Cont%':>7s} | {'Rev%':>7s} | {'N':>5s} | {'Fark':>8s}")
        print(f"    {'-'*75}")

        for q in range(1, 6):
            mask_q = sub['cvd_q'] == q
            group = sub[mask_q]
            if len(group) == 0:
                continue
            cont = is_cont[mask_q].sum()
            cont_rate = cont / len(group) * 100
            rev_rate = 100 - cont_rate
            diff = cont_rate - base_cont
            cvd_min = group['cvd_aligned'].min()
            cvd_max = group['cvd_aligned'].max()
            print(f"    Q{q:>7d} | [{cvd_min:>+10.0f}, {cvd_max:>+10.0f}] | {cont_rate:>6.1f}% | {rev_rate:>6.1f}% | {len(group):>5d} | {diff:>+7.1f}pp")

print()
print("=" * 120)
print("YORUM:")
print("  Q1 = CVD sweep yonune karsi (divergence)")
print("  Q5 = CVD sweep yonunde guclu (confirmation)")
print("  Eger Q1'de cont dusuk, Q5'te cont yuksekse → pre-sweep CVD trendi sinyal veriyor")
print("  Eger Q1-Q5 arasinda fark yoksa → pre-sweep CVD bilgi vermiyor")
print("=" * 120)
