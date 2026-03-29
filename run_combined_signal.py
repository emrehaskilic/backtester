"""Pre-Sweep CVD + Erken Mum Sinyali Kombinasyonu
t-1 bilgisi (CVD tukenmisligi) + t mumunun ilk dakika/saatlerindeki feature'lar
4H ETH Perp, 5 yillik veri
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

BP = 48  # 4H
n_candles = n // BP

# 4H mumlar
candles = []
for i in range(n_candles):
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

    if sig is None:
        continue

    events.append({
        'idx': i, 'bar': curr['start'], 'signal': sig,
    })

print("PRE-SWEEP CVD + ERKEN MUM SINYALI KOMBINASYONU")
print("4H ETH Perp, 5 yillik veri")
print(f"Toplam sweep event: {len(events)}")
print("=" * 120)
print()

# ══════════════════════════════════════════════════════════════
# Her event icin metrikleri hesapla
# ══════════════════════════════════════════════════════════════

for ev in events:
    bar = ev['bar']

    # 1) Pre-sweep CVD degisimi (t mumundan ONCE)
    for lb in [24, 48]:  # 2h ve 4h oncesi
        if bar >= lb:
            cvd_before = cvd[bar-lb:bar]
            ev[f'pre_cvd_{lb}'] = cvd_before[-1] - cvd_before[0]
        else:
            ev[f'pre_cvd_{lb}'] = np.nan

    # 2) Erken mum sinyalleri (t mumunun ICI)
    # CVD degisimi ilk N bar'da
    for early_bars in [1, 3, 6, 12]:  # 5dk, 15dk, 30dk, 1h
        end_bar = bar + early_bars
        if end_bar < n:
            ev[f'early_cvd_{early_bars}'] = cvd[end_bar] - cvd[bar]
            ev[f'early_price_{early_bars}'] = (c[end_bar] - c[bar]) / c[bar] * 100
            # Imbalance (buy_vol / total_vol) ilk N bar'da
            total_v = np.sum(bv[bar:end_bar] + sv[bar:end_bar])
            buy_v = np.sum(bv[bar:end_bar])
            ev[f'early_imb_{early_bars}'] = buy_v / total_v if total_v > 0 else 0.5
        else:
            ev[f'early_cvd_{early_bars}'] = np.nan
            ev[f'early_price_{early_bars}'] = np.nan
            ev[f'early_imb_{early_bars}'] = np.nan

edf = pd.DataFrame(events)

# ══════════════════════════════════════════════════════════════
# Sweep yonune gore aligned metrikleri hesapla
# ══════════════════════════════════════════════════════════════

for col_base in ['pre_cvd_24', 'pre_cvd_48', 'early_cvd_1', 'early_cvd_3', 'early_cvd_6', 'early_cvd_12',
                  'early_price_1', 'early_price_3', 'early_price_6', 'early_price_12',
                  'early_imb_1', 'early_imb_3', 'early_imb_6', 'early_imb_12']:
    # High sweep: pozitif = sweep yonunde, negatif = ters
    # Low sweep: ters cevir
    edf[f'{col_base}_aligned'] = edf[col_base].copy()
    low_mask = edf['signal'].str.startswith('low')
    if 'imb' in col_base:
        edf.loc[low_mask, f'{col_base}_aligned'] = 1.0 - edf.loc[low_mask, col_base]
    else:
        edf.loc[low_mask, f'{col_base}_aligned'] = -edf.loc[low_mask, col_base]

# ══════════════════════════════════════════════════════════════
# 2D Analiz: Pre-CVD × Early Signal → Cont Rate
# ══════════════════════════════════════════════════════════════

print("2D ANALIZ: PRE-SWEEP CVD × ERKEN MUM SINYALI → CONT RATE")
print("=" * 120)

for sweep_type in ["high", "low"]:
    mask_type = edf['signal'].str.startswith(sweep_type) & ~edf['signal'].str.contains('ambig')
    sub = edf[mask_type].copy()
    is_cont = sub['signal'].str.contains('cont')
    base_cont = is_cont.mean() * 100

    print(f"\n{'#'*120}")
    print(f"  {sweep_type.upper()} SWEEP (base cont = {base_cont:.1f}%, N = {len(sub)})")
    print(f"{'#'*120}")

    # Pre-CVD lookback'ler × early signal lookback'ler
    for pre_lb in [24, 48]:
        pre_col = f'pre_cvd_{pre_lb}_aligned'
        pre_time = f"{pre_lb*5//60}h oncesi CVD"

        for early_lb, early_label in [(1, "ilk 5dk"), (3, "ilk 15dk"), (6, "ilk 30dk"), (12, "ilk 1h")]:
            for early_type, early_name in [('cvd', 'CVD'), ('price', 'fiyat'), ('imb', 'imbalance')]:
                early_col = f'early_{early_type}_{early_lb}_aligned'

                valid = sub.dropna(subset=[pre_col, early_col])
                if len(valid) < 100:
                    continue

                # 3x3 grid: pre_cvd (tercile) × early_signal (tercile)
                try:
                    valid['pre_q'] = pd.qcut(valid[pre_col], 3, labels=['Ters','Notr','Ayni'], duplicates='drop')
                    valid['early_q'] = pd.qcut(valid[early_col], 3, labels=['Ters','Notr','Ayni'], duplicates='drop')
                except:
                    continue

                is_cont_v = valid['signal'].str.contains('cont')

                # Tabloyu olustur
                has_signal = False
                rows_data = []
                for pq in ['Ters', 'Notr', 'Ayni']:
                    row = []
                    for eq in ['Ters', 'Notr', 'Ayni']:
                        mask = (valid['pre_q'] == pq) & (valid['early_q'] == eq)
                        group = valid[mask]
                        if len(group) < 15:
                            row.append(('---', 0))
                        else:
                            cr = is_cont_v[mask].mean() * 100
                            row.append((f'{cr:.0f}%', len(group)))
                            if abs(cr - base_cont) > 10:
                                has_signal = True
                    rows_data.append((pq, row))

                if not has_signal:
                    continue

                print(f"\n  {pre_time} × {early_label} {early_name}")
                print(f"  {'':>15s} | {'Erken ' + early_name:^45s}")
                print(f"  {'Pre-CVD':>15s} | {'Ters':>13s} | {'Notr':>13s} | {'Ayni':>13s}")
                print(f"  {'-'*60}")
                for pq, row in rows_data:
                    cells = [f"{r[0]:>6s} N={r[1]:<4d}" if r[1] > 0 else f"{'---':>6s}      " for r in row]
                    print(f"  {pq:>15s} | {cells[0]:>13s} | {cells[1]:>13s} | {cells[2]:>13s}")
                print(f"  {'Base':>15s} | {base_cont:.0f}%")

# ══════════════════════════════════════════════════════════════
# En guclu kombinasyonlar — 5-quantile detay
# ══════════════════════════════════════════════════════════════

print()
print()
print("=" * 120)
print("EN GUCLU KOMBINASYONLAR — 5×5 QUANTILE DETAY")
print("=" * 120)

best_combos = [
    (48, 'cvd', 12, "4h oncesi CVD × ilk 1h CVD"),
    (24, 'cvd', 6, "2h oncesi CVD × ilk 30dk CVD"),
    (48, 'price', 12, "4h oncesi CVD × ilk 1h fiyat"),
    (24, 'imb', 6, "2h oncesi CVD × ilk 30dk imbalance"),
    (48, 'cvd', 3, "4h oncesi CVD × ilk 15dk CVD"),
    (24, 'cvd', 1, "2h oncesi CVD × ilk 5dk CVD"),
]

for pre_lb, early_type, early_lb, combo_label in best_combos:
    pre_col = f'pre_cvd_{pre_lb}_aligned'
    early_col = f'early_{early_type}_{early_lb}_aligned'

    for sweep_type in ["high", "low"]:
        mask_type = edf['signal'].str.startswith(sweep_type) & ~edf['signal'].str.contains('ambig')
        sub = edf[mask_type].copy()
        valid = sub.dropna(subset=[pre_col, early_col])
        if len(valid) < 100:
            continue

        is_cont = valid['signal'].str.contains('cont')
        base_cont = is_cont.mean() * 100

        try:
            valid['pre_q'] = pd.qcut(valid[pre_col], 5, labels=[1,2,3,4,5], duplicates='drop')
            valid['early_q'] = pd.qcut(valid[early_col], 5, labels=[1,2,3,4,5], duplicates='drop')
        except:
            continue

        print(f"\n  {combo_label} — {sweep_type.upper()} SWEEP (base={base_cont:.1f}%)")
        print(f"  {'':>10s} |  {'Erken sinyal Q1 (ters)':>15s} | {'Q2':>10s} | {'Q3':>10s} | {'Q4':>10s} | {'Q5 (ayni)':>15s}")
        print(f"  {'-'*85}")

        for pq in range(1, 6):
            cells = []
            for eq in range(1, 6):
                mask = (valid['pre_q'] == pq) & (valid['early_q'] == eq)
                group = valid[mask]
                if len(group) < 10:
                    cells.append(f"{'':>10s}")
                else:
                    cr = is_cont[mask].mean() * 100
                    marker = "***" if cr > base_cont + 15 else ("!!!" if cr < base_cont - 15 else "   ")
                    cells.append(f"{cr:>5.0f}% {len(group):>3d}{marker}")
            pq_label = "Pre Q1 ters" if pq == 1 else (f"Pre Q{pq}" if pq < 5 else "Pre Q5 ayni")
            print(f"  {pq_label:>10s} | {cells[0]:>15s} | {cells[1]:>10s} | {cells[2]:>10s} | {cells[3]:>10s} | {cells[4]:>15s}")

print()
print("=" * 120)
print("OKUMA REHBERI:")
print("  Pre Q1 (ters) = sweep oncesi CVD ters yone gitmis (TAZE momentum)")
print("  Pre Q5 (ayni) = sweep oncesi CVD ayni yone cok gitmis (TUKENMIS momentum)")
print("  Early Q1 (ters) = mumun ilk dakikalarinda CVD/fiyat ters yone gidiyor")
print("  Early Q5 (ayni) = mumun ilk dakikalarinda CVD/fiyat sweep yonunde gidiyor")
print("  *** = base rate'den 15pp+ yukarida (GUCLU cont sinyali)")
print("  !!! = base rate'den 15pp+ asagida (GUCLU rev sinyali)")
print("=" * 120)
