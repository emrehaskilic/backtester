"""Sabit Kasa Backtest
Kasa: $1000
Margin per trade: $100
Kaldıraç: 25x → Pozisyon: $2500
Her hafta başı kasa $1000'a sıfırlanır (kar çekilir)
Komisyon: %0.08 round trip
PMAX 3m çatışma filtresi ile sadece CONT sinyalleri
"""
import numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import rust_engine

df5 = pd.read_parquet('data/ETHUSDT_5m_5y.parquet')
df5['dt'] = pd.to_datetime(df5['open_time'], unit='ms')
c = df5['close'].values.astype(np.float64)
h = df5['high'].values.astype(np.float64)
l = df5['low'].values.astype(np.float64)
bv = df5['buy_vol'].values.astype(np.float64)
sv = df5['sell_vol'].values.astype(np.float64)
n = len(df5)
delta = bv - sv
cvd = np.cumsum(delta)

# PMAX 3m
df3 = pd.read_parquet('data/ETHUSDT_3m_5y_perp.parquet')
df3['dt'] = pd.to_datetime(df3['open_time'], unit='ms')
c3 = np.ascontiguousarray(df3['close'].values, dtype=np.float64)
h3 = np.ascontiguousarray(df3['high'].values, dtype=np.float64)
l3 = np.ascontiguousarray(df3['low'].values, dtype=np.float64)
src3 = np.ascontiguousarray((h3 + l3) / 2, dtype=np.float64)

pmax_result = rust_engine.compute_adaptive_pmax(
    src3, h3, l3, c3,
    10, 3.0, 10, 580, 440, 4.0, 1.25, 3, 5.5, 19, 1.5, 29,
)
pmax_dir_3m = np.array(pmax_result['direction'])
pmax_ts_3m = df3['open_time'].values
ts5 = df5['open_time'].values

pmax_at_5m = np.zeros(n)
j = 0
for i in range(n):
    while j < len(pmax_ts_3m) - 1 and pmax_ts_3m[j + 1] <= ts5[i]:
        j += 1
    pmax_at_5m[i] = pmax_dir_3m[j]

# Sweep sinyalleri
BP = 48; EARLY_BARS = 12; COMMISSION_PCT = 0.08
KASA = 1000.0
MARGIN = 100.0
LEVERAGE = 25
POZISYON = MARGIN * LEVERAGE  # $2500

candles = []
for i in range(n // BP):
    s = i * BP; e = s + BP
    candles.append({'start': s, 'open': c[s], 'high': np.max(h[s:e]), 'low': np.min(l[s:e]), 'close': c[e-1], 'dt': df5['dt'].iloc[s]})

CONT_CELLS = {
    ('Ters','Ters','Ayni'), ('Ters','Notr','Ayni'), ('Ters','Ayni','Ayni'),
    ('Ters','Ters','Notr'), ('Ters','Ters','Ters'), ('Ters','Ayni','Notr'),
    ('Notr','Ters','Ayni'), ('Notr','Notr','Ayni'),
    ('Ayni','Ayni','Ayni'), ('Ayni','Notr','Ayni'),
}

def tercile_from_th(val, th_33, th_67):
    if val <= th_33: return 'Ters'
    elif val <= th_67: return 'Notr'
    else: return 'Ayni'

# Walk-forward grid sinyalleri + PMAX filtre
TRAIN = 540; UPDATE = 180
trades = []
current_th = None; last_train_end = 0
edf_list = []

for i in range(1, len(candles)):
    prev = candles[i-1]; curr = candles[i]; bar = curr['start']
    swept_high = curr['high'] > prev['high']; swept_low = curr['low'] < prev['low']
    is_green = curr['close'] > curr['open']; is_red = curr['close'] < curr['open']
    sig, sweep_dir = None, None
    if swept_high and not swept_low:
        sig = 'high_cont' if curr['close'] > prev['high'] else ('high_rev' if is_red else None); sweep_dir = 'high'
    elif swept_low and not swept_high:
        sig = 'low_cont' if curr['close'] < prev['low'] else ('low_rev' if is_green else None); sweep_dir = 'low'
    elif swept_high and swept_low:
        if curr['close'] > prev['high']: sig, sweep_dir = 'high_cont', 'high'
        elif curr['close'] < prev['low']: sig, sweep_dir = 'low_cont', 'low'
        elif is_red: sig, sweep_dir = 'high_rev', 'high'
        elif is_green: sig, sweep_dir = 'low_rev', 'low'

    roc = np.nan
    if bar >= 36 and c[bar-36] > 0:
        roc = (c[bar]-c[bar-36])/c[bar-36]*100
        if sweep_dir == 'low': roc = -roc
    pcv = np.nan
    if bar >= 48:
        pcv = cvd[bar]-cvd[bar-48]
        if sweep_dir == 'low': pcv = -pcv
    early_p = np.nan
    entry_bar = bar + EARLY_BARS
    if entry_bar < n:
        early_p = (c[entry_bar]-c[bar])/c[bar]*100
        if sweep_dir == 'low': early_p = -early_p

    entry_price = c[entry_bar] if entry_bar < n else np.nan
    exit_price = c[bar + BP - 1] if bar + BP - 1 < n else np.nan
    pmax_entry = pmax_at_5m[entry_bar] if entry_bar < n else 0

    edf_list.append({
        'candle_idx': i, 'bar': bar, 'dt': curr['dt'],
        'signal': sig, 'sweep_dir': sweep_dir,
        'roc': roc, 'pre_cvd': pcv, 'early_p': early_p,
        'entry_price': entry_price, 'exit_price': exit_price,
        'pmax_dir': pmax_entry,
    })

    # Threshold guncelle
    if i >= TRAIN and (current_th is None or i - last_train_end >= UPDATE):
        td = [e for e in edf_list if e['candle_idx'] >= i-TRAIN and e['candle_idx'] < i and e['signal'] is not None]
        if len(td) >= 50:
            rocs = [e['roc'] for e in td if not np.isnan(e['roc'])]
            cvds = [e['pre_cvd'] for e in td if not np.isnan(e['pre_cvd'])]
            earlies = [e['early_p'] for e in td if not np.isnan(e['early_p'])]
            if len(rocs)>=30 and len(cvds)>=30 and len(earlies)>=30:
                current_th = {'roc_33':np.percentile(rocs,33.3),'roc_67':np.percentile(rocs,66.7),
                              'cvd_33':np.percentile(cvds,33.3),'cvd_67':np.percentile(cvds,66.7),
                              'early_33':np.percentile(earlies,33.3),'early_67':np.percentile(earlies,66.7)}
                last_train_end = i

    if current_th is None or sig is None or sweep_dir is None: continue
    if pd.isna(roc) or pd.isna(pcv) or pd.isna(early_p): continue
    if pd.isna(entry_price) or pd.isna(exit_price): continue

    rt = tercile_from_th(roc, current_th['roc_33'], current_th['roc_67'])
    ct = tercile_from_th(pcv, current_th['cvd_33'], current_th['cvd_67'])
    et = tercile_from_th(early_p, current_th['early_33'], current_th['early_67'])

    if (rt,ct,et) not in CONT_CELLS: continue

    # PMAX catisma filtresi
    pmax_agrees = (pmax_entry > 0 and sweep_dir == 'high') or (pmax_entry < 0 and sweep_dir == 'low')
    if pmax_agrees: continue  # sadece catisma

    direction = "LONG" if sweep_dir == "high" else "SHORT"

    if direction == "LONG":
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100

    # $ cinsinden
    pnl_leveraged_pct = pnl_pct * LEVERAGE  # 25x
    commission_dollar = POZISYON * COMMISSION_PCT / 100  # $2500 * 0.08% = $2
    pnl_dollar = POZISYON * pnl_pct / 100 - commission_dollar

    # MFE/MAE $ cinsinden
    mfe_pct = 0; mae_pct = 0
    for b in range(entry_bar, min(bar + BP, n)):
        if direction == "LONG":
            fav = (h[b] - entry_price) / entry_price * 100
            adv = (entry_price - l[b]) / entry_price * 100
        else:
            fav = (entry_price - l[b]) / entry_price * 100
            adv = (h[b] - entry_price) / entry_price * 100
        if fav > mfe_pct: mfe_pct = fav
        if adv > mae_pct: mae_pct = adv

    # Liq check: margin $100, 25x → liq at ~4% adverse move (margin / pozisyon)
    liq_pct = MARGIN / POZISYON * 100  # 4%
    liquidated = mae_pct >= liq_pct

    trades.append({
        'dt': curr['dt'], 'direction': direction,
        'entry': entry_price, 'exit': exit_price,
        'pnl_pct': pnl_pct, 'pnl_dollar': pnl_dollar,
        'pnl_leveraged_pct': pnl_leveraged_pct,
        'mfe_pct': mfe_pct, 'mae_pct': mae_pct,
        'liquidated': liquidated,
        'commission': commission_dollar,
        'year': pd.Timestamp(curr['dt']).year,
        'week': pd.Timestamp(curr['dt']).isocalendar()[1],
        'year_week': pd.Timestamp(curr['dt']).strftime('%Y-W%W'),
    })

tdf = pd.DataFrame(trades)

print("SABIT KASA BACKTEST — PMAX 3m CATISMA")
print("=" * 120)
print(f"Kasa: ${KASA:.0f} | Margin/trade: ${MARGIN:.0f} | Kaldıraç: {LEVERAGE}x | Pozisyon: ${POZISYON:.0f}")
print(f"Komisyon: %{COMMISSION_PCT} → ${POZISYON * COMMISSION_PCT / 100:.2f}/trade")
print(f"Liq seviyesi: ~%{MARGIN/POZISYON*100:.1f} adverse move")
print(f"Her hafta kasa ${KASA:.0f}'a sifirlanir")
print()

# Liq kontrolu
liq_count = tdf['liquidated'].sum()
print(f"Toplam trade: {len(tdf)}")
print(f"Likide olan: {liq_count} ({liq_count/len(tdf)*100:.1f}%)")
print()

# Liq olanlari -$100 (margin kaybi) olarak isle
tdf['pnl_final'] = tdf.apply(lambda r: -MARGIN if r['liquidated'] else r['pnl_dollar'], axis=1)

print(f"GENEL SONUCLAR:")
print(f"  Toplam trade: {len(tdf)}")
print(f"  Win rate: {(tdf['pnl_final'] > 0).mean() * 100:.1f}%")
print(f"  Avg PnL/trade: ${tdf['pnl_final'].mean():.2f}")
print(f"  Median PnL/trade: ${tdf['pnl_final'].median():.2f}")
print(f"  Toplam PnL: ${tdf['pnl_final'].sum():.2f}")
print(f"  Max single loss: ${tdf['pnl_final'].min():.2f}")
print(f"  Max single win: ${tdf['pnl_final'].max():.2f}")
print()

# Haftalik (her hafta $1000 ile basla)
tdf['week_key'] = pd.to_datetime(tdf['dt']).dt.to_period('W')
weekly_results = []

for week, grp in tdf.groupby('week_key'):
    kasa = KASA
    week_pnl = 0
    week_trades = 0
    week_wins = 0
    week_liq = 0

    for _, trade in grp.iterrows():
        if kasa < MARGIN:
            break  # kasa margin'i karsilamiyor, hafta bitti

        pnl = trade['pnl_final']
        kasa += pnl
        week_pnl += pnl
        week_trades += 1
        if pnl > 0: week_wins += 1
        if trade['liquidated']: week_liq += 1

    weekly_results.append({
        'week': str(week),
        'trades': week_trades,
        'wins': week_wins,
        'wr': week_wins / week_trades * 100 if week_trades > 0 else 0,
        'pnl': week_pnl,
        'end_kasa': KASA + week_pnl,
        'kar_cekilir': max(0, week_pnl),  # pozitifse cekilir
        'liq': week_liq,
    })

wdf = pd.DataFrame(weekly_results)

print(f"HAFTALIK SONUCLAR:")
print(f"  Toplam hafta: {len(wdf)}")
print(f"  Pozitif hafta: {(wdf['pnl'] > 0).sum()} ({(wdf['pnl'] > 0).mean()*100:.0f}%)")
print(f"  Negatif hafta: {(wdf['pnl'] < 0).sum()} ({(wdf['pnl'] < 0).mean()*100:.0f}%)")
print(f"  Sifir hafta: {(wdf['pnl'] == 0).sum()}")
print(f"  Ort haftalik PnL: ${wdf['pnl'].mean():.2f}")
print(f"  Median haftalik PnL: ${wdf['pnl'].median():.2f}")
print(f"  Toplam cekilen kar: ${wdf['kar_cekilir'].sum():.2f}")
print(f"  En iyi hafta: ${wdf['pnl'].max():.2f}")
print(f"  En kotu hafta: ${wdf['pnl'].min():.2f}")
print(f"  Liq olan hafta: {(wdf['liq'] > 0).sum()}")
print()

# Yillik
print(f"YILLIK SONUCLAR:")
print(f"  {'Yil':>6s} | {'Hafta':>6s} | {'Trade':>6s} | {'WR':>6s} | {'Toplam $':>10s} | {'Ort hafta':>10s} | {'Liq':>4s}")
print(f"  {'-'*60}")

wdf['year'] = wdf['week'].str[:4].astype(int)
for year, grp in wdf.groupby('year'):
    total_pnl = grp['pnl'].sum()
    total_trades = grp['trades'].sum()
    total_wins = grp['wins'].sum()
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    total_liq = grp['liq'].sum()
    print(f"  {year:>6d} | {len(grp):>6d} | {total_trades:>6d} | {wr:>5.1f}% | ${total_pnl:>9.2f} | ${grp['pnl'].mean():>9.2f} | {total_liq:>4d}")

print(f"\n  5 yil toplam cekilen kar: ${wdf['kar_cekilir'].sum():,.2f}")
print(f"  Aylik ortalama: ${wdf['kar_cekilir'].sum() / 60:,.2f}")

# Haftalik detay (ilk ve son 10 hafta)
print(f"\nILK 10 HAFTA:")
print(f"  {'Hafta':>15s} | {'Trd':>4s} | {'WR':>5s} | {'PnL $':>9s} | {'Kasa sonu':>10s} | {'Liq':>3s}")
print(f"  {'-'*55}")
for _, w in wdf.head(10).iterrows():
    print(f"  {w['week']:>15s} | {w['trades']:>4.0f} | {w['wr']:>4.0f}% | ${w['pnl']:>8.2f} | ${w['end_kasa']:>9.2f} | {w['liq']:>3.0f}")

print(f"\nSON 10 HAFTA:")
print(f"  {'Hafta':>15s} | {'Trd':>4s} | {'WR':>5s} | {'PnL $':>9s} | {'Kasa sonu':>10s} | {'Liq':>3s}")
print(f"  {'-'*55}")
for _, w in wdf.tail(10).iterrows():
    print(f"  {w['week']:>15s} | {w['trades']:>4.0f} | {w['wr']:>4.0f}% | ${w['pnl']:>8.2f} | ${w['end_kasa']:>9.2f} | {w['liq']:>3.0f}")

# Mae dagilimi
print(f"\nMAE (Max Adverse Excursion) DAGILIMI:")
print(f"  MAE < %1: {(tdf['mae_pct'] < 1).sum()} ({(tdf['mae_pct'] < 1).mean()*100:.0f}%)")
print(f"  MAE < %2: {(tdf['mae_pct'] < 2).sum()} ({(tdf['mae_pct'] < 2).mean()*100:.0f}%)")
print(f"  MAE < %3: {(tdf['mae_pct'] < 3).sum()} ({(tdf['mae_pct'] < 3).mean()*100:.0f}%)")
print(f"  MAE < %4 (liq): {(tdf['mae_pct'] < 4).sum()} ({(tdf['mae_pct'] < 4).mean()*100:.0f}%)")
print(f"  MAE >= %4 (LIQ): {(tdf['mae_pct'] >= 4).sum()} ({(tdf['mae_pct'] >= 4).mean()*100:.0f}%)")
print(f"  Ort MAE: {tdf['mae_pct'].mean():.2f}%")
print(f"  Max MAE: {tdf['mae_pct'].max():.2f}%")

print()
