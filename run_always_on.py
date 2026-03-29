"""Her zaman pozisyonda — Sweep + PMAX 3m
Sweep aktif → sweep yonu
Sweep bitti → PMAX 3m yonu
Yon degistiginde pozisyon cevirilir.
$1000 kasa, $100 margin, 25x kaldirac.
Haftalik kar cekilir.
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

# Sweep sinyalleri (4H grid CONT + PMAX catisma)
BP = 48; EARLY_BARS = 12
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

candles = []
for i in range(n // BP):
    s = i * BP; e = s + BP
    candles.append({'start': s, 'open': c[s], 'high': np.max(h[s:e]), 'low': np.min(l[s:e]), 'close': c[e-1]})

# Sweep sinyal dizisi: her 5m bar icin +1 (LONG), -1 (SHORT), 0 (yok)
sweep_signal = np.zeros(n)

TRAIN = 540; UPDATE = 180
edf_list = []
current_th = None; last_train_end = 0

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

    edf_list.append({'candle_idx': i, 'bar': bar, 'signal': sig, 'sweep_dir': sweep_dir,
                     'roc': roc, 'pre_cvd': pcv, 'early_p': early_p, 'entry_bar': entry_bar})

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

    rt = tercile_from_th(roc, current_th['roc_33'], current_th['roc_67'])
    ct = tercile_from_th(pcv, current_th['cvd_33'], current_th['cvd_67'])
    et = tercile_from_th(early_p, current_th['early_33'], current_th['early_67'])

    if (rt,ct,et) not in CONT_CELLS: continue

    # PMAX catisma filtresi
    pmax_entry = pmax_at_5m[entry_bar] if entry_bar < n else 0
    pmax_agrees = (pmax_entry > 0 and sweep_dir == 'high') or (pmax_entry < 0 and sweep_dir == 'low')
    if pmax_agrees: continue

    # Sinyal aktif: entry_bar'dan mum kapanisina kadar
    direction = 1 if sweep_dir == 'high' else -1
    signal_end = bar + BP - 1
    for b in range(entry_bar, min(signal_end + 1, n)):
        sweep_signal[b] = direction

# Bias dizisi: sweep varsa sweep, yoksa PMAX
bias = np.zeros(n)
for i in range(n):
    if sweep_signal[i] != 0:
        bias[i] = sweep_signal[i]
    else:
        bias[i] = pmax_at_5m[i]

print("HER ZAMAN POZISYONDA — SWEEP + PMAX 3m")
print("=" * 120)

# Kapsam
sweep_bars = (sweep_signal != 0).sum()
pmax_bars = n - sweep_bars
print(f"Sweep aktif: {sweep_bars/n*100:.1f}% | PMAX dolduruyor: {pmax_bars/n*100:.1f}%")
print(f"Bias LONG: {(bias > 0).sum()/n*100:.1f}% | SHORT: {(bias < 0).sum()/n*100:.1f}% | Notr: {(bias == 0).sum()/n*100:.1f}%")
print()

# Her 5m bar'da yon degisimi = trade
COMMISSION_PCT = 0.08
KASA = 1000.0; MARGIN = 100.0; LEVERAGE = 25
POZISYON = MARGIN * LEVERAGE

trades = []
current_dir = 0
entry_price = 0
entry_bar_idx = 0
entry_source = ""

for i in range(1, n):
    new_dir = 1 if bias[i] > 0 else (-1 if bias[i] < 0 else 0)

    if new_dir != current_dir and new_dir != 0:
        # Onceki pozisyonu kapat
        if current_dir != 0 and entry_price > 0:
            exit_price = c[i]
            if current_dir == 1:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

            commission = POZISYON * COMMISSION_PCT / 100
            pnl_dollar = POZISYON * pnl_pct / 100 - commission

            duration_bars = i - entry_bar_idx
            trades.append({
                'entry_bar': entry_bar_idx, 'exit_bar': i,
                'dt': df5['dt'].iloc[entry_bar_idx],
                'exit_dt': df5['dt'].iloc[i],
                'direction': 'LONG' if current_dir == 1 else 'SHORT',
                'source': entry_source,
                'entry': entry_price, 'exit': exit_price,
                'pnl_pct': pnl_pct, 'pnl_dollar': pnl_dollar,
                'duration_min': duration_bars * 5,
                'year': df5['dt'].iloc[entry_bar_idx].year,
            })

        # Yeni pozisyon ac
        current_dir = new_dir
        entry_price = c[i]
        entry_bar_idx = i
        entry_source = "SWEEP" if sweep_signal[i] != 0 else "PMAX"

tdf = pd.DataFrame(trades)

print(f"GENEL SONUCLAR:")
print(f"  Toplam trade: {len(tdf)}")
print(f"  Win rate: {(tdf['pnl_dollar'] > 0).mean()*100:.1f}%")
print(f"  Avg PnL/trade: ${tdf['pnl_dollar'].mean():.2f}")
print(f"  Toplam PnL: ${tdf['pnl_dollar'].sum():,.2f}")
print(f"  Ort trade suresi: {tdf['duration_min'].mean():.0f}dk ({tdf['duration_min'].mean()/60:.1f}h)")
print()

# Kaynak bazinda
for src in ["SWEEP", "PMAX"]:
    sub = tdf[tdf['source'] == src]
    if len(sub) == 0: continue
    print(f"  {src}: {len(sub)} trade | WR: {(sub['pnl_dollar']>0).mean()*100:.1f}% | Avg: ${sub['pnl_dollar'].mean():.2f} | Sum: ${sub['pnl_dollar'].sum():,.2f} | Ort sure: {sub['duration_min'].mean():.0f}dk")

# LONG vs SHORT
print()
for d in ["LONG", "SHORT"]:
    sub = tdf[tdf['direction'] == d]
    print(f"  {d}: {len(sub)} | WR: {(sub['pnl_dollar']>0).mean()*100:.1f}% | Sum: ${sub['pnl_dollar'].sum():,.2f}")

# Haftalik (kasa $1000'a sifirlanir)
tdf['week_key'] = pd.to_datetime(tdf['dt']).dt.to_period('W')
weekly_results = []

for week, grp in tdf.groupby('week_key'):
    kasa = KASA
    week_pnl = 0
    for _, trade in grp.iterrows():
        pnl = trade['pnl_dollar']
        # Liq check: margin $100, pozisyon $2500
        if trade['pnl_pct'] * LEVERAGE <= -(MARGIN / KASA * 100):
            pnl = -MARGIN  # liq
        kasa += pnl
        week_pnl += pnl
        if kasa < MARGIN:
            break

    weekly_results.append({
        'week': str(week), 'trades': len(grp),
        'pnl': week_pnl,
        'year': int(str(week)[:4]),
    })

wdf = pd.DataFrame(weekly_results)

print(f"\nHAFTALIK ($1000 kasa, kar cekilir):")
print(f"  Pozitif hafta: {(wdf['pnl']>0).sum()}/{len(wdf)} ({(wdf['pnl']>0).mean()*100:.0f}%)")
print(f"  Ort hafta: ${wdf['pnl'].mean():.2f}")
print(f"  En iyi: ${wdf['pnl'].max():.2f}")
print(f"  En kotu: ${wdf['pnl'].min():.2f}")
print(f"  Toplam cekilen: ${wdf[wdf['pnl']>0]['pnl'].sum():,.2f}")
print(f"  Toplam kayip: ${wdf[wdf['pnl']<0]['pnl'].sum():,.2f}")
print(f"  Net: ${wdf['pnl'].sum():,.2f}")
print(f"  Aylik ort: ${wdf['pnl'].sum() / 60:,.2f}")

# Yillik
print(f"\nYILLIK:")
print(f"  {'Yil':>6s} | {'Hafta':>6s} | {'Trade':>6s} | {'Toplam $':>10s} | {'Ort hft $':>10s}")
print(f"  {'-'*50}")
for year, grp in wdf.groupby('year'):
    print(f"  {year:>6d} | {len(grp):>6d} | {grp['trades'].sum():>6d} | ${grp['pnl'].sum():>9,.2f} | ${grp['pnl'].mean():>9.2f}")

# Sweep vs PMAX kaynak bazinda haftalik
print(f"\nKAYNAK BAZINDA DETAY:")
sweep_trades = tdf[tdf['source'] == 'SWEEP']
pmax_trades = tdf[tdf['source'] == 'PMAX']

if len(sweep_trades) > 0 and len(pmax_trades) > 0:
    print(f"  SWEEP: {len(sweep_trades)} trade, ${sweep_trades['pnl_dollar'].sum():,.2f} toplam")
    print(f"    WR: {(sweep_trades['pnl_dollar']>0).mean()*100:.1f}% | Avg: ${sweep_trades['pnl_dollar'].mean():.2f} | Ort sure: {sweep_trades['duration_min'].mean():.0f}dk")
    print(f"  PMAX:  {len(pmax_trades)} trade, ${pmax_trades['pnl_dollar'].sum():,.2f} toplam")
    print(f"    WR: {(pmax_trades['pnl_dollar']>0).mean()*100:.1f}% | Avg: ${pmax_trades['pnl_dollar'].mean():.2f} | Ort sure: {pmax_trades['duration_min'].mean():.0f}dk")

print()
