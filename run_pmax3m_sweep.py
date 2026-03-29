"""PMAX 3m TF + Sweep Bias Entegrasyonu
PMAX 3m'de calisir (orijinal WF optimize parametreler).
Sweep sinyalleri 5m'de calisir.
Timestamp bazli eslestirme.
"""
import numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import rust_engine

# 3m veri (PMAX icin)
df3 = pd.read_parquet('data/ETHUSDT_3m_5y_perp.parquet')
df3['dt'] = pd.to_datetime(df3['open_time'], unit='ms')
c3 = np.ascontiguousarray(df3['close'].values, dtype=np.float64)
h3 = np.ascontiguousarray(df3['high'].values, dtype=np.float64)
l3 = np.ascontiguousarray(df3['low'].values, dtype=np.float64)
src3 = np.ascontiguousarray((h3 + l3) / 2, dtype=np.float64)

# 5m veri (sweep icin)
df5 = pd.read_parquet('data/ETHUSDT_5m_5y.parquet')
df5['dt'] = pd.to_datetime(df5['open_time'], unit='ms')
c5 = df5['close'].values.astype(np.float64)
h5 = df5['high'].values.astype(np.float64)
l5 = df5['low'].values.astype(np.float64)
bv5 = df5['buy_vol'].values.astype(np.float64)
sv5 = df5['sell_vol'].values.astype(np.float64)
n5 = len(df5)

delta5 = bv5 - sv5
cvd5 = np.cumsum(delta5)

print("PMAX 3m + SWEEP BIAS ENTEGRASYONU")
print("=" * 120)
print(f"3m data: {len(df3):,} bars | 5m data: {n5:,} bars")
print()

# PMAX 3m'de hesapla
print("PMAX 3m hesaplaniyor (WF consensus parametreleri)...")
pmax_result = rust_engine.compute_adaptive_pmax(
    src3, h3, l3, c3,
    10, 3.0, 10,    # base params
    580, 440,        # lookback, flip_window
    4.0, 1.25,       # mult_base, mult_scale
    3, 5.5,          # ma_base, ma_scale
    19, 1.5,         # atr_base, atr_scale
    29,              # update_interval
)
pmax_dir_3m = np.array(pmax_result['direction'])
pmax_ts_3m = df3['open_time'].values  # timestamp'ler

long_pct = (pmax_dir_3m > 0).sum() / len(pmax_dir_3m) * 100
short_pct = (pmax_dir_3m < 0).sum() / len(pmax_dir_3m) * 100
print(f"  PMAX 3m: LONG {long_pct:.1f}% | SHORT {short_pct:.1f}%")

# PMAX yonunu 5m bar'lara esle (timestamp bazli)
print("PMAX 3m → 5m eslestirme...")
pmax_dir_at_5m = np.zeros(n5)
ts5 = df5['open_time'].values

j = 0  # 3m index
for i in range(n5):
    # 5m bar'in timestamp'ine en yakin 3m bar'i bul
    while j < len(pmax_ts_3m) - 1 and pmax_ts_3m[j + 1] <= ts5[i]:
        j += 1
    pmax_dir_at_5m[i] = pmax_dir_3m[j]

long_5m = (pmax_dir_at_5m > 0).sum() / n5 * 100
short_5m = (pmax_dir_at_5m < 0).sum() / n5 * 100
print(f"  PMAX 5m'de: LONG {long_5m:.1f}% | SHORT {short_5m:.1f}%")

# ══════════════════════════════════════════════════════════════
# Sweep sinyalleri (ayni kod)
# ══════════════════════════════════════════════════════════════

BP = 48; EARLY_BARS = 12; COMMISSION = 0.08

candles = []
for i in range(n5 // BP):
    s = i * BP; e = s + BP
    candles.append({'start': s, 'open': c5[s], 'high': np.max(h5[s:e]), 'low': np.min(l5[s:e]), 'close': c5[e-1], 'dt': df5['dt'].iloc[s]})

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
    if bar >= 36 and c5[bar-36] > 0:
        roc = (c5[bar]-c5[bar-36])/c5[bar-36]*100
        if sweep_dir == 'low': roc = -roc
    pcv = np.nan
    if bar >= 48:
        pcv = cvd5[bar]-cvd5[bar-48]
        if sweep_dir == 'low': pcv = -pcv
    early_p = np.nan
    entry_bar = bar + EARLY_BARS
    if entry_bar < n5:
        early_p = (c5[entry_bar]-c5[bar])/c5[bar]*100
        if sweep_dir == 'low': early_p = -early_p

    pmax_at_entry = pmax_dir_at_5m[entry_bar] if entry_bar < n5 else 0

    edf_list.append({
        'candle_idx': i, 'bar': bar, 'dt': curr['dt'],
        'signal': sig, 'sweep_dir': sweep_dir,
        'roc': roc, 'pre_cvd': pcv, 'early_p': early_p,
        'entry_bar': entry_bar,
        'entry_price': c5[entry_bar] if entry_bar < n5 else np.nan,
        'exit_price': c5[bar + BP - 1] if bar + BP - 1 < n5 else np.nan,
        'pmax_dir': pmax_at_entry,
    })

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

edf = pd.DataFrame(edf_list)

# Grid sinyali
current_th = None; last_train_end = 0
for idx in range(len(edf)):
    ev = edf.iloc[idx]; ci = ev['candle_idx']
    if ci >= TRAIN and (current_th is None or ci - last_train_end >= UPDATE):
        td = edf[(edf['candle_idx'] >= ci-TRAIN) & (edf['candle_idx'] < ci)]
        tv = td[td['signal'].notna()].dropna(subset=['roc','pre_cvd','early_p'])
        if len(tv) >= 50:
            current_th = {'roc_33':np.percentile(tv['roc'],33.3),'roc_67':np.percentile(tv['roc'],66.7),
                          'cvd_33':np.percentile(tv['pre_cvd'],33.3),'cvd_67':np.percentile(tv['pre_cvd'],66.7),
                          'early_33':np.percentile(tv['early_p'],33.3),'early_67':np.percentile(tv['early_p'],66.7)}
            last_train_end = ci
    if current_th is None or ev['signal'] is None: continue
    if pd.isna(ev['roc']) or pd.isna(ev['pre_cvd']) or pd.isna(ev['early_p']): continue
    if pd.isna(ev['entry_price']) or pd.isna(ev['exit_price']): continue
    rt = tercile_from_th(ev['roc'], current_th['roc_33'], current_th['roc_67'])
    ct = tercile_from_th(ev['pre_cvd'], current_th['cvd_33'], current_th['cvd_67'])
    et = tercile_from_th(ev['early_p'], current_th['early_33'], current_th['early_67'])
    if (rt,ct,et) in CONT_CELLS:
        edf.at[edf.index[idx], 'grid_signal'] = 'CONT'

cont = edf[edf.get('grid_signal') == 'CONT'].copy()
cont['pmax_agrees'] = cont.apply(lambda r: (r['pmax_dir'] > 0 and r['sweep_dir'] == 'high') or
                                             (r['pmax_dir'] < 0 and r['sweep_dir'] == 'low'), axis=1)

print(f"\nToplam CONT sinyal: {len(cont)}")
print(f"PMAX 3m uyumlu: {cont['pmax_agrees'].sum()} ({cont['pmax_agrees'].mean()*100:.1f}%)")
print(f"PMAX 3m catisma: {(~cont['pmax_agrees']).sum()} ({(~cont['pmax_agrees']).mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# Sonuclar
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*120}")
print("PMAX 3m FILTRE — YON DOGRULUGU ZAMANA GORE")
print(f"{'='*120}")

horizons = [(6,"30dk"), (12,"1h"), (24,"2h"), (36,"3h"), (48,"4h"),
            (72,"6h"), (96,"8h"), (144,"12h"), (288,"1d"), (576,"2d")]

groups = {
    "Tum CONT": cont,
    "PMAX 3m catisma": cont[~cont['pmax_agrees']],
    "PMAX 3m uyumlu": cont[cont['pmax_agrees']],
}

print(f"\n  {'Grup':>20s} | {'Horizon':>8s} | {'WR':>7s} | {'Avg ret':>9s} | {'N':>6s}")
print(f"  {'-'*65}")

for group_name, group_df in groups.items():
    for hz, hz_label in horizons:
        rets = []
        for _, row in group_df.iterrows():
            eb = row['entry_bar']
            if eb + hz >= n5 or c5[eb] <= 0: continue
            ret = (c5[eb + hz] - c5[eb]) / c5[eb] * 100
            if row['sweep_dir'] == 'low': ret = -ret
            rets.append(ret)
        if len(rets) < 50: continue
        wr = np.mean([1 if r > 0 else 0 for r in rets]) * 100
        avg = np.mean(rets)
        print(f"  {group_name:>20s} | {hz_label:>8s} | {wr:>6.1f}% | {avg:>+8.3f}% | {len(rets):>6d}")
    print()

# Backtest — PMAX catisma
print(f"\n{'='*120}")
print("BACKTEST KARSILASTIRMA (komisyon %0.08)")
print(f"{'='*120}")

def backtest(df_trades, label):
    df_t = df_trades.copy()
    def calc_pnl(row):
        if row['sweep_dir'] == 'high':
            return (row['exit_price'] - row['entry_price']) / row['entry_price'] * 100 - COMMISSION
        else:
            return (row['entry_price'] - row['exit_price']) / row['entry_price'] * 100 - COMMISSION
    df_t['pnl'] = df_t.apply(calc_pnl, axis=1)

    equity = [100.0]
    for p in df_t['pnl']:
        equity.append(equity[-1] * (1 + p / 100))
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100

    df_t['week'] = pd.to_datetime(df_t['dt']).dt.to_period('W')
    weekly = df_t.groupby('week')['pnl'].sum()
    df_t['year'] = pd.to_datetime(df_t['dt']).dt.year

    print(f"\n  {label}:")
    print(f"    N: {len(df_t)} | WR: {(df_t['pnl']>0).mean()*100:.1f}% | Avg: {df_t['pnl'].mean():.4f}% | Sum: {df_t['pnl'].sum():.1f}%")
    print(f"    Max DD: {dd.max():.1f}% | Poz hafta: {(weekly>0).sum()}/{len(weekly)} ({(weekly>0).mean()*100:.0f}%)")

    print(f"    Yillik:")
    for year, grp in df_t.groupby('year'):
        print(f"      {year}: N={len(grp)} WR={((grp['pnl']>0).mean()*100):.1f}% Sum={grp['pnl'].sum():+.1f}%")

backtest(cont, "Tum CONT")
backtest(cont[~cont['pmax_agrees']], "PMAX 3m catisma")
backtest(cont[cont['pmax_agrees']], "PMAX 3m uyumlu")

# PMAX 5m ile karsilastir
print(f"\n  --- PMAX 5m (onceki test) vs PMAX 3m karsilastirma ---")
print(f"  PMAX 3m catisma: {(~cont['pmax_agrees']).sum()} trade | {(~cont['pmax_agrees']).mean()*100:.1f}%")

# PMAX 5m'yi de hesapla karsilastirma icin
src5 = np.ascontiguousarray((h5 + l5) / 2, dtype=np.float64)
pmax5 = rust_engine.compute_adaptive_pmax(
    src5, np.ascontiguousarray(h5), np.ascontiguousarray(l5), np.ascontiguousarray(c5),
    10, 3.0, 10, 580, 440, 4.0, 1.25, 3, 5.5, 19, 1.5, 29,
)
pmax_dir_5m = np.array(pmax5['direction'])

cont5 = cont.copy()
cont5['pmax5_agrees'] = cont5.apply(lambda r: (pmax_dir_5m[int(r['entry_bar'])] > 0 and r['sweep_dir'] == 'high') or
                                                (pmax_dir_5m[int(r['entry_bar'])] < 0 and r['sweep_dir'] == 'low'), axis=1)

print(f"  PMAX 5m catisma: {(~cont5['pmax5_agrees']).sum()} trade | {(~cont5['pmax5_agrees']).mean()*100:.1f}%")

# Her ikisinin WR karsilastirmasi
catisma_3m = cont[~cont['pmax_agrees']]
catisma_5m = cont5[~cont5['pmax5_agrees']]

for hz, lbl in [(36,"3h"), (48,"4h"), (96,"8h"), (288,"1d")]:
    rets_3m = [(c5[int(r['entry_bar'])+hz]-c5[int(r['entry_bar'])])/c5[int(r['entry_bar'])]*100 * (1 if r['sweep_dir']=='high' else -1)
               for _,r in catisma_3m.iterrows() if int(r['entry_bar'])+hz < n5]
    rets_5m = [(c5[int(r['entry_bar'])+hz]-c5[int(r['entry_bar'])])/c5[int(r['entry_bar'])]*100 * (1 if r['sweep_dir']=='high' else -1)
               for _,r in catisma_5m.iterrows() if int(r['entry_bar'])+hz < n5]
    wr3 = np.mean([1 if r>0 else 0 for r in rets_3m])*100 if rets_3m else 0
    wr5 = np.mean([1 if r>0 else 0 for r in rets_5m])*100 if rets_5m else 0
    print(f"  {lbl}: PMAX 3m WR={wr3:.1f}% (N={len(rets_3m)}) | PMAX 5m WR={wr5:.1f}% (N={len(rets_5m)})")

print()
