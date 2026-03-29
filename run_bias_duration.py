"""Bias Gecerlilik Suresi Testi
Sweep sinyalinden sonra yon dogrulugu ne kadar sure korunuyor?
PMAX catisma vs uyumlu ayri ayri.
"""
import numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import rust_engine

df = pd.read_parquet('data/ETHUSDT_5m_5y.parquet')
df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
c = df['close'].values.astype(np.float64)
h = df['high'].values.astype(np.float64)
l = df['low'].values.astype(np.float64)
bv = df['buy_vol'].values.astype(np.float64)
sv = df['sell_vol'].values.astype(np.float64)
n = len(df)

delta = bv - sv
cvd = np.cumsum(delta)
sa = np.ascontiguousarray
BP = 48; EARLY_BARS = 12

# PMAX
src = sa((h + l) / 2, dtype=np.float64)
pmax_result = rust_engine.compute_adaptive_pmax(
    src, sa(h), sa(l), sa(c),
    10, 3.0, 10, 580, 440, 4.0, 1.25, 3, 5.5, 19, 1.5, 29,
)
pmax_dir = np.array(pmax_result['direction'])

# Mumlar + eventler
candles = []
for i in range(n // BP):
    s = i * BP; e = s + BP
    candles.append({'start': s, 'open': c[s], 'high': np.max(h[s:e]), 'low': np.min(l[s:e]), 'close': c[e-1], 'dt': df['dt'].iloc[s]})

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

# Walk-forward ile grid sinyalleri
events = []
current_th = None; last_train_end = 0
TRAIN = 540; UPDATE = 180

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

    edf_list.append({
        'candle_idx': i, 'bar': bar, 'dt': curr['dt'],
        'signal': sig, 'sweep_dir': sweep_dir,
        'roc': roc, 'pre_cvd': pcv, 'early_p': early_p,
        'entry_bar': entry_bar,
        'pmax_at_entry': pmax_dir[entry_bar] if entry_bar < n else 0,
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

# Grid sinyal hesapla
current_th = None; last_train_end = 0
for idx in range(len(edf)):
    ev = edf.iloc[idx]
    ci = ev['candle_idx']
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
    rt = tercile_from_th(ev['roc'], current_th['roc_33'], current_th['roc_67'])
    ct = tercile_from_th(ev['pre_cvd'], current_th['cvd_33'], current_th['cvd_67'])
    et = tercile_from_th(ev['early_p'], current_th['early_33'], current_th['early_67'])
    if (rt,ct,et) in CONT_CELLS:
        edf.at[edf.index[idx], 'grid_signal'] = 'CONT'

# CONT sinyallerini filtrele
cont = edf[edf.get('grid_signal') == 'CONT'].copy()
cont['sweep_long'] = cont['sweep_dir'] == 'high'
cont['pmax_agrees'] = cont.apply(lambda r: (r['pmax_at_entry'] > 0 and r['sweep_dir'] == 'high') or
                                             (r['pmax_at_entry'] < 0 and r['sweep_dir'] == 'low'), axis=1)

print("BIAS GECERLILIK SURESI TESTI")
print("=" * 120)
print(f"Toplam CONT sinyal: {len(cont)}")
print()

# Horizonlar (5m bar cinsinden)
horizons = [
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
]

groups = {
    "Tum CONT": cont,
    "PMAX catisma": cont[~cont['pmax_agrees']],
    "PMAX uyumlu": cont[cont['pmax_agrees']],
}

print(f"  {'Grup':>20s} | {'Horizon':>8s} | {'WR':>7s} | {'Avg ret':>9s} | {'N':>6s}")
print(f"  {'-'*65}")

for group_name, group_df in groups.items():
    for hz, hz_label in horizons:
        rets = []
        for _, row in group_df.iterrows():
            entry_bar = row['entry_bar']
            if entry_bar + hz >= n: continue
            if np.isnan(c[entry_bar]) or c[entry_bar] <= 0: continue

            ret = (c[entry_bar + hz] - c[entry_bar]) / c[entry_bar] * 100
            if row['sweep_dir'] == 'low':
                ret = -ret  # SHORT icin ters cevir
            rets.append(ret)

        if len(rets) < 50: continue
        wr = np.mean([1 if r > 0 else 0 for r in rets]) * 100
        avg = np.mean(rets)
        print(f"  {group_name:>20s} | {hz_label:>8s} | {wr:>6.1f}% | {avg:>+8.3f}% | {len(rets):>6d}")

    print()

# Yillik tutarlilik — PMAX catisma, farkli horizonlarda
print(f"\n{'='*120}")
print("YILLIK TUTARLILIK — PMAX CATISMA GRUBU")
print(f"{'='*120}")

catisma = cont[~cont['pmax_agrees']].copy()
catisma['year'] = pd.to_datetime(catisma['dt']).dt.year

for hz, hz_label in [(36, "3h"), (48, "4h"), (96, "8h"), (288, "1d"), (576, "2d")]:
    print(f"\n  Horizon: {hz_label}")
    print(f"  {'Yil':>6s} | {'WR':>7s} | {'Avg ret':>9s} | {'N':>5s}")
    print(f"  {'-'*35}")

    for year, grp in catisma.groupby('year'):
        rets = []
        for _, row in grp.iterrows():
            eb = row['entry_bar']
            if eb + hz >= n: continue
            ret = (c[eb + hz] - c[eb]) / c[eb] * 100
            if row['sweep_dir'] == 'low': ret = -ret
            rets.append(ret)
        if len(rets) < 10: continue
        wr = np.mean([1 if r > 0 else 0 for r in rets]) * 100
        avg = np.mean(rets)
        print(f"  {year:>6d} | {wr:>6.1f}% | {avg:>+8.3f}% | {len(rets):>5d}")

print()
