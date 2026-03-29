"""PMAX Filtre Testi
Sweep sinyali + PMAX ters yonde = guclendirilmis sinyal
Sweep sinyali + PMAX ayni yonde = zayiflamis sinyal
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
BP = 48; EARLY_BARS = 12; COMMISSION = 0.08

# PMAX
src = sa((h + l) / 2, dtype=np.float64)
pmax_result = rust_engine.compute_adaptive_pmax(
    src, sa(h), sa(l), sa(c),
    10, 3.0, 10, 580, 440, 4.0, 1.25, 3, 5.5, 19, 1.5, 29,
)
pmax_dir = np.array(pmax_result['direction'])

# Mumlar
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

# Event'ler + grid sinyal + PMAX durumu
all_events = []
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

    entry_price = c[entry_bar] if entry_bar < n else np.nan
    exit_price = c[bar + BP - 1] if bar + BP - 1 < n else np.nan

    # PMAX yonu entry aninda
    pmax_at_entry = pmax_dir[entry_bar] if entry_bar < n else 0

    edf_list.append({
        'candle_idx': i, 'bar': bar, 'dt': curr['dt'],
        'signal': sig, 'sweep_dir': sweep_dir,
        'roc': roc, 'pre_cvd': pcv, 'early_p': early_p,
        'entry_price': entry_price, 'exit_price': exit_price,
        'pmax_dir': pmax_at_entry,
    })

    # Threshold guncelle
    if i >= TRAIN and (current_th is None or i - last_train_end >= UPDATE):
        td = [e for e in edf_list if e['candle_idx'] >= i-TRAIN and e['candle_idx'] < i and e['signal'] is not None]
        if len(td) >= 50:
            rocs = []; cvds = []; earlies = []
            for e in td:
                b = e['bar']
                if b >= 36 and c[b-36]>0:
                    r = (c[b]-c[b-36])/c[b-36]*100
                    if e['sweep_dir']=='low': r=-r
                    rocs.append(r)
                if b >= 48:
                    cv = cvd[b]-cvd[b-48]
                    if e['sweep_dir']=='low': cv=-cv
                    cvds.append(cv)
                eb = b+EARLY_BARS
                if eb < n:
                    ep = (c[eb]-c[b])/c[b]*100
                    if e['sweep_dir']=='low': ep=-ep
                    earlies.append(ep)
            if len(rocs)>=30 and len(cvds)>=30 and len(earlies)>=30:
                current_th = {'roc_33':np.percentile(rocs,33.3),'roc_67':np.percentile(rocs,66.7),
                              'cvd_33':np.percentile(cvds,33.3),'cvd_67':np.percentile(cvds,66.7),
                              'early_33':np.percentile(earlies,33.3),'early_67':np.percentile(earlies,66.7)}
                last_train_end = i

edf = pd.DataFrame(edf_list)

# Grid sinyali hesapla
edf['grid_signal'] = None
for idx, ev in edf.iterrows():
    if current_th is None: continue
    if ev['signal'] is None or pd.isna(ev['roc']) or pd.isna(ev['pre_cvd']) or pd.isna(ev['early_p']): continue
    # En son threshold'u kullan (basitlik icin)
    # Gercekte walk-forward'da o anki threshold kullanilmali ama yaklasik olarak yeterli
edf['grid_signal'] = None

# Tekrar walk-forward ile grid sinyal hesapla
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
    if pd.isna(ev['entry_price']) or pd.isna(ev['exit_price']): continue

    rt = tercile_from_th(ev['roc'], current_th['roc_33'], current_th['roc_67'])
    ct = tercile_from_th(ev['pre_cvd'], current_th['cvd_33'], current_th['cvd_67'])
    et = tercile_from_th(ev['early_p'], current_th['early_33'], current_th['early_67'])

    if (rt,ct,et) in CONT_CELLS:
        edf.at[edf.index[idx], 'grid_signal'] = 'CONT'

# Filtrele: sadece grid CONT olanlar
cont = edf[edf['grid_signal'] == 'CONT'].copy()
cont['sweep_long'] = cont['sweep_dir'] == 'high'
cont['pmax_agrees'] = cont.apply(lambda r: (r['pmax_dir'] > 0 and r['sweep_dir'] == 'high') or
                                             (r['pmax_dir'] < 0 and r['sweep_dir'] == 'low'), axis=1)

# PnL hesapla
def calc_pnl(row):
    if row['sweep_dir'] == 'high':
        return (row['exit_price'] - row['entry_price']) / row['entry_price'] * 100 - COMMISSION
    else:
        return (row['entry_price'] - row['exit_price']) / row['entry_price'] * 100 - COMMISSION

cont['pnl'] = cont.apply(calc_pnl, axis=1)
cont['year'] = pd.to_datetime(cont['dt']).dt.year

print("PMAX FILTRE TESTI")
print("=" * 120)
print(f"Toplam CONT sinyal: {len(cont)}")
print(f"PMAX uyumlu: {cont['pmax_agrees'].sum()} ({cont['pmax_agrees'].mean()*100:.1f}%)")
print(f"PMAX catisma: {(~cont['pmax_agrees']).sum()} ({(~cont['pmax_agrees']).mean()*100:.1f}%)")
print()

# 3 mod karsilastir
modes = {
    "A: Tum CONT (filtre yok)": cont,
    "B: Sadece PMAX uyumlu": cont[cont['pmax_agrees']],
    "C: Sadece PMAX catisma": cont[~cont['pmax_agrees']],
}

print(f"{'Mod':>30s} | {'N':>6s} | {'WR':>6s} | {'Avg PnL':>9s} | {'Sum PnL':>9s} | {'$/trade':>8s}")
print(f"{'-'*80}")

for mod_name, mod_df in modes.items():
    if len(mod_df) == 0: continue
    wr = (mod_df['pnl'] > 0).mean() * 100
    avg = mod_df['pnl'].mean()
    total = mod_df['pnl'].sum()

    equity = [100.0]
    for p in mod_df['pnl']:
        equity.append(equity[-1] * (1 + p / 100))
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100

    print(f"{mod_name:>30s} | {len(mod_df):>6d} | {wr:>5.1f}% | {avg:>+8.4f}% | {total:>+8.1f}% | DD:{dd.max():.1f}%")

# Yillik detay her mod icin
print(f"\nYILLIK DETAY:")
for mod_name, mod_df in modes.items():
    if len(mod_df) == 0: continue
    print(f"\n  {mod_name}:")
    print(f"  {'Yil':>6s} | {'N':>5s} | {'WR':>6s} | {'Sum':>9s}")
    print(f"  {'-'*35}")
    for year, grp in mod_df.groupby('year'):
        print(f"  {year:>6d} | {len(grp):>5d} | {(grp['pnl']>0).mean()*100:>5.1f}% | {grp['pnl'].sum():>+8.1f}%")

# Haftalik
print(f"\nHAFTALIK OZET:")
for mod_name, mod_df in modes.items():
    if len(mod_df) == 0: continue
    mod_w = mod_df.copy()
    mod_w['week'] = pd.to_datetime(mod_w['dt']).dt.to_period('W')
    weekly = mod_w.groupby('week')['pnl'].sum()
    neg = (weekly < 0).sum()
    print(f"  {mod_name:>30s}: {(weekly>0).sum()}/{len(weekly)} poz hft ({(weekly>0).mean()*100:.0f}%) | En kotu: {weekly.min():.2f}%")

print()
