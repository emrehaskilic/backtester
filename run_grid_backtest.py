"""3D Grid Strateji Backtest
Giris: 4H mumun 1. saatinin sonu (12. bar)
Cikis: 4H mumun kapanisi (48. bar)
Yon: Grid sinyaline gore
"""
import numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
        'dt': df['dt'].iloc[s],
    })

# Walk-forward: threshold'lari rolling hesapla
# Train window: 540 mum (~3 ay), her 180 mumda guncelle (~1 ay)
TRAIN = 540
UPDATE = 180

# Tum event'leri once topla
all_events = []
for i in range(1, len(candles)):
    prev = candles[i-1]
    curr = candles[i]
    bar = curr['start']

    swept_high = curr['high'] > prev['high']
    swept_low = curr['low'] < prev['low']
    is_green = curr['close'] > curr['open']
    is_red = curr['close'] < curr['open']

    if swept_high and not swept_low:
        sig = "high_cont" if curr['close'] > prev['high'] else ("high_rev" if is_red else None)
        sweep_dir = "high"
    elif swept_low and not swept_high:
        sig = "low_cont" if curr['close'] < prev['low'] else ("low_rev" if is_green else None)
        sweep_dir = "low"
    elif swept_high and swept_low:
        if curr['close'] > prev['high']: sig, sweep_dir = "high_cont", "high"
        elif curr['close'] < prev['low']: sig, sweep_dir = "low_cont", "low"
        elif is_red: sig, sweep_dir = "high_rev", "high"
        elif is_green: sig, sweep_dir = "low_rev", "low"
        else: sig, sweep_dir = None, None
    else:
        sig, sweep_dir = None, None

    # ROC 3h
    roc = np.nan
    if bar >= 36 and c[bar - 36] > 0:
        roc = (c[bar] - c[bar - 36]) / c[bar - 36] * 100
        if sweep_dir == "low": roc = -roc

    # Pre-CVD 4h
    pre_cvd = np.nan
    if bar >= 48:
        pre_cvd = cvd[bar] - cvd[bar - 48]
        if sweep_dir == "low": pre_cvd = -pre_cvd

    # Ilk 1h fiyat
    early_p = np.nan
    entry_bar = bar + EARLY_BARS
    if entry_bar < n:
        early_p = (c[entry_bar] - c[bar]) / c[bar] * 100
        if sweep_dir == "low": early_p = -early_p

    # Entry ve exit fiyatlari
    entry_price = c[entry_bar] if entry_bar < n else np.nan
    exit_bar = bar + BP - 1  # mum kapanisi
    exit_price = c[exit_bar] if exit_bar < n else np.nan

    # MFE/MAE (giris ile cikis arasi)
    mfe = 0.0
    mae = 0.0
    if entry_bar < n and exit_bar < n and entry_price > 0:
        for b in range(entry_bar, exit_bar + 1):
            if b >= n: break
            if sweep_dir == "high":
                fav = (h[b] - entry_price) / entry_price * 100
                adv = (entry_price - l[b]) / entry_price * 100
            else:
                fav = (entry_price - l[b]) / entry_price * 100
                adv = (h[b] - entry_price) / entry_price * 100
            if fav > mfe: mfe = fav
            if adv > mae: mae = adv

    all_events.append({
        'candle_idx': i, 'bar': bar, 'dt': curr['dt'],
        'signal': sig, 'sweep_dir': sweep_dir,
        'roc': roc, 'pre_cvd': pre_cvd, 'early_p': early_p,
        'entry_price': entry_price, 'exit_price': exit_price,
        'mfe': mfe, 'mae': mae,
    })

edf = pd.DataFrame(all_events)

def tercile_from_th(val, th_33, th_67):
    if val <= th_33: return "Ters"
    elif val <= th_67: return "Notr"
    else: return "Ayni"

# Grid sinyal tablosu
GRID = {
    ("Ters", "Ters", "Ayni"): "CONT",
    ("Ters", "Notr", "Ayni"): "CONT",
    ("Ters", "Ayni", "Ayni"): "CONT",
    ("Ters", "Ters", "Notr"): "CONT",
    ("Ters", "Ters", "Ters"): "CONT",
    ("Ters", "Ayni", "Notr"): "CONT",
    ("Notr", "Ters", "Ayni"): "CONT",
    ("Notr", "Notr", "Ayni"): "CONT",
    ("Ayni", "Ayni", "Ayni"): "CONT",
    ("Ayni", "Notr", "Ayni"): "CONT",
    ("Ayni", "Ayni", "Ters"): "REV",
    ("Ayni", "Notr", "Ters"): "REV",
    ("Ayni", "Ters", "Ters"): "REV",
    ("Notr", "Notr", "Notr"): "REV",
}

print("3D GRID STRATEJI BACKTEST")
print("=" * 120)
print(f"Giris: 4H mumun 1. saatinin sonu | Cikis: 4H mum kapanisi")
print(f"Walk-forward: {TRAIN} mum train, {UPDATE} mumda bir guncelle")
print()

# ══════════════════════════════════════════════════════════════
# Walk-forward backtest
# ══════════════════════════════════════════════════════════════

trades = []
current_th = None  # (roc_33, roc_67, cvd_33, cvd_67, early_33, early_67)
last_train_end = 0

for _, ev in edf.iterrows():
    ci = ev['candle_idx']

    # Threshold guncelle
    if ci >= TRAIN and (current_th is None or ci - last_train_end >= UPDATE):
        train_data = edf[(edf['candle_idx'] >= ci - TRAIN) & (edf['candle_idx'] < ci)]
        train_valid = train_data.dropna(subset=['roc', 'pre_cvd', 'early_p'])
        train_sweep = train_valid[train_valid['signal'].notna()]

        if len(train_sweep) >= 50:
            current_th = {
                'roc_33': np.percentile(train_sweep['roc'], 33.3),
                'roc_67': np.percentile(train_sweep['roc'], 66.7),
                'cvd_33': np.percentile(train_sweep['pre_cvd'], 33.3),
                'cvd_67': np.percentile(train_sweep['pre_cvd'], 66.7),
                'early_33': np.percentile(train_sweep['early_p'], 33.3),
                'early_67': np.percentile(train_sweep['early_p'], 66.7),
            }
            last_train_end = ci

    if current_th is None: continue
    if ev['signal'] is None: continue
    if pd.isna(ev['roc']) or pd.isna(ev['pre_cvd']) or pd.isna(ev['early_p']): continue
    if pd.isna(ev['entry_price']) or pd.isna(ev['exit_price']): continue

    # Tercile
    roc_t = tercile_from_th(ev['roc'], current_th['roc_33'], current_th['roc_67'])
    cvd_t = tercile_from_th(ev['pre_cvd'], current_th['cvd_33'], current_th['cvd_67'])
    early_t = tercile_from_th(ev['early_p'], current_th['early_33'], current_th['early_67'])

    grid_key = (roc_t, cvd_t, early_t)
    grid_signal = GRID.get(grid_key, "PAS")

    if grid_signal == "PAS": continue

    # Trade
    if grid_signal == "CONT":
        direction = "LONG" if ev['sweep_dir'] == "high" else "SHORT"
    else:  # REV
        direction = "SHORT" if ev['sweep_dir'] == "high" else "LONG"

    if direction == "LONG":
        pnl = (ev['exit_price'] - ev['entry_price']) / ev['entry_price'] * 100
    else:
        pnl = (ev['entry_price'] - ev['exit_price']) / ev['entry_price'] * 100

    trades.append({
        'dt': ev['dt'], 'direction': direction, 'signal': grid_signal,
        'sweep_dir': ev['sweep_dir'],
        'entry': ev['entry_price'], 'exit': ev['exit_price'],
        'pnl': pnl, 'mfe': ev['mfe'], 'mae': ev['mae'],
        'roc_t': roc_t, 'cvd_t': cvd_t, 'early_t': early_t,
        'year': ev['dt'].year if hasattr(ev['dt'], 'year') else pd.Timestamp(ev['dt']).year,
    })

tdf = pd.DataFrame(trades)

if len(tdf) == 0:
    print("Trade yok!")
    sys.exit()

# ══════════════════════════════════════════════════════════════
# Sonuclar
# ══════════════════════════════════════════════════════════════

print(f"GENEL SONUCLAR:")
print(f"  Toplam trade: {len(tdf)}")
print(f"  Win rate: {(tdf['pnl'] > 0).mean() * 100:.1f}%")
print(f"  Ortalama PnL: {tdf['pnl'].mean():.4f}%")
print(f"  Median PnL: {tdf['pnl'].median():.4f}%")
print(f"  Toplam PnL: {tdf['pnl'].sum():.1f}%")
print(f"  Avg MFE: {tdf['mfe'].mean():.3f}%")
print(f"  Avg MAE: {tdf['mae'].mean():.3f}%")
print()

# Equity curve
equity = [100.0]
for pnl in tdf['pnl']:
    equity.append(equity[-1] * (1 + pnl / 100))
equity = np.array(equity)
peak = np.maximum.accumulate(equity)
dd = (peak - equity) / peak * 100

print(f"  $100 → ${equity[-1]:.2f}")
print(f"  Max drawdown: {dd.max():.1f}%")
print()

# CONT vs REV
cont_trades = tdf[tdf['signal'] == 'CONT']
rev_trades = tdf[tdf['signal'] == 'REV']
print(f"  CONT trades: {len(cont_trades)} | WR: {(cont_trades['pnl']>0).mean()*100:.1f}% | Avg: {cont_trades['pnl'].mean():.4f}% | Sum: {cont_trades['pnl'].sum():.1f}%")
print(f"  REV trades:  {len(rev_trades)} | WR: {(rev_trades['pnl']>0).mean()*100:.1f}% | Avg: {rev_trades['pnl'].mean():.4f}% | Sum: {rev_trades['pnl'].sum():.1f}%")
print()

# LONG vs SHORT
longs = tdf[tdf['direction'] == 'LONG']
shorts = tdf[tdf['direction'] == 'SHORT']
print(f"  LONG:  {len(longs)} | WR: {(longs['pnl']>0).mean()*100:.1f}% | Avg: {longs['pnl'].mean():.4f}% | Sum: {longs['pnl'].sum():.1f}%")
print(f"  SHORT: {len(shorts)} | WR: {(shorts['pnl']>0).mean()*100:.1f}% | Avg: {shorts['pnl'].mean():.4f}% | Sum: {shorts['pnl'].sum():.1f}%")
print()

# Yillik
print(f"YILLIK PERFORMANS:")
print(f"  {'Yil':>6s} | {'Trade':>6s} | {'WR':>6s} | {'Avg PnL':>9s} | {'Sum PnL':>9s} | {'$100→':>10s}")
print(f"  {'-'*55}")
for year, grp in tdf.groupby('year'):
    eq_y = [100.0]
    for p in grp['pnl']:
        eq_y.append(eq_y[-1] * (1 + p / 100))
    print(f"  {year:>6d} | {len(grp):>6d} | {(grp['pnl']>0).mean()*100:>5.1f}% | {grp['pnl'].mean():>+8.4f}% | {grp['pnl'].sum():>+8.1f}% | ${eq_y[-1]:>9.2f}")

# Aylik
print(f"\nAYLIK PERFORMANS:")
tdf['month'] = pd.to_datetime(tdf['dt']).dt.to_period('M')
monthly = tdf.groupby('month')['pnl'].agg(['sum', 'count', lambda x: (x>0).mean()*100])
monthly.columns = ['sum', 'count', 'wr']
print(f"  Pozitif ay: {(monthly['sum'] > 0).sum()} / {len(monthly)} ({(monthly['sum'] > 0).mean()*100:.0f}%)")
print(f"  Ortalama ay: {monthly['sum'].mean():.2f}%")
print(f"  En iyi ay: {monthly['sum'].max():.2f}%")
print(f"  En kotu ay: {monthly['sum'].min():.2f}%")
print()

# Grid hucre bazinda performans
print(f"GRID HUCRE BAZINDA PERFORMANS:")
print(f"  {'ROC':>6s} {'CVD':>6s} {'Early':>6s} {'Sinyal':>6s} | {'N':>5s} | {'WR':>6s} | {'Avg PnL':>9s} | {'Sum':>8s}")
print(f"  {'-'*65}")
for (rt, ct, et, sig), grp in tdf.groupby(['roc_t', 'cvd_t', 'early_t', 'signal']):
    if len(grp) < 10: continue
    print(f"  {rt:>6s} {ct:>6s} {et:>6s} {sig:>6s} | {len(grp):>5d} | {(grp['pnl']>0).mean()*100:>5.1f}% | {grp['pnl'].mean():>+8.4f}% | {grp['pnl'].sum():>+7.1f}%")

print()
