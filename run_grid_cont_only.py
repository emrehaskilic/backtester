"""3D Grid Strateji — Sadece CONT sinyalleri
REV cikarildi. Komisyon dahil.
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
EARLY_BARS = 12
COMMISSION = 0.08  # %0.08 round trip (maker 0.02 x2 + slippage)

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

all_events = []
for i in range(1, len(candles)):
    prev = candles[i-1]
    curr = candles[i]
    bar = curr['start']

    swept_high = curr['high'] > prev['high']
    swept_low = curr['low'] < prev['low']
    is_green = curr['close'] > curr['open']
    is_red = curr['close'] < curr['open']

    sig, sweep_dir = None, None
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

    roc = np.nan
    if bar >= 36 and c[bar - 36] > 0:
        roc = (c[bar] - c[bar - 36]) / c[bar - 36] * 100
        if sweep_dir == "low": roc = -roc

    pre_cvd = np.nan
    if bar >= 48:
        pre_cvd = cvd[bar] - cvd[bar - 48]
        if sweep_dir == "low": pre_cvd = -pre_cvd

    early_p = np.nan
    entry_bar = bar + EARLY_BARS
    if entry_bar < n:
        early_p = (c[entry_bar] - c[bar]) / c[bar] * 100
        if sweep_dir == "low": early_p = -early_p

    entry_price = c[entry_bar] if entry_bar < n else np.nan
    exit_bar = bar + BP - 1
    exit_price = c[exit_bar] if exit_bar < n else np.nan

    all_events.append({
        'candle_idx': i, 'bar': bar, 'dt': curr['dt'],
        'signal': sig, 'sweep_dir': sweep_dir,
        'roc': roc, 'pre_cvd': pre_cvd, 'early_p': early_p,
        'entry_price': entry_price, 'exit_price': exit_price,
    })

edf = pd.DataFrame(all_events)

def tercile_from_th(val, th_33, th_67):
    if val <= th_33: return "Ters"
    elif val <= th_67: return "Notr"
    else: return "Ayni"

# Sadece CONT grid hucreleri
CONT_CELLS = {
    ("Ters", "Ters", "Ayni"),
    ("Ters", "Notr", "Ayni"),
    ("Ters", "Ayni", "Ayni"),
    ("Ters", "Ters", "Notr"),
    ("Ters", "Ters", "Ters"),
    ("Ters", "Ayni", "Notr"),
    ("Notr", "Ters", "Ayni"),
    ("Notr", "Notr", "Ayni"),
    ("Ayni", "Ayni", "Ayni"),
    ("Ayni", "Notr", "Ayni"),
}

print("3D GRID STRATEJI — SADECE CONT")
print(f"Komisyon: %{COMMISSION} round trip")
print("=" * 120)

TRAIN = 540
UPDATE = 180

trades = []
current_th = None
last_train_end = 0

for _, ev in edf.iterrows():
    ci = ev['candle_idx']

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

    roc_t = tercile_from_th(ev['roc'], current_th['roc_33'], current_th['roc_67'])
    cvd_t = tercile_from_th(ev['pre_cvd'], current_th['cvd_33'], current_th['cvd_67'])
    early_t = tercile_from_th(ev['early_p'], current_th['early_33'], current_th['early_67'])

    grid_key = (roc_t, cvd_t, early_t)
    if grid_key not in CONT_CELLS: continue

    direction = "LONG" if ev['sweep_dir'] == "high" else "SHORT"

    if direction == "LONG":
        pnl_gross = (ev['exit_price'] - ev['entry_price']) / ev['entry_price'] * 100
    else:
        pnl_gross = (ev['entry_price'] - ev['exit_price']) / ev['entry_price'] * 100

    pnl_net = pnl_gross - COMMISSION

    trades.append({
        'dt': ev['dt'], 'direction': direction,
        'sweep_dir': ev['sweep_dir'],
        'entry': ev['entry_price'], 'exit': ev['exit_price'],
        'pnl_gross': pnl_gross, 'pnl': pnl_net,
        'roc_t': roc_t, 'cvd_t': cvd_t, 'early_t': early_t,
        'year': pd.Timestamp(ev['dt']).year,
    })

tdf = pd.DataFrame(trades)

print(f"\nGENEL SONUCLAR:")
print(f"  Toplam trade: {len(tdf)}")
print(f"  Win rate: {(tdf['pnl'] > 0).mean() * 100:.1f}%")
print(f"  Avg PnL (gross): {tdf['pnl_gross'].mean():.4f}%")
print(f"  Avg PnL (net):   {tdf['pnl'].mean():.4f}%")
print(f"  Median PnL:      {tdf['pnl'].median():.4f}%")
print(f"  Toplam PnL:      {tdf['pnl'].sum():.1f}%")
print()

equity = [100.0]
for pnl in tdf['pnl']:
    equity.append(equity[-1] * (1 + pnl / 100))
equity = np.array(equity)
peak = np.maximum.accumulate(equity)
dd = (peak - equity) / peak * 100

print(f"  $100 → ${equity[-1]:.2f}")
print(f"  Max drawdown: {dd.max():.1f}%")
print()

longs = tdf[tdf['direction'] == 'LONG']
shorts = tdf[tdf['direction'] == 'SHORT']
print(f"  LONG:  {len(longs)} | WR: {(longs['pnl']>0).mean()*100:.1f}% | Avg: {longs['pnl'].mean():.4f}% | Sum: {longs['pnl'].sum():.1f}%")
print(f"  SHORT: {len(shorts)} | WR: {(shorts['pnl']>0).mean()*100:.1f}% | Avg: {shorts['pnl'].mean():.4f}% | Sum: {shorts['pnl'].sum():.1f}%")
print()

print(f"YILLIK PERFORMANS:")
print(f"  {'Yil':>6s} | {'Trade':>6s} | {'WR':>6s} | {'Avg PnL':>9s} | {'Sum PnL':>9s} | {'$100→':>10s}")
print(f"  {'-'*58}")
for year, grp in tdf.groupby('year'):
    eq_y = [100.0]
    for p in grp['pnl']:
        eq_y.append(eq_y[-1] * (1 + p / 100))
    print(f"  {year:>6d} | {len(grp):>6d} | {(grp['pnl']>0).mean()*100:>5.1f}% | {grp['pnl'].mean():>+8.4f}% | {grp['pnl'].sum():>+8.1f}% | ${eq_y[-1]:>9.2f}")

tdf['month'] = pd.to_datetime(tdf['dt']).dt.to_period('M')
monthly = tdf.groupby('month')['pnl'].sum()
print(f"\nAYLIK:")
print(f"  Pozitif ay: {(monthly > 0).sum()} / {len(monthly)} ({(monthly > 0).mean()*100:.0f}%)")
print(f"  Ort ay: {monthly.mean():.2f}%")
print(f"  En iyi: {monthly.max():.2f}%")
print(f"  En kotu: {monthly.min():.2f}%")
print()

# Grid hucre bazinda
print(f"GRID HUCRE PERFORMANSI:")
print(f"  {'ROC':>6s} {'CVD':>6s} {'Early':>6s} | {'N':>5s} | {'WR':>6s} | {'Avg':>9s} | {'Sum':>8s}")
print(f"  {'-'*55}")
for (rt, ct, et), grp in tdf.groupby(['roc_t', 'cvd_t', 'early_t']):
    if len(grp) < 10: continue
    print(f"  {rt:>6s} {ct:>6s} {et:>6s} | {len(grp):>5d} | {(grp['pnl']>0).mean()*100:>5.1f}% | {grp['pnl'].mean():>+8.4f}% | {grp['pnl'].sum():>+7.1f}%")

# Komisyon sensitivity
print(f"\nKOMISYON DUYARLILIGI:")
for comm in [0.0, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
    net = tdf['pnl_gross'] - comm
    total = net.sum()
    wr = (net > 0).mean() * 100
    eq = [100.0]
    for p in net:
        eq.append(eq[-1] * (1 + p / 100))
    print(f"  %{comm:.2f} komisyon: WR={wr:.1f}% | Toplam={total:>+8.1f}% | $100→${eq[-1]:.2f}")

print()
