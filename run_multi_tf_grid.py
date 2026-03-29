"""Multi-TF Grid Strateji — 1H, 4H, 8H sweep'leri ayni 3D grid mantigi
Her timeframe kendi sweep'ini tespit eder, ayni 3 sinyal (ROC 3h + Pre-CVD 4h + ilk 1h fiyat)
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
COMMISSION = 0.08

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

def run_tf_backtest(bp, tf_name):
    """Tek timeframe icin 3D grid backtest"""
    # ROC lookback = bp'nin %75'i (sweet spot orani)
    # 4H=48 → ROC 36 (3h), 1H=12 → ROC 9 (45dk), 8H=96 → ROC 72 (6h)
    roc_lb = max(6, int(bp * 0.75))
    # Pre-CVD lookback = bp (1 mum oncesi CVD)
    cvd_lb = bp
    # Early bars = bp'nin %25'i (mumun ilk ceyreği)
    early_bars = max(3, bp // 4)

    early_time = f"{early_bars * 5}dk" if early_bars * 5 < 60 else f"{early_bars * 5 // 60}h"
    roc_time = f"{roc_lb * 5}dk" if roc_lb * 5 < 60 else f"{roc_lb * 5 / 60:.1f}h"

    n_candles = n // bp
    candles = []
    for i in range(n_candles):
        s = i * bp; e = s + bp
        candles.append({
            'start': s, 'open': c[s],
            'high': np.max(h[s:e]), 'low': np.min(l[s:e]),
            'close': c[e-1], 'dt': df['dt'].iloc[s],
        })

    all_events = []
    for i in range(1, len(candles)):
        prev = candles[i-1]; curr = candles[i]; bar = curr['start']
        swept_high = curr['high'] > prev['high']; swept_low = curr['low'] < prev['low']
        is_green = curr['close'] > curr['open']; is_red = curr['close'] < curr['open']

        sig, sweep_dir = None, None
        if swept_high and not swept_low:
            sig = 'high_cont' if curr['close'] > prev['high'] else ('high_rev' if is_red else None)
            sweep_dir = 'high'
        elif swept_low and not swept_high:
            sig = 'low_cont' if curr['close'] < prev['low'] else ('low_rev' if is_green else None)
            sweep_dir = 'low'
        elif swept_high and swept_low:
            if curr['close'] > prev['high']: sig, sweep_dir = 'high_cont', 'high'
            elif curr['close'] < prev['low']: sig, sweep_dir = 'low_cont', 'low'
            elif is_red: sig, sweep_dir = 'high_rev', 'high'
            elif is_green: sig, sweep_dir = 'low_rev', 'low'

        roc = np.nan
        if bar >= roc_lb and c[bar - roc_lb] > 0:
            roc = (c[bar] - c[bar - roc_lb]) / c[bar - roc_lb] * 100
            if sweep_dir == 'low': roc = -roc

        pre_cvd_val = np.nan
        if bar >= cvd_lb:
            pre_cvd_val = cvd[bar] - cvd[bar - cvd_lb]
            if sweep_dir == 'low': pre_cvd_val = -pre_cvd_val

        early_p = np.nan
        entry_bar = bar + early_bars
        if entry_bar < n:
            early_p = (c[entry_bar] - c[bar]) / c[bar] * 100
            if sweep_dir == 'low': early_p = -early_p

        entry_price = c[entry_bar] if entry_bar < n else np.nan
        exit_bar = bar + bp - 1
        exit_price = c[exit_bar] if exit_bar < n else np.nan

        all_events.append({
            'candle_idx': i, 'bar': bar, 'dt': curr['dt'],
            'signal': sig, 'sweep_dir': sweep_dir,
            'roc': roc, 'pre_cvd': pre_cvd_val, 'early_p': early_p,
            'entry_price': entry_price, 'exit_price': exit_price,
        })

    edf = pd.DataFrame(all_events)

    # Walk-forward
    TRAIN = 540; UPDATE = 180
    trades = []
    current_th = None; last_train_end = 0

    for _, ev in edf.iterrows():
        ci = ev['candle_idx']
        if ci >= TRAIN and (current_th is None or ci - last_train_end >= UPDATE):
            td = edf[(edf['candle_idx'] >= ci-TRAIN) & (edf['candle_idx'] < ci)]
            tv = td[td['signal'].notna()].dropna(subset=['roc','pre_cvd','early_p'])
            if len(tv) >= 50:
                current_th = {
                    'roc_33': np.percentile(tv['roc'],33.3), 'roc_67': np.percentile(tv['roc'],66.7),
                    'cvd_33': np.percentile(tv['pre_cvd'],33.3), 'cvd_67': np.percentile(tv['pre_cvd'],66.7),
                    'early_33': np.percentile(tv['early_p'],33.3), 'early_67': np.percentile(tv['early_p'],66.7),
                }
                last_train_end = ci
        if current_th is None or ev['signal'] is None: continue
        if pd.isna(ev['roc']) or pd.isna(ev['pre_cvd']) or pd.isna(ev['early_p']): continue
        if pd.isna(ev['entry_price']) or pd.isna(ev['exit_price']): continue

        rt = tercile_from_th(ev['roc'], current_th['roc_33'], current_th['roc_67'])
        ct = tercile_from_th(ev['pre_cvd'], current_th['cvd_33'], current_th['cvd_67'])
        et = tercile_from_th(ev['early_p'], current_th['early_33'], current_th['early_67'])

        if (rt, ct, et) not in CONT_CELLS: continue

        direction = "LONG" if ev['sweep_dir'] == "high" else "SHORT"
        if direction == "LONG":
            pnl = (ev['exit_price'] - ev['entry_price']) / ev['entry_price'] * 100
        else:
            pnl = (ev['entry_price'] - ev['exit_price']) / ev['entry_price'] * 100

        trades.append({
            'dt': ev['dt'], 'direction': direction,
            'pnl': pnl - COMMISSION, 'pnl_gross': pnl,
            'tf': tf_name,
            'year': pd.Timestamp(ev['dt']).year,
        })

    return pd.DataFrame(trades), roc_time, early_time

# ══════════════════════════════════════════════════════════════
# Her TF'yi calistir
# ══════════════════════════════════════════════════════════════

print("MULTI-TF GRID STRATEJI")
print("=" * 120)

timeframes = [
    (12, "1H"),
    (48, "4H"),
    (96, "8H"),
]

all_trades = []
tf_results = {}

for bp, tf_name in timeframes:
    tdf, roc_time, early_time = run_tf_backtest(bp, tf_name)
    tf_results[tf_name] = tdf
    all_trades.append(tdf)

    if len(tdf) == 0:
        print(f"\n  {tf_name}: Trade yok!")
        continue

    equity = [100.0]
    for p in tdf['pnl']:
        equity.append(equity[-1] * (1 + p / 100))
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100

    tdf_m = tdf.copy()
    tdf_m['month'] = pd.to_datetime(tdf_m['dt']).dt.to_period('M')
    monthly = tdf_m.groupby('month')['pnl'].sum()

    tdf_w = tdf.copy()
    tdf_w['week'] = pd.to_datetime(tdf_w['dt']).dt.to_period('W')
    weekly = tdf_w.groupby('week')['pnl'].sum()

    print(f"\n  {tf_name} (ROC {roc_time}, early {early_time}):")
    print(f"    Trade: {len(tdf)} | WR: {(tdf['pnl']>0).mean()*100:.1f}% | Avg: {tdf['pnl'].mean():.4f}% | Sum: {tdf['pnl'].sum():.1f}%")
    print(f"    $100 → ${eq[-1]:.2f} | Max DD: {dd.max():.1f}%")
    print(f"    Pozitif hafta: {(weekly>0).sum()}/{len(weekly)} ({(weekly>0).mean()*100:.0f}%) | En kotu: {weekly.min():.2f}%")
    print(f"    Pozitif ay: {(monthly>0).sum()}/{len(monthly)} ({(monthly>0).mean()*100:.0f}%) | En kotu: {monthly.min():.2f}%")

    print(f"    Yillik:")
    for year, grp in tdf.groupby('year'):
        print(f"      {year}: {len(grp)} trade | WR {(grp['pnl']>0).mean()*100:.1f}% | {grp['pnl'].sum():+.1f}%")

# ══════════════════════════════════════════════════════════════
# Birlestir
# ══════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print("BIRLESIK SONUC (1H + 4H + 8H)")
print("=" * 120)

combined = pd.concat(all_trades, ignore_index=True)
combined = combined.sort_values('dt').reset_index(drop=True)

if len(combined) > 0:
    equity = [100.0]
    for p in combined['pnl']:
        equity.append(equity[-1] * (1 + p / 100))
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100

    print(f"  Toplam trade: {len(combined)}")
    print(f"  TF dagilimi: 1H={len(tf_results.get('1H',pd.DataFrame()))} | 4H={len(tf_results.get('4H',pd.DataFrame()))} | 8H={len(tf_results.get('8H',pd.DataFrame()))}")
    print(f"  WR: {(combined['pnl']>0).mean()*100:.1f}%")
    print(f"  Avg PnL: {combined['pnl'].mean():.4f}%")
    print(f"  Toplam PnL: {combined['pnl'].sum():.1f}%")
    print(f"  $100 → ${eq[-1]:.2f}")
    print(f"  Max DD: {dd.max():.1f}%")

    combined_m = combined.copy()
    combined_m['month'] = pd.to_datetime(combined_m['dt']).dt.to_period('M')
    monthly = combined_m.groupby('month')['pnl'].sum()
    print(f"  Pozitif ay: {(monthly>0).sum()}/{len(monthly)} ({(monthly>0).mean()*100:.0f}%) | En kotu: {monthly.min():.2f}%")

    combined_w = combined.copy()
    combined_w['week'] = pd.to_datetime(combined_w['dt']).dt.to_period('W')
    weekly = combined_w.groupby('week')['pnl'].sum()
    print(f"  Pozitif hafta: {(weekly>0).sum()}/{len(weekly)} ({(weekly>0).mean()*100:.0f}%) | En kotu: {weekly.min():.2f}%")

    print(f"\n  Yillik:")
    for year, grp in combined.groupby('year'):
        eq_y = [100.0]
        for p in grp['pnl']:
            eq_y.append(eq_y[-1] * (1 + p / 100))
        print(f"    {year}: {len(grp)} trade | WR {(grp['pnl']>0).mean()*100:.1f}% | {grp['pnl'].sum():+.1f}% | $100→${eq_y[-1]:.2f}")

    # Gunluk trade sayisi
    combined_d = combined.copy()
    combined_d['date'] = pd.to_datetime(combined_d['dt']).dt.date
    daily_count = combined_d.groupby('date').size()
    print(f"\n  Gunluk trade: ort {daily_count.mean():.1f} | max {daily_count.max()}")

print()
