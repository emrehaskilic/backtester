"""State-Based Sweep Strategy
Pre-CVD (4h oncesi) + Ilk 1h fiyat → CONT/REV/BELIRSIZ
Belirsiz = onceki state'i koru
4H ETH Perp, 5 yillik veri
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

BP = 48  # 4H
n_candles = n // BP
PRE_LB = 48  # 4h oncesi CVD lookback
EARLY_BARS = 12  # ilk 1h = 12 x 5m bar

print("STATE-BASED SWEEP STRATEGY")
print("=" * 120)
print(f"Data: {n:,} bar (5m) | {df['dt'].iloc[0].strftime('%Y-%m-%d')} - {df['dt'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"4H candle: {n_candles:,} | Pre-CVD lookback: {PRE_LB} bar (4h) | Early signal: {EARLY_BARS} bar (1h)")
print()

# ══════════════════════════════════════════════════════════════
# 1) Mumlari olustur
# ══════════════════════════════════════════════════════════════
candles = []
for i in range(n_candles):
    s = i * BP
    e = s + BP
    candles.append({
        'start': s, 'open': c[s],
        'high': np.max(h[s:e]), 'low': np.min(l[s:e]),
        'close': c[e-1],
        'dt': df['dt'].iloc[s],
    })

# ══════════════════════════════════════════════════════════════
# 2) Walk-forward: quantile threshold'larini train'de hesapla
# ══════════════════════════════════════════════════════════════

# Once tum eventleri topla (threshold hesabi icin)
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
        sweep_dir = "high"
        if curr['close'] > prev['high']: outcome = "cont"
        elif is_red: outcome = "rev"
        else: outcome = "ambig"
    elif swept_low and not swept_high:
        sweep_dir = "low"
        if curr['close'] < prev['low']: outcome = "cont"
        elif is_green: outcome = "rev"
        else: outcome = "ambig"
    elif swept_high and swept_low:
        if curr['close'] > prev['high']: sweep_dir, outcome = "high", "cont"
        elif curr['close'] < prev['low']: sweep_dir, outcome = "low", "cont"
        elif is_red: sweep_dir, outcome = "high", "rev"
        elif is_green: sweep_dir, outcome = "low", "rev"
        else: sweep_dir, outcome = None, None
    else:
        sweep_dir, outcome = None, None

    # Pre-CVD
    if bar >= PRE_LB:
        pre_cvd = cvd[bar] - cvd[bar - PRE_LB]
        if sweep_dir == "low":
            pre_cvd = -pre_cvd  # align: pozitif = sweep yonunde
    else:
        pre_cvd = np.nan

    # Early price (ilk 1h)
    early_end = bar + EARLY_BARS
    if early_end < n:
        early_price = (c[early_end] - c[bar]) / c[bar] * 100
        if sweep_dir == "low":
            early_price = -early_price  # align
    else:
        early_price = np.nan

    # Entry price = ilk 1h sonundaki fiyat
    entry_price = c[early_end] if early_end < n else np.nan

    all_events.append({
        'candle_idx': i, 'bar': bar, 'dt': curr['dt'],
        'sweep_dir': sweep_dir, 'outcome': outcome,
        'pre_cvd': pre_cvd, 'early_price': early_price,
        'entry_price': entry_price,
        'candle_close': curr['close'],
    })

edf = pd.DataFrame(all_events)

# ══════════════════════════════════════════════════════════════
# 3) Walk-forward backtest
# ══════════════════════════════════════════════════════════════

# Window: 3 ay train (~540 4H mum), 1 ay test (~180 4H mum)
TRAIN_CANDLES = 540  # ~3 ay
TEST_CANDLES = 180   # ~1 ay

# Basit non-WF: tum veri uzerinde sabit quantile
# WF: rolling quantile
results_simple = []
results_wf = []

# --- A) Basit backtest (tum veri, sabit threshold) ---
sweep_events = edf[edf['sweep_dir'].notna() & edf['outcome'].notna()].copy()
sweep_events = sweep_events.dropna(subset=['pre_cvd', 'early_price', 'entry_price'])

# Quantile thresholds (5-quantile)
pre_thresholds = np.percentile(sweep_events['pre_cvd'], [20, 40, 60, 80])
early_thresholds = np.percentile(sweep_events['early_price'], [20, 40, 60, 80])

def classify_signal(pre_cvd, early_price, pre_th, early_th):
    """Pre-CVD quantile x Early price quantile → CONT/REV/AMBIG"""
    # Pre quantile (1-5)
    pre_q = 1
    for t in pre_th:
        if pre_cvd > t: pre_q += 1

    # Early quantile (1-5)
    early_q = 1
    for t in early_th:
        if early_price > t: early_q += 1

    # Cont zone: Pre Q1-Q2 x Early Q4-Q5, veya herhangi Pre x Early Q5
    if (pre_q <= 2 and early_q >= 4) or (early_q == 5 and pre_q <= 4):
        return "CONT"
    # Rev zone: Pre Q3-Q5 x Early Q1, veya Pre Q5 x Early Q1-Q2
    elif (pre_q >= 3 and early_q == 1) or (pre_q == 5 and early_q <= 2):
        return "REV"
    else:
        return "AMBIG"

# Sinyal uret
for idx, row in sweep_events.iterrows():
    sig = classify_signal(row['pre_cvd'], row['early_price'], pre_thresholds, early_thresholds)
    sweep_events.loc[idx, 'signal'] = sig

# --- B) State machine + backtest ---
def run_backtest(events_df, label=""):
    """State-based backtest: CONT/REV sinyalleri, AMBIG = onceki state"""
    trades = []
    current_state = None  # "LONG" or "SHORT" or None
    entry_price = None
    entry_time = None
    entry_reason = None

    # Tum mumlari sirayla isle
    sorted_events = events_df.sort_values('candle_idx').reset_index(drop=True)

    for _, row in sorted_events.iterrows():
        sig = row.get('signal', 'AMBIG')
        sweep_dir = row['sweep_dir']

        # Yonu belirle
        if sig == "CONT":
            new_state = "LONG" if sweep_dir == "high" else "SHORT"
        elif sig == "REV":
            new_state = "SHORT" if sweep_dir == "high" else "LONG"
        else:
            new_state = current_state  # AMBIG = onceki state

        # State degisti mi?
        if new_state != current_state and new_state is not None:
            # Onceki pozisyonu kapat
            if current_state is not None and entry_price is not None:
                exit_price = row['entry_price']
                if current_state == "LONG":
                    pnl = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - exit_price) / entry_price * 100

                trades.append({
                    'entry_time': entry_time, 'exit_time': row['dt'],
                    'direction': current_state, 'reason': entry_reason,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'duration_candles': row['candle_idx'] - trades[-1]['entry_candle'] if trades else 0,
                    'entry_candle': 0,
                })

            # Yeni pozisyon ac
            current_state = new_state
            entry_price = row['entry_price']
            entry_time = row['dt']
            entry_reason = f"{sig}_{sweep_dir}"

            if trades:
                pass
            trades.append({
                'entry_time': entry_time, 'exit_time': None,
                'direction': current_state, 'reason': entry_reason,
                'entry_price': entry_price, 'exit_price': None,
                'pnl_pct': None, 'entry_candle': row['candle_idx'],
            })

    # Son trades'i temizle (acik pozisyon)
    open_trades = [t for t in trades if t['pnl_pct'] is None]
    closed_trades = [t for t in trades if t['pnl_pct'] is not None]

    return closed_trades

# Daha temiz backtest
def run_backtest_v2(events_df):
    sorted_ev = events_df.sort_values('candle_idx').reset_index(drop=True)

    positions = []  # (direction, entry_price, entry_time, entry_candle_idx)
    closed = []
    current_dir = None
    entry_p = None
    entry_t = None
    entry_ci = None

    for _, row in sorted_ev.iterrows():
        sig = row.get('signal', 'AMBIG')
        sweep_dir = row['sweep_dir']

        if sig == "CONT":
            new_dir = "LONG" if sweep_dir == "high" else "SHORT"
        elif sig == "REV":
            new_dir = "SHORT" if sweep_dir == "high" else "LONG"
        else:
            continue  # AMBIG = state degistirme

        if new_dir == current_dir:
            continue  # ayni yon, degisiklik yok

        # Kapat + ac
        if current_dir is not None and entry_p is not None and not np.isnan(entry_p):
            exit_p = row['entry_price']
            if not np.isnan(exit_p):
                if current_dir == "LONG":
                    pnl = (exit_p - entry_p) / entry_p * 100
                else:
                    pnl = (entry_p - exit_p) / entry_p * 100
                closed.append({
                    'entry_time': entry_t, 'exit_time': row['dt'],
                    'dir': current_dir, 'entry': entry_p, 'exit': exit_p,
                    'pnl': pnl,
                    'dur': row['candle_idx'] - entry_ci,
                })

        current_dir = new_dir
        entry_p = row['entry_price']
        entry_t = row['dt']
        entry_ci = row['candle_idx']

    return closed

# ══════════════════════════════════════════════════════════════
# 4) Calistir
# ══════════════════════════════════════════════════════════════

# A) Sabit threshold backtest
trades = run_backtest_v2(sweep_events)
tdf = pd.DataFrame(trades)

if len(tdf) > 0:
    print("BACKTEST SONUCLARI — SABIT THRESHOLD (tum veri)")
    print("=" * 120)
    print(f"Toplam trade: {len(tdf)}")
    print(f"Win rate: {(tdf['pnl'] > 0).mean() * 100:.1f}%")
    print(f"Ortalama PnL: {tdf['pnl'].mean():.3f}%")
    print(f"Median PnL: {tdf['pnl'].median():.3f}%")
    print(f"Toplam PnL (kumulatif): {tdf['pnl'].sum():.1f}%")
    print(f"Ortalama trade suresi: {tdf['dur'].mean():.1f} mum ({tdf['dur'].mean()*4:.0f}h)")
    print(f"Max winning trade: {tdf['pnl'].max():.2f}%")
    print(f"Max losing trade: {tdf['pnl'].min():.2f}%")
    print()

    # Equity curve
    equity = [100.0]
    for pnl in tdf['pnl']:
        equity.append(equity[-1] * (1 + pnl / 100))
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100
    max_dd = dd.max()

    print(f"Baslangic: $100 → Son: ${equity[-1]:.2f}")
    print(f"Max drawdown: {max_dd:.1f}%")
    print()

    # LONG vs SHORT
    longs = tdf[tdf['dir'] == 'LONG']
    shorts = tdf[tdf['dir'] == 'SHORT']
    print(f"LONG trades:  {len(longs)} | WR: {(longs['pnl']>0).mean()*100:.1f}% | Avg: {longs['pnl'].mean():.3f}% | Sum: {longs['pnl'].sum():.1f}%")
    print(f"SHORT trades: {len(shorts)} | WR: {(shorts['pnl']>0).mean()*100:.1f}% | Avg: {shorts['pnl'].mean():.3f}% | Sum: {shorts['pnl'].sum():.1f}%")
    print()

    # Yillik performans
    tdf['year'] = pd.to_datetime(tdf['entry_time']).dt.year
    print("YILLIK PERFORMANS:")
    print(f"  {'Yil':>6s} | {'Trade':>6s} | {'WR':>6s} | {'Avg PnL':>8s} | {'Toplam':>8s}")
    print(f"  {'-'*45}")
    for year, grp in tdf.groupby('year'):
        print(f"  {year:>6d} | {len(grp):>6d} | {(grp['pnl']>0).mean()*100:>5.1f}% | {grp['pnl'].mean():>+7.3f}% | {grp['pnl'].sum():>+7.1f}%")
    print()

    # Aylik performans
    tdf['month'] = pd.to_datetime(tdf['entry_time']).dt.to_period('M')
    monthly = tdf.groupby('month')['pnl'].sum()
    print(f"AYLIK PERFORMANS:")
    print(f"  Pozitif aylar: {(monthly > 0).sum()} / {len(monthly)} ({(monthly > 0).mean()*100:.0f}%)")
    print(f"  Ortalama ay: {monthly.mean():.2f}%")
    print(f"  En iyi ay: {monthly.max():.2f}%")
    print(f"  En kotu ay: {monthly.min():.2f}%")
    print()

# ══════════════════════════════════════════════════════════════
# B) Walk-Forward backtest
# ══════════════════════════════════════════════════════════════

print("=" * 120)
print("WALK-FORWARD BACKTEST")
print(f"Train: {TRAIN_CANDLES} mum (~3 ay) | Test: {TEST_CANDLES} mum (~1 ay)")
print("=" * 120)

wf_trades = []
window_results = []

start = 0
window_num = 0
while start + TRAIN_CANDLES + TEST_CANDLES <= len(edf):
    train = edf.iloc[start:start+TRAIN_CANDLES]
    test = edf.iloc[start+TRAIN_CANDLES:start+TRAIN_CANDLES+TEST_CANDLES]

    # Train: sweep event'leri ve threshold hesapla
    train_sweep = train[train['sweep_dir'].notna() & train['outcome'].notna()].dropna(subset=['pre_cvd', 'early_price'])

    if len(train_sweep) < 50:
        start += TEST_CANDLES
        continue

    # Train threshold'lari
    pre_th = np.percentile(train_sweep['pre_cvd'], [20, 40, 60, 80])
    early_th = np.percentile(train_sweep['early_price'], [20, 40, 60, 80])

    # Test: sinyal uret
    test_sweep = test[test['sweep_dir'].notna() & test['outcome'].notna()].dropna(subset=['pre_cvd', 'early_price', 'entry_price']).copy()

    for idx, row in test_sweep.iterrows():
        sig = classify_signal(row['pre_cvd'], row['early_price'], pre_th, early_th)
        test_sweep.loc[idx, 'signal'] = sig

    # Backtest
    window_trades = run_backtest_v2(test_sweep)

    if window_trades:
        wt = pd.DataFrame(window_trades)
        total_pnl = wt['pnl'].sum()
        wr = (wt['pnl'] > 0).mean() * 100
        train_start_dt = train['dt'].iloc[0] if 'dt' in train.columns and len(train) > 0 else "?"
        test_start_dt = test['dt'].iloc[0] if 'dt' in test.columns and len(test) > 0 else "?"

        window_results.append({
            'window': window_num, 'trades': len(wt),
            'wr': wr, 'total_pnl': total_pnl,
            'test_start': test_start_dt,
        })
        wf_trades.extend(window_trades)

    window_num += 1
    start += TEST_CANDLES

if window_results:
    wrdf = pd.DataFrame(window_results)
    print(f"\nToplam pencere: {len(wrdf)}")
    print(f"Toplam trade: {sum(w['trades'] for w in window_results)}")
    print(f"Pozitif pencere: {(wrdf['total_pnl'] > 0).sum()} / {len(wrdf)} ({(wrdf['total_pnl'] > 0).mean()*100:.0f}%)")
    print(f"Ortalama pencere PnL: {wrdf['total_pnl'].mean():.2f}%")
    print()

    print(f"{'Pencere':>8s} | {'Test baslangic':>20s} | {'Trade':>6s} | {'WR':>6s} | {'PnL':>8s}")
    print(f"{'-'*55}")
    for _, w in wrdf.iterrows():
        dt_str = str(w['test_start'])[:10] if w['test_start'] != "?" else "?"
        print(f"{w['window']:>8d} | {dt_str:>20s} | {w['trades']:>6d} | {w['wr']:>5.1f}% | {w['total_pnl']:>+7.2f}%")

    # WF equity curve
    if wf_trades:
        wf_tdf = pd.DataFrame(wf_trades)
        wf_equity = [100.0]
        for pnl in wf_tdf['pnl']:
            wf_equity.append(wf_equity[-1] * (1 + pnl / 100))
        wf_equity = np.array(wf_equity)
        wf_peak = np.maximum.accumulate(wf_equity)
        wf_dd = (wf_peak - wf_equity) / wf_peak * 100
        wf_max_dd = wf_dd.max()

        print(f"\nWF SONUC:")
        print(f"  Toplam trade: {len(wf_tdf)}")
        print(f"  Win rate: {(wf_tdf['pnl']>0).mean()*100:.1f}%")
        print(f"  Ortalama PnL: {wf_tdf['pnl'].mean():.3f}%")
        print(f"  Toplam PnL: {wf_tdf['pnl'].sum():.1f}%")
        print(f"  $100 → ${wf_equity[-1]:.2f}")
        print(f"  Max drawdown: {wf_max_dd:.1f}%")
        print(f"  WR LONG: {(wf_tdf[wf_tdf['dir']=='LONG']['pnl']>0).mean()*100:.1f}% | SHORT: {(wf_tdf[wf_tdf['dir']=='SHORT']['pnl']>0).mean()*100:.1f}%")
