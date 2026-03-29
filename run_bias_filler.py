"""Bias Bosluk Doldurucu — KAMA vs PMAX
Sweep bias aktifken sweep sinyali kullan.
Sweep bias yokken KAMA veya PMAX yonunu kullan.
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

# ══════════════════════════════════════════════════════════════
# KAMA hesapla
# ══════════════════════════════════════════════════════════════

def compute_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]

    fast_c = 2.0 / (fast_sc + 1.0)
    slow_c = 2.0 / (slow_sc + 1.0)

    for i in range(er_period + 1, n):
        direction = abs(close[i] - close[i - er_period])
        volatility = sum(abs(close[j] - close[j-1]) for j in range(i - er_period + 1, i + 1))

        if volatility == 0:
            er = 0
        else:
            er = direction / volatility

        sc = (er * (fast_c - slow_c) + slow_c) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])

    return kama

# ══════════════════════════════════════════════════════════════
# PMAX hesapla
# ══════════════════════════════════════════════════════════════

def compute_pmax(close, high, low, atr_period=10, atr_mult=3.0, ma_period=10):
    """PMAX: EMA + SuperTrend benzeri"""
    n = len(close)

    # EMA
    ema = np.full(n, np.nan)
    k = 2.0 / (ma_period + 1.0)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i-1] * (1 - k)

    # ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr = np.full(n, np.nan)
    atr[atr_period] = np.mean(tr[1:atr_period+1])
    for i in range(atr_period + 1, n):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period

    # PMAX
    pmax_val = np.full(n, np.nan)
    direction = np.ones(n)  # 1 = long, -1 = short

    start = atr_period + 1
    if start >= n: return direction

    upper = ema[start] + atr_mult * atr[start]
    lower = ema[start] - atr_mult * atr[start]
    pmax_val[start] = upper
    direction[start] = 1

    for i in range(start + 1, n):
        if np.isnan(ema[i]) or np.isnan(atr[i]):
            direction[i] = direction[i-1]
            continue

        new_upper = ema[i] + atr_mult * atr[i]
        new_lower = ema[i] - atr_mult * atr[i]

        if direction[i-1] == 1:
            lower = max(new_lower, lower) if ema[i] > lower else new_lower
            if ema[i] < lower:
                direction[i] = -1
                upper = new_upper
            else:
                direction[i] = 1
        else:
            upper = min(new_upper, upper) if ema[i] < upper else new_upper
            if ema[i] > upper:
                direction[i] = 1
                lower = new_lower
            else:
                direction[i] = -1

    return direction

# ══════════════════════════════════════════════════════════════
# Sweep bias (onceki koddan)
# ══════════════════════════════════════════════════════════════

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

def compute_tf_signals(bp):
    roc_lb = max(6, int(bp * 0.75))
    cvd_lb = bp
    early_bars = max(3, bp // 4)
    n_candles = n // bp
    candles = []
    for i in range(n_candles):
        s = i * bp; e = s + bp
        candles.append({'start': s, 'open': c[s], 'high': np.max(h[s:e]), 'low': np.min(l[s:e]), 'close': c[e-1]})

    TRAIN = 540; UPDATE = 180
    current_th = None; last_train_end = 0
    bar_signals = np.zeros(n)
    all_events = []

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
        all_events.append({'candle_idx': i, 'bar': bar, 'signal': sig, 'sweep_dir': sweep_dir})
        if i >= TRAIN and (current_th is None or i - last_train_end >= UPDATE):
            td_events = [e for e in all_events if e['candle_idx'] >= i-TRAIN and e['candle_idx'] < i and e['signal'] is not None]
            if len(td_events) >= 50:
                rocs, cvds, earlies = [], [], []
                for e in td_events:
                    b = e['bar']
                    if b >= roc_lb and c[b-roc_lb]>0:
                        r = (c[b]-c[b-roc_lb])/c[b-roc_lb]*100
                        if e['sweep_dir']=='low': r=-r
                        rocs.append(r)
                    if b >= cvd_lb:
                        cv = cvd[b]-cvd[b-cvd_lb]
                        if e['sweep_dir']=='low': cv=-cv
                        cvds.append(cv)
                    eb = b+early_bars
                    if eb < n:
                        ep = (c[eb]-c[b])/c[b]*100
                        if e['sweep_dir']=='low': ep=-ep
                        earlies.append(ep)
                if len(rocs)>=30 and len(cvds)>=30 and len(earlies)>=30:
                    current_th = {'roc_33':np.percentile(rocs,33.3),'roc_67':np.percentile(rocs,66.7),'cvd_33':np.percentile(cvds,33.3),'cvd_67':np.percentile(cvds,66.7),'early_33':np.percentile(earlies,33.3),'early_67':np.percentile(earlies,66.7)}
                    last_train_end = i
        if current_th is None or sig is None or sweep_dir is None: continue
        roc = np.nan
        if bar >= roc_lb and c[bar-roc_lb]>0:
            roc = (c[bar]-c[bar-roc_lb])/c[bar-roc_lb]*100
            if sweep_dir=='low': roc=-roc
        pre_cvd_val = np.nan
        if bar >= cvd_lb:
            pre_cvd_val = cvd[bar]-cvd[bar-cvd_lb]
            if sweep_dir=='low': pre_cvd_val=-pre_cvd_val
        early_p = np.nan
        entry_bar = bar+early_bars
        if entry_bar < n:
            early_p = (c[entry_bar]-c[bar])/c[bar]*100
            if sweep_dir=='low': early_p=-early_p
        if pd.isna(roc) or pd.isna(pre_cvd_val) or pd.isna(early_p): continue
        rt = tercile_from_th(roc, current_th['roc_33'], current_th['roc_67'])
        ct = tercile_from_th(pre_cvd_val, current_th['cvd_33'], current_th['cvd_67'])
        et = tercile_from_th(early_p, current_th['early_33'], current_th['early_67'])
        if (rt,ct,et) in CONT_CELLS:
            direction = 1 if sweep_dir == "high" else -1
            signal_start = bar + early_bars
            signal_end = bar + bp - 1
            for b in range(signal_start, min(signal_end+1, n)):
                bar_signals[b] = direction
    return bar_signals

print("BIAS BOSLUK DOLDURUCU — KAMA vs PMAX")
print("=" * 120)

print("\nSweep sinyalleri hesaplaniyor...")
sig_1h = compute_tf_signals(12); print("  1H")
sig_4h = compute_tf_signals(48); print("  4H")
sig_8h = compute_tf_signals(96); print("  8H")

sweep_bias = np.array([sig_1h[i] + sig_4h[i] + sig_8h[i] for i in range(n)])
has_sweep = sweep_bias != 0

print(f"\nSweep bias aktif: {has_sweep.sum()/n*100:.1f}% | Bos: {(~has_sweep).sum()/n*100:.1f}%")

# KAMA ve PMAX farkli parametrelerle
print("\nKAMA ve PMAX hesaplaniyor...")

configs = [
    ("KAMA 10", "kama", compute_kama(c, 10, 2, 30)),
    ("KAMA 20", "kama", compute_kama(c, 20, 2, 30)),
    ("KAMA 50", "kama", compute_kama(c, 50, 2, 30)),
    ("PMAX 10/3", "pmax", compute_pmax(c, h, l, 10, 3.0, 10)),
    ("PMAX 10/2", "pmax", compute_pmax(c, h, l, 10, 2.0, 10)),
    ("PMAX 14/3", "pmax", compute_pmax(c, h, l, 14, 3.0, 14)),
    ("PMAX 20/3", "pmax", compute_pmax(c, h, l, 20, 3.0, 20)),
]

horizons = [(12, "1h"), (48, "4h"), (96, "8h"), (288, "1d")]

print("\n" + "=" * 120)
print("1) KAMA / PMAX TEK BASINA (sweep olmadan)")
print("=" * 120)

for name, typ, indicator in configs:
    if typ == "kama":
        # KAMA: fiyat > KAMA → LONG, fiyat < KAMA → SHORT
        direction = np.zeros(n)
        for i in range(n):
            if not np.isnan(indicator[i]):
                direction[i] = 1 if c[i] > indicator[i] else -1
    else:
        direction = indicator  # PMAX zaten +1/-1

    long_mask = direction > 0
    short_mask = direction < 0

    long_n = long_mask.sum()
    short_n = short_mask.sum()

    results = []
    for hz, _ in horizons:
        # LONG
        l_rets = [(c[i+hz]-c[i])/c[i]*100 for i in np.where(long_mask)[0] if i+hz < n]
        s_rets = [(c[i]-c[i+hz])/c[i]*100 for i in np.where(short_mask)[0] if i+hz < n]
        all_rets = l_rets + s_rets
        wr = np.mean([1 if r>0 else 0 for r in all_rets])*100 if all_rets else 0
        avg = np.mean(all_rets) if all_rets else 0
        results.append((wr, avg))

    print(f"  {name:>12s} | L:{long_n/n*100:.0f}% S:{short_n/n*100:.0f}% | 1h:{results[0][0]:.1f}% 4h:{results[1][0]:.1f}% 8h:{results[2][0]:.1f}% 1d:{results[3][0]:.1f}% | 4h avg:{results[1][1]:+.3f}%")

print("\n" + "=" * 120)
print("2) SWEEP BIAS + DOLDURUCU (sweep varken sweep, yokken KAMA/PMAX)")
print("=" * 120)

for name, typ, indicator in configs:
    if typ == "kama":
        filler_dir = np.zeros(n)
        for i in range(n):
            if not np.isnan(indicator[i]):
                filler_dir[i] = 1 if c[i] > indicator[i] else -1
    else:
        filler_dir = indicator

    # Birlestir: sweep aktifse sweep, degilse filler
    combined = np.zeros(n)
    for i in range(n):
        if sweep_bias[i] != 0:
            combined[i] = 1 if sweep_bias[i] > 0 else -1
        else:
            combined[i] = filler_dir[i]

    long_mask = combined > 0
    short_mask = combined < 0

    results = []
    for hz, _ in horizons:
        l_rets = [(c[i+hz]-c[i])/c[i]*100 for i in np.where(long_mask)[0] if i+hz < n]
        s_rets = [(c[i]-c[i+hz])/c[i]*100 for i in np.where(short_mask)[0] if i+hz < n]
        all_rets = l_rets + s_rets
        wr = np.mean([1 if r>0 else 0 for r in all_rets])*100 if all_rets else 0
        avg = np.mean(all_rets) if all_rets else 0
        results.append((wr, avg))

    # Sweep aktif olan bolumun WR'si
    sweep_rets = []
    for i in range(n):
        if sweep_bias[i] != 0 and i + 48 < n:
            ret = (c[i+48]-c[i])/c[i]*100
            if sweep_bias[i] < 0: ret = -ret
            sweep_rets.append(ret)
    sweep_wr = np.mean([1 if r>0 else 0 for r in sweep_rets])*100 if sweep_rets else 0

    # Filler olan bolumun WR'si
    filler_rets = []
    for i in np.where(~has_sweep)[0]:
        if i + 48 < n:
            ret = (c[i+48]-c[i])/c[i]*100
            if filler_dir[i] < 0: ret = -ret
            filler_rets.append(ret)
    filler_wr = np.mean([1 if r>0 else 0 for r in filler_rets])*100 if filler_rets else 0

    print(f"  Sweep+{name:>10s} | 1h:{results[0][0]:.1f}% 4h:{results[1][0]:.1f}% 8h:{results[2][0]:.1f}% 1d:{results[3][0]:.1f}% | 4h avg:{results[1][1]:+.3f}% | Sweep:{sweep_wr:.1f}% Fill:{filler_wr:.1f}%")

print("\n" + "=" * 120)
print("3) REFERANS — SADECE SWEEP BIAS (bos yerlerde sinyal yok)")
print("=" * 120)

results = []
for hz, _ in horizons:
    rets = []
    for i in range(n):
        if sweep_bias[i] != 0 and i + hz < n:
            ret = (c[i+hz]-c[i])/c[i]*100
            if sweep_bias[i] < 0: ret = -ret
            rets.append(ret)
    wr = np.mean([1 if r>0 else 0 for r in rets])*100
    avg = np.mean(rets)
    results.append((wr, avg))

print(f"  Sadece sweep | 1h:{results[0][0]:.1f}% 4h:{results[1][0]:.1f}% 8h:{results[2][0]:.1f}% 1d:{results[3][0]:.1f}% | 4h avg:{results[1][1]:+.3f}% | Aktif:{has_sweep.sum()/n*100:.1f}%")

print()
