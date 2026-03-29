"""Bias Gostericisi — Son sinyali tasi
Bias 0 olunca onceki bias degerini koru.
Farkli tasima yontemleri test et.
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
    bar_signals = [0] * n  # +1=LONG, -1=SHORT, 0=yok

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

        all_events.append({'candle_idx': i, 'bar': bar, 'signal': sig, 'sweep_dir': sweep_dir})

        if i >= TRAIN and (current_th is None or i - last_train_end >= UPDATE):
            td_events = [e for e in all_events if e['candle_idx'] >= i-TRAIN and e['candle_idx'] < i and e['signal'] is not None]
            if len(td_events) >= 50:
                rocs, cvds, earlies = [], [], []
                for e in td_events:
                    b = e['bar']
                    if b >= roc_lb and c[b-roc_lb] > 0:
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
                    current_th = {
                        'roc_33': np.percentile(rocs,33.3), 'roc_67': np.percentile(rocs,66.7),
                        'cvd_33': np.percentile(cvds,33.3), 'cvd_67': np.percentile(cvds,66.7),
                        'early_33': np.percentile(earlies,33.3), 'early_67': np.percentile(earlies,66.7),
                    }
                    last_train_end = i

        if current_th is None or sig is None or sweep_dir is None: continue

        roc = np.nan
        if bar >= roc_lb and c[bar-roc_lb] > 0:
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

        if (rt, ct, et) in CONT_CELLS:
            direction = 1 if sweep_dir == "high" else -1
            signal_start = bar + early_bars
            signal_end = bar + bp - 1
            for b in range(signal_start, min(signal_end+1, n)):
                bar_signals[b] = direction

    return bar_signals

print("BIAS GOSTERICISI — TASIMA YONTEMLERI")
print("=" * 120)
print()

print("Sinyaller hesaplaniyor...")
sig_1h = compute_tf_signals(12)
print("  1H tamam")
sig_4h = compute_tf_signals(48)
print("  4H tamam")
sig_8h = compute_tf_signals(96)
print("  8H tamam")

# Ham bias (tasimasiz)
raw_bias = np.array([sig_1h[i] + sig_4h[i] + sig_8h[i] for i in range(n)], dtype=np.float64)

# ══════════════════════════════════════════════════════════════
# Tasima yontemleri
# ══════════════════════════════════════════════════════════════

def carry_last(raw):
    """Bias 0 olunca son sinyali tasi"""
    carried = np.zeros(n)
    last = 0
    for i in range(n):
        if raw[i] != 0:
            last = raw[i]
        carried[i] = last
    return carried

def carry_decay(raw, half_life_bars):
    """Bias 0 olunca son sinyali tasi ama zamanla zayiflat"""
    carried = np.zeros(n)
    last_val = 0
    last_time = 0
    for i in range(n):
        if raw[i] != 0:
            last_val = raw[i]
            last_time = i
            carried[i] = raw[i]
        else:
            age = i - last_time
            decay = 0.5 ** (age / half_life_bars)
            carried[i] = last_val * decay
    return carried

def carry_with_sign(raw):
    """Sadece yon tasi (+1 veya -1), buyuklugu koru"""
    carried = np.zeros(n)
    last_sign = 0
    for i in range(n):
        if raw[i] != 0:
            last_sign = 1 if raw[i] > 0 else -1
            carried[i] = raw[i]
        else:
            carried[i] = last_sign  # sadece yon
    return carried

methods = [
    ("Ham (tasimasiz)", raw_bias),
    ("Son sinyali tasi", carry_last(raw_bias)),
    ("Sadece yon tasi (±1)", carry_with_sign(raw_bias)),
    ("Decay 1h (half-life=12)", carry_decay(raw_bias, 12)),
    ("Decay 4h (half-life=48)", carry_decay(raw_bias, 48)),
    ("Decay 1d (half-life=288)", carry_decay(raw_bias, 288)),
]

# Her yontem icin test
horizons = [(12, "1h"), (48, "4h"), (96, "8h"), (288, "1d")]

for method_name, bias_arr in methods:
    print(f"\n{'='*120}")
    print(f"  {method_name}")
    print(f"{'='*120}")

    # Bias dagilimi
    if method_name in ["Ham (tasimasiz)", "Son sinyali tasi", "Sadece yon tasi (±1)"]:
        from collections import Counter
        bc = Counter(bias_arr)
        zero_pct = bc.get(0, 0) / n * 100
        pos_pct = sum(v for k, v in bc.items() if k > 0) / n * 100
        neg_pct = sum(v for k, v in bc.items() if k < 0) / n * 100
        print(f"  Bias 0 (notr): {zero_pct:.1f}% | Pozitif: {pos_pct:.1f}% | Negatif: {neg_pct:.1f}%")
    else:
        zero_pct = np.mean(np.abs(bias_arr) < 0.1) * 100
        pos_pct = np.mean(bias_arr > 0.1) * 100
        neg_pct = np.mean(bias_arr < -0.1) * 100
        print(f"  ~Notr (<0.1): {zero_pct:.1f}% | Pozitif: {pos_pct:.1f}% | Negatif: {neg_pct:.1f}%")

    # Bias yonune gore sonraki fiyat
    # Basitlestir: bias > 0 → LONG, bias < 0 → SHORT, bias = 0 → yok
    long_mask = bias_arr > 0
    short_mask = bias_arr < 0
    flat_mask = np.abs(bias_arr) < 0.01

    print(f"\n  {'Yon':>8s} | {'N':>8s} | {'1h WR':>7s} | {'4h WR':>7s} | {'8h WR':>7s} | {'1d WR':>7s} | {'4h avg':>8s}")
    print(f"  {'-'*65}")

    for label, mask in [("LONG", long_mask), ("SHORT", short_mask), ("NOTR", flat_mask)]:
        indices = np.where(mask)[0]
        if len(indices) < 100:
            print(f"  {label:>8s} | {len(indices):>8d} | N/A")
            continue

        cells = []
        for hz, _ in horizons:
            rets = []
            for idx in indices:
                if idx + hz < n:
                    ret = (c[idx + hz] - c[idx]) / c[idx] * 100
                    if label == "SHORT": ret = -ret
                    rets.append(ret)
            wr = np.mean([1 if r > 0 else 0 for r in rets]) * 100 if rets else 0
            avg = np.mean(rets) if rets else 0
            cells.append((wr, avg))

        print(f"  {label:>8s} | {len(indices):>8d} | {cells[0][0]:>6.1f}% | {cells[1][0]:>6.1f}% | {cells[2][0]:>6.1f}% | {cells[3][0]:>6.1f}% | {cells[1][1]:>+7.3f}%")

    # Guclu bias (|bias| >= 2)
    strong_long = bias_arr >= 2 if method_name in ["Ham (tasimasiz)", "Son sinyali tasi", "Sadece yon tasi (±1)"] else bias_arr >= 1.5
    strong_short = bias_arr <= -2 if method_name in ["Ham (tasimasiz)", "Son sinyali tasi", "Sadece yon tasi (±1)"] else bias_arr <= -1.5

    sl_idx = np.where(strong_long)[0]
    ss_idx = np.where(strong_short)[0]

    if len(sl_idx) > 100 and len(ss_idx) > 100:
        print(f"\n  Guclu bias (|b|>=2):")
        for label, indices in [("GUCLU LONG", sl_idx), ("GUCLU SHORT", ss_idx)]:
            cells = []
            for hz, _ in horizons:
                rets = []
                for idx in indices:
                    if idx + hz < n:
                        ret = (c[idx+hz]-c[idx])/c[idx]*100
                        if "SHORT" in label: ret = -ret
                        rets.append(ret)
                wr = np.mean([1 if r>0 else 0 for r in rets])*100
                avg = np.mean(rets)
                cells.append((wr, avg))
            print(f"  {label:>12s} | {len(indices):>8d} | {cells[0][0]:>6.1f}% | {cells[1][0]:>6.1f}% | {cells[2][0]:>6.1f}% | {cells[3][0]:>6.1f}% | {cells[1][1]:>+7.3f}%")

print()
