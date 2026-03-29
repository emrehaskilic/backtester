"""Bias Gostericisi Testi
Her 5m bar'da 1H/4H/8H aktif sinyallerinin net yonu → sonraki fiyat hareketi
Pmax yerine kullanilabilir mi?
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

def compute_tf_signals(bp):
    """Her mum icin sinyal hesapla, aktif sinyal suresi boyunca gecerli"""
    roc_lb = max(6, int(bp * 0.75))
    cvd_lb = bp
    early_bars = max(3, bp // 4)

    n_candles = n // bp
    candles = []
    for i in range(n_candles):
        s = i * bp; e = s + bp
        candles.append({
            'start': s, 'open': c[s],
            'high': np.max(h[s:e]), 'low': np.min(l[s:e]),
            'close': c[e-1],
        })

    # Walk-forward threshold'lari
    TRAIN = 540; UPDATE = 180
    current_th = None; last_train_end = 0

    # Her 5m bar icin aktif sinyal (None, "LONG", "SHORT")
    bar_signals = [None] * n

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

        all_events.append({
            'candle_idx': i, 'bar': bar, 'signal': sig, 'sweep_dir': sweep_dir,
        })

        # Threshold guncelle
        if i >= TRAIN and (current_th is None or i - last_train_end >= UPDATE):
            td_events = [e for e in all_events if e['candle_idx'] >= i - TRAIN and e['candle_idx'] < i and e['signal'] is not None]
            if len(td_events) >= 50:
                rocs, cvds, earlies = [], [], []
                for e in td_events:
                    b = e['bar']
                    if b >= roc_lb and c[b - roc_lb] > 0:
                        r = (c[b] - c[b - roc_lb]) / c[b - roc_lb] * 100
                        if e['sweep_dir'] == 'low': r = -r
                        rocs.append(r)
                    if b >= cvd_lb:
                        cv = cvd[b] - cvd[b - cvd_lb]
                        if e['sweep_dir'] == 'low': cv = -cv
                        cvds.append(cv)
                    eb = b + early_bars
                    if eb < n:
                        ep = (c[eb] - c[b]) / c[b] * 100
                        if e['sweep_dir'] == 'low': ep = -ep
                        earlies.append(ep)
                if len(rocs) >= 30 and len(cvds) >= 30 and len(earlies) >= 30:
                    current_th = {
                        'roc_33': np.percentile(rocs, 33.3), 'roc_67': np.percentile(rocs, 66.7),
                        'cvd_33': np.percentile(cvds, 33.3), 'cvd_67': np.percentile(cvds, 66.7),
                        'early_33': np.percentile(earlies, 33.3), 'early_67': np.percentile(earlies, 66.7),
                    }
                    last_train_end = i

        if current_th is None or sig is None or sweep_dir is None: continue

        # ROC + CVD hesapla
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

        if pd.isna(roc) or pd.isna(pre_cvd_val) or pd.isna(early_p): continue

        rt = tercile_from_th(roc, current_th['roc_33'], current_th['roc_67'])
        ct = tercile_from_th(pre_cvd_val, current_th['cvd_33'], current_th['cvd_67'])
        et = tercile_from_th(early_p, current_th['early_33'], current_th['early_67'])

        if (rt, ct, et) in CONT_CELLS:
            direction = "LONG" if sweep_dir == "high" else "SHORT"
            # Sinyal aktif: giris bar'indan mum kapanisina kadar
            signal_start = bar + early_bars
            signal_end = bar + bp - 1
            for b in range(signal_start, min(signal_end + 1, n)):
                bar_signals[b] = direction

    return bar_signals

print("BIAS GOSTERICISI TESTI")
print("=" * 120)
print("Her 5m bar'da 1H/4H/8H aktif sinyallerinin net yonu")
print()

# Her TF'nin sinyallerini hesapla
print("Sinyaller hesaplaniyor...")
signals_1h = compute_tf_signals(12)
print("  1H tamam")
signals_4h = compute_tf_signals(48)
print("  4H tamam")
signals_8h = compute_tf_signals(96)
print("  8H tamam")

# Her 5m bar icin bias skoru hesapla
# LONG = +1, SHORT = -1, None = 0
bias = np.zeros(n, dtype=np.float64)
for i in range(n):
    score = 0
    if signals_1h[i] == "LONG": score += 1
    elif signals_1h[i] == "SHORT": score -= 1
    if signals_4h[i] == "LONG": score += 1
    elif signals_4h[i] == "SHORT": score -= 1
    if signals_8h[i] == "LONG": score += 1
    elif signals_8h[i] == "SHORT": score -= 1
    bias[i] = score

# Bias dagilimi
from collections import Counter
bias_counts = Counter(bias)
print(f"\nBIAS DAGILIMI (tum 5m bar'lar):")
print(f"  {'Bias':>6s} | {'Anlam':>25s} | {'N':>8s} | {'%':>6s}")
print(f"  {'-'*55}")
for b in [-3, -2, -1, 0, 1, 2, 3]:
    cnt = bias_counts.get(b, 0)
    anlam = {-3: "3 TF SHORT", -2: "2 SHORT + 1 notr/ters", -1: "1 SHORT + diger notr",
             0: "Notr / karisik", 1: "1 LONG + diger notr", 2: "2 LONG + 1 notr/ters",
             3: "3 TF LONG"}.get(b, "?")
    print(f"  {b:>+6.0f} | {anlam:>25s} | {cnt:>8d} | {cnt/n*100:>5.1f}%")

# Her bias seviyesi icin sonraki N bar fiyat degisimi
print(f"\nBIAS → SONRAKI FIYAT DEGISIMI:")
print(f"  {'Bias':>6s} | {'1h sonra':>10s} | {'4h sonra':>10s} | {'8h sonra':>10s} | {'1d sonra':>10s} | {'N':>8s}")
print(f"  {'-'*65}")

horizons = [(12, "1h"), (48, "4h"), (96, "8h"), (288, "1d")]

for b in [-3, -2, -1, 0, 1, 2, 3]:
    mask = np.where(bias == b)[0]
    if len(mask) < 100: continue

    cells = []
    for hz, hz_label in horizons:
        returns = []
        for idx in mask:
            if idx + hz < n:
                ret = (c[idx + hz] - c[idx]) / c[idx] * 100
                if b < 0: ret = -ret  # SHORT bias icin ters cev
                returns.append(ret)
        if returns:
            avg = np.mean(returns)
            wr = np.mean([1 if r > 0 else 0 for r in returns]) * 100
            cells.append(f"{avg:>+.3f}% {wr:.0f}%")
        else:
            cells.append(f"{'N/A':>10s}")

    print(f"  {b:>+6.0f} | {cells[0]:>10s} | {cells[1]:>10s} | {cells[2]:>10s} | {cells[3]:>10s} | {len(mask):>8d}")

# Bias degisim anlarini analiz et
print(f"\n\nBIAS DEGISIM ANALIZI:")
print(f"Bias'in 0'dan pozitife veya negatife gecis ani")
print(f"  {'Gecis':>20s} | {'1h sonra':>10s} | {'4h sonra':>10s} | {'8h sonra':>10s} | {'N':>6s}")
print(f"  {'-'*60}")

transitions = [
    ("0/neg → +2/+3", lambda i: bias[i] >= 2 and (i == 0 or bias[i-1] <= 0)),
    ("0/neg → +1", lambda i: bias[i] == 1 and (i == 0 or bias[i-1] <= 0)),
    ("0/pos → -2/-3", lambda i: bias[i] <= -2 and (i == 0 or bias[i-1] >= 0)),
    ("0/pos → -1", lambda i: bias[i] == -1 and (i == 0 or bias[i-1] >= 0)),
]

for label, cond in transitions:
    indices = [i for i in range(1, n) if cond(i)]
    if len(indices) < 30: continue

    cells = []
    for hz, hz_label in [(12,"1h"), (48,"4h"), (96,"8h")]:
        returns = []
        for idx in indices:
            if idx + hz < n:
                ret = (c[idx + hz] - c[idx]) / c[idx] * 100
                if "neg" in label.split("→")[1] or "-" in label.split("→")[1]:
                    ret = -ret
                returns.append(ret)
        avg = np.mean(returns)
        wr = np.mean([1 if r > 0 else 0 for r in returns]) * 100
        cells.append(f"{avg:>+.3f}% {wr:.0f}%")

    print(f"  {label:>20s} | {cells[0]:>10s} | {cells[1]:>10s} | {cells[2]:>10s} | {len(indices):>6d}")

# Bias suresi analizi
print(f"\n\nBIAS SURESI (ne kadar suruyor?):")

for b_level in [3, 2, -2, -3]:
    durations = []
    in_bias = False
    start = 0
    for i in range(n):
        if bias[i] == b_level:
            if not in_bias:
                start = i
                in_bias = True
        else:
            if in_bias:
                dur = (i - start) * 5  # dakika
                durations.append(dur)
                in_bias = False

    if durations:
        print(f"  Bias {b_level:>+d}: {len(durations)} donem | ort {np.mean(durations):.0f}dk | med {np.median(durations):.0f}dk | max {np.max(durations):.0f}dk")

print()
