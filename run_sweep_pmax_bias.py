"""Sweep Bias + Adaptive PMAX Entegrasyonu
Sweep sinyali aktif → sweep bias kullan
Sweep sinyali yok → WF-optimized adaptive PMAX yonunu kullan
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

# ══════════════════════════════════════════════════════════════
# 1) Adaptive PMAX hesapla (WF consensus parametreleri)
# ══════════════════════════════════════════════════════════════

print("SWEEP BIAS + ADAPTIVE PMAX ENTEGRASYONU")
print("=" * 120)
print()

# src = hl2
src = sa((h + l) / 2, dtype=np.float64)

print("Adaptive PMAX hesaplaniyor...")
pmax_result = rust_engine.compute_adaptive_pmax(
    src, sa(h), sa(l), sa(c),
    10,      # base_atr_period
    3.0,     # base_atr_multiplier
    10,      # base_ma_length
    580,     # lookback (vol regime)
    440,     # flip_window
    4.0,     # mult_base
    1.25,    # mult_scale
    3,       # ma_base
    5.5,     # ma_scale
    19,      # atr_base
    1.5,     # atr_scale
    29,      # update_interval
)

pmax_dir = np.array(pmax_result['direction'])  # +1 = LONG, -1 = SHORT
print(f"  PMAX LONG: {(pmax_dir > 0).sum()/n*100:.1f}% | SHORT: {(pmax_dir < 0).sum()/n*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# 2) Sweep bias hesapla (1H + 4H + 8H)
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
        pcv = np.nan
        if bar >= cvd_lb:
            pcv = cvd[bar]-cvd[bar-cvd_lb]
            if sweep_dir=='low': pcv=-pcv
        early_p = np.nan
        entry_bar = bar+early_bars
        if entry_bar < n:
            early_p = (c[entry_bar]-c[bar])/c[bar]*100
            if sweep_dir=='low': early_p=-early_p
        if pd.isna(roc) or pd.isna(pcv) or pd.isna(early_p): continue
        rt = tercile_from_th(roc, current_th['roc_33'], current_th['roc_67'])
        ct = tercile_from_th(pcv, current_th['cvd_33'], current_th['cvd_67'])
        et = tercile_from_th(early_p, current_th['early_33'], current_th['early_67'])
        if (rt,ct,et) in CONT_CELLS:
            direction = 1 if sweep_dir == "high" else -1
            signal_start = bar + early_bars
            signal_end = bar + bp - 1
            for b in range(signal_start, min(signal_end+1, n)):
                bar_signals[b] = direction
    return bar_signals

print("Sweep sinyalleri hesaplaniyor...")
sig_1h = compute_tf_signals(12); print("  1H")
sig_4h = compute_tf_signals(48); print("  4H")
sig_8h = compute_tf_signals(96); print("  8H")

sweep_bias = np.array([sig_1h[i] + sig_4h[i] + sig_8h[i] for i in range(n)])
has_sweep = sweep_bias != 0

print(f"  Sweep aktif: {has_sweep.sum()/n*100:.1f}% | Bos: {(~has_sweep).sum()/n*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# 3) Birlestir: sweep aktifse sweep, degilse PMAX
# ══════════════════════════════════════════════════════════════

combined_bias = np.zeros(n)
sweep_used = 0
pmax_used = 0

for i in range(n):
    if sweep_bias[i] != 0:
        combined_bias[i] = 1 if sweep_bias[i] > 0 else -1
        sweep_used += 1
    else:
        combined_bias[i] = pmax_dir[i]
        pmax_used += 1

print(f"\n  Sweep kullanilan: {sweep_used/n*100:.1f}% | PMAX kullanilan: {pmax_used/n*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# 4) Tum yontemleri karsilastir
# ══════════════════════════════════════════════════════════════

horizons = [(12,"1h"), (48,"4h"), (96,"8h"), (288,"1d")]

def evaluate_bias(bias_arr, label):
    long_mask = bias_arr > 0
    short_mask = bias_arr < 0

    results = {}
    for hz, hz_label in horizons:
        l_rets = [(c[i+hz]-c[i])/c[i]*100 for i in np.where(long_mask)[0] if i+hz < n]
        s_rets = [(c[i]-c[i+hz])/c[i]*100 for i in np.where(short_mask)[0] if i+hz < n]
        all_rets = l_rets + s_rets
        wr = np.mean([1 if r>0 else 0 for r in all_rets])*100 if all_rets else 0
        avg = np.mean(all_rets) if all_rets else 0
        results[hz_label] = (wr, avg, len(all_rets))
    return results

print(f"\n{'='*120}")
print("KARSILASTIRMA")
print(f"{'='*120}")
print(f"\n  {'Yontem':>30s} | {'Kapsam':>7s} | {'1h WR':>7s} | {'4h WR':>7s} | {'8h WR':>7s} | {'1d WR':>7s} | {'4h avg':>8s}")
print(f"  {'-'*85}")

for label, bias_arr in [
    ("Sadece sweep", sweep_bias),
    ("Sadece PMAX", pmax_dir),
    ("Sweep + PMAX", combined_bias),
]:
    active = np.abs(bias_arr) > 0
    kapsam = active.sum() / n * 100
    res = evaluate_bias(bias_arr, label)
    print(f"  {label:>30s} | {kapsam:>6.1f}% | {res['1h'][0]:>6.1f}% | {res['4h'][0]:>6.1f}% | {res['8h'][0]:>6.1f}% | {res['1d'][0]:>6.1f}% | {res['4h'][1]:>+7.3f}%")

# ══════════════════════════════════════════════════════════════
# 5) Sweep + PMAX: kaynak bazinda detay
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*120}")
print("KAYNAK BAZINDA DETAY (Sweep + PMAX)")
print(f"{'='*120}")

# Sweep aktif bolgeler
sweep_long = (has_sweep) & (sweep_bias > 0)
sweep_short = (has_sweep) & (sweep_bias < 0)
pmax_long = (~has_sweep) & (pmax_dir > 0)
pmax_short = (~has_sweep) & (pmax_dir < 0)

print(f"\n  {'Kaynak':>20s} | {'N bars':>8s} | {'%':>6s} | {'4h WR':>7s} | {'8h WR':>7s} | {'4h avg':>8s}")
print(f"  {'-'*65}")

for label, mask in [("Sweep LONG", sweep_long), ("Sweep SHORT", sweep_short),
                     ("PMAX LONG", pmax_long), ("PMAX SHORT", pmax_short)]:
    indices = np.where(mask)[0]
    if len(indices) < 100: continue
    is_long = "LONG" in label
    rets_4h = [(c[i+48]-c[i])/c[i]*100 * (1 if is_long else -1) for i in indices if i+48 < n]
    rets_8h = [(c[i+96]-c[i])/c[i]*100 * (1 if is_long else -1) for i in indices if i+96 < n]
    wr4 = np.mean([1 if r>0 else 0 for r in rets_4h])*100
    wr8 = np.mean([1 if r>0 else 0 for r in rets_8h])*100
    avg4 = np.mean(rets_4h)
    print(f"  {label:>20s} | {len(indices):>8d} | {len(indices)/n*100:>5.1f}% | {wr4:>6.1f}% | {wr8:>6.1f}% | {avg4:>+7.3f}%")

# ══════════════════════════════════════════════════════════════
# 6) Uyum/Catisma analizi
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*120}")
print("UYUM / CATISMA ANALIZI")
print("Sweep aktifken PMAX ile ayni yonde mi ters yonde mi?")
print(f"{'='*120}")

agree = has_sweep & (np.sign(sweep_bias) == pmax_dir)
disagree = has_sweep & (np.sign(sweep_bias) != pmax_dir)

print(f"\n  Sweep aktif: {has_sweep.sum():,} bar")
print(f"  Uyumlu (sweep = PMAX): {agree.sum():,} ({agree.sum()/has_sweep.sum()*100:.1f}%)")
print(f"  Catisma (sweep ≠ PMAX): {disagree.sum():,} ({disagree.sum()/has_sweep.sum()*100:.1f}%)")

# Uyumlu bolgede vs catisma bolgesinde WR
for label, mask in [("Uyumlu", agree), ("Catisma", disagree)]:
    indices = np.where(mask)[0]
    if len(indices) < 100: continue
    rets = []
    for i in indices:
        if i + 48 < n:
            ret = (c[i+48]-c[i])/c[i]*100
            if sweep_bias[i] < 0: ret = -ret
            rets.append(ret)
    wr = np.mean([1 if r>0 else 0 for r in rets])*100
    avg = np.mean(rets)
    print(f"  {label:>10s}: 4h WR={wr:.1f}% | avg={avg:+.3f}% | N={len(indices):,}")

# Yillik performans
print(f"\n{'='*120}")
print("YILLIK PERFORMANS (Sweep + PMAX combined)")
print(f"{'='*120}")

df['year'] = df['dt'].dt.year
for year in sorted(df['year'].unique()):
    year_mask = df['year'].values == year
    year_indices = np.where(year_mask)[0]

    long_bars = sum(1 for i in year_indices if combined_bias[i] > 0)
    short_bars = sum(1 for i in year_indices if combined_bias[i] < 0)

    # 4h returns
    rets = []
    for i in year_indices[::48]:  # her 4h'de bir sample
        if i + 48 < n:
            ret = (c[i+48]-c[i])/c[i]*100
            if combined_bias[i] < 0: ret = -ret
            elif combined_bias[i] == 0: continue
            rets.append(ret)

    if rets:
        wr = np.mean([1 if r>0 else 0 for r in rets])*100
        avg = np.mean(rets)
        total = np.sum(rets)
        print(f"  {year}: L:{long_bars/len(year_indices)*100:.0f}% S:{short_bars/len(year_indices)*100:.0f}% | 4h WR:{wr:.1f}% | avg:{avg:+.3f}% | N:{len(rets)}")

print()
