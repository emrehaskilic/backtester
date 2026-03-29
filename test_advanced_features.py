"""
Advanced features - listen to the data first.
57. Market Structure (HH/HL/LH/LL)
58. Session Return
59. Cross-TF Trend
60. Correlation Regime

Each feature tested independently, then combined.
"""
import numpy as np
import pandas as pd

# === Load data ===
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12; n_1h = len(df) // PERIOD
ts = df["open_time"].values
o,h,l,c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv,sv = df["buy_vol"].values, df["sell_vol"].values

c_1h=np.zeros(n_1h);h_1h=np.zeros(n_1h);l_1h=np.zeros(n_1h);o_1h=np.zeros(n_1h)
bv_1h=np.zeros(n_1h);sv_1h=np.zeros(n_1h);ts_1h=np.zeros(n_1h,dtype=np.uint64)
for i in range(n_1h):
    s,e=i*PERIOD,i*PERIOD+PERIOD
    ts_1h[i]=ts[s];o_1h[i]=o[s];h_1h[i]=h[s:e].max()
    l_1h[i]=l[s:e].min();c_1h[i]=c[e-1]
    bv_1h[i]=bv[s:e].sum();sv_1h[i]=sv[s:e].sum()

# BTC data
dfb = pd.read_parquet("data/BTCUSDT_5m_5y_perp.parquet")
nb = len(dfb) // PERIOD
btc_c = np.zeros(nb); btc_h = np.zeros(nb); btc_l = np.zeros(nb)
for i in range(nb):
    s,e=i*PERIOD,i*PERIOD+PERIOD
    btc_c[i]=dfb["close"].values[s:e][-1]
    btc_h[i]=dfb["high"].values[s:e].max()
    btc_l[i]=dfb["low"].values[s:e].min()
n_common = min(n_1h, nb)

# Outcomes K=2 (optimal from TPE)
K = 2
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if c_1h[i+K] > c_1h[i] else 0
outcomes[-K:] = -1
WARMUP = 500
valid = (outcomes >= 0) & (np.arange(n_1h) >= WARMUP)

def test_signal(name, signal, min_n=100):
    mask = (signal != 0) & valid
    if mask.sum() < min_n: return
    actual = outcomes[mask] == 1
    pred = signal[mask] > 0
    acc = (pred == actual).sum() / mask.sum()
    cov = mask.sum() / valid.sum() * 100
    changes = np.sum(signal[1:] != signal[:-1])
    avg_h = n_1h / max(changes, 1)
    print(f"  {name:<50}: cov={cov:>5.1f}%, acc={acc:.4f}, hold={avg_h:>5.1f}H, n={mask.sum()}")

print(f"1H: {n_1h} bars, K={K}")

# ═══════════════════════════════════════════════════
# 57. MARKET STRUCTURE
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"57. MARKET STRUCTURE (HH/HL/LH/LL)")
print(f"{'='*60}")

def compute_market_structure(high, low, lookback):
    n = len(high)
    ms_score = np.zeros(n)

    # Detect confirmed swing highs and lows
    swing_highs = []  # (index, price)
    swing_lows = []

    for i in range(lookback * 2, n):
        # Swing high: highest in last 2*lookback, confirmed if at least lookback bars ago
        window_h = high[i - 2*lookback:i+1]
        max_idx_in_window = np.argmax(window_h)
        max_pos = i - 2*lookback + max_idx_in_window
        if max_pos <= i - lookback:
            if not swing_highs or swing_highs[-1][0] != max_pos:
                swing_highs.append((max_pos, high[max_pos]))

        # Swing low
        window_l = low[i - 2*lookback:i+1]
        min_idx_in_window = np.argmin(window_l)
        min_pos = i - 2*lookback + min_idx_in_window
        if min_pos <= i - lookback:
            if not swing_lows or swing_lows[-1][0] != min_pos:
                swing_lows.append((min_pos, low[min_pos]))

        # Score from last 4 swing highs
        sh_score = 0
        recent_sh = [p for idx, p in swing_highs if idx <= i][-4:]
        if len(recent_sh) >= 2:
            for j in range(1, len(recent_sh)):
                if recent_sh[j] > recent_sh[j-1]: sh_score += 1  # HH
                else: sh_score -= 1  # LH

        # Score from last 4 swing lows
        sl_score = 0
        recent_sl = [p for idx, p in swing_lows if idx <= i][-4:]
        if len(recent_sl) >= 2:
            for j in range(1, len(recent_sl)):
                if recent_sl[j] > recent_sl[j-1]: sl_score += 1  # HL
                else: sl_score -= 1  # LL

        ms_score[i] = sh_score + sl_score

    return ms_score

for lb in [3, 5, 7, 10]:
    ms = compute_market_structure(h_1h, l_1h, lb)
    test_signal(f"MS lookback={lb} (trend follow)", np.sign(ms))
    test_signal(f"MS lookback={lb} (mean revert)", -np.sign(ms))

    # By strength
    for thresh in [2, 4, 6]:
        strong = np.where(np.abs(ms) >= thresh, np.sign(ms), 0)
        if (strong != 0).sum() > 100:
            test_signal(f"  MS lb={lb} |score|>={thresh} (trend)", strong)
            test_signal(f"  MS lb={lb} |score|>={thresh} (MR)", -strong)

# ═══════════════════════════════════════════════════
# 59. CROSS-TF TREND
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"59. CROSS-TF TREND")
print(f"{'='*60}")

def ema(data, span):
    k=2.0/(span+1);out=np.zeros_like(data);out[0]=data[0]
    for i in range(1,len(data)): out[i]=data[i]*k+out[i-1]*(1-k)
    return out

# Build 4H and 1D from 1H
n_4h = n_1h // 4
c_4h = np.zeros(n_4h); h_4h = np.zeros(n_4h); l_4h = np.zeros(n_4h)
for i in range(n_4h):
    s,e = i*4, i*4+4
    c_4h[i]=c_1h[e-1]; h_4h[i]=h_1h[s:e].max(); l_4h[i]=l_1h[s:e].min()

n_1d = n_1h // 24
c_1d = np.zeros(n_1d); h_1d = np.zeros(n_1d); l_1d = np.zeros(n_1d)
for i in range(n_1d):
    s,e = i*24, i*24+24
    c_1d[i]=c_1h[e-1]; h_1d[i]=h_1h[s:e].max(); l_1d[i]=l_1h[s:e].min()

for ema_1h_span, ema_4h_span, ema_1d_span in [(20,10,10), (24,12,10), (12,6,5), (36,15,10)]:
    ema_1h = ema(c_1h, ema_1h_span)
    ema_4h_raw = ema(c_4h, ema_4h_span)
    ema_1d_raw = ema(c_1d, ema_1d_span)

    # Map back to 1H
    trend_1h = np.sign(c_1h - ema_1h)
    trend_4h = np.zeros(n_1h)
    for i in range(n_1h):
        idx_4h = min(i // 4, n_4h - 1)
        if idx_4h > 0: trend_4h[i] = np.sign(c_4h[idx_4h-1] - ema_4h_raw[idx_4h-1])  # lagged
    trend_1d = np.zeros(n_1h)
    for i in range(n_1h):
        idx_1d = min(i // 24, n_1d - 1)
        if idx_1d > 0: trend_1d[i] = np.sign(c_1d[idx_1d-1] - ema_1d_raw[idx_1d-1])

    alignment = trend_1h + trend_4h + trend_1d  # -3 to +3

    test_signal(f"TF align({ema_1h_span},{ema_4h_span},{ema_1d_span}) trend", np.sign(alignment))
    test_signal(f"TF align({ema_1h_span},{ema_4h_span},{ema_1d_span}) MR", -np.sign(alignment))

    # Strong alignment only
    strong_align = np.where(np.abs(alignment) >= 3, np.sign(alignment), 0)
    test_signal(f"  Full align (|3|) trend", strong_align)
    test_signal(f"  Full align (|3|) MR", -strong_align)

# ═══════════════════════════════════════════════════
# 58. SESSION RETURN
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"58. SESSION FEATURES")
print(f"{'='*60}")

dates = pd.to_datetime(ts_1h, unit='ms', utc=True)
hours = np.array([d.hour for d in dates])
dows = np.array([d.dayofweek for d in dates])

# Session classification
# Tokyo: 0-9 UTC, London: 7-16 UTC, NY: 12-21 UTC
session_id = np.zeros(n_1h, dtype=int)  # 0=tokyo, 1=london, 2=ny
for i in range(n_1h):
    hr = hours[i]
    if 0 <= hr < 7: session_id[i] = 0       # Tokyo only
    elif 7 <= hr < 12: session_id[i] = 1     # London (+ Tokyo overlap early)
    elif 12 <= hr < 16: session_id[i] = 2    # NY + London overlap
    elif 16 <= hr < 21: session_id[i] = 2    # NY only
    else: session_id[i] = 0                   # Post-NY / Pre-Tokyo

# Session return (from session open)
session_return = np.zeros(n_1h)
session_open_price = c_1h[0]
prev_session = session_id[0]
for i in range(n_1h):
    if session_id[i] != prev_session:
        session_open_price = o_1h[i]
        prev_session = session_id[i]
    if session_open_price > 0:
        session_return[i] = (c_1h[i] - session_open_price) / session_open_price * 100

# Session MR: if session is up so far, predict down for rest (and vice versa)
for thresh in [0.2, 0.5, 1.0]:
    sr_mr = np.where(session_return > thresh, -1, np.where(session_return < -thresh, 1, 0))
    test_signal(f"Session return MR (>{thresh}%)", sr_mr)
    sr_trend = np.where(session_return > thresh, 1, np.where(session_return < -thresh, -1, 0))
    test_signal(f"Session return trend (>{thresh}%)", sr_trend)

# Hour-based signal (from our analysis: 00-04 and 20-24 are best)
hour_signal = np.zeros(n_1h)
for i in range(n_1h):
    if 0 <= hours[i] < 4 or 20 <= hours[i] < 24:
        hour_signal[i] = 1.0   # good hours: boost existing signal
    elif 12 <= hours[i] < 16:
        hour_signal[i] = -0.5  # bad hours: dampen

# Day-of-week signal
dow_signal = np.zeros(n_1h)
for i in range(n_1h):
    if dows[i] in [5, 6]:  # weekend
        dow_signal[i] = 1.0
    elif dows[i] == 0:  # monday
        dow_signal[i] = -0.5

# ═══════════════════════════════════════════════════
# 60. CORRELATION REGIME
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"60. BTC-ETH CORRELATION REGIME")
print(f"{'='*60}")

# Rolling correlation
eth_ret = np.zeros(n_common)
btc_ret = np.zeros(n_common)
for i in range(1, n_common):
    eth_ret[i] = (c_1h[i] - c_1h[i-1]) / c_1h[i-1]
    btc_ret[i] = (btc_c[i] - btc_c[i-1]) / btc_c[i-1]

for corr_window in [24, 48, 72, 168]:
    corr = np.zeros(n_common)
    for i in range(corr_window, n_common):
        e_slice = eth_ret[i-corr_window:i]
        b_slice = btc_ret[i-corr_window:i]
        if np.std(e_slice) > 0 and np.std(b_slice) > 0:
            corr[i] = np.corrcoef(e_slice, b_slice)[0, 1]

    # When correlation is high, BTC direction = ETH direction
    btc_dir = np.sign(btc_ret)
    high_corr_btc = np.where(corr > 0.7, btc_dir, 0)[:n_1h]
    test_signal(f"BTC dir when corr>{0.7} (w={corr_window})", high_corr_btc)

    # When correlation breaks down, ETH might diverge
    low_corr = np.where(corr < 0.3, 1, 0)[:n_1h]  # flag only

    # Correlation change (dropping = regime change)
    corr_chg = np.zeros(n_common)
    for i in range(24, n_common):
        corr_chg[i] = corr[i] - corr[i-24]

    # Dropping correlation + BTC direction
    drop_corr_btc = np.where((corr_chg < -0.2) & (np.abs(btc_ret) > 0.005), -btc_dir, 0)[:n_1h]
    test_signal(f"Corr dropping + BTC opposite (w={corr_window})", drop_corr_btc)

# ═══════════════════════════════════════════════════
# COMBINED: ALL FEATURES TOGETHER
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"COMBINED: ALL ADVANCED FEATURES")
print(f"{'='*60}")

# Best from each category based on results above
ms_5 = compute_market_structure(h_1h, l_1h, 5)

ema_1h_v = ema(c_1h, 20)
ema_4h_raw = ema(c_4h, 10)
ema_1d_raw = ema(c_1d, 10)
trend_1h = np.sign(c_1h - ema_1h_v)
trend_4h = np.zeros(n_1h)
trend_1d = np.zeros(n_1h)
for i in range(n_1h):
    idx_4h = min(i//4, n_4h-1)
    if idx_4h > 0: trend_4h[i] = np.sign(c_4h[idx_4h-1] - ema_4h_raw[idx_4h-1])
    idx_1d = min(i//24, n_1d-1)
    if idx_1d > 0: trend_1d[i] = np.sign(c_1d[idx_1d-1] - ema_1d_raw[idx_1d-1])
tf_align = trend_1h + trend_4h + trend_1d

# Try different combinations
# Combo 1: MS trend + TF alignment
combo1 = np.sign(ms_5) + np.sign(tf_align)
test_signal("MS + TF (trend)", np.sign(combo1))
test_signal("MS + TF (MR)", -np.sign(combo1))

# Combo 2: MS MR (strong) only
ms_mr_strong = np.where(np.abs(ms_5) >= 4, -np.sign(ms_5), 0)
test_signal("MS strong MR (|4|+)", ms_mr_strong)

# Combo 3: TF aligned + MS disagrees (reversal setup)
reversal = np.where((np.abs(tf_align) >= 2) & (np.sign(ms_5) != np.sign(tf_align)), -np.sign(tf_align), 0)
test_signal("TF trend but MS reversing", reversal)

# Combo 4: Everything agrees (trend)
all_trend = np.where(
    (np.sign(ms_5) == np.sign(tf_align)) & (np.abs(ms_5) >= 2) & (np.abs(tf_align) >= 2),
    np.sign(ms_5), 0
)
test_signal("All agree trend (MS+TF)", all_trend)
test_signal("All agree MR (MS+TF)", -all_trend)

print(f"\n{'='*60}")
print(f"INDIVIDUAL FEATURE ACCURACY SUMMARY")
print(f"{'='*60}")
