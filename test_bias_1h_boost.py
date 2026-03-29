"""
Accuracy boosting techniques for hybrid bias system
"""
import numpy as np
import pandas as pd
import rust_engine

df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12
n_1h = len(df) // PERIOD
c_1h = np.zeros(n_1h); h_1h = np.zeros(n_1h); l_1h = np.zeros(n_1h); o_1h = np.zeros(n_1h)
bv_1h = np.zeros(n_1h); sv_1h = np.zeros(n_1h); oi_1h = np.zeros(n_1h)
ts_1h = np.zeros(n_1h, dtype=np.uint64)
ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

for i in range(n_1h):
    s, e = i * PERIOD, i * PERIOD + PERIOD
    ts_1h[i] = ts[s]; o_1h[i] = o[s]; h_1h[i] = h[s:e].max()
    l_1h[i] = l[s:e].min(); c_1h[i] = c[e-1]
    bv_1h[i] = bv[s:e].sum(); sv_1h[i] = sv[s:e].sum(); oi_1h[i] = oi[e-1]

K = 12
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
outcomes[-K:] = -1
valid = outcomes >= 0

print("Running bias engine...")
r = rust_engine.bias_engine_compute_bias(
    ts_1h.astype(np.uint64), o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h
)
fb = np.array(r['final_bias'])

def ema(data, span):
    k = 2.0 / (span + 1)
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

ema_24 = ema(c_1h, 24)
ema_48 = ema(c_1h, 48)
ema_72 = ema(c_1h, 72)

dist_24 = (c_1h - ema_24) / ema_24 * 100
dist_48 = (c_1h - ema_48) / ema_48 * 100
dist_72 = (c_1h - ema_72) / ema_72 * 100

mr_24 = -np.sign(dist_24)  # mean-reversion signal

# Volume z-score (1H)
vol_1h = bv_1h + sv_1h
vol_mean = ema(vol_1h, 72)
vol_std = np.zeros(n_1h)
for i in range(72, n_1h):
    vol_std[i] = np.std(vol_1h[max(0,i-72):i])
vol_z = np.where(vol_std > 0, (vol_1h - vol_mean) / np.maximum(vol_std, 1e-10), 0)

# CVD (1H)
cvd_1h = bv_1h - sv_1h
cvd_cum = np.cumsum(cvd_1h)
cvd_ema = ema(cvd_cum, 24)
cvd_dir = np.sign(cvd_cum - cvd_ema)  # CVD above its EMA → buying pressure

# OI change
oi_chg = np.zeros(n_1h)
for i in range(24, n_1h):
    oi_chg[i] = oi_1h[i] - oi_1h[i-24]
oi_expanding = oi_chg > 0

# RSI (14-period)
def compute_rsi(data, period=14):
    rsi = np.full(len(data), 50.0)
    gains = np.zeros(len(data))
    losses = np.zeros(len(data))
    for i in range(1, len(data)):
        d = data[i] - data[i-1]
        if d > 0: gains[i] = d
        else: losses[i] = -d
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    for i in range(period, len(data)):
        if avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - 100 / (1 + rs)
    return rsi

rsi = compute_rsi(c_1h, 14)

def test_signal(name, signal, min_bars=100):
    mask = (signal != 0) & valid
    if mask.sum() < min_bars: return
    actual = outcomes[mask] == 1
    pred = signal[mask] > 0
    acc = (pred == actual).sum() / mask.sum()
    cov = mask.sum() / valid.sum() * 100
    changes = np.sum(signal[1:] != signal[:-1])
    avg_h = n_1h / max(changes, 1)
    print(f"  {name:<55}: cov={cov:>5.1f}%, acc={acc:.4f}, hold={avg_h:>5.1f}H, n={mask.sum()}")

# === BOOST 1: MULTI-EMA CONFLUENCE ===
print(f"\n=== BOOST 1: MULTI-EMA CONFLUENCE ===")
# All 3 EMAs agree on MR direction
mr_confluence = np.where(
    (np.sign(dist_24) == np.sign(dist_48)) & (np.sign(dist_48) == np.sign(dist_72)),
    -np.sign(dist_24), 0  # all agree → strong MR signal
)
test_signal("3-EMA MR confluence", mr_confluence)

# At least 2 of 3 agree
mr_2of3 = -np.sign(np.sign(dist_24) + np.sign(dist_48) + np.sign(dist_72))
test_signal("2/3-EMA MR majority", mr_2of3)

# === BOOST 2: BIAS ENGINE AGREEMENT ===
print(f"\n=== BOOST 2: BIAS + MR AGREEMENT ===")
# When weak bias agrees with MR
agree = np.where(np.sign(fb) == mr_24, mr_24, 0)
test_signal("Bias agrees with MR-24", agree)

# When weak bias DISAGREES with MR → use MR anyway
disagree = np.where(np.sign(fb) != mr_24, mr_24, 0)
test_signal("Bias disagrees, use MR-24", disagree)

# Bias direction same as MR → confidence boost
agree_strong = np.where((np.sign(fb) == mr_24) & (np.abs(fb) > 0.01), mr_24 * 2, mr_24)
test_signal("MR + bias agreement boost", np.sign(agree_strong))

# === BOOST 3: VOLUME CONFIRMATION ===
print(f"\n=== BOOST 3: VOLUME CONFIRMATION ===")
# High volume + MR signal
for vz_thresh in [0.0, 0.5, 1.0]:
    vol_mr = np.where(vol_z > vz_thresh, mr_24, 0)
    test_signal(f"MR-24 + vol_z>{vz_thresh}", vol_mr)

# === BOOST 4: CVD CONFIRMATION ===
print(f"\n=== BOOST 4: CVD DIRECTION ===")
# MR signal confirmed by CVD (counter-intuitive: CVD opposing MR might be better)
cvd_confirms_mr = np.where(cvd_dir == mr_24, mr_24, 0)
test_signal("MR + CVD agrees", cvd_confirms_mr)
cvd_opposes_mr = np.where(cvd_dir != mr_24, mr_24, 0)
test_signal("MR + CVD opposes (exhaustion)", cvd_opposes_mr)

# === BOOST 5: RSI EXTREMES ===
print(f"\n=== BOOST 5: RSI EXTREMES ===")
# RSI oversold + MR bull, RSI overbought + MR bear
rsi_mr = np.where(
    ((rsi < 35) & (mr_24 > 0)) | ((rsi > 65) & (mr_24 < 0)),
    mr_24, 0
)
test_signal("MR + RSI extreme (35/65)", rsi_mr)

rsi_mr2 = np.where(
    ((rsi < 40) & (mr_24 > 0)) | ((rsi > 60) & (mr_24 < 0)),
    mr_24, 0
)
test_signal("MR + RSI moderate (40/60)", rsi_mr2)

# Just RSI mean-reversion
rsi_signal = np.where(rsi < 40, 1, np.where(rsi > 60, -1, 0))
test_signal("RSI MR alone (40/60)", rsi_signal)

# === BOOST 6: OI CONFIRMATION ===
print(f"\n=== BOOST 6: OI EXPANDING/CONTRACTING ===")
oi_mr = np.where(oi_expanding, mr_24, 0)
test_signal("MR + OI expanding", oi_mr)
oi_mr2 = np.where(~oi_expanding, mr_24, 0)
test_signal("MR + OI contracting", oi_mr2)

# === BOOST 7: DISTANCE FROM EMA (stronger when farther) ===
print(f"\n=== BOOST 7: DISTANCE FILTER ===")
for d_min, d_max in [(0.3, 2.0), (0.5, 3.0), (1.0, 5.0), (0.3, 999)]:
    dist_mr = np.where((np.abs(dist_24) >= d_min) & (np.abs(dist_24) <= d_max), mr_24, 0)
    test_signal(f"MR-24 dist [{d_min}-{d_max}]%", dist_mr)

# === COMBINED BEST ===
print(f"\n=== COMBINED: BEST BOOSTERS ===")
# Score system: each confirming factor adds to score
score = np.zeros(n_1h)
# Base: MR-24 direction
score += mr_24 * 1.0

# Boost: 2nd EMA agrees
score += np.where(np.sign(dist_48) == np.sign(dist_24), -np.sign(dist_24) * 0.5, 0)

# Boost: bias engine agrees
score += np.where(np.sign(fb) == mr_24, mr_24 * 0.5, 0)

# Boost: RSI confirms
score += np.where(((rsi < 40) & (mr_24 > 0)) | ((rsi > 60) & (mr_24 < 0)), mr_24 * 0.5, 0)

# Boost: strong bias engine overrides everything
score = np.where(np.abs(fb) >= 0.05, fb * 3.0, score)

score_dir = np.sign(score)

test_signal("Combined score (all factors)", score_dir)

# By score strength
for s_thresh in [0.5, 1.0, 1.5, 2.0]:
    s_mask_dir = np.where(np.abs(score) >= s_thresh, np.sign(score), 0)
    test_signal(f"Combined |score|>={s_thresh}", s_mask_dir)
