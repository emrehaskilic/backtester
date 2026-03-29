"""
Can we improve weak-signal bars by adding TREND features?
Current 7 features = order flow microstructure (short-term pressure)
Missing: medium-term trend direction

Test: simple trend indicators as bias enhancers
"""
import numpy as np
import pandas as pd
import rust_engine

df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12
n_1h = len(df) // PERIOD
ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

c_1h = np.zeros(n_1h); h_1h = np.zeros(n_1h); l_1h = np.zeros(n_1h); o_1h = np.zeros(n_1h)
bv_1h = np.zeros(n_1h); sv_1h = np.zeros(n_1h); oi_1h = np.zeros(n_1h)
ts_1h = np.zeros(n_1h, dtype=np.uint64)

for i in range(n_1h):
    s, e = i * PERIOD, i * PERIOD + PERIOD
    ts_1h[i] = ts[s]; o_1h[i] = o[s]; h_1h[i] = h[s:e].max()
    l_1h[i] = l[s:e].min(); c_1h[i] = c[e-1]
    bv_1h[i] = bv[s:e].sum(); sv_1h[i] = sv[s:e].sum(); oi_1h[i] = oi[e-1]

# Outcomes K=12
K = 12
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
outcomes[-K:] = -1
valid = outcomes >= 0

# Get bias engine output
print("Running bias engine...")
r = rust_engine.bias_engine_compute_bias(
    ts_1h.astype(np.uint64), o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h
)
fb = np.array(r['final_bias'])

# === TREND FEATURES ===
def ema(data, span):
    k = 2.0 / (span + 1)
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

ema_24 = ema(c_1h, 24)    # 1 day EMA
ema_72 = ema(c_1h, 72)    # 3 day EMA
ema_168 = ema(c_1h, 168)  # 1 week EMA

# Trend signals
trend_short = (c_1h - ema_24) / ema_24 * 100   # % above/below 1D EMA
trend_med = (c_1h - ema_72) / ema_72 * 100      # % above/below 3D EMA
trend_long = (c_1h - ema_168) / ema_168 * 100   # % above/below 1W EMA
ema_slope = (ema_24 - np.roll(ema_24, 12)) / np.roll(ema_24, 12) * 100  # 12H slope

# Simple trend direction: majority vote of 3 EMAs
trend_vote = np.sign(trend_short) + np.sign(trend_med) + np.sign(trend_long)
trend_dir = np.sign(trend_vote)  # -1, 0, or +1

print(f"\n=== TREND INDICATORS ALONE ===")
for name, signal in [("EMA-24", np.sign(trend_short)),
                      ("EMA-72", np.sign(trend_med)),
                      ("EMA-168", np.sign(trend_long)),
                      ("EMA slope", np.sign(ema_slope)),
                      ("3-EMA vote", trend_dir)]:
    mask = (signal != 0) & valid
    if mask.sum() > 0:
        actual = outcomes[mask] == 1
        pred = signal[mask] > 0
        acc = (pred == actual).sum() / mask.sum()
        cov = mask.sum() / valid.sum() * 100
        changes = np.sum(signal[1:] != signal[:-1])
        avg_h = n_1h / max(changes, 1)
        print(f"  {name:>12}: cov={cov:.1f}%, acc={acc:.4f}, avg_hold={avg_h:.1f}H")

# === COMBINE: bias engine + trend ===
print(f"\n=== BIAS ENGINE + TREND COMBINATION ===")

# Strategy: use bias engine when strong, trend when bias is weak
for bias_thresh in [0.03, 0.05]:
    combined = np.zeros(n_1h)
    for i in range(n_1h):
        if abs(fb[i]) >= bias_thresh:
            combined[i] = np.sign(fb[i])  # bias engine dominant
        else:
            combined[i] = trend_dir[i]     # trend fallback

    mask = (combined != 0) & valid
    actual = outcomes[mask] == 1
    pred = combined[mask] > 0
    acc = (pred == actual).sum() / mask.sum()
    cov = mask.sum() / valid.sum() * 100
    changes = np.sum(combined[1:] != combined[:-1])
    avg_h = n_1h / max(changes, 1)
    print(f"  bias>{bias_thresh} + trend fallback: cov={cov:.1f}%, acc={acc:.4f}, avg_hold={avg_h:.1f}H")

# Strategy: weighted blend
for w_bias, w_trend in [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
    # Normalize trend to similar scale as bias (±0.10 range)
    trend_norm = np.clip(trend_short / 5.0, -0.15, 0.15)  # 5% above EMA → 0.15 bias
    blended = fb * w_bias + trend_norm * w_trend
    blended_dir = np.sign(blended)

    mask = (blended_dir != 0) & valid
    actual = outcomes[mask] == 1
    pred = blended_dir[mask] > 0
    acc = (pred == actual).sum() / mask.sum()
    cov = mask.sum() / valid.sum() * 100
    changes = np.sum(blended_dir[1:] != blended_dir[:-1])
    avg_h = n_1h / max(changes, 1)

    # Strong signal accuracy
    strong = np.abs(blended) > 0.05
    smask = strong & valid
    if smask.sum() > 0:
        sa = (np.sign(blended[smask]) > 0) == (outcomes[smask] == 1)
        sacc = sa.sum() / smask.sum()
    else:
        sacc = 0

    print(f"  blend w_bias={w_bias} w_trend={w_trend}: cov={cov:.1f}%, acc={acc:.4f}, "
          f"avg_hold={avg_h:.1f}H, strong_acc={sacc:.4f} ({smask.sum()} bars)")

# === BEST APPROACH: trend direction + bias as confidence ===
print(f"\n=== TREND DIR + BIAS CONFIDENCE ===")
# Direction always from 3-EMA vote (continuous)
# Confidence from bias engine (when strong, high conf; when weak, low conf)
final_dir = trend_dir.copy()
# Override with bias engine when strong AND disagrees with trend
for bias_thresh in [0.05, 0.07, 0.10]:
    override = final_dir.copy()
    n_overrides = 0
    for i in range(n_1h):
        if abs(fb[i]) >= bias_thresh and np.sign(fb[i]) != trend_dir[i]:
            override[i] = np.sign(fb[i])
            n_overrides += 1

    mask = (override != 0) & valid
    actual = outcomes[mask] == 1
    pred = override[mask] > 0
    acc = (pred == actual).sum() / mask.sum()
    cov = mask.sum() / valid.sum() * 100
    changes = np.sum(override[1:] != override[:-1])
    avg_h = n_1h / max(changes, 1)
    print(f"  trend + bias override>{bias_thresh}: cov={cov:.1f}%, acc={acc:.4f}, "
          f"avg_hold={avg_h:.1f}H, overrides={n_overrides}")
