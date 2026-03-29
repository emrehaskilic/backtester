"""
Mean reversion as continuous bias:
- Price above EMA → expect pullback → BEARISH
- Price below EMA → expect bounce → BULLISH
- Bias engine overrides when strong signal
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

# === MEAN REVERSION SIGNALS ===
print(f"\n=== MEAN REVERSION (COUNTER-TREND) ===")
for ema_span in [12, 24, 48, 72, 168]:
    ema_val = ema(c_1h, ema_span)
    # COUNTER-trend: above EMA → short, below EMA → long
    mr_signal = -np.sign(c_1h - ema_val)

    mask = (mr_signal != 0) & valid
    actual = outcomes[mask] == 1
    pred = mr_signal[mask] > 0
    acc = (pred == actual).sum() / mask.sum()
    changes = np.sum(mr_signal[1:] != mr_signal[:-1])
    avg_h = n_1h / max(changes, 1)
    print(f"  MR EMA-{ema_span:>3}: cov=100%, acc={acc:.4f}, avg_hold={avg_h:.1f}H")

# === DISTANCE-WEIGHTED MEAN REVERSION ===
print(f"\n=== DISTANCE-WEIGHTED MR (farther from EMA = stronger signal) ===")
for ema_span in [24, 48, 72]:
    ema_val = ema(c_1h, ema_span)
    dist_pct = (c_1h - ema_val) / ema_val * 100  # % distance from EMA
    mr_bias = -dist_pct / 10.0  # normalize: 1% above EMA → -0.10 bias

    mr_dir = np.sign(mr_bias)
    mask = (mr_dir != 0) & valid
    actual = outcomes[mask] == 1
    pred = mr_dir[mask] > 0
    acc = (pred == actual).sum() / mask.sum()

    # Accuracy by distance
    for dist_thresh in [0.5, 1.0, 2.0, 3.0]:
        dmask = (np.abs(dist_pct) > dist_thresh) & valid
        if dmask.sum() > 100:
            a = outcomes[dmask] == 1
            p = -np.sign(dist_pct[dmask]) > 0  # counter-trend
            da = (p == a).sum() / dmask.sum()
            print(f"  EMA-{ema_span}, |dist|>{dist_thresh}%: {dmask.sum()} bars, MR acc={da:.4f}")

# === HYBRID: bias engine (strong) + mean reversion (weak) ===
print(f"\n=== HYBRID: BIAS ENGINE + MEAN REVERSION ===")
ema_48 = ema(c_1h, 48)
dist_48 = (c_1h - ema_48) / ema_48 * 100
mr_bias_48 = np.clip(-dist_48 / 10.0, -0.15, 0.15)

for bias_thresh in [0.03, 0.05, 0.07]:
    hybrid = np.zeros(n_1h)
    source = np.zeros(n_1h, dtype=int)  # 0=none, 1=bias_engine, 2=mean_rev

    for i in range(n_1h):
        if abs(fb[i]) >= bias_thresh:
            hybrid[i] = fb[i]
            source[i] = 1
        else:
            hybrid[i] = mr_bias_48[i]
            source[i] = 2

    h_dir = np.sign(hybrid)
    mask = (h_dir != 0) & valid
    actual = outcomes[mask] == 1
    pred = h_dir[mask] > 0
    acc = (pred == actual).sum() / mask.sum()
    cov = mask.sum() / valid.sum() * 100
    changes = np.sum(h_dir[1:] != h_dir[:-1])
    avg_h = n_1h / max(changes, 1)

    # Per-source accuracy
    bias_mask = (source == 1) & valid & (h_dir != 0)
    mr_mask = (source == 2) & valid & (h_dir != 0)

    bias_acc = ((h_dir[bias_mask] > 0) == (outcomes[bias_mask] == 1)).sum() / max(bias_mask.sum(), 1)
    mr_acc = ((h_dir[mr_mask] > 0) == (outcomes[mr_mask] == 1)).sum() / max(mr_mask.sum(), 1)

    print(f"\n  bias_thresh={bias_thresh}:")
    print(f"    Total: cov={cov:.1f}%, acc={acc:.4f}, avg_hold={avg_h:.1f}H")
    print(f"    Bias engine bars: {bias_mask.sum()} ({bias_mask.sum()/valid.sum()*100:.1f}%), acc={bias_acc:.4f}")
    print(f"    Mean-rev bars:    {mr_mask.sum()} ({mr_mask.sum()/valid.sum()*100:.1f}%), acc={mr_acc:.4f}")

    # Strong hybrid signals
    for st in [0.03, 0.05, 0.10]:
        smask = (np.abs(hybrid) > st) & valid
        if smask.sum() > 0:
            sa = ((np.sign(hybrid[smask]) > 0) == (outcomes[smask] == 1)).sum() / smask.sum()
            print(f"    |hybrid|>{st}: {smask.sum()} bars, acc={sa:.4f}")
