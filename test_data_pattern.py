"""
Data Pattern Analysis: TF doğru/yanlış olduğunda datada ne farklı?

Veriyi konuşturalım. Her durumda data nasıl görünüyor:
- 1H doğru vs yanlış
- 4H kurtardığında vs kurtarmadığında
- 8H kurtardığında vs kurtarmadığında
- Üçü birden doğru vs üçü birden yanlış
"""
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

def aggregate(period):
    n = len(df) // period
    ts_a = np.zeros(n, dtype=np.uint64)
    o_a = np.zeros(n); c_a = np.zeros(n); h_a = np.zeros(n); l_a = np.zeros(n)
    bv_a = np.zeros(n); sv_a = np.zeros(n); oi_a = np.zeros(n)
    for i in range(n):
        s, e = i * period, i * period + period
        ts_a[i] = ts[s]; o_a[i] = o[s]; c_a[i] = c[e - 1]
        h_a[i] = h[s:e].max(); l_a[i] = l[s:e].min()
        bv_a[i] = bv[s:e].sum(); sv_a[i] = sv[s:e].sum(); oi_a[i] = oi[e - 1]
    return n, ts_a, o_a, c_a, h_a, l_a, bv_a, sv_a, oi_a

n_1h, ts_1h, o_1h, c_1h, h_1h, l_1h, bv_1h, sv_1h, oi_1h = aggregate(12)
n_4h, ts_4h, o_4h, c_4h, h_4h, l_4h, bv_4h, sv_4h, oi_4h = aggregate(48)
n_8h, ts_8h, o_8h, c_8h, h_8h, l_8h, bv_8h, sv_8h, oi_8h = aggregate(96)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def ema(data, span):
    k = 2.0 / (span + 1)
    out = np.zeros_like(data, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

def compute_rsi(data, period):
    rsi = np.full(len(data), 50.0)
    gains = np.zeros(len(data)); losses = np.zeros(len(data))
    for i in range(1, len(data)):
        d = data[i] - data[i-1]
        if d > 0: gains[i] = d
        else: losses[i] = -d
    avg_gain = ema(gains, period); avg_loss = ema(losses, period)
    for i in range(period, len(data)):
        if avg_loss[i] > 1e-15:
            rsi[i] = 100 - 100 / (1 + avg_gain[i] / avg_loss[i])
    return rsi

def cvd_zscore(buy_vol, sell_vol, window):
    cvd_bar = buy_vol - sell_vol
    cvd_ema = ema(cvd_bar, window)
    std_arr = np.zeros(len(buy_vol))
    for i in range(window, len(buy_vol)):
        diff = cvd_bar[i-window:i] - cvd_ema[i-window:i]
        std_arr[i] = np.sqrt(np.mean(diff**2))
    z = np.zeros(len(buy_vol))
    for i in range(window, len(buy_vol)):
        if std_arr[i] > 1e-15:
            z[i] = (cvd_bar[i] - cvd_ema[i]) / std_arr[i]
    return z

def combined_scoring(close, bv, sv, mr1, mr2, rsi_p, cvd_w):
    n = len(close)
    e1 = ema(close, mr1); e2 = ema(close, mr2)
    mr1s = np.where(close > e1, -1.0, 1.0)
    mr2s = np.where(close > e2, -1.0, 1.0)
    rsi = compute_rsi(close, rsi_p)
    cvd_z = cvd_zscore(bv, sv, cvd_w)
    score = np.zeros(n)
    score += mr1s * 0.9
    agree = (mr1s == mr2s)
    score += np.where(agree, mr1s * 0.6, 0)
    score += np.where(((rsi < 35) & (mr1s > 0)) | ((rsi > 65) & (mr1s < 0)), mr1s * 1.0, 0)
    score += np.where(cvd_z > 0.5, 1.0, np.where(cvd_z < -0.5, -1.0, 0))
    score += np.where(agree & (np.abs(cvd_z) > 0.5) & (np.sign(cvd_z) == mr1s), mr1s * 1.4, 0)
    return score

# ═══════════════════════════════════════════════════════════════
# SCORES + OUTCOMES + CORRECT/WRONG
# ═══════════════════════════════════════════════════════════════
score_1h = combined_scoring(c_1h, bv_1h, sv_1h, 32, 56, 18, 24)
score_4h = combined_scoring(c_4h, bv_4h, sv_4h, 8, 14, 5, 6)
score_8h = combined_scoring(c_8h, bv_8h, sv_8h, 4, 7, 4, 4)

K = 2
outcomes_1h = np.full(n_1h, -1, dtype=np.int8)
for i in range(n_1h - K): outcomes_1h[i] = 1 if c_1h[i+K] > c_1h[i] else 0
outcomes_4h = np.full(n_4h, -1, dtype=np.int8)
for i in range(n_4h - K): outcomes_4h[i] = 1 if c_4h[i+K] > c_4h[i] else 0
outcomes_8h = np.full(n_8h, -1, dtype=np.int8)
for i in range(n_8h - K): outcomes_8h[i] = 1 if c_8h[i+K] > c_8h[i] else 0

dir_1h = np.sign(score_1h)
dir_4h = np.sign(score_4h)
dir_8h = np.sign(score_8h)

# Map to 1H
correct_1h = np.full(n_1h, False)
valid_1h = (outcomes_1h >= 0) & (dir_1h != 0)
correct_1h[valid_1h] = (dir_1h[valid_1h] > 0) == (outcomes_1h[valid_1h] == 1)

correct_4h_on_1h = np.full(n_1h, False)
for i in range(n_4h):
    s, e = i*4, min(i*4+4, n_1h)
    if outcomes_4h[i] >= 0 and dir_4h[i] != 0:
        correct_4h_on_1h[s:e] = (dir_4h[i] > 0) == (outcomes_4h[i] == 1)

correct_8h_on_1h = np.full(n_1h, False)
for i in range(n_8h):
    s, e = i*8, min(i*8+8, n_1h)
    if outcomes_8h[i] >= 0 and dir_8h[i] != 0:
        correct_8h_on_1h[s:e] = (dir_8h[i] > 0) == (outcomes_8h[i] == 1)

active = valid_1h & (np.repeat(dir_4h, 4)[:n_1h] != 0) & (np.repeat(dir_8h, 8)[:n_1h] != 0)

# ═══════════════════════════════════════════════════════════════
# DATA FEATURES — her 1H bar için ölçülebilir özellikler
# ═══════════════════════════════════════════════════════════════
print("Computing data features...")

# 1. Volatility: ATR / close
atr_1h = (h_1h - l_1h)
atr_pct = atr_1h / (c_1h + 1e-10)

# 2. Rolling volatility (son 12 bar = 12H)
vol_12h = np.zeros(n_1h)
for i in range(12, n_1h):
    vol_12h[i] = np.std(np.diff(np.log(c_1h[i-12:i+1] + 1e-10)))

# 3. Volume: buy/sell ratio
vol_ratio = bv_1h / (sv_1h + 1e-10)

# 4. Volume relative to recent average
vol_rel = np.zeros(n_1h)
for i in range(24, n_1h):
    avg = np.mean(bv_1h[i-24:i] + sv_1h[i-24:i])
    if avg > 0: vol_rel[i] = (bv_1h[i] + sv_1h[i]) / avg

# 5. Trend strength: EMA distance / ATR
ema_20 = ema(c_1h, 20)
trend_str = (c_1h - ema_20) / (atr_1h.clip(1e-10) + 1e-10)

# 6. RSI
rsi_1h = compute_rsi(c_1h, 14)

# 7. CVD z-score
cvd_z = cvd_zscore(bv_1h, sv_1h, 24)

# 8. OI change
oi_change = np.zeros(n_1h)
for i in range(1, n_1h):
    if oi_1h[i-1] > 0: oi_change[i] = (oi_1h[i] - oi_1h[i-1]) / oi_1h[i-1]

# 9. Bar body ratio (close-open) / range
body_ratio = np.zeros(n_1h)
for i in range(n_1h):
    rng = h_1h[i] - l_1h[i]
    if rng > 0: body_ratio[i] = (c_1h[i] - o_1h[i]) / rng

# 10. Consecutive same-direction bars
consec = np.zeros(n_1h)
for i in range(1, n_1h):
    if (c_1h[i] > c_1h[i-1]) == (c_1h[i-1] > c_1h[max(0,i-2)]):
        consec[i] = consec[i-1] + 1

# 11. Recent return (son 4 bar)
ret_4 = np.zeros(n_1h)
for i in range(4, n_1h):
    if c_1h[i-4] > 0: ret_4[i] = (c_1h[i] - c_1h[i-4]) / c_1h[i-4]

# 12. Wick ratio (upper wick - lower wick) / range
wick_ratio = np.zeros(n_1h)
for i in range(n_1h):
    rng = h_1h[i] - l_1h[i]
    if rng > 0:
        upper = h_1h[i] - max(c_1h[i], o_1h[i])
        lower = min(c_1h[i], o_1h[i]) - l_1h[i]
        wick_ratio[i] = (upper - lower) / rng

# 13. Hour of day
hours = ((ts_1h // 1000) % 86400) // 3600

# ═══════════════════════════════════════════════════════════════
# PATTERN ANALYSIS: Compare feature distributions
# ═══════════════════════════════════════════════════════════════
features = {
    'atr_pct': atr_pct,
    'vol_12h': vol_12h,
    'vol_ratio': vol_ratio,
    'vol_rel': vol_rel,
    'trend_str': trend_str,
    'rsi': rsi_1h,
    'cvd_z': cvd_z,
    'oi_change': oi_change,
    'body_ratio': body_ratio,
    'consec': consec,
    'ret_4': ret_4,
    'wick_ratio': wick_ratio,
    'hour': hours.astype(float),
}

def compare_groups(name_a, mask_a, name_b, mask_b):
    """Compare feature distributions between two groups."""
    print(f"\n  {name_a} ({mask_a.sum():,}) vs {name_b} ({mask_b.sum():,})")
    print(f"  {'Feature':>15s}  {'Mean A':>8s}  {'Mean B':>8s}  {'Diff':>8s}  {'Std A':>8s}  {'Std B':>8s}  {'Effect':>8s}")
    print(f"  {'-'*70}")

    results = []
    for fname, fdata in features.items():
        va = fdata[mask_a]
        vb = fdata[mask_b]
        # Remove NaN/inf
        va = va[np.isfinite(va)]
        vb = vb[np.isfinite(vb)]
        if len(va) < 100 or len(vb) < 100:
            continue
        ma, mb = va.mean(), vb.mean()
        sa, sb = va.std(), vb.std()
        pooled_std = np.sqrt((sa**2 + sb**2) / 2) if (sa + sb) > 0 else 1
        effect = (ma - mb) / pooled_std if pooled_std > 1e-10 else 0
        results.append((abs(effect), fname, ma, mb, ma-mb, sa, sb, effect))

    # Sort by effect size
    results.sort(reverse=True)
    for _, fname, ma, mb, diff, sa, sb, effect in results:
        marker = " ***" if abs(effect) > 0.10 else (" **" if abs(effect) > 0.05 else (" *" if abs(effect) > 0.02 else ""))
        print(f"  {fname:>15s}  {ma:>8.4f}  {mb:>8.4f}  {diff:>+8.4f}  {sa:>8.4f}  {sb:>8.4f}  {effect:>+8.4f}{marker}")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 1: 1H doğru vs yanlış — data nasıl farklı?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  ANALYSIS 1: 1H doğru vs yanlış")
print(f"{'='*70}")
compare_groups("1H doğru", active & correct_1h, "1H yanlış", active & ~correct_1h)

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 2: 1H yanlış + 4H kurtarıyor vs kurtarmıyor
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  ANALYSIS 2: 1H yanlış → 4H kurtarıyor vs kurtarmıyor")
print(f"{'='*70}")
wrong_1h = active & ~correct_1h
compare_groups("4H kurtarıyor", wrong_1h & correct_4h_on_1h, "4H de yanlış", wrong_1h & ~correct_4h_on_1h)

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 3: 1H yanlış + 8H kurtarıyor vs kurtarmıyor
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  ANALYSIS 3: 1H yanlış → 8H kurtarıyor vs kurtarmıyor")
print(f"{'='*70}")
compare_groups("8H kurtarıyor", wrong_1h & correct_8h_on_1h, "8H de yanlış", wrong_1h & ~correct_8h_on_1h)

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 4: Üçü birden doğru vs üçü birden yanlış
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  ANALYSIS 4: Üçü birden doğru vs üçü birden yanlış")
print(f"{'='*70}")
all_correct = active & correct_1h & correct_4h_on_1h & correct_8h_on_1h
all_wrong = active & ~correct_1h & ~correct_4h_on_1h & ~correct_8h_on_1h
compare_groups("Hepsi doğru", all_correct, "Hepsi yanlış", all_wrong)

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 5: Feature quintile breakdown
# En etkili feature'ları quintile'lara böl, accuracy farkını göster
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  ANALYSIS 5: Feature Quintile Breakdown (en etkili feature'lar)")
print(f"{'='*70}")

# Top features to deep dive
top_features = ['vol_12h', 'atr_pct', 'trend_str', 'cvd_z', 'rsi', 'vol_ratio', 'consec', 'ret_4', 'oi_change', 'body_ratio']

for fname in top_features:
    fdata = features[fname]
    valid_f = active & np.isfinite(fdata)
    if valid_f.sum() < 1000:
        continue

    vals = fdata[valid_f]
    percentiles = np.percentile(vals, [20, 40, 60, 80])

    print(f"\n  {fname}:")
    print(f"  {'Quintile':>10s}  {'Range':>20s}  {'1H acc':>8s}  {'4H acc':>8s}  {'8H acc':>8s}  {'Best':>5s}  {'n':>6s}")
    print(f"  {'-'*70}")

    bounds = [(-np.inf, percentiles[0]), (percentiles[0], percentiles[1]),
              (percentiles[1], percentiles[2]), (percentiles[2], percentiles[3]),
              (percentiles[3], np.inf)]

    for qi, (lo, hi) in enumerate(bounds):
        qmask = valid_f & (fdata > lo) & (fdata <= hi)
        if qi == 0:
            qmask = valid_f & (fdata <= hi)
        nq = qmask.sum()
        if nq < 100:
            continue
        a1 = correct_1h[qmask].mean()
        a4 = correct_4h_on_1h[qmask].mean()
        a8 = correct_8h_on_1h[qmask].mean()
        best = "1H" if a1 >= a4 and a1 >= a8 else ("4H" if a4 >= a8 else "8H")

        lo_s = f"{lo:.4f}" if lo > -np.inf else "-inf"
        hi_s = f"{hi:.4f}" if hi < np.inf else "+inf"
        print(f"  Q{qi+1:>8d}  ({lo_s:>8s}, {hi_s:>8s}]  {a1:.4f}  {a4:.4f}  {a8:.4f}  {best:>5s}  {nq:>6,}")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 6: 2D interaction — en etkili 2 feature'ın cross'u
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  ANALYSIS 6: 2D Interaction (volatility × trend)")
print(f"{'='*70}")

# Volatility terciles
vol_valid = active & np.isfinite(vol_12h)
vol_p33 = np.percentile(vol_12h[vol_valid], 33)
vol_p66 = np.percentile(vol_12h[vol_valid], 66)

# Trend terciles
trend_valid = active & np.isfinite(trend_str)
trend_p33 = np.percentile(trend_str[trend_valid], 33)
trend_p66 = np.percentile(trend_str[trend_valid], 66)

print(f"  {'Vol':>10s}  {'Trend':>10s}  {'1H acc':>8s}  {'4H acc':>8s}  {'8H acc':>8s}  {'Best':>5s}  {'n':>6s}")
print(f"  {'-'*60}")

for vname, vlo, vhi in [("Low", -np.inf, vol_p33), ("Mid", vol_p33, vol_p66), ("High", vol_p66, np.inf)]:
    for tname, tlo, thi in [("Bear", -np.inf, trend_p33), ("Neutral", trend_p33, trend_p66), ("Bull", trend_p66, np.inf)]:
        mask = vol_valid & trend_valid
        if vlo == -np.inf:
            mask &= (vol_12h <= vhi)
        elif vhi == np.inf:
            mask &= (vol_12h > vlo)
        else:
            mask &= (vol_12h > vlo) & (vol_12h <= vhi)

        if tlo == -np.inf:
            mask &= (trend_str <= thi)
        elif thi == np.inf:
            mask &= (trend_str > tlo)
        else:
            mask &= (trend_str > tlo) & (trend_str <= thi)

        n = mask.sum()
        if n < 100:
            continue
        a1 = correct_1h[mask].mean()
        a4 = correct_4h_on_1h[mask].mean()
        a8 = correct_8h_on_1h[mask].mean()
        best = "1H" if a1 >= a4 and a1 >= a8 else ("4H" if a4 >= a8 else "8H")
        print(f"  {vname:>10s}  {tname:>10s}  {a1:.4f}  {a4:.4f}  {a8:.4f}  {best:>5s}  {n:>6,}")

print(f"\nDone!")
