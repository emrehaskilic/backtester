"""
HTF Trend Confirmation: Use 4H/8H as FILTERS for 1H combined scoring.

Approach: 1H combined scoring is the base signal (%53 OOS).
Higher TF trend direction acts as confirmation:
  - 1H long + 4H uptrend → boost
  - 1H long + 4H downtrend → dampen
  - Always %100 coverage (never skip, just re-weight)
"""
import time
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

PERIOD = 12
n_1h = len(df) // PERIOD
ts_1h = np.zeros(n_1h, dtype=np.uint64)
o_1h = np.zeros(n_1h); h_1h = np.zeros(n_1h); l_1h = np.zeros(n_1h); c_1h = np.zeros(n_1h)
bv_1h = np.zeros(n_1h); sv_1h = np.zeros(n_1h); oi_1h = np.zeros(n_1h)

for i in range(n_1h):
    s, e = i * PERIOD, i * PERIOD + PERIOD
    ts_1h[i] = ts[s]; o_1h[i] = o[s]; h_1h[i] = h[s:e].max()
    l_1h[i] = l[s:e].min(); c_1h[i] = c[e - 1]
    bv_1h[i] = bv[s:e].sum(); sv_1h[i] = sv[s:e].sum(); oi_1h[i] = oi[e - 1]

# Also build 4H and 8H close for trend computation
n_4h = len(df) // 48
c_4h = np.zeros(n_4h)
for i in range(n_4h):
    c_4h[i] = c[i * 48 + 47]

n_8h = len(df) // 96
c_8h = np.zeros(n_8h)
for i in range(n_8h):
    c_8h[i] = c[i * 96 + 95]

print(f"1H: {n_1h:,} | 4H: {n_4h:,} | 8H: {n_8h:,}")

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
        if avg_loss[i] > 1e-15:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - 100 / (1 + rs)
    return rsi

def cvd_zscore(buy_vol, sell_vol, window=24):
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

# ═══════════════════════════════════════════════════════════════
# 1H COMBINED SCORING (TPE optimal — baseline)
# ═══════════════════════════════════════════════════════════════
def compute_1h_score():
    ema1 = ema(c_1h, 32)
    ema2 = ema(c_1h, 56)
    mr1_sign = np.where(c_1h > ema1, -1.0, 1.0)
    mr2_sign = np.where(c_1h > ema2, -1.0, 1.0)
    rsi = compute_rsi(c_1h, 18)
    cvd_z = cvd_zscore(bv_1h, sv_1h, 24)

    score = np.zeros(n_1h)
    score += mr1_sign * 0.9       # MR primary
    agree_mr = (mr1_sign == mr2_sign)
    score += np.where(agree_mr, mr1_sign * 0.6, 0)  # MR secondary
    rsi_oversold = rsi < 35
    rsi_overbought = rsi > 65
    rsi_confirm = ((rsi_oversold & (mr1_sign > 0)) | (rsi_overbought & (mr1_sign < 0)))
    score += np.where(rsi_confirm, mr1_sign * 1.0, 0)  # RSI confirm
    score += np.where(cvd_z > 0.5, 1.0, np.where(cvd_z < -0.5, -1.0, 0))  # CVD
    score += np.where(agree_mr & (np.abs(cvd_z) > 0.5) & (np.sign(cvd_z) == mr1_sign),
                      mr1_sign * 1.4, 0)  # Agreement bonus
    return score

score_1h = compute_1h_score()

# ═══════════════════════════════════════════════════════════════
# HTF TREND FEATURES (mapped to 1H bars)
# ═══════════════════════════════════════════════════════════════

def compute_htf_trend_on_1h(htf_close, htf_n, ratio, ema_spans):
    """Compute HTF trend directions mapped onto 1H bars.
    Returns dict of trend features on 1H timebase."""
    features = {}
    for span in ema_spans:
        ema_htf = ema(htf_close, span)
        # Trend direction: price above EMA = uptrend (+1), below = downtrend (-1)
        trend_htf = np.where(htf_close > ema_htf, 1.0, -1.0)
        # Trend strength: distance from EMA as % of price
        strength_htf = (htf_close - ema_htf) / (ema_htf + 1e-10)

        # Map to 1H
        trend_1h = np.zeros(n_1h)
        strength_1h = np.zeros(n_1h)
        for i in range(htf_n):
            start = i * ratio
            end = min(start + ratio, n_1h)
            trend_1h[start:end] = trend_htf[i]
            strength_1h[start:end] = strength_htf[i]

        features[f'trend_{span}'] = trend_1h
        features[f'strength_{span}'] = strength_1h

    # Momentum: ROC over N bars
    for lookback in [3, 6]:
        mom = np.zeros(htf_n)
        for i in range(lookback, htf_n):
            if htf_close[i - lookback] > 0:
                mom[i] = (htf_close[i] - htf_close[i - lookback]) / htf_close[i - lookback]
        mom_1h = np.zeros(n_1h)
        for i in range(htf_n):
            start = i * ratio
            end = min(start + ratio, n_1h)
            mom_1h[start:end] = mom[i]
        features[f'mom_{lookback}'] = np.sign(mom_1h)

    return features

# 4H trend features
htf_4h = compute_htf_trend_on_1h(c_4h, n_4h, 4, ema_spans=[5, 10, 20])
# 8H trend features
htf_8h = compute_htf_trend_on_1h(c_8h, n_8h, 8, ema_spans=[3, 6, 12])

# ═══════════════════════════════════════════════════════════════
# OUTCOMES
# ═══════════════════════════════════════════════════════════════
K = 2
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
outcomes[-K:] = -1
valid = outcomes >= 0

# ═══════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════
def evaluate(name, score, split_pct=50):
    """Evaluate with walk-forward OOS."""
    direction = np.sign(score)
    neutral = direction == 0
    direction[neutral] = 1  # default bullish for coverage

    # In-sample (all data)
    non_neutral = valid & (direction != 0)
    pred = direction[non_neutral] > 0
    actual = outcomes[non_neutral] == 1
    is_acc = (pred == actual).mean()

    # OOS: train first split_pct%, test rest in chunks
    train_end = int(n_1h * split_pct / 100)
    chunk_size = n_1h // 10
    chunk_accs = []
    idx = train_end
    while idx + chunk_size <= n_1h:
        mask = np.zeros(n_1h, dtype=bool)
        mask[idx:idx+chunk_size] = True
        cv = mask & valid & (direction != 0)
        if cv.sum() > 100:
            p = direction[cv] > 0
            a = outcomes[cv] == 1
            chunk_accs.append((p == a).mean())
        idx += chunk_size

    oos_acc = np.mean(chunk_accs) if chunk_accs else 0.5
    oos_std = np.std(chunk_accs) if chunk_accs else 0
    n_chunks = len(chunk_accs)

    print(f"  {name:40s}: IS={is_acc:.4f}  OOS={oos_acc:.4f} ({oos_acc*100:.2f}%) ±{oos_std:.4f}  chunks={n_chunks}")
    return oos_acc

print(f"\n{'='*60}")
print(f"  BASELINE")
print(f"{'='*60}")
evaluate("1H combined scoring (baseline)", score_1h.copy())

# ═══════════════════════════════════════════════════════════════
# STRATEGY 1: HTF TREND CONFIRMATION (boost/dampen)
# When 1H signal aligns with HTF trend → boost
# When 1H signal conflicts with HTF trend → dampen (but don't flip)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STRATEGY 1: HTF Trend Confirmation (boost/dampen)")
print(f"{'='*60}")

dir_1h = np.sign(score_1h)

for tf_name, htf_feats in [("4H", htf_4h), ("8H", htf_8h)]:
    for feat_name, trend in htf_feats.items():
        if not feat_name.startswith('trend_') and not feat_name.startswith('mom_'):
            continue
        for boost, dampen in [(1.5, 0.5), (2.0, 0.5), (1.5, 0.7), (2.0, 0.3)]:
            aligned = (dir_1h == trend)
            conflict = (dir_1h == -trend)

            modified = score_1h.copy()
            modified[aligned] *= boost
            modified[conflict] *= dampen

            name = f"{tf_name} {feat_name} b={boost} d={dampen}"
            evaluate(name, modified)

# ═══════════════════════════════════════════════════════════════
# STRATEGY 2: HTF TREND AS ADDITIONAL SCORE COMPONENT
# Add HTF trend direction as a weighted component
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STRATEGY 2: HTF Trend as Score Component")
print(f"{'='*60}")

for tf_name, htf_feats in [("4H", htf_4h), ("8H", htf_8h)]:
    for feat_name in ['trend_10', 'trend_20', 'mom_3', 'mom_6'] if tf_name == '4H' else ['trend_6', 'trend_12', 'mom_3', 'mom_6']:
        if feat_name not in htf_feats:
            continue
        trend = htf_feats[feat_name]
        for w in [0.3, 0.5, 0.8, 1.0]:
            modified = score_1h + w * trend
            name = f"{tf_name} {feat_name} w={w}"
            evaluate(name, modified.copy())

# ═══════════════════════════════════════════════════════════════
# STRATEGY 3: MULTI-HTF COMBINATION
# Use both 4H and 8H trend together
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STRATEGY 3: Multi-HTF (4H + 8H) Combinations")
print(f"{'='*60}")

for feat_4h_name in ['trend_10', 'mom_3']:
    for feat_8h_name in ['trend_6', 'mom_3']:
        if feat_4h_name not in htf_4h or feat_8h_name not in htf_8h:
            continue
        t4 = htf_4h[feat_4h_name]
        t8 = htf_8h[feat_8h_name]

        for w4, w8 in [(0.3, 0.3), (0.3, 0.5), (0.5, 0.3), (0.5, 0.5), (0.5, 0.8)]:
            modified = score_1h + w4 * t4 + w8 * t8
            name = f"4H:{feat_4h_name}({w4})+8H:{feat_8h_name}({w8})"
            evaluate(name, modified.copy())

# ═══════════════════════════════════════════════════════════════
# STRATEGY 4: CONDITIONAL SWITCHING
# Use 1H by default, switch to HTF-dampened when HTF disagrees strongly
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STRATEGY 4: Conditional HTF Override")
print(f"{'='*60}")

for tf_name, htf_feats in [("4H", htf_4h), ("8H", htf_8h)]:
    for feat_name in ['strength_10', 'strength_20'] if tf_name == '4H' else ['strength_6', 'strength_12']:
        if feat_name not in htf_feats:
            continue
        strength = htf_feats[feat_name]
        trend_key = feat_name.replace('strength_', 'trend_')
        if trend_key not in htf_feats:
            continue
        trend = htf_feats[trend_key]

        for strength_thresh in [0.005, 0.01, 0.02]:
            modified = score_1h.copy()
            # When HTF has strong trend AND conflicts with 1H → dampen
            strong_conflict = (dir_1h != 0) & (dir_1h == -trend) & (np.abs(strength) > strength_thresh)
            modified[strong_conflict] *= 0.3  # heavy dampen

            # When HTF has strong trend AND agrees with 1H → boost
            strong_agree = (dir_1h != 0) & (dir_1h == trend) & (np.abs(strength) > strength_thresh)
            modified[strong_agree] *= 1.5  # moderate boost

            name = f"{tf_name} {feat_name} thresh={strength_thresh}"
            evaluate(name, modified)

# ═══════════════════════════════════════════════════════════════
# STRATEGY 5: Hour-of-day filter
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STRATEGY 5: Hour-of-day Adjustment")
print(f"{'='*60}")

hours_1h = ((ts_1h // 1000) % 86400) // 3600

# Compute per-hour accuracy in-sample (first 50%) to find good/bad hours
train_end = n_1h // 2
hour_acc = {}
for hour in range(24):
    mask = (hours_1h[:train_end] == hour) & valid[:train_end] & (np.sign(score_1h[:train_end]) != 0)
    if mask.sum() > 100:
        pred = np.sign(score_1h[:train_end])[mask] > 0
        actual = outcomes[:train_end][mask] == 1
        hour_acc[hour] = (pred == actual).mean()

if hour_acc:
    mean_acc = np.mean(list(hour_acc.values()))
    print(f"  Per-hour accuracy (train set):")
    for hour in sorted(hour_acc.keys()):
        marker = "+" if hour_acc[hour] > mean_acc + 0.01 else ("-" if hour_acc[hour] < mean_acc - 0.01 else " ")
        print(f"    {hour:02d}:00  {hour_acc[hour]:.4f} {marker}")

    # Strategy: boost good hours, dampen bad hours
    for boost, dampen in [(1.3, 0.7), (1.5, 0.5), (2.0, 0.3)]:
        modified = score_1h.copy()
        for hour in range(24):
            if hour not in hour_acc:
                continue
            mask = hours_1h == hour
            if hour_acc[hour] > mean_acc + 0.005:
                modified[mask] *= boost
            elif hour_acc[hour] < mean_acc - 0.005:
                modified[mask] *= dampen
        name = f"Hour filter b={boost} d={dampen}"
        evaluate(name, modified)

# ═══════════════════════════════════════════════════════════════
# SUMMARY: Top 10 strategies by OOS accuracy
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  Done! Check results above for best OOS accuracy.")
print(f"{'='*60}")
