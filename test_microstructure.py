"""Microstructure LightGBM — 30s tick data -> 1H features -> direction prediction
Extracts intra-bar order flow patterns invisible at bar level.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 1. LOAD 30s DATA
# ====================================================================
print("Loading 30s data...", flush=True)
df = pd.read_parquet("data/ETHUSDT_30s_11mo.parquet")
print(f"  30s bars: {len(df)}")

SUB = 120  # 30s per 1H
n_1h = len(df) // SUB
print(f"  1H bars: {n_1h} ({n_1h/(24*30):.1f} months)")

ot  = df['open_time'].values
op  = df['open'].values
hi  = df['high'].values
lo  = df['low'].values
cl  = df['close'].values
vol = df['volume'].values
bv  = df['buy_vol'].values
sv  = df['sell_vol'].values
tc  = df['trade_count'].values

# ====================================================================
# 2. FEATURE ENGINEERING: 1H bars from 30s sub-bars
# ====================================================================
print("Computing microstructure features...", flush=True)
t0 = time()

# Pre-allocate all features
feat = {}

# Standard 1H OHLCV (for baseline features)
ts_1h   = np.zeros(n_1h, dtype=np.uint64)
o_1h    = np.zeros(n_1h)
h_1h    = np.zeros(n_1h)
l_1h    = np.zeros(n_1h)
c_1h    = np.zeros(n_1h)
bv_1h   = np.zeros(n_1h)
sv_1h   = np.zeros(n_1h)
vol_1h  = np.zeros(n_1h)
tc_1h   = np.zeros(n_1h)

# Microstructure features (per 1H bar)
# A. Order flow
f_cvd_trend        = np.zeros(n_1h)  # CVD trajectory: correlation with time
f_cvd_reversal     = np.zeros(n_1h)  # CVD reversal count within bar
f_cvd_max_streak   = np.zeros(n_1h)  # longest consecutive buy/sell streak
f_cvd_first_half   = np.zeros(n_1h)  # CVD in first 60 sub-bars
f_cvd_second_half  = np.zeros(n_1h)  # CVD in last 60 sub-bars
f_cvd_acceleration = np.zeros(n_1h)  # 2nd half CVD - 1st half CVD

# B. Volume distribution
f_vol_gini         = np.zeros(n_1h)  # volume concentration (0=even, 1=concentrated)
f_vol_max_ratio    = np.zeros(n_1h)  # max sub-bar vol / mean vol
f_vol_spike_count  = np.zeros(n_1h)  # sub-bars with vol > 3*median
f_vol_top5_pct     = np.zeros(n_1h)  # % of volume in top 5 sub-bars

# C. Large trade detection
f_large_buy_vol    = np.zeros(n_1h)  # volume from large-buy sub-bars
f_large_sell_vol   = np.zeros(n_1h)  # volume from large-sell sub-bars
f_large_imbalance  = np.zeros(n_1h)  # (large_buy - large_sell) / total

# D. Price impact / absorption
f_price_impact     = np.zeros(n_1h)  # price move per unit volume
f_absorption       = np.zeros(n_1h)  # high vol but small price move (count)
f_vwap_position    = np.zeros(n_1h)  # close vs intra-bar VWAP

# E. Momentum within bar
f_first_half_ret   = np.zeros(n_1h)  # first 60 bars return
f_second_half_ret  = np.zeros(n_1h)  # last 60 bars return
f_intrabar_reversal= np.zeros(n_1h)  # first half up, second half down (or vice versa)

# F. Trade count features
f_tc_ratio         = np.zeros(n_1h)  # trade_count / volume (small trades indicator)
f_tc_spike         = np.zeros(n_1h)  # max trade_count sub-bar / median
f_tc_vol_corr      = np.zeros(n_1h)  # correlation between trade_count and volume

# G. Aggressor patterns
f_buy_cluster      = np.zeros(n_1h)  # max consecutive buy-dominated sub-bars
f_sell_cluster     = np.zeros(n_1h)  # max consecutive sell-dominated sub-bars
f_flip_count       = np.zeros(n_1h)  # how many times dominance flips

time_axis = np.arange(SUB, dtype=float)
half = SUB // 2

for i in range(n_1h):
    s, e = i * SUB, i * SUB + SUB

    # Sub-bar arrays
    sub_bv  = bv[s:e]
    sub_sv  = sv[s:e]
    sub_vol = vol[s:e]
    sub_cl  = cl[s:e]
    sub_hi  = hi[s:e]
    sub_lo  = lo[s:e]
    sub_op  = op[s:e]
    sub_tc  = tc[s:e]
    sub_cvd = sub_bv - sub_sv  # per-sub-bar CVD
    cum_cvd = np.cumsum(sub_cvd)

    # Standard 1H
    ts_1h[i]  = ot[s]
    o_1h[i]   = op[s]
    h_1h[i]   = sub_hi.max()
    l_1h[i]   = sub_lo.min()
    c_1h[i]   = sub_cl[-1]
    bv_1h[i]  = sub_bv.sum()
    sv_1h[i]  = sub_sv.sum()
    vol_1h[i] = sub_vol.sum()
    tc_1h[i]  = sub_tc.sum()

    total_vol = vol_1h[i]
    if total_vol < 1e-10:
        continue

    # ── A. Order Flow ──
    # CVD trend: correlation of cumulative CVD with time
    if cum_cvd.std() > 1e-10:
        f_cvd_trend[i] = np.corrcoef(time_axis, cum_cvd)[0, 1]

    # CVD reversals: count sign changes in cum_cvd
    cvd_sign = np.sign(sub_cvd)
    sign_changes = np.sum(np.abs(np.diff(cvd_sign)) > 0)
    f_cvd_reversal[i] = sign_changes / (SUB - 1)

    # Max buy/sell streak
    buy_dom = (sub_bv > sub_sv).astype(int)
    max_buy = max_sell = cur_buy = cur_sell = 0
    for j in range(SUB):
        if buy_dom[j]:
            cur_buy += 1; cur_sell = 0
        else:
            cur_sell += 1; cur_buy = 0
        max_buy = max(max_buy, cur_buy)
        max_sell = max(max_sell, cur_sell)
    f_cvd_max_streak[i] = (max_buy - max_sell) / SUB

    # CVD halves
    f_cvd_first_half[i]  = sub_cvd[:half].sum() / total_vol
    f_cvd_second_half[i] = sub_cvd[half:].sum() / total_vol
    f_cvd_acceleration[i] = f_cvd_second_half[i] - f_cvd_first_half[i]

    # ── B. Volume Distribution ──
    sorted_vol = np.sort(sub_vol)
    cum_vol = np.cumsum(sorted_vol)
    n_sub = len(sorted_vol)
    # Gini coefficient
    gini_num = 2 * np.sum((np.arange(1, n_sub+1)) * sorted_vol) - (n_sub + 1) * cum_vol[-1]
    f_vol_gini[i] = gini_num / (n_sub * cum_vol[-1]) if cum_vol[-1] > 0 else 0

    med_vol = np.median(sub_vol)
    mean_vol = sub_vol.mean()
    f_vol_max_ratio[i] = sub_vol.max() / mean_vol if mean_vol > 0 else 0
    f_vol_spike_count[i] = np.sum(sub_vol > 3 * med_vol) / SUB if med_vol > 0 else 0
    # Top 5 sub-bars % of total
    top5 = np.sort(sub_vol)[-5:].sum()
    f_vol_top5_pct[i] = top5 / total_vol

    # ── C. Large Trade Detection ──
    vol_threshold = np.percentile(sub_vol, 90)
    large_mask = sub_vol > vol_threshold
    f_large_buy_vol[i]  = sub_bv[large_mask].sum() / total_vol
    f_large_sell_vol[i] = sub_sv[large_mask].sum() / total_vol
    f_large_imbalance[i] = f_large_buy_vol[i] - f_large_sell_vol[i]

    # ── D. Price Impact / Absorption ──
    bar_range = h_1h[i] - l_1h[i]
    f_price_impact[i] = bar_range / total_vol if total_vol > 0 else 0

    # Absorption: sub-bars with high volume but small price move
    sub_range = sub_hi - sub_lo
    sub_range_med = np.median(sub_range) if np.median(sub_range) > 0 else 1e-10
    sub_vol_med = med_vol if med_vol > 0 else 1e-10
    absorb_mask = (sub_vol > 2 * sub_vol_med) & (sub_range < 0.5 * sub_range_med)
    f_absorption[i] = absorb_mask.sum() / SUB

    # VWAP
    vwap = np.sum(sub_cl * sub_vol) / total_vol if total_vol > 0 else c_1h[i]
    f_vwap_position[i] = (c_1h[i] - vwap) / (bar_range + 1e-10)

    # ── E. Momentum Within Bar ──
    mid_close = sub_cl[half - 1]
    f_first_half_ret[i]  = (mid_close - o_1h[i]) / o_1h[i] if o_1h[i] > 0 else 0
    f_second_half_ret[i] = (c_1h[i] - mid_close) / mid_close if mid_close > 0 else 0
    # Reversal: first half and second half have opposite signs
    f_intrabar_reversal[i] = -1.0 if (f_first_half_ret[i] * f_second_half_ret[i] < 0) else 1.0

    # ── F. Trade Count ──
    f_tc_ratio[i] = tc_1h[i] / total_vol if total_vol > 0 else 0
    tc_med = np.median(sub_tc) if np.median(sub_tc) > 0 else 1
    f_tc_spike[i] = sub_tc.max() / tc_med
    if sub_tc.std() > 0 and sub_vol.std() > 0:
        f_tc_vol_corr[i] = np.corrcoef(sub_tc.astype(float), sub_vol)[0, 1]

    # ── G. Aggressor Patterns ──
    buy_dominant = sub_bv > sub_sv
    max_bc = max_sc = cb = cs = 0
    flips = 0
    prev = None
    for j in range(SUB):
        cur = buy_dominant[j]
        if cur:
            cb += 1; cs = 0
        else:
            cs += 1; cb = 0
        max_bc = max(max_bc, cb)
        max_sc = max(max_sc, cs)
        if prev is not None and cur != prev:
            flips += 1
        prev = cur
    f_buy_cluster[i]  = max_bc / SUB
    f_sell_cluster[i] = max_sc / SUB
    f_flip_count[i]   = flips / (SUB - 1)

elapsed_feat = time() - t0
print(f"  Feature computation: {elapsed_feat:.1f}s")

# ====================================================================
# 3. BUILD FEATURE MATRIX
# ====================================================================

def ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values

def rolling_zscore(arr, window):
    s = pd.Series(arr)
    m = s.rolling(window, min_periods=window).mean()
    st = s.rolling(window, min_periods=window).std()
    return ((s - m) / st.replace(0, np.nan)).fillna(0).values

def rsi(close, period):
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).values

def returns(arr, k):
    r = np.zeros(len(arr))
    r[k:] = (arr[k:] - arr[:-k]) / arr[:-k]
    return r

features = {}

# ── Microstructure features (raw) ──
features['cvd_trend']        = f_cvd_trend
features['cvd_reversal']     = f_cvd_reversal
features['cvd_max_streak']   = f_cvd_max_streak
features['cvd_first_half']   = f_cvd_first_half
features['cvd_second_half']  = f_cvd_second_half
features['cvd_acceleration'] = f_cvd_acceleration
features['vol_gini']         = f_vol_gini
features['vol_max_ratio']    = f_vol_max_ratio
features['vol_spike_count']  = f_vol_spike_count
features['vol_top5_pct']     = f_vol_top5_pct
features['large_buy_vol']    = f_large_buy_vol
features['large_sell_vol']   = f_large_sell_vol
features['large_imbalance']  = f_large_imbalance
features['price_impact']     = f_price_impact
features['absorption']       = f_absorption
features['vwap_position']    = f_vwap_position
features['first_half_ret']   = f_first_half_ret
features['second_half_ret']  = f_second_half_ret
features['intrabar_reversal']= f_intrabar_reversal
features['tc_ratio']         = f_tc_ratio
features['tc_spike']         = f_tc_spike
features['tc_vol_corr']      = f_tc_vol_corr
features['buy_cluster']      = f_buy_cluster
features['sell_cluster']     = f_sell_cluster
features['flip_count']       = f_flip_count

# ── Microstructure rolling features ──
for w in [6, 12, 24]:
    features[f'cvd_trend_ema_{w}']    = ema(f_cvd_trend, w)
    features[f'large_imb_ema_{w}']    = ema(f_large_imbalance, w)
    features[f'absorption_ema_{w}']   = ema(f_absorption, w)
    features[f'vwap_pos_ema_{w}']     = ema(f_vwap_position, w)
    features[f'cvd_accel_ema_{w}']    = ema(f_cvd_acceleration, w)

# ── Standard price/volume features (baseline) ──
for w in [1, 2, 3, 6, 12, 24]:
    features[f'ret_{w}'] = returns(c_1h, w)

cvd_1h = bv_1h - sv_1h
for w in [6, 12, 24]:
    features[f'cvd_zscore_{w}'] = rolling_zscore(cvd_1h, w)
    features[f'vol_zscore_{w}'] = rolling_zscore(vol_1h, w)

for span in [8, 16, 32]:
    e = ema(c_1h, span)
    features[f'ema_dist_{span}'] = (c_1h - e) / e

features['imbalance'] = cvd_1h / (vol_1h + 1e-10)
features['imbalance_ema_12'] = ema(cvd_1h / (vol_1h + 1e-10), 12)

for p in [6, 14]:
    features[f'rsi_{p}'] = rsi(c_1h, p)

features['bar_range_pct'] = (h_1h - l_1h) / c_1h

# Time features
hour = (ts_1h % (86400_000)) / 3_600_000
features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

feature_names = sorted(features.keys())
X = np.column_stack([features[f] for f in feature_names])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Total features: {len(feature_names)}")

# Outcome: K=2 direction
K = 2
y = np.full(n_1h, -1, dtype=int)
for i in range(n_1h - K):
    y[i] = 1 if c_1h[i + K] > c_1h[i] else 0

valid_outcomes = np.sum(y >= 0)
print(f"Valid outcome bars: {valid_outcomes}")
print(f"Bull rate: {y[y>=0].mean():.3f}")

# ====================================================================
# 4. WALK-FORWARD
# ====================================================================

# With 8015 bars (~11 months):
# Train: 4000 bars (~5.5 months)
# Test chunks: ~1000 bars each (~1.4 months)
TRAIN_BARS = 4000
CHUNK_SIZE = 1000
WARMUP = 48

print(f"\nWalk-Forward: train={TRAIN_BARS}, chunk={CHUNK_SIZE}")

chunks = []
idx = TRAIN_BARS
while idx + CHUNK_SIZE <= n_1h:
    chunks.append((idx, idx + CHUNK_SIZE))
    idx += CHUNK_SIZE

print(f"Test chunks: {len(chunks)}")

t0 = time()
chunk_results = []
all_probs = np.full(n_1h, np.nan)

for ci, (test_start, test_end) in enumerate(chunks):
    train_mask = (np.arange(n_1h) >= WARMUP) & (np.arange(n_1h) < test_start) & (y >= 0)
    test_mask  = (np.arange(n_1h) >= test_start) & (np.arange(n_1h) < test_end) & (y >= 0)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test   = X[test_mask], y[test_mask]

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'seed': 42,
    }

    model = lgb.train(params, dtrain, num_boost_round=300)
    probs = model.predict(X_test)
    preds = (probs > 0.5).astype(int)
    acc = (preds == y_test).mean()

    test_indices = np.where(test_mask)[0]
    all_probs[test_indices] = probs

    chunk_results.append({'chunk': ci, 'n': len(y_test), 'acc': acc, 'bull_rate': y_test.mean()})
    print(f"  Chunk {ci}: acc={acc:.4f} (n={len(y_test)}, bull={y_test.mean():.3f})")

elapsed = time() - t0

# ====================================================================
# 5. RESULTS
# ====================================================================
print(f"\n{'='*60}")
print(f"MICROSTRUCTURE WALK-FORWARD RESULTS ({elapsed:.1f}s)")
print(f"{'='*60}")

accs = [c['acc'] for c in chunk_results]
total_n = sum(c['n'] for c in chunk_results)
total_correct = sum(c['acc'] * c['n'] for c in chunk_results)
overall = total_correct / total_n if total_n > 0 else 0.5

print(f"Overall OOS accuracy: {overall:.4f} ({overall*100:.2f}%)")
print(f"Per-chunk: {[f'{a:.4f}' for a in accs]}")
print(f"Mean: {np.mean(accs):.4f}  Std: {np.std(accs):.4f}")
print(f"Total test bars: {total_n}")

# Confidence buckets
print(f"\nConfidence Buckets:")
valid_mask = (~np.isnan(all_probs)) & (y >= 0)
vp = all_probs[valid_mask]
vy = y[valid_mask]
vpred = (vp > 0.5).astype(int)

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = np.abs(vp - 0.5) >= (thresh - 0.5)
    if mask.sum() > 50:
        acc_t = (vpred[mask] == vy[mask]).mean()
        cov = mask.mean()
        print(f"  P>={thresh:.2f}: acc={acc_t:.4f} cov={cov:.1%} n={mask.sum()}")

# Feature importance
print(f"\nTop 25 Feature Importances:")
imp = model.feature_importance(importance_type='gain')
imp_pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
for name, importance in imp_pairs[:25]:
    micro = '*' if name in [
        'cvd_trend','cvd_reversal','cvd_max_streak','cvd_first_half','cvd_second_half',
        'cvd_acceleration','vol_gini','vol_max_ratio','vol_spike_count','vol_top5_pct',
        'large_buy_vol','large_sell_vol','large_imbalance','price_impact','absorption',
        'vwap_position','first_half_ret','second_half_ret','intrabar_reversal',
        'tc_ratio','tc_spike','tc_vol_corr','buy_cluster','sell_cluster','flip_count',
    ] or 'cvd_trend_ema' in name or 'large_imb_ema' in name or 'absorption_ema' in name \
      or 'vwap_pos_ema' in name or 'cvd_accel_ema' in name else ' '
    print(f"  {micro} {name:30s}: {importance:,.0f}")

# Count micro vs standard in top 25
micro_names = set()
for name, _ in imp_pairs[:25]:
    if name in ['cvd_trend','cvd_reversal','cvd_max_streak','cvd_first_half','cvd_second_half',
        'cvd_acceleration','vol_gini','vol_max_ratio','vol_spike_count','vol_top5_pct',
        'large_buy_vol','large_sell_vol','large_imbalance','price_impact','absorption',
        'vwap_position','first_half_ret','second_half_ret','intrabar_reversal',
        'tc_ratio','tc_spike','tc_vol_corr','buy_cluster','sell_cluster','flip_count'] \
      or any(x in name for x in ['cvd_trend_ema','large_imb_ema','absorption_ema','vwap_pos_ema','cvd_accel_ema']):
        micro_names.add(name)

print(f"\nMicrostructure features in top 25: {len(micro_names)}/25")
print(f"\nBaseline (5Y OHLCV, bias engine): 0.5385")
print(f"Baseline (5Y OHLCV, LightGBM):    0.5322")
print(f"Microstructure (11M, LightGBM):    {overall:.4f}")
