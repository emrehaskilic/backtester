"""LightGBM Bias Engine — 50+ features, walk-forward validated
Target: 1H bar direction (K=2, 2-bar = 2H forward return sign)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING & 1H AGGREGATION
# ══════════════════════════════════════════════════════════════

print("Loading data...", flush=True)
eth = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
btc = pd.read_parquet("data/BTCUSDT_5m_5y_perp.parquet")
merged = pd.merge(eth, btc, on="open_time", suffixes=("_eth", "_btc"), how="inner")

PERIOD = 12  # 5m -> 1H
n_1h = len(merged) // PERIOD

def agg_1h(df, suffix):
    n = n_1h
    out = {}
    vals = {}
    for col in ['open_time', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'open_interest']:
        key = col if col == 'open_time' else f"{col}{suffix}"
        vals[col] = df[key].values

    ts = np.zeros(n, dtype=np.uint64)
    o = np.zeros(n); h = np.zeros(n); l = np.zeros(n); c = np.zeros(n)
    bv = np.zeros(n); sv = np.zeros(n); oi = np.zeros(n)
    vol = np.zeros(n)

    for i in range(n):
        s, e = i * PERIOD, i * PERIOD + PERIOD
        ts[i] = vals['open_time'][s]
        o[i] = vals['open'][s]
        h[i] = vals['high'][s:e].max()
        l[i] = vals['low'][s:e].min()
        c[i] = vals['close'][e-1]
        bv[i] = vals['buy_vol'][s:e].sum()
        sv[i] = vals['sell_vol'][s:e].sum()
        oi[i] = vals['open_interest'][e-1]
        vol[i] = bv[i] + sv[i]

    return ts, o, h, l, c, bv, sv, np.nan_to_num(oi, nan=0.0), vol

ts, o_e, h_e, l_e, c_e, bv_e, sv_e, oi_e, vol_e = agg_1h(merged, "_eth")
_, o_b, h_b, l_b, c_b, bv_b, sv_b, oi_b, vol_b = agg_1h(merged, "_btc")

print(f"1H bars: {n_1h} (~{n_1h/8760:.1f} years)")

# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (50+ features)
# ══════════════════════════════════════════════════════════════

print("Computing features...", flush=True)

def rolling_zscore(arr, window):
    s = pd.Series(arr)
    m = s.rolling(window, min_periods=window).mean()
    st = s.rolling(window, min_periods=window).std()
    return ((s - m) / st.replace(0, np.nan)).fillna(0).values

def rolling_rank(arr, window):
    s = pd.Series(arr)
    return s.rolling(window, min_periods=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).fillna(0.5).values

def ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values

def rsi(close, period):
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).values

def returns(close, period=1):
    r = np.zeros(len(close))
    r[period:] = (close[period:] - close[:-period]) / close[:-period]
    return r

def atr(high, low, close, period):
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period, min_periods=1).mean().values

features = {}

# ── GROUP 1: Price Returns & Momentum (12 features) ──
for w in [1, 2, 3, 6, 12, 24, 48, 72]:
    features[f'ret_{w}'] = returns(c_e, w)

for w in [6, 12, 24, 48]:
    ret = returns(c_e, 1)
    features[f'ret_zscore_{w}'] = rolling_zscore(ret, w)

# ── GROUP 2: Mean Reversion (8 features) ──
for span in [8, 16, 32, 64]:
    e = ema(c_e, span)
    features[f'ema_dist_{span}'] = (c_e - e) / e  # pct distance from EMA
    features[f'ema_cross_{span}'] = np.sign(c_e - e)  # above/below

# ── GROUP 3: RSI at multiple periods (4 features) ──
for p in [6, 14, 21, 28]:
    features[f'rsi_{p}'] = rsi(c_e, p)

# ── GROUP 4: Volatility (8 features) ──
for w in [6, 12, 24, 48]:
    a = atr(h_e, l_e, c_e, w)
    features[f'atr_{w}'] = a
    features[f'atr_pct_{w}'] = a / c_e  # ATR as % of price

# Bar range relative features
features['bar_range_pct'] = (h_e - l_e) / c_e
features['upper_shadow'] = (h_e - np.maximum(o_e, c_e)) / (h_e - l_e + 1e-10)
features['lower_shadow'] = (np.minimum(o_e, c_e) - l_e) / (h_e - l_e + 1e-10)
features['body_pct'] = np.abs(c_e - o_e) / (h_e - l_e + 1e-10)

# Volatility regime
features['atr_ratio_6_24'] = atr(h_e, l_e, c_e, 6) / (atr(h_e, l_e, c_e, 24) + 1e-10)
features['atr_rank_48'] = rolling_rank(atr(h_e, l_e, c_e, 12), 48)

# ── GROUP 5: Volume & CVD (10 features) ──
cvd_e = bv_e - sv_e
for w in [6, 12, 24, 48]:
    features[f'cvd_zscore_{w}'] = rolling_zscore(cvd_e, w)

features['imbalance'] = cvd_e / (vol_e + 1e-10)
features['imbalance_ema_12'] = ema(cvd_e / (vol_e + 1e-10), 12)

for w in [6, 12, 24]:
    features[f'vol_zscore_{w}'] = rolling_zscore(vol_e, w)

features['vol_ratio_6_24'] = ema(vol_e, 6) / (ema(vol_e, 24) + 1e-10)

# ── GROUP 6: Open Interest (4 features) ──
for w in [6, 12, 24, 48]:
    oi_change = np.zeros(len(oi_e))
    oi_change[w:] = (oi_e[w:] - oi_e[:-w]) / (oi_e[:-w] + 1e-10)
    features[f'oi_change_{w}'] = oi_change

# ── GROUP 7: BTC Cross-Asset (8 features) ──
btc_ret = returns(c_b, 1)
eth_ret = returns(c_e, 1)

for w in [6, 12, 24]:
    features[f'btc_ret_zscore_{w}'] = rolling_zscore(btc_ret, w)

# BTC lead-lag
for w in [3, 6, 12]:
    btc_cum = pd.Series(btc_ret).rolling(w).sum().fillna(0).values
    eth_cum = pd.Series(eth_ret).rolling(w).sum().fillna(0).values
    features[f'btc_lead_{w}'] = btc_cum - eth_cum

# BTC-ETH return correlation
for w in [24, 72]:
    corr = pd.Series(btc_ret).rolling(w).corr(pd.Series(eth_ret)).fillna(0).values
    features[f'btc_eth_corr_{w}'] = corr

# ── GROUP 8: Time Features (3 features) ──
# Hour of day (cyclical)
hour = (ts % (86400 * 1000)) / (3600 * 1000)
features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

# Day of week
day = ((ts // (86400 * 1000)) + 4) % 7  # 0=Monday
features['dow_sin'] = np.sin(2 * np.pi * day / 7)

# ── GROUP 9: Price Structure (6 features) ──
# Higher high / lower low counts
for w in [6, 12, 24]:
    hh = np.zeros(len(c_e))
    ll = np.zeros(len(c_e))
    for i in range(w, len(c_e)):
        hh[i] = np.sum(h_e[i-w+1:i+1] > h_e[i-w:i]) / w
        ll[i] = np.sum(l_e[i-w+1:i+1] < l_e[i-w:i]) / w
    features[f'hh_ratio_{w}'] = hh
    features[f'll_ratio_{w}'] = ll

# ── GROUP 10: BTC Volume (2 features) ──
btc_cvd = bv_b - sv_b
features['btc_cvd_zscore_12'] = rolling_zscore(btc_cvd, 12)
features['btc_cvd_zscore_24'] = rolling_zscore(btc_cvd, 24)

# ══════════════════════════════════════════════════════════════
# 3. BUILD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════

feature_names = sorted(features.keys())
X = np.column_stack([features[f] for f in feature_names])
print(f"Features: {len(feature_names)}")

# Outcome: K=2 direction (2 bars = 2H forward)
K = 2
y = np.full(n_1h, -1, dtype=int)
for i in range(n_1h - K):
    y[i] = 1 if c_e[i + K] > c_e[i] else 0

# ══════════════════════════════════════════════════════════════
# 4. WALK-FORWARD EVALUATION
# ══════════════════════════════════════════════════════════════

TRAIN_BARS = 17_520   # ~2 years
CHUNK_SIZE = 4_380    # ~6 months
WARMUP = 72           # skip first 72 bars (feature warmup)

print(f"\nWalk-Forward: train={TRAIN_BARS} bars (~2Y), chunk={CHUNK_SIZE} bars (~6M)")
print(f"Total bars for test: {n_1h - TRAIN_BARS}")

t0 = time()

chunks = []
idx = TRAIN_BARS
while idx + CHUNK_SIZE <= n_1h:
    chunks.append((idx, idx + CHUNK_SIZE))
    idx += CHUNK_SIZE

print(f"Test chunks: {len(chunks)}")

chunk_results = []
all_preds = np.full(n_1h, -1, dtype=int)
all_probs = np.full(n_1h, np.nan)

for ci, (test_start, test_end) in enumerate(chunks):
    # Train on everything before test_start
    train_mask = (np.arange(n_1h) >= WARMUP) & (np.arange(n_1h) < test_start) & (y >= 0)
    test_mask = (np.arange(n_1h) >= test_start) & (np.arange(n_1h) < test_end) & (y >= 0)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # LightGBM
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
        'min_child_samples': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
    }

    model = lgb.train(params, dtrain, num_boost_round=500)

    # Predict
    probs = model.predict(X_test)
    preds = (probs > 0.5).astype(int)

    acc = (preds == y_test).mean()
    bull_rate = y_test.mean()
    n_test = len(y_test)

    # Store
    test_indices = np.where(test_mask)[0]
    all_preds[test_indices] = preds
    all_probs[test_indices] = probs

    chunk_results.append({
        'chunk': ci,
        'start': test_start,
        'end': test_end,
        'n_test': n_test,
        'accuracy': acc,
        'bull_rate': bull_rate,
    })
    print(f"  Chunk {ci}: acc={acc:.4f} (n={n_test}, bull_rate={bull_rate:.3f})")

elapsed = time() - t0

# ══════════════════════════════════════════════════════════════
# 5. RESULTS
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"WALK-FORWARD RESULTS ({elapsed:.1f}s)")
print(f"{'='*60}")

accs = [c['accuracy'] for c in chunk_results]
n_bars = [c['n_test'] for c in chunk_results]
overall_correct = sum(c['accuracy'] * c['n_test'] for c in chunk_results)
overall_total = sum(c['n_test'] for c in chunk_results)
overall_acc = overall_correct / overall_total if overall_total > 0 else 0.5

print(f"Overall OOS accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print(f"Per-chunk accuracies: {[f'{a:.4f}' for a in accs]}")
print(f"Accuracy mean: {np.mean(accs):.4f}")
print(f"Accuracy std:  {np.std(accs):.4f}")
print(f"Total test bars: {overall_total}")
print(f"Chunks: {len(chunk_results)}")

# Confidence-based analysis
print(f"\n── Confidence Buckets ──")
valid = all_probs[~np.isnan(all_probs)]
valid_y = y[(~np.isnan(all_probs)) & (y >= 0)]
valid_p = all_probs[(~np.isnan(all_probs)) & (y >= 0)]
valid_pred = (valid_p > 0.5).astype(int)

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = np.abs(valid_p - 0.5) >= (thresh - 0.5)
    if mask.sum() > 100:
        acc_t = (valid_pred[mask] == valid_y[mask]).mean()
        cov = mask.mean()
        print(f"  P>={thresh:.2f} or P<={1-thresh:.2f}: acc={acc_t:.4f} cov={cov:.1%} n={mask.sum()}")

# Feature importance
print(f"\n── Top 20 Feature Importances (last model) ──")
imp = model.feature_importance(importance_type='gain')
imp_pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
for name, importance in imp_pairs[:20]:
    print(f"  {name:25s}: {importance:,.0f}")

print(f"\nPrevious best (bias engine): 0.5385")
print(f"LightGBM OOS:               {overall_acc:.4f}")
if overall_acc > 0.5385:
    print(f"Improvement: +{(overall_acc - 0.5385)*100:.2f}pp")
else:
    print(f"Difference:  {(overall_acc - 0.5385)*100:.2f}pp")
