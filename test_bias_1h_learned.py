"""
Learned weights for continuous 1H bias system.
Walk-forward logistic regression on multiple signal sources.
"""
import numpy as np
import pandas as pd
import rust_engine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

t_start = time.time()

# === DATA PREP ===
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12
n_1h = len(df) // PERIOD
ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

c_1h = np.zeros(n_1h); h_1h = np.zeros(n_1h); l_1h = np.zeros(n_1h); o_1h = np.zeros(n_1h)
bv_1h = np.zeros(n_1h); sv_1h = np.zeros(n_1h); oi_1h = np.zeros(n_1h)
ts_1h = np.zeros(n_1h, dtype=np.uint64)
vol_1h = np.zeros(n_1h)

for i in range(n_1h):
    s, e = i * PERIOD, i * PERIOD + PERIOD
    ts_1h[i] = ts[s]; o_1h[i] = o[s]; h_1h[i] = h[s:e].max()
    l_1h[i] = l[s:e].min(); c_1h[i] = c[e-1]
    bv_1h[i] = bv[s:e].sum(); sv_1h[i] = sv[s:e].sum()
    oi_1h[i] = oi[e-1]; vol_1h[i] = bv_1h[i] + sv_1h[i]

print(f"1H: {n_1h} bars")

# Outcomes K=12
K = 12
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if c_1h[i + K] > c_1h[i] else 0
outcomes[-K:] = -1

# === BIAS ENGINE ===
print("Running bias engine...")
r = rust_engine.bias_engine_compute_bias(
    ts_1h.astype(np.uint64), o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h
)
fb = np.array(r['final_bias'])

# === FEATURE ENGINEERING ===
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

# Feature 1: Bias engine output
f_bias = fb

# Feature 2-4: MR distances (negative = price above EMA = bearish MR)
f_mr24 = -(c_1h - ema_24) / np.maximum(ema_24, 1) * 100
f_mr48 = -(c_1h - ema_48) / np.maximum(ema_48, 1) * 100
f_mr72 = -(c_1h - ema_72) / np.maximum(ema_72, 1) * 100

# Feature 5: RSI (centered around 0: negative=overbought=bearish, positive=oversold=bullish)
def compute_rsi(data, period=14):
    rsi = np.full(len(data), 50.0)
    delta = np.diff(data, prepend=data[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_g = ema(gains, period)
    avg_l = ema(losses, period)
    for i in range(period, len(data)):
        if avg_l[i] > 0:
            rsi[i] = 100 - 100 / (1 + avg_g[i] / avg_l[i])
    return rsi

rsi_raw = compute_rsi(c_1h, 14)
f_rsi = -(rsi_raw - 50) / 50  # -1 to +1, positive = oversold = bullish MR

# Feature 6: Volume z-score
vol_ema = ema(vol_1h, 72)
vol_std = np.zeros(n_1h)
for i in range(72, n_1h):
    vol_std[i] = np.std(vol_1h[max(0,i-72):i])
f_volz = np.where(vol_std > 0, (vol_1h - vol_ema) / np.maximum(vol_std, 1e-10), 0)

# Feature 7: CVD momentum (positive = buying pressure)
cvd_1h = bv_1h - sv_1h
cvd_ema24 = ema(cvd_1h, 24)
cvd_std = np.zeros(n_1h)
for i in range(24, n_1h):
    cvd_std[i] = np.std(cvd_1h[max(0,i-24):i])
f_cvd = np.where(cvd_std > 0, (cvd_1h - cvd_ema24) / np.maximum(cvd_std, 1e-10), 0)

# Feature 8: OI change (normalized)
f_oi = np.zeros(n_1h)
for i in range(24, n_1h):
    f_oi[i] = (oi_1h[i] - oi_1h[i-24]) / max(oi_1h[i-24], 1e-10) * 100

# Feature 9: Bias + MR agreement (interaction)
f_agree = np.sign(fb) * np.sign(f_mr24)  # +1 if agree, -1 if disagree

# Feature 10: Price momentum (short term)
f_mom = np.zeros(n_1h)
for i in range(6, n_1h):
    f_mom[i] = (c_1h[i] - c_1h[i-6]) / c_1h[i-6] * 100

# Stack features
WARMUP = 300  # skip warmup period
feature_names = ['bias', 'mr24', 'mr48', 'mr72', 'rsi', 'vol_z', 'cvd', 'oi_chg', 'agree', 'momentum']
X_all = np.column_stack([f_bias, f_mr24, f_mr48, f_mr72, f_rsi, f_volz, f_cvd, f_oi, f_agree, f_mom])
y_all = outcomes

# Valid mask (warmup done + outcome available)
valid = (outcomes >= 0) & (np.arange(n_1h) >= WARMUP)
valid_idx = np.where(valid)[0]
print(f"Valid bars: {len(valid_idx)}")

# === WALK-FORWARD LOGISTIC REGRESSION ===
TRAIN_SIZE = 8760  # 1 year
VAL_SIZE = 2190    # 3 months
TEST_SIZE = 2190   # 3 months
STEP = 2190        # 3 months

print(f"\nWalk-forward: train={TRAIN_SIZE}, val={VAL_SIZE}, test={TEST_SIZE}, step={STEP}")

all_preds = np.full(n_1h, np.nan)
all_probs = np.full(n_1h, np.nan)
fold_results = []

fold = 0
start = valid_idx[0]

while start + TRAIN_SIZE + VAL_SIZE + TEST_SIZE <= valid_idx[-1]:
    train_end = start + TRAIN_SIZE
    val_end = train_end + VAL_SIZE
    test_end = val_end + TEST_SIZE

    # Train indices
    train_mask = valid & (np.arange(n_1h) >= start) & (np.arange(n_1h) < train_end)
    val_mask = valid & (np.arange(n_1h) >= train_end) & (np.arange(n_1h) < val_end)
    test_mask = valid & (np.arange(n_1h) >= val_end) & (np.arange(n_1h) < test_end)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    if len(X_train) < 100 or len(X_test) < 100:
        start += STEP
        fold += 1
        continue

    # Fit scaler on train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Find best C on validation
    best_c, best_val_acc = 0.01, 0
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
        model.fit(X_train_s, y_train)
        val_acc = model.score(X_val_s, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_c = C

    # Retrain on train+val with best C
    X_trainval = np.vstack([X_train_s, X_val_s])
    y_trainval = np.concatenate([y_train, y_val])
    model = LogisticRegression(C=best_c, max_iter=1000, solver='lbfgs')
    model.fit(X_trainval, y_trainval)

    # Predict on test
    test_probs = model.predict_proba(X_test_s)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    test_acc = (test_preds == y_test).mean()

    # Store predictions
    test_indices = np.where(test_mask)[0]
    all_preds[test_indices] = test_preds
    all_probs[test_indices] = test_probs

    # Feature importance
    coefs = model.coef_[0]

    fold_results.append({
        'fold': fold,
        'train_bars': len(X_train),
        'test_bars': len(X_test),
        'test_acc': test_acc,
        'best_C': best_c,
        'val_acc': best_val_acc,
        'coefs': coefs,
    })

    print(f"  Fold {fold}: test_acc={test_acc:.4f}, val_acc={best_val_acc:.4f}, C={best_c}, n_test={len(X_test)}")

    start += STEP
    fold += 1

# === RESULTS ===
print(f"\n{'='*60}")
print(f"WALK-FORWARD RESULTS")
print(f"{'='*60}")

accs = [f['test_acc'] for f in fold_results]
print(f"Folds: {len(fold_results)}")
print(f"Overall OOS accuracy: {np.mean(accs):.4f}")
print(f"Accuracy std: {np.std(accs):.4f}")
print(f"Best fold: {max(accs):.4f}")
print(f"Worst fold: {min(accs):.4f}")

# Average feature importance
avg_coefs = np.mean([f['coefs'] for f in fold_results], axis=0)
print(f"\nFeature importance (avg coefs):")
sorted_idx = np.argsort(np.abs(avg_coefs))[::-1]
for idx in sorted_idx:
    print(f"  {feature_names[idx]:>12}: {avg_coefs[idx]:>+.4f}")

# Accuracy by prediction confidence
predicted = ~np.isnan(all_probs)
pred_mask = predicted & valid
if pred_mask.sum() > 0:
    print(f"\nAccuracy by confidence:")
    probs_valid = all_probs[pred_mask]
    preds_valid = all_preds[pred_mask]
    y_valid = y_all[pred_mask]

    for conf_thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
        # Bars where model is confident
        confident = (probs_valid > conf_thresh) | (probs_valid < (1 - conf_thresh))
        if confident.sum() > 50:
            c_acc = (preds_valid[confident] == y_valid[confident]).mean()
            print(f"  |prob-0.5|>{conf_thresh-0.5:.2f}: {confident.sum():>6} bars ({confident.sum()/len(probs_valid)*100:.1f}%), acc={c_acc:.4f}")

    # Direction changes
    pred_dir = np.sign(all_probs[pred_mask] - 0.5)
    changes = np.sum(pred_dir[1:] != pred_dir[:-1])
    avg_hold = pred_mask.sum() / max(changes, 1)
    print(f"\nDirection changes: {changes}, avg hold: {avg_hold:.1f}H")

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed:.1f}s")
