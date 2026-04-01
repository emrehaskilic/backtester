"""
Multi-TF Combined Scoring: Cross-rescue analysis
1H / 4H / 8H combined scoring → who rescues whom, when?

All analysis at %100 coverage.
"""
import time
import numpy as np
import pandas as pd
import rust_engine

# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
ts = df["open_time"].values
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

def aggregate(period):
    n = len(df) // period
    ts_a = np.zeros(n, dtype=np.uint64)
    o_a = np.zeros(n); h_a = np.zeros(n); l_a = np.zeros(n); c_a = np.zeros(n)
    bv_a = np.zeros(n); sv_a = np.zeros(n); oi_a = np.zeros(n)
    for i in range(n):
        s, e = i * period, i * period + period
        ts_a[i] = ts[s]; o_a[i] = o[s]; h_a[i] = h[s:e].max()
        l_a[i] = l[s:e].min(); c_a[i] = c[e - 1]
        bv_a[i] = bv[s:e].sum(); sv_a[i] = sv[s:e].sum(); oi_a[i] = oi[e - 1]
    return n, ts_a, o_a, h_a, l_a, c_a, bv_a, sv_a, oi_a

# Aggregate all timeframes
n_1h, ts_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h = aggregate(12)
n_4h, ts_4h, o_4h, h_4h, l_4h, c_4h, bv_4h, sv_4h, oi_4h = aggregate(48)
n_8h, ts_8h, o_8h, h_8h, l_8h, c_8h, bv_8h, sv_8h, oi_8h = aggregate(96)

print(f"1H: {n_1h:,} bars | 4H: {n_4h:,} bars | 8H: {n_8h:,} bars")

# ═══════════════════════════════════════════════════════════════
# COMBINED SCORING FUNCTION
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

def combined_scoring(close, buy_vol, sell_vol, scale=1):
    """Combined scoring with TPE optimal params, scaled for timeframe."""
    n = len(close)
    # Scale EMA windows by timeframe (fewer bars = shorter windows)
    mr_span1 = max(4, 32 // scale)
    mr_span2 = max(8, 56 // scale)
    rsi_period = max(4, 18 // scale)
    rsi_threshold = 15.0
    cvd_window = max(4, 24 // scale)

    ema1 = ema(close, mr_span1)
    ema2 = ema(close, mr_span2)
    mr1_sign = np.where(close > ema1, -1.0, 1.0)
    mr2_sign = np.where(close > ema2, -1.0, 1.0)
    rsi = compute_rsi(close, rsi_period)
    cvd_z = cvd_zscore(buy_vol, sell_vol, cvd_window)

    score = np.zeros(n)
    # MR primary (w_mr1=0.9)
    score += mr1_sign * 0.9
    # MR secondary agreement (w_mr2=0.6)
    agree_mr = (mr1_sign == mr2_sign)
    score += np.where(agree_mr, mr1_sign * 0.6, 0)
    # RSI confirmation (w_rsi=1.0)
    rsi_oversold = rsi < (50 - rsi_threshold)
    rsi_overbought = rsi > (50 + rsi_threshold)
    rsi_confirm = ((rsi_oversold & (mr1_sign > 0)) | (rsi_overbought & (mr1_sign < 0)))
    score += np.where(rsi_confirm, mr1_sign * 1.0, 0)
    # CVD (w_cvd=1.0)
    score += np.where(cvd_z > 0.5, 1.0, np.where(cvd_z < -0.5, -1.0, 0))
    # Agreement bonus (w_agree=1.4) — bias is 0, so use MR agreement as proxy
    score += np.where(agree_mr & (np.abs(cvd_z) > 0.5) & (np.sign(cvd_z) == mr1_sign), mr1_sign * 1.4, 0)

    return score

# ═══════════════════════════════════════════════════════════════
# COMPUTE SCORES FOR ALL TIMEFRAMES
# ═══════════════════════════════════════════════════════════════
print(f"\nComputing combined scores...")
score_1h = combined_scoring(c_1h, bv_1h, sv_1h, scale=1)
score_4h = combined_scoring(c_4h, bv_4h, sv_4h, scale=4)
score_8h = combined_scoring(c_8h, bv_8h, sv_8h, scale=8)

# ═══════════════════════════════════════════════════════════════
# OUTCOMES: K=2 for each TF (same time horizon: 2 bars lookahead)
# ═══════════════════════════════════════════════════════════════
K = 2

def compute_outcomes(close, k):
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n - k):
        out[i] = 1 if close[i + k] > close[i] else 0
    out[-k:] = -1
    return out

outcomes_1h = compute_outcomes(c_1h, K)
outcomes_4h = compute_outcomes(c_4h, K)
outcomes_8h = compute_outcomes(c_8h, K)

# ═══════════════════════════════════════════════════════════════
# MAP ALL SIGNALS TO 1H TIMEBASE
# Each 4H bar covers 4 consecutive 1H bars
# Each 8H bar covers 8 consecutive 1H bars
# ═══════════════════════════════════════════════════════════════
# 4H signal mapped to 1H (hold signal for 4 bars)
score_4h_on_1h = np.zeros(n_1h)
for i in range(n_4h):
    start = i * 4
    end = min(start + 4, n_1h)
    score_4h_on_1h[start:end] = score_4h[i]

# 8H signal mapped to 1H (hold signal for 8 bars)
score_8h_on_1h = np.zeros(n_1h)
for i in range(n_8h):
    start = i * 8
    end = min(start + 8, n_1h)
    score_8h_on_1h[start:end] = score_8h[i]

# Use 1H outcomes as ground truth (K=2 at 1H = 2 hour lookahead)
valid = outcomes_1h >= 0

# Directions
dir_1h = np.sign(score_1h)
dir_4h = np.sign(score_4h_on_1h)
dir_8h = np.sign(score_8h_on_1h)

# Correct/incorrect per bar
actual_bull = outcomes_1h == 1
correct_1h = (dir_1h > 0) == actual_bull
correct_4h = (dir_4h > 0) == actual_bull
correct_8h = (dir_8h > 0) == actual_bull

# Only consider valid bars with non-neutral signals
active = valid & (dir_1h != 0) & (dir_4h != 0) & (dir_8h != 0)
n_active = active.sum()

print(f"\nActive bars (all 3 TFs non-neutral): {n_active:,} / {valid.sum():,} ({n_active/valid.sum()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# INDIVIDUAL ACCURACY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  INDIVIDUAL ACCURACY (on active bars)")
print(f"{'='*60}")

acc_1h = correct_1h[active].mean()
acc_4h = correct_4h[active].mean()
acc_8h = correct_8h[active].mean()
print(f"  1H: {acc_1h:.4f} ({acc_1h*100:.2f}%)")
print(f"  4H: {acc_4h:.4f} ({acc_4h*100:.2f}%)")
print(f"  8H: {acc_8h:.4f} ({acc_8h*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════
# CROSS-RESCUE ANALYSIS
# When TF_X is wrong, does TF_Y rescue?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  CROSS-RESCUE ANALYSIS")
print(f"{'='*60}")

def rescue_analysis(wrong_name, wrong_mask, other_name, other_correct):
    """When wrong_name is wrong, what % of time does other_name get it right?"""
    wrong_bars = active & wrong_mask
    n_wrong = wrong_bars.sum()
    if n_wrong == 0:
        return 0.0
    rescued = other_correct[wrong_bars].sum()
    rate = rescued / n_wrong
    print(f"  {wrong_name} yanlış ({n_wrong:,} bar) → {other_name} doğru: {rescued:,} ({rate*100:.2f}%)")
    return rate

wrong_1h = ~correct_1h
wrong_4h = ~correct_4h
wrong_8h = ~correct_8h

print(f"\n  1H yanlış olduğunda:")
rescue_analysis("1H", wrong_1h, "4H", correct_4h)
rescue_analysis("1H", wrong_1h, "8H", correct_8h)

print(f"\n  4H yanlış olduğunda:")
rescue_analysis("4H", wrong_4h, "1H", correct_1h)
rescue_analysis("4H", wrong_4h, "8H", correct_8h)

print(f"\n  8H yanlış olduğunda:")
rescue_analysis("8H", wrong_8h, "1H", correct_1h)
rescue_analysis("8H", wrong_8h, "4H", correct_4h)

# All 3 wrong at same time
all_wrong = active & wrong_1h & wrong_4h & wrong_8h
print(f"\n  Üçü birden yanlış: {all_wrong.sum():,} bar ({all_wrong.sum()/n_active*100:.1f}%)")
# At least one correct
any_correct = active & (correct_1h | correct_4h | correct_8h)
print(f"  En az biri doğru:  {any_correct.sum():,} bar ({any_correct.sum()/n_active*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# AGREEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  AGREEMENT ANALYSIS (consensus strength)")
print(f"{'='*60}")

# All 3 agree
all_agree = active & (dir_1h == dir_4h) & (dir_4h == dir_8h)
n_agree = all_agree.sum()
if n_agree > 0:
    acc_agree = correct_1h[all_agree].mean()  # all same direction, so any TF accuracy = same
    print(f"  3/3 agree: {n_agree:,} bars ({n_agree/n_active*100:.1f}%), accuracy={acc_agree:.4f} ({acc_agree*100:.2f}%)")

# 2 of 3 agree (majority)
agree_1h_4h = active & (dir_1h == dir_4h) & (dir_1h != dir_8h)
agree_1h_8h = active & (dir_1h == dir_8h) & (dir_1h != dir_4h)
agree_4h_8h = active & (dir_4h == dir_8h) & (dir_4h != dir_1h)

for name, mask, dir_majority in [
    ("1H+4H agree, 8H disagrees", agree_1h_4h, dir_1h),
    ("1H+8H agree, 4H disagrees", agree_1h_8h, dir_1h),
    ("4H+8H agree, 1H disagrees", agree_4h_8h, dir_4h),
]:
    n = mask.sum()
    if n > 100:
        pred_bull = dir_majority[mask] > 0
        actual = actual_bull[mask]
        acc = (pred_bull == actual).mean()
        print(f"  {name}: {n:,} bars ({n/n_active*100:.1f}%), majority acc={acc:.4f} ({acc*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════
# CONDITIONAL RESCUE: WHEN does rescue happen?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  CONDITIONAL RESCUE: Hour-of-day analysis")
print(f"{'='*60}")

# Extract hour from timestamps
hours_1h = ((ts_1h // 1000) % 86400) // 3600

for hour_block_name, hour_range in [("Asia (0-8 UTC)", range(0,9)), ("EU (9-16 UTC)", range(9,17)), ("US (17-23 UTC)", range(17,24))]:
    hour_mask = active & np.isin(hours_1h, list(hour_range))
    n_h = hour_mask.sum()
    if n_h < 100:
        continue
    a1 = correct_1h[hour_mask].mean()
    a4 = correct_4h[hour_mask].mean()
    a8 = correct_8h[hour_mask].mean()
    # Majority vote
    majority_dir = np.sign(score_1h + score_4h_on_1h + score_8h_on_1h)
    majority_correct = (majority_dir[hour_mask] > 0) == actual_bull[hour_mask]
    # Handle neutral (score sum = 0) — use 1H as tiebreaker
    neutral_mask = majority_dir[hour_mask] == 0
    if neutral_mask.sum() > 0:
        majority_correct[neutral_mask] = correct_1h[hour_mask][neutral_mask]
    a_maj = majority_correct.mean()
    print(f"  {hour_block_name}: 1H={a1:.4f} 4H={a4:.4f} 8H={a8:.4f} Majority={a_maj:.4f}  (n={n_h:,})")

print(f"\n{'='*60}")
print(f"  CONDITIONAL RESCUE: Volatility regime")
print(f"{'='*60}")

# ATR as volatility proxy
atr_1h = h_1h - l_1h
atr_pct = np.zeros(n_1h)
window = 168  # 1 week
for i in range(window, n_1h):
    sorted_atr = np.sort(atr_1h[i-window:i])
    rank = np.searchsorted(sorted_atr, atr_1h[i])
    atr_pct[i] = rank / window

for vol_name, vol_lo, vol_hi in [("Low vol (0-33%)", 0.0, 0.33), ("Mid vol (33-66%)", 0.33, 0.66), ("High vol (66-100%)", 0.66, 1.01)]:
    vol_mask = active & (atr_pct >= vol_lo) & (atr_pct < vol_hi)
    n_v = vol_mask.sum()
    if n_v < 100:
        continue
    a1 = correct_1h[vol_mask].mean()
    a4 = correct_4h[vol_mask].mean()
    a8 = correct_8h[vol_mask].mean()
    majority_dir = np.sign(score_1h + score_4h_on_1h + score_8h_on_1h)
    majority_correct = (majority_dir[vol_mask] > 0) == actual_bull[vol_mask]
    neutral = majority_dir[vol_mask] == 0
    if neutral.sum() > 0:
        majority_correct[neutral] = correct_1h[vol_mask][neutral]
    a_maj = majority_correct.mean()
    print(f"  {vol_name}: 1H={a1:.4f} 4H={a4:.4f} 8H={a8:.4f} Majority={a_maj:.4f}  (n={n_v:,})")

# ═══════════════════════════════════════════════════════════════
# ENSEMBLE STRATEGIES (full %100 coverage)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  ENSEMBLE STRATEGIES (%100 coverage)")
print(f"{'='*60}")

def eval_ensemble(name, score_arr):
    direction = np.sign(score_arr)
    # For neutral, use slight bias toward 1H
    neutral = direction == 0
    direction[neutral] = dir_1h[neutral]
    still_neutral = direction == 0
    direction[still_neutral] = 1  # last resort: bullish

    non_neutral = valid & (direction != 0)
    n_nn = non_neutral.sum()
    coverage = n_nn / valid.sum() * 100
    pred_bull = direction[non_neutral] > 0
    actual = outcomes_1h[non_neutral] == 1
    acc = (pred_bull == actual).mean()
    print(f"  {name}: acc={acc:.4f} ({acc*100:.2f}%), coverage={coverage:.1f}%, n={n_nn:,}")
    return acc

# 1. Individual baselines
print(f"\n  --- Baselines ---")
eval_ensemble("1H only", score_1h.copy())
eval_ensemble("4H only", score_4h_on_1h.copy())
eval_ensemble("8H only", score_8h_on_1h.copy())

# 2. Equal weight ensemble
print(f"\n  --- Ensemble strategies ---")
eval_ensemble("Equal weight (1+1+1)", (score_1h + score_4h_on_1h + score_8h_on_1h).copy())

# 3. Higher TF weighted more
eval_ensemble("HTF heavy (1+2+3)", (score_1h + 2*score_4h_on_1h + 3*score_8h_on_1h).copy())
eval_ensemble("HTF heavy (1+1+2)", (score_1h + score_4h_on_1h + 2*score_8h_on_1h).copy())

# 4. Lower TF weighted more
eval_ensemble("LTF heavy (3+2+1)", (3*score_1h + 2*score_4h_on_1h + score_8h_on_1h).copy())

# 5. Majority vote (sign-based)
majority_score = np.sign(dir_1h + dir_4h + dir_8h)
eval_ensemble("Majority vote", majority_score.copy())

# 6. Confidence-weighted: stronger signals get more weight
# Normalize each score to [-1, 1] range
s1_norm = score_1h / (np.abs(score_1h).max() + 1e-10)
s4_norm = score_4h_on_1h / (np.abs(score_4h_on_1h).max() + 1e-10)
s8_norm = score_8h_on_1h / (np.abs(score_8h_on_1h).max() + 1e-10)
eval_ensemble("Normalized equal", (s1_norm + s4_norm + s8_norm).copy())
eval_ensemble("Normalized HTF (1+2+3)", (s1_norm + 2*s4_norm + 3*s8_norm).copy())

# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD OOS TEST: Best ensemble vs baseline
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  WALK-FORWARD OOS TEST")
print(f"{'='*60}")

# Simple WF: train first 50%, test rest in chunks
train_end = n_1h // 2
chunk_size = n_1h // 10  # ~6 month chunks

strategies = {
    "1H only": score_1h,
    "Equal (1+1+1)": score_1h + score_4h_on_1h + score_8h_on_1h,
    "HTF heavy (1+2+3)": score_1h + 2*score_4h_on_1h + 3*score_8h_on_1h,
    "HTF (1+1+2)": score_1h + score_4h_on_1h + 2*score_8h_on_1h,
    "Majority vote": np.sign(dir_1h + dir_4h + dir_8h).astype(float),
    "Norm HTF (1+2+3)": s1_norm + 2*s4_norm + 3*s8_norm,
}

for name, score_arr in strategies.items():
    direction = np.sign(score_arr)
    # Fill neutrals with 1H
    neutral = direction == 0
    direction[neutral] = dir_1h[neutral]
    still_neutral = direction == 0
    direction[still_neutral] = 1

    chunk_accs = []
    idx = train_end
    while idx + chunk_size <= n_1h:
        chunk_mask = np.zeros(n_1h, dtype=bool)
        chunk_mask[idx:idx+chunk_size] = True
        chunk_valid = chunk_mask & valid & (direction != 0)
        n_cv = chunk_valid.sum()
        if n_cv > 100:
            pred = direction[chunk_valid] > 0
            actual = outcomes_1h[chunk_valid] == 1
            acc = (pred == actual).mean()
            chunk_accs.append(acc)
        idx += chunk_size

    if chunk_accs:
        mean_acc = np.mean(chunk_accs)
        std_acc = np.std(chunk_accs)
        print(f"  {name:25s}: OOS={mean_acc:.4f} ({mean_acc*100:.2f}%) ±{std_acc:.4f}  chunks={[round(a,4) for a in chunk_accs]}")
    else:
        print(f"  {name:25s}: No OOS chunks")

print(f"\nDone!")
