"""
Dual Source Bias Engine: Binance + CME ETH1!

Hypothesis: CME (institutional) and Binance (retail) have different
predictive power at different hours of the day.

1. Run bias engine on both datasets independently
2. Analyze accuracy by hour-of-day for each source
3. Discover which source is better at which hours
4. Build hybrid system selecting the best source per hour
5. Measure accuracy improvement
"""
import time
import numpy as np
import pandas as pd
import rust_engine

# ═══════════════════════════════════════════════════════════════
# DATA LOADING & 1H AGGREGATION
# ═══════════════════════════════════════════════════════════════

def load_and_aggregate(path, name):
    """Load 5m parquet, aggregate to 1H."""
    df = pd.read_parquet(path)
    PERIOD = 12
    n_1h = len(df) // PERIOD

    ts = df["open_time"].values
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    bv, sv, oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

    ts_1h = np.zeros(n_1h, dtype=np.uint64)
    o_1h = np.zeros(n_1h); h_1h = np.zeros(n_1h); l_1h = np.zeros(n_1h)
    c_1h = np.zeros(n_1h); bv_1h = np.zeros(n_1h); sv_1h = np.zeros(n_1h); oi_1h = np.zeros(n_1h)

    for i in range(n_1h):
        s, e = i * PERIOD, i * PERIOD + PERIOD
        ts_1h[i] = ts[s]; o_1h[i] = o[s]; h_1h[i] = h[s:e].max()
        l_1h[i] = l[s:e].min(); c_1h[i] = c[e - 1]
        bv_1h[i] = bv[s:e].sum(); sv_1h[i] = sv[s:e].sum(); oi_1h[i] = oi[e - 1]

    print(f"  {name}: {n_1h:,} 1H bars")
    return ts_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h, oi_1h

print("Loading data...")
bin_ts, bin_o, bin_h, bin_l, bin_c, bin_bv, bin_sv, bin_oi = \
    load_and_aggregate("data/ETHUSDT_5m_5y.parquet", "Binance")
cme_ts, cme_o, cme_h, cme_l, cme_c, cme_bv, cme_sv, cme_oi = \
    load_and_aggregate("data/CME_ETH1_5m_5y.parquet", "CME")

n_1h = len(bin_c)
assert len(cme_c) == n_1h, "Data length mismatch"

# ═══════════════════════════════════════════════════════════════
# RUN BIAS ENGINE ON BOTH SOURCES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  BIAS ENGINE: Binance")
print(f"{'='*60}")
t0 = time.time()
r_bin = rust_engine.bias_engine_compute_bias(
    bin_ts, bin_o, bin_h, bin_l, bin_c, bin_bv, bin_sv, bin_oi
)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Validated states: {r_bin['n_validated']}")
print(f"  Coverage: {r_bin['coverage']['coverage_pct']}%")
print(f"  Direction accuracy: {r_bin['accuracy']['direction_accuracy']:.4f}")

fb_bin = np.array(r_bin['final_bias'])
dir_bin = np.array(r_bin['direction'])

print(f"\n{'='*60}")
print(f"  BIAS ENGINE: CME ETH1!")
print(f"{'='*60}")
t0 = time.time()
r_cme = rust_engine.bias_engine_compute_bias(
    cme_ts, cme_o, cme_h, cme_l, cme_c, cme_bv, cme_sv, cme_oi
)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Validated states: {r_cme['n_validated']}")
print(f"  Coverage: {r_cme['coverage']['coverage_pct']}%")
print(f"  Direction accuracy: {r_cme['accuracy']['direction_accuracy']:.4f}")

fb_cme = np.array(r_cme['final_bias'])
dir_cme = np.array(r_cme['direction'])

# ═══════════════════════════════════════════════════════════════
# COMBINED SCORING FOR BOTH SOURCES
# ═══════════════════════════════════════════════════════════════

def ema(data, span):
    k = 2.0 / (span + 1)
    out = np.zeros_like(data, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

def compute_rsi(data, period=18):
    n = len(data)
    rsi = np.full(n, 50.0)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = data[i] - data[i-1]
        if d > 0: gains[i] = d
        else: losses[i] = -d
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    for i in range(period, n):
        if avg_loss[i] > 1e-15:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - 100 / (1 + rs)
    return rsi

def compute_combined_score(close, buy_vol, sell_vol, final_bias, params):
    """Compute combined score for a single data source."""
    n = len(close)
    ema1 = ema(close, params['mr_span1'])
    ema2 = ema(close, params['mr_span2'])
    mr1 = np.where(close > ema1, -1.0, 1.0)
    mr2 = np.where(close > ema2, -1.0, 1.0)
    rsi = compute_rsi(close, params['rsi_period'])

    # CVD z-score
    cvd_bar = buy_vol - sell_vol
    cvd_ema_arr = ema(cvd_bar, 24)
    cvd_std = np.zeros(n)
    for i in range(24, n):
        diff = cvd_bar[i-24:i] - cvd_ema_arr[i-24:i]
        cvd_std[i] = np.sqrt(np.mean(diff**2))
    cvd_z = np.zeros(n)
    for i in range(24, n):
        if cvd_std[i] > 1e-15:
            cvd_z[i] = (cvd_bar[i] - cvd_ema_arr[i]) / cvd_std[i]

    score = np.zeros(n)
    score += mr1 * params['w_mr1']
    agree_mr = (mr1 == mr2)
    score += np.where(agree_mr, mr1 * params['w_mr2'], 0)

    rsi_thresh = params['rsi_threshold']
    rsi_confirm = ((rsi < (50 - rsi_thresh)) & (mr1 > 0)) | ((rsi > (50 + rsi_thresh)) & (mr1 < 0))
    score += np.where(rsi_confirm, mr1 * params['w_rsi'], 0)

    bias_dir = np.sign(final_bias)
    bias_mr_agree = (bias_dir == mr1) & (bias_dir != 0)
    score += np.where(bias_mr_agree & (mr1 > 0), params['w_agree'], 0)
    score -= np.where(bias_mr_agree & (mr1 < 0), params['w_agree'], 0)

    score += np.where(cvd_z > 0.5, params['w_cvd'], np.where(cvd_z < -0.5, -params['w_cvd'], 0))

    return score

# TPE optimal params
params = {
    'mr_span1': 32, 'mr_span2': 56, 'rsi_period': 18, 'rsi_threshold': 15.0,
    'w_mr1': 0.9, 'w_mr2': 0.6, 'w_rsi': 1.0, 'w_agree': 1.4, 'w_cvd': 1.0,
}

print(f"\nComputing combined scores...")
score_bin = compute_combined_score(bin_c, bin_bv, bin_sv, fb_bin, params)
score_cme = compute_combined_score(cme_c, cme_bv, cme_sv, fb_cme, params)

# ═══════════════════════════════════════════════════════════════
# HOUR-OF-DAY ACCURACY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  HOUR-OF-DAY ACCURACY ANALYSIS")
print(f"{'='*60}")

K = 12  # 12H lookahead (bias engine default)
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if bin_c[i + K] > bin_c[i] else 0
outcomes[-K:] = -1
valid = outcomes >= 0

# Get hour for each 1H bar
hours = np.array([pd.to_datetime(t, unit='ms', utc=True).hour for t in bin_ts])

# Per-hour accuracy for each source
print(f"\n  {'Hour':>4} | {'Binance':>10} {'n':>6} | {'CME':>10} {'n':>6} | {'Best':>7} | {'Delta':>6}")
print(f"  {'-'*4} | {'-'*10} {'-'*6} | {'-'*10} {'-'*6} | {'-'*7} | {'-'*6}")

hour_acc_bin = {}
hour_acc_cme = {}
hour_best = {}

for h in range(24):
    mask = (hours == h) & valid

    # Binance: bias engine direction
    mask_bin = mask & (dir_bin != 0)
    if mask_bin.sum() > 50:
        actual = outcomes[mask_bin] == 1
        pred = fb_bin[mask_bin] > 0
        acc_bin = (pred == actual).sum() / mask_bin.sum()
        n_bin = mask_bin.sum()
    else:
        acc_bin = 0.5
        n_bin = 0

    # CME: bias engine direction
    mask_cme = mask & (dir_cme != 0)
    if mask_cme.sum() > 50:
        actual = outcomes[mask_cme] == 1
        pred = fb_cme[mask_cme] > 0
        acc_cme = (pred == actual).sum() / mask_cme.sum()
        n_cme = mask_cme.sum()
    else:
        acc_cme = 0.5
        n_cme = 0

    hour_acc_bin[h] = acc_bin
    hour_acc_cme[h] = acc_cme

    if acc_bin >= acc_cme:
        best = "BIN"
        delta = acc_bin - acc_cme
    else:
        best = "CME"
        delta = acc_cme - acc_bin

    hour_best[h] = best
    marker = " ***" if delta > 0.02 else ""
    print(f"  {h:>4} | {acc_bin:>10.4f} {n_bin:>6} | {acc_cme:>10.4f} {n_cme:>6} | {best:>7} | {delta:>5.3f}{marker}")

# ═══════════════════════════════════════════════════════════════
# COMBINED SCORING: HOUR-OF-DAY ACCURACY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  COMBINED SCORING: HOUR-OF-DAY ACCURACY")
print(f"{'='*60}")

dir_bin_score = np.sign(score_bin)
dir_cme_score = np.sign(score_cme)

print(f"\n  {'Hour':>4} | {'Binance':>10} {'n':>6} | {'CME':>10} {'n':>6} | {'Best':>7} | {'Delta':>6}")
print(f"  {'-'*4} | {'-'*10} {'-'*6} | {'-'*10} {'-'*6} | {'-'*7} | {'-'*6}")

hour_score_acc_bin = {}
hour_score_acc_cme = {}
hour_score_best = {}

for h in range(24):
    mask = (hours == h) & valid

    # Binance combined score
    mask_b = mask & (dir_bin_score != 0)
    if mask_b.sum() > 50:
        actual = outcomes[mask_b] == 1
        pred = score_bin[mask_b] > 0
        acc_b = (pred == actual).sum() / mask_b.sum()
        n_b = mask_b.sum()
    else:
        acc_b = 0.5; n_b = 0

    # CME combined score
    mask_c = mask & (dir_cme_score != 0)
    if mask_c.sum() > 50:
        actual = outcomes[mask_c] == 1
        pred = score_cme[mask_c] > 0
        acc_c = (pred == actual).sum() / mask_c.sum()
        n_c = mask_c.sum()
    else:
        acc_c = 0.5; n_c = 0

    hour_score_acc_bin[h] = acc_b
    hour_score_acc_cme[h] = acc_c

    if acc_b >= acc_c:
        best = "BIN"
        delta = acc_b - acc_c
    else:
        best = "CME"
        delta = acc_c - acc_b

    hour_score_best[h] = best
    marker = " ***" if delta > 0.02 else ""
    print(f"  {h:>4} | {acc_b:>10.4f} {n_b:>6} | {acc_c:>10.4f} {n_c:>6} | {best:>7} | {delta:>5.3f}{marker}")

# ═══════════════════════════════════════════════════════════════
# SESSION-BASED ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  SESSION-BASED ANALYSIS")
print(f"{'='*60}")

sessions = {
    "Asian (0-8 UTC)": list(range(0, 8)),
    "EU (8-14 UTC)": list(range(8, 14)),
    "US (14-21 UTC)": list(range(14, 21)),
    "Late (21-24 UTC)": list(range(21, 24)),
}

for sname, shours in sessions.items():
    mask = np.isin(hours, shours) & valid

    # Binance bias engine
    mask_b = mask & (dir_bin != 0)
    if mask_b.sum() > 100:
        actual = outcomes[mask_b] == 1
        pred = fb_bin[mask_b] > 0
        acc_b = (pred == actual).sum() / mask_b.sum()
    else:
        acc_b = 0.5

    # CME bias engine
    mask_c = mask & (dir_cme != 0)
    if mask_c.sum() > 100:
        actual = outcomes[mask_c] == 1
        pred = fb_cme[mask_c] > 0
        acc_c = (pred == actual).sum() / mask_c.sum()
    else:
        acc_c = 0.5

    delta = acc_b - acc_c
    best = "BIN" if acc_b >= acc_c else "CME"
    print(f"  {sname:<25} Bin={acc_b:.4f}  CME={acc_c:.4f}  Best={best} (Δ={abs(delta):.4f})")

# ═══════════════════════════════════════════════════════════════
# HYBRID SYSTEM: Best source per hour
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  HYBRID SYSTEM: Best Source Per Hour")
print(f"{'='*60}")

# Method 1: Bias engine — pick best source per hour
hybrid_correct = 0
hybrid_total = 0
bin_only_correct = 0
cme_only_correct = 0

for i in range(n_1h):
    if not valid[i]:
        continue

    h = hours[i]
    actual = outcomes[i] == 1

    # Use the source that's historically better at this hour
    if hour_acc_bin[h] >= hour_acc_cme[h]:
        pred = fb_bin[i] > 0
    else:
        pred = fb_cme[i] > 0

    if pred == actual:
        hybrid_correct += 1
    hybrid_total += 1

    # Track individual source accuracy
    if (fb_bin[i] > 0) == actual:
        bin_only_correct += 1
    if (fb_cme[i] > 0) == actual:
        cme_only_correct += 1

hybrid_acc = hybrid_correct / hybrid_total
bin_acc = bin_only_correct / hybrid_total
cme_acc = cme_only_correct / hybrid_total

print(f"\n  Bias Engine (K={K}):")
print(f"    Binance only:   {bin_acc:.4f} ({bin_acc*100:.2f}%)")
print(f"    CME only:       {cme_acc:.4f} ({cme_acc*100:.2f}%)")
print(f"    Hybrid (best/h):{hybrid_acc:.4f} ({hybrid_acc*100:.2f}%)")
print(f"    Improvement:    +{(hybrid_acc-max(bin_acc,cme_acc))*100:.2f}% over best single")

# Method 2: Combined scoring — pick best source per hour
hybrid2_correct = 0
hybrid2_total = 0
bin2_correct = 0
cme2_correct = 0

for i in range(n_1h):
    if not valid[i]:
        continue

    h = hours[i]
    actual = outcomes[i] == 1

    if hour_score_acc_bin[h] >= hour_score_acc_cme[h]:
        pred = score_bin[i] > 0
    else:
        pred = score_cme[i] > 0

    if pred == actual:
        hybrid2_correct += 1
    hybrid2_total += 1

    if (score_bin[i] > 0) == actual:
        bin2_correct += 1
    if (score_cme[i] > 0) == actual:
        cme2_correct += 1

hybrid2_acc = hybrid2_correct / hybrid2_total
bin2_acc = bin2_correct / hybrid2_total
cme2_acc = cme2_correct / hybrid2_total

print(f"\n  Combined Scoring (K={K}):")
print(f"    Binance only:   {bin2_acc:.4f} ({bin2_acc*100:.2f}%)")
print(f"    CME only:       {cme2_acc:.4f} ({cme2_acc*100:.2f}%)")
print(f"    Hybrid (best/h):{hybrid2_acc:.4f} ({hybrid2_acc*100:.2f}%)")
print(f"    Improvement:    +{(hybrid2_acc-max(bin2_acc,cme2_acc))*100:.2f}% over best single")

# ═══════════════════════════════════════════════════════════════
# METHOD 3: Weighted ensemble (confidence-based)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  WEIGHTED ENSEMBLE")
print(f"{'='*60}")

# Weight each source by its historical accuracy at that hour
ensemble_correct = 0
ensemble_total = 0

for i in range(n_1h):
    if not valid[i]:
        continue

    h = hours[i]
    actual = outcomes[i] == 1

    # Weighted combination: source weight = its hour accuracy
    w_b = hour_acc_bin[h] - 0.5  # excess accuracy
    w_c = hour_acc_cme[h] - 0.5

    # Weighted score: bias from each source * confidence weight
    combined = fb_bin[i] * max(w_b, 0) + fb_cme[i] * max(w_c, 0)

    # If both have zero excess, fall back to Binance
    if combined == 0:
        combined = fb_bin[i]

    if (combined > 0) == actual:
        ensemble_correct += 1
    ensemble_total += 1

ensemble_acc = ensemble_correct / ensemble_total
print(f"  Confidence-weighted ensemble: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
print(f"  vs Binance only:              {bin_acc:.4f}")
print(f"  Improvement:                  +{(ensemble_acc-bin_acc)*100:.2f}%")

# ═══════════════════════════════════════════════════════════════
# METHOD 4: Agreement boost — both sources agree
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  AGREEMENT ANALYSIS")
print(f"{'='*60}")

agree_mask = (np.sign(fb_bin) == np.sign(fb_cme)) & (np.sign(fb_bin) != 0) & valid
disagree_mask = (np.sign(fb_bin) != np.sign(fb_cme)) & (np.sign(fb_bin) != 0) & (np.sign(fb_cme) != 0) & valid

if agree_mask.sum() > 100:
    actual = outcomes[agree_mask] == 1
    pred = fb_bin[agree_mask] > 0  # both agree, doesn't matter which
    agree_acc = (pred == actual).sum() / agree_mask.sum()
    print(f"  Both sources AGREE:    acc={agree_acc:.4f}, n={agree_mask.sum():,} ({agree_mask.sum()/valid.sum()*100:.1f}%)")

if disagree_mask.sum() > 100:
    # When they disagree, try each
    actual = outcomes[disagree_mask] == 1
    pred_bin = fb_bin[disagree_mask] > 0
    pred_cme = fb_cme[disagree_mask] > 0
    dis_acc_bin = (pred_bin == actual).sum() / disagree_mask.sum()
    dis_acc_cme = (pred_cme == actual).sum() / disagree_mask.sum()
    print(f"  Sources DISAGREE:      Bin={dis_acc_bin:.4f}, CME={dis_acc_cme:.4f}, n={disagree_mask.sum():,} ({disagree_mask.sum()/valid.sum()*100:.1f}%)")

# Method: Use signal when agree, use hour-best when disagree
agree_disagree_correct = 0
agree_disagree_total = 0

for i in range(n_1h):
    if not valid[i]:
        continue

    actual = outcomes[i] == 1
    h = hours[i]

    if np.sign(fb_bin[i]) == np.sign(fb_cme[i]) and np.sign(fb_bin[i]) != 0:
        # Both agree → use the signal
        pred = fb_bin[i] > 0
    elif np.sign(fb_bin[i]) != 0 and np.sign(fb_cme[i]) != 0:
        # Disagree → use hour-best
        if hour_acc_bin[h] >= hour_acc_cme[h]:
            pred = fb_bin[i] > 0
        else:
            pred = fb_cme[i] > 0
    else:
        # One is neutral → use the non-neutral one
        if np.sign(fb_bin[i]) != 0:
            pred = fb_bin[i] > 0
        else:
            pred = fb_cme[i] > 0

    if pred == actual:
        agree_disagree_correct += 1
    agree_disagree_total += 1

ad_acc = agree_disagree_correct / agree_disagree_total
print(f"\n  Agree+HourBest hybrid: {ad_acc:.4f} ({ad_acc*100:.2f}%)")
print(f"  vs Binance only:       {bin_acc:.4f}")
print(f"  Improvement:           +{(ad_acc-bin_acc)*100:.2f}%")

# ═══════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  FINAL COMPARISON — ALL METHODS")
print(f"{'='*60}")

results = [
    ("Binance bias engine only", bin_acc),
    ("CME bias engine only", cme_acc),
    ("Hybrid: best source/hour", hybrid_acc),
    ("Weighted ensemble", ensemble_acc),
    ("Agree+HourBest hybrid", ad_acc),
    ("Binance combined scoring", bin2_acc),
    ("CME combined scoring", cme2_acc),
    ("Hybrid scoring: best/hour", hybrid2_acc),
]

results.sort(key=lambda x: x[1], reverse=True)
print(f"\n  {'Method':<35} {'Accuracy':>10} {'vs Base':>8}")
print(f"  {'-'*35} {'-'*10} {'-'*8}")
base = bin_acc
for name, acc in results:
    delta = acc - base
    marker = " ✓" if delta > 0.005 else ""
    print(f"  {name:<35} {acc:>10.4f} {delta:>+7.3f}{marker}")

print(f"\n  Coverage: 100.0% (all methods, bias engine fallback)")
print(f"  K={K} ({K}H lookahead)")
