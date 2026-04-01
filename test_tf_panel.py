"""
TF Panel Analysis: Her TF kendi outcome'ına göre değerlendirilir.
Her 8 saatlik periyot bir panel. İçinde:
  - 8H: 1 sinyal (8 saatin tamamı)
  - 4H: 2 sinyal (ilk 4H + son 4H)
  - 1H: 8 sinyal (her saat)

Her saat için: 3 TF'nin doğru/yanlış durumu yanyana.
Birinin eksiğini diğerleri kapatıyor mu?
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
    c_a = np.zeros(n); h_a = np.zeros(n); l_a = np.zeros(n)
    bv_a = np.zeros(n); sv_a = np.zeros(n)
    for i in range(n):
        s, e = i * period, i * period + period
        ts_a[i] = ts[s]; c_a[i] = c[e - 1]
        h_a[i] = h[s:e].max(); l_a[i] = l[s:e].min()
        bv_a[i] = bv[s:e].sum(); sv_a[i] = sv[s:e].sum()
    return n, ts_a, c_a, h_a, l_a, bv_a, sv_a

n_1h, ts_1h, c_1h, h_1h, l_1h, bv_1h, sv_1h = aggregate(12)
n_4h, ts_4h, c_4h, h_4h, l_4h, bv_4h, sv_4h = aggregate(48)
n_8h, ts_8h, c_8h, h_8h, l_8h, bv_8h, sv_8h = aggregate(96)

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

def compute_rsi(data, period):
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

def combined_scoring(close, buy_vol, sell_vol, mr_span1, mr_span2, rsi_period, cvd_window):
    """Combined scoring with given params."""
    n = len(close)
    ema1 = ema(close, mr_span1)
    ema2 = ema(close, mr_span2)
    mr1_sign = np.where(close > ema1, -1.0, 1.0)
    mr2_sign = np.where(close > ema2, -1.0, 1.0)
    rsi = compute_rsi(close, rsi_period)
    cvd_z = cvd_zscore(buy_vol, sell_vol, cvd_window)

    score = np.zeros(n)
    score += mr1_sign * 0.9
    agree_mr = (mr1_sign == mr2_sign)
    score += np.where(agree_mr, mr1_sign * 0.6, 0)
    rsi_oversold = rsi < 35
    rsi_overbought = rsi > 65
    rsi_confirm = ((rsi_oversold & (mr1_sign > 0)) | (rsi_overbought & (mr1_sign < 0)))
    score += np.where(rsi_confirm, mr1_sign * 1.0, 0)
    score += np.where(cvd_z > 0.5, 1.0, np.where(cvd_z < -0.5, -1.0, 0))
    score += np.where(agree_mr & (np.abs(cvd_z) > 0.5) & (np.sign(cvd_z) == mr1_sign),
                      mr1_sign * 1.4, 0)
    return score

# ═══════════════════════════════════════════════════════════════
# COMPUTE SCORES — each TF with proportional params
# ═══════════════════════════════════════════════════════════════
print("Computing scores...")

# 1H: TPE optimal params
score_1h = combined_scoring(c_1h, bv_1h, sv_1h, mr_span1=32, mr_span2=56, rsi_period=18, cvd_window=24)
# 4H: scaled params (divide windows by 4)
score_4h = combined_scoring(c_4h, bv_4h, sv_4h, mr_span1=8, mr_span2=14, rsi_period=5, cvd_window=6)
# 8H: scaled params (divide windows by 8)
score_8h = combined_scoring(c_8h, bv_8h, sv_8h, mr_span1=4, mr_span2=7, rsi_period=4, cvd_window=4)

# ═══════════════════════════════════════════════════════════════
# OUTCOMES — each TF with its own K=2 outcome
# 1H K=2 → 2 saat sonra yukarı mı?
# 4H K=2 → 8 saat sonra yukarı mı?
# 8H K=2 → 16 saat sonra yukarı mı?
# ═══════════════════════════════════════════════════════════════
K = 2

outcomes_1h = np.full(n_1h, -1, dtype=np.int8)
for i in range(n_1h - K):
    outcomes_1h[i] = 1 if c_1h[i + K] > c_1h[i] else 0

outcomes_4h = np.full(n_4h, -1, dtype=np.int8)
for i in range(n_4h - K):
    outcomes_4h[i] = 1 if c_4h[i + K] > c_4h[i] else 0

outcomes_8h = np.full(n_8h, -1, dtype=np.int8)
for i in range(n_8h - K):
    outcomes_8h[i] = 1 if c_8h[i + K] > c_8h[i] else 0

# Direction signals
dir_1h = np.sign(score_1h)
dir_4h = np.sign(score_4h)
dir_8h = np.sign(score_8h)

# ═══════════════════════════════════════════════════════════════
# PER-TF ACCURACY (each TF judged by its own outcome)
# ═══════════════════════════════════════════════════════════════
def tf_accuracy(direction, outcomes, name):
    valid = outcomes >= 0
    non_neutral = valid & (direction != 0)
    pred = direction[non_neutral] > 0
    actual = outcomes[non_neutral] == 1
    acc = (pred == actual).mean()
    print(f"  {name}: {acc:.4f} ({acc*100:.2f}%)  n={non_neutral.sum():,}")
    return acc

print(f"\n{'='*60}")
print(f"  PER-TF ACCURACY (each judged by own outcome)")
print(f"{'='*60}")
tf_accuracy(dir_1h, outcomes_1h, "1H (K=2 → 2H lookahead)")
tf_accuracy(dir_4h, outcomes_4h, "4H (K=2 → 8H lookahead)")
tf_accuracy(dir_8h, outcomes_8h, "8H (K=2 → 16H lookahead)")

# ═══════════════════════════════════════════════════════════════
# MAP EVERYTHING TO 1H TIMEBASE — per-hour panel
# For each 1H bar, we know:
#   - 1H signal & 1H outcome (native)
#   - 4H signal & 4H outcome (from parent 4H bar)
#   - 8H signal & 8H outcome (from parent 8H bar)
# ═══════════════════════════════════════════════════════════════

# 4H → 1H: each 4H bar covers 4 consecutive 1H bars
dir_4h_on_1h = np.zeros(n_1h)
correct_4h_own = np.full(n_1h, False)  # 4H correct by 4H outcome
for i in range(n_4h):
    s = i * 4
    e = min(s + 4, n_1h)
    dir_4h_on_1h[s:e] = dir_4h[i]
    if outcomes_4h[i] >= 0 and dir_4h[i] != 0:
        is_correct = (dir_4h[i] > 0) == (outcomes_4h[i] == 1)
        correct_4h_own[s:e] = is_correct

# 8H → 1H: each 8H bar covers 8 consecutive 1H bars
dir_8h_on_1h = np.zeros(n_1h)
correct_8h_own = np.full(n_1h, False)  # 8H correct by 8H outcome
for i in range(n_8h):
    s = i * 8
    e = min(s + 8, n_1h)
    dir_8h_on_1h[s:e] = dir_8h[i]
    if outcomes_8h[i] >= 0 and dir_8h[i] != 0:
        is_correct = (dir_8h[i] > 0) == (outcomes_8h[i] == 1)
        correct_8h_own[s:e] = is_correct

# 1H correct by 1H outcome
correct_1h_own = np.full(n_1h, False)
valid_1h = (outcomes_1h >= 0) & (dir_1h != 0)
correct_1h_own[valid_1h] = (dir_1h[valid_1h] > 0) == (outcomes_1h[valid_1h] == 1)

# Active: all 3 TFs have valid signals
active = valid_1h & (dir_4h_on_1h != 0) & (dir_8h_on_1h != 0)
n_active = active.sum()
print(f"\nActive bars (all 3 TFs valid): {n_active:,}")

# ═══════════════════════════════════════════════════════════════
# CROSS-RESCUE: Her TF kendi outcome'ında yanlışken,
#               diğerleri kendi outcome'larında doğru mu?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  CROSS-RESCUE ANALYSIS")
print(f"  (Her TF kendi outcome'ına göre doğru/yanlış)")
print(f"{'='*60}")

wrong_1h = active & ~correct_1h_own
wrong_4h = active & ~correct_4h_own
wrong_8h = active & ~correct_8h_own

n_wrong_1h = wrong_1h.sum()
n_wrong_4h = wrong_4h.sum()
n_wrong_8h = wrong_8h.sum()

print(f"\n  1H yanlış: {n_wrong_1h:,} bar ({n_wrong_1h/n_active*100:.1f}%)")
r_4h = correct_4h_own[wrong_1h].sum()
r_8h = correct_8h_own[wrong_1h].sum()
r_both = (correct_4h_own & correct_8h_own)[wrong_1h].sum()
r_any = (correct_4h_own | correct_8h_own)[wrong_1h].sum()
print(f"    → 4H kurtarıyor: {r_4h:,} ({r_4h/n_wrong_1h*100:.1f}%)")
print(f"    → 8H kurtarıyor: {r_8h:,} ({r_8h/n_wrong_1h*100:.1f}%)")
print(f"    → İkisi birden:  {r_both:,} ({r_both/n_wrong_1h*100:.1f}%)")
print(f"    → En az biri:    {r_any:,} ({r_any/n_wrong_1h*100:.1f}%)")
print(f"    → Hiçbiri:       {n_wrong_1h - r_any:,} ({(n_wrong_1h - r_any)/n_wrong_1h*100:.1f}%)")

print(f"\n  4H yanlış: {n_wrong_4h:,} bar ({n_wrong_4h/n_active*100:.1f}%)")
r_1h = correct_1h_own[wrong_4h].sum()
r_8h = correct_8h_own[wrong_4h].sum()
r_both = (correct_1h_own & correct_8h_own)[wrong_4h].sum()
r_any = (correct_1h_own | correct_8h_own)[wrong_4h].sum()
print(f"    → 1H kurtarıyor: {r_1h:,} ({r_1h/n_wrong_4h*100:.1f}%)")
print(f"    → 8H kurtarıyor: {r_8h:,} ({r_8h/n_wrong_4h*100:.1f}%)")
print(f"    → İkisi birden:  {r_both:,} ({r_both/n_wrong_4h*100:.1f}%)")
print(f"    → En az biri:    {r_any:,} ({r_any/n_wrong_4h*100:.1f}%)")
print(f"    → Hiçbiri:       {n_wrong_4h - r_any:,} ({(n_wrong_4h - r_any)/n_wrong_4h*100:.1f}%)")

print(f"\n  8H yanlış: {n_wrong_8h:,} bar ({n_wrong_8h/n_active*100:.1f}%)")
r_1h = correct_1h_own[wrong_8h].sum()
r_4h = correct_4h_own[wrong_8h].sum()
r_both = (correct_1h_own & correct_4h_own)[wrong_8h].sum()
r_any = (correct_1h_own | correct_4h_own)[wrong_8h].sum()
print(f"    → 1H kurtarıyor: {r_1h:,} ({r_1h/n_wrong_8h*100:.1f}%)")
print(f"    → 4H kurtarıyor: {r_4h:,} ({r_4h/n_wrong_8h*100:.1f}%)")
print(f"    → İkisi birden:  {r_both:,} ({r_both/n_wrong_8h*100:.1f}%)")
print(f"    → En az biri:    {r_any:,} ({r_any/n_wrong_8h*100:.1f}%)")
print(f"    → Hiçbiri:       {n_wrong_8h - r_any:,} ({(n_wrong_8h - r_any)/n_wrong_8h*100:.1f}%)")

# All 3 wrong simultaneously
all_wrong = active & ~correct_1h_own & ~correct_4h_own & ~correct_8h_own
all_correct = active & correct_1h_own & correct_4h_own & correct_8h_own
print(f"\n  Üçü birden yanlış: {all_wrong.sum():,} ({all_wrong.sum()/n_active*100:.1f}%)")
print(f"  Üçü birden doğru:  {all_correct.sum():,} ({all_correct.sum()/n_active*100:.1f}%)")
print(f"  En az biri doğru:  {(n_active - all_wrong.sum()):,} ({(n_active - all_wrong.sum())/n_active*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# DETAYLI PANEL: 8 saatlik periyotlar
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  8H PANEL DETAYI: Her periyotta saat saat analiz")
print(f"{'='*60}")

# Her 8H periyottaki doğruluk oranları
n_panels = n_1h // 8
panel_1h_acc = np.zeros(n_panels)
panel_4h_acc = np.zeros(n_panels)
panel_8h_acc = np.zeros(n_panels)
panel_best_tf = np.zeros(n_panels, dtype=int)  # 0=none, 1=1H, 4=4H, 8=8H

for p in range(n_panels):
    s = p * 8
    e = s + 8

    # 1H: 8 bar'ın accuracy'si
    mask = active[s:e]
    if mask.sum() > 0:
        panel_1h_acc[p] = correct_1h_own[s:e][mask].mean()
        panel_4h_acc[p] = correct_4h_own[s:e][mask].mean()
        panel_8h_acc[p] = correct_8h_own[s:e][mask].mean()

        accs = [panel_1h_acc[p], panel_4h_acc[p], panel_8h_acc[p]]
        best_idx = np.argmax(accs)
        panel_best_tf[p] = [1, 4, 8][best_idx]

valid_panels = (panel_1h_acc > 0) | (panel_4h_acc > 0) | (panel_8h_acc > 0)
n_valid_panels = valid_panels.sum()

print(f"  Toplam panel: {n_panels:,}, geçerli: {n_valid_panels:,}")
print(f"\n  Panel bazında ortalama accuracy:")
print(f"    1H: {panel_1h_acc[valid_panels].mean():.4f}")
print(f"    4H: {panel_4h_acc[valid_panels].mean():.4f}")
print(f"    8H: {panel_8h_acc[valid_panels].mean():.4f}")

print(f"\n  En iyi TF dağılımı (panel bazında):")
for tf in [1, 4, 8]:
    n = (panel_best_tf[valid_panels] == tf).sum()
    print(f"    {tf}H en iyi: {n:,} panel ({n/n_valid_panels*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# COMPLEMENTARITY: Korelasyon analizi
# Doğru/yanlış pattern'leri ne kadar farklı?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  COMPLEMENTARITY (doğru/yanlış korelasyonu)")
print(f"{'='*60}")

c1 = correct_1h_own[active].astype(float)
c4 = correct_4h_own[active].astype(float)
c8 = correct_8h_own[active].astype(float)

corr_14 = np.corrcoef(c1, c4)[0, 1]
corr_18 = np.corrcoef(c1, c8)[0, 1]
corr_48 = np.corrcoef(c4, c8)[0, 1]

print(f"  Correlation (1H, 4H): {corr_14:.4f}")
print(f"  Correlation (1H, 8H): {corr_18:.4f}")
print(f"  Correlation (4H, 8H): {corr_48:.4f}")
print(f"  (Düşük korelasyon = daha iyi tamamlayıcılık)")

# ═══════════════════════════════════════════════════════════════
# SAAT BAZLI DETAY: Her saatte hangi TF daha başarılı?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  SAAT BAZLI ACCURACY")
print(f"{'='*60}")

hours = ((ts_1h // 1000) % 86400) // 3600

print(f"  {'Saat':>5s}  {'1H':>7s}  {'4H':>7s}  {'8H':>7s}  {'En iyi':>7s}  {'n':>6s}")
print(f"  {'-'*45}")

hour_best = {}
for hour in range(24):
    mask = active & (hours == hour)
    n_h = mask.sum()
    if n_h < 50:
        continue
    a1 = correct_1h_own[mask].mean()
    a4 = correct_4h_own[mask].mean()
    a8 = correct_8h_own[mask].mean()

    best = "1H" if a1 >= a4 and a1 >= a8 else ("4H" if a4 >= a8 else "8H")
    hour_best[hour] = best

    print(f"  {hour:02d}:00  {a1:.4f}  {a4:.4f}  {a8:.4f}  {best:>7s}  {n_h:>6,}")

print(f"\n  En iyi TF dağılımı (saat bazlı):")
for tf in ["1H", "4H", "8H"]:
    n = sum(1 for v in hour_best.values() if v == tf)
    print(f"    {tf}: {n} saat")

# ═══════════════════════════════════════════════════════════════
# OPTIMAL SWITCHING: Her saatte en iyi TF'yi seç
# (in-sample oracle — üst sınır)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  OPTIMAL SWITCHING (oracle — üst sınır)")
print(f"{'='*60}")

# Oracle: her bar için en iyi TF'nin sonucunu al
oracle_correct = correct_1h_own | correct_4h_own | correct_8h_own
oracle_acc = oracle_correct[active].mean()
print(f"  Oracle (her bar en iyi TF): {oracle_acc:.4f} ({oracle_acc*100:.2f}%)")
print(f"  1H baseline:                {correct_1h_own[active].mean():.4f}")
print(f"  Potansiyel iyileşme:        +{(oracle_acc - correct_1h_own[active].mean())*100:.2f}pp")

# Walk-forward style: train first 50%, learn hourly best TF, test on rest
train_end = n_active // 2
active_idx = np.where(active)[0]
train_idx = active_idx[:train_end]
test_idx = active_idx[train_end:]

# Learn: per-hour best TF from train set
learned_best = {}
for hour in range(24):
    mask_train = np.isin(np.arange(n_1h), train_idx) & (hours == hour)
    n_t = mask_train.sum()
    if n_t < 50:
        learned_best[hour] = "1H"
        continue
    a1 = correct_1h_own[mask_train].mean()
    a4 = correct_4h_own[mask_train].mean()
    a8 = correct_8h_own[mask_train].mean()
    if a1 >= a4 and a1 >= a8:
        learned_best[hour] = "1H"
    elif a4 >= a8:
        learned_best[hour] = "4H"
    else:
        learned_best[hour] = "8H"

# Test: apply learned switching
test_correct = 0
test_total = 0
for idx in test_idx:
    hour = hours[idx]
    best_tf = learned_best.get(hour, "1H")
    if best_tf == "1H":
        test_correct += correct_1h_own[idx]
    elif best_tf == "4H":
        test_correct += correct_4h_own[idx]
    else:
        test_correct += correct_8h_own[idx]
    test_total += 1

if test_total > 0:
    switching_acc = test_correct / test_total
    baseline_acc = correct_1h_own[test_idx].mean()
    print(f"\n  WF Test — Saat bazlı TF switching:")
    print(f"    Switching accuracy: {switching_acc:.4f} ({switching_acc*100:.2f}%)")
    print(f"    1H baseline:       {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"    Fark:              {(switching_acc - baseline_acc)*100:+.2f}pp")
    print(f"    Test bars:         {test_total:,}")

print(f"\nDone!")
