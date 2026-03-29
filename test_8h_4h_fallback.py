"""
8H prediction + 4H fallback when 8H is weak.
When 8H signal is strong -> use 8H
When 8H signal is weak/wrong -> zoom into 4H for better signal
"""
import numpy as np
import pandas as pd

df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
o,h,l,c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv,sv = df["buy_vol"].values, df["sell_vol"].values

def ema(d,s):
    k=2.0/(s+1);out=np.zeros_like(d);out[0]=d[0]
    for i in range(1,len(d)):out[i]=d[i]*k+out[i-1]*(1-k)
    return out

def rsi(data,period):
    r=np.full(len(data),50.0);d=np.diff(data,prepend=data[0])
    g=np.where(d>0,d,0);lo=np.where(d<0,-d,0)
    ag=ema(g,period);al=ema(lo,period)
    for i in range(period,len(data)):
        if al[i]>0: r[i]=100-100/(1+ag[i]/al[i])
    return r

# Build 4H bars
P4=48; n_4h=len(df)//P4
c_4h=np.zeros(n_4h);h_4h=np.zeros(n_4h);l_4h=np.zeros(n_4h);o_4h=np.zeros(n_4h)
bv_4h=np.zeros(n_4h);sv_4h=np.zeros(n_4h)
for i in range(n_4h):
    s,e=i*P4,i*P4+P4
    o_4h[i]=o[s];c_4h[i]=c[e-1];h_4h[i]=h[s:e].max();l_4h[i]=l[s:e].min()
    bv_4h[i]=bv[s:e].sum();sv_4h[i]=sv[s:e].sum()

# Build 8H bars
P8=96; n_8h=len(df)//P8
c_8h=np.zeros(n_8h);h_8h=np.zeros(n_8h);l_8h=np.zeros(n_8h);o_8h=np.zeros(n_8h)
bv_8h=np.zeros(n_8h);sv_8h=np.zeros(n_8h)
for i in range(n_8h):
    s,e=i*P8,i*P8+P8
    o_8h[i]=o[s];c_8h[i]=c[e-1];h_8h[i]=h[s:e].max();l_8h[i]=l[s:e].min()
    bv_8h[i]=bv[s:e].sum();sv_8h[i]=sv[s:e].sum()

print(f"4H: {n_4h} bars, 8H: {n_8h} bars")

# 8H outcomes (K=1: sonraki 8H mum)
outcomes_8h = np.zeros(n_8h, dtype=np.int8)
for i in range(n_8h-1):
    outcomes_8h[i] = 1 if c_8h[i+1] > c_8h[i] else 0
outcomes_8h[-1] = -1
valid_8h = (outcomes_8h >= 0) & (np.arange(n_8h) >= 50)

# 8H signals
ema_8h = ema(c_8h, 4)
mr_8h = np.where(c_8h > ema_8h, -1.0, 1.0)
rsi_8h = rsi(c_8h, 4)
candle_ret_8h = np.where(o_8h > 0, (c_8h - o_8h) / o_8h * 100, 0)

# 8H combined score
score_8h = np.zeros(n_8h)
score_8h += mr_8h * 1.0
score_8h += np.where(rsi_8h < 35, 1, np.where(rsi_8h > 65, -1, 0)) * 1.0
score_8h += np.where(candle_ret_8h > 1, -1, np.where(candle_ret_8h < -1, 1, 0)) * 0.5

# 8H signal strength
score_8h_abs = np.abs(score_8h)

# 4H signals (2 bars per 8H bar)
ema_4h = ema(c_4h, 8)
mr_4h = np.where(c_4h > ema_4h, -1.0, 1.0)
rsi_4h = rsi(c_4h, 6)
candle_ret_4h = np.where(o_4h > 0, (c_4h - o_4h) / o_4h * 100, 0)

score_4h = np.zeros(n_4h)
score_4h += mr_4h * 1.0
score_4h += np.where(rsi_4h < 35, 1, np.where(rsi_4h > 65, -1, 0)) * 1.0
score_4h += np.where(candle_ret_4h > 1, -1, np.where(candle_ret_4h < -1, 1, 0)) * 1.0

# Map 4H signal to 8H: use the SECOND 4H bar within each 8H (more recent)
# Each 8H bar = 2x 4H bars
score_4h_at_8h = np.zeros(n_8h)
for i in range(n_8h):
    idx_4h = i * 2 + 1  # second 4H bar in this 8H
    if idx_4h < n_4h:
        score_4h_at_8h[i] = score_4h[idx_4h]

print(f"\n{'='*60}")
print(f"8H ALONE vs 8H+4H FALLBACK")
print(f"{'='*60}")

# Baseline: 8H alone
dir_8h = np.sign(score_8h)
mask = valid_8h & (dir_8h != 0)
acc_8h = ((dir_8h[mask] > 0) == (outcomes_8h[mask] == 1)).sum() / mask.sum()
print(f"\n8H alone: acc={acc_8h:.4f}, n={mask.sum()}")

# Accuracy by 8H score strength
print(f"\n=== 8H accuracy by signal strength ===")
for lo_s, hi_s in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]:
    m = valid_8h & (score_8h_abs >= lo_s) & (score_8h_abs < hi_s) & (dir_8h != 0)
    if m.sum() > 30:
        a = ((dir_8h[m] > 0) == (outcomes_8h[m] == 1)).sum() / m.sum()
        print(f"  |score| [{lo_s}-{hi_s}): acc={a:.4f}, n={m.sum()}")

# Strategy: when 8H is weak, use 4H
print(f"\n=== HYBRID: 8H strong -> use 8H, weak -> use 4H ===")
for threshold in [0.5, 1.0, 1.5, 2.0]:
    hybrid = np.zeros(n_8h)
    source = np.zeros(n_8h, dtype=int)  # 0=none, 1=8h, 2=4h

    for i in range(n_8h):
        if score_8h_abs[i] >= threshold:
            hybrid[i] = np.sign(score_8h[i])
            source[i] = 1
        else:
            hybrid[i] = np.sign(score_4h_at_8h[i])
            source[i] = 2

    h_dir = np.sign(hybrid)
    mask = valid_8h & (h_dir != 0)
    if mask.sum() < 50: continue
    acc = ((h_dir[mask] > 0) == (outcomes_8h[mask] == 1)).sum() / mask.sum()

    # Per-source
    m8 = mask & (source == 1)
    m4 = mask & (source == 2)
    a8 = ((h_dir[m8] > 0) == (outcomes_8h[m8] == 1)).sum() / max(m8.sum(), 1)
    a4 = ((h_dir[m4] > 0) == (outcomes_8h[m4] == 1)).sum() / max(m4.sum(), 1)

    print(f"  threshold={threshold}: total_acc={acc:.4f} | 8H({m8.sum()})={a8:.4f}, 4H({m4.sum()})={a4:.4f}")

# Strategy: 8H direction + 4H confirmation
print(f"\n=== 8H direction + 4H agreement ===")
agree = np.sign(score_8h) == np.sign(score_4h_at_8h)
disagree = ~agree & (np.sign(score_8h) != 0) & (np.sign(score_4h_at_8h) != 0)

m_agree = valid_8h & agree & (np.sign(score_8h) != 0)
m_disagree = valid_8h & disagree

if m_agree.sum() > 50:
    a = ((np.sign(score_8h[m_agree]) > 0) == (outcomes_8h[m_agree] == 1)).sum() / m_agree.sum()
    print(f"  8H+4H AGREE:    acc={a:.4f}, n={m_agree.sum()}")

if m_disagree.sum() > 50:
    # When they disagree, who is right?
    a8 = ((np.sign(score_8h[m_disagree]) > 0) == (outcomes_8h[m_disagree] == 1)).sum() / m_disagree.sum()
    a4 = ((np.sign(score_4h_at_8h[m_disagree]) > 0) == (outcomes_8h[m_disagree] == 1)).sum() / m_disagree.sum()
    print(f"  8H+4H DISAGREE: 8H_acc={a8:.4f}, 4H_acc={a4:.4f}, n={m_disagree.sum()}")

# Strategy: 8H weak + 4H strong -> use 4H
print(f"\n=== 8H weak + 4H strong ===")
for weak_t, strong_t in [(0.5, 1.5), (1.0, 1.5), (0.5, 2.0), (1.0, 2.0)]:
    combo = np.zeros(n_8h)
    for i in range(n_8h):
        if score_8h_abs[i] >= 1.5:
            combo[i] = np.sign(score_8h[i])  # 8H strong, trust it
        elif score_8h_abs[i] < weak_t and abs(score_4h_at_8h[i]) >= strong_t:
            combo[i] = np.sign(score_4h_at_8h[i])  # 8H weak, 4H strong
        elif score_8h_abs[i] >= 0.5:
            combo[i] = np.sign(score_8h[i])  # 8H medium, still use
        else:
            combo[i] = np.sign(score_4h_at_8h[i])  # default 4H

    d = np.sign(combo)
    mask = valid_8h & (d != 0)
    if mask.sum() > 50:
        acc = ((d[mask] > 0) == (outcomes_8h[mask] == 1)).sum() / mask.sum()
        print(f"  8H_weak<{weak_t} + 4H_strong>{strong_t}: acc={acc:.4f}, n={mask.sum()}")
