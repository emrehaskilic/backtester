"""
Where does accuracy break down?
Analyze by: time period, regime, signal strength, factor agreement, volatility
"""
import numpy as np
import pandas as pd
import rust_engine

df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12; n_1h = len(df) // PERIOD
ts = df["open_time"].values
o,h,l,c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv,sv,oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

c_1h=np.zeros(n_1h);h_1h=np.zeros(n_1h);l_1h=np.zeros(n_1h);o_1h=np.zeros(n_1h)
bv_1h=np.zeros(n_1h);sv_1h=np.zeros(n_1h);oi_1h=np.zeros(n_1h)
ts_1h=np.zeros(n_1h,dtype=np.uint64)
for i in range(n_1h):
    s,e=i*PERIOD,i*PERIOD+PERIOD
    ts_1h[i]=ts[s];o_1h[i]=o[s];h_1h[i]=h[s:e].max()
    l_1h[i]=l[s:e].min();c_1h[i]=c[e-1]
    bv_1h[i]=bv[s:e].sum();sv_1h[i]=sv[s:e].sum();oi_1h[i]=oi[e-1]

# Optimal params'tan scoring
def ema(data, span):
    k=2.0/(span+1);out=np.zeros_like(data);out[0]=data[0]
    for i in range(1,len(data)): out[i]=data[i]*k+out[i-1]*(1-k)
    return out

def rsi(data, period):
    r=np.full(len(data),50.0);d=np.diff(data,prepend=data[0])
    g=np.where(d>0,d,0);lo=np.where(d<0,-d,0)
    ag=ema(g,period);al=ema(lo,period)
    for i in range(period,len(data)):
        if al[i]>0: r[i]=100-100/(1+ag[i]/al[i])
    return r

# Bias engine
print("Running bias engine...", flush=True)
r = rust_engine.bias_engine_compute_bias(ts_1h.astype(np.uint64),o_1h,h_1h,l_1h,c_1h,bv_1h,sv_1h,oi_1h)
fb = np.array(r['final_bias'])

# Optimal scoring (from TPE results)
ema1 = ema(c_1h, 32)  # mr_ema_span1
ema2 = ema(c_1h, 56)  # mr_ema_span2
rsi_vals = rsi(c_1h, 18)  # rsi_period=18

mr1 = np.where(c_1h > ema1, -1.0, 1.0)
mr2 = np.where(c_1h > ema2, -1.0, 1.0)

cvd_1h = bv_1h - sv_1h
cvd_ema = ema(cvd_1h, 24)
cvd_std = np.zeros(n_1h)
for i in range(24, n_1h): cvd_std[i] = np.std(cvd_1h[max(0,i-24):i])
cvd_z = np.where(cvd_std>0, (cvd_1h-cvd_ema)/np.maximum(cvd_std,1e-10), 0)

# Combined score with optimal weights
score = np.zeros(n_1h)
score += fb * 0.0           # w_bias=0
score += mr1 * 0.9          # w_mr1
score += np.where(mr1==mr2, mr1*0.6, 0)  # w_mr2
rsi_os = rsi_vals < 35      # 50-15
rsi_ob = rsi_vals > 65      # 50+15
score += np.where((rsi_os & (mr1>0)) | (rsi_ob & (mr1<0)), mr1*1.0, 0)  # w_rsi
score += np.where((fb>0)&(mr1>0), 1.4, np.where((fb<0)&(mr1<0), -1.4, 0))  # w_agree
score += np.where(cvd_z>0.5, 1.0, np.where(cvd_z<-0.5, -1.0, 0))  # w_cvd
# Override
override = np.abs(fb) >= 0.04
score = np.where(override, fb * 0.0 * 3.0, score)  # w_bias=0 so override is 0 too

direction = np.sign(score)

# Outcomes K=2
K = 2
outcomes = np.zeros(n_1h, dtype=np.int8)
for i in range(n_1h - K):
    outcomes[i] = 1 if c_1h[i+K] > c_1h[i] else 0
outcomes[-K:] = -1
valid = (outcomes >= 0) & (np.arange(n_1h) >= 600)  # skip warmup

# ========================================
print(f"\n{'='*60}")
print(f"FAILURE ANALYSIS")
print(f"{'='*60}")

# 1. ACCURACY BY 6-MONTH CHUNKS
print(f"\n=== 1. ACCURACY BY TIME PERIOD (6-month chunks) ===")
chunk_size = 4380
for ci in range(n_1h // chunk_size):
    s = ci * chunk_size
    e = min(s + chunk_size, n_1h)
    mask = valid & (np.arange(n_1h) >= s) & (np.arange(n_1h) < e) & (direction != 0)
    if mask.sum() > 100:
        acc = ((direction[mask] > 0) == (outcomes[mask] == 1)).sum() / mask.sum()
        price_chg = (c_1h[min(e-1,n_1h-1)] - c_1h[s]) / c_1h[s] * 100
        date_s = pd.to_datetime(ts_1h[s], unit='ms').strftime('%Y-%m')
        date_e = pd.to_datetime(ts_1h[min(e-1,n_1h-1)], unit='ms').strftime('%Y-%m')
        print(f"  {date_s} to {date_e}: acc={acc:.4f}, price={price_chg:+.1f}%, n={mask.sum()}")

# 2. ACCURACY BY MARKET DIRECTION (bull vs bear periods)
print(f"\n=== 2. ACCURACY BY MARKET DIRECTION ===")
# Rolling 168H (1 week) return
ret_168 = np.zeros(n_1h)
for i in range(168, n_1h):
    ret_168[i] = (c_1h[i] - c_1h[i-168]) / c_1h[i-168] * 100

for label, cond in [
    ("Strong bull (>5%/week)", ret_168 > 5),
    ("Mild bull (1-5%/week)", (ret_168 > 1) & (ret_168 <= 5)),
    ("Flat (-1% to 1%)", (ret_168 >= -1) & (ret_168 <= 1)),
    ("Mild bear (-5 to -1%)", (ret_168 >= -5) & (ret_168 < -1)),
    ("Strong bear (<-5%/week)", ret_168 < -5),
]:
    mask = valid & cond & (direction != 0)
    if mask.sum() > 100:
        acc = ((direction[mask] > 0) == (outcomes[mask] == 1)).sum() / mask.sum()
        print(f"  {label:<30}: acc={acc:.4f}, n={mask.sum()}")

# 3. ACCURACY BY VOLATILITY
print(f"\n=== 3. ACCURACY BY VOLATILITY ===")
atr_1h = h_1h - l_1h
atr_pct = np.zeros(n_1h)
for i in range(288, n_1h):
    window = atr_1h[max(0,i-288):i]
    atr_pct[i] = np.searchsorted(np.sort(window), atr_1h[i]) / len(window)

for label, lo_pct, hi_pct in [
    ("Low vol (0-20%)", 0, 0.2),
    ("Normal vol (20-50%)", 0.2, 0.5),
    ("High vol (50-80%)", 0.5, 0.8),
    ("Extreme vol (80-100%)", 0.8, 1.01),
]:
    mask = valid & (atr_pct >= lo_pct) & (atr_pct < hi_pct) & (direction != 0)
    if mask.sum() > 100:
        acc = ((direction[mask] > 0) == (outcomes[mask] == 1)).sum() / mask.sum()
        print(f"  {label:<30}: acc={acc:.4f}, n={mask.sum()}")

# 4. ACCURACY BY FACTOR AGREEMENT
print(f"\n=== 4. ACCURACY BY FACTOR AGREEMENT ===")
n_agree = np.zeros(n_1h)
# Count how many factors agree with final direction
for i in range(n_1h):
    d = direction[i]
    if d == 0: continue
    cnt = 0
    if mr1[i] == d: cnt += 1
    if mr2[i] == d: cnt += 1
    if (rsi_vals[i] < 35 and d > 0) or (rsi_vals[i] > 65 and d < 0): cnt += 1
    if (cvd_z[i] > 0.5 and d > 0) or (cvd_z[i] < -0.5 and d < 0): cnt += 1
    if np.sign(fb[i]) == d: cnt += 1
    n_agree[i] = cnt

for n_ag in range(6):
    mask = valid & (n_agree == n_ag) & (direction != 0)
    if mask.sum() > 50:
        acc = ((direction[mask] > 0) == (outcomes[mask] == 1)).sum() / mask.sum()
        print(f"  {n_ag} factors agree: acc={acc:.4f}, n={mask.sum()}")

# 5. ACCURACY BY SCORE STRENGTH
print(f"\n=== 5. ACCURACY BY SCORE STRENGTH ===")
abs_score = np.abs(score)
for lo_s, hi_s in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 99)]:
    mask = valid & (abs_score >= lo_s) & (abs_score < hi_s) & (direction != 0)
    if mask.sum() > 50:
        acc = ((direction[mask] > 0) == (outcomes[mask] == 1)).sum() / mask.sum()
        print(f"  |score| [{lo_s:.1f}-{hi_s:.1f}): acc={acc:.4f}, n={mask.sum()}")

# 6. ACCURACY BY HOUR OF DAY
print(f"\n=== 6. ACCURACY BY HOUR OF DAY ===")
hours = np.array([pd.to_datetime(t, unit='ms').hour for t in ts_1h])
for hour_start in range(0, 24, 4):
    hour_end = hour_start + 4
    mask = valid & (hours >= hour_start) & (hours < hour_end) & (direction != 0)
    if mask.sum() > 100:
        acc = ((direction[mask] > 0) == (outcomes[mask] == 1)).sum() / mask.sum()
        print(f"  {hour_start:02d}:00-{hour_end:02d}:00 UTC: acc={acc:.4f}, n={mask.sum()}")

# 7. ACCURACY BY DAY OF WEEK
print(f"\n=== 7. ACCURACY BY DAY OF WEEK ===")
days = np.array([pd.to_datetime(t, unit='ms').dayofweek for t in ts_1h])
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for d in range(7):
    mask = valid & (days == d) & (direction != 0)
    if mask.sum() > 100:
        acc = ((direction[mask] > 0) == (outcomes[mask] == 1)).sum() / mask.sum()
        print(f"  {day_names[d]}: acc={acc:.4f}, n={mask.sum()}")

# 8. WORST STREAKS
print(f"\n=== 8. WORST LOSING STREAKS ===")
correct = ((direction > 0) == (outcomes == 1)) & valid & (direction != 0)
streak = 0
worst_streak = 0
worst_streak_start = 0
current_start = 0
for i in range(n_1h):
    if not valid[i] or direction[i] == 0:
        streak = 0
        continue
    if not correct[i]:
        if streak == 0: current_start = i
        streak += 1
        if streak > worst_streak:
            worst_streak = streak
            worst_streak_start = current_start
    else:
        streak = 0

print(f"  Worst streak: {worst_streak} consecutive wrong predictions")
if worst_streak > 0:
    ws = worst_streak_start
    we = ws + worst_streak
    date_s = pd.to_datetime(ts_1h[ws], unit='ms')
    date_e = pd.to_datetime(ts_1h[min(we, n_1h-1)], unit='ms')
    price_chg = (c_1h[min(we, n_1h-1)] - c_1h[ws]) / c_1h[ws] * 100
    print(f"  Period: {date_s} to {date_e}")
    print(f"  Price change: {price_chg:+.1f}%")
