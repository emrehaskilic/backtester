"""
4H prediction with 2H and 1H sub-signals.
Predict next 4H candle direction using multi-TF information.
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

def build_bars(period):
    n = len(df) // period
    _c=np.zeros(n);_h=np.zeros(n);_l=np.zeros(n);_o=np.zeros(n)
    _bv=np.zeros(n);_sv=np.zeros(n)
    for i in range(n):
        s,e=i*period,i*period+period
        _o[i]=o[s];_c[i]=c[e-1];_h[i]=h[s:e].max();_l[i]=l[s:e].min()
        _bv[i]=bv[s:e].sum();_sv[i]=sv[s:e].sum()
    return n, _o, _h, _l, _c, _bv, _sv

# Build all TF bars
n_1h, o_1h, h_1h, l_1h, c_1h, bv_1h, sv_1h = build_bars(12)
n_2h, o_2h, h_2h, l_2h, c_2h, bv_2h, sv_2h = build_bars(24)
n_4h, o_4h, h_4h, l_4h, c_4h, bv_4h, sv_4h = build_bars(48)

print(f"1H: {n_1h}, 2H: {n_2h}, 4H: {n_4h}")

# 4H outcomes (K=1)
outcomes = np.zeros(n_4h, dtype=np.int8)
for i in range(n_4h-1):
    outcomes[i] = 1 if c_4h[i+1] > c_4h[i] else 0
outcomes[-1] = -1
valid = (outcomes >= 0) & (np.arange(n_4h) >= 100)

# ── 4H signals ──
ema_4h = ema(c_4h, 8)
mr_4h = np.where(c_4h > ema_4h, -1.0, 1.0)
rsi_4h = rsi(c_4h, 6)
ret_4h = np.where(o_4h > 0, (c_4h - o_4h) / o_4h * 100, 0)
cvd_4h = bv_4h - sv_4h
imb_4h = cvd_4h / np.maximum(bv_4h + sv_4h, 1e-10)

score_4h = np.zeros(n_4h)
score_4h += mr_4h * 1.0
score_4h += np.where(rsi_4h < 35, 1, np.where(rsi_4h > 65, -1, 0)) * 1.0
score_4h += np.where(ret_4h > 1, -1, np.where(ret_4h < -1, 1, 0)) * 1.0

# ── 2H signals mapped to 4H ──
# Each 4H bar = 2x 2H bars. Use the LAST (2nd) 2H bar as most recent info
ema_2h = ema(c_2h, 12)
mr_2h = np.where(c_2h > ema_2h, -1.0, 1.0)
rsi_2h = rsi(c_2h, 8)
ret_2h = np.where(o_2h > 0, (c_2h - o_2h) / o_2h * 100, 0)
imb_2h = (bv_2h - sv_2h) / np.maximum(bv_2h + sv_2h, 1e-10)

# Map to 4H index
mr_2h_last = np.zeros(n_4h)     # last 2H MR signal
rsi_2h_last = np.zeros(n_4h)
ret_2h_last = np.zeros(n_4h)    # last 2H candle return
imb_2h_last = np.zeros(n_4h)
ret_2h_first = np.zeros(n_4h)   # first 2H candle return
momentum_2h = np.zeros(n_4h)    # 2nd 2H vs 1st 2H (acceleration)

for i in range(n_4h):
    i2_first = i * 2
    i2_last = i * 2 + 1
    if i2_last < n_2h:
        mr_2h_last[i] = mr_2h[i2_last]
        rsi_2h_last[i] = rsi_2h[i2_last]
        ret_2h_last[i] = ret_2h[i2_last]
        ret_2h_first[i] = ret_2h[i2_first]
        imb_2h_last[i] = imb_2h[i2_last]
        momentum_2h[i] = ret_2h[i2_last] - ret_2h[i2_first]  # accelerating?

# ── 1H signals mapped to 4H ──
ema_1h = ema(c_1h, 20)
mr_1h = np.where(c_1h > ema_1h, -1.0, 1.0)
rsi_1h = rsi(c_1h, 10)
ret_1h = np.where(o_1h > 0, (c_1h - o_1h) / o_1h * 100, 0)
imb_1h = (bv_1h - sv_1h) / np.maximum(bv_1h + sv_1h, 1e-10)

# Map to 4H: last 1H bar, and trend of 4 bars
mr_1h_last = np.zeros(n_4h)
rsi_1h_last = np.zeros(n_4h)
ret_1h_last = np.zeros(n_4h)
imb_1h_last = np.zeros(n_4h)
trend_1h_4bar = np.zeros(n_4h)  # HH/HL or LL/LH pattern
momentum_1h = np.zeros(n_4h)    # last 1H vs first 1H

for i in range(n_4h):
    i1_start = i * 4
    i1_end = i * 4 + 3
    if i1_end < n_1h:
        mr_1h_last[i] = mr_1h[i1_end]
        rsi_1h_last[i] = rsi_1h[i1_end]
        ret_1h_last[i] = ret_1h[i1_end]
        imb_1h_last[i] = imb_1h[i1_end]
        momentum_1h[i] = ret_1h[i1_end] - ret_1h[i1_start]

        # 4-bar trend: count up vs down closes
        ups = sum(1 for j in range(i1_start, i1_end+1) if c_1h[j] > o_1h[j])
        trend_1h_4bar[i] = (ups - 2) / 2  # -1 to +1

def test(name, signal, mn=50):
    mask = (signal != 0) & valid
    if mask.sum() < mn: return
    actual = outcomes[mask] == 1
    pred = signal[mask] > 0
    acc = (pred == actual).sum() / mask.sum()
    cov = mask.sum() / valid.sum() * 100
    print(f"  {name:<55}: cov={cov:>5.1f}%, acc={acc:.4f}, n={mask.sum()}")

print(f"\n{'='*60}")
print(f"TEST 1: BASELINE - 4H ALONE")
print(f"{'='*60}")
test("4H MR EMA-8", mr_4h)
test("4H RSI-6 (35/65)", np.where(rsi_4h<35,1,np.where(rsi_4h>65,-1,0)))
test("4H Candle MR >1%", np.where(ret_4h>1,-1,np.where(ret_4h<-1,1,0)))
test("4H Combined", np.sign(score_4h))

print(f"\n{'='*60}")
print(f"TEST 2: 4H + LAST 2H SIGNAL")
print(f"{'='*60}")
test("Last 2H MR", np.sign(mr_2h_last))
test("Last 2H MR (mean-revert last 2H ret)", np.where(ret_2h_last>0.3,-1,np.where(ret_2h_last<-0.3,1,0)))
# 4H + 2H agreement
agree_4h_2h = np.where(np.sign(score_4h) == mr_2h_last, np.sign(score_4h), 0)
test("4H + 2H agree", agree_4h_2h)
disagree_4h_2h = np.where((np.sign(score_4h) != mr_2h_last) & (np.sign(score_4h)!=0) & (mr_2h_last!=0), mr_2h_last, 0)
test("4H disagree, use 2H", disagree_4h_2h)

# 2H momentum (2nd half vs 1st half of 4H candle)
test("2H momentum (accelerating)", np.where(momentum_2h>0.3,1,np.where(momentum_2h<-0.3,-1,0)))
test("2H momentum MR", np.where(momentum_2h>0.3,-1,np.where(momentum_2h<-0.3,1,0)))

print(f"\n{'='*60}")
print(f"TEST 3: 4H + LAST 1H SIGNAL")
print(f"{'='*60}")
test("Last 1H MR", np.sign(mr_1h_last))
test("Last 1H ret MR", np.where(ret_1h_last>0.3,-1,np.where(ret_1h_last<-0.3,1,0)))
test("Last 1H imbalance", np.sign(imb_1h_last))
# 4H + 1H agreement
agree_4h_1h = np.where(np.sign(score_4h) == mr_1h_last, np.sign(score_4h), 0)
test("4H + 1H agree", agree_4h_1h)
# 1H trend within 4H candle
test("1H 4-bar trend", np.sign(trend_1h_4bar))
test("1H 4-bar trend MR", -np.sign(trend_1h_4bar))
# 1H momentum
test("1H momentum (last vs first)", np.where(momentum_1h>0.3,1,np.where(momentum_1h<-0.3,-1,0)))
test("1H momentum MR", np.where(momentum_1h>0.3,-1,np.where(momentum_1h<-0.3,1,0)))

print(f"\n{'='*60}")
print(f"TEST 4: ALL TF AGREE/DISAGREE")
print(f"{'='*60}")
all_agree = np.where(
    (np.sign(score_4h) == mr_2h_last) & (mr_2h_last == mr_1h_last) & (np.sign(score_4h) != 0),
    np.sign(score_4h), 0
)
test("4H+2H+1H ALL AGREE (trend)", all_agree)
test("4H+2H+1H ALL AGREE (MR)", -all_agree)

# 4H says one thing, but 2H and 1H both say opposite
lower_override = np.where(
    (mr_2h_last == mr_1h_last) & (mr_2h_last != np.sign(score_4h)) & (mr_2h_last != 0) & (np.sign(score_4h) != 0),
    mr_2h_last, 0
)
test("2H+1H override 4H", lower_override)

print(f"\n{'='*60}")
print(f"TEST 5: 4H WEAK + LOWER TF STRONG")
print(f"{'='*60}")
for weak_t in [0.5, 1.0]:
    # 4H weak
    weak_4h = np.abs(score_4h) < weak_t

    # 2H strong signal
    strong_2h = np.abs(ret_2h_last) > 0.5
    combo_2h = np.where(weak_4h & strong_2h, np.where(ret_2h_last>0,-1,1), np.sign(score_4h))
    test(f"4H weak(<{weak_t}) + 2H candle MR", np.sign(combo_2h))

    # 1H strong signal
    strong_1h = np.abs(ret_1h_last) > 0.3
    combo_1h = np.where(weak_4h & strong_1h, np.where(ret_1h_last>0,-1,1), np.sign(score_4h))
    test(f"4H weak(<{weak_t}) + 1H candle MR", np.sign(combo_1h))

print(f"\n{'='*60}")
print(f"TEST 6: 4H CANDLE MR + 1H REVERSAL CONFIRMATION")
print(f"{'='*60}")
# 4H big candle (>1%) + last 1H shows reversal (opposite direction)
big_bull_4h = ret_4h > 1.0
big_bear_4h = ret_4h < -1.0
last_1h_bear = ret_1h_last < -0.1
last_1h_bull = ret_1h_last > 0.1

# 4H big bull + last 1H turning bear = MR confirmed
mr_confirmed_bear = np.where(big_bull_4h & last_1h_bear, -1, 0)
mr_confirmed_bull = np.where(big_bear_4h & last_1h_bull, 1, 0)
mr_confirmed = mr_confirmed_bear + mr_confirmed_bull
test("4H big candle + 1H reversal starting", mr_confirmed)

# Opposite: 4H big + 1H continuing = trend continuation
cont_bull = np.where(big_bull_4h & last_1h_bull, 1, 0)
cont_bear = np.where(big_bear_4h & last_1h_bear, -1, 0)
continuation = cont_bull + cont_bear
test("4H big candle + 1H continuing (trend)", continuation)

print(f"\n{'='*60}")
print(f"TEST 7: BEST COMBINED SCORE")
print(f"{'='*60}")
# Combine best signals
best_score = np.zeros(n_4h)
best_score += mr_4h * 0.5                          # 4H MR
best_score += np.where(rsi_4h<35,1,np.where(rsi_4h>65,-1,0)) * 0.7  # 4H RSI
best_score += np.where(ret_4h>1,-1,np.where(ret_4h<-1,1,0)) * 1.0   # 4H candle MR
best_score += np.where(ret_2h_last>0.3,-1,np.where(ret_2h_last<-0.3,1,0)) * 0.5  # 2H MR
best_score += np.where(ret_1h_last>0.2,-1,np.where(ret_1h_last<-0.2,1,0)) * 0.5  # 1H MR
best_score += np.where(rsi_1h_last<35,1,np.where(rsi_1h_last>65,-1,0)) * 0.3     # 1H RSI

test("Best combined (4H+2H+1H)", np.sign(best_score))
for s in [0.5, 1.0, 1.5, 2.0, 2.5]:
    test(f"  |score|>={s}", np.where(np.abs(best_score)>=s, np.sign(best_score), 0))
