"""
Basic Strategy: 1H MR+RSI+CVD + 4H trend + 8H trend
%100 coverage, %54 OOS accuracy baseline

Strateji:
- Her 1H bar'da long veya short pozisyon
- Sinyal: combined score > 0 → long, < 0 → short
- K=2 bar (2 saat) holding period
- PnL: sign(score) * (close[i+2] - close[i]) / close[i]
"""
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv, sv = df["buy_vol"].values, df["sell_vol"].values

def aggregate(period):
    n = len(df) // period
    o_a = np.zeros(n); c_a = np.zeros(n); h_a = np.zeros(n); l_a = np.zeros(n)
    bv_a = np.zeros(n); sv_a = np.zeros(n)
    for i in range(n):
        s, e = i * period, i * period + period
        o_a[i] = o[s]; c_a[i] = c[e-1]; h_a[i] = h[s:e].max(); l_a[i] = l[s:e].min()
        bv_a[i] = bv[s:e].sum(); sv_a[i] = sv[s:e].sum()
    return n, o_a, c_a, h_a, l_a, bv_a, sv_a

n_1h, o_1h, c_1h, h_1h, l_1h, bv_1h, sv_1h = aggregate(12)
n_4h, o_4h, c_4h, h_4h, l_4h, bv_4h, sv_4h = aggregate(48)
n_8h, o_8h, c_8h, h_8h, l_8h, bv_8h, sv_8h = aggregate(96)

# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════
def ema(data, span):
    k = 2.0 / (span + 1)
    out = np.zeros_like(data, dtype=np.float64); out[0] = data[0]
    for i in range(1, len(data)): out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

def compute_rsi(data, period):
    rsi = np.full(len(data), 50.0)
    gains = np.zeros(len(data)); losses = np.zeros(len(data))
    for i in range(1, len(data)):
        d = data[i] - data[i-1]
        if d > 0: gains[i] = d
        else: losses[i] = -d
    ag = ema(gains, period); al = ema(losses, period)
    for i in range(period, len(data)):
        if al[i] > 1e-15: rsi[i] = 100 - 100 / (1 + ag[i] / al[i])
    return rsi

def cvd_zscore(bv, sv, window):
    cvd_bar = bv - sv; ce = ema(cvd_bar, window)
    std_arr = np.zeros(len(bv))
    for i in range(window, len(bv)):
        diff = cvd_bar[i-window:i] - ce[i-window:i]
        std_arr[i] = np.sqrt(np.mean(diff**2))
    z = np.zeros(len(bv))
    for i in range(window, len(bv)):
        if std_arr[i] > 1e-15: z[i] = (cvd_bar[i] - ce[i]) / std_arr[i]
    return z

# ═══════════════════════════════════════════════════════════════
# 1H BASE SCORING: MR + RSI + CVD + Agreement
# ═══════════════════════════════════════════════════════════════
e1 = ema(c_1h, 32); e2 = ema(c_1h, 56)
mr1 = np.where(c_1h > e1, -1.0, 1.0)
mr2 = np.where(c_1h > e2, -1.0, 1.0)
rsi = compute_rsi(c_1h, 18)
cvd_z = cvd_zscore(bv_1h, sv_1h, 24)

score = mr1 * 0.9
agree = (mr1 == mr2)
score += np.where(agree, mr1 * 0.6, 0)
score += np.where(((rsi < 35) & (mr1 > 0)) | ((rsi > 65) & (mr1 < 0)), mr1 * 1.0, 0)
score += np.where(cvd_z > 0.5, 1.0, np.where(cvd_z < -0.5, -1.0, 0))
score += np.where(agree & (np.abs(cvd_z) > 0.5) & (np.sign(cvd_z) == mr1), mr1 * 1.4, 0)

# ═══════════════════════════════════════════════════════════════
# HTF TREND: 4H EMA(10) + 8H EMA(6) → mapped to 1H
# ═══════════════════════════════════════════════════════════════
trend_4h = np.sign(c_4h - ema(c_4h, 10))  # +1 = bullish, -1 = bearish
trend_8h = np.sign(c_8h - ema(c_8h, 6))

# Map to 1H bars
trend_4h_on_1h = np.zeros(n_1h)
for i in range(n_4h):
    s, e = i * 4, min((i + 1) * 4, n_1h)
    trend_4h_on_1h[s:e] = trend_4h[i]

trend_8h_on_1h = np.zeros(n_1h)
for i in range(n_8h):
    s, e = i * 8, min((i + 1) * 8, n_1h)
    trend_8h_on_1h[s:e] = trend_8h[i]

# Final score
score_final = score + trend_4h_on_1h * 0.5 + trend_8h_on_1h * 0.5

# ═══════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════
K = 2  # 2 bar holding
direction = np.sign(score_final)

# Walk-forward: train 60%, test in 5 chunks
warmup = 100
train_end = int(n_1h * 0.6)
test_bars = n_1h - K - train_end
chunk_size = test_bars // 5

print(f"Data: {n_1h:,} bars ({n_1h/24/365:.1f} yıl)")
print(f"Train: 0-{train_end:,} ({train_end:,} bar)")
print(f"Test:  {train_end:,}-{n_1h:,} ({test_bars:,} bar, {5} chunk)")
print()

# ── In-sample metrics ──
print("=" * 70)
print("IN-SAMPLE (train)")
print("=" * 70)
is_correct = 0; is_total = 0; is_pnl = 0.0
is_pnl_list = []
for i in range(warmup, train_end):
    if i + K >= n_1h: break
    d = direction[i]
    if d == 0: d = 1  # force long if neutral (shouldn't happen)
    ret = (c_1h[i + K] - c_1h[i]) / c_1h[i]
    pnl = d * ret
    is_pnl_list.append(pnl)
    is_pnl += pnl
    is_correct += 1 if (d > 0) == (c_1h[i + K] > c_1h[i]) else 0
    is_total += 1

is_acc = is_correct / is_total
is_pnl_arr = np.array(is_pnl_list)
print(f"  Accuracy:    {is_acc:.4f} ({is_acc*100:.2f}%)")
print(f"  Total PnL:   {is_pnl*100:.2f}%")
print(f"  Bars:        {is_total:,}")
print(f"  Avg PnL/bar: {is_pnl/is_total*10000:.2f} bps")
neutral_is = ((direction[warmup:train_end] == 0).sum())
print(f"  Neutral bars: {neutral_is} ({neutral_is/is_total*100:.1f}%)")
print()

# ── OOS metrics per chunk ──
print("=" * 70)
print("OUT-OF-SAMPLE (5 chunks)")
print("=" * 70)
chunk_accs = []; chunk_pnls = []; chunk_sharpes = []
total_oos_correct = 0; total_oos_bars = 0; total_oos_pnl = 0.0
all_oos_pnl = []

for ch in range(5):
    cs = train_end + ch * chunk_size
    ce = cs + chunk_size if ch < 4 else n_1h - K

    correct = 0; total = 0; pnl_list = []
    for i in range(cs, ce):
        if i + K >= n_1h: break
        d = direction[i]
        if d == 0: d = 1
        ret = (c_1h[i + K] - c_1h[i]) / c_1h[i]
        pnl = d * ret
        pnl_list.append(pnl)
        correct += 1 if (d > 0) == (c_1h[i + K] > c_1h[i]) else 0
        total += 1

    acc = correct / total if total > 0 else 0
    pnl_arr = np.array(pnl_list)
    total_pnl = pnl_arr.sum()
    sharpe = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(24 * 365)) if pnl_arr.std() > 0 else 0

    chunk_accs.append(acc)
    chunk_pnls.append(total_pnl)
    chunk_sharpes.append(sharpe)
    total_oos_correct += correct
    total_oos_bars += total
    total_oos_pnl += total_pnl
    all_oos_pnl.extend(pnl_list)

    print(f"  Chunk {ch+1}: acc={acc:.4f} ({acc*100:.2f}%)  pnl={total_pnl*100:+.2f}%  sharpe={sharpe:.2f}  bars={total:,}")

oos_acc = total_oos_correct / total_oos_bars
oos_pnl_arr = np.array(all_oos_pnl)
oos_sharpe = (oos_pnl_arr.mean() / oos_pnl_arr.std() * np.sqrt(24 * 365)) if oos_pnl_arr.std() > 0 else 0

print(f"\n  Overall OOS:")
print(f"    Accuracy:     {oos_acc:.4f} ({oos_acc*100:.2f}%)")
print(f"    Total PnL:    {total_oos_pnl*100:+.2f}%")
print(f"    Avg PnL/bar:  {total_oos_pnl/total_oos_bars*10000:.2f} bps")
print(f"    Sharpe (ann.): {oos_sharpe:.2f}")
print(f"    Bars:         {total_oos_bars:,}")
neutral_oos = ((direction[train_end:n_1h-K] == 0).sum())
print(f"    Neutral bars: {neutral_oos} ({neutral_oos/total_oos_bars*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# EQUITY CURVE (OOS only)
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("EQUITY CURVE (OOS)")
print("=" * 70)
cum_pnl = np.cumsum(oos_pnl_arr)
max_dd = 0; peak = 0
for v in cum_pnl:
    if v > peak: peak = v
    dd = peak - v
    if dd > max_dd: max_dd = dd

print(f"  Final equity:  {cum_pnl[-1]*100:+.2f}%")
print(f"  Max drawdown:  {max_dd*100:.2f}%")
print(f"  Peak equity:   {peak*100:.2f}%")

# Monthly breakdown
print(f"\n  Aylık PnL:")
dates_oos = pd.to_datetime(df["open_time"].values[::12][train_end:train_end+len(oos_pnl_arr)], unit='ms')
monthly = pd.Series(oos_pnl_arr, index=dates_oos).resample('ME').sum()
for date, pnl in monthly.items():
    bar = "+" * int(abs(pnl) * 1000) if pnl > 0 else "-" * int(abs(pnl) * 1000)
    print(f"    {date.strftime('%Y-%m')}: {pnl*100:+6.2f}%  {bar}")

# ═══════════════════════════════════════════════════════════════
# TRADE STATS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TRADE STATS (OOS)")
print("=" * 70)
wins = oos_pnl_arr[oos_pnl_arr > 0]
losses = oos_pnl_arr[oos_pnl_arr < 0]
print(f"  Win rate:      {len(wins)/len(oos_pnl_arr)*100:.2f}%")
print(f"  Avg win:       {wins.mean()*100:.4f}%")
print(f"  Avg loss:      {losses.mean()*100:.4f}%")
print(f"  Win/Loss ratio: {abs(wins.mean()/losses.mean()):.3f}")
print(f"  Profit factor: {wins.sum()/abs(losses.sum()):.3f}")
print(f"  Total trades:  {len(oos_pnl_arr):,}")

# Longs vs Shorts
long_mask = np.array([direction[train_end + i] > 0 for i in range(len(oos_pnl_arr))])
short_mask = ~long_mask
long_pnl = oos_pnl_arr[long_mask]
short_pnl = oos_pnl_arr[short_mask]
print(f"\n  Long trades:   {long_mask.sum():,}  acc={((long_pnl>0).sum()/len(long_pnl)*100):.2f}%  pnl={long_pnl.sum()*100:+.2f}%")
print(f"  Short trades:  {short_mask.sum():,}  acc={((short_pnl>0).sum()/len(short_pnl)*100):.2f}%  pnl={short_pnl.sum()*100:+.2f}%")

print("\nDone!")
