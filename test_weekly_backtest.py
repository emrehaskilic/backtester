"""
Haftalık Backtest: 1H MR+RSI+CVD + 4H/8H trend
Kasa: $1000, Margin: $25, Kaldıraç: 25x → Pozisyon: $625
Her hafta sonu kar çekilir, $1000 ile tekrar başlanır.
Liq: Kasa $0'a düşerse → o hafta biter, kayıp = kalan kasa - 1000
Fee: %0.04 taker (giriş + çıkış)
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
# SCORING
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

trend_4h = np.sign(c_4h - ema(c_4h, 10))
trend_8h = np.sign(c_8h - ema(c_8h, 6))

trend_4h_on_1h = np.zeros(n_1h)
for i in range(n_4h):
    s, e = i * 4, min((i + 1) * 4, n_1h)
    trend_4h_on_1h[s:e] = trend_4h[i]

trend_8h_on_1h = np.zeros(n_1h)
for i in range(n_8h):
    s, e = i * 8, min((i + 1) * 8, n_1h)
    trend_8h_on_1h[s:e] = trend_8h[i]

score_final = score + trend_4h_on_1h * 0.5 + trend_8h_on_1h * 0.5
direction = np.sign(score_final)

# ═══════════════════════════════════════════════════════════════
# BACKTEST PARAMS
# ═══════════════════════════════════════════════════════════════
INITIAL_CAPITAL = 1000.0
MARGIN = 25.0
LEVERAGE = 25
POSITION_SIZE = MARGIN * LEVERAGE  # $625
FEE_RATE = 0.0004  # %0.04 taker
K = 2  # 2 bar holding

# OOS start
train_end = int(n_1h * 0.6)

# Timestamps for week grouping
ts_1h = pd.to_datetime(df["open_time"].values[::12], unit='ms')

# ═══════════════════════════════════════════════════════════════
# HAFTALIK BACKTEST
# ═══════════════════════════════════════════════════════════════
print(f"Kasa: ${INITIAL_CAPITAL}")
print(f"Margin: ${MARGIN}, Kaldıraç: {LEVERAGE}x, Pozisyon: ${POSITION_SIZE}")
print(f"Fee: %{FEE_RATE*100:.2f} taker (giriş+çıkış)")
print(f"OOS başlangıç: {ts_1h[train_end].strftime('%Y-%m-%d')}")
print(f"K={K} bar holding")
print()

# Group OOS bars by week (Monday-Sunday)
oos_dates = ts_1h[train_end:n_1h]
week_labels = oos_dates.isocalendar().week.values + oos_dates.year.values * 100

weeks = []
current_week = week_labels[0]
week_start = 0
for i in range(1, len(week_labels)):
    if week_labels[i] != current_week:
        weeks.append((week_start, i))
        current_week = week_labels[i]
        week_start = i
weeks.append((week_start, len(week_labels)))

print(f"Toplam hafta: {len(weeks)}")
print()
print("=" * 90)
print(f"{'Hafta':>5} {'Tarih':>12} {'Kasa Başı':>10} {'Kasa Sonu':>10} {'Kar/Zarar':>10} {'Kar%':>7} {'Trade':>6} {'WR%':>6} {'Liq':>4}")
print("=" * 90)

total_withdrawn = 0.0
total_weeks = 0
win_weeks = 0
lose_weeks = 0
liq_weeks = 0
weekly_pnls = []

for w_idx, (ws, we) in enumerate(weeks):
    capital = INITIAL_CAPITAL
    n_trades = 0
    n_wins = 0
    liquidated = False

    i = ws
    while i < we:
        bar_idx = train_end + i
        if bar_idx + K >= n_1h:
            break

        d = direction[bar_idx]
        if d == 0:
            d = 1  # default long

        entry_price = c_1h[bar_idx]
        exit_price = c_1h[bar_idx + K]

        # PnL calculation
        ret = (exit_price - entry_price) / entry_price
        trade_pnl = d * ret * POSITION_SIZE

        # Fees: entry + exit
        fee = POSITION_SIZE * FEE_RATE * 2

        capital += trade_pnl - fee
        n_trades += 1
        if trade_pnl > 0:
            n_wins += 1

        # Liquidation check: kasa <= 0
        if capital <= 0:
            capital = 0
            liquidated = True
            break

        i += K  # next trade after holding period

    pnl = capital - INITIAL_CAPITAL
    pnl_pct = pnl / INITIAL_CAPITAL * 100
    wr = n_wins / n_trades * 100 if n_trades > 0 else 0

    week_date = oos_dates[ws].strftime('%Y-%m-%d')
    liq_mark = "LIQ" if liquidated else ""

    weekly_pnls.append(pnl)
    total_weeks += 1
    if pnl > 0:
        win_weeks += 1
        total_withdrawn += pnl  # kar çekilir
    elif pnl < 0:
        lose_weeks += 1
    if liquidated:
        liq_weeks += 1

    print(f"{w_idx+1:>5} {week_date:>12} ${INITIAL_CAPITAL:>9.2f} ${capital:>9.2f} ${pnl:>+9.2f} {pnl_pct:>+6.1f}% {n_trades:>5} {wr:>5.1f}% {liq_mark:>4}")

# ═══════════════════════════════════════════════════════════════
# ÖZET
# ═══════════════════════════════════════════════════════════════
print("=" * 90)
print()
print("=" * 60)
print("ÖZET")
print("=" * 60)

weekly_pnls = np.array(weekly_pnls)
print(f"  Toplam hafta:       {total_weeks}")
print(f"  Karlı hafta:        {win_weeks} ({win_weeks/total_weeks*100:.1f}%)")
print(f"  Zararlı hafta:      {lose_weeks} ({lose_weeks/total_weeks*100:.1f}%)")
print(f"  Likide olan hafta:  {liq_weeks}")
print()
print(f"  Toplam çekilen kar: ${total_withdrawn:,.2f}")
print(f"  Toplam zarar:       ${weekly_pnls[weekly_pnls<0].sum():,.2f}")
print(f"  Net kar:            ${total_withdrawn + weekly_pnls[weekly_pnls<0].sum():,.2f}")
print()
print(f"  Ort. haftalık kar:  ${weekly_pnls.mean():,.2f} ({weekly_pnls.mean()/INITIAL_CAPITAL*100:+.2f}%)")
print(f"  En iyi hafta:       ${weekly_pnls.max():,.2f} ({weekly_pnls.max()/INITIAL_CAPITAL*100:+.2f}%)")
print(f"  En kötü hafta:      ${weekly_pnls.min():,.2f} ({weekly_pnls.min()/INITIAL_CAPITAL*100:+.2f}%)")
print(f"  Std haftalık:       ${weekly_pnls.std():,.2f}")
print()

# Arka arkaya zarar serisi
max_losing_streak = 0
current_streak = 0
for p in weekly_pnls:
    if p < 0:
        current_streak += 1
        max_losing_streak = max(max_losing_streak, current_streak)
    else:
        current_streak = 0
print(f"  Max losing streak:  {max_losing_streak} hafta")

# Arka arkaya kar serisi
max_winning_streak = 0
current_streak = 0
for p in weekly_pnls:
    if p > 0:
        current_streak += 1
        max_winning_streak = max(max_winning_streak, current_streak)
    else:
        current_streak = 0
print(f"  Max winning streak: {max_winning_streak} hafta")

print("\nDone!")
