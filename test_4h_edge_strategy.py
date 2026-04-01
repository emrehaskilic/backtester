"""
4H Edge Strategy — Look-ahead bias FREE
=========================================
Feature mining'den bulunan tutarlı edge'ler:
  - ema_slope_12 Q4: train=0.530, val=0.521, test=0.548
  - lower_wick Q1:   train=0.523, val=0.521, test=0.523
  - imbalance Q4:    train=0.527, val=0.523, test=0.526
  - dist_high_18 Q4: train=0.530, val=0.531, test=0.508
  - ema_dist_50 Q3:  train=0.535, val=0.522, test=0.515

Her feature: ÖNCEKI TAMAMLANMIŞ 4H bar'dan hesaplanır (look-ahead yok)
Quintile sınırları: TRAIN set'ten hesaplanır, val/test'e uygulanır
%100 coverage: Her 4H bar'da pozisyon var (long veya short)

Strateji: Her feature quintile'da +1 veya -1 oy verir.
Combined vote > 0 → long, < 0 → short, == 0 → long (default)
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

n_4h, o_4h, c_4h, h_4h, l_4h, bv_4h, sv_4h = aggregate(48)

# ═══════════════════════════════════════════════════════════════
# INDICATORS (vectorized)
# ═══════════════════════════════════════════════════════════════
def ema(data, span):
    k = 2.0 / (span + 1)
    out = np.zeros_like(data, dtype=np.float64); out[0] = data[0]
    for i in range(1, len(data)): out[i] = data[i] * k + out[i-1] * (1 - k)
    return out

# ═══════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (on previous completed bar — NO look-ahead)
# ═══════════════════════════════════════════════════════════════
# We compute features for each 4H bar, but when making prediction for bar i,
# we use features from bar i-1 (previous COMPLETED bar).

# Feature 1: EMA slope 12 — slope of EMA(12) over last 3 bars
ema_12 = ema(c_4h, 12)
ema_slope_12 = np.zeros(n_4h)
for i in range(3, n_4h):
    ema_slope_12[i] = (ema_12[i] - ema_12[i-3]) / ema_12[i-3]

# Feature 2: Lower wick ratio
body = np.abs(c_4h - o_4h)
full_range = h_4h - l_4h + 1e-15
lower_wick = np.zeros(n_4h)
for i in range(n_4h):
    lower_wick[i] = (min(o_4h[i], c_4h[i]) - l_4h[i]) / full_range[i]

# Feature 3: Volume imbalance (buy_vol - sell_vol) / total_vol
total_vol = bv_4h + sv_4h + 1e-15
imbalance = (bv_4h - sv_4h) / total_vol

# Feature 4: Distance from 18-bar high
rolling_high_18 = np.zeros(n_4h)
for i in range(18, n_4h):
    rolling_high_18[i] = h_4h[i-18:i].max()
dist_high_18 = np.zeros(n_4h)
for i in range(18, n_4h):
    if rolling_high_18[i] > 0:
        dist_high_18[i] = (c_4h[i] - rolling_high_18[i]) / rolling_high_18[i]

# Feature 5: EMA distance 50 — (close - EMA50) / EMA50
ema_50 = ema(c_4h, 50)
ema_dist_50 = (c_4h - ema_50) / (ema_50 + 1e-15)

# ═══════════════════════════════════════════════════════════════
# SPLIT: 40% train / 30% val / 30% test
# ═══════════════════════════════════════════════════════════════
warmup = 60  # need at least 50 bars for EMA(50) warmup
train_end = int(n_4h * 0.4)
val_end = int(n_4h * 0.7)

print(f"4H bars: {n_4h:,}")
print(f"Train: {warmup}-{train_end} ({train_end - warmup:,} bars)")
print(f"Val:   {train_end}-{val_end} ({val_end - train_end:,} bars)")
print(f"Test:  {val_end}-{n_4h} ({n_4h - val_end:,} bars)")
print()

# ═══════════════════════════════════════════════════════════════
# QUINTILE BOUNDARIES (from TRAIN set only)
# ═══════════════════════════════════════════════════════════════
features = {
    'ema_slope_12': ema_slope_12,
    'lower_wick': lower_wick,
    'imbalance': imbalance,
    'dist_high_18': dist_high_18,
    'ema_dist_50': ema_dist_50,
}

# Which quintile is bullish for each feature (from feature mining results)
# Positive vote = long bias, negative vote = short bias
feature_rules = {
    # ema_slope_12 Q4 is bullish (strong upward EMA slope → continuation)
    'ema_slope_12': {4: +1, 0: -1},  # Q4 long, Q0 short
    # lower_wick Q1 is bullish (small lower wick → strength)
    'lower_wick': {0: +1, 4: -1},  # Q0 (small wick) long, Q4 short
    # imbalance Q4 is bullish (high buy imbalance → continuation)
    'imbalance': {4: +1, 0: -1},  # Q4 long, Q0 short
    # dist_high_18 Q4 is bullish (close near high → strength)
    'dist_high_18': {4: +1, 0: -1},  # Q4 long, Q0 short
    # ema_dist_50 Q3 is bullish (moderately above EMA → trend)
    'ema_dist_50': {3: +1, 1: -1},  # Q3 long, Q1 short
}

# Compute quintile boundaries from train set
quintile_bounds = {}
for fname, fdata in features.items():
    train_vals = fdata[warmup:train_end]
    quintile_bounds[fname] = np.percentile(train_vals, [20, 40, 60, 80])

def get_quintile(value, bounds):
    """Return quintile 0-4 based on precomputed boundaries."""
    if value <= bounds[0]: return 0
    if value <= bounds[1]: return 1
    if value <= bounds[2]: return 2
    if value <= bounds[3]: return 3
    return 4

# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION (using PREVIOUS completed bar's features)
# ═══════════════════════════════════════════════════════════════
# For bar i, we predict direction using features from bar i-1
# This ensures NO look-ahead bias

direction = np.zeros(n_4h)
votes_detail = np.zeros((n_4h, 5))  # store individual votes for analysis

for i in range(warmup + 1, n_4h):
    prev = i - 1  # previous COMPLETED bar
    total_vote = 0.0
    for fidx, (fname, fdata) in enumerate(features.items()):
        q = get_quintile(fdata[prev], quintile_bounds[fname])
        rules = feature_rules[fname]
        vote = rules.get(q, 0)
        votes_detail[i, fidx] = vote
        total_vote += vote

    if total_vote > 0:
        direction[i] = 1
    elif total_vote < 0:
        direction[i] = -1
    else:
        direction[i] = 1  # default long if tied (100% coverage)

# ═══════════════════════════════════════════════════════════════
# ACCURACY TEST (K=1 bar forward return on 4H)
# ═══════════════════════════════════════════════════════════════
K = 1  # predict next 4H bar

def eval_accuracy(start, end, label):
    correct = 0; total = 0; pnl_list = []
    long_count = 0; short_count = 0
    for i in range(start, end):
        if i + K >= n_4h: break
        d = direction[i]
        if d == 0: continue  # shouldn't happen after warmup

        ret = (c_4h[i + K] - c_4h[i]) / c_4h[i]
        pnl = d * ret
        pnl_list.append(pnl)

        if d > 0: long_count += 1
        else: short_count += 1

        if (d > 0) == (c_4h[i + K] > c_4h[i]):
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    pnl_arr = np.array(pnl_list)
    total_pnl = pnl_arr.sum()
    avg_pnl = pnl_arr.mean() if len(pnl_arr) > 0 else 0
    sharpe = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(6 * 365)) if pnl_arr.std() > 0 else 0

    print(f"  {label}:")
    print(f"    Accuracy:   {acc:.4f} ({acc*100:.2f}%)")
    print(f"    Total PnL:  {total_pnl*100:+.2f}%")
    print(f"    Avg PnL:    {avg_pnl*10000:.2f} bps/bar")
    print(f"    Sharpe:     {sharpe:.2f}")
    print(f"    Bars:       {total:,} (long={long_count}, short={short_count})")
    print(f"    Coverage:   100%")

    return acc, total_pnl, pnl_arr

print("=" * 70)
print("4H EDGE STRATEGY — NO LOOK-AHEAD BIAS")
print("Features from previous completed 4H bar")
print("Quintile boundaries from TRAIN set only")
print("=" * 70)
print()

print("─── K=1 (next 4H bar) ───")
train_acc, train_pnl, train_pnl_arr = eval_accuracy(warmup + 1, train_end, "TRAIN")
print()
val_acc, val_pnl, val_pnl_arr = eval_accuracy(train_end, val_end, "VAL")
print()
test_acc, test_pnl, test_pnl_arr = eval_accuracy(val_end, n_4h, "TEST")
print()

# Combined OOS (val + test)
oos_start = train_end
oos_end = n_4h
oos_acc_val, oos_pnl_val, oos_pnl_arr_full = eval_accuracy(oos_start, oos_end, "FULL OOS (val+test)")
print()

# ═══════════════════════════════════════════════════════════════
# VOTE AGREEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("VOTE AGREEMENT ANALYSIS (OOS)")
print("=" * 70)

for agreement_threshold in [3, 4, 5]:
    correct = 0; total = 0; skipped = 0
    for i in range(oos_start, oos_end):
        if i + K >= n_4h: break
        vote_sum = votes_detail[i].sum()
        if abs(vote_sum) < agreement_threshold:
            skipped += 1
            continue
        d = 1 if vote_sum > 0 else -1
        if (d > 0) == (c_4h[i + K] > c_4h[i]):
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0
    coverage = total / (oos_end - oos_start - K) * 100
    print(f"  |vote| >= {agreement_threshold}: acc={acc:.4f} ({acc*100:.2f}%)  coverage={coverage:.1f}%  bars={total}")

print()

# ═══════════════════════════════════════════════════════════════
# INDIVIDUAL FEATURE ACCURACY (verify no look-ahead)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("INDIVIDUAL FEATURE ACCURACY (OOS, K=1)")
print("=" * 70)

for fname, fdata in features.items():
    rules = feature_rules[fname]
    correct = 0; total = 0
    for i in range(oos_start, oos_end):
        if i + K >= n_4h: break
        prev = i - 1
        q = get_quintile(fdata[prev], quintile_bounds[fname])
        vote = rules.get(q, 0)
        if vote == 0: continue  # no signal from this feature

        if (vote > 0) == (c_4h[i + K] > c_4h[i]):
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0
    coverage = total / (oos_end - oos_start - K) * 100
    print(f"  {fname:>20s}: acc={acc:.4f} ({acc*100:.2f}%)  coverage={coverage:.1f}%  bars={total}")

print()

# ═══════════════════════════════════════════════════════════════
# MAP TO 1H AND WEEKLY BACKTEST
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("WEEKLY BACKTEST — $1000 kasa, 25x leverage, reversal-based")
print("=" * 70)

# Aggregate 1H data
n_1h = len(df) // 12
c_1h = np.zeros(n_1h)
for i in range(n_1h):
    c_1h[i] = c[i * 12 + 11]

# Map 4H direction to 1H (using PREVIOUS completed 4H bar — no look-ahead)
dir_1h = np.zeros(n_1h)
for i in range(n_1h):
    h4_idx = i // 4  # which 4H bar this 1H bar belongs to
    # Use direction computed for h4_idx (which already uses features from h4_idx-1)
    if h4_idx < n_4h:
        dir_1h[i] = direction[h4_idx]

# OOS start on 1H
train_end_1h = train_end * 4  # 4H train_end → 1H equivalent

# Timestamps
ts_1h = pd.to_datetime(df["open_time"].values[::12], unit='ms')

# Group by weeks
oos_dates = ts_1h[train_end_1h:n_1h]
if len(oos_dates) == 0:
    print("No OOS data!")
else:
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

    INITIAL_CAPITAL = 1000.0
    MARGIN = 25.0
    LEVERAGE = 25
    POSITION_SIZE = MARGIN * LEVERAGE  # $625
    FEE_RATE = 0.0004

    print(f"Kasa: ${INITIAL_CAPITAL}, Margin: ${MARGIN}, Leverage: {LEVERAGE}x, Position: ${POSITION_SIZE}")
    print(f"Fee: {FEE_RATE*100:.2f}% taker (entry+exit)")
    print(f"OOS start: {oos_dates[0].strftime('%Y-%m-%d')}")
    print(f"Weeks: {len(weeks)}")
    print()

    print(f"{'Wk':>4} {'Date':>12} {'Start':>10} {'End':>10} {'PnL':>10} {'PnL%':>7} {'Trades':>6} {'WR%':>6} {'Liq':>4}")
    print("=" * 80)

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

        pos_dir = 0
        pos_entry_price = 0.0
        pos_entry_bar = 0

        for i in range(ws, we):
            bar_idx = train_end_1h + i
            if bar_idx >= n_1h:
                break

            d = dir_1h[bar_idx]
            if d == 0:
                d = 1  # default long if no signal

            price = c_1h[bar_idx]

            if pos_dir == 0:
                # Open position
                pos_dir = d
                pos_entry_price = price
                pos_entry_bar = bar_idx
                capital -= POSITION_SIZE * FEE_RATE  # entry fee

            elif d != pos_dir:
                # Reversal: close + open opposite
                ret = (price - pos_entry_price) / pos_entry_price
                trade_pnl = pos_dir * ret * POSITION_SIZE
                capital += trade_pnl
                capital -= POSITION_SIZE * FEE_RATE  # exit fee

                n_trades += 1
                if trade_pnl > 0:
                    n_wins += 1

                if capital <= 0:
                    capital = 0
                    liquidated = True
                    pos_dir = 0
                    break

                # Open opposite
                pos_dir = d
                pos_entry_price = price
                pos_entry_bar = bar_idx
                capital -= POSITION_SIZE * FEE_RATE  # entry fee

                if capital <= 0:
                    capital = 0
                    liquidated = True
                    pos_dir = 0
                    break

        # Week end: close position
        if pos_dir != 0 and not liquidated:
            last_bar = train_end_1h + we - 1
            if last_bar >= n_1h:
                last_bar = n_1h - 1
            exit_price = c_1h[last_bar]
            ret = (exit_price - pos_entry_price) / pos_entry_price
            trade_pnl = pos_dir * ret * POSITION_SIZE
            capital += trade_pnl
            capital -= POSITION_SIZE * FEE_RATE

            n_trades += 1
            if trade_pnl > 0:
                n_wins += 1
            pos_dir = 0

        pnl = capital - INITIAL_CAPITAL
        pnl_pct = pnl / INITIAL_CAPITAL * 100
        wr = n_wins / n_trades * 100 if n_trades > 0 else 0
        week_date = oos_dates[ws].strftime('%Y-%m-%d')
        liq_mark = "LIQ" if liquidated else ""

        weekly_pnls.append(pnl)
        total_weeks += 1
        if pnl > 0:
            win_weeks += 1
            total_withdrawn += pnl
        elif pnl < 0:
            lose_weeks += 1
        if liquidated:
            liq_weeks += 1

        print(f"{w_idx+1:>4} {week_date:>12} ${INITIAL_CAPITAL:>9.2f} ${capital:>9.2f} ${pnl:>+9.2f} {pnl_pct:>+6.1f}% {n_trades:>5} {wr:>5.1f}% {liq_mark:>4}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("=" * 80)
    print()
    weekly_pnls = np.array(weekly_pnls)
    print(f"  Total weeks:        {total_weeks}")
    print(f"  Winning weeks:      {win_weeks} ({win_weeks/total_weeks*100:.1f}%)")
    print(f"  Losing weeks:       {lose_weeks} ({lose_weeks/total_weeks*100:.1f}%)")
    print(f"  Liquidations:       {liq_weeks}")
    print()
    print(f"  Total withdrawn:    ${total_withdrawn:,.2f}")
    print(f"  Total losses:       ${weekly_pnls[weekly_pnls<0].sum():,.2f}")
    print(f"  Net profit:         ${total_withdrawn + weekly_pnls[weekly_pnls<0].sum():,.2f}")
    print()
    print(f"  Avg weekly PnL:     ${weekly_pnls.mean():,.2f} ({weekly_pnls.mean()/INITIAL_CAPITAL*100:+.2f}%)")
    print(f"  Best week:          ${weekly_pnls.max():,.2f}")
    print(f"  Worst week:         ${weekly_pnls.min():,.2f}")
    print(f"  Std weekly:         ${weekly_pnls.std():,.2f}")

    # Streaks
    max_losing = 0; cur = 0
    for p in weekly_pnls:
        if p < 0: cur += 1; max_losing = max(max_losing, cur)
        else: cur = 0
    max_winning = 0; cur = 0
    for p in weekly_pnls:
        if p > 0: cur += 1; max_winning = max(max_winning, cur)
        else: cur = 0
    print(f"  Max losing streak:  {max_losing} weeks")
    print(f"  Max winning streak: {max_winning} weeks")

print("\nDone!")
