"""
Session-based prediction: predict NEXT session's direction.
DST-aware session boundaries.
"""
import numpy as np
import pandas as pd
import rust_engine
from datetime import datetime, timezone, timedelta

# === Load 1H data ===
df = pd.read_parquet("data/ETHUSDT_5m_5y.parquet")
PERIOD = 12; n_1h = len(df) // PERIOD
ts = df["open_time"].values
o,h,l,c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
bv,sv,oi = df["buy_vol"].values, df["sell_vol"].values, df["open_interest"].values

c_1h=np.zeros(n_1h);h_1h=np.zeros(n_1h);l_1h=np.zeros(n_1h);o_1h=np.zeros(n_1h)
bv_1h=np.zeros(n_1h);sv_1h=np.zeros(n_1h);oi_1h=np.zeros(n_1h)
ts_1h=np.zeros(n_1h,dtype=np.uint64);vol_1h=np.zeros(n_1h)
for i in range(n_1h):
    s,e=i*PERIOD,i*PERIOD+PERIOD
    ts_1h[i]=ts[s];o_1h[i]=o[s];h_1h[i]=h[s:e].max()
    l_1h[i]=l[s:e].min();c_1h[i]=c[e-1]
    bv_1h[i]=bv[s:e].sum();sv_1h[i]=sv[s:e].sum()
    oi_1h[i]=oi[e-1];vol_1h[i]=bv_1h[i]+sv_1h[i]

dates_1h = pd.to_datetime(ts_1h, unit='ms', utc=True)
print(f"1H data: {n_1h} bars, {dates_1h[0]} to {dates_1h[-1]}")

# === DST-aware session definitions ===
# US DST: 2nd Sunday March → 1st Sunday November
# EU DST: Last Sunday March → Last Sunday October
# We use simplified rules based on month

def get_sessions_utc(dt):
    """Return session boundaries in UTC hours for a given date, DST-aware."""
    month = dt.month
    # US summer: Apr-Oct (simplified), EU summer: Apr-Oct
    us_summer = 3 <= month <= 10
    eu_summer = 3 <= month <= 10

    tokyo_start = 0   # always 00:00 UTC (no DST)
    tokyo_end = 9      # always 09:00 UTC

    london_start = 7 if eu_summer else 8
    london_end = 16 if eu_summer else 17

    ny_start = 12 if us_summer else 13
    ny_end = 21 if us_summer else 22

    return {
        'tokyo': (tokyo_start, tokyo_end),
        'london': (london_start, london_end),
        'newyork': (ny_start, ny_end),
    }

# === Build session bars ===
# Each session: open, high, low, close, volume, buy_vol, sell_vol, OI, return
sessions = []  # list of dicts

i = 0
while i < n_1h:
    dt = dates_1h[i]
    sess_def = get_sessions_utc(dt)
    hour = dt.hour

    # Determine which session this bar belongs to
    for sess_name, (s_start, s_end) in sess_def.items():
        if s_start <= hour < s_end:
            # Collect all bars in this session
            sess_bars = []
            j = i
            while j < n_1h:
                h_j = dates_1h[j].hour
                # Same session if same date portion and hour in range
                if s_start <= h_j < s_end and (dates_1h[j].date() == dt.date() or
                    (s_start > s_end and dates_1h[j].date() == dt.date())):
                    sess_bars.append(j)
                    j += 1
                else:
                    break

            if len(sess_bars) >= 2:
                idxs = sess_bars
                sess_open = o_1h[idxs[0]]
                sess_high = h_1h[idxs].max()
                sess_low = l_1h[idxs].min()
                sess_close = c_1h[idxs[-1]]
                sess_vol = vol_1h[idxs].sum()
                sess_bv = bv_1h[idxs].sum()
                sess_sv = sv_1h[idxs].sum()
                sess_oi = oi_1h[idxs[-1]]
                sess_ret = (sess_close - sess_open) / sess_open * 100
                sess_cvd = sess_bv - sess_sv
                sess_imb = sess_cvd / max(sess_vol, 1e-10)
                sess_range = (sess_high - sess_low) / sess_open * 100

                sessions.append({
                    'name': sess_name,
                    'date': dt.date(),
                    'start_idx': idxs[0],
                    'end_idx': idxs[-1],
                    'open': sess_open,
                    'high': sess_high,
                    'low': sess_low,
                    'close': sess_close,
                    'volume': sess_vol,
                    'buy_vol': sess_bv,
                    'sell_vol': sess_sv,
                    'oi': sess_oi,
                    'return': sess_ret,
                    'cvd': sess_cvd,
                    'imbalance': sess_imb,
                    'range': sess_range,
                    'n_bars': len(idxs),
                    'ts': ts_1h[idxs[0]],
                })
            i = j
            break
    else:
        i += 1

print(f"Total sessions: {len(sessions)}")

# Convert to DataFrame
sdf = pd.DataFrame(sessions)
print(f"Sessions per type:")
for name in ['tokyo', 'london', 'newyork']:
    cnt = (sdf['name'] == name).sum()
    print(f"  {name}: {cnt}")

# === Outcome: next session's direction ===
sdf['next_return'] = sdf['return'].shift(-1)
sdf['next_bull'] = (sdf['next_return'] > 0).astype(int)
sdf['next_name'] = sdf['name'].shift(-1)

# Drop last row (no next session)
sdf = sdf.iloc[:-1].copy()

# === Features for prediction ===
# Current session features
sdf['f_return'] = sdf['return']
sdf['f_range'] = sdf['range']
sdf['f_imbalance'] = sdf['imbalance']
sdf['f_cvd_norm'] = sdf['cvd'] / sdf['volume'].clip(lower=1)
sdf['f_vol_ratio'] = sdf['volume'] / sdf.groupby('name')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())

# Previous session features
sdf['f_prev_return'] = sdf['return'].shift(1)
sdf['f_prev_range'] = sdf['range'].shift(1)
sdf['f_prev_imbalance'] = sdf['imbalance'].shift(1)

# 2 sessions ago
sdf['f_prev2_return'] = sdf['return'].shift(2)

# Session sequence encoding
sess_map = {'tokyo': 0, 'london': 1, 'newyork': 2}
sdf['f_sess_id'] = sdf['name'].map(sess_map)
sdf['f_next_sess_id'] = sdf['next_name'].map(sess_map)

# Day of week
sdf['f_dow'] = pd.to_datetime(sdf['date']).dt.dayofweek

# OI change from session start
sdf['f_oi_change'] = sdf['oi'].pct_change()

# Rolling stats
for w in [5, 10]:
    sdf[f'f_ret_mean_{w}'] = sdf['return'].rolling(w, min_periods=3).mean()
    sdf[f'f_ret_std_{w}'] = sdf['return'].rolling(w, min_periods=3).std()
    sdf[f'f_imb_mean_{w}'] = sdf['imbalance'].rolling(w, min_periods=3).mean()

# Fill NaN with 0 for rolling features, drop only critical NaN
sdf = sdf.fillna(0)
sdf = sdf[sdf['next_return'].notna() & (sdf['volume'] > 0)].reset_index(drop=True)
print(f"Valid sessions for prediction: {len(sdf)}")

# === Simple accuracy tests ===
print(f"\n{'='*60}")
print(f"SESSION PREDICTION ANALYSIS")
print(f"{'='*60}")

# Baseline: predict same as current session
same_dir = ((sdf['f_return'] > 0) == sdf['next_bull']).mean()
print(f"\nBaseline (same direction): {same_dir:.4f}")

# Reverse: predict opposite
rev_dir = ((sdf['f_return'] < 0) == sdf['next_bull']).mean()
print(f"Reverse (mean reversion): {rev_dir:.4f}")

# By session transition
print(f"\n=== ACCURACY BY SESSION TRANSITION ===")
for curr in ['tokyo', 'london', 'newyork']:
    for nxt in ['tokyo', 'london', 'newyork']:
        mask = (sdf['name'] == curr) & (sdf['next_name'] == nxt)
        if mask.sum() > 50:
            sub = sdf[mask]
            # Same direction
            same = ((sub['f_return'] > 0) == sub['next_bull']).mean()
            # Reverse
            rev = ((sub['f_return'] < 0) == sub['next_bull']).mean()
            print(f"  {curr:>8} -> {nxt:<8}: n={mask.sum():>5}, same={same:.4f}, reverse={rev:.4f}")

# By current session strength
print(f"\n=== NEXT SESSION BY CURRENT RETURN STRENGTH ===")
for lo, hi, label in [(-99, -2, "strong bear"), (-2, -0.5, "mild bear"),
                        (-0.5, 0.5, "flat"), (0.5, 2, "mild bull"), (2, 99, "strong bull")]:
    mask = (sdf['f_return'] >= lo) & (sdf['f_return'] < hi)
    if mask.sum() > 50:
        sub = sdf[mask]
        next_bull_pct = sub['next_bull'].mean()
        print(f"  Current {label:>12}: n={mask.sum():>5}, next_bull={next_bull_pct:.4f}")

# By imbalance
print(f"\n=== NEXT SESSION BY CURRENT IMBALANCE ===")
for lo, hi, label in [(-1, -0.1, "strong sell"), (-0.1, -0.02, "mild sell"),
                        (-0.02, 0.02, "neutral"), (0.02, 0.1, "mild buy"), (0.1, 1, "strong buy")]:
    mask = (sdf['f_imbalance'] >= lo) & (sdf['f_imbalance'] < hi)
    if mask.sum() > 50:
        sub = sdf[mask]
        next_bull_pct = sub['next_bull'].mean()
        print(f"  Imbalance {label:>12}: n={mask.sum():>5}, next_bull={next_bull_pct:.4f}")

# By volume ratio (high vs low volume sessions)
print(f"\n=== NEXT SESSION BY VOLUME RATIO ===")
for lo, hi, label in [(0, 0.7, "low vol"), (0.7, 1.0, "below avg"), (1.0, 1.5, "above avg"), (1.5, 99, "high vol")]:
    mask = (sdf['f_vol_ratio'] >= lo) & (sdf['f_vol_ratio'] < hi)
    if mask.sum() > 50:
        sub = sdf[mask]
        next_bull_pct = sub['next_bull'].mean()
        rev = ((sub['f_return'] < 0) == sub['next_bull']).mean()
        print(f"  Volume {label:>10}: n={mask.sum():>5}, next_bull={next_bull_pct:.4f}, MR_acc={rev:.4f}")

# Combined: strong imbalance + MR
print(f"\n=== COMBINED SIGNALS ===")
# Strong sell imbalance → next session bull (MR)
mask = (sdf['f_imbalance'] < -0.05) & (sdf['f_return'] < -0.5)
if mask.sum() > 30:
    acc = sdf.loc[mask, 'next_bull'].mean()
    print(f"  Strong sell + bear return → next bull: {acc:.4f} (n={mask.sum()})")

mask = (sdf['f_imbalance'] > 0.05) & (sdf['f_return'] > 0.5)
if mask.sum() > 30:
    acc = 1 - sdf.loc[mask, 'next_bull'].mean()
    print(f"  Strong buy + bull return → next bear: {acc:.4f} (n={mask.sum()})")

# Cross-session: Tokyo bearish → London bull?
for curr, nxt in [('tokyo','london'), ('london','newyork'), ('newyork','tokyo')]:
    mask = (sdf['name']==curr) & (sdf['next_name']==nxt) & (sdf['f_return'] < -0.5)
    if mask.sum() > 30:
        acc = sdf.loc[mask, 'next_bull'].mean()
        print(f"  {curr} bear → {nxt} bull: {acc:.4f} (n={mask.sum()})")
    mask = (sdf['name']==curr) & (sdf['next_name']==nxt) & (sdf['f_return'] > 0.5)
    if mask.sum() > 30:
        acc = 1 - sdf.loc[mask, 'next_bull'].mean()
        print(f"  {curr} bull → {nxt} bear: {acc:.4f} (n={mask.sum()})")
