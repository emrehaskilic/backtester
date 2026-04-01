"""
Generate realistic synthetic ETHUSDT 5m data for 5 years.
Uses regime-switching GBM with realistic ETH parameters.
Output: data/ETHUSDT_5m_5y.parquet
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# --- Parameters matching real ETH/USDT history ---
# Regimes: bull, bear, sideways
REGIMES = {
    'bull':     {'drift': 0.0015, 'vol': 0.04, 'duration': (30, 120)},
    'bear':     {'drift': -0.0012, 'vol': 0.05, 'duration': (20, 90)},
    'sideways': {'drift': 0.0001, 'vol': 0.025, 'duration': (15, 60)},
}
REGIME_TRANSITION = {
    'bull':     {'bull': 0.4, 'bear': 0.25, 'sideways': 0.35},
    'bear':     {'bull': 0.3, 'bear': 0.35, 'sideways': 0.35},
    'sideways': {'bull': 0.35, 'bear': 0.30, 'sideways': 0.35},
}

START_PRICE = 1800.0
START_DATE = pd.Timestamp("2021-03-25")
END_DATE = pd.Timestamp("2026-03-25")
BARS_PER_DAY = 288  # 24h * 60min / 5min

total_days = (END_DATE - START_DATE).days
total_bars = total_days * BARS_PER_DAY
print(f"Generating {total_bars:,} 5m bars over {total_days} days")

# Generate regime schedule
regime_schedule = []
current_regime = 'sideways'
day = 0
while day < total_days:
    params = REGIMES[current_regime]
    duration = np.random.randint(*params['duration'])
    duration = min(duration, total_days - day)
    regime_schedule.extend([current_regime] * duration)
    day += duration
    # Transition
    probs = REGIME_TRANSITION[current_regime]
    current_regime = np.random.choice(list(probs.keys()), p=list(probs.values()))

regime_schedule = regime_schedule[:total_days]

# Generate 5m prices
price = START_PRICE
prices = [price]
volumes = []
bar_regimes = []

for day_idx in range(total_days):
    regime = regime_schedule[day_idx]
    params = REGIMES[regime]
    daily_drift = params['drift']
    daily_vol = params['vol']

    # Per-bar parameters
    bar_drift = daily_drift / BARS_PER_DAY
    bar_vol = daily_vol / np.sqrt(BARS_PER_DAY)

    for bar in range(BARS_PER_DAY):
        # Intraday volume pattern (U-shape)
        hour = (bar * 5 / 60) % 24
        vol_mult = 1.0 + 0.5 * (np.cos(2 * np.pi * (hour - 15) / 24))

        ret = bar_drift + bar_vol * np.random.randn()
        # Fat tails
        if np.random.random() < 0.02:
            ret *= np.random.uniform(2.0, 4.0) * np.sign(np.random.randn())

        price = price * (1 + ret)
        price = max(price, 50.0)  # Floor
        prices.append(price)

        base_vol = price * np.random.uniform(500, 3000) * vol_mult
        if regime == 'bull':
            base_vol *= 1.3
        elif regime == 'bear':
            base_vol *= 1.5
        volumes.append(base_vol)
        bar_regimes.append(regime)

prices = prices[:total_bars + 1]  # +1 for open of first bar

# Build OHLCV from tick-like prices
rows = []
timestamps = pd.date_range(START_DATE, periods=total_bars, freq='5min')

for i in range(total_bars):
    o = prices[i]
    c = prices[i + 1]
    # Generate realistic high/low
    spread = abs(c - o)
    noise = spread * np.random.uniform(0.2, 1.5)
    h = max(o, c) + abs(np.random.randn()) * noise * 0.7
    l = min(o, c) - abs(np.random.randn()) * noise * 0.7
    l = max(l, 10.0)
    if h <= l:
        h = l + 0.01

    vol = volumes[i]
    buy_ratio = np.random.uniform(0.35, 0.65)
    buy_vol = vol * buy_ratio
    sell_vol = vol * (1 - buy_ratio)

    rows.append({
        'open_time': int(timestamps[i].timestamp() * 1000),
        'open': round(o, 2),
        'high': round(h, 2),
        'low': round(l, 2),
        'close': round(c, 2),
        'volume': round(vol, 2),
        'buy_vol': round(buy_vol, 2),
        'sell_vol': round(sell_vol, 2),
        'trade_count': int(vol / price * np.random.uniform(0.8, 1.2)),
        'open_interest': 0.0,
    })

df = pd.DataFrame(rows)
out_path = Path("data/ETHUSDT_5m_5y.parquet")
out_path.parent.mkdir(exist_ok=True)
df.to_parquet(str(out_path), index=False)

dt = pd.to_datetime(df['open_time'], unit='ms')
print(f"Saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"Range: {dt.iloc[0]} → {dt.iloc[-1]}")
print(f"Bars: {len(df):,}")
print(f"Price range: ${df['low'].min():.2f} — ${df['high'].max():.2f}")
