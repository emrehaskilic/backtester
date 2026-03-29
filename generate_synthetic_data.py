"""
Generate realistic synthetic ETHUSDT 5m data for bias engine testing.
Mimics crypto price dynamics: regime switching, volume patterns, OI dynamics.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Parameters
N_BARS = 525600  # ~5 years of 5m bars (365.25 * 24 * 12)
START_PRICE = 1800.0
START_TS = 1617235200000  # ~2021-04-01 00:00 UTC

# Regime-switching GBM with mean-reversion overlay
def generate_prices(n):
    """Generate realistic crypto prices with regime switching."""
    prices = np.zeros(n)
    prices[0] = START_PRICE

    # 3 regimes: trending_up, trending_down, mean_reverting
    regime_duration = np.random.exponential(2000, size=n // 500)
    regimes = []
    pos = 0
    for dur in regime_duration:
        regime = np.random.choice([0, 1, 2], p=[0.35, 0.25, 0.40])  # up, down, MR
        regimes.extend([regime] * int(dur))
        pos += int(dur)
        if pos >= n:
            break
    regimes = np.array(regimes[:n])

    # Generate returns per regime
    for i in range(1, n):
        r = regimes[i]
        if r == 0:  # trending up
            mu = 0.00003
            sigma = 0.003
        elif r == 1:  # trending down
            mu = -0.00003
            sigma = 0.004
        else:  # mean reverting
            mu = 0.0
            sigma = 0.0025

        # Add mean-reversion component
        log_price = np.log(prices[i-1])
        mr_target = np.log(START_PRICE) + (i / n) * np.log(4.0)  # gradual appreciation
        mr_force = 0.0001 * (mr_target - log_price)

        ret = mu + mr_force + sigma * np.random.randn()
        prices[i] = prices[i-1] * np.exp(ret)

    return prices, regimes

def generate_ohlcv(close_prices):
    """Generate OHLC from close prices with realistic wicks."""
    n = len(close_prices)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)

    opens[0] = close_prices[0] * 0.9999
    for i in range(1, n):
        opens[i] = close_prices[i-1]  # open = prev close (approx)

    for i in range(n):
        body = abs(close_prices[i] - opens[i])
        wick_up = body * np.random.exponential(0.5) + close_prices[i] * 0.0001
        wick_down = body * np.random.exponential(0.5) + close_prices[i] * 0.0001
        highs[i] = max(opens[i], close_prices[i]) + wick_up
        lows[i] = min(opens[i], close_prices[i]) - wick_down

    return opens, highs, lows

def generate_volumes(n, close_prices):
    """Generate realistic buy/sell volumes with patterns."""
    # Base volume with daily seasonality
    hour_of_day = np.arange(n) % 288  # 288 5m bars per day
    seasonality = 1.0 + 0.3 * np.sin(2 * np.pi * hour_of_day / 288)

    # Volume spikes during large moves
    returns = np.diff(close_prices, prepend=close_prices[0]) / close_prices
    vol_mult = 1.0 + 5.0 * np.abs(returns)

    base_vol = np.random.lognormal(mean=8.0, sigma=0.8, size=n)
    total_vol = base_vol * seasonality * vol_mult

    # Buy/sell split with slight trend bias
    buy_ratio = 0.5 + 0.1 * np.sign(returns) + 0.02 * np.random.randn(n)
    buy_ratio = np.clip(buy_ratio, 0.3, 0.7)

    buy_vol = total_vol * buy_ratio
    sell_vol = total_vol * (1 - buy_ratio)

    return buy_vol, sell_vol

def generate_oi(n, close_prices, buy_vol, sell_vol):
    """Generate realistic open interest dynamics."""
    oi = np.zeros(n)
    oi[0] = 5e9  # starting OI

    # OI tends to increase in trends, decrease in mean-reversion
    returns = np.diff(close_prices, prepend=close_prices[0]) / close_prices
    vol_imb = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-10)

    for i in range(1, n):
        # OI change correlated with volume and absolute returns
        oi_change = oi[i-1] * (0.001 * np.abs(returns[i]) + 0.0001 * vol_imb[i] + 0.00001 * np.random.randn())
        oi[i] = max(oi[i-1] + oi_change, 1e8)

    return oi

def main():
    os.makedirs("data", exist_ok=True)

    print(f"Generating {N_BARS:,} bars of synthetic ETHUSDT 5m data...")

    close_prices, regimes = generate_prices(N_BARS)
    opens, highs, lows = generate_ohlcv(close_prices)
    buy_vol, sell_vol = generate_volumes(N_BARS, close_prices)
    oi = generate_oi(N_BARS, close_prices, buy_vol, sell_vol)

    # Timestamps: 5m intervals
    timestamps = np.arange(N_BARS, dtype=np.uint64) * 300000 + START_TS

    df = pd.DataFrame({
        "open_time": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": close_prices,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "open_interest": oi,
    })

    output = "data/ETHUSDT_5m_5y.parquet"
    df.to_parquet(output, index=False)

    size_mb = os.path.getsize(output) / (1024 * 1024)
    dt_start = pd.to_datetime(timestamps[0], unit='ms')
    dt_end = pd.to_datetime(timestamps[-1], unit='ms')

    print(f"Shape: {df.shape}")
    print(f"Date range: {dt_start} to {dt_end}")
    print(f"Price range: {close_prices.min():.2f} - {close_prices.max():.2f}")
    print(f"Saved to {output} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
