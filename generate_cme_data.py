"""
CME ETH Futures (ETH1!) Synthetic Data Generator

CME ETH futures characteristics:
- Trading hours: Sun 23:00 UTC - Fri 21:00 UTC (summer)
- Daily maintenance break: 21:00-22:00 UTC
- Institutional flow patterns (different from Binance retail)
- Settlement dynamics, basis/premium over spot
- Higher volume during US hours (13:30-21:00 UTC)
"""
import numpy as np
import pandas as pd
import os

np.random.seed(123)  # Different seed from Binance data

def is_cme_trading_hour(ts_ms):
    """Check if a timestamp falls within CME trading hours."""
    dt = pd.to_datetime(ts_ms, unit='ms', utc=True)
    hour = dt.hour
    weekday = dt.weekday()  # 0=Mon, 6=Sun

    # Weekend: only Sunday after 23:00 UTC
    if weekday == 5:  # Saturday - closed
        return False
    if weekday == 6 and hour < 23:  # Sunday before 23:00 - closed
        return False

    # Friday: close at 21:00 UTC
    if weekday == 4 and hour >= 21:
        return False

    # Daily break: 21:00-22:00 UTC
    if hour == 21:
        return False

    return True


def generate_cme_data(binance_path="data/ETHUSDT_5m_5y.parquet"):
    """Generate CME ETH futures data aligned with Binance timestamps."""

    # Load Binance data for reference (price correlation)
    df_bin = pd.read_parquet(binance_path)
    n = len(df_bin)
    timestamps = df_bin["open_time"].values

    # CME prices: correlated with Binance but with basis/premium
    bin_close = df_bin["close"].values

    # CME basis: institutional premium, mean-reverts around +0.1% to +0.5%
    basis = np.zeros(n)
    basis[0] = 0.002  # 0.2% premium
    for i in range(1, n):
        # Basis mean-reverts to ~0.2%, with noise
        basis[i] = basis[i-1] + 0.0001 * (0.002 - basis[i-1]) + 0.0003 * np.random.randn()
        basis[i] = np.clip(basis[i], -0.005, 0.01)

    cme_close = bin_close * (1 + basis)

    # Generate OHLC
    cme_open = np.zeros(n)
    cme_high = np.zeros(n)
    cme_low = np.zeros(n)
    cme_open[0] = cme_close[0]
    for i in range(1, n):
        cme_open[i] = cme_close[i-1] * (1 + basis[i] - basis[i-1])
        body = abs(cme_close[i] - cme_open[i])
        wick = body * np.random.exponential(0.4) + cme_close[i] * 0.0001
        cme_high[i] = max(cme_open[i], cme_close[i]) + wick
        cme_low[i] = min(cme_open[i], cme_close[i]) - wick

    # Volume: institutional pattern — heavy during US hours, light during Asian
    cme_buy_vol = np.zeros(n)
    cme_sell_vol = np.zeros(n)

    for i in range(n):
        dt = pd.to_datetime(timestamps[i], unit='ms', utc=True)
        hour = dt.hour

        if not is_cme_trading_hour(timestamps[i]):
            # CME closed: zero volume
            cme_buy_vol[i] = 0
            cme_sell_vol[i] = 0
            continue

        # Volume by hour (institutional pattern):
        # Peak: 13:30-20:00 UTC (US session)
        # Medium: 8:00-13:30 UTC (EU session)
        # Low: 22:00-8:00 UTC (Asian session)
        if 14 <= hour <= 20:
            vol_mult = 3.0 + 1.0 * np.random.rand()  # US session: heavy
        elif 8 <= hour <= 13:
            vol_mult = 1.5 + 0.5 * np.random.rand()  # EU session: medium
        else:
            vol_mult = 0.3 + 0.3 * np.random.rand()  # Asian: light

        base_vol = np.random.lognormal(7.0, 0.6)
        total_vol = base_vol * vol_mult

        # Institutional flow: more directional (less noisy buy/sell split)
        ret = (cme_close[i] - cme_open[i]) / max(cme_open[i], 1e-10)
        # Institutional orders are more aligned with price direction
        buy_ratio = 0.5 + 0.15 * np.sign(ret) * min(abs(ret) * 100, 1.0) + 0.01 * np.random.randn()
        buy_ratio = np.clip(buy_ratio, 0.3, 0.7)

        cme_buy_vol[i] = total_vol * buy_ratio
        cme_sell_vol[i] = total_vol * (1 - buy_ratio)

    # OI: CME OI dynamics (different from perps — expiry-driven)
    cme_oi = np.zeros(n)
    cme_oi[0] = 2e9
    for i in range(1, n):
        dt = pd.to_datetime(timestamps[i], unit='ms', utc=True)
        # OI grows before expiry, drops after
        day_of_month = dt.day
        if day_of_month > 20:  # approaching expiry
            oi_drift = -0.0001
        else:
            oi_drift = 0.00005

        vol_total = cme_buy_vol[i] + cme_sell_vol[i]
        oi_change = cme_oi[i-1] * (oi_drift + 0.00002 * np.random.randn())
        if vol_total > 0:
            oi_change += vol_total * 0.001 * np.random.randn()
        cme_oi[i] = max(cme_oi[i-1] + oi_change, 1e8)

    df_cme = pd.DataFrame({
        "open_time": timestamps,
        "open": cme_open,
        "high": cme_high,
        "low": cme_low,
        "close": cme_close,
        "buy_vol": cme_buy_vol,
        "sell_vol": cme_sell_vol,
        "open_interest": cme_oi,
        "basis": basis,
        "cme_trading": [is_cme_trading_hour(t) for t in timestamps],
    })

    output = "data/CME_ETH1_5m_5y.parquet"
    df_cme.to_parquet(output, index=False)
    size_mb = os.path.getsize(output) / (1024 * 1024)

    # Stats
    trading_pct = df_cme["cme_trading"].sum() / len(df_cme) * 100
    print(f"CME ETH1! data generated: {len(df_cme):,} bars")
    print(f"Trading hours: {trading_pct:.1f}% of bars")
    print(f"Saved to {output} ({size_mb:.1f} MB)")

    return df_cme


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_cme_data()
