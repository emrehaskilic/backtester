"""Data fetching — Binance klines (OHLCV) for PMax+KC strategy."""

import json
import logging
import time
import urllib.parse
import urllib.request

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_klines(symbol: str, interval: str = "3m", days: int = 180) -> pd.DataFrame:
    """Fetch historical klines from Binance Futures."""
    base = "https://fapi.binance.com"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    all_candles = []
    current = start_ms

    logger.info("Fetching %s %s data (%d days)...", symbol, interval, days)

    while current < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current,
            "limit": 1500,
        }
        url = f"{base}/fapi/v1/klines?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "BacTester/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning("Fetch error: %s", e)
            break

        if not raw:
            break

        for k in raw:
            all_candles.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
            })

        current = int(raw[-1][6]) + 1
        if len(raw) < 1500:
            break

    df = pd.DataFrame(all_candles)
    logger.info("Fetched %d candles for %s", len(df), symbol)
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into in-sample and out-of-sample."""
    split_idx = int(len(df) * train_ratio)
    df_is = df.iloc[:split_idx].reset_index(drop=True)
    df_oos = df.iloc[split_idx:].reset_index(drop=True)
    return df_is, df_oos
