"""
Candle Builder — aggTrades CSV'den 3dk OHLCV mumları oluşturur.
Dual-loop destekli: tick bazlı iterasyon + mum kapanışı eventi.
"""

import pandas as pd
import numpy as np
import os


def load_aggtrades(csv_paths):
    """Birden fazla aggTrades CSV'yi birleştirip yükler."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("transact_time").reset_index(drop=True)

    # Normalize columns
    df.rename(columns={
        "transact_time": "timestamp",
        "price": "price",
        "quantity": "quantity",
        "is_buyer_maker": "is_buyer_maker",
    }, inplace=True)

    df["price"] = df["price"].astype(np.float64)
    df["quantity"] = df["quantity"].astype(np.float64)

    # is_buyer_maker: true/True/TRUE -> taker SELL, maker BUY
    # is_buyer_maker = True  -> satıcı taker (fiyat düşürücü)
    # is_buyer_maker = False -> alıcı taker (fiyat yükseltici)
    if df["is_buyer_maker"].dtype == object:
        df["is_buyer_maker"] = df["is_buyer_maker"].str.lower().map({"true": True, "false": False})

    return df


class CandleBuilder:
    """
    Tick verilerinden belirli periyotta OHLCV mumları oluşturur.
    Her tick'te güncellenir, mum kapanışında callback tetikler.
    """

    def __init__(self, period_sec=180):
        self.period_sec = period_sec  # 3dk = 180sn
        self.period_ms = period_sec * 1000

        # Mevcut mum state
        self.current_candle_start = 0
        self.candle_open = 0.0
        self.candle_high = -np.inf
        self.candle_low = np.inf
        self.candle_close = 0.0
        self.candle_volume = 0.0
        self.candle_buy_volume = 0.0
        self.candle_sell_volume = 0.0
        self.candle_trade_count = 0

        # Tamamlanmış mumlar
        self.candles = []

        self._initialized = False

    def _align_candle_start(self, ts_ms):
        """Timestamp'i periyoda hizala."""
        return (ts_ms // self.period_ms) * self.period_ms

    def process_tick(self, ts_ms, price, quantity, is_buyer_maker):
        """
        Tek bir tick işle.
        Returns: (candle_closed, completed_candle_dict or None)
        """
        aligned = self._align_candle_start(ts_ms)

        if not self._initialized:
            self.current_candle_start = aligned
            self.candle_open = price
            self.candle_high = price
            self.candle_low = price
            self.candle_close = price
            self.candle_volume = 0.0
            self.candle_buy_volume = 0.0
            self.candle_sell_volume = 0.0
            self.candle_trade_count = 0
            self._initialized = True

        completed = None

        # Yeni periyoda geçiş — mum kapandı
        if aligned > self.current_candle_start:
            completed = {
                "timestamp": self.current_candle_start,
                "open": self.candle_open,
                "high": self.candle_high,
                "low": self.candle_low,
                "close": self.candle_close,
                "volume": self.candle_volume,
                "buy_volume": self.candle_buy_volume,
                "sell_volume": self.candle_sell_volume,
                "trade_count": self.candle_trade_count,
            }
            self.candles.append(completed)

            # Yeni mum başlat
            self.current_candle_start = aligned
            self.candle_open = price
            self.candle_high = price
            self.candle_low = price
            self.candle_close = price
            self.candle_volume = 0.0
            self.candle_buy_volume = 0.0
            self.candle_sell_volume = 0.0
            self.candle_trade_count = 0

        # Mumu güncelle
        if price > self.candle_high:
            self.candle_high = price
        if price < self.candle_low:
            self.candle_low = price
        self.candle_close = price
        self.candle_volume += quantity
        self.candle_trade_count += 1

        if is_buyer_maker:
            self.candle_sell_volume += quantity
        else:
            self.candle_buy_volume += quantity

        return (completed is not None, completed)

    def get_candles_df(self):
        """Tamamlanmış mumları DataFrame olarak döndür."""
        if not self.candles:
            return pd.DataFrame()
        return pd.DataFrame(self.candles)
