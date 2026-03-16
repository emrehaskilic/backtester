#!/Users/deneme/miniconda/bin/python
"""
Tick Replay Engine — NinjaTrader 8 Tick Replay benzeri.
aggTrades verisinden DFS bileşenlerini hesaplar.
"""

import numpy as np
import pandas as pd
from collections import deque
import math


class RollingStats:
    """Hızlı rolling mean/std hesaplama."""
    def __init__(self, window):
        self.window = window
        self.values = deque(maxlen=window)

    def update(self, val):
        self.values.append(val)

    def mean(self):
        if not self.values:
            return 0
        return sum(self.values) / len(self.values)

    def std(self):
        if len(self.values) < 2:
            return 1
        m = self.mean()
        var = sum((x - m) ** 2 for x in self.values) / (len(self.values) - 1)
        return max(math.sqrt(var), 1e-10)

    def zscore(self, val):
        s = self.std()
        return (val - self.mean()) / s if s > 0 else 0

    def percentile(self, val):
        if not self.values:
            return 0.5
        below = sum(1 for x in self.values if x <= val)
        return below / len(self.values)

    @property
    def count(self):
        return len(self.values)


class TickReplayEngine:
    """
    aggTrades'ten DFS 8 bileşenini hesaplar.
    Her tick'te güncellenen state machine.
    """

    def __init__(self, rolling_window_sec=3600):
        self.rolling_window = rolling_window_sec  # normalizasyon penceresi (saniye)

        # 1-saniyelik bucket'lar
        self.current_bucket_ts = 0
        self.bucket_buy_vol = 0
        self.bucket_sell_vol = 0
        self.bucket_trades = 0
        self.bucket_open = 0
        self.bucket_close = 0
        self.bucket_high = -1e18
        self.bucket_low = 1e18

        # Rolling istatistikler (bucket bazlı)
        self.delta_stats = RollingStats(rolling_window_sec)
        self.cvd_slope_stats = RollingStats(rolling_window_sec)
        self.log_pressure_stats = RollingStats(rolling_window_sec)
        self.obi_stats = RollingStats(rolling_window_sec)
        self.sweep_stats = RollingStats(rolling_window_sec)
        self.burst_stats = RollingStats(rolling_window_sec)
        self.oi_stats = RollingStats(rolling_window_sec)
        self.vol_stats = RollingStats(rolling_window_sec)

        # CVD tracking
        self.cvd = 0
        self.cvd_history = deque(maxlen=30)  # son 30 saniye

        # Burst tracking
        self.last_sides = deque(maxlen=10)

        # VWAP
        self.vwap_cum_pv = 0
        self.vwap_cum_v = 0
        self.vwap = 0

        # State
        self.last_price = 0
        self.tick_count = 0
        self.warmup_done = False

        # DFS bileşenleri (son hesaplanan)
        self.dfs_components = {
            "zDelta": 0, "zCvd": 0, "zLogP": 0, "zObiW": 0,
            "zObiD": 0, "sweepSigned": 0, "burstSigned": 0, "oiImpulse": 0
        }
        self.dfs_percentile = 0.5
        self.prints_per_sec = 0

        # Bucket history for candle building
        self.buckets = []

    def _flush_bucket(self):
        """Mevcut 1-saniyelik bucket'ı hesapla ve kaydet."""
        if self.current_bucket_ts == 0:
            return

        delta = self.bucket_buy_vol - self.bucket_sell_vol
        total_vol = self.bucket_buy_vol + self.bucket_sell_vol

        # CVD güncelle
        self.cvd += delta
        self.cvd_history.append(self.cvd)

        # CVD slope (son 10 saniye)
        cvd_slope = 0
        if len(self.cvd_history) >= 2:
            cvd_slope = self.cvd_history[-1] - self.cvd_history[0]

        # Log pressure
        log_pressure = 0
        if self.bucket_buy_vol > 0 and self.bucket_sell_vol > 0:
            log_pressure = math.log(self.bucket_buy_vol / self.bucket_sell_vol)

        # Sweep (price movement magnitude * direction)
        sweep = 0
        if self.bucket_open > 0:
            sweep = (self.bucket_close - self.bucket_open) / self.bucket_open * 10000  # bps

        # Burst (consecutive same-side trades)
        burst_count = 0
        if self.last_sides:
            last = self.last_sides[-1]
            for s in reversed(self.last_sides):
                if s == last:
                    burst_count += 1
                else:
                    break
            burst_count *= (1 if last == "BUY" else -1)

        # Volatility (high-low range)
        vol = 0
        if self.bucket_low < 1e18 and self.bucket_high > -1e18:
            vol = (self.bucket_high - self.bucket_low) / self.bucket_open * 10000 if self.bucket_open > 0 else 0

        # Rolling stats güncelle
        self.delta_stats.update(delta)
        self.cvd_slope_stats.update(cvd_slope)
        self.log_pressure_stats.update(log_pressure)
        self.sweep_stats.update(sweep)
        self.burst_stats.update(burst_count)
        self.vol_stats.update(vol)

        # OBI yaklaşık (aggTrades'ten orderbook yok, delta bazlı yaklaşım)
        obi_approx = delta / (total_vol + 1e-10)
        self.obi_stats.update(obi_approx)

        # Z-score hesapla
        self.dfs_components["zDelta"] = self.delta_stats.zscore(delta)
        self.dfs_components["zCvd"] = self.cvd_slope_stats.zscore(cvd_slope)
        self.dfs_components["zLogP"] = self.log_pressure_stats.zscore(log_pressure)
        self.dfs_components["zObiW"] = self.obi_stats.zscore(obi_approx)
        self.dfs_components["zObiD"] = self.obi_stats.zscore(obi_approx) * 0.8  # deep approx
        self.dfs_components["sweepSigned"] = self.sweep_stats.zscore(sweep)
        self.dfs_components["burstSigned"] = self.burst_stats.zscore(burst_count)
        self.dfs_components["oiImpulse"] = 0  # OI data yok aggTrades'te

        # Prints per second
        self.prints_per_sec = self.bucket_trades

        # Warmup kontrolü
        if self.delta_stats.count >= 60:
            self.warmup_done = True

        # Bucket kaydet
        self.buckets.append({
            "ts": self.current_bucket_ts,
            "open": self.bucket_open,
            "high": self.bucket_high,
            "low": self.bucket_low,
            "close": self.bucket_close,
            "volume": total_vol,
            "buy_vol": self.bucket_buy_vol,
            "sell_vol": self.bucket_sell_vol,
            "delta": delta,
            "cvd": self.cvd,
            "prints": self.bucket_trades,
        })

    def _reset_bucket(self, ts, price):
        """Yeni bucket başlat."""
        self.current_bucket_ts = ts
        self.bucket_buy_vol = 0
        self.bucket_sell_vol = 0
        self.bucket_trades = 0
        self.bucket_open = price
        self.bucket_close = price
        self.bucket_high = price
        self.bucket_low = price

    def process_tick(self, timestamp_ms, price, quantity, side):
        """
        Tek bir aggTrade'i işle.
        Returns: (warmup_done, dfs_percentile, dfs_components)
        """
        ts_sec = timestamp_ms // 1000
        self.tick_count += 1

        # VWAP güncelle
        self.vwap_cum_pv += price * quantity
        self.vwap_cum_v += quantity
        self.vwap = self.vwap_cum_pv / self.vwap_cum_v if self.vwap_cum_v > 0 else price

        # Burst tracking
        self.last_sides.append(side)

        # Bucket yönetimi (1 saniyelik)
        if ts_sec != self.current_bucket_ts:
            self._flush_bucket()
            self._reset_bucket(ts_sec, price)

        # Bucket güncelle
        if side == "BUY":
            self.bucket_buy_vol += quantity
        else:
            self.bucket_sell_vol += quantity

        self.bucket_trades += 1
        self.bucket_close = price
        self.bucket_high = max(self.bucket_high, price)
        self.bucket_low = min(self.bucket_low, price)
        self.last_price = price

    def get_dfs(self, weights=None):
        """
        DFS percentile hesapla.
        weights: dict of component weights (sum to 1.0)
        """
        if weights is None:
            weights = {
                "zDelta": 0.22, "zCvd": 0.18, "zLogP": 0.12, "zObiW": 0.14,
                "zObiD": 0.12, "sweepSigned": 0.08, "burstSigned": 0.08, "oiImpulse": 0.06
            }

        # Ağırlıklı toplam
        raw_score = sum(weights.get(k, 0) * v for k, v in self.dfs_components.items())

        # Sigmoid ile [0, 1] aralığına dönüştür
        self.dfs_percentile = 1 / (1 + math.exp(-raw_score))
        return self.dfs_percentile

    def get_vwap(self):
        return self.vwap

    def get_vwap_deviation_pct(self):
        if self.vwap <= 0 or self.last_price <= 0:
            return 0
        return (self.last_price - self.vwap) / self.vwap * 100

    def reset_vwap(self):
        """Günlük VWAP sıfırlama."""
        self.vwap_cum_pv = 0
        self.vwap_cum_v = 0
