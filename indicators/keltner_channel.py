"""
Keltner Channel Indicator — 3 Parametre.

Parametreler:
  kc_length     : EMA uzunluğu (3)
  kc_multiplier : ATR çarpanı (0.5)
  kc_atr_period : ATR periyodu (2)

Üst Band = EMA + ATR * multiplier
Alt Band = EMA - ATR * multiplier
"""

import numpy as np
from collections import deque


class KeltnerChannel:
    """
    Keltner Channel — EMA + ATR bazlı band hesaplama.
    Her mum kapanışında update() çağrılır.
    """

    def __init__(self, params=None):
        p = params or {}

        self.kc_length = int(p.get("kc_length", 3))
        self.kc_multiplier = float(p.get("kc_multiplier", 0.5))
        self.kc_atr_period = int(p.get("kc_atr_period", 2))

        # EMA state
        self.ema = 0.0
        self.ema_k = 2.0 / (self.kc_length + 1)  # EMA smoothing factor
        self.ema_initialized = False

        # ATR için fiyat geçmişi
        self.highs = deque(maxlen=self.kc_atr_period + 1)
        self.lows = deque(maxlen=self.kc_atr_period + 1)
        self.closes = deque(maxlen=self.kc_atr_period + 1)

        # Band değerleri
        self.upper = 0.0
        self.lower = 0.0
        self.middle = 0.0
        self.atr = 0.0

        self.ready = False
        self.candle_count = 0

    def update(self, high, low, close):
        """
        Yeni mum kapanışında çağrılır.
        Returns: (upper, middle, lower, atr)
        """
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.candle_count += 1

        # EMA hesapla
        if not self.ema_initialized:
            self.ema = close
            self.ema_initialized = True
        else:
            self.ema = close * self.ema_k + self.ema * (1 - self.ema_k)

        self.middle = self.ema

        # ATR hesapla
        if len(self.closes) >= 2:
            trs = []
            closes_list = list(self.closes)
            highs_list = list(self.highs)
            lows_list = list(self.lows)

            n = min(self.kc_atr_period, len(closes_list) - 1)
            for i in range(len(closes_list) - n, len(closes_list)):
                tr = max(
                    highs_list[i] - lows_list[i],
                    abs(highs_list[i] - closes_list[i - 1]),
                    abs(lows_list[i] - closes_list[i - 1])
                )
                trs.append(tr)

            self.atr = np.mean(trs) if trs else 0.0
        else:
            self.atr = high - low

        # Band hesapla
        self.upper = self.middle + self.atr * self.kc_multiplier
        self.lower = self.middle - self.atr * self.kc_multiplier

        if self.candle_count >= max(self.kc_length, self.kc_atr_period):
            self.ready = True

        return self.upper, self.middle, self.lower, self.atr
