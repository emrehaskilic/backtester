"""
Adaptive PMax Indicator — 9 Parametre.

PMax (Profit Maximizer): ATR tabanlı trend takip indikatörü.
Adaptive versiyonu volatilite rejimine göre ATR period, MA length ve
multiplier'ı dinamik olarak ayarlar.

Parametreler:
  vol_lookback   : Volatilite rejimi hesaplama penceresi (mum sayısı)
  flip_window    : PMax flip teyidi için gereken mum sayısı
  mult_base      : ATR multiplier taban değeri
  mult_scale     : Volatilite ile multiplier ölçekleme faktörü
  ma_base        : MA length taban değeri
  ma_scale       : Volatilite ile MA ölçekleme faktörü
  atr_base       : ATR period taban değeri
  atr_scale      : Volatilite ile ATR ölçekleme faktörü
  update_interval: Adaptif parametreleri güncelleme sıklığı (mum sayısı)

Sabit (optimize edilmeyen):
  pmax_atr_period     : 10
  pmax_atr_multiplier : 3.0
  pmax_ma_length      : 10
"""

import numpy as np
from collections import deque


class AdaptivePMax:
    """
    Adaptive PMax — volatilite rejimine göre parametreleri ayarlar.
    Her mum kapanışında update() çağrılır.
    """

    def __init__(self, params=None):
        p = params or {}

        # Adaptive parametreler (9 parametre)
        self.vol_lookback = int(p.get("vol_lookback", 260))
        self.flip_window = int(p.get("flip_window", 360))
        self.mult_base = float(p.get("mult_base", 3.25))
        self.mult_scale = float(p.get("mult_scale", 2.0))
        self.ma_base = int(p.get("ma_base", 11))
        self.ma_scale = float(p.get("ma_scale", 4.5))
        self.atr_base = int(p.get("atr_base", 15))
        self.atr_scale = float(p.get("atr_scale", 2.0))
        self.update_interval = int(p.get("update_interval", 55))

        # Sabit PMax parametreleri (başlangıç değerleri, adaptif olarak değişir)
        self.atr_period = int(p.get("pmax_atr_period", 10))
        self.atr_multiplier = float(p.get("pmax_atr_multiplier", 3.0))
        self.ma_length = int(p.get("pmax_ma_length", 10))

        # Aktif (adaptif) parametreler
        self.active_atr_period = self.atr_period
        self.active_ma_length = self.ma_length
        self.active_multiplier = self.atr_multiplier

        # Fiyat geçmişi
        self.highs = deque(maxlen=max(self.vol_lookback, 500))
        self.lows = deque(maxlen=max(self.vol_lookback, 500))
        self.closes = deque(maxlen=max(self.vol_lookback, 500))

        # PMax state
        self.direction = 0  # 1=LONG, -1=SHORT, 0=belirsiz
        self.pmax_stop = 0.0
        self.pmax_line = 0.0

        # Flip tracking
        self.pending_direction = 0
        self.flip_counter = 0
        self.candle_count = 0

        # Son flip bilgisi
        self.flipped = False
        self.flip_price = 0.0

    def _calc_volatility_regime(self):
        """
        Volatilite rejimi: son vol_lookback mum üzerinden normalize edilmiş
        ATR / close ortalaması. 0-1 arasında percentile döndürür.
        """
        if len(self.closes) < min(self.vol_lookback, 20):
            return 0.5  # yeterli veri yok, nötr

        lookback = min(len(self.closes), self.vol_lookback)
        closes = list(self.closes)[-lookback:]
        highs = list(self.highs)[-lookback:]
        lows = list(self.lows)[-lookback:]

        # ATR hesapla (basit: high - low ortalaması)
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            trs.append(tr)

        if not trs:
            return 0.5

        avg_tr = np.mean(trs[-14:]) if len(trs) >= 14 else np.mean(trs)
        avg_close = np.mean(closes[-14:]) if len(closes) >= 14 else np.mean(closes)

        if avg_close == 0:
            return 0.5

        current_vol = avg_tr / avg_close

        # Tüm lookback üzerinden percentile
        all_vols = []
        window = min(14, len(trs))
        for i in range(window, len(trs) + 1):
            chunk_tr = np.mean(trs[i - window:i])
            chunk_cl = np.mean(closes[i - window + 1:i + 1])
            if chunk_cl > 0:
                all_vols.append(chunk_tr / chunk_cl)

        if not all_vols:
            return 0.5

        below = sum(1 for v in all_vols if v <= current_vol)
        return below / len(all_vols)

    def _update_adaptive_params(self, vol_regime):
        """Volatilite rejimine göre PMax parametrelerini güncelle."""
        # vol_regime: 0=düşük vol, 1=yüksek vol
        # Yüksek volatilitede: daha uzun ATR, daha yüksek multiplier, daha uzun MA
        self.active_atr_period = max(2, int(self.atr_base + self.atr_scale * vol_regime))
        self.active_ma_length = max(2, int(self.ma_base + self.ma_scale * vol_regime))
        self.active_multiplier = self.mult_base + self.mult_scale * vol_regime

    def _calc_atr(self, period):
        """Son N mum üzerinden ATR hesapla."""
        if len(self.closes) < 2:
            return 0.0

        n = min(period, len(self.closes) - 1)
        closes = list(self.closes)
        highs = list(self.highs)
        lows = list(self.lows)

        trs = []
        for i in range(len(closes) - n, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            trs.append(tr)

        return np.mean(trs) if trs else 0.0

    def _calc_ma(self, period):
        """Son N mum üzerinden SMA hesapla."""
        if len(self.closes) < period:
            return self.closes[-1] if self.closes else 0.0

        return np.mean(list(self.closes)[-period:])

    def update(self, high, low, close):
        """
        Yeni mum kapanışında çağrılır.
        Returns: (flipped: bool, direction: int, pmax_stop: float)
          flipped: bu mumda yön değişti mi
          direction: 1=LONG, -1=SHORT, 0=belirsiz
          pmax_stop: PMax stop seviyesi
        """
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.candle_count += 1
        self.flipped = False

        # Yeterli veri yoksa bekle
        if len(self.closes) < max(self.active_atr_period, self.active_ma_length) + 1:
            return False, 0, 0.0

        # Adaptif parametre güncelleme
        if self.candle_count % self.update_interval == 0:
            vol_regime = self._calc_volatility_regime()
            self._update_adaptive_params(vol_regime)

        # ATR ve MA hesapla
        atr = self._calc_atr(self.active_atr_period)
        ma = self._calc_ma(self.active_ma_length)

        if atr == 0:
            return False, self.direction, self.pmax_stop

        # PMax hesaplama
        # Long stop: MA - ATR * multiplier
        # Short stop: MA + ATR * multiplier
        long_stop = ma - atr * self.active_multiplier
        short_stop = ma + atr * self.active_multiplier

        # Önceki stop ile karşılaştır (sadece iyileştirme yönünde güncelle)
        if self.direction == 1:
            # Long'dayız: stop sadece yukarı çekilir
            if long_stop > self.pmax_stop:
                self.pmax_stop = long_stop
            # Fiyat stop'un altına düştü mü?
            if close < self.pmax_stop:
                self._try_flip(-1, close)
        elif self.direction == -1:
            # Short'tayız: stop sadece aşağı çekilir
            if short_stop < self.pmax_stop:
                self.pmax_stop = short_stop
            # Fiyat stop'un üstüne çıktı mı?
            if close > self.pmax_stop:
                self._try_flip(1, close)
        else:
            # İlk yön belirleme
            if close > ma:
                self.direction = 1
                self.pmax_stop = long_stop
            else:
                self.direction = -1
                self.pmax_stop = short_stop

        self.pmax_line = self.pmax_stop
        return self.flipped, self.direction, self.pmax_stop

    def _try_flip(self, new_direction, price):
        """
        PMax yön değişikliği — flip_window teyidi.
        flip_window=1 ise anında flip, >1 ise o kadar mum teyit bekler.
        """
        if self.flip_window <= 1:
            # Anında flip
            self._execute_flip(new_direction, price)
            return

        if new_direction == self.pending_direction:
            self.flip_counter += 1
        else:
            self.pending_direction = new_direction
            self.flip_counter = 1

        if self.flip_counter >= self.flip_window:
            self._execute_flip(new_direction, price)
            self.pending_direction = 0
            self.flip_counter = 0

    def _execute_flip(self, new_direction, price):
        """Yön değişikliğini uygula."""
        self.direction = new_direction
        self.flipped = True
        self.flip_price = price

        # Yeni yöne göre stop ayarla
        atr = self._calc_atr(self.active_atr_period)
        ma = self._calc_ma(self.active_ma_length)

        if new_direction == 1:
            self.pmax_stop = ma - atr * self.active_multiplier
        else:
            self.pmax_stop = ma + atr * self.active_multiplier
