"""Adaptive PMax — volatiliteye gore parametrelerini otomatik ayarlayan PMax.

Orijinal PMax'i Scalper Bot'tan import eder, uzerine adaptif katman ekler.
Scalper Bot koduna DOKUNMAZ.

3 adaptasyon modu:
  1. ATR-Normalized: Sadece atr_multiplier volatiliteye gore kayar
  2. Preset: 3 rejim preset'i arasindan otomatik secer
  3. Continuous: Her bar'da tum parametreleri surekli ayarlar
"""

from __future__ import annotations

import sys
import math
import numpy as np
import pandas as pd

from strategies.pmax_kc.config import SCALPER_BOT_PATH

if SCALPER_BOT_PATH not in sys.path:
    sys.path.append(SCALPER_BOT_PATH)
from core.strategy.indicators import (
    pmax as original_pmax,
    atr_rma,
    variant,
    ema,
)


# =====================================================================
# Mode 1: ATR-Normalized Multiplier (en basit)
# =====================================================================

def adaptive_pmax_normalized(
    src: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    base_atr_period: int = 10,
    base_atr_multiplier: float = 3.0,
    ma_type: str = "EMA",
    ma_length: int = 10,
    lookback: int = 480,  # 24 saat (3m bar)
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """PMax with ATR-normalized multiplier.

    atr_multiplier = base * (current_atr / median_atr_24h)

    Volatilite yukseldikce multiplier genisler (daha az sinyal, daha saglam).
    Volatilite dustukce multiplier daralir (daha cok sinyal, daha hassas).

    Returns: (pmax_line, mavg, direction, adaptive_multiplier_series)
    """
    # ATR hesapla
    atr_vals = atr_rma(high, low, close, base_atr_period)
    atr_arr = atr_vals.values
    n = len(atr_arr)

    # MA hesapla
    mavg = variant(ma_type, src, ma_length)
    mavg_vals = mavg.values.copy()
    close_vals = close.values.copy()

    # Adaptive multiplier hesapla
    adaptive_mult = np.full(n, base_atr_multiplier)
    for i in range(lookback, n):
        window = atr_arr[max(0, i - lookback):i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 10:
            median_atr = np.median(valid)
            current_atr = atr_arr[i]
            if not np.isnan(current_atr) and median_atr > 0:
                ratio = current_atr / median_atr
                # Clamp: 0.5x - 2.0x base
                ratio = max(0.5, min(2.0, ratio))
                adaptive_mult[i] = base_atr_multiplier * ratio

    # PMax hesapla (orijinal lojigin aynisi, adaptive multiplier ile)
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    direction = np.ones(n)
    pmax_line = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(mavg_vals[i]) or np.isnan(atr_arr[i]):
            continue

        mult = adaptive_mult[i]
        atr_component = atr_arr[i]

        ls = mavg_vals[i] - mult * atr_component
        prev_ls = long_stop[i - 1] if not np.isnan(long_stop[i - 1]) else ls
        long_stop[i] = max(ls, prev_ls) if mavg_vals[i] > prev_ls else ls

        ss = mavg_vals[i] + mult * atr_component
        prev_ss = short_stop[i - 1] if not np.isnan(short_stop[i - 1]) else ss
        short_stop[i] = min(ss, prev_ss) if mavg_vals[i] < prev_ss else ss

        prev_dir = direction[i - 1]
        if prev_dir == -1 and mavg_vals[i] > short_stop[i - 1]:
            direction[i] = 1
        elif prev_dir == 1 and mavg_vals[i] < long_stop[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_dir

        pmax_line[i] = long_stop[i] if direction[i] == 1 else short_stop[i]

    return (
        pd.Series(pmax_line, index=src.index),
        mavg,
        pd.Series(direction, index=src.index),
        pd.Series(adaptive_mult, index=src.index),
    )


# =====================================================================
# Mode 2: Preset Regime Selector
# =====================================================================

PRESETS = {
    "trending": {"atr_multiplier": 3.5, "ma_length": 12, "atr_period": 10},
    "ranging":  {"atr_multiplier": 2.0, "ma_length": 7,  "atr_period": 8},
    "volatile": {"atr_multiplier": 4.0, "ma_length": 15, "atr_period": 14},
}


def detect_regime(
    atr_arr: np.ndarray,
    close_arr: np.ndarray,
    mavg_arr: np.ndarray,
    dir_arr: np.ndarray,
    idx: int,
    lookback: int = 480,
) -> str:
    """Detect market regime at bar index."""
    start = max(0, idx - lookback)

    # Vol level
    atr_window = atr_arr[start:idx + 1]
    valid_atr = atr_window[~np.isnan(atr_window)]
    if len(valid_atr) < 10:
        return "trending"  # default

    median_atr = np.median(valid_atr)
    current_atr = atr_arr[idx] if not np.isnan(atr_arr[idx]) else median_atr
    vol_ratio = current_atr / median_atr if median_atr > 0 else 1.0

    # Trend strength
    close_v = close_arr[idx]
    mavg_v = mavg_arr[idx] if not np.isnan(mavg_arr[idx]) else close_v
    trend_dist = abs(close_v - mavg_v) / current_atr if current_atr > 0 else 0

    # Momentum (flip count in last 120 bars)
    flip_start = max(0, idx - 120)
    dir_window = dir_arr[flip_start:idx + 1]
    valid_dir = dir_window[~np.isnan(dir_window)]
    flips = int(np.sum(np.diff(valid_dir) != 0)) if len(valid_dir) > 1 else 0

    # Classify
    if vol_ratio > 1.5 or flips >= 4:
        return "volatile"
    elif trend_dist < 0.5 and vol_ratio < 0.8:
        return "ranging"
    else:
        return "trending"


def adaptive_pmax_preset(
    src: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ma_type: str = "EMA",
    base_atr_period: int = 10,
    regime_check_interval: int = 20,  # her 20 bar'da rejim kontrol
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """PMax with preset-based regime switching.

    Her regime_check_interval bar'da rejimi kontrol eder,
    uygun preset'i secer ve PMax parametrelerini degistirir.

    Returns: (pmax_line, mavg, direction, regime_series)
    """
    n = len(close)
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values

    # Baslangic icin full ATR ve MA hesapla (rejim tespiti icin)
    base_atr = atr_rma(high, low, close, base_atr_period).values
    base_mavg = variant(ma_type, src, 10).values
    base_dir = np.ones(n)

    # Once basit bir direction hesapla (rejim tespiti icin)
    for i in range(1, n):
        if np.isnan(base_mavg[i]) or np.isnan(base_atr[i]):
            continue
        ls = base_mavg[i] - 3.0 * base_atr[i]
        prev_ls = ls
        ss = base_mavg[i] + 3.0 * base_atr[i]
        if i > 0:
            base_dir[i] = base_dir[i - 1]

    # Simdi adaptive pmax hesapla
    regime_labels = np.full(n, "", dtype=object)
    current_regime = "trending"
    current_preset = PRESETS["trending"]

    # Segment-based: her rejim degisiminde yeni PMax hesapla
    pmax_line = np.full(n, np.nan)
    mavg_out = np.full(n, np.nan)
    direction = np.ones(n)
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)

    # Aktif parametreler
    active_mult = current_preset["atr_multiplier"]
    active_ma_len = current_preset["ma_length"]
    active_atr_period = current_preset["atr_period"]

    # ATR ve MA'yi aktif parametrelerle hesapla
    atr_arr = atr_rma(high, low, close, active_atr_period).values
    mavg_arr = variant(ma_type, src, active_ma_len).values

    for i in range(1, n):
        # Rejim kontrolu
        if i >= 480 and i % regime_check_interval == 0:
            new_regime = detect_regime(base_atr, close_arr, base_mavg, direction, i)
            if new_regime != current_regime:
                current_regime = new_regime
                current_preset = PRESETS[new_regime]
                active_mult = current_preset["atr_multiplier"]
                active_ma_len = current_preset["ma_length"]
                active_atr_period = current_preset["atr_period"]
                # Yeni parametrelerle ATR ve MA guncelle
                atr_arr = atr_rma(high, low, close, active_atr_period).values
                mavg_arr = variant(ma_type, src, active_ma_len).values

        regime_labels[i] = current_regime

        if np.isnan(mavg_arr[i]) or np.isnan(atr_arr[i]):
            continue

        mavg_out[i] = mavg_arr[i]
        atr_component = atr_arr[i]

        ls = mavg_arr[i] - active_mult * atr_component
        prev_ls = long_stop[i - 1] if not np.isnan(long_stop[i - 1]) else ls
        long_stop[i] = max(ls, prev_ls) if mavg_arr[i] > prev_ls else ls

        ss = mavg_arr[i] + active_mult * atr_component
        prev_ss = short_stop[i - 1] if not np.isnan(short_stop[i - 1]) else ss
        short_stop[i] = min(ss, prev_ss) if mavg_arr[i] < prev_ss else ss

        prev_dir = direction[i - 1]
        if prev_dir == -1 and mavg_arr[i] > short_stop[i - 1]:
            direction[i] = 1
        elif prev_dir == 1 and mavg_arr[i] < long_stop[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_dir

        pmax_line[i] = long_stop[i] if direction[i] == 1 else short_stop[i]

    return (
        pd.Series(pmax_line, index=src.index),
        pd.Series(mavg_out, index=src.index),
        pd.Series(direction, index=src.index),
        pd.Series(regime_labels, index=src.index),
    )


# =====================================================================
# Mode 3: Continuous Adaptive (en akilli)
# =====================================================================

def adaptive_pmax_continuous(
    src: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ma_type: str = "EMA",
    base_atr_period: int = 10,
    base_atr_multiplier: float = 3.0,
    base_ma_length: int = 10,
    lookback: int = 480,
    # Tunable parameters
    flip_window: int = 120,
    mult_base: float = 2.0,
    mult_scale: float = 1.0,
    ma_base: int = 8,
    ma_scale: float = 3.0,
    atr_base: int = 8,
    atr_scale: float = 1.5,
    update_interval: int = 1,  # kac bar'da bir parametre guncelle (1=her bar, 20=her saat)
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """PMax with continuously adaptive parameters.

    Tunable params:
      - lookback:    vol median penceresi (bar)
      - flip_window: yon degisimi sayma penceresi (bar)
      - mult_base/scale: atr_multiplier = mult_base + vol_ratio * mult_scale
      - ma_base/scale:   ma_length = round(ma_base + trend_dist * ma_scale)
      - atr_base/scale:  atr_period = round(atr_base + flip_count * atr_scale)

    Returns: (pmax_line, mavg, direction, mult_series, ma_len_series, atr_period_series)
    """
    n = len(close)
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values

    # Base ATR for regime detection
    base_atr = atr_rma(high, low, close, base_atr_period).values

    # Pre-compute MA variants for different lengths (cache)
    ma_cache: dict[int, np.ndarray] = {}
    for ml in range(5, 25):
        ma_cache[ml] = variant(ma_type, src, ml).values

    # Pre-compute ATR variants for different periods
    atr_cache: dict[int, np.ndarray] = {}
    for ap in range(5, 25):
        atr_cache[ap] = atr_rma(high, low, close, ap).values

    # Output arrays
    pmax_line = np.full(n, np.nan)
    mavg_out = np.full(n, np.nan)
    direction = np.ones(n)
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)

    mult_series = np.full(n, base_atr_multiplier)
    ma_len_series = np.full(n, float(base_ma_length))
    atr_p_series = np.full(n, float(base_atr_period))

    for i in range(1, n):
        # Adaptive parameters
        # Onceki bar'in degerlerini koru (update olmayan bar'larda)
        if i > 1:
            active_mult = mult_series[i - 1]
            active_ma_len = int(ma_len_series[i - 1])
            active_atr_p = int(atr_p_series[i - 1])
        else:
            active_mult = base_atr_multiplier
            active_ma_len = base_ma_length
            active_atr_p = base_atr_period

        if i >= lookback and (update_interval <= 1 or i % update_interval == 0):
            # 1. Vol ratio -> atr_multiplier
            window = base_atr[max(0, i - lookback):i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 10:
                median_atr = np.median(valid)
                current_atr = base_atr[i]
                if not np.isnan(current_atr) and median_atr > 0:
                    vol_ratio = current_atr / median_atr
                    vol_ratio = max(0.5, min(2.0, vol_ratio))
                    active_mult = mult_base + vol_ratio * mult_scale

            # 2. Trend dist -> ma_length
            mavg_base = ma_cache.get(base_ma_length, ma_cache[10])
            mavg_v = mavg_base[i] if not np.isnan(mavg_base[i]) else close_arr[i]
            c_atr = base_atr[i] if not np.isnan(base_atr[i]) else 1.0
            trend_dist = abs(close_arr[i] - mavg_v) / c_atr if c_atr > 0 else 0
            trend_dist = min(4.0, trend_dist)
            active_ma_len = int(round(ma_base + trend_dist * ma_scale))
            active_ma_len = max(5, min(24, active_ma_len))

            # 3. Flip count -> atr_period
            flip_start = max(0, i - flip_window)
            dir_window = direction[flip_start:i]
            flips = int(np.sum(np.diff(dir_window) != 0)) if len(dir_window) > 1 else 0
            active_atr_p = int(round(atr_base + flips * atr_scale))
            active_atr_p = max(5, min(24, active_atr_p))

        mult_series[i] = active_mult
        ma_len_series[i] = float(active_ma_len)
        atr_p_series[i] = float(active_atr_p)

        # Get cached values
        mavg_arr = ma_cache.get(active_ma_len, ma_cache[10])
        atr_arr = atr_cache.get(active_atr_p, atr_cache[10])

        if np.isnan(mavg_arr[i]) or np.isnan(atr_arr[i]):
            continue

        mavg_out[i] = mavg_arr[i]

        ls = mavg_arr[i] - active_mult * atr_arr[i]
        prev_ls = long_stop[i - 1] if not np.isnan(long_stop[i - 1]) else ls
        long_stop[i] = max(ls, prev_ls) if mavg_arr[i] > prev_ls else ls

        ss = mavg_arr[i] + active_mult * atr_arr[i]
        prev_ss = short_stop[i - 1] if not np.isnan(short_stop[i - 1]) else ss
        short_stop[i] = min(ss, prev_ss) if mavg_arr[i] < prev_ss else ss

        prev_dir = direction[i - 1]
        if prev_dir == -1 and mavg_arr[i] > short_stop[i - 1]:
            direction[i] = 1
        elif prev_dir == 1 and mavg_arr[i] < long_stop[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_dir

        pmax_line[i] = long_stop[i] if direction[i] == 1 else short_stop[i]

    return (
        pd.Series(pmax_line, index=src.index),
        pd.Series(mavg_out, index=src.index),
        pd.Series(direction, index=src.index),
        pd.Series(mult_series, index=src.index),
        pd.Series(ma_len_series, index=src.index),
        pd.Series(atr_p_series, index=src.index),
    )


# =====================================================================
# Adaptive Keltner Channel
# =====================================================================

def adaptive_keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    direction_arr: np.ndarray,
    # Base KC params (Round 4 best)
    base_kc_length: int = 5,
    base_kc_mult: float = 0.7,
    base_kc_atr: int = 26,
    base_max_dca: int = 5,
    base_tp_pct: float = 0.50,
    # Adaptive tuning
    kc_vol_lookback: int = 80,
    kc_update_interval: int = 40,
    # KC multiplier adaptive: kc_mult = kc_mult_base + vol_ratio * kc_mult_scale
    kc_mult_base: float = 0.3,
    kc_mult_scale: float = 0.4,
    # DCA adaptive: max_dca = dca_base + trend_bonus
    kc_dca_base: int = 3,
    kc_dca_trend_bonus: int = 2,
    # TP adaptive: tp_pct = tp_base + vol_ratio * tp_scale
    kc_tp_base: float = 0.30,
    kc_tp_scale: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive Keltner Channel - piyasa kosullarina gore KC parametrelerini ayarlar.

    Returns: (kc_upper, kc_lower, kc_mid, adaptive_kc_mult, adaptive_max_dca, adaptive_tp_pct)
    """
    n = len(close)
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values

    base_atr = atr_rma(high, low, close, base_kc_atr).values

    # Pre-compute KC EMA variants
    kc_ema_cache = {}
    for kl in [3, 5, 7, 10, 13, 16, 20]:
        kc_ema_cache[kl] = ema(close, kl).values

    # Pre-compute KC ATR variants
    kc_atr_cache = {}
    for ka in [10, 14, 18, 22, 26, 30]:
        kc_atr_cache[ka] = atr_rma(high, low, close, ka).values

    # Output arrays
    kc_upper_out = np.full(n, np.nan)
    kc_lower_out = np.full(n, np.nan)
    kc_mid_out = np.full(n, np.nan)
    kc_mult_out = np.full(n, base_kc_mult)
    max_dca_out = np.full(n, float(base_max_dca))
    tp_pct_out = np.full(n, base_tp_pct)

    # Active adaptive values
    active_kc_mult = base_kc_mult
    active_max_dca = base_max_dca
    active_tp_pct = base_tp_pct

    for i in range(1, n):
        # Keep previous values by default
        if i > 1:
            active_kc_mult = kc_mult_out[i - 1]
            active_max_dca = int(max_dca_out[i - 1])
            active_tp_pct = tp_pct_out[i - 1]

        # Update at intervals
        if i >= kc_vol_lookback and (kc_update_interval <= 1 or i % kc_update_interval == 0):
            # Vol ratio
            window = base_atr[max(0, i - kc_vol_lookback):i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 10:
                median_atr = np.median(valid)
                current_atr = base_atr[i]
                if not np.isnan(current_atr) and median_atr > 0:
                    vol_ratio = current_atr / median_atr
                    vol_ratio = max(0.5, min(2.0, vol_ratio))

                    # 1. KC multiplier: vol yuksek = genis bant, vol dusuk = dar
                    active_kc_mult = kc_mult_base + vol_ratio * kc_mult_scale
                    active_kc_mult = max(0.3, min(2.0, active_kc_mult))

                    # 2. TP pct: vol yuksek = buyuk TP, vol dusuk = kucuk TP
                    active_tp_pct = kc_tp_base + vol_ratio * kc_tp_scale
                    active_tp_pct = max(0.10, min(0.80, active_tp_pct))

            # 3. DCA steps: trend gucune gore
            # Son 120 bar'da PMax direction degisimi
            flip_start = max(0, i - 120)
            dir_window = direction_arr[flip_start:i + 1]
            valid_dir = dir_window[~np.isnan(dir_window)]
            flips = int(np.sum(np.diff(valid_dir) != 0)) if len(valid_dir) > 1 else 0

            if flips <= 1:  # sakin piyasa, guclu trend
                active_max_dca = kc_dca_base + kc_dca_trend_bonus
            elif flips <= 3:  # orta
                active_max_dca = kc_dca_base + 1
            else:  # choppy, dikkatli ol
                active_max_dca = kc_dca_base
            active_max_dca = max(1, min(7, active_max_dca))

        kc_mult_out[i] = active_kc_mult
        max_dca_out[i] = float(active_max_dca)
        tp_pct_out[i] = active_tp_pct

        # Compute KC bands with active multiplier
        kc_mid_val = kc_ema_cache.get(base_kc_length, kc_ema_cache[5])[i]
        kc_atr_val = kc_atr_cache.get(base_kc_atr, kc_atr_cache[26])[i]

        if not np.isnan(kc_mid_val) and not np.isnan(kc_atr_val):
            kc_mid_out[i] = kc_mid_val
            kc_upper_out[i] = kc_mid_val + active_kc_mult * kc_atr_val
            kc_lower_out[i] = kc_mid_val - active_kc_mult * kc_atr_val

    return kc_upper_out, kc_lower_out, kc_mid_out, kc_mult_out, max_dca_out, tp_pct_out
