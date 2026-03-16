"""Unified Data Manager — strateji tipine göre doğru downloader'ı kullanır."""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_data(symbol: str, strategy: str, timeframe: str = "3m",
                days: int = 180, progress_callback=None) -> str:
    """Veri yoksa indir, varsa path'ini döndür.

    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        strategy: "pmax_kc" veya "swinginess"
        timeframe: "1m", "3m", "5m", "15m" (sadece pmax_kc için)
        days: Lookback gün sayısı
        progress_callback: İlerleme callback(symbol, progress_pct)

    Returns:
        Veri dosyasının path'i
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if strategy == "pmax_kc":
        return _ensure_klines(symbol, timeframe, days, data_dir, progress_callback)
    elif strategy == "swinginess":
        return _ensure_aggtrades(symbol, data_dir, progress_callback)
    else:
        raise ValueError(f"Bilinmeyen strateji: {strategy}")


def _ensure_klines(symbol: str, timeframe: str, days: int,
                   data_dir: Path, progress_callback=None) -> str:
    """Klines (OHLCV) verisini kontrol et/indir."""
    path = data_dir / f"{symbol}_{timeframe}_{days}d.parquet"

    if path.exists():
        logger.info(f"Klines cache mevcut: {path}")
        return str(path)

    logger.info(f"Klines indiriliyor: {symbol} {timeframe} ({days} gün)...")
    from data.downloader_klines import fetch_klines

    df = fetch_klines(symbol, timeframe, days)
    df.to_parquet(str(path), index=False)
    logger.info(f"Klines kaydedildi: {len(df)} mum → {path}")

    if progress_callback:
        progress_callback(symbol, 100)

    return str(path)


def _ensure_aggtrades(symbol: str, data_dir: Path,
                      progress_callback=None) -> str:
    """AggTrades (tick) verisini kontrol et/indir."""
    path = data_dir / f"{symbol.lower()}_aggtrades.parquet"

    if path.exists():
        logger.info(f"AggTrades cache mevcut: {path}")
        return str(path)

    logger.info(f"AggTrades indiriliyor: {symbol}...")
    from data.downloader_aggtrades import download_symbol

    # download_symbol fonksiyonu varsa kullan, yoksa ana script'ten çağır
    try:
        download_symbol(symbol, str(data_dir), progress_callback)
    except (ImportError, AttributeError):
        logger.warning(f"AggTrades downloader bulunamadı — manual download gerekebilir")
        raise FileNotFoundError(
            f"AggTrades verisi bulunamadı: {path}\n"
            f"Manuel indirme: python download_aggtrades.py"
        )

    return str(path)


def get_data_status(symbols: list[str] = None) -> list[dict]:
    """Tüm sembollerin veri durumunu döndür."""
    import pandas as pd

    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]

    data_dir = Path("data")
    results = []

    for symbol in symbols:
        status = {"symbol": symbol, "klines": {}, "aggtrades": {}}

        # Klines kontrol
        for tf in ["1m", "3m", "5m", "15m"]:
            for days in [180]:
                path = data_dir / f"{symbol}_{tf}_{days}d.parquet"
                if path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    try:
                        df = pd.read_parquet(path)
                        status["klines"][tf] = {
                            "available": True,
                            "size_mb": round(size_mb, 1),
                            "candles": len(df),
                            "date_from": str(pd.to_datetime(df["open_time"].iloc[0], unit="ms").date()),
                            "date_to": str(pd.to_datetime(df["open_time"].iloc[-1], unit="ms").date()),
                        }
                    except:
                        status["klines"][tf] = {"available": True, "size_mb": round(size_mb, 1)}

        # AggTrades kontrol
        agg_path = data_dir / f"{symbol.lower()}_aggtrades.parquet"
        if agg_path.exists():
            size_mb = agg_path.stat().st_size / (1024 * 1024)
            try:
                df = pd.read_parquet(agg_path)
                status["aggtrades"] = {
                    "available": True,
                    "size_mb": round(size_mb, 1),
                    "tick_count": len(df),
                    "date_from": str(pd.to_datetime(df["timestamp"].iloc[0], unit="ms").date()),
                    "date_to": str(pd.to_datetime(df["timestamp"].iloc[-1], unit="ms").date()),
                }
            except:
                status["aggtrades"] = {"available": True, "size_mb": round(size_mb, 1)}
        else:
            status["aggtrades"] = {"available": False}

        results.append(status)

    return results
