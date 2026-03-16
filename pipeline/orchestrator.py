"""Pipeline Orchestrator — tam otomatik multi-pair optimization coordinator.

Tek bir 'start' komutuyla:
1. Kuyruktan sıradaki pair'i al
2. Veri yoksa indir
3. Strateji seçimine göre R1→R5 pipeline'ı çalıştır (PMax+KC) veya single-pass (Swinginess)
4. Her round sonunda otomatik en iyi trial'ı seç
5. Sonucu kaydet, PDF oluştur
6. Bir sonraki pair'e geç
7. Kuyruk bitene kadar tekrarla

Crash recovery: Backend restart edilirse, kaldığı yerden devam eder.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from pipeline import state_manager, queue_manager
from data.manager import ensure_data

logger = logging.getLogger(__name__)

# Global orchestrator state
_orchestrator_thread: Optional[threading.Thread] = None
_stop_flag = threading.Event()
_event_callback: Optional[Callable] = None


def set_event_callback(callback: Callable):
    """WebSocket event callback'i ayarla."""
    global _event_callback
    _event_callback = callback


def _emit(event_type: str, data: dict):
    """Event broadcast."""
    if _event_callback:
        try:
            _event_callback(event_type, data)
        except Exception as e:
            logger.warning(f"Event broadcast hatası: {e}")


def is_running() -> bool:
    """Pipeline çalışıyor mu?"""
    return _orchestrator_thread is not None and _orchestrator_thread.is_alive()


def start_pipeline(pairs: list[dict] = None):
    """Pipeline'ı başlat.

    Args:
        pairs: [{"symbol": "BTCUSDT", "strategy": "pmax_kc", "timeframes": ["3m"]}, ...]
               None ise mevcut kuyruğu kullanır.
    """
    global _orchestrator_thread

    if is_running():
        logger.warning("Pipeline zaten çalışıyor!")
        return {"error": "Pipeline zaten çalışıyor"}

    # Yeni pair'ler ekle
    if pairs:
        queue_manager.add_batch(pairs)

    _stop_flag.clear()
    _orchestrator_thread = threading.Thread(target=_pipeline_loop, daemon=True)
    _orchestrator_thread.start()

    logger.info("Pipeline başlatıldı")
    _emit("pipeline_status", {"running": True, **queue_manager.get_status()})
    return {"status": "started"}


def stop_pipeline():
    """Pipeline'ı gracefully durdur."""
    _stop_flag.set()
    logger.info("Pipeline durdurma sinyali gönderildi")
    _emit("pipeline_status", {"running": False, "stopping": True})
    return {"status": "stopping"}


def _pipeline_loop():
    """Ana pipeline döngüsü."""
    logger.info("Pipeline döngüsü başladı")

    settings = state_manager.get_settings()

    while not _stop_flag.is_set():
        # Sıradaki job'ı al
        job = queue_manager.get_next()
        if job is None:
            # Kuyruk boş — kontrol et, bitmiş olabilir
            logger.info("Kuyruk boş — pipeline tamamlandı")
            _emit("pipeline_status", {"running": False, "completed": True, **queue_manager.get_status()})
            break

        try:
            _run_job(job, settings)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Job hatası: {job['symbol']} — {e}")
            state_manager.mark_job_failed(job["id"], str(e))
            _emit("error", {"message": f"{job['symbol']} başarısız: {e}"})

    logger.info("Pipeline döngüsü bitti")


def _run_job(job: dict, settings: dict):
    """Tek bir job'ı çalıştır."""
    symbol = job["symbol"]
    strategy = job["strategy"]
    timeframes = job["timeframes"]

    logger.info(f"=== JOB BAŞLIYOR: {symbol} ({strategy}) ===")
    _emit("pipeline_status", {
        "running": True,
        "current_symbol": symbol,
        "current_strategy": strategy,
        **queue_manager.get_status(),
    })

    best_result = None
    best_timeframe = None
    best_ratio = -1

    for tf in timeframes:
        if _stop_flag.is_set():
            logger.info("Durdurma sinyali — job yarıda kesildi")
            return

        state_manager.mark_job_running(job["id"], tf)
        _emit("log", {"message": f"{symbol} {tf} başlıyor...", "level": "info"})

        # 1. Veri kontrolü
        try:
            data_path = ensure_data(symbol, strategy, tf, settings.get("days", 180))
        except FileNotFoundError as e:
            _emit("error", {"message": f"Veri bulunamadı: {e}"})
            continue

        # 2. Strateji çalıştır
        result = _run_strategy(job, strategy, symbol, tf, settings)

        if result and "error" not in result:
            # En iyi timeframe'i bul
            metrics = result.get("selected_metrics", {})
            # Son round'un metriklerinden ratio al
            last_round_key = max(metrics.keys()) if metrics else None
            if last_round_key:
                ratio = metrics[last_round_key].get("ratio", 0)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = result
                    best_timeframe = tf

    if best_result:
        best_result["best_timeframe"] = best_timeframe
        state_manager.mark_job_completed(job["id"], best_result)
        _emit("pair_completed", {
            "symbol": symbol, "strategy": strategy,
            "timeframe": best_timeframe,
            "final_result": best_result,
        })
        logger.info(f"=== JOB TAMAMLANDI: {symbol} (en iyi TF: {best_timeframe}) ===")
    else:
        state_manager.mark_job_failed(job["id"], "Hiçbir timeframe başarılı olmadı")
        _emit("error", {"message": f"{symbol} için başarılı sonuç alınamadı"})


def _run_strategy(job: dict, strategy: str, symbol: str, timeframe: str,
                  settings: dict) -> Optional[dict]:
    """Strateji tipine göre doğru optimizer'ı çalıştır."""
    leverage = settings.get("leverage", 25)
    days = settings.get("days", 180)
    n_jobs = settings.get("n_jobs", 4)
    trial_counts = settings.get("trial_counts", {})

    # Round checkpoint callback
    def on_event(event_type, data):
        _emit(event_type, data)
        # Round tamamlandığında checkpoint kaydet
        if event_type == "round_completed":
            round_num = data.get("round", 0)
            selected = data.get("selected", {})
            state_manager.save_round_checkpoint(
                job["id"], round_num,
                selected.get("params", {}),
                {k: selected.get(k) for k in ["oos_net", "dd", "wr", "ratio", "trades"] if k in selected},
            )

    if strategy == "pmax_kc":
        from strategies.pmax_kc.optimizer import PMaxKCOptimizer

        optimizer = PMaxKCOptimizer(
            symbol=symbol, timeframe=timeframe, days=days,
            leverage=leverage, event_callback=on_event, n_jobs=n_jobs,
        )

        # Crash recovery: önceki round'dan devam
        if job.get("selected_params_per_round"):
            optimizer.selected_params = job["selected_params_per_round"]
            optimizer.selected_metrics = job.get("selected_metrics_per_round", {})

        # Trial counts'u int key'lere çevir
        tc = {int(k.replace("r", "")): v for k, v in trial_counts.items()} if trial_counts else None
        return optimizer.run_full_pipeline(trial_counts=tc)

    elif strategy == "swinginess":
        # Swinginess: single-pass optimization
        _emit("log", {"message": f"Swinginess optimizer henüz pipeline'a entegre değil", "level": "warn"})
        return None

    return None


def resume_on_startup():
    """Backend restart sonrası — kaldığı yerden devam et."""
    running_job = state_manager.get_running_job()
    if running_job:
        logger.info(f"Crash recovery: {running_job['symbol']} kaldığı yerden devam edecek")
        _emit("log", {"message": f"Crash recovery: {running_job['symbol']} devam ediyor"})

        # Running job'ı queued'a geri al (pipeline loop tekrar alacak)
        state_manager.update_job(running_job["id"], {"status": "queued"})

        # Pipeline'ı otomatik başlat
        start_pipeline()
        return True
    return False
