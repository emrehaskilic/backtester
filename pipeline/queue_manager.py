"""Queue Manager — multi-pair sequential queue management."""

import logging
from typing import Optional

from pipeline import state_manager

logger = logging.getLogger(__name__)


def add_to_queue(symbol: str, strategy: str, timeframes: list[str]) -> dict:
    """Kuyruğa yeni pair ekle."""
    job = state_manager.add_job(symbol, strategy, timeframes)
    logger.info(f"Kuyruğa eklendi: {symbol} ({strategy}) timeframes={timeframes} id={job['id']}")
    return job


def add_batch(pairs: list[dict]) -> list[dict]:
    """Birden fazla pair'i kuyruğa ekle.

    Args:
        pairs: [{"symbol": "BTCUSDT", "strategy": "pmax_kc", "timeframes": ["3m"]}, ...]

    Returns:
        Eklenen job'ların listesi
    """
    jobs = []
    for p in pairs:
        job = add_to_queue(p["symbol"], p["strategy"], p.get("timeframes", ["3m"]))
        jobs.append(job)
    return jobs


def remove_from_queue(job_id: str) -> bool:
    """Kuyruğtan job sil."""
    return state_manager.remove_job(job_id)


def get_next() -> Optional[dict]:
    """Sıradaki job'ı al."""
    return state_manager.get_next_job()


def get_status() -> dict:
    """Kuyruk durumunu döndür."""
    queue = state_manager.get_queue()
    running = [j for j in queue if j["status"] == "running"]
    queued = [j for j in queue if j["status"] == "queued"]
    completed = [j for j in queue if j["status"] == "completed"]
    failed = [j for j in queue if j["status"] == "failed"]

    return {
        "total": len(queue),
        "running": len(running),
        "queued": len(queued),
        "completed": len(completed),
        "failed": len(failed),
        "current": running[0] if running else None,
        "queue": queue,
    }
