"""Pipeline State Manager — JSON-based persistence + crash recovery."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = Path("results") / "pipeline_state.json"

DEFAULT_SETTINGS = {
    "leverage": 25,
    "days": 180,
    "trial_counts": {"r1": 200, "r2": 300, "r3": 400, "r4": 300, "r5": 200},
    "auto_select_method": "ratio",
    "compounding": {"<50K": 10, "50-100K": 10, "100-200K": 5, "200K+": 2},
    "n_jobs": 4,
}


def _ensure_dir():
    STATE_FILE.parent.mkdir(exist_ok=True)


def load_state() -> dict:
    """Pipeline state'ini diskten yükle."""
    _ensure_dir()
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"State dosyası okunamadı: {e}")
    return {"queue": [], "settings": DEFAULT_SETTINGS.copy(), "last_saved": None}


def save_state(state: dict):
    """Pipeline state'ini diske kaydet."""
    _ensure_dir()
    state["last_saved"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_settings() -> dict:
    """Mevcut ayarları döndür."""
    state = load_state()
    return state.get("settings", DEFAULT_SETTINGS.copy())


def update_settings(new_settings: dict):
    """Ayarları güncelle."""
    state = load_state()
    state["settings"] = {**state.get("settings", DEFAULT_SETTINGS.copy()), **new_settings}
    save_state(state)
    return state["settings"]


def add_job(symbol: str, strategy: str, timeframes: list[str]) -> dict:
    """Kuyruğa yeni job ekle."""
    state = load_state()
    job = {
        "id": str(uuid.uuid4())[:8],
        "symbol": symbol,
        "strategy": strategy,
        "timeframes": timeframes,
        "status": "queued",  # queued | downloading | running | completed | failed
        "current_round": 0,
        "current_timeframe": None,
        "total_rounds": 5 if strategy == "pmax_kc" else 1,
        "selected_params_per_round": {},
        "selected_metrics_per_round": {},
        "final_result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
    }
    state["queue"].append(job)
    save_state(state)
    return job


def remove_job(job_id: str) -> bool:
    """Kuyruğtan job sil (sadece queued olanlar)."""
    state = load_state()
    for i, job in enumerate(state["queue"]):
        if job["id"] == job_id and job["status"] == "queued":
            state["queue"].pop(i)
            save_state(state)
            return True
    return False


def get_next_job() -> Optional[dict]:
    """Sıradaki 'queued' job'ı döndür."""
    state = load_state()
    for job in state["queue"]:
        if job["status"] == "queued":
            return job
    return None


def update_job(job_id: str, updates: dict):
    """Job'ı güncelle."""
    state = load_state()
    for job in state["queue"]:
        if job["id"] == job_id:
            job.update(updates)
            break
    save_state(state)


def mark_job_running(job_id: str, timeframe: str = None):
    """Job'ı running olarak işaretle."""
    update_job(job_id, {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "current_timeframe": timeframe,
    })


def mark_job_completed(job_id: str, result: dict):
    """Job'ı completed olarak işaretle."""
    update_job(job_id, {
        "status": "completed",
        "completed_at": datetime.now().isoformat(),
        "final_result": result,
    })


def mark_job_failed(job_id: str, error: str):
    """Job'ı failed olarak işaretle."""
    update_job(job_id, {
        "status": "failed",
        "error": error,
        "completed_at": datetime.now().isoformat(),
    })


def save_round_checkpoint(job_id: str, round_num: int, params: dict, metrics: dict):
    """Round tamamlandığında checkpoint kaydet."""
    state = load_state()
    for job in state["queue"]:
        if job["id"] == job_id:
            job["current_round"] = round_num
            job["selected_params_per_round"][f"r{round_num}"] = params
            job["selected_metrics_per_round"][f"r{round_num}"] = metrics
            break
    save_state(state)


def get_running_job() -> Optional[dict]:
    """Çalışmakta olan job'ı bul (crash recovery için)."""
    state = load_state()
    for job in state["queue"]:
        if job["status"] == "running":
            return job
    return None


def get_completed_jobs() -> list[dict]:
    """Tamamlanan tüm job'ları döndür."""
    state = load_state()
    return [j for j in state["queue"] if j["status"] == "completed"]


def get_queue() -> list[dict]:
    """Tüm kuyruğu döndür."""
    state = load_state()
    return state["queue"]
