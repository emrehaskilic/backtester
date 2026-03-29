"""Pipeline State Manager — JSON-based persistence + crash recovery.

V2: Interactive step-based pipeline with top-10 selection.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = Path("results") / "pipeline_state.json"
V2_STATE_FILE = Path("results") / "pipeline_v2_state.json"

DEFAULT_SETTINGS = {
    "leverage": 25,
    "days": 180,
    "trial_counts": {"r1": 200, "r2": 300, "r3": 400, "r4": 300, "r5": 200},
    "auto_select_method": "ratio",
    "compounding": {"<50K": 10, "50-100K": 10, "100-200K": 5, "200K+": 2},
    "n_jobs": 4,
}

# ============================================================
# Pipeline V2 — Interactive Step-Based State
# ============================================================

PIPELINE_V2_STEPS = [
    {"key": "pmax_discovery", "label": "PMax Keşif", "strategy": "unified_pmax"},
    {"key": "kc_optimize", "label": "KC Optimize", "strategy": "unified_kc"},
    {"key": "kelly_dyncomp", "label": "Kelly DynComp", "strategy": "unified_kelly"},
    {"key": "dynsl_test", "label": "DynSL Test", "strategy": "unified_dynsl"},
]

DEFAULT_V2_STATE = {
    "active": False,
    "current_step": 0,  # 0-3 (index into PIPELINE_V2_STEPS)
    "step_status": "idle",  # idle | running | awaiting_selection | completed
    "symbol": "ETHUSDT",
    "timeframe": "3m",
    "locked_params": {},  # Önceki adımlardan kilitli parametreler
    "step_results": {},  # Her adımın top-10 sonuçları: {"pmax_discovery": [...], ...}
    "selected_indices": {},  # Kullanıcının her adımda seçtiği index: {"pmax_discovery": 3, ...}
    "n_trials": 1000,
    "started_at": None,
    "last_updated": None,
}


def load_v2_state() -> dict:
    """V2 pipeline state'ini yükle."""
    V2_STATE_FILE.parent.mkdir(exist_ok=True)
    if V2_STATE_FILE.exists():
        try:
            with open(V2_STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"V2 state dosyası okunamadı: {e}")
    return DEFAULT_V2_STATE.copy()


def save_v2_state(state: dict):
    """V2 pipeline state'ini kaydet."""
    V2_STATE_FILE.parent.mkdir(exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    with open(V2_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def init_v2_pipeline(symbol: str, timeframe: str = "3m", n_trials: int = 1000) -> dict:
    """V2 pipeline'ı başlat — tüm state sıfırla."""
    state = DEFAULT_V2_STATE.copy()
    state["active"] = True
    state["symbol"] = symbol
    state["timeframe"] = timeframe
    state["n_trials"] = n_trials
    state["current_step"] = 0
    state["step_status"] = "idle"
    state["locked_params"] = {}
    state["step_results"] = {}
    state["selected_indices"] = {}
    state["started_at"] = datetime.now().isoformat()
    save_v2_state(state)
    return state


def set_v2_step_running(step_index: int):
    """Adımı running olarak işaretle."""
    state = load_v2_state()
    state["current_step"] = step_index
    state["step_status"] = "running"
    save_v2_state(state)


def set_v2_step_results(step_key: str, top10: list[dict]):
    """Adım tamamlandı — top-10 sonuçları kaydet, seçim bekle."""
    state = load_v2_state()
    state["step_results"][step_key] = top10
    state["step_status"] = "awaiting_selection"
    save_v2_state(state)


def select_v2_result(step_key: str, selected_index: int) -> dict:
    """Kullanıcı top-10'dan birini seçti — kilitli parametreleri güncelle."""
    state = load_v2_state()
    top10 = state["step_results"].get(step_key, [])
    if selected_index < 0 or selected_index >= len(top10):
        return {"error": f"Geçersiz index: {selected_index}, top10 uzunluğu: {len(top10)}"}

    selected = top10[selected_index]
    state["selected_indices"][step_key] = selected_index

    # Seçilen parametreleri locked_params'a ekle
    if step_key == "pmax_discovery":
        state["locked_params"]["pmax"] = selected["params"]
    elif step_key == "kc_optimize":
        state["locked_params"]["kc"] = selected["params"]
    elif step_key == "kelly_dyncomp":
        state["locked_params"]["kelly"] = selected["params"]
    elif step_key == "dynsl_test":
        state["locked_params"]["dynsl"] = selected["params"]

    # Sonraki adıma geç
    current = state["current_step"]
    if current < len(PIPELINE_V2_STEPS) - 1:
        state["current_step"] = current + 1
        state["step_status"] = "idle"
    else:
        state["step_status"] = "completed"
        state["active"] = False

    save_v2_state(state)
    return state


def update_v2_settings(timeframe: str = None, n_trials: int = None):
    """V2 pipeline ayarlarını güncelle (sıfırlamadan)."""
    state = load_v2_state()
    if timeframe:
        state["timeframe"] = timeframe
    if n_trials:
        state["n_trials"] = n_trials
    save_v2_state(state)
    return state


def reset_v2_pipeline():
    """V2 pipeline'ı sıfırla."""
    state = DEFAULT_V2_STATE.copy()
    save_v2_state(state)
    return state


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
    total_rounds = 5 if strategy == "pmax_kc" else (13 if strategy == "walkforward_pmax" else 1)
    job = {
        "id": str(uuid.uuid4())[:8],
        "symbol": symbol,
        "strategy": strategy,
        "timeframes": timeframes,
        "status": "queued",  # queued | downloading | running | completed | failed
        "current_round": 0,
        "current_timeframe": None,
        "total_rounds": total_rounds,
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
