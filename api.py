"""
Tam Otomatik Optimizasyon Canavarı — Unified FastAPI Backend.
PMax+KC + Swinginess stratejileri, WebSocket real-time updates, pipeline orchestration.

Port: 8055
"""

import os
import json
import asyncio
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("api")

app = FastAPI(title="Optimizasyon Canavarı API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]

# ============================================================
# WebSocket Manager
# ============================================================

class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WS bağlandı ({len(self.active_connections)} aktif)")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WS ayrıldı ({len(self.active_connections)} aktif)")

    async def broadcast(self, event_type: str, data: dict):
        message = json.dumps({"type": event_type, "data": data}, default=str)
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_text(message)
            except:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    def sync_broadcast(self, event_type: str, data: dict):
        """Thread'den çağrılabilir senkron broadcast."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.broadcast(event_type, data))
            else:
                loop.run_until_complete(self.broadcast(event_type, data))
        except RuntimeError:
            # Event loop yoksa yeni oluştur
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.broadcast(event_type, data))


ws_manager = WebSocketManager()


@app.websocket("/ws/pipeline")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Client'tan gelen mesajları işle (gerekirse)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ============================================================
# In-memory state (eski endpoint'ler için)
# ============================================================

backtest_state = {"running": False, "progress": 0, "result": None, "symbol": None}
optimize_state = {"running": False, "progress": 0, "trial": 0, "total": 0, "symbol": None, "error": None}
download_state = {"running": False, "progress": 0, "symbol": None, "month": ""}


# ============================================================
# Models
# ============================================================

class BacktestParams(BaseModel):
    symbol: str = "BTCUSDT"
    strategy: str = "swinginess"  # "swinginess" | "pmax_kc"
    # Swinginess params
    w_delta: float = 0.22
    w_cvd: float = 0.18
    w_logp: float = 0.12
    w_obi_w: float = 0.14
    w_obi_d: float = 0.12
    w_sweep: float = 0.08
    w_burst: float = 0.08
    w_oi: float = 0.06
    trs_confirm_ticks: int = 90
    trs_bullish_zone: float = 0.65
    trs_bearish_zone: float = 0.35
    trs_agreement: float = 0.40
    stop_loss_pct: float = 1.5
    trailing_activation_pct: float = 0.8
    trailing_distance_pct: float = 0.5
    exit_score_hard: float = 0.85
    exit_score_soft: float = 0.70
    min_prints_per_sec: float = 1.0
    entry_cooldown_sec: int = 300
    rolling_window_sec: int = 3600
    time_flat_sec: int = 14400
    margin_per_trade: float = 100
    leverage: int = 50
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005


class OptimizeRequest(BaseModel):
    symbol: str = "BTCUSDT"
    n_trials: int = 100


class PipelineStartRequest(BaseModel):
    pairs: list[dict]  # [{"symbol": "BTCUSDT", "strategy": "pmax_kc", "timeframes": ["3m"]}]


class PipelineAddRequest(BaseModel):
    symbol: str
    strategy: str = "pmax_kc"
    timeframes: list[str] = ["3m"]


class SettingsRequest(BaseModel):
    leverage: Optional[int] = None
    days: Optional[int] = None
    trial_counts: Optional[dict] = None
    auto_select_method: Optional[str] = None
    compounding: Optional[dict] = None
    n_jobs: Optional[int] = None


# ============================================================
# HEALTH & META
# ============================================================

@app.get("/api/health")
def health():
    from pipeline.orchestrator import is_running
    return {"status": "ok", "version": "2.0.0", "pipeline_running": is_running()}


@app.get("/api/symbols")
def get_symbols():
    return {"symbols": SYMBOLS}


# ============================================================
# PIPELINE (Full-Auto)
# ============================================================

@app.post("/api/pipeline/start")
def pipeline_start(req: PipelineStartRequest, background_tasks: BackgroundTasks):
    from pipeline.orchestrator import start_pipeline, is_running, set_event_callback
    if is_running():
        return {"error": "Pipeline zaten çalışıyor"}

    set_event_callback(ws_manager.sync_broadcast)
    background_tasks.add_task(start_pipeline, req.pairs)
    return {"status": "started", "pairs": len(req.pairs)}


@app.post("/api/pipeline/stop")
def pipeline_stop():
    from pipeline.orchestrator import stop_pipeline
    return stop_pipeline()


@app.get("/api/pipeline/status")
def pipeline_status():
    from pipeline.orchestrator import is_running
    from pipeline.queue_manager import get_status
    status = get_status()
    status["pipeline_running"] = is_running()
    return status


@app.post("/api/pipeline/add")
def pipeline_add(req: PipelineAddRequest):
    from pipeline.queue_manager import add_to_queue
    job = add_to_queue(req.symbol, req.strategy, req.timeframes)
    return {"status": "added", "job": job}


@app.delete("/api/pipeline/remove/{job_id}")
def pipeline_remove(job_id: str):
    from pipeline.queue_manager import remove_from_queue
    success = remove_from_queue(job_id)
    if success:
        return {"status": "removed"}
    return {"error": "Job bulunamadı veya çalışıyor"}


@app.get("/api/pipeline/history")
def pipeline_history():
    from pipeline.state_manager import get_completed_jobs
    return {"results": get_completed_jobs()}


# ============================================================
# SETTINGS
# ============================================================

@app.get("/api/settings")
def get_settings():
    from pipeline.state_manager import get_settings
    return get_settings()


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    from pipeline.state_manager import update_settings
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    return update_settings(updates)


# ============================================================
# DATA STATUS & DOWNLOAD
# ============================================================

@app.get("/api/data/status")
def data_status():
    from data.manager import get_data_status
    return {"data": get_data_status(SYMBOLS)}


@app.post("/api/data/download/{symbol}")
def start_download(symbol: str, background_tasks: BackgroundTasks):
    symbol = symbol.upper()
    if download_state["running"]:
        return {"error": "Bir indirme zaten devam ediyor"}

    download_state["running"] = True
    download_state["progress"] = 0
    download_state["symbol"] = symbol
    download_state["month"] = ""
    background_tasks.add_task(_run_download, symbol)
    return {"status": "started", "symbol": symbol}


def _run_download(symbol: str):
    try:
        from download_aggtrades import download_symbol, merge_symbol, get_months, MONTHS_BACK
        months = get_months(MONTHS_BACK)
        total_months = len(months)

        def on_month_progress(month_index: int, year_month: str):
            download_state["progress"] = int((month_index + 1) / total_months * 80)
            download_state["month"] = year_month

        download_symbol(symbol, on_progress=on_month_progress)
        download_state["progress"] = 85
        download_state["month"] = "Birleştiriliyor..."
        merge_symbol(symbol)
        download_state["progress"] = 100
        download_state["month"] = "Tamamlandı"
    except Exception as e:
        download_state["progress"] = -1
        download_state["month"] = str(e)
    finally:
        download_state["running"] = False


@app.get("/api/data/download/progress")
def download_progress():
    return download_state


# ============================================================
# BACKTEST (Manual — her iki strateji)
# ============================================================

@app.post("/api/backtest/run")
def run_backtest(params: BacktestParams, background_tasks: BackgroundTasks):
    if backtest_state["running"]:
        return {"error": "Bir backtest zaten çalışıyor"}

    backtest_state["running"] = True
    backtest_state["progress"] = 0
    backtest_state["result"] = None
    backtest_state["symbol"] = params.symbol
    background_tasks.add_task(_run_backtest, params)
    return {"status": "started", "symbol": params.symbol}


def _run_backtest(params: BacktestParams):
    try:
        if params.strategy == "swinginess":
            _run_swinginess_backtest(params)
        elif params.strategy == "pmax_kc":
            _run_pmax_backtest(params)
        else:
            backtest_state["result"] = {"error": f"Bilinmeyen strateji: {params.strategy}"}
    except Exception as e:
        backtest_state["result"] = {"error": str(e)}
    finally:
        backtest_state["running"] = False


def _run_swinginess_backtest(params: BacktestParams):
    """Swinginess tick-replay backtest."""
    from strategies.swinginess.tick_engine import TickReplayEngine
    from strategies.swinginess.strategy import SwingingessStrategy

    symbol = params.symbol.upper()
    parquet_path = os.path.join(DATA_DIR, f"{symbol.lower()}_aggtrades.parquet")

    if not os.path.exists(parquet_path):
        backtest_state["result"] = {"error": f"Veri bulunamadı: {symbol}"}
        return

    df = pd.read_parquet(parquet_path)
    n = len(df)

    p = params.model_dump()
    p.pop("symbol", None)
    p.pop("strategy", None)

    rolling_window = p.get("rolling_window_sec", 3600)
    engine = TickReplayEngine(rolling_window_sec=rolling_window)
    strategy = SwingingessStrategy(p)

    ts_arr = df["timestamp"].values
    price_arr = df["price"].values
    qty_arr = df["quantity"].values
    side_arr = df["side"].values

    last_bucket_ts = 0
    processed = 0
    equity_curve = []

    for i in range(n):
        ts_ms = int(ts_arr[i])
        price = float(price_arr[i])
        qty = float(qty_arr[i])
        side = str(side_arr[i])

        engine.process_tick(ts_ms, price, qty, side)

        ts_sec = ts_ms // 1000
        if ts_sec == last_bucket_ts:
            continue
        last_bucket_ts = ts_sec
        processed += 1

        if not engine.warmup_done:
            continue

        strategy.on_second(ts_sec, price, engine)

        if strategy.equity <= 0:
            break

        if processed % 300 == 0:
            equity_curve.append({"ts": ts_sec, "equity": round(strategy.equity, 2)})

        if processed % 10000 == 0:
            backtest_state["progress"] = min(99, int(i / n * 100))

    if strategy.position != 0:
        strategy._close_position(float(price_arr[-1]), int(ts_arr[-1]) // 1000, "CLOSE")

    results = strategy.get_results()
    trades = []
    for idx, t in enumerate(strategy.trades):
        trades.append({
            "id": idx + 1,
            "pnl": round(t["pnl"], 2),
            "pnl_pct": round(t["pnl_pct"], 2),
            "type": t["type"],
            "hold_sec": t["hold"],
        })

    if equity_curve and strategy.equity != equity_curve[-1]["equity"]:
        equity_curve.append({"ts": last_bucket_ts, "equity": round(strategy.equity, 2)})

    backtest_state["result"] = {
        "symbol": symbol,
        "strategy": "swinginess",
        "metrics": results,
        "equity_curve": equity_curve,
        "trades": trades,
        "params": p,
    }
    backtest_state["progress"] = 100


def _run_pmax_backtest(params: BacktestParams):
    """PMax+KC backtest (placeholder — full implementation in optimizer)."""
    backtest_state["result"] = {
        "error": "PMax+KC backtest henüz manuel modda desteklenmiyor. Pipeline kullanın."
    }
    backtest_state["progress"] = 100


@app.get("/api/backtest/progress")
def backtest_progress():
    return {
        "running": backtest_state["running"],
        "progress": backtest_state["progress"],
        "symbol": backtest_state["symbol"],
    }


@app.get("/api/backtest/results")
def backtest_results():
    return backtest_state["result"] or {"error": "Henüz backtest çalıştırılmadı"}


# ============================================================
# OPTIMIZATION (Manual — legacy endpoint)
# ============================================================

@app.post("/api/optimize/start")
def start_optimization(req: OptimizeRequest, background_tasks: BackgroundTasks):
    if optimize_state["running"]:
        return {"error": "Bir optimizasyon zaten çalışıyor"}

    optimize_state["running"] = True
    optimize_state["progress"] = 0
    optimize_state["trial"] = 0
    optimize_state["total"] = req.n_trials
    optimize_state["symbol"] = req.symbol
    optimize_state["error"] = None
    background_tasks.add_task(_run_optimization, req.symbol, req.n_trials)
    return {"status": "started", "symbol": req.symbol}


def _run_optimization(symbol: str, n_trials: int):
    try:
        import optuna
        from strategies.swinginess.optimizer import load_data, objective, run_trial_from_bars, pre_aggregate_bars

        os.makedirs(RESULTS_DIR, exist_ok=True)
        in_sample, out_sample = load_data(symbol)
        in_bars = pre_aggregate_bars(in_sample)

        study = optuna.create_study(direction="maximize")

        def callback(study, trial):
            optimize_state["trial"] = len(study.trials)
            optimize_state["progress"] = min(99, int(len(study.trials) / n_trials * 100))

        study.optimize(
            lambda trial: objective(trial, in_bars),
            n_trials=n_trials,
            callbacks=[callback],
        )

        optimize_state["progress"] = 100
    except Exception as e:
        import traceback
        traceback.print_exc()
        optimize_state["progress"] = -1
        optimize_state["error"] = str(e)
    finally:
        optimize_state["running"] = False


@app.get("/api/optimize/progress")
def optimize_progress():
    return optimize_state


@app.post("/api/optimize/reset")
def optimize_reset():
    optimize_state.update({"running": False, "progress": 0, "trial": 0, "total": 0, "symbol": None, "error": None})
    return {"status": "reset"}


@app.get("/api/optimize/results/{symbol}")
def optimize_results(symbol: str):
    symbol = symbol.upper()
    results_path = os.path.join(RESULTS_DIR, f"{symbol.lower()}_results.json")
    if not os.path.exists(results_path):
        return {"error": f"{symbol} için optimizasyon sonucu bulunamadı", "results": []}
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {"symbol": symbol, "results": results}


# ============================================================
# RESULTS BROWSER
# ============================================================

@app.get("/api/results")
def get_all_results():
    """Tüm tamamlanan optimizasyon sonuçlarını döndür."""
    results = []
    results_dir = Path(RESULTS_DIR)

    # Pipeline completed jobs
    from pipeline.state_manager import get_completed_jobs
    for job in get_completed_jobs():
        results.append({
            "id": job["id"],
            "symbol": job["symbol"],
            "strategy": job["strategy"],
            "timeframe": job.get("current_timeframe", "3m"),
            "date": job.get("completed_at", ""),
            "source": "pipeline",
            "metrics": job.get("selected_metrics_per_round", {}),
            "final_result": job.get("final_result"),
        })

    # Legacy JSON results
    for f in results_dir.glob("*_full_optimization.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            results.append({
                "id": f.stem,
                "symbol": data.get("symbol", f.stem.split("_")[0]),
                "strategy": "pmax_kc",
                "timeframe": data.get("timeframe", "3m"),
                "date": "",
                "source": "legacy",
                "final_result": data,
            })
        except:
            pass

    return {"results": results}


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Backend başlatıldığında crash recovery kontrol et."""
    from pipeline.orchestrator import resume_on_startup, set_event_callback
    set_event_callback(ws_manager.sync_broadcast)
    resume_on_startup()


# ============================================================
# SERVE FRONTEND (production)
# ============================================================

frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8055)
