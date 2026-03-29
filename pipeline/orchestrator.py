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
_wf_optimizer_ref = [None]  # Walk-forward optimizer referansı (stop için)


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
    # Walk-forward optimizer varsa onu da durdur
    if _wf_optimizer_ref[0] is not None:
        try:
            _wf_optimizer_ref[0].stop()
        except Exception:
            pass
        _wf_optimizer_ref[0] = None
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
            # Unified optimizer: aggregate varsa direkt kullan
            agg = result.get("aggregate")
            if agg:
                ratio = agg.get("total_net", 0) / max(agg.get("max_dd", 1), 1)
            else:
                metrics = result.get("selected_metrics", {})
                last_round_key = max(metrics.keys()) if metrics else None
                ratio = metrics[last_round_key].get("ratio", 0) if last_round_key else 0
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

    elif strategy == "walkforward_pmax":
        from strategies.pmax_kc.walkforward import WalkForwardPMaxOptimizer

        optimizer = WalkForwardPMaxOptimizer(
            symbol=symbol, timeframe=timeframe, days=days,
            leverage=leverage, event_callback=on_event, n_jobs=n_jobs,
        )

        # Stop flag bağla
        _wf_optimizer_ref[0] = optimizer

        return optimizer.run()

    elif strategy == "unified_pmax":
        from strategies.pmax_kc.unified_wf import UnifiedWFOptimizer
        optimizer = UnifiedWFOptimizer(
            symbol=symbol, timeframe=timeframe, days=270,
            leverage=leverage, event_callback=on_event, n_jobs=n_jobs,
        )
        _wf_optimizer_ref[0] = optimizer
        return optimizer.run_pmax_optimization()

    elif strategy == "unified_kc":
        from strategies.pmax_kc.unified_wf import UnifiedWFOptimizer
        optimizer = UnifiedWFOptimizer(
            symbol=symbol, timeframe=timeframe, days=270,
            leverage=leverage, event_callback=on_event, n_jobs=n_jobs,
        )
        _wf_optimizer_ref[0] = optimizer
        return optimizer.run_kc_optimization()

    elif strategy == "unified_kelly":
        from strategies.pmax_kc.unified_wf import UnifiedWFOptimizer
        optimizer = UnifiedWFOptimizer(
            symbol=symbol, timeframe=timeframe, days=270,
            leverage=leverage, event_callback=on_event, n_jobs=n_jobs,
        )
        _wf_optimizer_ref[0] = optimizer
        return optimizer.run_kelly_optimization()

    elif strategy == "walkforward_kc":
        from strategies.pmax_kc.walkforward import WalkForwardKCOptimizer

        optimizer = WalkForwardKCOptimizer(
            symbol=symbol, timeframe=timeframe, days=days,
            leverage=leverage, event_callback=on_event, n_jobs=n_jobs,
        )
        _wf_optimizer_ref[0] = optimizer
        return optimizer.run()

    elif strategy == "swinginess":
        # Swinginess: single-pass optimization
        _emit("log", {"message": f"Swinginess optimizer henüz pipeline'a entegre değil", "level": "warn"})
        return None

    return None


# ============================================================
# Pipeline V2 — Interactive Step-Based
# ============================================================

_v2_thread: Optional[threading.Thread] = None


def is_v2_running() -> bool:
    """V2 pipeline step çalışıyor mu?"""
    return _v2_thread is not None and _v2_thread.is_alive()


def start_v2_step(step_key: str):
    """V2 pipeline'da belirli bir adımı başlat.

    State manager'dan locked_params okunur, ilgili optimizer çalıştırılır.
    Tamamlandığında top-10 state'e yazılır.
    """
    global _v2_thread

    if is_v2_running():
        return {"error": "Bir V2 adımı zaten çalışıyor"}

    # State'i hemen running'e çek — HTTP response'dan önce
    from pipeline.state_manager import load_v2_state, set_v2_step_running, PIPELINE_V2_STEPS
    step_index = next(
        (i for i, s in enumerate(PIPELINE_V2_STEPS) if s["key"] == step_key), None
    )
    if step_index is not None:
        set_v2_step_running(step_index)

    _stop_flag.clear()
    _v2_thread = threading.Thread(target=_run_v2_step, args=(step_key,), daemon=True)
    _v2_thread.start()
    return {"status": "started", "step": step_key}


def stop_v2_step():
    """V2 adımını durdur."""
    _stop_flag.set()
    if _wf_optimizer_ref[0] is not None:
        try:
            _wf_optimizer_ref[0].stop()
        except Exception:
            pass
        _wf_optimizer_ref[0] = None
    return {"status": "stopping"}


def _run_v2_step(step_key: str):
    """V2 pipeline tek adım çalıştırıcı."""
    from pipeline.state_manager import (
        load_v2_state, set_v2_step_running, set_v2_step_results,
        PIPELINE_V2_STEPS,
    )

    v2 = load_v2_state()
    symbol = v2["symbol"]
    timeframe = v2.get("timeframe", "3m")
    n_trials = v2.get("n_trials", 1000)
    locked = v2.get("locked_params", {})
    settings = state_manager.get_settings()
    n_jobs = settings.get("n_jobs", 4)

    # Hangi adım?
    step_index = next(
        (i for i, s in enumerate(PIPELINE_V2_STEPS) if s["key"] == step_key), None
    )
    if step_index is None:
        _emit("error", {"message": f"Bilinmeyen adım: {step_key}"})
        return

    # Not: set_v2_step_running zaten start_v2_step'te yapıldı (race condition önleme)
    step_info = PIPELINE_V2_STEPS[step_index]
    _emit("v2_step_started", {"step": step_key, "label": step_info["label"]})
    _emit("log", {"message": f"V2 Adım başlıyor: {step_info['label']}", "level": "info"})

    try:
        from strategies.pmax_kc.unified_wf import UnifiedWFOptimizer

        def on_event(event_type, data):
            _emit(event_type, data)

        optimizer = UnifiedWFOptimizer(
            symbol=symbol, timeframe=timeframe, days=270,
            leverage=settings.get("leverage", 25),
            event_callback=on_event, n_jobs=n_jobs,
            n_trials=n_trials,
        )

        _wf_optimizer_ref[0] = optimizer

        if step_key == "pmax_discovery":
            result = optimizer.run_pmax_optimization()

        elif step_key == "kc_optimize":
            pmax_params = locked.get("pmax")
            if not pmax_params:
                _emit("error", {"message": "PMax parametreleri bulunamadı — önce PMax adımını tamamlayın"})
                return
            result = optimizer.run_kc_optimization(locked_pmax_params=pmax_params)

        elif step_key == "kelly_dyncomp":
            pmax_params = locked.get("pmax")
            kc_params = locked.get("kc")
            if not pmax_params or not kc_params:
                _emit("error", {"message": "PMax/KC parametreleri bulunamadı — önceki adımları tamamlayın"})
                return
            result = optimizer.run_kelly_optimization(
                locked_pmax_params=pmax_params,
                locked_kc_params=kc_params,
            )

        elif step_key == "dynsl_test":
            pmax_params = locked.get("pmax")
            kc_params = locked.get("kc")
            kelly_params = locked.get("kelly")
            if not pmax_params or not kc_params:
                _emit("error", {"message": "PMax/KC parametreleri bulunamadı — önceki adımları tamamlayın"})
                return
            result = _run_dynsl_grid(symbol, timeframe, pmax_params, kc_params,
                                     kelly_params, settings, on_event)
        else:
            _emit("error", {"message": f"Bilinmeyen strateji: {step_key}"})
            return

        _wf_optimizer_ref[0] = None

        if result and "error" not in result:
            top10 = result.get("top10", [])
            if top10:
                set_v2_step_results(step_key, top10)
                _emit("v2_step_completed", {
                    "step": step_key,
                    "top10": top10,
                    "aggregate": result.get("aggregate"),
                })
                _emit("log", {"message": f"V2 {step_info['label']} tamamlandı — top-10 hazır, seçim bekleniyor", "level": "success"})
            else:
                _emit("error", {"message": f"{step_info['label']} tamamlandı ama top-10 üretilemedi"})
        else:
            error_msg = result.get("error", "Bilinmeyen hata") if result else "Sonuç yok"
            _emit("error", {"message": f"{step_info['label']} başarısız: {error_msg}"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        _emit("error", {"message": f"V2 adım hatası: {e}"})
    finally:
        _wf_optimizer_ref[0] = None


def _run_dynsl_grid(symbol, timeframe, pmax_params, kc_params, kelly_params,
                    settings, on_event) -> dict:
    """DynSL grid search — hard_stop yüzdeleri dener."""
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from strategies.pmax_kc.backtest import _get_source
    from strategies.pmax_kc.config import DEFAULT_PARAMS

    try:
        import rust_engine
    except ImportError:
        return {"error": "Rust engine bulunamadı"}

    data_path = Path(f"data/{symbol}_{timeframe}_270d.parquet")
    if not data_path.exists():
        return {"error": f"Veri dosyası bulunamadı: {data_path}"}

    df = pd.read_parquet(data_path)
    bars_per_day = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96}.get(timeframe, 480)
    train_bars = 90 * bars_per_day
    test_df = df.iloc[train_bars:].reset_index(drop=True)

    sa = np.ascontiguousarray
    src = sa(_get_source(test_df, DEFAULT_PARAMS.get("source", "hl2")).values, dtype=np.float64)
    h = sa(test_df["high"].values, dtype=np.float64)
    l = sa(test_df["low"].values, dtype=np.float64)
    c = sa(test_df["close"].values, dtype=np.float64)

    # PMax compute
    atr_period = DEFAULT_PARAMS.get("atr_period", 10)
    atr_mult = DEFAULT_PARAMS.get("atr_multiplier", 3.0)
    ma_length = DEFAULT_PARAMS.get("ma_length", 10)

    pmax_result = rust_engine.compute_adaptive_pmax(
        src, h, l, c,
        atr_period, atr_mult, ma_length,
        pmax_params["vol_lookback"], pmax_params["flip_window"],
        pmax_params["mult_base"], pmax_params["mult_scale"],
        pmax_params["ma_base"], pmax_params["ma_scale"],
        pmax_params["atr_base"], pmax_params["atr_scale"],
        pmax_params["update_interval"],
    )
    pi = sa(pmax_result["pmax_line"], dtype=np.float64)
    mi = sa(pmax_result["mavg"], dtype=np.float64)
    di = sa(pmax_result["direction"], dtype=np.float64)

    ind = rust_engine.precompute_indicators(
        h, l, c,
        DEFAULT_PARAMS.get("ema_filter_period", 144),
        kc_params["kc_length"], kc_params["kc_multiplier"], kc_params["kc_atr_period"],
    )

    # Grid search: hard_stop 0.5% — 5.0% (0.25 adım)
    hs_values = [round(x * 0.25, 2) for x in range(2, 21)]  # 0.5 — 5.0
    results = []

    on_event("log", {"message": f"DynSL grid search: {len(hs_values)} hard_stop değeri test ediliyor", "level": "info"})
    on_event("log", {"message": "NOT: Rust engine hard_stop parametresini henüz dışarıdan almıyor — sabit 2.5% kullanılıyor. Sonuçlar referans amaçlıdır.", "level": "warn"})

    for i, hs in enumerate(hs_values):
        if _stop_flag.is_set():
            break

        if kelly_params:
            r = rust_engine.run_backtest_dynamic(
                c, h, l, pi, mi, di,
                sa(ind["rsi_vals"]), sa(ind["ema_filter"]),
                sa(ind["rsi_ema_vals"]), sa(ind["atr_vol"]),
                sa(ind["kc_upper_arr"]), sa(ind["kc_lower_arr"]),
                DEFAULT_PARAMS.get("rsi_overbought", 65.0),
                kc_params["max_dca_steps"], kc_params["tp_close_percent"],
                kelly_params["base_margin_pct"],
                kelly_params["tier1_threshold"], kelly_params["tier1_pct"],
                kelly_params["tier2_threshold"], kelly_params["tier2_pct"],
            )
        else:
            r = rust_engine.run_backtest(
                c, h, l, pi, mi, di,
                sa(ind["rsi_vals"]), sa(ind["ema_filter"]),
                sa(ind["rsi_ema_vals"]), sa(ind["atr_vol"]),
                sa(ind["kc_upper_arr"]), sa(ind["kc_lower_arr"]),
                DEFAULT_PARAMS.get("rsi_overbought", 65.0),
                kc_params["max_dca_steps"], kc_params["tp_close_percent"],
            )

        score = r["balance"] / max(r["max_drawdown"], 5.0)
        results.append({
            "tid": i,
            "score": round(score, 4),
            "params": {"hard_stop": hs},
            "total_net": round(r["net_pct"], 2),
            "max_dd": round(r["max_drawdown"], 2),
            "avg_wr": round(r["win_rate"], 1),
            "total_trades": r["total_trades"],
            "balance": round(r["balance"], 2),
            "oos_net": round(r["net_pct"], 2),
            "dd": round(r["max_drawdown"], 2),
            "wr": round(r["win_rate"], 1),
            "is_net": round(r["net_pct"], 2),
        })

        on_event("trial_completed", {
            "fold": 0, "trial_num": i,
            "score": round(score, 4),
            "oos_net": round(r["net_pct"], 2),
            "dd": round(r["max_drawdown"], 2),
            "ratio": round(score, 1),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top10 = results[:10]

    # Aggregate hesapla (en iyi sonuçtan)
    best = top10[0] if top10 else {}
    aggregate = {
        "total_net": best.get("total_net", 0),
        "max_dd": best.get("max_dd", 0),
        "avg_wr": best.get("avg_wr", 0),
        "total_trades": best.get("total_trades", 0),
        "total_weeks": 0,
        "profitable_weeks": 0,
        "win_rate_weeks": 0,
        "avg_weekly_net": 0,
        "best_week_net": 0,
        "worst_week_net": 0,
    } if top10 else {}

    import json
    summary = {
        "symbol": symbol,
        "step": "dynsl",
        "top10": top10,
        "aggregate": aggregate,
        "total_trials": len(results),
    }
    path = Path("results") / f"{symbol}_unified_dynsl.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    on_event("walkforward_completed", {"summary": summary})
    return summary


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
