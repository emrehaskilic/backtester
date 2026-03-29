"""Unified Walk-Forward Optimizer — tek parametre seti, tum haftalarda test.

Her trial'da 12 haftanin HEPSINDE backtest calisir.
Skor = toplam net / max DD * tutarlilik.
Sonuc: 6 ayda tutarli buyume, minimum DD.

Rust engine ile ~48ms/trial, 1000 trial ~1 dakika.
"""

import json
import logging
import math
import sys
import time
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import optuna
import pandas as pd

from strategies.pmax_kc.config import (
    DEFAULT_PARAMS, SCALPER_BOT_PATH,
)
from data.downloader_klines import fetch_klines
from strategies.pmax_kc.backtest import _get_source

try:
    import rust_engine
    USE_RUST = True
except ImportError:
    USE_RUST = False

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("unified_wf")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Walk-Forward sabitleri
TRAIN_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7
TRIALS = 1000


class UnifiedWFOptimizer:
    """Tek parametre seti, tum haftalarda test.

    Her trial:
      1. Parametre onerisi (Optuna)
      2. Train penceresi (90 gun) backtest — pruning icin
      3. 12 test haftasinin HEPSINDE backtest
      4. Skor = toplam_net / max_dd * tutarlilik

    Sonuc: Tek PMax (veya KC) parametresi, 12 haftada test edilmis.
    """

    def __init__(self, symbol: str, timeframe: str = "3m", days: int = 270,
                 leverage: int = 25, event_callback: Optional[Callable] = None,
                 n_jobs: int = 6, n_trials: int = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.days = days
        self.leverage = leverage
        self.event_callback = event_callback
        self.n_jobs = n_jobs
        self.n_trials = n_trials or TRIALS
        self.running = True
        self._params = DEFAULT_PARAMS.copy()
        self._log_buffer = []
        self._lock = threading.Lock()

        self.live_state = {
            "is_walkforward": True,
            "running": True,
            "symbol": symbol,
            "current_fold": 0,
            "total_folds": 0,
            "completed_trials": 0,
            "total_trials": self.n_trials,
            "best_oos_net": 0,
            "best_dd": 0,
            "best_ratio": 0,
            "fold_results": [],
        }

    def emit(self, event_type: str, data: dict):
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception:
                pass

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self._log_buffer.append(f"[{ts}] {msg}")
            if len(self._log_buffer) > 500:
                self._log_buffer = self._log_buffer[-500:]
        logger.info(msg)
        self.emit("log", {"message": msg, "level": "info"})

    def _compute_pmax(self, src, h, l, c, pmax_params):
        """Rust ile adaptive PMax hesapla."""
        atr_period = self._params.get("atr_period", 10)
        atr_mult = self._params.get("atr_multiplier", 3.0)
        ma_length = self._params.get("ma_length", 10)

        sa = np.ascontiguousarray
        result = rust_engine.compute_adaptive_pmax(
            sa(src.values if hasattr(src, 'values') else src, dtype=np.float64),
            sa(h.values if hasattr(h, 'values') else h, dtype=np.float64),
            sa(l.values if hasattr(l, 'values') else l, dtype=np.float64),
            sa(c.values if hasattr(c, 'values') else c, dtype=np.float64),
            atr_period, atr_mult, ma_length,
            pmax_params["vol_lookback"], pmax_params["flip_window"],
            pmax_params["mult_base"], pmax_params["mult_scale"],
            pmax_params["ma_base"], pmax_params["ma_scale"],
            pmax_params["atr_base"], pmax_params["atr_scale"],
            pmax_params["update_interval"],
        )
        return (
            np.ascontiguousarray(result["pmax_line"]),
            np.ascontiguousarray(result["mavg"]),
            np.ascontiguousarray(result["direction"]),
        )

    def _build_windows(self, df: pd.DataFrame) -> list[dict]:
        """Train + test pencerelerini olustur."""
        bars_per_day = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96}.get(self.timeframe, 480)
        train_bars = TRAIN_DAYS * bars_per_day
        test_bars = TEST_DAYS * bars_per_day
        step_bars = STEP_DAYS * bars_per_day
        total = len(df)

        windows = []
        num = 1
        start = 0

        while start + train_bars + test_bars <= total:
            train_end = start + train_bars
            test_end = train_end + test_bars
            windows.append({
                "num": num,
                "train_df": df.iloc[start:train_end].reset_index(drop=True),
                "test_df": df.iloc[train_end:min(test_end, total)].reset_index(drop=True),
            })
            num += 1
            start += step_bars

        return windows

    def _precompute_window(self, df, kc_length=20, kc_multiplier=1.5, kc_atr_period=10):
        """Bir pencere icin sabit indikatorleri pre-compute et."""
        sa = np.ascontiguousarray
        h = sa(df["high"].values, dtype=np.float64)
        l = sa(df["low"].values, dtype=np.float64)
        c = sa(df["close"].values, dtype=np.float64)
        ind = rust_engine.precompute_indicators(
            h, l, c,
            self._params.get("ema_filter_period", 144),
            kc_length, kc_multiplier, kc_atr_period,
        )
        return {
            "high": h, "low": l, "close": c,
            "src": sa(_get_source(df, self._params.get("source", "hl2")).values, dtype=np.float64),
            **ind,
        }

    def _run_backtest(self, pre, pi, mi, di, max_dca=2, tp_pct=0.20):
        """Rust backtest calistir."""
        sa = np.ascontiguousarray
        return rust_engine.run_backtest(
            pre["close"], pre["high"], pre["low"],
            sa(pi, dtype=np.float64), sa(mi, dtype=np.float64), sa(di, dtype=np.float64),
            sa(pre["rsi_vals"]), sa(pre["ema_filter"]),
            sa(pre["rsi_ema_vals"]), sa(pre["atr_vol"]),
            sa(pre["kc_upper_arr"]), sa(pre["kc_lower_arr"]),
            self._params.get("rsi_overbought", 65.0),
            max_dca, tp_pct,
        )

    def run_pmax_optimization(self) -> dict:
        """ADIM 1: PMax parametrelerini optimize et — tum haftalarda."""
        self.running = True
        start_time = time.time()

        try:
            self._log(f"=== UNIFIED PMax OPTIMIZATION: {self.symbol} ===")

            # Veri yukle
            data_path = Path(f"data/{self.symbol}_{self.timeframe}_{self.days}d.parquet")
            if data_path.exists():
                df = pd.read_parquet(data_path)
                self._log(f"Cache: {len(df)} mum")
            else:
                df = fetch_klines(self.symbol, self.timeframe, self.days)
                df.to_parquet(data_path, index=False)

            windows = self._build_windows(df)
            n_weeks = len(windows)
            self._log(f"{n_weeks} hafta olusturuldu, {TRIALS} trial basliyor")

            # Her haftanin test verisini pre-compute et
            test_caches = []
            for w in windows:
                test_caches.append(self._precompute_window(w["test_df"]))

            # Ilk pencere icin train pre-compute (pruning icin)
            train_cache = self._precompute_window(windows[0]["train_df"])

            self.live_state.update({
                "total_folds": n_weeks,
                "total_trials": self.n_trials,
                "completed_trials": 0,
            })
            self.emit("walkforward_started", {
                "symbol": self.symbol,
                "total_folds": n_weeks,
                "train_days": TRAIN_DAYS,
                "test_days": TEST_DAYS,
                "step_days": STEP_DAYS,
                "trials_per_fold": self.n_trials,
            })

            # Optuna
            sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=30)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            trials_data = []
            trial_lock = threading.Lock()
            best_score_ever = [float("-inf")]

            def objective(trial):
                cp = {
                    "vol_lookback": trial.suggest_int("vol_lookback", 60, 1440, step=20),
                    "flip_window": trial.suggest_int("flip_window", 20, 480, step=10),
                    "mult_base": trial.suggest_float("mult_base", 0.5, 4.0, step=0.25),
                    "mult_scale": trial.suggest_float("mult_scale", 0.25, 3.0, step=0.25),
                    "ma_base": trial.suggest_int("ma_base", 3, 20),
                    "ma_scale": trial.suggest_float("ma_scale", 0.5, 6.0, step=0.5),
                    "atr_base": trial.suggest_int("atr_base", 3, 20),
                    "atr_scale": trial.suggest_float("atr_scale", 0.5, 4.0, step=0.5),
                    "update_interval": trial.suggest_int("update_interval", 1, 60, step=1),
                }
                try:
                    # Hizli pruning: ilk train penceresinde kontrol
                    pi, mi, di = self._compute_pmax(
                        train_cache["src"], train_cache["high"],
                        train_cache["low"], train_cache["close"], cp)
                    train_r = self._run_backtest(train_cache, pi, mi, di)
                    if train_r["total_trades"] < 5:
                        return -999

                    trial.report(train_r["net_pct"], step=0)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    # 12 haftanin HEPSINDE test
                    week_nets = []
                    week_dds = []
                    week_wrs = []
                    week_trades = []

                    for wi, tc in enumerate(test_caches):
                        pi_t, mi_t, di_t = self._compute_pmax(
                            tc["src"], tc["high"], tc["low"], tc["close"], cp)
                        r = self._run_backtest(tc, pi_t, mi_t, di_t)
                        week_nets.append(r["net_pct"])
                        week_dds.append(r["max_drawdown"])
                        week_wrs.append(r["win_rate"])
                        week_trades.append(r["total_trades"])

                    total_net = sum(week_nets)
                    max_dd = max(week_dds) if week_dds else 100
                    profitable = sum(1 for n in week_nets if n > 0)
                    consistency = profitable / n_weeks if n_weeks > 0 else 0
                    avg_wr = np.mean(week_wrs)

                    # Skor: toplam getiri / risk * tutarlilik
                    score = (total_net / max(max_dd, 5.0)) * consistency * math.sqrt(avg_wr / 100.0)

                    trial.set_user_attr("total_net", round(total_net, 2))
                    trial.set_user_attr("max_dd", round(max_dd, 2))
                    trial.set_user_attr("profitable_weeks", profitable)
                    trial.set_user_attr("avg_wr", round(avg_wr, 1))
                    trial.set_user_attr("week_nets", [round(n, 2) for n in week_nets])
                    trial.set_user_attr("worst_week", round(min(week_nets), 2))
                    trial.set_user_attr("best_week", round(max(week_nets), 2))
                    return score

                except optuna.TrialPruned:
                    raise
                except Exception:
                    return -999

            def callback(study, trial):
                if not self.running:
                    study.stop()
                    return
                if trial.value is not None and trial.value > -999 and "total_net" in trial.user_attrs:
                    a = trial.user_attrs
                    td = {
                        "tid": trial.number,
                        "score": round(trial.value, 4),
                        "total_net": a["total_net"],
                        "max_dd": a["max_dd"],
                        "profitable_weeks": a["profitable_weeks"],
                        "avg_wr": a["avg_wr"],
                        "worst_week": a["worst_week"],
                        "best_week": a["best_week"],
                        "week_nets": a["week_nets"],
                        "params": trial.params,
                        # Uyumluluk icin
                        "is_net": a["total_net"],
                        "oos_net": a["total_net"],
                        "dd": a["max_dd"],
                        "wr": a["avg_wr"],
                        "trades": a["profitable_weeks"],
                    }
                    with trial_lock:
                        trials_data.append(td)
                        n_valid = len(trials_data)
                        self.live_state["completed_trials"] = n_valid
                        if td["total_net"] > self.live_state["best_oos_net"]:
                            self.live_state["best_oos_net"] = td["total_net"]
                        if td["max_dd"] > 0 and (self.live_state["best_dd"] == 0 or td["max_dd"] < self.live_state["best_dd"]):
                            self.live_state["best_dd"] = td["max_dd"]
                        ratio = round(td["total_net"] / max(td["max_dd"], 1), 1)
                        if ratio > self.live_state["best_ratio"]:
                            self.live_state["best_ratio"] = ratio

                    self.emit("trial_completed", {
                        "fold": 0,
                        "trial_num": trial.number,
                        "score": td["score"],
                        "oos_net": td["total_net"],
                        "dd": td["max_dd"],
                        "ratio": ratio,
                    })

                    # Yeni en iyi bulunduysa log
                    if trial.value > best_score_ever[0]:
                        best_score_ever[0] = trial.value
                        self._log(
                            f"YENi EN IYI #{trial.number}: "
                            f"Net={td['total_net']:+.1f}% DD={td['max_dd']:.1f}% "
                            f"Karli={td['profitable_weeks']}/{n_weeks} "
                            f"WR={td['avg_wr']:.0f}% "
                            f"En kotu hafta={td['worst_week']:+.1f}%"
                        )

            # Optimize
            study.optimize(objective, n_trials=self.n_trials, callbacks=[callback],
                           n_jobs=self.n_jobs)

            if not trials_data:
                return {"error": "Hicbir trial basarili olmadi"}

            # Top-10 sıralama (score'a göre)
            sorted_trials = sorted(trials_data, key=lambda x: x["score"], reverse=True)
            top10 = sorted_trials[:10]

            # En iyi sonuc
            best = top10[0]
            best_params = best["params"]

            # Her top-10 entry icin haftalik detay hesapla
            for entry in top10:
                entry_folds = []
                for wi, w in enumerate(windows):
                    tc = test_caches[wi]
                    pi, mi, di = self._compute_pmax(
                        tc["src"], tc["high"], tc["low"], tc["close"], entry["params"])
                    r = self._run_backtest(tc, pi, mi, di)
                    test_start = pd.to_datetime(
                        w["test_df"]["open_time"].iloc[0], unit="ms").strftime("%Y-%m-%d")
                    test_end = pd.to_datetime(
                        w["test_df"]["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
                    entry_folds.append({
                        "fold": wi + 1,
                        "test_period": f"{test_start} -> {test_end}",
                        "test_net": round(r["net_pct"], 2),
                        "test_dd": round(r["max_drawdown"], 2),
                        "test_wr": round(r["win_rate"], 2),
                        "test_trades": r["total_trades"],
                    })
                entry["folds"] = entry_folds

            # En iyi parametrenin fold_results'i (ana tablo icin)
            fold_results = top10[0]["folds"]

            self.live_state["fold_results"] = fold_results
            elapsed = time.time() - start_time

            # Aggregate (en iyi parametreden)
            nets = [f["test_net"] for f in fold_results]
            dds = [f["test_dd"] for f in fold_results]
            profitable = sum(1 for n in nets if n > 0)

            summary = {
                "symbol": self.symbol,
                "step": "pmax",
                "best_params": best_params,
                "best_score": best["score"],
                "top10": top10,
                "aggregate": {
                    "total_weeks": n_weeks,
                    "profitable_weeks": profitable,
                    "win_rate_weeks": round(profitable / n_weeks * 100, 1),
                    "total_net": round(sum(nets), 2),
                    "avg_weekly_net": round(np.mean(nets), 2),
                    "best_week_net": round(max(nets), 2),
                    "worst_week_net": round(min(nets), 2),
                    "max_dd": round(max(dds), 2),
                    "avg_dd": round(np.mean(dds), 2),
                    "avg_wr": round(np.mean([f["test_wr"] for f in fold_results]), 1),
                    "total_trades": sum(f["test_trades"] for f in fold_results),
                },
                "folds": fold_results,
                "total_trials": len(trials_data),
                "elapsed_seconds": round(elapsed, 1),
            }

            # Kaydet
            path = RESULTS_DIR / f"{self.symbol}_unified_pmax.json"
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)

            self._log(f"=== PMax TAMAMLANDI ({elapsed:.0f}s) ===")
            self._log(
                f"Sonuc: {profitable}/{n_weeks} hafta karli, "
                f"Toplam: {sum(nets):+.1f}%, Max DD: {max(dds):.1f}%"
            )
            self.emit("walkforward_completed", {"summary": summary})
            return summary

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"HATA: {e}")
            self.emit("error", {"message": str(e)})
            return {"error": str(e)}
        finally:
            self.running = False

    def run_kc_optimization(self, locked_pmax_params: dict = None) -> dict:
        """ADIM 2: KC parametrelerini optimize et — PMax kilitli, tum haftalarda."""
        self.running = True
        start_time = time.time()

        try:
            # PMax parametrelerini yukle
            if locked_pmax_params is None:
                pmax_path = RESULTS_DIR / f"{self.symbol}_unified_pmax.json"
                if not pmax_path.exists():
                    return {"error": "Once PMax optimizasyonu calistirilmali"}
                with open(pmax_path) as f:
                    pmax_data = json.load(f)
                locked_pmax_params = pmax_data["best_params"]

            self._log(f"=== UNIFIED KC OPTIMIZATION: {self.symbol} ===")
            self._log(f"PMax kilitli: {json.dumps(locked_pmax_params)}")

            # Veri yukle
            data_path = Path(f"data/{self.symbol}_{self.timeframe}_{self.days}d.parquet")
            df = pd.read_parquet(data_path)

            windows = self._build_windows(df)
            n_weeks = len(windows)

            # Her hafta icin PMax pre-compute (1 kez, kilitli)
            test_pmax_caches = []
            for w in windows:
                tc = w["test_df"]
                src = _get_source(tc, self._params.get("source", "hl2"))
                pi, mi, di = self._compute_pmax(src, tc["high"], tc["low"], tc["close"], locked_pmax_params)
                test_pmax_caches.append({"pi": pi, "mi": mi, "di": di, "df": tc})

            # Train icin de PMax pre-compute (pruning icin)
            train_df = windows[0]["train_df"]
            train_src = _get_source(train_df, self._params.get("source", "hl2"))
            train_pi, train_mi, train_di = self._compute_pmax(
                train_src, train_df["high"], train_df["low"], train_df["close"], locked_pmax_params)

            self.live_state.update({
                "total_folds": n_weeks,
                "total_trials": self.n_trials,
                "completed_trials": 0,
                "best_oos_net": 0, "best_dd": 0, "best_ratio": 0,
                "fold_results": [],
            })
            self.emit("walkforward_started", {
                "symbol": self.symbol,
                "total_folds": n_weeks,
                "train_days": TRAIN_DAYS,
                "test_days": TEST_DAYS,
                "step_days": STEP_DAYS,
                "trials_per_fold": self.n_trials,
            })

            sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=30)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            trials_data = []
            trial_lock = threading.Lock()
            best_score_ever = [float("-inf")]
            sa = np.ascontiguousarray

            def objective(trial):
                kc = {
                    "kc_length": trial.suggest_int("kc_length", 5, 50),
                    "kc_multiplier": trial.suggest_float("kc_multiplier", 0.5, 4.0, step=0.1),
                    "kc_atr_period": trial.suggest_int("kc_atr_period", 3, 30),
                    "max_dca_steps": trial.suggest_int("max_dca_steps", 1, 5),
                    "tp_close_percent": trial.suggest_float("tp_close_percent", 0.05, 0.50, step=0.05),
                }
                try:
                    # Pruning: train'de kontrol
                    train_ind = rust_engine.precompute_indicators(
                        sa(train_df["high"].values, dtype=np.float64),
                        sa(train_df["low"].values, dtype=np.float64),
                        sa(train_df["close"].values, dtype=np.float64),
                        self._params.get("ema_filter_period", 144),
                        kc["kc_length"], kc["kc_multiplier"], kc["kc_atr_period"],
                    )
                    train_r = rust_engine.run_backtest(
                        sa(train_df["close"].values, dtype=np.float64),
                        sa(train_df["high"].values, dtype=np.float64),
                        sa(train_df["low"].values, dtype=np.float64),
                        sa(train_pi, dtype=np.float64),
                        sa(train_mi, dtype=np.float64),
                        sa(train_di, dtype=np.float64),
                        sa(train_ind["rsi_vals"]), sa(train_ind["ema_filter"]),
                        sa(train_ind["rsi_ema_vals"]), sa(train_ind["atr_vol"]),
                        sa(train_ind["kc_upper_arr"]), sa(train_ind["kc_lower_arr"]),
                        self._params.get("rsi_overbought", 65.0),
                        kc["max_dca_steps"], kc["tp_close_percent"],
                    )
                    if train_r["total_trades"] < 5:
                        return -999
                    trial.report(train_r["net_pct"], step=0)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    # 12 haftada test
                    week_nets = []
                    week_dds = []
                    week_wrs = []

                    for wi, pc in enumerate(test_pmax_caches):
                        tc = pc["df"]
                        ind = rust_engine.precompute_indicators(
                            sa(tc["high"].values, dtype=np.float64),
                            sa(tc["low"].values, dtype=np.float64),
                            sa(tc["close"].values, dtype=np.float64),
                            self._params.get("ema_filter_period", 144),
                            kc["kc_length"], kc["kc_multiplier"], kc["kc_atr_period"],
                        )
                        r = rust_engine.run_backtest(
                            sa(tc["close"].values, dtype=np.float64),
                            sa(tc["high"].values, dtype=np.float64),
                            sa(tc["low"].values, dtype=np.float64),
                            sa(pc["pi"], dtype=np.float64),
                            sa(pc["mi"], dtype=np.float64),
                            sa(pc["di"], dtype=np.float64),
                            sa(ind["rsi_vals"]), sa(ind["ema_filter"]),
                            sa(ind["rsi_ema_vals"]), sa(ind["atr_vol"]),
                            sa(ind["kc_upper_arr"]), sa(ind["kc_lower_arr"]),
                            self._params.get("rsi_overbought", 65.0),
                            kc["max_dca_steps"], kc["tp_close_percent"],
                        )
                        week_nets.append(r["net_pct"])
                        week_dds.append(r["max_drawdown"])
                        week_wrs.append(r["win_rate"])

                    total_net = sum(week_nets)
                    max_dd = max(week_dds) if week_dds else 100
                    profitable = sum(1 for n in week_nets if n > 0)
                    consistency = profitable / n_weeks
                    avg_wr = np.mean(week_wrs)

                    score = (total_net / max(max_dd, 5.0)) * consistency * math.sqrt(avg_wr / 100.0)

                    trial.set_user_attr("total_net", round(total_net, 2))
                    trial.set_user_attr("max_dd", round(max_dd, 2))
                    trial.set_user_attr("profitable_weeks", profitable)
                    trial.set_user_attr("avg_wr", round(avg_wr, 1))
                    trial.set_user_attr("worst_week", round(min(week_nets), 2))
                    trial.set_user_attr("best_week", round(max(week_nets), 2))
                    return score

                except optuna.TrialPruned:
                    raise
                except Exception:
                    return -999

            def callback(study, trial):
                if not self.running:
                    study.stop()
                    return
                if trial.value is not None and trial.value > -999 and "total_net" in trial.user_attrs:
                    a = trial.user_attrs
                    td = {
                        "tid": trial.number,
                        "score": round(trial.value, 4),
                        "total_net": a["total_net"],
                        "max_dd": a["max_dd"],
                        "profitable_weeks": a["profitable_weeks"],
                        "avg_wr": a["avg_wr"],
                        "worst_week": a["worst_week"],
                        "best_week": a["best_week"],
                        "params": trial.params,
                        "oos_net": a["total_net"],
                        "dd": a["max_dd"],
                        "wr": a["avg_wr"],
                        "is_net": a["total_net"],
                    }
                    with trial_lock:
                        trials_data.append(td)
                        n_valid = len(trials_data)
                        self.live_state["completed_trials"] = n_valid
                        if td["total_net"] > self.live_state["best_oos_net"]:
                            self.live_state["best_oos_net"] = td["total_net"]
                        if td["max_dd"] > 0 and (self.live_state["best_dd"] == 0 or td["max_dd"] < self.live_state["best_dd"]):
                            self.live_state["best_dd"] = td["max_dd"]
                        ratio = round(td["total_net"] / max(td["max_dd"], 1), 1)
                        if ratio > self.live_state["best_ratio"]:
                            self.live_state["best_ratio"] = ratio

                    self.emit("trial_completed", {
                        "fold": 0, "trial_num": trial.number,
                        "score": td["score"], "oos_net": td["total_net"],
                        "dd": td["max_dd"], "ratio": ratio,
                    })

                    if trial.value > best_score_ever[0]:
                        best_score_ever[0] = trial.value
                        self._log(
                            f"YENi EN IYI #{trial.number}: "
                            f"Net={td['total_net']:+.1f}% DD={td['max_dd']:.1f}% "
                            f"Karli={td['profitable_weeks']}/{n_weeks} WR={td['avg_wr']:.0f}%"
                        )

            study.optimize(objective, n_trials=self.n_trials, callbacks=[callback],
                           n_jobs=self.n_jobs)

            if not trials_data:
                return {"error": "Hicbir trial basarili olmadi"}

            # Top-10 sıralama
            sorted_trials = sorted(trials_data, key=lambda x: x["score"], reverse=True)
            top10 = sorted_trials[:10]

            best = top10[0]
            best_kc = best["params"]

            # Her top-10 entry icin haftalik detay hesapla
            for entry in top10:
                kc_p = entry["params"]
                entry_folds = []
                for wi, pc in enumerate(test_pmax_caches):
                    tc = pc["df"]
                    ind = rust_engine.precompute_indicators(
                        sa(tc["high"].values, dtype=np.float64),
                        sa(tc["low"].values, dtype=np.float64),
                        sa(tc["close"].values, dtype=np.float64),
                        self._params.get("ema_filter_period", 144),
                        kc_p["kc_length"], kc_p["kc_multiplier"], kc_p["kc_atr_period"],
                    )
                    r = rust_engine.run_backtest(
                        sa(tc["close"].values, dtype=np.float64),
                        sa(tc["high"].values, dtype=np.float64),
                        sa(tc["low"].values, dtype=np.float64),
                        sa(pc["pi"], dtype=np.float64), sa(pc["mi"], dtype=np.float64),
                        sa(pc["di"], dtype=np.float64),
                        sa(ind["rsi_vals"]), sa(ind["ema_filter"]),
                        sa(ind["rsi_ema_vals"]), sa(ind["atr_vol"]),
                        sa(ind["kc_upper_arr"]), sa(ind["kc_lower_arr"]),
                        self._params.get("rsi_overbought", 65.0),
                        kc_p["max_dca_steps"], kc_p["tp_close_percent"],
                    )
                    w = windows[wi]
                    test_start = pd.to_datetime(
                        w["test_df"]["open_time"].iloc[0], unit="ms").strftime("%Y-%m-%d")
                    test_end = pd.to_datetime(
                        w["test_df"]["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
                    entry_folds.append({
                        "fold": wi + 1,
                        "test_period": f"{test_start} -> {test_end}",
                        "test_net": round(r["net_pct"], 2),
                        "test_dd": round(r["max_drawdown"], 2),
                        "test_wr": round(r["win_rate"], 2),
                        "test_trades": r["total_trades"],
                    })
                entry["folds"] = entry_folds

            fold_results = top10[0]["folds"]

            self.live_state["fold_results"] = fold_results
            elapsed = time.time() - start_time
            nets = [f["test_net"] for f in fold_results]
            dds = [f["test_dd"] for f in fold_results]
            profitable = sum(1 for n in nets if n > 0)

            summary = {
                "symbol": self.symbol,
                "step": "kc",
                "pmax_params": locked_pmax_params,
                "best_kc_params": best_kc,
                "top10": top10,
                "aggregate": {
                    "total_weeks": n_weeks,
                    "profitable_weeks": profitable,
                    "win_rate_weeks": round(profitable / n_weeks * 100, 1),
                    "total_net": round(sum(nets), 2),
                    "avg_weekly_net": round(np.mean(nets), 2),
                    "best_week_net": round(max(nets), 2),
                    "worst_week_net": round(min(nets), 2),
                    "max_dd": round(max(dds), 2),
                    "avg_dd": round(np.mean(dds), 2),
                    "avg_wr": round(np.mean([f["test_wr"] for f in fold_results]), 1),
                    "total_trades": sum(f["test_trades"] for f in fold_results),
                },
                "folds": fold_results,
                "elapsed_seconds": round(elapsed, 1),
            }

            path = RESULTS_DIR / f"{self.symbol}_unified_kc.json"
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)

            self._log(f"=== KC TAMAMLANDI ({elapsed:.0f}s) ===")
            self._log(f"Sonuc: {profitable}/{n_weeks} karli, Toplam: {sum(nets):+.1f}%")
            self.emit("walkforward_completed", {"summary": summary})
            return summary

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"HATA: {e}")
            self.emit("error", {"message": str(e)})
            return {"error": str(e)}
        finally:
            self.running = False

    def run_kelly_optimization(self, locked_pmax_params: dict = None,
                               locked_kc_params: dict = None) -> dict:
        """ADIM 3: Kelly/DynComp — PMax+KC kilitli, pozisyon buyuklugu optimize."""
        self.running = True
        start_time = time.time()

        try:
            # PMax + KC parametrelerini yukle (disaridan verilmediyse dosyadan)
            if locked_pmax_params is not None and locked_kc_params is not None:
                locked_pmax = locked_pmax_params
                locked_kc = locked_kc_params
            else:
                pmax_path = RESULTS_DIR / f"{self.symbol}_unified_pmax.json"
                kc_path = RESULTS_DIR / f"{self.symbol}_unified_kc.json"
                if not pmax_path.exists() or not kc_path.exists():
                    return {"error": "Once PMax ve KC optimizasyonu calistirilmali"}

                with open(pmax_path) as f:
                    pmax_data = json.load(f)
                with open(kc_path) as f:
                    kc_data = json.load(f)

                locked_pmax = pmax_data["best_params"] if locked_pmax_params is None else locked_pmax_params
                locked_kc = kc_data["best_kc_params"] if locked_kc_params is None else locked_kc_params

            self._log(f"=== UNIFIED KELLY/DYNCOMP: {self.symbol} ===")
            self._log(f"PMax kilitli: {json.dumps(locked_pmax)}")
            self._log(f"KC kilitli: {json.dumps(locked_kc)}")

            # Veri yukle — 270 gunun TEST kismini tek parca olarak kullan
            data_path = Path(f"data/{self.symbol}_{self.timeframe}_{self.days}d.parquet")
            df = pd.read_parquet(data_path)
            bars_per_day = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96}.get(self.timeframe, 480)
            train_bars = TRAIN_DAYS * bars_per_day

            # Test verisi: train sonrasi tum veri (180 gun = 6 ay)
            test_df = df.iloc[train_bars:].reset_index(drop=True)
            self._log(f"Test verisi: {len(test_df)} bar ({len(test_df) // bars_per_day} gun)")

            # PMax + KC + indicators pre-compute (1 kez — kilitli parametreler)
            sa = np.ascontiguousarray
            src = sa(_get_source(test_df, self._params.get("source", "hl2")).values, dtype=np.float64)
            h = sa(test_df["high"].values, dtype=np.float64)
            l = sa(test_df["low"].values, dtype=np.float64)
            c = sa(test_df["close"].values, dtype=np.float64)

            pi, mi, di = self._compute_pmax(src, h, l, c, locked_pmax)
            pi_c = sa(pi, dtype=np.float64)
            mi_c = sa(mi, dtype=np.float64)
            di_c = sa(di, dtype=np.float64)

            ind = rust_engine.precompute_indicators(
                h, l, c,
                self._params.get("ema_filter_period", 144),
                locked_kc["kc_length"], locked_kc["kc_multiplier"], locked_kc["kc_atr_period"],
            )
            ind_rsi = sa(ind["rsi_vals"])
            ind_ema = sa(ind["ema_filter"])
            ind_rsi_ema = sa(ind["rsi_ema_vals"])
            ind_atr = sa(ind["atr_vol"])
            ind_kc_u = sa(ind["kc_upper_arr"])
            ind_kc_l = sa(ind["kc_lower_arr"])

            n_weeks = (len(test_df) // bars_per_day) // 7
            self.live_state.update({
                "total_folds": n_weeks,
                "total_trials": self.n_trials,
                "completed_trials": 0,
                "best_oos_net": 0, "best_dd": 0, "best_ratio": 0,
                "fold_results": [],
            })
            self.emit("walkforward_started", {
                "symbol": self.symbol,
                "total_folds": n_weeks,
                "train_days": TRAIN_DAYS,
                "test_days": len(test_df) // bars_per_day,
                "step_days": 0,
                "trials_per_fold": self.n_trials,
            })

            sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=30)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            trials_data = []
            trial_lock = threading.Lock()
            best_score_ever = [float("-inf")]

            def objective(trial):
                kelly_params = {
                    "base_margin_pct": trial.suggest_float("base_margin_pct", 1.0, 10.0, step=0.5),
                    "tier1_threshold": trial.suggest_float("tier1_threshold", 20000, 50000, step=5000),
                    "tier1_pct": trial.suggest_float("tier1_pct", 0.5, 5.0, step=0.25),
                    "tier2_threshold": trial.suggest_float("tier2_threshold", 50000, 150000, step=10000),
                    "tier2_pct": trial.suggest_float("tier2_pct", 0.25, 3.0, step=0.25),
                }
                try:
                    r = rust_engine.run_backtest_dynamic(
                        c, h, l, pi_c, mi_c, di_c,
                        ind_rsi, ind_ema, ind_rsi_ema, ind_atr, ind_kc_u, ind_kc_l,
                        self._params.get("rsi_overbought", 65.0),
                        locked_kc["max_dca_steps"],
                        locked_kc["tp_close_percent"],
                        kelly_params["base_margin_pct"],
                        kelly_params["tier1_threshold"],
                        kelly_params["tier1_pct"],
                        kelly_params["tier2_threshold"],
                        kelly_params["tier2_pct"],
                    )

                    if r["total_trades"] < 10:
                        return -999

                    # DD siniri: %25'i gecen trial direkt cop
                    if r["max_drawdown"] > 25.0:
                        return -999

                    balance = r["balance"]
                    max_dd = max(r["max_drawdown"], 5.0)
                    if balance <= 0:
                        return -999

                    growth = balance / 10000.0
                    score = growth / max_dd * math.sqrt(r["win_rate"] / 100.0)

                    trial.set_user_attr("balance", round(balance, 2))
                    trial.set_user_attr("net_pct", round(r["net_pct"], 2))
                    trial.set_user_attr("max_dd", round(r["max_drawdown"], 2))
                    trial.set_user_attr("total_trades", r["total_trades"])
                    trial.set_user_attr("win_rate", round(r["win_rate"], 1))
                    return score

                except optuna.TrialPruned:
                    raise
                except Exception:
                    return -999

            def callback(study, trial):
                if not self.running:
                    study.stop()
                    return
                if trial.value is not None and trial.value > -999 and "balance" in trial.user_attrs:
                    a = trial.user_attrs
                    td = {
                        "tid": trial.number,
                        "score": round(trial.value, 4),
                        "balance": a["balance"],
                        "total_net": a["net_pct"],
                        "max_dd": a["max_dd"],
                        "total_trades": a["total_trades"],
                        "avg_wr": a["win_rate"],
                        "params": trial.params,
                        "oos_net": a["net_pct"],
                        "dd": a["max_dd"],
                        "wr": a["win_rate"],
                        "is_net": a["net_pct"],
                    }
                    with trial_lock:
                        trials_data.append(td)
                        n_valid = len(trials_data)
                        self.live_state["completed_trials"] = n_valid
                        if td["total_net"] > self.live_state["best_oos_net"]:
                            self.live_state["best_oos_net"] = td["total_net"]
                        if td["max_dd"] > 0 and (self.live_state["best_dd"] == 0 or td["max_dd"] < self.live_state["best_dd"]):
                            self.live_state["best_dd"] = td["max_dd"]
                        ratio = round(td["total_net"] / max(td["max_dd"], 1), 1)
                        if ratio > self.live_state["best_ratio"]:
                            self.live_state["best_ratio"] = ratio

                    self.emit("trial_completed", {
                        "fold": 0, "trial_num": trial.number,
                        "score": td["score"], "oos_net": td["total_net"],
                        "dd": td["max_dd"], "ratio": ratio,
                    })

                    if trial.value > best_score_ever[0]:
                        best_score_ever[0] = trial.value
                        self._log(
                            f"YENi EN IYI #{trial.number}: "
                            f"$10K -> ${td['balance']:,.0f} ({td['total_net']:+.1f}%) "
                            f"DD={td['max_dd']:.1f}% WR={td['avg_wr']:.0f}% "
                            f"Trades={td['total_trades']}"
                        )

            study.optimize(objective, n_trials=self.n_trials, callbacks=[callback],
                           n_jobs=self.n_jobs)

            if not trials_data:
                return {"error": "Hicbir trial basarili olmadi"}

            # Top-10 sıralama
            sorted_trials = sorted(trials_data, key=lambda x: x["score"], reverse=True)
            top10 = sorted_trials[:10]

            best = top10[0]
            best_kelly = best["params"]

            elapsed = time.time() - start_time
            summary = {
                "symbol": self.symbol,
                "step": "kelly",
                "pmax_params": locked_pmax,
                "kc_params": locked_kc,
                "best_kelly_params": best_kelly,
                "top10": top10,
                "result": {
                    "balance": best["balance"],
                    "net_pct": best["total_net"],
                    "max_dd": best["max_dd"],
                    "total_trades": best["total_trades"],
                    "win_rate": best["avg_wr"],
                },
                "elapsed_seconds": round(elapsed, 1),
            }

            path = RESULTS_DIR / f"{self.symbol}_unified_kelly.json"
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)

            self._log(f"=== KELLY TAMAMLANDI ({elapsed:.0f}s) ===")
            self._log(f"$10K -> ${best['balance']:,.0f} ({best['total_net']:+.1f}%) DD={best['max_dd']:.1f}%")
            self.emit("walkforward_completed", {"summary": summary})
            return summary

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"HATA: {e}")
            self.emit("error", {"message": str(e)})
            return {"error": str(e)}
        finally:
            self.running = False

    def stop(self):
        self.running = False
