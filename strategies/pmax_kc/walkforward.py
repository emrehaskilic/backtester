"""Walk-Forward PMax Optimizer — Haftalık fold'larla Optuna optimizasyonu.

180 gün veri, 90 gün rolling train, 7 gün test, 7 gün adım.
13 fold, her fold'da bağımsız Optuna PMax optimizasyonu.
Hızlandırma: TPESampler(multivariate=True) + MedianPruner + warm-start + n_jobs=4.

Kullanım:
    Pipeline üzerinden: strategy="walkforward_pmax"
    Direkt: python -m strategies.pmax_kc.walkforward ETHUSDT
"""

import json
import logging
import sys
import time
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import optuna
import pandas as pd

from strategies.pmax_kc.config import (
    DEFAULT_PARAMS, INITIAL_BALANCE, MAKER_FEE, TAKER_FEE,
    SCALPER_BOT_PATH, LEVERAGE, MARGIN_PER_TRADE,
)
from data.downloader_klines import fetch_klines
from strategies.pmax_kc.backtest import _get_source

# Rust engine — 178x hizlanma
try:
    import rust_engine
    USE_RUST = True
except ImportError:
    USE_RUST = False
    from strategies.pmax_kc.adaptive_pmax import adaptive_pmax_continuous
    from strategies.pmax_kc.backtest import run_backtest_with_pmax, precompute_indicators

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("walkforward_pmax")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Walk-Forward sabitleri
TRAIN_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7

# Trial limiti — her fold sabit 1000 trial çalışacak, erken kesme yok
TRIALS_PER_FOLD = 1000


def auto_select_best(trials: list[dict]) -> tuple[int, dict]:
    """En iyi trial'ı ratio bazlı otomatik seç."""
    for t in trials:
        t["ratio"] = round(t["oos_net"] / t["dd"], 2) if t["dd"] > 0 else 0.0

    by_ratio = sorted(trials, key=lambda x: x["ratio"], reverse=True)
    valid = [t for t in by_ratio if t["oos_net"] > 0 and t["dd"] > 0 and t["trades"] >= 5]
    if not valid:
        valid = [t for t in by_ratio if t["dd"] > 0]
    if not valid:
        valid = by_ratio

    selected = valid[0]
    idx = by_ratio.index(selected)
    return idx, selected


class WalkForwardPMaxOptimizer:
    """Walk-Forward PMax optimizer — fold bazlı, haftalık test.

    Her fold:
      1. Training verisinde Optuna PMax optimizasyonu (150 trial)
      2. En iyi parametrelerle test haftasını backtest et
      3. Haftalık sonucu kaydet

    Hızlandırma:
      - TPESampler(multivariate=True, n_startup_trials=15)
      - MedianPruner(n_startup_trials=10)
      - n_jobs=4 paralel trial
      - Warm-start: önceki fold'un top-5'i sonraki fold'a enjekte
      - Sabit indikatörler fold başına bir kez pre-compute
    """

    def __init__(self, symbol: str, timeframe: str = "3m", days: int = 180,
                 leverage: int = 25, event_callback: Optional[Callable] = None,
                 n_jobs: int = 6):
        self.symbol = symbol
        self.timeframe = timeframe
        self.days = days
        self.leverage = leverage
        self.event_callback = event_callback
        self.n_jobs = n_jobs
        self.running = True

        self._params = DEFAULT_PARAMS.copy()
        self._log_buffer = []
        self._lock = threading.Lock()

        # Live state — frontend'in her an sorgulayabilecegi durum
        self.live_state = {
            "is_walkforward": True,
            "running": True,
            "symbol": symbol,
            "current_fold": 0,
            "total_folds": 0,
            "completed_trials": 0,
            "total_trials": TRIALS_PER_FOLD,
            "best_oos_net": 0,
            "best_dd": 0,
            "best_ratio": 0,
            "fold_results": [],
        }

    def emit(self, event_type: str, data: dict):
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Event broadcast hatası: {e}")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self._log_buffer.append(f"[{ts}] {msg}")
            if len(self._log_buffer) > 500:
                self._log_buffer = self._log_buffer[-500:]
        logger.info(msg)
        self.emit("log", {"message": msg, "level": "info"})

    def _compute_pmax(self, src, h, l, c, pmax_params):
        """Adaptive PMax hesapla — Rust veya Python."""
        atr_period = self._params.get("atr_period", 10)
        atr_mult = self._params.get("atr_multiplier", 3.0)
        ma_length = self._params.get("ma_length", 10)

        if USE_RUST:
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
            return np.ascontiguousarray(result["pmax_line"]), np.ascontiguousarray(result["mavg"]), np.ascontiguousarray(result["direction"])
        else:
            p, m, d, _, _, _ = adaptive_pmax_continuous(
                src, h, l, c, ma_type="EMA",
                base_atr_period=atr_period, base_atr_multiplier=atr_mult, base_ma_length=ma_length,
                lookback=pmax_params["vol_lookback"], flip_window=pmax_params["flip_window"],
                mult_base=pmax_params["mult_base"], mult_scale=pmax_params["mult_scale"],
                ma_base=pmax_params["ma_base"], ma_scale=pmax_params["ma_scale"],
                atr_base=pmax_params["atr_base"], atr_scale=pmax_params["atr_scale"],
                update_interval=pmax_params["update_interval"],
            )
            return p.values, m.values, d.values

    def _build_folds(self, df: pd.DataFrame) -> list[dict]:
        """Veriyi fold'lara böl. Her fold train + test penceresi."""
        # 3m kline = 480 mum/gün
        bars_per_day = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96}.get(self.timeframe, 480)

        train_bars = TRAIN_DAYS * bars_per_day
        test_bars = TEST_DAYS * bars_per_day
        step_bars = STEP_DAYS * bars_per_day
        total_bars = len(df)

        folds = []
        fold_num = 1
        start = 0

        while start + train_bars + test_bars <= total_bars:
            train_end = start + train_bars
            test_end = train_end + test_bars

            folds.append({
                "fold": fold_num,
                "train_start": start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": min(test_end, total_bars),
                "train_df": df.iloc[start:train_end].reset_index(drop=True),
                "test_df": df.iloc[train_end:test_end].reset_index(drop=True),
            })

            fold_num += 1
            start += step_bars

        return folds

    def _run_fold(self, fold: dict, warm_start_params: list[dict] = None) -> Optional[dict]:
        """Tek bir fold'u çalıştır: sadece train'de optimize et, test'i 1 kez çalıştır.

        Walk-Forward prensibi: test verisi optimization sırasında GÖRÜLMEZ.
        Sadece seçilen en iyi parametrelerle 1 kez test edilir.
        """
        fold_num = fold["fold"]
        train_df = fold["train_df"]
        test_df = fold["test_df"]

        if len(train_df) < 500 or len(test_df) < 100:
            self._log(f"Fold {fold_num}: Yetersiz veri (train={len(train_df)}, test={len(test_df)})")
            return None

        source = self._params.get("source", "hl2")
        src_train = _get_source(train_df, source)
        h_tr, l_tr, c_tr = train_df["high"], train_df["low"], train_df["close"]
        params = self._params

        # Sabit indikatörleri fold başına 1 kez hesapla
        if USE_RUST:
            sa = np.ascontiguousarray
            train_indicators = rust_engine.precompute_indicators(
                sa(h_tr.values, dtype=np.float64),
                sa(l_tr.values, dtype=np.float64),
                sa(c_tr.values, dtype=np.float64),
                params.get("ema_filter_period", 144),
                params.get("kc_length", 20),
                params.get("kc_multiplier", 1.5),
                params.get("kc_atr_period", 10),
            )
        else:
            train_indicators = precompute_indicators(train_df, params)

        self._log(f"=== FOLD {fold_num}: Train {len(train_df)} bar, Test {len(test_df)} bar ({TRIALS_PER_FOLD} trial) ===")
        self.live_state.update({
            "current_fold": fold_num,
            "completed_trials": 0,
            "total_trials": TRIALS_PER_FOLD,
            "best_oos_net": 0, "best_dd": 0, "best_ratio": 0,
        })
        self.emit("fold_started", {
            "fold": fold_num,
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "n_trials": TRIALS_PER_FOLD,
        })

        # Optuna study
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=20,
            seed=fold_num * 42,
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        # Warm-start: önceki fold'un top-5 parametrelerini enjekte et
        if warm_start_params:
            for wp in warm_start_params[:5]:
                try:
                    study.enqueue_trial(wp)
                except Exception:
                    pass

        trials_data = []
        trial_lock = threading.Lock()

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
                # SADECE train backtest — test verisi görülmüyor
                pi, mi, di = self._compute_pmax(src_train, h_tr, l_tr, c_tr, cp)
                if USE_RUST:
                    sa = np.ascontiguousarray
                    train_r = rust_engine.run_backtest(
                        sa(c_tr.values, dtype=np.float64),
                        sa(h_tr.values, dtype=np.float64),
                        sa(l_tr.values, dtype=np.float64),
                        sa(pi, dtype=np.float64), sa(mi, dtype=np.float64), sa(di, dtype=np.float64),
                        sa(train_indicators["rsi_vals"]), sa(train_indicators["ema_filter"]),
                        sa(train_indicators["rsi_ema_vals"]), sa(train_indicators["atr_vol"]),
                        sa(train_indicators["kc_upper_arr"]), sa(train_indicators["kc_lower_arr"]),
                        params.get("rsi_overbought", 65.0),
                        params.get("max_dca_steps", 2),
                        params.get("tp_close_percent", 0.20),
                    )
                else:
                    train_r = run_backtest_with_pmax(
                        train_df, pi, mi, di, params, "wf_train",
                        precomputed=train_indicators,
                    )
                if train_r["total_trades"] < 5:
                    return -999

                # Pruning
                trial.report(train_r["net_pct"], step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Risk-adjusted skor: Calmar-benzeri ratio
                # net / max(dd, 5) * sqrt(wr/100) — düşük DD + yüksek WR + iyi getiri
                import math
                train_dd = max(train_r["max_drawdown"], 5.0)
                train_wr = max(train_r["win_rate"], 1.0)
                score = (train_r["net_pct"] / train_dd) * math.sqrt(train_wr / 100.0)

                trial.set_user_attr("is_net", train_r["net_pct"])
                trial.set_user_attr("is_dd", train_r["max_drawdown"])
                trial.set_user_attr("is_wr", train_r["win_rate"])
                trial.set_user_attr("is_trades", train_r["total_trades"])
                return score
            except optuna.TrialPruned:
                raise
            except Exception:
                return -999

        def callback(study, trial):
            if not self.running:
                study.stop()
                return
            if trial.value is not None and trial.value > -999 and "is_net" in trial.user_attrs:
                a = trial.user_attrs
                td = {
                    "tid": trial.number,
                    "score": round(trial.value, 4),
                    "is_net": round(a.get("is_net", 0), 1),
                    "is_dd": round(a.get("is_dd", 0), 1),
                    "wr": round(a.get("is_wr", 0), 1),
                    "dd": round(a.get("is_dd", 0), 1),
                    "trades": a.get("is_trades", 0),
                    "oos_net": 0,  # OOS henüz bilinmiyor
                    "params": trial.params,
                }
                with trial_lock:
                    trials_data.append(td)

                n_valid = len(trials_data)
                is_net = td["is_net"]
                is_dd = td["dd"]
                ratio = round(is_net / is_dd, 1) if is_dd > 0 else 0

                # Live state guncelle
                self.live_state["completed_trials"] = n_valid
                if is_net > self.live_state["best_oos_net"]:
                    self.live_state["best_oos_net"] = is_net
                if is_dd > 0 and (self.live_state["best_dd"] == 0 or is_dd < self.live_state["best_dd"]):
                    self.live_state["best_dd"] = is_dd
                if ratio > self.live_state["best_ratio"]:
                    self.live_state["best_ratio"] = ratio
                self.emit("trial_completed", {
                    "fold": fold_num,
                    "trial_num": trial.number,
                    "score": td["score"],
                    "oos_net": is_net,  # Aslında train net, OOS henüz yok
                    "dd": is_dd,
                    "ratio": ratio,
                })

        # 1000 trial — erken kesme yok
        study.optimize(objective, n_trials=TRIALS_PER_FOLD, callbacks=[callback],
                       n_jobs=self.n_jobs)

        if not self.running or not trials_data:
            return None

        # En iyi trial'ı seç (train score bazlı)
        best_trial = max(trials_data, key=lambda x: x["score"])
        best_params = best_trial["params"]

        # TEST: Seçilen parametrelerle test haftasını 1 kez çalıştır (ilk kez görülüyor)
        src_test = _get_source(test_df, source)
        h_te, l_te, c_te = test_df["high"], test_df["low"], test_df["close"]
        po, mo, do2 = self._compute_pmax(src_test, h_te, l_te, c_te, best_params)
        if USE_RUST:
            sa = np.ascontiguousarray
            test_indicators = rust_engine.precompute_indicators(
                sa(h_te.values, dtype=np.float64), sa(l_te.values, dtype=np.float64),
                sa(c_te.values, dtype=np.float64),
                params.get("ema_filter_period", 144), params.get("kc_length", 20),
                params.get("kc_multiplier", 1.5), params.get("kc_atr_period", 10),
            )
            test_result = rust_engine.run_backtest(
                sa(c_te.values, dtype=np.float64), sa(h_te.values, dtype=np.float64),
                sa(l_te.values, dtype=np.float64),
                sa(po, dtype=np.float64), sa(mo, dtype=np.float64), sa(do2, dtype=np.float64),
                sa(test_indicators["rsi_vals"]), sa(test_indicators["ema_filter"]),
                sa(test_indicators["rsi_ema_vals"]), sa(test_indicators["atr_vol"]),
                sa(test_indicators["kc_upper_arr"]), sa(test_indicators["kc_lower_arr"]),
                params.get("rsi_overbought", 65.0), params.get("max_dca_steps", 2),
                params.get("tp_close_percent", 0.20),
            )
        else:
            test_indicators = precompute_indicators(test_df, params)
            test_result = run_backtest_with_pmax(
                test_df, po, mo, do2, params, "wf_final", precomputed=test_indicators,
            )

        # Tarih bilgisi
        train_start_date = pd.to_datetime(train_df["open_time"].iloc[0], unit="ms").strftime("%Y-%m-%d")
        train_end_date = pd.to_datetime(train_df["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
        test_start_date = pd.to_datetime(test_df["open_time"].iloc[0], unit="ms").strftime("%Y-%m-%d")
        test_end_date = pd.to_datetime(test_df["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")

        # Top-5 parametreleri warm-start için kaydet
        top5_by_score = sorted(trials_data, key=lambda x: x["score"], reverse=True)[:5]
        top5_params = [t["params"] for t in top5_by_score if t.get("params")]

        fold_result = {
            "fold": fold_num,
            "train_period": f"{train_start_date} → {train_end_date}",
            "test_period": f"{test_start_date} → {test_end_date}",
            "params": best_params,
            "train_net": round(best_trial["is_net"], 2),
            "test_net": round(test_result["net_pct"], 2),
            "test_dd": round(test_result["max_drawdown"], 2),
            "test_wr": round(test_result["win_rate"], 2),
            "test_trades": test_result["total_trades"],
            "test_balance": round(test_result["balance"], 2),
            "ratio": round(test_result["net_pct"] / max(test_result["max_drawdown"], 1), 2),
            "total_valid_trials": len(trials_data),
            "top5_params": top5_params,
        }

        conv_label = f"{len(trials_data)} gecerli / {TRIALS_PER_FOLD} trial"
        self._log(
            f"Fold {fold_num} TAMAMLANDI ({conv_label}) — "
            f"Test: {test_result['net_pct']:+.1f}% DD:{test_result['max_drawdown']:.1f}% "
            f"WR:{test_result['win_rate']:.0f}% Trades:{test_result['total_trades']} "
            f"[{test_start_date} → {test_end_date}]"
        )

        self.live_state["fold_results"].append(fold_result)
        self.emit("fold_completed", {
            "fold": fold_num,
            "result": fold_result,
            "top5": sorted(trials_data, key=lambda x: x.get("ratio", 0), reverse=True)[:5],
        })

        return fold_result

    def run(self) -> dict:
        """Tam Walk-Forward analizi çalıştır.

        Returns:
            Tüm fold sonuçlarını ve aggregate metrikleri içeren dict
        """
        self.running = True
        start_time = time.time()

        try:
            # 1. Veri yükle
            self._log(f"=== WALK-FORWARD BAŞLIYOR: {self.symbol} {self.timeframe} ({self.days} gün) ===")
            data_path = Path(f"data/{self.symbol}_{self.timeframe}_{self.days}d.parquet")
            data_path.parent.mkdir(exist_ok=True)

            if data_path.exists():
                df = pd.read_parquet(data_path)
                self._log(f"Cache: {len(df)} mum yüklendi")
            else:
                self._log(f"Veri indiriliyor: {self.symbol} {self.timeframe} {self.days} gün...")
                self.emit("log", {"message": f"Veri indiriliyor...", "level": "info"})
                df = fetch_klines(self.symbol, self.timeframe, self.days)
                df.to_parquet(data_path, index=False)
                self._log(f"İndirildi: {len(df)} mum")

            if len(df) < 1000:
                return {"error": f"Yetersiz veri: {len(df)} mum (min 1000)"}

            # 2. Fold'ları oluştur
            folds = self._build_folds(df)
            total_folds = len(folds)
            self._log(f"{total_folds} fold oluşturuldu (Train:{TRAIN_DAYS}d, Test:{TEST_DAYS}d, Step:{STEP_DAYS}d)")

            self.live_state["total_folds"] = total_folds
            self.live_state["total_trials"] = TRIALS_PER_FOLD
            self.emit("walkforward_started", {
                "symbol": self.symbol,
                "total_folds": total_folds,
                "train_days": TRAIN_DAYS,
                "test_days": TEST_DAYS,
                "step_days": STEP_DAYS,
                "trials_per_fold": TRIALS_PER_FOLD,
            })

            # 3. Her fold'u çalıştır
            fold_results = []
            warm_start_params = None

            for fold in folds:
                if not self.running:
                    self._log("Durdurma sinyali alındı")
                    break

                result = self._run_fold(fold, warm_start_params)

                if result:
                    fold_results.append(result)

                    # Warm-start: bu fold'un top-5 parametresini sonraki fold'a aktar
                    warm_start_params = result.get("top5_params", [result["params"]])

                    # Fold checkpoint kaydet
                    self._save_fold_checkpoint(fold_results)

            # 4. Aggregate sonuçlar
            if not fold_results:
                return {"error": "Hiçbir fold başarılı olmadı"}

            summary = self._compute_summary(fold_results, time.time() - start_time)

            # 5. Final kaydet
            result_path = RESULTS_DIR / f"{self.symbol}_walkforward.json"
            with open(result_path, "w") as f:
                json.dump(summary, f, indent=2)

            self._log(f"=== WALK-FORWARD TAMAMLANDI — {result_path} ===")
            self._log(
                f"Sonuç: {summary['aggregate']['profitable_weeks']}/{summary['aggregate']['total_weeks']} hafta kârlı, "
                f"Ort: {summary['aggregate']['avg_weekly_net']:+.1f}%, "
                f"En kötü: {summary['aggregate']['worst_week_net']:+.1f}%"
            )

            self.emit("walkforward_completed", {"summary": summary})
            return summary

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"HATA: {e}")
            self.emit("error", {"message": f"Walk-Forward hatası: {e}"})
            return {"error": str(e)}
        finally:
            self.running = False

    def _save_fold_checkpoint(self, fold_results: list[dict]):
        """Her fold sonrası checkpoint kaydet."""
        try:
            checkpoint_path = RESULTS_DIR / f"{self.symbol}_wf_checkpoint.json"
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "symbol": self.symbol,
                    "completed_folds": len(fold_results),
                    "folds": fold_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Checkpoint kayıt hatası: {e}")

    def _compute_summary(self, fold_results: list[dict], elapsed_sec: float) -> dict:
        """Tüm fold sonuçlarını aggregate et."""
        nets = [f["test_net"] for f in fold_results]
        dds = [f["test_dd"] for f in fold_results]
        wrs = [f["test_wr"] for f in fold_results]
        trades = [f["test_trades"] for f in fold_results]
        ratios = [f["ratio"] for f in fold_results]

        profitable_weeks = sum(1 for n in nets if n > 0)
        total_weeks = len(nets)

        # Parametre stabilitesi — standart sapma (düşük = stabil)
        all_params = [f["params"] for f in fold_results]
        param_stability = {}
        if all_params:
            for key in all_params[0]:
                vals = [p[key] for p in all_params if key in p]
                if vals and isinstance(vals[0], (int, float)):
                    mean = np.mean(vals)
                    std = np.std(vals)
                    cv = std / mean if mean != 0 else 0  # Coefficient of Variation
                    param_stability[key] = {
                        "mean": round(float(mean), 2),
                        "std": round(float(std), 2),
                        "cv": round(float(cv), 3),  # < 0.3 = stabil
                    }

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "days": self.days,
            "leverage": self.leverage,
            "config": {
                "train_days": TRAIN_DAYS,
                "test_days": TEST_DAYS,
                "step_days": STEP_DAYS,
                "trials_per_fold": TRIALS_PER_FOLD,
                "n_jobs": self.n_jobs,
            },
            "aggregate": {
                "total_weeks": total_weeks,
                "profitable_weeks": profitable_weeks,
                "win_rate_weeks": round(profitable_weeks / total_weeks * 100, 1) if total_weeks > 0 else 0,
                "avg_weekly_net": round(float(np.mean(nets)), 2),
                "median_weekly_net": round(float(np.median(nets)), 2),
                "best_week_net": round(float(max(nets)), 2),
                "worst_week_net": round(float(min(nets)), 2),
                "total_net": round(float(sum(nets)), 2),
                "avg_dd": round(float(np.mean(dds)), 2),
                "max_dd": round(float(max(dds)), 2),
                "avg_wr": round(float(np.mean(wrs)), 1),
                "avg_trades_per_week": round(float(np.mean(trades)), 1),
                "total_trades": sum(trades),
                "avg_ratio": round(float(np.mean(ratios)), 2),
                "consistency_score": round(profitable_weeks / total_weeks * 100, 1) if total_weeks > 0 else 0,
            },
            "param_stability": param_stability,
            "folds": fold_results,
            "elapsed_seconds": round(elapsed_sec, 1),
        }

    def stop(self):
        """Gracefully durdur."""
        self.running = False
        self._log("Durdurma sinyali alındı — mevcut fold bittikten sonra duracak")


class WalkForwardKCOptimizer(WalkForwardPMaxOptimizer):
    """Walk-Forward KC Optimizer — PMax kilitli, KC parametreleri optimize.

    ADIM 1'den gelen PMax parametrelerini yükler, kilitler.
    Her fold'da KC parametrelerini (kc_length, kc_multiplier, kc_atr_period,
    max_dca_steps, tp_close_percent) optimize eder.
    """

    def __init__(self, symbol: str, timeframe: str = "3m", days: int = 180,
                 leverage: int = 25, event_callback: Optional[Callable] = None,
                 n_jobs: int = 6):
        super().__init__(symbol, timeframe, days, leverage, event_callback, n_jobs)
        self._pmax_results = None  # ADIM 1 fold sonuçları

    def _load_pmax_results(self) -> list[dict]:
        """ADIM 1 PMax WF sonuçlarını yükle."""
        path = RESULTS_DIR / f"{self.symbol}_walkforward.json"
        if not path.exists():
            raise FileNotFoundError(f"PMax WF sonuçları bulunamadı: {path}")
        with open(path) as f:
            data = json.load(f)
        folds = data.get("folds", [])
        if not folds:
            raise ValueError("PMax WF sonuçlarında fold bulunamadı")
        self._log(f"PMax sonuçları yüklendi: {len(folds)} fold")
        return folds

    def _get_pmax_params_for_fold(self, fold_num: int) -> dict:
        """Belirli fold için en iyi PMax parametrelerini döndür."""
        if not self._pmax_results:
            return {}
        # Aynı fold numarası varsa onu kullan
        for f in self._pmax_results:
            if f["fold"] == fold_num:
                return f["params"]
        # Yoksa en iyi (en yüksek ratio) fold'un parametrelerini kullan
        best = max(self._pmax_results, key=lambda x: x.get("ratio", 0))
        self._log(f"Fold {fold_num} için PMax param yok, en iyi fold {best['fold']} kullanılıyor")
        return best["params"]

    def _run_fold(self, fold: dict, warm_start_params: list[dict] = None) -> Optional[dict]:
        """KC fold: PMax kilitli, KC optimize."""
        fold_num = fold["fold"]
        train_df = fold["train_df"]
        test_df = fold["test_df"]

        if len(train_df) < 500 or len(test_df) < 100:
            self._log(f"Fold {fold_num}: Yetersiz veri")
            return None

        source = self._params.get("source", "hl2")
        src_train = _get_source(train_df, source)
        h_tr, l_tr, c_tr = train_df["high"], train_df["low"], train_df["close"]
        params = self._params

        # Bu fold'un PMax parametrelerini al ve kilitle
        pmax_params = self._get_pmax_params_for_fold(fold_num)
        if not pmax_params:
            self._log(f"Fold {fold_num}: PMax parametresi bulunamadı, atlanıyor")
            return None

        # PMax'i bir kez hesapla ve kilitle (tüm trial'lar bu PMax'i kullanacak)
        pi, mi, di = self._compute_pmax(src_train, h_tr, l_tr, c_tr, pmax_params)

        self._log(f"=== KC FOLD {fold_num}: PMax kilitli, KC optimize ({TRIALS_PER_FOLD} trial) ===")
        self.live_state.update({
            "current_fold": fold_num,
            "completed_trials": 0,
            "total_trials": TRIALS_PER_FOLD,
            "best_oos_net": 0, "best_dd": 0, "best_ratio": 0,
        })
        self.emit("fold_started", {
            "fold": fold_num,
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "n_trials": TRIALS_PER_FOLD,
        })

        # Optuna study
        sampler = optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=20, seed=fold_num * 77,
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        if warm_start_params:
            for wp in warm_start_params[:5]:
                try:
                    study.enqueue_trial(wp)
                except Exception:
                    pass

        trials_data = []
        trial_lock = threading.Lock()

        # PMax ve sabit indikatörler pre-compute (bir kez)
        sa = np.ascontiguousarray
        pi_c = sa(pi, dtype=np.float64)
        mi_c = sa(mi, dtype=np.float64)
        di_c = sa(di, dtype=np.float64)
        c_tr_arr = sa(c_tr.values, dtype=np.float64)
        h_tr_arr = sa(h_tr.values, dtype=np.float64)
        l_tr_arr = sa(l_tr.values, dtype=np.float64)

        # RSI ve EMA filter sabit — KC parametrelerine bağlı değil
        # Ama KC indikatörleri (kc_upper, kc_lower) her trial'da farklı olacak
        # Bu yüzden RSI/EMA'yı pre-compute et, KC'yi trial içinde hesapla
        if USE_RUST:
            # Sadece RSI + EMA + ATR_VOL pre-compute (KC her trial'da değişecek)
            base_ind = rust_engine.precompute_indicators(
                h_tr_arr, l_tr_arr, c_tr_arr,
                params.get("ema_filter_period", 144),
                20, 1.5, 10,  # dummy KC params — kullanılmayacak
            )

        def objective(trial):
            kc_params = {
                "kc_length": trial.suggest_int("kc_length", 5, 50),
                "kc_multiplier": trial.suggest_float("kc_multiplier", 0.5, 4.0, step=0.1),
                "kc_atr_period": trial.suggest_int("kc_atr_period", 3, 30),
                "max_dca_steps": trial.suggest_int("max_dca_steps", 1, 5),
                "tp_close_percent": trial.suggest_float("tp_close_percent", 0.05, 0.50, step=0.05),
            }
            try:
                if USE_RUST:
                    # KC indikatörlerini bu trial'ın parametreleriyle hesapla
                    trial_ind = rust_engine.precompute_indicators(
                        h_tr_arr, l_tr_arr, c_tr_arr,
                        params.get("ema_filter_period", 144),
                        kc_params["kc_length"],
                        kc_params["kc_multiplier"],
                        kc_params["kc_atr_period"],
                    )
                    train_r = rust_engine.run_backtest(
                        c_tr_arr, h_tr_arr, l_tr_arr,
                        pi_c, mi_c, di_c,
                        sa(trial_ind["rsi_vals"]), sa(trial_ind["ema_filter"]),
                        sa(trial_ind["rsi_ema_vals"]), sa(trial_ind["atr_vol"]),
                        sa(trial_ind["kc_upper_arr"]), sa(trial_ind["kc_lower_arr"]),
                        params.get("rsi_overbought", 65.0),
                        kc_params["max_dca_steps"],
                        kc_params["tp_close_percent"],
                    )
                else:
                    tp = params.copy()
                    tp.update(kc_params)
                    from strategies.pmax_kc.backtest import run_backtest_with_pmax
                    train_r = run_backtest_with_pmax(train_df, pi, mi, di, tp, "kc_train")

                if train_r["total_trades"] < 5:
                    return -999

                trial.report(train_r["net_pct"], step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                import math
                train_dd = max(train_r["max_drawdown"], 5.0)
                train_wr = max(train_r["win_rate"], 1.0)
                score = (train_r["net_pct"] / train_dd) * math.sqrt(train_wr / 100.0)

                trial.set_user_attr("is_net", train_r["net_pct"])
                trial.set_user_attr("is_dd", train_r["max_drawdown"])
                trial.set_user_attr("is_wr", train_r["win_rate"])
                trial.set_user_attr("is_trades", train_r["total_trades"])
                return score
            except optuna.TrialPruned:
                raise
            except Exception:
                return -999

        def callback(study, trial):
            if not self.running:
                study.stop()
                return
            if trial.value is not None and trial.value > -999 and "is_net" in trial.user_attrs:
                a = trial.user_attrs
                td = {
                    "tid": trial.number,
                    "score": round(trial.value, 4),
                    "is_net": round(a.get("is_net", 0), 1),
                    "is_dd": round(a.get("is_dd", 0), 1),
                    "wr": round(a.get("is_wr", 0), 1),
                    "dd": round(a.get("is_dd", 0), 1),
                    "trades": a.get("is_trades", 0),
                    "oos_net": 0,
                    "params": trial.params,
                }
                with trial_lock:
                    trials_data.append(td)

                n_valid = len(trials_data)
                is_net = td["is_net"]
                is_dd = td["dd"]
                ratio = round(is_net / is_dd, 1) if is_dd > 0 else 0

                self.live_state["completed_trials"] = n_valid
                if is_net > self.live_state["best_oos_net"]:
                    self.live_state["best_oos_net"] = is_net
                if is_dd > 0 and (self.live_state["best_dd"] == 0 or is_dd < self.live_state["best_dd"]):
                    self.live_state["best_dd"] = is_dd
                if ratio > self.live_state["best_ratio"]:
                    self.live_state["best_ratio"] = ratio
                self.emit("trial_completed", {
                    "fold": fold_num,
                    "trial_num": trial.number,
                    "score": td["score"],
                    "oos_net": is_net,
                    "dd": is_dd,
                    "ratio": ratio,
                })

        study.optimize(objective, n_trials=TRIALS_PER_FOLD, callbacks=[callback],
                       n_jobs=self.n_jobs)

        if not self.running or not trials_data:
            return None

        # En iyi KC parametreleri
        best_trial = max(trials_data, key=lambda x: x["score"])
        best_kc_params = best_trial["params"]

        # TEST: Kilitli PMax + en iyi KC ile test haftası
        src_test = _get_source(test_df, source)
        h_te, l_te, c_te = test_df["high"], test_df["low"], test_df["close"]
        po, mo, do2 = self._compute_pmax(src_test, h_te, l_te, c_te, pmax_params)

        if USE_RUST:
            test_ind = rust_engine.precompute_indicators(
                sa(h_te.values, dtype=np.float64), sa(l_te.values, dtype=np.float64),
                sa(c_te.values, dtype=np.float64),
                params.get("ema_filter_period", 144),
                best_kc_params["kc_length"],
                best_kc_params["kc_multiplier"],
                best_kc_params["kc_atr_period"],
            )
            test_result = rust_engine.run_backtest(
                sa(c_te.values, dtype=np.float64), sa(h_te.values, dtype=np.float64),
                sa(l_te.values, dtype=np.float64),
                sa(po, dtype=np.float64), sa(mo, dtype=np.float64), sa(do2, dtype=np.float64),
                sa(test_ind["rsi_vals"]), sa(test_ind["ema_filter"]),
                sa(test_ind["rsi_ema_vals"]), sa(test_ind["atr_vol"]),
                sa(test_ind["kc_upper_arr"]), sa(test_ind["kc_lower_arr"]),
                params.get("rsi_overbought", 65.0),
                best_kc_params["max_dca_steps"],
                best_kc_params["tp_close_percent"],
            )
        else:
            from strategies.pmax_kc.backtest import run_backtest_with_pmax, precompute_indicators
            tp = params.copy()
            tp.update(best_kc_params)
            test_ind = precompute_indicators(test_df, tp)
            test_result = run_backtest_with_pmax(test_df, po, mo, do2, tp, "kc_final", precomputed=test_ind)

        # Tarih bilgisi
        train_start = pd.to_datetime(train_df["open_time"].iloc[0], unit="ms").strftime("%Y-%m-%d")
        train_end = pd.to_datetime(train_df["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
        test_start = pd.to_datetime(test_df["open_time"].iloc[0], unit="ms").strftime("%Y-%m-%d")
        test_end = pd.to_datetime(test_df["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")

        top5_by_score = sorted(trials_data, key=lambda x: x["score"], reverse=True)[:5]
        top5_params = [t["params"] for t in top5_by_score if t.get("params")]

        fold_result = {
            "fold": fold_num,
            "train_period": f"{train_start} → {train_end}",
            "test_period": f"{test_start} → {test_end}",
            "pmax_params": pmax_params,
            "params": best_kc_params,
            "train_net": round(best_trial["is_net"], 2),
            "test_net": round(test_result["net_pct"], 2),
            "test_dd": round(test_result["max_drawdown"], 2),
            "test_wr": round(test_result["win_rate"], 2),
            "test_trades": test_result["total_trades"],
            "test_balance": round(test_result["balance"], 2),
            "ratio": round(test_result["net_pct"] / max(test_result["max_drawdown"], 1), 2),
            "total_valid_trials": len(trials_data),
            "top5_params": top5_params,
        }

        conv_label = f"{len(trials_data)} gecerli / {TRIALS_PER_FOLD} trial"
        self._log(
            f"KC Fold {fold_num} TAMAMLANDI ({conv_label}) — "
            f"Test: {test_result['net_pct']:+.1f}% DD:{test_result['max_drawdown']:.1f}% "
            f"WR:{test_result['win_rate']:.0f}% Trades:{test_result['total_trades']} "
            f"[{test_start} → {test_end}]"
        )

        self.live_state["fold_results"].append(fold_result)
        self.emit("fold_completed", {
            "fold": fold_num,
            "result": fold_result,
            "top5": sorted(trials_data, key=lambda x: x.get("score", 0), reverse=True)[:5],
        })

        return fold_result

    def run(self) -> dict:
        """KC Walk-Forward çalıştır — PMax kilitli."""
        self.running = True
        start_time = time.time()

        try:
            self._log(f"=== KC WALK-FORWARD BAŞLIYOR: {self.symbol} ===")

            # PMax sonuçlarını yükle
            self._pmax_results = self._load_pmax_results()

            # Veri yükle
            data_path = Path(f"data/{self.symbol}_{self.timeframe}_{self.days}d.parquet")
            if data_path.exists():
                df = pd.read_parquet(data_path)
                self._log(f"Cache: {len(df)} mum")
            else:
                df = fetch_klines(self.symbol, self.timeframe, self.days)
                df.to_parquet(data_path, index=False)
                self._log(f"İndirildi: {len(df)} mum")

            if len(df) < 1000:
                return {"error": f"Yetersiz veri: {len(df)} mum"}

            folds = self._build_folds(df)
            total_folds = len(folds)
            self._log(f"{total_folds} fold oluşturuldu")

            self.live_state["total_folds"] = total_folds
            self.live_state["total_trials"] = TRIALS_PER_FOLD
            self.emit("walkforward_started", {
                "symbol": self.symbol,
                "total_folds": total_folds,
                "train_days": TRAIN_DAYS,
                "test_days": TEST_DAYS,
                "step_days": STEP_DAYS,
                "trials_per_fold": TRIALS_PER_FOLD,
            })

            fold_results = []
            warm_start_params = None

            for fold in folds:
                if not self.running:
                    self._log("Durdurma sinyali alındı")
                    break

                result = self._run_fold(fold, warm_start_params)

                if result:
                    fold_results.append(result)
                    warm_start_params = result.get("top5_params", [result["params"]])
                    self._save_kc_checkpoint(fold_results)

            if not fold_results:
                return {"error": "Hiçbir fold başarılı olmadı"}

            summary = self._compute_summary(fold_results, time.time() - start_time)

            result_path = RESULTS_DIR / f"{self.symbol}_walkforward_kc.json"
            with open(result_path, "w") as f:
                json.dump(summary, f, indent=2)

            self._log(f"=== KC WALK-FORWARD TAMAMLANDI — {result_path} ===")
            self._log(
                f"Sonuç: {summary['aggregate']['profitable_weeks']}/{summary['aggregate']['total_weeks']} hafta kârlı, "
                f"Ort: {summary['aggregate']['avg_weekly_net']:+.1f}%"
            )

            self.emit("walkforward_completed", {"summary": summary})
            return summary

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"HATA: {e}")
            self.emit("error", {"message": f"KC Walk-Forward hatası: {e}"})
            return {"error": str(e)}
        finally:
            self.running = False

    def _save_kc_checkpoint(self, fold_results: list[dict]):
        try:
            path = RESULTS_DIR / f"{self.symbol}_wf_kc_checkpoint.json"
            with open(path, "w") as f:
                json.dump({
                    "symbol": self.symbol,
                    "completed_folds": len(fold_results),
                    "folds": fold_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"KC Checkpoint kayıt hatası: {e}")


# CLI çalıştırma
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward PMax Optimizer")
    parser.add_argument("symbol", nargs="?", default="ETHUSDT", help="Trading pair")
    parser.add_argument("--timeframe", default="3m", help="Kline timeframe")
    parser.add_argument("--days", type=int, default=180, help="Lookback days")
    parser.add_argument("--trials", type=int, default=150, help="Trials per fold")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel trials")
    args = parser.parse_args()

    optimizer = WalkForwardPMaxOptimizer(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        n_jobs=args.n_jobs,
    )

    result = optimizer.run(trial_count=args.trials)

    if "error" in result:
        print(f"\nHATA: {result['error']}")
    else:
        agg = result["aggregate"]
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD SONUCU: {args.symbol}")
        print(f"{'='*60}")
        print(f"Kârlı Hafta: {agg['profitable_weeks']}/{agg['total_weeks']} ({agg['win_rate_weeks']:.0f}%)")
        print(f"Ortalama Haftalık: {agg['avg_weekly_net']:+.1f}%")
        print(f"En İyi Hafta: {agg['best_week_net']:+.1f}%")
        print(f"En Kötü Hafta: {agg['worst_week_net']:+.1f}%")
        print(f"Toplam Net: {agg['total_net']:+.1f}%")
        print(f"Ort DD: {agg['avg_dd']:.1f}%")
        print(f"Ort WR: {agg['avg_wr']:.0f}%")
        print(f"Toplam Trade: {agg['total_trades']}")
        print(f"Süre: {result['elapsed_seconds']:.0f}s")
        print(f"\nDetaylı sonuç: results/{args.symbol}_walkforward.json")
