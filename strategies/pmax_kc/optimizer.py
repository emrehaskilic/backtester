"""PMax+KC Full-Auto Optimizer — round bazlı, sıfır insan müdahalesi.

Orijinal: claude optimizator/auto_optimize.py
Değişiklikler:
  - _wait_for_approval() KALDIRILDI → auto_select_best() ile değiştirildi
  - PMax cache R4-R5-Kelly arası paylaşılıyor
  - n_jobs ile paralel trial desteği
  - Optuna MedianPruner ile erken durdurma
  - StrategyOptimizer interface'e uyumlu
"""

import json
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import optuna
import pandas as pd

from strategies.pmax_kc.config import (
    DEFAULT_PARAMS, TRAIN_RATIO, INITIAL_BALANCE, MAKER_FEE, TAKER_FEE,
    SCALPER_BOT_PATH,
)
from data.downloader_klines import fetch_klines, split_data
from strategies.pmax_kc.adaptive_pmax import adaptive_pmax_continuous
from strategies.pmax_kc.backtest import run_backtest_with_pmax, _get_source
from strategies.base import StrategyOptimizer, RoundConfig, TrialResult, RoundResult

if SCALPER_BOT_PATH not in sys.path:
    sys.path.append(SCALPER_BOT_PATH)
from core.strategy.indicators import pmax as original_pmax, rsi, ema, atr, keltner_channel

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pmax_optimizer")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def auto_select_best(trials: list[dict], method: str = "ratio") -> tuple[int, dict]:
    """En iyi trial'ı otomatik seç. İnsan müdahalesi yok.

    Returns:
        (index_in_top10, selected_trial_dict)
    """
    for t in trials:
        t["ratio"] = round(t["oos_net"] / t["dd"], 2) if t["dd"] > 0 else 0.0

    by_ratio = sorted(trials, key=lambda x: x["ratio"], reverse=True)

    # Filtrele: OOS pozitif, en az 10 trade, DD > 0
    valid = [t for t in by_ratio if t["oos_net"] > 0 and t["dd"] > 0 and t["trades"] >= 10]
    if not valid:
        # Fallback: DD > 0 olan herhangi biri
        valid = [t for t in by_ratio if t["dd"] > 0]
    if not valid:
        valid = by_ratio  # absolute fallback

    selected = valid[0]
    idx = by_ratio.index(selected)
    return idx, selected


def make_narrow(val, pct=0.3, mn=None, mx=None, step=None, is_int=False):
    """Parametre aralığını daralt."""
    lo = val * (1 - pct)
    hi = val * (1 + pct)
    if mn is not None: lo = max(lo, mn)
    if mx is not None: hi = min(hi, mx)
    if lo > hi: lo, hi = hi, lo
    if lo == hi: hi = lo + (1 if is_int else 0.25)
    if is_int: return int(lo), int(hi)
    if step: lo = round(lo / step) * step; hi = round(hi / step) * step
    if lo >= hi: hi = lo + (step if step else 0.25)
    return lo, hi


class PMaxKCOptimizer(StrategyOptimizer):
    """PMax + Keltner Channel tam otomatik optimizer.

    6 round pipeline:
      R1: PMax Keşif (geniş arama)
      R2: PMax Fine-Tune (dar arama)
      R3: DD Optimize (drawdown penalty)
      R4: KC Optimize (PMax kilitli)
      R5: KC Fine-Tune (ratio-based scoring)
      R6: Kelly/DynComp Test
    """

    def __init__(self, symbol: str, timeframe: str = "3m", days: int = 180,
                 leverage: int = 25, event_callback: Optional[Callable] = None,
                 n_jobs: int = 1):
        super().__init__(symbol, timeframe, days, leverage, event_callback)
        self.n_jobs = n_jobs
        self.running = True
        self._log_buffer = []
        self._lock = threading.Lock()

        # Pipeline state
        self.current_round = 0
        self.progress = 0

        # Data cache (prepare_data'da doldurulur)
        self._df_is = None
        self._df_oos = None
        self._src_is = None
        self._src_oos = None
        self._params = None

        # PMax cache (R3 sonrası kilitlenir)
        self._locked_pmax_is = None  # (pmax, mavg, direction) tuple
        self._locked_pmax_oos = None

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        with self._lock:
            self._log_buffer.append(entry)
            if len(self._log_buffer) > 300:
                self._log_buffer = self._log_buffer[-300:]
        logger.info(msg)
        self.emit("log", {"message": msg, "level": "info"})

    def get_rounds(self) -> list[RoundConfig]:
        return [
            RoundConfig(1, "PMAX KESIF", 200, "Geniş PMax parametre araması"),
            RoundConfig(2, "FINE-TUNE", 300, "PMax parametrelerini daralt", skippable=True),
            RoundConfig(3, "DD OPTIMIZE", 400, "Drawdown penalty ile optimize"),
            RoundConfig(4, "KC OPTIMIZE", 300, "Keltner Channel araması (PMax kilitli)"),
            RoundConfig(5, "KC FINE-TUNE", 200, "KC ratio-based fine-tune", skippable=True),
        ]

    def get_data_type(self) -> str:
        return "klines"

    def get_available_timeframes(self) -> list[str]:
        return ["1m", "3m", "5m", "15m"]

    def prepare_data(self, data_path: str = None) -> dict:
        """Veri yükle, IS/OOS böl, sabit indikatörleri pre-compute et."""
        if data_path is None:
            data_path = f"data/{self.symbol}_{self.timeframe}_{self.days}d.parquet"

        dp = Path(data_path)
        dp.parent.mkdir(exist_ok=True)

        if dp.exists():
            df = pd.read_parquet(dp)
            self._log(f"Cache: {len(df)} mum")
        else:
            df = fetch_klines(self.symbol, self.timeframe, self.days)
            df.to_parquet(dp, index=False)
            self._log(f"İndirildi: {len(df)} mum")

        if len(df) < 500:
            raise ValueError(f"Yetersiz veri: {len(df)} mum")

        split_idx = int(len(df) * TRAIN_RATIO)
        self._df_is = df.iloc[:split_idx].reset_index(drop=True)
        self._df_oos = df.iloc[split_idx:].reset_index(drop=True)
        self._log(f"IS: {len(self._df_is)} | OOS: {len(self._df_oos)}")

        self._params = DEFAULT_PARAMS.copy()
        source = self._params.get("source", "hl2")
        self._src_is = _get_source(self._df_is, source)
        self._src_oos = _get_source(self._df_oos, source)

        return {"df_is": self._df_is, "df_oos": self._df_oos}

    def _compute_pmax(self, src, h, l, c, pmax_params):
        """Adaptive PMax hesapla."""
        ma_type = self._params.get("ma_type", "EMA")
        atr_period = self._params.get("atr_period", 10)
        atr_mult = self._params.get("atr_multiplier", 3.0)
        ma_length = self._params.get("ma_length", 10)

        p, m, d, _, _, _ = adaptive_pmax_continuous(
            src, h, l, c, ma_type=ma_type,
            base_atr_period=atr_period, base_atr_multiplier=atr_mult, base_ma_length=ma_length,
            lookback=pmax_params["vol_lookback"], flip_window=pmax_params["flip_window"],
            mult_base=pmax_params["mult_base"], mult_scale=pmax_params["mult_scale"],
            ma_base=pmax_params["ma_base"], ma_scale=pmax_params["ma_scale"],
            atr_base=pmax_params["atr_base"], atr_scale=pmax_params["atr_scale"],
            update_interval=pmax_params["update_interval"],
        )
        return p.values, m.values, d.values

    def _lock_pmax(self, pmax_params):
        """PMax'i kilitle (R3 sonrası). R4, R5, Kelly bu cache'i kullanır."""
        self._log("PMax kilitleniyor ve cache'leniyor...")
        h_is, l_is, c_is = self._df_is["high"], self._df_is["low"], self._df_is["close"]
        h_oos, l_oos, c_oos = self._df_oos["high"], self._df_oos["low"], self._df_oos["close"]

        self._locked_pmax_is = self._compute_pmax(self._src_is, h_is, l_is, c_is, pmax_params)
        self._locked_pmax_oos = self._compute_pmax(self._src_oos, h_oos, l_oos, c_oos, pmax_params)
        self._log("PMax cache hazır")

    def _run_pmax_round(self, round_num, n_trials, objective_fn) -> Optional[RoundResult]:
        """Tek bir Optuna round'u çalıştır, en iyi trial'ı otomatik seç."""
        round_cfg = self.get_rounds()[round_num - 1]
        self.current_round = round_num
        self._log(f"=== ROUND {round_num}: {round_cfg.name} ({n_trials} trial) ===")
        self.emit("round_started", {
            "round": round_num, "name": round_cfg.name, "n_trials": n_trials
        })

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0),
        )
        trials_data = []
        trial_lock = threading.Lock()

        def callback(study, trial):
            if not self.running:
                study.stop()
                return
            if trial.value and trial.value > -999:
                a = trial.user_attrs
                td = {
                    "tid": trial.number,
                    "score": round(trial.value, 2),
                    "is_net": round(a.get("is_net", 0), 1),
                    "oos_net": round(a.get("oos_net", 0), 1),
                    "wr": round(a.get("oos_wr", 0), 1),
                    "dd": round(a.get("oos_dd", 0), 1),
                    "trades": a.get("oos_trades", 0),
                    "params": trial.params,
                }
                with trial_lock:
                    trials_data.append(td)
                ratio = round(td["oos_net"] / td["dd"], 1) if td["dd"] > 0 else 0
                self.emit("trial_completed", {
                    "round": round_num, "trial_num": trial.number,
                    "score": td["score"], "oos_net": td["oos_net"],
                    "dd": td["dd"], "ratio": ratio,
                })

        study.optimize(objective_fn, n_trials=n_trials, callbacks=[callback],
                       n_jobs=self.n_jobs)

        if not self.running or not trials_data:
            return None

        # Otomatik seçim — İNSAN MÜDAHALESİ YOK
        idx, selected = auto_select_best(trials_data)
        self._log(f"=== {round_cfg.name} TAMAMLANDI — {len(trials_data)} trial ===")
        self._log(f"Otomatik seçim: #{selected['tid']} OOS={selected['oos_net']:+.1f}% "
                  f"DD={selected['dd']:.1f}% Ratio={selected['ratio']}")

        # Trial parametrelerini al
        params = None
        for t in study.trials:
            if t.number == selected["tid"]:
                params = t.params
                break
        if params is None:
            params = study.best_params

        # Top 10'u broadcast et
        top10 = sorted(trials_data, key=lambda x: x.get("ratio", 0), reverse=True)[:10]
        self.emit("round_completed", {
            "round": round_num, "name": round_cfg.name,
            "top10": top10, "selected": selected,
        })

        # Kaydet
        round_key = f"r{round_num}"
        self.selected_params[round_key] = params
        self.selected_metrics[round_key] = {
            "tid": selected["tid"],
            "oos_net": selected["oos_net"],
            "dd": selected["dd"],
            "wr": selected["wr"],
            "ratio": selected["ratio"],
            "trades": selected["trades"],
        }

        save_path = RESULTS_DIR / f"{self.symbol}_round{round_num}_trials.json"
        with open(save_path, "w") as f:
            json.dump({"trials": trials_data, "selected": selected, "params": params}, f, indent=2)

        trial_result = TrialResult(
            tid=selected["tid"], score=selected["score"],
            is_net=selected["is_net"], oos_net=selected["oos_net"],
            wr=selected["wr"], dd=selected["dd"],
            trades=selected["trades"], params=params,
        )
        return RoundResult(round_num, round_cfg.name,
                          [TrialResult(**td) for td in trials_data[:10]],
                          trial_result, params)

    def run_full_pipeline(self, trial_counts: Optional[dict] = None) -> dict:
        """Tam otomatik R1→R5 pipeline. Sıfır insan müdahalesi.

        Args:
            trial_counts: Round bazlı trial sayıları {1: 200, 2: 300, ...}

        Returns:
            Final sonuç dict'i
        """
        self.running = True
        self.selected_params = {}
        self.selected_metrics = {}

        defaults = {1: 200, 2: 300, 3: 400, 4: 300, 5: 200}
        tc = trial_counts or defaults

        try:
            # STAGE 0: DATA
            self._log(f"=== VERI HAZIRLANIYOR ({self.symbol}, {self.timeframe}, {self.days} gün) ===")
            self.prepare_data()

            h_is, l_is, c_is = self._df_is["high"], self._df_is["low"], self._df_is["close"]
            h_oos, l_oos, c_oos = self._df_oos["high"], self._df_oos["low"], self._df_oos["close"]
            ma_type = self._params.get("ma_type", "EMA")
            atr_period = self._params.get("atr_period", 10)
            atr_mult = self._params.get("atr_multiplier", 3.0)
            ma_length = self._params.get("ma_length", 10)
            params = self._params

            # ============================================================
            # ROUND 1: PMAX KESIF
            # ============================================================
            self.progress = 5

            def r1_obj(trial):
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
                    pi, mi, di = self._compute_pmax(self._src_is, h_is, l_is, c_is, cp)
                    is_r = run_backtest_with_pmax(self._df_is, pi, mi, di, params, "r1")
                    if is_r["total_trades"] < 10: return -999
                    # IS filtre — kötüyse OOS yapma (pruning)
                    trial.report(is_r["net_pct"], step=0)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    po, mo, do2 = self._compute_pmax(self._src_oos, h_oos, l_oos, c_oos, cp)
                    oos_r = run_backtest_with_pmax(self._df_oos, po, mo, do2, params, "r1")
                    gap = abs(is_r["net_pct"] - oos_r["net_pct"])
                    score = (oos_r["net_pct"] * 0.6) + (is_r["net_pct"] * 0.2) + (oos_r["win_rate"] * 0.3) - gap * 0.2
                    trial.set_user_attr("is_net", is_r["net_pct"])
                    trial.set_user_attr("oos_net", oos_r["net_pct"])
                    trial.set_user_attr("oos_wr", oos_r["win_rate"])
                    trial.set_user_attr("oos_dd", oos_r["max_drawdown"])
                    trial.set_user_attr("oos_trades", oos_r["total_trades"])
                    return score
                except optuna.TrialPruned:
                    raise
                except:
                    return -999

            r1_result = self._run_pmax_round(1, tc.get(1, 200), r1_obj)
            if r1_result is None: return self._make_error("R1 durdu")
            r1_params = r1_result.selected_params
            self.progress = 20

            # ============================================================
            # ROUND 2: FINE-TUNE
            # ============================================================
            def r2_obj(trial):
                base = r1_params
                lo, hi = make_narrow(base["vol_lookback"], 0.3, 20, 1500, is_int=True)
                cp = {"vol_lookback": trial.suggest_int("vol_lookback", lo, hi, step=10)}
                lo, hi = make_narrow(base["flip_window"], 0.3, 20, 500, is_int=True)
                cp["flip_window"] = trial.suggest_int("flip_window", lo, hi, step=10)
                lo, hi = make_narrow(base["mult_base"], 0.25, 0.5, 4.0, step=0.25)
                cp["mult_base"] = trial.suggest_float("mult_base", lo, hi, step=0.25)
                lo, hi = make_narrow(base["mult_scale"], 0.3, 0.25, 3.0, step=0.25)
                cp["mult_scale"] = trial.suggest_float("mult_scale", lo, hi, step=0.25)
                lo, hi = make_narrow(base["ma_base"], 0.25, 3, 20, is_int=True)
                cp["ma_base"] = trial.suggest_int("ma_base", lo, hi)
                lo, hi = make_narrow(base["ma_scale"], 0.25, 0.5, 6.0, step=0.5)
                cp["ma_scale"] = trial.suggest_float("ma_scale", lo, hi, step=0.5)
                lo, hi = make_narrow(base["atr_base"], 0.25, 3, 20, is_int=True)
                cp["atr_base"] = trial.suggest_int("atr_base", lo, hi)
                lo, hi = make_narrow(base["atr_scale"], 0.3, 0.5, 4.0, step=0.5)
                cp["atr_scale"] = trial.suggest_float("atr_scale", lo, hi, step=0.5)
                lo, hi = make_narrow(base["update_interval"], 0.25, 1, 60, is_int=True)
                cp["update_interval"] = trial.suggest_int("update_interval", lo, hi)
                try:
                    pi, mi, di = self._compute_pmax(self._src_is, h_is, l_is, c_is, cp)
                    is_r = run_backtest_with_pmax(self._df_is, pi, mi, di, params, "r2")
                    if is_r["total_trades"] < 10: return -999
                    trial.report(is_r["net_pct"], step=0)
                    if trial.should_prune(): raise optuna.TrialPruned()
                    po, mo, do2 = self._compute_pmax(self._src_oos, h_oos, l_oos, c_oos, cp)
                    oos_r = run_backtest_with_pmax(self._df_oos, po, mo, do2, params, "r2")
                    gap = abs(is_r["net_pct"] - oos_r["net_pct"])
                    score = (oos_r["net_pct"] * 0.6) + (is_r["net_pct"] * 0.2) + (oos_r["win_rate"] * 0.3) - gap * 0.3
                    trial.set_user_attr("is_net", is_r["net_pct"])
                    trial.set_user_attr("oos_net", oos_r["net_pct"])
                    trial.set_user_attr("oos_wr", oos_r["win_rate"])
                    trial.set_user_attr("oos_dd", oos_r["max_drawdown"])
                    trial.set_user_attr("oos_trades", oos_r["total_trades"])
                    return score
                except optuna.TrialPruned:
                    raise
                except:
                    return -999

            r2_result = self._run_pmax_round(2, tc.get(2, 300), r2_obj)
            r2_params = r2_result.selected_params if r2_result else r1_params
            self.progress = 40

            # ============================================================
            # ROUND 3: DD OPTIMIZE
            # ============================================================
            def r3_obj(trial):
                base = r2_params
                lo, hi = make_narrow(base["vol_lookback"], 0.2, 20, 1500, is_int=True)
                cp = {"vol_lookback": trial.suggest_int("vol_lookback", lo, hi, step=5)}
                lo, hi = make_narrow(base["flip_window"], 0.2, 20, 500, is_int=True)
                cp["flip_window"] = trial.suggest_int("flip_window", lo, hi, step=5)
                lo, hi = make_narrow(base["mult_base"], 0.15, 0.5, 4.0, step=0.25)
                cp["mult_base"] = trial.suggest_float("mult_base", lo, hi, step=0.25)
                lo, hi = make_narrow(base["mult_scale"], 0.2, 0.25, 3.0, step=0.25)
                cp["mult_scale"] = trial.suggest_float("mult_scale", lo, hi, step=0.25)
                lo, hi = make_narrow(base["ma_base"], 0.15, 3, 20, is_int=True)
                cp["ma_base"] = trial.suggest_int("ma_base", lo, hi)
                lo, hi = make_narrow(base["ma_scale"], 0.15, 0.5, 6.0, step=0.5)
                cp["ma_scale"] = trial.suggest_float("ma_scale", lo, hi, step=0.5)
                lo, hi = make_narrow(base["atr_base"], 0.15, 3, 20, is_int=True)
                cp["atr_base"] = trial.suggest_int("atr_base", lo, hi)
                lo, hi = make_narrow(base["atr_scale"], 0.2, 0.5, 4.0, step=0.5)
                cp["atr_scale"] = trial.suggest_float("atr_scale", lo, hi, step=0.5)
                lo, hi = make_narrow(base["update_interval"], 0.15, 1, 60, is_int=True)
                cp["update_interval"] = trial.suggest_int("update_interval", lo, hi)
                try:
                    pi, mi, di = self._compute_pmax(self._src_is, h_is, l_is, c_is, cp)
                    is_r = run_backtest_with_pmax(self._df_is, pi, mi, di, params, "r3")
                    if is_r["total_trades"] < 10: return -999
                    trial.report(is_r["net_pct"], step=0)
                    if trial.should_prune(): raise optuna.TrialPruned()
                    po, mo, do2 = self._compute_pmax(self._src_oos, h_oos, l_oos, c_oos, cp)
                    oos_r = run_backtest_with_pmax(self._df_oos, po, mo, do2, params, "r3")
                    gap = abs(is_r["net_pct"] - oos_r["net_pct"])
                    dd_pen = 0
                    if oos_r["max_drawdown"] > 50: dd_pen = (oos_r["max_drawdown"] - 50) * 1.5
                    if is_r["max_drawdown"] > 50: dd_pen += (is_r["max_drawdown"] - 50) * 0.5
                    score = (oos_r["net_pct"] * 0.7) + (is_r["net_pct"] * 0.15) + (oos_r["win_rate"] * 0.3) - dd_pen - gap * 0.3
                    trial.set_user_attr("is_net", is_r["net_pct"])
                    trial.set_user_attr("oos_net", oos_r["net_pct"])
                    trial.set_user_attr("oos_wr", oos_r["win_rate"])
                    trial.set_user_attr("oos_dd", oos_r["max_drawdown"])
                    trial.set_user_attr("oos_trades", oos_r["total_trades"])
                    return score
                except optuna.TrialPruned:
                    raise
                except:
                    return -999

            r3_result = self._run_pmax_round(3, tc.get(3, 400), r3_obj)
            r3_params = r3_result.selected_params if r3_result else r2_params
            self._log(f"PMax LOCKED: {json.dumps(r3_params)}")
            self.progress = 65

            # PMax'i kilitle — R4, R5, Kelly hep bu cache'i kullanacak
            self._lock_pmax(r3_params)
            pi, mi, di = self._locked_pmax_is
            po, mo, do2 = self._locked_pmax_oos

            # ============================================================
            # ROUND 4: KC OPTIMIZE
            # ============================================================
            def r4_obj(trial):
                kp = {
                    "kc_length": trial.suggest_int("kc_length", 5, 50),
                    "kc_multiplier": trial.suggest_float("kc_multiplier", 0.5, 4.0, step=0.1),
                    "kc_atr_period": trial.suggest_int("kc_atr_period", 3, 30),
                    "max_dca_steps": trial.suggest_int("max_dca_steps", 1, 5),
                    "tp_close_percent": trial.suggest_float("tp_close_percent", 0.05, 0.50, step=0.05),
                }
                tp = params.copy(); tp.update(kp)
                try:
                    is_r = run_backtest_with_pmax(self._df_is, pi, mi, di, tp, "r4")
                    if is_r["total_trades"] < 10: return -999
                    trial.report(is_r["net_pct"], step=0)
                    if trial.should_prune(): raise optuna.TrialPruned()
                    oos_r = run_backtest_with_pmax(self._df_oos, po, mo, do2, tp, "r4")
                    gap = abs(is_r["net_pct"] - oos_r["net_pct"])
                    dd_pen = 0
                    if oos_r["max_drawdown"] > 50: dd_pen = (oos_r["max_drawdown"] - 50) * 1.5
                    if is_r["max_drawdown"] > 50: dd_pen += (is_r["max_drawdown"] - 50) * 0.5
                    score = (oos_r["net_pct"] * 0.7) + (is_r["net_pct"] * 0.15) + (oos_r["win_rate"] * 0.3) - dd_pen - gap * 0.3
                    trial.set_user_attr("is_net", is_r["net_pct"])
                    trial.set_user_attr("oos_net", oos_r["net_pct"])
                    trial.set_user_attr("oos_wr", oos_r["win_rate"])
                    trial.set_user_attr("oos_dd", oos_r["max_drawdown"])
                    trial.set_user_attr("oos_trades", oos_r["total_trades"])
                    return score
                except optuna.TrialPruned:
                    raise
                except:
                    return -999

            r4_result = self._run_pmax_round(4, tc.get(4, 300), r4_obj)
            if r4_result is None: return self._make_error("R4 durdu")
            r4_params = r4_result.selected_params
            self.progress = 85

            # ============================================================
            # ROUND 5: KC FINE-TUNE
            # ============================================================
            def r5_obj(trial):
                base = r4_params
                lo, hi = make_narrow(base["kc_length"], 0.3, 3, 50, is_int=True)
                kp = {"kc_length": trial.suggest_int("kc_length", lo, hi)}
                lo, hi = make_narrow(base["kc_multiplier"], 0.25, 0.3, 4.0, step=0.1)
                kp["kc_multiplier"] = trial.suggest_float("kc_multiplier", lo, hi, step=0.1)
                lo, hi = make_narrow(base["kc_atr_period"], 0.25, 3, 30, is_int=True)
                kp["kc_atr_period"] = trial.suggest_int("kc_atr_period", lo, hi)
                lo, hi = make_narrow(base["max_dca_steps"], 0.3, 1, 7, is_int=True)
                kp["max_dca_steps"] = trial.suggest_int("max_dca_steps", lo, hi)
                lo, hi = make_narrow(base["tp_close_percent"], 0.25, 0.05, 0.60, step=0.05)
                kp["tp_close_percent"] = trial.suggest_float("tp_close_percent", lo, hi, step=0.05)
                tp2 = params.copy(); tp2.update(kp)
                try:
                    is_r = run_backtest_with_pmax(self._df_is, pi, mi, di, tp2, "r5")
                    if is_r["total_trades"] < 10: return -999
                    trial.report(is_r["net_pct"], step=0)
                    if trial.should_prune(): raise optuna.TrialPruned()
                    oos_r = run_backtest_with_pmax(self._df_oos, po, mo, do2, tp2, "r5")
                    oos_dd = max(oos_r["max_drawdown"], 1)
                    ratio_score = (oos_r["net_pct"] / oos_dd) * oos_r["win_rate"] * 0.01
                    gap = abs(is_r["net_pct"] - oos_r["net_pct"])
                    score = ratio_score - gap * 0.05
                    trial.set_user_attr("is_net", is_r["net_pct"])
                    trial.set_user_attr("oos_net", oos_r["net_pct"])
                    trial.set_user_attr("oos_wr", oos_r["win_rate"])
                    trial.set_user_attr("oos_dd", oos_r["max_drawdown"])
                    trial.set_user_attr("oos_trades", oos_r["total_trades"])
                    return score
                except optuna.TrialPruned:
                    raise
                except:
                    return -999

            r5_result = self._run_pmax_round(5, tc.get(5, 200), r5_obj)
            if r5_result and r5_result.selected_params:
                r4_params = r5_result.selected_params  # R5 sonucu R4'ü override eder
                self.selected_params["r4"] = r4_params
                self._log("KC Fine-Tune sonucu R4'ü override etti")

            # ============================================================
            # SAVE FINAL
            # ============================================================
            self.progress = 100
            final = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "days": self.days,
                "leverage": self.leverage,
                "pmax_params": r3_params,
                "kc_params": r4_params,
                "selected_per_round": self.selected_params,
                "selected_metrics": self.selected_metrics,
            }

            result_path = RESULTS_DIR / f"{self.symbol}_full_optimization.json"
            with open(result_path, "w") as f:
                json.dump(final, f, indent=2)

            self._log(f"=== TAMAMLANDI — {result_path} ===")
            self.emit("pair_completed", {
                "symbol": self.symbol,
                "strategy": "pmax_kc",
                "timeframe": self.timeframe,
                "final_metrics": self.selected_metrics,
                "pmax_params": r3_params,
                "kc_params": r4_params,
            })

            return final

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"HATA: {e}")
            return self._make_error(str(e))
        finally:
            self.running = False

    def _make_error(self, msg: str) -> dict:
        return {"error": msg, "symbol": self.symbol}

    # StrategyOptimizer abstract methods (delegated to run_full_pipeline)
    def create_objective(self, round_num, prepared_data):
        raise NotImplementedError("Use run_full_pipeline() directly")

    def extract_params(self, round_num, trial_params):
        return trial_params

    def run_final_backtest(self, prepared_data, params):
        raise NotImplementedError("Use run_full_pipeline() directly")

    def stop(self):
        """Pipeline'ı gracefully durdur."""
        self.running = False
        self._log("Durdurma sinyali alındı — mevcut trial bittikten sonra duracak")
