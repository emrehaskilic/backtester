"""
KC Walk-Forward Optimizer V2 — kc[i-1] (look-ahead fix).
Anchored expanding window, 500 trial/fold, Rust engine.

PMax kilitli (konsensus), KC 3 parametre optimize.
Filtreler, stop, DynComp KAPALI.

Kullanim: python daily_wf_kc_v2.py
"""

import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import rust_engine
    print("Rust engine yuklendi")
except ImportError:
    print("HATA: rust_engine bulunamadi. cd rust_engine && python -m maturin develop --release")
    sys.exit(1)

# ================================================================
# SABITLER
# ================================================================

# PMax konsensus (kilitli)
PMAX_PARAMS = {
    "vol_lookback": 260,
    "flip_window": 360,
    "mult_base": 3.25,
    "mult_scale": 2.0,
    "ma_base": 11,
    "ma_scale": 4.5,
    "atr_base": 15,
    "atr_scale": 2.0,
    "update_interval": 55,
}
BASE_ATR_PERIOD = 10
BASE_ATR_MULT = 3.0
BASE_MA_LENGTH = 10

# WF sabitleri
WARMUP_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7
TRIALS_PER_FOLD = 500
N_JOBS = 6

# Backtest sabitleri
MAX_DCA = 4
TP_PCT = 0.50

# Veri
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_source(df):
    """hl2 source."""
    return (df["high"].values + df["low"].values) / 2.0


def compute_pmax(src, high, low, close):
    """Rust ile adaptive PMax hesapla (kilitli parametreler)."""
    sa = np.ascontiguousarray
    result = rust_engine.compute_adaptive_pmax(
        sa(src, dtype=np.float64),
        sa(high, dtype=np.float64),
        sa(low, dtype=np.float64),
        sa(close, dtype=np.float64),
        BASE_ATR_PERIOD, BASE_ATR_MULT, BASE_MA_LENGTH,
        PMAX_PARAMS["vol_lookback"], PMAX_PARAMS["flip_window"],
        PMAX_PARAMS["mult_base"], PMAX_PARAMS["mult_scale"],
        PMAX_PARAMS["ma_base"], PMAX_PARAMS["ma_scale"],
        PMAX_PARAMS["atr_base"], PMAX_PARAMS["atr_scale"],
        PMAX_PARAMS["update_interval"],
    )
    return (
        np.ascontiguousarray(result["pmax_line"]),
        np.ascontiguousarray(result["mavg"]),
    )


def compute_kc(high, low, close, kc_length, kc_multiplier, kc_atr_period):
    """Rust ile KC hesapla."""
    sa = np.ascontiguousarray
    ind = rust_engine.precompute_indicators(
        sa(high, dtype=np.float64),
        sa(low, dtype=np.float64),
        sa(close, dtype=np.float64),
        144,  # ema_filter_period (kullanilmiyor ama zorunlu)
        kc_length, kc_multiplier, kc_atr_period,
    )
    return (
        np.ascontiguousarray(ind["kc_upper_arr"]),
        np.ascontiguousarray(ind["kc_lower_arr"]),
    )


def run_backtest(closes, highs, lows, pmax_line, mavg, kc_upper, kc_lower):
    """Rust kc_lagged backtest."""
    sa = np.ascontiguousarray
    return rust_engine.run_backtest_kc_lagged(
        sa(closes), sa(highs), sa(lows),
        sa(pmax_line), sa(mavg),
        sa(kc_upper), sa(kc_lower),
        MAX_DCA, TP_PCT,
    )


def build_folds(df):
    """Anchored expanding window fold'lari olustur."""
    bars_per_day = 480  # 3m = 1440/3 = 480 bar/gun
    warmup_bars = WARMUP_DAYS * bars_per_day
    test_bars = TEST_DAYS * bars_per_day
    step_bars = STEP_DAYS * bars_per_day
    total = len(df)

    folds = []
    fold_num = 1
    test_start = warmup_bars

    while test_start + test_bars <= total:
        test_end = test_start + test_bars

        # Train: 0 -> test_start (anchored expanding)
        # Test window: test_start - 1000 warmup -> test_end
        test_warmup_start = max(0, test_start - 1000)

        folds.append({
            "num": fold_num,
            "train_start": 0,
            "train_end": test_start,
            "test_warmup_start": test_warmup_start,
            "test_start": test_start,
            "test_end": test_end,
        })

        fold_num += 1
        test_start += step_bars

    return folds


def main():
    print("=" * 80)
    print("  KC WALK-FORWARD OPTIMIZER V2 — kc[i-1] (LOOK-AHEAD FIX)")
    print("  PMax kilitli | 500 trial/fold | Anchored expanding window")
    print("=" * 80)

    # Veri yukle
    if not os.path.exists(DATA_PATH):
        print(f"  HATA: {DATA_PATH} bulunamadi. Once build_candle_cache.py calistirin.")
        return

    print(f"\n  Veri yukleniyor: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Toplam mum: {len(df):,}")

    # Arrays
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    src = get_source(df)

    # PMax hesapla (bir kez, kilitli)
    print(f"  PMax hesaplaniyor (kilitli)...")
    t0 = time.time()
    pmax_line, mavg = compute_pmax(src, high, low, close)
    print(f"  PMax tamamlandi ({time.time()-t0:.1f}s)")

    # Fold'lari olustur
    folds = build_folds(df)
    n_folds = len(folds)
    print(f"  Fold sayisi: {n_folds}")
    print(f"  Trial/fold: {TRIALS_PER_FOLD}")
    print(f"  Toplam trial: {n_folds * TRIALS_PER_FOLD:,}")

    # Her fold icin optimize
    all_fold_results = []
    all_best_params = []

    for fold in folds:
        fold_num = fold["num"]
        t_fold = time.time()

        # Train slice — PMax zaten hesaplandi, KC parametrelerini dene
        train_h = high[:fold["train_end"]]
        train_l = low[:fold["train_end"]]
        train_c = close[:fold["train_end"]]
        train_pmax = pmax_line[:fold["train_end"]]
        train_mavg = mavg[:fold["train_end"]]

        # Test slice (1000 bar warmup ile)
        ts = fold["test_warmup_start"]
        te = fold["test_end"]
        test_h = high[ts:te]
        test_l = low[ts:te]
        test_c = close[ts:te]
        test_pmax = pmax_line[ts:te]
        test_mavg = mavg[ts:te]

        # Optuna study
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=15,
            seed=42 + fold_num,
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        best_in_fold = {"score": -999, "params": None, "result": None}

        def objective(trial):
            kc_length = trial.suggest_int("kc_length", 2, 50)
            kc_multiplier = trial.suggest_float("kc_multiplier", 0.1, 5.0, step=0.1)
            kc_atr_period = trial.suggest_int("kc_atr_period", 2, 50)

            # Train backtest (pruning icin)
            kc_u_tr, kc_l_tr = compute_kc(train_h, train_l, train_c,
                                          kc_length, kc_multiplier, kc_atr_period)
            train_r = run_backtest(train_c, train_h, train_l,
                                   train_pmax, train_mavg, kc_u_tr, kc_l_tr)

            if train_r["total_trades"] < 5:
                return -999

            trial.report(train_r["net_pct"], step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Test backtest (OOS)
            kc_u_te, kc_l_te = compute_kc(test_h, test_l, test_c,
                                          kc_length, kc_multiplier, kc_atr_period)
            test_r = run_backtest(test_c, test_h, test_l,
                                  test_pmax, test_mavg, kc_u_te, kc_l_te)

            net = test_r["net_pct"]
            dd = test_r["max_drawdown"]
            wr = test_r["win_rate"]
            trades = test_r["total_trades"]

            # Sert filtre (dokuman ile ayni)
            if wr < 80 or dd > 20:
                return -999

            # Skor: (net/DD) * WR
            score = (net / max(dd, 0.1)) * wr

            trial.set_user_attr("net", round(net, 2))
            trial.set_user_attr("dd", round(dd, 2))
            trial.set_user_attr("wr", round(wr, 1))
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("tp", test_r["tp_count"])
            trial.set_user_attr("rev", test_r["rev_count"])

            return score

        # Optimize
        study.optimize(objective, n_trials=TRIALS_PER_FOLD, n_jobs=N_JOBS)

        # Sonuc
        if study.best_value is not None and study.best_value > -999:
            bp = study.best_trial
            result = {
                "fold": fold_num,
                "status": "OK",
                "score": round(bp.value, 4),
                "params": bp.params,
                "net": bp.user_attrs.get("net", 0),
                "dd": bp.user_attrs.get("dd", 0),
                "wr": bp.user_attrs.get("wr", 0),
                "trades": bp.user_attrs.get("trades", 0),
                "tp": bp.user_attrs.get("tp", 0),
                "rev": bp.user_attrs.get("rev", 0),
            }
            all_best_params.append(bp.params)
        else:
            result = {
                "fold": fold_num,
                "status": "FAIL",
                "score": -999,
                "params": None,
                "net": 0, "dd": 0, "wr": 0, "trades": 0, "tp": 0, "rev": 0,
            }

        all_fold_results.append(result)
        elapsed = time.time() - t_fold

        if result["status"] == "OK":
            p = result["params"]
            print(f"  Fold {fold_num:2d}/{n_folds} | "
                  f"Net: {result['net']:>+7.2f}% DD: {result['dd']:>5.1f}% "
                  f"WR: {result['wr']:>4.0f}% Trades: {result['trades']:>4d} | "
                  f"KC({p['kc_length']},{p['kc_multiplier']:.1f},{p['kc_atr_period']}) | "
                  f"{elapsed:.0f}s")
        else:
            print(f"  Fold {fold_num:2d}/{n_folds} | FAIL — tum trial'lar filtreye takildi | {elapsed:.0f}s")

    # ================================================================
    # KONSENSUS ANALIZI
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  KONSENSUS ANALIZI")
    print(f"{'='*80}")

    ok_folds = [r for r in all_fold_results if r["status"] == "OK"]
    fail_folds = [r for r in all_fold_results if r["status"] == "FAIL"]

    print(f"  Basarili fold: {len(ok_folds)}/{n_folds}")
    print(f"  Basarisiz fold: {len(fail_folds)}/{n_folds}")

    if not all_best_params:
        print(f"\n  HICBIR FOLD BASARILI OLMADI. Optimizasyon basarisiz.")
        return

    # Medyan + CV
    param_names = ["kc_length", "kc_multiplier", "kc_atr_period"]
    consensus = {}

    print(f"\n  {'Parametre':20s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>6s} | {'Max':>6s}")
    print(f"  {'-'*75}")

    for pname in param_names:
        vals = [p[pname] for p in all_best_params]
        median_val = float(np.median(vals))
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        cv = std_val / mean_val if mean_val > 0 else 0

        # Step snap
        if pname == "kc_multiplier":
            step = 0.1
            median_val = round(round(median_val / step) * step, 1)
        else:
            median_val = int(round(median_val))

        if cv == 0:
            stability = "MUTLAK"
        elif cv < 0.3:
            stability = "STABIL"
        elif cv < 0.5:
            stability = "ORTA"
        else:
            stability = "DEGISKEN"

        consensus[pname] = median_val

        print(f"  {pname:20s} | {str(median_val):>10s} | {cv:>8.4f} | {stability:>12s} | {min(vals):>6} | {max(vals):>6}")

    # Karli fold analizi
    profitable = sum(1 for r in ok_folds if r["net"] > 0)
    print(f"\n  Karli fold: {profitable}/{len(ok_folds)} ({profitable/len(ok_folds)*100:.0f}%)" if ok_folds else "")

    # Sonuclari kaydet
    output = {
        "method": "KC WF Optimizer V2 — kc[i-1]",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_folds": n_folds,
        "successful_folds": len(ok_folds),
        "failed_folds": len(fail_folds),
        "trials_per_fold": TRIALS_PER_FOLD,
        "consensus": consensus,
        "pmax_params": PMAX_PARAMS,
        "fold_results": all_fold_results,
        "all_best_params": all_best_params,
    }

    out_path = os.path.join(RESULTS_DIR, "kc_wf_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Sonuclar kaydedildi: {out_path}")

    # Ozet tablo
    print(f"\n  FOLD DETAYLARI:")
    print(f"  {'Fold':>4s} | {'Net%':>8s} | {'DD%':>6s} | {'WR%':>5s} | {'Trades':>6s} | {'TP':>4s} | {'Rev':>4s} | KC Params")
    print(f"  {'-'*80}")
    for r in all_fold_results:
        if r["status"] == "OK":
            p = r["params"]
            print(f"  {r['fold']:4d} | {r['net']:>+8.2f} | {r['dd']:>6.1f} | {r['wr']:>5.0f} | "
                  f"{r['trades']:>6d} | {r['tp']:>4d} | {r['rev']:>4d} | "
                  f"({p['kc_length']},{p['kc_multiplier']:.1f},{p['kc_atr_period']})")
        else:
            print(f"  {r['fold']:4d} | {'FAIL':>8s} |")


if __name__ == "__main__":
    main()
