"""
Graduated DCA Walk-Forward Optimizer V2 — kc[i-1].
PMax + KC kilitli. 4 DCA multiplier optimize.
Kisit: m1 <= m2 <= m3 <= m4 (artan sira).

Kullanim: python daily_wf_dca_v2.py
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
    print("HATA: rust_engine bulunamadi.")
    sys.exit(1)

# ================================================================
# KILITLI PARAMETRELER
# ================================================================

# PMax konsensus (Adim 1)
PMAX_PARAMS = {
    "vol_lookback": 260, "flip_window": 360,
    "mult_base": 3.25, "mult_scale": 2.0,
    "ma_base": 11, "ma_scale": 4.5,
    "atr_base": 15, "atr_scale": 2.0,
    "update_interval": 55,
}
BASE_ATR_PERIOD = 10
BASE_ATR_MULT = 3.0
BASE_MA_LENGTH = 10

# KC konsensus (Adim 2, kc[i-1])
KC_LENGTH = 20
KC_MULTIPLIER = 2.0
KC_ATR_PERIOD = 28

# Sabitler
MAX_DCA = 4
TP_PCT = 0.50
WARMUP_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7
TRIALS_PER_FOLD = 500
N_JOBS = 6

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_source(df):
    return (df["high"].values + df["low"].values) / 2.0


def compute_pmax(src, high, low, close):
    sa = np.ascontiguousarray
    result = rust_engine.compute_adaptive_pmax(
        sa(src, dtype=np.float64), sa(high, dtype=np.float64),
        sa(low, dtype=np.float64), sa(close, dtype=np.float64),
        BASE_ATR_PERIOD, BASE_ATR_MULT, BASE_MA_LENGTH,
        PMAX_PARAMS["vol_lookback"], PMAX_PARAMS["flip_window"],
        PMAX_PARAMS["mult_base"], PMAX_PARAMS["mult_scale"],
        PMAX_PARAMS["ma_base"], PMAX_PARAMS["ma_scale"],
        PMAX_PARAMS["atr_base"], PMAX_PARAMS["atr_scale"],
        PMAX_PARAMS["update_interval"],
    )
    return np.ascontiguousarray(result["pmax_line"]), np.ascontiguousarray(result["mavg"])


def compute_kc(high, low, close):
    sa = np.ascontiguousarray
    ind = rust_engine.precompute_indicators(
        sa(high, dtype=np.float64), sa(low, dtype=np.float64),
        sa(close, dtype=np.float64),
        144, KC_LENGTH, KC_MULTIPLIER, KC_ATR_PERIOD,
    )
    return np.ascontiguousarray(ind["kc_upper_arr"]), np.ascontiguousarray(ind["kc_lower_arr"])


def run_backtest(closes, highs, lows, pmax_line, mavg, kc_upper, kc_lower,
                 dca_m1, dca_m2, dca_m3, dca_m4):
    sa = np.ascontiguousarray
    return rust_engine.run_backtest_kc_lagged_graduated(
        sa(closes), sa(highs), sa(lows),
        sa(pmax_line), sa(mavg),
        sa(kc_upper), sa(kc_lower),
        MAX_DCA, TP_PCT,
        dca_m1, dca_m2, dca_m3, dca_m4,
    )


def build_folds(df):
    bars_per_day = 480
    warmup_bars = WARMUP_DAYS * bars_per_day
    test_bars = TEST_DAYS * bars_per_day
    step_bars = STEP_DAYS * bars_per_day
    total = len(df)

    folds = []
    fold_num = 1
    test_start = warmup_bars

    while test_start + test_bars <= total:
        test_end = test_start + test_bars
        test_warmup_start = max(0, test_start - 1000)
        folds.append({
            "num": fold_num,
            "train_end": test_start,
            "test_warmup_start": test_warmup_start,
            "test_end": test_end,
        })
        fold_num += 1
        test_start += step_bars

    return folds


def main():
    print("=" * 80)
    print("  GRADUATED DCA WALK-FORWARD OPTIMIZER V2 -- kc[i-1]")
    print("  PMax+KC kilitli | 500 trial/fold | Anchored expanding window")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        print(f"  HATA: {DATA_PATH} bulunamadi.")
        return

    print(f"\n  Veri yukleniyor...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Toplam mum: {len(df):,}")

    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    src = get_source(df)

    # PMax (kilitli, bir kez)
    print(f"  PMax hesaplaniyor...")
    pmax_line, mavg = compute_pmax(src, high, low, close)

    # KC (kilitli, bir kez)
    print(f"  KC hesaplaniyor (length={KC_LENGTH}, mult={KC_MULTIPLIER}, atr={KC_ATR_PERIOD})...")
    kc_upper, kc_lower = compute_kc(high, low, close)

    folds = build_folds(df)
    n_folds = len(folds)
    print(f"  Fold sayisi: {n_folds}")
    print(f"  Trial/fold: {TRIALS_PER_FOLD}")
    print(f"  Toplam trial: {n_folds * TRIALS_PER_FOLD:,}\n")

    all_fold_results = []
    all_best_params = []

    for fold in folds:
        fold_num = fold["num"]
        t_fold = time.time()

        # Train slice
        tr_end = fold["train_end"]
        tr_c, tr_h, tr_l = close[:tr_end], high[:tr_end], low[:tr_end]
        tr_pmax, tr_mavg = pmax_line[:tr_end], mavg[:tr_end]
        tr_kcu, tr_kcl = kc_upper[:tr_end], kc_lower[:tr_end]

        # Test slice (1000 bar warmup)
        ts = fold["test_warmup_start"]
        te = fold["test_end"]
        te_c, te_h, te_l = close[ts:te], high[ts:te], low[ts:te]
        te_pmax, te_mavg = pmax_line[ts:te], mavg[ts:te]
        te_kcu, te_kcl = kc_upper[ts:te], kc_lower[ts:te]

        # Optuna
        sampler = optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=15,
            seed=42 + fold_num, warn_independent_sampling=False,
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial):
            # Artan sira kisiti
            m1 = trial.suggest_float("dca_m1", 0.25, 2.0, step=0.25)
            m2 = trial.suggest_float("dca_m2", m1, 3.0, step=0.25)
            m3 = trial.suggest_float("dca_m3", m2, 4.0, step=0.25)
            m4 = trial.suggest_float("dca_m4", m3, 5.0, step=0.25)

            # Train backtest (pruning)
            train_r = run_backtest(tr_c, tr_h, tr_l, tr_pmax, tr_mavg,
                                   tr_kcu, tr_kcl, m1, m2, m3, m4)
            if train_r["total_trades"] < 5:
                return -999

            trial.report(train_r["net_pct"], step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Test backtest (OOS)
            test_r = run_backtest(te_c, te_h, te_l, te_pmax, te_mavg,
                                  te_kcu, te_kcl, m1, m2, m3, m4)

            net = test_r["net_pct"]
            dd = test_r["max_drawdown"]
            wr = test_r["win_rate"]

            # Sert filtre
            if wr < 80 or dd > 20:
                return -999

            score = (net / max(dd, 0.1)) * wr

            trial.set_user_attr("net", round(net, 2))
            trial.set_user_attr("dd", round(dd, 2))
            trial.set_user_attr("wr", round(wr, 1))
            trial.set_user_attr("trades", test_r["total_trades"])

            return score

        study.optimize(objective, n_trials=TRIALS_PER_FOLD, n_jobs=N_JOBS)

        if study.best_value is not None and study.best_value > -999:
            bp = study.best_trial
            result = {
                "fold": fold_num, "status": "OK",
                "score": round(bp.value, 4),
                "params": bp.params,
                "net": bp.user_attrs.get("net", 0),
                "dd": bp.user_attrs.get("dd", 0),
                "wr": bp.user_attrs.get("wr", 0),
                "trades": bp.user_attrs.get("trades", 0),
            }
            all_best_params.append(bp.params)
        else:
            result = {
                "fold": fold_num, "status": "FAIL",
                "score": -999, "params": None,
                "net": 0, "dd": 0, "wr": 0, "trades": 0,
            }

        all_fold_results.append(result)
        elapsed = time.time() - t_fold

        if result["status"] == "OK":
            p = result["params"]
            print(f"  Fold {fold_num:2d}/{n_folds} | "
                  f"Net: {result['net']:>+7.2f}% DD: {result['dd']:>5.1f}% "
                  f"WR: {result['wr']:>4.0f}% Trd: {result['trades']:>4d} | "
                  f"DCA[{p['dca_m1']:.2f},{p['dca_m2']:.2f},{p['dca_m3']:.2f},{p['dca_m4']:.2f}] | "
                  f"{elapsed:.0f}s")
        else:
            print(f"  Fold {fold_num:2d}/{n_folds} | FAIL | {elapsed:.0f}s")

    # ================================================================
    # KONSENSUS
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  KONSENSUS ANALIZI")
    print(f"{'='*80}")

    ok_folds = [r for r in all_fold_results if r["status"] == "OK"]
    fail_folds = [r for r in all_fold_results if r["status"] == "FAIL"]

    print(f"  Basarili fold: {len(ok_folds)}/{n_folds}")
    print(f"  Basarisiz fold: {len(fail_folds)}/{n_folds}")

    if not all_best_params:
        print(f"\n  HICBIR FOLD BASARILI OLMADI.")
        return

    param_names = ["dca_m1", "dca_m2", "dca_m3", "dca_m4"]
    consensus = {}

    print(f"\n  {'Parametre':12s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>6s} | {'Max':>6s}")
    print(f"  {'-'*65}")

    for pname in param_names:
        vals = [p[pname] for p in all_best_params]
        median_val = float(np.median(vals))
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        cv = std_val / mean_val if mean_val > 0 else 0

        # Step snap (0.25)
        median_val = round(round(median_val / 0.25) * 0.25, 2)

        if cv == 0:
            stability = "MUTLAK"
        elif cv < 0.3:
            stability = "STABIL"
        elif cv < 0.5:
            stability = "ORTA"
        else:
            stability = "DEGISKEN"

        consensus[pname] = median_val
        print(f"  {pname:12s} | {median_val:>10.2f} | {cv:>8.4f} | {stability:>12s} | {min(vals):>6.2f} | {max(vals):>6.2f}")

    profitable = sum(1 for r in ok_folds if r["net"] > 0)
    print(f"\n  Karli fold: {profitable}/{len(ok_folds)} ({profitable/len(ok_folds)*100:.0f}%)" if ok_folds else "")

    # Kaydet
    output = {
        "method": "Graduated DCA WF V2 -- kc[i-1]",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "locked_pmax": PMAX_PARAMS,
        "locked_kc": {"kc_length": KC_LENGTH, "kc_multiplier": KC_MULTIPLIER, "kc_atr_period": KC_ATR_PERIOD},
        "total_folds": n_folds,
        "successful_folds": len(ok_folds),
        "consensus": consensus,
        "fold_results": all_fold_results,
        "all_best_params": all_best_params,
    }

    out_path = os.path.join(RESULTS_DIR, "dca_wf_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Sonuclar: {out_path}")

    # Fold detaylari
    print(f"\n  FOLD DETAYLARI:")
    print(f"  {'Fold':>4s} | {'Net%':>8s} | {'DD%':>6s} | {'WR%':>5s} | {'Trd':>5s} | DCA Multipliers")
    print(f"  {'-'*70}")
    for r in all_fold_results:
        if r["status"] == "OK":
            p = r["params"]
            print(f"  {r['fold']:4d} | {r['net']:>+8.2f} | {r['dd']:>6.1f} | {r['wr']:>5.0f} | "
                  f"{r['trades']:>5d} | [{p['dca_m1']:.2f}, {p['dca_m2']:.2f}, {p['dca_m3']:.2f}, {p['dca_m4']:.2f}]")
        else:
            print(f"  {r['fold']:4d} | {'FAIL':>8s} |")


if __name__ == "__main__":
    main()
