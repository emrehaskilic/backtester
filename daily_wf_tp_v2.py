"""
Graduated TP Walk-Forward Optimizer V2 — kc[i-1].
PMax + KC + DCA kilitli. 4 TP yuzdesi optimize.
Kisit: tp1 <= tp2 <= tp3 <= tp4 (artan sira).

Kullanim: python daily_wf_tp_v2.py
"""

import json, math, os, sys, time
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

PMAX = {"vol_lookback": 260, "flip_window": 360, "mult_base": 3.25, "mult_scale": 2.0,
        "ma_base": 11, "ma_scale": 4.5, "atr_base": 15, "atr_scale": 2.0, "update_interval": 55}
KC_LENGTH, KC_MULT, KC_ATR = 20, 2.0, 28
DCA_M = [0.25, 0.25, 0.25, 0.50]  # DCA konsensus
MAX_DCA = 4

WARMUP_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7
TRIALS_PER_FOLD = 500
N_JOBS = 6

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_pmax(src, h, l, c):
    sa = np.ascontiguousarray
    r = rust_engine.compute_adaptive_pmax(
        sa(src, dtype=np.float64), sa(h, dtype=np.float64),
        sa(l, dtype=np.float64), sa(c, dtype=np.float64),
        10, 3.0, 10,
        PMAX["vol_lookback"], PMAX["flip_window"],
        PMAX["mult_base"], PMAX["mult_scale"],
        PMAX["ma_base"], PMAX["ma_scale"],
        PMAX["atr_base"], PMAX["atr_scale"],
        PMAX["update_interval"])
    return sa(r["pmax_line"]), sa(r["mavg"])


def compute_kc(h, l, c):
    sa = np.ascontiguousarray
    ind = rust_engine.precompute_indicators(
        sa(h, dtype=np.float64), sa(l, dtype=np.float64),
        sa(c, dtype=np.float64), 144, KC_LENGTH, KC_MULT, KC_ATR)
    return sa(ind["kc_upper_arr"]), sa(ind["kc_lower_arr"])


def run_bt(c, h, l, pm, mv, ku, kl, tp1, tp2, tp3, tp4):
    sa = np.ascontiguousarray
    return rust_engine.run_backtest_kc_lagged_graduated_tp(
        sa(c), sa(h), sa(l), sa(pm), sa(mv), sa(ku), sa(kl),
        MAX_DCA,
        DCA_M[0], DCA_M[1], DCA_M[2], DCA_M[3],
        tp1, tp2, tp3, tp4)


def build_folds(df):
    bpd = 480
    warmup = WARMUP_DAYS * bpd
    test = TEST_DAYS * bpd
    step = STEP_DAYS * bpd
    total = len(df)
    folds, num, ts = [], 1, warmup
    while ts + test <= total:
        folds.append({"num": num, "train_end": ts,
                       "test_ws": max(0, ts - 1000), "test_end": ts + test})
        num += 1
        ts += step
    return folds


def main():
    print("=" * 80)
    print("  GRADUATED TP WALK-FORWARD OPTIMIZER V2 -- kc[i-1]")
    print("  PMax+KC+DCA kilitli | 500 trial/fold")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Mum: {len(df):,}")

    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    src = (h + l) / 2.0

    print("  PMax + KC hesaplaniyor...")
    pmax_line, mavg = compute_pmax(src, h, l, c)
    kc_u, kc_l = compute_kc(h, l, c)

    folds = build_folds(df)
    nf = len(folds)
    print(f"  Fold: {nf} | Trial/fold: {TRIALS_PER_FOLD}\n")

    all_results = []
    all_params = []

    for fold in folds:
        fn = fold["num"]
        t0 = time.time()

        te, tws, ted = fold["train_end"], fold["test_ws"], fold["test_end"]

        # Train/test slices
        tr_c, tr_h, tr_l = c[:te], h[:te], l[:te]
        tr_pm, tr_mv, tr_ku, tr_kl = pmax_line[:te], mavg[:te], kc_u[:te], kc_l[:te]
        te_c, te_h, te_l = c[tws:ted], h[tws:ted], l[tws:ted]
        te_pm, te_mv, te_ku, te_kl = pmax_line[tws:ted], mavg[tws:ted], kc_u[tws:ted], kc_l[tws:ted]

        sampler = optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=15,
            seed=42 + fn, warn_independent_sampling=False)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial):
            t1 = trial.suggest_float("tp1", 0.10, 0.70, step=0.05)
            t2 = trial.suggest_float("tp2", t1, 0.90, step=0.05)
            t3 = trial.suggest_float("tp3", t2, 0.95, step=0.05)
            t4 = trial.suggest_float("tp4", t3, 1.00, step=0.05)

            # Train
            tr_r = run_bt(tr_c, tr_h, tr_l, tr_pm, tr_mv, tr_ku, tr_kl, t1, t2, t3, t4)
            if tr_r["total_trades"] < 5:
                return -999
            trial.report(tr_r["net_pct"], step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Test (OOS)
            r = run_bt(te_c, te_h, te_l, te_pm, te_mv, te_ku, te_kl, t1, t2, t3, t4)
            net, dd, wr = r["net_pct"], r["max_drawdown"], r["win_rate"]

            if wr < 80 or dd > 20:
                return -999

            score = (net / max(dd, 0.1)) * wr
            trial.set_user_attr("net", round(net, 2))
            trial.set_user_attr("dd", round(dd, 2))
            trial.set_user_attr("wr", round(wr, 1))
            trial.set_user_attr("trades", r["total_trades"])
            return score

        study.optimize(objective, n_trials=TRIALS_PER_FOLD, n_jobs=N_JOBS)

        if study.best_value is not None and study.best_value > -999:
            bp = study.best_trial
            res = {"fold": fn, "status": "OK", "score": round(bp.value, 4),
                   "params": bp.params,
                   "net": bp.user_attrs.get("net", 0),
                   "dd": bp.user_attrs.get("dd", 0),
                   "wr": bp.user_attrs.get("wr", 0),
                   "trades": bp.user_attrs.get("trades", 0)}
            all_params.append(bp.params)
        else:
            res = {"fold": fn, "status": "FAIL", "score": -999, "params": None,
                   "net": 0, "dd": 0, "wr": 0, "trades": 0}

        all_results.append(res)
        el = time.time() - t0

        if res["status"] == "OK":
            p = res["params"]
            print(f"  Fold {fn:2d}/{nf} | Net: {res['net']:>+7.2f}% DD: {res['dd']:>5.1f}% "
                  f"WR: {res['wr']:>4.0f}% | "
                  f"TP[{p['tp1']:.2f},{p['tp2']:.2f},{p['tp3']:.2f},{p['tp4']:.2f}] | {el:.0f}s")
        else:
            print(f"  Fold {fn:2d}/{nf} | FAIL | {el:.0f}s")

    # Konsensus
    print(f"\n{'='*80}")
    print(f"  KONSENSUS")
    print(f"{'='*80}")

    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"  Basarili: {len(ok)}/{nf} | Basarisiz: {nf-len(ok)}/{nf}")

    if not all_params:
        print("  HICBIR FOLD BASARILI OLMADI.")
        return

    consensus = {}
    print(f"\n  {'Param':8s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>6s} | {'Max':>6s}")
    print(f"  {'-'*60}")

    for pn in ["tp1", "tp2", "tp3", "tp4"]:
        vals = [p[pn] for p in all_params]
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cv = std / mean if mean > 0 else 0
        med = round(round(med / 0.05) * 0.05, 2)
        stab = "MUTLAK" if cv == 0 else "STABIL" if cv < 0.3 else "ORTA" if cv < 0.5 else "DEGISKEN"
        consensus[pn] = med
        print(f"  {pn:8s} | {med:>10.2f} | {cv:>8.4f} | {stab:>12s} | {min(vals):>6.2f} | {max(vals):>6.2f}")

    profitable = sum(1 for r in ok if r["net"] > 0)
    print(f"\n  Karli fold: {profitable}/{len(ok)} ({profitable/len(ok)*100:.0f}%)" if ok else "")

    out = {"method": "Graduated TP WF V2 -- kc[i-1]",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "locked": {"pmax": PMAX, "kc": {"length": KC_LENGTH, "mult": KC_MULT, "atr": KC_ATR},
                      "dca": DCA_M},
           "consensus": consensus, "folds": nf, "ok": len(ok),
           "results": all_results, "params": all_params}

    op = os.path.join(RESULTS_DIR, "tp_wf_v2_results.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Kaydedildi: {op}")


if __name__ == "__main__":
    main()
