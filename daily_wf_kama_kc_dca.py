"""
KAMA+KC+Graduated DCA WF Optimizer.
KAMA+KC kilitli, 4 DCA multiplier optimize.
"""
import json, os, sys, time
import numpy as np, pandas as pd, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import rust_engine
print("Rust engine yuklendi")

KAMA_P, KAMA_F, KAMA_S, SLOPE_LB, SLOPE_TH = 22, 8, 31, 14, 0.80
KC_L, KC_M, KC_A = 17, 3.4, 13
MAX_DCA, TP_PCT = 4, 0.50
TRIALS, N_JOBS = 500, 6

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_folds(total):
    bpd, w, t, s = 480, 90*480, 7*480, 7*480
    folds, num, pos = [], 1, w
    while pos + t <= total:
        folds.append({"num": num, "te": pos, "tws": max(0, pos-1000), "ted": pos+t})
        num += 1; pos += s
    return folds

def main():
    print("=" * 80)
    print("  KAMA+KC+Graduated DCA WF OPTIMIZER")
    print("  KAMA+KC kilitli | DCA 4 param | 500 trial/fold")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)
    sa = np.ascontiguousarray
    c, h, l = sa(df["close"].values, dtype=np.float64), sa(df["high"].values, dtype=np.float64), sa(df["low"].values, dtype=np.float64)
    folds = build_folds(len(df))
    nf = len(folds)
    print(f"  Mum: {len(df):,} | Fold: {nf}\n")

    all_results, all_params = [], []

    for fold in folds:
        fn, t0 = fold["num"], time.time()
        te, tws, ted = fold["te"], fold["tws"], fold["ted"]
        tr_c, tr_h, tr_l = c[:te], h[:te], l[:te]
        te_c, te_h, te_l = c[tws:ted], h[tws:ted], l[tws:ted]

        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15, seed=42+fn, warn_independent_sampling=False)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=optuna.pruners.MedianPruner(n_startup_trials=10))

        def objective(trial):
            m1 = trial.suggest_float("dca_m1", 0.25, 2.0, step=0.25)
            m2 = trial.suggest_float("dca_m2", m1, 3.0, step=0.25)
            m3 = trial.suggest_float("dca_m3", m2, 4.0, step=0.25)
            m4 = trial.suggest_float("dca_m4", m3, 5.0, step=0.25)

            tr = rust_engine.run_kama_kc_grad_dca(tr_c, tr_h, tr_l, KAMA_P, KAMA_F, KAMA_S, SLOPE_LB, SLOPE_TH, KC_L, KC_M, KC_A, MAX_DCA, TP_PCT, m1, m2, m3, m4)
            if tr["total_trades"] < 5: return -999
            trial.report(tr["net_pct"], step=0)
            if trial.should_prune(): raise optuna.TrialPruned()

            r = rust_engine.run_kama_kc_grad_dca(te_c, te_h, te_l, KAMA_P, KAMA_F, KAMA_S, SLOPE_LB, SLOPE_TH, KC_L, KC_M, KC_A, MAX_DCA, TP_PCT, m1, m2, m3, m4)
            net, dd, wr = r["net_pct"], r["max_drawdown"], r["win_rate"]
            if wr < 80 or dd > 20: return -999
            score = (net / max(dd, 0.1)) * wr
            trial.set_user_attr("net", round(net, 2)); trial.set_user_attr("dd", round(dd, 2))
            trial.set_user_attr("wr", round(wr, 1)); trial.set_user_attr("trades", r["total_trades"])
            return score

        study.optimize(objective, n_trials=TRIALS, n_jobs=N_JOBS)

        if study.best_value is not None and study.best_value > -999:
            bp = study.best_trial
            res = {"fold": fn, "status": "OK", "score": round(bp.value, 4), "params": bp.params,
                   "net": bp.user_attrs.get("net", 0), "dd": bp.user_attrs.get("dd", 0),
                   "wr": bp.user_attrs.get("wr", 0), "trades": bp.user_attrs.get("trades", 0)}
            all_params.append(bp.params)
        else:
            res = {"fold": fn, "status": "FAIL", "params": None, "net": 0, "dd": 0, "wr": 0, "trades": 0}

        all_results.append(res)
        el = time.time() - t0
        if res["status"] == "OK":
            p = res["params"]
            print(f"  Fold {fn:2d}/{nf} | Net: {res['net']:>+7.2f}% DD: {res['dd']:>5.1f}% WR: {res['wr']:>4.0f}% | DCA[{p['dca_m1']:.2f},{p['dca_m2']:.2f},{p['dca_m3']:.2f},{p['dca_m4']:.2f}] | {el:.0f}s")
        else:
            print(f"  Fold {fn:2d}/{nf} | FAIL | {el:.0f}s")

    # Konsensus
    print(f"\n{'='*80}\n  KONSENSUS\n{'='*80}")
    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"  Basarili: {len(ok)}/{nf}")
    if not all_params: print("  HICBIR FOLD BASARILI OLMADI."); return

    consensus = {}
    print(f"\n  {'Param':12s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>6s} | {'Max':>6s}")
    print(f"  {'-'*65}")
    for pn in ["dca_m1", "dca_m2", "dca_m3", "dca_m4"]:
        vals = [p[pn] for p in all_params]
        med, mean, std = float(np.median(vals)), float(np.mean(vals)), float(np.std(vals))
        cv = std / mean if mean > 0 else 0
        med = round(round(med / 0.25) * 0.25, 2)
        stab = "MUTLAK" if cv == 0 else "STABIL" if cv < 0.3 else "ORTA" if cv < 0.5 else "DEGISKEN"
        consensus[pn] = med
        print(f"  {pn:12s} | {med:>10.2f} | {cv:>8.4f} | {stab:>12s} | {min(vals):>6.2f} | {max(vals):>6.2f}")

    profitable = sum(1 for r in ok if r["net"] > 0)
    print(f"\n  Karli fold: {profitable}/{len(ok)} ({profitable/len(ok)*100:.0f}%)")

    out = {"method": "KAMA+KC+GradDCA WF", "consensus": consensus, "folds": nf, "ok": len(ok),
           "results": all_results, "params": all_params}
    op = os.path.join(RESULTS_DIR, "kama_kc_dca_wf_results.json")
    with open(op, "w") as f: json.dump(out, f, indent=2, default=str)
    print(f"\n  Kaydedildi: {op}")

if __name__ == "__main__":
    main()
