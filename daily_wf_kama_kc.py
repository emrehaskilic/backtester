"""
KAMA + KC Walk-Forward Optimizer — KAMA kilitli, KC optimize.
kc[i-1] (look-ahead yok). 3 KC param optimize.
"""

import json, os, sys, time
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import rust_engine
print("Rust engine yuklendi")

# KAMA konsensus (kilitli)
KAMA_P, KAMA_F, KAMA_S = 20, 6, 33
SLOPE_LB, SLOPE_TH = 14, 0.75

MAX_DCA = 4
TP_PCT = 0.50
WARMUP_DAYS, TEST_DAYS, STEP_DAYS = 90, 7, 7
TRIALS = 500
N_JOBS = 6

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_folds(total):
    bpd = 480
    w, t, s = WARMUP_DAYS * bpd, TEST_DAYS * bpd, STEP_DAYS * bpd
    folds, num, pos = [], 1, w
    while pos + t <= total:
        folds.append({"num": num, "te": pos, "tws": max(0, pos-1000), "ted": pos+t})
        num += 1; pos += s
    return folds


def main():
    print("=" * 80)
    print("  KAMA + KC WALK-FORWARD OPTIMIZER — kc[i-1]")
    print("  KAMA kilitli | KC 3 param optimize | 500 trial/fold")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)
    sa = np.ascontiguousarray
    c = sa(df["close"].values, dtype=np.float64)
    h = sa(df["high"].values, dtype=np.float64)
    l = sa(df["low"].values, dtype=np.float64)
    print(f"  Mum: {len(df):,}")

    folds = build_folds(len(df))
    nf = len(folds)
    print(f"  Fold: {nf} | Trial: {TRIALS}\n")

    all_results, all_params = [], []

    for fold in folds:
        fn = fold["num"]
        t0 = time.time()
        te, tws, ted = fold["te"], fold["tws"], fold["ted"]

        tr_c, tr_h, tr_l = c[:te], h[:te], l[:te]
        te_c, te_h, te_l = c[tws:ted], h[tws:ted], l[tws:ted]

        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15, seed=42+fn)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial):
            kl = trial.suggest_int("kc_length", 2, 50)
            km = trial.suggest_float("kc_multiplier", 0.1, 5.0, step=0.1)
            ka = trial.suggest_int("kc_atr_period", 2, 50)

            # Train
            tr = rust_engine.run_kama_kc_backtest(
                tr_c, tr_h, tr_l,
                KAMA_P, KAMA_F, KAMA_S, SLOPE_LB, SLOPE_TH,
                kl, km, ka, MAX_DCA, TP_PCT)
            if tr["total_trades"] < 5: return -999
            if tr["net_pct"] < 30: return -999
            trial.report(tr["net_pct"], step=0)
            if trial.should_prune(): raise optuna.TrialPruned()

            # Test (OOS)
            r = rust_engine.run_kama_kc_backtest(
                te_c, te_h, te_l,
                KAMA_P, KAMA_F, KAMA_S, SLOPE_LB, SLOPE_TH,
                kl, km, ka, MAX_DCA, TP_PCT)

            net, dd, wr = r["net_pct"], r["max_drawdown"], r["win_rate"]
            if wr < 80 or dd > 20: return -999

            score = (net / max(dd, 0.1)) * wr
            trial.set_user_attr("net", round(net, 2))
            trial.set_user_attr("dd", round(dd, 2))
            trial.set_user_attr("wr", round(wr, 1))
            trial.set_user_attr("trades", r["total_trades"])
            trial.set_user_attr("tp", r["tp_count"])
            trial.set_user_attr("rev", r["rev_count"])
            return score

        study.optimize(objective, n_trials=TRIALS, n_jobs=N_JOBS)

        if study.best_value is not None and study.best_value > -999:
            bp = study.best_trial
            res = {"fold": fn, "status": "OK", "score": round(bp.value, 4),
                   "params": bp.params,
                   "net": bp.user_attrs.get("net", 0), "dd": bp.user_attrs.get("dd", 0),
                   "wr": bp.user_attrs.get("wr", 0), "trades": bp.user_attrs.get("trades", 0),
                   "tp": bp.user_attrs.get("tp", 0), "rev": bp.user_attrs.get("rev", 0)}
            all_params.append(bp.params)
        else:
            res = {"fold": fn, "status": "FAIL", "score": -999, "params": None,
                   "net": 0, "dd": 0, "wr": 0, "trades": 0, "tp": 0, "rev": 0}

        all_results.append(res)
        el = time.time() - t0

        if res["status"] == "OK":
            p = res["params"]
            print(f"  Fold {fn:2d}/{nf} | Net: {res['net']:>+7.2f}% DD: {res['dd']:>5.1f}% "
                  f"WR: {res['wr']:>4.0f}% Trd: {res['trades']:>4d} TP:{res['tp']:>3d} Rev:{res['rev']:>3d} | "
                  f"KC({p['kc_length']},{p['kc_multiplier']:.1f},{p['kc_atr_period']}) | {el:.0f}s")
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
    print(f"\n  {'Param':18s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>6s} | {'Max':>6s}")
    print(f"  {'-'*70}")
    for pn in ["kc_length", "kc_multiplier", "kc_atr_period"]:
        vals = [p[pn] for p in all_params]
        med = float(np.median(vals))
        mean, std = float(np.mean(vals)), float(np.std(vals))
        cv = std / mean if mean > 0 else 0
        if pn == "kc_multiplier": med = round(round(med / 0.1) * 0.1, 1)
        else: med = int(round(med))
        stab = "MUTLAK" if cv == 0 else "STABIL" if cv < 0.3 else "ORTA" if cv < 0.5 else "DEGISKEN"
        consensus[pn] = med
        print(f"  {pn:18s} | {str(med):>10s} | {cv:>8.4f} | {stab:>12s} | {min(vals):>6} | {max(vals):>6}")

    profitable = sum(1 for r in ok if r["net"] > 0)
    print(f"\n  Karli fold: {profitable}/{len(ok)} ({profitable/len(ok)*100:.0f}%)")

    out = {"method": "KAMA+KC WF — kc[i-1]", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "kama": {"period": KAMA_P, "fast": KAMA_F, "slow": KAMA_S, "slope_lb": SLOPE_LB, "slope_th": SLOPE_TH},
           "consensus": consensus, "folds": nf, "ok": len(ok),
           "results": all_results, "params": all_params}
    op = os.path.join(RESULTS_DIR, "kama_kc_wf_results.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Kaydedildi: {op}")


if __name__ == "__main__":
    main()
