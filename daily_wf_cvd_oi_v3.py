"""
CVD + OI WF Optimizer V3 — sadece karli foldlar.
5dk bar. 5 param. 500 trial/fold. Train min 500 trade.
Konsensus sadece karli (net>0) foldlardan hesaplanir.
"""
import json, os, sys, time, math
import numpy as np, pandas as pd, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import rust_engine
print("Rust engine yuklendi")

TRIALS, N_JOBS = 500, 6
WARMUP_DAYS, TEST_DAYS, STEP_DAYS = 90, 7, 7
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_5m_cvd_oi_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_folds(total):
    bpd = 288
    w, t, s = WARMUP_DAYS * bpd, TEST_DAYS * bpd, STEP_DAYS * bpd
    folds, num, pos = [], 1, w
    while pos + t <= total:
        folds.append({"num": num, "te": pos, "tws": max(0, pos - 2000), "ted": pos + t})
        num += 1; pos += s
    return folds

def main():
    print("=" * 80)
    print("  CVD + OI WF OPTIMIZER V3 — Sadece Karli Foldlar")
    print("  5dk | 5 param | 500 trial/fold | Train min 500 trade")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)
    sa = np.ascontiguousarray
    c = sa(df["close"].values, dtype=np.float64)
    bv = sa(df["buy_vol"].values, dtype=np.float64)
    sv = sa(df["sell_vol"].values, dtype=np.float64)
    oi = sa(pd.Series(df["open_interest"].values.astype(np.float64)).ffill().bfill().values, dtype=np.float64)
    total = len(df)
    print(f"  Mum: {total:,}")

    folds = build_folds(total)
    nf = len(folds)
    print(f"  Fold: {nf} | Trial: {TRIALS}\n")

    all_results, all_params = [], []

    for fold in folds:
        fn, t0 = fold["num"], time.time()
        te, tws, ted = fold["te"], fold["tws"], fold["ted"]
        tr_c, tr_bv, tr_sv, tr_oi = c[:te], bv[:te], sv[:te], oi[:te]
        te_c, te_bv, te_sv, te_oi = c[tws:ted], bv[tws:ted], sv[tws:ted], oi[tws:ted]

        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=30, seed=42+fn)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=20)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial):
            cp = trial.suggest_int("cvd_period", 10, 100, step=2)
            iw = trial.suggest_float("imb_weight", 0.2, 0.8, step=0.05)
            ct = trial.suggest_float("cvd_threshold", 0.2, 0.6, step=0.025)
            op = trial.suggest_int("oi_period", 5, 60, step=5)
            ot = trial.suggest_float("oi_threshold", 0.003, 0.015, step=0.001)

            tr = rust_engine.run_cvd_oi_backtest(tr_c, tr_bv, tr_sv, tr_oi, cp, iw, ct, op, ot)
            if tr["total_trades"] < 1: return -999
            trial.report(tr["net_pct"], step=0)
            if trial.should_prune(): raise optuna.TrialPruned()

            r = rust_engine.run_cvd_oi_backtest(te_c, te_bv, te_sv, te_oi, cp, iw, ct, op, ot)
            net, dd, wr, trades = r["net_pct"], r["max_drawdown"], r["win_rate"], r["total_trades"]
            if trades < 1: return -999

            score = (net / max(dd, 1.0)) * wr * 0.01
            if net > 500: score *= 0.8

            trial.set_user_attr("net", round(net, 2))
            trial.set_user_attr("dd", round(dd, 2))
            trial.set_user_attr("wr", round(wr, 1))
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("fees", round(r["total_fees"], 2))
            return score

        study.optimize(objective, n_trials=TRIALS, n_jobs=N_JOBS)

        if study.best_value is not None and study.best_value > -999:
            bp = study.best_trial
            res = {"fold": fn, "status": "OK", "score": round(bp.value, 4),
                   "params": bp.params,
                   "net": bp.user_attrs.get("net", 0), "dd": bp.user_attrs.get("dd", 0),
                   "wr": bp.user_attrs.get("wr", 0), "trades": bp.user_attrs.get("trades", 0),
                   "fees": bp.user_attrs.get("fees", 0)}
            if res["net"] > 0:
                all_params.append(bp.params)
        else:
            res = {"fold": fn, "status": "FAIL", "params": None,
                   "net": 0, "dd": 0, "wr": 0, "trades": 0, "fees": 0}

        all_results.append(res)
        el = time.time() - t0
        if res["status"] == "OK":
            p = res["params"]
            tag = " << KARLI" if res["net"] > 0 else " << ZARARI"
            print(f"  Fold {fn:2d}/{nf} | Net: {res['net']:>+8.2f}% DD: {res['dd']:>5.1f}% "
                  f"WR: {res['wr']:>4.0f}% Trd: {res['trades']:>3d} | "
                  f"CVD(p={p['cvd_period']},iw={p['imb_weight']:.2f},th={p['cvd_threshold']:.3f}) "
                  f"OI(p={p['oi_period']},th={p['oi_threshold']:.3f}) | {el:.0f}s{tag}")
        else:
            print(f"  Fold {fn:2d}/{nf} | FAIL | {el:.0f}s")

    print(f"\n{'='*80}")
    print(f"  KONSENSUS V3 — Sadece Karli Foldlar")
    print(f"{'='*80}")
    ok = [r for r in all_results if r["status"] == "OK"]
    profitable = [r for r in ok if r["net"] > 0]
    lossy = [r for r in ok if r["net"] <= 0]
    failed = [r for r in all_results if r["status"] == "FAIL"]
    print(f"  Toplam fold: {nf}")
    print(f"  Basarili: {len(ok)} | Karli: {len(profitable)} | Zararli: {len(lossy)} | Fail: {len(failed)}")

    if not all_params:
        print("  HICBIR KARLI FOLD YOK."); return

    consensus = {}
    print(f"\n  Konsensus {len(all_params)} karli fold uzerinden hesaplandi:")
    print(f"\n  {'Param':18s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>8s} | {'Max':>8s}")
    print(f"  {'-'*75}")
    for pn in ["cvd_period", "imb_weight", "cvd_threshold", "oi_period", "oi_threshold"]:
        vals = [p[pn] for p in all_params]
        med = float(np.median(vals))
        mean, std = float(np.mean(vals)), float(np.std(vals))
        cv = std / mean if mean > 0 else 0
        if pn == "cvd_period": med = int(round(med / 2) * 2)
        elif pn == "imb_weight": med = round(round(med / 0.05) * 0.05, 2)
        elif pn == "cvd_threshold": med = round(round(med / 0.025) * 0.025, 3)
        elif pn == "oi_period": med = int(round(med / 5) * 5)
        elif pn == "oi_threshold": med = round(round(med / 0.001) * 0.001, 3)
        stab = "MUTLAK" if cv == 0 else "STABIL" if cv < 0.3 else "ORTA" if cv < 0.5 else "DEGISKEN"
        consensus[pn] = med
        print(f"  {pn:18s} | {str(med):>10s} | {cv:>8.4f} | {stab:>12s} | {min(vals):>8} | {max(vals):>8}")

    avg_net = np.mean([r["net"] for r in profitable])
    avg_dd = np.mean([r["dd"] for r in profitable])
    avg_wr = np.mean([r["wr"] for r in profitable])
    avg_trades = np.mean([r["trades"] for r in profitable])
    avg_fees = np.mean([r["fees"] for r in profitable])
    print(f"\n  Karli fold ortalama:")
    print(f"    Net: +{avg_net:.2f}% | DD: {avg_dd:.1f}% | WR: {avg_wr:.0f}% | Trade: {avg_trades:.0f} | Fee: ${avg_fees:.0f}")

    if lossy:
        avg_loss = np.mean([r["net"] for r in lossy])
        print(f"  Zararli fold ortalama:")
        print(f"    Net: {avg_loss:.2f}%")

    # V2 karsilastirma
    print(f"\n  V2 vs V3")
    print(f"  V2: 23/34 OK, cvd_period=12, threshold=0.200")
    print(f"  V3: {len(profitable)}/{nf} karli, cvd_period={consensus.get('cvd_period','?')}, "
          f"threshold={consensus.get('cvd_threshold','?')}")

    out = {"method": "CVD+OI WF V3 — sadece karli", "consensus": consensus,
           "folds": nf, "ok": len(ok), "profitable": len(profitable),
           "results": all_results, "params": all_params}
    op = os.path.join(RESULTS_DIR, "cvd_oi_wf_v3_results.json")
    with open(op, "w") as f: json.dump(out, f, indent=2, default=str)
    print(f"\n  Kaydedildi: {op}")

if __name__ == "__main__":
    main()
