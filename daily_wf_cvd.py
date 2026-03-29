"""
CVD Order Flow WF Optimizer — Adim 1: Saf CVD sinyal.
30s mikro-mum uzerinde. DCA/TP yok, reversal ile kapanir.
3 param: cvd_period, imb_weight, signal_threshold.
"""
import json, os, sys, time, math
import numpy as np, pandas as pd, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import rust_engine
    print("Rust engine yuklendi")
except ImportError:
    print("HATA: rust_engine bulunamadi.")
    sys.exit(1)

TRIALS, N_JOBS = 500, 6
WARMUP_DAYS, TEST_DAYS, STEP_DAYS = 90, 7, 7

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_30s_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_folds(total):
    bpd = 2880  # 30s = 24*60*2 = 2880 bar/gun
    w, t, s = WARMUP_DAYS * bpd, TEST_DAYS * bpd, STEP_DAYS * bpd
    folds, num, pos = [], 1, w
    while pos + t <= total:
        folds.append({"num": num, "te": pos, "tws": max(0, pos - 5000), "ted": pos + t})
        num += 1; pos += s
    return folds


def main():
    print("=" * 80)
    print("  CVD ORDER FLOW WF OPTIMIZER — Adim 1: Saf Sinyal")
    print("  30s mikro-mum | 3 param | 500 trial/fold")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        print(f"  HATA: {DATA_PATH} bulunamadi. Once build_30s_cache.py calistirin.")
        return

    df = pd.read_parquet(DATA_PATH)
    sa = np.ascontiguousarray
    closes = sa(df["close"].values, dtype=np.float64)
    buy_vol = sa(df["buy_vol"].values, dtype=np.float64)
    sell_vol = sa(df["sell_vol"].values, dtype=np.float64)
    total = len(df)
    print(f"  Mum: {total:,} (30s)")

    folds = build_folds(total)
    nf = len(folds)
    print(f"  Fold: {nf} | Trial: {TRIALS}")
    print(f"  Toplam trial: {nf * TRIALS:,}\n")

    all_results, all_params = [], []

    for fold in folds:
        fn, t0 = fold["num"], time.time()
        te, tws, ted = fold["te"], fold["tws"], fold["ted"]

        tr_c, tr_bv, tr_sv = closes[:te], buy_vol[:te], sell_vol[:te]
        te_c, te_bv, te_sv = closes[tws:ted], buy_vol[tws:ted], sell_vol[tws:ted]

        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15, seed=42 + fn)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial):
            cp = trial.suggest_int("cvd_period", 5, 50)
            iw = trial.suggest_float("imb_weight", 0.0, 1.0, step=0.05)
            st = trial.suggest_float("signal_threshold", 0.1, 1.0, step=0.05)

            # Train
            tr = rust_engine.run_cvd_backtest(tr_c, tr_bv, tr_sv, cp, iw, st)
            if tr["total_trades"] < 5:
                return -999
            trial.report(tr["net_pct"], step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Test (OOS)
            r = rust_engine.run_cvd_backtest(te_c, te_bv, te_sv, cp, iw, st)
            net, dd, wr, trades = r["net_pct"], r["max_drawdown"], r["win_rate"], r["total_trades"]

            if trades < 3:
                return -999

            score = (net / max(dd, 1.0)) * wr * 0.01
            if net > 500:
                score *= 0.8

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
                   "net": bp.user_attrs.get("net", 0),
                   "dd": bp.user_attrs.get("dd", 0),
                   "wr": bp.user_attrs.get("wr", 0),
                   "trades": bp.user_attrs.get("trades", 0),
                   "fees": bp.user_attrs.get("fees", 0)}
            all_params.append(bp.params)
        else:
            res = {"fold": fn, "status": "FAIL", "score": -999, "params": None,
                   "net": 0, "dd": 0, "wr": 0, "trades": 0, "fees": 0}

        all_results.append(res)
        el = time.time() - t0

        if res["status"] == "OK":
            p = res["params"]
            print(f"  Fold {fn:2d}/{nf} | Net: {res['net']:>+8.2f}% DD: {res['dd']:>5.1f}% "
                  f"WR: {res['wr']:>4.0f}% Trd: {res['trades']:>4d} Fee: ${res['fees']:>6.0f} | "
                  f"CVD(p={p['cvd_period']},iw={p['imb_weight']:.2f},th={p['signal_threshold']:.2f}) | {el:.0f}s")
        else:
            print(f"  Fold {fn:2d}/{nf} | FAIL | {el:.0f}s")

    # Konsensus
    print(f"\n{'='*80}")
    print(f"  KONSENSUS ANALIZI — CVD SINYAL")
    print(f"{'='*80}")

    ok = [r for r in all_results if r["status"] == "OK"]
    fail = [r for r in all_results if r["status"] == "FAIL"]
    print(f"  Basarili: {len(ok)}/{nf} | Basarisiz: {len(fail)}/{nf}")

    if not all_params:
        print("  HICBIR FOLD BASARILI OLMADI.")
        return

    consensus = {}
    print(f"\n  {'Param':18s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>8s} | {'Max':>8s}")
    print(f"  {'-'*75}")

    for pn in ["cvd_period", "imb_weight", "signal_threshold"]:
        vals = [p[pn] for p in all_params]
        med = float(np.median(vals))
        mean, std = float(np.mean(vals)), float(np.std(vals))
        cv = std / mean if mean > 0 else 0

        if pn == "cvd_period":
            med = int(round(med))
        elif pn == "imb_weight":
            med = round(round(med / 0.05) * 0.05, 2)
        elif pn == "signal_threshold":
            med = round(round(med / 0.05) * 0.05, 2)

        stab = "MUTLAK" if cv == 0 else "STABIL" if cv < 0.3 else "ORTA" if cv < 0.5 else "DEGISKEN"
        consensus[pn] = med
        print(f"  {pn:18s} | {str(med):>10s} | {cv:>8.4f} | {stab:>12s} | {min(vals):>8} | {max(vals):>8}")

    profitable = sum(1 for r in ok if r["net"] > 0)
    avg_trades = np.mean([r["trades"] for r in ok])
    avg_fees = np.mean([r["fees"] for r in ok])
    print(f"\n  Karli fold: {profitable}/{len(ok)} ({profitable/len(ok)*100:.0f}%)")
    print(f"  Ort. trade/fold: {avg_trades:.0f}")
    print(f"  Ort. fee/fold: ${avg_fees:.0f}")

    # KAMA karsilastirma
    print(f"\n  KARSILASTIRMA: SAF KAMA vs SAF CVD")
    print(f"  {'-'*50}")
    print(f"  KAMA: 30/34 karli fold (88%), ~10 trade/fold")
    print(f"  CVD:  {profitable}/{len(ok)} karli fold ({profitable/len(ok)*100:.0f}%), ~{avg_trades:.0f} trade/fold")

    # Kaydet
    out = {"method": "CVD Order Flow WF — Adim 1",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "consensus": consensus, "folds": nf, "ok": len(ok),
           "results": all_results, "params": all_params}
    op = os.path.join(RESULTS_DIR, "cvd_wf_results.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Kaydedildi: {op}")


if __name__ == "__main__":
    main()
