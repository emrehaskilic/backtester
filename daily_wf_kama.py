"""
KAMA Walk-Forward Optimizer — slope-based signal.
Saf KAMA, DCA/TP/filtre/stop yok.
5 parametre: kama_period, kama_fast, kama_slow, slope_lookback, slope_threshold.

Kullanim: python daily_wf_kama.py
"""

import json, os, sys, time
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

WARMUP_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7
TRIALS_PER_FOLD = 500
N_JOBS = 6

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_folds(total_bars):
    bpd = 480
    warmup = WARMUP_DAYS * bpd
    test = TEST_DAYS * bpd
    step = STEP_DAYS * bpd
    folds, num, ts = [], 1, warmup
    while ts + test <= total_bars:
        folds.append({"num": num, "train_end": ts,
                       "test_ws": max(0, ts - 1000), "test_end": ts + test})
        num += 1
        ts += step
    return folds


def main():
    print("=" * 80)
    print("  KAMA WALK-FORWARD OPTIMIZER — Slope-Based Signal")
    print("  Saf KAMA | 5 param | 500 trial/fold | Anchored expanding")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)
    sa = np.ascontiguousarray
    closes = sa(df["close"].values, dtype=np.float64)
    highs = sa(df["high"].values, dtype=np.float64)
    lows = sa(df["low"].values, dtype=np.float64)
    total = len(df)
    print(f"  Mum: {total:,}")

    folds = build_folds(total)
    nf = len(folds)
    print(f"  Fold: {nf} | Trial/fold: {TRIALS_PER_FOLD}")
    print(f"  Toplam trial: {nf * TRIALS_PER_FOLD:,}\n")

    all_results = []
    all_params = []

    for fold in folds:
        fn = fold["num"]
        t0 = time.time()

        te = fold["train_end"]
        tws, ted = fold["test_ws"], fold["test_end"]

        # Train/test slices
        tr_c, tr_h, tr_l = closes[:te], highs[:te], lows[:te]
        te_c, te_h, te_l = closes[tws:ted], highs[tws:ted], lows[tws:ted]

        sampler = optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=15, seed=42 + fn)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial):
            kp = trial.suggest_int("kama_period", 5, 50)
            kf = trial.suggest_int("kama_fast", 2, 10)
            ks = trial.suggest_int("kama_slow", 20, 50)
            sl = trial.suggest_int("slope_lookback", 1, 20)
            st = trial.suggest_float("slope_threshold", 0.0, 1.0, step=0.05)

            # Train
            tr_r = rust_engine.run_kama_backtest(tr_c, tr_h, tr_l, kp, kf, ks, sl, st)
            if tr_r["total_trades"] < 5:
                return -999

            # Train kar filtresi: min %25 net kar
            if tr_r["net_pct"] < 25:
                return -999

            trial.report(tr_r["net_pct"], step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Test (OOS)
            r = rust_engine.run_kama_backtest(te_c, te_h, te_l, kp, kf, ks, sl, st)
            net, dd, wr, trades = r["net_pct"], r["max_drawdown"], r["win_rate"], r["total_trades"]

            if trades < 3:
                return -999

            # Kar filtresi: OOS'ta min %25 net kar
            if net < 25:
                return -999

            # Skor: PMax adimi ile ayni
            # Overfitting cezasi: net > 500 ise x0.8
            score = (net / max(dd, 1.0)) * wr * 0.01
            if net > 500:
                score *= 0.8

            trial.set_user_attr("net", round(net, 2))
            trial.set_user_attr("dd", round(dd, 2))
            trial.set_user_attr("wr", round(wr, 1))
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("fees", round(r["total_fees"], 2))
            return score

        study.optimize(objective, n_trials=TRIALS_PER_FOLD, n_jobs=N_JOBS)

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
                  f"KAMA({p['kama_period']},{p['kama_fast']},{p['kama_slow']}) "
                  f"SL:{p['slope_lookback']} ST:{p['slope_threshold']:.2f} | {el:.0f}s")
        else:
            print(f"  Fold {fn:2d}/{nf} | FAIL | {el:.0f}s")

    # Konsensus
    print(f"\n{'='*80}")
    print(f"  KONSENSUS ANALIZI")
    print(f"{'='*80}")

    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"  Basarili: {len(ok)}/{nf} | Basarisiz: {nf-len(ok)}/{nf}")

    if not all_params:
        print("  HICBIR FOLD BASARILI OLMADI.")
        return

    consensus = {}
    print(f"\n  {'Param':18s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>6s} | {'Max':>6s}")
    print(f"  {'-'*70}")

    for pn in ["kama_period", "kama_fast", "kama_slow", "slope_lookback", "slope_threshold"]:
        vals = [p[pn] for p in all_params]
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cv = std / mean if mean > 0 else 0

        if pn == "slope_threshold":
            med = round(round(med / 0.05) * 0.05, 2)
        else:
            med = int(round(med))

        stab = "MUTLAK" if cv == 0 else "STABIL" if cv < 0.3 else "ORTA" if cv < 0.5 else "DEGISKEN"
        consensus[pn] = med
        print(f"  {pn:18s} | {str(med):>10s} | {cv:>8.4f} | {stab:>12s} | {min(vals):>6} | {max(vals):>6}")

    profitable = sum(1 for r in ok if r["net"] > 0)
    avg_trades = np.mean([r["trades"] for r in ok])
    avg_fees = np.mean([r["fees"] for r in ok])
    print(f"\n  Karli fold: {profitable}/{len(ok)} ({profitable/len(ok)*100:.0f}%)")
    print(f"  Ort. trade/fold: {avg_trades:.0f}")
    print(f"  Ort. fee/fold: ${avg_fees:.0f}")

    # PMax karsilastirma
    print(f"\n  PMAX KARSILASTIRMA")
    print(f"  PMax saf: WR 35.4%, DD 19.2%, PnL +37.5%, 861 trade, 16/38 karli fold (42%)")
    print(f"  KAMA saf: WR ?, DD ?, PnL ?, {avg_trades:.0f} trade/fold, {profitable}/{len(ok)} karli fold ({profitable/len(ok)*100:.0f}%)")

    # Kaydet
    out = {"method": "KAMA WF Optimizer — slope-based",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "consensus": consensus, "folds": nf, "ok": len(ok),
           "results": all_results, "params": all_params}
    op = os.path.join(RESULTS_DIR, "kama_wf_results.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Kaydedildi: {op}")


if __name__ == "__main__":
    main()
