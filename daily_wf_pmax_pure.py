"""
Saf PMax Walk-Forward Optimizer — Rust-native TPE + Rayon
Sadece adaptive PMax crossover, reversal ile çıkış.
12 parametre. KC/DCA/TP/CVD/OI yok.

Kullanim: python daily_wf_pmax_pure.py
"""

import json, os, sys, time
import numpy as np
import pandas as pd

try:
    import rust_engine
    print("Rust engine yuklendi")
except ImportError:
    print("HATA: rust_engine bulunamadi.")
    sys.exit(1)

WARMUP_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7
TRIALS_PER_FOLD = 5000

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAM_NAMES = [
    "pmax_atr_period", "pmax_atr_mult", "pmax_ma_length", "pmax_lookback",
    "pmax_flip_window", "pmax_mult_base", "pmax_mult_scale", "pmax_ma_base",
    "pmax_ma_scale", "pmax_atr_base", "pmax_atr_scale", "pmax_update_interval",
]


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
    t_start = time.time()

    print("=" * 80)
    print("  SAF PMAX WF OPTIMIZER — Rust-native TPE + Rayon")
    print(f"  12 param | {TRIALS_PER_FOLD:,} trial/fold | Warm-start top-5")
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
    print(f"  Fold: {nf} | Trial/fold: {TRIALS_PER_FOLD:,}")
    print(f"  Toplam trial: {nf * TRIALS_PER_FOLD:,}\n")

    all_results = []
    all_params = []
    warm_start_flat = np.array([], dtype=np.float64)
    n_warm = 0

    for fold in folds:
        fn = fold["num"]
        t0 = time.time()

        te = fold["train_end"]
        tws, ted = fold["test_ws"], fold["test_end"]

        r = rust_engine.run_pmax_wf_fold(
            closes[:te], highs[:te], lows[:te],
            closes[tws:ted], highs[tws:ted], lows[tws:ted],
            TRIALS_PER_FOLD, 42 + fn,
            warm_start_flat, n_warm,
        )

        el = time.time() - t0
        elapsed_total = time.time() - t_start

        if r["status"] == "OK":
            params_list = list(r["params"])
            params_dict = dict(zip(PARAM_NAMES, params_list))

            res = {
                "fold": fn, "status": "OK", "score": round(r["score"], 4),
                "params": params_dict,
                "net": round(r["net_pct"], 2),
                "dd": round(r["max_drawdown"], 2),
                "wr": round(r["win_rate"], 1),
                "trades": r["total_trades"],
                "fees": round(r["total_fees"], 2),
            }
            all_params.append(params_dict)

            warm_start_flat = np.array(r["top5_params"], dtype=np.float64)
            n_warm = r["n_top5"]

            print(f"  Fold {fn:2d}/{nf} | Net: {res['net']:>+8.2f}% DD: {res['dd']:>5.1f}% "
                  f"WR: {res['wr']:>4.0f}% Trd: {res['trades']:>4d} "
                  f"| Score: {res['score']:>7.3f} | {el:.0f}s (tot: {elapsed_total/60:.0f}m)", flush=True)
        else:
            res = {"fold": fn, "status": "FAIL", "score": -999, "params": None,
                   "net": 0, "dd": 0, "wr": 0, "trades": 0, "fees": 0}
            print(f"  Fold {fn:2d}/{nf} | FAIL | {el:.0f}s (tot: {elapsed_total/60:.0f}m)", flush=True)

        all_results.append(res)

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
    float_step = {'pmax_atr_mult': 0.1, 'pmax_mult_base': 0.1, 'pmax_mult_scale': 0.1,
                  'pmax_ma_scale': 0.1, 'pmax_atr_scale': 0.1}

    print(f"\n  {'Param':22s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>6s} | {'Max':>6s}")
    print(f"  {'-'*75}")

    for pn in PARAM_NAMES:
        vals = [p[pn] for p in all_params]
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cv = std / mean if mean > 0 else 0

        if pn in float_step:
            step = float_step[pn]
            med = round(round(med / step) * step, 2)
        else:
            med = int(round(med))

        stab = "MUTLAK" if cv == 0 else "STABIL" if cv < 0.3 else "ORTA" if cv < 0.5 else "DEGISKEN"
        consensus[pn] = med
        print(f"  {pn:22s} | {str(med):>10s} | {cv:>8.4f} | {stab:>12s} | {min(vals):>6} | {max(vals):>6}")

    profitable = sum(1 for r in ok if r["net"] > 0)
    total_net = sum(r["net"] for r in ok)
    avg_dd = np.mean([r["dd"] for r in ok])

    print(f"\n  Karli fold      : {profitable}/{len(ok)} ({profitable/len(ok)*100:.0f}%)")
    print(f"  Toplam net (OK) : {total_net:+.2f}%")
    print(f"  Ort. net/fold   : {total_net/len(ok):+.2f}%")
    print(f"  Ort. DD         : {avg_dd:.2f}%")

    elapsed_total = time.time() - t_start
    print(f"\n  Toplam sure: {elapsed_total/60:.0f} dk")

    out = {
        "method": "Pure PMax WF — Rust-native TPE + Rayon",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"trials_per_fold": TRIALS_PER_FOLD},
        "consensus": consensus, "folds": nf, "ok": len(ok),
        "results": all_results, "params": all_params,
        "total_seconds": round(elapsed_total),
    }
    op = os.path.join(RESULTS_DIR, "pmax_pure_wf_results.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Kaydedildi: {op}")


if __name__ == "__main__":
    main()
