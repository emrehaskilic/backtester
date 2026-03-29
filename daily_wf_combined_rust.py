"""
COMBINED Walk-Forward Optimizer — Rust-native TPE + Rayon
KAMA + PMax + CVD+OI + KC Lagged + Graduated DCA/TP
Tum optimizasyon Rust icinde, Python sadece data yukler ve sonuc yazar.

Kullanim: python daily_wf_combined_rust.py
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

# ── Sabitler ──
KAMA_P, KAMA_F, KAMA_S = 20, 6, 33
SLOPE_LB, SLOPE_TH = 14, 0.75

WARMUP_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7
TRIALS_PER_FOLD = 5000
TOP_K_WARMSTART = 5

DATA_VOL = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_vol_11mo.parquet")
DATA_OI = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_OI_5m_11mo.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAM_NAMES = [
    "pmax_atr_period", "pmax_atr_mult", "pmax_ma_length", "pmax_lookback",
    "pmax_flip_window", "pmax_mult_base", "pmax_mult_scale", "pmax_ma_base",
    "pmax_ma_scale", "pmax_atr_base", "pmax_atr_scale", "pmax_update_interval",
    "cvd_period", "imb_weight", "cvd_threshold", "oi_period", "oi_threshold",
    "kc_length", "kc_multiplier", "kc_atr_period",
    "max_dca", "dca_m1", "dca_m2", "dca_m3", "dca_m4",
    "tp1", "tp2", "tp3", "tp4",
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


def load_data():
    df = pd.read_parquet(DATA_VOL)
    oi5 = pd.read_parquet(DATA_OI)
    df['oi'] = np.nan
    oi_map = dict(zip(oi5['open_time'], oi5['open_interest']))
    for idx in range(len(df)):
        ot = df['open_time'].iloc[idx]
        if ot in oi_map:
            df.iloc[idx, df.columns.get_loc('oi')] = oi_map[ot]
    df['oi'] = df['oi'].ffill().fillna(0.0)

    sa = np.ascontiguousarray
    return {
        'closes': sa(df['close'].values, dtype=np.float64),
        'highs': sa(df['high'].values, dtype=np.float64),
        'lows': sa(df['low'].values, dtype=np.float64),
        'buy_vol': sa(df['buy_vol'].values, dtype=np.float64),
        'sell_vol': sa(df['sell_vol'].values, dtype=np.float64),
        'oi': sa(df['oi'].values, dtype=np.float64),
        'total': len(df),
    }


def main():
    t_start = time.time()

    print("=" * 90)
    print("  COMBINED WF OPTIMIZER — Rust-native TPE + Rayon")
    print(f"  KAMA(sabit) + PMax + CVD+OI + KC + GrDCA + GrTP")
    print(f"  29 param | {TRIALS_PER_FOLD:,} trial/fold | Warm-start top-{TOP_K_WARMSTART}")
    print("=" * 90)

    data = load_data()
    total = data['total']
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

        # Tek Rust cagirisi — tum optimizasyon icerde
        r = rust_engine.run_wf_fold(
            data['closes'][:te], data['highs'][:te], data['lows'][:te],
            data['buy_vol'][:te], data['sell_vol'][:te], data['oi'][:te],
            data['closes'][tws:ted], data['highs'][tws:ted], data['lows'][tws:ted],
            data['buy_vol'][tws:ted], data['sell_vol'][tws:ted], data['oi'][tws:ted],
            KAMA_P, KAMA_F, KAMA_S, SLOPE_LB, SLOPE_TH,
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
                "tp": r["tp_count"],
                "rev": r["rev_count"],
                "fees": round(r["total_fees"], 2),
            }
            all_params.append(params_dict)

            # Warm-start for next fold
            warm_start_flat = np.array(r["top5_params"], dtype=np.float64)
            n_warm = r["n_top5"]

            print(f"  Fold {fn:2d}/{nf} | Net: {res['net']:>+8.2f}% DD: {res['dd']:>5.1f}% "
                  f"WR: {res['wr']:>4.0f}% Trd: {res['trades']:>4d} TP:{res['tp']:>3d} Rev:{res['rev']:>3d} "
                  f"| Score: {res['score']:>7.3f} | {el:.0f}s (tot: {elapsed_total/60:.0f}m)", flush=True)
        else:
            res = {"fold": fn, "status": "FAIL", "score": -999, "params": None,
                   "net": 0, "dd": 0, "wr": 0, "trades": 0, "tp": 0, "rev": 0, "fees": 0}
            print(f"  Fold {fn:2d}/{nf} | FAIL | {el:.0f}s (tot: {elapsed_total/60:.0f}m)", flush=True)

        all_results.append(res)

    # ── Konsensus ──
    print(f"\n{'='*90}")
    print(f"  KONSENSUS ANALIZI")
    print(f"{'='*90}")

    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"  Basarili: {len(ok)}/{nf} | Basarisiz: {nf-len(ok)}/{nf}")

    if not all_params:
        print("  HICBIR FOLD BASARILI OLMADI.")
        return

    consensus = {}
    float_step_params = {
        'pmax_atr_mult': 0.1, 'pmax_mult_base': 0.1, 'pmax_mult_scale': 0.1,
        'pmax_ma_scale': 0.1, 'pmax_atr_scale': 0.1,
        'imb_weight': 0.05, 'cvd_threshold': 0.001, 'oi_threshold': 0.001,
        'kc_multiplier': 0.1,
        'dca_m1': 0.1, 'dca_m2': 0.1, 'dca_m3': 0.1, 'dca_m4': 0.1,
        'tp1': 0.05, 'tp2': 0.05, 'tp3': 0.05, 'tp4': 0.05,
    }

    print(f"\n  {'Param':22s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>8s} | {'Max':>8s}")
    print(f"  {'-'*80}")

    for pn in PARAM_NAMES:
        vals = [p[pn] for p in all_params]
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cv = std / mean if mean > 0 else 0

        if pn in float_step_params:
            step = float_step_params[pn]
            med = round(round(med / step) * step, 4)
        else:
            med = int(round(med))

        stab = "MUTLAK" if cv == 0 else "STABIL" if cv < 0.3 else "ORTA" if cv < 0.5 else "DEGISKEN"
        consensus[pn] = med
        print(f"  {pn:22s} | {str(med):>10s} | {cv:>8.4f} | {stab:>12s} | {str(round(min(vals),3)):>8s} | {str(round(max(vals),3)):>8s}")

    profitable = sum(1 for r in ok if r["net"] > 0)
    total_net = sum(r["net"] for r in ok)
    avg_dd = np.mean([r["dd"] for r in ok])
    avg_trades = np.mean([r["trades"] for r in ok])

    print(f"\n  Karli fold      : {profitable}/{len(ok)} ({profitable/len(ok)*100:.0f}%)")
    print(f"  Toplam net (OK) : {total_net:+.2f}%")
    print(f"  Ort. net/fold   : {total_net/len(ok):+.2f}%")
    print(f"  Ort. DD         : {avg_dd:.2f}%")
    print(f"  Ort. trade/fold : {avg_trades:.0f}")

    elapsed_total = time.time() - t_start
    print(f"\n  Toplam sure: {elapsed_total/60:.0f} dk ({elapsed_total/3600:.1f} saat)")

    # ── Kaydet ──
    out = {
        "method": "COMBINED WF — Rust-native TPE + Rayon",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kama": {"period": KAMA_P, "fast": KAMA_F, "slow": KAMA_S,
                 "slope_lb": SLOPE_LB, "slope_th": SLOPE_TH},
        "config": {"trials_per_fold": TRIALS_PER_FOLD, "warmstart_k": TOP_K_WARMSTART},
        "consensus": consensus, "folds": nf, "ok": len(ok),
        "results": all_results, "params": all_params,
        "total_seconds": round(elapsed_total),
    }
    op = os.path.join(RESULTS_DIR, "combined_wf_rust_results.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Kaydedildi: {op}")


if __name__ == "__main__":
    main()
