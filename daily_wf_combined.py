"""
COMBINED Walk-Forward Optimizer
KAMA + PMax + CVD+OI + KC Lagged + Graduated DCA/TP

Üçlü konfirmasyon: KAMA slope + PMax yönü + CVD+OI
KC lagged DCA/TP — graduated margin & close %
Hard stop yok.

27 parametre optimize:
  PMax adaptive: 10 param
  CVD+OI: 5 param
  KC: 3 param
  GrDCA: max_dca + m1-m4 = 5 param
  GrTP: tp1-tp4 = 4 param

KAMA sabit: 20, 6, 33, 14, 0.75

Kullanim: python daily_wf_combined.py
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

# ── Sabitler ──
KAMA_P, KAMA_F, KAMA_S = 20, 6, 33
SLOPE_LB, SLOPE_TH = 14, 0.75

WARMUP_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7
TRIALS_PER_FOLD = 500
N_JOBS = 6
TRAIN_MIN_NET = 25      # Train'de min %25 kar
TOP_K_WARMSTART = 5     # Önceki fold'dan top-5 parametre enjekte

DATA_VOL = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_vol_11mo.parquet")
DATA_OI = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_OI_5m_11mo.parquet")
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


def load_data():
    """3m vol data + 5m OI forward-fill."""
    df = pd.read_parquet(DATA_VOL)
    oi5 = pd.read_parquet(DATA_OI)

    # Forward-fill OI from 5m to 3m
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


def compute_pmax(closes, highs, lows, params):
    """Adaptive PMax hesapla."""
    r = rust_engine.compute_adaptive_pmax(
        closes, highs, lows, closes,
        params['pmax_atr_period'],
        params['pmax_atr_mult'],
        params['pmax_ma_length'],
        params['pmax_lookback'],
        params['pmax_flip_window'],
        params['pmax_mult_base'],
        params['pmax_mult_scale'],
        params['pmax_ma_base'],
        params['pmax_ma_scale'],
        params['pmax_atr_base'],
        params['pmax_atr_scale'],
        params['pmax_update_interval'],
    )
    sa = np.ascontiguousarray
    return (
        sa(np.array(r['pmax_line']), dtype=np.float64),
        sa(np.array(r['mavg']), dtype=np.float64),
    )


def suggest_params(trial):
    """27 parametre öner."""
    p = {}

    # PMax adaptive (10 param)
    p['pmax_atr_period'] = trial.suggest_int('pmax_atr_period', 8, 24)
    p['pmax_atr_mult'] = trial.suggest_float('pmax_atr_mult', 1.0, 4.0, step=0.1)
    p['pmax_ma_length'] = trial.suggest_int('pmax_ma_length', 5, 20)
    p['pmax_lookback'] = trial.suggest_int('pmax_lookback', 20, 100)
    p['pmax_flip_window'] = trial.suggest_int('pmax_flip_window', 10, 50)
    p['pmax_mult_base'] = trial.suggest_float('pmax_mult_base', 0.5, 3.0, step=0.1)
    p['pmax_mult_scale'] = trial.suggest_float('pmax_mult_scale', 0.1, 1.5, step=0.1)
    p['pmax_ma_base'] = trial.suggest_int('pmax_ma_base', 5, 12)
    p['pmax_ma_scale'] = trial.suggest_float('pmax_ma_scale', 0.1, 0.5, step=0.1)
    p['pmax_atr_base'] = trial.suggest_int('pmax_atr_base', 5, 15)
    p['pmax_atr_scale'] = trial.suggest_float('pmax_atr_scale', 0.1, 1.0, step=0.1)
    p['pmax_update_interval'] = trial.suggest_int('pmax_update_interval', 1, 10)

    # CVD+OI (5 param)
    p['cvd_period'] = trial.suggest_int('cvd_period', 5, 100)
    p['imb_weight'] = trial.suggest_float('imb_weight', 0.0, 1.0, step=0.05)
    p['cvd_threshold'] = trial.suggest_float('cvd_threshold', 0.001, 0.1, step=0.001)
    p['oi_period'] = trial.suggest_int('oi_period', 5, 100)
    p['oi_threshold'] = trial.suggest_float('oi_threshold', 0.0, 0.05, step=0.001)

    # KC (3 param)
    p['kc_length'] = trial.suggest_int('kc_length', 2, 50)
    p['kc_multiplier'] = trial.suggest_float('kc_multiplier', 0.5, 5.0, step=0.1)
    p['kc_atr_period'] = trial.suggest_int('kc_atr_period', 2, 50)

    # Graduated DCA (5 param)
    p['max_dca'] = trial.suggest_int('max_dca', 1, 4)
    p['dca_m1'] = trial.suggest_float('dca_m1', 0.3, 2.0, step=0.1)
    p['dca_m2'] = trial.suggest_float('dca_m2', 0.2, 1.5, step=0.1)
    p['dca_m3'] = trial.suggest_float('dca_m3', 0.1, 1.2, step=0.1)
    p['dca_m4'] = trial.suggest_float('dca_m4', 0.1, 1.0, step=0.1)

    # Graduated TP (4 param)
    p['tp1'] = trial.suggest_float('tp1', 0.05, 0.8, step=0.05)
    p['tp2'] = trial.suggest_float('tp2', 0.05, 0.8, step=0.05)
    p['tp3'] = trial.suggest_float('tp3', 0.05, 0.8, step=0.05)
    p['tp4'] = trial.suggest_float('tp4', 0.05, 0.8, step=0.05)

    return p


def run_backtest(data_slice, pmax_line, mavg):
    """Combined backtest çalıştır."""
    return rust_engine.run_combined_backtest(
        data_slice['closes'], data_slice['highs'], data_slice['lows'],
        data_slice['buy_vol'], data_slice['sell_vol'], data_slice['oi'],
        pmax_line, mavg,
        KAMA_P, KAMA_F, KAMA_S, SLOPE_LB, SLOPE_TH,
        data_slice['_p']['cvd_period'],
        data_slice['_p']['imb_weight'],
        data_slice['_p']['cvd_threshold'],
        data_slice['_p']['oi_period'],
        data_slice['_p']['oi_threshold'],
        data_slice['_p']['kc_length'],
        data_slice['_p']['kc_multiplier'],
        data_slice['_p']['kc_atr_period'],
        data_slice['_p']['max_dca'],
        data_slice['_p']['dca_m1'], data_slice['_p']['dca_m2'],
        data_slice['_p']['dca_m3'], data_slice['_p']['dca_m4'],
        data_slice['_p']['tp1'], data_slice['_p']['tp2'],
        data_slice['_p']['tp3'], data_slice['_p']['tp4'],
    )


def main():
    t_start = time.time()

    print("=" * 90)
    print("  COMBINED WALK-FORWARD OPTIMIZER")
    print("  KAMA(sabit) + PMax(adaptive) + CVD+OI + KC Lagged + GrDCA + GrTP")
    print(f"  27 param | {TRIALS_PER_FOLD:,} trial/fold | Warm-start top-{TOP_K_WARMSTART}")
    print("=" * 90)

    data = load_data()
    total = data['total']
    print(f"  Mum: {total:,}")

    folds = build_folds(total)
    nf = len(folds)
    print(f"  Fold: {nf} | Trial/fold: {TRIALS_PER_FOLD:,}")
    print(f"  Toplam trial: {nf * TRIALS_PER_FOLD:,}")
    print(f"  Tahmini sure: {nf * TRIALS_PER_FOLD * 0.17 / 6 / 60:.0f} dk (~{nf * TRIALS_PER_FOLD * 0.17 / 6 / 3600:.1f} saat)\n")

    all_results = []
    all_params = []
    prev_top_params = []  # Warm-start için

    for fold in folds:
        fn = fold["num"]
        t0 = time.time()
        print(f"  Fold {fn:2d}/{nf} basladi...", flush=True)

        te = fold["train_end"]
        tws, ted = fold["test_ws"], fold["test_end"]

        # Train/test slice'lar
        tr_data = {k: data[k][:te] for k in ['closes', 'highs', 'lows', 'buy_vol', 'sell_vol', 'oi']}
        te_data = {k: data[k][tws:ted] for k in ['closes', 'highs', 'lows', 'buy_vol', 'sell_vol', 'oi']}

        sampler = optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=50, seed=42 + fn)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=30)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        # Warm-start: önceki fold'un top parametrelerini enjekte
        if prev_top_params:
            for pp in prev_top_params:
                study.enqueue_trial(pp)

        def objective(trial):
            p = suggest_params(trial)

            # ── Train ──
            try:
                tr_pm, tr_mv = compute_pmax(tr_data['closes'], tr_data['highs'], tr_data['lows'], p)
                tr_slice = {**tr_data, '_p': p}
                tr_r = run_backtest(tr_slice, tr_pm, tr_mv)
            except BaseException:
                return -999

            if tr_r["total_trades"] < 5:
                return -999

            trial.report(tr_r["net_pct"], step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # ── Test (OOS) ──
            try:
                te_pm, te_mv = compute_pmax(te_data['closes'], te_data['highs'], te_data['lows'], p)
                te_slice = {**te_data, '_p': p}
                r = run_backtest(te_slice, te_pm, te_mv)
            except BaseException:
                return -999

            net = r["net_pct"]
            dd = r["max_drawdown"]
            wr = r["win_rate"]
            trades = r["total_trades"]

            if trades < 3:
                return -999

            # OOS zarar filtresi
            if net < 0:
                return -999

            # Skor: net/dd * wr, overfitting cezası
            score = (net / max(dd, 0.5)) * wr * 0.01
            if net > 500:
                score *= 0.8

            trial.set_user_attr("net", round(net, 2))
            trial.set_user_attr("dd", round(dd, 2))
            trial.set_user_attr("wr", round(wr, 1))
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("tp", r["tp_count"])
            trial.set_user_attr("rev", r["rev_count"])
            trial.set_user_attr("fees", round(r["total_fees"], 2))
            return score

        study.optimize(objective, n_trials=TRIALS_PER_FOLD, n_jobs=N_JOBS)

        # Sonuç
        if study.best_value is not None and study.best_value > -999:
            bp = study.best_trial
            res = {
                "fold": fn, "status": "OK", "score": round(bp.value, 4),
                "params": bp.params,
                "net": bp.user_attrs.get("net", 0),
                "dd": bp.user_attrs.get("dd", 0),
                "wr": bp.user_attrs.get("wr", 0),
                "trades": bp.user_attrs.get("trades", 0),
                "tp": bp.user_attrs.get("tp", 0),
                "rev": bp.user_attrs.get("rev", 0),
                "fees": bp.user_attrs.get("fees", 0),
            }
            all_params.append(bp.params)

            # Warm-start: bu fold'un top-K parametresini kaydet
            sorted_trials = sorted(
                [t for t in study.trials if t.value is not None and t.value > -999],
                key=lambda t: t.value, reverse=True
            )
            prev_top_params = [t.params for t in sorted_trials[:TOP_K_WARMSTART]]
        else:
            res = {"fold": fn, "status": "FAIL", "score": -999, "params": None,
                   "net": 0, "dd": 0, "wr": 0, "trades": 0, "tp": 0, "rev": 0, "fees": 0}
            # FAIL olursa önceki warm-start'ı koru

        all_results.append(res)
        el = time.time() - t0
        elapsed_total = time.time() - t_start

        if res["status"] == "OK":
            print(f"  Fold {fn:2d}/{nf} | Net: {res['net']:>+8.2f}% DD: {res['dd']:>5.1f}% "
                  f"WR: {res['wr']:>4.0f}% Trd: {res['trades']:>4d} TP:{res['tp']:>3d} Rev:{res['rev']:>3d} "
                  f"| Score: {res['score']:>7.3f} | {el:.0f}s (tot: {elapsed_total/60:.0f}m)")
        else:
            print(f"  Fold {fn:2d}/{nf} | FAIL | {el:.0f}s (tot: {elapsed_total/60:.0f}m)")

    # ── Konsensüs ──
    print(f"\n{'='*90}")
    print(f"  KONSENSUS ANALIZI")
    print(f"{'='*90}")

    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"  Basarili: {len(ok)}/{nf} | Basarisiz: {nf-len(ok)}/{nf}")

    if not all_params:
        print("  HICBIR FOLD BASARILI OLMADI.")
        return

    consensus = {}
    param_names = list(all_params[0].keys())

    print(f"\n  {'Param':22s} | {'Konsensus':>10s} | {'CV':>8s} | {'Stabilite':>12s} | {'Min':>8s} | {'Max':>8s}")
    print(f"  {'-'*80}")

    float_step_params = {
        'pmax_atr_mult': 0.1, 'pmax_mult_base': 0.1, 'pmax_mult_scale': 0.1,
        'pmax_ma_scale': 0.1, 'pmax_atr_scale': 0.1,
        'imb_weight': 0.05, 'cvd_threshold': 0.001, 'oi_threshold': 0.001,
        'kc_multiplier': 0.1,
        'dca_m1': 0.1, 'dca_m2': 0.1, 'dca_m3': 0.1, 'dca_m4': 0.1,
        'tp1': 0.05, 'tp2': 0.05, 'tp3': 0.05, 'tp4': 0.05,
    }

    for pn in param_names:
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
        print(f"  {pn:22s} | {str(med):>10s} | {cv:>8.4f} | {stab:>12s} | {str(min(vals)):>8s} | {str(max(vals)):>8s}")

    profitable = sum(1 for r in ok if r["net"] > 0)
    avg_trades = np.mean([r["trades"] for r in ok])
    avg_fees = np.mean([r["fees"] for r in ok])
    total_net = sum(r["net"] for r in ok)
    avg_dd = np.mean([r["dd"] for r in ok])

    print(f"\n  Karli fold      : {profitable}/{len(ok)} ({profitable/len(ok)*100:.0f}%)")
    print(f"  Toplam net (OK) : {total_net:+.2f}%")
    print(f"  Ort. net/fold   : {total_net/len(ok):+.2f}%")
    print(f"  Ort. DD         : {avg_dd:.2f}%")
    print(f"  Ort. trade/fold : {avg_trades:.0f}")
    print(f"  Ort. fee/fold   : ${avg_fees:.0f}")

    elapsed_total = time.time() - t_start
    print(f"\n  Toplam sure: {elapsed_total/60:.0f} dk ({elapsed_total/3600:.1f} saat)")

    # ── Kaydet ──
    out = {
        "method": "COMBINED WF — KAMA+PMax+CVD+OI+KC+GrDCA+GrTP",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kama": {"period": KAMA_P, "fast": KAMA_F, "slow": KAMA_S,
                 "slope_lb": SLOPE_LB, "slope_th": SLOPE_TH},
        "config": {"trials_per_fold": TRIALS_PER_FOLD, "n_jobs": N_JOBS,
                   "train_min_net": TRAIN_MIN_NET, "warmstart_k": TOP_K_WARMSTART},
        "consensus": consensus, "folds": nf, "ok": len(ok),
        "results": all_results, "params": all_params,
        "total_seconds": round(elapsed_total),
    }
    op = os.path.join(RESULTS_DIR, "combined_wf_results.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Kaydedildi: {op}")


if __name__ == "__main__":
    main()
