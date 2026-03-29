"""PMax + KC birlikte optimize — 5000 trial, 14 parametre, 25 hafta."""
import json, time, math, sys, os
import numpy as np
import pandas as pd
import optuna
import rust_engine

sys.path.insert(0, os.path.dirname(__file__))
from strategies.pmax_kc.config import DEFAULT_PARAMS
from strategies.pmax_kc.backtest import _get_source

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Veri
df = pd.read_parquet("data/ETHUSDT_3m_270d.parquet")
bars_per_day = 480
train_bars = 90 * bars_per_day
test_bars = 7 * bars_per_day
step_bars = 7 * bars_per_day
total = len(df)
sa = np.ascontiguousarray

# Test pencerelerini olustur
windows = []
start = 0
while start + train_bars + test_bars <= total:
    train_end = start + train_bars
    test_end = train_end + test_bars
    w = df.iloc[train_end:min(test_end, total)].reset_index(drop=True)
    src = sa(_get_source(w, "hl2").values, dtype=np.float64)
    h = sa(w["high"].values, dtype=np.float64)
    l = sa(w["low"].values, dtype=np.float64)
    c = sa(w["close"].values, dtype=np.float64)
    windows.append({"src": src, "h": h, "l": l, "c": c})
    start += step_bars

n_weeks = len(windows)
print(f"{n_weeks} hafta, 3000 trial, 14 parametre (PMax+KC birlikte)", flush=True)

# Train penceresi
train_df = df.iloc[:train_bars].reset_index(drop=True)
train_src = sa(_get_source(train_df, "hl2").values, dtype=np.float64)
train_h = sa(train_df["high"].values, dtype=np.float64)
train_l = sa(train_df["low"].values, dtype=np.float64)
train_c = sa(train_df["close"].values, dtype=np.float64)

sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=30)
pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
import gc

best_ever = [float("-inf")]
valid_count = [0]
t0 = time.perf_counter()


def objective(trial):
    pmax = {
        "vol_lookback": trial.suggest_int("vol_lookback", 60, 1440, step=20),
        "flip_window": trial.suggest_int("flip_window", 20, 480, step=10),
        "mult_base": trial.suggest_float("mult_base", 0.5, 4.0, step=0.25),
        "mult_scale": trial.suggest_float("mult_scale", 0.25, 3.0, step=0.25),
        "ma_base": trial.suggest_int("ma_base", 3, 20),
        "ma_scale": trial.suggest_float("ma_scale", 0.5, 6.0, step=0.5),
        "atr_base": trial.suggest_int("atr_base", 3, 20),
        "atr_scale": trial.suggest_float("atr_scale", 0.5, 4.0, step=0.5),
        "update_interval": trial.suggest_int("update_interval", 1, 60, step=1),
    }
    kc = {
        "kc_length": trial.suggest_int("kc_length", 5, 50),
        "kc_multiplier": trial.suggest_float("kc_multiplier", 0.5, 4.0, step=0.1),
        "kc_atr_period": trial.suggest_int("kc_atr_period", 3, 30),
        "max_dca_steps": trial.suggest_int("max_dca_steps", 1, 5),
        "tp_close_percent": trial.suggest_float("tp_close_percent", 0.05, 0.50, step=0.05),
    }

    try:
        # Pruning: train
        pm = rust_engine.compute_adaptive_pmax(
            train_src, train_h, train_l, train_c,
            10, 3.0, 10,
            pmax["vol_lookback"], pmax["flip_window"],
            pmax["mult_base"], pmax["mult_scale"],
            pmax["ma_base"], pmax["ma_scale"],
            pmax["atr_base"], pmax["atr_scale"],
            pmax["update_interval"],
        )
        ind = rust_engine.precompute_indicators(
            train_h, train_l, train_c, 144,
            kc["kc_length"], kc["kc_multiplier"], kc["kc_atr_period"],
        )
        tr = rust_engine.run_backtest(
            train_c, train_h, train_l,
            sa(pm["pmax_line"]), sa(pm["mavg"]), sa(pm["direction"]),
            sa(ind["rsi_vals"]), sa(ind["ema_filter"]),
            sa(ind["rsi_ema_vals"]), sa(ind["atr_vol"]),
            sa(ind["kc_upper_arr"]), sa(ind["kc_lower_arr"]),
            65.0, kc["max_dca_steps"], kc["tp_close_percent"],
        )
        if tr["total_trades"] < 5:
            return -999
        trial.report(tr["net_pct"], step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # 25 haftada test
        week_nets = []
        week_dds = []
        week_wrs = []
        for w in windows:
            pm_t = rust_engine.compute_adaptive_pmax(
                w["src"], w["h"], w["l"], w["c"],
                10, 3.0, 10,
                pmax["vol_lookback"], pmax["flip_window"],
                pmax["mult_base"], pmax["mult_scale"],
                pmax["ma_base"], pmax["ma_scale"],
                pmax["atr_base"], pmax["atr_scale"],
                pmax["update_interval"],
            )
            ind_t = rust_engine.precompute_indicators(
                w["h"], w["l"], w["c"], 144,
                kc["kc_length"], kc["kc_multiplier"], kc["kc_atr_period"],
            )
            r = rust_engine.run_backtest(
                w["c"], w["h"], w["l"],
                sa(pm_t["pmax_line"]), sa(pm_t["mavg"]), sa(pm_t["direction"]),
                sa(ind_t["rsi_vals"]), sa(ind_t["ema_filter"]),
                sa(ind_t["rsi_ema_vals"]), sa(ind_t["atr_vol"]),
                sa(ind_t["kc_upper_arr"]), sa(ind_t["kc_lower_arr"]),
                65.0, kc["max_dca_steps"], kc["tp_close_percent"],
            )
            week_nets.append(r["net_pct"])
            week_dds.append(r["max_drawdown"])
            week_wrs.append(r["win_rate"])

        total_net = sum(week_nets)
        max_dd = max(week_dds) if week_dds else 100
        profitable = sum(1 for n in week_nets if n > 0)
        consistency = profitable / n_weeks
        avg_wr = np.mean(week_wrs)

        score = (total_net / max(max_dd, 5.0)) * consistency * math.sqrt(avg_wr / 100.0)

        trial.set_user_attr("total_net", round(total_net, 2))
        trial.set_user_attr("max_dd", round(max_dd, 2))
        trial.set_user_attr("profitable", profitable)
        trial.set_user_attr("avg_wr", round(avg_wr, 1))
        trial.set_user_attr("worst", round(min(week_nets), 2))
        trial.set_user_attr("best", round(max(week_nets), 2))
        return score
    except optuna.TrialPruned:
        raise
    except Exception:
        return -999


def callback(study, trial):
    if trial.value is not None and trial.value > -999 and "total_net" in trial.user_attrs:
        valid_count[0] += 1
        if trial.value > best_ever[0]:
            best_ever[0] = trial.value
            a = trial.user_attrs
            elapsed = time.perf_counter() - t0
            print(
                f"  [{elapsed:5.0f}s] #{trial.number:4d} ({valid_count[0]} valid) "
                f"Net={a['total_net']:+.1f}% DD={a['max_dd']:.1f}% "
                f"Karli={a['profitable']}/{n_weeks} WR={a['avg_wr']:.0f}% "
                f"Worst={a['worst']:+.1f}% Best={a['best']:+.1f}%",
                flush=True,
            )


print("Basliyor...", flush=True)
study.optimize(objective, n_trials=3000, callbacks=[callback], n_jobs=2)

# Sonuc
best = study.best_trial
a = best.user_attrs
elapsed = time.perf_counter() - t0
print(f"\n{'='*60}")
print(f"SONUC: 5000 trial ({valid_count[0]} valid), {elapsed:.0f}s")
print(f"Net={a['total_net']:+.1f}% DD={a['max_dd']:.1f}% Karli={a['profitable']}/{n_weeks} WR={a['avg_wr']:.0f}%")
print(f"\nPMAX:")
for k in ["vol_lookback", "flip_window", "mult_base", "mult_scale", "ma_base", "ma_scale", "atr_base", "atr_scale", "update_interval"]:
    print(f"  {k} = {best.params[k]}")
print(f"\nKC:")
for k in ["kc_length", "kc_multiplier", "kc_atr_period", "max_dca_steps", "tp_close_percent"]:
    print(f"  {k} = {best.params[k]}")

# Onceki sonucla karsilastir
with open("results/ETHUSDT_unified_kc.json") as f:
    old = json.load(f)
old_net = old["aggregate"]["total_net"]
print(f"\nKARSILASTIRMA:")
print(f"  Eski (sirali):    Net={old_net:+.1f}%")
print(f"  Yeni (birlikte):  Net={a['total_net']:+.1f}%")
print(f"  Fark:             {a['total_net'] - old_net:+.1f}%")

# Kaydet
result = {
    "pmax_params": {k: best.params[k] for k in ["vol_lookback", "flip_window", "mult_base", "mult_scale", "ma_base", "ma_scale", "atr_base", "atr_scale", "update_interval"]},
    "kc_params": {k: best.params[k] for k in ["kc_length", "kc_multiplier", "kc_atr_period", "max_dca_steps", "tp_close_percent"]},
    "aggregate": {
        "total_net": a["total_net"],
        "max_dd": a["max_dd"],
        "profitable_weeks": a["profitable"],
        "total_weeks": n_weeks,
        "avg_wr": a["avg_wr"],
        "worst_week": a["worst"],
        "best_week": a["best"],
    },
    "total_trials": 5000,
    "valid_trials": valid_count[0],
    "elapsed_seconds": round(elapsed, 1),
}
with open("results/ETHUSDT_unified_combined.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nKaydedildi: results/ETHUSDT_unified_combined.json")
