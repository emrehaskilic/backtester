"""Kelly DynComp — combined PMax+KC kilitli, DD siniri %25, 3000 trial."""
import json, time, math, sys, os
import numpy as np
import pandas as pd
import optuna
import rust_engine

sys.path.insert(0, os.path.dirname(__file__))
from strategies.pmax_kc.config import DEFAULT_PARAMS
from strategies.pmax_kc.backtest import _get_source

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Combined PMax+KC yukle
with open("results/ETHUSDT_unified_combined.json") as f:
    combined = json.load(f)
pmax_p = combined["pmax_params"]
kc_p = combined["kc_params"]
print(f"PMax: {json.dumps(pmax_p)}", flush=True)
print(f"KC: {json.dumps(kc_p)}", flush=True)

# Veri — 270 gun, test kismi (180 gun)
df = pd.read_parquet("data/ETHUSDT_3m_270d.parquet")
bars_per_day = 480
train_bars = 90 * bars_per_day
test_df = df.iloc[train_bars:].reset_index(drop=True)
n = len(test_df)
print(f"Test: {n} bar ({n // bars_per_day} gun)", flush=True)

params = DEFAULT_PARAMS.copy()
sa = np.ascontiguousarray

# PMax + KC pre-compute (1 kez — kilitli)
src = sa(_get_source(test_df, "hl2").values, dtype=np.float64)
h = sa(test_df["high"].values, dtype=np.float64)
l = sa(test_df["low"].values, dtype=np.float64)
c = sa(test_df["close"].values, dtype=np.float64)

pm = rust_engine.compute_adaptive_pmax(
    src, h, l, c, 10, 3.0, 10,
    pmax_p["vol_lookback"], pmax_p["flip_window"],
    pmax_p["mult_base"], pmax_p["mult_scale"],
    pmax_p["ma_base"], pmax_p["ma_scale"],
    pmax_p["atr_base"], pmax_p["atr_scale"],
    pmax_p["update_interval"],
)
pi = sa(pm["pmax_line"])
mi = sa(pm["mavg"])
di = sa(pm["direction"])

ind = rust_engine.precompute_indicators(
    h, l, c, 144, kc_p["kc_length"], kc_p["kc_multiplier"], kc_p["kc_atr_period"],
)
ind_rsi = sa(ind["rsi_vals"])
ind_ema = sa(ind["ema_filter"])
ind_rsi_ema = sa(ind["rsi_ema_vals"])
ind_atr = sa(ind["atr_vol"])
ind_kc_u = sa(ind["kc_upper_arr"])
ind_kc_l = sa(ind["kc_lower_arr"])

print("Pre-compute tamam, 3000 trial basliyor...", flush=True)

sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=30)
pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

best_ever = [float("-inf")]
valid_count = [0]
t0 = time.perf_counter()


def objective(trial):
    kp = {
        "base_margin_pct": trial.suggest_float("base_margin_pct", 1.0, 10.0, step=0.5),
        "tier1_threshold": trial.suggest_float("tier1_threshold", 20000, 50000, step=5000),
        "tier1_pct": trial.suggest_float("tier1_pct", 0.5, 5.0, step=0.25),
        "tier2_threshold": trial.suggest_float("tier2_threshold", 50000, 150000, step=10000),
        "tier2_pct": trial.suggest_float("tier2_pct", 0.25, 3.0, step=0.25),
    }
    try:
        r = rust_engine.run_backtest_dynamic(
            c, h, l, pi, mi, di,
            ind_rsi, ind_ema, ind_rsi_ema, ind_atr, ind_kc_u, ind_kc_l,
            65.0, kc_p["max_dca_steps"], kc_p["tp_close_percent"],
            kp["base_margin_pct"], kp["tier1_threshold"], kp["tier1_pct"],
            kp["tier2_threshold"], kp["tier2_pct"],
        )
        if r["total_trades"] < 10:
            return -999
        if r["max_drawdown"] > 25.0:
            return -999
        if r["balance"] <= 0:
            return -999

        growth = r["balance"] / 10000.0
        score = growth / max(r["max_drawdown"], 5.0) * math.sqrt(r["win_rate"] / 100.0)

        trial.set_user_attr("balance", round(r["balance"], 2))
        trial.set_user_attr("net_pct", round(r["net_pct"], 2))
        trial.set_user_attr("max_dd", round(r["max_drawdown"], 2))
        trial.set_user_attr("trades", r["total_trades"])
        trial.set_user_attr("wr", round(r["win_rate"], 1))
        return score
    except optuna.TrialPruned:
        raise
    except Exception:
        return -999


def callback(study, trial):
    if trial.value is not None and trial.value > -999 and "balance" in trial.user_attrs:
        valid_count[0] += 1
        if trial.value > best_ever[0]:
            best_ever[0] = trial.value
            a = trial.user_attrs
            elapsed = time.perf_counter() - t0
            print(
                f"  [{elapsed:5.0f}s] #{trial.number:4d} ({valid_count[0]} valid) "
                f"$10K -> ${a['balance']:,.0f} ({a['net_pct']:+.1f}%) "
                f"DD={a['max_dd']:.1f}% WR={a['wr']:.0f}% Trades={a['trades']}",
                flush=True,
            )


study.optimize(objective, n_trials=3000, callbacks=[callback], n_jobs=2)

# Sonuc
best = study.best_trial
a = best.user_attrs
elapsed = time.perf_counter() - t0
print(f"\n{'='*60}", flush=True)
print(f"SONUC: 3000 trial ({valid_count[0]} valid), {elapsed:.0f}s", flush=True)
print(f"$10K -> ${a['balance']:,.0f} ({a['net_pct']:+.1f}%) DD={a['max_dd']:.1f}% WR={a['wr']:.0f}%", flush=True)
print(f"\nKELLY PARAMETRELERI:", flush=True)
for k, v in best.params.items():
    print(f"  {k} = {v}", flush=True)

# Onceki ile karsilastir
print(f"\nKARSILASTIRMA:", flush=True)
print(f"  Eski Kelly (DD siniri yok):  $992K, DD %46.6", flush=True)
print(f"  Eski Kelly (DD %25 siniri):  $365K, DD %23.3", flush=True)
print(f"  Yeni Combined+Kelly (DD %25): ${a['balance']:,.0f}, DD {a['max_dd']:.1f}%", flush=True)

# Kaydet
result = {
    "pmax_params": pmax_p,
    "kc_params": kc_p,
    "best_kelly_params": best.params,
    "result": {
        "balance": a["balance"],
        "net_pct": a["net_pct"],
        "max_dd": a["max_dd"],
        "total_trades": a["trades"],
        "win_rate": a["wr"],
    },
    "total_trials": 3000,
    "valid_trials": valid_count[0],
    "elapsed_seconds": round(elapsed, 1),
}
with open("results/ETHUSDT_combined_kelly.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nKaydedildi: results/ETHUSDT_combined_kelly.json", flush=True)
