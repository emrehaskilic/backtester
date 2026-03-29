"""Leverage testi — eski (sirali) ve yeni (birlikte) modeller, 25x/40x/50x."""
import json, math, sys, os
import numpy as np
import pandas as pd
import rust_engine

sys.path.insert(0, os.path.dirname(__file__))
from strategies.pmax_kc.config import DEFAULT_PARAMS
from strategies.pmax_kc.backtest import _get_source

sa = np.ascontiguousarray

# Veri
df = pd.read_parquet("data/ETHUSDT_3m_270d.parquet")
train_bars = 90 * 480
test_df = df.iloc[train_bars:].reset_index(drop=True)
h = sa(test_df["high"].values, dtype=np.float64)
l = sa(test_df["low"].values, dtype=np.float64)
c = sa(test_df["close"].values, dtype=np.float64)
src = sa(_get_source(test_df, "hl2").values, dtype=np.float64)

# Model 1: Eski (sirali) parametreler
with open("results/ETHUSDT_unified_pmax.json") as f:
    old_pmax = json.load(f)["best_params"]
with open("results/ETHUSDT_unified_kc.json") as f:
    old_kc = json.load(f)["best_kc_params"]
old_kelly = {"base_margin_pct": 3.0, "tier1_threshold": 30000.0, "tier1_pct": 5.0,
             "tier2_threshold": 150000.0, "tier2_pct": 2.25}

# Model 2: Yeni (birlikte) parametreler
with open("results/ETHUSDT_unified_combined.json") as f:
    comb = json.load(f)
new_pmax = comb["pmax_params"]
new_kc = comb["kc_params"]
new_kelly = {"base_margin_pct": 3.5, "tier1_threshold": 25000.0, "tier1_pct": 3.25,
             "tier2_threshold": 120000.0, "tier2_pct": 2.25}

models = [
    ("Eski (sirali)", old_pmax, old_kc, old_kelly),
    ("Yeni (birlikte)", new_pmax, new_kc, new_kelly),
]
leverages = [25, 40, 50]


def run_test(pmax_p, kc_p, kelly_p, leverage):
    """Rust backtest — belirli leverage ile."""
    # PMax
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

    # Indicators
    ind = rust_engine.precompute_indicators(
        h, l, c, 144, kc_p["kc_length"], kc_p["kc_multiplier"], kc_p["kc_atr_period"],
    )

    # Rust backtest.rs LEVERAGE sabiti 25x — degistiremeyiz direkt
    # Ama dynamic margin'da margin_pct * leverage etkisi var
    # Leverage artirmak = margin_pct artirmak ile ayni etki
    # 40x/25x = 1.6 carpan, 50x/25x = 2.0 carpan
    lev_mult = leverage / 25.0
    adj_kelly = {
        "base_margin_pct": kelly_p["base_margin_pct"] * lev_mult,
        "tier1_threshold": kelly_p["tier1_threshold"],
        "tier1_pct": kelly_p["tier1_pct"] * lev_mult,
        "tier2_threshold": kelly_p["tier2_threshold"],
        "tier2_pct": kelly_p["tier2_pct"] * lev_mult,
    }

    r = rust_engine.run_backtest_dynamic(
        c, h, l, pi, mi, di,
        sa(ind["rsi_vals"]), sa(ind["ema_filter"]),
        sa(ind["rsi_ema_vals"]), sa(ind["atr_vol"]),
        sa(ind["kc_upper_arr"]), sa(ind["kc_lower_arr"]),
        65.0, kc_p["max_dca_steps"], kc_p["tp_close_percent"],
        adj_kelly["base_margin_pct"], adj_kelly["tier1_threshold"], adj_kelly["tier1_pct"],
        adj_kelly["tier2_threshold"], adj_kelly["tier2_pct"],
    )
    return r


print(f"{'Model':<20} {'Lev':>4} {'Balance':>12} {'Net%':>10} {'DD%':>6} {'WR%':>5} {'Trades':>7}", flush=True)
print("-" * 70, flush=True)

for name, pmax_p, kc_p, kelly_p in models:
    for lev in leverages:
        r = run_test(pmax_p, kc_p, kelly_p, lev)
        print(
            f"{name:<20} {lev:>3}x ${r['balance']:>10,.0f} {r['net_pct']:>+9.1f}% {r['max_drawdown']:>5.1f} {r['win_rate']:>4.0f}% {r['total_trades']:>6}",
            flush=True,
        )
    print(flush=True)
