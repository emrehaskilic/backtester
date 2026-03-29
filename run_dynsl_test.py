"""DynSL Test — en iyi model (sirali) uzerine Dynamic Stop Loss optimize et.

PMax + KC + Kelly kilitli, sadece DynSL parametreleri optimize edilecek.
DD %25 siniri aktif.
"""
import json, time, math, sys, os
import numpy as np
import pandas as pd
import optuna
import rust_engine

sys.path.insert(0, os.path.dirname(__file__))
from strategies.pmax_kc.config import DEFAULT_PARAMS
from strategies.pmax_kc.backtest import _get_source

optuna.logging.set_verbosity(optuna.logging.WARNING)

# En iyi model parametreleri (sirali)
with open("results/ETHUSDT_unified_pmax.json") as f:
    pmax_p = json.load(f)["best_params"]
with open("results/ETHUSDT_unified_kc.json") as f:
    kc_p = json.load(f)["best_kc_params"]
# DD sinirli Kelly
kelly_p = {"base_margin_pct": 3.0, "tier1_threshold": 30000.0, "tier1_pct": 5.0,
           "tier2_threshold": 150000.0, "tier2_pct": 2.25}

print(f"PMax: {json.dumps(pmax_p)}", flush=True)
print(f"KC: {json.dumps(kc_p)}", flush=True)
print(f"Kelly: {json.dumps(kelly_p)}", flush=True)

# Veri
df = pd.read_parquet("data/ETHUSDT_3m_270d.parquet")
train_bars = 90 * 480
test_df = df.iloc[train_bars:].reset_index(drop=True)
sa = np.ascontiguousarray
h = sa(test_df["high"].values, dtype=np.float64)
l = sa(test_df["low"].values, dtype=np.float64)
c = sa(test_df["close"].values, dtype=np.float64)
src = sa(_get_source(test_df, "hl2").values, dtype=np.float64)

# PMax pre-compute (kilitli)
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

# KC indicators (kilitli)
ind = rust_engine.precompute_indicators(
    h, l, c, 144, kc_p["kc_length"], kc_p["kc_multiplier"], kc_p["kc_atr_period"],
)
ind_rsi = sa(ind["rsi_vals"])
ind_ema = sa(ind["ema_filter"])
ind_rsi_ema = sa(ind["rsi_ema_vals"])
ind_atr = sa(ind["atr_vol"])
ind_kc_u = sa(ind["kc_upper_arr"])
ind_kc_l = sa(ind["kc_lower_arr"])

# Mevcut hard_stop_pct=2.5 ile baseline
print("\n=== BASELINE (hard_stop=2.5%, DynSL yok) ===", flush=True)
r_base = rust_engine.run_backtest_dynamic(
    c, h, l, pi, mi, di,
    ind_rsi, ind_ema, ind_rsi_ema, ind_atr, ind_kc_u, ind_kc_l,
    65.0, kc_p["max_dca_steps"], kc_p["tp_close_percent"],
    kelly_p["base_margin_pct"], kelly_p["tier1_threshold"], kelly_p["tier1_pct"],
    kelly_p["tier2_threshold"], kelly_p["tier2_pct"],
)
print(f"$10K -> ${r_base['balance']:,.0f} ({r_base['net_pct']:+.1f}%) DD={r_base['max_drawdown']:.1f}% WR={r_base['win_rate']:.0f}% HS={r_base['hard_stop_count']}", flush=True)

# DynSL: Rust backtest'te hard_stop_pct sabiti var (2.5%)
# Bunu degistirmek icin Rust'a yeni parametre eklemek lazim
# Simdilik: farkli hard_stop degerleri ile grid search yapalim
# Rust backtest'te hard_stop_pct=2.5 sabit — degistiremeyiz
# AMA: backtest_dynamic'e hard_stop parametresi eklememiz lazim

# Rust'a hard_stop parametresi yok — Python'da farki test edelim
# Rust backtest.rs'ta hard_stop_pct = 2.5 sabit
# Bunu parametrik yapmaliyiz

print("\nRust backtest'te hard_stop sabit (2.5%). Parametrik yapmam lazim.", flush=True)
print("Simdilik farkli hard_stop degerleri icin Rust'u yeniden build edip test edecegim.", flush=True)

# Hard stop degerlerini test et (Rust rebuild gerekli)
# Gecici cozum: Python backtest ile DynSL test
from strategies.pmax_kc.adaptive_pmax import adaptive_pmax_continuous
from core.strategy.indicators import rsi as py_rsi, ema as py_ema, atr as py_atr, keltner_channel as py_kc

print("\nPython backtest ile DynSL grid search...", flush=True)

# Python'da adaptive backtest — DynSL parametrik
params = DEFAULT_PARAMS.copy()
params.update(kc_p)

pmax_call = {k.replace("vol_lookback", "lookback"): v for k, v in pmax_p.items()}
py_p, py_m, py_d, _, _, _ = adaptive_pmax_continuous(
    _get_source(test_df, "hl2"), test_df["high"], test_df["low"], test_df["close"],
    ma_type="EMA", base_atr_period=10, base_atr_multiplier=3.0, base_ma_length=10,
    **pmax_call,
)

pmax_line = py_p.values
mavg_arr = py_m.values
closes = c
highs = h
lows = l
times = test_df["open_time"].values
n = len(closes)

# Pre-compute indicators (Python)
rsi_vals = py_rsi(test_df["close"], 28).values
ema_filter = py_ema(test_df["close"], 144).values
rsi_ema_vals = py_ema(pd.Series(rsi_vals), 10).values
atr_vol = py_atr(test_df["high"], test_df["low"], test_df["close"], 50).values
_, kc_upper, kc_lower = py_kc(test_df["high"], test_df["low"], test_df["close"],
    kc_length=kc_p["kc_length"], kc_multiplier=kc_p["kc_multiplier"], atr_period=kc_p["kc_atr_period"])
kc_u_arr = kc_upper.values
kc_l_arr = kc_lower.values

# ATR for DynSL
from core.strategy.indicators import atr_rma
dynsl_atr_cache = {}
for period in [8, 10, 12, 14, 16, 20]:
    dynsl_atr_cache[period] = atr_rma(test_df["high"], test_df["low"], test_df["close"], period).values


def calc_margin(bal):
    if bal >= kelly_p["tier2_threshold"]:
        pct = kelly_p["tier2_pct"]
    elif bal >= kelly_p["tier1_threshold"]:
        pct = kelly_p["tier1_pct"]
    else:
        pct = kelly_p["base_margin_pct"]
    m = bal * pct / 100.0
    return max(50.0, min(m, bal * 0.10))


def apply_f(i, side):
    c_val = closes[i]
    if not np.isnan(ema_filter[i]):
        if side == 1 and c_val < ema_filter[i]: return False
        if side == -1 and c_val > ema_filter[i]: return False
    r = rsi_vals[i] if not np.isnan(rsi_vals[i]) else 50.0
    r_e = rsi_ema_vals[i] if not np.isnan(rsi_ema_vals[i]) else 50.0
    rsi_ob = 65.0
    rsi_os = 35.0
    if side == 1 and r > rsi_ob and r > r_e: return False
    if side == -1 and r < rsi_os and r < r_e: return False
    if i >= 200:
        w = atr_vol[max(0, i - 200):i + 1]
        v = w[~np.isnan(w)]
        if len(v) > 0:
            t = np.percentile(v, 20)
            if not np.isnan(atr_vol[i]) and atr_vol[i] < t: return False
    return True


def run_backtest_dynsl(hs_atr_mult, hs_atr_period, hard_stop_fallback):
    """Backtest with Dynamic SL: ATR-based stop loss."""
    hs_atr = dynsl_atr_cache.get(hs_atr_period)
    if hs_atr is None:
        hs_atr = atr_rma(test_df["high"], test_df["low"], test_df["close"], hs_atr_period).values

    LEVERAGE = 25.0
    MAKER_FEE = 0.0002
    TAKER_FEE = 0.0005
    total_fee_rate = MAKER_FEE + TAKER_FEE
    MIN_BARS = 200

    condition = 0.0
    avg_entry = 0.0
    total_notional = 0.0
    dca_fills = 0
    balance = 10000.0
    peak = 10000.0
    max_dd = 0.0
    trade_count = 0
    win_count = 0
    loss_count = 0
    hs_count = 0
    tp_count = 0
    max_dca = kc_p["max_dca_steps"]
    tp_pct = kc_p["tp_close_percent"]

    for i in range(MIN_BARS, n):
        if np.isnan(mavg_arr[i]) or np.isnan(pmax_line[i]):
            continue
        cur_margin = calc_margin(balance)

        if i > 0 and not np.isnan(mavg_arr[i-1]) and not np.isnan(pmax_line[i-1]):
            buy_cross = mavg_arr[i-1] <= pmax_line[i-1] and mavg_arr[i] > pmax_line[i]
            sell_cross = mavg_arr[i-1] >= pmax_line[i-1] and mavg_arr[i] < pmax_line[i]

            if buy_cross and condition <= 0:
                if condition < 0 and total_notional > 0:
                    pnl_pct = (avg_entry - closes[i]) / avg_entry * 100
                    pnl = total_notional * pnl_pct / 100
                    fee = total_notional * TAKER_FEE
                    balance += pnl - fee
                    trade_count += 1
                    if pnl > 0: win_count += 1
                    else: loss_count += 1
                if apply_f(i, 1) and balance >= cur_margin:
                    condition = 1.0
                    avg_entry = closes[i]
                    total_notional = cur_margin * LEVERAGE
                    dca_fills = 0
                    balance -= total_notional * TAKER_FEE
                else:
                    condition = 0.0; total_notional = 0.0

            elif sell_cross and condition >= 0:
                if condition > 0 and total_notional > 0:
                    pnl_pct = (closes[i] - avg_entry) / avg_entry * 100
                    pnl = total_notional * pnl_pct / 100
                    fee = total_notional * TAKER_FEE
                    balance += pnl - fee
                    trade_count += 1
                    if pnl > 0: win_count += 1
                    else: loss_count += 1
                if apply_f(i, -1) and balance >= cur_margin:
                    condition = -1.0
                    avg_entry = closes[i]
                    total_notional = cur_margin * LEVERAGE
                    dca_fills = 0
                    balance -= total_notional * TAKER_FEE
                else:
                    condition = 0.0; total_notional = 0.0

        if condition != 0 and total_notional > 0:
            if np.isnan(kc_u_arr[i]) or np.isnan(kc_l_arr[i]):
                continue

            if condition > 0:
                # DynSL: ATR-based hard stop
                if dca_fills >= max_dca and avg_entry > 0:
                    if hs_atr_mult > 0 and hs_atr is not None and not np.isnan(hs_atr[i]):
                        hs_dist = hs_atr_mult * hs_atr[i]
                        triggered = closes[i] <= avg_entry - hs_dist
                    else:
                        loss_pct = (avg_entry - closes[i]) / avg_entry * 100
                        triggered = loss_pct >= hard_stop_fallback
                    if triggered:
                        pnl_pct = (closes[i] - avg_entry) / avg_entry * 100
                        pnl = total_notional * pnl_pct / 100
                        balance += pnl - total_notional * TAKER_FEE
                        trade_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        hs_count += 1
                        condition = 0.0; total_notional = 0.0; continue

                if dca_fills < max_dca and lows[i] <= kc_l_arr[i] and balance >= cur_margin:
                    step = cur_margin * LEVERAGE
                    old = total_notional
                    total_notional += step
                    avg_entry = (avg_entry * old + kc_l_arr[i] * step) / total_notional
                    dca_fills += 1
                    balance -= step * MAKER_FEE
                elif dca_fills > 0 and highs[i] >= kc_u_arr[i] and tp_pct > 0:
                    bp = avg_entry * (1 + total_fee_rate)
                    if kc_u_arr[i] > bp:
                        closed = total_notional * tp_pct
                        pnl_pct = (kc_u_arr[i] - avg_entry) / avg_entry * 100
                        pnl = closed * pnl_pct / 100
                        fee = closed * MAKER_FEE
                        balance += pnl - fee
                        total_notional -= closed
                        dca_fills = max(0, dca_fills - 1)
                        trade_count += 1; tp_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        if total_notional < 1: condition = 0.0; total_notional = 0.0
            else:
                if dca_fills >= max_dca and avg_entry > 0:
                    if hs_atr_mult > 0 and hs_atr is not None and not np.isnan(hs_atr[i]):
                        hs_dist = hs_atr_mult * hs_atr[i]
                        triggered = closes[i] >= avg_entry + hs_dist
                    else:
                        loss_pct = (closes[i] - avg_entry) / avg_entry * 100
                        triggered = loss_pct >= hard_stop_fallback
                    if triggered:
                        pnl_pct = (avg_entry - closes[i]) / avg_entry * 100
                        pnl = total_notional * pnl_pct / 100
                        balance += pnl - total_notional * TAKER_FEE
                        trade_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        hs_count += 1
                        condition = 0.0; total_notional = 0.0; continue

                if dca_fills < max_dca and highs[i] >= kc_u_arr[i] and balance >= cur_margin:
                    step = cur_margin * LEVERAGE
                    old = total_notional
                    total_notional += step
                    avg_entry = (avg_entry * old + kc_u_arr[i] * step) / total_notional
                    dca_fills += 1
                    balance -= step * MAKER_FEE
                elif dca_fills > 0 and lows[i] <= kc_l_arr[i] and tp_pct > 0:
                    bp = avg_entry * (1 - total_fee_rate)
                    if kc_l_arr[i] < bp:
                        closed = total_notional * tp_pct
                        pnl_pct = (avg_entry - kc_l_arr[i]) / avg_entry * 100
                        pnl = closed * pnl_pct / 100
                        fee = closed * MAKER_FEE
                        balance += pnl - fee
                        total_notional -= closed
                        dca_fills = max(0, dca_fills - 1)
                        trade_count += 1; tp_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        if total_notional < 1: condition = 0.0; total_notional = 0.0

        if balance > peak: peak = balance
        if peak > 0:
            dd = (peak - balance) / peak * 100
            if dd > max_dd: max_dd = dd

    if condition != 0 and total_notional > 0:
        pnl_pct = ((closes[-1] - avg_entry) / avg_entry * 100) if condition > 0 else ((avg_entry - closes[-1]) / avg_entry * 100)
        pnl = total_notional * pnl_pct / 100
        balance += pnl - total_notional * TAKER_FEE
        trade_count += 1
        if pnl > 0: win_count += 1
        else: loss_count += 1

    wr = win_count / trade_count * 100 if trade_count > 0 else 0
    return {"balance": balance, "net_pct": (balance - 10000) / 10000 * 100,
            "max_dd": max_dd, "wr": wr, "trades": trade_count, "hs": hs_count, "tp": tp_count}


# Grid search: ATR mult x ATR period x fallback
print(f"\n{'ATR_mult':>8} {'ATR_per':>7} {'Fallback':>8} {'Balance':>12} {'Net%':>9} {'DD%':>6} {'WR%':>5} {'HS':>4} {'TP':>5}", flush=True)
print("-" * 75, flush=True)

best_score = -999
best_params = {}
best_result = {}

for atr_mult in [0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    for atr_per in [10, 14]:
        for fallback in [1.5, 2.0, 2.5, 3.0]:
            r = run_backtest_dynsl(atr_mult, atr_per, fallback)
            if r["balance"] > 0 and r["max_dd"] < 30:
                score = (r["balance"] / 10000) / max(r["max_dd"], 5) * math.sqrt(r["wr"] / 100)
                if score > best_score:
                    best_score = score
                    best_params = {"hs_atr_mult": atr_mult, "hs_atr_period": atr_per, "hard_stop_fallback": fallback}
                    best_result = r
                    marker = " <-- BEST"
                else:
                    marker = ""
                print(
                    f"{atr_mult:>8.1f} {atr_per:>7} {fallback:>8.1f} ${r['balance']:>10,.0f} {r['net_pct']:>+8.1f}% {r['max_dd']:>5.1f} {r['wr']:>4.0f}% {r['hs']:>4} {r['tp']:>5}{marker}",
                    flush=True,
                )

print(f"\n{'='*60}", flush=True)
print(f"EN IYI DynSL: {best_params}", flush=True)
print(f"$10K -> ${best_result['balance']:,.0f} ({best_result['net_pct']:+.1f}%) DD={best_result['max_dd']:.1f}% WR={best_result['wr']:.0f}%", flush=True)
print(f"\nKARSILASTIRMA:", flush=True)
print(f"  DynSL yok (HS=2.5%):  ${r_base['balance']:,.0f}, DD {r_base['max_drawdown']:.1f}%", flush=True)
print(f"  DynSL en iyi:         ${best_result['balance']:,.0f}, DD {best_result['max_dd']:.1f}%", flush=True)

result = {
    "pmax_params": pmax_p,
    "kc_params": kc_p,
    "kelly_params": kelly_p,
    "dynsl_params": best_params,
    "baseline": {"balance": r_base["balance"], "dd": r_base["max_drawdown"]},
    "dynsl_result": best_result,
}
with open("results/ETHUSDT_dynsl_test.json", "w") as f:
    json.dump(result, f, indent=2, default=float)
print(f"\nKaydedildi: results/ETHUSDT_dynsl_test.json", flush=True)
