"""DynSL Test — eski sirali model, 40x leverage."""
import json, time, math, sys, os
import numpy as np
import pandas as pd
import rust_engine

sys.path.insert(0, os.path.dirname(__file__))
from strategies.pmax_kc.config import DEFAULT_PARAMS
from strategies.pmax_kc.backtest import _get_source
from strategies.pmax_kc.adaptive_pmax import adaptive_pmax_continuous
from core.strategy.indicators import rsi as py_rsi, ema as py_ema, atr as py_atr, keltner_channel as py_kc, atr_rma

# Parametreler (sirali model)
with open("results/ETHUSDT_unified_pmax.json") as f:
    pmax_p = json.load(f)["best_params"]
with open("results/ETHUSDT_unified_kc.json") as f:
    kc_p = json.load(f)["best_kc_params"]
kelly_p = {"base_margin_pct": 3.0, "tier1_threshold": 30000.0, "tier1_pct": 5.0,
           "tier2_threshold": 150000.0, "tier2_pct": 2.25}

# 40x = margin_pct * 1.6
LEV_MULT = 40.0 / 25.0
kelly_40x = {
    "base_margin_pct": kelly_p["base_margin_pct"] * LEV_MULT,
    "tier1_threshold": kelly_p["tier1_threshold"],
    "tier1_pct": kelly_p["tier1_pct"] * LEV_MULT,
    "tier2_threshold": kelly_p["tier2_threshold"],
    "tier2_pct": kelly_p["tier2_pct"] * LEV_MULT,
}

print(f"40x Kelly (adjusted): {json.dumps(kelly_40x)}", flush=True)

# Veri
df = pd.read_parquet("data/ETHUSDT_3m_270d.parquet")
train_bars = 90 * 480
test_df = df.iloc[train_bars:].reset_index(drop=True)
n = len(test_df)
sa = np.ascontiguousarray
params = DEFAULT_PARAMS.copy()
params.update(kc_p)

# PMax
pmax_call = {k.replace("vol_lookback", "lookback"): v for k, v in pmax_p.items()}
py_p, py_m, py_d, _, _, _ = adaptive_pmax_continuous(
    _get_source(test_df, "hl2"), test_df["high"], test_df["low"], test_df["close"],
    ma_type="EMA", base_atr_period=10, base_atr_multiplier=3.0, base_ma_length=10,
    **pmax_call,
)
pmax_line = py_p.values
mavg_arr = py_m.values
closes = test_df["close"].values
highs = test_df["high"].values
lows = test_df["low"].values

# Indicators
rsi_vals = py_rsi(test_df["close"], 28).values
ema_filter = py_ema(test_df["close"], 144).values
rsi_ema_vals = py_ema(pd.Series(rsi_vals), 10).values
atr_vol = py_atr(test_df["high"], test_df["low"], test_df["close"], 50).values
_, kc_upper, kc_lower = py_kc(test_df["high"], test_df["low"], test_df["close"],
    kc_length=kc_p["kc_length"], kc_multiplier=kc_p["kc_multiplier"], atr_period=kc_p["kc_atr_period"])
kc_u_arr = kc_upper.values
kc_l_arr = kc_lower.values

# ATR cache for DynSL
dynsl_atr_cache = {}
for period in [8, 10, 12, 14, 16, 20]:
    dynsl_atr_cache[period] = atr_rma(test_df["high"], test_df["low"], test_df["close"], period).values


def calc_margin(bal):
    if bal >= kelly_40x["tier2_threshold"]:
        pct = kelly_40x["tier2_pct"]
    elif bal >= kelly_40x["tier1_threshold"]:
        pct = kelly_40x["tier1_pct"]
    else:
        pct = kelly_40x["base_margin_pct"]
    return max(50.0, min(bal * pct / 100.0, bal * 0.10))


def apply_f(i, side):
    c_val = closes[i]
    if not np.isnan(ema_filter[i]):
        if side == 1 and c_val < ema_filter[i]: return False
        if side == -1 and c_val > ema_filter[i]: return False
    r = rsi_vals[i] if not np.isnan(rsi_vals[i]) else 50.0
    r_e = rsi_ema_vals[i] if not np.isnan(rsi_ema_vals[i]) else 50.0
    if side == 1 and r > 65 and r > r_e: return False
    if side == -1 and r < 35 and r < r_e: return False
    if i >= 200:
        w = atr_vol[max(0, i-200):i+1]
        v = w[~np.isnan(w)]
        if len(v) > 0:
            t = np.percentile(v, 20)
            if not np.isnan(atr_vol[i]) and atr_vol[i] < t: return False
    return True


def run_test(hs_atr_mult, hs_atr_period, hard_stop_fallback):
    hs_atr = dynsl_atr_cache.get(hs_atr_period)
    LEVERAGE = 25.0  # Rust sabit, margin_pct ile ayarladik
    MAKER_FEE = 0.0002
    TAKER_FEE = 0.0005
    total_fee_rate = MAKER_FEE + TAKER_FEE
    MIN_BARS = 200

    condition = 0.0; avg_entry = 0.0; total_notional = 0.0; dca_fills = 0
    balance = 10000.0; peak = 10000.0; max_dd = 0.0
    trade_count = 0; win_count = 0; loss_count = 0; hs_count = 0; tp_count = 0
    max_dca = kc_p["max_dca_steps"]; tp_pct = kc_p["tp_close_percent"]

    for i in range(MIN_BARS, n):
        if np.isnan(mavg_arr[i]) or np.isnan(pmax_line[i]): continue
        cur_margin = calc_margin(balance)

        if i > 0 and not np.isnan(mavg_arr[i-1]) and not np.isnan(pmax_line[i-1]):
            buy_cross = mavg_arr[i-1] <= pmax_line[i-1] and mavg_arr[i] > pmax_line[i]
            sell_cross = mavg_arr[i-1] >= pmax_line[i-1] and mavg_arr[i] < pmax_line[i]

            if buy_cross and condition <= 0:
                if condition < 0 and total_notional > 0:
                    pnl = total_notional * ((avg_entry - closes[i]) / avg_entry * 100) / 100
                    balance += pnl - total_notional * TAKER_FEE
                    trade_count += 1
                    if pnl > 0: win_count += 1
                    else: loss_count += 1
                if apply_f(i, 1) and balance >= cur_margin:
                    condition = 1.0; avg_entry = closes[i]
                    total_notional = cur_margin * LEVERAGE; dca_fills = 0
                    balance -= total_notional * TAKER_FEE
                else: condition = 0.0; total_notional = 0.0

            elif sell_cross and condition >= 0:
                if condition > 0 and total_notional > 0:
                    pnl = total_notional * ((closes[i] - avg_entry) / avg_entry * 100) / 100
                    balance += pnl - total_notional * TAKER_FEE
                    trade_count += 1
                    if pnl > 0: win_count += 1
                    else: loss_count += 1
                if apply_f(i, -1) and balance >= cur_margin:
                    condition = -1.0; avg_entry = closes[i]
                    total_notional = cur_margin * LEVERAGE; dca_fills = 0
                    balance -= total_notional * TAKER_FEE
                else: condition = 0.0; total_notional = 0.0

        if condition != 0 and total_notional > 0:
            if np.isnan(kc_u_arr[i]) or np.isnan(kc_l_arr[i]): continue

            if condition > 0:
                if dca_fills >= max_dca and avg_entry > 0:
                    if hs_atr_mult > 0 and hs_atr is not None and not np.isnan(hs_atr[i]):
                        triggered = closes[i] <= avg_entry - hs_atr_mult * hs_atr[i]
                    else:
                        triggered = (avg_entry - closes[i]) / avg_entry * 100 >= hard_stop_fallback
                    if triggered:
                        pnl = total_notional * ((closes[i] - avg_entry) / avg_entry * 100) / 100
                        balance += pnl - total_notional * TAKER_FEE
                        trade_count += 1; hs_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        condition = 0.0; total_notional = 0.0; continue
                if dca_fills < max_dca and lows[i] <= kc_l_arr[i] and balance >= cur_margin:
                    step = cur_margin * LEVERAGE; old = total_notional
                    total_notional += step
                    avg_entry = (avg_entry * old + kc_l_arr[i] * step) / total_notional
                    dca_fills += 1; balance -= step * MAKER_FEE
                elif dca_fills > 0 and highs[i] >= kc_u_arr[i] and tp_pct > 0:
                    bp = avg_entry * (1 + total_fee_rate)
                    if kc_u_arr[i] > bp:
                        closed = total_notional * tp_pct
                        pnl = closed * ((kc_u_arr[i] - avg_entry) / avg_entry * 100) / 100
                        balance += pnl - closed * MAKER_FEE
                        total_notional -= closed; dca_fills = max(0, dca_fills - 1)
                        trade_count += 1; tp_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        if total_notional < 1: condition = 0.0; total_notional = 0.0
            else:
                if dca_fills >= max_dca and avg_entry > 0:
                    if hs_atr_mult > 0 and hs_atr is not None and not np.isnan(hs_atr[i]):
                        triggered = closes[i] >= avg_entry + hs_atr_mult * hs_atr[i]
                    else:
                        triggered = (closes[i] - avg_entry) / avg_entry * 100 >= hard_stop_fallback
                    if triggered:
                        pnl = total_notional * ((avg_entry - closes[i]) / avg_entry * 100) / 100
                        balance += pnl - total_notional * TAKER_FEE
                        trade_count += 1; hs_count += 1
                        if pnl > 0: win_count += 1
                        else: loss_count += 1
                        condition = 0.0; total_notional = 0.0; continue
                if dca_fills < max_dca and highs[i] >= kc_u_arr[i] and balance >= cur_margin:
                    step = cur_margin * LEVERAGE; old = total_notional
                    total_notional += step
                    avg_entry = (avg_entry * old + kc_u_arr[i] * step) / total_notional
                    dca_fills += 1; balance -= step * MAKER_FEE
                elif dca_fills > 0 and lows[i] <= kc_l_arr[i] and tp_pct > 0:
                    bp = avg_entry * (1 - total_fee_rate)
                    if kc_l_arr[i] < bp:
                        closed = total_notional * tp_pct
                        pnl = closed * ((avg_entry - kc_l_arr[i]) / avg_entry * 100) / 100
                        balance += pnl - closed * MAKER_FEE
                        total_notional -= closed; dca_fills = max(0, dca_fills - 1)
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
        balance += total_notional * pnl_pct / 100 - total_notional * TAKER_FEE
        trade_count += 1
        if total_notional * pnl_pct / 100 > 0: win_count += 1
        else: loss_count += 1

    wr = win_count / trade_count * 100 if trade_count > 0 else 0
    return {"balance": balance, "net_pct": (balance-10000)/10000*100, "max_dd": max_dd,
            "wr": wr, "trades": trade_count, "hs": hs_count, "tp": tp_count}


# Grid search
print(f"\n{'ATR_m':>6} {'ATR_p':>5} {'HS%':>5} {'Balance':>12} {'Net%':>10} {'DD%':>6} {'WR%':>5} {'HS':>4}", flush=True)
print("-" * 60, flush=True)

best_score = -999
best_params = {}
best_r = {}

for atr_mult in [0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    for atr_per in [10, 14]:
        for fallback in [1.0, 1.5, 2.0, 2.5, 3.0]:
            r = run_test(atr_mult, atr_per, fallback)
            if r["balance"] > 0:
                score = (r["balance"]/10000) / max(r["max_dd"], 5) * math.sqrt(r["wr"]/100)
                if score > best_score:
                    best_score = score
                    best_params = {"hs_atr_mult": atr_mult, "hs_atr_period": atr_per, "hs_fallback": fallback}
                    best_r = r
                    m = " <-- BEST"
                else: m = ""
                print(f"{atr_mult:>6.1f} {atr_per:>5} {fallback:>5.1f} ${r['balance']:>10,.0f} {r['net_pct']:>+9.1f}% {r['max_dd']:>5.1f} {r['wr']:>4.0f}% {r['hs']:>4}{m}", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"EN IYI 40x DynSL: {best_params}", flush=True)
print(f"$10K -> ${best_r['balance']:,.0f} ({best_r['net_pct']:+.1f}%) DD={best_r['max_dd']:.1f}% WR={best_r['wr']:.0f}%", flush=True)
print(f"\nBaseline 40x (HS=2.5%): $1,215,272 DD=48.0%", flush=True)
