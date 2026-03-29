"""Trade CSV + PDF rapor olustur — masaustune kaydet."""
import json, csv, numpy as np, pandas as pd, sys, os

sys.path.insert(0, os.path.dirname(__file__))

from strategies.pmax_kc.config import DEFAULT_PARAMS, INITIAL_BALANCE, LEVERAGE, MAKER_FEE, TAKER_FEE
from strategies.pmax_kc.backtest import _get_source
from strategies.pmax_kc.adaptive_pmax import adaptive_pmax_continuous
from core.strategy.indicators import rsi, ema, atr, keltner_channel

DESKTOP = r"C:\Users\emrehaskilic\Desktop"
DD_CAP = True  # DD sinirli versiyon
DYNSL = True   # DynSL versiyonu

# Load params
with open("results/ETHUSDT_unified_pmax.json") as f:
    pmax_p = json.load(f)["best_params"]
with open("results/ETHUSDT_unified_kc.json") as f:
    kc_p = json.load(f)["best_kc_params"]
with open("results/ETHUSDT_unified_kelly.json") as f:
    kelly_data = json.load(f)
    kelly_p = kelly_data["best_kelly_params"]

suffix = "_dynsl" if DYNSL else ("_dd25" if DD_CAP else "")

# Load data
df = pd.read_parquet("data/ETHUSDT_3m_270d.parquet")
train_bars = 90 * 480
test_df = df.iloc[train_bars:].reset_index(drop=True)
n = len(test_df)

params = DEFAULT_PARAMS.copy()
params.update(kc_p)
src = _get_source(test_df, params.get("source", "hl2"))
h, l, c = test_df["high"], test_df["low"], test_df["close"]

# PMax
pmax_call = {k.replace("vol_lookback", "lookback"): v for k, v in pmax_p.items()}
p_arr, m_arr, d_arr, _, _, _ = adaptive_pmax_continuous(
    src, h, l, c, ma_type="EMA",
    base_atr_period=params.get("atr_period", 10),
    base_atr_multiplier=params.get("atr_multiplier", 3.0),
    base_ma_length=params.get("ma_length", 10),
    **pmax_call,
)
pmax_line = p_arr.values
mavg_arr = m_arr.values
closes = c.values
highs = h.values
lows = l.values
times = test_df["open_time"].values

# Indicators
rsi_vals = rsi(c, 28).values
ema_filter = ema(c, params.get("ema_filter_period", 144)).values
rsi_ema_vals = ema(pd.Series(rsi_vals), 10).values
atr_vol = atr(h, l, c, 50).values
_, kc_upper, kc_lower = keltner_channel(h, l, c,
    kc_length=kc_p["kc_length"], kc_multiplier=kc_p["kc_multiplier"], atr_period=kc_p["kc_atr_period"])
kc_u = kc_upper.values
kc_l = kc_lower.values


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
    rsi_ob = params.get("rsi_overbought", 65)
    rsi_os = 100 - rsi_ob
    if side == 1 and r > rsi_ob and r > r_e: return False
    if side == -1 and r < rsi_os and r < r_e: return False
    if i >= 200:
        w = atr_vol[max(0, i - 200):i + 1]
        v = w[~np.isnan(w)]
        if len(v) > 0:
            t = np.percentile(v, 20)
            if not np.isnan(atr_vol[i]) and atr_vol[i] < t: return False
    return True


# Backtest
MIN_BARS = 200
condition = 0.0
avg_entry = 0.0
total_notional = 0.0
dca_fills = 0
balance = INITIAL_BALANCE
peak = INITIAL_BALANCE
max_dd = 0.0
trades = []
max_dca = kc_p["max_dca_steps"]
tp_pct = kc_p["tp_close_percent"]
hard_stop_pct = 2.0 if DYNSL else 2.5
total_fee_rate = MAKER_FEE + TAKER_FEE
entry_bar = 0
win_count = 0
loss_count = 0


def do_close(i, exit_price, reason):
    global balance, condition, total_notional, win_count, loss_count
    if total_notional <= 0:
        return
    if condition > 0:
        pnl_pct = (exit_price - avg_entry) / avg_entry * 100
    else:
        pnl_pct = (avg_entry - exit_price) / avg_entry * 100
    pnl = total_notional * pnl_pct / 100
    fee = total_notional * TAKER_FEE
    balance += pnl - fee
    if pnl > 0:
        win_count += 1
    else:
        loss_count += 1
    side = "LONG" if condition > 0 else "SHORT"
    ts = pd.to_datetime(int(times[i]), unit="ms")
    entry_ts = pd.to_datetime(int(times[entry_bar]), unit="ms")
    trades.append({
        "entry_time": str(entry_ts),
        "exit_time": str(ts),
        "side": side,
        "entry_price": round(avg_entry, 2),
        "exit_price": round(exit_price, 2),
        "margin": round(calc_margin(balance), 2),
        "notional": round(total_notional, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "fee": round(fee, 2),
        "balance_after": round(balance, 2),
        "dca_fills": dca_fills,
        "reason": reason,
    })


for i in range(MIN_BARS, n):
    if np.isnan(mavg_arr[i]) or np.isnan(pmax_line[i]):
        continue

    cur_margin = calc_margin(balance)

    if i > 0 and not np.isnan(mavg_arr[i - 1]) and not np.isnan(pmax_line[i - 1]):
        buy_cross = mavg_arr[i - 1] <= pmax_line[i - 1] and mavg_arr[i] > pmax_line[i]
        sell_cross = mavg_arr[i - 1] >= pmax_line[i - 1] and mavg_arr[i] < pmax_line[i]

        if buy_cross and condition <= 0:
            if condition < 0 and total_notional > 0:
                do_close(i, closes[i], "REVERSAL")
            if apply_f(i, 1) and balance >= cur_margin:
                condition = 1.0
                avg_entry = closes[i]
                total_notional = cur_margin * LEVERAGE
                dca_fills = 0
                entry_bar = i
                balance -= total_notional * TAKER_FEE
            else:
                condition = 0.0
                total_notional = 0.0

        elif sell_cross and condition >= 0:
            if condition > 0 and total_notional > 0:
                do_close(i, closes[i], "REVERSAL")
            if apply_f(i, -1) and balance >= cur_margin:
                condition = -1.0
                avg_entry = closes[i]
                total_notional = cur_margin * LEVERAGE
                dca_fills = 0
                entry_bar = i
                balance -= total_notional * TAKER_FEE
            else:
                condition = 0.0
                total_notional = 0.0

    # KC DCA/TP/HS
    if condition != 0 and total_notional > 0:
        if np.isnan(kc_u[i]) or np.isnan(kc_l[i]):
            continue
        if condition > 0:
            if dca_fills >= max_dca and avg_entry > 0:
                loss = (avg_entry - closes[i]) / avg_entry * 100
                if loss >= hard_stop_pct:
                    do_close(i, closes[i], "HARD_STOP")
                    condition = 0.0
                    total_notional = 0.0
                    continue
            if dca_fills < max_dca and lows[i] <= kc_l[i] and balance >= cur_margin:
                step = cur_margin * LEVERAGE
                old = total_notional
                total_notional += step
                avg_entry = (avg_entry * old + kc_l[i] * step) / total_notional
                dca_fills += 1
                balance -= step * MAKER_FEE
            elif dca_fills > 0 and highs[i] >= kc_u[i] and tp_pct > 0:
                bp = avg_entry * (1 + total_fee_rate)
                if kc_u[i] > bp:
                    closed = total_notional * tp_pct
                    pnl_pct = (kc_u[i] - avg_entry) / avg_entry * 100
                    pnl = closed * pnl_pct / 100
                    fee = closed * MAKER_FEE
                    balance += pnl - fee
                    total_notional -= closed
                    dca_fills = max(0, dca_fills - 1)
                    if pnl > 0: win_count += 1
                    else: loss_count += 1
                    trades.append({
                        "entry_time": str(pd.to_datetime(int(times[entry_bar]), unit="ms")),
                        "exit_time": str(pd.to_datetime(int(times[i]), unit="ms")),
                        "side": "LONG", "entry_price": round(avg_entry, 2),
                        "exit_price": round(kc_u[i], 2), "margin": round(cur_margin, 2),
                        "notional": round(closed, 2), "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2), "fee": round(fee, 2),
                        "balance_after": round(balance, 2), "dca_fills": dca_fills, "reason": "TP",
                    })
                    if total_notional < 1:
                        condition = 0.0
                        total_notional = 0.0
        else:
            if dca_fills >= max_dca and avg_entry > 0:
                loss = (closes[i] - avg_entry) / avg_entry * 100
                if loss >= hard_stop_pct:
                    do_close(i, closes[i], "HARD_STOP")
                    condition = 0.0
                    total_notional = 0.0
                    continue
            if dca_fills < max_dca and highs[i] >= kc_u[i] and balance >= cur_margin:
                step = cur_margin * LEVERAGE
                old = total_notional
                total_notional += step
                avg_entry = (avg_entry * old + kc_u[i] * step) / total_notional
                dca_fills += 1
                balance -= step * MAKER_FEE
            elif dca_fills > 0 and lows[i] <= kc_l[i] and tp_pct > 0:
                bp = avg_entry * (1 - total_fee_rate)
                if kc_l[i] < bp:
                    closed = total_notional * tp_pct
                    pnl_pct = (avg_entry - kc_l[i]) / avg_entry * 100
                    pnl = closed * pnl_pct / 100
                    fee = closed * MAKER_FEE
                    balance += pnl - fee
                    total_notional -= closed
                    dca_fills = max(0, dca_fills - 1)
                    if pnl > 0: win_count += 1
                    else: loss_count += 1
                    trades.append({
                        "entry_time": str(pd.to_datetime(int(times[entry_bar]), unit="ms")),
                        "exit_time": str(pd.to_datetime(int(times[i]), unit="ms")),
                        "side": "SHORT", "entry_price": round(avg_entry, 2),
                        "exit_price": round(kc_l[i], 2), "margin": round(cur_margin, 2),
                        "notional": round(closed, 2), "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2), "fee": round(fee, 2),
                        "balance_after": round(balance, 2), "dca_fills": dca_fills, "reason": "TP",
                    })
                    if total_notional < 1:
                        condition = 0.0
                        total_notional = 0.0

    if balance > peak:
        peak = balance
    if peak > 0:
        dd = (peak - balance) / peak * 100
        if dd > max_dd:
            max_dd = dd

# Close remaining
if condition != 0 and total_notional > 0:
    do_close(n - 1, closes[-1], "END")

print(f"Trades: {len(trades)}")
print(f"Balance: ${balance:,.0f}")
print(f"Max DD: {max_dd:.1f}%")
print(f"Win: {win_count}, Loss: {loss_count}")

# Save CSV
csv_path = os.path.join(DESKTOP, f"ETHUSDT_trades{suffix}.csv")
if trades:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=trades[0].keys())
        w.writeheader()
        w.writerows(trades)
    print(f"CSV saved: {csv_path}")

# Save PDF (text-based report)
report_path = os.path.join(DESKTOP, f"ETHUSDT_strategy_report{suffix}.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("ETHUSDT UNIFIED WALK-FORWARD STRATEGY REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("OZET\n")
    f.write("-" * 40 + "\n")
    f.write(f"Sembol:          ETHUSDT\n")
    f.write(f"Timeframe:       3m\n")
    f.write(f"Test Donemi:     6 ay (180 gun)\n")
    f.write(f"Baslangic Kasa:  $10,000\n")
    f.write(f"Final Kasa:      ${balance:,.0f}\n")
    f.write(f"Net Getiri:      {(balance - 10000) / 10000 * 100:+,.1f}%\n")
    f.write(f"Max Drawdown:    {max_dd:.1f}%\n")
    f.write(f"Toplam Trade:    {len(trades)}\n")
    total = win_count + loss_count
    wr = win_count / total * 100 if total > 0 else 0
    f.write(f"Win Rate:        {wr:.1f}% ({win_count}W / {loss_count}L)\n")
    f.write(f"Leverage:        25x\n\n")

    f.write("PMAX PARAMETRELERI (Adaptive Continuous)\n")
    f.write("-" * 40 + "\n")
    for k, v in pmax_p.items():
        f.write(f"  {k:20s} = {v}\n")
    f.write(f"\n  Base: atr_period=10, atr_multiplier=3.0, ma_length=10, ma_type=EMA, source=hl2\n\n")

    f.write("KELTNER CHANNEL PARAMETRELERI\n")
    f.write("-" * 40 + "\n")
    for k, v in kc_p.items():
        f.write(f"  {k:20s} = {v}\n")
    f.write("\n")

    f.write("KELLY / DYNAMIC COMPOUNDING PARAMETRELERI\n")
    f.write("-" * 40 + "\n")
    for k, v in kelly_p.items():
        f.write(f"  {k:20s} = {v}\n")
    f.write(f"\n  Kasa < ${kelly_p['tier1_threshold']:,.0f}  -> margin = kasa * {kelly_p['base_margin_pct']}%\n")
    f.write(f"  Kasa ${kelly_p['tier1_threshold']:,.0f}-${kelly_p['tier2_threshold']:,.0f} -> margin = kasa * {kelly_p['tier1_pct']}%\n")
    f.write(f"  Kasa > ${kelly_p['tier2_threshold']:,.0f} -> margin = kasa * {kelly_p['tier2_pct']}%\n\n")

    f.write("FILTRELER\n")
    f.write("-" * 40 + "\n")
    f.write(f"  EMA Filter:    EMA(144)\n")
    f.write(f"  RSI Filter:    RSI(28), overbought=65, oversold=35\n")
    f.write(f"  ATR Volume:    ATR(50), bottom 20% filtre\n")
    f.write(f"  Hard Stop:     {hard_stop_pct}% (DCA full sonrasi)\n\n")

    f.write("OPTIMIZASYON SURECI\n")
    f.write("-" * 40 + "\n")
    f.write(f"  ADIM 1 - PMax Kesif:     1000 trial x 25 hafta unified test\n")
    f.write(f"  ADIM 2 - KC Optimize:    1000 trial x 25 hafta, PMax kilitli\n")
    f.write(f"  ADIM 3 - Kelly DynComp:  1000 trial x 180 gun tek backtest\n")
    f.write(f"  Engine: Rust (PyO3) — 178x Python hizlanmasi\n")
    f.write(f"  Sampler: TPESampler(multivariate=True)\n")
    f.write(f"  Pruner: MedianPruner\n")
    f.write(f"  Paralel: n_jobs=6\n\n")

    f.write("=" * 70 + "\n")
    f.write("NOT: Bu backtest sonuclari gecmis veriye dayalidir.\n")
    f.write("Gercek piyasada slippage, funding rate, likidite farklari\n")
    f.write("nedeniyle sonuclar farkli olabilir.\n")
    f.write("=" * 70 + "\n")

print(f"Report saved: {report_path}")
