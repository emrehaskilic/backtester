"""
FINAL BACKTEST — 4 Katman Tam Konsensus (kc[i-1])
PMax + KC + Graduated DCA + Graduated TP
$1,000 | 25x | Haftalik kar cekimi | 9 ay aktif (2 ay warmup)
"""

import os, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

try:
    import rust_engine
except ImportError:
    print("HATA: rust_engine bulunamadi")
    exit(1)

# ================================================================
# 4 KATMAN TAM KONSENSUS (kc[i-1])
# ================================================================
PMAX = {"vol_lookback": 260, "flip_window": 360, "mult_base": 3.25, "mult_scale": 2.0,
        "ma_base": 11, "ma_scale": 4.5, "atr_base": 15, "atr_scale": 2.0, "update_interval": 55}
KC_LENGTH, KC_MULT, KC_ATR = 20, 2.0, 28
DCA_M = [0.25, 0.25, 0.25, 0.50]
TP_P = [0.20, 0.60, 0.80, 0.95]
MAX_DCA = 4
INITIAL = 1000.0

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")


def main():
    print("=" * 105)
    print("  FINAL BACKTEST — 4 Katman Tam Konsensus (kc[i-1], look-ahead yok)")
    print("  PMax + KC(20,2.0,28) + DCA[0.25,0.25,0.25,0.50] + TP[0.20,0.60,0.80,0.95]")
    print("  $1,000 | 25x | Haftalik kar cekimi")
    print("=" * 105)

    df = pd.read_parquet(DATA_PATH)
    sa = np.ascontiguousarray
    h = sa(df["high"].values, dtype=np.float64)
    l = sa(df["low"].values, dtype=np.float64)
    c = sa(df["close"].values, dtype=np.float64)
    src = (h + l) / 2.0
    ts = df["open_time"].values

    # PMax
    pm = rust_engine.compute_adaptive_pmax(
        sa(src), sa(h), sa(l), sa(c), 10, 3.0, 10,
        PMAX["vol_lookback"], PMAX["flip_window"],
        PMAX["mult_base"], PMAX["mult_scale"],
        PMAX["ma_base"], PMAX["ma_scale"],
        PMAX["atr_base"], PMAX["atr_scale"],
        PMAX["update_interval"])
    pmax_line, mavg = sa(pm["pmax_line"]), sa(pm["mavg"])

    # KC
    ind = rust_engine.precompute_indicators(sa(h), sa(l), sa(c), 144, KC_LENGTH, KC_MULT, KC_ATR)
    kc_u, kc_l = sa(ind["kc_upper_arr"]), sa(ind["kc_lower_arr"])

    # Warmup: 90 gun
    warmup = 90 * 480
    active_ts = int(ts[warmup])
    first_dt = datetime.fromtimestamp(active_ts / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(int(ts[-1]) / 1000, tz=timezone.utc)

    # Hafta sinirlari
    d = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while d.weekday() != 0:
        d += timedelta(days=1)

    weeks = []
    wn = 1
    while d < last_dt:
        we = d + timedelta(days=7)
        weeks.append({"label": f"W{wn:02d}",
                       "range": f"{d.strftime('%d %b')} - {(we-timedelta(days=1)).strftime('%d %b %Y')}",
                       "s": int(d.timestamp() * 1000), "e": int(we.timestamp() * 1000)})
        d = we
        wn += 1

    print(f"  Mum: {len(df):,} | Hafta: {len(weeks)}\n")

    # Header
    print(f"  {'Hafta':5s} | {'Tarih':24s} | {'Bitis':>9s} | {'PnL':>10s} | {'PnL%':>7s} | "
          f"{'Cekilen':>9s} | {'Trd':>4s} | {'WR%':>5s} | {'TP':>4s} | {'DD%':>6s}")
    print(f"  {'-'*103}")

    weekly = []
    for week in weeks:
        ws = int(np.searchsorted(ts, week["s"]))
        we = int(np.searchsorted(ts, week["e"]))
        if we <= ws:
            continue

        ss = max(0, ws - 1000)
        r = rust_engine.run_backtest_kc_lagged_graduated_tp(
            sa(c[ss:we]), sa(h[ss:we]), sa(l[ss:we]),
            sa(pmax_line[ss:we]), sa(mavg[ss:we]),
            sa(kc_u[ss:we]), sa(kc_l[ss:we]),
            MAX_DCA,
            DCA_M[0], DCA_M[1], DCA_M[2], DCA_M[3],
            TP_P[0], TP_P[1], TP_P[2], TP_P[3])

        net = r["net_pct"]
        eq = INITIAL * (1 + net / 100)
        pnl = eq - INITIAL
        profit = max(0, pnl)

        wr = {"label": week["label"], "range": week["range"],
              "eq": round(eq, 2), "pnl": round(pnl, 2), "pnl_pct": round(net, 2),
              "withdrawn": round(profit, 2), "trades": r["total_trades"],
              "wr": round(r["win_rate"], 1), "dd": round(r["max_drawdown"], 2),
              "tp": r["tp_count"], "fees": round(r["total_fees"], 2)}
        weekly.append(wr)

        sign = "+" if wr["pnl"] >= 0 else ""
        print(f"  {wr['label']:5s} | {wr['range']:24s} | "
              f"${wr['eq']:>8,.2f} | {sign}${wr['pnl']:>8,.2f} | {sign}{wr['pnl_pct']:>5.1f}% | "
              f"${wr['withdrawn']:>8,.2f} | {wr['trades']:>4d} | {wr['wr']:>4.0f}% | "
              f"{wr['tp']:>4d} | {wr['dd']:>5.1f}%")

    # Rapor
    n = len(weekly)
    total_pnl = sum(w["pnl"] for w in weekly)
    total_withdrawn = sum(w["withdrawn"] for w in weekly)
    total_fees = sum(w["fees"] for w in weekly)
    win_w = sum(1 for w in weekly if w["pnl"] > 0)
    loss_w = sum(1 for w in weekly if w["pnl"] < 0)
    best = max(weekly, key=lambda w: w["pnl"])
    worst = min(weekly, key=lambda w: w["pnl"])

    # Consecutive
    cw = cl = mcw = mcl = 0
    for w in weekly:
        if w["pnl"] > 0:
            cw += 1; cl = 0
            if cw > mcw: mcw = cw
        elif w["pnl"] < 0:
            cl += 1; cw = 0
            if cl > mcl: mcl = cl

    print(f"  {'-'*103}")

    print(f"\n  {'='*60}")
    print(f"  FINAL OZET")
    print(f"  {'='*60}")
    print(f"  Strateji            : PMax + KC(i-1) + Grad DCA + Grad TP")
    print(f"  Konsensus KC        : length=20, mult=2.0, atr=28")
    print(f"  Konsensus DCA       : [{DCA_M}]")
    print(f"  Konsensus TP        : [{TP_P}]")
    print(f"  {'-'*60}")
    print(f"  Toplam Hafta        : {n}")
    print(f"  Karli / Zararli     : {win_w} ({win_w/n*100:.0f}%) / {loss_w} ({loss_w/n*100:.0f}%)")
    print(f"  Toplam Net PnL      : ${total_pnl:>10,.2f}")
    print(f"  Toplam Cekilen Kar  : ${total_withdrawn:>10,.2f}")
    print(f"  Toplam Komisyon     : ${total_fees:>10,.2f}")
    print(f"  Ort. Haftalik PnL   : ${total_pnl/n:>10,.2f} ({total_pnl/n/INITIAL*100:.2f}%)")
    print(f"  En Iyi Hafta        : {best['label']} +${best['pnl']:,.2f}")
    print(f"  En Kotu Hafta       : {worst['label']} ${worst['pnl']:,.2f}")
    print(f"  Ust Uste Karli      : {mcw} hafta")
    print(f"  Ust Uste Zararli    : {mcl} hafta")

    # Aylik
    print(f"\n  AYLIK OZET")
    print(f"  {'-'*60}")
    monthly = {}
    for w in weekly:
        parts = w["range"].split(" - ")
        tokens = parts[0].strip().split(" ")
        mk = tokens[1] if len(tokens) > 1 else "?"
        rt = parts[1].strip().split(" ")
        yk = rt[-1] if rt else ""
        key = f"{mk} {yk}"
        if key not in monthly:
            monthly[key] = {"pnl": 0, "withdrawn": 0, "weeks": 0, "wins": 0, "losses": 0}
        monthly[key]["pnl"] += w["pnl"]
        monthly[key]["withdrawn"] += w["withdrawn"]
        monthly[key]["weeks"] += 1
        if w["pnl"] > 0: monthly[key]["wins"] += 1
        elif w["pnl"] < 0: monthly[key]["losses"] += 1

    for mk, d in monthly.items():
        sign = "+" if d["pnl"] >= 0 else ""
        print(f"  {mk:10s} | {sign}${d['pnl']:>9,.2f} | Cekilen: ${d['withdrawn']:>8,.2f} | "
              f"W/L: {d['wins']}/{d['losses']}")


if __name__ == "__main__":
    main()
