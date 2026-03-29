"""
Saf KAMA 1 Senelik Backtest — konsensus parametreleri.
$1,000 | 25x | Haftalik kar cekimi | 9 ay aktif (2 ay warmup)
"""

import os, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import rust_engine

KAMA_PERIOD = 22
KAMA_FAST = 8
KAMA_SLOW = 31
SLOPE_LB = 14
SLOPE_TH = 0.80
INITIAL = 1000.0

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")

def main():
    print("=" * 100)
    print("  SAF KAMA BACKTEST — Konsensus (period=22, fast=8, slow=31, slope_lb=14, th=0.80)")
    print("  $1,000 | 25x | Haftalik kar cekimi")
    print("=" * 100)

    df = pd.read_parquet(DATA_PATH)
    sa = np.ascontiguousarray
    c = sa(df["close"].values, dtype=np.float64)
    h = sa(df["high"].values, dtype=np.float64)
    l = sa(df["low"].values, dtype=np.float64)
    ts = df["open_time"].values
    print(f"  Mum: {len(df):,}")

    warmup = 90 * 480
    active_ts = int(ts[warmup])
    first_dt = datetime.fromtimestamp(active_ts / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(int(ts[-1]) / 1000, tz=timezone.utc)

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

    print(f"  Hafta: {len(weeks)}\n")
    print(f"  {'Hafta':5s} | {'Tarih':24s} | {'Bitis':>9s} | {'PnL':>10s} | {'PnL%':>7s} | "
          f"{'Cekilen':>9s} | {'Trd':>4s} | {'WR%':>5s} | {'DD%':>6s} | {'Fee':>7s}")
    print(f"  {'-'*100}")

    weekly = []
    total_trades_all = 0

    for week in weeks:
        ws = int(np.searchsorted(ts, week["s"]))
        we = int(np.searchsorted(ts, week["e"]))
        if we <= ws:
            continue

        ss = max(0, ws - 1000)
        r = rust_engine.run_kama_backtest(
            sa(c[ss:we]), sa(h[ss:we]), sa(l[ss:we]),
            KAMA_PERIOD, KAMA_FAST, KAMA_SLOW, SLOPE_LB, SLOPE_TH)

        net = r["net_pct"]
        eq = INITIAL * (1 + net / 100)
        pnl = eq - INITIAL
        profit = max(0, pnl)
        total_trades_all += r["total_trades"]

        wr = {"label": week["label"], "range": week["range"],
              "eq": round(eq, 2), "pnl": round(pnl, 2), "pnl_pct": round(net, 2),
              "withdrawn": round(profit, 2), "trades": r["total_trades"],
              "wr": round(r["win_rate"], 1), "dd": round(r["max_drawdown"], 2),
              "fees": round(r["total_fees"], 2)}
        weekly.append(wr)

        sign = "+" if wr["pnl"] >= 0 else ""
        print(f"  {wr['label']:5s} | {wr['range']:24s} | "
              f"${wr['eq']:>8,.2f} | {sign}${wr['pnl']:>8,.2f} | {sign}{wr['pnl_pct']:>5.1f}% | "
              f"${wr['withdrawn']:>8,.2f} | {wr['trades']:>4d} | {wr['wr']:>4.0f}% | "
              f"{wr['dd']:>5.1f}% | ${wr['fees']:>6.0f}")

    n = len(weekly)
    total_pnl = sum(w["pnl"] for w in weekly)
    total_withdrawn = sum(w["withdrawn"] for w in weekly)
    total_fees = sum(w["fees"] for w in weekly)
    win_w = sum(1 for w in weekly if w["pnl"] > 0)
    loss_w = sum(1 for w in weekly if w["pnl"] < 0)
    best = max(weekly, key=lambda w: w["pnl"])
    worst = min(weekly, key=lambda w: w["pnl"])

    cw = cl = mcw = mcl = 0
    for w in weekly:
        if w["pnl"] > 0:
            cw += 1; cl = 0
            if cw > mcw: mcw = cw
        elif w["pnl"] < 0:
            cl += 1; cw = 0
            if cl > mcl: mcl = cl

    print(f"  {'-'*100}")
    print(f"\n  {'='*60}")
    print(f"  OZET — SAF KAMA")
    print(f"  {'='*60}")
    print(f"  Toplam Hafta          : {n}")
    print(f"  Karli / Zararli       : {win_w} ({win_w/n*100:.0f}%) / {loss_w} ({loss_w/n*100:.0f}%)")
    print(f"  Toplam Net PnL        : ${total_pnl:>10,.2f}")
    print(f"  Toplam Cekilen Kar    : ${total_withdrawn:>10,.2f}")
    print(f"  Toplam Komisyon       : ${total_fees:>10,.2f}")
    print(f"  Toplam Trade          : {total_trades_all}")
    print(f"  Ort. Trade/Hafta      : {total_trades_all/n:.1f}")
    print(f"  Ort. Haftalik PnL     : ${total_pnl/n:>10,.2f} ({total_pnl/n/INITIAL*100:.2f}%)")
    print(f"  En Iyi Hafta          : {best['label']} +${best['pnl']:,.2f}")
    print(f"  En Kotu Hafta         : {worst['label']} ${worst['pnl']:,.2f}")
    print(f"  Ust Uste Karli        : {mcw} hafta")
    print(f"  Ust Uste Zararli      : {mcl} hafta")


if __name__ == "__main__":
    main()
