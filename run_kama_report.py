"""
SAF KAMA — 1 Senelik Haftalik Detayli Rapor.
Her hafta ayri ayri, sonunda ozet istatistikler.
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
    df = pd.read_parquet(DATA_PATH)
    sa = np.ascontiguousarray
    c = sa(df["close"].values, dtype=np.float64)
    h = sa(df["high"].values, dtype=np.float64)
    l = sa(df["low"].values, dtype=np.float64)
    ts = df["open_time"].values

    warmup = 90 * 480
    first_dt = datetime.fromtimestamp(int(ts[warmup]) / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(int(ts[-1]) / 1000, tz=timezone.utc)

    d = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while d.weekday() != 0:
        d += timedelta(days=1)

    weeks = []
    wn = 1
    while d < last_dt:
        we = d + timedelta(days=7)
        weeks.append({"num": wn,
                       "range": f"{d.strftime('%d %b %Y')} - {(we-timedelta(days=1)).strftime('%d %b %Y')}",
                       "s": int(d.timestamp() * 1000), "e": int(we.timestamp() * 1000),
                       "month": d.strftime("%b %Y")})
        d = we
        wn += 1

    # Header
    print()
    print("  " + "=" * 64)
    print("  SIGMA KAPITAL — SAF KAMA HAFTALIK BACKTEST RAPORU")
    print("  " + "=" * 64)
    print(f"  ETHUSDT.P | 3m | 25x | $1,000 Baslangic")
    print(f"  KAMA(period={KAMA_PERIOD}, fast={KAMA_FAST}, slow={KAMA_SLOW})")
    print(f"  Slope(lookback={SLOPE_LB}, threshold={SLOPE_TH})")
    print(f"  DCA/TP/Filtre/Stop: KAPALI — Saf trend sinyali")
    print(f"  Veri: {len(df):,} mum | 2 ay warmup + 9 ay aktif")
    print("  " + "=" * 64)

    results = []
    total_trades = 0

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
        total_trades += r["total_trades"]

        wr = {"num": week["num"], "range": week["range"], "month": week["month"],
              "eq": round(eq, 2), "pnl": round(pnl, 2), "pnl_pct": round(net, 2),
              "withdrawn": round(profit, 2), "trades": r["total_trades"],
              "wr": round(r["win_rate"], 1), "dd": round(r["max_drawdown"], 2),
              "fees": round(r["total_fees"], 2)}
        results.append(wr)

    # Her hafta detayli
    for w in results:
        sign = "+" if w["pnl"] >= 0 else ""
        status = "KARLI" if w["pnl"] > 0 else "ZARARLI" if w["pnl"] < 0 else "BASABAS"

        print()
        print(f"  {'=' * 64}")
        print(f"  HAFTA {w['num']:2d}  |  {w['range']}")
        print(f"  {'-' * 64}")
        print(f"  Baslangic    : $1,000.00")
        print(f"  Bitis        : ${w['eq']:>10,.2f}")
        print(f"  PnL          : {sign}${w['pnl']:>10,.2f} ({sign}{w['pnl_pct']:.1f}%)")
        print(f"  Cekilen Kar  : ${w['withdrawn']:>10,.2f}")
        print(f"  Trade Sayisi : {w['trades']}")
        print(f"  Win Rate     : {w['wr']:.0f}%")
        print(f"  Max Drawdown : {w['dd']:.1f}%")
        print(f"  Komisyon     : ${w['fees']:>10,.2f}")
        print(f"  Durum        : {status}")

    # Ozet
    n = len(results)
    total_pnl = sum(w["pnl"] for w in results)
    total_withdrawn = sum(w["withdrawn"] for w in results)
    total_fees = sum(w["fees"] for w in results)
    win_w = sum(1 for w in results if w["pnl"] > 0)
    loss_w = sum(1 for w in results if w["pnl"] < 0)
    even_w = sum(1 for w in results if w["pnl"] == 0)
    best = max(results, key=lambda w: w["pnl"])
    worst = min(results, key=lambda w: w["pnl"])
    avg_pnl = total_pnl / n if n > 0 else 0
    avg_win = np.mean([w["pnl"] for w in results if w["pnl"] > 0]) if win_w > 0 else 0
    avg_loss = np.mean([w["pnl"] for w in results if w["pnl"] < 0]) if loss_w > 0 else 0

    cw = cl = mcw = mcl = 0
    for w in results:
        if w["pnl"] > 0:
            cw += 1; cl = 0
            if cw > mcw: mcw = cw
        elif w["pnl"] < 0:
            cl += 1; cw = 0
            if cl > mcl: mcl = cl
        else:
            cw = 0; cl = 0

    print()
    print()
    print("  " + "=" * 64)
    print("  OZET ISTATISTIKLER")
    print("  " + "=" * 64)
    print()
    print(f"  GENEL")
    print(f"  {'-' * 50}")
    print(f"  Toplam Hafta            : {n}")
    print(f"  Karli Hafta             : {win_w} ({win_w/n*100:.0f}%)")
    print(f"  Zararli Hafta           : {loss_w} ({loss_w/n*100:.0f}%)")
    print(f"  Basabas Hafta           : {even_w}")
    print()
    print(f"  FINANSAL")
    print(f"  {'-' * 50}")
    print(f"  Toplam Net PnL          : ${total_pnl:>10,.2f}")
    print(f"  Toplam Cekilen Kar      : ${total_withdrawn:>10,.2f}")
    print(f"  Toplam Komisyon         : ${total_fees:>10,.2f}")
    print(f"  Net PnL (Komisyon hari) : ${total_pnl + total_fees:>10,.2f}")
    print()
    print(f"  ORTALAMALAR")
    print(f"  {'-' * 50}")
    print(f"  Ort. Haftalik PnL       : ${avg_pnl:>10,.2f} ({avg_pnl/INITIAL*100:.2f}%)")
    print(f"  Ort. Karli Hafta        : ${avg_win:>10,.2f}")
    print(f"  Ort. Zararli Hafta      : ${avg_loss:>10,.2f}")
    print(f"  Kar/Zarar Orani         : {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "")
    print()
    print(f"  TRADE")
    print(f"  {'-' * 50}")
    print(f"  Toplam Trade            : {total_trades}")
    print(f"  Ort. Trade/Hafta        : {total_trades/n:.1f}")
    print()
    print(f"  EXTREMLER")
    print(f"  {'-' * 50}")
    print(f"  En Iyi Hafta            : Hafta {best['num']} +${best['pnl']:,.2f} ({best['range']})")
    print(f"  En Kotu Hafta           : Hafta {worst['num']} ${worst['pnl']:,.2f} ({worst['range']})")
    print(f"  Ust Uste Karli          : {mcw} hafta")
    print(f"  Ust Uste Zararli        : {mcl} hafta")

    # Aylik ozet
    print()
    print(f"  AYLIK OZET")
    print(f"  {'-' * 64}")
    print(f"  {'Ay':12s} | {'PnL':>10s} | {'Cekilen':>10s} | {'W/L':>5s} | {'Trade':>5s} | {'Komisyon':>8s}")
    print(f"  {'-' * 64}")

    monthly = {}
    for w in results:
        mk = w["month"]
        if mk not in monthly:
            monthly[mk] = {"pnl": 0, "withdrawn": 0, "wins": 0, "losses": 0, "trades": 0, "fees": 0}
        monthly[mk]["pnl"] += w["pnl"]
        monthly[mk]["withdrawn"] += w["withdrawn"]
        monthly[mk]["trades"] += w["trades"]
        monthly[mk]["fees"] += w["fees"]
        if w["pnl"] > 0: monthly[mk]["wins"] += 1
        elif w["pnl"] < 0: monthly[mk]["losses"] += 1

    total_m_pnl = 0
    total_m_withdrawn = 0
    for mk, d in monthly.items():
        sign = "+" if d["pnl"] >= 0 else ""
        total_m_pnl += d["pnl"]
        total_m_withdrawn += d["withdrawn"]
        print(f"  {mk:12s} | {sign}${d['pnl']:>9,.2f} | ${d['withdrawn']:>9,.2f} | "
              f"{d['wins']}/{d['losses']}   | {d['trades']:>5d} | ${d['fees']:>7,.0f}")

    print(f"  {'-' * 64}")
    tsign = "+" if total_m_pnl >= 0 else ""
    print(f"  {'TOPLAM':12s} | {tsign}${total_m_pnl:>9,.2f} | ${total_m_withdrawn:>9,.2f}")

    print()
    print("  " + "=" * 64)
    print("  Rapor sonu")
    print("  " + "=" * 64)


if __name__ == "__main__":
    main()
