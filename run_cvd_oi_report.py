"""
CVD + OI Order Flow — 1 Senelik Haftalik Rapor.
5dk bar. Saf sinyal, DCA/TP yok.
"""
import os, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import rust_engine

CVD_PERIOD = 16
IMB_WEIGHT = 0.20
CVD_TH = 0.10
OI_PERIOD = 15
OI_TH = 0.007
INITIAL = 1000.0

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_5m_cvd_oi_11mo.parquet")

def main():
    df = pd.read_parquet(DATA_PATH)
    sa = np.ascontiguousarray
    c = sa(df["close"].values, dtype=np.float64)
    bv = sa(df["buy_vol"].values, dtype=np.float64)
    sv = sa(df["sell_vol"].values, dtype=np.float64)
    oi_raw = df["open_interest"].values.astype(np.float64)
    oi = sa(pd.Series(oi_raw).ffill().bfill().values, dtype=np.float64)
    ts = df["open_time"].values

    warmup = 90 * 288
    first_dt = datetime.fromtimestamp(int(ts[warmup]) / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(int(ts[-1]) / 1000, tz=timezone.utc)
    d = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while d.weekday() != 0:
        d += timedelta(days=1)

    weeks = []
    wn = 1
    while d < last_dt:
        we = d + timedelta(days=7)
        weeks.append({"num": wn, "range": f"{d.strftime('%d %b %Y')} - {(we-timedelta(days=1)).strftime('%d %b %Y')}",
                       "s": int(d.timestamp()*1000), "e": int(we.timestamp()*1000), "month": d.strftime("%b %Y")})
        d = we; wn += 1

    print()
    print("  " + "=" * 64)
    print("  SIGMA KAPITAL — CVD + OI ORDER FLOW HAFTALIK RAPOR")
    print("  " + "=" * 64)
    print(f"  ETHUSDT.P | 5dk | 25x | $1,000 | kasa/40")
    print(f"  CVD(period={CVD_PERIOD}, imb_weight={IMB_WEIGHT}, threshold={CVD_TH})")
    print(f"  OI(period={OI_PERIOD}, threshold={OI_TH})")
    print(f"  DCA/TP: KAPALI — Saf order flow + OI sinyali")
    print(f"  Veri: {len(df):,} bar (5dk) | OI coverage: 100%")
    print("  " + "=" * 64)

    results = []
    tt = 0
    for week in weeks:
        ws = int(np.searchsorted(ts, week["s"]))
        we = int(np.searchsorted(ts, week["e"]))
        if we <= ws: continue
        ss = max(0, ws - 2000)
        r = rust_engine.run_cvd_oi_backtest(
            sa(c[ss:we]), sa(bv[ss:we]), sa(sv[ss:we]), sa(oi[ss:we]),
            CVD_PERIOD, IMB_WEIGHT, CVD_TH, OI_PERIOD, OI_TH)
        net = r["net_pct"]
        eq = INITIAL * (1 + net / 100)
        pnl = eq - INITIAL
        tt += r["total_trades"]
        results.append({"num": week["num"], "range": week["range"], "month": week["month"],
                         "eq": round(eq,2), "pnl": round(pnl,2), "pnl_pct": round(net,2),
                         "withdrawn": round(max(0,pnl),2), "trades": r["total_trades"],
                         "wr": round(r["win_rate"],1), "dd": round(r["max_drawdown"],2),
                         "fees": round(r["total_fees"],2)})

    for w in results:
        sign = "+" if w["pnl"] >= 0 else ""
        st = "KARLI" if w["pnl"] > 0 else "ZARARLI" if w["pnl"] < 0 else "BASABAS"
        print()
        print(f"  {'='*64}")
        print(f"  HAFTA {w['num']:2d}  |  {w['range']}")
        print(f"  {'-'*64}")
        print(f"  Baslangic    : $1,000.00")
        print(f"  Bitis        : ${w['eq']:>10,.2f}")
        print(f"  PnL          : {sign}${w['pnl']:>10,.2f} ({sign}{w['pnl_pct']:.1f}%)")
        print(f"  Cekilen Kar  : ${w['withdrawn']:>10,.2f}")
        print(f"  Trade        : {w['trades']}")
        print(f"  Win Rate     : {w['wr']:.0f}%")
        print(f"  Max Drawdown : {w['dd']:.1f}%")
        print(f"  Komisyon     : ${w['fees']:>10,.2f}")
        print(f"  Durum        : {st}")

    n = len(results)
    tp = sum(w["pnl"] for w in results)
    tw = sum(w["withdrawn"] for w in results)
    tf = sum(w["fees"] for w in results)
    ww = sum(1 for w in results if w["pnl"] > 0)
    lw = sum(1 for w in results if w["pnl"] < 0)
    aw = np.mean([w["pnl"] for w in results if w["pnl"] > 0]) if ww > 0 else 0
    al = np.mean([w["pnl"] for w in results if w["pnl"] < 0]) if lw > 0 else 0
    b = max(results, key=lambda w: w["pnl"])
    wo = min(results, key=lambda w: w["pnl"])
    cw = cl = mcw = mcl = 0
    for w in results:
        if w["pnl"] > 0: cw += 1; cl = 0; mcw = max(mcw, cw)
        elif w["pnl"] < 0: cl += 1; cw = 0; mcl = max(mcl, cl)

    print(f"\n\n  {'='*64}")
    print(f"  OZET ISTATISTIKLER")
    print(f"  {'='*64}")
    print(f"\n  GENEL\n  {'-'*50}")
    print(f"  Toplam Hafta            : {n}")
    print(f"  Karli Hafta             : {ww} ({ww/n*100:.0f}%)")
    print(f"  Zararli Hafta           : {lw} ({lw/n*100:.0f}%)")
    print(f"\n  FINANSAL\n  {'-'*50}")
    print(f"  Toplam Net PnL          : ${tp:>10,.2f}")
    print(f"  Toplam Cekilen Kar      : ${tw:>10,.2f}")
    print(f"  Toplam Komisyon         : ${tf:>10,.2f}")
    print(f"\n  ORTALAMALAR\n  {'-'*50}")
    print(f"  Ort. Haftalik PnL       : ${tp/n:>10,.2f} ({tp/n/INITIAL*100:.2f}%)")
    print(f"  Ort. Karli Hafta        : ${aw:>10,.2f}")
    print(f"  Ort. Zararli Hafta      : ${al:>10,.2f}")
    if al != 0: print(f"  Kar/Zarar Orani         : {abs(aw/al):.2f}x")
    print(f"\n  TRADE\n  {'-'*50}")
    print(f"  Toplam Trade            : {tt}")
    print(f"  Ort. Trade/Hafta        : {tt/n:.1f}")
    print(f"\n  EXTREMLER\n  {'-'*50}")
    print(f"  En Iyi Hafta            : Hafta {b['num']} +${b['pnl']:,.2f}")
    print(f"  En Kotu Hafta           : Hafta {wo['num']} ${wo['pnl']:,.2f}")
    print(f"  Ust Uste Karli          : {mcw} hafta")
    print(f"  Ust Uste Zararli        : {mcl} hafta")

    print(f"\n  KARSILASTIRMA\n  {'-'*55}")
    print(f"  {'':20s} | {'Saf KAMA':>10s} | {'CVD+OI':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'Karli Hafta':20s} | {'17/35':>10s} | {f'{ww}/{n}':>10s}")
    print(f"  {'Net PnL':20s} | {'$272.75':>10s} | ${tp:>9,.2f}")
    print(f"  {'Cekilen Kar':20s} | {'$1,242.83':>10s} | ${tw:>9,.2f}")
    print(f"  {'Toplam Trade':20s} | {'312':>10s} | {tt:>10d}")
    print(f"  {'Komisyon':20s} | {'$2,340':>10s} | ${tf:>9,.0f}")
    if al != 0: print(f"  {'Kar/Zarar':20s} | {'1.36x':>10s} | {abs(aw/al):>9.2f}x")

    print(f"\n  AYLIK OZET\n  {'-'*64}")
    print(f"  {'Ay':12s} | {'PnL':>10s} | {'Cekilen':>10s} | {'W/L':>5s} | {'Trade':>5s}")
    print(f"  {'-'*64}")
    monthly = {}
    for w in results:
        mk = w["month"]
        if mk not in monthly: monthly[mk] = {"pnl":0,"withdrawn":0,"wins":0,"losses":0,"trades":0}
        monthly[mk]["pnl"] += w["pnl"]; monthly[mk]["withdrawn"] += w["withdrawn"]
        monthly[mk]["trades"] += w["trades"]
        if w["pnl"] > 0: monthly[mk]["wins"] += 1
        elif w["pnl"] < 0: monthly[mk]["losses"] += 1
    for mk, d in monthly.items():
        s = "+" if d["pnl"] >= 0 else ""
        print(f"  {mk:12s} | {s}${d['pnl']:>9,.2f} | ${d['withdrawn']:>9,.2f} | {d['wins']}/{d['losses']}   | {d['trades']:>5d}")
    print(f"  {'-'*64}")
    s = "+" if tp >= 0 else ""
    print(f"  {'TOPLAM':12s} | {s}${tp:>9,.2f} | ${tw:>9,.2f}")
    print(f"\n  {'='*64}\n  Rapor sonu\n  {'='*64}")

if __name__ == "__main__":
    main()
