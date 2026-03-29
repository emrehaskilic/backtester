"""
KAMA + KC — 1 Senelik Haftalik Detayli Rapor.
KAMA kilitli + KC konsensus (kc[i-1]).
"""

import os, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import rust_engine

# KAMA konsensus
KAMA_P, KAMA_F, KAMA_S = 22, 8, 31
SLOPE_LB, SLOPE_TH = 14, 0.80

# KC konsensus (kc[i-1])
KC_L, KC_M, KC_A = 17, 3.4, 13

MAX_DCA = 4
TP_PCT = 0.50
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

    print()
    print("  " + "=" * 64)
    print("  SIGMA KAPITAL — KAMA + KC HAFTALIK BACKTEST RAPORU")
    print("  " + "=" * 64)
    print(f"  ETHUSDT.P | 3m | 25x | $1,000 Baslangic")
    print(f"  KAMA(period={KAMA_P}, fast={KAMA_F}, slow={KAMA_S})")
    print(f"  Slope(lookback={SLOPE_LB}, threshold={SLOPE_TH})")
    print(f"  KC(length={KC_L}, multiplier={KC_M}, atr_period={KC_A}) — kc[i-1]")
    print(f"  DCA: max_dca=4, sabit $300 margin | TP: %50 sabit")
    print(f"  Graduated DCA/TP: KAPALI")
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
        r = rust_engine.run_kama_kc_backtest(
            sa(c[ss:we]), sa(h[ss:we]), sa(l[ss:we]),
            KAMA_P, KAMA_F, KAMA_S, SLOPE_LB, SLOPE_TH,
            KC_L, KC_M, KC_A, MAX_DCA, TP_PCT)

        net = r["net_pct"]
        eq = INITIAL * (1 + net / 100)
        pnl = eq - INITIAL
        profit = max(0, pnl)
        total_trades += r["total_trades"]

        wr = {"num": week["num"], "range": week["range"], "month": week["month"],
              "eq": round(eq, 2), "pnl": round(pnl, 2), "pnl_pct": round(net, 2),
              "withdrawn": round(profit, 2), "trades": r["total_trades"],
              "wr": round(r["win_rate"], 1), "dd": round(r["max_drawdown"], 2),
              "fees": round(r["total_fees"], 2),
              "tp": r["tp_count"], "rev": r["rev_count"]}
        results.append(wr)

    # Her hafta
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
        print(f"  Trade        : {w['trades']} (TP: {w['tp']}, Rev: {w['rev']})")
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
    avg_pnl = total_pnl / n
    avg_win = np.mean([w["pnl"] for w in results if w["pnl"] > 0]) if win_w > 0 else 0
    avg_loss = np.mean([w["pnl"] for w in results if w["pnl"] < 0]) if loss_w > 0 else 0
    best = max(results, key=lambda w: w["pnl"])
    worst = min(results, key=lambda w: w["pnl"])

    cw = cl = mcw = mcl = 0
    for w in results:
        if w["pnl"] > 0: cw += 1; cl = 0; mcw = max(mcw, cw)
        elif w["pnl"] < 0: cl += 1; cw = 0; mcl = max(mcl, cl)
        else: cw = 0; cl = 0

    total_tp = sum(w["tp"] for w in results)
    total_rev = sum(w["rev"] for w in results)

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
    print()
    print(f"  FINANSAL")
    print(f"  {'-' * 50}")
    print(f"  Toplam Net PnL          : ${total_pnl:>10,.2f}")
    print(f"  Toplam Cekilen Kar      : ${total_withdrawn:>10,.2f}")
    print(f"  Toplam Komisyon         : ${total_fees:>10,.2f}")
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
    print(f"  Toplam TP               : {total_tp}")
    print(f"  Toplam Reversal         : {total_rev}")
    print()
    print(f"  EXTREMLER")
    print(f"  {'-' * 50}")
    print(f"  En Iyi Hafta            : Hafta {best['num']} +${best['pnl']:,.2f}")
    print(f"  En Kotu Hafta           : Hafta {worst['num']} ${worst['pnl']:,.2f}")
    print(f"  Ust Uste Karli          : {mcw} hafta")
    print(f"  Ust Uste Zararli        : {mcl} hafta")

    # Saf KAMA karsilastirma
    print()
    print(f"  KARSILASTIRMA: SAF KAMA vs KAMA+KC")
    print(f"  {'-' * 50}")
    print(f"  {'':20s} | {'Saf KAMA':>10s} | {'KAMA+KC':>10s}")
    print(f"  {'-' * 50}")
    print(f"  {'Karli Hafta':20s} | {'17/35':>10s} | {f'{win_w}/{n}':>10s}")
    print(f"  {'Net PnL':20s} | {'$272.75':>10s} | ${total_pnl:>9,.2f}")
    print(f"  {'Cekilen Kar':20s} | {'$1,242.83':>10s} | ${total_withdrawn:>9,.2f}")
    print(f"  {'Toplam Trade':20s} | {'312':>10s} | {total_trades:>10d}")
    print(f"  {'Komisyon':20s} | {'$2,340':>10s} | ${total_fees:>9,.0f}")

    # Aylik
    print()
    print(f"  AYLIK OZET")
    print(f"  {'-' * 64}")
    print(f"  {'Ay':12s} | {'PnL':>10s} | {'Cekilen':>10s} | {'W/L':>5s} | {'Trade':>5s}")
    print(f"  {'-' * 64}")

    monthly = {}
    for w in results:
        mk = w["month"]
        if mk not in monthly:
            monthly[mk] = {"pnl": 0, "withdrawn": 0, "wins": 0, "losses": 0, "trades": 0}
        monthly[mk]["pnl"] += w["pnl"]
        monthly[mk]["withdrawn"] += w["withdrawn"]
        monthly[mk]["trades"] += w["trades"]
        if w["pnl"] > 0: monthly[mk]["wins"] += 1
        elif w["pnl"] < 0: monthly[mk]["losses"] += 1

    for mk, d in monthly.items():
        sign = "+" if d["pnl"] >= 0 else ""
        print(f"  {mk:12s} | {sign}${d['pnl']:>9,.2f} | ${d['withdrawn']:>9,.2f} | "
              f"{d['wins']}/{d['losses']}   | {d['trades']:>5d}")

    print(f"  {'-' * 64}")
    tsign = "+" if total_pnl >= 0 else ""
    print(f"  {'TOPLAM':12s} | {tsign}${total_pnl:>9,.2f} | ${total_withdrawn:>9,.2f}")

    print()
    print("  " + "=" * 64)
    print("  Rapor sonu")
    print("  " + "=" * 64)


if __name__ == "__main__":
    main()
