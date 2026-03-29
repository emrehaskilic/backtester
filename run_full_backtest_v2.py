"""
Full backtest — PMax + KC(kc[i-1]) + Graduated DCA konsensus.
11 ay veri, haftalik kar cekimi, $1,000 baslangic.
"""

import os, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import rust_engine
except ImportError:
    print("HATA: rust_engine bulunamadi")
    exit(1)

# Kilitli parametreler
PMAX = {"vol_lookback": 260, "flip_window": 360, "mult_base": 3.25, "mult_scale": 2.0,
        "ma_base": 11, "ma_scale": 4.5, "atr_base": 15, "atr_scale": 2.0, "update_interval": 55}
KC_LENGTH, KC_MULT, KC_ATR = 20, 2.0, 28
DCA_M = [0.25, 0.25, 0.25, 0.50]
MAX_DCA, TP_PCT = 4, 0.50
INITIAL = 1000.0

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ETHUSDT_3m_aggtrades_11mo.parquet")


def main():
    print("=" * 100)
    print("  FULL BACKTEST — PMax + KC(i-1) + Graduated DCA konsensus")
    print("  $1,000 | 25x | Haftalik kar cekimi | 11 ay")
    print("=" * 100)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Mum: {len(df):,}")

    sa = np.ascontiguousarray
    h = sa(df["high"].values, dtype=np.float64)
    l = sa(df["low"].values, dtype=np.float64)
    c = sa(df["close"].values, dtype=np.float64)
    src = (h + l) / 2.0
    ts = df["open_time"].values

    # PMax
    pm = rust_engine.compute_adaptive_pmax(
        sa(src), sa(h), sa(l), sa(c),
        10, 3.0, 10,
        PMAX["vol_lookback"], PMAX["flip_window"],
        PMAX["mult_base"], PMAX["mult_scale"],
        PMAX["ma_base"], PMAX["ma_scale"],
        PMAX["atr_base"], PMAX["atr_scale"],
        PMAX["update_interval"])
    pmax_line = sa(pm["pmax_line"])
    mavg = sa(pm["mavg"])

    # KC
    ind = rust_engine.precompute_indicators(sa(h), sa(l), sa(c), 144, KC_LENGTH, KC_MULT, KC_ATR)
    kc_u = sa(ind["kc_upper_arr"])
    kc_l = sa(ind["kc_lower_arr"])

    print(f"  Indikatorler hazir")

    # Warmup: ilk 90 gun (43200 bar)
    warmup_bars = 90 * 480
    active_start_ts = int(ts[warmup_bars])

    # Hafta sinirlari
    from datetime import timezone
    first_dt = datetime.fromtimestamp(active_start_ts / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(int(ts[-1]) / 1000, tz=timezone.utc)

    # Ilk Pazartesi
    d = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while d.weekday() != 0:
        d += timedelta(days=1)

    weeks = []
    wn = 1
    while d < last_dt:
        we = d + timedelta(days=7)
        weeks.append({"label": f"W{wn:02d}",
                       "date_range": f"{d.strftime('%d %b')} - {(we-timedelta(days=1)).strftime('%d %b %Y')}",
                       "start_ms": int(d.timestamp() * 1000),
                       "end_ms": int(we.timestamp() * 1000)})
        d = we
        wn += 1

    print(f"  Hafta: {len(weeks)}\n")

    # Haftalik backtest
    weekly = []
    for week in weeks:
        ws = int(np.searchsorted(ts, week["start_ms"]))
        we = int(np.searchsorted(ts, week["end_ms"]))
        if we <= ws:
            continue

        # Slice with warmup
        ss = max(0, ws - 1000)
        r = rust_engine.run_backtest_kc_lagged_graduated(
            sa(c[ss:we]), sa(h[ss:we]), sa(l[ss:we]),
            sa(pmax_line[ss:we]), sa(mavg[ss:we]),
            sa(kc_u[ss:we]), sa(kc_l[ss:we]),
            MAX_DCA, TP_PCT,
            DCA_M[0], DCA_M[1], DCA_M[2], DCA_M[3])

        # $1000 normalize (Rust $10K ile calisir)
        scale = INITIAL / 10000.0
        net = r["net_pct"]
        eq = INITIAL * (1 + net / 100)
        pnl = eq - INITIAL
        profit = max(0, pnl)

        wr = {"label": week["label"], "date_range": week["date_range"],
              "end_equity": round(eq, 2), "pnl": round(pnl, 2),
              "pnl_pct": round(net, 2), "withdrawn": round(profit, 2),
              "trades": r["total_trades"], "wr": round(r["win_rate"], 1),
              "dd": round(r["max_drawdown"], 2),
              "tp": r["tp_count"], "rev": r["rev_count"]}
        weekly.append(wr)

        sign = "+" if wr["pnl"] >= 0 else ""
        print(f"  {wr['label']} | {wr['date_range']:24s} | "
              f"PnL: {sign}${wr['pnl']:>8.2f} ({sign}{wr['pnl_pct']:>5.1f}%) | "
              f"Trd: {wr['trades']:>3d} WR: {wr['wr']:>4.0f}% | "
              f"TP:{wr['tp']:>3d} Rev:{wr['rev']:>3d} | DD: {wr['dd']:>5.1f}%")

    # Rapor
    print(f"\n{'='*100}")
    print(f"  RAPOR — PMax + KC(i-1) + Graduated DCA [{DCA_M}]")
    print(f"{'='*100}")

    n = len(weekly)
    total_pnl = sum(w["pnl"] for w in weekly)
    total_withdrawn = sum(w["withdrawn"] for w in weekly)
    win_w = sum(1 for w in weekly if w["pnl"] > 0)
    loss_w = sum(1 for w in weekly if w["pnl"] < 0)

    print(f"  Toplam Hafta          : {n}")
    print(f"  Karli / Zararli       : {win_w} ({win_w/n*100:.0f}%) / {loss_w} ({loss_w/n*100:.0f}%)")
    print(f"  Toplam Net PnL        : ${total_pnl:>10,.2f}")
    print(f"  Toplam Cekilen Kar    : ${total_withdrawn:>10,.2f}")
    print(f"  Ort. Haftalik PnL     : ${total_pnl/n:>10,.2f} ({total_pnl/n/INITIAL*100:.2f}%)")

    best = max(weekly, key=lambda w: w["pnl"])
    worst = min(weekly, key=lambda w: w["pnl"])
    print(f"  En Iyi Hafta          : {best['label']} ${best['pnl']:>10,.2f}")
    print(f"  En Kotu Hafta         : {worst['label']} ${worst['pnl']:>10,.2f}")


if __name__ == "__main__":
    main()
