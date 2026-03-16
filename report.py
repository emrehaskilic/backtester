"""
Optimizasyon sonuç raporu.
Kullanım: python report.py
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    results_path = os.path.join(RESULTS_DIR, "optimization_results.json")

    if not os.path.exists(results_path):
        print("Hata: Once optimize.py calistirin!")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    print("=" * 70)
    print("  SWINGINESS TICK REPLAY OPTIMIZASYON RAPORU")
    print("=" * 70)

    for symbol, results in all_results.items():
        if not results:
            print(f"\n  {symbol}: Sonuc yok")
            continue

        robust = [r for r in results
                  if r["out_of_sample"]["profit_factor"] > 1.0
                  and r["in_sample"]["profit_factor"] > 1.0]

        best_list = robust[:3] if robust else results[:3]

        for r in best_list:
            p = r["params"]
            is_r = r["in_sample"]
            os_r = r["out_of_sample"]

            print(f"\n{'─'*70}")
            print(f"  {symbol} — #{r['rank']}")
            print(f"{'─'*70}")

            print(f"\n  ┌─────────────────┬──────────────┬──────────────┐")
            print(f"  │ Metrik          │  In-Sample   │  Out-Sample  │")
            print(f"  ├─────────────────┼──────────────┼──────────────┤")
            print(f"  │ Net P&L         │ {is_r['net_pnl']:>+11,.0f} │ {os_r['net_pnl']:>+11,.0f} │")
            print(f"  │ Profit Factor   │ {is_r['profit_factor']:>11.3f} │ {os_r['profit_factor']:>11.3f} │")
            print(f"  │ Win Rate        │ {is_r['win_rate']:>10.1f}% │ {os_r['win_rate']:>10.1f}% │")
            print(f"  │ Max Drawdown    │ {is_r['max_drawdown']:>10.2f}% │ {os_r['max_drawdown']:>10.2f}% │")
            print(f"  │ Total Trades    │ {is_r['total_trades']:>11,} │ {os_r['total_trades']:>11,} │")
            print(f"  └─────────────────┴──────────────┴──────────────┘")

            print(f"\n  DFS Agirliklari:")
            print(f"    Delta={p.get('w_delta',0):.2f}  CVD={p.get('w_cvd',0):.2f}  LogP={p.get('w_logp',0):.2f}  OBI_W={p.get('w_obi_w',0):.2f}")
            print(f"    OBI_D={p.get('w_obi_d',0):.2f}  Sweep={p.get('w_sweep',0):.2f}  Burst={p.get('w_burst',0):.2f}  OI={p.get('w_oi',0):.2f}")

            print(f"\n  TRS:")
            print(f"    Confirm={p.get('trs_confirm_ticks')}  Bull={p.get('trs_bullish_zone')}  Bear={p.get('trs_bearish_zone')}  Agree={p.get('trs_agreement')}")

            print(f"\n  Risk:")
            print(f"    SL={p.get('stop_loss_pct')}%  Trail Act={p.get('trailing_activation_pct')}%  Trail Dist={p.get('trailing_distance_pct')}%")
            print(f"    Exit Hard={p.get('exit_score_hard')}  Soft={p.get('exit_score_soft')}")
            print(f"    Cooldown={p.get('entry_cooldown_sec')}s  Margin={p.get('margin_per_trade')} USDT")


if __name__ == "__main__":
    main()
