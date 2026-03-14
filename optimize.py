"""
Swinginess Optuna Optimizasyon — 4 pair bağımsız.
Windows Server (Contabo VPS) uyumlu.

Kullanım: python optimize.py
"""

import optuna
import pandas as pd
import numpy as np
import json
import os
import sys
from tick_engine import TickReplayEngine
from swinginess_strategy import SwingingessStrategy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
N_TRIALS = 500


def load_data(symbol):
    """Veriyi yükle ve %70/%30 böl."""
    path = os.path.join(DATA_DIR, f"{symbol.lower()}_aggtrades.parquet")
    df = pd.read_parquet(path)
    split = int(len(df) * 0.7)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def run_trial(df, params):
    """Tek bir parametre seti ile backtest çalıştır."""
    rolling_window = params.get("rolling_window_sec", 3600)
    engine = TickReplayEngine(rolling_window_sec=rolling_window)
    strategy = SwingingessStrategy(params)

    ts_arr = df["timestamp"].values
    price_arr = df["price"].values
    qty_arr = df["quantity"].values
    side_arr = df["side"].values

    last_bucket_ts = 0

    for i in range(len(df)):
        ts_ms = int(ts_arr[i])
        price = float(price_arr[i])
        qty = float(qty_arr[i])
        side = str(side_arr[i])

        engine.process_tick(ts_ms, price, qty, side)

        ts_sec = ts_ms // 1000
        if ts_sec == last_bucket_ts:
            continue
        last_bucket_ts = ts_sec

        if not engine.warmup_done:
            continue

        strategy.on_second(ts_sec, price, engine)

        if strategy.equity <= 0:
            break

    if strategy.position != 0:
        strategy._close_position(float(price_arr[-1]), int(ts_arr[-1]) // 1000, "CLOSE")

    return strategy.get_results()


def objective(trial, df):
    """Optuna objective fonksiyonu."""
    params = {
        # DFS ağırlıkları
        "w_delta": trial.suggest_float("w_delta", 0.05, 0.40, step=0.05),
        "w_cvd": trial.suggest_float("w_cvd", 0.05, 0.35, step=0.05),
        "w_logp": trial.suggest_float("w_logp", 0.05, 0.25, step=0.05),
        "w_obi_w": trial.suggest_float("w_obi_w", 0.05, 0.25, step=0.05),
        "w_obi_d": trial.suggest_float("w_obi_d", 0.05, 0.25, step=0.05),
        "w_sweep": trial.suggest_float("w_sweep", 0.02, 0.15, step=0.02),
        "w_burst": trial.suggest_float("w_burst", 0.02, 0.15, step=0.02),
        "w_oi": trial.suggest_float("w_oi", 0.02, 0.15, step=0.02),

        # TRS parametreleri
        "trs_confirm_ticks": trial.suggest_int("trs_confirm_ticks", 30, 180, step=10),
        "trs_bullish_zone": trial.suggest_float("trs_bullish_zone", 0.55, 0.85, step=0.05),
        "trs_bearish_zone": trial.suggest_float("trs_bearish_zone", 0.15, 0.45, step=0.05),
        "trs_agreement": trial.suggest_float("trs_agreement", 0.25, 0.60, step=0.05),

        # Risk / çıkış
        "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.5, 3.0, step=0.25),
        "trailing_activation_pct": trial.suggest_float("trailing_activation_pct", 0.3, 2.0, step=0.1),
        "trailing_distance_pct": trial.suggest_float("trailing_distance_pct", 0.2, 1.5, step=0.1),
        "exit_score_hard": trial.suggest_float("exit_score_hard", 0.70, 0.95, step=0.05),
        "exit_score_soft": trial.suggest_float("exit_score_soft", 0.50, 0.85, step=0.05),

        # Filtreler
        "min_prints_per_sec": trial.suggest_float("min_prints_per_sec", 0.5, 3.0, step=0.5),
        "entry_cooldown_sec": trial.suggest_int("entry_cooldown_sec", 60, 600, step=60),
        "rolling_window_sec": trial.suggest_int("rolling_window_sec", 1800, 7200, step=600),
        "time_flat_sec": trial.suggest_int("time_flat_sec", 3600, 28800, step=3600),

        # Sabit
        "leverage": 50,
        "margin_per_trade": trial.suggest_float("margin_per_trade", 10, 100, step=10),
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
    }

    # Exit soft < hard
    if params["exit_score_soft"] >= params["exit_score_hard"]:
        return -999

    result = run_trial(df, params)

    if result["total_trades"] < 10:
        return -999

    pf = min(result["profit_factor"], 10)
    wr = result["win_rate"]
    dd = result["max_drawdown"]
    net_pct = result["net_pnl_pct"]

    score = net_pct + pf * 5 - dd * 0.1

    # Kaydet
    trial.set_user_attr("net_pnl", result["net_pnl"])
    trial.set_user_attr("profit_factor", result["profit_factor"])
    trial.set_user_attr("win_rate", result["win_rate"])
    trial.set_user_attr("max_drawdown", result["max_drawdown"])
    trial.set_user_attr("total_trades", result["total_trades"])
    trial.set_user_attr("avg_hold_sec", result["avg_hold_sec"])

    return score


def optimize_pair(symbol):
    print(f"\n{'='*60}")
    print(f"  {symbol} OPTİMİZASYONU ({N_TRIALS} trial)")
    print(f"{'='*60}")

    in_sample, out_sample = load_data(symbol)
    print(f"  In-sample:  {len(in_sample):,} tick")
    print(f"  Out-sample: {len(out_sample):,} tick")

    # Optuna study (SQLite DB ile kaydet — crash recovery)
    db_path = os.path.join(RESULTS_DIR, f"{symbol.lower()}_study.db")
    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{db_path}",
        study_name=f"{symbol}_optimize",
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, in_sample), n_trials=N_TRIALS, show_progress_bar=True)

    # En iyi 5
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value and t.value > -999 else -9999, reverse=True)[:5]

    results = []
    for rank, trial in enumerate(top_trials, 1):
        if trial.value is None or trial.value <= -999:
            continue

        params = trial.params.copy()
        params["leverage"] = 50
        params["maker_fee"] = 0.0002
        params["taker_fee"] = 0.0005

        # Out-of-sample doğrulama
        oos = run_trial(out_sample, params)

        entry = {
            "rank": rank,
            "params": params,
            "in_sample": {k: trial.user_attrs.get(k, 0) for k in
                         ["net_pnl", "profit_factor", "win_rate", "max_drawdown", "total_trades", "avg_hold_sec"]},
            "out_of_sample": oos,
        }
        results.append(entry)

        print(f"\n  #{rank} {symbol}")
        print(f"    IS:  PF={entry['in_sample']['profit_factor']:.2f}  WR={entry['in_sample']['win_rate']:.1f}%  DD={entry['in_sample']['max_drawdown']:.1f}%  PnL={entry['in_sample']['net_pnl']:+,.0f}")
        print(f"    OOS: PF={oos['profit_factor']:.2f}  WR={oos['win_rate']:.1f}%  DD={oos['max_drawdown']:.1f}%  PnL={oos['net_pnl']:+,.0f}")

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  SWINGINESS TICK REPLAY OPTİMİZASYON")
    print(f"  {len(SYMBOLS)} pair | {N_TRIALS} trial/pair | Optuna Bayesian")
    print("=" * 60)

    all_results = {}

    for symbol in SYMBOLS:
        try:
            results = optimize_pair(symbol)
            all_results[symbol] = results
        except Exception as e:
            print(f"\n  HATA {symbol}: {e}")
            import traceback
            traceback.print_exc()
            all_results[symbol] = []

    # Sonuçları kaydet
    results_path = os.path.join(RESULTS_DIR, "optimization_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Özet rapor
    print(f"\n\n{'='*60}")
    print("  ÖZET — EN İYİ ROBUST SONUÇLAR")
    print(f"{'='*60}")

    for symbol in SYMBOLS:
        if not all_results.get(symbol):
            print(f"\n  {symbol}: Sonuç bulunamadı")
            continue

        robust = [r for r in all_results[symbol]
                  if r["out_of_sample"]["profit_factor"] > 1.0
                  and r["in_sample"]["profit_factor"] > 1.0]

        if robust:
            best = robust[0]
            p = best["params"]
            print(f"\n  {symbol}:")
            print(f"    TRS: confirm={p.get('trs_confirm_ticks')}  bull={p.get('trs_bullish_zone')}  bear={p.get('trs_bearish_zone')}")
            print(f"    SL: {p.get('stop_loss_pct')}%  Trail: {p.get('trailing_activation_pct')}%/{p.get('trailing_distance_pct')}%")
            print(f"    Exit Score: hard={p.get('exit_score_hard')}  soft={p.get('exit_score_soft')}")
            print(f"    IS:  PnL={best['in_sample']['net_pnl']:+,.0f}  PF={best['in_sample']['profit_factor']:.2f}")
            print(f"    OOS: PnL={best['out_of_sample']['net_pnl']:+,.0f}  PF={best['out_of_sample']['profit_factor']:.2f}")
        else:
            print(f"\n  {symbol}: Robust sonuç bulunamadı")

    print(f"\nSonuçlar: {results_path}")


if __name__ == "__main__":
    # Tek pair optimize etmek için: python optimize.py BTCUSDT
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results = optimize_pair(symbol)
        results_path = os.path.join(RESULTS_DIR, f"{symbol.lower()}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSonuçlar: {results_path}")
    else:
        main()
