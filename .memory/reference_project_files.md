---
name: reference_project_files
description: Bactester projesi dosya yapisi ve onemli modullerin konumlari
type: reference
---

Proje: C:\Users\emrehaskilic\Desktop\bactester

**Rust Engine:** rust_engine/src/
- backtest.rs — PMax backtests (look-ahead fix'li)
- kama.rs — KAMA + combined backtest
- adaptive_pmax.rs — Adaptive PMax
- cvd_oi.rs — CVD + OI
- optimizer.rs — Rust-native TPE + Rayon
- pmax_pure.rs — Saf PMax optimizer
- pattern_scanner.rs — Ilk pattern tarayici
- pattern_miner.rs — Quantile-based pattern mining
- pattern_miner_v2.rs — 7-task comprehensive analysis
- sweep_miner.rs — Multi-TF sweep mining (weekly/daily/8h/4h/1h/15m)
- sweep_candle_analysis.rs — Mum kapanisi bazli analiz
- sweep_candle_miner.rs — Candle pattern quantile mining
- sweep_candle_strategy.rs — Saf candle strateji
- sweep_candle_strategy_kc.rs — KC DCA/TP stratejisi
- sweep_strategy.rs — Multi-TF weighted voting
- sweep_strategy_1h.rs — 1H triple barrier
- sweep_strategy_1h_v2.rs — 1H ters sinyal cikis

**Data:** data/
- ETHUSDT_5m_cvd_oi_11mo.parquet — native 5m (OI dahil)
- ETHUSDT_3m_vol_11mo.parquet — 3m buy/sell vol
- ETHUSDT_OI_5m_11mo.parquet — 5m OI

**Sonuclar:** results/
- sweep_miner_results.json
- comprehensive_analysis.json
- combined_wf_rust_results.json

**TradingView:** Desktop/1h_sweep_indicator.pine
**PDF Raporlar:** Desktop/weekly_daily_8h_4h_1h_sweep.pdf
