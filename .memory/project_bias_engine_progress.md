---
name: bias_engine_progress
description: Continuous Bias Engine implementasyon ilerlemesi - tum sessionlar, sonuclar, optimal parametreler, sonraki adim
type: project
---

## Continuous Bias Engine — Implementation Progress

### Session 1-2 (2026-03-26): Temel Altyapi
- 7 feature computation, quantization, state mining, probability, robustness
- 62 validated state, 5m 11-ay veri uzerinde

### Session 3 (2026-03-26): Tam Pipeline
- Fallback, calibration, regime, sweep, final_bias, decay, monitoring, walkforward
- In-sample %53.7, OOS %51.0

### Session 4 (2026-03-26): OOS Iyilestirmesi
- FDR 0.01→0.05, baseline-relative significance, calibration guard
- OOS %51.0, WF-validated 58 state, accuracy std 0.008

### Session 5 (2026-03-26): 1H + Mean Reversion + Strateji
- 5m→1H gecis (43K bar, 5 yil veri)
- Mean reversion (EMA counter-trend) kesfedildi: %52.5 tek basina
- Combined scoring: bias engine + MR + RSI + CVD + agreement
- Hybrid system: %53.4 accuracy, %100 coverage

### Session 6 (2026-03-26~27): TPE Optimizasyon
- 38 parametre tanimlandi (Group A: 18 bias engine, Group B: 20 scoring)
- Rust TPE optimizer yazildi (2-phase nested: outer+inner)
- 2000 trial, 10.9 saat
- **Final OOS Score: 0.5385 (%53.85 accuracy, walk-forward validated)**
- 119 validated state

### OPTIMAL PARAMETRELER (2000 trial TPE sonucu)

**Group A:**
| Parametre | Deger |
|-----------|-------|
| cvd_micro_window | 26 |
| cvd_macro_window | 96 |
| vol_micro_window | 24 |
| vol_macro_window | 240 |
| imbalance_ema_span | 10 |
| atr_pct_window | 528 |
| oi_change_window | 480 |
| quant_window | 1000 |
| quantile_count | 7 |
| k_horizon | 2 |
| min_sample_size | 180 |
| min_edge | 0.045 |
| prior_strength | 15 |
| fdr_alpha | 0.05 |
| temporal_min_segments | 2 |
| temporal_max_reversals | 0 |
| min_noise_stability | 0.55 |
| ensemble_min_n | 90 |

**Group B:**
| Parametre | Deger |
|-----------|-------|
| mr_ema_span1 | 32 |
| mr_ema_span2 | 56 |
| rsi_period | 18 |
| rsi_threshold | 15 |
| w_bias | 0.0 |
| w_mr1 | 0.9 |
| w_mr2 | 0.6 |
| w_rsi | 1.0 |
| w_agree | 1.4 |
| w_cvd | 1.0 |
| bias_override_threshold | 0.04 |
| bias_override_mult | 3.0 |
| sweep_scale | 0.25 |
| sweep_aligned_weight | 0.50 |
| sweep_conflict_mult | 0.60 |
| regime_dir_lookback | 264 |
| trending_threshold | 1.75 |
| high_vol_threshold | 0.95 |
| regime_shift_lookback | 48 |
| regime_shift_penalty | 0.80 |

### KRITIK BULGULAR
1. **K=2 optimal** — 2 saat lookahead, 12 degil
2. **quantile_count=7** — 5 degil, daha ince ayrim
3. **w_bias=0.0** — bias engine direkt katki SIFIR
4. **w_agree=1.4** — ama agreement bonus EN GUCLU faktor
5. **MR (0.9) + RSI (1.0) + CVD (1.0)** — ana suruculer
6. **Agreement interaction** — bias+MR ayni yonu gosterdiginde en guclu sinyal
7. **min_edge=0.045** — sert edge filtresi, az ama kaliteli state
8. **quant_window=1000** — hizli adaptasyon (2016'dan kisa)

### DOSYALAR (Session 5-6 yenileri)
- `rust_engine/src/bias_engine/params.rs` — 38 parametre struct
- `rust_engine/src/bias_engine/scoring.rs` — MR + RSI + CVD combined scoring
- `rust_engine/src/bias_engine/optimizer.rs` — TPE 2-phase nested optimizer
- `rust_engine/src/bias_kc_strategy.rs` — Bias + KC strateji (denendi, -133 USDT)
- `test_bias_1h_5y.py` — 1H 5Y analiz
- `test_bias_1h_carry.py` — carry-forward testi
- `test_bias_1h_trend.py` — trend indicator testi
- `test_bias_1h_meanrev.py` — mean reversion testi
- `test_bias_1h_boost.py` — accuracy boosting
- `test_bias_1h_learned.py` — logistic regression
- `test_bias_1h_optimize.py` — Phase 1 Python optimizer
- `test_bias_optimize2.py` — Phase 2 Rust optimizer

### SONRAKI SESSION: Yeni Data Kaynaklari ile %80+ Hedefi
Kullanici her an %80+ accuracy istiyor. Mevcut OHLCV+OI ile limit %53.85.
Denenecek yeni data kaynaklari:
1. **Funding rate** — long/short kalabaligi (Binance API)
2. **Liquidation data** — cascade tetikleyicisi (Binance API)
3. **Order book depth** — bid/ask wall'lari (Binance API)
4. **BTC korelasyon** — BTC hareket edince ETH ne yapiyor

**Why:** %53.85 fee sonrasi karli ama kullanici %80 istiyor. Mevcut feature set'in limiti burasi — yeni bilgi kaynaklari lazim.
**How to apply:** Kullanici "yeni data kaynaklariyla devam et" dediginde bu memory'den devam et.

### Spec Dosyasi
`Desktop/CONTINUOUS_BIAS_ENGINE_PROMPT.md` — tam spesifikasyon (21 section)
