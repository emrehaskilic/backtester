---
name: sweep_strategy_journey
description: Multi-TF sweep candle analysis - continuation/reversal/ambiguous tanimlari (1D/8H/4H/2H/1H) ve strateji gelistirme sureci
type: project
---

## Proje: ETH Sweep-Based Trading Strategy

### Veri
- 5m ETHUSDT 11 ay (96K bar, native OI dahil)
- buy_vol, sell_vol, open_interest hepsi native 5m
- Tarih: 2025-04-01 - 2026-02-28

### Sweep Continuation / Reversal / Ambiguous Tanimlari (Tum TF'ler icin ayni mantik)

**High Sweep (onceki mumun high'i sweep edildi):**
- Continuation: Mum, prev_high ustunde kapandi → yukari devam etti
- Reversal: Mum, prev_high altinda kapandi + kirmizi kapanis → geri dondu
- Ambiguous: Mum, prev_high altinda kapandi ama yesil kapanis → belirsiz

**Low Sweep (onceki mumun low'u sweep edildi):**
- Continuation: Mum, prev_low altinda kapandi → asagi devam etti
- Reversal: Mum, prev_low ustunde kapandi + yesil kapanis → geri dondu
- Ambiguous: Mum, prev_low ustunde kapandi ama kirmizi kapanis → belirsiz

**Inside bar = pas**

### Multi-TF Ozet Tablo
| TF  | Sweep olusma suresi | Cont/Rev karari | Sonraki bias suresi |
|-----|---------------------|-----------------|---------------------|
| 1D  | Gun icinde          | Gunluk kapanista | 1-3 gun            |
| 8H  | 8 saat icinde       | 8H kapanista     | 8-24 saat          |
| 4H  | 4 saat icinde       | 4H kapanista     | 4-8 saat           |
| 2H  | 2 saat icinde       | 2H kapanista     | 2-4 saat           |
| 1H  | 1 saat icinde       | 1H kapanista     | 1-2 saat           |

Kucuk TF = sik sinyal, kisa omur. Buyuk TF = nadir sinyal, uzun omur.
Hepsinde ayni kural: sweep yonunde kapanis = continuation, ters kapanis = reversal.

### En Guclu Pattern'ler (8/8 Walk-Forward, FDR-corrected)

**LONG (High Continuation):**
- CVD_micro Q5 + Imbalance Q5 -> %97.8 (N=270, WF 8/8)
- CVD_micro Q5 + Vol_micro Q5 -> %98.4 (N=243, WF 8/8)
- CVD_micro Q4 + Imbalance Q4 -> %92.2 (N=153, WF 8/8)

**SHORT (Low Continuation):**
- CVD_micro Q1 + Vol_micro Q5 -> %99.6 (N=268, WF 8/8)
- CVD_micro Q1 + Imbalance Q1 -> %97.5 (N=239, WF 8/8)

**LONG (Low Reversal):**
- Vol_micro Q1 + Imbalance Q5 -> %88.0 (N=142, WF 8/8)

**SHORT (High Reversal):**
- CVD_micro Q1 + Imbalance Q1 -> %74.3 (N=249, WF 8/8)

### Feature'lar (7 adet, saf order flow)
1. CVD z-score micro (12-bar)
2. CVD z-score macro (288-bar)
3. OI change (288-bar, Daily ATR normalized)
4. Volume z-score micro (12-bar)
5. Volume z-score macro (288-bar)
6. Imbalance smoothed (EMA_12)
7. ATR percentile (288-bar rank)

### Dominant Feature
CVD_micro en guclu ayirici:
- High sweep: Q1 %36 -> Q5 %95 continuation (monoton)
- Low sweep: Q1 %95 -> Q5 %33 continuation (monoton)

### Strateji Denemeleri
1. Saf sweep (ters sinyal cikis): Net -31%, WR 41% — fee yiyor
2. Sweep + KC DCA/TP: Net -82% — DCA kayiplari katliyor
3. Multi-TF weighted voting: Net -19% — choppy piyasada yanlis yon

**Why:** Yon dogru bulunuyor ama pozisyon yonetimi (cikis, risk) cozulmedi. Giris sinyali guclu, cikis sinyali zayif.

**How to apply:** Giris icin sweep + candle + feature pattern kullan, cikis icin farkli mekanizma gerekli (KC TP, trailing stop, veya zaman bazli).

### Teknoloji
- Rust engine: backtest + pattern miner + strategy hepsi Rust'ta
- Rayon parallelism
- Rust-native TPE optimizer (Optuna'dan 30x hizli)
- TradingView Pine indicator yazildi

### Dosyalar
- rust_engine/src/sweep_candle_analysis.rs — mum bazli analiz
- rust_engine/src/sweep_candle_miner.rs — quantile pattern mining
- rust_engine/src/sweep_candle_strategy.rs — saf strateji
- rust_engine/src/sweep_candle_strategy_kc.rs — KC DCA/TP stratejisi
- Desktop/1h_sweep_indicator.pine — TradingView indikatoru
