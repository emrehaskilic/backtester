---
name: next_strategy_to_implement
description: Sonraki session'da yazilacak strateji - 1H sweep yon + 3m KC islem. Rust'ta backtest edilecek.
type: project
---

## Strateji: 1H Sweep Yon + 3m KC Islem

### Kasa ve Risk
- Kasa: 1,000 USDT (her hafta basinда sifirlanir, yeni hafta = yeni 1000 USDT)
- Margin: Kasanin %1'i = 10 USDT per entry
- Leverage: 25x
- Haftalik PnL ayri ayri raporlanir (haftalik sifirlama = compound yok)

### Yon Belirleme (1H mum kapanisinda)
- Continuation pattern match -> o yonde ol
- Reversal pattern match -> ters yonde ol
- Belirsiz / inside bar -> mevcut yonu koru

### Continuation/Reversal Tanimi (kullanicinin tanimladi)
- High sweep + close > prev_high = Continuation (LONG)
- High sweep + close < prev_high + kirmizi mum = Reversal (SHORT)
- High sweep + close < prev_high + yesil mum = Belirsiz (pas)
- Low sweep + close < prev_low = Continuation (SHORT)
- Low sweep + close > prev_low + yesil mum = Reversal (LONG)
- Low sweep + close > prev_low + kirmizi mum = Belirsiz (pas)
- Inside bar = pas

### Feature Pattern'ler (miner'dan, 8/8 walk-forward)
**LONG patterns:**
- CVD_micro Q5 + Imbalance Q5 (97.8%, N=270)
- CVD_micro Q5 + Vol_micro Q5 (98.4%, N=243)
- CVD_micro Q4 + Imbalance Q4 (92.2%, N=153)
- Vol_micro Q1 + Imbalance Q5 (88.0%, N=142, low reversal)

**SHORT patterns:**
- CVD_micro Q1 + Vol_micro Q5 (99.6%, N=268)
- CVD_micro Q1 + Imbalance Q1 (97.5%, N=239)
- CVD_micro Q2 + Vol_micro Q4 (90.8%, N=163)
- Vol_micro Q1 + Imbalance Q1 (75.2%, N=249, high reversal)

### Islem (3m grafik, KC bazli)
- KC standart: length=20, mult=2.0, atr_period=14
- LONG yondeyken: KC alt band -> AL (DCA), KC ust band -> SAT (TP)
- SHORT yondeyken: KC ust band -> SAT (DCA), KC alt band -> AL (TP)
- Tek kademe, basic DCA ve TP
- Her DCA = kasa %1 margin
- Her TP = tum pozisyonu kapat

### Cikis
- 1H'de reversal sinyal gelince -> tum pozisyon kapat + ters yone gec

### Features (7 adet, 5m native data uzerinde)
1. CVD z-score micro (12-bar)
2. CVD z-score macro (288-bar)
3. OI change (288-bar, Daily ATR normalized)
4. Volume z-score micro (12-bar)
5. Volume z-score macro (288-bar)
6. Imbalance smoothed (EMA_12)
7. ATR percentile (288-bar rank)

### Quantile
- Her feature 5 bucket (Q1-Q5)
- Rolling monthly update (720 bar)

### Veri
- 5m ETHUSDT: data/ETHUSDT_5m_cvd_oi_11mo.parquet (native OI dahil)
- 3m ETHUSDT: data/ETHUSDT_3m_vol_11mo.parquet (buy_vol, sell_vol)

### Teknoloji
- Rust engine: rust_engine/src/ (tum moduller mevcut)
- Feature hesaplama: sweep_miner.rs -> compute_5m_features()
- KC hesaplama: sweep_candle_strategy_kc.rs icinde mevcut
- PyO3 binding: lib.rs

### Onceki Denemeler (basarisiz)
- Saf sweep ters sinyal cikis: -31% (fee yiyor, WR 41%)
- Sweep + KC DCA/TP: -82% (DCA kayiplari katliyor)
- Sorun: giris dogru ama cikis/pozisyon yonetimi yanlis

### Bu Sefer Fark
- 1H yon + 3m islem (multi-timeframe)
- Basit DCA/TP (kademesiz)
- KC standart (optimize edilmemis)
- Reversal = cikis + ters giris (surekli islem)
