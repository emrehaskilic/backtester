# Session Memory — ETH/USDT Daily Pattern Pipeline

## SON DURUM (2026-04-01)

### Tamamlanan İşler:
1. **Look-ahead bias keşfi**: HTF trend mapping'de look-ahead bias vardı. Düzeltilince TÜM edge kayboldu.
2. **1H feature mining (37 feature)**: Hiçbir feature anlamlı edge vermedi.
3. **4H feature mining (38 feature)**: Küçük edge'ler bulundu ama composite sinyalde kayboldu.
4. **test_4h_edge_strategy.py**: Yazıldı, commit+push edildi. Sonuç: OOS %50.33 — edge yok.

### YAPILACAK İŞ — Günlük Pattern Pipeline
Kullanıcı aşağıdaki tam pipeline'ın kodlanmasını istiyor. HİÇ BAŞLANMADI.

**Script adı:** `test_daily_pattern_pipeline.py`

**Plan (10 adım):**

1. **Data**: 5m parquet → daily OHLCV (~1825 gün). Split %60/%20/%20.
2. **Feature Engineering (~25-30 feature):**
   - Candle: body_ratio, close_position, upper_wick, lower_wick
   - Trend: EMA slope 5/20/50, Close-EMA dist 5/20/50
   - Momentum: Return 1/3/5/7 gün, RSI(14)
   - Volatilite: ATR(14)/close, Bollinger width(20)
   - Volume: spike ratio (vol/SMA20), 3-gün trend
   - Multi-day (N optimize): N-gün avg body_ratio, max wick, return, vol trend, up streak
3. **Feature Selection:** Spearman corr → top 10-15, |r|>0.7 çiftlerden birini at, val doğrula
4. **Window Optimization:** N={3,4,5,6,7} dene, val'de en iyi N seç
5. **ML (GBM + RF):** max_depth=3-5, min_samples_leaf≥20, walk-forward 6ay train/1ay predict
6. **İstatistiksel:** Tercile bucket, 2'li kombinasyon, sample<20 filtre, train+val tutarlılık
7. **Ensemble:** ML prob + stat prob + feature agreement → confidence score
8. **Position sizing:** conf>0.60→$625, 0.55-0.60→$312, <0.55→$156. %100 coverage
9. **Backtest:** $1000 kasa, 25x leverage, fee %0.04, slippage %0.02, haftalık kar çekimi
10. **Output:** 4 tablo (günlük bias, pattern kataloğu, performans, feature importance)

**Kurallar:**
- Look-ahead bias YOK (T günü kapanışından hesapla)
- Train sınırları kullan
- Walk-forward overfitting kontrolü
- sklearn + lightgbm kurulu (.venv)

### Branch: `claude/bias-detection-coverage-WinWq`
### Data: `data/ETHUSDT_5m_5y.parquet`

### Kullanıcının Kırmızı Çizgileri:
- %100 coverage ŞART
- Look-ahead bias → ÖLÜM
- "Best practice" bahanesiyle talebi sulandırma
- Yarım yamalak implementasyon yapma
