# Swinginess Tick Replay Backtester & Optimizer

Swinginess trading stratejisi için tick bazlı backtest ve Optuna parametre optimizasyonu.

## Kurulum (Windows Server / Contabo VPS)

```powershell
# Python 3.11+ gerekli
pip install -r requirements.txt

# aggTrades verisi indir (6 ay, ~10-30GB)
python download_aggtrades.py

# Hızlı test (default parametreler)
python run_backtest.py

# Parametre optimizasyonu başlat (500 trial/pair)
python optimize.py
```

## Dosya Yapısı

```
backtester/
├── download_aggtrades.py   # Binance'ten tick verisi indir
├── tick_engine.py          # NT8 benzeri Tick Replay Engine
├── swinginess_strategy.py  # Strateji mantığı (TRS + DFS)
├── run_backtest.py         # Tek backtest çalıştır
├── optimize.py             # Optuna optimizasyon (4 pair bağımsız)
├── report.py               # Sonuç raporu
├── requirements.txt        # Python bağımlılıkları
└── data/                   # İndirilen aggTrades verisi (gitignore)
```

## Optimize Edilen Parametreler (~30)

- **DFS Ağırlıkları**: 8 bileşen (zDelta, zCvd, zLogP, zObi, sweep, burst, oi)
- **TRS Parametreleri**: confirm_ticks, bullish/bearish zone, agreement
- **Çıkış**: trailing activation/distance, exit score thresholds, SL
- **Filtreler**: chop score, prints/sec, entry cooldown, rolling window

## Çalışma Prensibi

1. Binance aggTrades (tick verisi) indirilir
2. Her tick sırayla işlenir (NT8 Tick Replay benzeri)
3. 1-saniyelik bucket'larda DFS bileşenleri hesaplanır
4. TRS reversal sinyalleri üretilir
5. Giriş/çıkış kararları verilir
6. Optuna en kârlı parametre setini bulur

## Pair'ler

- BTCUSDT
- ETHUSDT
- SOLUSDT
- XRPUSDT

Her pair için bağımsız optimizasyon yapılır.
