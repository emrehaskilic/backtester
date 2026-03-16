# Swinginess Backtester — UX Tasarim Dokumani

## 1. Genel Bakis

**Urun:** Swinginess Tick Replay Backtester & Optimizer Web UI
**Kullanici:** Solo trader (emrehaskilic)
**Platform:** React (Vite + TypeScript) frontend + FastAPI backend
**Tema:** Dark theme (trading platform hissi)
**Grafik:** Recharts + Lightweight Charts (TradingView)

---

## 2. Bilgi Mimarisi

```
App
├── Sidebar (sabit, sol taraf)
│   ├── Logo / Baslik
│   ├── Dashboard (/)
│   ├── Backtest (/backtest)
│   ├── Optimizasyon (/optimization)
│   ├── Veri Yonetimi (/data)
│   └── Ayarlar (/settings)
│
├── Header (ust bar)
│   ├── Sayfa basligi
│   ├── Aktif pair badge
│   └── Durum indikatoru (backend bagli/bagli degil)
│
└── Main Content (sag taraf, sayfa icerigi)
```

---

## 3. Sayfa Tasarimlari

### 3.1 Dashboard (Ana Sayfa)

**Amac:** Tek bakista genel durum.

```
┌─────────┬──────────────────────────────────────────────┐
│         │  DASHBOARD                                   │
│  LOGO   │                                              │
│         │  ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  -----  │  │ BTCUSDT  │ │ ETHUSDT  │ │ SOLUSDT  │ ... │
│ Dashbrd │  │ PF: 1.85 │ │ PF: 2.10 │ │ PF: 0.92 │     │
│ Backtst │  │ WR: 58%  │ │ WR: 62%  │ │ WR: 45%  │     │
│ Optimiz │  │ PnL:+420 │ │ PnL:+680 │ │ PnL:-120 │     │
│ Veri    │  │ DD: 8.5% │ │ DD: 6.2% │ │ DD: 15%  │     │
│ Ayarlar │  └──────────┘ └──────────┘ └──────────┘      │
│         │                                              │
│         │  ┌──────────────────────────────────────┐    │
│         │  │        EQUITY CURVE (en iyi pair)     │    │
│         │  │        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~   │    │
│         │  │   $1200 ┐                      /      │    │
│         │  │         │              ___/---/        │    │
│         │  │   $1000 │____/----\___/                │    │
│         │  │         └────────────────────────      │    │
│         │  └──────────────────────────────────────┘    │
│         │                                              │
│         │  ┌─────────────────┐ ┌──────────────────┐    │
│         │  │ SON ISLEMLER    │ │ VERI DURUMU       │    │
│         │  │ BTC LONG +$12  │ │ BTC: 180M tick OK │    │
│         │  │ ETH SHORT -$5  │ │ ETH: 150M tick OK │    │
│         │  │ SOL LONG +$8   │ │ SOL: Veri yok     │    │
│         │  └─────────────────┘ │ XRP: 90M tick OK  │    │
│         │                      └──────────────────┘    │
└─────────┴──────────────────────────────────────────────┘
```

**Bilesenler:**
- **Pair Kartlari:** Her pair icin ozet metrikler (son backtest veya optimizasyon sonucu). Kart rengi: yesil (karli), kirmizi (zararda).
- **Equity Curve:** En son calisan backtest'in equity grafigi (Recharts AreaChart).
- **Son Islemler:** Son 10 trade listesi (pair, yon, pnl, exit type).
- **Veri Durumu:** Her pair icin tick verisi durumu (mevcut/eksik, boyut).

---

### 3.2 Backtest Sayfasi

**Amac:** Parametre ayarla, backtest calistir, sonuclari gor.

```
┌─────────┬──────────────────────────────────────────────┐
│         │  BACKTEST                                    │
│ SIDEBAR │                                              │
│         │  ┌─ PARAMETRELER ─────────────────────────┐  │
│         │  │ Pair: [BTCUSDT ▼]                      │  │
│         │  │                                        │  │
│         │  │ -- DFS Agirliklari --                   │  │
│         │  │ Delta  [====|====] 0.22                 │  │
│         │  │ CVD    [====|====] 0.18                 │  │
│         │  │ LogP   [===|=====] 0.12                 │  │
│         │  │ OBI_W  [===|=====] 0.14                 │  │
│         │  │ OBI_D  [===|=====] 0.12                 │  │
│         │  │ Sweep  [==|======] 0.08                 │  │
│         │  │ Burst  [==|======] 0.08                 │  │
│         │  │ OI     [=|=======] 0.06                 │  │
│         │  │                                        │  │
│         │  │ -- TRS --                               │  │
│         │  │ Confirm Ticks  [  90  ]                 │  │
│         │  │ Bullish Zone   [ 0.65 ]                 │  │
│         │  │ Bearish Zone   [ 0.35 ]                 │  │
│         │  │ Agreement      [ 0.40 ]                 │  │
│         │  │                                        │  │
│         │  │ -- Risk / Cikis --                      │  │
│         │  │ Stop Loss %    [ 1.5  ]                 │  │
│         │  │ Trail Act %    [ 0.8  ]                 │  │
│         │  │ Trail Dist %   [ 0.5  ]                 │  │
│         │  │ Exit Hard      [ 0.85 ]                 │  │
│         │  │ Exit Soft      [ 0.70 ]                 │  │
│         │  │                                        │  │
│         │  │ -- Filtreler --                          │  │
│         │  │ Min Prints/s   [ 1.0  ]                 │  │
│         │  │ Cooldown (s)   [ 300  ]                 │  │
│         │  │ Rolling Win(s) [ 3600 ]                 │  │
│         │  │ Time Flat (s)  [14400 ]                 │  │
│         │  │ Margin (USDT)  [ 100  ]                 │  │
│         │  │                                        │  │
│         │  │ [  BACKTEST CALISTIR  ]  [RESET]        │  │
│         │  └────────────────────────────────────────┘  │
│         │                                              │
│         │  ┌─ SONUCLAR ────────────────────────────┐   │
│         │  │                                       │   │
│         │  │  Net PnL    Profit Factor   Win Rate  │   │
│         │  │  +420.50    1.854           58.3%     │   │
│         │  │                                       │   │
│         │  │  Max DD     Total Trades    Avg Hold  │   │
│         │  │  8.52%      142             12.5m     │   │
│         │  │                                       │   │
│         │  │  [SL: 35] [TRAIL: 42] [EXIT: 28]     │   │
│         │  │  [TRS_REV: 18] [TIME_FLAT: 12]       │   │
│         │  │  [BLEED: 7]                          │   │
│         │  └───────────────────────────────────────┘   │
│         │                                              │
│         │  ┌─ EQUITY CURVE ────────────────────────┐   │
│         │  │   (Recharts AreaChart - interaktif)    │   │
│         │  │   Hover ile deger goster              │   │
│         │  │   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~        │   │
│         │  └───────────────────────────────────────┘   │
│         │                                              │
│         │  ┌─ TRADE LISTESI ───────────────────────┐   │
│         │  │ #  | Yon   | PnL     | Hold  | Cikis  │   │
│         │  │ 1  | LONG  | +12.50  | 8m    | TRAIL  │   │
│         │  │ 2  | SHORT | -5.20   | 3m    | SL     │   │
│         │  │ 3  | LONG  | +18.00  | 22m   | EXIT   │   │
│         │  │ ...                                   │   │
│         │  └───────────────────────────────────────┘   │
└─────────┴──────────────────────────────────────────────┘
```

**Bilesenler:**
- **Parametre Paneli:** Accordion gruplari (DFS, TRS, Risk, Filtreler). Slider + input combo.
- **Calistir Butonu:** Tiklayinca loading spinner, progress bar (gun bazli).
- **Sonuc Kartlari:** 6 buyuk metrik karti (renkli — yesil/kirmizi).
- **Exit Types:** Chip/badge seklinde dagilim.
- **Equity Curve:** Recharts AreaChart, hover tooltip, gradient fill.
- **Trade Listesi:** Sortable tablo, renk kodlu PnL, sayfalama.

---

### 3.3 Optimizasyon Sayfasi

**Amac:** Optuna sonuclarini gor, karsilastir.

```
┌─────────┬──────────────────────────────────────────────┐
│         │  OPTIMIZASYON SONUCLARI                      │
│ SIDEBAR │                                              │
│         │  Pair: [BTCUSDT ▼]   [OPTIMIZASYON BASLAT]  │
│         │                                              │
│         │  ┌─ EN IYI 5 SONUC ──────────────────────┐   │
│         │  │ #  │ Score │ PF    │ WR    │ DD    │PnL│   │
│         │  │ 1  │ 48.2  │ 2.15  │ 62%   │ 5.2% │+820│  │
│         │  │ 2  │ 45.1  │ 1.98  │ 58%   │ 6.8% │+650│  │
│         │  │ 3  │ 42.8  │ 1.85  │ 61%   │ 7.1% │+580│  │
│         │  │ [Detay]                               │   │
│         │  └───────────────────────────────────────┘   │
│         │                                              │
│         │  ┌─ IN-SAMPLE vs OUT-OF-SAMPLE ──────────┐   │
│         │  │                                       │   │
│         │  │  Metrik      │ In-Sample │ Out-Sample │   │
│         │  │  Net PnL     │   +820    │   +340     │   │
│         │  │  PF          │   2.15    │   1.62     │   │
│         │  │  Win Rate    │   62%     │   55%      │   │
│         │  │  Max DD      │   5.2%    │   8.1%     │   │
│         │  │  Trades      │   186     │   72       │   │
│         │  └───────────────────────────────────────┘   │
│         │                                              │
│         │  ┌─ PARAMETRE DETAY ─────────────────────┐   │
│         │  │ Secili trial'in tum parametreleri      │   │
│         │  │ [BU PARAMETRELERLE BACKTEST CALISTIR]  │   │
│         │  └───────────────────────────────────────┘   │
│         │                                              │
│         │  ┌─ TRIAL HISTORY ───────────────────────┐   │
│         │  │ (Scatter plot: trial # vs score)       │   │
│         │  │ Hover ile parametre detayi             │   │
│         │  └───────────────────────────────────────┘   │
└─────────┴──────────────────────────────────────────────┘
```

---

### 3.4 Veri Yonetimi

**Amac:** aggTrades verisi durumu ve indirme kontrolu.

```
┌─────────┬──────────────────────────────────────────────┐
│         │  VERI YONETIMI                               │
│ SIDEBAR │                                              │
│         │  ┌──────────────────────────────────────────┐│
│         │  │ BTCUSDT                                  ││
│         │  │ Durum: ✅ Mevcut                         ││
│         │  │ Boyut: 2.4 GB (parquet)                  ││
│         │  │ Tick Sayisi: 180,452,000                 ││
│         │  │ Tarih Araligi: 2025-09 → 2026-02        ││
│         │  │ [GUNCELLE]                               ││
│         │  ├──────────────────────────────────────────┤│
│         │  │ ETHUSDT                                  ││
│         │  │ Durum: ✅ Mevcut                         ││
│         │  │ ...                                      ││
│         │  ├──────────────────────────────────────────┤│
│         │  │ SOLUSDT                                  ││
│         │  │ Durum: ❌ Veri Yok                       ││
│         │  │ [INDIR]                                  ││
│         │  ├──────────────────────────────────────────┤│
│         │  │ XRPUSDT                                  ││
│         │  │ Durum: ✅ Mevcut                         ││
│         │  │ ...                                      ││
│         │  └──────────────────────────────────────────┘│
│         │                                              │
│         │  [TUMU INDIR / GUNCELLE]                     │
│         │                                              │
│         │  ┌─ INDIRME DURUMU ──────────────────────┐   │
│         │  │ SOLUSDT: ████████░░░░░ 65% (4/6 ay)   │   │
│         │  └───────────────────────────────────────┘   │
└─────────┴──────────────────────────────────────────────┘
```

---

## 4. Renk Paleti (Dark Theme)

| Kullanim         | Renk      | Hex       |
|------------------|-----------|-----------|
| Background       | Koyu gri  | #0f1117   |
| Surface/Card     | Gri       | #1a1d29   |
| Surface Hover    | Acik gri  | #252836   |
| Sidebar          | Koyu      | #0d0f15   |
| Primary (accent) | Mavi      | #3b82f6   |
| Profit/Pozitif   | Yesil     | #22c55e   |
| Loss/Negatif     | Kirmizi   | #ef4444   |
| Warning          | Sari      | #eab308   |
| Text Primary     | Beyaz     | #e2e8f0   |
| Text Secondary   | Gri       | #94a3b8   |
| Border           | Koyu gri  | #2d3148   |

---

## 5. Teknik Stack

### Frontend
- **React 18** + TypeScript
- **Vite** (build tool)
- **React Router** (sayfa yonlendirme)
- **Recharts** (equity curve, bar chart, scatter plot)
- **TanStack Table** (trade listesi, sortable tablo)
- **Tailwind CSS** (hizli styling, dark theme)
- **Lucide React** (ikonlar)
- **Axios** (API istekleri)

### Backend
- **FastAPI** (Python)
- **Uvicorn** (ASGI server)
- **Mevcut backtester kodlari** (tick_engine, swinginess_strategy, optimize, download_aggtrades)

### API Endpoints (FastAPI)

| Method | Endpoint                    | Aciklama                          |
|--------|-----------------------------|-----------------------------------|
| GET    | /api/symbols                | Mevcut pair listesi               |
| GET    | /api/data/status            | Veri durumu (her pair)            |
| POST   | /api/data/download/{symbol} | Veri indirmeyi baslat             |
| GET    | /api/data/download/progress | Indirme durumu                    |
| POST   | /api/backtest/run           | Backtest calistir (params body)   |
| GET    | /api/backtest/results       | Son backtest sonuclari            |
| POST   | /api/optimize/start         | Optimizasyon baslat               |
| GET    | /api/optimize/results/{sym} | Optimizasyon sonuclari            |
| GET    | /api/optimize/progress      | Optimizasyon durumu               |

---

## 6. Kullanici Akislari

### Akis 1: Hizli Backtest
1. Sidebar'dan "Backtest" tikla
2. Pair sec (dropdown)
3. Parametreleri ayarla (veya default birak)
4. "Backtest Calistir" tikla
5. Loading spinner goster (progress: gun bazli)
6. Sonuclar, equity curve ve trade listesi gosterilir

### Akis 2: Optimizasyon
1. Sidebar'dan "Optimizasyon" tikla
2. Pair sec
3. "Optimizasyon Baslat" tikla
4. Progress bar (trial bazli)
5. Tamamlaninca en iyi 5 sonuc tablosu
6. Bir sonuc sec → IS vs OOS karsilastirma tablosu
7. "Bu parametrelerle backtest calistir" → Backtest sayfasina git

### Akis 3: Veri Indirme
1. Sidebar'dan "Veri Yonetimi" tikla
2. Eksik pair icin "Indir" tikla
3. Progress bar (ay bazli)
4. Tamamlaninca durum guncellenir

---

## 7. Responsive Davranisi

- **1200px+**: Tam layout (sidebar + content)
- **768-1199px**: Sidebar collapse (sadece ikonlar)
- **<768px**: Sidebar hamburger menu, tek kolon layout

---

## 8. Oncelik Sirasi (MVP)

1. FastAPI backend (API endpoints)
2. React projesi + Tailwind + Router setup
3. Layout (Sidebar + Header)
4. Backtest sayfasi (en kritik)
5. Dashboard
6. Optimizasyon sonuclari
7. Veri yonetimi
