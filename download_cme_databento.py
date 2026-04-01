"""
Databento CME ETH Futures (ETH1!) Data Downloader

Databento API ile CME Globex'ten ETH futures verisini çeker.
5m OHLCV + volume bilgisi ile parquet olarak kaydeder.

Kullanım:
  1. pip install databento
  2. export DATABENTO_API_KEY="db-xxxxx"  (veya script'e yaz)
  3. python download_cme_databento.py

Databento ücretsiz tier:
  - Günlük $0 kredi ile başlarsın, ilk kayıtta ücretsiz kredi veriyorlar
  - CME futures historical data ücretli ama makul

Docs: https://databento.com/docs
"""
import os
import sys
import numpy as np
import pandas as pd

try:
    import databento as db
except ImportError:
    print("databento paketi yüklü değil. Yüklemek için:")
    print("  pip install databento")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
API_KEY = os.environ.get("DATABENTO_API_KEY", "")
if not API_KEY:
    print("DATABENTO_API_KEY environment variable gerekli!")
    print("  export DATABENTO_API_KEY='db-xxxxx'")
    print("")
    print("API key almak için: https://databento.com/signup")
    sys.exit(1)

# CME Globex dataset
DATASET = "GLBX.MDP3"

# ETH futures continuous front month
# Databento symbology:
#   "ETH.FUT"     → tüm ETH futures contracts
#   "ETH.c.0"     → continuous front month (roll at expiry)
#   "ETHM5"       → specific contract (June 2025)
SYMBOL = "ETH.c.0"

# Date range (5 years)
START = "2021-04-01"
END = "2026-03-31"

OUTPUT = "data/CME_ETH1_5m_5y.parquet"

# ═══════════════════════════════════════════════════════════════
# OPTION 1: OHLCV-1m → aggregate to 5m
# (Databento 5m OHLCV doğrudan olmayabilir, 1m var)
# ═══════════════════════════════════════════════════════════════

def download_ohlcv():
    """Download 1-minute OHLCV and aggregate to 5m."""
    client = db.Historical(API_KEY)

    print(f"Downloading {SYMBOL} from {DATASET}...")
    print(f"  Period: {START} → {END}")
    print(f"  Schema: ohlcv-1m (will aggregate to 5m)")

    # İlk önce maliyet kontrolü yap
    print("\nChecking cost...")
    try:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema="ohlcv-1m",
            start=START,
            end=END,
        )
        print(f"  Estimated cost: ${cost:.2f}")
        response = input("  Devam etmek istiyor musun? (y/n): ")
        if response.lower() != 'y':
            print("İptal edildi.")
            return None
    except Exception as e:
        print(f"  Cost check failed: {e}")
        print("  Devam ediyorum...")

    # Download
    print("\nDownloading data...")
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[SYMBOL],
        schema="ohlcv-1m",
        start=START,
        end=END,
        stype_in="continuous",  # continuous contract
    )

    # Convert to DataFrame
    df = data.to_df()
    print(f"  Downloaded: {len(df):,} 1m bars")

    if len(df) == 0:
        print("Veri bulunamadı!")
        return None

    return df


def download_trades():
    """Alternative: Download trade data for buy/sell volume decomposition."""
    client = db.Historical(API_KEY)

    print(f"\nDownloading trades for buy/sell volume...")
    print("  (Bu daha pahalı olabilir, ama buy_vol/sell_vol ayrımı verir)")

    # Cost check
    try:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema="trades",
            start=START,
            end=END,
        )
        print(f"  Estimated cost: ${cost:.2f}")
        if cost > 50:
            print("  ⚠ Maliyet yüksek! trades yerine ohlcv kullanılacak.")
            return None
    except Exception:
        pass

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[SYMBOL],
        schema="trades",
        start=START,
        end=END,
        stype_in="continuous",
    )

    return data.to_df()


def aggregate_1m_to_5m(df_1m):
    """Aggregate 1-minute OHLCV to 5-minute."""
    # Databento OHLCV columns: open, high, low, close, volume
    df = df_1m.copy()

    # ts_event is the bar timestamp
    if 'ts_event' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ts_event'])
    elif df.index.name == 'ts_event':
        df['timestamp'] = df.index
        df = df.reset_index(drop=True)
    else:
        df['timestamp'] = df.index
        df = df.reset_index(drop=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Floor to 5-minute intervals
    df['bar_5m'] = df['timestamp'].dt.floor('5min')

    # Aggregate
    agg = df.groupby('bar_5m').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).reset_index()

    agg.rename(columns={'bar_5m': 'timestamp'}, inplace=True)

    print(f"  Aggregated: {len(df):,} 1m bars → {len(agg):,} 5m bars")
    return agg


def estimate_buy_sell_volume(df_5m):
    """
    Buy/sell volume tahmini (trade data olmadan).

    Yöntem: Close-Open rule
    - Close > Open → bar bullish → buy_vol = volume * ratio, sell_vol = volume * (1-ratio)
    - Ratio, fiyat hareketinin büyüklüğüne göre ayarlanır
    """
    n = len(df_5m)
    buy_vol = np.zeros(n)
    sell_vol = np.zeros(n)

    for i in range(n):
        vol = df_5m.iloc[i]['volume']
        o = df_5m.iloc[i]['open']
        c = df_5m.iloc[i]['close']
        h = df_5m.iloc[i]['high']
        l = df_5m.iloc[i]['low']

        if h == l or vol == 0:
            buy_vol[i] = vol * 0.5
            sell_vol[i] = vol * 0.5
            continue

        # Body position within range
        body_mid = (o + c) / 2
        range_mid = (h + l) / 2
        body_pos = (body_mid - l) / (h - l)  # 0=bottom, 1=top

        # Buy ratio: higher body position → more buying
        buy_ratio = 0.3 + 0.4 * body_pos  # range: 0.3 to 0.7
        buy_vol[i] = vol * buy_ratio
        sell_vol[i] = vol * (1 - buy_ratio)

    df_5m = df_5m.copy()
    df_5m['buy_vol'] = buy_vol
    df_5m['sell_vol'] = sell_vol

    return df_5m


def fetch_oi_from_databento():
    """Open Interest verisi çek (ayrı schema)."""
    client = db.Historical(API_KEY)

    try:
        # OI genellikle "definition" veya "statistics" schema'sında
        # CME daily OI
        print("\nFetching Open Interest data...")
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema="statistics",  # veya "definition"
            start=START,
            end=END,
            stype_in="continuous",
        )
        df_oi = data.to_df()
        print(f"  OI data points: {len(df_oi):,}")
        return df_oi
    except Exception as e:
        print(f"  OI fetch failed: {e}")
        print("  Volume-based OI estimate kullanılacak.")
        return None


def build_oi_series(df_5m, df_oi=None):
    """OI serisini oluştur veya estimate et."""
    n = len(df_5m)

    if df_oi is not None and len(df_oi) > 0:
        # Map OI to 5m bars (forward fill)
        # OI genellikle günlük gelir, 5m bar'lara ffill yaparız
        oi_series = pd.Series(index=df_5m['timestamp'], dtype=float)
        # ... mapping logic
        df_5m['open_interest'] = oi_series.ffill().values
    else:
        # Estimate: volume-based proxy
        vol_ema = pd.Series(df_5m['volume'].values).ewm(span=288).mean().values
        df_5m = df_5m.copy()
        df_5m['open_interest'] = vol_ema * 5000  # rough scaling

    return df_5m


def format_output(df_5m):
    """Final format: match Binance parquet structure."""
    # open_time as milliseconds
    df = df_5m.copy()
    df['open_time'] = (df['timestamp'].astype(np.int64) // 10**6).astype(np.uint64)

    result = pd.DataFrame({
        'open_time': df['open_time'],
        'open': df['open'].astype(np.float64),
        'high': df['high'].astype(np.float64),
        'low': df['low'].astype(np.float64),
        'close': df['close'].astype(np.float64),
        'buy_vol': df['buy_vol'].astype(np.float64),
        'sell_vol': df['sell_vol'].astype(np.float64),
        'open_interest': df['open_interest'].astype(np.float64),
    })

    return result


def main():
    os.makedirs("data", exist_ok=True)

    # Step 1: Download OHLCV (1m)
    df_1m = download_ohlcv()
    if df_1m is None or len(df_1m) == 0:
        print("Veri indirilemedi!")
        return

    # Step 2: Aggregate to 5m
    df_5m = aggregate_1m_to_5m(df_1m)

    # Step 3: Buy/sell volume (trade data varsa decompose, yoksa estimate)
    try:
        df_trades = download_trades()
        if df_trades is not None and len(df_trades) > 0:
            # Trade-level buy/sell decomposition
            print("  Trade-level buy/sell volume decomposition...")
            df_trades['timestamp_5m'] = df_trades.index.floor('5min')
            # Aggressor side: CME'de 'side' field
            if 'side' in df_trades.columns:
                buys = df_trades[df_trades['side'] == 'B'].groupby('timestamp_5m')['size'].sum()
                sells = df_trades[df_trades['side'] == 'S'].groupby('timestamp_5m')['size'].sum()
                df_5m['buy_vol'] = df_5m['timestamp'].map(buys).fillna(0).values
                df_5m['sell_vol'] = df_5m['timestamp'].map(sells).fillna(0).values
            else:
                df_5m = estimate_buy_sell_volume(df_5m)
        else:
            df_5m = estimate_buy_sell_volume(df_5m)
    except Exception as e:
        print(f"  Trade download failed: {e}")
        df_5m = estimate_buy_sell_volume(df_5m)

    # Step 4: Open Interest
    try:
        df_oi = fetch_oi_from_databento()
        df_5m = build_oi_series(df_5m, df_oi)
    except Exception:
        df_5m = build_oi_series(df_5m, None)

    # Step 5: Format and save
    result = format_output(df_5m)

    result.to_parquet(OUTPUT, index=False)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  CME ETH1! DATA SAVED")
    print(f"{'='*60}")
    print(f"  Bars:  {len(result):,}")
    print(f"  From:  {pd.to_datetime(result['open_time'].iloc[0], unit='ms')}")
    print(f"  To:    {pd.to_datetime(result['open_time'].iloc[-1], unit='ms')}")
    print(f"  File:  {OUTPUT} ({size_mb:.1f} MB)")
    print(f"  Cols:  {list(result.columns)}")


if __name__ == "__main__":
    main()
