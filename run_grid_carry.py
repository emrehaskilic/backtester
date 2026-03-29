"""3D Grid Strateji — Son sinyali tasi (sadece CONT sinyalleri)
Sinyal geldiginde yon degistir, sinyal yoksa onceki yonde kal.
REV sinyali = pozisyonu kapat, yeni yone girme.
"""
import numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_parquet('data/ETHUSDT_5m_5y.parquet')
df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
c = df['close'].values.astype(np.float64)
h = df['high'].values.astype(np.float64)
l = df['low'].values.astype(np.float64)
bv = df['buy_vol'].values.astype(np.float64)
sv = df['sell_vol'].values.astype(np.float64)
n = len(df)

delta = bv - sv
cvd = np.cumsum(delta)
BP = 48
EARLY_BARS = 12
COMMISSION = 0.08
TRAIN = 540
UPDATE = 180

candles = []
for i in range(n // BP):
    s = i * BP; e = s + BP
    candles.append({
        'start': s, 'open': c[s],
        'high': np.max(h[s:e]), 'low': np.min(l[s:e]),
        'close': c[e-1], 'dt': df['dt'].iloc[s],
    })

# Her mum icin sweep + sinyal bilgisi
all_candle_info = []
for i in range(1, len(candles)):
    prev = candles[i-1]; curr = candles[i]; bar = curr['start']

    swept_high = curr['high'] > prev['high']; swept_low = curr['low'] < prev['low']
    is_green = curr['close'] > curr['open']; is_red = curr['close'] < curr['open']

    sig, sweep_dir = None, None
    if swept_high and not swept_low:
        sig = 'high_cont' if curr['close'] > prev['high'] else ('high_rev' if is_red else None)
        sweep_dir = 'high'
    elif swept_low and not swept_high:
        sig = 'low_cont' if curr['close'] < prev['low'] else ('low_rev' if is_green else None)
        sweep_dir = 'low'
    elif swept_high and swept_low:
        if curr['close'] > prev['high']: sig, sweep_dir = 'high_cont', 'high'
        elif curr['close'] < prev['low']: sig, sweep_dir = 'low_cont', 'low'
        elif is_red: sig, sweep_dir = 'high_rev', 'high'
        elif is_green: sig, sweep_dir = 'low_rev', 'low'

    roc = np.nan
    if bar >= 36 and c[bar-36] > 0:
        roc = (c[bar] - c[bar-36]) / c[bar-36] * 100
        if sweep_dir == 'low': roc = -roc

    pre_cvd = np.nan
    if bar >= 48:
        pre_cvd = cvd[bar] - cvd[bar-48]
        if sweep_dir == 'low': pre_cvd = -pre_cvd

    early_p = np.nan
    entry_bar = bar + EARLY_BARS
    if entry_bar < n:
        early_p = (c[entry_bar] - c[bar]) / c[bar] * 100
        if sweep_dir == 'low': early_p = -early_p

    # 4H mum baslangic ve bitis fiyatlari (ilk 1h sonrasi → mum kapanisi)
    price_at_1h = c[entry_bar] if entry_bar < n else np.nan
    price_at_close = c[bar + BP - 1] if bar + BP - 1 < n else np.nan
    # 4H mum full fiyatlari (mum basi → mum sonu)
    price_at_open = c[bar] if bar < n else np.nan

    all_candle_info.append({
        'candle_idx': i, 'bar': bar, 'dt': curr['dt'],
        'signal': sig, 'sweep_dir': sweep_dir,
        'roc': roc, 'pre_cvd': pre_cvd, 'early_p': early_p,
        'price_open': price_at_open,
        'price_1h': price_at_1h,
        'price_close': price_at_close,
    })

edf = pd.DataFrame(all_candle_info)

def tercile_from_th(val, th_33, th_67):
    if val <= th_33: return 'Ters'
    elif val <= th_67: return 'Notr'
    else: return 'Ayni'

CONT_CELLS = {
    ('Ters','Ters','Ayni'), ('Ters','Notr','Ayni'), ('Ters','Ayni','Ayni'),
    ('Ters','Ters','Notr'), ('Ters','Ters','Ters'), ('Ters','Ayni','Notr'),
    ('Notr','Ters','Ayni'), ('Notr','Notr','Ayni'),
    ('Ayni','Ayni','Ayni'), ('Ayni','Notr','Ayni'),
}

REV_CELLS = {
    ('Ayni','Ayni','Ters'), ('Ayni','Notr','Ters'), ('Ayni','Ters','Ters'),
    ('Notr','Notr','Notr'),
}

print("3D GRID — SON SINYALI TASI")
print("=" * 120)
print("CONT sinyali → sweep yonunde pozisyon ac")
print("REV sinyali → pozisyonu kapat, beklemeye gec")
print("Sinyal yok → onceki pozisyonu tasi")
print()

# ══════════════════════════════════════════════════════════════
# 3 farkli mod test et
# ══════════════════════════════════════════════════════════════

modes = [
    ("A: Sadece CONT (onceki)", "cont_only"),       # onceki backtest — referans
    ("B: CONT tasi, sessizde tut", "carry"),          # CONT gelince gir, sessizde tut, REV'de cik
    ("C: CONT tasi, REV'de ters", "carry_rev"),       # CONT gelince gir, sessizde tut, REV'de ters gir
]

for mode_label, mode_name in modes:
    print(f"\n{'='*120}")
    print(f"  MOD {mode_label}")
    print(f"{'='*120}")

    current_th = None
    last_train_end = 0
    position = None  # None, "LONG", "SHORT"
    entry_price = None
    entry_time = None
    trades = []

    for _, ev in edf.iterrows():
        ci = ev['candle_idx']

        # Threshold guncelle
        if ci >= TRAIN and (current_th is None or ci - last_train_end >= UPDATE):
            td = edf[(edf['candle_idx'] >= ci-TRAIN) & (edf['candle_idx'] < ci)]
            tv = td[td['signal'].notna()].dropna(subset=['roc','pre_cvd','early_p'])
            if len(tv) >= 50:
                current_th = {
                    'roc_33': np.percentile(tv['roc'],33.3), 'roc_67': np.percentile(tv['roc'],66.7),
                    'cvd_33': np.percentile(tv['pre_cvd'],33.3), 'cvd_67': np.percentile(tv['pre_cvd'],66.7),
                    'early_33': np.percentile(tv['early_p'],33.3), 'early_67': np.percentile(tv['early_p'],66.7),
                }
                last_train_end = ci

        if current_th is None: continue

        # Grid sinyali hesapla (sadece sweep olan mumlarda)
        grid_signal = None
        if ev['signal'] is not None and not pd.isna(ev['roc']) and not pd.isna(ev['pre_cvd']) and not pd.isna(ev['early_p']):
            rt = tercile_from_th(ev['roc'], current_th['roc_33'], current_th['roc_67'])
            ct = tercile_from_th(ev['pre_cvd'], current_th['cvd_33'], current_th['cvd_67'])
            et = tercile_from_th(ev['early_p'], current_th['early_33'], current_th['early_67'])
            key = (rt, ct, et)
            if key in CONT_CELLS:
                grid_signal = "CONT"
            elif key in REV_CELLS:
                grid_signal = "REV"

        # Mod'a gore karar
        if mode_name == "cont_only":
            # Onceki backtest — sadece CONT sinyallerinde giris/cikis, her trade bagimsiz
            if grid_signal == "CONT" and not pd.isna(ev['price_1h']) and not pd.isna(ev['price_close']):
                direction = "LONG" if ev['sweep_dir'] == "high" else "SHORT"
                if direction == "LONG":
                    pnl = (ev['price_close'] - ev['price_1h']) / ev['price_1h'] * 100
                else:
                    pnl = (ev['price_1h'] - ev['price_close']) / ev['price_1h'] * 100
                trades.append({
                    'dt': ev['dt'], 'direction': direction, 'pnl': pnl - COMMISSION,
                    'year': pd.Timestamp(ev['dt']).year,
                })

        elif mode_name == "carry":
            # CONT → pozisyon ac, sessizde tut, REV → cik
            new_dir = None
            if grid_signal == "CONT":
                new_dir = "LONG" if ev['sweep_dir'] == "high" else "SHORT"
            elif grid_signal == "REV":
                new_dir = "FLAT"  # kapat

            if new_dir is not None and new_dir != position:
                # Onceki pozisyonu kapat
                if position is not None and position != "FLAT" and entry_price is not None:
                    exit_p = ev['price_1h'] if not pd.isna(ev['price_1h']) else ev['price_open']
                    if not pd.isna(exit_p):
                        if position == "LONG":
                            pnl = (exit_p - entry_price) / entry_price * 100
                        else:
                            pnl = (entry_price - exit_p) / entry_price * 100
                        trades.append({
                            'dt': entry_time, 'direction': position, 'pnl': pnl - COMMISSION,
                            'year': pd.Timestamp(entry_time).year,
                        })

                # Yeni pozisyon ac
                if new_dir != "FLAT" and not pd.isna(ev['price_1h']):
                    position = new_dir
                    entry_price = ev['price_1h']
                    entry_time = ev['dt']
                else:
                    position = None
                    entry_price = None

            # Sessizde → position degismez (tasi)

        elif mode_name == "carry_rev":
            # CONT → sweep yonunde, REV → ters yonde, sessizde tut
            new_dir = None
            if grid_signal == "CONT":
                new_dir = "LONG" if ev['sweep_dir'] == "high" else "SHORT"
            elif grid_signal == "REV":
                new_dir = "SHORT" if ev['sweep_dir'] == "high" else "LONG"

            if new_dir is not None and new_dir != position:
                if position is not None and entry_price is not None:
                    exit_p = ev['price_1h'] if not pd.isna(ev['price_1h']) else ev['price_open']
                    if not pd.isna(exit_p):
                        if position == "LONG":
                            pnl = (exit_p - entry_price) / entry_price * 100
                        else:
                            pnl = (entry_price - exit_p) / entry_price * 100
                        trades.append({
                            'dt': entry_time, 'direction': position, 'pnl': pnl - COMMISSION,
                            'year': pd.Timestamp(entry_time).year,
                        })

                if not pd.isna(ev['price_1h']):
                    position = new_dir
                    entry_price = ev['price_1h']
                    entry_time = ev['dt']

    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        print("  Trade yok!")
        continue

    equity = [100.0]
    for p in tdf['pnl']:
        equity.append(equity[-1] * (1 + p / 100))
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100

    print(f"  Trade: {len(tdf)} | WR: {(tdf['pnl']>0).mean()*100:.1f}% | Avg: {tdf['pnl'].mean():.4f}% | Sum: {tdf['pnl'].sum():.1f}%")
    print(f"  $100 → ${equity[-1]:.2f} | Max DD: {dd.max():.1f}%")

    longs = tdf[tdf['direction']=='LONG']
    shorts = tdf[tdf['direction']=='SHORT']
    print(f"  LONG:  {len(longs)} | WR: {(longs['pnl']>0).mean()*100:.1f}% | Sum: {longs['pnl'].sum():.1f}%")
    print(f"  SHORT: {len(shorts)} | WR: {(shorts['pnl']>0).mean()*100:.1f}% | Sum: {shorts['pnl'].sum():.1f}%")

    print(f"\n  YILLIK:")
    print(f"  {'Yil':>6s} | {'N':>5s} | {'WR':>6s} | {'Sum':>9s}")
    print(f"  {'-'*35}")
    for year, grp in tdf.groupby('year'):
        print(f"  {year:>6d} | {len(grp):>5d} | {(grp['pnl']>0).mean()*100:>5.1f}% | {grp['pnl'].sum():>+8.1f}%")

    tdf['week'] = pd.to_datetime(tdf['dt']).dt.to_period('W')
    weekly = tdf.groupby('week')['pnl'].sum()
    neg_weeks = (weekly < 0).sum()
    print(f"\n  HAFTALIK: {(weekly>0).sum()}/{len(weekly)} pozitif ({(weekly>0).mean()*100:.0f}%) | Negatif hafta: {neg_weeks}")
    print(f"  En kotu hafta: {weekly.min():.2f}%")

    tdf['month'] = pd.to_datetime(tdf['dt']).dt.to_period('M')
    monthly = tdf.groupby('month')['pnl'].sum()
    neg_months = (monthly < 0).sum()
    print(f"  AYLIK: {(monthly>0).sum()}/{len(monthly)} pozitif ({(monthly>0).mean()*100:.0f}%) | Negatif ay: {neg_months}")
    print(f"  En kotu ay: {monthly.min():.2f}%")

print()
