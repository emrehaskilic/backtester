"""
Liquidity Sweep Analysis — 4H high/low sweep sonrası ne olmuş?

1. 3m veriyi 4H'ye resample → swing high/low bul
2. 3m'de sweep anlarını tespit et
3. Sweep sonrası: continuation mu reversal mı?
4. Her sweep anında feature'ları hesapla
5. Continuation vs reversal ayıran koşulları bul
"""
import numpy as np, pandas as pd, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Data ──
df = pd.read_parquet('data/ETHUSDT_3m_vol_11mo.parquet')
df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
oi5 = pd.read_parquet('data/ETHUSDT_OI_5m_11mo.parquet')
df['oi'] = np.nan
oi_map = dict(zip(oi5['open_time'], oi5['open_interest']))
for idx in range(len(df)):
    ot = df['open_time'].iloc[idx]
    if ot in oi_map:
        df.iloc[idx, df.columns.get_loc('oi')] = oi_map[ot]
df['oi'] = df['oi'].ffill().fillna(0.0)

total = len(df)
closes = df['close'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
buy_vol = df['buy_vol'].values.astype(np.float64)
sell_vol = df['sell_vol'].values.astype(np.float64)
oi_arr = df['oi'].values.astype(np.float64)

print(f"Data: {total:,} bar (3m) | {df['dt'].iloc[0].strftime('%Y-%m-%d')} - {df['dt'].iloc[-1].strftime('%Y-%m-%d')}")
print()

# ── Step 1: 4H resample ──
df_4h = df.set_index('dt').resample('4h').agg({
    'open_time': 'first',
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'buy_vol': 'sum',
    'sell_vol': 'sum',
}).dropna()

print(f"4H bar: {len(df_4h):,}")

# ── Step 2: Swing high/low (sol 3, sag 3 bar'dan yuksek/alcak) ──
swing_lookback = 3
h4_highs = df_4h['high'].values
h4_lows = df_4h['low'].values
h4_times = df_4h.index

swing_highs = []  # (4h_index, price, timestamp)
swing_lows = []

for i in range(swing_lookback, len(df_4h) - swing_lookback):
    # Swing high: ortadaki bar en yuksek
    is_sh = True
    for j in range(1, swing_lookback + 1):
        if h4_highs[i - j] >= h4_highs[i] or h4_highs[i + j] >= h4_highs[i]:
            is_sh = False
            break
    if is_sh:
        swing_highs.append((i, h4_highs[i], h4_times[i]))

    # Swing low: ortadaki bar en alcak
    is_sl = True
    for j in range(1, swing_lookback + 1):
        if h4_lows[i - j] <= h4_lows[i] or h4_lows[i + j] <= h4_lows[i]:
            is_sl = False
            break
    if is_sl:
        swing_lows.append((i, h4_lows[i], h4_times[i]))

print(f"4H Swing High: {len(swing_highs)} | Swing Low: {len(swing_lows)}")
print()

# ── Step 3: 3m'de sweep tespit ──
# Sweep = fiyat swing seviyesini gectigi AN
# Her swing seviyesi sadece 1 kez sweep edilebilir

# 3m bar -> timestamp mapping
bar_times = df['dt'].values

# ATR (3m)
tr = np.zeros(total); tr[0] = highs[0] - lows[0]
for i in range(1, total):
    tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
atr = np.full(total, np.nan); atr[14] = np.mean(tr[1:15])
for i in range(15, total): atr[i] = (atr[i-1]*13 + tr[i]) / 14

# CVD
delta = buy_vol - sell_vol
cvd_20 = np.zeros(total)
for i in range(20, total):
    cvd_20[i] = np.sum(delta[i-19:i+1])

# Volume
total_vol = buy_vol + sell_vol
vol_mean = np.full(total, np.nan)
for i in range(20, total):
    vol_mean[i] = np.mean(total_vol[i-19:i+1])

# OI change
oi_change = np.zeros(total)
for i in range(20, total):
    if oi_arr[i-20] > 0 and not np.isnan(oi_arr[i]) and not np.isnan(oi_arr[i-20]):
        oi_change[i] = (oi_arr[i] - oi_arr[i-20]) / oi_arr[i-20] * 100

# EMA 20
ema20 = np.full(total, np.nan); k = 2/21; ema20[0] = closes[0]
for i in range(1, total): ema20[i] = closes[i]*k + ema20[i-1]*(1-k)

# Sonraki N bar icinde ne olmus: max favorable, max adverse, net
def measure_aftermath(bar_idx, direction, horizons=[80, 160, 320, 480]):
    """direction: 1=yukarı bekliyoruz (sweep low sonrası), -1=aşağı (sweep high sonrası)"""
    results = {}
    entry = closes[bar_idx]
    if entry <= 0: return results

    for h in horizons:
        end = min(bar_idx + h, total - 1)
        if bar_idx >= end:
            results[h] = {'net': 0, 'mfe': 0, 'mae': 0, 'continuation': False}
            continue

        future_closes = closes[bar_idx+1:end+1]
        future_highs = highs[bar_idx+1:end+1]
        future_lows = lows[bar_idx+1:end+1]

        if direction > 0:  # yukarı bekliyoruz
            net = (future_closes[-1] - entry) / entry * 100
            mfe = (np.max(future_highs) - entry) / entry * 100  # max favorable
            mae = (entry - np.min(future_lows)) / entry * 100   # max adverse
            continuation = net > 0.3  # net %0.3+ yukari gitmis
        else:  # asagi bekliyoruz
            net = (entry - future_closes[-1]) / entry * 100
            mfe = (entry - np.min(future_lows)) / entry * 100
            mae = (np.max(future_highs) - entry) / entry * 100
            continuation = net > 0.3

        results[h] = {'net': net, 'mfe': mfe, 'mae': mae, 'continuation': continuation}
    return results


# ── HIGH SWEEP tespiti ──
high_sweeps = []
used_swings = set()

for sh_idx, sh_price, sh_time in swing_highs:
    if sh_idx in used_swings: continue
    # Swing'den sonraki bar'lardan itibaren 3m'de ara
    # 4H swing olustugundan min 3*4h = 12h sonra sweep olabilir (sag taraf onayı)
    sh_ts = pd.Timestamp(sh_time)
    search_start_ts = sh_ts + pd.Timedelta(hours=12)

    for i in range(200, total - 480):
        bar_ts = pd.Timestamp(bar_times[i])
        if bar_ts < search_start_ts: continue
        if bar_ts > sh_ts + pd.Timedelta(days=14): break  # max 14 gun icinde sweep

        # Sweep: high bu bar'da swing high'i gecti
        if highs[i] > sh_price and highs[i-1] <= sh_price:
            # Feature'lar
            if np.isnan(atr[i]) or atr[i] <= 0: continue

            cvd_val = cvd_20[i]
            cvd_direction = 1 if cvd_val > 0 else -1
            vol_ratio = total_vol[i] / vol_mean[i] if not np.isnan(vol_mean[i]) and vol_mean[i] > 0 else 1
            oi_chg = oi_change[i]
            price_vs_ema = (closes[i] - ema20[i]) / atr[i] if not np.isnan(ema20[i]) else 0
            imbalance = (buy_vol[i] - sell_vol[i]) / (buy_vol[i] + sell_vol[i]) if (buy_vol[i] + sell_vol[i]) > 0 else 0
            sweep_depth = (highs[i] - sh_price) / atr[i]  # sweep ne kadar gecti

            aftermath = measure_aftermath(i, 1)  # sweep high sonrası yukarı devam mı?
            aftermath_rev = measure_aftermath(i, -1)  # yoksa asagi mi dondu?

            high_sweeps.append({
                'bar': i, 'date': df['dt'].iloc[i].strftime('%Y-%m-%d %H:%M'),
                'swing_price': sh_price, 'sweep_price': highs[i],
                'sweep_depth': sweep_depth,
                'cvd': cvd_val, 'cvd_dir': cvd_direction,
                'vol_ratio': vol_ratio, 'oi_change': oi_chg,
                'price_vs_ema': price_vs_ema, 'imbalance': imbalance,
                'aftermath_cont': aftermath,   # yukarı devam
                'aftermath_rev': aftermath_rev,  # asagi donus
            })
            used_swings.add(sh_idx)
            break

# ── LOW SWEEP tespiti ──
low_sweeps = []
used_swings_low = set()

for sl_idx, sl_price, sl_time in swing_lows:
    if sl_idx in used_swings_low: continue
    sl_ts = pd.Timestamp(sl_time)
    search_start_ts = sl_ts + pd.Timedelta(hours=12)

    for i in range(200, total - 480):
        bar_ts = pd.Timestamp(bar_times[i])
        if bar_ts < search_start_ts: continue
        if bar_ts > sl_ts + pd.Timedelta(days=14): break

        if lows[i] < sl_price and lows[i-1] >= sl_price:
            if np.isnan(atr[i]) or atr[i] <= 0: continue

            cvd_val = cvd_20[i]
            cvd_direction = 1 if cvd_val > 0 else -1
            vol_ratio = total_vol[i] / vol_mean[i] if not np.isnan(vol_mean[i]) and vol_mean[i] > 0 else 1
            oi_chg = oi_change[i]
            price_vs_ema = (closes[i] - ema20[i]) / atr[i] if not np.isnan(ema20[i]) else 0
            imbalance = (buy_vol[i] - sell_vol[i]) / (buy_vol[i] + sell_vol[i]) if (buy_vol[i] + sell_vol[i]) > 0 else 0
            sweep_depth = (sl_price - lows[i]) / atr[i]

            aftermath = measure_aftermath(i, -1)  # sweep low sonrası asagi devam mı?
            aftermath_rev = measure_aftermath(i, 1)   # yoksa yukari mi dondu?

            low_sweeps.append({
                'bar': i, 'date': df['dt'].iloc[i].strftime('%Y-%m-%d %H:%M'),
                'swing_price': sl_price, 'sweep_price': lows[i],
                'sweep_depth': sweep_depth,
                'cvd': cvd_val, 'cvd_dir': cvd_direction,
                'vol_ratio': vol_ratio, 'oi_change': oi_chg,
                'price_vs_ema': price_vs_ema, 'imbalance': imbalance,
                'aftermath_cont': aftermath,
                'aftermath_rev': aftermath_rev,
            })
            used_swings_low.add(sl_idx)
            break

print(f"High Sweep: {len(high_sweeps)} event | Low Sweep: {len(low_sweeps)} event")
print()

# ── Analiz ──
horizons = [80, 160, 320, 480]
h_names = ['4h', '8h', '16h', '24h']

# ═══ HIGH SWEEP ANALIZI ═══
print("=" * 100)
print("  4H HIGH SWEEP ANALIZI")
print("  High sweep sonrasi: yukari devam mi (continuation) yoksa asagi mi dondu (reversal)?")
print("=" * 100)
print()

for hi, (h, hname) in enumerate(zip(horizons, h_names)):
    cont = [s for s in high_sweeps if h in s['aftermath_cont'] and s['aftermath_cont'][h]['continuation']]
    rev = [s for s in high_sweeps if h in s['aftermath_rev'] and s['aftermath_rev'][h]['continuation']]
    neither = len(high_sweeps) - len(cont) - len(rev)

    cont_pct = len(cont)/max(len(high_sweeps),1)*100
    rev_pct = len(rev)/max(len(high_sweeps),1)*100

    print(f"  Horizon {hname}: Continuation (yukari): {len(cont)}/{len(high_sweeps)} ({cont_pct:.1f}%) | "
          f"Reversal (asagi): {len(rev)}/{len(high_sweeps)} ({rev_pct:.1f}%) | "
          f"Notr: {neither}")

    if cont:
        avg_mfe = np.mean([s['aftermath_cont'][h]['mfe'] for s in cont])
        avg_net = np.mean([s['aftermath_cont'][h]['net'] for s in cont])
        print(f"    Continuation: Avg Net: {avg_net:+.3f}% | Avg MFE: {avg_mfe:+.3f}%")
    if rev:
        avg_mfe = np.mean([s['aftermath_rev'][h]['mfe'] for s in rev])
        avg_net = np.mean([s['aftermath_rev'][h]['net'] for s in rev])
        print(f"    Reversal:     Avg Net: {avg_net:+.3f}% | Avg MFE: {avg_mfe:+.3f}%")

print()

# ═══ LOW SWEEP ANALIZI ═══
print("=" * 100)
print("  4H LOW SWEEP ANALIZI")
print("  Low sweep sonrasi: asagi devam mi (continuation) yoksa yukari mi dondu (reversal)?")
print("=" * 100)
print()

for hi, (h, hname) in enumerate(zip(horizons, h_names)):
    cont = [s for s in low_sweeps if h in s['aftermath_cont'] and s['aftermath_cont'][h]['continuation']]
    rev = [s for s in low_sweeps if h in s['aftermath_rev'] and s['aftermath_rev'][h]['continuation']]
    neither = len(low_sweeps) - len(cont) - len(rev)

    cont_pct = len(cont)/max(len(low_sweeps),1)*100
    rev_pct = len(rev)/max(len(low_sweeps),1)*100

    print(f"  Horizon {hname}: Continuation (asagi): {len(cont)}/{len(low_sweeps)} ({cont_pct:.1f}%) | "
          f"Reversal (yukari): {len(rev)}/{len(low_sweeps)} ({rev_pct:.1f}%) | "
          f"Notr: {neither}")

    if cont:
        avg_mfe = np.mean([s['aftermath_cont'][h]['mfe'] for s in cont])
        avg_net = np.mean([s['aftermath_cont'][h]['net'] for s in cont])
        print(f"    Continuation: Avg Net: {avg_net:+.3f}% | Avg MFE: {avg_mfe:+.3f}%")
    if rev:
        avg_mfe = np.mean([s['aftermath_rev'][h]['mfe'] for s in rev])
        avg_net = np.mean([s['aftermath_rev'][h]['net'] for s in rev])
        print(f"    Reversal:     Avg Net: {avg_net:+.3f}% | Avg MFE: {avg_mfe:+.3f}%")

print()

# ═══ FEATURE FARKI: Continuation vs Reversal ═══
print("=" * 100)
print("  CONTINUATION vs REVERSAL — Feature karsilastirma (8h horizon)")
print("=" * 100)
print()

h = 160  # 8h

for sweep_type, sweeps, cont_dir, rev_dir in [
    ("HIGH SWEEP", high_sweeps, "yukari devam", "asagi donus"),
    ("LOW SWEEP", low_sweeps, "asagi devam", "yukari donus"),
]:
    cont = [s for s in sweeps if h in s['aftermath_cont'] and s['aftermath_cont'][h]['continuation']]
    rev = [s for s in sweeps if h in s['aftermath_rev'] and s['aftermath_rev'][h]['continuation']]

    if not cont or not rev:
        print(f"  {sweep_type}: Yeterli ornek yok")
        continue

    print(f"  {sweep_type} ({cont_dir} vs {rev_dir}):")
    print(f"  {'Feature':20s} | {'Continuation':>15s} | {'Reversal':>15s} | {'Fark':>10s}")
    print(f"  {'-'*70}")

    for feat in ['sweep_depth', 'cvd', 'vol_ratio', 'oi_change', 'price_vs_ema', 'imbalance']:
        c_vals = [s[feat] for s in cont]
        r_vals = [s[feat] for s in rev]
        c_avg = np.mean(c_vals)
        r_avg = np.mean(r_vals)
        diff = c_avg - r_avg
        sig = "***" if abs(diff) > 0.5 * (abs(c_avg) + abs(r_avg) + 0.01) else "**" if abs(diff) > 0.3 * (abs(c_avg) + abs(r_avg) + 0.01) else ""
        print(f"  {feat:20s} | {c_avg:>+15.4f} | {r_avg:>+15.4f} | {diff:>+10.4f} {sig}")

    # CVD direction distribution
    c_cvd_pos = sum(1 for s in cont if s['cvd_dir'] > 0) / len(cont) * 100
    r_cvd_pos = sum(1 for s in rev if s['cvd_dir'] > 0) / len(rev) * 100
    print(f"  {'CVD pozitif %':20s} | {c_cvd_pos:>14.1f}% | {r_cvd_pos:>14.1f}% | {c_cvd_pos-r_cvd_pos:>+9.1f}%")
    print()

# ═══ PATTERN DISCOVERY ═══
print("=" * 100)
print("  SWEEP PATTERN DISCOVERY — Hangi kosullarda continuation, hangisinde reversal?")
print("=" * 100)
print()

for sweep_type, sweeps in [("HIGH SWEEP (short reversal)", high_sweeps), ("LOW SWEEP (long reversal)", low_sweeps)]:
    h = 160
    if sweep_type.startswith("HIGH"):
        # Reversal = asagi donus (short giris)
        labels = [1 if (h in s['aftermath_rev'] and s['aftermath_rev'][h]['continuation']) else 0 for s in sweeps]
    else:
        # Reversal = yukari donus (long giris)
        labels = [1 if (h in s['aftermath_rev'] and s['aftermath_rev'][h]['continuation']) else 0 for s in sweeps]

    base_wr = sum(labels) / max(len(labels), 1) * 100
    print(f"  {sweep_type}")
    print(f"  Base reversal rate (8h): {sum(labels)}/{len(labels)} ({base_wr:.1f}%)")
    print()

    # Feature-based filtering
    conditions = [
        ("CVD negatif (alicilar zayif)", lambda s: s['cvd'] < 0),
        ("CVD pozitif (alicilar guclu)", lambda s: s['cvd'] > 0),
        ("Yuksek volume (vol_ratio > 1.5)", lambda s: s['vol_ratio'] > 1.5),
        ("Dusuk volume (vol_ratio < 0.7)", lambda s: s['vol_ratio'] < 0.7),
        ("OI artiyor (> +0.3%)", lambda s: s['oi_change'] > 0.3),
        ("OI azaliyor (< -0.3%)", lambda s: s['oi_change'] < -0.3),
        ("Fiyat EMA uzerinde", lambda s: s['price_vs_ema'] > 0.5),
        ("Fiyat EMA altinda", lambda s: s['price_vs_ema'] < -0.5),
        ("Buyuk sweep (depth > 0.5 ATR)", lambda s: s['sweep_depth'] > 0.5),
        ("Kucuk sweep (depth < 0.2 ATR)", lambda s: s['sweep_depth'] < 0.2),
        ("Satis baskisi (imbalance < -0.1)", lambda s: s['imbalance'] < -0.1),
        ("Alis baskisi (imbalance > 0.1)", lambda s: s['imbalance'] > 0.1),
    ]

    print(f"  {'Kosul':45s} | {'N':>4s} | {'Rev':>4s} | {'RevWR%':>7s} | {'vs Base':>8s}")
    print(f"  {'-'*80}")

    for desc, cond_fn in conditions:
        filtered = [(s, l) for s, l in zip(sweeps, labels) if cond_fn(s)]
        if len(filtered) < 10: continue
        n = len(filtered)
        rev_cnt = sum(l for _, l in filtered)
        wr = rev_cnt / n * 100
        diff = wr - base_wr
        sig = "***" if abs(diff) > 10 else "**" if abs(diff) > 5 else "*" if abs(diff) > 3 else ""
        print(f"  {desc:45s} | {n:>4d} | {rev_cnt:>4d} | {wr:>7.1f} | {diff:>+7.1f}% {sig}")

    # 2-condition combos
    print()
    print(f"  IKILI KOMBINASYONLAR (en iyi 10):")
    combos = []
    for i, (d1, c1) in enumerate(conditions):
        for j, (d2, c2) in enumerate(conditions):
            if j <= i: continue
            filtered = [(s, l) for s, l in zip(sweeps, labels) if c1(s) and c2(s)]
            if len(filtered) < 15: continue
            n = len(filtered)
            rev_cnt = sum(l for _, l in filtered)
            wr = rev_cnt / n * 100
            diff = wr - base_wr
            combos.append((f"{d1} + {d2}", n, rev_cnt, wr, diff))

    combos.sort(key=lambda x: -x[3])
    print(f"  {'Kosul':70s} | {'N':>4s} | {'RevWR%':>7s} | {'vs Base':>8s}")
    print(f"  {'-'*100}")
    for desc, n, rev, wr, diff in combos[:10]:
        sig = "***" if diff > 10 else "**" if diff > 5 else ""
        print(f"  {desc:70s} | {n:>4d} | {wr:>7.1f} | {diff:>+7.1f}% {sig}")
    print()
