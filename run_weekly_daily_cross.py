"""Haftalik mum durumunun gunluk sweep sonuclarina etkisi
- Haftalik mum continuation ise, o haftadaki gunluk mumlar nasil kapaniyor?
- Haftalik mum reversal ise?
- Haftalik mum ambiguous/inside bar ise?
"""
import numpy as np, pandas as pd, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import rust_engine

def load(path):
    df = pd.read_parquet(path)
    sa = np.ascontiguousarray
    return {
        'c': sa(df['close'].values, dtype=np.float64),
        'h': sa(df['high'].values, dtype=np.float64),
        'l': sa(df['low'].values, dtype=np.float64),
        'bv': sa(df['buy_vol'].values, dtype=np.float64),
        'sv': sa(df['sell_vol'].values, dtype=np.float64),
        'oi': sa(df['open_interest'].fillna(0).values, dtype=np.float64),
        'n': len(df),
    }

eth_perp = load('data/ETHUSDT_5m_5y.parquet')

BP_1D = 288   # 5m bars per day
BP_1W = 2016  # 5m bars per week

n = eth_perp['n']

print("HAFTALIK → GUNLUK CROSS-TIMEFRAME ANALIZI")
print("ETH Perp, 5 yillik veri")
print("=" * 120)
print()

# ══════════════════════════════════════════════════════════════
# 1) Haftalik mumlari olustur ve siniflandir
# ══════════════════════════════════════════════════════════════

n_weeks = n // BP_1W
n_days = n // BP_1D

# Haftalik mumlar
weekly_candles = []  # (start_bar, open, high, low, close)
for w in range(n_weeks):
    start = w * BP_1W
    end = start + BP_1W
    o = eth_perp['c'][start]
    h = max(eth_perp['h'][start:end])
    l = min(eth_perp['l'][start:end])
    c = eth_perp['c'][end - 1]
    weekly_candles.append((start, o, h, l, c))

# Haftalik siniflandirma
weekly_signals = []  # (week_idx, signal_type)
for i in range(1, len(weekly_candles)):
    _, prev_o, prev_h, prev_l, prev_c = weekly_candles[i-1]
    start, curr_o, curr_h, curr_l, curr_c = weekly_candles[i]

    swept_high = curr_h > prev_h
    swept_low = curr_l < prev_l
    is_green = curr_c > curr_o
    is_red = curr_c < curr_o

    if swept_high and not swept_low:
        if curr_c > prev_h:
            sig = "high_cont"
        elif is_red:
            sig = "high_rev"
        else:
            sig = "high_ambig"
    elif swept_low and not swept_high:
        if curr_c < prev_l:
            sig = "low_cont"
        elif is_green:
            sig = "low_rev"
        else:
            sig = "low_ambig"
    elif swept_high and swept_low:
        if curr_c > prev_h:
            sig = "high_cont"
        elif curr_c < prev_l:
            sig = "low_cont"
        elif is_red:
            sig = "high_rev"
        elif is_green:
            sig = "low_rev"
        else:
            sig = "inside"
    else:
        sig = "inside"

    weekly_signals.append((i, sig, start))

# Haftalik dagilim
from collections import Counter
wcounts = Counter(s for _, s, _ in weekly_signals)
print(f"Haftalik mum sayisi: {len(weekly_signals)}")
for k in ["high_cont", "high_rev", "high_ambig", "low_cont", "low_rev", "low_ambig", "inside"]:
    print(f"  {k:15s}: {wcounts.get(k, 0):>4d} ({wcounts.get(k, 0)/len(weekly_signals)*100:.1f}%)")
print()

# ══════════════════════════════════════════════════════════════
# 2) Gunluk mumlari olustur ve siniflandir
# ══════════════════════════════════════════════════════════════

daily_candles = []
for d in range(n_days):
    start = d * BP_1D
    end = start + BP_1D
    o = eth_perp['c'][start]
    h = max(eth_perp['h'][start:end])
    l = min(eth_perp['l'][start:end])
    c = eth_perp['c'][end - 1]
    daily_candles.append((start, o, h, l, c))

daily_signals = []  # (day_idx, signal_type, start_bar)
for i in range(1, len(daily_candles)):
    _, prev_o, prev_h, prev_l, prev_c = daily_candles[i-1]
    start, curr_o, curr_h, curr_l, curr_c = daily_candles[i]

    swept_high = curr_h > prev_h
    swept_low = curr_l < prev_l
    is_green = curr_c > curr_o
    is_red = curr_c < curr_o

    if swept_high and not swept_low:
        if curr_c > prev_h:
            sig = "high_cont"
        elif is_red:
            sig = "high_rev"
        else:
            sig = "high_ambig"
    elif swept_low and not swept_high:
        if curr_c < prev_l:
            sig = "low_cont"
        elif is_green:
            sig = "low_rev"
        else:
            sig = "low_ambig"
    elif swept_high and swept_low:
        if curr_c > prev_h:
            sig = "high_cont"
        elif curr_c < prev_l:
            sig = "low_cont"
        elif is_red:
            sig = "high_rev"
        elif is_green:
            sig = "low_rev"
        else:
            sig = "inside"
    else:
        sig = "inside"

    daily_signals.append((i, sig, start))

# ══════════════════════════════════════════════════════════════
# 3) Her gunluk mum icin hangi haftadaki oldugunu bul
# ══════════════════════════════════════════════════════════════

# Onceki haftanin sinyalini bul (t-1 hafta → bu haftadaki gunluk mumlar)
# Hafta i'nin icindeki gunler: start_bar in [i*BP_1W, (i+1)*BP_1W)

def get_prev_week_signal(day_start_bar):
    """Bu gunun icinde oldugu haftanin ONCEKI haftasinin sinyalini dondur"""
    week_idx = day_start_bar // BP_1W
    # Onceki haftanin sinyalini bul
    for wi, sig, ws in weekly_signals:
        if wi == week_idx:
            return sig
    return None

def get_current_week_signal(day_start_bar):
    """Bu gunun icinde oldugu haftanin sinyalini dondur (hafta kapandiktan sonra bilinir)"""
    week_idx = day_start_bar // BP_1W
    # Bu haftanin sinyalini bul
    for wi, sig, ws in weekly_signals:
        if wi == week_idx:
            return sig
    return None

# ══════════════════════════════════════════════════════════════
# 4) Analiz: Onceki hafta sinyali → bu haftadaki gunluk sonuclar
# ══════════════════════════════════════════════════════════════

print("=" * 120)
print("ANALIZ 1: ONCEKI HAFTANIN SINYAL TIPI → BU HAFTADAKI GUNLUK SWEEP SONUCLARI")
print("(t-1 hafta kapandi, t haftasindaki gunluk mumlar nasil kapaniyor?)")
print("=" * 120)
print()

# Onceki hafta sinyal gruplari
groups = {}
for di, dsig, dstart in daily_signals:
    prev_w_sig = get_prev_week_signal(dstart)
    if prev_w_sig is None:
        continue
    if prev_w_sig not in groups:
        groups[prev_w_sig] = []
    groups[prev_w_sig].append(dsig)

# Haftalik sinyal bazinda gunluk sonuclari goster
for wsig in ["high_cont", "high_rev", "low_cont", "low_rev", "high_ambig", "low_ambig", "inside"]:
    if wsig not in groups:
        continue
    daily_sigs = groups[wsig]
    total = len(daily_sigs)
    dc = Counter(daily_sigs)

    # High sweep olan gunler
    h_total = dc.get("high_cont", 0) + dc.get("high_rev", 0) + dc.get("high_ambig", 0)
    h_cont = dc.get("high_cont", 0)
    h_rev = dc.get("high_rev", 0)
    h_cont_rate = h_cont / h_total * 100 if h_total > 0 else 0

    # Low sweep olan gunler
    l_total = dc.get("low_cont", 0) + dc.get("low_rev", 0) + dc.get("low_ambig", 0)
    l_cont = dc.get("low_cont", 0)
    l_rev = dc.get("low_rev", 0)
    l_cont_rate = l_cont / l_total * 100 if l_total > 0 else 0

    print(f"  Onceki hafta: {wsig:15s} (N={total} gun)")
    print(f"    Gunluk sinyal dagilimi:")
    for k in ["high_cont", "high_rev", "high_ambig", "low_cont", "low_rev", "low_ambig", "inside"]:
        if dc.get(k, 0) > 0:
            print(f"      {k:15s}: {dc[k]:>4d} ({dc[k]/total*100:.1f}%)")
    if h_total > 0:
        print(f"    High sweep: cont={h_cont_rate:.1f}% ({h_cont}/{h_total}) | rev={h_rev/h_total*100:.1f}%")
    if l_total > 0:
        print(f"    Low  sweep: cont={l_cont_rate:.1f}% ({l_cont}/{l_total}) | rev={l_rev/l_total*100:.1f}%")
    print()

# ══════════════════════════════════════════════════════════════
# 5) Analiz 2: Ayni haftanin sinyali → gunluk sonuclar
# ══════════════════════════════════════════════════════════════

print("=" * 120)
print("ANALIZ 2: AYNI HAFTANIN SINYAL TIPI → O HAFTADAKI GUNLUK SWEEP SONUCLARI")
print("(Hafta kapandiktan sonra geriye bakis — hafta ici nasil gitmis?)")
print("=" * 120)
print()

groups2 = {}
for di, dsig, dstart in daily_signals:
    curr_w_sig = get_current_week_signal(dstart)
    if curr_w_sig is None:
        continue
    if curr_w_sig not in groups2:
        groups2[curr_w_sig] = []
    groups2[curr_w_sig].append(dsig)

for wsig in ["high_cont", "high_rev", "low_cont", "low_rev", "high_ambig", "low_ambig", "inside"]:
    if wsig not in groups2:
        continue
    daily_sigs = groups2[wsig]
    total = len(daily_sigs)
    dc = Counter(daily_sigs)

    h_total = dc.get("high_cont", 0) + dc.get("high_rev", 0) + dc.get("high_ambig", 0)
    h_cont = dc.get("high_cont", 0)
    h_rev = dc.get("high_rev", 0)
    h_cont_rate = h_cont / h_total * 100 if h_total > 0 else 0

    l_total = dc.get("low_cont", 0) + dc.get("low_rev", 0) + dc.get("low_ambig", 0)
    l_cont = dc.get("low_cont", 0)
    l_rev = dc.get("low_rev", 0)
    l_cont_rate = l_cont / l_total * 100 if l_total > 0 else 0

    print(f"  Bu hafta: {wsig:15s} (N={total} gun)")
    if h_total > 0:
        print(f"    High sweep gunluk cont: {h_cont_rate:.1f}% ({h_cont}/{h_total})")
    if l_total > 0:
        print(f"    Low  sweep gunluk cont: {l_cont_rate:.1f}% ({l_cont}/{l_total})")
    print()

# ══════════════════════════════════════════════════════════════
# 6) Ozet karsilastirma tablosu
# ══════════════════════════════════════════════════════════════

print("=" * 120)
print("OZET KARSILASTIRMA")
print("=" * 120)

# Genel gunluk base rate
all_daily = [s for _, s, _ in daily_signals]
all_dc = Counter(all_daily)
all_h = all_dc.get("high_cont",0) + all_dc.get("high_rev",0) + all_dc.get("high_ambig",0)
all_h_cont = all_dc.get("high_cont",0)
all_l = all_dc.get("low_cont",0) + all_dc.get("low_rev",0) + all_dc.get("low_ambig",0)
all_l_cont = all_dc.get("low_cont",0)

print(f"\n  Genel gunluk base rate:")
print(f"    High sweep cont: {all_h_cont/all_h*100:.1f}% ({all_h_cont}/{all_h})")
print(f"    Low  sweep cont: {all_l_cont/all_l*100:.1f}% ({all_l_cont}/{all_l})")

print(f"\n  ONCEKI HAFTA SINYALI → GUNLUK HIGH SWEEP CONT RATE:")
print(f"  {'Onceki hafta':>20s} | {'H Cont%':>8s} | {'H Total':>7s} | {'L Cont%':>8s} | {'L Total':>7s}")
print(f"  {'-'*60}")
print(f"  {'GENEL (base)':>20s} | {all_h_cont/all_h*100:>7.1f}% | {all_h:>7d} | {all_l_cont/all_l*100:>7.1f}% | {all_l:>7d}")

for wsig in ["high_cont", "high_rev", "low_cont", "low_rev", "inside"]:
    if wsig not in groups:
        continue
    daily_sigs = groups[wsig]
    dc = Counter(daily_sigs)
    ht = dc.get("high_cont",0)+dc.get("high_rev",0)+dc.get("high_ambig",0)
    hc = dc.get("high_cont",0)
    lt = dc.get("low_cont",0)+dc.get("low_rev",0)+dc.get("low_ambig",0)
    lc = dc.get("low_cont",0)
    hcr = hc/ht*100 if ht>0 else 0
    lcr = lc/lt*100 if lt>0 else 0
    print(f"  {wsig:>20s} | {hcr:>7.1f}% | {ht:>7d} | {lcr:>7.1f}% | {lt:>7d}")

print(f"\n  AYNI HAFTA SINYALI → GUNLUK HIGH SWEEP CONT RATE:")
print(f"  {'Bu hafta':>20s} | {'H Cont%':>8s} | {'H Total':>7s} | {'L Cont%':>8s} | {'L Total':>7s}")
print(f"  {'-'*60}")
print(f"  {'GENEL (base)':>20s} | {all_h_cont/all_h*100:>7.1f}% | {all_h:>7d} | {all_l_cont/all_l*100:>7.1f}% | {all_l:>7d}")

for wsig in ["high_cont", "high_rev", "low_cont", "low_rev", "inside"]:
    if wsig not in groups2:
        continue
    daily_sigs = groups2[wsig]
    dc = Counter(daily_sigs)
    ht = dc.get("high_cont",0)+dc.get("high_rev",0)+dc.get("high_ambig",0)
    hc = dc.get("high_cont",0)
    lt = dc.get("low_cont",0)+dc.get("low_rev",0)+dc.get("low_ambig",0)
    lc = dc.get("low_cont",0)
    hcr = hc/ht*100 if ht>0 else 0
    lcr = lc/lt*100 if lt>0 else 0
    print(f"  {wsig:>20s} | {hcr:>7.1f}% | {ht:>7d} | {lcr:>7.1f}% | {lt:>7d}")

print()
