"""
Bias Engine — Step 1-2 Test Script
Verifies: feature computation, HTF aggregation, quantization, state enumeration.
"""
import numpy as np
import pandas as pd
import time
import sys
import os

# Build the Rust module first
print("=" * 60)
print("BIAS ENGINE — STEP 1-2 VERIFICATION")
print("=" * 60)

# Add the rust_engine build path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rust_engine"))

try:
    import rust_engine
except ImportError:
    print("\n[ERROR] rust_engine not found. Build it first:")
    print("  cd rust_engine && maturin develop --release")
    sys.exit(1)

# -- Load data --
print("\n[1] Loading data...")
df = pd.read_parquet("data/ETHUSDT_5m_cvd_oi_11mo.parquet")
print(f"    Bars: {len(df):,}")
print(f"    Columns: {list(df.columns)}")
print(f"    Date range: {pd.to_datetime(df['open_time'].iloc[0], unit='ms')} -> "
      f"{pd.to_datetime(df['open_time'].iloc[-1], unit='ms')}")

# Fill OI NaN with forward fill then 0
oi = df["open_interest"].fillna(method="ffill").fillna(0.0).values.astype(np.float64)

# -- Run Rust engine --
print("\n[2] Running bias_engine_step1_2...")
t0 = time.time()
result = rust_engine.bias_engine_step1_2(
    df["open_time"].values.astype(np.uint64),
    df["open"].values.astype(np.float64),
    df["high"].values.astype(np.float64),
    df["low"].values.astype(np.float64),
    df["close"].values.astype(np.float64),
    df["buy_vol"].values.astype(np.float64),
    df["sell_vol"].values.astype(np.float64),
    oi,
)
elapsed = time.time() - t0
print(f"    Completed in {elapsed:.2f}s")

# -- Summary --
print("\n[3] Summary:")
print(f"    Total bars:      {result['n_bars']:,}")
print(f"    Valid bars:      {result['n_valid_bars']:,} "
      f"({result['n_valid_bars']/result['n_bars']*100:.1f}%)")
print(f"    Quantized bars:  {result['n_quantized_bars']:,} "
      f"({result['n_quantized_bars']/result['n_bars']*100:.1f}%)")
print(f"    Warmup bars:     {result['warmup_bars']:,}")

# -- HTF bar counts --
print("\n[4] HTF Bar Counts:")
for tf, count in result["htf_bar_counts"].items():
    print(f"    {tf}: {count:,} bars")

# -- Feature verification --
print("\n[5] Feature Spot-Check (bar 3000):")
features = result["features"]
idx = 3000
for name, arr in features.items():
    val = arr[idx]
    nan_count = np.isnan(arr).sum()
    valid_count = len(arr) - nan_count
    print(f"    {name:20s}: val={val:+.4f}  valid={valid_count:,}  NaN={nan_count:,}")

# -- Quintile distribution --
print("\n[6] Quintile Distribution (should be ~20% each):")
dist = result["quintile_distribution_pct"]
all_ok = True
for name, pcts in dist.items():
    deviation = max(abs(p - 20.0) for p in pcts)
    status = "OK" if deviation < 3.0 else "WARN"
    if status == "WARN":
        all_ok = False
    print(f"    {name:20s}: Q1={pcts[0]:.1f}% Q2={pcts[1]:.1f}% "
          f"Q3={pcts[2]:.1f}% Q4={pcts[3]:.1f}% Q5={pcts[4]:.1f}%  [{status}]")
if all_ok:
    print("    OK: All features within ±3% of target 20%")

# -- State stats --
print("\n[7] State Statistics:")
print(f"    Total states with data: {result['total_states_with_data']:,}")
print(f"    Depth 1 active: {result['depth1_active']:,} / 35")
print(f"    Depth 2 active: {result['depth2_active']:,} / 525")
print(f"    Depth 3 active: {result['depth3_active']:,} / 4,375")

# State count distribution
print("\n[8] State Count Distribution:")
for depth in [1, 2, 3]:
    key = f"depth{depth}_count_above"
    d = result[key]
    parts = "  ".join(f"{k}:{v}" for k, v in d.items())
    print(f"    Depth {depth}: {parts}")

# -- Top states --
print("\n[9] Top 15 States by Frequency:")
print(f"    {'State':<40s} {'Count':>7s} {'Depth':>5s}")
print(f"    {'-'*40} {'-'*7} {'-'*5}")
for s in result["top_states"][:15]:
    print(f"    {s['state']:<40s} {s['count']:>7,} {s['depth']:>5}")

# -- Sanity checks --
print("\n[10] Sanity Checks:")
checks = []

# Check 1: Quantized bars should be roughly n_bars - warmup
expected_min = result["n_bars"] - result["warmup_bars"] - 500  # some margin
c1 = result["n_quantized_bars"] >= expected_min
checks.append(("Quantized bars >= expected", c1))

# Check 2: All depth-1 states should be active (35/35)
c2 = result["depth1_active"] == 35
checks.append(("All 35 depth-1 states active", c2))

# Check 3: Quintile distributions should be roughly uniform
c3 = all(
    all(abs(p - 20.0) < 5.0 for p in pcts)
    for pcts in result["quintile_distribution_pct"].values()
)
checks.append(("Quintile distributions ~uniform", c3))

# Check 4: HTF counts make sense
htf = result["htf_bar_counts"]
c4 = htf.get("1H", 0) > htf.get("1D", 0) > 0
checks.append(("HTF: 1H > 1D > 0", c4))

for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"    [{status}] {name}")

all_pass = all(ok for _, ok in checks)
print(f"\n{'='*60}")
if all_pass:
    print("ALL CHECKS PASSED — Step 1-2 verified.")
else:
    print("SOME CHECKS FAILED — review output above.")
print(f"{'='*60}")
