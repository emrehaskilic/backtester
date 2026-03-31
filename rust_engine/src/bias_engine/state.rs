/// Bias Engine — State Definition, Enumeration & Matching (Section 5 of spec)
///
/// A **state** is a combination of (feature_index, quintile) pairs.
/// 16 features × 7 quintile bins. Three depths:
///   Depth 1 — single feature     : C(16,1) × 7^1 = 112
///   Depth 2 — feature pair       : C(16,2) × 7^2 = 5 880
///   Depth 3 — feature triple     : C(16,3) × 7^3 = 192 080
///   Total                                          = 198 072
///
/// Compact encoding in u32:
///   bits 24-25 : depth (1, 2, or 3)
///   bits  0-7  : slot 0  →  feat(4 bits) | quint(4 bits)
///   bits  8-15 : slot 1
///   bits 16-23 : slot 2

use std::collections::HashMap;

use super::features::N_FEATURES;

/// Compact state identifier.
pub type StateKey = u32;

const N_QUINTILES: usize = 7;

// ── Encoding / Decoding ──

/// Encode sorted (feature, quintile) pairs into a compact u32 key.
/// Pairs **must** be sorted ascending by feature index.
#[inline]
pub fn encode_state(pairs: &[(usize, u8)]) -> StateKey {
    let depth = pairs.len() as u32;
    let mut key: u32 = depth << 24;
    for (slot, &(feat, quint)) in pairs.iter().enumerate() {
        key |= ((feat as u32) << 4 | quint as u32) << (slot * 8);
    }
    key
}

/// Decode a state key back to (feature_index, quintile) pairs.
pub fn decode_state(key: StateKey) -> Vec<(usize, u8)> {
    let depth = (key >> 24) as usize;
    let mut pairs = Vec::with_capacity(depth);
    for slot in 0..depth {
        let bits = (key >> (slot * 8)) & 0xFF;
        let feat = (bits >> 4) as usize;
        let quint = (bits & 0x0F) as u8;
        pairs.push((feat, quint));
    }
    pairs
}

/// Depth of a state (1, 2, or 3).
#[inline]
pub fn state_depth(key: StateKey) -> u32 {
    key >> 24
}

/// Human-readable label, e.g. "cvd_micro=Q5,vol_micro=Q5"
pub fn state_to_string(key: StateKey) -> String {
    let short = [
        "cvd_mi", "cvd_ma", "oi_chg", "vol_mi", "vol_ma", "imb_sm", "atr_pc",
        "vwap", "mom", "wick", "vp_div", "oi_vol", "autocr",
        "hour", "mtf4h", "mtf_d",
    ];
    let pairs = decode_state(key);
    pairs
        .iter()
        .map(|&(f, q)| format!("{}=Q{}", short[f], q))
        .collect::<Vec<_>>()
        .join(",")
}

// ── Enumeration ──

/// Total number of states across all depths (16 features × 7 quintiles).
/// Depth 1: C(16,1) × 7   = 112
/// Depth 2: C(16,2) × 49  = 5_880
/// Depth 3: C(16,3) × 343 = 192_080
/// Total                   = 198_072
pub const TOTAL_STATES: usize = 112 + 5_880 + 192_080; // 198_072

/// Generate every possible state key.
pub fn enumerate_all_states() -> Vec<StateKey> {
    let mut states = Vec::with_capacity(TOTAL_STATES);

    // Depth 1
    for f in 0..N_FEATURES {
        for q in 1..=N_QUINTILES as u8 {
            states.push(encode_state(&[(f, q)]));
        }
    }

    // Depth 2
    for f1 in 0..N_FEATURES {
        for f2 in (f1 + 1)..N_FEATURES {
            for q1 in 1..=N_QUINTILES as u8 {
                for q2 in 1..=N_QUINTILES as u8 {
                    states.push(encode_state(&[(f1, q1), (f2, q2)]));
                }
            }
        }
    }

    // Depth 3
    for f1 in 0..N_FEATURES {
        for f2 in (f1 + 1)..N_FEATURES {
            for f3 in (f2 + 1)..N_FEATURES {
                for q1 in 1..=N_QUINTILES as u8 {
                    for q2 in 1..=N_QUINTILES as u8 {
                        for q3 in 1..=N_QUINTILES as u8 {
                            states.push(encode_state(&[(f1, q1), (f2, q2), (f3, q3)]));
                        }
                    }
                }
            }
        }
    }

    debug_assert_eq!(states.len(), TOTAL_STATES);
    states
}

// ── Matching ──

/// For a bar with 7 quintile values, emit all state keys it belongs to.
///
/// Returns up to 7 + 21 + 35 = 63 keys.
/// If any quintile is 0 (invalid), returns empty.
pub fn match_bar_states(quintiles: &[u8; N_FEATURES]) -> Vec<StateKey> {
    for &q in quintiles.iter() {
        if q == 0 {
            return Vec::new();
        }
    }

    let mut matches = Vec::with_capacity(63);

    // Depth 1: 7 states
    for f in 0..N_FEATURES {
        matches.push(encode_state(&[(f, quintiles[f])]));
    }

    // Depth 2: 21 states
    for f1 in 0..N_FEATURES {
        for f2 in (f1 + 1)..N_FEATURES {
            matches.push(encode_state(&[(f1, quintiles[f1]), (f2, quintiles[f2])]));
        }
    }

    // Depth 3: 35 states
    for f1 in 0..N_FEATURES {
        for f2 in (f1 + 1)..N_FEATURES {
            for f3 in (f2 + 1)..N_FEATURES {
                matches.push(encode_state(&[
                    (f1, quintiles[f1]),
                    (f2, quintiles[f2]),
                    (f3, quintiles[f3]),
                ]));
            }
        }
    }

    matches
}

// ── Counting ──

/// Count how many bars match each state.
///
/// `all_quintiles` — 7 arrays of length n_bars (from quantize_all).
/// Returns HashMap<StateKey, count>.
pub fn count_states(all_quintiles: &[Vec<u8>], n_bars: usize) -> HashMap<StateKey, u32> {
    let mut counts: HashMap<StateKey, u32> = HashMap::with_capacity(TOTAL_STATES);

    for i in 0..n_bars {
        // Build quintile tuple for this bar
        let mut q = [0u8; N_FEATURES];
        let mut valid = true;
        for f in 0..N_FEATURES {
            q[f] = all_quintiles[f][i];
            if q[f] == 0 {
                valid = false;
                break;
            }
        }
        if !valid {
            continue;
        }

        // Accumulate counts for all 63 matching states
        let keys = match_bar_states(&q);
        for key in keys {
            *counts.entry(key).or_insert(0) += 1;
        }
    }

    counts
}
