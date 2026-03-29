/// Bias Engine — Step 9a: Confidence Score (Spec Section 14)
///
/// confidence = sample_conf × ci_conf × stability_conf × regime_conf
///
/// Each component is [0, 1]. Confidence does NOT modify bias directly —
/// it's a separate signal for downstream position-sizing / filtering.

/// Compute confidence score for a bar's bias.
///
/// `n_total`          — state sample size
/// `ci_width`         — 95% CI full width (2 × ci_half)
/// `noise_stability`  — from robustness (0–1)
/// `regime_match`     — true if current regime has sufficient data (N ≥ 30)
pub fn compute_confidence(
    n_total: u32,
    ci_width: f64,
    noise_stability: f64,
    regime_match: bool,
) -> f64 {
    // Sample confidence: saturates at N=500
    let sample_conf = (n_total as f64 / 500.0).min(1.0);

    // CI confidence: CI_width=0 → 1.0, CI_width=0.20 → 0.0
    let ci_conf = (1.0 - ci_width / 0.20).clamp(0.0, 1.0);

    // Stability confidence: direct from noise injection test
    let stability_conf = noise_stability.clamp(0.0, 1.0);

    // Regime confidence: 1.0 if regime has data, 0.50 otherwise
    let regime_conf = if regime_match { 1.0 } else { 0.50 };

    sample_conf * ci_conf * stability_conf * regime_conf
}

/// Compute confidence for a fallback (baseline) bar — lower confidence.
pub fn compute_fallback_confidence() -> f64 {
    // No validated state matched: very low confidence
    // sample_conf=0, so product would be 0. Use a floor of 0.10
    0.10
}
