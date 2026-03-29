/// Bias Engine — Step 9b: Final Bias Computation (Spec Section 15)
///
/// Combines state_bias + sweep_bias with alignment check and regime shift penalty.

/// Directional alignment between state and sweep bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alignment {
    Aligned,
    Conflicting,
    SweepNeutral,
}

/// Bias direction classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasDirection {
    Bullish,
    Bearish,
    Neutral,
}

/// Bias strength classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasStrength {
    Strong,
    Weak,
    Neutral,
}

/// Full per-bar bias output.
#[derive(Clone, Debug)]
pub struct FinalBiasOutput {
    /// Final bias after all adjustments (range ±0.50)
    pub final_bias: f64,
    pub direction: BiasDirection,
    pub strength: BiasStrength,
    pub confidence: f64,

    /// Component breakdown
    pub state_bias: f64,
    pub sweep_bias: f64,
    pub alignment: Alignment,
    pub regime_shift_penalty: bool,
}

/// Classify bias into direction + strength (Section 15.3).
fn classify_bias(bias: f64) -> (BiasDirection, BiasStrength) {
    if bias > 0.10 {
        (BiasDirection::Bullish, BiasStrength::Strong)
    } else if bias > 0.001 {
        (BiasDirection::Bullish, BiasStrength::Weak)
    } else if bias >= -0.001 {
        (BiasDirection::Neutral, BiasStrength::Neutral)
    } else if bias >= -0.10 {
        (BiasDirection::Bearish, BiasStrength::Weak)
    } else {
        (BiasDirection::Bearish, BiasStrength::Strong)
    }
}

/// Combine state bias and sweep bias into final bias (Section 15.2).
pub fn compute_final_bias(
    state_bias: f64,
    sweep_bias: f64,
    regime_shift: bool,
    confidence: f64,
) -> FinalBiasOutput {
    let (combined, alignment) = if sweep_bias.abs() < 1e-10 {
        // No sweep signal → pure state bias
        (state_bias, Alignment::SweepNeutral)
    } else {
        let aligned = (state_bias >= 0.0 && sweep_bias >= 0.0)
            || (state_bias < 0.0 && sweep_bias < 0.0);

        if aligned {
            // Same direction → reinforce (sweep contributes 30%)
            // Sweep is not robustness-validated, so give it less weight
            (state_bias + sweep_bias * 0.30, Alignment::Aligned)
        } else {
            // Conflicting → state always wins (it's validated, sweep is not)
            // But reduce confidence due to disagreement
            (state_bias * 0.70, Alignment::Conflicting)
        }
    };

    // Regime shift penalty
    let penalized = if regime_shift {
        combined * 0.50
    } else {
        combined
    };

    // Final clamp
    let final_bias = penalized.clamp(-0.50, 0.50);
    let (direction, strength) = classify_bias(final_bias);

    FinalBiasOutput {
        final_bias,
        direction,
        strength,
        confidence,
        state_bias,
        sweep_bias,
        alignment,
        regime_shift_penalty: regime_shift,
    }
}
