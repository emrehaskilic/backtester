/// Bias Engine — Step 6: Calibration via Isotonic Regression (Spec Section 13)
///
/// Pool Adjacent Violators Algorithm (PAVA):
///   1. Sort (predicted_prob, actual_outcome) pairs by predicted
///   2. Walk forward; whenever a later group has lower mean than the previous, merge them
///   3. Result: a monotonically non-decreasing step function
///
/// At inference: linear interpolation between breakpoints.
/// If the calibrator is invalid (too few data points or degenerate), it acts as identity.

/// A fitted isotonic regression calibrator.
#[derive(Clone, Debug)]
pub struct IsotonicCalibrator {
    /// Sorted breakpoints: (predicted_prob, calibrated_prob)
    pub points: Vec<(f64, f64)>,
    /// Whether this calibrator is valid (enough data + non-degenerate)
    pub valid: bool,
}

/// Minimum pairs needed for a valid calibrator.
const MIN_CALIBRATION_PAIRS: usize = 500;
/// Minimum predicted range needed (avoids degenerate flat input)
const MIN_PREDICTED_RANGE: f64 = 0.02;

impl IsotonicCalibrator {
    /// Create an identity calibrator (passthrough).
    pub fn identity() -> Self {
        IsotonicCalibrator {
            points: vec![(0.0, 0.0), (1.0, 1.0)],
            valid: false,
        }
    }

    /// Fit the calibrator using PAVA.
    ///
    /// `predicted` — smoothed probabilities (one per bar).
    /// `outcomes`  — binary outcomes (1=bull, 0=bear, 255=skip).
    /// Only bars where outcome in {0,1} are used.
    pub fn fit(predicted: &[f64], outcomes: &[u8]) -> Self {
        // Collect valid (predicted, actual) pairs
        let mut pairs: Vec<(f64, f64)> = predicted
            .iter()
            .zip(outcomes.iter())
            .filter(|(_, &o)| o != 255)
            .map(|(&p, &o)| (p, o as f64))
            .collect();

        // Guard: too few pairs
        if pairs.len() < MIN_CALIBRATION_PAIRS {
            return Self::identity();
        }

        // Sort by predicted probability
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Guard: degenerate range
        let pred_range = pairs.last().unwrap().0 - pairs.first().unwrap().0;
        if pred_range < MIN_PREDICTED_RANGE {
            return Self::identity();
        }

        // PAVA: pool adjacent violators
        // Each block: (sum_predicted, sum_actual, count)
        let mut blocks: Vec<(f64, f64, usize)> = pairs
            .iter()
            .map(|&(p, a)| (p, a, 1))
            .collect();

        let mut merged = true;
        while merged {
            merged = false;
            let mut new_blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(blocks.len());

            for &(sp, sa, sc) in &blocks {
                if let Some(last) = new_blocks.last_mut() {
                    let prev_mean = last.1 / last.2 as f64;
                    let curr_mean = sa / sc as f64;
                    if curr_mean < prev_mean {
                        // Violator — merge
                        last.0 += sp;
                        last.1 += sa;
                        last.2 += sc;
                        merged = true;
                        continue;
                    }
                }
                new_blocks.push((sp, sa, sc));
            }
            blocks = new_blocks;
        }

        // Convert blocks to breakpoints: (mean_predicted, mean_actual)
        let points: Vec<(f64, f64)> = blocks
            .iter()
            .map(|&(sp, sa, sc)| (sp / sc as f64, sa / sc as f64))
            .collect();

        // Validate: calibrator should have at least 3 breakpoints
        let valid = points.len() >= 3;

        IsotonicCalibrator { points, valid }
    }

    /// Transform a single predicted probability into a calibrated probability.
    /// Uses linear interpolation between breakpoints.
    /// If calibrator is invalid, returns input unchanged.
    pub fn transform(&self, prob: f64) -> f64 {
        if !self.valid || self.points.is_empty() {
            return prob;
        }
        if self.points.len() == 1 {
            return self.points[0].1;
        }

        // Clamp to range
        if prob <= self.points[0].0 {
            return self.points[0].1;
        }
        if prob >= self.points.last().unwrap().0 {
            return self.points.last().unwrap().1;
        }

        // Binary search for the segment
        let mut lo = 0;
        let mut hi = self.points.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.points[mid].0 <= prob {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let (x0, y0) = self.points[lo];
        let (x1, y1) = self.points[hi];
        let dx = x1 - x0;
        if dx.abs() < 1e-15 {
            return (y0 + y1) * 0.5;
        }
        let t = (prob - x0) / dx;
        y0 + t * (y1 - y0)
    }

    /// Transform a slice of probabilities.
    pub fn transform_all(&self, probs: &[f64]) -> Vec<f64> {
        probs.iter().map(|&p| self.transform(p)).collect()
    }
}
