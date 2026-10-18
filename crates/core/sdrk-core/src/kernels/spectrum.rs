//! Kernels of the spectrum display path: windowing, magnitude squared and power in dB.

use crate::error::Result;
use crate::types::{ComplexSample, RealSample};

/// Multiplies a complex signal by a real window.
///
/// ```
/// use sdrk_core::{kernels::spectrum::Window, Fallback, C32};
/// let input = [C32::new(2.0, 4.0), C32::new(1.0, 1.0)];
/// let mut output = [C32::default(); 2];
/// Fallback.window(&input, &mut output, &[0.5, 0.0])?;
/// assert_eq!(output, [C32::new(1.0, 2.0), C32::new(0.0, 0.0)]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Window<T: ComplexSample> {
    /// `output[n] = input[n] * w[n]`. `w.len()` and `output.len()` must equal `input.len()`.
    /// `f32` is a plain multiply per component; `Q15` uses `q15_mul`.
    fn window(&self, input: &[T], output: &mut [T], w: &[T::Real]) -> Result<()>;
}

/// Squared magnitude of complex samples.
///
/// ```
/// use sdrk_core::{kernels::spectrum::MagSq, Fallback, C32};
/// let mut output = [0.0f32; 1];
/// Fallback.mag_sq(&[C32::new(3.0, 4.0)], &mut output)?;
/// assert_eq!(output, [25.0]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait MagSq<T: ComplexSample> {
    /// `f32`: `re * re + im * im` (no FMA). `Q15`: `sat16(round_shift(re^2 + im^2, 15))`
    /// in `i64`; a full-scale sample saturates. The lengths must be equal.
    fn mag_sq(&self, input: &[T], output: &mut [T::Real]) -> Result<()>;
}

/// Linear power to decibels.
///
/// ```
/// use sdrk_core::{kernels::spectrum::PowerDb, Fallback};
/// let mut output = [0.0f32; 3];
/// Fallback.power_db(&[1.0f32, 0.01, f32::NAN], &mut output, 0.0)?;
/// assert_eq!(output[0], 0.0);
/// assert!((output[1] + 20.0).abs() < 1e-4);
/// assert!((output[2] + 200.0).abs() < 1e-3); // NaN is treated as 1e-20
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait PowerDb<T: RealSample> {
    /// `output[n] = 10 * log10(max(p, 1e-20)) + offset_db` with `p` the input as linear
    /// power (`Q15`: `raw / 32768`). Negative, zero and NaN input count as `1e-20`.
    /// The lengths must be equal.
    fn power_db(&self, input: &[T], output: &mut [f32], offset_db: f32) -> Result<()>;
}
