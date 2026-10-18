//! Basic same-rate kernels.

use crate::error::Result;
use crate::types::ComplexSample;

/// Multiplies a complex signal by a real gain.
///
/// This is the template for every later kernel: a trait generic over the sample
/// type, implemented by `Fallback`, `Reference` and the facade `Sdrk`.
///
/// ```
/// use sdrk_core::{kernels::basic::Scale, Fallback, C32};
/// let input = [C32::new(1.0, -2.0)];
/// let mut output = [C32::default()];
/// Fallback.scale(&input, &mut output, 0.5)?;
/// assert_eq!(output[0], C32::new(0.5, -1.0));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait Scale<T: ComplexSample> {
    /// `output[n] = input[n] * gain` (real gain). The lengths must be equal.
    fn scale(&self, input: &[T], output: &mut [T], gain: T::Real) -> Result<()>;
}
