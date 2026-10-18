//! Helper macros for kernel implementations.

/// Generates the body of a same-rate, stateless, per-sample kernel.
///
/// It validates `output.len() == input.len()` (else `Error::LengthMismatch`) and
/// then applies `$f` to every input sample. Use it as the starting point for
/// fallback kernels and for later SIMD "process one" helpers.
///
/// ```
/// use sdrk_core::{map_kernel, Result};
/// fn double(input: &[f32], output: &mut [f32]) -> Result<()> {
///     map_kernel!(input, output, |x: f32| x * 2.0)
/// }
/// let mut out = [0.0; 2];
/// double(&[1.0, 2.0], &mut out)?;
/// assert_eq!(out, [2.0, 4.0]);
/// assert!(double(&[1.0], &mut out).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[macro_export]
macro_rules! map_kernel {
    ($input:expr, $output:expr, $f:expr) => {{
        let input = $input;
        let output = $output;
        if output.len() != input.len() {
            return ::core::result::Result::Err($crate::Error::LengthMismatch {
                expected: input.len(),
                got: output.len(),
            });
        }
        let f = $f;
        for (o, &i) in output.iter_mut().zip(input) {
            *o = f(i);
        }
        ::core::result::Result::Ok(())
    }};
}
