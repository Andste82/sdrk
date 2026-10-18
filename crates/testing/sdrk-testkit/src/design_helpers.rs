//! Filter design helpers for tests.
//!
//! These exist only until the real designer of M09 is available; tests use them to build the
//! taps of the end-to-end chains.

use core::f64::consts::PI;

/// Windowed-sinc low-pass with a symmetric Hann window, normalised to a DC gain of 1.
///
/// `cutoff_norm` is the cutoff divided by the sample rate in cycles per sample
/// (`0 < c < 0.5`; this is *not* the `SciPy` Nyquist convention). With `N = taps.len()` and
/// `m = (N - 1) / 2`:
///
/// ```text
/// h[n] = 2 c sinc(2 c (n - m)) w[n],   sinc(x) = sin(pi x) / (pi x),  sinc(0) = 1
/// w[n] = 0.5 - 0.5 cos(2 pi n / (N - 1))
/// ```
///
/// and finally `h` is divided by `sum h`. A single tap is `[1.0]`. Convert to `f32` with `as f32`
/// and to Q15 with the rule of `sdrk_core::num::q15_from_f64`.
///
/// ```
/// let mut h = [0.0f64; 127];
/// sdrk_testkit::design_helpers::lowpass_hann(100e3 / 2.4e6, &mut h);
/// assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert_eq!(h[0], h[126]); // symmetric
/// ```
pub fn lowpass_hann(cutoff_norm: f64, taps: &mut [f64]) {
    let n = taps.len();
    if n == 0 {
        return;
    }
    let m = (n as f64 - 1.0) / 2.0;
    for (i, h) in taps.iter_mut().enumerate() {
        let x = 2.0 * cutoff_norm * (i as f64 - m);
        let sinc = if x == 0.0 {
            1.0
        } else {
            libm::sin(PI * x) / (PI * x)
        };
        let window = if n == 1 {
            1.0
        } else {
            0.5 - 0.5 * libm::cos(2.0 * PI * i as f64 / (n as f64 - 1.0))
        };
        *h = 2.0 * cutoff_norm * sinc * window;
    }
    let sum: f64 = taps.iter().sum();
    if sum != 0.0 {
        for h in taps.iter_mut() {
            *h /= sum;
        }
    }
}
