//! Analytic FM test signal.

use core::f64::consts::PI;
use num_complex::Complex64;

const TAU: f64 = 2.0 * PI;

/// Fractional part `x - floor(x)`, in `[0, 1)` for finite `x`.
fn frac(x: f64) -> f64 {
    x - libm::floor(x)
}

/// Writes an FM baseband signal of unit amplitude: a carrier at `carrier_offset_hz`, frequency-modulated
/// by a sine of `audio_hz` with the amplitude `audio_amp` and the peak deviation
/// `deviation_hz * audio_amp`.
///
/// The instantaneous frequency is `offset + deviation * audio_amp * sin(2 pi audio_hz n / rate)`.
/// The phase is integrated *analytically*,
///
/// ```text
/// phase(n) = 2 pi frac(offset n / rate) + beta (1 - cos(2 pi frac(audio_hz n / rate)))
/// beta     = deviation * audio_amp / audio_hz
/// ```
///
/// where the constant `beta` makes the phase start at 0. Because the phase depends only on the
/// absolute index `n = start + i`, generating a signal in chunks gives exactly the samples of
/// generating it at once. `audio_hz` must not be 0.
///
/// ```
/// use num_complex::Complex64;
/// let mut out = [Complex64::new(0.0, 0.0); 4];
/// // No modulation: a plain carrier at a quarter of the sample rate.
/// sdrk_testkit::fm::fm_baseband(&mut out, 1000.0, 250.0, 75.0, 1.0, 0.0, 0);
/// assert!((out[1].im - 1.0).abs() < 1e-12);
/// ```
pub fn fm_baseband(
    out: &mut [Complex64],
    rate_hz: f64,
    carrier_offset_hz: f64,
    deviation_hz: f64,
    audio_hz: f64,
    audio_amp: f64,
    start: u64,
) {
    let beta = deviation_hz * audio_amp / audio_hz;
    for (i, o) in out.iter_mut().enumerate() {
        let n = start.wrapping_add(i as u64) as f64;
        let carrier = TAU * frac(carrier_offset_hz * n / rate_hz);
        let audio = TAU * frac(audio_hz * n / rate_hz);
        let phase = carrier + beta * (1.0 - libm::cos(audio));
        *o = Complex64::new(libm::cos(phase), libm::sin(phase));
    }
}
