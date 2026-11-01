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

/// Like [`fm_baseband`] for two simultaneous audio tones, each given as `(peak deviation in Hz,
/// audio frequency in Hz)`:
///
/// ```text
/// phase(n) = 2 pi frac(offset n / rate) + sum_i beta_i (1 - cos(2 pi frac(audio_i n / rate)))
/// beta_i   = deviation_i / audio_i
/// ```
///
/// The phase depends only on the absolute index `start + i`, so chunked generation gives the same
/// samples as generating the signal at once. Neither audio frequency may be 0.
///
/// ```
/// use num_complex::Complex64;
/// let mut out = [Complex64::new(0.0, 0.0); 4];
/// // Without deviation: a plain carrier at a quarter of the sample rate.
/// sdrk_testkit::fm::fm_baseband_two_tones(&mut out, 1000.0, 250.0, [(0.0, 100.0), (0.0, 7.0)], 0);
/// assert!((out[1].im - 1.0).abs() < 1e-12);
/// ```
pub fn fm_baseband_two_tones(
    out: &mut [Complex64],
    rate_hz: f64,
    carrier_offset_hz: f64,
    tones: [(f64, f64); 2],
    start: u64,
) {
    for (i, o) in out.iter_mut().enumerate() {
        let n = start.wrapping_add(i as u64) as f64;
        let mut phase = TAU * frac(carrier_offset_hz * n / rate_hz);
        for (deviation_hz, audio_hz) in tones {
            let audio = TAU * frac(audio_hz * n / rate_hz);
            phase += deviation_hz / audio_hz * (1.0 - libm::cos(audio));
        }
        *o = Complex64::new(libm::cos(phase), libm::sin(phase));
    }
}

/// One cosine component of a multi-tone FM modulation signal: the instantaneous frequency
/// deviation contains `amplitude * cos(2 pi freq_hz t + phase_rad)` (in units of the deviation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tone {
    /// Amplitude, relative to the peak deviation of the signal.
    pub amplitude: f64,
    /// Frequency in Hz; must not be 0.
    pub freq_hz: f64,
    /// Phase at `t = 0` in radians.
    pub phase_rad: f64,
}

/// A tone of [`fm_multitone_baseband`] has the frequency 0 (the phase integral does not exist).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroToneFrequency;

impl core::fmt::Display for ZeroToneFrequency {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("a tone of the FM signal has the frequency 0")
    }
}

impl core::error::Error for ZeroToneFrequency {}

/// Writes an FM baseband signal of unit amplitude whose instantaneous frequency is
/// `carrier_offset_hz + deviation_hz * sum_i amplitude_i cos(2 pi freq_i t + phase_i)`.
///
/// The phase is integrated *analytically*,
///
/// ```text
/// phase(n) = 2 pi frac(offset n / rate)
///          + sum_i deviation amplitude_i / freq_i (sin(2 pi frac(freq_i n / rate) + phase_i) - sin(phase_i))
/// ```
///
/// (the constant makes the phase start at 0), so it depends only on the absolute index
/// `start + i` and generating a signal in chunks gives exactly the samples of generating it at
/// once. Tones with `amplitude` 0 are allowed.
///
/// # Errors
/// [`ZeroToneFrequency`] if a tone has `freq_hz == 0`; `out` is not written then.
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::fm::{Tone, fm_multitone_baseband};
/// let mut out = [Complex64::new(0.0, 0.0); 4];
/// let tones = [Tone { amplitude: 0.0, freq_hz: 100.0, phase_rad: 0.0 }];
/// // No deviation: a plain carrier at a quarter of the sample rate.
/// fm_multitone_baseband(&mut out, 1000.0, 250.0, 75.0, &tones, 0)?;
/// assert!((out[1].im - 1.0).abs() < 1e-12);
/// let bad = [Tone { amplitude: 1.0, freq_hz: 0.0, phase_rad: 0.0 }];
/// assert!(fm_multitone_baseband(&mut out, 1000.0, 250.0, 75.0, &bad, 0).is_err());
/// # Ok::<(), sdrk_testkit::fm::ZeroToneFrequency>(())
/// ```
pub fn fm_multitone_baseband(
    out: &mut [Complex64],
    rate_hz: f64,
    carrier_offset_hz: f64,
    deviation_hz: f64,
    tones: &[Tone],
    start: u64,
) -> Result<(), ZeroToneFrequency> {
    if tones.iter().any(|t| t.freq_hz == 0.0) {
        return Err(ZeroToneFrequency);
    }
    for (i, o) in out.iter_mut().enumerate() {
        let n = start.wrapping_add(i as u64) as f64;
        let mut phase = TAU * frac(carrier_offset_hz * n / rate_hz);
        for t in tones {
            let arg = TAU * frac(t.freq_hz * n / rate_hz) + t.phase_rad;
            phase +=
                deviation_hz * t.amplitude / t.freq_hz * (libm::sin(arg) - libm::sin(t.phase_rad));
        }
        *o = Complex64::new(libm::cos(phase), libm::sin(phase));
    }
    Ok(())
}

/// Frequency of the stereo pilot in Hz.
pub const PILOT_HZ: f64 = 19_000.0;

/// The multiplex signal of FM stereo (ITU-R BS.450, `docs/milestones/M24-fm-stereo.md` section 2)
/// for the sine tones `L = a_l sin(2 pi f_l t)` and `R = a_r sin(2 pi f_r t)` as cosine tones
/// (`left` and `right` are `(amplitude, freq_hz)`, an amplitude of 0 is allowed):
///
/// ```text
/// m = 0.45 (L + R) + 0.45 (L - R) sin(2 w_p t) + pilot sin(w_p t)
/// ```
///
/// The sum terms have the phase `-pi/2`, the products `sin * sin` are written as
/// `1/2 (cos(difference) - cos(sum))` around 38 kHz, the pilot has the phase `-pi/2`. The
/// frequencies of the products are `38 kHz -+ f`, so the tones must keep them non-zero.
/// The result has 7 tones (two audio, four products, the pilot), in this order.
///
/// ```
/// use sdrk_testkit::fm::stereo_mpx_tones;
/// let tones = stereo_mpx_tones((1.0, 1000.0), (1.0, 3000.0), 0.1);
/// assert_eq!(tones.len(), 7);
/// assert_eq!(tones[6].freq_hz, 19_000.0);
/// ```
#[must_use]
pub fn stereo_mpx_tones(left: (f64, f64), right: (f64, f64), pilot: f64) -> [Tone; 7] {
    let (a_l, f_l) = left;
    let (a_r, f_r) = right;
    let sub = 2.0 * PILOT_HZ;
    let t = |amplitude: f64, freq_hz: f64, phase_rad: f64| Tone {
        amplitude,
        freq_hz,
        phase_rad,
    };
    [
        t(0.45 * a_l, f_l, -PI / 2.0),
        t(0.45 * a_r, f_r, -PI / 2.0),
        // 0.45 L sin(2 w_p t) = 0.225 a_l (cos((38k - f_l) t) - cos((38k + f_l) t))
        t(0.225 * a_l, sub - f_l, 0.0),
        t(-0.225 * a_l, sub + f_l, 0.0),
        // -0.45 R sin(2 w_p t)
        t(-0.225 * a_r, sub - f_r, 0.0),
        t(0.225 * a_r, sub + f_r, 0.0),
        t(pilot, PILOT_HZ, -PI / 2.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_generation_equals_one_shot() {
        let tones = stereo_mpx_tones((1.0, 1000.0), (0.7, 3000.0), 0.1);
        let mut whole = [Complex64::new(0.0, 0.0); 200];
        fm_multitone_baseband(&mut whole, 2_400_000.0, 600_000.0, 75_000.0, &tones, 12_345)
            .unwrap();
        let mut chunked = [Complex64::new(0.0, 0.0); 200];
        for (k, piece) in chunked.chunks_mut(37).enumerate() {
            fm_multitone_baseband(
                piece,
                2_400_000.0,
                600_000.0,
                75_000.0,
                &tones,
                12_345 + 37 * k as u64,
            )
            .unwrap();
        }
        assert_eq!(whole, chunked);
    }

    #[test]
    fn the_tones_sum_to_the_multiplex_signal_of_the_milestone() {
        let (a_l, f_l, a_r, f_r, a_p) = (0.9, 1000.0, 0.6, 3000.0, 0.1);
        let tones = stereo_mpx_tones((a_l, f_l), (a_r, f_r), a_p);
        for n in 0..2000 {
            let t = f64::from(n) * 1.234e-5;
            let sum: f64 = tones
                .iter()
                .map(|k| k.amplitude * libm::cos(TAU * k.freq_hz * t + k.phase_rad))
                .sum();
            let l = a_l * libm::sin(TAU * f_l * t);
            let r = a_r * libm::sin(TAU * f_r * t);
            let w = TAU * PILOT_HZ * t;
            let direct = 0.45 * (l + r) + 0.45 * (l - r) * libm::sin(2.0 * w) + a_p * libm::sin(w);
            assert!((sum - direct).abs() < 1e-12, "n = {n}: {sum} vs {direct}");
        }
    }
}
