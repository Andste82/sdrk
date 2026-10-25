//! Signal generators.
//!
//! All generators compute in f64, write into a caller slice and depend only on
//! the absolute sample index `n = start + i`. Generating a signal in chunks
//! therefore gives exactly the same samples as generating it at once.
//!
//! The module is named `gen`, a reserved word in edition 2024; use it as
//! `sdrk_testkit::r#gen`.

use core::f64::consts::PI;
use num_complex::Complex64;
use sdrk_core::{C32, ComplexSample, Rate};

use crate::rng::Pcg32;

const TAU: f64 = 2.0 * PI;

/// Fractional part `x - floor(x)`, always in `[0, 1)` for finite `x`.
fn frac(x: f64) -> f64 {
    x - libm::floor(x)
}

fn index_of(start: u64, i: usize) -> f64 {
    start.wrapping_add(i as u64) as f64
}

fn unit(phase: f64, amp: f64) -> Complex64 {
    Complex64::new(amp * libm::cos(phase), amp * libm::sin(phase))
}

/// Converts f64 samples to the sample type `T` (each component rounded with
/// [`ComplexSample::from_c64_round`]). Converts `min(src.len(), dst.len())`
/// samples.
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_core::{CQ15, Q15};
/// let mut dst = [CQ15::default(); 1];
/// sdrk_testkit::r#gen::write_as(&[Complex64::new(0.5, -0.5)], &mut dst);
/// assert_eq!(dst[0], CQ15 { re: Q15(16384), im: Q15(-16384) });
/// ```
pub fn write_as<T: ComplexSample>(src: &[Complex64], dst: &mut [T]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = T::from_c64_round(*s);
    }
}

/// Complex tone `amp * exp(j(2 pi frac(freq n) + phase))`; `freq` is in
/// cycles per sample and may be negative.
///
/// ```
/// use num_complex::Complex64;
/// let mut out = [Complex64::new(0.0, 0.0); 4];
/// sdrk_testkit::r#gen::tone(&mut out, 0.25, 1.0, 0.0, 0);
/// assert!((out[1].im - 1.0).abs() < 1e-12);
/// ```
pub fn tone(out: &mut [Complex64], freq: f64, amp: f64, phase: f64, start: u64) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = unit(TAU * frac(freq * index_of(start, i)) + phase, amp);
    }
}

/// Real tone `amp * cos(2 pi frac(freq n) + phase)`.
pub fn real_tone(out: &mut [f64], freq: f64, amp: f64, phase: f64, start: u64) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = amp * libm::cos(TAU * frac(freq * index_of(start, i)) + phase);
    }
}

/// Sum of complex tones; every entry is `(freq, amp, phase)`.
pub fn multitone(out: &mut [Complex64], tones: &[(f64, f64, f64)], start: u64) {
    for (i, o) in out.iter_mut().enumerate() {
        let n = index_of(start, i);
        let mut acc = Complex64::new(0.0, 0.0);
        for &(freq, amp, phase) in tones {
            acc += unit(TAU * frac(freq * n) + phase, amp);
        }
        *o = acc;
    }
}

/// Complex Gaussian noise with `E|x|^2 = power`. Per sample the real part is
/// drawn first, then the imaginary part (four `next_u32` draws each).
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::rng::Pcg32;
/// let mut out = [Complex64::new(0.0, 0.0); 8];
/// sdrk_testkit::r#gen::noise(&mut out, &mut Pcg32::new(1, 1), 2.0);
/// ```
pub fn noise(out: &mut [Complex64], rng: &mut Pcg32, power: f64) {
    let sigma = libm::sqrt(power / 2.0);
    for o in out.iter_mut() {
        let re = sigma * rng.normal();
        let im = sigma * rng.normal();
        *o = Complex64::new(re, im);
    }
}

/// Real Gaussian noise with variance `power`.
pub fn real_noise(out: &mut [f64], rng: &mut Pcg32, power: f64) {
    let sigma = libm::sqrt(power);
    for o in out.iter_mut() {
        *o = sigma * rng.normal();
    }
}

/// Linear chirp over `len` samples: instantaneous frequency
/// `f0 + (f1 - f0) n / len`, phase `2 pi (f0 n + (f1 - f0) n^2 / (2 len))`.
/// A `len` of 0 is treated as 1.
pub fn chirp(out: &mut [Complex64], f0: f64, f1: f64, len: u64, amp: f64, start: u64) {
    let len = len.max(1) as f64;
    for (i, o) in out.iter_mut().enumerate() {
        let n = index_of(start, i);
        let cycles = f0 * n + (f1 - f0) * n * n / (2.0 * len);
        *o = unit(TAU * cycles, amp);
    }
}

/// A single `amp` at absolute index `at`, zero elsewhere.
pub fn impulse(out: &mut [Complex64], at: u64, amp: f64, start: u64) {
    for (i, o) in out.iter_mut().enumerate() {
        let hit = start.wrapping_add(i as u64) == at;
        *o = Complex64::new(if hit { amp } else { 0.0 }, 0.0);
    }
}

/// Zero before absolute index `at`, `amp` from `at` on.
pub fn step(out: &mut [Complex64], at: u64, amp: f64, start: u64) {
    for (i, o) in out.iter_mut().enumerate() {
        let on = start.wrapping_add(i as u64) >= at;
        *o = Complex64::new(if on { amp } else { 0.0 }, 0.0);
    }
}

/// Amplitude-modulated carrier `carrier * (1 + index * cos(2 pi audio_hz n / fs)) * exp(j phase)`
/// for `n = start + i`, computed in f64 and rounded once to f32. The modulation phase is
/// reduced to a fraction of a cycle first, so the samples depend only on the absolute index and
/// a signal generated in chunks equals the one generated at once.
///
/// ```
/// use sdrk_core::{Rate, C32};
/// let mut out = [C32::default(); 4];
/// // Carrier 0.5, 25 % modulation at a quarter of the sample rate, no phase shift.
/// sdrk_testkit::r#gen::am_tone(&mut out, Rate::hz(8_000).unwrap(), 2_000.0, 0.25, 0.5, 0.0, 0);
/// assert_eq!(out[0].re, 0.625);
/// assert!((out[1].re - 0.5).abs() < 1e-6);
/// assert_eq!(out[1].im, 0.0);
/// ```
pub fn am_tone(
    out: &mut [C32],
    rate: Rate,
    audio_hz: f64,
    index: f64,
    carrier: f64,
    phase: f64,
    start: u64,
) {
    let fs = rate.as_f64();
    let (pc, ps) = (libm::cos(phase), libm::sin(phase));
    for (i, o) in out.iter_mut().enumerate() {
        let cycles = frac(audio_hz * index_of(start, i) / fs);
        let env = carrier * (1.0 + index * libm::cos(TAU * cycles));
        *o = C32::new((env * pc) as f32, (env * ps) as f32);
    }
}

/// Extreme inputs for robustness tests.
pub mod extreme {
    use num_complex::Complex64;
    use sdrk_core::{C32, ComplexSample, Sample};

    /// All-zero signal.
    pub fn silence<T: Sample>(out: &mut [T]) {
        out.fill(T::ZERO);
    }

    /// Alternates `+(1 - 2^-15)` and `-(1 - 2^-15)` on both components,
    /// starting positive. For `CQ15` this is the raw pair `(32767, 32767)`,
    /// `(-32767, -32767)`, ...
    ///
    /// ```
    /// use sdrk_core::{CQ15, Q15};
    /// let mut out = [CQ15::default(); 2];
    /// sdrk_testkit::r#gen::extreme::full_scale_alternating(&mut out);
    /// assert_eq!(out[0].re, Q15(32767));
    /// assert_eq!(out[1].im, Q15(-32767));
    /// ```
    pub fn full_scale_alternating<T: ComplexSample>(out: &mut [T]) {
        let a = 1.0 - 1.0 / 32768.0;
        for (i, o) in out.iter_mut().enumerate() {
            let v = if i % 2 == 0 { a } else { -a };
            *o = T::from_c64_round(Complex64::new(v, v));
        }
    }

    /// The special f32 values cycled by [`f32_specials`].
    pub const F32_SPECIALS: [f32; 7] = [
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE / 2.0,
        f32::MAX,
    ];

    /// Fills with `0.0, -0.0, NaN, +inf, -inf, MIN_POSITIVE/2 (denormal),
    /// f32::MAX`, cycling.
    pub fn f32_specials(out: &mut [f32]) {
        for (o, v) in out.iter_mut().zip(F32_SPECIALS.iter().cycle()) {
            *o = *v;
        }
    }

    /// Like [`f32_specials`] for complex samples: the real part walks through
    /// the list, the imaginary part is shifted by three entries so that both
    /// components differ.
    pub fn f32_specials_complex(out: &mut [C32]) {
        let re = F32_SPECIALS.iter().cycle();
        let im = F32_SPECIALS.iter().cycle().skip(3);
        for (o, (r, i)) in out.iter_mut().zip(re.zip(im)) {
            *o = C32::new(*r, *i);
        }
    }
}
