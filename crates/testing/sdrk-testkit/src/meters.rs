//! Measurement meters: tone fit, SINAD, frequency estimation and spectral masks.
//!
//! All meters compute in f64 with `libm`, allocate nothing and are independent of the FFT
//! kernel under test (the DFTs here are direct sums).

// Short names are the notation of the least-squares and DFT formulas.
#![allow(clippy::many_single_char_names)]

use core::f64::consts::PI;
use core::fmt;

use num_complex::Complex64;

const TAU: f64 = 2.0 * PI;

/// Fractional part `x - floor(x)`.
fn frac(x: f64) -> f64 {
    x - libm::floor(x)
}

/// `(cos, sin)` of the phase of sample `n` of a tone at `freq` (Hz) for sample rate `rate`.
fn cos_sin(n: usize, rate: f64, freq: f64) -> (f64, f64) {
    let theta = TAU * frac(freq * n as f64 / rate);
    (libm::cos(theta), libm::sin(theta))
}

/// Result of a [`ToneMeter`] fit: `x[n] ~ amplitude cos(2 pi f n / rate + phase) + dc`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tone {
    /// Amplitude of the fitted tone (peak, not RMS).
    pub amplitude: f64,
    /// Phase in radians at sample 0, in `(-pi, pi]`.
    pub phase: f64,
    /// Fitted constant offset.
    pub dc: f64,
    /// Mean power of what is left after removing the tone and the offset.
    pub residual_power: f64,
}

/// Least-squares tone fit at a known frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToneMeter;

impl ToneMeter {
    /// Fits `a cos + b sin + dc` to `signal` at `freq` (Hz, sample rate `rate`) by least squares.
    ///
    /// If the three basis functions are linearly dependent (fewer than three samples, `freq` at
    /// 0 or at a multiple of the Nyquist frequency) only the offset is fitted and the tone
    /// amplitude is 0.
    ///
    /// ```
    /// use sdrk_testkit::meters::ToneMeter;
    /// let x: Vec<f64> = (0..1000)
    ///     .map(|n| 0.5 * (2.0 * std::f64::consts::PI * 100.0 * n as f64 / 8000.0).cos() + 0.1)
    ///     .collect();
    /// let t = ToneMeter::measure(&x, 8000.0, 100.0);
    /// assert!((t.amplitude - 0.5).abs() < 1e-12 && (t.dc - 0.1).abs() < 1e-12);
    /// assert!(t.residual_power < 1e-24);
    /// ```
    pub fn measure(signal: &[f64], rate: f64, freq: f64) -> Tone {
        let n = signal.len();
        if n == 0 {
            return Tone {
                amplitude: 0.0,
                phase: 0.0,
                dc: 0.0,
                residual_power: 0.0,
            };
        }
        // Normal equations of the basis (cos, sin, 1).
        let (mut scc, mut sss, mut scs, mut sc, mut ss) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut xc, mut xs, mut x1) = (0.0, 0.0, 0.0);
        for (i, &x) in signal.iter().enumerate() {
            let (c, s) = cos_sin(i, rate, freq);
            scc += c * c;
            sss += s * s;
            scs += c * s;
            sc += c;
            ss += s;
            xc += x * c;
            xs += x * s;
            x1 += x;
        }
        let nf = n as f64;
        let m = [[scc, scs, sc], [scs, sss, ss], [sc, ss, nf]];
        let (a, b, dc) = match solve3(m, [xc, xs, x1], nf) {
            Some(sol) => sol,
            None => (0.0, 0.0, x1 / nf),
        };
        let mut residual = 0.0;
        for (i, &x) in signal.iter().enumerate() {
            let (c, s) = cos_sin(i, rate, freq);
            let r = x - a * c - b * s - dc;
            residual += r * r;
        }
        Tone {
            amplitude: libm::hypot(a, b),
            phase: if a == 0.0 && b == 0.0 {
                0.0
            } else {
                libm::atan2(-b, a)
            },
            dc,
            residual_power: residual / nf,
        }
    }
}

/// Solves `m * v = rhs` by Cramer's rule; `None` when the matrix is (nearly) singular.
/// `scale` is the sample count, which scales the determinant of a well-conditioned system.
fn solve3(m: [[f64; 3]; 3], rhs: [f64; 3], scale: f64) -> Option<(f64, f64, f64)> {
    let det3 = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let det = det3(m);
    if det.is_nan() || libm::fabs(det) <= 1e-9 * scale * scale * scale {
        return None;
    }
    let col = |k: usize| {
        let mut a = m;
        for (row, r) in a.iter_mut().zip(rhs) {
            if let Some(cell) = row.get_mut(k) {
                *cell = r;
            }
        }
        det3(a) / det
    };
    Some((col(0), col(1), col(2)))
}

/// Signal-to-noise-and-distortion ratio of a tone at `freq`:
/// `10 log10(total_power / residual_power)`, where the total power is the mean power without
/// the fitted offset and the residual comes from [`ToneMeter`]. `+inf` for a perfect tone.
///
/// ```
/// let x: Vec<f64> = (0..4000)
///     .map(|n| (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / 48000.0).sin())
///     .collect();
/// assert!(sdrk_testkit::meters::sinad_db(&x, 48000.0, 1000.0) > 200.0);
/// ```
pub fn sinad_db(signal: &[f64], rate: f64, freq: f64) -> f64 {
    let t = ToneMeter::measure(signal, rate, freq);
    let n = signal.len().max(1) as f64;
    let total: f64 = signal.iter().map(|x| (x - t.dc) * (x - t.dc)).sum::<f64>() / n;
    if t.residual_power <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(total / t.residual_power)
    }
}

/// Frequency of a (nearly) pure tone in Hz: the frequency within `guess +- span` that
/// minimises the residual power of the [`ToneMeter`] fit (golden-section style ternary search).
///
/// The search is exact when the residual has a single minimum in the interval, which holds
/// for `span` below about half the reciprocal of the signal duration.
///
/// ```
/// let x: Vec<f64> = (0..4800)
///     .map(|n| (2.0 * std::f64::consts::PI * 1000.05 * n as f64 / 48000.0).sin())
///     .collect();
/// let f = sdrk_testkit::meters::fit_frequency(&x, 48000.0, 1000.0, 5.0);
/// assert!((f - 1000.05).abs() < 1e-6);
/// ```
pub fn fit_frequency(signal: &[f64], rate: f64, guess: f64, span: f64) -> f64 {
    let residual = |f: f64| ToneMeter::measure(signal, rate, f).residual_power;
    let (mut lo, mut hi) = (guess - span, guess + span);
    for _ in 0..60 {
        let third = (hi - lo) / 3.0;
        let (a, b) = (lo + third, hi - third);
        if residual(a) < residual(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    f64::midpoint(lo, hi)
}

/// Longest part of the signal that [`estimate_frequency`] analyses.
pub const ESTIMATE_MAX_SAMPLES: usize = 2048;

/// Frequency of the strongest component of a complex signal in Hz, in `(-rate/2, rate/2]`.
///
/// The peak of a 4x zero-padded f64 DFT of (at most the first [`ESTIMATE_MAX_SAMPLES`] samples of)
/// the signal, refined by parabolic interpolation of the magnitudes of the three bins around
/// the peak. It is a direct sum and does not use the FFT kernel under test.
///
/// ```
/// use num_complex::Complex64;
/// let x: Vec<Complex64> = (0..512)
///     .map(|n| {
///         let a = 2.0 * std::f64::consts::PI * 1234.5 * n as f64 / 48000.0;
///         Complex64::new(a.cos(), a.sin())
///     })
///     .collect();
/// let f = sdrk_testkit::meters::estimate_frequency(&x, 48000.0);
/// assert!((f - 1234.5).abs() < 2.0);
/// ```
pub fn estimate_frequency(signal: &[Complex64], rate: f64) -> f64 {
    let x = signal.get(..ESTIMATE_MAX_SAMPLES).unwrap_or(signal);
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let bins = 4 * n;
    let mag = |k: usize| -> f64 {
        // Direct sum with a phasor recurrence, the step is e^{-j 2 pi k / bins}.
        let theta = -TAU * k as f64 / bins as f64;
        let step = Complex64::new(libm::cos(theta), libm::sin(theta));
        let (mut w, mut acc) = (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        for (i, s) in x.iter().enumerate() {
            acc += s * w;
            w *= step;
            // Re-anchor the phasor now and then so that rounding errors cannot build up.
            if i % 256 == 255 {
                let theta = theta * (i + 1) as f64;
                w = Complex64::new(libm::cos(theta), libm::sin(theta));
            }
        }
        libm::hypot(acc.re, acc.im)
    };
    let mut best = (0usize, -1.0f64);
    for k in 0..bins {
        let m = mag(k);
        if m > best.1 {
            best = (k, m);
        }
    }
    let k = best.0;
    let (left, right) = (mag((k + bins - 1) % bins), mag((k + 1) % bins));
    let denom = left - 2.0 * best.1 + right;
    let delta = if denom == 0.0 {
        0.0
    } else {
        (0.5 * (left - right) / denom).clamp(-0.5, 0.5)
    };
    let mut pos = k as f64 + delta;
    if pos > bins as f64 / 2.0 {
        pos -= bins as f64;
    }
    pos / bins as f64 * rate
}

/// A bin of the spectrum that exceeds the mask.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskViolation {
    /// Frequency of the offending bin in Hz.
    pub freq_hz: f64,
    /// Its level in dB relative to the strongest bin.
    pub level_db: f64,
    /// The limit of the mask entry it violates.
    pub limit_db: f64,
}

impl fmt::Display for MaskViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "spectrum mask violated at {} Hz: {} dB above the limit of {} dB",
            self.freq_hz, self.level_db, self.limit_db
        )
    }
}

impl core::error::Error for MaskViolation {}

/// Length of the DFT segments of [`spectral_mask`].
pub const MASK_SEGMENT: usize = 1024;

/// Checks the averaged power spectrum of a complex signal against a mask.
///
/// The signal is cut into non-overlapping segments of [`MASK_SEGMENT`] samples (a shorter signal is
/// one zero-padded segment); each segment is multiplied by a symmetric Hann window and
/// transformed with a direct f64 DFT, and the powers are averaged per bin. Levels are in dB
/// relative to the strongest bin of the average. Every mask entry `(f_lo, f_hi, max_db)`
/// requires all bins with `f_lo <= f <= f_hi` (bin frequencies run from `-rate/2` up to just
/// below `rate/2`) to be at most `max_db`. Returns the first violation in bin order.
///
/// The cost is `2 * 1024 * length` complex multiply-adds, so pass an excerpt of a long signal.
///
/// ```
/// use num_complex::Complex64;
/// let x: Vec<Complex64> = (0..2048)
///     .map(|n| {
///         let a = 2.0 * std::f64::consts::PI * 0.125 * n as f64;
///         Complex64::new(a.cos(), a.sin())
///     })
///     .collect();
/// // Everything away from +rate/8 must be 40 dB down.
/// let mask = [(-8000.0, 1500.0, -40.0), (2500.0, 7999.0, -40.0)];
/// assert!(sdrk_testkit::meters::spectral_mask(&x, 16000.0, &mask).is_ok());
/// let tight = [(1000.0, 3000.0, -40.0)];
/// assert!(sdrk_testkit::meters::spectral_mask(&x, 16000.0, &tight).is_err());
/// ```
pub fn spectral_mask(
    signal: &[Complex64],
    rate: f64,
    mask: &[(f64, f64, f64)],
) -> Result<(), MaskViolation> {
    let segments = (signal.len() / MASK_SEGMENT).max(1);
    let window_step = TAU / (MASK_SEGMENT as f64 - 1.0);
    let window_cos = libm::cos(window_step);
    let power = |k: usize| -> f64 {
        let theta = -TAU * k as f64 / MASK_SEGMENT as f64;
        let step = Complex64::new(libm::cos(theta), libm::sin(theta));
        let mut total = 0.0;
        for s in 0..segments {
            let seg = signal.get(s * MASK_SEGMENT..).unwrap_or_default();
            let (mut w, mut acc) = (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
            // cos(2 pi i / (N - 1)) by the Chebyshev recurrence, re-anchored with `libm` now
            // and then, so that the window costs no `cos` per sample.
            let (mut prev, mut cur) = (window_cos, 1.0);
            for (i, x) in seg.iter().take(MASK_SEGMENT).enumerate() {
                acc += x * w * (0.5 - 0.5 * cur);
                w *= step;
                let next = 2.0 * window_cos * cur - prev;
                prev = cur;
                cur = next;
                if i % 256 == 255 {
                    let theta = theta * (i + 1) as f64;
                    w = Complex64::new(libm::cos(theta), libm::sin(theta));
                    let a = window_step * (i + 1) as f64;
                    cur = libm::cos(a);
                    prev = libm::cos(a - window_step);
                }
            }
            total += acc.re * acc.re + acc.im * acc.im;
        }
        total / segments as f64
    };
    let freq_of = |k: usize| {
        let signed = if k < MASK_SEGMENT / 2 {
            k as f64
        } else {
            k as f64 - MASK_SEGMENT as f64
        };
        signed / MASK_SEGMENT as f64 * rate
    };
    let peak = (0..MASK_SEGMENT).map(power).fold(0.0, f64::max);
    if peak <= 0.0 {
        return Ok(());
    }
    for k in 0..MASK_SEGMENT {
        let f = freq_of(k);
        let mut limit: Option<f64> = None;
        for &(lo, hi, max_db) in mask {
            if f >= lo && f <= hi {
                limit = Some(limit.map_or(max_db, |l: f64| l.min(max_db)));
            }
        }
        if let Some(limit_db) = limit {
            let level_db = 10.0 * libm::log10(power(k).max(1e-300) / peak);
            if level_db > limit_db {
                return Err(MaskViolation {
                    freq_hz: f,
                    level_db,
                    limit_db,
                });
            }
        }
    }
    Ok(())
}
