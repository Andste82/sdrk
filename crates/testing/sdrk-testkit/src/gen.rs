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

/// The sideband of an SSB test signal.
///
/// ```
/// use sdrk_testkit::r#gen::Sideband;
/// assert_ne!(Sideband::Usb, Sideband::Lsb);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sideband {
    /// Upper sideband: the tone lies above the carrier.
    Usb,
    /// Lower sideband: the tone lies below the carrier.
    Lsb,
}

/// SSB tone with the suppressed carrier at 0 Hz: `amplitude * exp(+-j 2 pi audio_hz n / fs)` for
/// `n = start + i` (`+` for USB, `-` for LSB), computed in f64 and rounded once to f32. The
/// phase is reduced to a fraction of a cycle first, so the samples depend only on the absolute
/// index and a signal generated in chunks equals the one generated at once.
///
/// ```
/// use sdrk_core::{Rate, C32};
/// use sdrk_testkit::r#gen::{Sideband, ssb_tone};
/// let mut usb = [C32::default(); 3];
/// let mut lsb = [C32::default(); 3];
/// // A quarter of the sample rate: 1, j, -1 for USB and the conjugate for LSB.
/// ssb_tone(&mut usb, Rate::hz(8_000).unwrap(), 2_000.0, Sideband::Usb, 0.5, 0);
/// ssb_tone(&mut lsb, Rate::hz(8_000).unwrap(), 2_000.0, Sideband::Lsb, 0.5, 0);
/// assert_eq!((usb[0].re, usb[1].im), (0.5, 0.5));
/// assert_eq!((lsb[1].im, lsb[2].re), (-0.5, -0.5));
/// ```
pub fn ssb_tone(
    out: &mut [C32],
    rate: Rate,
    audio_hz: f64,
    sideband: Sideband,
    amplitude: f64,
    start: u64,
) {
    let fs = rate.as_f64();
    let sign = match sideband {
        Sideband::Usb => 1.0,
        Sideband::Lsb => -1.0,
    };
    for (i, o) in out.iter_mut().enumerate() {
        let cycles = frac(audio_hz * index_of(start, i) / fs);
        let theta = sign * TAU * cycles;
        *o = C32::new(
            (amplitude * libm::cos(theta)) as f32,
            (amplitude * libm::sin(theta)) as f32,
        );
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

/// Modulation of [`PskTest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Psk {
    /// One bit per symbol: bit `b` gives the symbol `2b - 1` (bit 1 is `+1`).
    Bpsk,
    /// Two bits per symbol, Gray coded: bit `2k` gives the real part, bit `2k + 1` the
    /// imaginary part, each `(2b - 1) / sqrt(2)`.
    Qpsk,
}

impl Psk {
    /// Bits per symbol.
    ///
    /// ```
    /// use sdrk_testkit::r#gen::Psk;
    /// assert_eq!(Psk::Qpsk.bits_per_symbol(), 2);
    /// ```
    pub const fn bits_per_symbol(self) -> usize {
        match self {
            Self::Bpsk => 1,
            Self::Qpsk => 2,
        }
    }
}

/// Pulse shape of [`PskTest`]; the time `t` is in symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pulse {
    /// Raised cosine `sinc(t) cos(pi b t) / (1 - (2 b t)^2)` (Nyquist: zero at every other
    /// integer `t`).
    RaisedCosine,
    /// Root-raised cosine `(sin(pi t (1 - b)) + 4 b t cos(pi t (1 + b))) / (pi t (1 - (4 b t)^2))`;
    /// its convolution with itself (over `t` in symbols) is the raised cosine.
    RootRaisedCosine,
}

/// Largest pulse span of [`PskTest`] in symbols per side.
pub const PSK_MAX_SPAN: usize = 32;

/// Near-zero test of the pulse singularities.
const PULSE_EPS: f64 = 1e-9;

/// The pulse `p(t)` (time `t` in symbols, roll-off `b` in `0..=1`), evaluated in f64; the
/// removable singularities take their limits.
///
/// ```
/// use sdrk_testkit::r#gen::{pulse, Pulse};
/// assert_eq!(pulse(Pulse::RaisedCosine, 0.35, 0.0), 1.0);
/// assert!(pulse(Pulse::RaisedCosine, 0.35, 3.0).abs() < 1e-15);
/// ```
pub fn pulse(shape: Pulse, rolloff: f64, t: f64) -> f64 {
    let (a, b) = pulse_angles(shape, rolloff);
    pulse_from(shape, rolloff, t, libm::sin(a * t), libm::cos(b * t))
}

/// The angle factors `a`, `b` of the pulse terms `sin(a t)` and `cos(b t)`.
fn pulse_angles(shape: Pulse, rolloff: f64) -> (f64, f64) {
    match shape {
        Pulse::RaisedCosine => (PI, PI * rolloff),
        Pulse::RootRaisedCosine => (PI * (1.0 - rolloff), PI * (1.0 + rolloff)),
    }
}

/// The pulse from the precomputed terms `sin_at = sin(a t)` and `cos_bt = cos(b t)`.
fn pulse_from(shape: Pulse, beta: f64, t: f64, sin_at: f64, cos_bt: f64) -> f64 {
    match shape {
        Pulse::RaisedCosine => {
            let x = 2.0 * beta * t;
            let den = 1.0 - x * x;
            if libm::fabs(den) < PULSE_EPS {
                // t = +-1 / (2 b): (pi / 4) sinc(1 / (2 b))
                let u = 0.5 / beta;
                PI / 4.0 * libm::sin(PI * u) / (PI * u)
            } else if libm::fabs(t) < PULSE_EPS {
                1.0
            } else {
                sin_at / (PI * t) * cos_bt / den
            }
        }
        Pulse::RootRaisedCosine => {
            let x = 4.0 * beta * t;
            let den = 1.0 - x * x;
            if libm::fabs(t) < PULSE_EPS {
                1.0 - beta + 4.0 * beta / PI
            } else if libm::fabs(den) < PULSE_EPS {
                // t = +-1 / (4 b)
                let q = PI / (4.0 * beta);
                beta / core::f64::consts::SQRT_2
                    * ((1.0 + 2.0 / PI) * libm::sin(q) + (1.0 - 2.0 / PI) * libm::cos(q))
            } else {
                (sin_at + x * cos_bt) / (PI * t * den)
            }
        }
    }
}

/// A BPSK or QPSK test signal shaped with a raised-cosine (or root-raised-cosine) pulse,
/// computed in f64 (M38; the library pulse design comes in M40).
///
/// Symbol `k` (from the bits, see [`Psk`]) is centred at the sample position
/// [`PskTest::symbol_position`]`(k) = offset + k sps`; sample `n` is
/// `sum_k a_k p((n - offset) / sps - k)` over the symbols within `span` symbols of `n`.
/// Each sample depends only on its absolute index, so chunked generation equals generation
/// at once.
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::r#gen::{Psk, PskTest};
/// let bits = [1u8, 0, 0, 1];
/// let g = PskTest::new(Psk::Bpsk, 4.0, 0.35);
/// let mut out = vec![Complex64::new(0.0, 0.0); g.samples(&bits)];
/// g.generate(&bits, 0, &mut out);
/// assert!((out[0].re - 1.0).abs() < 1e-12); // symbol 0 (bit 1) at sample 0
/// assert!((out[4].re + 1.0).abs() < 1e-12); // symbol 1 (bit 0) at sample 4
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PskTest {
    /// Modulation.
    pub modulation: Psk,
    /// Pulse shape (default raised cosine).
    pub pulse: Pulse,
    /// Samples per symbol (`> 0`, need not be an integer).
    pub sps: f64,
    /// Roll-off factor `b` in `0..=1`.
    pub rolloff: f64,
    /// Sample position of symbol 0 (default 0.0).
    pub offset: f64,
    /// Truncation of the pulse in symbols per side (default 16, at most [`PSK_MAX_SPAN`]).
    pub span: usize,
}

impl PskTest {
    /// A raised-cosine signal with symbol 0 at sample 0 and a span of 16 symbols per side.
    ///
    /// ```
    /// use sdrk_testkit::r#gen::{Psk, Pulse, PskTest};
    /// let g = PskTest::new(Psk::Qpsk, 4.1, 0.35);
    /// assert_eq!((g.pulse, g.offset, g.span), (Pulse::RaisedCosine, 0.0, 16));
    /// ```
    pub fn new(modulation: Psk, sps: f64, rolloff: f64) -> Self {
        Self {
            modulation,
            pulse: Pulse::RaisedCosine,
            sps,
            rolloff,
            offset: 0.0,
            span: 16,
        }
    }

    /// Number of whole symbols in `bits` (left-over bits are ignored).
    ///
    /// ```
    /// use sdrk_testkit::r#gen::{Psk, PskTest};
    /// assert_eq!(PskTest::new(Psk::Qpsk, 2.0, 0.5).symbols(&[0; 7]), 3);
    /// ```
    pub fn symbols(&self, bits: &[u8]) -> usize {
        bits.len() / self.modulation.bits_per_symbol()
    }

    /// Symbol `k` of `bits` (zero beyond the last whole symbol); a non-zero bit counts as 1.
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use sdrk_testkit::r#gen::{Psk, PskTest};
    /// let g = PskTest::new(Psk::Qpsk, 2.0, 0.5);
    /// let h = core::f64::consts::FRAC_1_SQRT_2;
    /// assert_eq!(g.symbol(&[1, 0], 0), Complex64::new(h, -h));
    /// ```
    pub fn symbol(&self, bits: &[u8], k: usize) -> Complex64 {
        let level = |b: Option<&u8>| {
            if b.is_some_and(|&b| b != 0) {
                1.0
            } else {
                -1.0
            }
        };
        if k >= self.symbols(bits) {
            return Complex64::new(0.0, 0.0);
        }
        match self.modulation {
            Psk::Bpsk => Complex64::new(level(bits.get(k)), 0.0),
            Psk::Qpsk => {
                let h = core::f64::consts::FRAC_1_SQRT_2;
                Complex64::new(h * level(bits.get(2 * k)), h * level(bits.get(2 * k + 1)))
            }
        }
    }

    /// Sample position of the centre of symbol `k`: `offset + k sps`.
    ///
    /// ```
    /// use sdrk_testkit::r#gen::{Psk, PskTest};
    /// let mut g = PskTest::new(Psk::Bpsk, 4.1, 0.35);
    /// g.offset = 2.0;
    /// assert!((g.symbol_position(10) - 43.0).abs() < 1e-12);
    /// ```
    pub fn symbol_position(&self, k: usize) -> f64 {
        self.offset + k as f64 * self.sps
    }

    /// Number of samples that hold the whole signal including the pulse tail of the last
    /// symbol (from sample 0).
    ///
    /// ```
    /// use sdrk_testkit::r#gen::{Psk, PskTest};
    /// let g = PskTest::new(Psk::Bpsk, 4.0, 0.35);
    /// assert_eq!(g.samples(&[1; 10]), 36 + 64 + 1);
    /// ```
    pub fn samples(&self, bits: &[u8]) -> usize {
        let n = self.symbols(bits);
        if n == 0 {
            return 0;
        }
        let last = self.symbol_position(n - 1) + self.span.min(PSK_MAX_SPAN) as f64 * self.sps;
        if last < 0.0 {
            0
        } else {
            libm::floor(last) as usize + 1
        }
    }

    /// Writes the samples `start .. start + out.len()` of the signal of `bits`.
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use sdrk_testkit::r#gen::{Psk, PskTest};
    /// let bits = [1u8, 1, 0, 1, 0, 0, 1, 0];
    /// let g = PskTest::new(Psk::Qpsk, 4.1, 0.35);
    /// let mut whole = vec![Complex64::new(0.0, 0.0); 40];
    /// g.generate(&bits, 0, &mut whole);
    /// let mut tail = vec![Complex64::new(0.0, 0.0); 15];
    /// g.generate(&bits, 25, &mut tail);
    /// assert_eq!(&whole[25..], &tail[..]);
    /// ```
    pub fn generate(&self, bits: &[u8], start: u64, out: &mut [Complex64]) {
        let nsym = self.symbols(bits) as i64;
        let span = self.span.min(PSK_MAX_SPAN);
        let beta = self.rolloff;
        let (ang_a, ang_b) = pulse_angles(self.pulse, beta);
        // sin and cos of a j and b j for the symbol steps j = 0 ..= 2 span
        let mut tab = [(0.0f64, 0.0f64, 0.0f64, 0.0f64); 2 * PSK_MAX_SPAN + 1];
        for (j, e) in tab.iter_mut().enumerate() {
            let jf = j as f64;
            *e = (
                libm::sin(ang_a * jf),
                libm::cos(ang_a * jf),
                libm::sin(ang_b * jf),
                libm::cos(ang_b * jf),
            );
        }
        let spanf = span as f64;
        for (i, o) in out.iter_mut().enumerate() {
            let u = (index_of(start, i) - self.offset) / self.sps;
            let k_lo = (libm::ceil(u - spanf) as i64).max(0);
            let k_hi = (libm::floor(u + spanf) as i64).min(nsym - 1);
            let mut acc = Complex64::new(0.0, 0.0);
            if k_lo <= k_hi {
                let t0 = u - k_lo as f64;
                let (sa0, ca0) = (libm::sin(ang_a * t0), libm::cos(ang_a * t0));
                let (sb0, cb0) = (libm::sin(ang_b * t0), libm::cos(ang_b * t0));
                for (k, &(saj, caj, sbj, cbj)) in (k_lo..=k_hi).zip(tab.iter()) {
                    let t = u - k as f64;
                    // t = t0 - j: sin(a t) and cos(b t) by the angle-difference formulas
                    let sin_at = sa0 * caj - ca0 * saj;
                    let cos_bt = cb0 * cbj + sb0 * sbj;
                    let p = pulse_from(self.pulse, beta, t, sin_at, cos_bt);
                    acc += self.symbol(bits, k as usize) * p;
                }
            }
            *o = acc;
        }
    }
}

/// Writes the samples `0 .. out.len()` of a BPSK or QPSK signal of `bits` with the
/// raised-cosine pulse of roll-off `rolloff`, `sps` samples per symbol and symbol 0 at
/// sample 0 (the defaults of [`PskTest::new`]).
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_testkit::r#gen::{psk_test, Psk};
/// let mut out = [Complex64::new(0.0, 0.0); 16];
/// psk_test(Psk::Bpsk, 4.0, 0.35, &[1, 0, 1, 1], &mut out);
/// assert!((out[8].re - 1.0).abs() < 1e-12);
/// ```
pub fn psk_test(modulation: Psk, sps: f64, rolloff: f64, bits: &[u8], out: &mut [Complex64]) {
    PskTest::new(modulation, sps, rolloff).generate(bits, 0, out);
}
