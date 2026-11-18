//! Deterministic white Gaussian noise (`gauss_noise`).
//!
//! The generator is PCG32 (PCG-XSH-RR 64/32, `pcg_basic.c` of the PCG reference implementation
//! by Melissa O'Neill), the same generator as the testkit's `Pcg32`. Every real noise value
//! takes exactly **one** 32-bit draw `r`, which the inverse normal CDF maps to an integer normal
//! value `zq` in Q17 ([`normal_q17`]): the sign comes from bit 31, the magnitude from a 4 KiB
//! table of 32 octaves × 32 segments of `-Φ⁻¹` ([`gauss_icdf_table`]), interpolated linearly
//! with one Q15 multiply. There is no floating-point function, no rejection and no
//! data-dependent draw count, so the sequence is the same on every target and any sample can be
//! reached with [`GaussNoiseState::seek`] in `O(log n)` steps.
//!
//! The sequence is a seeded, deterministic function of the seed, the stream, the sample type and
//! the sample index, held in the caller's [`GaussNoiseState`]; it is not an entropy source.
//!
//! **Accuracy and tail.** The table value lies within 5.62e-5 (in units of sigma) of the ideal
//! `-Φ⁻¹(q)`, and `|z| ≤ 830729 · 2^-17 = 6.337959` sigma: the two-sided mass beyond it is
//! 2.33e-10 and the tail probabilities `P(Z > t)` are exact only to about 1.2e-10. The noise is
//! therefore **not for BER or false-alarm studies below about 1e-8**.
//!
//! ```
//! use sdrk_core::kernels::noise::{normal_q17, GaussNoise, GaussNoiseState};
//! use sdrk_core::Fallback;
//!
//! // The largest magnitude: draw 0 (and its mirror 0x8000_0000) maps to the first knot.
//! assert_eq!(normal_q17(0), 830_729);
//! assert_eq!(normal_q17(0x8000_0000), -830_729);
//! let zmax = 830_729.0 * (-17.0f64).exp2();
//! assert!((zmax - 6.337_959).abs() < 1e-6);
//!
//! // So no output exceeds 6.338 sigma.
//! let mut state = GaussNoiseState::new(1, 1, 0.5)?;
//! let mut out = [0.0f32; 4096];
//! Fallback.gauss_noise(&mut state, &mut out)?;
//! assert!(out.iter().all(|x| x.abs() <= 0.5 * 6.338));
//! # Ok::<(), sdrk_core::Error>(())
//! ```

use sdrk_device::Ratio;

use crate::error::{Error, Result};
use crate::tables::GAUSS_ICDF;
use crate::timing::{Delay, Timing};
use crate::types::{Q15, Sample, SampleType};

/// The multiplier of the PCG32 linear congruential generator.
pub const PCG32_MULT: u64 = 6_364_136_223_846_793_005;

/// The largest allowed `sigma`.
const SIGMA_MAX: f32 = 16.0;

/// `2^-17` as `f32`, the scale of the Q17 values of [`normal_q17`].
const TWO_POW_MINUS_17: f32 = 1.0 / 131_072.0;

/// The PCG32 output function (XSH-RR) of the state `old` before a step: the 32-bit draw that
/// [`Pcg32::next_u32`] returns while the generator stands at `old`. For backend crates that run
/// several generator states in lanes.
///
/// ```
/// use sdrk_core::kernels::noise::{pcg32_output, GaussNoiseState};
/// use sdrk_core::SampleType;
/// let mut state = GaussNoiseState::new(42, 54, 1.0)?;
/// let (rng, _) = state.begin(SampleType::F32, 1)?.expect("not empty");
/// let at = rng.state();
/// assert_eq!(pcg32_output(at), rng.next_u32());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[inline]
pub const fn pcg32_output(old: u64) -> u32 {
    let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
    let rot = (old >> 59) as u32;
    xorshifted.rotate_right(rot)
}

/// The PCG32 generator (PCG-XSH-RR 64/32) of a [`GaussNoiseState`], the core's own copy of the
/// testkit's generator. It is reached through [`GaussNoiseState::begin`], for kernel
/// implementations in backend crates; it cannot be created on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Creates a generator from a seed and a stream number (`pcg32_srandom_r`).
    pub(crate) fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(seed);
        rng.step();
        rng
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG32_MULT).wrapping_add(self.inc);
    }

    /// The 64-bit LCG state: the next draw is [`pcg32_output`]`(state)`.
    #[inline]
    pub fn state(&self) -> u64 {
        self.state
    }

    /// The largest number of chains of [`fill_interleaved`](Self::fill_interleaved): its `K`
    /// lane states (8 B each) are a stack array, so the bound keeps them at 128 B, far below the
    /// 4 KiB stack limit of a kernel call (R3.4).
    pub const MAX_CHAINS: usize = 16;

    /// The odd LCG increment `(stream << 1) | 1`.
    #[inline]
    pub fn increment(&self) -> u64 {
        self.inc
    }

    /// The next 32 random bits.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.step();
        pcg32_output(old)
    }

    /// The affine map `s -> mult · s + plus` (wrapping `u64`) that moves the LCG state `delta`
    /// draws ahead, as `(mult, plus)`: `mult = MULT^delta`, `plus = inc · (MULT^(delta-1) + … +
    /// 1)`, in `O(log delta)` steps (`pcg_advance_lcg_64`). Backends that run `k` chains in
    /// lanes step every lane with `jump(k)`.
    ///
    /// ```
    /// use sdrk_core::kernels::noise::GaussNoiseState;
    /// use sdrk_core::SampleType;
    /// let mut state = GaussNoiseState::new(3, 5, 1.0)?;
    /// let (rng, _) = state.begin(SampleType::F32, 8)?.expect("not empty");
    /// let (a, c) = rng.jump(4);
    /// let jumped = a.wrapping_mul(rng.state()).wrapping_add(c);
    /// for _ in 0..4 {
    ///     rng.next_u32();
    /// }
    /// assert_eq!(rng.state(), jumped);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn jump(&self, mut delta: u64) -> (u64, u64) {
        let mut cur_mult = PCG32_MULT;
        let mut cur_plus = self.inc;
        let mut acc_mult: u64 = 1;
        let mut acc_plus: u64 = 0;
        while delta > 0 {
            if delta & 1 == 1 {
                acc_mult = acc_mult.wrapping_mul(cur_mult);
                acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
            }
            cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
            cur_mult = cur_mult.wrapping_mul(cur_mult);
            delta >>= 1;
        }
        (acc_mult, acc_plus)
    }

    /// Skips `delta` draws in `O(log delta)` (`pcg32_advance_r`), wrapping at the period 2^64.
    ///
    /// ```
    /// use sdrk_core::kernels::noise::GaussNoiseState;
    /// use sdrk_core::SampleType;
    /// let mut a = GaussNoiseState::new(7, 3, 1.0)?;
    /// let mut b = a;
    /// let (ra, _) = a.begin(SampleType::F32, 1)?.expect("not empty");
    /// let (rb, _) = b.begin(SampleType::F32, 1)?.expect("not empty");
    /// let start = *ra;
    /// // `advance(n)` equals `n` draws.
    /// ra.advance(1000);
    /// for _ in 0..1000 {
    ///     rb.next_u32();
    /// }
    /// assert_eq!(*ra, *rb);
    /// // Twice half the period returns to the start.
    /// ra.advance(1 << 63);
    /// ra.advance(1 << 63);
    /// assert_eq!(*ra, *rb);
    /// rb.advance(1000_u64.wrapping_neg());
    /// assert_eq!(*rb, start);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn advance(&mut self, delta: u64) {
        let (mult, plus) = self.jump(delta);
        self.state = mult.wrapping_mul(self.state).wrapping_add(plus);
    }

    /// Writes the next `out.len()` draws in draw order with `K` interleaved chains (chain `i`
    /// starts `i` draws ahead and moves with [`jump`](Self::jump)`(K)`), and leaves the
    /// generator after the last draw: the same draws and the same state as `next_u32` in a loop,
    /// with `K` independent multiply chains for instruction-level parallelism. `K = 0` is
    /// treated as 1; `K` above [`MAX_CHAINS`](Self::MAX_CHAINS) does not compile (the lane
    /// states live on the stack, R3.4). For backend crates and tests.
    ///
    /// ```
    /// use sdrk_core::kernels::noise::GaussNoiseState;
    /// use sdrk_core::SampleType;
    /// let mut a = GaussNoiseState::new(9, 1, 1.0)?;
    /// let mut b = a;
    /// let (ra, _) = a.begin(SampleType::F32, 1)?.expect("not empty");
    /// let (rb, _) = b.begin(SampleType::F32, 1)?.expect("not empty");
    /// let mut x = [0u32; 11];
    /// ra.fill_interleaved::<4>(&mut x);
    /// let y: [u32; 11] = core::array::from_fn(|_| rb.next_u32());
    /// assert_eq!((x, *ra), (y, *rb));
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn fill_interleaved<const K: usize>(&mut self, out: &mut [u32]) {
        self.map_interleaved::<K, u32>(out, |r| r);
    }

    /// [`fill_interleaved`](Self::fill_interleaved) with the map `f` applied to every draw.
    #[inline]
    pub(crate) fn map_interleaved<const K: usize, O>(
        &mut self,
        out: &mut [O],
        f: impl Fn(u32) -> O,
    ) {
        const { assert!(K <= Self::MAX_CHAINS, "at most `Pcg32::MAX_CHAINS` chains") };
        if K <= 1 {
            for o in out.iter_mut() {
                *o = f(self.next_u32());
            }
            return;
        }
        let (a, c) = self.jump(K as u64);
        let mut lanes = [0_u64; K];
        let mut s = self.state;
        for lane in &mut lanes {
            *lane = s;
            s = s.wrapping_mul(PCG32_MULT).wrapping_add(self.inc);
        }
        let mut chunks = out.chunks_exact_mut(K);
        for chunk in &mut chunks {
            for (o, lane) in chunk.iter_mut().zip(lanes.iter_mut()) {
                let old = *lane;
                *lane = a.wrapping_mul(old).wrapping_add(c);
                *o = f(pcg32_output(old));
            }
        }
        let rest = chunks.into_remainder();
        for (o, lane) in rest.iter_mut().zip(lanes.iter()) {
            *o = f(pcg32_output(*lane));
        }
        // Lane `i` stands at draw `done + i`, so the state after the last draw is the lane at
        // the length of the remainder (always below `K`).
        if let Some(&next) = lanes.get(rest.len()) {
            self.state = next;
        }
    }
}

/// `next_u32` draws per output sample of the type `ty`: one per real value, so 2 for a complex
/// sample (the real part first) and 1 for a real sample.
///
/// ```
/// use sdrk_core::{kernels::noise::draws_per_sample, SampleType};
/// assert_eq!(draws_per_sample(SampleType::CQ15), 2);
/// assert_eq!(draws_per_sample(SampleType::F32), 1);
/// ```
pub const fn draws_per_sample(ty: SampleType) -> u64 {
    match ty {
        SampleType::C32 | SampleType::CQ15 => 2,
        _ => 1,
    }
}

/// The inverse-CDF table of the kernel (1024 entries, 4 KiB), generated by `build.rs` with
/// Wichura's AS241 in f64 and checked against `scipy.special.ndtri` by a golden case. For
/// backend crates and tests.
///
/// Entry `(L << 5) | j` (octave `L`, segment `j`) packs the Q17 knot
/// `B[L][j] = floor(-Φ⁻¹((32 + j) · 2^(L - 38)) · 2^17 + 0.5)` in bits 12 and up and the slope
/// `B[L][j] - B[L][j + 1]` (1 to 4095) in bits 0 to 11. The knots are continuous across the
/// segments and octaves, non-increasing, and end at `B[31][32] = 0` (`q = 1/2`).
///
/// On a microcontroller without a flash cache the random table access can add wait states;
/// placing the table in RAM is a linker choice of the application.
///
/// ```
/// use sdrk_core::kernels::noise::gauss_icdf_table;
/// let t = gauss_icdf_table();
/// assert_eq!(t[0] >> 12, 830_729);
/// // Continuity: base minus slope is the next base.
/// assert!(t.windows(2).all(|w| (w[0] >> 12) - (w[0] & 0xFFF) == w[1] >> 12));
/// assert_eq!((t[1023] >> 12) - (t[1023] & 0xFFF), 0);
/// ```
#[inline]
pub fn gauss_icdf_table() -> &'static [u32; 1024] {
    &GAUSS_ICDF
}

/// The integer normal value of the draw `r` in Q17 (`|zq| ≤ 830729`), the inverse CDF of the
/// contract. For backend crates and tests.
///
/// ```text
/// x    = (r << 1) | 1             // odd, so q = x / 2^33 is the midpoint of one of 2^31 cells
/// c    = clz(x);  L = 31 - c;  n = x << c
/// sub  = (n >> 26) & 31;  frac = (n >> 11) & 0x7FFF
/// e    = GAUSS_ICDF[(L << 5) | sub];  base = e >> 12;  slope = e & 0xFFF
/// mag  = base - ((slope · frac + 2^14) >> 15)
/// zq   = r >> 31 == 1 ? -mag : mag
/// ```
///
/// `zq · 2^-17` lies within 5.62e-5 of `-Φ⁻¹(q)` (`sign` applied), is non-increasing in `x`,
/// and the sign bit makes the distribution exactly symmetric: `normal_q17(r ^ 0x8000_0000) =
/// -normal_q17(r)`.
///
/// ```
/// use sdrk_core::kernels::noise::normal_q17;
/// assert_eq!(normal_q17(0x7FFF_FFFF), 0); // q just below 1/2: the median
/// for r in [1_u32, 12_345, 0x4000_0000, 0x7FFF_0000] {
///     assert_eq!(normal_q17(r ^ 0x8000_0000), -normal_q17(r));
/// }
/// // r = 2^30 gives x = 2^31 + 1, q just above 1/4: -Φ⁻¹(1/4) = 0.674490 sigma.
/// let z = f64::from(normal_q17(1 << 30)) / 131_072.0;
/// assert!((z - 0.674_490).abs() < 6e-5);
/// ```
// The names of the contract's formula (milestone M48 §2.1).
#[allow(clippy::many_single_char_names)]
#[inline]
pub fn normal_q17(r: u32) -> i32 {
    let x = (r << 1) | 1;
    let c = x.leading_zeros();
    let l = 31 - c;
    let n = x << c;
    let sub = (n >> 26) & 31;
    let frac = (n >> 11) & 0x7FFF;
    // `(l << 5) | sub` is below 1024, so the lookup always succeeds.
    let e = match GAUSS_ICDF.get(((l << 5) | sub) as usize) {
        Some(&e) => e,
        None => 0,
    };
    let base = e >> 12;
    let slope = e & 0xFFF;
    // `slope · frac < 2^27` and the rounded product is at most `slope`, so `mag ≥ 0`.
    let mag = base.wrapping_sub((slope * frac + (1 << 14)) >> 15) as i32;
    let s = (r as i32) >> 31;
    (mag ^ s).wrapping_sub(s)
}

/// The output scale of a [`GaussNoiseState`], derived from `sigma` once in
/// [`GaussNoiseState::new`] and [`GaussNoiseState::set_sigma`]. For backend crates: the exact
/// noise value of the Q17 value `zq` is `v = zq · 2^-17 · sigma_e` with
/// `sigma_e = num::flush_denormal(sigma)`, and every output format rounds `v` once.
///
/// - [`f32_value`](Self::f32_value): `(zq as f32) * s17` with `s17 = sigma_e · 2^-17` (exact, and
///   every nonzero product is normal, so no flush-to-zero mode changes a bit).
/// - [`q15_value`](Self::q15_value): the R7 rule `sat16(floor(v · 32768 + 0.5))` in integers,
///   from the 24-bit significand `ms` of `sigma_e` and the shift `sh`:
///   `v · 32768 = zq · ms · 2^-sh`.
///
/// ```
/// use sdrk_core::kernels::noise::GaussNoiseState;
/// let s = GaussNoiseState::new(1, 1, 0.75)?.scale();
/// assert_eq!(s.s17(), 0.75 / 131_072.0);
/// assert_eq!((s.ms(), s.sh()), (0xC0_0000, 26)); // 0.75 = 0xC00000 · 2^-24, sh = 2 + 24
/// assert_eq!(s.q15_value(131_072).0, 24_576); // 1.0 · 0.75 in Q15
/// assert_eq!(s.f32_value(-131_072), -0.75);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussScale {
    s17: f32,
    ms: u32,
    sh: u32,
}

impl GaussScale {
    /// The scale of `sigma` (already checked: finite, `0 ..= 16`, not `-0.0`).
    fn new(sigma: f32) -> Self {
        let sigma_e = crate::num::flush_denormal(sigma);
        if sigma_e == 0.0 {
            return Self {
                s17: 0.0,
                ms: 0,
                sh: 21,
            };
        }
        // `sigma_e` is a normal `f32` (at least 1e-30): `sigma_e = ms · 2^(e - 150)`.
        let bits = sigma_e.to_bits();
        let e = (bits >> 23) & 0xFF;
        let ms = (bits & 0x7F_FFFF) | 0x80_0000;
        // `v · 32768 = zq · 2^-17 · ms · 2^(e - 150) · 2^15 = zq · ms · 2^-(152 - e)`;
        // `sigma ≤ 16` gives `e ≤ 131`, so `sh ≥ 21`.
        let sh = 152_u32.saturating_sub(e);
        let (ms, sh) = if sh >= 45 { (0, 21) } else { (ms, sh) };
        Self {
            s17: sigma_e * TWO_POW_MINUS_17,
            ms,
            sh,
        }
    }

    /// `sigma_e · 2^-17`, the `f32` factor of the Q17 values (0 when `sigma` is flushed).
    #[inline]
    pub fn s17(&self) -> f32 {
        self.s17
    }

    /// The 24-bit significand of `sigma_e` (implicit bit included), or 0 when every `Q15`
    /// output is 0 (`sigma_e = 0`, or `sh ≥ 45`).
    #[inline]
    pub fn ms(&self) -> u32 {
        self.ms
    }

    /// The right shift of the `Q15` path, `21 ..= 44` (21 when [`ms`](Self::ms) is 0).
    #[inline]
    pub fn sh(&self) -> u32 {
        self.sh
    }

    /// True when `sigma` counts as 0 (`sigma < 1e-30`): every output is `+0.0` (or 0).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.s17 == 0.0
    }

    /// The `f32` output of the Q17 value `zq`: `v` rounded once to nearest-even. With
    /// [`is_zero`](Self::is_zero) the caller writes `+0.0` instead (the product would keep the
    /// sign of `zq`).
    #[inline]
    pub fn f32_value(&self, zq: i32) -> f32 {
        (zq as f32) * self.s17
    }

    /// The `Q15` output of the Q17 value `zq`: `sat16(floor(v · 32768 + 0.5))` in integers
    /// (`a = |zq| · ms`, `half = 2^(sh - 1)`; `(a + half) >> sh` for positive `zq`,
    /// `-((a + half - 1) >> sh)` for negative `zq`, then saturated).
    #[inline]
    pub fn q15_value(&self, zq: i32) -> Q15 {
        let a = u64::from(zq.unsigned_abs()) * u64::from(self.ms);
        let half = 1_u64 << (self.sh - 1);
        let neg = zq < 0;
        let m = (a + half - u64::from(neg)) >> self.sh;
        if neg {
            Q15(-(m.min(32_768) as i32) as i16)
        } else {
            Q15(m.min(32_767) as i16)
        }
    }
}

/// State of the Gaussian noise generator: the PCG32 generator, the seed and the stream it was
/// created from, the standard deviation `sigma` with its derived [`GaussScale`], the index of
/// the next output sample and the sample type of the last call.
///
/// Sample `n` of the output depends only on the seed, the stream, the sample type and `n`: it
/// is the normal value (or the two normal values, real part first, of a complex sample) of the
/// draws that follow `n · d` draws of the generator created from the seed and the stream, with
/// `d = 2` draws per complex and `d = 1` per real sample. [`seek`](Self::seek) therefore gives
/// random access to any sample, and the output does not depend on how the calls split the
/// stream. The period is 2^64 draws, so complex samples `n` and `n + 2^63` coincide.
///
/// The first call fixes the sample type; a call with another type is `InvalidState` until
/// [`reset`](Self::reset) or [`seek`](Self::seek) clears it.
///
/// This noise and the testkit's channel model `Awgn` are different sequences with the same
/// statistics.
///
/// ```
/// use sdrk_core::{kernels::noise::{GaussNoise, GaussNoiseState}, Fallback, C32};
/// let mut state = GaussNoiseState::new(680, 1, 0.5)?;
/// let mut whole = [C32::default(); 8];
/// Fallback.gauss_noise(&mut state, &mut whole)?;
/// assert_eq!(state.index(), 8);
///
/// // Random access: sample 5 on its own.
/// state.seek(5);
/// let mut one = [C32::default(); 1];
/// Fallback.gauss_noise(&mut state, &mut one)?;
/// assert_eq!(one[0], whole[5]);
/// # Ok::<(), sdrk_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussNoiseState {
    /// The generator. While `ty` is `None` it stands at draw 0 of `new(seed, stream)` and
    /// `index` is the position the next call advances it to.
    rng: Pcg32,
    seed: u64,
    stream: u64,
    sigma: f32,
    scale: GaussScale,
    index: u64,
    ty: Option<SampleType>,
}

impl GaussNoiseState {
    /// Creates a generator at sample index 0.
    ///
    /// `sigma` is the standard deviation of each real component: of the real value, or of the
    /// real and of the imaginary part of a complex value. It must be finite and lie in
    /// `0 ..= 16`, otherwise `InvalidParameter`; `-0.0` counts as `0.0`, and a `sigma` below
    /// 1e-30 gives zeros (it is flushed, R6.3; [`sigma`](Self::sigma) still returns it).
    /// `stream` selects one of 2^63 independent sequences (its top bit is not used, as in
    /// `pcg_basic.c`).
    ///
    /// **Power convention.** A complex sample has the power `N0 = 2 sigma²`, as in the
    /// testkit's channel model `Awgn`. Conversions from established libraries:
    ///
    /// - GNU Radio's complex `noise_source_c` with amplitude `A` (total power `A²`, variance
    ///   `A² / 2` per component): `sigma = A / sqrt(2)`. Its real `noise_source_f` with
    ///   amplitude `A`: `sigma = A`.
    /// - liquid-dsp's `crandnf` has variance 1 per component, and `cawgn` scales it by
    ///   `nstd / sqrt(2)`: `sigma = nstd / sqrt(2)`. Its real `randnf` times `s`: `sigma = s`.
    ///
    /// GNU Radio (xoroshiro128+) and liquid-dsp (libc `rand()`, Box–Muller: `randnf` the sine
    /// branch, `crandnf` both branches via `cexpf`) have other sequences; only the statistics
    /// agree.
    ///
    /// **Tail limit.** Every value lies within `6.338 sigma` (`830729 · 2^-17`); the noise is
    /// not for BER or false-alarm studies below about 1e-8.
    ///
    /// ```
    /// use sdrk_core::kernels::noise::GaussNoiseState;
    /// // The noise of a GNU Radio `noise_source_c` with amplitude 0.2.
    /// let a = 0.2f32;
    /// let state = GaussNoiseState::new(1, 1, a / core::f32::consts::SQRT_2)?;
    /// assert_eq!(state.index(), 0);
    /// // Per complex sample N0 = 2 sigma² = A².
    /// let n0 = 2.0 * state.sigma() * state.sigma();
    /// assert!((n0 - a * a).abs() < 1e-7);
    /// assert!(GaussNoiseState::new(1, 1, -0.1).is_err());
    /// assert!(GaussNoiseState::new(1, 1, f32::NAN).is_err());
    /// assert!(GaussNoiseState::new(1, 1, 16.5).is_err());
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn new(seed: u64, stream: u64, sigma: f32) -> Result<Self> {
        let sigma = check_sigma(sigma)?;
        Ok(Self {
            rng: Pcg32::new(seed, stream),
            seed,
            stream,
            sigma,
            scale: GaussScale::new(sigma),
            index: 0,
            ty: None,
        })
    }

    /// The standard deviation of each real component, as it was set.
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// The output scale derived from `sigma`.
    pub fn scale(&self) -> GaussScale {
        self.scale
    }

    /// Sets the standard deviation (the rule of [`new`](Self::new)) and recomputes the scale;
    /// the position is kept, and the next call uses the new value. On an error nothing
    /// changes.
    ///
    /// ```
    /// use sdrk_core::kernels::noise::GaussNoiseState;
    /// let mut state = GaussNoiseState::new(1, 1, 0.1)?;
    /// state.set_sigma(0.25)?;
    /// assert_eq!(state.sigma(), 0.25);
    /// assert!(state.set_sigma(f32::INFINITY).is_err());
    /// assert_eq!(state.sigma(), 0.25);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn set_sigma(&mut self, sigma: f32) -> Result<()> {
        self.sigma = check_sigma(sigma)?;
        self.scale = GaussScale::new(self.sigma);
        Ok(())
    }

    /// The seed the generator was created from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The stream the generator was created from.
    pub fn stream(&self) -> u64 {
        self.stream
    }

    /// The sample index of the next output sample (wrapping at 2^64).
    pub fn index(&self) -> u64 {
        self.index
    }

    /// The sample type fixed by the last call, `None` after construction, `seek` and `reset`.
    pub fn sample_type(&self) -> Option<SampleType> {
        self.ty
    }

    /// Moves to sample `index`: re-creates the generator from the seed and the stream and
    /// clears the sample type. The next call advances the generator by `index` times its draws
    /// per sample (2 for complex, 1 for real samples, wrapping at the period 2^64) in
    /// `O(log index)` steps before it generates.
    ///
    /// ```
    /// use sdrk_core::{kernels::noise::{GaussNoise, GaussNoiseState}, Fallback};
    /// let mut state = GaussNoiseState::new(7, 3, 1.0)?;
    /// state.seek(1 << 40);
    /// assert_eq!(state.index(), 1 << 40);
    /// let mut out = [0.0f32; 4];
    /// Fallback.gauss_noise(&mut state, &mut out)?;
    /// assert_eq!(state.index(), (1 << 40) + 4);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn seek(&mut self, index: u64) {
        self.rng = Pcg32::new(self.seed, self.stream);
        self.index = index;
        self.ty = None;
    }

    /// Returns to sample 0 (`seek(0)`); `sigma` is kept.
    pub fn reset(&mut self) {
        self.seek(0);
    }

    /// Starts a call of `len` samples of the type `ty`, for kernel implementations in backend
    /// crates: checks the type (`InvalidState` for another type than the fixed one), fixes it on
    /// the first call (advancing the generator to the stored index), and moves the index past
    /// the call. Returns the generator, standing at the first draw of the call, and the scale,
    /// or `None` for an empty call, which changes nothing. The implementation then takes
    /// exactly [`draws_per_sample`]`(ty)` draws per output sample, one per real value in draw
    /// order (re, im, re, im, … for complex samples), and leaves the generator after the last
    /// one.
    ///
    /// ```
    /// use sdrk_core::kernels::noise::{normal_q17, GaussNoiseState};
    /// use sdrk_core::SampleType;
    /// let mut state = GaussNoiseState::new(1, 1, 0.5)?;
    /// let (rng, scale) = state.begin(SampleType::F32, 2)?.expect("not empty");
    /// let first = scale.f32_value(normal_q17(rng.next_u32()));
    /// assert!(first.abs() <= 0.5 * 6.338);
    /// assert_eq!(state.index(), 2);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn begin(
        &mut self,
        ty: SampleType,
        len: usize,
    ) -> Result<Option<(&mut Pcg32, GaussScale)>> {
        if self.ty.is_some_and(|t| t != ty) {
            return Err(Error::InvalidState(
                "the sample type differs from the previous call",
            ));
        }
        if len == 0 {
            return Ok(None);
        }
        if self.ty.is_none() {
            self.ty = Some(ty);
            self.rng
                .advance(self.index.wrapping_mul(draws_per_sample(ty)));
        }
        self.index = self.index.wrapping_add(len as u64);
        Ok(Some((&mut self.rng, self.scale)))
    }
}

/// `sigma` when it is finite and lies in `0 ..= 16` (with `-0.0` made `+0.0`), else
/// `InvalidParameter`.
fn check_sigma(sigma: f32) -> Result<f32> {
    if !(sigma.is_finite() && (0.0..=SIGMA_MAX).contains(&sigma)) {
        return Err(Error::InvalidParameter("sigma must be finite and 0 to 16"));
    }
    // `-0.0 + 0.0` is `+0.0`; every other value is unchanged.
    Ok(sigma + 0.0)
}

impl Timing for GaussNoiseState {
    fn ratio(&self) -> Ratio {
        Ratio::ONE
    }

    fn group_delay(&self) -> Delay {
        Delay::ZERO
    }
}

/// Fills a buffer with white Gaussian noise of the standard deviation `sigma` per real
/// component, continuing the sequence of the state (bit-exact class).
///
/// Per real value one draw `r`, `zq = `[`normal_q17`]`(r)`, and the exact value
/// `v = zq · 2^-17 · sigma_e` (`sigma_e` = `sigma`, or 0 below 1e-30), rounded once: to the
/// nearest `f32` for `f32`, by the `Q15` rule of the numeric conventions
/// (`floor(v · 32768 + 0.5)`, saturated) for `Q15`; complex samples take two values, the real
/// part first. Values beyond full scale saturate (`sigma` 0.25: about 6.3e-5 of the `Q15`
/// values clip; `sigma` 0.6: 9.56e-2). `sigma_e = 0` gives `+0.0` (or 0) in every sample and
/// still advances the generator. An empty output changes nothing. A call with another sample
/// type than the previous one is `InvalidState`, and nothing changes.
///
/// ```
/// use sdrk_core::{kernels::noise::{GaussNoise, GaussNoiseState}, Fallback, Q15};
/// let mut state = GaussNoiseState::new(684, 3, 0.6)?;
/// let mut out = [Q15(0); 256];
/// Fallback.gauss_noise(&mut state, &mut out)?;
/// assert!(out.iter().any(|q| *q == Q15::MAX || *q == Q15::MIN)); // sigma 0.6 clips
/// // The type is fixed now.
/// assert!(Fallback.gauss_noise(&mut state, &mut [0.0f32; 4]).is_err());
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub trait GaussNoise<T: Sample> {
    /// Writes the next `output.len()` samples of the sequence and advances the state.
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [T]) -> Result<()>;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn pcg32_gives_the_known_answer_of_pcg_basic() {
        let mut rng = Pcg32::new(42, 54);
        let want = [
            0xa15c_02b7_u32,
            0x7b47_f409,
            0xba1d_3330,
            0x83d2_f293,
            0xbfa4_784b,
            0xcbed_606e,
        ];
        for w in want {
            assert_eq!(rng.next_u32(), w);
        }
    }

    #[test]
    fn advance_equals_stepping() {
        for delta in [0_u64, 1, 2, 3, 777, 4096] {
            let mut a = Pcg32::new(9, 11);
            let mut b = a;
            a.advance(delta);
            for _ in 0..delta {
                b.next_u32();
            }
            assert_eq!(a, b, "delta {delta}");
        }
        // The period is 2^64: advancing by 2^63 twice returns to the start.
        let start = Pcg32::new(5, 6);
        let mut c = start;
        c.advance(1 << 63);
        assert_ne!(c, start);
        c.advance(1 << 63);
        assert_eq!(c, start);
    }

    #[test]
    fn the_scale_of_the_limits() {
        // sigma 16 = 2^23 · 2^-19: e = 131, sh = 21.
        let s = GaussScale::new(16.0);
        assert_eq!((s.ms, s.sh), (0x80_0000, 21));
        // sh ≥ 45 stores ms = 0.
        let s = GaussScale::new(1e-12);
        assert_eq!((s.ms, s.sh), (0, 21));
        assert!(!s.is_zero());
        assert!(GaussScale::new(9e-31).is_zero());
        assert!(!GaussScale::new(1e-30).is_zero());
    }
}
