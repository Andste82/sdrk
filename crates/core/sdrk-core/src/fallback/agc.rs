//! Fallback implementation of `agc`.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::agc::{Agc, AgcParts, AgcState};
use crate::kernels::check_len;
use crate::num::{sat16, sqrt_f32, sqrt_f64};
use crate::types::{C32, CQ15, Q15};

/// Samples per iteration of the `C32` loop of the fallback. The loop is a recursion on the
/// gain, so nothing is vectorized; four samples per iteration share the counter, the end test
/// and the branch of the loop. Measured on every counted platform (x86-64, AArch64, wasm32,
/// Cortex-M4), one sample per iteration costs more everywhere (2.4 to 3.5 instructions per
/// sample), and eight per iteration more as well (the compiler then keeps the inner loop).
const UNROLL: usize = 4;

/// One sample of the `C32` fallback: writes `y = g x` and returns the gain of the next sample.
///
/// The output level is `level = g m` with `m = |x| = sqrtf(x.re^2 + x.im^2)`. The update of
/// the specification is evaluated per range of the level in the form with the fewest
/// operations (the same function; only the roundings differ, the constants `c_r = r_r ref` and
/// `b_r = r_r / lo` are those of [`Derived`](crate::kernels::agc::Derived)):
///
/// - attack, `level > ref`: `g d / (d + r_a (level - ref))` with `d = min(level, hi)`, as
///   specified, clamped to `[g_min, g_max]`;
/// - release, `lo <= level <= ref`: `g (level + r_r (ref - level)) / level` is
///   `g + (c_r / m - r_r g)`, because `g / level = 1 / m`; clamped to `[g_min, g_max]`;
/// - release, `0 < level < lo`: `g (lo + r_r (ref - level)) / lo` is
///   `g + g (b_r (ref - level))`; clamped to `g_max`;
/// - a zero or NaN level holds the gain (it fails all three tests).
///
/// The release forms add the small update to `g` in their last step, so the rate enters with
/// its full precision. The attack and the first release form can miss the old gain by one
/// rounding near the set point (the attack `g d / den` rounds up where `den` rounds to `d`, the
/// release rounds down where its update is a rounding below zero), so both clamp on both
/// sides. The second release form adds `g` times a product of non-negative factors to `g`,
/// which never lowers it, so its clamp to `g_min` would change no value and is left out.
///
/// Plain scalar code for every target: no target-specific path in the loop and no hint to the
/// compiler. One exception lies in a shared helper: [`sqrt_f32`] is one correctly rounded IEEE
/// operation everywhere, but on wasm32 with `simd128` it reaches the hardware instruction
/// through lane 0 of `f32x4_sqrt` (`libm` has only software code there: about 189 instead of
/// 62 fuel per sample; the count of commit c45da86 already included it); it gives the same
/// bits as on every other target.
///
/// The minimum and maximum are a compare and a select (the operands are never NaN and never two
/// zeros: the level is above the reference where it is limited, and the gain is positive and
/// not NaN in every branch; an infinite level gives zero in the attack); `f32::min` and
/// `f32::max` would add their NaN rule. Each two-sided clamp first takes the minimum with
/// `g_max`: the same values as the other order (`g_min <= g_max`), but the three branches then
/// do not end in the same operation, which the compiler would otherwise move behind the
/// branches and turn into a branch of its own on x86-64, and the branches stay branches on
/// AArch64 (`cargo xtask count`, docs/performance.md).
#[allow(
    clippy::inline_always,
    reason = "the recursion on the gain must stay in the registers of the unrolled loop"
)]
#[inline(always)]
fn sample_c32(par: &AgcParts<'_>, gain: f32, x: C32, o: &mut C32) -> f32 {
    *o = C32::new(gain * x.re, gain * x.im);
    let m = sqrt_f32(x.re * x.re + x.im * x.im);
    let level = gain * m;
    if level > par.ref_f {
        let div = if level > par.hi_f { par.hi_f } else { level };
        let n = gain * div / (div + par.ra_f * (level - par.ref_f));
        let n = if n < par.gmax_f { n } else { par.gmax_f };
        if n > par.gmin_f { n } else { par.gmin_f }
    } else if level >= par.lo_f {
        let n = gain + (par.d.cr / m - par.rr_f * gain);
        let n = if n < par.gmax_f { n } else { par.gmax_f };
        if n > par.gmin_f { n } else { par.gmin_f }
    } else if level > 0.0 {
        let n = gain + gain * (par.d.br * (par.ref_f - level));
        if n < par.gmax_f { n } else { par.gmax_f }
    } else {
        gain
    }
}

impl Agc<C32> for Fallback {
    fn agc(&self, state: &mut AgcState<C32>, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let par = state.parts_mut();
        let mut gain = *par.g_f;
        // Whole chunks zipped by value: one counter (a zip of `&mut ChunksExact` tests both
        // ends in every iteration).
        let (xc, xr) = input.as_chunks::<UNROLL>();
        let (oc, or) = output.as_chunks_mut::<UNROLL>();
        for (x, o) in xc.iter().zip(oc) {
            // The chunks have exactly `UNROLL` samples, so the compiler unrolls this loop.
            for (&x, o) in x.iter().zip(o.iter_mut()) {
                gain = sample_c32(&par, gain, x, o);
            }
        }
        for (&x, o) in xr.iter().zip(or) {
            gain = sample_c32(&par, gain, x, o);
        }
        *par.g_f = gain;
        *par.g_d = f64::from(gain);
        Ok(())
    }
}

// The recursion of the `C32` gain alone, for the SIMD backend (`sdrk-backend-simd`, `agc.rs`),
// which computes the magnitudes and the products in vectors around it. The fallback does not
// use it; it computes the same values (tests `c32_fallback_matches_the_model_*` below and
// `tests/agc.rs` of the backend).

/// Samples per iteration of [`AgcState::<C32>::gains_with`] (tuned for the backend): eight on
/// AArch64 (32 FP registers), four elsewhere (the 16 of x86-64).
#[cfg(target_arch = "aarch64")]
const GAINS_UNROLL: usize = 8;
#[cfg(not(target_arch = "aarch64"))]
const GAINS_UNROLL: usize = 4;

/// The smaller of `a` and `b` as a compare and a select, for operands that are never NaN and
/// never two zeros. `f32::min` costs more on wasm32 and at the baseline of x86-64 (its NaN
/// rule), and on AArch64 its `fminnm` comes with an `fminnm` of each operand that the compiler
/// cannot prove to be no signalling NaN (a value from another basic block, such as a limit or
/// the result of a branch), which is no shorter (a backend passes the bare instruction to
/// [`AgcState::<C32>::gains_with`]).
#[allow(
    clippy::inline_always,
    reason = "part of the recursion on the gain, which must stay in the registers of the unrolled loop"
)]
#[inline(always)]
fn min_f32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// The larger of `a` and `b`, with the conditions of [`min_f32`].
#[allow(
    clippy::inline_always,
    reason = "part of the recursion on the gain, which must stay in the registers of the unrolled loop"
)]
#[inline(always)]
fn max_f32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// One step of the recursion: the gain after a sample of magnitude `m = |x|` (output level
/// `level = gain * m`), with `min` and `max` for operands that are never NaN or two zeros. The
/// same function as the step of [`sample_c32`], written for the backend: the minimum and the
/// maximum of the caller and the optional test of the hold.
///
/// The update of the specification in the form with the fewest operations per range of the
/// level (the same function; only the roundings differ, the constants are those of
/// [`Derived`](crate::kernels::agc::Derived)):
///
/// - attack, `level > ref`: `g * d / (d + r_a (level - ref))` with `d = min(level, hi)`, as
///   specified;
/// - release, `lo <= level <= ref`: `g (level + r_r (ref - level)) / level` is
///   `g + (r_r ref / m - r_r g)`, because `g / level = 1 / m`;
/// - release, `0 < level < lo`: `g (lo + r_r (ref - level)) / lo` is
///   `g + g ((r_r / lo) (ref - level))`;
/// - a zero or NaN level holds the gain (both fail `level >= lo` and `level > 0`).
///
/// `HOLD = false` omits the last test, for a caller that knows that `m` is above zero (and so
/// is the level: `m` is at least `2^-74.5` where it is not zero and `g` at least `0.01`).
///
/// The release forms add the small update to `g` in their last step, so the rate enters with
/// its full precision (a constant such as `1 - r_r` rounded to f32 would change the effective
/// rate). The new gain is clamped to `[g_min, g_max]` as specified. The attack lowers the gain
/// and the release raises it, but the attack and the first release form can miss the old gain
/// by one rounding near the set point (the attack `g d / den` rounds up where `den` rounds to
/// `d`, the release `g + (c / m - r_r g)` rounds down where the update is a rounding below
/// zero), so both clamp on both sides. The second release form adds `g` times a product of
/// non-negative factors to `g`, which never lowers it, so its clamp to `g_min` is left out (it
/// would give the same values: the gain is at least `g_min` before the step).
#[allow(
    clippy::inline_always,
    reason = "the recursion on the gain must stay in the registers of the unrolled loop; the instruction counts of the benchmark depend on it"
)]
#[inline(always)]
fn step_c32<const HOLD: bool>(
    par: &AgcParts<'_>,
    gain: f32,
    m: f32,
    min: impl Fn(f32, f32) -> f32,
    max: impl Fn(f32, f32) -> f32,
) -> f32 {
    let d = &par.d;
    let level = gain * m;
    // The operands of `min` and `max` are never NaN or two zeros: the level is above the
    // reference where it is limited, the limits are positive, and the new gain is positive and
    // not NaN in every branch (an infinite level gives zero in the attack).
    if level > par.ref_f {
        let div = min(level, par.hi_f);
        min(
            max(
                gain * div / (div + par.ra_f * (level - par.ref_f)),
                par.gmin_f,
            ),
            par.gmax_f,
        )
    } else if level >= par.lo_f {
        min(
            max(gain + (d.cr / m - par.rr_f * gain), par.gmin_f),
            par.gmax_f,
        )
    } else if !HOLD || level > 0.0 {
        min(gain + gain * (d.br * (par.ref_f - level)), par.gmax_f)
    } else {
        // `black_box` (no instruction of its own on this rare path) keeps the compiler from
        // evaluating every branch and selecting: it cannot move this path into the others, so
        // it keeps the branches. It was measured when the fallback still ran this step (about
        // 37 instructions per sample on AArch64 without it instead of about 22); the backend
        // keeps it so that its counts do not depend on that choice of the compiler.
        core::hint::black_box(gain)
    }
}

impl AgcState<C32> {
    /// The recursion of the `C32` gain alone, for a backend that computes the magnitudes and
    /// the output outside it: writes the gain that sample `i` is scaled with to `gains[i]`,
    /// then steps the gain. `mag[i]` must be `sqrt_f32(x.re * x.re + x.im * x.im)` of the
    /// input sample (IEEE operations, the square root correctly rounded, so a vector computes
    /// the same values). With `y = gains[i] * x` as the output this gives the values of
    /// `Fallback` bit for bit. The lengths must be equal.
    ///
    /// ```
    /// use sdrk_core::{kernels::agc::{Agc, AgcState}, Fallback, Rate, C32};
    /// let rate = Rate::hz(48_000)?;
    /// let mut a = AgcState::<C32>::new(0.5, 0.001, 0.01, -20.0, 60.0, 0.0, rate)?;
    /// let mut b = a;
    /// let x = [C32::new(0.3, 0.4), C32::new(0.0, 0.0), C32::new(-1.2, 0.5)];
    /// let mut y = [C32::default(); 3];
    /// Fallback.agc(&mut a, &x, &mut y)?;
    /// let mut gains = [0.0; 3];
    /// b.gains(&[0.5, 0.0, 1.3], &mut gains)?;
    /// assert_eq!(a, b);
    /// assert_eq!(y[2], x[2] * gains[2]);
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    pub fn gains(&mut self, mag: &[f32], gains: &mut [f32]) -> Result<()> {
        self.gains_with(mag, gains, false, min_f32, max_f32)
    }

    /// [`AgcState::<C32>::gains`] with the minimum and the maximum of two `f32` given by the
    /// caller, for a target with an instruction that the compiler does not emit on its own (the
    /// `fminnm`/`fmaxnm` of AArch64 without the canonicalization of the operands). They are
    /// called only with operands that are never NaN and never two zeros, so any correct
    /// minimum and maximum give the same values.
    ///
    /// `positive` says that every magnitude is above zero (none is zero or NaN, which hold the
    /// gain): the step then omits the test of the hold. The values are the same as with
    /// `false` where it is true; where it is not, the gain is wrong (but no more).
    ///
    /// ```
    /// use sdrk_core::{kernels::agc::AgcState, Rate, C32};
    /// let rate = Rate::hz(48_000)?;
    /// let mut a = AgcState::<C32>::new(0.5, 0.001, 0.01, -20.0, 60.0, 0.0, rate)?;
    /// let mut b = a;
    /// let mag = [0.5, 0.0, 1.3, 1e-3];
    /// let (mut ga, mut gb) = ([0.0; 4], [0.0; 4]);
    /// a.gains(&mag, &mut ga)?;
    /// b.gains_with(&mag, &mut gb, false, f32::min, f32::max)?;
    /// assert_eq!((a, ga), (b, gb));
    /// // Without the zero the test of the hold may be left out.
    /// let mag = [0.5, 0.2, 1.3, 1e-3];
    /// a.gains(&mag, &mut ga)?;
    /// b.gains_with(&mag, &mut gb, true, f32::min, f32::max)?;
    /// assert_eq!((a, ga), (b, gb));
    /// # Ok::<(), sdrk_core::Error>(())
    /// ```
    #[allow(
        clippy::inline_always,
        reason = "the caller's `min` and `max` must be inlined into the recursion, with the target features of the caller"
    )]
    #[inline(always)]
    pub fn gains_with(
        &mut self,
        mag: &[f32],
        gains: &mut [f32],
        positive: bool,
        min: impl Fn(f32, f32) -> f32 + Copy,
        max: impl Fn(f32, f32) -> f32 + Copy,
    ) -> Result<()> {
        check_len(mag.len(), gains.len())?;
        let par = self.parts_mut();
        let gain = if positive {
            run_gains::<false>(&par, *par.g_f, mag, gains, min, max)
        } else {
            run_gains::<true>(&par, *par.g_f, mag, gains, min, max)
        };
        *par.g_f = gain;
        *par.g_d = f64::from(gain);
        Ok(())
    }
}

/// The loop of [`AgcState::<C32>::gains_with`]; returns the gain after the last sample.
#[allow(
    clippy::inline_always,
    reason = "the caller's `min` and `max` must be inlined into the recursion, with the target features of the caller"
)]
#[inline(always)]
fn run_gains<const HOLD: bool>(
    par: &AgcParts<'_>,
    mut gain: f32,
    mag: &[f32],
    gains: &mut [f32],
    min: impl Fn(f32, f32) -> f32 + Copy,
    max: impl Fn(f32, f32) -> f32 + Copy,
) -> f32 {
    let (mc, mr) = mag.as_chunks::<GAINS_UNROLL>();
    let (gc, gr) = gains.as_chunks_mut::<GAINS_UNROLL>();
    for (m, g) in mc.iter().zip(gc) {
        // Written out: the compiler unrolls a loop of four steps, but keeps one of eight.
        let mut out = [0.0_f32; GAINS_UNROLL];
        macro_rules! steps {
            ($($i:literal)*) => {$(
                out[$i] = gain;
                gain = step_c32::<HOLD>(par, gain, m[$i], min, max);
            )*};
        }
        #[cfg(target_arch = "aarch64")]
        steps!(0 1 2 3 4 5 6 7);
        #[cfg(not(target_arch = "aarch64"))]
        steps!(0 1 2 3);
        // The gains are collected and stored together.
        *g = out;
    }
    for (&m, g) in mr.iter().zip(gr) {
        *g = gain;
        gain = step_c32::<HOLD>(par, gain, m, min, max);
    }
    gain
}

/// `2^47` and `2^48` as f64 (exact).
const P47: f64 = 140_737_488_355_328.0;
const P48: f64 = 281_474_976_710_656.0;

/// The candidate `k` of a quotient from its f64 estimate `est` (`0 <= est < 2^50`), for
/// [`settle`]: for `est >= 1/2` an integer within `1/2` of `est + 1/2`; below `1/2` it is `0`
/// (`est < 1/4`) or `1`.
///
/// x86-64: `est + (2^52 - 1/2)` is at least `2^52` for `est >= 1/2`, where the ulp is 1, so it
/// rounds `est - 1/2` to an integer `j` and its bits minus those of `2^52` are `j`; `k = j + 1`
/// is the bits minus those of `2^52 - 1/2`, the float below `2^52`. Below `1/2` the sum rounds
/// to `2^52 - 1/2` (`k = 0`) or to `2^52` (`k = 1`). One addition, one move to an integer
/// register and one subtraction: a saturating `as` costs about five instructions more there,
/// and rounding `est + 1/2` with `2^52` (the same `k` for `est >= 1/2`) one addition more on
/// the path of the gain. Elsewhere `trunc(est) + 1` (`est + 1/2 - (trunc(est) + 1)` is
/// `frac(est) - 1/2`), one instruction and an addition on AArch64 (`fcvtzu`) and on wasm32.
#[allow(
    clippy::inline_always,
    reason = "part of the per-sample update of the `CQ15` gain"
)]
#[inline(always)]
fn candidate(est: f64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // 2^52 - 1/2, the largest float below 2^52.
        const MAGIC: f64 = 4_503_599_627_370_495.5;
        (est + MAGIC).to_bits().wrapping_sub(MAGIC.to_bits())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        est as u64 + 1
    }
}

/// The rounded square root `round_sqrt(s)` of the specification for `0 <= s <= 2^32`, with one
/// f64 square root: the integer nearest to `sqrt(s)`.
///
/// Why it is exact: `s` is exact in f64 and the square root (at most `2^16`) is correctly
/// rounded, with an ulp of at most `2^-36`. The rounded root steps from `n` to `n + 1`
/// between `s = n^2 + n` and `n^2 + n + 1`, where `sqrt(s)` is `n + 1/2 - 0.25 / (n + 1/2 +
/// sqrt(s))` and `n + 1/2 + 0.75 / (n + 1/2 + sqrt(s))`: at least `0.25 / (2^17 + 1) > 2^-20`
/// away from `n + 1/2`, far more than the half ulp (`2^-37`) by which the rounded square root
/// can move. So `fl(sqrt(s))` is never within `2^-21` of a half integer, and the only integer
/// within `1/2` of it is the rounded root. The unit test checks both sides of every step for
/// every `n` up to `2^16` (both functions are monotone in `s`, so that is every `s`).
///
/// x86-64: `r + 2^52` rounds the root `r` to the nearest integer `n` (the ulp in `[2^52,
/// 2^53)` is 1), and its bits are those of `2^52` plus `n`; the low 32 bits of `2^52` are
/// zero and `n <= 2^16`, so the low 32 bits are `n` (a 32-bit move to an integer register, no
/// subtraction; a saturating `as` costs about five instructions more there). Elsewhere
/// `trunc(r + 1/2)` (`r + 1/2` is exact), one instruction on AArch64 (`fcvtzs`) and one
/// operator on wasm32 (`i64.trunc_sat_f64_s`).
#[allow(
    clippy::inline_always,
    reason = "part of the per-sample update of the `CQ15` gain"
)]
#[inline(always)]
fn round_sqrt_u32(s: u64) -> i64 {
    // `s` is below 2^63, so the signed conversion (one instruction on x86-64) is exact.
    let r = sqrt_f64(s as i64 as f64);
    #[cfg(target_arch = "x86_64")]
    {
        const MAGIC: f64 = 4_503_599_627_370_496.0; // 2^52
        i64::from((r + MAGIC).to_bits() as u32)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (r + 0.5) as i64
    }
}

/// `floor(n / d)` from a candidate `k` with `|n - k d| < 2^63` and `k - 1 <= n / d < k + 1`:
/// `n - k d` is then exact as the wrapping difference of the low 64 bits (`n_lo`, `k d mod
/// 2^64`) read as `i64`, and its sign tells `k` from `k - 1`. Integer operations of 64 bits
/// only (no 128-bit product, no 128-bit division).
#[allow(
    clippy::inline_always,
    reason = "part of the per-sample update of the `CQ15` gain"
)]
#[inline(always)]
fn settle(k: u64, n_lo: u64, d: u64) -> u64 {
    let r = n_lo.wrapping_sub(k.wrapping_mul(d)) as i64;
    k - u64::from(r < 0)
}

/// The attack of the `CQ15` gain: `floor((a 2^48 + floor(den / 2)) / den)` with
/// `a = gain * div`, i.e. `floor(gain div 2^48 / den + 1/2)` of the specification.
///
/// Domain (from the parameter rules of `AgcState::new`): `0 < gain < 2^47` (the Q.32 gain is
/// at most 90 dB), `1025 <= div <= 49152`, `den = div 2^48 + r_a |e|` with `r_a <= 2^47` and
/// a level of at most `46341` (the root of `2 * 32768^2`), so `den / 2^48 <= max(1.5 level -
/// ref / 2, ref + level / 2) < 55939` and `den < 0.854 * 2^64`; `a < 2^63`.
///
/// Bound argument: the real quotient `x = a 2^48 / den` is at most `gain < 2^47`. The f64
/// estimate divides `fl(a) 2^47` by `fl(floor(den / 2))` (three roundings of at most `2^-53`
/// each, and `floor(den / 2)` differs from `den / 2` by less than `2^-58` relatively), so it
/// is within `2^47 (3.0001 * 2^-53 + 2^-58) < 0.05` of `x`. The real quotient is at least
/// `g_min div / (den / 2^48) > 2^25.3 * 1025 / 55939 > 2^19`, so the estimate is above `1/2`
/// and `k` is an integer within `1/2` of `estimate + 1/2` ([`candidate`]), so the exact
/// quotient `t = x + floor(den / 2) / den` (the added part is in `(1/2 - 2^-58, 1/2]`) lies
/// within `1/2 + 0.05` of `k`, so `|n - k den| < 0.55 * 0.854 * 2^64 < 2^63` and [`settle`]
/// gives `floor(t)`.
#[allow(
    clippy::inline_always,
    reason = "part of the per-sample update of the `CQ15` gain"
)]
#[inline(always)]
fn attack_q(a: u64, den: u64) -> u64 {
    let half = den >> 1;
    // Both operands are below 2^63, so the signed conversions (one instruction on x86-64)
    // give the same values as unsigned ones.
    let est = (a as i64 as f64) * P47 / (half as i64 as f64);
    // The estimate is below 2^47.
    let k = candidate(est);
    // `a 2^48 + half` modulo 2^64 (the shift drops the high bits).
    settle(k, (a << 48).wrapping_add(half), den)
}

/// The constants of [`release_q`] for a release rate `r_r` (`1 <= r_r <= 2^47`).
#[derive(Clone, Copy)]
struct Release {
    /// `r_r`.
    rr: u64,
    /// `2^48 / r_r` rounded to f64, the scale of the divisor of the estimate.
    scale: f64,
    /// `2^48`, hidden from the compiler (`black_box`, once per call): with the constant it
    /// turns `k * (div << 48)` into `((k << 48) - ...) * div` and puts a shift and a
    /// subtraction after the estimate `k` on the path of the gain; with a value it does not
    /// know, `div * 2^48` is one multiplication beside that path and `k * divisor` one on it.
    unit: u64,
}

impl Release {
    fn new(rr: u64) -> Self {
        Self {
            rr,
            // `r_r` is at most 2^47 (signed conversion, as in `attack_q`).
            scale: P48 / (rr as i64 as f64),
            unit: core::hint::black_box(1 << 48),
        }
    }
}

/// The release step of the `CQ15` gain: `floor((p + div 2^47) / (div 2^48))` with
/// `p = gain * r_r * e`, i.e. `floor(gain r_r e / (div 2^48) + 1/2)` of the specification
/// (the same value as `floor(floor((p + div 2^47) / 2^48) / div)`), with the constants `c` of
/// [`Release`].
///
/// Domain: `0 < gain < 2^47`, `1 <= r_r <= 2^47`, `0 <= e <= ref <= 2^15` (so `r_r e <
/// 2^62`), `256 <= div <= 2^15` with `e / div < 4.02` (`div = max(level, ref >> 2)`).
///
/// The estimate is `gain e / (div (2^48 / r_r))`: the rate is moved into the divisor, so that
/// neither the numerator nor the divisor waits for the product `r_r e` (an integer
/// multiplication and a conversion after the level), which only the exact correction needs.
///
/// Bound argument: the real value `x = p / (div 2^48)` is below `2^47 * 2^47 * 4.02 / 2^48 <
/// 2^48.01`. The f64 estimate has four roundings (the product `gain e`, the scale, the divisor
/// `div scale`, the quotient; `gain`, `e` and `div` are exact), so it is within `2^48.01 *
/// 4.0001 * 2^-53 < 0.13` of `x`. For an estimate of at least `1/2`, `k` is an integer within
/// `1/2` of `estimate + 1/2` ([`candidate`]), so the exact quotient `t = x + 1/2` lies within
/// `1/2 + 0.13` of `k`, so `|n - k div 2^48| < 0.63 * 2^63`. Below `1/2` (`e` zero or small)
/// `k` is `0` for an estimate below `1/4`, where `t < 0.88` and `n < 0.88 div 2^48 < 2^63`, and
/// `1` otherwise, where `0.62 < t < 1.13`; either way `k - 1 <= t < k + 1` and [`settle`]
/// gives `floor(t)`.
#[allow(
    clippy::inline_always,
    reason = "part of the per-sample update of the `CQ15` gain"
)]
#[inline(always)]
fn release_q(gain: u64, e: u64, div: u64, c: Release) -> u64 {
    // `gain` is below 2^47, `e` and `div` below 2^16 (signed conversions, as in `attack_q`).
    let est = (gain as i64 as f64) * (e as i64 as f64) / (div as i64 as f64 * c.scale);
    // The estimate is below 2^49.
    let k = candidate(est);
    // `div 2^48` (at most 2^63) and its half.
    let d = div * c.unit;
    settle(k, gain.wrapping_mul(c.rr * e).wrapping_add(d >> 1), d)
}

impl Agc<CQ15> for Fallback {
    fn agc(&self, state: &mut AgcState<CQ15>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let par = state.parts_mut();
        let mut gain = *par.g_q;
        let release = Release::new(par.rr_q as u64);
        for (o, &x) in output.iter_mut().zip(input) {
            // `round_shift(x * gain, 32)` without its saturation: `|x * gain|` is below 2^15 *
            // 2^47, so adding 2^31 cannot overflow.
            let shift = |v: Q15| (i64::from(v.0) * gain + (1 << 31)) >> 32;
            let (mut re, mut im) = (shift(x.re), shift(x.im));
            // `sat16` of both. x86-64: as a branch, because the output saturates rarely and a
            // compare and select per component would lie on the path from the gain to the
            // level; the `black_box` on the rare path keeps the compiler from turning it into
            // selects. Elsewhere the selects are faster (fewer instructions on AArch64 and
            // wasm32, and fewer cycles on the Cortex-A76 model).
            #[cfg(target_arch = "x86_64")]
            {
                let wide = |v: i64| !(-32768..=32767).contains(&v);
                if wide(re) || wide(im) {
                    re = i64::from(sat16(core::hint::black_box(re)));
                    im = i64::from(sat16(im));
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                re = i64::from(sat16(re));
                im = i64::from(sat16(im));
            }
            *o = CQ15 {
                re: Q15(re as i16),
                im: Q15(im as i16),
            };
            // An input of exactly zero carries no information: hold the gain.
            if x.re.0 == 0 && x.im.0 == 0 {
                continue;
            }
            // At most 2 * 32768^2 = 2^31.
            let level = round_sqrt_u32((re * re + im * im) as u64);
            let err = par.ref_q - level;
            // The same values as the `i128` formulas of `Reference` (which divide by `2 *
            // den` after adding `den`), computed from an f64 estimate and an exact correction
            // in 64-bit integers (`attack_q`, `release_q`).
            // `clamp(next, g_min, g_max)` of the specification on one side per branch: the
            // attack never raises the gain (its quotient is at most `gain`, which is at most
            // `g_max`) and the release never lowers it (it adds a quotient of non-negative
            // values to `gain`, which is at least `g_min`), so the other side would change no
            // value; one compare and select less on the path of the gain.
            gain = if err < 0 {
                let div = level.min(par.hi_q) as u64;
                // Below 0.854 * 2^64 (see `attack_q`).
                let den = (div << 48) + (par.ra_q as u64) * err.unsigned_abs();
                // The quotient is at most `gain` (below 2^47), so the narrowing is exact.
                (attack_q(gain as u64 * div, den) as i64).max(par.gmin_q)
            } else {
                let div = level.max(par.lo_q) as u64;
                // The sum is below 2^49, so the narrowing is exact.
                let step = release_q(gain as u64, err as u64, div, release);
                ((gain as u64 + step) as i64).min(par.gmax_q)
            };
        }
        *par.g_q = gain;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! The fast quotients of the `CQ15` update against plain `u128` division: every step of
    //! the rounded square root, the whole kernel domain at its edges, random points of it, and
    //! numerators placed next to a multiple of the divisor (where an estimate is off by one).

    use super::{Release, attack_q, candidate, release_q, round_sqrt_u32};
    use crate::num::round_sqrt;

    /// xorshift64.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// Uniform in `lo..=hi` (up to a negligible bias).
        fn range(&mut self, lo: u64, hi: u64) -> u64 {
            lo + self.next() % (hi - lo + 1)
        }
    }

    fn attack_exact(a: u64, den: u64) -> u64 {
        (((u128::from(a)) << 48) + u128::from(den >> 1)).div_euclid(u128::from(den)) as u64
    }

    fn release_exact(gain: u64, re: u64, div: u64) -> u64 {
        let num = u128::from(gain) * u128::from(re) + (u128::from(div) << 47);
        (num / (u128::from(div) << 48)) as u64
    }

    /// The Q.32 gain limits and Q.48 rates of the parameter rules: -40 and 90 dB, rates up to
    /// one half.
    const GMIN: u64 = 42_949_673;
    const GMAX: u64 = 135_818_791_317_412;
    const RMAX: u64 = 1 << 47;

    /// What [`settle`](super::settle) needs of the candidate: for an estimate of at least
    /// `1/2` an integer within `1/2` of `estimate + 1/2`, below `1/2` zero (below `1/4`, on
    /// x86-64) or one; at the switch points of the rounding and across the whole range.
    #[test]
    fn the_candidate_is_within_one_half_of_the_estimate_plus_one_half() {
        let check = |est: f64| {
            let k = candidate(est);
            if est >= 0.5 {
                let d = k as f64 - (est + 0.5);
                assert!(d.abs() <= 0.5, "{est}: {k}");
            } else {
                assert!(k <= 1, "{est}: {k}");
                if cfg!(target_arch = "x86_64") && est < 0.25 {
                    assert_eq!(k, 0, "{est}");
                }
            }
        };
        let mut rng = Rng(0x1357_9bdf_2468_ace0);
        for n in 0..4096u32 {
            let base = f64::from(n) / 8.0;
            for e in [
                base,
                base.next_up(),
                base.next_down().max(0.0),
                base + 0.5,
                base + 0.25,
            ] {
                check(e);
            }
        }
        for _ in 0..1_000_000 {
            // Log-uniform up to 2^49, the largest estimate of the kernel.
            let e = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
            check(e * (1u64 << (rng.next() % 50)) as f64);
        }
        check((1u64 << 49) as f64);
    }

    #[test]
    fn round_sqrt_u32_is_exact_on_both_sides_of_every_step() {
        for n in 0..(1u64 << 16) {
            for s in [n * n, n * n + n, n * n + n + 1, (n + 1) * (n + 1) - 1] {
                assert_eq!(round_sqrt_u32(s), round_sqrt(s as i64), "s = {s}");
            }
        }
        assert_eq!(round_sqrt_u32(1 << 32), 1 << 16);
        assert_eq!(round_sqrt_u32(2 * 32768 * 32768), 46341);
    }

    /// One attack of the kernel: the divisor and the quotient as `Fallback::agc` forms them.
    fn attack_case(gain: u64, level: u64, reference: u64, ra: u64) {
        let div = level.min(reference + (reference >> 1));
        let den = (div << 48) + ra * (level - reference);
        let a = gain * div;
        let q = attack_q(a, den);
        assert_eq!(
            q,
            attack_exact(a, den),
            "g {gain} l {level} ref {reference} ra {ra}"
        );
        assert!(q <= gain);
    }

    /// One release of the kernel.
    fn release_case(gain: u64, level: u64, reference: u64, rr: u64) {
        let div = level.max(reference >> 2);
        let e = reference - level;
        assert_eq!(
            release_q(gain, e, div, Release::new(rr)),
            release_exact(gain, rr * e, div),
            "g {gain} l {level} ref {reference} rr {rr}"
        );
    }

    #[test]
    fn the_quotients_are_exact_at_the_edges_of_the_domain() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for reference in [1024, 1025, 1027, 2047, 8191, 16384, 30894, 32767, 32768] {
            for _ in 0..3 {
                let g = [GMIN, GMAX, rng.range(GMIN, GMAX)];
                let r = [1, RMAX, rng.range(1, RMAX)];
                for (&gain, &rate) in g.iter().zip(&r) {
                    for level in reference + 1..=46341 {
                        attack_case(gain, level, reference, rate);
                    }
                    for level in 1..=reference {
                        release_case(gain, level, reference, rate);
                    }
                }
            }
            // Every pair of the edge values at the extreme levels.
            for gain in [GMIN, GMIN + 1, GMAX - 1, GMAX] {
                for rate in [1, 2, RMAX - 1, RMAX] {
                    for level in [reference + 1, reference + (reference >> 1), 46341] {
                        attack_case(gain, level, reference, rate);
                    }
                    for level in [1, (reference >> 2) - 1, reference >> 2, reference] {
                        release_case(gain, level, reference, rate);
                    }
                }
            }
        }
    }

    #[test]
    fn the_quotients_are_exact_at_random_points_of_the_domain() {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        for i in 0..2_000_000u32 {
            let reference = rng.range(1024, 32768);
            // Small rates as well as large ones (the slow loops of long time constants).
            let rate = rng.range(1, RMAX) >> (rng.next() % 48);
            let rate = rate.max(1);
            let gain = rng.range(GMIN, GMAX) >> (u64::from(i) % 12);
            let gain = gain.max(GMIN);
            if i % 2 == 0 {
                attack_case(gain, rng.range(reference + 1, 46341), reference, rate);
            } else {
                release_case(gain, rng.range(1, reference), reference, rate);
            }
        }
    }

    #[test]
    fn the_quotients_are_exact_next_to_a_multiple_of_the_divisor() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let mut checked = 0u32;
        for _ in 0..1_000_000u32 {
            // Attack: a numerator `a 2^48 + floor(den / 2)` within `2^48` of `q den`, for any
            // divisor of the domain (`1025 * 2^48 <= den < 55939 * 2^48`).
            let den = rng.range(1025 << 48, (55939 << 48) - 1);
            let q = rng.range(1, (1 << 47) - 2);
            let target = u128::from(q) * u128::from(den) - u128::from(den >> 1);
            let a0 = (target >> 48) as u64;
            for a in [a0.saturating_sub(1), a0, a0 + 1, a0 + 2] {
                if (u128::from(a) << 48) < (u128::from(den) << 47) {
                    assert_eq!(attack_q(a, den), attack_exact(a, den), "a {a} den {den}");
                    checked += 1;
                }
            }
            // Release: `gain * r_r * e + div 2^47` within `r_r e` of `q div 2^48`, with the
            // gain chosen for a random `q` of the gain range (small rates put the numerator
            // close to the multiple).
            let div = rng.range(256, 32768);
            let e = rng.range(1, (4 * div).min(32768));
            let rr = (rng.range(1, RMAX) >> (rng.next() % 48)).max(1);
            let re = u128::from(rr * e);
            let unit = u128::from(div) << 48;
            let q = u128::from(rng.range(GMIN, GMAX)) * re / unit + 1;
            let g0 = ((q * unit - (unit >> 1)) / re) as u64;
            for gain in [g0.saturating_sub(1), g0, g0 + 1] {
                // The domain: the gain range and `gain r_r e / (div 2^48) < 2^48.01`.
                if (GMIN..=GMAX).contains(&gain) && u128::from(gain) * re < unit << 48 {
                    assert_eq!(
                        release_q(gain, e, div, Release::new(rr)),
                        release_exact(gain, rr * e, div),
                        "g {gain} rr {rr} e {e} div {div}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 5_000_000, "{checked}");
    }
}

#[cfg(all(test, feature = "reference"))]
mod c32_tests {
    //! The `C32` fallback against a model of its contract written out once more (one sample at
    //! a time, the limits as `f32::clamp` and `f32::min`), bit for bit, and against the f64
    //! `Reference`, within the accuracy of f32: at the bounds of every branch (output levels at
    //! and next to `ref`, `lo` and `hi`), at zero, NaN, infinity and tiny and huge magnitudes,
    //! for corner parameters (both ends of the reference, the fastest and slow rates, gains at
    //! both limits, equal limits), and on random signals through the unrolled loop and its
    //! remainder. A longer random run is `#[ignore]`d: `cargo test -p sdrk-core --release --lib
    //! --features reference -- --ignored c32_` (the tests need `Reference`).
    #![allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::indexing_slicing,
        clippy::unwrap_used,
        clippy::panic,
        clippy::many_single_char_names
    )]

    use super::{AgcParts, sqrt_f32};
    use crate::kernels::agc::{Agc, AgcState};
    use crate::{C32, Fallback, Rate, Reference};
    use std::{format, vec, vec::Vec};

    const FS: u64 = 48_000;

    /// The update of the contract for one sample: the output and the next gain.
    fn model(par: &AgcParts<'_>, g: f32, x: C32) -> (C32, f32) {
        let y = C32::new(g * x.re, g * x.im);
        let m = sqrt_f32(x.re * x.re + x.im * x.im);
        let level = g * m;
        let next = if level > par.ref_f {
            let div = level.min(par.hi_f);
            (g * div / (div + par.ra_f * (level - par.ref_f))).clamp(par.gmin_f, par.gmax_f)
        } else if level >= par.lo_f {
            (g + (par.d.cr / m - par.rr_f * g)).clamp(par.gmin_f, par.gmax_f)
        } else if level > 0.0 {
            (g + g * (par.d.br * (par.ref_f - level))).min(par.gmax_f)
        } else {
            g
        };
        (y, next)
    }

    /// The model over `x`, with the state updated as the kernels do.
    fn run_model(state: &mut AgcState<C32>, x: &[C32]) -> Vec<C32> {
        let par = state.parts_mut();
        let mut g = *par.g_f;
        let mut y = Vec::with_capacity(x.len());
        for &s in x {
            let (o, n) = model(&par, g, s);
            y.push(o);
            g = n;
        }
        *par.g_f = g;
        *par.g_d = f64::from(g);
        y
    }

    fn gain_f(state: &mut AgcState<C32>) -> f32 {
        *state.parts_mut().g_f
    }

    fn gain_d(state: &mut AgcState<C32>) -> f64 {
        *state.parts_mut().g_d
    }

    /// The index of the first sample whose bits differ (two NaN count as equal: the payload of
    /// a NaN result is not specified).
    fn differs(a: &[C32], b: &[C32]) -> Option<usize> {
        let bits = |p: f32, q: f32| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan());
        a.iter()
            .zip(b)
            .position(|(p, q)| !(bits(p.re, q.re) && bits(p.im, q.im)))
    }

    /// Fallback and model from equal states over `x`, in calls of `chunk` samples (the unrolled
    /// loop and its remainder).
    fn check_model(state: AgcState<C32>, x: &[C32], chunk: usize, what: &str) {
        let (mut sf, mut sm) = (state, state);
        let want = run_model(&mut sm, x);
        let mut got = vec![C32::default(); x.len()];
        for (i, o) in x.chunks(chunk).zip(got.chunks_mut(chunk)) {
            Fallback.agc(&mut sf, i, o).unwrap();
        }
        if let Some(i) = differs(&want, &got) {
            panic!(
                "{what}: sample {i} of {}: {:?} against {:?}",
                x.len(),
                got[i],
                want[i]
            );
        }
        assert_eq!(sf, sm, "{what}: state");
    }

    fn state(reference: f64, attack: f64, release: f64, gains: (f64, f64, f64)) -> AgcState<C32> {
        let (min, max, init) = gains;
        AgcState::new(
            reference,
            attack,
            release,
            min,
            max,
            init,
            Rate::hz(FS).unwrap(),
        )
        .unwrap()
    }

    /// Corner parameter sets.
    fn params() -> Vec<AgcState<C32>> {
        // The fastest allowed loop (a rate of 0.5) and slow ones.
        let fastest = 1.0 / (FS as f64 * core::f64::consts::LN_2) * 1.000_000_1;
        let mut out = Vec::new();
        for reference in [0.03125, 0.0809, 0.5, 1.0] {
            for (attack, release) in [(fastest, fastest), (0.005, 0.05), (0.2, 2.0)] {
                for gains in [
                    (-40.0, 90.0, -40.0),
                    (-40.0, 90.0, 90.0),
                    (-40.0, 90.0, 0.0),
                    (-6.0, 12.0, 6.0),
                    (-25.3, 90.0, -25.3),
                    (-40.0, 83.7, 83.7),
                    (3.0, 3.0, 3.0),
                ] {
                    out.push(state(reference, attack, release, gains));
                }
            }
        }
        out
    }

    /// `v` moved by `k` floats (positive and finite `v`).
    fn ulps(v: f32, k: i32) -> f32 {
        f32::from_bits(v.to_bits().wrapping_add_signed(k))
    }

    /// Samples whose output level at gain `g` is at or next to `ref`, `lo` and `hi`, on both
    /// axes and off them, and the special magnitudes.
    fn edges(par: &AgcParts<'_>, g: f32) -> Vec<C32> {
        let mut out = Vec::new();
        for t in [par.ref_f, par.lo_f, par.hi_f] {
            let m = t / g;
            for k in -3..=3 {
                let m = ulps(m, k);
                out.extend([
                    C32::new(m, 0.0),
                    C32::new(0.0, -m),
                    C32::new(-m * 0.6, m * 0.8),
                ]);
            }
        }
        for v in [
            1e-45, 1e-40, 1e-23, 5.0e-23, 1e-22, 1e-19, 1e-15, 1.0e-6, 1e6, 1e15, 1e19,
        ] {
            out.extend([C32::new(v, 0.0), C32::new(v, v)]);
        }
        out.extend([
            C32::new(0.0, 0.0),
            C32::new(-0.0, 0.0),
            C32::new(0.0, -0.0),
            C32::new(f32::NAN, 0.0),
            C32::new(0.3, f32::NAN),
            C32::new(f32::INFINITY, 0.0),
            C32::new(-1.0, f32::NEG_INFINITY),
            C32::new(f32::INFINITY, f32::NAN),
            C32::new(2e19, 0.0),
            C32::new(3e19, 3e19),
            C32::new(f32::MAX, f32::MAX),
            C32::new(f32::MIN_POSITIVE, 0.0),
        ]);
        out
    }

    /// Whether the f32 kernels and the f64 model see the same kind of level: the squares of a
    /// finite `x` neither underflow nor overflow in f32 (where they do, `m` is zero or
    /// infinite and the f64 level is not, which the specification allows).
    fn ordinary(x: C32) -> bool {
        let m = f64::from(x.re).hypot(f64::from(x.im));
        m == 0.0 || (1e-18..=1e18).contains(&m) || !m.is_finite()
    }

    #[test]
    fn c32_fallback_matches_the_model_and_the_reference_at_the_branch_bounds() {
        let mut probes = 0;
        for st in params() {
            let mut st0 = st;
            let g = gain_f(&mut st0);
            let gmin = st0.parts_mut().gmin_f;
            let edge = edges(&st0.parts_mut(), g);
            for &x in &edge {
                // One sample from the state (the remainder of the loop) and the same sample in
                // every slot of the unrolled loop.
                check_model(st, &[x], 1, &format!("{x:?} alone"));
                for slot in 0..4 {
                    let mut block = [C32::new(0.01, 0.02); 9];
                    block[slot] = x;
                    check_model(st, &block, 9, &format!("{x:?} in slot {slot}"));
                }
                // Against the f64 model: one step from the same gain.
                let (mut sf, mut sr) = (st, st);
                let (mut yf, mut yr) = ([C32::default()], [C32::default()]);
                Fallback.agc(&mut sf, &[x], &mut yf).unwrap();
                Reference.agc(&mut sr, &[x], &mut yr).unwrap();
                let (gf, gr) = (f64::from(gain_f(&mut sf)), gain_d(&mut sr));
                if ordinary(x) {
                    let rel = (gf - gr).abs() / gr;
                    assert!(rel < 1e-5, "{x:?}: {gf:e} against {gr:e} of the reference");
                    let held = (x.re == 0.0 && x.im == 0.0) || x.re.is_nan() || x.im.is_nan();
                    let infinite = (x.re.is_infinite() || x.im.is_infinite()) && !held;
                    if held {
                        assert_eq!(gain_f(&mut sf), g, "{x:?} holds the gain");
                        assert_eq!(gr, gain_d(&mut st0), "{x:?} holds the reference gain");
                    }
                    if infinite {
                        assert_eq!(
                            gain_f(&mut sf),
                            gmin,
                            "{x:?} drives the gain to the minimum"
                        );
                    }
                } else {
                    // The f32 squares underflow (to zero, which holds the gain, or to a
                    // subnormal, an inexact low level) or overflow (an infinite level: the
                    // attack goes to the minimum); the gain stays within its limits.
                    let g1 = gain_f(&mut sf);
                    let par = st0.parts_mut();
                    assert!(g1 >= par.gmin_f && g1 <= par.gmax_f, "{x:?}: {g1:e}");
                }
                probes += 1;
            }
            // All bounds after one another, so that the gain moves between them.
            check_model(st, &edge, 4, "all bounds, chunks of 4");
            check_model(st, &edge, 7, "all bounds, chunks of 7");
        }
        assert!(probes > 8_000, "{probes}");
    }

    /// xorshift64.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// Uniform in `[0, 1)`.
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Noise whose amplitude changes every 1 to 300 samples, log-uniform over `10^-7` to
    /// `10^4`, with a special sample (zero, NaN, infinity, a tiny or a huge value) in `specials`
    /// of every 1000.
    fn signal(rng: &mut Rng, n: usize, specials: u64) -> Vec<C32> {
        let special = [
            C32::new(0.0, 0.0),
            C32::new(f32::NAN, 1.0),
            C32::new(f32::INFINITY, 0.0),
            C32::new(1e-30, -1e-30),
            C32::new(1e25, 0.0),
            C32::new(1e-41, 0.0),
        ];
        let mut out = Vec::with_capacity(n);
        let mut amp = 1.0;
        let mut left = 0;
        while out.len() < n {
            if left == 0 {
                amp = libm::pow(10.0, -7.0 + 11.0 * rng.unit());
                left = 1 + rng.next() % 300;
            }
            left -= 1;
            if rng.next() % 1000 < specials {
                out.push(special[(rng.next() % special.len() as u64) as usize]);
            } else {
                let (a, b) = (rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 - 1.0);
                out.push(C32::new((a * amp) as f32, (b * amp) as f32));
            }
        }
        out
    }

    fn snr_db(want: &[C32], got: &[C32]) -> f64 {
        let (mut s, mut e) = (0.0, 0.0);
        for (w, g) in want.iter().zip(got) {
            let (wr, wi) = (f64::from(w.re), f64::from(w.im));
            let (dr, di) = (wr - f64::from(g.re), wi - f64::from(g.im));
            s += wr * wr + wi * wi;
            e += dr * dr + di * di;
        }
        10.0 * libm::log10(s / e)
    }

    fn random_run(seed: u64, n: usize) {
        let mut rng = Rng(seed);
        for (k, st) in params().into_iter().enumerate() {
            let x = signal(&mut rng, n, 5);
            let chunk = [1, 3, 4, 5, 17, 256, 4096][k % 7];
            check_model(st, &x, chunk, &format!("params {k} chunk {chunk}"));
            // Against the f64 model, without the special samples.
            let x = signal(&mut rng, n, 0);
            let (mut sf, mut sr) = (st, st);
            let (mut yf, mut yr) = (vec![C32::default(); n], vec![C32::default(); n]);
            Fallback.agc(&mut sf, &x, &mut yf).unwrap();
            Reference.agc(&mut sr, &x, &mut yr).unwrap();
            let snr = snr_db(&yr, &yf);
            assert!(snr > 80.0, "params {k}: {snr} dB against the reference");
        }
    }

    #[test]
    fn c32_fallback_matches_the_model_on_random_signals() {
        random_run(0x5eed_a6c3_2000_0001, 3_000);
    }

    /// About 170 million samples (84 parameter sets of two signals of 10^6); slow without
    /// optimisation.
    #[test]
    #[ignore = "long; run in release"]
    fn c32_fallback_matches_the_model_on_long_random_signals() {
        random_run(0x5eed_a6c3_2000_0002, 1_000_000);
    }
}

#[cfg(all(test, feature = "reference"))]
mod cq15_tests {
    //! The `CQ15` fallback against `Reference` (`i128` arithmetic), bit for bit, where its
    //! update takes the less common paths: outputs that saturate in one component, in the
    //! other or in both (the saturation is a branch of its own), outputs at and next to the
    //! bounds `-32768` and `32767`, gains at both limits, and the release with a zero or small
    //! error (an estimate below one half), in calls of every length from 1 to 9 samples.
    #![allow(
        clippy::indexing_slicing,
        clippy::unwrap_used,
        clippy::panic,
        clippy::cast_possible_truncation
    )]

    use crate::kernels::agc::{Agc, AgcState};
    use crate::types::Sample;
    use crate::{CQ15, Fallback, Q15, Rate, Reference};
    use std::{format, vec, vec::Vec};

    /// xorshift64.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn sample(&mut self, amp: i64) -> i16 {
            if amp == 0 {
                return 0;
            }
            let span = (2 * amp + 1) as u64;
            (((self.next() % span) as i64) - amp).clamp(-32768, 32767) as i16
        }
    }

    fn state(reference: f64, attack: f64, release: f64, gains: (f64, f64, f64)) -> AgcState<CQ15> {
        let (min, max, init) = gains;
        AgcState::new(
            reference,
            attack,
            release,
            min,
            max,
            init,
            Rate::hz(48_000).unwrap(),
        )
        .unwrap()
    }

    /// Fallback and `Reference` from equal states over `x`, the fallback in calls of `chunk`
    /// samples.
    fn check(st: AgcState<CQ15>, x: &[CQ15], chunk: usize, what: &str) {
        let (mut sf, mut sr) = (st, st);
        let mut want = vec![CQ15::ZERO; x.len()];
        Reference.agc(&mut sr, x, &mut want).unwrap();
        let mut got = vec![CQ15::ZERO; x.len()];
        for (i, o) in x.chunks(chunk).zip(got.chunks_mut(chunk)) {
            Fallback.agc(&mut sf, i, o).unwrap();
        }
        if let Some(i) = want.iter().zip(&got).position(|(w, g)| w != g) {
            panic!("{what}: sample {i}: {:?} against {:?}", got[i], want[i]);
        }
        assert_eq!(sf, sr, "{what}: state");
    }

    /// Blocks in which the output saturates in the real part only, the imaginary part only,
    /// both or neither, with the gain moving between them; full-scale corners and samples whose
    /// output lands next to the bounds at a gain of one.
    fn signal(rng: &mut Rng, n: usize) -> Vec<CQ15> {
        let mut out = Vec::with_capacity(n);
        let corners = [
            (32767, 0),
            (-32768, 0),
            (0, 32767),
            (0, -32768),
            (32767, -32768),
            (-32768, -32768),
            (1, -32768),
            (32766, 1),
        ];
        while out.len() < n {
            let kind = rng.next() % 6;
            for _ in 0..=(rng.next() % 40) {
                let (re, im) = match kind {
                    0 => (rng.sample(32768), rng.sample(30)),
                    1 => (rng.sample(30), rng.sample(32768)),
                    2 => (rng.sample(32768), rng.sample(32768)),
                    3 => corners[(rng.next() % corners.len() as u64) as usize],
                    4 => (rng.sample(3), rng.sample(3)),
                    _ => (rng.sample(2000), rng.sample(2000)),
                };
                out.push(CQ15 {
                    re: Q15(re),
                    im: Q15(im),
                });
            }
        }
        out.truncate(n);
        out
    }

    #[test]
    fn cq15_fallback_matches_the_reference_where_the_output_saturates() {
        let mut rng = Rng(0x00c1_5a6c_05a7_u64);
        let fastest = 1.45 / 48_000.0;
        let params = [
            (0.5, 0.005, 0.05, (-20.0, 60.0, 0.0)),
            (1.0, fastest, fastest, (-40.0, 90.0, 90.0)),
            (1.0 / 32.0, fastest, 0.2, (-40.0, 90.0, -40.0)),
            (0.7, 0.0005, 0.01, (0.0, 0.0, 0.0)),
            (0.25, 0.1, fastest, (-6.0, 30.0, 30.0)),
            (0.999, 0.002, 2.0, (-40.0, 12.0, 6.0)),
        ];
        let mut saturated = 0usize;
        for (k, &(reference, attack, release, gains)) in params.iter().enumerate() {
            let st = state(reference, attack, release, gains);
            let x = signal(&mut rng, 6000);
            // Count the saturating outputs of the reference run, to be sure the test reaches
            // that path.
            let (mut sr, mut y) = (st, vec![CQ15::ZERO; x.len()]);
            Reference.agc(&mut sr, &x, &mut y).unwrap();
            saturated += y
                .iter()
                .filter(|s| [s.re.0, s.im.0].iter().any(|&v| v == 32767 || v == -32768))
                .count();
            for chunk in 1..=9 {
                check(st, &x, chunk, &format!("params {k} chunk {chunk}"));
            }
            // A single sample from the state, for every corner and gain limit.
            for &s in &x[..200] {
                check(st, &[s], 1, &format!("params {k} single {s:?}"));
            }
        }
        assert!(saturated > 1000, "{saturated}");
    }
}
