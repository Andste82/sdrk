//! Fallback implementation of `agc`.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::agc::{Agc, AgcParts, AgcState};
use crate::kernels::check_len;
use crate::num::{near_int_plus_half_f64, round_sqrt_u32, sat16, sqrt_f32};
use crate::types::{C32, CQ15, Q15};

/// Samples per iteration of the `C32` loops. The loops are a recursion on the gain, so nothing
/// is vectorized; four samples per iteration share the counter, the end test and the branch of
/// the loop, and more than four make the compiler spill registers at the baseline level of
/// x86-64.
const UNROLL: usize = 4;

/// Samples per iteration of [`AgcState::<C32>::gains_with`], whose loop holds less per sample:
/// eight on AArch64 (32 FP registers), four elsewhere (the 16 of x86-64).
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
/// `level = gain * m`), with `min` and `max` for operands that are never NaN or two zeros.
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
        // it keeps the branches. Without it the fallback ran about 37 instructions per sample
        // on AArch64 instead of about 22.
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

/// One sample of the fallback: writes `gain * x` and returns the next gain.
#[allow(
    clippy::inline_always,
    reason = "the recursion on the gain must stay in the registers of the unrolled loop; the instruction counts of the benchmark depend on it"
)]
#[inline(always)]
fn sample_c32(par: &AgcParts<'_>, gain: f32, x: C32, o: &mut C32) -> f32 {
    *o = C32::new(gain * x.re, gain * x.im);
    let m = sqrt_f32(x.re * x.re + x.im * x.im);
    step_c32::<true>(par, gain, m, min_f32, max_f32)
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

/// `2^47` and `2^48` as f64 (exact).
const P47: f64 = 140_737_488_355_328.0;
const P48: f64 = 281_474_976_710_656.0;

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
/// is within `2^47 (3.0001 * 2^-53 + 2^-58) < 0.05` of `x`. `k` is an integer within `1/2` of
/// `estimate + 1/2` ([`near_int_plus_half_f64`]), so the exact quotient `t = x + floor(den /
/// 2) / den` (the added part is in `(1/2 - 2^-58, 1/2]`) lies within `1/2 + 0.05` of `k`, so
/// `|n - k den| < 0.55 * 0.854 * 2^64 < 2^63` and [`settle`] gives `floor(t)`.
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
    let k = near_int_plus_half_f64(est);
    // `a 2^48 + half` modulo 2^64 (the shift drops the high bits).
    settle(k, (a << 48).wrapping_add(half), den)
}

/// The release step of the `CQ15` gain: `floor((p + div 2^47) / (div 2^48))` with
/// `p = gain * re` and `re = r_r e`, i.e. `floor(gain r_r e / (div 2^48) + 1/2)` of the specification
/// (the same value as `floor(floor((p + div 2^47) / 2^48) / div)`).
///
/// Domain: `0 < gain < 2^47`, `r_r <= 2^47`, `0 <= e <= ref <= 2^15` (so `re < 2^62`),
/// `256 <= div <= 2^15` with `e / div < 4.02` (`div = max(level, ref >> 2)`).
///
/// Bound argument: the real value `x = p / (div 2^48)` is below `2^47 * 2^47 * 4.02 / 2^48 <
/// 2^48.01`. The f64 estimate has three roundings (`fl(re)`, the product, the quotient; `div
/// 2^48` is exact), so it is within `2^48.01 * 3.0001 * 2^-53 < 0.1` of `x`. `k` is an integer
/// within `1/2` of `estimate + 1/2`, so the exact quotient `t = x + 1/2` lies within `1/2 +
/// 0.1` of `k`, so `|n - k div 2^48| < 0.6 * 2^63` and [`settle`] gives `floor(t)`.
#[allow(
    clippy::inline_always,
    reason = "part of the per-sample update of the `CQ15` gain"
)]
#[inline(always)]
fn release_q(gain: u64, re: u64, div: u64) -> u64 {
    // `gain` and `re` are below 2^63 (signed conversions, as in `attack_q`).
    let est = (gain as i64 as f64) * (re as i64 as f64) / (div as i64 as f64 * P48);
    // The estimate is below 2^49.
    let k = near_int_plus_half_f64(est);
    settle(k, gain.wrapping_mul(re).wrapping_add(div << 47), div << 48)
}

impl Agc<CQ15> for Fallback {
    fn agc(&self, state: &mut AgcState<CQ15>, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let par = state.parts_mut();
        let mut gain = *par.g_q;
        for (o, &x) in output.iter_mut().zip(input) {
            // `round_shift(x * gain, 32)` without its saturation: `|x * gain|` is below 2^15 *
            // 2^47, so adding 2^31 cannot overflow.
            let scale = |v: Q15| Q15(sat16((i64::from(v.0) * gain + (1 << 31)) >> 32));
            let y = CQ15 {
                re: scale(x.re),
                im: scale(x.im),
            };
            *o = y;
            // An input of exactly zero carries no information: hold the gain.
            if x.re.0 == 0 && x.im.0 == 0 {
                continue;
            }
            let (re, im) = (i64::from(y.re.0), i64::from(y.im.0));
            // At most 2 * 32768^2 = 2^31.
            let level = round_sqrt_u32((re * re + im * im) as u64);
            let err = par.ref_q - level;
            // The same values as the `i128` formulas of `Reference` (which divide by `2 *
            // den` after adding `den`), computed from an f64 estimate and an exact correction
            // in 64-bit integers (`attack_q`, `release_q`).
            let next = if err < 0 {
                let div = level.min(par.hi_q) as u64;
                // Below 0.854 * 2^64 (see `attack_q`).
                let den = (div << 48) + (par.ra_q as u64) * err.unsigned_abs();
                // The quotient is at most `gain`.
                attack_q(gain as u64 * div, den)
            } else {
                let div = level.max(par.lo_q) as u64;
                // `r_r e` is below 2^47 * 2^15.
                gain as u64 + release_q(gain as u64, par.rr_q as u64 * err as u64, div)
            };
            // The limits bound `next` (below 2^49) to the Q.32 range, so the narrowing is
            // exact; `max` then `min` is `clamp` for `gmin_q <= gmax_q` without its check.
            gain = (next as i64).max(par.gmin_q).min(par.gmax_q);
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

    use super::{attack_q, release_q};
    use crate::num::{round_sqrt, round_sqrt_u32};

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
        let re = rr * (reference - level);
        assert_eq!(
            release_q(gain, re, div),
            release_exact(gain, re, div),
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
            // Release: `gain * re + div 2^47` within `gain` of `q div 2^48`.
            let div = rng.range(256, 32768);
            let gain = rng.range(GMIN, GMAX);
            let q = rng.range(1, (1 << 48) - 1);
            let target = ((u128::from(q) * u128::from(div)) << 48) - (u128::from(div) << 47);
            let re0 = (target / u128::from(gain)) as u64;
            for re in [re0.saturating_sub(1), re0, re0 + 1] {
                // The domain: `re < 2^62` and `gain re / (div 2^48) < 2^48.01`.
                if re < (1 << 62) && u128::from(gain) * u128::from(re) < u128::from(div) << 96 {
                    assert_eq!(
                        release_q(gain, re, div),
                        release_exact(gain, re, div),
                        "g {gain} re {re} div {div}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 5_000_000, "{checked}");
    }
}
