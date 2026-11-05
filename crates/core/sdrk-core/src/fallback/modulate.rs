//! Fallback implementations of the `fm_mod` and `am_mod` kernels.

use crate::Fallback;
use crate::error::Result;
use crate::fallback::mix::{cos_sin_f32, cos_sin_q15};
use crate::kernels::check_len;
use crate::kernels::modulate::{AmMod, FmMod, FmModState, check_am_params};
use crate::types::{C32, CQ15, Q15, RealSample};

/// `1.5 * 2^52`: adding it to an `f64` of magnitude below `2^51` rounds to the nearest integer
/// and leaves that integer in two's complement in the low mantissa bits.
const MAGIC: f64 = 6_755_399_441_055_744.0;
/// `2^51`, the magnitude below which [`MAGIC`] gives the integer exactly.
const MAGIC_RANGE: f64 = 2_251_799_813_685_248.0;

/// The phase step `floor(x * k + 0.5)` as `i64`, wrapped into `u32`.
///
/// For `|v| < 2^51` (every finite step of an ordinary signal) the floor is taken without
/// `libm::floor`, whose portable bit manipulation has several data-dependent branches, and
/// without the saturating `f64` to `i64` conversion: `v + MAGIC` rounds to the nearest integer
/// `n`, whose low 32 bits are the wrapped step; where that rounded up (`n > v`) one is
/// subtracted. The result equals `(libm::floor(v) as i64) as u32` bit for bit; larger
/// magnitudes, the infinities and NaN take that formula.
#[inline]
pub(crate) fn phase_step(x: f64, k: f64) -> u32 {
    match phase_step_fast(x, k) {
        (step, true) => step,
        (_, false) => phase_step_wide(x * k + 0.5),
    }
}

/// The fast path of [`phase_step`] without a branch: the step and whether `|x k + 0.5|` lies in
/// its range (`false` for NaN); outside the range the step is meaningless.
#[inline]
fn phase_step_fast(x: f64, k: f64) -> (u32, bool) {
    let v = x * k + 0.5;
    let m = v + MAGIC;
    let up = u32::from(m - MAGIC > v);
    (
        (m.to_bits() as u32).wrapping_sub(up),
        libm::fabs(v) < MAGIC_RANGE,
    )
}

/// [`phase_step`] of `v = x * k + 0.5` outside the range of the fast path.
#[cold]
#[inline(never)]
fn phase_step_wide(v: f64) -> u32 {
    (libm::floor(v) as i64) as u32
}

/// Samples per run of [`modulate`].
const RUN: usize = 64;

/// The modulator in runs of [`RUN`] samples: first the phase steps of the run, then the running
/// phases, then the oscillator values `osc(phase)`. The steps take the fast path of
/// [`phase_step`] for the whole run without a branch, so the compiler vectorizes them; only a
/// run with a value outside its range (NaN, infinite or huge input) computes them again with
/// [`phase_step`]. The phase sum is the only chain from sample to sample; with the steps and the
/// lookups in loops of their own, an out-of-order core overlaps them freely instead of waiting
/// on the long chain of one sample (step, phase, table, interpolation).
#[inline]
fn modulate<X: RealSample, Y>(
    k: f64,
    phase: &mut u32,
    input: &[X],
    output: &mut [Y],
    osc: impl Fn(u32) -> Result<Y>,
) -> Result<()> {
    let mut p = *phase;
    let mut phases = [0u32; RUN];
    for (xr, yr) in input.chunks(RUN).zip(output.chunks_mut(RUN)) {
        let mut fast = true;
        for (q, &x) in phases.iter_mut().zip(xr) {
            let (step, ok) = phase_step_fast(x.to_f64(), k);
            *q = step;
            fast &= ok;
        }
        if !fast {
            for (q, &x) in phases.iter_mut().zip(xr) {
                *q = phase_step(x.to_f64(), k);
            }
        }
        for q in phases.iter_mut().take(xr.len()) {
            p = p.wrapping_add(*q);
            *q = p;
        }
        for (y, &q) in yr.iter_mut().zip(&phases) {
            *y = osc(q)?;
        }
    }
    *phase = p;
    Ok(())
}

impl FmMod<C32> for Fallback {
    fn fm_mod(&self, s: &mut FmModState, input: &[f32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (k, phase) = s.parts_mut();
        modulate(k, phase, input, output, |p| {
            let (c, sn) = cos_sin_f32(p)?;
            Ok(C32::new(c, sn))
        })
    }
}

impl FmMod<CQ15> for Fallback {
    fn fm_mod(&self, s: &mut FmModState, input: &[Q15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (k, phase) = s.parts_mut();
        modulate(k, phase, input, output, cos_sin_q15)
    }
}

#[cfg(test)]
mod tests {
    //! The fast path of the phase step and the runs of the modulator must agree with the
    //! per-sample formula of the contract bit for bit.

    use std::vec;
    use std::vec::Vec;

    use super::{MAGIC_RANGE, RUN, phase_step};
    use crate::Fallback;
    use crate::fallback::mix::{cos_sin_f32, cos_sin_q15};
    use crate::kernels::modulate::{FmMod, FmModState};
    use crate::timing::Rate;
    use crate::types::{C32, CQ15, Q15, RealSample};

    /// The formula of the contract, `(floor(x * k + 0.5) as i64) as u32`.
    fn exact(x: f64, k: f64) -> u32 {
        (libm::floor(x * k + 0.5) as i64) as u32
    }

    #[test]
    fn phase_step_equals_the_floor_formula() {
        let mut values = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            2.5,
            -2.5,
            0.499_999_999_999_999_9,
            -0.499_999_999_999_999_9,
            MAGIC_RANGE,
            -MAGIC_RANGE,
            MAGIC_RANGE - 0.5,
            -MAGIC_RANGE + 0.5,
            MAGIC_RANGE - 1.0,
            -MAGIC_RANGE + 1.0,
            MAGIC_RANGE * 2.0,
            -MAGIC_RANGE * 2.0,
            4_294_967_295.5,
            -4_294_967_296.5,
            9.3e18,
            -9.3e18,
            1e300,
            -1e300,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        // The neighbours of integers and half integers over the magnitudes of the fast path and
        // at its bound.
        for base in [1.0, 3.0, 1e6, 1.34e9, 4.3e9, 1e15, 2.2e15, MAGIC_RANGE] {
            for d in [-1.0f64, -0.5, 0.0, 0.5, 1.0] {
                let v: f64 = base + d;
                for w in [v, -v] {
                    values.extend([w, w.next_up(), w.next_down()]);
                }
            }
        }
        // A pseudo-random sweep over many magnitudes, with fractions.
        let mut s = 0x2545_f491_4f6c_dd1d_u64;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let e = (s >> 58) as i32 - 8;
            let m = (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            values.push(m * libm::exp2(f64::from(e)));
        }
        for x in values {
            // `k = 1`: the step of `x` itself; `k = 0.5`; the scale of 75 kHz at 240 kHz.
            for k in [1.0, 0.5, 1_342_177_280.0] {
                assert_eq!(phase_step(x, k), exact(x, k), "x = {x:e}, k = {k}");
            }
        }
    }

    /// The modulator one sample at a time with the formula of the contract.
    fn per_sample<X: RealSample, Y>(
        k: f64,
        mut p: u32,
        x: &[X],
        osc: impl Fn(u32) -> Y,
    ) -> (Vec<Y>, u32) {
        let y = x
            .iter()
            .map(|&x| {
                p = p.wrapping_add(exact(x.to_f64(), k));
                osc(p)
            })
            .collect();
        (y, p)
    }

    #[test]
    #[allow(
        clippy::many_single_char_names,
        reason = "the names of the maths: x, q, k, s, n"
    )]
    fn runs_equal_the_per_sample_loop() {
        let rate = Rate::hz(240_000).unwrap();
        let mut seed = 0x9E37_79B9u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };
        // Every length around the run boundaries, with values that leave the fast path.
        for n in (0..=3 * RUN + 1).chain([5 * RUN - 1, 5 * RUN, 7 * RUN + 3]) {
            let x: Vec<f32> = (0..n)
                .map(|i| match i % 41 {
                    7 => f32::NAN,
                    19 => 3.0e9,
                    23 => f32::NEG_INFINITY,
                    _ => (f64::from(next()) / 2_147_483_648.0 - 1.0) as f32,
                })
                .collect();
            let q: Vec<Q15> = (0..n).map(|_| Q15(next() as i16)).collect();
            let start = next();
            let mut s = FmModState::new(75_000.0, rate).unwrap();
            s.set_phase(start);
            let k = s.scale();
            let (want, end) = per_sample(k, start, &x, |p| {
                let (c, sn) = cos_sin_f32(p).unwrap();
                C32::new(c, sn)
            });
            let mut got = vec![C32::default(); n];
            Fallback.fm_mod(&mut s, &x, &mut got).unwrap();
            let bits = |v: &[C32]| -> Vec<(u32, u32)> {
                v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
            };
            assert_eq!(bits(&got), bits(&want), "c32 n = {n}");
            assert_eq!(s.phase(), end, "c32 n = {n}");
            s.set_phase(start);
            let (want, end) = per_sample(k, start, &q, |p| cos_sin_q15(p).unwrap());
            let mut got = vec![CQ15::default(); n];
            Fallback.fm_mod(&mut s, &q, &mut got).unwrap();
            assert_eq!(got, want, "cq15 n = {n}");
            assert_eq!(s.phase(), end, "cq15 n = {n}");
        }
    }
}

/// One `am_mod` `C32` value: `carrier * (1 + depth * clamp(x, -1, 1))` in `f32` in the order of
/// the contract, NaN counting as 0. The clamp passes NaN through, and the mask of `x == x` turns
/// it into `+0` without a branch (a `-0` from the clamp or an `x` of `-0` gives `t = -0` or `0`
/// and `u = 1` either way), so the compilers vectorize the loop (`maxps`, `minps`, `cmpordps`
/// and `andps` on x86, `fmax`, `fmin`, `fcmeq` and `and` on NEON).
#[inline]
fn am_f32(carrier: f32, depth: f32, x: f32) -> f32 {
    let mask = 0u32.wrapping_sub(u32::from(!x.is_nan()));
    let x = f32::from_bits(x.clamp(-1.0, 1.0).to_bits() & mask);
    let t = depth * x;
    let u = 1.0 + t;
    carrier * u
}

impl AmMod<C32> for Fallback {
    fn am_mod(&self, carrier: f32, depth: f32, input: &[f32], output: &mut [C32]) -> Result<()> {
        check_am_params(carrier, depth)?;
        check_len(input.len(), output.len())?;
        let output: &mut [[f32; 2]] = bytemuck::cast_slice_mut(output);
        for (o, &x) in output.iter_mut().zip(input) {
            *o = [am_f32(carrier, depth, x), 0.0];
        }
        Ok(())
    }
}

/// The `Q15` gain `round(v * 32768)` limited to 32767 of `am_mod`, for `v` in `0 ..= 1`
/// (checked by the caller). `v * 32768 + 0.5` lies in `[0.5, 32768.5]`, and the truncating
/// conversion of a positive value is its floor, so this is `floor(v * 32768 + 0.5)` without the
/// call of `libm::floor`. The product is exact in `f64`; where the sum is not (`v` below
/// `2^-38`), it lies in `[0.5, 1)` and rounds within that range, so the floor is 0 either way.
#[inline]
fn am_gain(v: f32) -> i32 {
    ((f64::from(v) * 32768.0 + 0.5) as i32).min(32767)
}

/// One `am_mod` `CQ15` value in `i32`, bit for bit the `i64` formula of the contract
/// `sat16(round_shift(c (32768 + t), 15))` with `t = round_shift(m x, 15)`.
///
/// With `0 <= c, m <= 32767` and `-32768 <= x <= 32767`: `m x` lies in `[-2^30 + 2^15, 2^30]`,
/// so `t` lies in `[-32767, 32766]`. Because `c 32768` is a multiple of `2^15`,
/// `round_shift(c 32768 + c t, 15) = c + round_shift(c t, 15)`, and `|c t| < 2^30` fits. The sum
/// `c + round_shift(c t, 15)` is at least 0 (`c t >= -32767 c`), so only the upper limit of
/// `sat16` can act. The products are of 16-bit values (`t` fits `i16`), the form the compilers
/// vectorize with 16-bit multiplies; the SIMD backend computes the same two rounding products
/// with `pmulhrsw` / `sqrdmulh` / `q15mulr` and a saturating add. Unit test
/// `am_mod_i32_form_equals_the_i64_formula` checks every `x` against the formula.
#[inline]
fn am_q15(c_q: i32, m_q: i32, x: i16) -> i16 {
    let t = (m_q * i32::from(x) + (1 << 14)) >> 15;
    let r = (c_q * t + (1 << 14)) >> 15;
    (c_q + r).min(32767) as i16
}

impl AmMod<CQ15> for Fallback {
    fn am_mod(&self, carrier: f32, depth: f32, input: &[Q15], output: &mut [CQ15]) -> Result<()> {
        check_am_params(carrier, depth)?;
        check_len(input.len(), output.len())?;
        let (c_q, m_q) = (am_gain(carrier), am_gain(depth));
        let input: &[i16] = bytemuck::cast_slice(input);
        let output: &mut [[i16; 2]] = bytemuck::cast_slice_mut(output);
        for (o, &x) in output.iter_mut().zip(input) {
            *o = [am_q15(c_q, m_q, x), 0];
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod am_tests {
    use super::am_q15;
    use crate::num::{round_shift, sat16};

    fn formula(c: i64, m: i64, x: i64) -> i16 {
        let t = round_shift(m * x, 15);
        sat16(round_shift(c * (32768 + t), 15))
    }

    #[test]
    fn am_mod_gain_equals_the_rounded_product() {
        let mut v = 0.0f32;
        // Every f32 step near the rounding boundaries of small gains, then a coarse sweep.
        let mut values = std::vec![
            0.0f32,
            -0.0,
            1.0,
            1.0e-45,
            1.0e-30,
            0.999_984_7,
            0.999_999_94
        ];
        while v <= 1.0 {
            values.push(v);
            v += 1.0 / 65536.0;
            values.push(v + f32::EPSILON * v);
            values.push(v - f32::EPSILON * v);
        }
        for v in values.into_iter().filter(|v| (0.0..=1.0).contains(v)) {
            let want = (libm::floor(f64::from(v) * 32768.0 + 0.5) as i64).min(32767);
            assert_eq!(i64::from(super::am_gain(v)), want, "v={v}");
        }
    }

    #[test]
    fn am_mod_i32_form_equals_the_i64_formula() {
        let gains = [0, 1, 2, 3, 16383, 16384, 16385, 26214, 32766, 32767];
        let mut seed = 0x9e37_79b9_u32;
        let mut pairs = std::vec::Vec::new();
        for &c in &gains {
            for &m in &gains {
                pairs.push((c, m));
            }
        }
        for _ in 0..40 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            pairs.push(((seed & 0x7fff) as i32, ((seed >> 15) & 0x7fff) as i32));
        }
        for (c, m) in pairs {
            for x in i16::MIN..=i16::MAX {
                assert_eq!(
                    am_q15(c, m, x),
                    formula(i64::from(c), i64::from(m), i64::from(x)),
                    "c={c} m={m} x={x}"
                );
            }
        }
    }
}
