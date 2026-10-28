//! Fallback implementation of the `fm_mod` kernel.

use crate::Fallback;
use crate::error::Result;
use crate::fallback::mix::{cos_sin_f32, cos_sin_q15};
use crate::kernels::check_len;
use crate::kernels::modulate::{FmMod, FmModState};
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
    let v = x * k + 0.5;
    if libm::fabs(v) < MAGIC_RANGE {
        let m = v + MAGIC;
        let up = u32::from(m - MAGIC > v);
        (m.to_bits() as u32).wrapping_sub(up)
    } else {
        phase_step_wide(v)
    }
}

/// [`phase_step`] of `v = x * k + 0.5` outside the range of the fast path.
#[cold]
#[inline(never)]
fn phase_step_wide(v: f64) -> u32 {
    (libm::floor(v) as i64) as u32
}

/// Samples per run of [`modulate`].
const RUN: usize = 64;

/// The modulator in runs of [`RUN`] samples: first the running phases of the run, then the
/// oscillator values `osc(phase)`. The phases are the only chain from sample to sample; with
/// the lookups in a loop of their own, an out-of-order core overlaps them freely instead of
/// waiting on the long chain of one sample (step, phase, table, interpolation).
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
        for (q, &x) in phases.iter_mut().zip(xr) {
            p = p.wrapping_add(phase_step(x.to_f64(), k));
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
