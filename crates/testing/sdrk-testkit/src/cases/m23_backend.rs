//! Suite cases of the SIMD backend for the M23 kernels, run through the facade:
//! `backend_vs_fallback_halfband` and `backend_vs_fallback_real_to_complex`.
//!
//! The same matrix as `m13`: the lengths 0 to 4096 that fit the arena, full-scale corners for
//! the `Q15` kernels, and the block-size invariance (which includes the phase across calls of
//! `real_to_complex`). The tolerance kernels are measured against the reference (110 dB), the
//! bit-exact ones against the fallback. The cases use no files, so they also run on the
//! emulators of the microcontroller targets, where the facade is the fallback.

#![allow(clippy::many_single_char_names, clippy::too_many_lines)]

use core::f64::consts::PI;

use sdrk::Sdrk;
use sdrk_core::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState,
};
use sdrk_core::{C32, CQ15, Fallback, Q15, Reference};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use super::m06::{flat, min_snr};
use super::m13::{fail, lengths};
use crate::arena::Arena;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Taps of a windowed-sinc halfband filter (`N = 4 k + 3`): zero at the even offsets from the
/// centre, exactly symmetric, summing to one.
fn halfband_f32(out: &mut [f32]) {
    let n = out.len();
    let m = n.saturating_sub(1) / 2;
    let mut sum = 0.0;
    for (k, t) in out.iter_mut().enumerate() {
        let d = k.abs_diff(m);
        let v = if d == 0 {
            0.5
        } else if d % 2 == 0 {
            0.0
        } else {
            let x = PI * d as f64 / 2.0;
            let w = 0.54 + 0.46 * libm::cos(PI * d as f64 / (m + 1) as f64);
            libm::sin(x) / x * w * 0.5
        };
        *t = v as f32;
        sum += f64::from(*t);
    }
    for t in out.iter_mut() {
        *t = (f64::from(*t) / sum) as f32;
    }
}

fn halfband_q15(out: &mut [Q15], scratch: &mut [f32]) {
    halfband_f32(scratch);
    for (q, t) in out.iter_mut().zip(scratch.iter()) {
        *q = sdrk_core::num::q15_from_f64(f64::from(*t));
    }
}

pub(super) fn backend_vs_fallback_halfband(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_halfband";
    const N: usize = 1000;
    const TAPS: usize = 47;
    let mut rng = Pcg32::new(1330, 1);
    // `C32` and `f32` against the reference (110 dB).
    for n in lengths(arena.remaining(), 8 * 3 + 8 + 64) {
        let mut c = arena.carver();
        let input = c.take::<C32>(n)?;
        let (rf, fac) = (c.take::<C32>(n / 2 + 2)?, c.take::<C32>(n / 2 + 2)?);
        let taps = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<C32>(TAPS - 1)?, c.take::<C32>(TAPS - 1)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 0.9);
        }
        halfband_f32(taps);
        let mut a = HalfbandState::<C32, _, _>::new(&*taps, h1)?;
        let mut d = HalfbandState::<C32, _, _>::new(&*taps, h2)?;
        let pb = HalfbandDecimate::<C32>::halfband_decimate(&Reference, &mut a, input, rf)?;
        let pd = HalfbandDecimate::<C32>::halfband_decimate(&Sdrk, &mut d, input, fac)?;
        ensure(pb == pd, || fail(CASE, "progress differs", n))?;
        let k = pb.produced;
        if k > 0 {
            min_snr(
                CASE,
                "c32 below 110 dB",
                flat(rf.get(..k).unwrap_or_default()),
                flat(fac.get(..k).unwrap_or_default()),
                110.0,
            )?;
        }
        ensure(
            a.history() == d.history() && a.countdown() == d.countdown(),
            || fail(CASE, "c32 states differ", n),
        )?;
    }
    for n in lengths(arena.remaining(), 4 * 3 + 4 + 64) {
        let mut c = arena.carver();
        let input = c.take::<f32>(n)?;
        let (rf, fac) = (c.take::<f32>(n / 2 + 2)?, c.take::<f32>(n / 2 + 2)?);
        let taps = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<f32>(TAPS - 1)?, c.take::<f32>(TAPS - 1)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 0.9).re;
        }
        halfband_f32(taps);
        let mut a = HalfbandRealState::<f32, _, _>::new(&*taps, h1)?;
        let mut d = HalfbandRealState::<f32, _, _>::new(&*taps, h2)?;
        let pb =
            HalfbandDecimateReal::<f32>::halfband_decimate_real(&Reference, &mut a, input, rf)?;
        let pd = HalfbandDecimateReal::<f32>::halfband_decimate_real(&Sdrk, &mut d, input, fac)?;
        ensure(pb == pd, || fail(CASE, "f32 progress differs", n))?;
        let k = pb.produced;
        if k > 0 {
            min_snr(
                CASE,
                "f32 below 110 dB",
                rf.get(..k).unwrap_or_default(),
                fac.get(..k).unwrap_or_default(),
                110.0,
            )?;
        }
        ensure(
            a.history() == d.history() && a.countdown() == d.countdown(),
            || fail(CASE, "f32 states differ", n),
        )?;
    }
    // `CQ15` and `Q15` bit for bit against the fallback, with full-scale corners.
    for n in lengths(arena.remaining(), 4 * 3 + 4 + 64) {
        let mut c = arena.carver();
        let input = c.take::<CQ15>(n)?;
        let (fb, fac) = (c.take::<CQ15>(n / 2 + 2)?, c.take::<CQ15>(n / 2 + 2)?);
        let taps = c.take::<Q15>(TAPS)?;
        let scratch = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<CQ15>(TAPS - 1)?, c.take::<CQ15>(TAPS - 1)?);
        for s in input.iter_mut() {
            *s = random_cq15(&mut rng, 1.0);
        }
        if let Some(s) = input.get_mut(1) {
            *s = CQ15 {
                re: Q15(-32768),
                im: Q15(-32768),
            };
        }
        halfband_q15(taps, scratch);
        let mut a = HalfbandState::<CQ15, _, _>::new(&*taps, h1)?;
        let mut d = HalfbandState::<CQ15, _, _>::new(&*taps, h2)?;
        let pb = HalfbandDecimate::<CQ15>::halfband_decimate(&Fallback, &mut a, input, fb)?;
        let pd = HalfbandDecimate::<CQ15>::halfband_decimate(&Sdrk, &mut d, input, fac)?;
        let k = pb.produced;
        ensure(pb == pd, || fail(CASE, "cq15 progress differs", n))?;
        ensure(
            bit_exact(
                fb.get(..k).unwrap_or_default(),
                fac.get(..k).unwrap_or_default(),
            )
            .is_none(),
            || fail(CASE, "cq15 differs from the fallback", n),
        )?;
        ensure(
            a.history() == d.history() && a.countdown() == d.countdown(),
            || fail(CASE, "cq15 states differ", n),
        )?;
    }
    for n in lengths(arena.remaining(), 2 * 3 + 2 + 64) {
        let mut c = arena.carver();
        let input = c.take::<Q15>(n)?;
        let (fb, fac) = (c.take::<Q15>(n / 2 + 2)?, c.take::<Q15>(n / 2 + 2)?);
        let taps = c.take::<Q15>(TAPS)?;
        let scratch = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<Q15>(TAPS - 1)?, c.take::<Q15>(TAPS - 1)?);
        for s in input.iter_mut() {
            *s = random_cq15(&mut rng, 1.0).re;
        }
        if let Some(s) = input.get_mut(2) {
            *s = Q15(-32768);
        }
        halfband_q15(taps, scratch);
        let mut a = HalfbandRealState::<Q15, _, _>::new(&*taps, h1)?;
        let mut d = HalfbandRealState::<Q15, _, _>::new(&*taps, h2)?;
        let pb = HalfbandDecimateReal::<Q15>::halfband_decimate_real(&Fallback, &mut a, input, fb)?;
        let pd = HalfbandDecimateReal::<Q15>::halfband_decimate_real(&Sdrk, &mut d, input, fac)?;
        let k = pb.produced;
        ensure(pb == pd, || fail(CASE, "q15 progress differs", n))?;
        ensure(
            bit_exact(
                fb.get(..k).unwrap_or_default(),
                fac.get(..k).unwrap_or_default(),
            )
            .is_none(),
            || fail(CASE, "q15 differs from the fallback", n),
        )?;
        ensure(
            a.history() == d.history() && a.countdown() == d.countdown(),
            || fail(CASE, "q15 states differ", n),
        )?;
    }
    // The input is resubmitted when the output space is short; the bits do not depend on it.
    let mut c = arena.carver();
    let input = c.take::<C32>(N)?;
    let (whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?);
    let taps = c.take::<f32>(TAPS)?;
    for s in input.iter_mut() {
        *s = random_c32(&mut rng, 1.0);
    }
    halfband_f32(taps);
    let taps: &[f32] = taps;
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(1331, 61),
        || HalfbandState::<C32, _, _>::new(taps, [C32::ZERO; TAPS - 1]),
        |state, i, o| match state {
            Ok(st) => HalfbandDecimate::<C32>::halfband_decimate(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        },
    )?;
    let mut c = arena.carver();
    let input = c.take::<Q15>(N)?;
    let (whole, chunked) = (c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    let (taps, scratch) = (c.take::<Q15>(TAPS)?, c.take::<f32>(TAPS)?);
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 1.0).re;
    }
    halfband_q15(taps, scratch);
    let taps: &[Q15] = taps;
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(1332, 61),
        || HalfbandRealState::<Q15, _, _>::new(taps, [Q15(0); TAPS - 1]),
        |state, i, o| match state {
            Ok(st) => HalfbandDecimateReal::<Q15>::halfband_decimate_real(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

pub(super) fn backend_vs_fallback_real_to_complex(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_real_to_complex";
    const N: usize = 1000;
    const TAPS: usize = 63;
    let mut rng = Pcg32::new(1333, 1);
    for n in lengths(arena.remaining(), 4 * 3 + 8 + 64) {
        let mut c = arena.carver();
        let input = c.take::<f32>(n)?;
        let (rf, fac) = (c.take::<C32>(n / 2 + 2)?, c.take::<C32>(n / 2 + 2)?);
        let taps = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<C32>(TAPS - 1)?, c.take::<C32>(TAPS - 1)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 0.9).re;
        }
        halfband_f32(taps);
        let mut a = RealToComplexState::<C32, _, _>::new(&*taps, h1)?;
        let mut d = RealToComplexState::<C32, _, _>::new(&*taps, h2)?;
        let pb = RealToComplex::<C32>::real_to_complex(&Reference, &mut a, input, rf)?;
        let pd = RealToComplex::<C32>::real_to_complex(&Sdrk, &mut d, input, fac)?;
        ensure(pb == pd, || fail(CASE, "progress differs", n))?;
        let k = pb.produced;
        if k > 0 {
            min_snr(
                CASE,
                "c32 below 110 dB",
                flat(rf.get(..k).unwrap_or_default()),
                flat(fac.get(..k).unwrap_or_default()),
                110.0,
            )?;
        }
        ensure(
            a.history() == d.history() && a.n_mod_4() == d.n_mod_4(),
            || fail(CASE, "c32 states differ", n),
        )?;
    }
    for n in lengths(arena.remaining(), 2 * 3 + 4 + 64) {
        let mut c = arena.carver();
        let input = c.take::<Q15>(n)?;
        let (fb, fac) = (c.take::<CQ15>(n / 2 + 2)?, c.take::<CQ15>(n / 2 + 2)?);
        let taps = c.take::<Q15>(TAPS)?;
        let scratch = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<CQ15>(TAPS - 1)?, c.take::<CQ15>(TAPS - 1)?);
        for s in input.iter_mut() {
            *s = random_cq15(&mut rng, 1.0).re;
        }
        // The saturating negation of the shift at every phase.
        for (i, s) in input.iter_mut().enumerate().take(8) {
            if i % 2 == 0 {
                *s = Q15(-32768);
            }
        }
        halfband_q15(taps, scratch);
        let mut a = RealToComplexState::<CQ15, _, _>::new(&*taps, h1)?;
        let mut d = RealToComplexState::<CQ15, _, _>::new(&*taps, h2)?;
        let pb = RealToComplex::<CQ15>::real_to_complex(&Fallback, &mut a, input, fb)?;
        let pd = RealToComplex::<CQ15>::real_to_complex(&Sdrk, &mut d, input, fac)?;
        let k = pb.produced;
        ensure(pb == pd, || fail(CASE, "cq15 progress differs", n))?;
        ensure(
            bit_exact(
                fb.get(..k).unwrap_or_default(),
                fac.get(..k).unwrap_or_default(),
            )
            .is_none(),
            || fail(CASE, "cq15 differs from the fallback", n),
        )?;
        ensure(
            a.history() == d.history() && a.n_mod_4() == d.n_mod_4(),
            || fail(CASE, "cq15 states differ", n),
        )?;
    }
    // Block-size invariance, which includes the phase across calls.
    let mut c = arena.carver();
    let input = c.take::<f32>(N)?;
    let (whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?);
    let taps = c.take::<f32>(TAPS)?;
    for s in input.iter_mut() {
        *s = random_c32(&mut rng, 1.0).re;
    }
    halfband_f32(taps);
    let taps: &[f32] = taps;
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(1334, 61),
        || RealToComplexState::<C32, _, _>::new(taps, [C32::ZERO; TAPS - 1]),
        |state, i, o| match state {
            Ok(st) => RealToComplex::<C32>::real_to_complex(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        },
    )?;
    let mut c = arena.carver();
    let input = c.take::<Q15>(N)?;
    let (whole, chunked) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    let (taps, scratch) = (c.take::<Q15>(TAPS)?, c.take::<f32>(TAPS)?);
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 1.0).re;
    }
    halfband_q15(taps, scratch);
    let taps: &[Q15] = taps;
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(1335, 61),
        || RealToComplexState::<CQ15, _, _>::new(taps, [CQ15::default(); TAPS - 1]),
        |state, i, o| match state {
            Ok(st) => RealToComplex::<CQ15>::real_to_complex(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}
