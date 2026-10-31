//! Suite cases of M23: the kernels `halfband_decimate`, `halfband_decimate_real` and
//! `real_to_complex`, their extreme inputs, the blocks `halfband` and `real_to_complex`, and the
//! end-to-end FM receive from a real 16-bit input.
//!
//! Every kernel case compares `Fallback` with `Reference`, the facade with `Fallback` and runs
//! the facade in 20 chunk plans; no files are read, so the cases run on microcontrollers too.
//! The `f32` kernels are tolerance kernels (110 dB against the reference and the fallback), the
//! Q15 kernels are bit-exact. A chunked run of the facade must equal the whole run bit for bit
//! for every type (R6.2).

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::large_stack_frames
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::design::{halfband_taps, kaiser_beta, to_f32, to_q15};
use sdrk_blocks::{
    ARENA_ALIGN, ArenaStorage, Block, BlockReal, BlockSample, HalfbandBlock, HalfbandConfig,
    HalfbandRealBlock, RealToComplexBlock,
};
use sdrk_core::kernels::convert::ConvertReal;
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate, ResampleReal};
use sdrk_core::kernels::demod::QuadDemod;
use sdrk_core::kernels::fused::{Ddc, FmDemod};
use sdrk_core::kernels::halfband::{
    HalfbandDecimate, HalfbandDecimateReal, HalfbandRealState, HalfbandState, RealToComplex,
    RealToComplexState,
};
use sdrk_core::{
    C32, CQ15, ComplexSample, Fallback, Q15, Rate, RealSample, Reference, Sample, Timing,
};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use super::m06::{flat, hand_out, min_snr};
use crate::arena::Arena;
use crate::e2e::E2eSample;
use crate::e2e_real::{
    REAL_FULL_SAMPLES, REAL_MIN_SINAD_F32_DB, REAL_MIN_SINAD_Q15_DB, REAL_QUICK_SAMPLES,
    measure_real_rx,
};
use crate::r#gen::extreme;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Input samples per kernel case (odd, so that the last output comes from an even index).
const N: usize = 1025;
/// Outputs of `N` inputs.
const OUT: usize = N.div_ceil(2);
/// An empty history.
const NO_HISTORY: [C32; 0] = [];
/// Chunk plans per case.
const PLANS: u64 = 20;
/// The tolerance of the `f32` kernels against the reference and the fallback, in dB.
const KERNEL_DB: f64 = 110.0;

#[allow(clippy::needless_pass_by_value)]
fn design_error(e: sdrk_blocks::Error) -> Failure {
    Failure::new("halfband design", "the filter design failed").detail(format_args!("{e}"))
}

/// The halfband taps (60 dB) of `N` taps as `f32`.
fn taps_f32<const M: usize>() -> Result<[f32; M], Failure> {
    let mut h = [0.0f64; M];
    halfband_taps(kaiser_beta(60.0), &mut h).map_err(design_error)?;
    let mut out = [0.0f32; M];
    to_f32(&h, &mut out).map_err(design_error)?;
    Ok(out)
}

/// The same taps as Q15 (the zero taps stay zero).
fn taps_q15<const M: usize>() -> Result<[Q15; M], Failure> {
    let mut h = [0.0f64; M];
    halfband_taps(kaiser_beta(60.0), &mut h).map_err(design_error)?;
    let mut out = [Q15(0); M];
    to_q15(&h, &mut out).map_err(design_error)?;
    Ok(out)
}

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Requires the SNR of `got` against `want` (flattened `f32` components) to reach `min` dB.
fn snr_ok(
    case: &'static str,
    msg: &'static str,
    want: &[f32],
    got: &[f32],
    min: f64,
) -> Result<(), Failure> {
    min_snr(case, msg, want, got, min)
}

fn real_flat(x: &[f32]) -> &[f32] {
    x
}

fn q15_exact<T: Sample>(
    case: &'static str,
    msg: &'static str,
    a: &[T],
    b: &[T],
) -> Result<(), Failure> {
    ensure(bit_exact(a, b).is_none(), || Failure::new(case, msg))
}

// ---------------------------------------------------------------------------------------
// halfband_decimate
// ---------------------------------------------------------------------------------------

/// Fallback against Reference, the facade against the fallback and 20 chunk plans of the facade
/// for `halfband_decimate` on the sample type `T`. `tolerant` selects the `f32` rules.
fn complex_case<T, const M: usize>(
    arena: &mut Arena<'_>,
    case: &'static str,
    taps: [T::Real; M],
    fill: fn(&mut [T], &mut Pcg32),
    tolerant: bool,
    seed: u64,
) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: HalfbandDecimate<T>,
    Fallback: HalfbandDecimate<T>,
    Reference: HalfbandDecimate<T>,
{
    let mut c = arena.carver();
    let x = c.take::<T>(N)?;
    let (fb, rf, fac) = (c.take::<T>(OUT)?, c.take::<T>(OUT)?, c.take::<T>(OUT)?);
    let (h1, h2, h3) = (
        c.take::<T>(M - 1)?,
        c.take::<T>(M - 1)?,
        c.take::<T>(M - 1)?,
    );
    fill(x, &mut Pcg32::new(seed, 1));
    let mut a = HalfbandState::<T, _, _>::new(taps, h1)?;
    let mut b = HalfbandState::<T, _, _>::new(taps, h2)?;
    let mut d = HalfbandState::<T, _, _>::new(taps, h3)?;
    let pa = HalfbandDecimate::<T>::halfband_decimate(&Fallback, &mut a, x, fb)?;
    let pb = HalfbandDecimate::<T>::halfband_decimate(&Reference, &mut b, x, rf)?;
    let pd = HalfbandDecimate::<T>::halfband_decimate(&Sdrk, &mut d, x, fac)?;
    ensure(
        [pa, pb, pd]
            .iter()
            .all(|p| p.consumed == N && p.produced == OUT),
        || Failure::new(case, "wrong progress"),
    )?;
    ensure(
        a.countdown() == b.countdown() && a.countdown() == d.countdown(),
        || Failure::new(case, "phase differs"),
    )?;
    if tolerant {
        let (r, f, s) = (as_f32(rf), as_f32(fb), as_f32(fac));
        snr_ok(case, "fallback below 110 dB", r, f, KERNEL_DB)?;
        snr_ok(case, "facade below 110 dB", f, s, KERNEL_DB)?;
    } else {
        q15_exact(case, "fallback differs from the reference", rf, fb)?;
        q15_exact(case, "facade differs from the fallback", fb, fac)?;
        ensure(
            a.history() == b.history() && a.history() == d.history(),
            || Failure::new(case, "history differs"),
        )?;
    }

    // Chunk plans: the output does not depend on how the input is cut.
    let (whole, chunked) = (c.take::<T>(OUT)?, c.take::<T>(OUT)?);
    let (g1, g2) = (c.take::<T>(M - 1)?, c.take::<T>(M - 1)?);
    let input: &[T] = x;
    for plan in 0..PLANS {
        let mut slots = hand_out(&mut *g1, &mut *g2);
        let p = ChunkPlan::new(seed + 100 + plan, 1 + (plan as usize * 37) % 400);
        let make = || HalfbandState::<T, _, _>::new(taps, slots());
        let step = |st: &mut Result<HalfbandState<T, [T::Real; M], &mut [T]>, sdrk_core::Error>,
                    i: &[T],
                    o: &mut [T]| match st {
            Ok(st) => HalfbandDecimate::<T>::halfband_decimate(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        };
        check_block_invariance(input, whole, chunked, p, make, step)?;
    }
    Ok(())
}

fn as_f32<T: bytemuck::Pod>(x: &[T]) -> &[f32] {
    bytemuck::try_cast_slice(x).unwrap_or_default()
}

fn fill_c32(x: &mut [C32], rng: &mut Pcg32) {
    for s in x.iter_mut() {
        *s = random_c32(rng, 0.3);
    }
}

fn fill_cq15(x: &mut [CQ15], rng: &mut Pcg32) {
    for s in x.iter_mut() {
        *s = random_cq15(rng, 1.0);
    }
}

pub(super) fn halfband_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    complex_case::<C32, 23>(
        arena,
        "halfband_c32",
        taps_f32::<23>()?,
        fill_c32,
        true,
        2310,
    )?;
    // A constant settles at the DC gain of 1 (the taps sum to 1).
    let mut c = arena.carver();
    let (x, y) = (c.take::<C32>(100)?, c.take::<C32>(50)?);
    x.fill(C32::new(0.5, -0.25));
    let mut st = HalfbandState::<C32, _, _>::new(taps_f32::<23>()?, c.take::<C32>(22)?)?;
    HalfbandDecimate::<C32>::halfband_decimate(&Sdrk, &mut st, x, y)?;
    let last = y.last().copied().unwrap_or_default();
    ensure(
        (last.re - 0.5).abs() < 1e-5 && (last.im + 0.25).abs() < 1e-5,
        || Failure::new("halfband_c32", "the DC gain is not 1"),
    )
}

pub(super) fn halfband_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "halfband_cq15";
    complex_case::<CQ15, 23>(arena, CASE, taps_q15::<23>()?, fill_cq15, false, 2311)?;
    // The same result as `fir_decimate` by 2 with the same (expanded) taps, bit for bit.
    let taps = taps_q15::<23>()?;
    let mut c = arena.carver();
    let x = c.take::<CQ15>(N)?;
    let (hb, fir) = (c.take::<CQ15>(OUT)?, c.take::<CQ15>(OUT)?);
    fill_cq15(x, &mut Pcg32::new(2312, 1));
    let mut a = HalfbandState::<CQ15, _, _>::new(taps, c.take::<CQ15>(22)?)?;
    let mut b = FirDecimState::<CQ15, _, _>::new(taps, c.take::<CQ15>(22)?, 2)?;
    HalfbandDecimate::<CQ15>::halfband_decimate(&Sdrk, &mut a, x, hb)?;
    FirDecimate::<CQ15>::fir_decimate(&Fallback, &mut b, x, fir)?;
    q15_exact(CASE, "differs from fir_decimate by 2", hb, fir)?;
    ensure(a.timing_matches(&b), || {
        Failure::new(CASE, "timing differs")
    })
}

trait TimingMatches<B> {
    fn timing_matches(&self, other: &B) -> bool;
}

impl<A: Timing, B: Timing> TimingMatches<B> for A {
    fn timing_matches(&self, other: &B) -> bool {
        self.ratio() == other.ratio()
            && self.group_delay().reduced() == other.group_delay().reduced()
    }
}

// ---------------------------------------------------------------------------------------
// halfband_decimate_real
// ---------------------------------------------------------------------------------------

fn real_case<T, const M: usize>(
    arena: &mut Arena<'_>,
    case: &'static str,
    taps: [T; M],
    fill: fn(&mut [T], &mut Pcg32),
    tolerant: bool,
    seed: u64,
) -> Result<(), Failure>
where
    T: RealSample,
    Sdrk: HalfbandDecimateReal<T>,
    Fallback: HalfbandDecimateReal<T>,
    Reference: HalfbandDecimateReal<T>,
{
    let mut c = arena.carver();
    let x = c.take::<T>(N)?;
    let (fb, rf, fac) = (c.take::<T>(OUT)?, c.take::<T>(OUT)?, c.take::<T>(OUT)?);
    let (h1, h2, h3) = (
        c.take::<T>(M - 1)?,
        c.take::<T>(M - 1)?,
        c.take::<T>(M - 1)?,
    );
    fill(x, &mut Pcg32::new(seed, 1));
    let mut a = HalfbandRealState::<T, _, _>::new(taps, h1)?;
    let mut b = HalfbandRealState::<T, _, _>::new(taps, h2)?;
    let mut d = HalfbandRealState::<T, _, _>::new(taps, h3)?;
    let pa = HalfbandDecimateReal::<T>::halfband_decimate_real(&Fallback, &mut a, x, fb)?;
    let pb = HalfbandDecimateReal::<T>::halfband_decimate_real(&Reference, &mut b, x, rf)?;
    let pd = HalfbandDecimateReal::<T>::halfband_decimate_real(&Sdrk, &mut d, x, fac)?;
    ensure(
        [pa, pb, pd]
            .iter()
            .all(|p| p.consumed == N && p.produced == OUT),
        || Failure::new(case, "wrong progress"),
    )?;
    if tolerant {
        let (r, f, s) = (as_f32(rf), as_f32(fb), as_f32(fac));
        snr_ok(case, "fallback below 110 dB", r, f, KERNEL_DB)?;
        snr_ok(case, "facade below 110 dB", f, s, KERNEL_DB)?;
    } else {
        q15_exact(case, "fallback differs from the reference", rf, fb)?;
        q15_exact(case, "facade differs from the fallback", fb, fac)?;
        ensure(
            a.history() == b.history() && a.history() == d.history(),
            || Failure::new(case, "history differs"),
        )?;
    }
    let (whole, chunked) = (c.take::<T>(OUT)?, c.take::<T>(OUT)?);
    let (g1, g2) = (c.take::<T>(M - 1)?, c.take::<T>(M - 1)?);
    let input: &[T] = x;
    for plan in 0..PLANS {
        let mut slots = hand_out(&mut *g1, &mut *g2);
        let p = ChunkPlan::new(seed + 100 + plan, 1 + (plan as usize * 41) % 400);
        let make = || HalfbandRealState::<T, _, _>::new(taps, slots());
        let step = |st: &mut Result<HalfbandRealState<T, [T; M], &mut [T]>, sdrk_core::Error>,
                    i: &[T],
                    o: &mut [T]| match st {
            Ok(st) => HalfbandDecimateReal::<T>::halfband_decimate_real(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        };
        check_block_invariance(input, whole, chunked, p, make, step)?;
    }
    Ok(())
}

pub(super) fn halfband_real_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    real_case::<f32, 23>(
        arena,
        "halfband_real_f32",
        taps_f32::<23>()?,
        |x, rng| {
            for s in x.iter_mut() {
                *s = random_c32(rng, 0.3).re;
            }
        },
        true,
        2320,
    )?;
    let mut c = arena.carver();
    let (x, y) = (c.take::<f32>(100)?, c.take::<f32>(50)?);
    x.fill(0.5);
    let mut st = HalfbandRealState::<f32, _, _>::new(taps_f32::<23>()?, c.take::<f32>(22)?)?;
    HalfbandDecimateReal::<f32>::halfband_decimate_real(&Sdrk, &mut st, x, y)?;
    ensure(
        (y.last().copied().unwrap_or_default() - 0.5).abs() < 1e-5,
        || Failure::new("halfband_real_f32", "the DC gain is not 1"),
    )
}

pub(super) fn halfband_real_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    real_case::<Q15, 23>(
        arena,
        "halfband_real_q15",
        taps_q15::<23>()?,
        |x, rng| {
            for s in x.iter_mut() {
                *s = random_cq15(rng, 1.0).re;
            }
        },
        false,
        2321,
    )
}

// ---------------------------------------------------------------------------------------
// real_to_complex
// ---------------------------------------------------------------------------------------

/// `x[n] (-j)^n` with the saturating negation of Q15.
fn shift<T: ComplexSample>(x: &[T::Real], out: &mut [T])
where
    T::Real: Neg,
{
    let zero = <T::Real as Sample>::ZERO;
    for (n, (v, o)) in x.iter().zip(out.iter_mut()).enumerate() {
        *o = match n & 3 {
            0 => T::new(*v, zero),
            1 => T::new(zero, v.neg_sat()),
            2 => T::new(v.neg_sat(), zero),
            _ => T::new(zero, *v),
        };
    }
}

trait Neg: Copy {
    fn neg_sat(self) -> Self;
}

impl Neg for f32 {
    fn neg_sat(self) -> f32 {
        -self
    }
}

impl Neg for Q15 {
    fn neg_sat(self) -> Q15 {
        Q15(self.0.saturating_neg())
    }
}

fn r2c_case<T, const M: usize>(
    arena: &mut Arena<'_>,
    case: &'static str,
    taps: [T::Real; M],
    fill: fn(&mut [T::Real], &mut Pcg32),
    tolerant: bool,
    seed: u64,
) -> Result<(), Failure>
where
    T: ComplexSample,
    T::Real: Neg,
    Sdrk: RealToComplex<T> + HalfbandDecimate<T>,
    Fallback: RealToComplex<T> + HalfbandDecimate<T>,
    Reference: RealToComplex<T>,
{
    let mut c = arena.carver();
    let x = c.take::<T::Real>(N)?;
    let (fb, rf, fac) = (c.take::<T>(OUT)?, c.take::<T>(OUT)?, c.take::<T>(OUT)?);
    let (h1, h2, h3) = (
        c.take::<T>(M - 1)?,
        c.take::<T>(M - 1)?,
        c.take::<T>(M - 1)?,
    );
    fill(x, &mut Pcg32::new(seed, 1));
    let mut a = RealToComplexState::<T, _, _>::new(taps, h1)?;
    let mut b = RealToComplexState::<T, _, _>::new(taps, h2)?;
    let mut d = RealToComplexState::<T, _, _>::new(taps, h3)?;
    let pa = RealToComplex::<T>::real_to_complex(&Fallback, &mut a, x, fb)?;
    let pb = RealToComplex::<T>::real_to_complex(&Reference, &mut b, x, rf)?;
    let pd = RealToComplex::<T>::real_to_complex(&Sdrk, &mut d, x, fac)?;
    ensure(
        [pa, pb, pd]
            .iter()
            .all(|p| p.consumed == N && p.produced == OUT),
        || Failure::new(case, "wrong progress"),
    )?;
    ensure(
        a.n_mod_4() == N % 4 && b.n_mod_4() == N % 4 && d.n_mod_4() == N % 4,
        || Failure::new(case, "n mod 4 is not carried"),
    )?;

    // The definition: the shift, then `halfband_decimate` with the same taps.
    let (u, chain) = (c.take::<T>(N)?, c.take::<T>(OUT)?);
    shift::<T>(x, u);
    let mut e = HalfbandState::<T, _, _>::new(taps, c.take::<T>(M - 1)?)?;
    HalfbandDecimate::<T>::halfband_decimate(&Fallback, &mut e, u, chain)?;
    if tolerant {
        let (r, f, s, k) = (as_f32(rf), as_f32(fb), as_f32(fac), as_f32(chain));
        snr_ok(case, "fallback below 110 dB", r, f, KERNEL_DB)?;
        snr_ok(case, "facade below 110 dB", f, s, KERNEL_DB)?;
        snr_ok(
            case,
            "differs from shift and halfband_decimate",
            k,
            f,
            KERNEL_DB,
        )?;
    } else {
        q15_exact(case, "fallback differs from the reference", rf, fb)?;
        q15_exact(case, "facade differs from the fallback", fb, fac)?;
        q15_exact(case, "differs from shift and halfband_decimate", chain, fb)?;
        ensure(
            a.history() == b.history() && a.history() == d.history(),
            || Failure::new(case, "history differs"),
        )?;
    }

    let (whole, chunked) = (c.take::<T>(OUT)?, c.take::<T>(OUT)?);
    let (g1, g2) = (c.take::<T>(M - 1)?, c.take::<T>(M - 1)?);
    let input: &[T::Real] = x;
    for plan in 0..PLANS {
        let mut slots = hand_out(&mut *g1, &mut *g2);
        let p = ChunkPlan::new(seed + 100 + plan, 1 + (plan as usize * 43) % 400);
        let make = || RealToComplexState::<T, _, _>::new(taps, slots());
        let step =
            |st: &mut Result<RealToComplexState<T, [T::Real; M], &mut [T]>, sdrk_core::Error>,
             i: &[T::Real],
             o: &mut [T]| match st {
                Ok(st) => RealToComplex::<T>::real_to_complex(&Sdrk, st, i, o),
                Err(e) => Err(e.clone()),
            };
        check_block_invariance(input, whole, chunked, p, make, step)?;
    }
    Ok(())
}

pub(super) fn real_to_complex_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    r2c_case::<C32, 31>(
        arena,
        "real_to_complex_c32",
        taps_f32::<31>()?,
        |x, rng| {
            for s in x.iter_mut() {
                *s = random_c32(rng, 0.3).re;
            }
        },
        true,
        2330,
    )
}

pub(super) fn real_to_complex_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    r2c_case::<CQ15, 31>(
        arena,
        "real_to_complex_cq15",
        taps_q15::<31>()?,
        |x, rng| {
            for s in x.iter_mut() {
                *s = random_cq15(rng, 1.0).re;
            }
            // The most negative value, where the negation saturates, at every phase.
            for k in [1usize, 2, 5, 6, 9, 10, 100, 101] {
                if let Some(v) = x.get_mut(k) {
                    *v = Q15(-32768);
                }
            }
        },
        false,
        2331,
    )
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

pub(super) fn extremes_m23(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m23";
    const M: usize = 70;
    const T: usize = 23;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<C32>(M)?;
    let (fb, fac, rf) = (c.take::<C32>(M)?, c.take::<C32>(M)?, c.take::<C32>(M)?);
    let (rx, rfb, rfac) = (c.take::<f32>(M)?, c.take::<f32>(M)?, c.take::<f32>(M)?);
    let (zc, zr) = (c.take::<C32>(M)?, c.take::<C32>(M)?);
    let hf = taps_f32::<T>()?;
    let hq = taps_q15::<T>()?;

    // f32: silence gives silence, full scale and the special values give no panic and the same
    // bits in the facade and the fallback where they are finite.
    for kind in 0..3 {
        match kind {
            0 => extreme::silence(x),
            1 => {
                for (i, s) in x.iter_mut().enumerate() {
                    *s = if i % 2 == 0 {
                        C32::new(1.0, -1.0)
                    } else {
                        C32::new(-1.0, 1.0)
                    };
                }
            }
            _ => extreme::f32_specials_complex(x),
        }
        let mut a = HalfbandState::<C32, _, _>::new(hf, [C32::default(); T - 1])?;
        let mut b = HalfbandState::<C32, _, _>::new(hf, [C32::default(); T - 1])?;
        let mut d = HalfbandState::<C32, _, _>::new(hf, [C32::default(); T - 1])?;
        HalfbandDecimate::<C32>::halfband_decimate(&Fallback, &mut a, x, fb)?;
        HalfbandDecimate::<C32>::halfband_decimate(&Reference, &mut b, x, rf)?;
        HalfbandDecimate::<C32>::halfband_decimate(&Sdrk, &mut d, x, fac)?;
        match kind {
            0 => ensure(
                fb.iter()
                    .chain(fac.iter())
                    .chain(rf.iter())
                    .take(M / 2)
                    .all(|s| *s == C32::default())
                    && fac.iter().take(M / 2).all(|s| s.re == 0.0 && s.im == 0.0),
                || fail("halfband_decimate: silence did not give silence"),
            )?,
            1 => {
                ensure(
                    fb.iter()
                        .chain(fac.iter())
                        .take(M)
                        .all(|s| s.re.is_finite() && s.im.is_finite()),
                    || fail("halfband_decimate: full scale did not stay finite"),
                )?;
                min_snr(
                    CASE,
                    "full scale: facade differs",
                    flat(fb),
                    flat(fac),
                    KERNEL_DB,
                )?;
            }
            _ => {}
        }
        // real_to_complex and the real kernel on the same data.
        for (r, s) in rx.iter_mut().zip(x.iter()) {
            *r = s.re;
        }
        let mut s1 = RealToComplexState::<C32, _, _>::new(hf, [C32::default(); T - 1])?;
        let mut s2 = RealToComplexState::<C32, _, _>::new(hf, [C32::default(); T - 1])?;
        RealToComplex::<C32>::real_to_complex(&Fallback, &mut s1, rx, zc)?;
        RealToComplex::<C32>::real_to_complex(&Sdrk, &mut s2, rx, zr)?;
        if kind == 0 {
            ensure(
                zc.iter().take(M / 2).all(|s| s.re == 0.0 && s.im == 0.0),
                || fail("real_to_complex: silence did not give silence"),
            )?;
        }
        let mut r1 = HalfbandRealState::<f32, _, _>::new(hf, [0.0f32; T - 1])?;
        let mut r2 = HalfbandRealState::<f32, _, _>::new(hf, [0.0f32; T - 1])?;
        HalfbandDecimateReal::<f32>::halfband_decimate_real(&Fallback, &mut r1, rx, rfb)?;
        HalfbandDecimateReal::<f32>::halfband_decimate_real(&Sdrk, &mut r2, rx, rfac)?;
        if kind == 1 {
            ensure(
                rfb.iter().take(M / 2).all(|v| v.is_finite())
                    && rfac.iter().take(M / 2).all(|v| v.is_finite()),
                || fail("halfband_decimate_real: full scale did not stay finite"),
            )?;
            min_snr(
                CASE,
                "real full scale: facade differs",
                real_flat(rfb.get(..M / 2).unwrap_or_default()),
                real_flat(rfac.get(..M / 2).unwrap_or_default()),
                KERNEL_DB,
            )?;
        }
        // NaN or infinity poisons the output of a finite window only locally: a `reset` clears
        // the state, and silence afterwards is silence.
        if kind == 2 {
            r1.reset();
            let mut out = [1.0f32; 8];
            HalfbandDecimateReal::<f32>::halfband_decimate_real(
                &Fallback,
                &mut r1,
                &[0.0f32; 16],
                &mut out,
            )?;
            ensure(out.iter().all(|v| v.to_bits() == 0), || {
                fail("halfband_decimate_real: reset did not clear the state")
            })?;
            ensure(
                fb.iter().zip(fac.iter()).all(|(p, q)| {
                    // Where the fallback is finite, so is the facade.
                    !(p.re.is_finite() && p.im.is_finite())
                        || (q.re.is_finite() && q.im.is_finite())
                        || same_f32(p.re, q.re)
                }),
                || fail("halfband_decimate: facade not finite where the fallback is"),
            )?;
        }
    }

    // Q15: silence, full scale and the corners of the raw range; all three kernels equal the
    // reference bit for bit.
    let corners = [-32768i16, -32767, -1, 0, 1, 32766, 32767];
    let q = c.take::<CQ15>(M)?;
    let (q_fb, q_rf, q_fac) = (c.take::<CQ15>(M)?, c.take::<CQ15>(M)?, c.take::<CQ15>(M)?);
    let (rq, rq_fb, rq_rf, rq_fac) = (
        c.take::<Q15>(M)?,
        c.take::<Q15>(M)?,
        c.take::<Q15>(M)?,
        c.take::<Q15>(M)?,
    );
    for kind in 0..3 {
        match kind {
            0 => extreme::silence(q),
            1 => extreme::full_scale_alternating(q),
            _ => {
                let mut k = 0usize;
                'fill: for re in corners {
                    for im in corners {
                        let Some(slot) = q.get_mut(k) else {
                            break 'fill;
                        };
                        *slot = CQ15 {
                            re: Q15(re),
                            im: Q15(im),
                        };
                        k += 1;
                    }
                }
            }
        }
        let mut a = HalfbandState::<CQ15, _, _>::new(hq, [CQ15::default(); T - 1])?;
        let mut b = HalfbandState::<CQ15, _, _>::new(hq, [CQ15::default(); T - 1])?;
        let mut d = HalfbandState::<CQ15, _, _>::new(hq, [CQ15::default(); T - 1])?;
        HalfbandDecimate::<CQ15>::halfband_decimate(&Fallback, &mut a, q, q_fb)?;
        HalfbandDecimate::<CQ15>::halfband_decimate(&Reference, &mut b, q, q_rf)?;
        HalfbandDecimate::<CQ15>::halfband_decimate(&Sdrk, &mut d, q, q_fac)?;
        ensure(
            bit_exact(q_fb, q_rf).is_none()
                && bit_exact(q_fb, q_fac).is_none()
                && a.history() == b.history()
                && a.history() == d.history(),
            || fail("halfband_decimate: Q15 differs between implementations"),
        )?;
        for (r, s) in rq.iter_mut().zip(q.iter()) {
            *r = s.re;
        }
        let mut r1 = HalfbandRealState::<Q15, _, _>::new(hq, [Q15(0); T - 1])?;
        let mut r2 = HalfbandRealState::<Q15, _, _>::new(hq, [Q15(0); T - 1])?;
        let mut r3 = HalfbandRealState::<Q15, _, _>::new(hq, [Q15(0); T - 1])?;
        HalfbandDecimateReal::<Q15>::halfband_decimate_real(&Fallback, &mut r1, rq, rq_fb)?;
        HalfbandDecimateReal::<Q15>::halfband_decimate_real(&Reference, &mut r2, rq, rq_rf)?;
        HalfbandDecimateReal::<Q15>::halfband_decimate_real(&Sdrk, &mut r3, rq, rq_fac)?;
        ensure(
            bit_exact(rq_fb, rq_rf).is_none() && bit_exact(rq_fb, rq_fac).is_none(),
            || fail("halfband_decimate_real: Q15 differs between implementations"),
        )?;
        let mut t1 = RealToComplexState::<CQ15, _, _>::new(hq, [CQ15::default(); T - 1])?;
        let mut t2 = RealToComplexState::<CQ15, _, _>::new(hq, [CQ15::default(); T - 1])?;
        let mut t3 = RealToComplexState::<CQ15, _, _>::new(hq, [CQ15::default(); T - 1])?;
        RealToComplex::<CQ15>::real_to_complex(&Fallback, &mut t1, rq, q_fb)?;
        RealToComplex::<CQ15>::real_to_complex(&Reference, &mut t2, rq, q_rf)?;
        RealToComplex::<CQ15>::real_to_complex(&Sdrk, &mut t3, rq, q_fac)?;
        ensure(
            bit_exact(q_fb, q_rf).is_none()
                && bit_exact(q_fb, q_fac).is_none()
                && t1.history() == t2.history()
                && t1.history() == t3.history(),
            || fail("real_to_complex: Q15 differs between implementations"),
        )?;
        if kind == 0 {
            ensure(
                q_fb.iter().take(M / 2).all(|s| *s == CQ15::default()),
                || fail("real_to_complex: Q15 silence did not give silence"),
            )?;
        }
    }

    // Invalid construction is an error, not a panic: a tap count other than 4 k + 3, a non-zero
    // tap at an even offset from the centre, a wrong history length.
    let bad_len = HalfbandState::<C32, _, _>::new([0.5f32; 5], [C32::default(); 4]).is_err()
        && HalfbandState::<C32, _, _>::new([0.0f32; 1], NO_HISTORY).is_err();
    let bad_taps = HalfbandState::<C32, _, _>::new(
        [0.0f32, 0.1, 0.0, 0.5, 0.0, 0.0, 0.0],
        [C32::default(); 6],
    )
    .is_err()
        && HalfbandRealState::<f32, _, _>::new([0.0f32, 0.0, 0.0, 0.5, 0.0, 0.1, 0.0], [0.0f32; 6])
            .is_err()
        && RealToComplexState::<CQ15, _, _>::new(
            [Q15(0), Q15(1), Q15(0), Q15(0), Q15(0), Q15(0), Q15(0)],
            [CQ15::default(); 6],
        )
        .is_err();
    let bad_history =
        HalfbandState::<C32, _, _>::new([0.25f32, 0.5, 0.25], [C32::default(); 3]).is_err();
    ensure(bad_len && bad_taps && bad_history, || {
        fail("invalid construction accepted")
    })?;
    ensure(
        HalfbandState::<C32, _, _>::new([0.25f32, 0.5, 0.25], [C32::default(); 2]).is_ok(),
        || fail("a valid three-tap halfband was rejected"),
    )
}

// ---------------------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------------------

const FS: u64 = 48_000;
/// Input samples of the block cases.
const BN: usize = 4096;
/// Output samples measured at the end of a block run.
const WINDOW: usize = 1024;

fn rate() -> Result<Rate, Failure> {
    Ok(Rate::hz(FS)?)
}

/// Mean power of the last [`WINDOW`] samples.
fn tail_power<T: ComplexSample>(y: &[T]) -> f64 {
    let tail = y.get(y.len().saturating_sub(WINDOW)..).unwrap_or_default();
    tail.iter().map(|s| s.to_c64().norm_sqr()).sum::<f64>() / tail.len().max(1) as f64
}

fn tail_power_real<T: RealSample>(y: &[T]) -> f64 {
    let tail = y.get(y.len().saturating_sub(WINDOW)..).unwrap_or_default();
    tail.iter().map(|s| s.to_f64() * s.to_f64()).sum::<f64>() / tail.len().max(1) as f64
}

/// Peak position of an impulse response against the promised group delay (in input samples).
fn delay_error(peak_out_index: usize, impulse_at: usize, delay: sdrk_core::Delay) -> f64 {
    let reduced = delay.reduced();
    let promised = reduced.num() as f64 / reduced.den() as f64;
    (2.0 * peak_out_index as f64 - impulse_at as f64) - promised
}

const ALIAS: &str = "block_halfband_alias";

/// A tone in the stop band is attenuated by at least `attenuation_db - 3` dB; the pass band is
/// flat; the impulse peak sits at the promised delay (+-0.5 sample).
fn alias_complex<T: BlockSample>(arena: &mut Arena<'_>) -> Result<(), Failure>
where
    Sdrk: HalfbandDecimate<T>,
{
    let fail = |msg: &'static str| Failure::new(ALIAS, msg);
    let cfg = HalfbandConfig::default();
    let r = rate()?;
    let fs = FS as f64;
    let bytes = HalfbandBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, r)? + ARENA_ALIGN;
    let mut c = arena.carver();
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let mut block = HalfbandBlock::<T, _>::new(&cfg, r, &mut store)?;
    let (x, y) = (c.take::<T>(BN)?, c.take::<T>(BN / 2 + 2)?);
    let limit = libm::pow(10.0, -(cfg.attenuation_db - 3.0) / 10.0);
    // The stop band starts at fs / 4 + transition / 2 = 0.3 fs.
    for (f, stop) in [(0.31, true), (0.38, true), (0.45, true), (0.15, false)] {
        block.reset();
        for (i, s) in x.iter_mut().enumerate() {
            let p = core::f64::consts::TAU * f * i as f64;
            *s = T::from_c64_round(Complex64::new(0.5 * libm::cos(p), 0.5 * libm::sin(p)));
        }
        let p = block.process(x, y)?;
        let out = y.get(..p.produced).unwrap_or_default();
        let ratio = tail_power(out) / 0.25;
        if stop {
            ensure(ratio <= limit, || {
                fail("a stop-band tone is attenuated by less than attenuation - 3 dB")
                    .detail(format_args!("{} Hz: power ratio {ratio}", f * fs))
            })?;
        } else {
            ensure((ratio - 1.0).abs() < 0.02, || {
                fail("the pass band is not flat").detail(format_args!("{} Hz: {ratio}", f * fs))
            })?;
        }
    }
    // Timing: an impulse at input index 1 peaks at the promised delay.
    block.reset();
    x.fill(T::default());
    if let Some(s) = x.get_mut(1) {
        *s = T::from_c64_round(Complex64::new(0.9, 0.0));
    }
    let p = block.process(x, y)?;
    let out = y.get(..p.produced).unwrap_or_default();
    let peak = out
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.to_c64().norm_sqr().total_cmp(&b.1.to_c64().norm_sqr()))
        .map_or(0, |(i, _)| i);
    let err = delay_error(peak, 1, block.timing().delay);
    ensure(err.abs() <= 0.5, || {
        fail("the impulse peak is off the promised group delay")
            .detail(format_args!("{err} samples"))
    })
}

fn alias_real<R: BlockReal>(arena: &mut Arena<'_>) -> Result<(), Failure>
where
    Sdrk: HalfbandDecimateReal<R>,
{
    let fail = |msg: &'static str| Failure::new(ALIAS, msg);
    let cfg = HalfbandConfig::default();
    let r = rate()?;
    let bytes = HalfbandRealBlock::<R, ArenaStorage<'_>>::storage_bytes(&cfg, r)? + ARENA_ALIGN;
    let mut c = arena.carver();
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let mut block = HalfbandRealBlock::<R, _>::new(&cfg, r, &mut store)?;
    let (x, y) = (c.take::<R>(BN)?, c.take::<R>(BN / 2 + 2)?);
    let limit = libm::pow(10.0, -(cfg.attenuation_db - 3.0) / 10.0);
    // A real tone at f in (0.3, 0.5) fs aliases into the pass band.
    for f in [0.31, 0.38, 0.45] {
        block.reset();
        for (i, s) in x.iter_mut().enumerate() {
            *s = R::from_f64_round(0.5 * libm::sin(core::f64::consts::TAU * f * i as f64));
        }
        let p = block.process(x, y)?;
        let ratio = tail_power_real(y.get(..p.produced).unwrap_or_default()) / 0.125;
        ensure(ratio <= limit, || {
            fail("a real stop-band tone is attenuated by less than attenuation - 3 dB")
                .detail(format_args!("{} Hz: power ratio {ratio}", f * FS as f64))
        })?;
    }
    Ok(())
}

pub(super) fn block_halfband_alias(arena: &mut Arena<'_>) -> Result<(), Failure> {
    alias_complex::<C32>(arena)?;
    alias_complex::<CQ15>(arena)?;
    alias_real::<f32>(arena)?;
    alias_real::<Q15>(arena)
}

const TONE: &str = "block_real_to_complex_tone";

/// Power of `y` at the normalised frequency `g` (cycles per sample), single-bin DFT over the
/// last [`WINDOW`] samples, as an amplitude.
fn bin_amplitude<T: ComplexSample>(y: &[T], g: f64) -> f64 {
    let tail = y.get(y.len().saturating_sub(WINDOW)..).unwrap_or_default();
    let mut acc = Complex64::new(0.0, 0.0);
    for (i, s) in tail.iter().enumerate() {
        let p = -core::f64::consts::TAU * g * i as f64;
        acc += s.to_c64() * Complex64::new(libm::cos(p), libm::sin(p));
    }
    libm::sqrt(acc.norm_sqr()) / tail.len().max(1) as f64
}

/// A real tone at `f` appears at `f - fs/4` of the output (rate `fs/2`) with half its amplitude;
/// the mirror at `-f` is at least `attenuation_db - 3` dB down.
fn tone_for<T: BlockSample>(arena: &mut Arena<'_>) -> Result<(), Failure>
where
    Sdrk: RealToComplex<T>,
    T::Real: BlockReal,
{
    let fail = |msg: &'static str| Failure::new(TONE, msg);
    let cfg = HalfbandConfig::default();
    let r = rate()?;
    let fs = FS as f64;
    let bytes = RealToComplexBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, r)? + ARENA_ALIGN;
    let mut c = arena.carver();
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let mut block = RealToComplexBlock::<T, _>::new(&cfg, r, &mut store)?;
    let (x, y) = (c.take::<T::Real>(BN)?, c.take::<T>(BN / 2 + 2)?);
    // The usable band is fs/4 +- (fs/4 - transition/2) around the output centre.
    for f in [0.10, 0.33, 0.40] {
        block.reset();
        for (i, s) in x.iter_mut().enumerate() {
            *s = <T::Real as RealSample>::from_f64_round(
                0.5 * libm::cos(core::f64::consts::TAU * f * i as f64),
            );
        }
        let p = block.process(x, y)?;
        let out = y.get(..p.produced).unwrap_or_default();
        // Output frequency in cycles per output sample: (f - 1/4) / (1/2).
        let wanted_bin = (f - 0.25) * 2.0;
        let wanted = bin_amplitude(out, wanted_bin);
        let mirror = bin_amplitude(out, (-f - 0.25) * 2.0);
        let off = bin_amplitude(out, wanted_bin + 4.0 / WINDOW as f64);
        let limit = libm::pow(10.0, -(cfg.attenuation_db - 3.0) / 20.0);
        let in_band = libm::fabs(f - 0.25) <= 0.25 - 0.05;
        ensure(!in_band || libm::fabs(wanted - 0.25) <= 0.01, || {
            fail("the tone is not at f - fs/4 with half the amplitude")
                .detail(format_args!("{} Hz: {wanted}", f * fs))
        })?;
        ensure(off < 0.3 * wanted.max(1e-9) || !in_band, || {
            fail("the tone is not narrow at f - fs/4")
        })?;
        ensure(mirror <= limit * 0.5, || {
            fail("the mirror is not attenuated by attenuation - 3 dB")
                .detail(format_args!("{} Hz: {mirror}", f * fs))
        })?;
    }
    Ok(())
}

pub(super) fn block_real_to_complex_tone(arena: &mut Arena<'_>) -> Result<(), Failure> {
    tone_for::<C32>(arena)?;
    tone_for::<CQ15>(arena)
}

// ---------------------------------------------------------------------------------------
// end to end
// ---------------------------------------------------------------------------------------

fn real_rx_case<T>(
    case: &'static str,
    arena: &mut Arena<'_>,
    samples: usize,
    min_sinad_db: f64,
) -> Result<(), Failure>
where
    T: E2eSample + BlockSample,
    <T as ComplexSample>::Real: BlockReal,
    Sdrk: ConvertReal<<T as ComplexSample>::Real>
        + RealToComplex<T>
        + Ddc<T>
        + FmDemod<T>
        + QuadDemod<T>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    let m = measure_real_rx::<T>(arena, samples)?;
    ensure(m.audio_samples == samples / 100, || {
        Failure::new(case, "wrong audio sample count").detail(format_args!("{}", m.audio_samples))
    })?;
    ensure(m.sinad_db >= min_sinad_db, || {
        Failure::new(case, "SINAD below the threshold")
            .detail(format_args!("{} dB, need {min_sinad_db}", m.sinad_db))
    })?;
    ensure(libm::fabs(m.tone_hz - crate::e2e::AUDIO_HZ) <= 0.5, || {
        Failure::new(case, "the audio tone is not at 1 kHz")
            .detail(format_args!("{} Hz", m.tone_hz))
    })
}

pub(super) fn e2e_real_fm_rx_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    real_rx_case::<C32>(
        "e2e_real_fm_rx_f32",
        arena,
        REAL_QUICK_SAMPLES,
        REAL_MIN_SINAD_F32_DB,
    )
}

pub(super) fn e2e_real_fm_rx_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    real_rx_case::<CQ15>(
        "e2e_real_fm_rx_q15",
        arena,
        REAL_QUICK_SAMPLES,
        REAL_MIN_SINAD_Q15_DB,
    )
}

pub(super) fn e2e_real_fm_rx_f32_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    real_rx_case::<C32>(
        "e2e_real_fm_rx_f32_1s",
        arena,
        REAL_FULL_SAMPLES,
        REAL_MIN_SINAD_F32_DB,
    )
}

pub(super) fn e2e_real_fm_rx_q15_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    real_rx_case::<CQ15>(
        "e2e_real_fm_rx_q15_1s",
        arena,
        REAL_FULL_SAMPLES,
        REAL_MIN_SINAD_Q15_DB,
    )
}
