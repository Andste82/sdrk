//! Suite cases of M24: the kernels `stereo_pll` and `stereo_matrix` and their extreme inputs.
//!
//! Every kernel case compares `Fallback` with `Reference`, the facade with `Fallback` and runs
//! the facade in 20 chunk plans; no files are read, so the cases run on microcontrollers too.
//! Both kernels are bit-exact in both types (R6.2: the output does not depend on the block
//! size).

#![allow(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_blocks::{BiquadElem, BlockSample};
use sdrk_core::kernels::biquad::{Biquad, BiquadReal};
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::FirDecimate;
use sdrk_core::kernels::demod::QuadDemod;
use sdrk_core::kernels::fused::Ddc;
use sdrk_core::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use sdrk_core::{
    C32, CQ15, ComplexSample, Error, Fallback, Q15, Rate, RealSample, Reference, SampleType,
};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use crate::arena::Arena;
use crate::e2e::E2eSample;
use crate::e2e_stereo::{
    FULL_SAMPLES, MIN_SEPARATION_DB, QUICK_SAMPLES, RATE, StereoSignal, check_mono, check_stereo,
    check_switching, measure_block_mono, measure_block_separation, measure_stereo,
};
use crate::invariance::ChunkPlan;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Sample rate of the kernel cases (the pilot is below `fs / 4`).
const FS: u64 = 80_000;
/// Length of the pilot signals: the lock detector (50 ms) needs about 2000 samples to switch on.
const N: usize = 2800;
/// Length of the matrix signals.
const NM: usize = 1500;
/// Chunk plans per type.
const PLANS: u64 = 20;

fn rate() -> Result<Rate, Failure> {
    Ok(Rate::hz(FS)?)
}

/// Nominal pilot level: 0.1 for `f32`; for `Q15` 0.1 of a signal scaled by 0.625.
fn pilot_level<T: RealSample>() -> f64 {
    if T::TYPE == SampleType::Q15 {
        0.0625
    } else {
        0.1
    }
}

fn state<T: RealSample>() -> Result<StereoPllState<T>, Failure> {
    Ok(StereoPllState::new(
        19_000.0,
        30.0,
        pilot_level::<T>(),
        rate()?,
    )?)
}

/// `sin(2 pi turns)` with the turns reduced exactly enough for the short signals.
fn sin_turns(turns: f64) -> f64 {
    libm::sin(2.0 * core::f64::consts::PI * (turns - libm::floor(turns)))
}

/// The multiplex signal of the milestone (L 1 kHz, R 3 kHz, pilot at 19 000.5 Hz, start phase
/// 1 rad) and the pilot term alone.
fn fill<T: RealSample>(mpx: &mut [T], pilot: &mut [T]) {
    let scale = if T::TYPE == SampleType::Q15 {
        0.625
    } else {
        1.0
    };
    let amp = pilot_level::<T>();
    for (n, (m, p)) in mpx.iter_mut().zip(pilot.iter_mut()).enumerate() {
        let t = n as f64 / FS as f64;
        let left = sin_turns(1000.0 * t);
        let right = sin_turns(3000.0 * t);
        let theta = 19_000.5 * t + 1.0 / (2.0 * core::f64::consts::PI);
        let pil = sin_turns(theta);
        let sub = sin_turns(2.0 * theta);
        *m = T::from_f64_round(scale * 0.45 * ((left + right) + (left - right) * sub) + amp * pil);
        *p = T::from_f64_round(amp * pil);
    }
}

fn split<X>(s: &[X], n: usize) -> Result<(&[X], &[X]), Failure> {
    s.split_at_checked(n)
        .ok_or_else(|| Failure::new("stereo", "slice split"))
}

fn split_mut<X>(s: &mut [X], n: usize) -> Result<(&mut [X], &mut [X]), Failure> {
    s.split_at_mut_checked(n)
        .ok_or_else(|| Failure::new("stereo", "slice split"))
}

/// Runs the facade over the signal in the chunks of `plan`.
fn pll_chunked<T: RealSample>(
    st: &mut StereoPllState<T>,
    mpx: &[T],
    pilot: &[T],
    out: &mut [T::Complex],
    plan: ChunkPlan,
) -> Result<(), Failure>
where
    Sdrk: StereoPll<T>,
{
    let (mut m, mut p, mut o) = (mpx, pilot, out);
    for size in plan.sizes() {
        if m.is_empty() {
            break;
        }
        let n = size.min(m.len());
        let (mh, mt) = split(m, n)?;
        let (ph, pt) = split(p, n)?;
        let (oh, ot) = split_mut(o, n)?;
        StereoPll::<T>::stereo_pll(&Sdrk, st, mh, ph, oh)?;
        (m, p, o) = (mt, pt, ot);
    }
    Ok(())
}

/// Fallback and Reference agree bit for bit (output and state), the loop locked and switched
/// to stereo, and the facade output does not depend on the block size (20 plans).
fn stereo_pll_for<T: RealSample>(arena: &mut Arena<'_>, case: &'static str) -> Result<(), Failure>
where
    Sdrk: StereoPll<T>,
    Fallback: StereoPll<T>,
    Reference: StereoPll<T>,
{
    let mut c = arena.carver();
    let mpx = c.take::<T>(N)?;
    let pilot = c.take::<T>(N)?;
    let fb = c.take::<T::Complex>(N)?;
    let rf = c.take::<T::Complex>(N)?;
    fill(mpx, pilot);
    let (mut a, mut b) = (state::<T>()?, state::<T>()?);
    StereoPll::<T>::stereo_pll(&Fallback, &mut a, mpx, pilot, fb)?;
    StereoPll::<T>::stereo_pll(&Reference, &mut b, mpx, pilot, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "reference differs from the fallback")
    })?;
    ensure(a == b, || Failure::new(case, "final state differs"))?;
    // The loop must have locked: stereo at the end, difference zero before the lock and
    // non-zero after it.
    let silent = fb.iter().take(200).all(|s| s.im().to_f64() == 0.0);
    let active = fb.iter().filter(|s| s.im().to_f64() != 0.0).count();
    ensure(a.is_stereo() && silent && active > 300, || {
        Failure::new(case, "the pilot loop did not lock")
            .detail(format_args!("{active} non-zero samples"))
    })?;
    for seed in 0..PLANS {
        let plan = ChunkPlan::new(4300 + seed, 1 + (seed as usize * 37) % 500);
        let mut st = state::<T>()?;
        pll_chunked(&mut st, mpx, pilot, rf, plan)?;
        ensure(bit_exact(fb, rf).is_none() && st == a, || {
            Failure::new(case, "output depends on the block size")
        })?;
    }
    Ok(())
}

pub(super) fn stereo_pll_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    stereo_pll_for::<f32>(arena, "stereo_pll_f32")
}

pub(super) fn stereo_pll_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    stereo_pll_for::<Q15>(arena, "stereo_pll_q15")
}

fn matrix_chunked<T: ComplexSample>(
    input: &[T],
    out: &mut [T],
    plan: ChunkPlan,
) -> Result<(), Failure>
where
    Sdrk: StereoMatrix<T>,
{
    let (mut i, mut o) = (input, out);
    for size in plan.sizes() {
        if i.is_empty() {
            break;
        }
        let n = size.min(i.len());
        let (ih, it) = split(i, n)?;
        let (oh, ot) = split_mut(o, n)?;
        StereoMatrix::<T>::stereo_matrix(&Sdrk, ih, oh)?;
        (i, o) = (it, ot);
    }
    Ok(())
}

fn stereo_matrix_for<T: ComplexSample>(
    arena: &mut Arena<'_>,
    case: &'static str,
    fill: fn(&mut Pcg32) -> T,
) -> Result<(), Failure>
where
    Sdrk: StereoMatrix<T>,
    Fallback: StereoMatrix<T>,
    Reference: StereoMatrix<T>,
{
    let mut c = arena.carver();
    let input = c.take::<T>(NM)?;
    let (fb, rf) = (c.take::<T>(NM)?, c.take::<T>(NM)?);
    let mut rng = Pcg32::new(4400, 1);
    for s in input.iter_mut() {
        *s = fill(&mut rng);
    }
    StereoMatrix::<T>::stereo_matrix(&Fallback, input, fb)?;
    StereoMatrix::<T>::stereo_matrix(&Reference, input, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "reference differs from the fallback")
    })?;
    StereoMatrix::<T>::stereo_matrix(&Sdrk, input, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "facade differs from the fallback")
    })?;
    // Unequal lengths are an error and change nothing.
    let short = rf.get_mut(..NM - 1).unwrap_or_default();
    ensure(
        matches!(
            StereoMatrix::<T>::stereo_matrix(&Fallback, input, short),
            Err(Error::LengthMismatch { .. })
        ),
        || Failure::new(case, "unequal lengths were accepted"),
    )?;
    for seed in 0..PLANS {
        let plan = ChunkPlan::new(4500 + seed, 1 + (seed as usize * 37) % 500);
        matrix_chunked(input, rf, plan)?;
        ensure(bit_exact(fb, rf).is_none(), || {
            Failure::new(case, "output depends on the block size")
        })?;
    }
    Ok(())
}

pub(super) fn stereo_matrix_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    stereo_matrix_for::<C32>(arena, "stereo_matrix_c32", |r| random_c32(r, 0.6))
}

pub(super) fn stereo_matrix_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    // The whole range: s + 2 d leaves it for a quarter of the samples.
    stereo_matrix_for::<CQ15>(arena, "stereo_matrix_cq15", |r| random_cq15(r, 1.0))
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

fn same_c32(a: C32, b: C32) -> bool {
    same_f32(a.re, b.re) && same_f32(a.im, b.im)
}

/// The f32 specials (`0`, `-0`, NaN, `+-inf`, a denormal, `f32::MAX`) cycled with an offset.
const SPECIALS: [f32; 7] = [
    0.0,
    -0.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE / 2.0,
    f32::MAX,
];

/// The Q15 corner values.
const CORNERS: [i16; 7] = [-32768, -32767, -1, 0, 1, 32766, 32767];

pub(super) fn extremes_m24(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m24";
    const M: usize = 120;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<f32>(M)?;
    let p = c.take::<f32>(M)?;
    let o1 = c.take::<C32>(M)?;
    let o2 = c.take::<C32>(M)?;

    // Silence: not stereo, difference exactly +0.
    x.fill(0.0);
    p.fill(0.0);
    let mut st = state::<f32>()?;
    StereoPll::<f32>::stereo_pll(&Sdrk, &mut st, x, p, o1)?;
    ensure(
        !st.is_stereo()
            && o1
                .iter()
                .all(|s| s.re.to_bits() == 0 && s.im.to_bits() == 0),
        || fail("silence did not give a mono zero output"),
    )?;

    // Non-finite and extreme values in both inputs: Fallback and Reference agree bit for bit
    // (NaN with NaN) and the phase, the decision and the level agree.
    for (offset, other) in [(0usize, 3usize), (2, 5), (4, 1)] {
        for (i, (xv, pv)) in x.iter_mut().zip(p.iter_mut()).enumerate() {
            *xv = SPECIALS.get((i + offset) % 7).copied().unwrap_or(0.0);
            *pv = SPECIALS.get((i / 3 + other) % 7).copied().unwrap_or(0.0);
        }
        let (mut a, mut b) = (state::<f32>()?, state::<f32>()?);
        StereoPll::<f32>::stereo_pll(&Fallback, &mut a, x, p, o1)?;
        StereoPll::<f32>::stereo_pll(&Reference, &mut b, x, p, o2)?;
        ensure(
            o1.iter().zip(o2.iter()).all(|(u, v)| same_c32(*u, *v))
                && a.phase() == b.phase()
                && a.is_stereo() == b.is_stereo()
                && a.integrator().to_bits() == b.integrator().to_bits()
                && a.lock_level().to_bits() == b.lock_level().to_bits(),
            || fail("implementations differ on non-finite input"),
        )?;
    }

    // A non-finite pilot sample leaves the level unchanged, while the phase keeps running.
    let mut st = state::<f32>()?;
    for (i, (xv, pv)) in x.iter_mut().zip(p.iter_mut()).enumerate() {
        *xv = 0.3;
        *pv = (0.1 * sin_turns(19_000.0 * i as f64 / FS as f64 + 0.1)) as f32;
    }
    StereoPll::<f32>::stereo_pll(&Sdrk, &mut st, x, p, o1)?;
    ensure(st.lock_level() != 0.0, || fail("the level did not move"))?;
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let before = (st.lock_level().to_bits(), st.phase());
        let mut out = [C32::ZERO];
        StereoPll::<f32>::stereo_pll(&Sdrk, &mut st, &[0.3], &[bad], &mut out)?;
        ensure(st.lock_level().to_bits() == before.0, || {
            fail("a non-finite pilot sample changed the level")
        })?;
        ensure(st.phase() != before.1, || fail("the phase did not advance"))?;
    }

    // Full scale in both signals, f32 and Q15.
    for (i, (xv, pv)) in x.iter_mut().zip(p.iter_mut()).enumerate() {
        *xv = if i % 2 == 0 { 1.0 } else { -1.0 };
        *pv = if i % 3 == 0 { 1.0 } else { -1.0 };
    }
    let (mut a, mut b) = (state::<f32>()?, state::<f32>()?);
    StereoPll::<f32>::stereo_pll(&Fallback, &mut a, x, p, o1)?;
    StereoPll::<f32>::stereo_pll(&Reference, &mut b, x, p, o2)?;
    ensure(bit_exact(o1, o2).is_none() && a == b, || {
        fail("full scale differs between implementations")
    })?;
    ensure(a.integrator().abs() <= a.integrator_limit(), || {
        fail("the integrator left its limit")
    })?;

    // Q15 corners of the loop: every pair of corner values as mpx and pilot, then the same
    // pairs in the other order.
    let n = CORNERS.len() * CORNERS.len();
    let (qx, qp) = (c.take::<Q15>(n)?, c.take::<Q15>(n)?);
    let (q1, q2) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
    for (k, (xv, pv)) in qx.iter_mut().zip(qp.iter_mut()).enumerate() {
        *xv = Q15(CORNERS.get(k / 7).copied().unwrap_or(0));
        *pv = Q15(CORNERS.get(k % 7).copied().unwrap_or(0));
    }
    for (xs, ps) in [(&*qx, &*qp), (&*qp, &*qx)] {
        let (mut a, mut b) = (state::<Q15>()?, state::<Q15>()?);
        StereoPll::<Q15>::stereo_pll(&Fallback, &mut a, xs, ps, q1)?;
        StereoPll::<Q15>::stereo_pll(&Reference, &mut b, xs, ps, q2)?;
        ensure(bit_exact(q1, q2).is_none() && a == b, || {
            fail("Q15 loop corners differ between implementations")
        })?;
    }

    // Matrix corners: C32 specials and every pair of Q15 corners.
    let (mi, m1, m2) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
    for (k, s) in mi.iter_mut().enumerate() {
        *s = CQ15 {
            re: Q15(CORNERS.get(k / 7).copied().unwrap_or(0)),
            im: Q15(CORNERS.get(k % 7).copied().unwrap_or(0)),
        };
    }
    StereoMatrix::<CQ15>::stereo_matrix(&Fallback, mi, m1)?;
    StereoMatrix::<CQ15>::stereo_matrix(&Reference, mi, m2)?;
    ensure(bit_exact(m1, m2).is_none(), || {
        fail("Q15 matrix corners differ between implementations")
    })?;
    // Saturation: s = -32768, d = 32767 gives 32766 (the sum is formed before the clamp).
    let hit = mi
        .iter()
        .zip(m1.iter())
        .find(|(i, _)| i.re.0 == -32768 && i.im.0 == 32767);
    ensure(hit.is_some_and(|(_, o)| o.re.0 == 32766), || {
        fail("the matrix saturated before the sum")
    })?;
    let (ci, c1, c2o) = (c.take::<C32>(M)?, c.take::<C32>(M)?, c.take::<C32>(M)?);
    for (i, s) in ci.iter_mut().enumerate() {
        *s = C32::new(
            SPECIALS.get(i % 7).copied().unwrap_or(0.0),
            SPECIALS.get((i / 7 + 2) % 7).copied().unwrap_or(0.0),
        );
    }
    StereoMatrix::<C32>::stereo_matrix(&Fallback, ci, c1)?;
    StereoMatrix::<C32>::stereo_matrix(&Reference, ci, c2o)?;
    ensure(
        c1.iter().zip(c2o.iter()).all(|(u, v)| same_c32(*u, *v)),
        || fail("C32 matrix specials differ between implementations"),
    )
}

// ---------------------------------------------------------------------------------------
// block and end to end (E2E-STEREO)
// ---------------------------------------------------------------------------------------

const BLOCK_SEP: &str = "block_fm_stereo_separation";

fn separation_case<T>(arena: &mut Arena<'_>, min_db: f64) -> Result<(), Failure>
where
    T: E2eSample + BlockSample + BiquadElem,
    <T as ComplexSample>::Real: BiquadElem,
    Sdrk: QuadDemod<T>
        + BiquadReal<<T as ComplexSample>::Real>
        + StereoPll<<T as ComplexSample>::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    let (l, r) = measure_block_separation::<T>(arena)?;
    ensure(l >= min_db && r >= min_db, || {
        Failure::new(BLOCK_SEP, "channel separation below the threshold")
            .detail(format_args!("{l} dB and {r} dB, need {min_db}"))
    })
}

pub(super) fn block_fm_stereo_separation(arena: &mut Arena<'_>) -> Result<(), Failure> {
    separation_case::<C32>(arena, MIN_SEPARATION_DB.0)?;
    separation_case::<CQ15>(arena, MIN_SEPARATION_DB.1)
}

const BLOCK_MONO: &str = "block_fm_stereo_mono";

pub(super) fn block_fm_stereo_mono(arena: &mut Arena<'_>) -> Result<(), Failure> {
    for q15 in [false, true] {
        let (differing, locked) = if q15 {
            measure_block_mono::<CQ15>(arena)?
        } else {
            measure_block_mono::<C32>(arena)?
        };
        ensure(!locked, || {
            Failure::new(
                BLOCK_MONO,
                "the block reported a pilot in a mono signal or in noise",
            )
        })?;
        ensure(differing == 0, || {
            Failure::new(BLOCK_MONO, "left and right differ without a pilot")
                .detail(format_args!("{differing} samples"))
        })?;
    }
    Ok(())
}

fn stereo_case<T>(
    case: &'static str,
    arena: &mut Arena<'_>,
    samples: usize,
    q15: bool,
) -> Result<(), Failure>
where
    T: E2eSample + BlockSample + BiquadElem,
    <T as ComplexSample>::Real: BiquadElem,
    Sdrk: Convert<T>
        + Ddc<T>
        + QuadDemod<T>
        + BiquadReal<<T as ComplexSample>::Real>
        + StereoPll<<T as ComplexSample>::Real>
        + FirDecimate<T>
        + StereoMatrix<T>
        + Biquad<T>,
{
    let m = measure_stereo::<T>(arena, samples, &StereoSignal::STEREO)?;
    ensure(m.audio_samples == samples / 50, || {
        Failure::new(case, "wrong audio sample count").detail(format_args!("{}", m.audio_samples))
    })?;
    check_stereo(case, &m, q15)
}

pub(super) fn e2e_fm_stereo_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    stereo_case::<C32>("e2e_fm_stereo_f32", arena, QUICK_SAMPLES, false)
}

pub(super) fn e2e_fm_stereo_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    stereo_case::<CQ15>("e2e_fm_stereo_q15", arena, QUICK_SAMPLES, true)
}

pub(super) fn e2e_fm_stereo_f32_2s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    stereo_case::<C32>("e2e_fm_stereo_f32_2s", arena, FULL_SAMPLES, false)
}

pub(super) fn e2e_fm_stereo_q15_2s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    stereo_case::<CQ15>("e2e_fm_stereo_q15_2s", arena, FULL_SAMPLES, true)
}

pub(super) fn e2e_fm_stereo_switch_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    let m = measure_stereo::<C32>(arena, RATE as usize, &StereoSignal::switching(0.3, 0.7))?;
    check_switching("e2e_fm_stereo_switch_f32", &m)
}

/// A mono transmission and noise without a carrier never report stereo and give identical
/// channels. The noise runs for 0.25 s here (1.0 s in the host test `e2e_stereo_m24`).
pub(super) fn e2e_fm_stereo_mono(arena: &mut Arena<'_>) -> Result<(), Failure> {
    let half = QUICK_SAMPLES / 2;
    let m = measure_stereo::<C32>(arena, half, &StereoSignal::MONO)?;
    check_mono("e2e_fm_stereo_mono", &m)?;
    let m = measure_stereo::<CQ15>(arena, half, &StereoSignal::MONO)?;
    check_mono("e2e_fm_stereo_mono", &m)?;
    let m = measure_stereo::<C32>(arena, half, &StereoSignal::NOISE)?;
    check_mono("e2e_fm_stereo_mono", &m)?;
    let m = measure_stereo::<CQ15>(arena, half, &StereoSignal::NOISE)?;
    check_mono("e2e_fm_stereo_mono", &m)
}
