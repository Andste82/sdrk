//! Suite cases of M35: the kernel `am_mod` and its extreme inputs.
//!
//! Every kernel case compares `Fallback` with `Reference` and the facade with `Fallback` (bit
//! for bit, the kernel is bit-exact in both types), checks the length and parameter errors and
//! runs the facade in 20 chunk plans; no files are read, so the cases run on microcontrollers
//! too.

#![allow(clippy::cast_possible_truncation, clippy::too_many_lines)]

use sdrk::Sdrk;
use sdrk_core::kernels::modulate::AmMod;
use sdrk_core::{C32, CQ15, ComplexSample, Error, Fallback, Q15, Reference, Sample};

use super::ensure;
use crate::arena::Arena;
use crate::invariance::ChunkPlan;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Length of the signals of the kernel cases.
const N: usize = 500;
/// Chunk plans per case.
const PLANS: u64 = 20;
/// The parameters of the kernel cases (the defaults of the block).
const CARRIER: f32 = 0.5;
const DEPTH: f32 = 0.8;

/// One implementation of the kernel as a function pointer.
type AmModFn<T> = fn(f32, f32, &[<T as ComplexSample>::Real], &mut [T]) -> sdrk_core::Result<()>;

/// Uniform in `[-1.5, 1.5)`, with a NaN every 61st draw.
fn random_f32(rng: &mut Pcg32) -> f32 {
    let v = ((rng.next_f64() * 2.0 - 1.0) * 1.5) as f32;
    if rng.below(61) == 0 { f32::NAN } else { v }
}

/// Any Q15 value.
fn random_q15(rng: &mut Pcg32) -> Q15 {
    Q15((rng.next_u32() >> 16) as i16)
}

fn range<'a, X>(
    s: &'a [X],
    from: usize,
    to: usize,
    case: &'static str,
) -> Result<&'a [X], Failure> {
    s.get(from..to)
        .ok_or_else(|| Failure::new(case, "slice range"))
}

fn range_mut<'a, X>(
    s: &'a mut [X],
    from: usize,
    to: usize,
    case: &'static str,
) -> Result<&'a mut [X], Failure> {
    s.get_mut(from..to)
        .ok_or_else(|| Failure::new(case, "slice range"))
}

/// Fallback and Reference agree bit for bit, the facade equals the fallback, unequal lengths
/// and parameters outside `0 ..= 1` are errors that change nothing, and the facade does not
/// depend on the block size (20 plans).
fn am_mod_case<T>(
    arena: &mut Arena<'_>,
    case: &'static str,
    seed: u64,
    fill: fn(&mut Pcg32) -> T::Real,
) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: AmMod<T>,
    Fallback: AmMod<T>,
    Reference: AmMod<T>,
{
    let mut c = arena.carver();
    let input = c.take::<T::Real>(N)?;
    let (fb, rf, fac) = (c.take::<T>(N)?, c.take::<T>(N)?, c.take::<T>(N)?);
    let mut rng = Pcg32::new(seed, 1);
    for s in input.iter_mut() {
        *s = fill(&mut rng);
    }
    AmMod::<T>::am_mod(&Fallback, CARRIER, DEPTH, input, fb)?;
    AmMod::<T>::am_mod(&Reference, CARRIER, DEPTH, input, rf)?;
    AmMod::<T>::am_mod(&Sdrk, CARRIER, DEPTH, input, fac)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "reference differs from the fallback")
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(case, "facade differs from the fallback")
    })?;
    ensure(
        fb.iter()
            .all(|s| bit_exact(&[s.im()], &[T::Real::ZERO]).is_none()),
        || Failure::new(case, "the imaginary part is not zero"),
    )?;
    // Errors: unequal lengths and parameters outside 0 ..= 1; the output stays as it was.
    let calls: [AmModFn<T>; 3] = [
        |c, m, i, o| AmMod::<T>::am_mod(&Fallback, c, m, i, o),
        |c, m, i, o| AmMod::<T>::am_mod(&Reference, c, m, i, o),
        |c, m, i, o| AmMod::<T>::am_mod(&Sdrk, c, m, i, o),
    ];
    for f in calls {
        let short = range_mut(rf, 0, N - 1, case)?;
        ensure(
            matches!(
                f(CARRIER, DEPTH, input, short),
                Err(Error::LengthMismatch { .. })
            ),
            || Failure::new(case, "a shorter output was accepted"),
        )?;
        for (cr, dp) in [
            (1.5, DEPTH),
            (-0.1, DEPTH),
            (f32::NAN, DEPTH),
            (CARRIER, 1.01),
            (CARRIER, f32::INFINITY),
        ] {
            ensure(
                matches!(f(cr, dp, input, rf), Err(Error::InvalidParameter(_))),
                || Failure::new(case, "a parameter outside 0 to 1 was accepted"),
            )?;
        }
        ensure(bit_exact(fb, rf).is_none(), || {
            Failure::new(case, "a rejected call wrote to the output")
        })?;
    }
    for plan_seed in 0..PLANS {
        let plan = ChunkPlan::new(seed * 100 + plan_seed, 1 + (plan_seed as usize * 37) % 300);
        fac.fill(T::ZERO);
        let mut pos = 0;
        for size in plan.sizes() {
            if pos >= N {
                break;
            }
            let end = (pos + size).min(N);
            AmMod::<T>::am_mod(
                &Sdrk,
                CARRIER,
                DEPTH,
                range(input, pos, end, case)?,
                range_mut(fac, pos, end, case)?,
            )?;
            pos = end;
        }
        ensure(bit_exact(fb, fac).is_none(), || {
            Failure::new(case, "output depends on the block size")
        })?;
    }
    Ok(())
}

pub(super) fn am_mod_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    am_mod_case::<C32>(arena, "am_mod_c32", 4800, random_f32)
}

pub(super) fn am_mod_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    am_mod_case::<CQ15>(arena, "am_mod_cq15", 4801, random_q15)
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

/// The f32 specials (`0`, `-0`, NaN, `+-inf`, a denormal, `+-f32::MAX`, `+-1`, just beyond 1).
const SPECIALS: [f32; 11] = [
    0.0,
    -0.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE / 2.0,
    f32::MAX,
    f32::MIN,
    1.0,
    -1.0,
    1.0 + f32::EPSILON,
];

/// The Q15 corner values.
const CORNERS: [i16; 7] = [-32768, -32767, -1, 0, 1, 32766, 32767];

/// Parameter pairs `(carrier, depth)`: the ends of the range and the defaults.
const GAINS: [(f32, f32); 6] = [
    (0.0, 0.0),
    (1.0, 1.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.8),
    (-0.0, -0.0),
];

/// All three implementations give the same output (bit for bit); returns the fallback's.
fn all_agree<T>(
    case: &'static str,
    carrier: f32,
    depth: f32,
    input: &[T::Real],
    outs: (&mut [T], &mut [T], &mut [T]),
) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: AmMod<T>,
    Fallback: AmMod<T>,
    Reference: AmMod<T>,
{
    let (o1, o2, o3) = outs;
    AmMod::<T>::am_mod(&Fallback, carrier, depth, input, &mut *o1)?;
    AmMod::<T>::am_mod(&Reference, carrier, depth, input, &mut *o2)?;
    AmMod::<T>::am_mod(&Sdrk, carrier, depth, input, &mut *o3)?;
    ensure(
        bit_exact(o1, o2).is_none() && bit_exact(o1, o3).is_none(),
        || Failure::new(case, "implementations differ"),
    )
}

pub(super) fn extremes_m35(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m35";
    const M: usize = 64;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<f32>(M)?;
    let (g1, g2, g3) = (c.take::<C32>(M)?, c.take::<C32>(M)?, c.take::<C32>(M)?);
    let q = c.take::<Q15>(M)?;
    let (h1, h2, h3) = (c.take::<CQ15>(M)?, c.take::<CQ15>(M)?, c.take::<CQ15>(M)?);

    // Silence: the bare carrier.
    x.fill(0.0);
    q.fill(Q15(0));
    all_agree::<C32>(CASE, CARRIER, DEPTH, x, (&mut *g1, &mut *g2, &mut *g3))?;
    ensure(g1.iter().all(|s| *s == C32::new(CARRIER, 0.0)), || {
        fail("silence did not give the carrier")
    })?;
    all_agree::<CQ15>(CASE, CARRIER, DEPTH, q, (&mut *h1, &mut *h2, &mut *h3))?;
    ensure(
        h1.iter().all(|s| s.re == Q15(16384) && s.im == Q15(0)),
        || fail("Q15 silence did not give the carrier"),
    )?;

    // Full scale alternating: the two ends of the envelope, carrier (1 +- depth).
    for (i, v) in x.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    all_agree::<C32>(CASE, CARRIER, DEPTH, x, (&mut *g1, &mut *g2, &mut *g3))?;
    let (low, high) = (CARRIER * (1.0 - DEPTH), CARRIER * (1.0 + DEPTH));
    let ends = g1
        .iter()
        .enumerate()
        .all(|(i, s)| s.re == if i % 2 == 0 { high } else { low } && s.im.to_bits() == 0);
    ensure(ends, || {
        fail("full scale did not give carrier (1 +- depth)")
    })?;
    for (i, v) in q.iter_mut().enumerate() {
        *v = if i % 2 == 0 { Q15::MAX } else { Q15::MIN };
    }
    all_agree::<CQ15>(CASE, CARRIER, DEPTH, q, (&mut *h1, &mut *h2, &mut *h3))?;
    // c_q = 16384, m_q = 26214: t = 26213 and -26214, re = 29491 and 3277.
    let ends = h1
        .iter()
        .enumerate()
        .all(|(i, s)| s.re == Q15(if i % 2 == 0 { 29491 } else { 3277 }) && s.im == Q15(0));
    ensure(ends, || {
        fail("Q15 full scale did not give carrier (1 +- depth)")
    })?;

    // NaN and infinity: the implementations agree, the output is finite, NaN gives the bare
    // carrier and the clamp maps +-inf and +-f32::MAX to the ends.
    for (i, v) in x.iter_mut().enumerate() {
        *v = SPECIALS.get(i % SPECIALS.len()).copied().unwrap_or(0.0);
    }
    for (cr, dp) in GAINS {
        all_agree::<C32>(CASE, cr, dp, x, (&mut *g1, &mut *g2, &mut *g3))?;
        ensure(
            g1.iter().all(|s| s.re.is_finite() && s.im.to_bits() == 0),
            || fail("a special input gave a non-finite output"),
        )?;
        let ends = x.iter().zip(g1.iter()).all(|(v, s)| {
            let want = if v.is_nan() {
                cr
            } else if *v >= 1.0 {
                cr * (1.0 + dp)
            } else if *v <= -1.0 {
                cr * (1.0 - dp)
            } else {
                s.re
            };
            s.re == want
        });
        ensure(ends, || fail("NaN or a value beyond +-1 was not clamped"))?;
    }

    // Q15 corners with every gain pair: the implementations agree, re lies in 0 ..= 32767.
    for (i, v) in q.iter_mut().enumerate() {
        *v = Q15(CORNERS.get(i % CORNERS.len()).copied().unwrap_or(0));
    }
    for (cr, dp) in GAINS {
        all_agree::<CQ15>(CASE, cr, dp, q, (&mut *h1, &mut *h2, &mut *h3))?;
        ensure(h1.iter().all(|s| s.re.0 >= 0 && s.im == Q15(0)), || {
            fail("a Q15 corner gave a negative envelope")
        })?;
    }
    // carrier 1, depth 1: -32768 comes down to one LSB, 32767 saturates.
    all_agree::<CQ15>(CASE, 1.0, 1.0, q, (&mut *h1, &mut *h2, &mut *h3))?;
    ensure(
        h1.first().is_some_and(|s| s.re == Q15(1)) && h1.get(6).is_some_and(|s| s.re == Q15::MAX),
        || fail("Q15 corners at carrier 1 and depth 1"),
    )?;

    // Depth 0: the constant carrier whatever the input.
    let mut rng = Pcg32::new(4802, 1);
    for v in x.iter_mut() {
        *v = random_f32(&mut rng);
    }
    for v in q.iter_mut() {
        *v = random_q15(&mut rng);
    }
    all_agree::<C32>(CASE, 0.7, 0.0, x, (&mut *g1, &mut *g2, &mut *g3))?;
    ensure(g1.iter().all(|s| *s == C32::new(0.7, 0.0)), || {
        fail("depth 0 did not give the constant carrier")
    })?;
    all_agree::<CQ15>(CASE, 0.7, 0.0, q, (&mut *h1, &mut *h2, &mut *h3))?;
    // round(0.7f32 32768) = 22938.
    ensure(
        h1.iter().all(|s| s.re == Q15(22938) && s.im == Q15(0)),
        || fail("Q15 depth 0 did not give the constant carrier"),
    )?;

    // Depth 1 with a full-scale tone (cosine at fs / 8, which reaches -1 exactly): the envelope
    // touches 0 (Q15: one LSB at carrier 0.5, round_shift(c_q, 15)) and peaks at 2 carrier.
    for (i, v) in x.iter_mut().enumerate() {
        *v = libm::cos(2.0 * core::f64::consts::PI * (i % 8) as f64 / 8.0) as f32;
    }
    for (o, v) in q.iter_mut().zip(x.iter()) {
        *o = Q15::from_f32(*v);
    }
    all_agree::<C32>(CASE, CARRIER, 1.0, x, (&mut *g1, &mut *g2, &mut *g3))?;
    let (lo, hi) = g1.iter().fold((f32::INFINITY, 0.0f32), |(l, h), s| {
        (l.min(s.re), h.max(s.re))
    });
    ensure(
        lo == 0.0 && hi == 2.0 * CARRIER && g1.iter().all(|s| s.re >= 0.0),
        || fail("depth 1 did not touch 0"),
    )?;
    all_agree::<CQ15>(CASE, CARRIER, 1.0, q, (&mut *h1, &mut *h2, &mut *h3))?;
    let (lo, hi) = h1
        .iter()
        .fold((i16::MAX, 0i16), |(l, h), s| (l.min(s.re.0), h.max(s.re.0)));
    ensure(lo == 1 && hi == 32767, || {
        fail("Q15 depth 1 did not touch 0")
    })?;
    Ok(())
}
