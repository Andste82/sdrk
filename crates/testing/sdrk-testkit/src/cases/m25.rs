//! Suite cases of M25: the kernel `preemph_limit` (pre-emphasis and peak limiter), its extreme
//! inputs and the block.
//!
//! The kernel is bit-exact in both types and not accelerated (the facade is the fallback). The
//! kernel cases compare `Fallback` with `Reference` (output and final state) and run the facade
//! in 20 chunk plans; no files are read, so the cases run on microcontrollers too. The loop
//! tests of the milestone need the pipeline (`std`) and live in `tests/system`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use sdrk::Sdrk;
use sdrk_blocks::design::preemphasis_coefs;
use sdrk_blocks::{
    ARENA_ALIGN, ArenaStorage, Block, BlockReal, ParamValue, PreemphLimitBlock, PreemphLimitConfig,
};
use sdrk_core::kernels::preemph_limit::{PreemphLimit, PreemphLimitState};
use sdrk_core::{Fallback, Q15, Rate, RealSample, Reference, SampleType};

use super::ensure;
use crate::arena::Arena;
use crate::invariance::ChunkPlan;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Sample rate of the kernel cases.
const FS: u64 = 48_000;
/// Length of the kernel signals.
const N: usize = 2048;
/// Chunk plans per type.
const PLANS: u64 = 20;
/// Output peak of the kernel cases.
const LIMIT: f64 = 0.5;
/// Hold and release of the kernel cases: short, so that the signal visits the held and the
/// released state and ends idle.
const HOLD_S: f64 = 0.01;
const RELEASE_S: f64 = 0.005;

fn rate(hz: u64) -> Result<Rate, Failure> {
    Ok(Rate::hz(hz)?)
}

fn state<T: RealSample>() -> Result<PreemphLimitState<T>, Failure> {
    let coefs = preemphasis_coefs(50e-6, FS as f64)?;
    Ok(PreemphLimitState::new(
        coefs,
        LIMIT,
        HOLD_S,
        RELEASE_S,
        rate(FS)?,
    )?)
}

/// The largest output value the type can hold for the limit `limit`.
fn bound<T: RealSample>(limit: f64) -> f64 {
    if T::TYPE == SampleType::Q15 {
        libm::floor(limit * 32768.0 + 0.5).min(32767.0) / 32768.0
    } else {
        f64::from(limit as f32)
    }
}

fn sin_turns(turns: f64) -> f64 {
    libm::sin(2.0 * core::f64::consts::PI * (turns - libm::floor(turns)))
}

/// Noise of sigma 0.02 with tone bursts: 1 kHz at samples 200 to 500 and 8 kHz at 700 to 900.
fn fill<T: RealSample>(x: &mut [T]) {
    let amp = if T::TYPE == SampleType::Q15 {
        0.95
    } else {
        1.5
    };
    let mut rng = Pcg32::new(4500, 1);
    for (n, v) in x.iter_mut().enumerate() {
        let t = n as f64 / FS as f64;
        let burst = if (200..500).contains(&n) {
            amp * sin_turns(1000.0 * t)
        } else if (700..900).contains(&n) {
            amp * sin_turns(8000.0 * t)
        } else {
            0.0
        };
        *v = T::from_f64_round(burst + 0.02 * rng.normal());
    }
}

fn split<X>(s: &[X], n: usize) -> Result<(&[X], &[X]), Failure> {
    s.split_at_checked(n)
        .ok_or_else(|| Failure::new("preemph_limit", "slice split"))
}

fn split_mut<X>(s: &mut [X], n: usize) -> Result<(&mut [X], &mut [X]), Failure> {
    s.split_at_mut_checked(n)
        .ok_or_else(|| Failure::new("preemph_limit", "slice split"))
}

/// Runs the facade over the signal in the chunks of `plan`.
fn run_chunked<T: RealSample>(
    st: &mut PreemphLimitState<T>,
    input: &[T],
    out: &mut [T],
    plan: ChunkPlan,
) -> Result<(), Failure>
where
    Sdrk: PreemphLimit<T>,
{
    let (mut i, mut o) = (input, out);
    for size in plan.sizes() {
        if i.is_empty() {
            break;
        }
        let n = size.min(i.len());
        let (ih, it) = split(i, n)?;
        let (oh, ot) = split_mut(o, n)?;
        PreemphLimit::<T>::preemph_limit(&Sdrk, st, ih, oh)?;
        (i, o) = (it, ot);
    }
    Ok(())
}

/// Fallback and Reference agree bit for bit (output and state), the limiter attacked, held and
/// released (gain below 1 in the middle, 1.0 at the end), no output exceeds the limit and the
/// facade output does not depend on the block size (20 plans).
fn preemph_limit_for<T: RealSample>(
    arena: &mut Arena<'_>,
    case: &'static str,
) -> Result<(), Failure>
where
    Sdrk: PreemphLimit<T>,
    Fallback: PreemphLimit<T>,
    Reference: PreemphLimit<T>,
{
    let mut c = arena.carver();
    let input = c.take::<T>(N)?;
    let fb = c.take::<T>(N)?;
    let rf = c.take::<T>(N)?;
    fill(input);
    let (mut a, mut b) = (state::<T>()?, state::<T>()?);
    PreemphLimit::<T>::preemph_limit(&Fallback, &mut a, input, fb)?;
    PreemphLimit::<T>::preemph_limit(&Reference, &mut b, input, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "reference differs from the fallback")
    })?;
    ensure(a == b, || Failure::new(case, "final state differs"))?;
    let top = bound::<T>(LIMIT);
    ensure(fb.iter().all(|v| v.to_f64().abs() <= top), || {
        Failure::new(case, "an output sample exceeds the limit")
    })?;
    ensure(fb.iter().any(|v| v.to_f64().abs() > 0.9 * top), || {
        Failure::new(case, "the limiter never reached the limit")
    })?;
    // The limiter was active in the middle of the run and idle at the end.
    let mut mid = state::<T>()?;
    let (head, _) = split(input, 900)?;
    let (scratch, _) = split_mut(rf, 900)?;
    PreemphLimit::<T>::preemph_limit(&Fallback, &mut mid, head, scratch)?;
    ensure(mid.gain() < 0.9 && a.gain() == 1.0, || {
        Failure::new(case, "the limiter did not attack and release").detail(format_args!(
            "gain {} and {}",
            mid.gain(),
            a.gain()
        ))
    })?;
    for seed in 0..PLANS {
        let plan = ChunkPlan::new(4600 + seed, 1 + (seed as usize * 37) % 500);
        let mut st = state::<T>()?;
        run_chunked(&mut st, input, rf, plan)?;
        ensure(bit_exact(fb, rf).is_none() && st == a, || {
            Failure::new(case, "output depends on the block size")
        })?;
    }
    // The facade is the fallback.
    let mut st = state::<T>()?;
    PreemphLimit::<T>::preemph_limit(&Sdrk, &mut st, input, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "facade differs from the fallback")
    })
}

pub(super) fn preemph_limit_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    preemph_limit_for::<f32>(arena, "preemph_limit_f32")
}

pub(super) fn preemph_limit_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    preemph_limit_for::<Q15>(arena, "preemph_limit_q15")
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

/// The f32 specials cycled with an offset: NaN, infinities and a denormal are all handled by the
/// specified arithmetic (NaN and infinities are replaced by zero).
const SPECIALS: [f32; 7] = [
    0.0,
    -0.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE / 2.0,
    1.0,
];

/// Finite `f32` values beyond the input limit of 1024, both signs.
const HUGE: [f32; 8] = [1025.0, -2000.0, 1e30, -1e36, 3e38, f32::MAX, f32::MIN, -1e6];

/// The Q15 corner values.
const CORNERS: [i16; 7] = [-32768, -32767, -1, 0, 1, 32766, 32767];

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

pub(super) fn extremes_m25(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m25";
    const M: usize = 200;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<f32>(M)?;
    let o1 = c.take::<f32>(M)?;
    let o2 = c.take::<f32>(M)?;

    // Silence gives exact zeros and an idle limiter, in both types.
    x.fill(0.0);
    let mut st = state::<f32>()?;
    PreemphLimit::<f32>::preemph_limit(&Sdrk, &mut st, x, o1)?;
    ensure(
        o1.iter().all(|v| v.to_bits() == 0) && st.gain() == 1.0,
        || fail("f32 silence did not give zeros"),
    )?;
    let q = c.take::<Q15>(M)?;
    let qo = c.take::<Q15>(M)?;
    q.fill(Q15(0));
    let mut qs = state::<Q15>()?;
    PreemphLimit::<Q15>::preemph_limit(&Sdrk, &mut qs, q, qo)?;
    ensure(qo.iter().all(|v| v.0 == 0) && qs.gain() == 1.0, || {
        fail("Q15 silence did not give zeros")
    })?;

    // Full scale, NaN, infinities and denormals in every phase: the implementations agree bit
    // for bit, the output is finite and never above the limit.
    let top = bound::<f32>(LIMIT);
    for offset in 0..7 {
        for (i, v) in x.iter_mut().enumerate() {
            *v = if offset == 6 {
                if i % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                SPECIALS.get((i / 3 + offset) % 7).copied().unwrap_or(0.0)
            };
        }
        let (mut a, mut b) = (state::<f32>()?, state::<f32>()?);
        PreemphLimit::<f32>::preemph_limit(&Fallback, &mut a, x, o1)?;
        PreemphLimit::<f32>::preemph_limit(&Reference, &mut b, x, o2)?;
        ensure(same_bits(o1, o2) && a == b, || {
            fail("implementations differ on non-finite or full-scale input")
        })?;
        ensure(
            o1.iter()
                .all(|v| v.is_finite() && f64::from(v.abs()) <= top),
            || fail("the output is not finite or exceeds the limit"),
        )?;
    }

    // Finite values far beyond full scale (the input is clamped to +-1024): the implementations
    // agree, the output stays finite and below the limit, and the state recovers (the gain is
    // finite and positive afterwards).
    for offset in 0..HUGE.len() {
        for (i, v) in x.iter_mut().enumerate() {
            *v = if i % 5 == 0 {
                HUGE.get((i / 5 + offset) % HUGE.len())
                    .copied()
                    .unwrap_or(0.0)
            } else {
                0.25
            };
        }
        let (mut a, mut b) = (state::<f32>()?, state::<f32>()?);
        PreemphLimit::<f32>::preemph_limit(&Fallback, &mut a, x, o1)?;
        PreemphLimit::<f32>::preemph_limit(&Reference, &mut b, x, o2)?;
        ensure(same_bits(o1, o2) && a == b, || {
            fail("implementations differ on huge finite input")
        })?;
        ensure(
            o1.iter()
                .all(|v| v.is_finite() && f64::from(v.abs()) <= top)
                && a.gain().is_finite()
                && a.gain() > 0.0,
            || fail("a huge finite input left a non-finite output or state"),
        )?;
    }

    // Q15: every pair of corner values, as a sequence and in the other order, with limit 1.0
    // and limit 0.5.
    let n = CORNERS.len() * CORNERS.len();
    let (qx, qy) = (c.take::<Q15>(n)?, c.take::<Q15>(n)?);
    let (q1, q2) = (c.take::<Q15>(n)?, c.take::<Q15>(n)?);
    for (k, (a, b)) in qx.iter_mut().zip(qy.iter_mut()).enumerate() {
        *a = Q15(CORNERS.get(k / 7).copied().unwrap_or(0));
        *b = Q15(CORNERS.get(k % 7).copied().unwrap_or(0));
    }
    for limit in [1.0, LIMIT] {
        for input in [&*qx, &*qy] {
            let coefs = preemphasis_coefs(530e-6, FS as f64)?;
            let rt = rate(FS)?;
            let mk = || PreemphLimitState::<Q15>::new(coefs, limit, HOLD_S, RELEASE_S, rt);
            let (mut a, mut b) = (mk()?, mk()?);
            PreemphLimit::<Q15>::preemph_limit(&Fallback, &mut a, input, q1)?;
            PreemphLimit::<Q15>::preemph_limit(&Reference, &mut b, input, q2)?;
            ensure(bit_exact(q1, q2).is_none() && a == b, || {
                fail("Q15 corners differ between implementations")
            })?;
            let top = bound::<Q15>(limit);
            ensure(q1.iter().all(|v| v.to_f64().abs() <= top), || {
                fail("a Q15 corner output exceeds the limit")
            })?;
        }
    }

    // The largest pre-emphasis gain, 1000 us at 240 kHz (b0 about 272), with a full-scale
    // square wave in Q15: the i128 products must not overflow, and the output stays within the
    // limit.
    const SQ: usize = 400;
    let (sx, s1, s2) = (c.take::<Q15>(SQ)?, c.take::<Q15>(SQ)?, c.take::<Q15>(SQ)?);
    for (i, v) in sx.iter_mut().enumerate() {
        *v = if (i / 5) % 2 == 0 {
            Q15(32767)
        } else {
            Q15(-32768)
        };
    }
    let coefs = preemphasis_coefs(1e-3, 240_000.0)?;
    for limit in [1.0, 0.1] {
        let rt = rate(240_000)?;
        let mk = || PreemphLimitState::<Q15>::new(coefs, limit, HOLD_S, RELEASE_S, rt);
        let (mut a, mut b) = (mk()?, mk()?);
        PreemphLimit::<Q15>::preemph_limit(&Fallback, &mut a, sx, s1)?;
        PreemphLimit::<Q15>::preemph_limit(&Reference, &mut b, sx, s2)?;
        ensure(bit_exact(s1, s2).is_none() && a == b, || {
            fail("the square wave at 1000 us differs between implementations")
        })?;
        let top = bound::<Q15>(limit);
        ensure(s1.iter().all(|v| v.to_f64().abs() <= top), || {
            fail("the square wave output exceeds the limit")
        })?;
        ensure(s1.iter().any(|v| v.0 != 0), || {
            fail("the square wave gave silence")
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// block
// ---------------------------------------------------------------------------------------

const BLOCK: &str = "block_preemph_limit_peak";
const BLOCK_N: usize = 1500;

fn block_peak_for<T: BlockReal>(arena: &mut Arena<'_>) -> Result<(), Failure>
where
    Sdrk: PreemphLimit<T>,
{
    let mut c = arena.carver();
    let x = c.take::<T>(BLOCK_N)?;
    let y = c.take::<T>(BLOCK_N)?;
    // Random input up to 20 times full scale (saturated in Q15).
    let mut rng = Pcg32::new(4700, 2);
    for v in x.iter_mut() {
        *v = T::from_f64_round(((rng.next_f64() * 2.0) - 1.0) * 20.0);
    }
    let cfg = PreemphLimitConfig {
        limit: 0.8,
        ..PreemphLimitConfig::default()
    };
    let r = rate(240_000)?;
    let bytes = PreemphLimitBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, r)? + ARENA_ALIGN;
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let mut block = PreemphLimitBlock::<T, _>::new(&cfg, r, &mut store)?;
    for limit in [0.8, 0.3] {
        if limit < 0.8 {
            block.set_param("limit", ParamValue::F64(limit))?;
        }
        let p = block.process(x, y)?;
        ensure(p.consumed == BLOCK_N && p.produced == BLOCK_N, || {
            Failure::new(BLOCK, "the block did not process the whole input")
        })?;
        let top = bound::<T>(limit);
        ensure(y.iter().all(|v| v.to_f64().abs() <= top), || {
            Failure::new(BLOCK, "an output sample exceeds the limit")
                .detail(format_args!("limit {limit}"))
        })?;
        ensure(y.iter().any(|v| v.to_f64().abs() > 0.9 * top), || {
            Failure::new(BLOCK, "the output never came close to the limit")
        })?;
    }
    Ok(())
}

pub(super) fn block_preemph_limit_peak(arena: &mut Arena<'_>) -> Result<(), Failure> {
    block_peak_for::<f32>(arena)?;
    block_peak_for::<Q15>(arena)
}
