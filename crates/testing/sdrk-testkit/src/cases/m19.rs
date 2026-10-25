//! Suite cases of M19: the `squelch` kernel, its extreme inputs, the `squelch` block and the
//! end-to-end narrowband FM receive.

#![allow(clippy::many_single_char_names, clippy::too_many_lines)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::{ArenaStorage, Block, BlockSample, ParamValue, SquelchBlock, SquelchConfig};
use sdrk_core::kernels::demod::QuadDemod;
use sdrk_core::kernels::fused::{Ddc, FmDemod};
use sdrk_core::kernels::squelch::{Squelch, SquelchState};
use sdrk_core::kernels::{convert::Convert, decim::ResampleReal};
use sdrk_core::{C32, CQ15, ComplexSample, Fallback, Q15, Rate, RealSample, Reference, SampleType};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use super::m06::same_rate;
use crate::arena::Arena;
use crate::e2e::E2eSample;
use crate::e2e_nfm::{
    NFM_FULL_SAMPLES, NFM_MAX_CLOSE_DELAY_S, NFM_MAX_OPEN_DELAY_S, NFM_MIN_SINAD_F32_DB,
    NFM_MIN_SINAD_Q15_DB, NFM_QUICK_SAMPLES, measure_nfm,
};
use crate::r#gen::extreme;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

const FS: u64 = 48_000;
/// Time constant of the kernel cases (24 samples: the gate follows the stretches below).
const TAU_S: f64 = 0.0005;
/// Time constant of the block case.
const BLOCK_TAU_S: f64 = 0.002;
const THRESHOLD_DB: f64 = -40.0;
const HYSTERESIS_DB: f64 = 3.0;
/// Length of the test signals.
const N: usize = 1536;
/// Length of one loud or quiet stretch.
const STRETCH: usize = 150;
/// Chunk plans per type in the invariance check.
const PLANS: u64 = 20;

fn rate() -> Result<Rate, Failure> {
    Ok(Rate::hz(FS)?)
}

/// Noise amplitude of the stretch that contains sample `i`: quiet (-60 dBFS), loud (-13 dBFS),
/// very quiet, medium (-25 dBFS), so the gate opens and closes several times.
fn amplitude(i: usize) -> f64 {
    match (i / STRETCH) % 4 {
        0 => 0.001,
        1 => 0.15,
        2 => 0.0006,
        _ => 0.04,
    }
}

fn state<T: ComplexSample>() -> Result<SquelchState<T>, Failure> {
    Ok(SquelchState::<T>::new(
        THRESHOLD_DB,
        HYSTERESIS_DB,
        TAU_S,
        rate()?,
    )?)
}

fn fill_c32(input: &mut [C32], seed: u64) {
    let mut rng = Pcg32::new(seed, 1);
    for (i, s) in input.iter_mut().enumerate() {
        let n = random_c32(&mut rng, 1.0);
        let a = amplitude(i);
        *s = C32::new(n.re * a as f32, n.im * a as f32);
    }
}

fn fill_cq15(input: &mut [CQ15], seed: u64) {
    let mut rng = Pcg32::new(seed, 1);
    for (i, s) in input.iter_mut().enumerate() {
        let n = random_cq15(&mut rng, 1.0);
        let a = amplitude(i);
        *s = CQ15::from_c64_round(Complex64::new(n.re.to_f64() * a, n.im.to_f64() * a));
    }
}

/// Fallback, Reference and facade agree bit for bit (output and state), the gate opened and
/// closed during the signal, and the output does not depend on the block size (20 plans).
fn squelch_for<T>(
    arena: &mut Arena<'_>,
    case: &'static str,
    fill: fn(&mut [T], u64),
) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: Squelch<T>,
    Fallback: Squelch<T>,
    Reference: Squelch<T>,
{
    let mut c = arena.carver();
    let input = c.take::<T>(N)?;
    let (fb, rf, fac) = (c.take::<T>(N)?, c.take::<T>(N)?, c.take::<T>(N)?);
    fill(input, 4100);
    let (mut a, mut b, mut d) = (state::<T>()?, state::<T>()?, state::<T>()?);
    Squelch::<T>::squelch(&Fallback, &mut a, input, fb)?;
    Squelch::<T>::squelch(&Reference, &mut b, input, rf)?;
    Squelch::<T>::squelch(&Sdrk, &mut d, input, fac)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "reference differs from the fallback")
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(case, "facade differs from the fallback")
    })?;
    ensure(
        a.average_power() == b.average_power()
            && a.average_power() == d.average_power()
            && a.is_open() == b.is_open()
            && a.is_open() == d.is_open(),
        || Failure::new(case, "final state differs"),
    )?;
    // The output is zero or the input, and both happen.
    let (mut passed, mut zeroed) = (0usize, 0usize);
    for (i, o) in input.iter().zip(fb.iter()) {
        if bit_exact(core::slice::from_ref(o), core::slice::from_ref(&T::ZERO)).is_none() {
            zeroed += 1;
        } else if bit_exact(core::slice::from_ref(o), core::slice::from_ref(i)).is_none() {
            passed += 1;
        } else {
            return Err(Failure::new(case, "output is neither zero nor the input"));
        }
    }
    ensure(passed > 200 && zeroed > 200, || {
        Failure::new(case, "the gate did not open and close")
            .detail(format_args!("{passed} passed, {zeroed} zeroed"))
    })?;
    let fs = rate()?;
    for seed in 0..PLANS {
        let plan = ChunkPlan::new(4200 + seed, 1 + (seed as usize * 37) % 500);
        check_block_invariance(
            input,
            fb,
            rf,
            plan,
            || SquelchState::<T>::new(THRESHOLD_DB, HYSTERESIS_DB, TAU_S, fs),
            |st, i, o| match st {
                Ok(st) => same_rate(i, o, |i, o| Squelch::<T>::squelch(&Sdrk, st, i, o)),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    Ok(())
}

pub(super) fn squelch_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    squelch_for::<C32>(arena, "squelch_c32", fill_c32)
}

pub(super) fn squelch_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    squelch_for::<CQ15>(arena, "squelch_cq15", fill_cq15)
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

pub(super) fn extremes_m19(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m19";
    const M: usize = 70;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<C32>(M)?;
    let out = c.take::<C32>(M)?;
    let q = c.take::<CQ15>(M)?;
    let qout = c.take::<CQ15>(M)?;
    let qref = c.take::<CQ15>(M)?;

    // Silence keeps the gate closed and the output zero, even at the lowest threshold.
    for threshold in [-40.0, -150.0] {
        let mut st = SquelchState::<C32>::new(threshold, 3.0, TAU_S, rate()?)?;
        extreme::silence(x);
        Squelch::<C32>::squelch(&Sdrk, &mut st, x, out)?;
        ensure(
            !st.is_open()
                && out
                    .iter()
                    .all(|s| s.re.to_bits() == 0 && s.im.to_bits() == 0),
            || fail("silence did not give a closed gate"),
        )?;
        // A Q15 threshold below about -90 dB rounds to a zero open level (always open).
        let mut stq = SquelchState::<CQ15>::new(threshold.max(-80.0), 3.0, TAU_S, rate()?)?;
        extreme::silence(q);
        Squelch::<CQ15>::squelch(&Sdrk, &mut stq, q, qout)?;
        ensure(
            !stq.is_open() && qout.iter().all(|s| *s == CQ15::default()),
            || fail("Q15 silence did not give a closed gate"),
        )?;
    }

    // Full scale opens the gate; the output follows the input once p has crossed the level.
    let mut st = state::<C32>()?;
    for s in x.iter_mut() {
        *s = C32::new(1.0, 1.0);
    }
    Squelch::<C32>::squelch(&Sdrk, &mut st, x, out)?;
    ensure(st.is_open() && out.last() == x.last(), || {
        fail("full scale did not open the gate")
    })?;
    ensure(
        st.average_power().is_finite() && st.average_power() <= 2.0,
        || fail("average power out of range"),
    )?;

    // NaN and infinity, one sample at a time: a non-finite power leaves p and the gate alone,
    // and the sample is passed or zeroed by the gate state.
    extreme::f32_specials_complex(x);
    let mut st = state::<C32>()?;
    let loud = [C32::new(0.5, 0.5); 400];
    let mut warm = [C32::default(); 400];
    Squelch::<C32>::squelch(&Sdrk, &mut st, &loud, &mut warm)?;
    ensure(st.is_open(), || fail("warm-up did not open the gate"))?;
    let mut nonfinite = 0usize;
    for (s, o) in x.iter().zip(out.iter_mut()) {
        let (p_before, open_before) = (st.average_power(), st.is_open());
        let power = s.re * s.re + s.im * s.im;
        Squelch::<C32>::squelch(
            &Sdrk,
            &mut st,
            core::slice::from_ref(s),
            core::slice::from_mut(o),
        )?;
        if !power.is_finite() {
            nonfinite += 1;
            ensure(
                st.average_power() == p_before && st.is_open() == open_before,
                || fail("a non-finite power changed the state"),
            )?;
        }
        let want = if st.is_open() { *s } else { C32::new(0.0, 0.0) };
        ensure(same_f32(o.re, want.re) && same_f32(o.im, want.im), || {
            fail("output is neither the input nor zero")
        })?;
    }
    ensure(nonfinite > 20, || fail("too few non-finite samples"))?;
    // The same stream in one call gives the same result as the facade and the reference.
    let (mut a, mut b, mut d) = (state::<C32>()?, state::<C32>()?, state::<C32>()?);
    let (o1, o2) = (c.take::<C32>(M)?, c.take::<C32>(M)?);
    Squelch::<C32>::squelch(&Fallback, &mut a, x, out)?;
    Squelch::<C32>::squelch(&Reference, &mut b, x, o1)?;
    Squelch::<C32>::squelch(&Sdrk, &mut d, x, o2)?;
    let all_same = out.iter().zip(o1.iter()).zip(o2.iter()).all(|((p, r), s)| {
        same_f32(p.re, r.re) && same_f32(p.im, r.im) && same_f32(p.re, s.re) && same_f32(p.im, s.im)
    });
    ensure(
        all_same && a.is_open() == b.is_open() && a.is_open() == d.is_open(),
        || fail("implementations differ on non-finite input"),
    )?;

    // Q15 corners: the extreme raw values in every combination.
    let corners = [-32768i16, -32767, -1, 0, 1, 32766, 32767];
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
    let (mut a, mut b, mut d) = (state::<CQ15>()?, state::<CQ15>()?, state::<CQ15>()?);
    let q = q.get(..k).unwrap_or_default();
    let (fb, rf) = (
        qout.get_mut(..k).unwrap_or_default(),
        qref.get_mut(..k).unwrap_or_default(),
    );
    Squelch::<CQ15>::squelch(&Fallback, &mut a, q, fb)?;
    Squelch::<CQ15>::squelch(&Reference, &mut b, q, rf)?;
    let fac = c.take::<CQ15>(k)?;
    Squelch::<CQ15>::squelch(&Sdrk, &mut d, q, fac)?;
    ensure(
        bit_exact(fb, rf).is_none()
            && bit_exact(fb, fac).is_none()
            && a.average_power() == b.average_power()
            && a.average_power() == d.average_power(),
        || fail("Q15 corners differ between implementations"),
    )?;
    ensure(a.average_power() <= 2.0, || {
        fail("Q15 average power above full scale")
    })?;
    // Alternating full-scale Q15 opens the gate and passes the last sample.
    let mut st = state::<CQ15>()?;
    extreme::full_scale_alternating(qout);
    Squelch::<CQ15>::squelch(&Sdrk, &mut st, qout, qref)?;
    ensure(st.is_open() && qref.last() == qout.last(), || {
        fail("Q15 full scale did not open the gate")
    })?;
    // Invalid parameters are errors.
    ensure(
        SquelchState::<C32>::new(1.0, 3.0, TAU_S, rate()?).is_err()
            && SquelchState::<C32>::new(-40.0, -1.0, TAU_S, rate()?).is_err()
            && SquelchState::<C32>::new(-40.0, 3.0, 0.0, rate()?).is_err()
            && SquelchState::<C32>::new(f64::NAN, 3.0, TAU_S, rate()?).is_err()
            && SquelchState::<C32>::new(-40.0, 3.0, 1e-9, rate()?).is_err(),
        || fail("invalid parameters accepted"),
    )
}

// ---------------------------------------------------------------------------------------
// block
// ---------------------------------------------------------------------------------------

/// Burst amplitude (-20 dBFS) and lengths of the gate test.
const BURST_AMPLITUDE: f64 = 0.1;
const BURST: usize = 1500;
const TAIL: usize = 1500;

/// `C32`/`CQ15` sample of the constant burst.
fn burst_sample<T: ComplexSample>(on: bool) -> T {
    T::from_c64_round(Complex64::new(if on { BURST_AMPLITUDE } else { 0.0 }, 0.0))
}

/// `(open_at, close_at)` of the burst input: the first sample with the gate open and the first
/// sample after the burst with the gate closed, from the recurrence of the kernel written out
/// here (`f32` for `C32`, the Q30 integer for `CQ15`).
fn switching_samples<T: ComplexSample>(input: &[T]) -> (usize, usize) {
    let alpha = 1.0 - libm::exp(-1.0 / (FS as f64 * BLOCK_TAU_S));
    let (open_db, close_db) = (THRESHOLD_DB, THRESHOLD_DB - HYSTERESIS_DB);
    let lin = |db: f64| libm::pow(10.0, db / 10.0);
    let q30 = |p: f64| libm::floor(p * 1_073_741_824.0 + 0.5) as i64;
    let (mut gate, mut open_at, mut close_at) = (false, usize::MAX, usize::MAX);
    let (mut pf, mut pq) = (0.0f32, 0i64);
    for (k, s) in input.iter().enumerate() {
        let c = s.to_c64();
        if T::TYPE == SampleType::CQ15 {
            let alpha_raw = libm::floor(alpha * 32768.0 + 0.5) as i64;
            let re = libm::round(c.re * 32768.0) as i64;
            let im = libm::round(c.im * 32768.0) as i64;
            pq += (alpha_raw * (re * re + im * im - pq) + 16384) >> 15;
            if pq >= q30(lin(open_db)) {
                gate = true;
            } else if pq < q30(lin(close_db)) {
                gate = false;
            }
        } else {
            let (re, im) = (c.re as f32, c.im as f32);
            pf += (alpha as f32) * (re * re + im * im - pf);
            if pf >= lin(open_db) as f32 {
                gate = true;
            } else if pf < lin(close_db) as f32 {
                gate = false;
            }
        }
        if gate && open_at == usize::MAX {
            open_at = k;
        }
        if !gate && k >= BURST && close_at == usize::MAX {
            close_at = k;
        }
    }
    (open_at, close_at)
}

fn gate_for<T>(arena: &mut Arena<'_>) -> Result<(), Failure>
where
    T: BlockSample + ComplexSample + PartialEq,
    Sdrk: Squelch<T>,
{
    const CASE: &str = "block_squelch_gate";
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let cfg = SquelchConfig {
        threshold_db: THRESHOLD_DB,
        hysteresis_db: HYSTERESIS_DB,
        tau_s: BLOCK_TAU_S,
    };
    let mut c = arena.carver();
    let input = c.take::<T>(BURST + TAIL)?;
    let out = c.take::<T>(BURST + TAIL)?;
    for (i, s) in input.iter_mut().enumerate() {
        *s = burst_sample::<T>(i < BURST);
    }
    let mut block = SquelchBlock::<T, _>::new(&cfg, rate()?, &mut ArenaStorage::new(&mut []))?;
    let p = block.process(input, out)?;
    ensure(
        p.consumed == input.len() && p.produced == input.len(),
        || fail("block did not process the whole input"),
    )?;
    // Documented switching samples, exactly: a model of the recurrence written for this case
    // (f32 for C32, the Q30 integer for CQ15) gives the first sample with the gate open and
    // the first one after the burst with the gate closed.
    let (open_at, close_at) = switching_samples::<T>(input);
    let first = out
        .iter()
        .zip(input.iter())
        .position(|(o, i)| *o == *i && *i != burst_sample::<T>(false))
        .ok_or_else(|| fail("the gate never opened"))?;
    ensure(first == open_at, || {
        fail("gate opened at the wrong sample").detail(format_args!("{first}, expected {open_at}"))
    })?;
    let zero = burst_sample::<T>(false);
    ensure(
        out.iter().take(first).all(|s| *s == zero)
            && out
                .iter()
                .skip(first)
                .take(BURST - first)
                .all(|s| *s == burst_sample::<T>(true)),
        || fail("output is not zero before and the input after the opening"),
    )?;
    ensure(close_at > BURST + 100 && close_at < BURST + TAIL, || {
        fail("closing sample out of range")
    })?;
    block.reset();
    ensure(!block.is_open(), || fail("reset left the gate open"))?;
    // Feed everything before the closing sample: still open; with that sample: closed.
    let mut pos = 0usize;
    while pos < close_at {
        let n = (close_at - pos).min(out.len());
        let head = input.get(pos..pos + n).unwrap_or_default();
        block.process(head, out.get_mut(..n).unwrap_or_default())?;
        pos += n;
    }
    ensure(block.is_open(), || fail("the gate closed too early"))?;
    let mut sink = [T::ZERO; 1];
    block.process(
        input.get(close_at..=close_at).unwrap_or_default(),
        &mut sink,
    )?;
    ensure(!block.is_open(), || {
        fail("the gate did not close at the documented sample")
    })?;
    // A runtime threshold takes effect from the next call: -10 dB is above the burst power.
    block.reset();
    block.set_param("threshold_db", ParamValue::F64(-10.0))?;
    let n = BURST.min(out.len());
    block.process(
        input.get(..n).unwrap_or_default(),
        out.get_mut(..n).unwrap_or_default(),
    )?;
    ensure(
        !block.is_open() && out.iter().take(n).all(|s| *s == zero),
        || fail("threshold change was not applied"),
    )
}

pub(super) fn block_squelch_gate(arena: &mut Arena<'_>) -> Result<(), Failure> {
    gate_for::<C32>(arena)?;
    gate_for::<CQ15>(arena)
}

// ---------------------------------------------------------------------------------------
// end to end
// ---------------------------------------------------------------------------------------

fn nfm_case<T>(
    case: &'static str,
    arena: &mut Arena<'_>,
    samples: usize,
    min_sinad_db: f64,
) -> Result<(), Failure>
where
    T: E2eSample + BlockSample,
    Sdrk: Convert<T>
        + Ddc<T>
        + FmDemod<T>
        + QuadDemod<T>
        + Squelch<T>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    let m = measure_nfm::<T>(arena, samples)?;
    ensure(m.audio_samples == samples / 50, || {
        Failure::new(case, "wrong audio sample count").detail(format_args!("{}", m.audio_samples))
    })?;
    ensure(m.sinad_db >= min_sinad_db, || {
        Failure::new(case, "SINAD below the threshold")
            .detail(format_args!("{} dB, need {min_sinad_db}", m.sinad_db))
    })?;
    ensure(
        m.open_delay_s <= NFM_MAX_OPEN_DELAY_S && !m.opened_early,
        || {
            Failure::new(case, "gate opened too late or too early").detail(format_args!(
                "{} s, early {}",
                m.open_delay_s, m.opened_early
            ))
        },
    )?;
    ensure(m.close_delay_s <= NFM_MAX_CLOSE_DELAY_S, || {
        Failure::new(case, "gate closed too late").detail(format_args!("{} s", m.close_delay_s))
    })?;
    ensure(m.zero_checked > 0 && m.zero_violations == 0, || {
        Failure::new(case, "audio is not exactly zero outside the transmission").detail(
            format_args!("{} of {} samples", m.zero_violations, m.zero_checked),
        )
    })
}

pub(super) fn e2e_nfm_rx_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    nfm_case::<C32>(
        "e2e_nfm_rx_f32",
        arena,
        NFM_QUICK_SAMPLES,
        NFM_MIN_SINAD_F32_DB,
    )
}

pub(super) fn e2e_nfm_rx_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    nfm_case::<CQ15>(
        "e2e_nfm_rx_q15",
        arena,
        NFM_QUICK_SAMPLES,
        NFM_MIN_SINAD_Q15_DB,
    )
}

pub(super) fn e2e_nfm_rx_f32_2s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    nfm_case::<C32>(
        "e2e_nfm_rx_f32_2s",
        arena,
        NFM_FULL_SAMPLES,
        NFM_MIN_SINAD_F32_DB,
    )
}

pub(super) fn e2e_nfm_rx_q15_2s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    nfm_case::<CQ15>(
        "e2e_nfm_rx_q15_2s",
        arena,
        NFM_FULL_SAMPLES,
        NFM_MIN_SINAD_Q15_DB,
    )
}
