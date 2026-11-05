//! Suite cases of M33: the kernels `noise_blank` and `auto_notch` and their extreme inputs.
//!
//! Every kernel case compares `Fallback` with `Reference` and the facade with `Fallback` (bit
//! for bit, both kernels are bit-exact in every type) and runs the facade in 20 chunk plans; no
//! files are read, so the cases run on microcontrollers too.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::auto_notch::{AutoNotch, AutoNotchState};
use sdrk_core::kernels::noise_blank::{NoiseBlank, NoiseBlankState};
use sdrk_core::{C32, CQ15, ComplexSample, Fallback, Q15, Rate, Reference};

use super::ensure;
use super::m05::random_c32;
use super::m06::same_rate;
use crate::arena::Arena;
use crate::r#gen::extreme;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Chunk plans per case.
const PLANS: u64 = 20;

// ---------------------------------------------------------------------------------------
// noise_blank
// ---------------------------------------------------------------------------------------

const NB_FS: u64 = 48_000;
const NB_THRESHOLD_DB: f64 = 15.0;
const NB_TAU_S: f64 = 0.002;
/// Hold 0.2 ms: `H = 10` samples at 48 kHz.
const NB_HOLD_S: f64 = 0.0002;
/// Lookahead `D = 5` samples (0.1 ms at 48 kHz).
const NB_D: usize = 5;
/// Length of the test signals.
const NB_N: usize = 1536;
/// The noise amplitude rises by 20 dB (more than the threshold) at this sample.
const NB_STEP_AT: usize = 1000;

fn nb_rate() -> Result<Rate, Failure> {
    Ok(Rate::hz(NB_FS)?)
}

fn nb_state<T: ComplexSample>() -> Result<NoiseBlankState<T, [T; NB_D]>, Failure> {
    Ok(NoiseBlankState::new(
        NB_THRESHOLD_DB,
        NB_TAU_S,
        NB_HOLD_S,
        [T::ZERO; NB_D],
        nb_rate()?,
    )?)
}

/// Noise of amplitude 0.02 (0.2 from `NB_STEP_AT` on) with an impulse of 3 samples and
/// amplitude 0.9 at a random offset in every stretch of 128 samples.
fn nb_signal<T: ComplexSample>(out: &mut [T], seed: u64) {
    let mut rng = Pcg32::new(seed, 1);
    let mut impulse_at = 0usize;
    for (i, o) in out.iter_mut().enumerate() {
        if i % 128 == 0 {
            impulse_at = i + 8 + rng.below(100) as usize;
        }
        let n = random_c32(&mut rng, if i < NB_STEP_AT { 0.02 } else { 0.2 });
        let mut v = Complex64::new(f64::from(n.re), f64::from(n.im));
        if (impulse_at..impulse_at + 3).contains(&i) {
            v += Complex64::new(0.6, -0.6);
        }
        *o = T::from_c64_round(v);
    }
}

/// The output is the input delayed by `NB_D` or zero, both happen, the power step of
/// `NB_STEP_AT` passes again after a while (D363), and the count of zero outputs is the
/// state's `blanked()`.
fn nb_check_output<T: ComplexSample>(
    case: &'static str,
    input: &[T],
    output: &[T],
    blanked: u64,
) -> Result<(), Failure> {
    let (mut passed, mut zeroed) = (0usize, 0u64);
    let delayed = core::iter::repeat_n(T::ZERO, NB_D).chain(input.iter().copied());
    let mut last_zero = 0usize;
    for (i, (o, d)) in output.iter().zip(delayed).enumerate() {
        let one = core::slice::from_ref(o);
        if bit_exact(one, core::slice::from_ref(&d)).is_none() {
            passed += 1;
        } else if bit_exact(one, &[T::ZERO]).is_none() {
            zeroed += 1;
            last_zero = i;
        } else {
            return Err(
                Failure::new(case, "output is neither zero nor the delayed input")
                    .detail(format_args!("index {i}")),
            );
        }
    }
    ensure(passed > 1000 && zeroed > 100, || {
        Failure::new(case, "the blanker did not pass and blank")
            .detail(format_args!("{passed} passed, {zeroed} zeroed"))
    })?;
    ensure(zeroed == blanked, || {
        Failure::new(case, "blanked() differs from the count of zero outputs")
    })?;
    // The last impulse is at most 108 + 3 samples into its stretch; afterwards everything passes.
    ensure(last_zero < NB_N - 8, || {
        Failure::new(case, "the output stayed zero after the power step")
    })
}

fn noise_blank_for<T>(arena: &mut Arena<'_>, case: &'static str, seed: u64) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: NoiseBlank<T>,
    Fallback: NoiseBlank<T>,
    Reference: NoiseBlank<T>,
{
    let mut c = arena.carver();
    let input = c.take::<T>(NB_N)?;
    let (fb, rf, fac) = (c.take::<T>(NB_N)?, c.take::<T>(NB_N)?, c.take::<T>(NB_N)?);
    nb_signal(input, seed);
    let (mut a, mut b, mut d) = (nb_state::<T>()?, nb_state::<T>()?, nb_state::<T>()?);
    NoiseBlank::<T>::noise_blank(&Fallback, &mut a, input, fb)?;
    NoiseBlank::<T>::noise_blank(&Reference, &mut b, input, rf)?;
    NoiseBlank::<T>::noise_blank(&Sdrk, &mut d, input, fac)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "reference differs from the fallback")
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(case, "facade differs from the fallback")
    })?;
    ensure(
        a.average_power() == b.average_power()
            && a.average_power() == d.average_power()
            && a.blanked() == b.blanked()
            && a.blanked() == d.blanked(),
        || Failure::new(case, "final state differs"),
    )?;
    nb_check_output(case, input, fb, a.blanked())?;
    // Unequal lengths are an error.
    let short = fac.get_mut(..NB_N - 1).unwrap_or_default();
    ensure(
        NoiseBlank::<T>::noise_blank(&Sdrk, &mut d, input, short).is_err(),
        || Failure::new(case, "a shorter output was accepted"),
    )?;
    for plan_seed in 0..PLANS {
        let plan = ChunkPlan::new(seed * 100 + plan_seed, 1 + (plan_seed as usize * 37) % 400);
        check_block_invariance(input, fb, rf, plan, nb_state::<T>, |st, i, o| match st {
            Ok(st) => same_rate(i, o, |i, o| NoiseBlank::<T>::noise_blank(&Sdrk, st, i, o)),
            Err(_) => Err(sdrk_core::Error::Internal("noise blank state")),
        })?;
    }
    Ok(())
}

pub(super) fn noise_blank_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    noise_blank_for::<C32>(arena, "noise_blank_c32", 4710)
}

pub(super) fn noise_blank_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    noise_blank_for::<CQ15>(arena, "noise_blank_cq15", 4711)
}

// ---------------------------------------------------------------------------------------
// auto_notch
// ---------------------------------------------------------------------------------------

const AN_TAPS: usize = 16;
const AN_DELAY: usize = 4;
const AN_MU: f32 = 0.05;
const AN_N: usize = 2048;

fn an_state() -> Result<AutoNotchState<[f32; AN_TAPS], [f32; AN_TAPS + AN_DELAY]>, Failure> {
    Ok(AutoNotchState::new(
        [0.0; AN_TAPS],
        [0.0; AN_TAPS + AN_DELAY],
        AN_DELAY,
        AN_MU,
    )?)
}

/// A tone of amplitude 0.5 at 1/8 of the sample rate plus uniform noise of amplitude 0.1.
fn an_signal(out: &mut [f32], seed: u64) {
    let mut rng = Pcg32::new(seed, 1);
    for (i, o) in out.iter_mut().enumerate() {
        let tone = 0.5 * libm::sin(2.0 * core::f64::consts::PI * 0.125 * i as f64);
        *o = (tone + (rng.next_f64() * 2.0 - 1.0) * 0.1) as f32;
    }
}

fn power(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len().max(1) as f64
}

pub(super) fn auto_notch_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "auto_notch_f32";
    let mut c = arena.carver();
    let input = c.take::<f32>(AN_N)?;
    let (fb, rf, fac) = (
        c.take::<f32>(AN_N)?,
        c.take::<f32>(AN_N)?,
        c.take::<f32>(AN_N)?,
    );
    an_signal(input, 4740);
    let (mut a, mut b, mut d) = (an_state()?, an_state()?, an_state()?);
    AutoNotch::auto_notch(&Fallback, &mut a, input, fb)?;
    AutoNotch::auto_notch(&Reference, &mut b, input, rf)?;
    AutoNotch::auto_notch(&Sdrk, &mut d, input, fac)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(CASE, "reference differs from the fallback")
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs from the fallback")
    })?;
    ensure(
        bit_exact(a.weights(), b.weights()).is_none()
            && bit_exact(a.weights(), d.weights()).is_none(),
        || Failure::new(CASE, "final weights differ"),
    )?;
    // The tone is removed: over the last quarter the output keeps about the noise power
    // (0.0033) of an input power of about 0.128.
    let tail = AN_N - AN_N / 4;
    let (p_in, p_out) = (
        power(input.get(tail..).unwrap_or_default()),
        power(fb.get(tail..).unwrap_or_default()),
    );
    ensure(p_out < p_in / 10.0, || {
        Failure::new(CASE, "the tone was not removed")
            .detail(format_args!("in {p_in}, out {p_out}"))
    })?;
    let short = fac.get_mut(..AN_N - 1).unwrap_or_default();
    ensure(
        AutoNotch::auto_notch(&Sdrk, &mut d, input, short).is_err(),
        || Failure::new(CASE, "a shorter output was accepted"),
    )?;
    for plan_seed in 0..PLANS {
        let plan = ChunkPlan::new(474_000 + plan_seed, 1 + (plan_seed as usize * 37) % 400);
        check_block_invariance(input, fb, rf, plan, an_state, |st, i, o| match st {
            Ok(st) => same_rate(i, o, |i, o| AutoNotch::auto_notch(&Sdrk, st, i, o)),
            Err(_) => Err(sdrk_core::Error::Internal("auto notch state")),
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The extreme raw values in every combination; returns how many samples were written.
fn q15_corners(q: &mut [CQ15]) -> usize {
    let corners = [-32768i16, -32767, -1, 0, 1, 32766, 32767];
    let mut k = 0usize;
    for re in corners {
        for im in corners {
            if let Some(slot) = q.get_mut(k) {
                *slot = CQ15 {
                    re: Q15(re),
                    im: Q15(im),
                };
                k += 1;
            }
        }
    }
    k
}

/// `noise_blank` with lookahead 0 and hold 0: an impulse blanks exactly its own sample.
fn nb_no_lookahead_no_hold<T>(case: &'static str, x: &mut [T], out: &mut [T]) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: NoiseBlank<T>,
{
    let small = T::from_c64_round(Complex64::new(0.01, 0.0));
    let big = T::from_c64_round(Complex64::new(0.9, 0.0));
    x.fill(small);
    if let Some(s) = x.get_mut(40) {
        *s = big;
    }
    let mut st = NoiseBlankState::<T, [T; 0]>::new(15.0, 0.002, 0.0, [], nb_rate()?)?;
    NoiseBlank::<T>::noise_blank(&Sdrk, &mut st, x, out)?;
    let only_40 = x.iter().zip(out.iter()).enumerate().all(|(i, (a, b))| {
        let want = if i == 40 { T::ZERO } else { *a };
        bit_exact(core::slice::from_ref(b), core::slice::from_ref(&want)).is_none()
    });
    ensure(only_40 && st.blanked() == 1, || {
        Failure::new(
            case,
            "lookahead 0 and hold 0 did not blank exactly the impulse",
        )
    })
}

pub(super) fn extremes_m33(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m33";
    // A tone at fs/8, amplitude 0.5, for the huge-input check of `auto_notch`.
    const TONE: [f32; 8] = [0.5, 0.354, 0.0, -0.354, -0.5, -0.354, 0.0, 0.354];
    const M: usize = 120;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<C32>(M)?;
    let (out, o1, o2) = (c.take::<C32>(M)?, c.take::<C32>(M)?, c.take::<C32>(M)?);
    let q = c.take::<CQ15>(M)?;
    let (qout, qref, qfac) = (c.take::<CQ15>(M)?, c.take::<CQ15>(M)?, c.take::<CQ15>(M)?);
    let r = c.take::<f32>(M)?;
    let (rout, r1, r2) = (c.take::<f32>(M)?, c.take::<f32>(M)?, c.take::<f32>(M)?);

    // ---- noise_blank ----
    // Silence: the output is zero, nothing is blanked and the average stays zero.
    extreme::silence(x);
    let mut st = nb_state::<C32>()?;
    NoiseBlank::<C32>::noise_blank(&Sdrk, &mut st, x, out)?;
    ensure(
        out.iter()
            .all(|s| s.re.to_bits() == 0 && s.im.to_bits() == 0)
            && st.blanked() == 0
            && st.average_power() == 0.0,
        || fail("silence was not passed as zero"),
    )?;
    extreme::silence(q);
    let mut stq = nb_state::<CQ15>()?;
    NoiseBlank::<CQ15>::noise_blank(&Sdrk, &mut stq, q, qout)?;
    ensure(
        qout.iter().all(|s| *s == CQ15::default()) && stq.blanked() == 0,
        || fail("Q15 silence was not passed as zero"),
    )?;

    // Full scale: a steady full-scale signal is no impulse and passes, delayed by D.
    x.fill(C32::new(1.0, 1.0));
    let mut st = nb_state::<C32>()?;
    NoiseBlank::<C32>::noise_blank(&Sdrk, &mut st, x, out)?;
    ensure(
        st.blanked() == 0
            && out
                .get(NB_D..)
                .is_some_and(|o| o.iter().all(|s| *s == C32::new(1.0, 1.0)))
            && st.average_power() == 2.0,
        || fail("full scale was blanked"),
    )?;
    extreme::full_scale_alternating(q);
    let mut stq = nb_state::<CQ15>()?;
    NoiseBlank::<CQ15>::noise_blank(&Sdrk, &mut stq, q, qout)?;
    ensure(
        stq.blanked() == 0
            && qout
                .get(NB_D..)
                .zip(q.get(..M - NB_D))
                .is_some_and(|(o, i)| o == i)
            && stq.average_power() <= 2.0,
        || fail("Q15 full scale was blanked"),
    )?;

    // NaN and infinity after a warm-up: every non-finite power is an impulse, so no non-finite
    // value reaches the output, and the implementations agree.
    let warm = [C32::new(0.5, 0.5); 64];
    let mut warm_out = [C32::default(); 64];
    extreme::f32_specials_complex(x);
    let mut sts = [nb_state::<C32>()?, nb_state::<C32>()?, nb_state::<C32>()?];
    let [s_fb, s_rf, s_fac] = &mut sts;
    NoiseBlank::<C32>::noise_blank(&Fallback, s_fb, &warm, &mut warm_out)?;
    NoiseBlank::<C32>::noise_blank(&Reference, s_rf, &warm, &mut warm_out)?;
    NoiseBlank::<C32>::noise_blank(&Sdrk, s_fac, &warm, &mut warm_out)?;
    NoiseBlank::<C32>::noise_blank(&Fallback, s_fb, x, out)?;
    NoiseBlank::<C32>::noise_blank(&Reference, s_rf, x, o1)?;
    NoiseBlank::<C32>::noise_blank(&Sdrk, s_fac, x, o2)?;
    let all_same = out.iter().zip(o1.iter()).zip(o2.iter()).all(|((p, r), s)| {
        same_f32(p.re, r.re) && same_f32(p.im, r.im) && same_f32(p.re, s.re) && same_f32(p.im, s.im)
    });
    ensure(
        all_same && s_fb.blanked() == s_rf.blanked() && s_fb.blanked() == s_fac.blanked(),
        || fail("implementations differ on non-finite input"),
    )?;
    ensure(
        out.iter().all(|s| s.re.is_finite() && s.im.is_finite())
            && s_fb.average_power().is_finite(),
        || fail("a non-finite input reached the output or the average"),
    )?;

    // Q15 corners: the implementations agree and the average stays at most full scale.
    let k = q15_corners(q);
    let qk = q.get(..k).unwrap_or_default();
    let (fb, rf, fac) = (
        qout.get_mut(..k).unwrap_or_default(),
        qref.get_mut(..k).unwrap_or_default(),
        qfac.get_mut(..k).unwrap_or_default(),
    );
    let (mut a, mut b, mut d) = (
        nb_state::<CQ15>()?,
        nb_state::<CQ15>()?,
        nb_state::<CQ15>()?,
    );
    NoiseBlank::<CQ15>::noise_blank(&Fallback, &mut a, qk, fb)?;
    NoiseBlank::<CQ15>::noise_blank(&Reference, &mut b, qk, rf)?;
    NoiseBlank::<CQ15>::noise_blank(&Sdrk, &mut d, qk, fac)?;
    ensure(
        bit_exact(fb, rf).is_none()
            && bit_exact(fb, fac).is_none()
            && a.average_power() == b.average_power()
            && a.average_power() == d.average_power()
            && a.blanked() == b.blanked()
            && a.blanked() == d.blanked(),
        || fail("Q15 corners differ between implementations"),
    )?;
    ensure(a.average_power() <= 2.0, || {
        fail("Q15 average power above full scale")
    })?;

    // Lookahead 0 and hold 0.
    nb_no_lookahead_no_hold::<C32>(CASE, x, out)?;
    nb_no_lookahead_no_hold::<CQ15>(CASE, q, qout)?;

    // Invalid parameters are errors.
    let rate = nb_rate()?;
    let h = || [C32::default(); NB_D];
    ensure(
        NoiseBlankState::<C32, _>::new(5.9, NB_TAU_S, NB_HOLD_S, h(), rate).is_err()
            && NoiseBlankState::<C32, _>::new(40.1, NB_TAU_S, NB_HOLD_S, h(), rate).is_err()
            && NoiseBlankState::<C32, _>::new(f64::NAN, NB_TAU_S, NB_HOLD_S, h(), rate).is_err()
            && NoiseBlankState::<C32, _>::new(15.0, 0.0009, NB_HOLD_S, h(), rate).is_err()
            && NoiseBlankState::<C32, _>::new(15.0, NB_TAU_S, 0.011, h(), rate).is_err()
            && NoiseBlankState::<C32, _>::new(
                15.0,
                NB_TAU_S,
                NB_HOLD_S,
                [C32::default(); 49],
                rate,
            )
            .is_err(),
        || fail("invalid noise blank parameters accepted"),
    )?;

    // ---- auto_notch ----
    // Silence: zero output and zero weights.
    extreme::silence(r);
    let mut st = an_state()?;
    AutoNotch::auto_notch(&Sdrk, &mut st, r, rout)?;
    ensure(
        rout.iter().all(|v| v.to_bits() == 0) && st.weights().iter().all(|w| *w == 0.0),
        || fail("auto notch silence"),
    )?;
    // Full scale alternating (a tone at fs/2): finite output and weights, and the tone is
    // removed after the adaptation.
    for (i, v) in r.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let mut st = an_state()?;
    for _ in 0..8 {
        AutoNotch::auto_notch(&Sdrk, &mut st, r, rout)?;
    }
    ensure(
        rout.iter().all(|v| v.is_finite())
            && st.weights().iter().all(|w| w.is_finite())
            && power(rout) < 0.01,
        || fail("auto notch full scale"),
    )?;
    // NaN and infinity: a non-finite input gives +0.0, and the implementations agree.
    extreme::f32_specials(r);
    let (mut a, mut b, mut d) = (an_state()?, an_state()?, an_state()?);
    AutoNotch::auto_notch(&Fallback, &mut a, r, rout)?;
    AutoNotch::auto_notch(&Reference, &mut b, r, r1)?;
    AutoNotch::auto_notch(&Sdrk, &mut d, r, r2)?;
    let all_same = rout
        .iter()
        .zip(r1.iter())
        .zip(r2.iter())
        .all(|((p, q), s)| same_f32(*p, *q) && same_f32(*p, *s));
    ensure(all_same, || {
        fail("auto notch implementations differ on non-finite input")
    })?;
    ensure(
        r.iter()
            .zip(rout.iter())
            .all(|(i, o)| i.is_finite() || o.to_bits() == 0),
        || fail("a non-finite input did not give +0.0"),
    )?;
    // A huge finite input (1e35 after silence, f32::MAX inside a tone) must not turn the
    // weights or the following outputs non-finite (D422); the implementations agree.
    for (huge, tone) in [(1e35f32, false), (f32::MAX, true)] {
        for (v, t) in r.iter_mut().zip(TONE.iter().cycle()) {
            *v = if tone { *t } else { 0.0 };
        }
        let at = if tone { M / 2 } else { 0 };
        if let Some(v) = r.get_mut(at) {
            *v = huge;
        }
        let (mut a, mut b, mut d) = (an_state()?, an_state()?, an_state()?);
        for round in 0..2 {
            AutoNotch::auto_notch(&Fallback, &mut a, r, rout)?;
            AutoNotch::auto_notch(&Reference, &mut b, r, r1)?;
            AutoNotch::auto_notch(&Sdrk, &mut d, r, r2)?;
            let all_same = rout
                .iter()
                .zip(r1.iter())
                .zip(r2.iter())
                .all(|((p, q), s)| same_f32(*p, *q) && same_f32(*p, *s));
            ensure(all_same, || {
                fail("auto notch implementations differ on a huge input")
            })?;
            ensure(
                rout.iter().all(|v| v.is_finite())
                    && a.weights().iter().all(|w| w.is_finite())
                    && b.weights().iter().all(|w| w.is_finite()),
                || fail("a huge finite input made the auto notch non-finite"),
            )?;
            // The second round feeds only finite, moderate samples.
            if round == 0
                && let Some(v) = r.get_mut(at)
            {
                *v = 0.25;
            }
        }
    }
    // Invalid parameters are errors.
    ensure(
        AutoNotchState::new([0.0f32; 7], [0.0f32; 8], 1, 0.01).is_err()
            && AutoNotchState::new([0.0f32; 8], [0.0f32; 8], 0, 0.01).is_err()
            && AutoNotchState::new([0.0f32; 8], [0.0f32; 73], 65, 0.01).is_err()
            && AutoNotchState::new([0.0f32; 8], [0.0f32; 9], 1, 0.2).is_err()
            && AutoNotchState::new([0.0f32; 8], [0.0f32; 9], 1, f32::NAN).is_err()
            && AutoNotchState::new([0.0f32; 8], [0.0f32; 10], 1, 0.01).is_err(),
        || fail("invalid auto notch parameters accepted"),
    )
}
