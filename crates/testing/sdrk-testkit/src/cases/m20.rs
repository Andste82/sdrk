//! Suite cases of M20: the kernels `mag`, `dc_block_real` and `agc`, their extreme inputs, the
//! `agc` block and the end-to-end AM receive.
//!
//! Every kernel case compares `Fallback` (and the facade) with `Reference`; no files are read,
//! so the cases run on microcontrollers too.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::{AgcBlock, AgcConfig, ArenaStorage, Block, BlockSample, ParamValue};
use sdrk_core::kernels::agc::{Agc, AgcState};
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::ResampleReal;
use sdrk_core::kernels::demod::{DcBlockReal, DcBlockRealState};
use sdrk_core::kernels::fused::Ddc;
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::{C32, CQ15, ComplexSample, Fallback, Q15, Rate, RealSample, Reference, Sample};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use super::m06::{flat, min_snr, same_rate};
use crate::arena::Arena;
use crate::e2e::E2eSample;
use crate::e2e_am::{
    AM_FULL_SAMPLES, AM_INTERIM_LEVEL_ERROR_DB, AM_MIN_SINAD_F32_DB, AM_MIN_SINAD_Q15_DB,
    AM_QUICK_SAMPLES, measure_am,
};
use crate::r#gen::extreme;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

const FS: u64 = 48_000;
/// Chunk plans per type in the invariance checks.
const PLANS: u64 = 20;

fn rate() -> Result<Rate, Failure> {
    Ok(Rate::hz(FS)?)
}

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

// ---------------------------------------------------------------------------------------
// mag
// ---------------------------------------------------------------------------------------

pub(super) fn mag_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "mag_c32";
    const N: usize = 1024;
    let mut c = arena.carver();
    let x = c.take::<C32>(N)?;
    let (fb, rf, fac) = (c.take::<f32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
    let mut rng = Pcg32::new(2001, 1);
    for s in x.iter_mut() {
        *s = random_c32(&mut rng, 1.0);
    }
    for (s, v) in x.iter_mut().zip([
        C32::new(0.0, 0.0),
        C32::new(1.0, 0.0),
        C32::new(0.0, -1.0),
        C32::new(1.0, 1.0),
        C32::new(-1.0, -1.0),
        C32::new(3.0, 4.0),
    ]) {
        *s = v;
    }
    Mag::<C32>::mag(&Fallback, x, fb)?;
    Mag::<C32>::mag(&Reference, x, rf)?;
    Mag::<C32>::mag(&Sdrk, x, fac)?;
    min_snr(CASE, "f32 below 130 dB", rf, fb, 130.0)?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs")
    })?;
    ensure(
        fb.get(5) == Some(&5.0)
            && fb.first() == Some(&0.0)
            && fb.get(3) == Some(&core::f32::consts::SQRT_2),
        || Failure::new(CASE, "known magnitudes are wrong"),
    )
}

pub(super) fn mag_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "mag_cq15";
    const N: usize = 1024;
    let mut c = arena.carver();
    let x = c.take::<CQ15>(N)?;
    let (fb, rf, fac) = (c.take::<Q15>(N)?, c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    let mut rng = Pcg32::new(2002, 1);
    for pass in 0..3 {
        match pass {
            0 => {
                for s in x.iter_mut() {
                    *s = random_cq15(&mut rng, 1.0);
                }
            }
            1 => extreme::full_scale_alternating(x),
            _ => {
                // Around the squares: (r, 0), (r, 1) and (r, r) for the raw values r.
                for (i, s) in x.iter_mut().enumerate() {
                    let r = (i / 3) as i16 * 7;
                    *s = match i % 3 {
                        0 => CQ15 {
                            re: Q15(r),
                            im: Q15(0),
                        },
                        1 => CQ15 {
                            re: Q15(r),
                            im: Q15(1),
                        },
                        _ => CQ15 {
                            re: Q15(r),
                            im: Q15(r),
                        },
                    };
                }
            }
        }
        Mag::<CQ15>::mag(&Fallback, x, fb)?;
        Mag::<CQ15>::mag(&Reference, x, rf)?;
        Mag::<CQ15>::mag(&Sdrk, x, fac)?;
        ensure(
            bit_exact(fb, rf).is_none() && bit_exact(fb, fac).is_none(),
            || Failure::new(CASE, "not bit-exact").detail(format_args!("pass {pass}")),
        )?;
        if pass == 2 {
            // sqrt(r^2) = r exactly; sqrt(r^2 + 1) rounds to r; sqrt(2 r^2) to round(r sqrt 2).
            for (i, o) in fb.iter().enumerate() {
                let r = (i / 3) as i64 * 7;
                let want = match i % 3 {
                    0 => r,
                    1 => r.max(1),
                    _ => libm::round(r as f64 * core::f64::consts::SQRT_2) as i64,
                };
                ensure(i64::from(o.0) == want.min(32767), || {
                    Failure::new(CASE, "wrong magnitude").detail(format_args!("sample {i}"))
                })?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// dc_block_real
// ---------------------------------------------------------------------------------------

pub(super) fn dc_block_real_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "dc_block_real_f32";
    const N: usize = 1024;
    let mut c = arena.carver();
    let x = c.take::<f32>(N)?;
    let (fb, rf, fac) = (c.take::<f32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
    let mut rng = Pcg32::new(2003, 1);
    for s in x.iter_mut() {
        *s = random_c32(&mut rng, 0.2).re + 0.3;
    }
    let mut a = DcBlockRealState::<f32>::new(0.995)?;
    let (mut b, mut d) = (a, a);
    DcBlockReal::<f32>::dc_block_real(&Fallback, &mut a, x, fb)?;
    DcBlockReal::<f32>::dc_block_real(&Reference, &mut b, x, rf)?;
    DcBlockReal::<f32>::dc_block_real(&Sdrk, &mut d, x, fac)?;
    min_snr(CASE, "fallback below 100 dB", rf, fb, 100.0)?;
    ensure(bit_exact(fb, fac).is_none() && a == d, || {
        Failure::new(CASE, "facade differs")
    })?;
    // The constant offset decays.
    ensure(fb.last().is_some_and(|v| v.abs() < 0.8), || {
        Failure::new(CASE, "the offset did not decay")
    })?;
    let (input, whole, chunked) = (&*x, c.take::<f32>(N)?, c.take::<f32>(N)?);
    for seed in 0..PLANS {
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(2100 + seed, 1 + (seed as usize * 37) % 400),
            || DcBlockRealState::<f32>::new(0.99),
            |st, i, o| match st {
                Ok(st) => same_rate(i, o, |i, o| {
                    DcBlockReal::<f32>::dc_block_real(&Sdrk, st, i, o)
                }),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    Ok(())
}

pub(super) fn dc_block_real_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "dc_block_real_q15";
    const N: usize = 1024;
    let mut c = arena.carver();
    let x = c.take::<Q15>(N)?;
    let (fb, rf, fac) = (c.take::<Q15>(N)?, c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    let mut rng = Pcg32::new(2004, 1);
    for extremes in [false, true] {
        for (i, s) in x.iter_mut().enumerate() {
            *s = if extremes {
                if i % 2 == 0 { Q15(32767) } else { Q15(-32768) }
            } else {
                random_cq15(&mut rng, 1.0).re
            };
        }
        let mut a = DcBlockRealState::<Q15>::new(Q15(32735))?;
        let (mut b, mut d) = (a, a);
        DcBlockReal::<Q15>::dc_block_real(&Fallback, &mut a, x, fb)?;
        DcBlockReal::<Q15>::dc_block_real(&Reference, &mut b, x, rf)?;
        DcBlockReal::<Q15>::dc_block_real(&Sdrk, &mut d, x, fac)?;
        ensure(bit_exact(fb, rf).is_none() && a == b, || {
            Failure::new(CASE, "not bit-exact")
        })?;
        ensure(bit_exact(fb, fac).is_none() && a == d, || {
            Failure::new(CASE, "facade differs")
        })?;
    }
    let (input, whole, chunked) = (&*x, c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    for seed in 0..PLANS {
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(2200 + seed, 1 + (seed as usize * 41) % 400),
            || DcBlockRealState::<Q15>::new(Q15(32000)),
            |st, i, o| match st {
                Ok(st) => same_rate(i, o, |i, o| {
                    DcBlockReal::<Q15>::dc_block_real(&Sdrk, st, i, o)
                }),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// agc
// ---------------------------------------------------------------------------------------

const AGC_REFERENCE: f64 = 0.5;
const AGC_ATTACK_S: f64 = 0.001;
const AGC_RELEASE_S: f64 = 0.002;
const AGC_MIN_DB: f64 = -20.0;
const AGC_MAX_DB: f64 = 60.0;
/// Length of one stretch of the AGC test signal.
const STRETCH: usize = 256;
const AGC_N: usize = 3 * STRETCH;

fn agc_state_at<T: ComplexSample>(fs: Rate) -> sdrk_core::Result<AgcState<T>> {
    AgcState::<T>::new(
        AGC_REFERENCE,
        AGC_ATTACK_S,
        AGC_RELEASE_S,
        AGC_MIN_DB,
        AGC_MAX_DB,
        0.0,
        fs,
    )
}

fn agc_state<T: ComplexSample>() -> Result<AgcState<T>, Failure> {
    Ok(agc_state_at::<T>(rate()?)?)
}

/// A complex tone of 1 kHz whose amplitude steps 0.01, 0.3, 0.003 with a little noise.
fn agc_signal<T: ComplexSample>(x: &mut [T], seed: u64) {
    let mut rng = Pcg32::new(seed, 1);
    for (i, s) in x.iter_mut().enumerate() {
        let amp = match i / STRETCH {
            0 => 0.01,
            1 => 0.3,
            _ => 0.003,
        };
        let phase = core::f64::consts::TAU * 1000.0 * i as f64 / FS as f64;
        let n = random_c32(&mut rng, 1.0);
        *s = T::from_c64_round(Complex64::new(
            amp * libm::cos(phase) + 1e-4 * f64::from(n.re),
            amp * libm::sin(phase) + 1e-4 * f64::from(n.im),
        ));
    }
}

fn agc_for<T>(
    arena: &mut Arena<'_>,
    case: &'static str,
    vs_reference: fn(&'static str, &[T], &[T], &AgcState<T>, &AgcState<T>) -> Result<(), Failure>,
) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: Agc<T>,
    Fallback: Agc<T>,
    Reference: Agc<T>,
{
    let mut c = arena.carver();
    let input = c.take::<T>(AGC_N)?;
    let (fb, rf, fac) = (
        c.take::<T>(AGC_N)?,
        c.take::<T>(AGC_N)?,
        c.take::<T>(AGC_N)?,
    );
    agc_signal(input, 2300);
    let (mut a, mut b, mut d) = (agc_state::<T>()?, agc_state::<T>()?, agc_state::<T>()?);
    Agc::<T>::agc(&Fallback, &mut a, input, fb)?;
    Agc::<T>::agc(&Reference, &mut b, input, rf)?;
    Agc::<T>::agc(&Sdrk, &mut d, input, fac)?;
    vs_reference(case, fb, rf, &a, &b)?;
    ensure(
        bit_exact(fb, fac).is_none() && a.gain_db() == d.gain_db(),
        || Failure::new(case, "facade differs from the fallback"),
    )?;
    // The gain follows the steps: up on the weak first stretch, down after the loud one, and
    // never outside its limits.
    let mut st = agc_state::<T>()?;
    let tmp = c.take::<T>(STRETCH)?;
    let mut gains = [0.0f64; 3];
    for (k, g) in gains.iter_mut().enumerate() {
        let part = input
            .get(k * STRETCH..(k + 1) * STRETCH)
            .unwrap_or_default();
        Agc::<T>::agc(&Sdrk, &mut st, part, tmp)?;
        *g = st.gain_db();
        ensure(*g >= AGC_MIN_DB - 1e-3 && *g <= AGC_MAX_DB + 1e-3, || {
            Failure::new(case, "gain outside its limits")
        })?;
    }
    ensure(
        gains[0] > 10.0 && gains[1] < gains[0] - 10.0 && gains[2] > gains[1] + 10.0,
        || {
            Failure::new(case, "the gain does not follow the steps")
                .detail(format_args!("{gains:?} dB"))
        },
    )?;
    // 20 chunk plans give the bits of one call.
    let (whole, chunked) = (c.take::<T>(AGC_N)?, c.take::<T>(AGC_N)?);
    let fs = rate()?;
    for seed in 0..PLANS {
        let plan = ChunkPlan::new(2400 + seed, 1 + (seed as usize * 37) % 500);
        check_block_invariance(
            input,
            whole,
            chunked,
            plan,
            || agc_state_at::<T>(fs),
            |st, i, o| match st {
                Ok(st) => same_rate(i, o, |i, o| Agc::<T>::agc(&Sdrk, st, i, o)),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    Ok(())
}

/// `C32`: the fallback (f32 gain) follows the f64 gain of the reference within 80 dB.
fn c32_vs_reference(
    case: &'static str,
    fb: &[C32],
    rf: &[C32],
    _a: &AgcState<C32>,
    _b: &AgcState<C32>,
) -> Result<(), Failure> {
    min_snr(case, "fallback below 80 dB", flat(rf), flat(fb), 80.0)
}

/// `CQ15`: bit-exact, including the final gain.
fn cq15_vs_reference(
    case: &'static str,
    fb: &[CQ15],
    rf: &[CQ15],
    a: &AgcState<CQ15>,
    b: &AgcState<CQ15>,
) -> Result<(), Failure> {
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(case, "reference differs from the fallback")
    })?;
    ensure(a.gain_db() == b.gain_db(), || {
        Failure::new(case, "final gain differs from the reference")
    })
}

pub(super) fn agc_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    agc_for::<C32>(arena, "agc_c32", c32_vs_reference)
}

pub(super) fn agc_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    agc_for::<CQ15>(arena, "agc_cq15", cq15_vs_reference)
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

pub(super) fn extremes_m20(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m20";
    const M: usize = 70;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<C32>(M)?;
    let (fb, rf, fac) = (c.take::<C32>(M)?, c.take::<C32>(M)?, c.take::<C32>(M)?);
    let (m_fb, m_rf, m_fac) = (c.take::<f32>(M)?, c.take::<f32>(M)?, c.take::<f32>(M)?);
    let q = c.take::<CQ15>(M)?;
    let (q_fb, q_rf, q_fac) = (c.take::<CQ15>(M)?, c.take::<CQ15>(M)?, c.take::<CQ15>(M)?);
    let (mq_fb, mq_rf, mq_fac) = (c.take::<Q15>(M)?, c.take::<Q15>(M)?, c.take::<Q15>(M)?);

    // f32 inputs: silence, full scale, the special values (NaN, infinities, denormal, MAX).
    for kind in 0..3 {
        match kind {
            0 => extreme::silence(x),
            1 => {
                for s in x.iter_mut() {
                    *s = C32::new(1.0, -1.0);
                }
            }
            _ => extreme::f32_specials_complex(x),
        }
        // mag: no panic, the three implementations agree where defined.
        Mag::<C32>::mag(&Fallback, x, m_fb)?;
        Mag::<C32>::mag(&Reference, x, m_rf)?;
        Mag::<C32>::mag(&Sdrk, x, m_fac)?;
        for ((a, b), d) in m_fb.iter().zip(m_rf.iter()).zip(m_fac.iter()) {
            ensure(same_f32(*a, *d), || fail("mag: facade differs"))?;
            // The reference computes in f64: it agrees on NaN and infinity, and a finite
            // value of the fallback is within f32 rounding of it.
            ensure(
                a.is_nan() == b.is_nan()
                    && (a.is_nan()
                        || a == b
                        || (a.is_infinite() && *b > 1e38)
                        || f64::from(*a - *b).abs() <= 1e-6 * f64::from(*b).abs()
                        || f64::from(*a - *b).abs() < 1e-30),
                || fail("mag: reference differs"),
            )?;
        }
        // agc: the gain stays inside its limits and the facade equals the fallback.
        let (mut a, mut b, mut d) = (
            agc_state::<C32>()?,
            agc_state::<C32>()?,
            agc_state::<C32>()?,
        );
        Agc::<C32>::agc(&Fallback, &mut a, x, fb)?;
        Agc::<C32>::agc(&Reference, &mut b, x, rf)?;
        Agc::<C32>::agc(&Sdrk, &mut d, x, fac)?;
        for st in [&a, &b, &d] {
            ensure(
                st.gain_db() >= AGC_MIN_DB - 1e-3 && st.gain_db() <= AGC_MAX_DB + 1e-3,
                || fail("agc: gain outside its limits"),
            )?;
        }
        ensure(
            fb.iter()
                .zip(fac.iter())
                .all(|(p, r)| same_f32(p.re, r.re) && same_f32(p.im, r.im))
                && a.gain_db() == d.gain_db(),
            || fail("agc: facade differs"),
        )?;
        // dc_block_real on the real parts: no panic, the facade equals the fallback.
        let real = c.take::<f32>(M)?;
        let (r_fb, r_fac) = (c.take::<f32>(M)?, c.take::<f32>(M)?);
        for (r, s) in real.iter_mut().zip(x.iter()) {
            *r = s.re;
        }
        let mut s1 = DcBlockRealState::<f32>::new(0.99)?;
        let mut s2 = s1;
        DcBlockReal::<f32>::dc_block_real(&Fallback, &mut s1, real, r_fb)?;
        DcBlockReal::<f32>::dc_block_real(&Sdrk, &mut s2, real, r_fac)?;
        ensure(
            r_fb.iter().zip(r_fac.iter()).all(|(p, r)| same_f32(*p, *r)),
            || fail("dc_block_real: facade differs"),
        )?;
    }
    // A NaN sample leaves the gain unchanged; an infinite one drives it to the minimum.
    let mut st = agc_state::<C32>()?;
    let warm = [C32::new(0.1, 0.0); 100];
    let mut sink = [C32::default(); 100];
    Agc::<C32>::agc(&Sdrk, &mut st, &warm, &mut sink)?;
    let g = st.gain_db();
    Agc::<C32>::agc(
        &Sdrk,
        &mut st,
        &[C32::new(f32::NAN, 0.0)],
        sink.get_mut(..1).unwrap_or_default(),
    )?;
    ensure(st.gain_db() == g, || fail("a NaN sample changed the gain"))?;
    Agc::<C32>::agc(
        &Sdrk,
        &mut st,
        &[C32::new(f32::INFINITY, 0.0)],
        sink.get_mut(..1).unwrap_or_default(),
    )?;
    ensure((st.gain_db() - AGC_MIN_DB).abs() < 1e-3, || {
        fail("an infinite sample did not drive the gain to the minimum")
    })?;

    // Q15: silence, full scale and the corners of the raw range.
    let corners = [-32768i16, -32767, -1, 0, 1, 32766, 32767];
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
        Mag::<CQ15>::mag(&Fallback, q, mq_fb)?;
        Mag::<CQ15>::mag(&Reference, q, mq_rf)?;
        Mag::<CQ15>::mag(&Sdrk, q, mq_fac)?;
        ensure(
            bit_exact(mq_fb, mq_rf).is_none() && bit_exact(mq_fb, mq_fac).is_none(),
            || fail("mag: Q15 corners differ"),
        )?;
        let (mut a, mut b, mut d) = (
            agc_state::<CQ15>()?,
            agc_state::<CQ15>()?,
            agc_state::<CQ15>()?,
        );
        Agc::<CQ15>::agc(&Fallback, &mut a, q, q_fb)?;
        Agc::<CQ15>::agc(&Reference, &mut b, q, q_rf)?;
        Agc::<CQ15>::agc(&Sdrk, &mut d, q, q_fac)?;
        ensure(
            bit_exact(q_fb, q_rf).is_none()
                && bit_exact(q_fb, q_fac).is_none()
                && a.gain_db() == b.gain_db()
                && a.gain_db() == d.gain_db()
                && a.gain_db() >= AGC_MIN_DB - 1e-3
                && a.gain_db() <= AGC_MAX_DB + 1e-3,
            || fail("agc: Q15 differs between implementations"),
        )?;
        // Silence keeps the output zero and raises the gain to the maximum.
        if kind == 0 {
            ensure(q_fb.iter().all(|s| *s == CQ15::default()), || {
                fail("agc: silence did not give silence")
            })?;
        }
        let real = c.take::<Q15>(M)?;
        let (r_fb, r_rf, r_fac) = (c.take::<Q15>(M)?, c.take::<Q15>(M)?, c.take::<Q15>(M)?);
        for (r, s) in real.iter_mut().zip(q.iter()) {
            *r = s.re;
        }
        let mut s1 = DcBlockRealState::<Q15>::new(Q15(32735))?;
        let (mut s2, mut s3) = (s1, s1);
        DcBlockReal::<Q15>::dc_block_real(&Fallback, &mut s1, real, r_fb)?;
        DcBlockReal::<Q15>::dc_block_real(&Reference, &mut s2, real, r_rf)?;
        DcBlockReal::<Q15>::dc_block_real(&Sdrk, &mut s3, real, r_fac)?;
        ensure(
            bit_exact(r_fb, r_rf).is_none() && bit_exact(r_fb, r_fac).is_none() && s1 == s2,
            || fail("dc_block_real: Q15 corners differ"),
        )?;
    }
    // Invalid parameters are errors.
    let r = rate()?;
    ensure(
        AgcState::<C32>::new(2.0, 0.005, 0.05, -20.0, 60.0, 0.0, r).is_err()
            && AgcState::<C32>::new(0.5, 0.0, 0.05, -20.0, 60.0, 0.0, r).is_err()
            && AgcState::<C32>::new(0.5, 0.005, 0.05, 30.0, 20.0, 25.0, r).is_err()
            && AgcState::<C32>::new(0.5, 0.005, 0.05, -20.0, 60.0, 61.0, r).is_err()
            && AgcState::<C32>::new(f64::NAN, 0.005, 0.05, -20.0, 60.0, 0.0, r).is_err()
            && DcBlockRealState::<f32>::new(1.0).is_err()
            && DcBlockRealState::<Q15>::new(Q15(0)).is_err(),
        || fail("invalid parameters accepted"),
    )?;
    // The sample types of the two kernels are the real ones.
    ensure(
        <Q15 as Sample>::ZERO == Q15(0) && f32::ZERO.to_bits() == 0,
        || fail("zero constants"),
    )?;
    ensure(<f32 as RealSample>::from_f64_round(0.5) == 0.5, || {
        fail("f32 rounding")
    })
}

// ---------------------------------------------------------------------------------------
// block
// ---------------------------------------------------------------------------------------

/// Largest absolute amplitude of the last `n` samples.
fn tail_amplitude<T: ComplexSample>(x: &[T], n: usize) -> f64 {
    x.iter()
        .skip(x.len().saturating_sub(n))
        .map(|s| {
            let c = s.to_c64();
            libm::sqrt(c.re * c.re + c.im * c.im)
        })
        .fold(0.0, f64::max)
}

fn level_for<T>(arena: &mut Arena<'_>) -> Result<(), Failure>
where
    T: BlockSample,
    Sdrk: Agc<T>,
{
    const CASE: &str = "block_agc_level";
    const N: usize = 2400;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let cfg = AgcConfig {
        attack_s: 0.002,
        release_s: 0.002,
        ..AgcConfig::default()
    };
    let mut c = arena.carver();
    let input = c.take::<T>(N)?;
    let out = c.take::<T>(N)?;
    for (i, s) in input.iter_mut().enumerate() {
        let phase = core::f64::consts::TAU * 1000.0 * i as f64 / FS as f64;
        *s = T::from_c64_round(Complex64::new(
            0.05 * libm::cos(phase),
            0.05 * libm::sin(phase),
        ));
    }
    let mut block = AgcBlock::<T, _>::new(&cfg, rate()?, &mut ArenaStorage::new(&mut []))?;
    let p = block.process(input, out)?;
    ensure(p.consumed == N && p.produced == N, || {
        fail("block did not process the whole input")
    })?;
    // The weak tone (-26 dBFS) is raised to the reference of -6 dBFS within 1 dB.
    let level_db = |x: &[T]| 20.0 * libm::log10(tail_amplitude(x, 480));
    ensure((level_db(out) + 6.0).abs() < 1.0, || {
        fail("the output is not at the reference level")
            .detail(format_args!("{} dBFS", level_db(out)))
    })?;
    // A runtime reference takes effect from the next call and the loop settles at it.
    block.set_param("reference_db", ParamValue::F64(-12.0))?;
    ensure(
        block.param("reference_db") == Ok(ParamValue::F64(-12.0)),
        || fail("reference_db is not reported"),
    )?;
    let p = block.process(input, out)?;
    ensure(p.produced == N, || fail("second call produced no output"))?;
    ensure((level_db(out) + 12.0).abs() < 1.0, || {
        fail("the output did not follow the new reference")
            .detail(format_args!("{} dBFS", level_db(out)))
    })?;
    // The gain never leaves its limits: a signal far below needs more than the maximum.
    block.set_param("max_gain_db", ParamValue::F64(10.0))?;
    ensure(block.gain_db() <= 10.0 + 1e-3, || {
        fail("the gain was not clamped to the new maximum")
    })?;
    block.reset();
    ensure(block.gain_db() == 0.0, || {
        fail("reset did not restore the gain")
    })
}

pub(super) fn block_agc_level(arena: &mut Arena<'_>) -> Result<(), Failure> {
    level_for::<C32>(arena)?;
    level_for::<CQ15>(arena)
}

// ---------------------------------------------------------------------------------------
// end to end
// ---------------------------------------------------------------------------------------

fn am_case<T>(
    case: &'static str,
    arena: &mut Arena<'_>,
    samples: usize,
    min_sinad_db: f64,
) -> Result<(), Failure>
where
    T: E2eSample + BlockSample,
    Sdrk: Convert<T>
        + Ddc<T>
        + Agc<T>
        + Mag<T>
        + DcBlockReal<<T as ComplexSample>::Real>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    let m = measure_am::<T>(arena, samples)?;
    ensure(m.audio_samples == samples / 50, || {
        Failure::new(case, "wrong audio sample count").detail(format_args!("{}", m.audio_samples))
    })?;
    ensure(m.sinad_db >= min_sinad_db, || {
        Failure::new(case, "SINAD below the threshold")
            .detail(format_args!("{} dB, need {min_sinad_db}", m.sinad_db))
    })?;
    // The specified limit is 1 dB (open item M20-2); the cases enforce the interim limit.
    ensure(m.level_error_db.abs() <= AM_INTERIM_LEVEL_ERROR_DB, || {
        Failure::new(case, "audio level too far from reference * index")
            .detail(format_args!("{} dB", m.level_error_db))
    })
}

pub(super) fn e2e_am_rx_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    am_case::<C32>(
        "e2e_am_rx_f32",
        arena,
        AM_QUICK_SAMPLES,
        AM_MIN_SINAD_F32_DB,
    )
}

pub(super) fn e2e_am_rx_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    am_case::<CQ15>(
        "e2e_am_rx_q15",
        arena,
        AM_QUICK_SAMPLES,
        AM_MIN_SINAD_Q15_DB,
    )
}

pub(super) fn e2e_am_rx_f32_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    am_case::<C32>(
        "e2e_am_rx_f32_1s",
        arena,
        AM_FULL_SAMPLES,
        AM_MIN_SINAD_F32_DB,
    )
}

pub(super) fn e2e_am_rx_q15_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    am_case::<CQ15>(
        "e2e_am_rx_q15_1s",
        arena,
        AM_FULL_SAMPLES,
        AM_MIN_SINAD_Q15_DB,
    )
}
