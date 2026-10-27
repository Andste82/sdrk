//! Suite cases of M22: the kernels `biquad` and `biquad_real` and their extreme inputs.
//!
//! Every kernel case compares `Fallback` (and the facade) with `Reference`, runs the facade in
//! 20 chunk plans and checks one property of the design; no files are read, so the cases run
//! on microcontrollers too.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::design::{BiquadResponse, biquad_coefs, butterworth_sos, to_f32_sos, to_q14};
use sdrk_blocks::{
    ARENA_ALIGN, ArenaStorage, BiquadBlock, BiquadConfig, BiquadElem, BiquadRealBlock, Block,
    BlockReal, BlockSample, ParamValue,
};
use sdrk_core::kernels::biquad::{
    Biquad, BiquadReal, BiquadRealState, BiquadSample, BiquadState, MAX_SECTIONS,
};
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::ResampleReal;
use sdrk_core::kernels::demod::QuadDemod;
use sdrk_core::kernels::fused::{Ddc, FmDemod};
use sdrk_core::kernels::squelch::Squelch;
use sdrk_core::{C32, CQ15, ComplexSample, Fallback, Q15, Rate, RealSample, Reference, Sample};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use super::m06::{flat, hand_out, min_snr, same_rate};
use crate::arena::Arena;
use crate::e2e::E2eSample;
use crate::e2e_nfm::{
    NFM_MAX_CLOSE_DELAY_S, NFM_MAX_OPEN_DELAY_S, NFM_MAX_SINAD_LOSS_DB_F32,
    NFM_MAX_SINAD_LOSS_DB_Q15, NFM_MIN_SUBTONE_DB_F32, NFM_MIN_SUBTONE_DB_Q15, NFM_QUICK_SAMPLES,
    measure_nfm, measure_nfm_ctcss,
};
use crate::r#gen::extreme;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::meters::ToneAccumulator;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

const FS: f64 = 48_000.0;
/// Samples per kernel case.
const N: usize = 1024;
/// Sections of the cascades of the cases (a fourth-order Butterworth filter).
const SECTIONS: usize = 2;
/// Chunk plans per case.
const PLANS: u64 = 20;

type Coefs<T> = [[<T as BiquadSample>::Coef; 5]; SECTIONS];

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// A fourth-order Butterworth low-pass at 4.8 kHz of the 48 kHz rate, as `f32` coefficients.
fn lowpass4_f32() -> Result<Coefs<C32>, Failure> {
    let mut sos = [[0.0f64; 5]; SECTIONS];
    butterworth_sos(BiquadResponse::Lowpass, 4, 4_800.0, FS, &mut sos).map_err(design_error)?;
    let mut out = [[0.0f32; 5]; SECTIONS];
    to_f32_sos(&sos, &mut out).map_err(design_error)?;
    Ok(out)
}

/// The same filter as Q2.14 coefficients.
fn lowpass4_q14() -> Result<Coefs<CQ15>, Failure> {
    let mut sos = [[0.0f64; 5]; SECTIONS];
    butterworth_sos(BiquadResponse::Lowpass, 4, 4_800.0, FS, &mut sos).map_err(design_error)?;
    let mut out = [[0i16; 5]; SECTIONS];
    to_q14(&sos, &mut out).map_err(design_error)?;
    Ok(out)
}

#[allow(clippy::needless_pass_by_value)]
fn design_error(e: sdrk_blocks::Error) -> Failure {
    Failure::new("biquad design", "the filter design failed").detail(format_args!("{e}"))
}

/// Largest power of the last `n` samples relative to the first `n` (in dB), for the notch.
fn power_db(x: &[f32]) -> f64 {
    let sum: f64 = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    10.0 * libm::log10(sum / x.len().max(1) as f64)
}

// ---------------------------------------------------------------------------------------
// complex
// ---------------------------------------------------------------------------------------

/// Fallback against Reference, the facade against the fallback and 20 chunk plans of the
/// facade for the complex kernel on sample type `T`.
fn complex_case<T>(
    arena: &mut Arena<'_>,
    case: &'static str,
    coefs: Coefs<T>,
    fill: fn(&mut [T], &mut Pcg32),
    compare: fn(&'static str, &[T], &[T]) -> Result<(), Failure>,
    seed: u64,
) -> Result<(), Failure>
where
    T: ComplexSample + BiquadSample,
    Sdrk: Biquad<T>,
    Fallback: Biquad<T>,
    Reference: Biquad<T>,
{
    let mut c = arena.carver();
    let x = c.take::<T>(N)?;
    let (fb, rf, fac) = (c.take::<T>(N)?, c.take::<T>(N)?, c.take::<T>(N)?);
    let (h1, h2, h3) = (
        c.take::<T>(4 * SECTIONS)?,
        c.take::<T>(4 * SECTIONS)?,
        c.take::<T>(4 * SECTIONS)?,
    );
    fill(x, &mut Pcg32::new(seed, 1));
    let mut a = BiquadState::<T, _, _>::new(coefs, h1)?;
    let mut b = BiquadState::<T, _, _>::new(coefs, h2)?;
    let mut d = BiquadState::<T, _, _>::new(coefs, h3)?;
    Biquad::<T>::biquad(&Fallback, &mut a, x, fb)?;
    Biquad::<T>::biquad(&Reference, &mut b, x, rf)?;
    Biquad::<T>::biquad(&Sdrk, &mut d, x, fac)?;
    compare(case, rf, fb)?;
    ensure(
        bit_exact(fb, fac).is_none() && a.history() == d.history(),
        || Failure::new(case, "facade differs"),
    )?;

    // Chunk plans: the output does not depend on how the input is cut.
    let (input, whole, chunked) = (&*x, c.take::<T>(N)?, c.take::<T>(N)?);
    let (g1, g2) = (c.take::<T>(4 * SECTIONS)?, c.take::<T>(4 * SECTIONS)?);
    for plan in 0..PLANS {
        let mut slots = hand_out(&mut *g1, &mut *g2);
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(seed + 100 + plan, 1 + (plan as usize * 37) % 400),
            || BiquadState::<T, _, _>::new(coefs, slots()),
            |st, i, o| match st {
                Ok(st) => same_rate(i, o, |i, o| Biquad::<T>::biquad(&Sdrk, st, i, o)),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    Ok(())
}

fn compare_c32(case: &'static str, want: &[C32], got: &[C32]) -> Result<(), Failure> {
    min_snr(case, "f32 below 100 dB", flat(want), flat(got), 100.0)
}

fn compare_cq15(case: &'static str, want: &[CQ15], got: &[CQ15]) -> Result<(), Failure> {
    ensure(bit_exact(want, got).is_none(), || {
        Failure::new(case, "not bit-exact with the reference")
    })
}

pub(super) fn biquad_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "biquad_c32";
    complex_case::<C32>(
        arena,
        CASE,
        lowpass4_f32()?,
        |x, rng| {
            for s in x.iter_mut() {
                *s = random_c32(rng, 0.3);
            }
        },
        compare_c32,
        2400,
    )?;
    // A constant input settles at the DC gain of the low-pass, which is 1.
    let coefs = lowpass4_f32()?;
    let mut c = arena.carver();
    let (x, y) = (c.take::<C32>(600)?, c.take::<C32>(600)?);
    x.fill(C32::new(0.5, -0.25));
    let mut st = BiquadState::<C32, _, _>::new(coefs, c.take::<C32>(4 * SECTIONS)?)?;
    Biquad::<C32>::biquad(&Sdrk, &mut st, x, y)?;
    let last = y.last().copied().unwrap_or_default();
    ensure(
        (last.re - 0.5).abs() < 1e-4 && (last.im + 0.25).abs() < 1e-4,
        || Failure::new(CASE, "the DC gain of the low-pass is not 1"),
    )
}

pub(super) fn biquad_cq15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "biquad_cq15";
    complex_case::<CQ15>(
        arena,
        CASE,
        lowpass4_q14()?,
        |x, rng| {
            // Near full scale, so that the sections saturate now and then.
            for s in x.iter_mut() {
                *s = random_cq15(rng, 1.0);
            }
        },
        compare_cq15,
        2500,
    )?;
    let coefs = lowpass4_q14()?;
    let mut c = arena.carver();
    let (x, y) = (c.take::<CQ15>(600)?, c.take::<CQ15>(600)?);
    x.fill(CQ15 {
        re: Q15(8192),
        im: Q15(-4096),
    });
    let mut st = BiquadState::<CQ15, _, _>::new(coefs, c.take::<CQ15>(4 * SECTIONS)?)?;
    Biquad::<CQ15>::biquad(&Sdrk, &mut st, x, y)?;
    let last = y.last().copied().unwrap_or_default();
    ensure(
        (i32::from(last.re.0) - 8192).abs() <= 8 && (i32::from(last.im.0) + 4096).abs() <= 8,
        || Failure::new(CASE, "the DC gain of the low-pass is not 1"),
    )
}

// ---------------------------------------------------------------------------------------
// real
// ---------------------------------------------------------------------------------------

/// The real counterpart of [`complex_case`].
fn real_case<T>(
    arena: &mut Arena<'_>,
    case: &'static str,
    coefs: [[T::Coef; 5]; 1],
    fill: fn(&mut [T], &mut Pcg32),
    compare: fn(&'static str, &[T], &[T]) -> Result<(), Failure>,
    seed: u64,
) -> Result<(), Failure>
where
    T: RealSample + BiquadSample,
    Sdrk: BiquadReal<T>,
    Fallback: BiquadReal<T>,
    Reference: BiquadReal<T>,
{
    let mut c = arena.carver();
    let x = c.take::<T>(N)?;
    let (fb, rf, fac) = (c.take::<T>(N)?, c.take::<T>(N)?, c.take::<T>(N)?);
    let (h1, h2, h3) = (c.take::<T>(4)?, c.take::<T>(4)?, c.take::<T>(4)?);
    fill(x, &mut Pcg32::new(seed, 1));
    let mut a = BiquadRealState::<T, _, _>::new(coefs, h1)?;
    let mut b = BiquadRealState::<T, _, _>::new(coefs, h2)?;
    let mut d = BiquadRealState::<T, _, _>::new(coefs, h3)?;
    BiquadReal::<T>::biquad_real(&Fallback, &mut a, x, fb)?;
    BiquadReal::<T>::biquad_real(&Reference, &mut b, x, rf)?;
    BiquadReal::<T>::biquad_real(&Sdrk, &mut d, x, fac)?;
    compare(case, rf, fb)?;
    ensure(
        bit_exact(fb, fac).is_none() && a.history() == d.history(),
        || Failure::new(case, "facade differs"),
    )?;

    let (input, whole, chunked) = (&*x, c.take::<T>(N)?, c.take::<T>(N)?);
    let (g1, g2) = (c.take::<T>(4)?, c.take::<T>(4)?);
    for plan in 0..PLANS {
        let mut slots = hand_out(&mut *g1, &mut *g2);
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(seed + 100 + plan, 1 + (plan as usize * 41) % 400),
            || BiquadRealState::<T, _, _>::new(coefs, slots()),
            |st, i, o| match st {
                Ok(st) => same_rate(i, o, |i, o| BiquadReal::<T>::biquad_real(&Sdrk, st, i, o)),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    Ok(())
}

pub(super) fn biquad_real_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "biquad_real_f32";
    // A notch at 2.4 kHz with q 10.
    let notch = biquad_coefs(BiquadResponse::Notch, 2_400.0, 10.0, FS).map_err(design_error)?;
    let mut coefs = [[0.0f32; 5]; 1];
    to_f32_sos(&[notch], &mut coefs).map_err(design_error)?;
    real_case::<f32>(
        arena,
        CASE,
        coefs,
        |x, rng| {
            for s in x.iter_mut() {
                *s = random_c32(rng, 0.3).re;
            }
        },
        |case, want, got| min_snr(case, "f32 below 90 dB", want, got, 90.0),
        2600,
    )?;
    // The notch removes its centre frequency by at least 40 dB.
    let mut c = arena.carver();
    let (x, y) = (c.take::<f32>(1024)?, c.take::<f32>(1024)?);
    let mut st = BiquadRealState::<f32, _, _>::new(coefs, c.take::<f32>(4)?)?;
    for (n, v) in x.iter_mut().enumerate() {
        *v = 0.5 * libm::sinf(core::f32::consts::TAU * 0.05 * n as f32);
    }
    BiquadReal::<f32>::biquad_real(&Sdrk, &mut st, x, y)?;
    let drop =
        power_db(x.get(768..).unwrap_or_default()) - power_db(y.get(768..).unwrap_or_default());
    ensure(drop >= 40.0, || {
        Failure::new(CASE, "the notch removes less than 40 dB").detail(format_args!("{drop} dB"))
    })
}

pub(super) fn biquad_real_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "biquad_real_q15";
    // A second-order Butterworth high-pass at 480 Hz (fs / 100), inside the Q15 limit.
    let mut sos = [[0.0f64; 5]; 1];
    butterworth_sos(BiquadResponse::Highpass, 2, 480.0, FS, &mut sos).map_err(design_error)?;
    let mut coefs = [[0i16; 5]; 1];
    to_q14(&sos, &mut coefs).map_err(design_error)?;
    real_case::<Q15>(
        arena,
        CASE,
        coefs,
        |x, rng| {
            for s in x.iter_mut() {
                *s = random_cq15(rng, 1.0).re;
            }
        },
        |case, want, got| {
            ensure(bit_exact(want, got).is_none(), || {
                Failure::new(case, "not bit-exact with the reference")
            })
        },
        2700,
    )?;
    // A constant is blocked down to the Q15 rounding floor: at fs / 100 the feedback error is
    // amplified by 1 / |1 + a1 + a2| (about 260), which leaves a constant offset of 123 LSB
    // (1.5 % of the input of 8192); the limit of the kernel is documented in docs/kernels.md.
    let mut c = arena.carver();
    let (x, y) = (c.take::<Q15>(4000)?, c.take::<Q15>(4000)?);
    x.fill(Q15(8192));
    let mut st = BiquadRealState::<Q15, _, _>::new(coefs, c.take::<Q15>(4)?)?;
    BiquadReal::<Q15>::biquad_real(&Sdrk, &mut st, x, y)?;
    ensure(
        y.iter().skip(3000).all(|v| i32::from(v.0).abs() <= 164),
        || Failure::new(CASE, "the high-pass does not block DC"),
    )
}

// ---------------------------------------------------------------------------------------
// extremes
// ---------------------------------------------------------------------------------------

pub(super) fn extremes_m22(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_m22";
    const M: usize = 70;
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let mut c = arena.carver();
    let x = c.take::<C32>(M)?;
    let (fb, rf, fac) = (c.take::<C32>(M)?, c.take::<C32>(M)?, c.take::<C32>(M)?);
    let (r_in, r_fb, r_fac) = (c.take::<f32>(M)?, c.take::<f32>(M)?, c.take::<f32>(M)?);
    let q = c.take::<CQ15>(M)?;
    let (q_fb, q_rf, q_fac) = (c.take::<CQ15>(M)?, c.take::<CQ15>(M)?, c.take::<CQ15>(M)?);
    let (rq_in, rq_fb, rq_rf, rq_fac) = (
        c.take::<Q15>(M)?,
        c.take::<Q15>(M)?,
        c.take::<Q15>(M)?,
        c.take::<Q15>(M)?,
    );
    let f32_coefs = lowpass4_f32()?;
    let q_coefs = lowpass4_q14()?;
    let (h1, h2, h3) = (
        c.take::<C32>(4 * SECTIONS)?,
        c.take::<C32>(4 * SECTIONS)?,
        c.take::<C32>(4 * SECTIONS)?,
    );
    let first = f32_coefs.first().copied().unwrap_or_default();

    // f32: silence, full scale and the special values (NaN, infinities, denormal, MAX).
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
        let mut a = BiquadState::<C32, _, _>::new(f32_coefs, &mut *h1)?;
        let mut b = BiquadState::<C32, _, _>::new(f32_coefs, &mut *h2)?;
        let mut d = BiquadState::<C32, _, _>::new(f32_coefs, &mut *h3)?;
        Biquad::<C32>::biquad(&Fallback, &mut a, x, fb)?;
        Biquad::<C32>::biquad(&Reference, &mut b, x, rf)?;
        Biquad::<C32>::biquad(&Sdrk, &mut d, x, fac)?;
        ensure(
            fb.iter()
                .zip(fac.iter())
                .all(|(p, r)| same_f32(p.re, r.re) && same_f32(p.im, r.im)),
            || fail("biquad: facade differs"),
        )?;
        match kind {
            0 => {
                ensure(fb.iter().all(|s| *s == C32::default()), || {
                    fail("biquad: silence did not give silence")
                })?;
                ensure(rf.iter().all(|s| *s == C32::default()), || {
                    fail("biquad: reference silence did not give silence")
                })?;
            }
            1 => {
                // A bounded input gives a bounded output: no overflow to infinity.
                ensure(
                    fb.iter()
                        .chain(rf.iter())
                        .all(|s| s.re.is_finite() && s.im.is_finite()),
                    || fail("biquad: full scale did not stay finite"),
                )?;
                min_snr(
                    CASE,
                    "full scale: reference differs",
                    flat(rf),
                    flat(fb),
                    100.0,
                )?;
            }
            _ => {}
        }
        // The real kernel on the real parts, same properties.
        for (r, s) in r_in.iter_mut().zip(x.iter()) {
            *r = s.re;
        }
        let coefs1 = [first];
        let mut s1 = BiquadRealState::<f32, _, _>::new(coefs1, [0.0f32; 4])?;
        let mut s2 = BiquadRealState::<f32, _, _>::new(coefs1, [0.0f32; 4])?;
        BiquadReal::<f32>::biquad_real(&Fallback, &mut s1, r_in, r_fb)?;
        BiquadReal::<f32>::biquad_real(&Sdrk, &mut s2, r_in, r_fac)?;
        ensure(
            r_fb.iter().zip(r_fac.iter()).all(|(p, r)| same_f32(*p, *r)),
            || fail("biquad_real: facade differs"),
        )?;
    }

    // A NaN poisons the recursion, `reset` clears it, and the next silence is silence again.
    let mut st = BiquadRealState::<f32, _, _>::new([first], [0.0f32; 4])?;
    BiquadReal::<f32>::biquad_real(&Sdrk, &mut st, &[f32::NAN; 4], &mut [0.0f32; 4])?;
    st.reset();
    let mut out = [1.0f32; 4];
    BiquadReal::<f32>::biquad_real(&Sdrk, &mut st, &[0.0f32; 4], &mut out)?;
    ensure(out.iter().all(|v| v.to_bits() == 0), || {
        fail("biquad_real: reset did not clear the state")
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
        let mut a = BiquadState::<CQ15, _, _>::new(q_coefs, [CQ15::default(); 4 * SECTIONS])?;
        let mut b = BiquadState::<CQ15, _, _>::new(q_coefs, [CQ15::default(); 4 * SECTIONS])?;
        let mut d = BiquadState::<CQ15, _, _>::new(q_coefs, [CQ15::default(); 4 * SECTIONS])?;
        Biquad::<CQ15>::biquad(&Fallback, &mut a, q, q_fb)?;
        Biquad::<CQ15>::biquad(&Reference, &mut b, q, q_rf)?;
        Biquad::<CQ15>::biquad(&Sdrk, &mut d, q, q_fac)?;
        ensure(
            bit_exact(q_fb, q_rf).is_none()
                && bit_exact(q_fb, q_fac).is_none()
                && a.history() == b.history()
                && a.history() == d.history(),
            || fail("biquad: Q15 differs between implementations"),
        )?;
        if kind == 0 {
            ensure(q_fb.iter().all(|s| *s == CQ15::default()), || {
                fail("biquad: Q15 silence did not give silence")
            })?;
        }
        // The real kernel with the corners of the raw range as coefficients as well: the
        // extreme Q2.14 values must neither panic nor differ from the reference.
        for (r, s) in rq_in.iter_mut().zip(q.iter()) {
            *r = s.re;
        }
        for coef in [
            [-32768i16; 5],
            [32767; 5],
            [-32768, 32767, -32768, 32767, -32768],
        ] {
            let mut s1 = BiquadRealState::<Q15, _, _>::new([coef], [Q15(0); 4])?;
            let mut s2 = BiquadRealState::<Q15, _, _>::new([coef], [Q15(0); 4])?;
            let mut s3 = BiquadRealState::<Q15, _, _>::new([coef], [Q15(0); 4])?;
            BiquadReal::<Q15>::biquad_real(&Fallback, &mut s1, rq_in, rq_fb)?;
            BiquadReal::<Q15>::biquad_real(&Reference, &mut s2, rq_in, rq_rf)?;
            BiquadReal::<Q15>::biquad_real(&Sdrk, &mut s3, rq_in, rq_fac)?;
            ensure(
                bit_exact(rq_fb, rq_rf).is_none()
                    && bit_exact(rq_fb, rq_fac).is_none()
                    && s1.history() == s2.history(),
                || fail("biquad_real: Q15 corners differ"),
            )?;
        }
    }

    // Invalid construction is an error, not a panic.
    ensure(
        BiquadRealState::<f32, _, _>::new([[0.0f32; 5]; 0], [0.0f32; 0]).is_err()
            && BiquadRealState::<f32, _, _>::new(
                [[0.0f32; 5]; MAX_SECTIONS + 1],
                [0.0f32; 4 * (MAX_SECTIONS + 1)],
            )
            .is_err()
            && BiquadRealState::<f32, _, _>::new([[0.0f32; 5]; 2], [0.0f32; 7]).is_err()
            && BiquadState::<CQ15, _, _>::new([[0i16; 5]; 1], [CQ15::default(); 5]).is_err(),
        || fail("invalid construction accepted"),
    )?;
    ensure(
        BiquadRealState::<f32, _, _>::new([[0.0f32; 5]; MAX_SECTIONS], [0.0f32; 4 * MAX_SECTIONS])
            .is_ok(),
        || fail("32 sections rejected"),
    )?;
    ensure(
        sdrk_core::num::q14_from_f64(-2.0) == Some(-32768)
            && sdrk_core::num::q14_from_f64(2.0).is_none()
            && sdrk_core::num::q14_from_f64(f64::NAN).is_none(),
        || fail("q14_from_f64 corners"),
    )
}

// ---------------------------------------------------------------------------------------
// block
// ---------------------------------------------------------------------------------------

/// Samples of the block case (50 ms at 48 kHz).
const BLOCK_N: usize = 2400;
/// The last samples over which the level of a tone is fitted (25 ms).
const LEVEL_WINDOW: usize = 1200;
const BLOCK_CASE: &str = "block_biquad_response";

/// Writes a sine of `hz` with `amp` (relative to full scale) into `out`.
fn sine<T: RealSample>(out: &mut [T], hz: f64, amp: f64) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = T::from_f64_round(amp * libm::sin(core::f64::consts::TAU * hz * i as f64 / FS));
    }
}

/// Level of the tone `hz` in the last [`LEVEL_WINDOW`] samples of `x`, as an amplitude.
fn tone_level<T: RealSample>(x: &[T], hz: f64) -> f64 {
    let start = x.len().saturating_sub(LEVEL_WINDOW);
    let mut acc = ToneAccumulator::new(FS, hz);
    // The accumulator indexes from 0; the fit takes the phase from the data.
    for v in x.iter().skip(start) {
        acc.push(v.to_f64());
    }
    acc.amplitude()
}

/// Gain of `out` against an input tone of amplitude `amp`, in dB.
fn gain_db<T: RealSample>(out: &[T], hz: f64, amp: f64) -> f64 {
    20.0 * libm::log10(tone_level(out, hz).max(1e-12) / amp)
}

fn block_fail(msg: &'static str) -> Failure {
    Failure::new(BLOCK_CASE, msg)
}

fn block_rate() -> Result<Rate, Failure> {
    Ok(Rate::hz(FS as u64)?)
}

/// The tone and notch properties of the real block on sample type `T`.
fn real_block_for<T>(arena: &mut Arena<'_>, notch_min_db: f64) -> Result<(), Failure>
where
    T: RealSample + BiquadElem + BlockReal,
    Sdrk: BiquadReal<T>,
{
    let rate = block_rate()?;
    let mut c = arena.carver();
    let (x, y) = (c.take::<T>(BLOCK_N)?, c.take::<T>(BLOCK_N)?);
    let lowpass = BiquadConfig {
        response: BiquadResponse::Lowpass,
        cutoff_hz: 2_000.0,
        order: 4,
        ..BiquadConfig::default()
    };
    let bytes =
        BiquadRealBlock::<T, ArenaStorage<'_>>::storage_bytes(&lowpass, rate)? + ARENA_ALIGN;
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let mut block = BiquadRealBlock::<T, _>::new(&lowpass, rate, &mut store)?;
    // 1 kHz passes (less than 0.1 dB), 8 kHz is attenuated by at least 48 dB.
    sine(x, 1_000.0, 0.5);
    block.process(x, y)?;
    let pass = gain_db(y, 1_000.0, 0.5);
    ensure(pass.abs() < 0.1, || {
        block_fail("the 1 kHz tone is attenuated").detail(format_args!("{pass} dB"))
    })?;
    block.reset();
    sine(x, 8_000.0, 0.5);
    block.process(x, y)?;
    let stop = gain_db(y, 8_000.0, 0.5);
    ensure(stop <= -48.0, || {
        block_fail("the 8 kHz tone is attenuated by less than 48 dB")
            .detail(format_args!("{stop} dB"))
    })?;
    // Runtime: a lower cutoff takes effect from the next call and the output stays finite.
    block.reset();
    block.set_param("cutoff_hz", ParamValue::F64(500.0))?;
    sine(x, 1_000.0, 0.5);
    block.process(x, y)?;
    let lowered = gain_db(y, 1_000.0, 0.5);
    ensure(
        lowered <= -20.0 && y.iter().all(|v| v.to_f64().is_finite()),
        || {
            block_fail("the lowered cutoff did not take effect")
                .detail(format_args!("{lowered} dB"))
        },
    )?;
    ensure(block.param("cutoff_hz")? == ParamValue::F64(500.0), || {
        block_fail("the cutoff is not reported")
    })?;

    // The notch at 1 kHz with q 10 removes the tone by at least 40 dB; q is a runtime parameter.
    let notch = BiquadConfig {
        response: BiquadResponse::Notch,
        cutoff_hz: 1_000.0,
        q: 10.0,
        ..BiquadConfig::default()
    };
    let bytes = BiquadRealBlock::<T, ArenaStorage<'_>>::storage_bytes(&notch, rate)? + ARENA_ALIGN;
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let mut block = BiquadRealBlock::<T, _>::new(&notch, rate, &mut store)?;
    sine(x, 1_000.0, 0.5);
    block.process(x, y)?;
    let notched = gain_db(y, 1_000.0, 0.5);
    ensure(notched <= -notch_min_db, || {
        block_fail("the notch is not deep enough").detail(format_args!("{notched} dB"))
    })?;
    // A tone 100 Hz away is hurt less by the narrow notch than by a wide one (q 1).
    block.reset();
    sine(x, 1_100.0, 0.5);
    block.process(x, y)?;
    let narrow = gain_db(y, 1_100.0, 0.5);
    block.reset();
    block.set_param("q", ParamValue::F64(1.0))?;
    block.process(x, y)?;
    let wide = gain_db(y, 1_100.0, 0.5);
    ensure(wide < narrow - 3.0, || {
        block_fail("a lower q did not widen the notch")
            .detail(format_args!("{narrow} dB with q 10, {wide} dB with q 1"))
    })?;
    // A rejected change leaves the block as it was.
    ensure(
        block
            .set_param("cutoff_hz", ParamValue::F64(30_000.0))
            .is_err()
            && block.param("cutoff_hz")? == ParamValue::F64(1_000.0),
        || block_fail("an invalid cutoff was accepted"),
    )
}

/// Runs `block` over `input`: in one call, or in the chunks of `plan`.
fn drive<B: Block>(
    block: &mut B,
    input: &[B::In],
    output: &mut [B::Out],
    plan: Option<ChunkPlan>,
) -> Result<(), Failure> {
    let Some(plan) = plan else {
        block.process(input, output)?;
        return Ok(());
    };
    let mut pos = 0usize;
    for size in plan.sizes() {
        if pos >= input.len() {
            break;
        }
        let n = size.min(input.len() - pos);
        let range = pos..pos + n;
        block.process(
            input.get(range.clone()).unwrap_or_default(),
            output.get_mut(range).unwrap_or_default(),
        )?;
        pos += n;
    }
    Ok(())
}

/// The output of a whole run equals the output of 20 chunk plans, bit for bit. `run` builds a
/// fresh block over the memory and drives it.
fn invariance_for<T>(
    arena: &mut Arena<'_>,
    bytes: usize,
    mut sample: impl FnMut(&mut Pcg32) -> T,
    mut run: impl FnMut(&mut [u8], &[T], &mut [T], Option<ChunkPlan>) -> Result<(), Failure>,
) -> Result<(), Failure>
where
    T: Sample + Copy + PartialEq,
{
    let mut c = arena.carver();
    let (x, whole, chunked) = (
        c.take::<T>(BLOCK_N)?,
        c.take::<T>(BLOCK_N)?,
        c.take::<T>(BLOCK_N)?,
    );
    let mem = c.take::<u8>(bytes)?;
    let mut rng = Pcg32::new(4300, 1);
    for s in x.iter_mut() {
        *s = sample(&mut rng);
    }
    run(mem, x, whole, None)?;
    for seed in 0..PLANS {
        let plan = ChunkPlan::new(4300 + seed, 1 + (seed as usize * 37) % 500);
        run(mem, x, chunked, Some(plan))?;
        ensure(bit_exact(whole, chunked).is_none(), || {
            block_fail("the output depends on the chunk sizes").detail(format_args!("plan {seed}"))
        })?;
    }
    Ok(())
}

fn real_invariance<T>(arena: &mut Arena<'_>) -> Result<(), Failure>
where
    T: RealSample + BiquadElem + BlockReal,
    Sdrk: BiquadReal<T>,
{
    let rate = block_rate()?;
    let cfg = BiquadConfig {
        response: BiquadResponse::Highpass,
        cutoff_hz: 300.0,
        order: 4,
        ..BiquadConfig::default()
    };
    let bytes = BiquadRealBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + ARENA_ALIGN;
    invariance_for::<T>(
        arena,
        bytes,
        |rng| T::from_f64_round(0.3 * rng.normal()),
        |mem, x, y, plan| {
            let mut store = ArenaStorage::new(mem);
            let mut block = BiquadRealBlock::<T, _>::new(&cfg, rate, &mut store)?;
            drive(&mut block, x, y, plan)
        },
    )
}

fn complex_invariance<T>(arena: &mut Arena<'_>) -> Result<(), Failure>
where
    T: ComplexSample + BiquadElem + BlockSample,
    Sdrk: Biquad<T>,
{
    let rate = block_rate()?;
    let cfg = BiquadConfig {
        response: BiquadResponse::Bandpass,
        cutoff_hz: 3_000.0,
        q: 2.0,
        ..BiquadConfig::default()
    };
    let bytes = BiquadBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + ARENA_ALIGN;
    invariance_for::<T>(
        arena,
        bytes,
        |rng| T::from_c64_round(Complex64::new(0.2 * rng.normal(), 0.2 * rng.normal())),
        |mem, x, y, plan| {
            let mut store = ArenaStorage::new(mem);
            let mut block = BiquadBlock::<T, _>::new(&cfg, rate, &mut store)?;
            drive(&mut block, x, y, plan)
        },
    )
}

pub(super) fn block_biquad_response(arena: &mut Arena<'_>) -> Result<(), Failure> {
    // Notch depth: 40 dB for f32, 34 dB for Q15 (the separate Q15 threshold of decision D23;
    // the Q2.14 coefficients limit the Q15 notch to 34.9 dB).
    real_block_for::<f32>(arena, 40.0)?;
    real_block_for::<Q15>(arena, 34.0)?;
    real_invariance::<f32>(arena)?;
    real_invariance::<Q15>(arena)?;
    complex_invariance::<C32>(arena)?;
    complex_invariance::<CQ15>(arena)
}

// ---------------------------------------------------------------------------------------
// end to end
// ---------------------------------------------------------------------------------------

fn ctcss_case<T>(
    case: &'static str,
    arena: &mut Arena<'_>,
    samples: usize,
    min_subtone_db: f64,
    max_sinad_loss_db: f64,
) -> Result<(), Failure>
where
    T: E2eSample + BlockSample,
    <T as ComplexSample>::Real: BiquadElem + BlockReal,
    Sdrk: Convert<T>
        + Ddc<T>
        + FmDemod<T>
        + QuadDemod<T>
        + Squelch<T>
        + ResampleReal<<T as ComplexSample>::Real>
        + BiquadReal<<T as ComplexSample>::Real>,
{
    let base = measure_nfm::<T>(arena, samples)?;
    let m = measure_nfm_ctcss::<T>(arena, samples)?;
    ensure(m.audio_samples == samples / 50, || {
        Failure::new(case, "wrong audio sample count").detail(format_args!("{}", m.audio_samples))
    })?;
    ensure(m.subtone_below_db >= min_subtone_db, || {
        Failure::new(case, "the 88.5 Hz sub-tone is not attenuated enough").detail(format_args!(
            "{} dB below the 1 kHz tone, need {min_subtone_db}",
            m.subtone_below_db
        ))
    })?;
    ensure(m.sinad_db >= base.sinad_db - max_sinad_loss_db, || {
        Failure::new(case, "SINAD too far below E2E-NFM").detail(format_args!(
            "{} dB, E2E-NFM {} dB, allowed loss {max_sinad_loss_db} dB",
            m.sinad_db, base.sinad_db
        ))
    })?;
    ensure(
        m.open_delay_s <= NFM_MAX_OPEN_DELAY_S && m.close_delay_s <= NFM_MAX_CLOSE_DELAY_S,
        || Failure::new(case, "the gate switched too late"),
    )?;
    ensure(m.zero_checked > 0 && m.zero_violations == 0, || {
        Failure::new(case, "audio is not exactly zero before the transmission")
    })
}

pub(super) fn e2e_nfm_ctcss_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    ctcss_case::<C32>(
        "e2e_nfm_ctcss_f32",
        arena,
        NFM_QUICK_SAMPLES,
        NFM_MIN_SUBTONE_DB_F32,
        NFM_MAX_SINAD_LOSS_DB_F32,
    )
}

/// Q15 has its own SINAD threshold of 1.1 dB (decision D23; measured 1.07 dB).
pub(super) fn e2e_nfm_ctcss_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    ctcss_case::<CQ15>(
        "e2e_nfm_ctcss_q15",
        arena,
        NFM_QUICK_SAMPLES,
        NFM_MIN_SUBTONE_DB_Q15,
        NFM_MAX_SINAD_LOSS_DB_Q15,
    )
}
