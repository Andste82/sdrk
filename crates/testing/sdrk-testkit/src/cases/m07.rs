//! Suite cases of the FFT kernels (`fft`, `rfft`, `irfft`): the fallback and the facade against
//! the naive f64 DFT of `Reference`, round trips, Parseval, the weak-tone property of block
//! floating point and extreme inputs. No files are read, so the cases run on microcontrollers.

#![allow(clippy::many_single_char_names, clippy::too_many_lines)]

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, FftScale, RealFft, RealFftPlan, Scaling};
use sdrk_core::{C32, CQ15, ComplexSample, Error, Fallback, Q15, Reference};

use super::ensure;
use crate::arena::Arena;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

const FWD: Direction = Direction::Forward;
const INV: Direction = Direction::Inverse;

/// `10 log10(sum |want|^2 / sum |want - got|^2)`; `+inf` for identical signals.
fn snr_db(want: impl Iterator<Item = Complex64>, got: impl Iterator<Item = Complex64>) -> f64 {
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (w, g) in want.zip(got) {
        sig += w.norm_sqr();
        err += (w - g).norm_sqr();
    }
    if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(sig / err)
    }
}

fn min_snr(case: &'static str, msg: &'static str, db: f64, min: f64) -> Result<(), Failure> {
    ensure(db >= min, || {
        Failure::new(case, msg).detail(format_args!("{db} dB, need {min}"))
    })
}

/// The samples as complex f64 multiplied by `2^exponent` (the unnormalised transform).
fn scaled<T: ComplexSample>(x: &[T], exponent: i32) -> impl Iterator<Item = Complex64> + '_ {
    let g = libm::ldexp(1.0, exponent);
    x.iter().map(move |v| v.to_c64() * g)
}

fn to_c64<T: ComplexSample>(x: &[T]) -> impl Iterator<Item = Complex64> + '_ {
    x.iter().map(|v| v.to_c64())
}

fn real_c64(x: &[f32]) -> impl Iterator<Item = Complex64> + '_ {
    x.iter().map(|&v| Complex64::new(f64::from(v), 0.0))
}

fn noise_c32(out: &mut [C32], rng: &mut Pcg32, sigma: f64) {
    for s in out {
        *s = C32::new((rng.normal() * sigma) as f32, (rng.normal() * sigma) as f32);
    }
}

fn noise_cq15(out: &mut [CQ15], rng: &mut Pcg32, sigma: f64) {
    for s in out {
        *s = CQ15::from_c64_round(Complex64::new(rng.normal() * sigma, rng.normal() * sigma));
    }
}

/// Full-scale (or weaker) tone at `bin` with amplitude `amp`.
fn tone_cq15(out: &mut [CQ15], bin: usize, amp: f64) {
    let n = out.len();
    for (i, s) in out.iter_mut().enumerate() {
        let th = 2.0 * PI * ((bin * i) % n) as f64 / n as f64;
        *s = CQ15::from_c64_round(Complex64::new(amp * libm::cos(th), amp * libm::sin(th)));
    }
}

fn log2(n: usize) -> i32 {
    n.trailing_zeros() as i32
}

// ----------------------------------------------------------------- C32 vs reference

fn c32_vs_reference(
    arena: &mut Arena<'_>,
    case: &'static str,
    n: usize,
    seed: u64,
) -> Result<(), Failure> {
    let mut c = arena.carver();
    let (x, rf, fb, fac) = (
        c.take::<C32>(n)?,
        c.take::<C32>(n)?,
        c.take::<C32>(n)?,
        c.take::<C32>(n)?,
    );
    let tw = c.take::<C32>(n / 2)?;
    let plan = FftPlan::<C32, _>::new(n, tw)?;
    let mut rng = Pcg32::new(seed, 7);
    noise_c32(x, &mut rng, 0.3);
    for dir in [FWD, INV] {
        let want = Reference.dft(x, rf, dir, Scaling::Default)?;
        fb.copy_from_slice(x);
        let got = Fft::<C32>::fft(&Fallback, &plan, fb, dir, Scaling::Default)?;
        fac.copy_from_slice(x);
        let via = Fft::<C32>::fft(&Sdrk, &plan, fac, dir, Scaling::Default)?;
        ensure(want == got && got == via, || {
            Failure::new(case, "exponent differs").detail(format_args!("{want:?} {got:?} {via:?}"))
        })?;
        ensure(bit_exact(fb, fac).is_none(), || {
            Failure::new(case, "facade differs from the fallback")
        })?;
        min_snr(
            case,
            "below 110 dB against the reference",
            snr_db(to_c64(rf), to_c64(fb)),
            110.0,
        )?;
    }
    Ok(())
}

pub(super) fn fft_c32_vs_reference_64(arena: &mut Arena<'_>) -> Result<(), Failure> {
    c32_vs_reference(arena, "fft_c32_vs_reference_64", 64, 701)
}

pub(super) fn fft_c32_vs_reference_1024(arena: &mut Arena<'_>) -> Result<(), Failure> {
    c32_vs_reference(arena, "fft_c32_vs_reference_1024", 1024, 702)
}

// -------------------------------------------------------------- CQ15 vs reference

pub(super) fn fft_cq15_fixed_vs_reference_1024(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fft_cq15_fixed_vs_reference_1024";
    const N: usize = 1024;
    let mut c = arena.carver();
    let (x, rf, fb, fac) = (
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
    );
    let plan = FftPlan::<CQ15, _>::new(N, c.take::<CQ15>(N / 2)?)?;
    let mut rng = Pcg32::new(703, 7);
    for (min, tone) in [(40.0, false), (55.0, true)] {
        if tone {
            tone_cq15(x, 37, 1.0 - libm::pow(2.0, -15.0));
        } else {
            noise_cq15(x, &mut rng, 0.3);
        }
        let want = Reference.dft(x, rf, FWD, Scaling::Default)?;
        fb.copy_from_slice(x);
        let got = Fft::<CQ15>::fft(&Fallback, &plan, fb, FWD, Scaling::Default)?;
        fac.copy_from_slice(x);
        let via = Fft::<CQ15>::fft(&Sdrk, &plan, fac, FWD, Scaling::Default)?;
        ensure(want == got && got == via && got.exponent == log2(N), || {
            Failure::new(CASE, "exponent differs").detail(format_args!("{want:?} {got:?}"))
        })?;
        ensure(bit_exact(fb, fac).is_none(), || {
            Failure::new(CASE, "facade differs from the fallback")
        })?;
        min_snr(
            CASE,
            "fixed scaling below the contract",
            snr_db(to_c64(rf), to_c64(fb)),
            min,
        )?;
    }
    Ok(())
}

pub(super) fn fft_cq15_bfp_vs_reference_1024(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fft_cq15_bfp_vs_reference_1024";
    const N: usize = 1024;
    let mut c = arena.carver();
    let (x, rf, fb, fac) = (
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
    );
    let plan = FftPlan::<CQ15, _>::new(N, c.take::<CQ15>(N / 2)?)?;
    let mut rng = Pcg32::new(704, 7);
    // The tone threshold is 58 dB: the exact algorithm reaches 59.05 dB (decisions log, M07).
    for (min, tone) in [(55.0, false), (58.0, true)] {
        if tone {
            tone_cq15(x, 37, 1.0 - libm::pow(2.0, -15.0));
        } else {
            noise_cq15(x, &mut rng, 0.3);
        }
        let want = Reference.dft(x, rf, FWD, Scaling::BlockFloat)?;
        fb.copy_from_slice(x);
        let got = Fft::<CQ15>::fft(&Fallback, &plan, fb, FWD, Scaling::BlockFloat)?;
        fac.copy_from_slice(x);
        let via = Fft::<CQ15>::fft(&Sdrk, &plan, fac, FWD, Scaling::BlockFloat)?;
        ensure(got == via && bit_exact(fb, fac).is_none(), || {
            Failure::new(CASE, "facade differs from the fallback")
        })?;
        ensure((0..=2 * log2(N)).contains(&got.exponent), || {
            Failure::new(CASE, "exponent out of range").detail(format_args!("{got:?}"))
        })?;
        // Both sides are compared after multiplying by their own `2^exponent`.
        min_snr(
            CASE,
            "block floating point below the contract",
            snr_db(scaled(rf, want.exponent), scaled(fb, got.exponent)),
            min,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------- round trips

pub(super) fn fft_roundtrip_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fft_roundtrip_c32";
    const N: usize = 1024;
    let mut c = arena.carver();
    let (x, y) = (c.take::<C32>(N)?, c.take::<C32>(N)?);
    let plan = FftPlan::<C32, _>::new(N, c.take::<C32>(N / 2)?)?;
    let mut rng = Pcg32::new(705, 7);
    noise_c32(x, &mut rng, 0.3);
    y.copy_from_slice(x);
    Fft::<C32>::fft(&Sdrk, &plan, y, FWD, Scaling::Default)?;
    let s = Fft::<C32>::fft(&Sdrk, &plan, y, INV, Scaling::Default)?;
    ensure(s.exponent == log2(N), || {
        Failure::new(CASE, "wrong inverse exponent")
    })?;
    min_snr(
        CASE,
        "inverse(forward(x)) differs from x",
        snr_db(to_c64(x), to_c64(y)),
        110.0,
    )
}

pub(super) fn fft_roundtrip_cq15_bfp(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fft_roundtrip_cq15_bfp";
    const N: usize = 1024;
    let mut c = arena.carver();
    let (x, y) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    let plan = FftPlan::<CQ15, _>::new(N, c.take::<CQ15>(N / 2)?)?;
    let mut rng = Pcg32::new(706, 7);
    noise_cq15(x, &mut rng, 0.3);
    y.copy_from_slice(x);
    let f = Fft::<CQ15>::fft(&Sdrk, &plan, y, FWD, Scaling::BlockFloat)?;
    let i = Fft::<CQ15>::fft(&Sdrk, &plan, y, INV, Scaling::BlockFloat)?;
    // out * 2^(e_fwd + e_inv) / N approximates x.
    min_snr(
        CASE,
        "round trip below 40 dB",
        snr_db(to_c64(x), scaled(y, f.exponent + i.exponent - log2(N))),
        40.0,
    )
}

pub(super) fn fft_parseval_c32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fft_parseval_c32";
    let mut c = arena.carver();
    for n in [64usize, 1024] {
        let x = c.take::<C32>(n)?;
        let y = c.take::<C32>(n)?;
        let plan = FftPlan::<C32, _>::new(n, c.take::<C32>(n / 2)?)?;
        let mut rng = Pcg32::new(707, n as u64);
        noise_c32(x, &mut rng, 0.3);
        y.copy_from_slice(x);
        Fft::<C32>::fft(&Sdrk, &plan, y, FWD, Scaling::Default)?;
        let ex: f64 = to_c64(x).map(|v| v.norm_sqr()).sum::<f64>() * n as f64;
        let ef: f64 = to_c64(y).map(|v| v.norm_sqr()).sum();
        let rel = libm::fabs(ex - ef) / ef;
        ensure(rel < 1e-5, || {
            Failure::new(CASE, "Parseval violated").detail(format_args!("n={n}: relative {rel}"))
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------- real FFT

pub(super) fn rfft_vs_reference_1024(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "rfft_vs_reference_1024";
    const N: usize = 1024;
    let mut c = arena.carver();
    let x = c.take::<f32>(N)?;
    let (rf, fb, fac) = (
        c.take::<C32>(N / 2 + 1)?,
        c.take::<C32>(N / 2 + 1)?,
        c.take::<C32>(N / 2 + 1)?,
    );
    let scratch = c.take::<C32>(N / 2)?;
    let plan = RealFftPlan::new(N, c.take::<C32>(N / 4)?, c.take::<C32>(N / 2)?)?;
    let mut rng = Pcg32::new(708, 7);
    for v in x.iter_mut() {
        *v = (rng.normal() * 0.3) as f32;
    }
    let want = Reference.rfft(&plan, x, rf, scratch)?;
    let got = Fallback.rfft(&plan, x, fb, scratch)?;
    let via = Sdrk.rfft(&plan, x, fac, scratch)?;
    ensure(want == got && got == via, || {
        Failure::new(CASE, "exponent differs")
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs from the fallback")
    })?;
    min_snr(
        CASE,
        "below 110 dB against the reference",
        snr_db(to_c64(rf), to_c64(fb)),
        110.0,
    )
}

pub(super) fn irfft_vs_reference_1024(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "irfft_vs_reference_1024";
    const N: usize = 1024;
    let mut c = arena.carver();
    let x = c.take::<f32>(N)?;
    let (rf, fb, fac) = (c.take::<f32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
    let bins = c.take::<C32>(N / 2 + 1)?;
    let scratch = c.take::<C32>(N / 2)?;
    let plan = RealFftPlan::new(N, c.take::<C32>(N / 4)?, c.take::<C32>(N / 2)?)?;
    let mut rng = Pcg32::new(709, 7);
    for v in x.iter_mut() {
        *v = (rng.normal() * 0.3) as f32;
    }
    Fallback.rfft(&plan, x, bins, scratch)?;
    // Bin 0 and bin N/2 carry an imaginary part that must be ignored.
    if let Some(b) = bins.first_mut() {
        b.im = 0.5;
    }
    if let Some(b) = bins.get_mut(N / 2) {
        b.im = 0.5;
    }
    let want = Reference.irfft(&plan, bins, rf, scratch)?;
    let got = Fallback.irfft(&plan, bins, fb, scratch)?;
    let via = Sdrk.irfft(&plan, bins, fac, scratch)?;
    ensure(want == got && got == via && got.exponent == log2(N), || {
        Failure::new(CASE, "exponent differs")
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs from the fallback")
    })?;
    min_snr(
        CASE,
        "below 110 dB against the reference",
        snr_db(real_c64(rf), real_c64(fb)),
        110.0,
    )?;
    min_snr(
        CASE,
        "does not return the original signal",
        snr_db(real_c64(x), real_c64(fb)),
        110.0,
    )
}

pub(super) fn rfft_roundtrip_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "rfft_roundtrip_f32";
    const N: usize = 1024;
    let mut c = arena.carver();
    let (x, y) = (c.take::<f32>(N)?, c.take::<f32>(N)?);
    let bins = c.take::<C32>(N / 2 + 1)?;
    let scratch = c.take::<C32>(N / 2)?;
    let plan = RealFftPlan::new(N, c.take::<C32>(N / 4)?, c.take::<C32>(N / 2)?)?;
    let mut rng = Pcg32::new(710, 7);
    for v in x.iter_mut() {
        *v = (rng.normal() * 0.3) as f32;
    }
    Sdrk.rfft(&plan, x, bins, scratch)?;
    Sdrk.irfft(&plan, bins, y, scratch)?;
    min_snr(
        CASE,
        "irfft(rfft(x)) differs from x",
        snr_db(real_c64(x), real_c64(y)),
        110.0,
    )
}

// ------------------------------------------------------------------------ weak tone

pub(super) fn fft_weak_tone_bfp(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fft_weak_tone_bfp";
    const N: usize = 1024;
    const BIN: usize = 100;
    let mut c = arena.carver();
    let (x, y) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    let mags = c.take::<f32>(N)?;
    let plan = FftPlan::<CQ15, _>::new(N, c.take::<CQ15>(N / 2)?)?;
    // A tone at -60 dBFS without noise.
    tone_cq15(x, BIN, libm::pow(10.0, -60.0 / 20.0));
    y.copy_from_slice(x);
    let s = Fft::<CQ15>::fft(&Sdrk, &plan, y, FWD, Scaling::BlockFloat)?;
    let scale = libm::ldexp(1.0, s.exponent);
    for (m, v) in mags.iter_mut().zip(y.iter()) {
        *m = (libm::sqrt(v.to_c64().norm_sqr()) * scale) as f32;
    }
    let tone = f64::from(mags.get(BIN).copied().unwrap_or(0.0));
    // Median of all other bins: move the tone out of the way and sort in place (insertion sort
    // would be quadratic, so use a selection by repeated partition of the sorted prefix).
    if let Some(m) = mags.get_mut(BIN) {
        *m = 0.0;
    }
    mags.sort_unstable_by(f32::total_cmp);
    // The zeroed tone bin sorts first; the median of the remaining N - 1 values is at N / 2.
    let median = f64::from(mags.get(N / 2).copied().unwrap_or(0.0));
    ensure(tone >= median * libm::pow(10.0, 30.0 / 20.0), || {
        Failure::new(CASE, "tone not 30 dB above the median").detail(format_args!(
            "tone {tone}, median {median}, exponent {}",
            s.exponent
        ))
    })?;
    // The exact tone is N * amplitude: block floating point must keep it within a few percent.
    let want = N as f64 * libm::pow(10.0, -60.0 / 20.0);
    ensure(libm::fabs(tone - want) < 0.05 * want, || {
        Failure::new(CASE, "tone level wrong").detail(format_args!("{tone} instead of {want}"))
    })
}

// ------------------------------------------------------------------------- extremes

pub(super) fn extremes_fft(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_fft";
    const N: usize = 256;
    let mut c = arena.carver();
    let (xf, xq) = (c.take::<C32>(N)?, c.take::<CQ15>(N)?);
    let plan_f = FftPlan::<C32, _>::new(N, c.take::<C32>(N / 2)?)?;
    let plan_q = FftPlan::<CQ15, _>::new(N, c.take::<CQ15>(N / 2)?)?;
    let has_nan = |s: &[C32]| s.iter().any(|v| v.re.is_nan() || v.im.is_nan());

    // f32 specials (NaN, infinities, denormals, f32::MAX): no panic, NaN reaches the output.
    crate::r#gen::extreme::f32_specials_complex(xf);
    Fft::<C32>::fft(&Sdrk, &plan_f, xf, FWD, Scaling::Default)?;
    ensure(has_nan(xf), || Failure::new(CASE, "fft swallowed NaN"))?;
    // Silence stays silent in every mode.
    xf.fill(C32::default());
    Fft::<C32>::fft(&Sdrk, &plan_f, xf, INV, Scaling::Default)?;
    ensure(xf.iter().all(|v| *v == C32::default()), || {
        Failure::new(CASE, "silence changed (c32)")
    })?;
    for scaling in [Scaling::Default, Scaling::BlockFloat] {
        for dir in [FWD, INV] {
            xq.fill(CQ15::default());
            let s = Fft::<CQ15>::fft(&Sdrk, &plan_q, xq, dir, scaling)?;
            ensure(xq.iter().all(|v| *v == CQ15::default()), || {
                Failure::new(CASE, "silence changed (cq15)")
            })?;
            let want = if scaling == Scaling::Default {
                log2(N)
            } else {
                0
            };
            ensure(s.exponent == want, || {
                Failure::new(CASE, "wrong exponent for silence").detail(format_args!("{s:?}"))
            })?;
            // Full-scale alternating values (complex magnitude sqrt 2) and the Q15 corners:
            // no panic, saturated results stay in range by construction.
            crate::r#gen::extreme::full_scale_alternating(xq);
            Fft::<CQ15>::fft(&Sdrk, &plan_q, xq, dir, scaling)?;
            xq.fill(CQ15 {
                re: Q15::MIN,
                im: Q15::MIN,
            });
            Fft::<CQ15>::fft(&Sdrk, &plan_q, xq, dir, scaling)?;
        }
    }
    // Block floating point keeps full-scale alternating input exact: its spectrum is a single
    // line of height N * (1 - 2^-15) per component.
    crate::r#gen::extreme::full_scale_alternating(xq);
    let s = Fft::<CQ15>::fft(&Sdrk, &plan_q, xq, FWD, Scaling::BlockFloat)?;
    let peak = xq
        .iter()
        .map(|v| i32::from(v.re.0).abs())
        .max()
        .unwrap_or(0);
    ensure(
        f64::from(peak) * libm::ldexp(1.0, s.exponent) > 0.9 * N as f64 * 32767.0,
        || Failure::new(CASE, "alternating input lost its line").detail(format_args!("{s:?}")),
    )?;
    // The unsupported combination is an error, not a panic.
    let e = Fft::<C32>::fft(&Sdrk, &plan_f, xf, FWD, Scaling::BlockFloat);
    ensure(matches!(e, Err(Error::Unsupported(_))), || {
        Failure::new(CASE, "c32 block floating point must be unsupported")
    })?;
    let _: FftScale = FftScale::default();
    Ok(())
}
