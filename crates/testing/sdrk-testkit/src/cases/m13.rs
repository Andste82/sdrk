//! Suite cases of the SIMD backend, run through the facade: `backend_vs_fallback_*`.
//!
//! Every accelerated kernel is run with the lengths 0, 1, 3, 15, 16, 17, 255 and 4096 (as far
//! as the arena allows), so that every tail of the vector loops is taken, and with extreme
//! inputs. The bit-exact kernels are compared bit for bit with the fallback; the tolerance
//! kernels against the contract of `docs/kernels.md`, measured against the reference. The
//! stateful kernels are also run in random chunks and must give the same bits as one call.
//!
//! The cases use no files, so they also run on the emulators of the microcontroller targets,
//! where the facade is the fallback and the comparison holds trivially.

#![allow(clippy::many_single_char_names, clippy::too_many_lines)]

use core::f64::consts::PI;

use sdrk::Sdrk;
use sdrk_core::kernels::biquad::{Biquad, BiquadState};
use sdrk_core::kernels::convert::{Convert, ConvertOut};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::demod::{QuadDemod, QuadDemodState};
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::interp::{FirInterpState, FirInterpolate};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::kernels::spectrum::{Mag, MagSq};
use sdrk_core::{C32, CQ15, Fallback, Q15, Rate, Reference};
use sdrk_device::{SampleFormat, formats};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use super::m06::{flat, min_snr, same_rate};
use crate::arena::Arena;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// The lengths of the matrix; the tails of 4, 8 and 16 lanes are all covered.
const LENGTHS: [usize; 8] = [0, 1, 3, 15, 16, 17, 255, 4096];

/// The lengths that fit an arena with `bytes_per_sample` bytes of buffers per sample.
fn lengths(remaining: usize, bytes_per_sample: usize) -> impl Iterator<Item = usize> {
    let room = remaining.saturating_sub(4096);
    LENGTHS
        .into_iter()
        .filter(move |&n| n.saturating_mul(bytes_per_sample) <= room)
}

fn fail(case: &'static str, msg: &'static str, n: usize) -> Failure {
    Failure::new(case, msg).detail(format_args!("n={n}"))
}

// ------------------------------------------------------------------------------- convert

/// The formats with a vector path and a few that go to the fallback.
const CONVERT_FORMATS: [SampleFormat; 5] = [
    formats::CU8,
    formats::CS8,
    formats::CS16,
    formats::CS12_LSB16,
    formats::CS16_PLANAR,
];

pub(super) fn backend_vs_fallback_convert(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_convert";
    let mut rng = Pcg32::new(1301, 1);
    for n in lengths(arena.remaining(), 4 + 8 + 8 + 8) {
        for format in CONVERT_FORMATS {
            let Ok(bytes_n) = format.bytes_for_samples(n) else {
                continue;
            };
            let mut c = arena.carver();
            let raw = c.take::<u8>(bytes_n)?;
            let (fb, fac) = (c.take::<C32>(n)?, c.take::<C32>(n)?);
            let (qb, qac) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
            for (i, b) in raw.iter_mut().enumerate() {
                // Random bytes, with every byte value and the extremes in the first 256.
                *b = if i < 256 {
                    i as u8
                } else {
                    rng.next_u32() as u8
                };
            }
            let nb = Convert::<C32>::convert(&Fallback, &format, raw, fb)?;
            let na = Convert::<C32>::convert(&Sdrk, &format, raw, fac)?;
            ensure(nb == na && bit_exact(fb, fac).is_none(), || {
                fail(CASE, "c32 differs from the fallback", n)
            })?;
            let nb = Convert::<CQ15>::convert(&Fallback, &format, raw, qb)?;
            let na = Convert::<CQ15>::convert(&Sdrk, &format, raw, qac)?;
            ensure(nb == na && bit_exact(qb, qac).is_none(), || {
                fail(CASE, "cq15 differs from the fallback", n)
            })?;
        }
    }
    // The error cases are the same: a partial sample and a short output.
    let mut out = [C32::ZERO; 2];
    for (len, room) in [(3usize, 2usize), (8, 2)] {
        let raw = [0u8; 8];
        let a = Convert::<C32>::convert(
            &Fallback,
            &formats::CS16,
            raw.get(..len).unwrap_or(&raw),
            out.get_mut(..room).unwrap_or_default(),
        );
        let b = Convert::<C32>::convert(
            &Sdrk,
            &formats::CS16,
            raw.get(..len).unwrap_or(&raw),
            out.get_mut(..room).unwrap_or_default(),
        );
        ensure(a == b, || Failure::new(CASE, "errors differ"))?;
    }
    Ok(())
}

pub(super) fn backend_vs_fallback_convert_out(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_convert_out";
    let mut rng = Pcg32::new(1302, 1);
    let specials = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.0 - f32::EPSILON,
        1.000_01,
        -1.000_01,
        3.9e-3,
        -3.9e-3,
        127.5 / 128.0,
        32767.5 / 32768.0,
        -32768.5 / 32768.0,
        1.0e-30,
        3.4e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for n in lengths(arena.remaining(), 8 + 4 + 8 + 8 + 4) {
        for format in [formats::CU8, formats::CS8, formats::CS16] {
            let Ok(bytes_n) = format.bytes_for_samples(n) else {
                continue;
            };
            let mut c = arena.carver();
            let (input, qin) = (c.take::<C32>(n)?, c.take::<CQ15>(n)?);
            let (fb, fac) = (c.take::<u8>(bytes_n)?, c.take::<u8>(bytes_n)?);
            for (i, s) in input.iter_mut().enumerate() {
                let pick = |r: &mut Pcg32, k: usize| match r.below(4) {
                    0 => specials.get(k % specials.len()).copied().unwrap_or(0.0),
                    1 => (r.next_f64() * 2.4 - 1.2) as f32,
                    2 => (f64::from(r.below(65536)) - 32768.0 + 0.5) as f32 / 32768.0,
                    _ => (f64::from(r.below(512)) - 256.0 + 0.5) as f32 / 128.0,
                };
                *s = C32::new(pick(&mut rng, i), pick(&mut rng, i / 3 + 1));
            }
            for s in qin.iter_mut() {
                *s = random_cq15(&mut rng, 1.0);
            }
            ConvertOut::<C32>::convert_out(&Fallback, &format, input, fb)?;
            ConvertOut::<C32>::convert_out(&Sdrk, &format, input, fac)?;
            ensure(fb == fac, || fail(CASE, "c32 differs from the fallback", n))?;
            ConvertOut::<CQ15>::convert_out(&Fallback, &format, qin, fb)?;
            ConvertOut::<CQ15>::convert_out(&Sdrk, &format, qin, fac)?;
            ensure(fb == fac, || {
                fail(CASE, "cq15 differs from the fallback", n)
            })?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------- rotate

pub(super) fn backend_vs_fallback_rotate(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_rotate";
    const N: usize = 1000;
    let mut rng = Pcg32::new(1303, 1);
    for n in lengths(arena.remaining(), 8 * 4) {
        let mut c = arena.carver();
        let (input, fb, rf, fac) = (
            c.take::<C32>(n)?,
            c.take::<C32>(n)?,
            c.take::<C32>(n)?,
            c.take::<C32>(n)?,
        );
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 1.0);
        }
        for inc in [0x0123_4567u32, 0xC000_0001, 1 << 30] {
            let mut a = Nco::from_increment(inc);
            let mut b = a;
            let mut d = a;
            Rotate::<C32>::rotate(&Fallback, &mut a, input, fb)?;
            Rotate::<C32>::rotate(&Reference, &mut b, input, rf)?;
            Rotate::<C32>::rotate(&Sdrk, &mut d, input, fac)?;
            if n > 0 {
                min_snr(CASE, "c32 below 120 dB", flat(rf), flat(fac), 120.0)?;
            }
            ensure(a == d && b == d, || {
                fail(CASE, "phase advanced differently", n)
            })?;
        }
    }
    for n in lengths(arena.remaining(), 4 * 4) {
        let mut c = arena.carver();
        let (input, rf, fac) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
        for (k, s) in input.iter_mut().enumerate() {
            *s = if k % 5 == 0 {
                CQ15 {
                    re: Q15(-32768),
                    im: Q15(-32768),
                }
            } else {
                random_cq15(&mut rng, 1.0)
            };
        }
        let mut b = Nco::from_increment(0x0765_4321);
        let mut d = b;
        Rotate::<CQ15>::rotate(&Reference, &mut b, input, rf)?;
        Rotate::<CQ15>::rotate(&Sdrk, &mut d, input, fac)?;
        // 80 dB against the reference (the integer rules of the fallback give 1 LSB at most).
        let worst = rf
            .iter()
            .zip(fac.iter())
            .map(|(x, y)| {
                (i32::from(x.re.0) - i32::from(y.re.0))
                    .abs()
                    .max((i32::from(x.im.0) - i32::from(y.im.0)).abs())
            })
            .max()
            .unwrap_or(0);
        ensure(worst <= 4, || {
            fail(CASE, "cq15 more than 4 LSB from the reference", n)
        })?;
        ensure(b == d, || fail(CASE, "cq15 phase advanced differently", n))?;
    }
    // Random chunks give the bits of one call.
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
    for s in input.iter_mut() {
        *s = random_c32(&mut rng, 1.0);
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(1304, 47),
        || Nco::from_increment(0x0123_4567),
        |nco, i, o| same_rate(i, o, |i, o| Rotate::<C32>::rotate(&Sdrk, nco, i, o)),
    )?;
    Ok(())
}

// -------------------------------------------------------------------------------------- fir

fn lowpass_f32(out: &mut [f32], fc: f64) {
    let n = out.len();
    let mid = (n.max(1) - 1) as f64 / 2.0;
    let mut sum = 0.0;
    for (k, t) in out.iter_mut().enumerate() {
        let x = k as f64 - mid;
        let sinc = if x == 0.0 {
            2.0 * fc
        } else {
            libm::sin(2.0 * PI * fc * x) / (PI * x)
        };
        let w = if n > 1 {
            0.54 - 0.46 * libm::cos(2.0 * PI * k as f64 / (n - 1) as f64)
        } else {
            1.0
        };
        *t = (sinc * w) as f32;
        sum += sinc * w;
    }
    for t in out.iter_mut() {
        *t = (f64::from(*t) / sum) as f32;
    }
}

fn lowpass_q15(out: &mut [Q15], fc: f64, scratch: &mut [f32]) {
    lowpass_f32(scratch, fc);
    for (q, t) in out.iter_mut().zip(scratch.iter()) {
        *q = sdrk_core::num::q15_from_f64(f64::from(*t));
    }
}

pub(super) fn backend_vs_fallback_fir(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_fir";
    const N: usize = 1000;
    const TAPS: usize = 31;
    let mut rng = Pcg32::new(1305, 1);
    for n in lengths(arena.remaining(), 8 * 4 + 64) {
        let mut c = arena.carver();
        let (input, rf, fac) = (c.take::<C32>(n)?, c.take::<C32>(n)?, c.take::<C32>(n)?);
        let taps = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<C32>(TAPS - 1)?, c.take::<C32>(TAPS - 1)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 0.9);
        }
        lowpass_f32(taps, 0.2);
        let mut a = FirState::<C32, _, _>::new(&*taps, h1)?;
        let mut d = FirState::<C32, _, _>::new(&*taps, h2)?;
        Fir::<C32>::fir(&Reference, &mut a, input, rf)?;
        Fir::<C32>::fir(&Sdrk, &mut d, input, fac)?;
        if n > 0 {
            min_snr(CASE, "c32 below 110 dB", flat(rf), flat(fac), 110.0)?;
        }
        ensure(a.history() == d.history(), || {
            fail(CASE, "histories differ", n)
        })?;
    }
    for n in lengths(arena.remaining(), 4 * 3 + 64) {
        let mut c = arena.carver();
        let (input, fb, fac) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
        let taps = c.take::<Q15>(TAPS)?;
        let scratch = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<CQ15>(TAPS - 1)?, c.take::<CQ15>(TAPS - 1)?);
        for (k, s) in input.iter_mut().enumerate() {
            *s = if k % 7 == 0 {
                CQ15 {
                    re: Q15(-32768),
                    im: Q15(32767),
                }
            } else {
                random_cq15(&mut rng, 1.0)
            };
        }
        lowpass_q15(taps, 0.2, scratch);
        let mut a = FirState::<CQ15, _, _>::new(&*taps, h1)?;
        let mut d = FirState::<CQ15, _, _>::new(&*taps, h2)?;
        Fir::<CQ15>::fir(&Fallback, &mut a, input, fb)?;
        Fir::<CQ15>::fir(&Sdrk, &mut d, input, fac)?;
        ensure(bit_exact(fb, fac).is_none(), || {
            fail(CASE, "cq15 differs from the fallback", n)
        })?;
        ensure(a.history() == d.history(), || {
            fail(CASE, "cq15 histories differ", n)
        })?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
    let taps = c.take::<f32>(TAPS)?;
    for s in input.iter_mut() {
        *s = random_c32(&mut rng, 1.0);
    }
    lowpass_f32(taps, 0.2);
    let taps: &[f32] = taps;
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(1306, 53),
        || FirState::<C32, _, _>::new(taps, [C32::ZERO; TAPS - 1]),
        |state, i, o| match state {
            Ok(st) => same_rate(i, o, |i, o| Fir::<C32>::fir(&Sdrk, st, i, o)),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

pub(super) fn backend_vs_fallback_fir_decimate(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_fir_decimate";
    const N: usize = 1000;
    const TAPS: usize = 63;
    const M: usize = 10;
    let mut rng = Pcg32::new(1307, 1);
    for n in lengths(arena.remaining(), 8 * 3 + 8 + 64) {
        let mut c = arena.carver();
        let (input, rf, fac) = (c.take::<C32>(n)?, c.take::<C32>(n)?, c.take::<C32>(n)?);
        let taps = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<C32>(TAPS - 1)?, c.take::<C32>(TAPS - 1)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 0.9);
        }
        lowpass_f32(taps, 0.04);
        let mut a = FirDecimState::<C32, _, _>::new(&*taps, h1, M)?;
        let mut d = FirDecimState::<C32, _, _>::new(&*taps, h2, M)?;
        let pb = FirDecimate::<C32>::fir_decimate(&Reference, &mut a, input, rf)?;
        let pd = FirDecimate::<C32>::fir_decimate(&Sdrk, &mut d, input, fac)?;
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
            || fail(CASE, "states differ", n),
        )?;
    }
    for n in lengths(arena.remaining(), 4 * 3 + 4 + 64) {
        let mut c = arena.carver();
        let (input, fb, fac) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
        let taps = c.take::<Q15>(TAPS)?;
        let scratch = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<CQ15>(TAPS - 1)?, c.take::<CQ15>(TAPS - 1)?);
        for s in input.iter_mut() {
            *s = random_cq15(&mut rng, 1.0);
        }
        lowpass_q15(taps, 0.04, scratch);
        let mut a = FirDecimState::<CQ15, _, _>::new(&*taps, h1, M)?;
        let mut d = FirDecimState::<CQ15, _, _>::new(&*taps, h2, M)?;
        let pb = FirDecimate::<CQ15>::fir_decimate(&Fallback, &mut a, input, fb)?;
        let pd = FirDecimate::<CQ15>::fir_decimate(&Sdrk, &mut d, input, fac)?;
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
    // The unconsumed input is resubmitted when the output space is short.
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
    let taps = c.take::<f32>(TAPS)?;
    for s in input.iter_mut() {
        *s = random_c32(&mut rng, 1.0);
    }
    lowpass_f32(taps, 0.04);
    let taps: &[f32] = taps;
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(1308, 61),
        || FirDecimState::<C32, _, _>::new(taps, [C32::ZERO; TAPS - 1], M),
        |state, i, o| match state {
            Ok(st) => FirDecimate::<C32>::fir_decimate(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

pub(super) fn backend_vs_fallback_fir_interpolate(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_fir_interpolate";
    const L: usize = 4;
    const TAPS: usize = 32;
    const HIST: usize = TAPS / L;
    let mut rng = Pcg32::new(1309, 1);
    for n in lengths(arena.remaining(), 8 + 8 * L * 2 + 64) {
        let mut c = arena.carver();
        let input = c.take::<C32>(n)?;
        let (rf, fac) = (c.take::<C32>(n * L)?, c.take::<C32>(n * L)?);
        let taps = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<C32>(HIST)?, c.take::<C32>(HIST)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 0.9);
        }
        lowpass_f32(taps, 0.1);
        for t in taps.iter_mut() {
            *t *= L as f32;
        }
        let mut a = FirInterpState::<C32, _, _>::new(&*taps, h1, L)?;
        let mut d = FirInterpState::<C32, _, _>::new(&*taps, h2, L)?;
        let pb = FirInterpolate::<C32>::fir_interpolate(&Reference, &mut a, input, rf)?;
        let pd = FirInterpolate::<C32>::fir_interpolate(&Sdrk, &mut d, input, fac)?;
        ensure(pb == pd && pb.produced == n * L, || {
            fail(CASE, "progress differs", n)
        })?;
        if n > 0 {
            min_snr(CASE, "c32 below 110 dB", flat(rf), flat(fac), 110.0)?;
        }
        ensure(a.history() == d.history(), || {
            fail(CASE, "histories differ", n)
        })?;
    }
    for n in lengths(arena.remaining(), 4 + 4 * L * 2 + 64) {
        let mut c = arena.carver();
        let input = c.take::<CQ15>(n)?;
        let (fb, fac) = (c.take::<CQ15>(n * L)?, c.take::<CQ15>(n * L)?);
        let taps = c.take::<Q15>(TAPS)?;
        let scratch = c.take::<f32>(TAPS)?;
        let (h1, h2) = (c.take::<CQ15>(HIST)?, c.take::<CQ15>(HIST)?);
        for s in input.iter_mut() {
            *s = random_cq15(&mut rng, 1.0);
        }
        lowpass_q15(taps, 0.1, scratch);
        let mut a = FirInterpState::<CQ15, _, _>::new(&*taps, h1, L)?;
        let mut d = FirInterpState::<CQ15, _, _>::new(&*taps, h2, L)?;
        FirInterpolate::<CQ15>::fir_interpolate(&Fallback, &mut a, input, fb)?;
        FirInterpolate::<CQ15>::fir_interpolate(&Sdrk, &mut d, input, fac)?;
        ensure(bit_exact(fb, fac).is_none(), || {
            fail(CASE, "cq15 differs from the fallback", n)
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------- mag_sq

pub(super) fn backend_vs_fallback_mag_sq(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_mag_sq";
    let mut rng = Pcg32::new(1310, 1);
    for n in lengths(arena.remaining(), 8 + 4 * 3 + 4 * 2) {
        let mut c = arena.carver();
        let (input, qin) = (c.take::<C32>(n)?, c.take::<CQ15>(n)?);
        let (fb, fac) = (c.take::<f32>(n)?, c.take::<f32>(n)?);
        let (qb, qac) = (c.take::<Q15>(n)?, c.take::<Q15>(n)?);
        for (k, s) in input.iter_mut().enumerate() {
            *s = match k % 11 {
                3 => C32::new(f32::NAN, 1.0),
                5 => C32::new(f32::INFINITY, 0.0),
                _ => random_c32(&mut rng, 2.0),
            };
        }
        for (k, s) in qin.iter_mut().enumerate() {
            *s = if k % 4 == 0 {
                CQ15 {
                    re: Q15(-32768),
                    im: Q15(-32768),
                }
            } else {
                random_cq15(&mut rng, 1.0)
            };
        }
        MagSq::<C32>::mag_sq(&Fallback, input, fb)?;
        MagSq::<C32>::mag_sq(&Sdrk, input, fac)?;
        // The squares are exact in both: the same bits, NaN payloads aside.
        ensure(
            fb.iter()
                .zip(fac.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
            || fail(CASE, "c32 differs from the fallback", n),
        )?;
        MagSq::<CQ15>::mag_sq(&Fallback, qin, qb)?;
        MagSq::<CQ15>::mag_sq(&Sdrk, qin, qac)?;
        ensure(bit_exact(qb, qac).is_none(), || {
            fail(CASE, "cq15 differs from the fallback", n)
        })?;
    }
    Ok(())
}

// ------------------------------------------------------------------------------------- mag

pub(super) fn backend_vs_fallback_mag(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_mag";
    let mut rng = Pcg32::new(2010, 1);
    for n in lengths(arena.remaining(), 8 + 4 + 4 * 2 + 2 * 2) {
        let mut c = arena.carver();
        let (input, qin) = (c.take::<C32>(n)?, c.take::<CQ15>(n)?);
        let (fb, fac) = (c.take::<f32>(n)?, c.take::<f32>(n)?);
        let (qb, qac) = (c.take::<Q15>(n)?, c.take::<Q15>(n)?);
        for (k, s) in input.iter_mut().enumerate() {
            *s = match k % 11 {
                3 => C32::new(f32::NAN, 1.0),
                5 => C32::new(f32::INFINITY, 0.0),
                7 => C32::new(0.0, 0.0),
                _ => random_c32(&mut rng, 2.0),
            };
        }
        for (k, s) in qin.iter_mut().enumerate() {
            *s = match k % 5 {
                0 => CQ15 {
                    re: Q15(-32768),
                    im: Q15(-32768),
                },
                1 => CQ15 {
                    re: Q15(0),
                    im: Q15(0),
                },
                _ => random_cq15(&mut rng, 1.0),
            };
        }
        Mag::<C32>::mag(&Fallback, input, fb)?;
        Mag::<C32>::mag(&Sdrk, input, fac)?;
        // The square root is correctly rounded in both: the same bits, NaN payloads aside.
        ensure(
            fb.iter()
                .zip(fac.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
            || fail(CASE, "c32 differs from the fallback", n),
        )?;
        Mag::<CQ15>::mag(&Fallback, qin, qb)?;
        Mag::<CQ15>::mag(&Sdrk, qin, qac)?;
        ensure(bit_exact(qb, qac).is_none(), || {
            fail(CASE, "cq15 differs from the fallback", n)
        })?;
    }
    Ok(())
}

// ------------------------------------------------------------------------------- quad_demod

fn fm_signal_c32(out: &mut [C32], rng: &mut Pcg32) {
    let mut phase = 0.0f64;
    for (k, s) in out.iter_mut().enumerate() {
        phase += 0.3 * libm::sin(k as f64 * 0.01) + 0.02 * (rng.next_f64() - 0.5);
        let amp = 0.5 + 0.4 * rng.next_f64();
        *s = C32::new(
            (amp * libm::cos(phase)) as f32,
            (amp * libm::sin(phase)) as f32,
        );
    }
}

pub(super) fn backend_vs_fallback_quad_demod(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_quad_demod";
    const N: usize = 1000;
    let mut rng = Pcg32::new(1311, 1);
    for n in lengths(arena.remaining(), 8 + 4 * 2) {
        let mut c = arena.carver();
        let input = c.take::<C32>(n)?;
        let (rf, fac) = (c.take::<f32>(n)?, c.take::<f32>(n)?);
        fm_signal_c32(input, &mut rng);
        let (mut a, mut d) = (QuadDemodState::<C32>::new(), QuadDemodState::<C32>::new());
        QuadDemod::<C32>::quad_demod(&Reference, &mut a, input, rf, 1.5)?;
        QuadDemod::<C32>::quad_demod(&Sdrk, &mut d, input, fac, 1.5)?;
        if n > 1 {
            min_snr(CASE, "c32 below 100 dB", rf, fac, 100.0)?;
        }
        ensure(a == d, || fail(CASE, "states differ", n))?;
    }
    for n in lengths(arena.remaining(), 4 + 2 * 2) {
        let mut c = arena.carver();
        let input = c.take::<CQ15>(n)?;
        let (rf, fac) = (c.take::<Q15>(n)?, c.take::<Q15>(n)?);
        for (k, s) in input.iter_mut().enumerate() {
            let p = k as f64 * 0.4;
            *s = if k % 9 == 0 {
                CQ15 {
                    re: Q15(-32768),
                    im: Q15(-32768),
                }
            } else {
                CQ15 {
                    re: Q15((20000.0 * libm::cos(p)) as i16),
                    im: Q15((20000.0 * libm::sin(p)) as i16),
                }
            };
        }
        let (mut a, mut d) = (QuadDemodState::<CQ15>::new(), QuadDemodState::<CQ15>::new());
        QuadDemod::<CQ15>::quad_demod(&Reference, &mut a, input, rf, Q15(30000))?;
        QuadDemod::<CQ15>::quad_demod(&Sdrk, &mut d, input, fac, Q15(30000))?;
        let worst = rf
            .iter()
            .zip(fac.iter())
            .map(|(x, y)| (i32::from(x.0) - i32::from(y.0)).abs())
            .max()
            .unwrap_or(0);
        // 70 dB against the reference leaves room for far more than this.
        ensure(worst <= 2, || {
            fail(CASE, "cq15 more than 2 LSB from the reference", n)
        })?;
        ensure(a == d, || fail(CASE, "cq15 states differ", n))?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
    fm_signal_c32(input, &mut rng);
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(1312, 41),
        QuadDemodState::<C32>::new,
        |state, i, o| {
            same_rate(i, o, |i, o| {
                QuadDemod::<C32>::quad_demod(&Sdrk, state, i, o, 1.0)
            })
        },
    )?;
    Ok(())
}

// ----------------------------------------------------------------------------------- fm_mod

pub(super) fn backend_vs_fallback_fm_mod(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_fm_mod";
    const N: usize = 1000;
    let mut rng = Pcg32::new(1313, 1);
    let state = || FmModState::new(75_000.0, Rate::hz(240_000)?);
    for n in lengths(arena.remaining(), 4 + 8 * 2) {
        let mut c = arena.carver();
        let input = c.take::<f32>(n)?;
        let (rf, fac) = (c.take::<C32>(n)?, c.take::<C32>(n)?);
        for (k, s) in input.iter_mut().enumerate() {
            *s = match k % 13 {
                4 => f32::NAN,
                7 => 1.0,
                8 => -1.0,
                _ => (rng.next_f64() * 2.2 - 1.1) as f32,
            };
        }
        let (mut a, mut d) = (state()?, state()?);
        FmMod::<C32>::fm_mod(&Reference, &mut a, input, rf)?;
        FmMod::<C32>::fm_mod(&Sdrk, &mut d, input, fac)?;
        // A NaN input has no defined phase step: compare the stretch before the first one.
        let end = input.iter().position(|v| v.is_nan()).unwrap_or(n);
        if end > 0 {
            min_snr(
                CASE,
                "c32 below 120 dB",
                flat(rf.get(..end).unwrap_or_default()),
                flat(fac.get(..end).unwrap_or_default()),
                120.0,
            )?;
        }
        ensure(a.phase() == d.phase(), || fail(CASE, "phases differ", n))?;
    }
    for n in lengths(arena.remaining(), 2 + 4 * 2) {
        let mut c = arena.carver();
        let input = c.take::<Q15>(n)?;
        let (fb, fac) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
        for (k, s) in input.iter_mut().enumerate() {
            *s = if k % 5 == 0 {
                Q15(-32768)
            } else {
                Q15(rng.next_u32() as i16)
            };
        }
        let (mut a, mut d) = (state()?, state()?);
        FmMod::<CQ15>::fm_mod(&Fallback, &mut a, input, fb)?;
        FmMod::<CQ15>::fm_mod(&Sdrk, &mut d, input, fac)?;
        ensure(
            bit_exact(fb, fac).is_none() && a.phase() == d.phase(),
            || fail(CASE, "cq15 differs from the fallback", n),
        )?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<f32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
    for s in input.iter_mut() {
        *s = (rng.next_f64() * 2.0 - 1.0) as f32;
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(1314, 43),
        state,
        |st, i, o| match st {
            Ok(st) => same_rate(i, o, |i, o| FmMod::<C32>::fm_mod(&Sdrk, st, i, o)),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------------------- fft

fn snr_c64<W: sdrk_core::ComplexSample, G: sdrk_core::ComplexSample>(
    want: &[W],
    got: &[G],
    want_exp: i32,
    got_exp: i32,
) -> f64 {
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    let (gw, gg) = (libm::ldexp(1.0, want_exp), libm::ldexp(1.0, got_exp));
    for (w, g) in want.iter().zip(got) {
        let (w, g) = (w.to_c64() * gw, g.to_c64() * gg);
        sig += w.norm_sqr();
        err += (w - g).norm_sqr();
    }
    if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(sig / err)
    }
}

pub(super) fn backend_vs_fallback_fft(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_fft";
    let mut rng = Pcg32::new(1315, 1);
    for n in [16usize, 64, 256, 1024, 4096] {
        if n * 8 * 4 + n * 4 > arena.remaining().saturating_sub(4096) {
            continue;
        }
        let mut c = arena.carver();
        let (x, rf, fac) = (c.take::<C32>(n)?, c.take::<C32>(n)?, c.take::<C32>(n)?);
        let plan = FftPlan::<C32, _>::new(n, c.take::<C32>(n / 2)?)?;
        for dir in [Direction::Forward, Direction::Inverse] {
            for (k, s) in x.iter_mut().enumerate() {
                *s = if k % 31 == 7 {
                    C32::new(0.99, -0.99)
                } else {
                    random_c32(&mut rng, 0.5)
                };
            }
            let want = Reference.dft(x, rf, dir, Scaling::Default)?;
            fac.copy_from_slice(x);
            let got = Fft::<C32>::fft(&Sdrk, &plan, fac, dir, Scaling::Default)?;
            ensure(want == got, || fail(CASE, "c32 exponent differs", n))?;
            let snr = snr_c64(rf, fac, 0, 0);
            ensure(snr >= 110.0, || {
                Failure::new(CASE, "c32 below 110 dB").detail(format_args!("n={n} {snr} dB"))
            })?;
        }
    }
    // The CQ15 FFT of the backend is not bit-exact with the fallback (decision D10): both are
    // measured against the exact transform (the C32 reference of the same input, unnormalised).
    // Fixed scaling must stay within 1 dB of the fallback and give its exponent. The inputs
    // here (near full scale with corner values, or a few LSB) are outside the block floating
    // point contract, which `fft_cq15_bfp_vs_reference_1024` checks: the backend, which picks
    // one shift per stage from the largest norm, may be up to 8 dB below the fallback there.
    for n in [16usize, 64, 256, 1024, 4096] {
        if n * 4 * 4 + n * 2 + n * 8 * 2 > arena.remaining().saturating_sub(4096) {
            continue;
        }
        let mut c = arena.carver();
        let (x, fb, fac) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
        let (xc, rc) = (c.take::<C32>(n)?, c.take::<C32>(n)?);
        let plan = FftPlan::<CQ15, _>::new(n, c.take::<CQ15>(n / 2)?)?;
        for weak in [false, true] {
            for (k, s) in x.iter_mut().enumerate() {
                *s = if weak {
                    CQ15 {
                        re: Q15((rng.below(64) as i16) - 32),
                        im: Q15(0),
                    }
                } else if k % 9 == 0 {
                    CQ15 {
                        re: Q15(-32768),
                        im: Q15(32767),
                    }
                } else {
                    random_cq15(&mut rng, 0.9)
                };
            }
            for (d, s) in xc.iter_mut().zip(x.iter()) {
                *d = C32::new(f32::from(s.re.0) / 32768.0, f32::from(s.im.0) / 32768.0);
            }
            let exact = Reference.dft(xc, rc, Direction::Forward, Scaling::Default)?;
            for scaling in [Scaling::Default, Scaling::BlockFloat] {
                fb.copy_from_slice(x);
                fac.copy_from_slice(x);
                let a = Fft::<CQ15>::fft(&Fallback, &plan, fb, Direction::Forward, scaling)?;
                let b = Fft::<CQ15>::fft(&Sdrk, &plan, fac, Direction::Forward, scaling)?;
                let want = snr_c64(rc, fb, exact.exponent, a.exponent);
                let got = snr_c64(rc, fac, exact.exponent, b.exponent);
                let (ok, what) = if scaling == Scaling::Default {
                    (
                        a == b && got >= want - 1.0,
                        "cq15 fixed scaling differs from the fallback",
                    )
                } else {
                    (
                        got >= want - 8.0,
                        "cq15 block floating point far below the fallback",
                    )
                };
                ensure(ok, || {
                    Failure::new(CASE, what).detail(format_args!(
                        "n={n} weak={weak} {got} dB (fallback {want} dB), exponents {} {}",
                        a.exponent, b.exponent
                    ))
                })?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------- biquad

/// The fourth-order Butterworth low-pass at 0.1 of the sample rate (two sections), in `f32` and
/// Q2.14.
const BIQUAD_F32: [[f32; 5]; 2] = [
    [
        0.077_956_34,
        0.155_912_68,
        0.077_956_34,
        -1.320_913_4,
        0.632_738_8,
    ],
    [
        0.061_885_195,
        0.123_770_39,
        0.061_885_195,
        -1.048_599_6,
        0.296_140_36,
    ],
];
const BIQUAD_Q14: [[i16; 5]; 2] = [
    [1277, 2554, 1277, -21642, 10367],
    [1014, 2028, 1014, -17180, 4852],
];
/// Sections that overflow and saturate: poles outside the unit circle.
const BIQUAD_WILD_F32: [[f32; 5]; 2] = [[1.5, -1.9, 1.7, -1.9, 1.5], [-1.2, 1.8, 0.9, 1.9, -1.8]];
const BIQUAD_WILD_Q14: [[i16; 5]; 2] = [
    [i16::MAX, i16::MIN, i16::MAX, i16::MIN, i16::MAX],
    [i16::MIN, i16::MAX, 20000, i16::MAX, i16::MIN],
];

/// The two-lane path of `biquad` on `C32` and `CQ15` (re and im in the two lanes of a vector)
/// against the fallback: the same bits, also with NaN, infinities and subnormal samples, with
/// coefficients that overflow and saturate, and in random chunks.
pub(super) fn backend_vs_fallback_biquad(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "backend_vs_fallback_biquad";
    const N: usize = 1000;
    let mut rng = Pcg32::new(2201, 1);
    for n in lengths(arena.remaining(), 8 * 3 + 64 * 2) {
        for coefs in [BIQUAD_F32, BIQUAD_WILD_F32] {
            let mut c = arena.carver();
            let (input, fb, fac) = (c.take::<C32>(n)?, c.take::<C32>(n)?, c.take::<C32>(n)?);
            let (h1, h2) = (c.take::<C32>(8)?, c.take::<C32>(8)?);
            for (k, s) in input.iter_mut().enumerate() {
                *s = match k % 13 {
                    3 => C32::new(f32::NAN, 1.0),
                    5 => C32::new(f32::INFINITY, 0.0),
                    7 => C32::new(1e-40, -1e-33),
                    9 => C32::new(0.0, 0.0),
                    _ => random_c32(&mut rng, 1.5),
                };
            }
            let mut a = BiquadState::<C32, _, _>::new(coefs, h1)?;
            let mut d = BiquadState::<C32, _, _>::new(coefs, h2)?;
            Biquad::<C32>::biquad(&Fallback, &mut a, input, fb)?;
            Biquad::<C32>::biquad(&Sdrk, &mut d, input, fac)?;
            let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
            ensure(
                fb.iter()
                    .zip(fac.iter())
                    .all(|(a, b)| same(a.re, b.re) && same(a.im, b.im)),
                || fail(CASE, "c32 differs from the fallback", n),
            )?;
            ensure(
                a.history()
                    .iter()
                    .zip(d.history())
                    .all(|(a, b)| same(a.re, b.re) && same(a.im, b.im)),
                || fail(CASE, "c32 histories differ", n),
            )?;
        }
    }
    for n in lengths(arena.remaining(), 4 * 3 + 32 * 2) {
        for coefs in [BIQUAD_Q14, BIQUAD_WILD_Q14] {
            let mut c = arena.carver();
            let (input, fb, fac) = (c.take::<CQ15>(n)?, c.take::<CQ15>(n)?, c.take::<CQ15>(n)?);
            let (h1, h2) = (c.take::<CQ15>(8)?, c.take::<CQ15>(8)?);
            for (k, s) in input.iter_mut().enumerate() {
                *s = if k % 7 == 0 {
                    CQ15 {
                        re: Q15(-32768),
                        im: Q15(32767),
                    }
                } else {
                    random_cq15(&mut rng, 1.0)
                };
            }
            let mut a = BiquadState::<CQ15, _, _>::new(coefs, h1)?;
            let mut d = BiquadState::<CQ15, _, _>::new(coefs, h2)?;
            Biquad::<CQ15>::biquad(&Fallback, &mut a, input, fb)?;
            Biquad::<CQ15>::biquad(&Sdrk, &mut d, input, fac)?;
            ensure(bit_exact(fb, fac).is_none(), || {
                fail(CASE, "cq15 differs from the fallback", n)
            })?;
            ensure(a.history() == d.history(), || {
                fail(CASE, "cq15 histories differ", n)
            })?;
        }
    }
    // Random chunks give the bits of one call.
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
    for s in input.iter_mut() {
        *s = random_c32(&mut rng, 1.0);
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(2202, 53),
        || BiquadState::<C32, _, _>::new(BIQUAD_F32, [C32::ZERO; 8]),
        |state, i, o| match state {
            Ok(st) => same_rate(i, o, |i, o| Biquad::<C32>::biquad(&Sdrk, st, i, o)),
            Err(e) => Err(e.clone()),
        },
    )?;
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 1.0);
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(2203, 53),
        || BiquadState::<CQ15, _, _>::new(BIQUAD_Q14, [CQ15::default(); 8]),
        |state, i, o| match state {
            Ok(st) => same_rate(i, o, |i, o| Biquad::<CQ15>::biquad(&Sdrk, st, i, o)),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}
