//! Suite cases of the transmit and fused kernels and of the end-to-end chains: `convert_out`,
//! `fir_interpolate`, `fm_mod`, `ddc`, `duc`, `fm_demod`, the FM receive chain and the FM
//! transmit-then-receive chain.
//!
//! The kernel cases compare `Fallback` (and the facade) with `Reference` and the independent
//! encoder; no files are read, so the cases run on microcontrollers too.

#![allow(clippy::many_single_char_names, clippy::too_many_lines)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::convert::{Convert, ConvertOut, ConvertOutReal, ConvertReal};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::demod::{Deemphasis, DeemphasisState, QuadDemod, QuadDemodState};
use sdrk_core::kernels::fused::{Ddc, Duc, FmDemod};
use sdrk_core::kernels::interp::{
    FirInterpRealState, FirInterpState, FirInterpolate, FirInterpolateReal,
};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::num::q15_from_f64;
use sdrk_core::{
    C32, CQ15, ComplexSample, Delay, Fallback, Progress, Q15, Rate, Ratio, RealSample, Reference,
    Timing,
};
use sdrk_device::{Align, Container, Endian, FullScale, Layout, SampleFormat, formats};

use super::m05::{FORMATS, random_c32, random_cq15};
use super::m06::{flat, hand_out, min_snr, same_rate};
use super::{ensure, half_lsb};
use crate::arena::{Arena, Carver};
use crate::e2e::{self, E2eSample, RxKernels, TxKernels, design};
use crate::encode::{decode_reference, encode};
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

// ---------------------------------------------------------------------------------------
// convert_out
// ---------------------------------------------------------------------------------------

/// Formats beyond the constants of `docs/device.md`: byte order, alignment and I/Q order.
fn extra_formats() -> [SampleFormat; 4] {
    let make = |c, bits, layout, align, endian| {
        SampleFormat::new(c, bits, c != Container::U8, layout, align, endian)
            .unwrap_or(formats::CS16)
    };
    [
        make(
            Container::I16,
            16,
            Layout::Interleaved,
            Align::Lsb,
            Endian::Big,
        ),
        make(
            Container::I8,
            6,
            Layout::Interleaved,
            Align::Msb,
            Endian::Little,
        ),
        formats::CU8.with_swap_iq(true),
        formats::CU8
            .with_full_scale(FullScale::Explicit(200))
            .unwrap_or(formats::CU8),
    ]
}

/// The amplitude range `(2^(bits-1) / full_scale)` that a format represents.
fn range_of(f: &SampleFormat) -> f64 {
    (1u64 << (f.bits() - 1)) as f64 / f.full_scale_value()
}

pub(super) fn convert_out_all_formats_roundtrip(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "convert_out_all_formats_roundtrip";
    const N: usize = 64;
    let mut c = arena.carver();
    let x32 = c.take::<C32>(N)?;
    let xq = c.take::<CQ15>(N)?;
    let oracle = c.take::<Complex64>(N)?;
    let (b_fb, b_rf, b_fac, b_or) = (
        c.take::<u8>(N * 8)?,
        c.take::<u8>(N * 8)?,
        c.take::<u8>(N * 8)?,
        c.take::<u8>(N * 8)?,
    );
    let back32 = c.take::<C32>(N)?;
    let again = c.take::<u8>(N * 8)?;
    let decoded = c.take::<Complex64>(N)?;
    let mut rng = Pcg32::new(801, 8);
    let formats_to_test = FORMATS.into_iter().chain(extra_formats());
    for (index, format) in formats_to_test.enumerate() {
        let format = &format;
        let fail =
            |msg: &'static str| Failure::new(CASE, msg).detail(format_args!("format {index}"));
        let real = format.layout() == Layout::Real;
        let float = format.container() == Container::F32;
        // 1. Random samples beyond full scale, with the awkward values first.
        for s in x32.iter_mut() {
            *s = random_c32(&mut rng, 1.3);
        }
        let half = (0.5 / format.full_scale_value()) as f32;
        let specials = [
            C32::new(f32::NAN, f32::INFINITY),
            C32::new(f32::NEG_INFINITY, -0.0),
            C32::new(1.0, -1.0),
            C32::new(half, -half),
            C32::new(3.0 * half, -3.0 * half),
            C32::new(f32::MAX, f32::MIN),
        ];
        for (d, s) in x32.iter_mut().zip(specials) {
            *d = s;
        }
        for (o, s) in oracle.iter_mut().zip(x32.iter()) {
            *o = Complex64::new(f64::from(s.re), f64::from(s.im));
        }
        let want = encode(format, oracle, b_or)?;
        let (nf, nr, nd);
        if real {
            let mut re = [0f32; N];
            for (r, s) in re.iter_mut().zip(x32.iter()) {
                *r = s.re;
            }
            nf = ConvertOutReal::<f32>::convert_out_real(&Fallback, format, &re, b_fb)?;
            nr = ConvertOutReal::<f32>::convert_out_real(&Reference, format, &re, b_rf)?;
            nd = ConvertOutReal::<f32>::convert_out_real(&Sdrk, format, &re, b_fac)?;
        } else {
            nf = ConvertOut::<C32>::convert_out(&Fallback, format, x32, b_fb)?;
            nr = ConvertOut::<C32>::convert_out(&Reference, format, x32, b_rf)?;
            nd = ConvertOut::<C32>::convert_out(&Sdrk, format, x32, b_fac)?;
        }
        ensure(nf == want && nr == want && nd == want, || {
            fail("byte count differs from the oracle")
        })?;
        let (bf, br, bd, bo) = (
            b_fb.get(..want).unwrap_or_default(),
            b_rf.get(..want).unwrap_or_default(),
            b_fac.get(..want).unwrap_or_default(),
            b_or.get(..want).unwrap_or_default(),
        );
        ensure(bf == bo, || fail("c32: fallback differs from the oracle"))?;
        ensure(br == bo, || fail("c32: reference differs from the oracle"))?;
        ensure(bd == bo, || fail("c32: facade differs from the oracle"))?;

        // 2. Random raw Q15 values: the integer rules equal the f64 formula of the oracle on q / 32768.
        for s in xq.iter_mut() {
            *s = random_cq15(&mut rng, 1.0);
        }
        for (d, s) in xq.iter_mut().zip([
            CQ15 {
                re: Q15::MIN,
                im: Q15::MAX,
            },
            CQ15 {
                re: Q15::MAX,
                im: Q15::MIN,
            },
            CQ15 {
                re: Q15(0),
                im: Q15(-1),
            },
        ]) {
            *d = s;
        }
        for (o, s) in oracle.iter_mut().zip(xq.iter()) {
            *o = s.to_c64();
        }
        let want_q = encode(format, oracle, b_or)?;
        if real {
            let mut re = [Q15(0); N];
            for (r, s) in re.iter_mut().zip(xq.iter()) {
                *r = s.re;
            }
            ConvertOutReal::<Q15>::convert_out_real(&Fallback, format, &re, b_fb)?;
            ConvertOutReal::<Q15>::convert_out_real(&Reference, format, &re, b_rf)?;
            ConvertOutReal::<Q15>::convert_out_real(&Sdrk, format, &re, b_fac)?;
        } else {
            ConvertOut::<CQ15>::convert_out(&Fallback, format, xq, b_fb)?;
            ConvertOut::<CQ15>::convert_out(&Reference, format, xq, b_rf)?;
            ConvertOut::<CQ15>::convert_out(&Sdrk, format, xq, b_fac)?;
        }
        let bo = b_or.get(..want_q).unwrap_or_default();
        ensure(b_fb.get(..want_q) == Some(bo), || {
            fail("cq15: fallback differs from the oracle")
        })?;
        ensure(b_rf.get(..want_q) == Some(bo), || {
            fail("cq15: reference differs from the oracle")
        })?;
        ensure(b_fac.get(..want_q) == Some(bo), || {
            fail("cq15: facade differs from the oracle")
        })?;

        // 3. Round trip: encode, decode with the independent decoder, stay within half an LSB.
        let scale = 0.95 * range_of(format).min(1.0);
        for s in x32.iter_mut() {
            *s = random_c32(&mut rng, scale);
        }
        for (o, s) in oracle.iter_mut().zip(x32.iter()) {
            *o = Complex64::new(f64::from(s.re), f64::from(s.im));
        }
        let n_bytes = if real {
            let mut re = [0f32; N];
            for (r, s) in re.iter_mut().zip(x32.iter()) {
                *r = s.re;
            }
            ConvertOutReal::<f32>::convert_out_real(&Sdrk, format, &re, b_fac)?
        } else {
            ConvertOut::<C32>::convert_out(&Sdrk, format, x32, b_fac)?
        };
        let raw = b_fac.get(..n_bytes).unwrap_or_default();
        ensure(decode_reference(format, raw, decoded)? == N, || {
            fail("decoded sample count")
        })?;
        let tol = half_lsb(format).max(if float { 0.0 } else { 1e-6 });
        for (want, got) in oracle.iter().zip(decoded.iter()) {
            let want_im = if real { 0.0 } else { want.im };
            let err = libm::fabs(want.re - got.re).max(libm::fabs(want_im - got.im));
            ensure(err <= tol, || {
                fail("round-trip error above half an LSB")
                    .detail(format_args!("format {index}: {err} > {tol}"))
            })?;
        }

        // 4. Decoding and encoding again reproduces the bytes (formats of at most 24 bits).
        if format.bits() <= 24 {
            if real {
                let mut f = [0f32; N];
                ConvertReal::<f32>::convert_real(&Fallback, format, raw, &mut f)?;
                let n = ConvertOutReal::<f32>::convert_out_real(&Fallback, format, &f, again)?;
                ensure(again.get(..n) == Some(raw), || {
                    fail("decode then encode differs")
                })?;
            } else {
                Convert::<C32>::convert(&Fallback, format, raw, back32)?;
                let n = ConvertOut::<C32>::convert_out(&Fallback, format, back32, again)?;
                ensure(again.get(..n) == Some(raw), || {
                    fail("decode then encode differs")
                })?;
            }
        }
    }
    // Error cases.
    let tiny = b_fb.get_mut(..3).unwrap_or_default();
    ensure(
        matches!(
            ConvertOut::<C32>::convert_out(&Fallback, &formats::CS16, x32, tiny),
            Err(sdrk_core::Error::BufferTooSmall { .. })
        ),
        || Failure::new(CASE, "a small output buffer is not reported"),
    )?;
    ensure(
        matches!(
            ConvertOut::<C32>::convert_out(
                &Fallback,
                &formats::PACKED10,
                x32.get(..3).unwrap_or_default(),
                b_fb
            ),
            Err(sdrk_core::Error::InvalidParameter(_))
        ),
        || Failure::new(CASE, "an incomplete packing group is not reported"),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------------------
// fir_interpolate
// ---------------------------------------------------------------------------------------

const L: usize = 5;
const INTERP_TAPS: usize = 241;
const INTERP_IN: usize = 120;

fn check_timing<S: Timing>(
    case: &'static str,
    state: &S,
    l: usize,
    taps: usize,
) -> Result<(), Failure> {
    ensure(state.ratio() == Ratio::new(l as u64, 1)?, || {
        Failure::new(case, "wrong ratio")
    })?;
    let delay = Delay::new(taps as u64 - 1, 2 * l as u64)?;
    ensure(state.group_delay().reduced() == delay.reduced(), || {
        Failure::new(case, "wrong group delay")
    })
}

pub(super) fn fir_interpolate_c32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fir_interpolate_c32_vs_reference";
    let mut c = arena.carver();
    let taps = design::<f32>(&mut c, 15e3 / 240e3, INTERP_TAPS, L as f64)?;
    let input = c.take::<C32>(INTERP_IN)?;
    let (fb, rf, fac) = (
        c.take::<C32>(INTERP_IN * L)?,
        c.take::<C32>(INTERP_IN * L)?,
        c.take::<C32>(INTERP_IN * L)?,
    );
    let hist_n = INTERP_TAPS.div_ceil(L);
    let (h1, h2, h3) = (
        c.take::<C32>(hist_n)?,
        c.take::<C32>(hist_n)?,
        c.take::<C32>(hist_n)?,
    );
    let mut rng = Pcg32::new(802, 1);
    for s in input.iter_mut() {
        *s = random_c32(&mut rng, 0.5);
    }
    let mut a = FirInterpState::<C32, _, _>::new(taps, h1, L)?;
    let mut b = FirInterpState::<C32, _, _>::new(taps, h2, L)?;
    let mut d = FirInterpState::<C32, _, _>::new(taps, h3, L)?;
    let pa = FirInterpolate::<C32>::fir_interpolate(&Fallback, &mut a, input, fb)?;
    let pb = FirInterpolate::<C32>::fir_interpolate(&Reference, &mut b, input, rf)?;
    let pd = FirInterpolate::<C32>::fir_interpolate(&Sdrk, &mut d, input, fac)?;
    let want = Progress {
        consumed: INTERP_IN,
        produced: INTERP_IN * L,
    };
    ensure(pa == want && pb == want && pd == want, || {
        Failure::new(CASE, "every input sample must yield L outputs")
            .detail(format_args!("{pa:?} {pb:?} {pd:?}"))
    })?;
    min_snr(CASE, "fallback below 110 dB", flat(rf), flat(fb), 110.0)?;
    // The SIMD backend sums in a different order: it meets the contract of the kernel.
    min_snr(CASE, "facade below 110 dB", flat(rf), flat(fac), 110.0)?;
    ensure(
        a.history() == b.history() && a.history() == d.history(),
        || Failure::new(CASE, "histories differ"),
    )?;
    check_timing(CASE, &a, L, INTERP_TAPS)
}

pub(super) fn fir_interpolate_cq15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fir_interpolate_cq15_vs_reference";
    let mut c = arena.carver();
    // The Q15 taps carry the gain L as well: no polyphase branch sums above 1.
    let taps = design::<Q15>(&mut c, 15e3 / 240e3, INTERP_TAPS, L as f64)?;
    let input = c.take::<CQ15>(INTERP_IN)?;
    let (fb, rf, fac) = (
        c.take::<CQ15>(INTERP_IN * L)?,
        c.take::<CQ15>(INTERP_IN * L)?,
        c.take::<CQ15>(INTERP_IN * L)?,
    );
    let hist_n = INTERP_TAPS.div_ceil(L);
    let (h1, h2, h3) = (
        c.take::<CQ15>(hist_n)?,
        c.take::<CQ15>(hist_n)?,
        c.take::<CQ15>(hist_n)?,
    );
    let mut rng = Pcg32::new(803, 1);
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 0.9);
    }
    let mut a = FirInterpState::<CQ15, _, _>::new(taps, h1, L)?;
    let mut b = FirInterpState::<CQ15, _, _>::new(taps, h2, L)?;
    let mut d = FirInterpState::<CQ15, _, _>::new(taps, h3, L)?;
    FirInterpolate::<CQ15>::fir_interpolate(&Fallback, &mut a, input, fb)?;
    FirInterpolate::<CQ15>::fir_interpolate(&Reference, &mut b, input, rf)?;
    FirInterpolate::<CQ15>::fir_interpolate(&Sdrk, &mut d, input, fac)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(CASE, "fallback differs from the reference")
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs")
    })?;
    ensure(a.history() == b.history(), || {
        Failure::new(CASE, "histories differ")
    })?;
    check_timing(CASE, &a, L, INTERP_TAPS)
}

pub(super) fn fir_interpolate_real_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fir_interpolate_real_vs_reference";
    let hist_n = INTERP_TAPS.div_ceil(L);
    let mut rng = Pcg32::new(804, 1);
    {
        let mut c = arena.carver();
        let taps = design::<f32>(&mut c, 15e3 / 240e3, INTERP_TAPS, L as f64)?;
        let input = c.take::<f32>(INTERP_IN)?;
        let (fb, rf) = (c.take::<f32>(INTERP_IN * L)?, c.take::<f32>(INTERP_IN * L)?);
        let (h1, h2) = (c.take::<f32>(hist_n)?, c.take::<f32>(hist_n)?);
        for s in input.iter_mut() {
            *s = ((rng.next_f64() * 2.0 - 1.0) * 0.5) as f32;
        }
        let mut a = FirInterpRealState::<f32, _, _>::new(taps, h1, L)?;
        let mut b = FirInterpRealState::<f32, _, _>::new(taps, h2, L)?;
        FirInterpolateReal::<f32>::fir_interpolate_real(&Sdrk, &mut a, input, fb)?;
        FirInterpolateReal::<f32>::fir_interpolate_real(&Reference, &mut b, input, rf)?;
        min_snr(CASE, "f32 below 110 dB", rf, fb, 110.0)?;
        check_timing(CASE, &a, L, INTERP_TAPS)?;
    }
    let mut c = arena.carver();
    let taps = design::<Q15>(&mut c, 15e3 / 240e3, INTERP_TAPS, L as f64)?;
    let input = c.take::<Q15>(INTERP_IN)?;
    let (fb, rf) = (c.take::<Q15>(INTERP_IN * L)?, c.take::<Q15>(INTERP_IN * L)?);
    let (h1, h2) = (c.take::<Q15>(hist_n)?, c.take::<Q15>(hist_n)?);
    for s in input.iter_mut() {
        *s = q15_from_f64((rng.next_f64() * 2.0 - 1.0) * 0.9);
    }
    let mut a = FirInterpRealState::<Q15, _, _>::new(taps, h1, L)?;
    let mut b = FirInterpRealState::<Q15, _, _>::new(taps, h2, L)?;
    FirInterpolateReal::<Q15>::fir_interpolate_real(&Sdrk, &mut a, input, fb)?;
    FirInterpolateReal::<Q15>::fir_interpolate_real(&Reference, &mut b, input, rf)?;
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(CASE, "q15 differs from the reference")
    })
}

pub(super) fn fir_interpolate_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fir_interpolate_block_invariance";
    const IN: usize = 200;
    const TAPS: usize = 61;
    let hist_n = TAPS.div_ceil(L);
    let mut rng = Pcg32::new(805, 2);
    {
        let mut c = arena.carver();
        let taps = design::<f32>(&mut c, 0.05, TAPS, L as f64)?;
        let (input, whole, chunked) = (
            c.take::<C32>(IN)?,
            c.take::<C32>(IN * L)?,
            c.take::<C32>(IN * L)?,
        );
        let mut hist = hand_out(c.take::<C32>(hist_n)?, c.take::<C32>(hist_n)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 1.0);
        }
        let report = check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(806, 37),
            || FirInterpState::<C32, _, _>::new(taps, hist(), L),
            |state, i, o| match state {
                Ok(st) => FirInterpolate::<C32>::fir_interpolate(&Sdrk, st, i, o),
                Err(e) => Err(e.clone()),
            },
        )?;
        ensure(report.outputs == IN * L, || {
            Failure::new(CASE, "wrong output count")
        })?;
    }
    {
        let mut c = arena.carver();
        let taps = design::<Q15>(&mut c, 0.05, TAPS, L as f64)?;
        let (input, whole, chunked) = (
            c.take::<CQ15>(IN)?,
            c.take::<CQ15>(IN * L)?,
            c.take::<CQ15>(IN * L)?,
        );
        let mut hist = hand_out(c.take::<CQ15>(hist_n)?, c.take::<CQ15>(hist_n)?);
        for s in input.iter_mut() {
            *s = random_cq15(&mut rng, 1.0);
        }
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(807, 41),
            || FirInterpState::<CQ15, _, _>::new(taps, hist(), L),
            |state, i, o| match state {
                Ok(st) => FirInterpolate::<CQ15>::fir_interpolate(&Sdrk, st, i, o),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    let mut c = arena.carver();
    let taps = design::<f32>(&mut c, 0.05, TAPS, L as f64)?;
    let (input, whole, chunked) = (
        c.take::<f32>(IN)?,
        c.take::<f32>(IN * L)?,
        c.take::<f32>(IN * L)?,
    );
    let mut hist = hand_out(c.take::<f32>(hist_n)?, c.take::<f32>(hist_n)?);
    for s in input.iter_mut() {
        *s = (rng.next_f64() * 2.0 - 1.0) as f32;
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(808, 29),
        || FirInterpRealState::<f32, _, _>::new(taps, hist(), L),
        |state, i, o| match state {
            Ok(st) => FirInterpolateReal::<f32>::fir_interpolate_real(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------------------
// fm_mod
// ---------------------------------------------------------------------------------------

const FM_N: usize = 1024;

fn fm_state() -> Result<FmModState, Failure> {
    Ok(FmModState::new(75_000.0, Rate::hz(240_000)?)?)
}

pub(super) fn fm_mod_c32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fm_mod_c32_vs_reference";
    let mut c = arena.carver();
    let input = c.take::<f32>(FM_N)?;
    let (fb, rf, fac) = (
        c.take::<C32>(FM_N)?,
        c.take::<C32>(FM_N)?,
        c.take::<C32>(FM_N)?,
    );
    let mut rng = Pcg32::new(809, 1);
    for s in input.iter_mut() {
        *s = (rng.normal() * 0.3) as f32;
    }
    let (mut a, mut b, mut d) = (fm_state()?, fm_state()?, fm_state()?);
    FmMod::<C32>::fm_mod(&Fallback, &mut a, input, fb)?;
    FmMod::<C32>::fm_mod(&Reference, &mut b, input, rf)?;
    FmMod::<C32>::fm_mod(&Sdrk, &mut d, input, fac)?;
    min_snr(CASE, "fallback below 120 dB", flat(rf), flat(fb), 120.0)?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs")
    })?;
    ensure(a == b && a == d, || Failure::new(CASE, "phase differs"))?;
    // The phase integrates the input: the last phase is the sum of the rounded steps.
    let k = a.scale();
    let mut phase = 0u32;
    for x in input.iter() {
        phase = phase.wrapping_add((libm::floor(f64::from(*x) * k + 0.5) as i64) as u32);
    }
    ensure(a.phase() == phase, || {
        Failure::new(CASE, "phase is not the sum of the steps")
    })?;
    ensure(
        a.ratio() == Ratio::ONE && a.group_delay() == Delay::ZERO,
        || Failure::new(CASE, "wrong timing"),
    )
}

pub(super) fn fm_mod_cq15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fm_mod_cq15_vs_reference";
    let mut c = arena.carver();
    let input = c.take::<Q15>(FM_N)?;
    let (fb, rf, fac) = (
        c.take::<CQ15>(FM_N)?,
        c.take::<CQ15>(FM_N)?,
        c.take::<CQ15>(FM_N)?,
    );
    let mut rng = Pcg32::new(810, 1);
    for s in input.iter_mut() {
        *s = q15_from_f64(rng.normal() * 0.3);
    }
    let (mut a, mut b, mut d) = (fm_state()?, fm_state()?, fm_state()?);
    FmMod::<CQ15>::fm_mod(&Fallback, &mut a, input, fb)?;
    FmMod::<CQ15>::fm_mod(&Reference, &mut b, input, rf)?;
    FmMod::<CQ15>::fm_mod(&Sdrk, &mut d, input, fac)?;
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (r, f) in rf.iter().zip(fb.iter()) {
        sig += r.to_c64().norm_sqr();
        err += (r.to_c64() - f.to_c64()).norm_sqr();
    }
    let snr = if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(sig / err)
    };
    ensure(snr >= 80.0, || {
        Failure::new(CASE, "fallback below 80 dB").detail(format_args!("{snr} dB"))
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs")
    })?;
    ensure(a == b && a == d, || Failure::new(CASE, "phase differs"))
}

pub(super) fn fm_mod_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const IN: usize = 1000;
    let mut rng = Pcg32::new(811, 2);
    {
        let mut c = arena.carver();
        let (input, whole, chunked) = (c.take::<f32>(IN)?, c.take::<C32>(IN)?, c.take::<C32>(IN)?);
        for s in input.iter_mut() {
            *s = (rng.next_f64() * 2.0 - 1.0) as f32;
        }
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(812, 53),
            fm_state,
            |state, i, o| match state {
                Ok(st) => same_rate(i, o, |i, o| FmMod::<C32>::fm_mod(&Sdrk, st, i, o)),
                Err(e) => Err(sdrk_core::Error::Internal(e.msg)),
            },
        )?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<Q15>(IN)?, c.take::<CQ15>(IN)?, c.take::<CQ15>(IN)?);
    for s in input.iter_mut() {
        *s = q15_from_f64(rng.next_f64() * 2.0 - 1.0);
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(813, 47),
        fm_state,
        |state, i, o| match state {
            Ok(st) => same_rate(i, o, |i, o| FmMod::<CQ15>::fm_mod(&Sdrk, st, i, o)),
            Err(e) => Err(sdrk_core::Error::Internal(e.msg)),
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Fused kernels
// ---------------------------------------------------------------------------------------

/// A random sample of the type: both components uniform in `+-scale`.
fn random_sample<T: ComplexSample>(rng: &mut Pcg32, scale: f64) -> T {
    T::from_c64_round(Complex64::new(
        (rng.next_f64() * 2.0 - 1.0) * scale,
        (rng.next_f64() * 2.0 - 1.0) * scale,
    ))
}

fn real_taps<'s, R: RealSample>(
    c: &mut Carver<'s>,
    cutoff: f64,
    n: usize,
    gain: f64,
) -> Result<&'s [R], Failure> {
    design::<R>(c, cutoff, n, gain)
}

const DDC_IN: usize = 600;
const DDC_M: usize = 10;
const DDC_TAPS: usize = 31;

/// `ddc` through small output and scratch buffers and random chunk sizes equals the chain of
/// the single kernels run in one piece, bit for bit, including all states.
fn ddc_equals_chain_for<T>(
    arena: &mut Arena<'_>,
    case: &'static str,
    seed: u64,
) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: Rotate<T> + FirDecimate<T> + Ddc<T>,
{
    let mut c = arena.carver();
    let taps = real_taps::<T::Real>(&mut c, 0.04, DDC_TAPS, 1.0)?;
    let input = c.take::<T>(DDC_IN)?;
    let rotated = c.take::<T>(DDC_IN)?;
    let want = c.take::<T>(DDC_IN / DDC_M + 2)?;
    let got = c.take::<T>(DDC_IN / DDC_M + 2)?;
    let (small_out, small_scratch) = (c.take::<T>(7)?, c.take::<T>(40)?);
    let (h1, h2) = (c.take::<T>(DDC_TAPS - 1)?, c.take::<T>(DDC_TAPS - 1)?);
    let mut rng = Pcg32::new(seed, 3);
    for s in input.iter_mut() {
        *s = random_sample::<T>(&mut rng, 0.8);
    }
    let rate = Rate::hz(2_400_000)?;
    let mut nco_a = Nco::new(-600_000.0, rate)?;
    let mut nco_b = nco_a;
    let mut fir_a = FirDecimState::<T, _, _>::new(taps, h1, DDC_M)?;
    let mut fir_b = FirDecimState::<T, _, _>::new(taps, h2, DDC_M)?;
    // The chain of the single kernels in one piece.
    Rotate::<T>::rotate(&Sdrk, &mut nco_a, input, rotated)?;
    let pw = FirDecimate::<T>::fir_decimate(&Sdrk, &mut fir_a, rotated, want)?;
    // The fused kernel in random chunks with small buffers.
    let (mut produced, mut pos) = (0usize, 0usize);
    for size in ChunkPlan::new(seed + 1, 97).sizes() {
        if pos >= DDC_IN {
            break;
        }
        let end = (pos + size).min(DDC_IN);
        let mut rest = input.get(pos..end).unwrap_or_default();
        while !rest.is_empty() {
            let p = Ddc::<T>::ddc(
                &Sdrk,
                &mut nco_b,
                &mut fir_b,
                rest,
                small_out,
                small_scratch,
            )?;
            if p.consumed == 0 && p.produced == 0 {
                return Err(Failure::new(case, "ddc made no progress"));
            }
            let room = got.get_mut(produced..).unwrap_or_default();
            let part = small_out.get(..p.produced).unwrap_or_default();
            room.get_mut(..p.produced)
                .ok_or_else(|| Failure::new(case, "output overflow"))?
                .copy_from_slice(part);
            produced += p.produced;
            rest = rest.get(p.consumed..).unwrap_or_default();
        }
        pos = end;
    }
    ensure(produced == pw.produced, || {
        Failure::new(case, "output count differs")
            .detail(format_args!("{produced} vs {}", pw.produced))
    })?;
    ensure(
        bit_exact(
            want.get(..produced).unwrap_or_default(),
            got.get(..produced).unwrap_or_default(),
        )
        .is_none(),
        || Failure::new(case, "fused output differs from the chain"),
    )?;
    ensure(nco_a == nco_b, || {
        Failure::new(case, "oscillator phase differs")
    })?;
    ensure(
        fir_a.history() == fir_b.history() && fir_a.countdown() == fir_b.countdown(),
        || Failure::new(case, "decimator state differs"),
    )
}

pub(super) fn ddc_equals_chain(arena: &mut Arena<'_>) -> Result<(), Failure> {
    ddc_equals_chain_for::<C32>(arena, "ddc_equals_chain", 820)?;
    ddc_equals_chain_for::<CQ15>(arena, "ddc_equals_chain", 821)
}

const DUC_IN: usize = 60;
const DUC_L: usize = 10;
const DUC_TAPS: usize = 61;

fn duc_equals_chain_for<T>(
    arena: &mut Arena<'_>,
    case: &'static str,
    seed: u64,
) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: Rotate<T> + FirInterpolate<T> + Duc<T>,
{
    let hist_n = DUC_TAPS.div_ceil(DUC_L);
    let mut c = arena.carver();
    let taps = real_taps::<T::Real>(&mut c, 0.04, DUC_TAPS, DUC_L as f64)?;
    let input = c.take::<T>(DUC_IN)?;
    let upsampled = c.take::<T>(DUC_IN * DUC_L)?;
    let want = c.take::<T>(DUC_IN * DUC_L)?;
    let got = c.take::<T>(DUC_IN * DUC_L)?;
    let (small_out, small_scratch) = (c.take::<T>(25)?, c.take::<T>(33)?);
    let (h1, h2) = (c.take::<T>(hist_n)?, c.take::<T>(hist_n)?);
    let mut rng = Pcg32::new(seed, 4);
    for s in input.iter_mut() {
        *s = random_sample::<T>(&mut rng, 0.8);
    }
    let rate = Rate::hz(2_400_000)?;
    let mut nco_a = Nco::new(600_000.0, rate)?;
    let mut nco_b = nco_a;
    let mut fir_a = FirInterpState::<T, _, _>::new(taps, h1, DUC_L)?;
    let mut fir_b = FirInterpState::<T, _, _>::new(taps, h2, DUC_L)?;
    let pi = FirInterpolate::<T>::fir_interpolate(&Sdrk, &mut fir_a, input, upsampled)?;
    Rotate::<T>::rotate(&Sdrk, &mut nco_a, upsampled, want)?;
    let (mut produced, mut pos) = (0usize, 0usize);
    for size in ChunkPlan::new(seed + 1, 17).sizes() {
        if pos >= DUC_IN {
            break;
        }
        let end = (pos + size).min(DUC_IN);
        let mut rest = input.get(pos..end).unwrap_or_default();
        while !rest.is_empty() {
            let p = Duc::<T>::duc(
                &Sdrk,
                &mut fir_b,
                &mut nco_b,
                rest,
                small_out,
                small_scratch,
            )?;
            if p.consumed == 0 {
                return Err(Failure::new(case, "duc made no progress"));
            }
            ensure(p.produced == p.consumed * DUC_L, || {
                Failure::new(case, "duc must produce L outputs per input sample")
            })?;
            got.get_mut(produced..produced + p.produced)
                .ok_or_else(|| Failure::new(case, "output overflow"))?
                .copy_from_slice(small_out.get(..p.produced).unwrap_or_default());
            produced += p.produced;
            rest = rest.get(p.consumed..).unwrap_or_default();
        }
        pos = end;
    }
    ensure(produced == pi.produced, || {
        Failure::new(case, "output count differs")
    })?;
    ensure(bit_exact(want, got).is_none(), || {
        Failure::new(case, "fused output differs from the chain")
    })?;
    ensure(nco_a == nco_b, || {
        Failure::new(case, "oscillator phase differs")
    })?;
    ensure(fir_a.history() == fir_b.history(), || {
        Failure::new(case, "interpolator state differs")
    })
}

pub(super) fn duc_equals_chain(arena: &mut Arena<'_>) -> Result<(), Failure> {
    duc_equals_chain_for::<C32>(arena, "duc_equals_chain", 830)?;
    duc_equals_chain_for::<CQ15>(arena, "duc_equals_chain", 831)
}

fn fm_demod_equals_chain_for<T>(
    arena: &mut Arena<'_>,
    case: &'static str,
    seed: u64,
    gain: T::Real,
) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: QuadDemod<T> + Deemphasis<T::Real> + FmDemod<T>,
{
    const N: usize = 600;
    let rate = Rate::hz(240_000)?;
    let mut c = arena.carver();
    let input = c.take::<T>(N)?;
    let (tmp, want, got) = (
        c.take::<T::Real>(N)?,
        c.take::<T::Real>(N)?,
        c.take::<T::Real>(N)?,
    );
    let scratch = c.take::<T::Real>(64)?;
    let mut rng = Pcg32::new(seed, 5);
    // A frequency-modulated signal with a little noise.
    let mut phase = 0.0f64;
    for s in input.iter_mut() {
        phase += 0.6 * (rng.next_f64() * 2.0 - 1.0);
        let n = Complex64::new(rng.normal() * 0.01, rng.normal() * 0.01);
        *s = T::from_c64_round(Complex64::new(libm::cos(phase) * 0.8, libm::sin(phase) * 0.8) + n);
    }
    let (mut qa, mut qb) = (QuadDemodState::<T>::new(), QuadDemodState::<T>::new());
    let mut da = DeemphasisState::<T::Real>::new(50e-6, rate)?;
    let mut db = da;
    QuadDemod::<T>::quad_demod(&Sdrk, &mut qa, input, tmp, gain)?;
    Deemphasis::<T::Real>::deemphasis(&Sdrk, &mut da, tmp, want)?;
    let mut pos = 0usize;
    for size in ChunkPlan::new(seed + 1, 150).sizes() {
        if pos >= N {
            break;
        }
        let end = (pos + size).min(N);
        FmDemod::<T>::fm_demod(
            &Sdrk,
            &mut qb,
            &mut db,
            input.get(pos..end).unwrap_or_default(),
            got.get_mut(pos..end).unwrap_or_default(),
            scratch,
            gain,
        )?;
        pos = end;
    }
    ensure(bit_exact(want, got).is_none(), || {
        Failure::new(case, "fused output differs from the chain")
    })?;
    ensure(qa == qb && da == db, || Failure::new(case, "states differ"))
}

pub(super) fn fm_demod_equals_chain(arena: &mut Arena<'_>) -> Result<(), Failure> {
    fm_demod_equals_chain_for::<C32>(arena, "fm_demod_equals_chain", 840, 2.0)?;
    fm_demod_equals_chain_for::<CQ15>(arena, "fm_demod_equals_chain", 841, Q15(32767))
}

fn partial_output_for<T>(
    arena: &mut Arena<'_>,
    case: &'static str,
    seed: u64,
) -> Result<(), Failure>
where
    T: ComplexSample,
    Sdrk: Ddc<T>,
{
    const N: usize = 500;
    const M: usize = 7;
    let mut c = arena.carver();
    let taps = real_taps::<T::Real>(&mut c, 0.05, 21, 1.0)?;
    let input = c.take::<T>(N)?;
    let (whole, parts) = (c.take::<T>(N / M + 2)?, c.take::<T>(N / M + 2)?);
    let scratch = c.take::<T>(N)?;
    let (small_out, small_scratch) = (c.take::<T>(3)?, c.take::<T>(50)?);
    let (h1, h2) = (c.take::<T>(20)?, c.take::<T>(20)?);
    let mut rng = Pcg32::new(seed, 6);
    for s in input.iter_mut() {
        *s = random_sample::<T>(&mut rng, 0.8);
    }
    let rate = Rate::hz(1_000_000)?;
    let mut nco_w = Nco::new(123_456.0, rate)?;
    let mut fir_w = FirDecimState::<T, _, _>::new(taps, h1, M)?;
    let pw = Ddc::<T>::ddc(&Sdrk, &mut nco_w, &mut fir_w, input, whole, scratch)?;
    ensure(pw.consumed == N, || {
        Failure::new(case, "one call must consume everything")
    })?;
    let mut nco = Nco::new(123_456.0, rate)?;
    let inc = nco.increment();
    let mut fir = FirDecimState::<T, _, _>::new(taps, h2, M)?;
    let (mut pos, mut produced) = (0usize, 0usize);
    while pos < N {
        let before = fir.max_input_for_output(small_out.len());
        let p = Ddc::<T>::ddc(
            &Sdrk,
            &mut nco,
            &mut fir,
            input.get(pos..).unwrap_or_default(),
            small_out,
            small_scratch,
        )?;
        ensure(p.consumed > 0, || Failure::new(case, "no input consumed"))?;
        ensure(p.consumed <= before.min(small_scratch.len()), || {
            Failure::new(case, "consumed more than the output space allows")
        })?;
        pos += p.consumed;
        // The oscillator advanced by exactly the consumed samples.
        let expected_phase = (pos as u32).wrapping_mul(inc);
        ensure(nco.phase() == expected_phase, || {
            Failure::new(
                case,
                "oscillator phase is not input position times increment",
            )
            .detail(format_args!("position {pos}"))
        })?;
        parts
            .get_mut(produced..produced + p.produced)
            .ok_or_else(|| Failure::new(case, "output overflow"))?
            .copy_from_slice(small_out.get(..p.produced).unwrap_or_default());
        produced += p.produced;
    }
    ensure(produced == pw.produced, || {
        Failure::new(case, "output count differs")
    })?;
    ensure(
        bit_exact(
            whole.get(..produced).unwrap_or_default(),
            parts.get(..produced).unwrap_or_default(),
        )
        .is_none(),
        || Failure::new(case, "output depends on the output buffer size"),
    )?;
    ensure(nco == nco_w, || Failure::new(case, "final phase differs"))
}

pub(super) fn ddc_partial_output_keeps_phase(arena: &mut Arena<'_>) -> Result<(), Failure> {
    partial_output_for::<C32>(arena, "ddc_partial_output_keeps_phase", 850)?;
    partial_output_for::<CQ15>(arena, "ddc_partial_output_keeps_phase", 851)
}

// ---------------------------------------------------------------------------------------
// End-to-end chains
// ---------------------------------------------------------------------------------------

/// Thresholds of the end-to-end cases in dB (see `docs/kernels.md`).
const RX_MIN_DB: [f64; 2] = [40.0, 35.0];
const TXRX_MIN_DB: [f64; 2] = [35.0, 30.0];
/// Largest allowed deviation of the measured audio frequency from 1 kHz.
const TONE_TOLERANCE_HZ: f64 = 0.1;

fn check_measurement(
    case: &'static str,
    m: &e2e::Measurement,
    samples: usize,
    min_db: f64,
) -> Result<(), Failure> {
    ensure(m.audio_samples == samples / 50, || {
        Failure::new(case, "wrong audio sample count").detail(format_args!("{}", m.audio_samples))
    })?;
    ensure(m.sinad_db >= min_db, || {
        Failure::new(case, "SINAD below the threshold")
            .detail(format_args!("{} dB, need {min_db}", m.sinad_db))
    })?;
    ensure(
        libm::fabs(m.tone_hz - e2e::AUDIO_HZ) <= TONE_TOLERANCE_HZ,
        || Failure::new(case, "audio frequency off").detail(format_args!("{} Hz", m.tone_hz)),
    )?;
    ensure(m.amplitude > 0.3, || {
        Failure::new(case, "audio amplitude too low").detail(format_args!("{}", m.amplitude))
    })
}

fn rx_case<T: E2eSample>(
    case: &'static str,
    arena: &mut Arena<'_>,
    samples: usize,
    min_db: f64,
) -> Result<(), Failure>
where
    Sdrk: RxKernels<T>,
{
    let m = e2e::measure_rx::<T, Sdrk>(&Sdrk, arena, samples)?;
    check_measurement(case, &m, samples, min_db)
}

fn txrx_case<T: E2eSample>(
    case: &'static str,
    arena: &mut Arena<'_>,
    samples: usize,
    min_db: f64,
) -> Result<(), Failure>
where
    Sdrk: TxKernels<T>,
{
    let m = e2e::measure_txrx::<T, Sdrk>(&Sdrk, arena, samples)?;
    check_measurement(case, &m, samples, min_db)
}

/// 0.1 s of signal at 2.4 MHz.
const QUICK_SAMPLES: usize = 240_000;
/// 1.0 s of signal at 2.4 MHz.
const FULL_SAMPLES: usize = 2_400_000;

pub(super) fn e2e_fm_rx_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    rx_case::<C32>("e2e_fm_rx_f32", arena, QUICK_SAMPLES, RX_MIN_DB[0])
}

pub(super) fn e2e_fm_rx_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    rx_case::<CQ15>("e2e_fm_rx_q15", arena, QUICK_SAMPLES, RX_MIN_DB[1])
}

pub(super) fn e2e_fm_txrx_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    txrx_case::<C32>("e2e_fm_txrx_f32", arena, QUICK_SAMPLES, TXRX_MIN_DB[0])
}

pub(super) fn e2e_fm_txrx_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    txrx_case::<CQ15>("e2e_fm_txrx_q15", arena, QUICK_SAMPLES, TXRX_MIN_DB[1])
}

pub(super) fn e2e_fm_rx_f32_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    rx_case::<C32>("e2e_fm_rx_f32_1s", arena, FULL_SAMPLES, RX_MIN_DB[0])
}

pub(super) fn e2e_fm_rx_q15_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    rx_case::<CQ15>("e2e_fm_rx_q15_1s", arena, FULL_SAMPLES, RX_MIN_DB[1])
}

pub(super) fn e2e_fm_txrx_f32_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    txrx_case::<C32>("e2e_fm_txrx_f32_1s", arena, FULL_SAMPLES, TXRX_MIN_DB[0])
}

pub(super) fn e2e_fm_txrx_q15_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    txrx_case::<CQ15>("e2e_fm_txrx_q15_1s", arena, FULL_SAMPLES, TXRX_MIN_DB[1])
}

// ---------------------------------------------------------------------------------------
// Extreme inputs
// ---------------------------------------------------------------------------------------

pub(super) fn extremes_tx_kernels(arena: &mut Arena<'_>) -> Result<(), Failure> {
    use crate::r#gen::extreme;
    const CASE: &str = "extremes_tx_kernels";
    const N: usize = 140;
    let fail = |msg: &'static str| Failure::new(CASE, msg);

    // convert_out: NaN, infinities, denormals and the largest floats in every format.
    {
        let mut c = arena.carver();
        let x = c.take::<C32>(N)?;
        let q = c.take::<CQ15>(N)?;
        let (b_fb, b_rf, b_fac) = (
            c.take::<u8>(N * 8)?,
            c.take::<u8>(N * 8)?,
            c.take::<u8>(N * 8)?,
        );
        extreme::f32_specials_complex(x);
        extreme::full_scale_alternating(q);
        for (index, f) in FORMATS.iter().enumerate() {
            let detail = |msg: &'static str| fail(msg).detail(format_args!("format {index}"));
            if f.layout() == Layout::Real {
                let mut re = [0f32; N];
                for (r, s) in re.iter_mut().zip(x.iter()) {
                    *r = s.re;
                }
                let n = ConvertOutReal::<f32>::convert_out_real(&Fallback, f, &re, b_fb)?;
                ConvertOutReal::<f32>::convert_out_real(&Reference, f, &re, b_rf)?;
                ConvertOutReal::<f32>::convert_out_real(&Sdrk, f, &re, b_fac)?;
                ensure(
                    b_fb.get(..n) == b_rf.get(..n) && b_fb.get(..n) == b_fac.get(..n),
                    || detail("f32 specials: implementations differ"),
                )?;
            } else {
                let n = ConvertOut::<C32>::convert_out(&Fallback, f, x, b_fb)?;
                ConvertOut::<C32>::convert_out(&Reference, f, x, b_rf)?;
                ConvertOut::<C32>::convert_out(&Sdrk, f, x, b_fac)?;
                ensure(
                    b_fb.get(..n) == b_rf.get(..n) && b_fb.get(..n) == b_fac.get(..n),
                    || detail("c32 specials: implementations differ"),
                )?;
                let n = ConvertOut::<CQ15>::convert_out(&Fallback, f, q, b_fb)?;
                ConvertOut::<CQ15>::convert_out(&Reference, f, q, b_rf)?;
                ensure(b_fb.get(..n) == b_rf.get(..n), || {
                    detail("full scale: implementations differ")
                })?;
            }
        }
    }

    // fir_interpolate: silence stays silent, full scale matches the reference, NaN does not panic.
    {
        const TAPS: usize = 25;
        const LL: usize = 4;
        let mut c = arena.carver();
        let taps_q = design::<Q15>(&mut c, 0.1, TAPS, LL as f64)?;
        let taps_f = design::<f32>(&mut c, 0.1, TAPS, LL as f64)?;
        let input_q = c.take::<CQ15>(N)?;
        let input_f = c.take::<C32>(N)?;
        let (fb_q, rf_q) = (c.take::<CQ15>(N * LL)?, c.take::<CQ15>(N * LL)?);
        let (fb_f, fac_f) = (c.take::<C32>(N * LL)?, c.take::<C32>(N * LL)?);
        let hist_n = TAPS.div_ceil(LL);
        let (h1, h2, h5) = (
            c.take::<CQ15>(hist_n)?,
            c.take::<CQ15>(hist_n)?,
            c.take::<CQ15>(hist_n)?,
        );
        let (h3, h4) = (c.take::<C32>(hist_n)?, c.take::<C32>(hist_n)?);
        extreme::silence(input_q);
        let mut a = FirInterpState::<CQ15, _, _>::new(taps_q, h1, LL)?;
        FirInterpolate::<CQ15>::fir_interpolate(&Sdrk, &mut a, input_q, fb_q)?;
        ensure(fb_q.iter().all(|s| *s == CQ15::default()), || {
            fail("silence in, noise out (cq15)")
        })?;
        extreme::full_scale_alternating(input_q);
        let mut a = FirInterpState::<CQ15, _, _>::new(taps_q, h5, LL)?;
        let mut b = FirInterpState::<CQ15, _, _>::new(taps_q, h2, LL)?;
        FirInterpolate::<CQ15>::fir_interpolate(&Sdrk, &mut a, input_q, fb_q)?;
        FirInterpolate::<CQ15>::fir_interpolate(&Reference, &mut b, input_q, rf_q)?;
        ensure(bit_exact(fb_q, rf_q).is_none(), || {
            fail("full scale: fallback differs from the reference (cq15)")
        })?;
        extreme::f32_specials_complex(input_f);
        let mut a = FirInterpState::<C32, _, _>::new(taps_f, h3, LL)?;
        let mut b = FirInterpState::<C32, _, _>::new(taps_f, h4, LL)?;
        FirInterpolate::<C32>::fir_interpolate(&Fallback, &mut a, input_f, fb_f)?;
        FirInterpolate::<C32>::fir_interpolate(&Sdrk, &mut b, input_f, fac_f)?;
        // Sums of non-finite and huge values depend on the order of the additions; require the
        // same classification and, where the fallback result is moderate, the same value.
        let agree = |w: f32, g: f32| {
            !w.is_finite() || w.abs() > 1e30 || (w - g).abs() <= 1e-4 * (1.0 + w.abs())
        };
        ensure(
            fb_f.iter()
                .zip(fac_f.iter())
                .all(|(w, g)| agree(w.re, g.re) && agree(w.im, g.im)),
            || fail("specials: facade differs (c32)"),
        )?;
    }

    // fm_mod: any f32 input leaves a finite, bounded output and equal phases.
    {
        let mut c = arena.carver();
        let input = c.take::<f32>(N)?;
        let (fb, rf) = (c.take::<C32>(N)?, c.take::<C32>(N)?);
        extreme::f32_specials(input);
        let (mut a, mut b) = (fm_state()?, fm_state()?);
        FmMod::<C32>::fm_mod(&Fallback, &mut a, input, fb)?;
        FmMod::<C32>::fm_mod(&Reference, &mut b, input, rf)?;
        ensure(a == b, || fail("specials: phases differ"))?;
        ensure(
            fb.iter().all(|s| {
                s.re.is_finite() && s.im.is_finite() && s.re.abs() <= 1.0001 && s.im.abs() <= 1.0001
            }),
            || fail("fm_mod output left the unit circle"),
        )?;
        let q = c.take::<Q15>(N)?;
        let out = c.take::<CQ15>(N)?;
        for (i, s) in q.iter_mut().enumerate() {
            *s = match i % 3 {
                0 => Q15::MIN,
                1 => Q15::MAX,
                _ => Q15(0),
            };
        }
        FmMod::<CQ15>::fm_mod(&Sdrk, &mut fm_state()?, q, out)?;
        ensure(
            out.iter().all(|s| {
                let (re, im) = (i64::from(s.re.0), i64::from(s.im.0));
                (re * re + im * im - 32767 * 32767).abs() < 4 * 32768
            }),
            || fail("cq15 fm_mod output is off the unit circle"),
        )?;
    }

    // Fused kernels on full-scale and silent input equal the chains (tested for random data in
    // the dedicated cases) and do not panic on NaN.
    {
        let mut c = arena.carver();
        let taps = design::<f32>(&mut c, 0.05, 21, 1.0)?;
        let input = c.take::<C32>(N)?;
        let (out, scratch) = (c.take::<C32>(N)?, c.take::<C32>(N)?);
        let hist = c.take::<C32>(20)?;
        extreme::f32_specials_complex(input);
        let mut fir = FirDecimState::<C32, _, _>::new(taps, hist, 4)?;
        let mut nco = Nco::new(1000.0, Rate::hz(48_000)?)?;
        let p = Ddc::<C32>::ddc(&Sdrk, &mut nco, &mut fir, input, out, scratch)?;
        ensure(p.consumed == N, || fail("ddc must consume the whole block"))?;
        let (audio, tmp) = (c.take::<f32>(N)?, c.take::<f32>(N)?);
        let mut quad = QuadDemodState::<C32>::new();
        let mut deemph = DeemphasisState::<f32>::new(50e-6, Rate::hz(240_000)?)?;
        FmDemod::<C32>::fm_demod(&Sdrk, &mut quad, &mut deemph, input, audio, tmp, 1.0)?;
    }
    Ok(())
}
