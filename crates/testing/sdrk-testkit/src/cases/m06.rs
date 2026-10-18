//! Suite cases of the receive kernels: `fir_decimate`, `resample`, `quad_demod`,
//! `deemphasis`, `dc_block`, `window`, `mag_sq` and `power_db`.
//!
//! Every case compares `Fallback` (and the facade) with `Reference`; no files are
//! read, so the cases run on microcontrollers too.

#![allow(clippy::many_single_char_names, clippy::too_many_lines)]

use core::f64::consts::PI;

use sdrk::Sdrk;
use sdrk_core::kernels::decim::{
    FirDecimState, FirDecimate, Resample, ResampleReal, ResampleRealState, ResampleState,
};
use sdrk_core::kernels::demod::{
    DcBlock, DcBlockState, Deemphasis, DeemphasisState, QuadDemod, QuadDemodState,
};
use sdrk_core::kernels::spectrum::{MagSq, PowerDb, Window};
use sdrk_core::num::{q15_from_f64, q15_mul};
use sdrk_core::window::{Symmetry, blackman_harris, hann, kaiser, window_gain};
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Rate, Reference, Sample, Timing};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use crate::arena::Arena;
use crate::r#gen;
use crate::invariance::{ChunkPlan, check_block_invariance};
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

/// Hamming-windowed sinc low-pass with `n` taps and cutoff `fc` (cycles per sample), DC gain `gain`.
fn lowpass(n: usize, fc: f64, gain: f64, mut put: impl FnMut(usize, f64)) {
    let mid = (n.max(1) - 1) as f64 / 2.0;
    let tap = |k: usize| {
        let x = k as f64 - mid;
        let sinc = if x == 0.0 {
            2.0 * fc
        } else {
            libm::sin(2.0 * PI * fc * x) / (PI * x)
        };
        let window = if n > 1 {
            0.54 - 0.46 * libm::cos(2.0 * PI * k as f64 / (n - 1) as f64)
        } else {
            1.0
        };
        sinc * window
    };
    let sum: f64 = (0..n).map(tap).sum();
    for k in 0..n {
        put(k, gain * tap(k) / sum);
    }
}

fn taps_f32(out: &mut [f32], fc: f64, gain: f64) {
    let n = out.len();
    lowpass(n, fc, gain, |k, v| {
        if let Some(t) = out.get_mut(k) {
            *t = v as f32;
        }
    });
}

fn taps_q15(out: &mut [Q15], fc: f64, gain: f64) {
    let n = out.len();
    lowpass(n, fc, gain, |k, v| {
        if let Some(t) = out.get_mut(k) {
            *t = q15_from_f64(v);
        }
    });
}

/// Runs a same-rate kernel on as many samples as both buffers hold.
fn same_rate<I: Copy, O>(
    input: &[I],
    output: &mut [O],
    kernel: impl FnOnce(&[I], &mut [O]) -> sdrk_core::Result<()>,
) -> sdrk_core::Result<Progress> {
    let n = input.len().min(output.len());
    let internal = || sdrk_core::Error::Internal("slice split");
    let (i, _) = input.split_at_checked(n).ok_or_else(internal)?;
    let (o, _) = output.split_at_mut_checked(n).ok_or_else(internal)?;
    kernel(i, o)?;
    Ok(Progress {
        consumed: n,
        produced: n,
    })
}

/// Hands out two history buffers one after the other (and empty slices afterwards), so that
/// a `make_state` closure can build two states over caller-provided storage.
fn hand_out<'a, T>(first: &'a mut [T], second: &'a mut [T]) -> impl FnMut() -> &'a mut [T] {
    let mut slots = [Some(first), Some(second)];
    let mut next = 0usize;
    move || {
        let slot = slots
            .get_mut(next)
            .and_then(Option::take)
            .unwrap_or_default();
        next += 1;
        slot
    }
}

fn flat(x: &[C32]) -> &[f32] {
    bytemuck::try_cast_slice(x).unwrap_or_default()
}

/// `10 log10(sum want^2 / sum (want - got)^2)`; `+inf` for identical signals.
fn snr_db(want: &[f32], got: &[f32]) -> f64 {
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (&w, &g) in want.iter().zip(got) {
        let (w, g) = (f64::from(w), f64::from(g));
        sig += w * w;
        err += (w - g) * (w - g);
    }
    if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(sig / err)
    }
}

fn min_snr(
    case: &'static str,
    msg: &'static str,
    want: &[f32],
    got: &[f32],
    min: f64,
) -> Result<(), Failure> {
    let snr = snr_db(want, got);
    ensure(snr >= min, || {
        Failure::new(case, msg).detail(format_args!("{snr} dB, need {min}"))
    })
}

/// Output room that suffices for `n` input samples at `up / down`.
const fn room(n: usize, up: usize, down: usize) -> usize {
    n * up / down + up + 2
}

/// Number of outputs for `n` input samples after a reset: `floor((n L - 1) / M) + 1`.
const fn out_count(n: usize, up: usize, down: usize) -> usize {
    (n * up - 1) / down + 1
}

/// `(L, M, taps)` of the resampler cases.
const RATIOS: [(usize, usize, usize); 3] = [(1, 5, 61), (3, 2, 48), (147, 160, 294)];

fn ratio_cutoff(up: usize, down: usize) -> f64 {
    0.45 / up.max(down) as f64
}

// ------------------------------------------------------------- fir_decimate

pub(super) fn fir_decimate_c32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fir_decimate_c32_vs_reference";
    const N: usize = 1024;
    let mut rng = Pcg32::new(601, 1);
    for (taps_n, m) in [(63usize, 10usize), (31, 3), (9, 1)] {
        let mut c = arena.carver();
        let input = c.take::<C32>(N)?;
        let (fb, rf, fac) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
        let taps = c.take::<f32>(taps_n)?;
        let (h1, h2, h3) = (
            c.take::<C32>(taps_n - 1)?,
            c.take::<C32>(taps_n - 1)?,
            c.take::<C32>(taps_n - 1)?,
        );
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 0.9);
        }
        taps_f32(taps, 0.4 / m as f64, 1.0);
        let mut a = FirDecimState::<C32, _, _>::new(&*taps, h1, m)?;
        let mut b = FirDecimState::<C32, _, _>::new(&*taps, h2, m)?;
        let mut d = FirDecimState::<C32, _, _>::new(&*taps, h3, m)?;
        let pa = FirDecimate::<C32>::fir_decimate(&Fallback, &mut a, input, fb)?;
        let pb = FirDecimate::<C32>::fir_decimate(&Reference, &mut b, input, rf)?;
        let pd = FirDecimate::<C32>::fir_decimate(&Sdrk, &mut d, input, fac)?;
        ensure(pa == pb && pa == pd, || {
            Failure::new(CASE, "progress differs").detail(format_args!("M={m}"))
        })?;
        ensure(pa.consumed == N && pa.produced == N.div_ceil(m), || {
            Failure::new(CASE, "wrong output count").detail(format_args!("M={m}: {pa:?}"))
        })?;
        let n = pa.produced;
        min_snr(
            CASE,
            "fallback below 110 dB",
            flat(rf.get(..n).unwrap_or_default()),
            flat(fb.get(..n).unwrap_or_default()),
            110.0,
        )?;
        ensure(
            bit_exact(
                fb.get(..n).unwrap_or_default(),
                fac.get(..n).unwrap_or_default(),
            )
            .is_none(),
            || Failure::new(CASE, "facade differs"),
        )?;
        ensure(
            a.history() == b.history()
                && a.history() == d.history()
                && a.countdown() == b.countdown(),
            || Failure::new(CASE, "states differ"),
        )?;
    }
    Ok(())
}

pub(super) fn fir_decimate_cq15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "fir_decimate_cq15_vs_reference";
    const N: usize = 1024;
    const TAPS: usize = 33;
    let mut rng = Pcg32::new(602, 1);
    for extremes in [false, true] {
        for m in [1usize, 4, 10] {
            let mut c = arena.carver();
            let input = c.take::<CQ15>(N)?;
            let (fb, rf, fac) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
            let taps = c.take::<Q15>(TAPS)?;
            let (h1, h2, h3) = (
                c.take::<CQ15>(TAPS - 1)?,
                c.take::<CQ15>(TAPS - 1)?,
                c.take::<CQ15>(TAPS - 1)?,
            );
            if extremes {
                r#gen::extreme::full_scale_alternating(input);
            } else {
                for s in input.iter_mut() {
                    *s = random_cq15(&mut rng, 1.0);
                }
            }
            taps_q15(taps, 0.4 / m as f64, 1.0);
            let mut a = FirDecimState::<CQ15, _, _>::new(&*taps, h1, m)?;
            let mut b = FirDecimState::<CQ15, _, _>::new(&*taps, h2, m)?;
            let mut d = FirDecimState::<CQ15, _, _>::new(&*taps, h3, m)?;
            let pa = FirDecimate::<CQ15>::fir_decimate(&Fallback, &mut a, input, fb)?;
            let pb = FirDecimate::<CQ15>::fir_decimate(&Reference, &mut b, input, rf)?;
            let pd = FirDecimate::<CQ15>::fir_decimate(&Sdrk, &mut d, input, fac)?;
            ensure(pa == pb && pa == pd && pa.produced == N.div_ceil(m), || {
                Failure::new(CASE, "progress differs").detail(format_args!("M={m}"))
            })?;
            let n = pa.produced;
            let (fb, rf, fac) = (
                fb.get(..n).unwrap_or_default(),
                rf.get(..n).unwrap_or_default(),
                fac.get(..n).unwrap_or_default(),
            );
            ensure(bit_exact(fb, rf).is_none(), || {
                Failure::new(CASE, "not bit-exact to the reference").detail(format_args!("M={m}"))
            })?;
            ensure(bit_exact(fb, fac).is_none(), || {
                Failure::new(CASE, "facade differs")
            })?;
            ensure(
                a.history() == b.history() && a.countdown() == b.countdown(),
                || Failure::new(CASE, "states differ"),
            )?;
        }
    }
    Ok(())
}

pub(super) fn fir_decimate_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const N: usize = 1024;
    const TAPS: usize = 21;
    const M: usize = 7;
    let mut rng = Pcg32::new(603, 2);
    {
        let mut c = arena.carver();
        let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
        let taps = c.take::<f32>(TAPS)?;
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 1.0);
        }
        taps_f32(taps, 0.1, 1.0);
        let taps: &[f32] = taps;
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(631, 53),
            || FirDecimState::<C32, _, _>::new(taps, [C32::ZERO; TAPS - 1], M),
            |state, i, o| match state {
                Ok(st) => FirDecimate::<C32>::fir_decimate(&Sdrk, st, i, o),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    let taps = c.take::<Q15>(TAPS)?;
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 1.0);
    }
    taps_q15(taps, 0.1, 1.0);
    let taps: &[Q15] = taps;
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(632, 47),
        || FirDecimState::<CQ15, _, _>::new(taps, [CQ15::ZERO; TAPS - 1], M),
        |state, i, o| match state {
            Ok(st) => FirDecimate::<CQ15>::fir_decimate(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

// ----------------------------------------------------------------- resample

const RS_IN: usize = 320;

pub(super) fn resample_c32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "resample_c32_vs_reference";
    let mut rng = Pcg32::new(611, 1);
    for (up, down, taps_n) in RATIOS {
        let out_n = room(RS_IN, up, down);
        let hist_n = taps_n.div_ceil(up);
        let mut c = arena.carver();
        let input = c.take::<C32>(RS_IN)?;
        let (fb, rf, fac) = (
            c.take::<C32>(out_n)?,
            c.take::<C32>(out_n)?,
            c.take::<C32>(out_n)?,
        );
        let taps = c.take::<f32>(taps_n)?;
        let (h1, h2, h3) = (
            c.take::<C32>(hist_n)?,
            c.take::<C32>(hist_n)?,
            c.take::<C32>(hist_n)?,
        );
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 0.9);
        }
        taps_f32(taps, ratio_cutoff(up, down), up as f64);
        let mut a = ResampleState::<C32, _, _>::new(&*taps, h1, up, down)?;
        let mut b = ResampleState::<C32, _, _>::new(&*taps, h2, up, down)?;
        let mut d = ResampleState::<C32, _, _>::new(&*taps, h3, up, down)?;
        let pa = Resample::<C32>::resample(&Fallback, &mut a, input, fb)?;
        let pb = Resample::<C32>::resample(&Reference, &mut b, input, rf)?;
        let pd = Resample::<C32>::resample(&Sdrk, &mut d, input, fac)?;
        ensure(pa == pb && pa == pd, || {
            Failure::new(CASE, "progress differs").detail(format_args!("{up}/{down}"))
        })?;
        ensure(
            pa.consumed == RS_IN && pa.produced == out_count(RS_IN, up, down),
            || Failure::new(CASE, "wrong output count").detail(format_args!("{up}/{down}: {pa:?}")),
        )?;
        let n = pa.produced;
        min_snr(
            CASE,
            "fallback below 110 dB",
            flat(rf.get(..n).unwrap_or_default()),
            flat(fb.get(..n).unwrap_or_default()),
            110.0,
        )?;
        ensure(
            bit_exact(
                fb.get(..n).unwrap_or_default(),
                fac.get(..n).unwrap_or_default(),
            )
            .is_none(),
            || Failure::new(CASE, "facade differs"),
        )?;
        ensure(
            a.history() == b.history() && a.history() == d.history() && a.phase() == b.phase(),
            || Failure::new(CASE, "states differ"),
        )?;
        // Timing declarations: ratio L / M and group delay (N - 1) / (2 L).
        ensure(
            a.ratio() == sdrk_core::Ratio::new(up as u64, down as u64)?,
            || Failure::new(CASE, "wrong ratio").detail(format_args!("{up}/{down}")),
        )?;
        let delay = sdrk_core::Delay::new(taps_n as u64 - 1, 2 * up as u64)?;
        ensure(a.group_delay().reduced() == delay.reduced(), || {
            Failure::new(CASE, "wrong group delay").detail(format_args!("{up}/{down}"))
        })?;
    }
    Ok(())
}

pub(super) fn resample_cq15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "resample_cq15_vs_reference";
    let mut rng = Pcg32::new(612, 1);
    for extremes in [false, true] {
        for (up, down, taps_n) in RATIOS {
            let out_n = room(RS_IN, up, down);
            let hist_n = taps_n.div_ceil(up);
            let mut c = arena.carver();
            let input = c.take::<CQ15>(RS_IN)?;
            let (fb, rf, fac) = (
                c.take::<CQ15>(out_n)?,
                c.take::<CQ15>(out_n)?,
                c.take::<CQ15>(out_n)?,
            );
            let taps = c.take::<Q15>(taps_n)?;
            let (h1, h2, h3) = (
                c.take::<CQ15>(hist_n)?,
                c.take::<CQ15>(hist_n)?,
                c.take::<CQ15>(hist_n)?,
            );
            if extremes {
                r#gen::extreme::full_scale_alternating(input);
            } else {
                for s in input.iter_mut() {
                    *s = random_cq15(&mut rng, 1.0);
                }
            }
            // The gain L does not fit into Q15 taps; the caller would restore it afterwards.
            taps_q15(taps, ratio_cutoff(up, down), 1.0);
            let mut a = ResampleState::<CQ15, _, _>::new(&*taps, h1, up, down)?;
            let mut b = ResampleState::<CQ15, _, _>::new(&*taps, h2, up, down)?;
            let mut d = ResampleState::<CQ15, _, _>::new(&*taps, h3, up, down)?;
            let pa = Resample::<CQ15>::resample(&Fallback, &mut a, input, fb)?;
            let pb = Resample::<CQ15>::resample(&Reference, &mut b, input, rf)?;
            let pd = Resample::<CQ15>::resample(&Sdrk, &mut d, input, fac)?;
            ensure(
                pa == pb && pa == pd && pa.produced == out_count(RS_IN, up, down),
                || Failure::new(CASE, "progress differs").detail(format_args!("{up}/{down}")),
            )?;
            let n = pa.produced;
            let (fb, rf, fac) = (
                fb.get(..n).unwrap_or_default(),
                rf.get(..n).unwrap_or_default(),
                fac.get(..n).unwrap_or_default(),
            );
            ensure(bit_exact(fb, rf).is_none(), || {
                Failure::new(CASE, "not bit-exact to the reference")
                    .detail(format_args!("{up}/{down}"))
            })?;
            ensure(bit_exact(fb, fac).is_none(), || {
                Failure::new(CASE, "facade differs")
            })?;
            ensure(a.history() == b.history() && a.phase() == b.phase(), || {
                Failure::new(CASE, "states differ")
            })?;
        }
    }
    Ok(())
}

pub(super) fn resample_real_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "resample_real_vs_reference";
    const TAPS: usize = 48;
    const UP: usize = 3;
    const DOWN: usize = 2;
    let hist_n = TAPS.div_ceil(UP);
    let out_n = room(RS_IN, UP, DOWN);
    let mut rng = Pcg32::new(613, 1);
    {
        let mut c = arena.carver();
        let input = c.take::<f32>(RS_IN)?;
        let (fb, rf, fac) = (
            c.take::<f32>(out_n)?,
            c.take::<f32>(out_n)?,
            c.take::<f32>(out_n)?,
        );
        let taps = c.take::<f32>(TAPS)?;
        let (h1, h2, h3) = (
            c.take::<f32>(hist_n)?,
            c.take::<f32>(hist_n)?,
            c.take::<f32>(hist_n)?,
        );
        for s in input.iter_mut() {
            *s = (rng.next_f64() * 1.8 - 0.9) as f32;
        }
        taps_f32(taps, ratio_cutoff(UP, DOWN), UP as f64);
        let mut a = ResampleRealState::<f32, _, _>::new(&*taps, h1, UP, DOWN)?;
        let mut b = ResampleRealState::<f32, _, _>::new(&*taps, h2, UP, DOWN)?;
        let mut d = ResampleRealState::<f32, _, _>::new(&*taps, h3, UP, DOWN)?;
        let pa = ResampleReal::<f32>::resample_real(&Fallback, &mut a, input, fb)?;
        let pb = ResampleReal::<f32>::resample_real(&Reference, &mut b, input, rf)?;
        let pd = ResampleReal::<f32>::resample_real(&Sdrk, &mut d, input, fac)?;
        ensure(
            pa == pb && pa == pd && pa.produced == out_count(RS_IN, UP, DOWN),
            || Failure::new(CASE, "progress differs"),
        )?;
        let n = pa.produced;
        min_snr(
            CASE,
            "f32 below 110 dB",
            rf.get(..n).unwrap_or_default(),
            fb.get(..n).unwrap_or_default(),
            110.0,
        )?;
        ensure(
            bit_exact(
                fb.get(..n).unwrap_or_default(),
                fac.get(..n).unwrap_or_default(),
            )
            .is_none(),
            || Failure::new(CASE, "facade differs"),
        )?;
    }
    let mut c = arena.carver();
    let input = c.take::<Q15>(RS_IN)?;
    let (fb, rf, fac) = (
        c.take::<Q15>(out_n)?,
        c.take::<Q15>(out_n)?,
        c.take::<Q15>(out_n)?,
    );
    let taps = c.take::<Q15>(TAPS)?;
    let (h1, h2, h3) = (
        c.take::<Q15>(hist_n)?,
        c.take::<Q15>(hist_n)?,
        c.take::<Q15>(hist_n)?,
    );
    for s in input.iter_mut() {
        *s = Q15((rng.next_u32() >> 16) as i16);
    }
    taps_q15(taps, ratio_cutoff(UP, DOWN), 1.0);
    let mut a = ResampleRealState::<Q15, _, _>::new(&*taps, h1, UP, DOWN)?;
    let mut b = ResampleRealState::<Q15, _, _>::new(&*taps, h2, UP, DOWN)?;
    let mut d = ResampleRealState::<Q15, _, _>::new(&*taps, h3, UP, DOWN)?;
    let pa = ResampleReal::<Q15>::resample_real(&Fallback, &mut a, input, fb)?;
    let pb = ResampleReal::<Q15>::resample_real(&Reference, &mut b, input, rf)?;
    let pd = ResampleReal::<Q15>::resample_real(&Sdrk, &mut d, input, fac)?;
    ensure(pa == pb && pa == pd, || {
        Failure::new(CASE, "q15 progress differs")
    })?;
    let n = pa.produced;
    let (fb, rf, fac) = (
        fb.get(..n).unwrap_or_default(),
        rf.get(..n).unwrap_or_default(),
        fac.get(..n).unwrap_or_default(),
    );
    ensure(bit_exact(fb, rf).is_none(), || {
        Failure::new(CASE, "q15 not bit-exact")
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "q15 facade differs")
    })
}

pub(super) fn resample_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const IN: usize = 640;
    let mut rng = Pcg32::new(614, 2);
    for (k, (up, down, taps_n)) in RATIOS.into_iter().enumerate() {
        let out_n = room(IN, up, down);
        let hist_n = taps_n.div_ceil(up);
        {
            let mut c = arena.carver();
            let (input, whole, chunked) = (
                c.take::<C32>(IN)?,
                c.take::<C32>(out_n)?,
                c.take::<C32>(out_n)?,
            );
            let taps = c.take::<f32>(taps_n)?;
            let mut hist = hand_out(c.take::<C32>(hist_n)?, c.take::<C32>(hist_n)?);
            for s in input.iter_mut() {
                *s = random_c32(&mut rng, 1.0);
            }
            taps_f32(taps, ratio_cutoff(up, down), up as f64);
            let taps: &[f32] = taps;
            let report = check_block_invariance(
                input,
                whole,
                chunked,
                ChunkPlan::new(641 + k as u64, 61),
                || ResampleState::<C32, _, _>::new(taps, hist(), up, down),
                |state, i, o| match state {
                    Ok(st) => Resample::<C32>::resample(&Sdrk, st, i, o),
                    Err(e) => Err(e.clone()),
                },
            )?;
            ensure(report.outputs == out_count(IN, up, down), || {
                Failure::new("resample_block_invariance", "wrong output count")
            })?;
        }
        let mut c = arena.carver();
        let (input, whole, chunked) = (
            c.take::<CQ15>(IN)?,
            c.take::<CQ15>(out_n)?,
            c.take::<CQ15>(out_n)?,
        );
        let taps = c.take::<Q15>(taps_n)?;
        let mut hist = hand_out(c.take::<CQ15>(hist_n)?, c.take::<CQ15>(hist_n)?);
        for s in input.iter_mut() {
            *s = random_cq15(&mut rng, 1.0);
        }
        taps_q15(taps, ratio_cutoff(up, down), 1.0);
        let taps: &[Q15] = taps;
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(651 + k as u64, 67),
            || ResampleState::<CQ15, _, _>::new(taps, hist(), up, down),
            |state, i, o| match state {
                Ok(st) => Resample::<CQ15>::resample(&Sdrk, st, i, o),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    // The real variant (f32 and Q15) with the 3/2 ratio.
    let (up, down, taps_n) = RATIOS[1];
    let hist_n = taps_n.div_ceil(up);
    let out_n = room(IN, up, down);
    {
        let mut c = arena.carver();
        let (input, whole, chunked) = (
            c.take::<f32>(IN)?,
            c.take::<f32>(out_n)?,
            c.take::<f32>(out_n)?,
        );
        let taps = c.take::<f32>(taps_n)?;
        let mut hist = hand_out(c.take::<f32>(hist_n)?, c.take::<f32>(hist_n)?);
        for s in input.iter_mut() {
            *s = (rng.next_f64() * 2.0 - 1.0) as f32;
        }
        taps_f32(taps, ratio_cutoff(up, down), up as f64);
        let taps: &[f32] = taps;
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(661, 59),
            || ResampleRealState::<f32, _, _>::new(taps, hist(), up, down),
            |state, i, o| match state {
                Ok(st) => ResampleReal::<f32>::resample_real(&Sdrk, st, i, o),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (
        c.take::<Q15>(IN)?,
        c.take::<Q15>(out_n)?,
        c.take::<Q15>(out_n)?,
    );
    let taps = c.take::<Q15>(taps_n)?;
    let mut hist = hand_out(c.take::<Q15>(hist_n)?, c.take::<Q15>(hist_n)?);
    for s in input.iter_mut() {
        *s = Q15((rng.next_u32() >> 16) as i16);
    }
    taps_q15(taps, ratio_cutoff(up, down), 1.0);
    let taps: &[Q15] = taps;
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(662, 71),
        || ResampleRealState::<Q15, _, _>::new(taps, hist(), up, down),
        |state, i, o| match state {
            Ok(st) => ResampleReal::<Q15>::resample_real(&Sdrk, st, i, o),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------- quad_demod

/// A phase-modulated signal `exp(j phi[n])` with `phi` the running sum of `step(n)`.
fn fm_signal<T: sdrk_core::ComplexSample>(
    out: &mut [T],
    mut step: impl FnMut(usize) -> f64,
    amp: f64,
) {
    let mut phase = 0.0f64;
    for (n, o) in out.iter_mut().enumerate() {
        phase += step(n);
        *o = T::from_c64_round(num_complex::Complex64::new(
            amp * libm::cos(phase),
            amp * libm::sin(phase),
        ));
    }
}

pub(super) fn quad_demod_c32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "quad_demod_c32_vs_reference";
    const N: usize = 1024;
    let mut c = arena.carver();
    let input = c.take::<C32>(N)?;
    let (fb, rf, fac) = (c.take::<f32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
    fm_signal(
        input,
        |n| 2.0 * PI * 0.05 * libm::sin(2.0 * PI * 0.004 * n as f64),
        1.0,
    );
    let gain = 1.0f32 / (2.0 * PI as f32 * 0.05);
    let (mut a, mut b, mut d) = (
        QuadDemodState::<C32>::new(),
        QuadDemodState::<C32>::new(),
        QuadDemodState::<C32>::new(),
    );
    QuadDemod::<C32>::quad_demod(&Fallback, &mut a, input, fb, gain)?;
    QuadDemod::<C32>::quad_demod(&Reference, &mut b, input, rf, gain)?;
    QuadDemod::<C32>::quad_demod(&Sdrk, &mut d, input, fac, gain)?;
    min_snr(CASE, "fallback below 100 dB", rf, fb, 100.0)?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs")
    })?;
    ensure(a == b && a == d, || Failure::new(CASE, "states differ"))?;

    // A pure tone gives the constant 2 pi f * gain (the first output has a zero predecessor).
    let f = 0.0731f64;
    fm_signal(input, |_| 2.0 * PI * f, 1.0);
    let mut s = QuadDemodState::<C32>::new();
    QuadDemod::<C32>::quad_demod(&Sdrk, &mut s, input, fb, 1.0)?;
    let want = (2.0 * PI * f) as f32;
    ensure(
        fb.first() == Some(&0.0) && fb.iter().skip(1).all(|v| (v - want).abs() < 1e-5),
        || Failure::new(CASE, "tone is not constant"),
    )
}

pub(super) fn quad_demod_cq15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "quad_demod_cq15_vs_reference";
    const N: usize = 1024;
    let mut c = arena.carver();
    let input = c.take::<CQ15>(N)?;
    let (fb, rf, fac) = (c.take::<Q15>(N)?, c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    let mut rng = Pcg32::new(621, 1);
    for pass in 0..3 {
        match pass {
            0 => fm_signal(
                input,
                |n| 2.0 * PI * 0.05 * libm::sin(2.0 * PI * 0.004 * n as f64),
                0.9,
            ),
            1 => {
                for s in input.iter_mut() {
                    *s = random_cq15(&mut rng, 1.0);
                }
            }
            _ => r#gen::extreme::full_scale_alternating(input),
        }
        let gain = Q15(32767 - 5000 * pass);
        let (mut a, mut b, mut d) = (
            QuadDemodState::<CQ15>::new(),
            QuadDemodState::<CQ15>::new(),
            QuadDemodState::<CQ15>::new(),
        );
        QuadDemod::<CQ15>::quad_demod(&Fallback, &mut a, input, fb, gain)?;
        QuadDemod::<CQ15>::quad_demod(&Reference, &mut b, input, rf, gain)?;
        QuadDemod::<CQ15>::quad_demod(&Sdrk, &mut d, input, fac, gain)?;
        // The fallback rounds twice (phase, then gain), the reference once.
        let worst = fb
            .iter()
            .zip(rf.iter())
            .map(|(x, y)| (i32::from(x.0) - i32::from(y.0)).abs())
            .max()
            .unwrap_or(0);
        ensure(worst <= 1, || {
            Failure::new(CASE, "more than one LSB from the reference")
                .detail(format_args!("pass {pass}: {worst}"))
        })?;
        ensure(bit_exact(fb, fac).is_none(), || {
            Failure::new(CASE, "facade differs")
        })?;
        ensure(a == b && a == d, || Failure::new(CASE, "states differ"))?;
    }
    Ok(())
}

pub(super) fn quad_demod_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const N: usize = 1024;
    let mut rng = Pcg32::new(622, 2);
    {
        let mut c = arena.carver();
        let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 1.0);
        }
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(623, 61),
            QuadDemodState::<C32>::new,
            |st, i, o| {
                same_rate(i, o, |i, o| {
                    QuadDemod::<C32>::quad_demod(&Sdrk, st, i, o, 1.7)
                })
            },
        )?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<CQ15>(N)?, c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 1.0);
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(624, 43),
        QuadDemodState::<CQ15>::new,
        |st, i, o| {
            same_rate(i, o, |i, o| {
                QuadDemod::<CQ15>::quad_demod(&Sdrk, st, i, o, Q15(20000))
            })
        },
    )?;
    Ok(())
}

// --------------------------------------------------------------- deemphasis

pub(super) fn deemphasis_f32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "deemphasis_f32_vs_reference";
    const N: usize = 2048;
    let mut c = arena.carver();
    let input = c.take::<f32>(N)?;
    let (fb, rf, fac) = (c.take::<f32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
    let mut rng = Pcg32::new(631, 1);
    for s in input.iter_mut() {
        *s = (rng.next_f64() * 1.8 - 0.9) as f32;
    }
    let rate = Rate::hz(48_000)?;
    let mut a = DeemphasisState::<f32>::new(50e-6, rate)?;
    let (mut b, mut d) = (a, a);
    Deemphasis::<f32>::deemphasis(&Fallback, &mut a, input, fb)?;
    Deemphasis::<f32>::deemphasis(&Reference, &mut b, input, rf)?;
    Deemphasis::<f32>::deemphasis(&Sdrk, &mut d, input, fac)?;
    min_snr(CASE, "fallback below 100 dB", rf, fb, 100.0)?;
    ensure(bit_exact(fb, fac).is_none() && a == d, || {
        Failure::new(CASE, "facade differs")
    })
}

pub(super) fn deemphasis_q15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "deemphasis_q15_vs_reference";
    const N: usize = 2048;
    let mut c = arena.carver();
    let input = c.take::<Q15>(N)?;
    let (fb, rf, fac) = (c.take::<Q15>(N)?, c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    let mut rng = Pcg32::new(632, 1);
    let rate = Rate::hz(48_000)?;
    for (pass, tau) in [50e-6, 75e-6, 2e-6].into_iter().enumerate() {
        for (i, s) in input.iter_mut().enumerate() {
            *s = match pass {
                2 if i % 2 == 0 => Q15::MAX,
                2 => Q15::MIN,
                _ => Q15((rng.next_u32() >> 16) as i16),
            };
        }
        let mut a = DeemphasisState::<Q15>::new(tau, rate)?;
        let (mut b, mut d) = (a, a);
        Deemphasis::<Q15>::deemphasis(&Fallback, &mut a, input, fb)?;
        Deemphasis::<Q15>::deemphasis(&Reference, &mut b, input, rf)?;
        Deemphasis::<Q15>::deemphasis(&Sdrk, &mut d, input, fac)?;
        ensure(bit_exact(fb, rf).is_none() && a == b, || {
            Failure::new(CASE, "not bit-exact to the reference").detail(format_args!("pass {pass}"))
        })?;
        ensure(bit_exact(fb, fac).is_none() && a == d, || {
            Failure::new(CASE, "facade differs")
        })?;
    }
    Ok(())
}

pub(super) fn deemphasis_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const N: usize = 1024;
    let mut rng = Pcg32::new(633, 2);
    let rate = Rate::hz(48_000)?;
    {
        let mut c = arena.carver();
        let (input, whole, chunked) = (c.take::<f32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
        for s in input.iter_mut() {
            *s = (rng.next_f64() * 2.0 - 1.0) as f32;
        }
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(634, 57),
            || DeemphasisState::<f32>::new(75e-6, rate),
            |st, i, o| match st {
                Ok(st) => same_rate(i, o, |i, o| Deemphasis::<f32>::deemphasis(&Sdrk, st, i, o)),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<Q15>(N)?, c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    for s in input.iter_mut() {
        *s = Q15((rng.next_u32() >> 16) as i16);
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(635, 49),
        || DeemphasisState::<Q15>::new(75e-6, rate),
        |st, i, o| match st {
            Ok(st) => same_rate(i, o, |i, o| Deemphasis::<Q15>::deemphasis(&Sdrk, st, i, o)),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

// ------------------------------------------------------------------ dc_block

pub(super) fn dc_block_c32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "dc_block_c32_vs_reference";
    const N: usize = 1024;
    let mut c = arena.carver();
    let input = c.take::<C32>(N)?;
    let (fb, rf, fac) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
    let mut rng = Pcg32::new(641, 1);
    for s in input.iter_mut() {
        let n = random_c32(&mut rng, 0.2);
        *s = C32::new(n.re + 0.3, n.im + 0.2);
    }
    let mut a = DcBlockState::<C32>::new(0.995)?;
    let (mut b, mut d) = (a, a);
    DcBlock::<C32>::dc_block(&Fallback, &mut a, input, fb)?;
    DcBlock::<C32>::dc_block(&Reference, &mut b, input, rf)?;
    DcBlock::<C32>::dc_block(&Sdrk, &mut d, input, fac)?;
    min_snr(CASE, "fallback below 100 dB", flat(rf), flat(fb), 100.0)?;
    ensure(bit_exact(fb, fac).is_none() && a == d, || {
        Failure::new(CASE, "facade differs")
    })
}

pub(super) fn dc_block_cq15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "dc_block_cq15_vs_reference";
    const N: usize = 2048;
    let mut c = arena.carver();
    let input = c.take::<CQ15>(N)?;
    let (fb, rf, fac) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    let mut rng = Pcg32::new(642, 1);
    for extremes in [false, true] {
        if extremes {
            r#gen::extreme::full_scale_alternating(input);
        } else {
            for s in input.iter_mut() {
                *s = random_cq15(&mut rng, 1.0);
            }
        }
        let mut a = DcBlockState::<CQ15>::new(Q15(32735))?;
        let (mut b, mut d) = (a, a);
        DcBlock::<CQ15>::dc_block(&Fallback, &mut a, input, fb)?;
        DcBlock::<CQ15>::dc_block(&Reference, &mut b, input, rf)?;
        DcBlock::<CQ15>::dc_block(&Sdrk, &mut d, input, fac)?;
        ensure(bit_exact(fb, rf).is_none() && a == b, || {
            Failure::new(CASE, "not bit-exact")
        })?;
        ensure(bit_exact(fb, fac).is_none() && a == d, || {
            Failure::new(CASE, "facade differs")
        })?;
    }
    Ok(())
}

pub(super) fn dc_block_block_invariance(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const N: usize = 1024;
    let mut rng = Pcg32::new(643, 2);
    {
        let mut c = arena.carver();
        let (input, whole, chunked) = (c.take::<C32>(N)?, c.take::<C32>(N)?, c.take::<C32>(N)?);
        for s in input.iter_mut() {
            *s = random_c32(&mut rng, 1.0);
        }
        check_block_invariance(
            input,
            whole,
            chunked,
            ChunkPlan::new(644, 51),
            || DcBlockState::<C32>::new(0.99),
            |st, i, o| match st {
                Ok(st) => same_rate(i, o, |i, o| DcBlock::<C32>::dc_block(&Sdrk, st, i, o)),
                Err(e) => Err(e.clone()),
            },
        )?;
    }
    let mut c = arena.carver();
    let (input, whole, chunked) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N)?, c.take::<CQ15>(N)?);
    for s in input.iter_mut() {
        *s = random_cq15(&mut rng, 1.0);
    }
    check_block_invariance(
        input,
        whole,
        chunked,
        ChunkPlan::new(645, 45),
        || DcBlockState::<CQ15>::new(Q15(32000)),
        |st, i, o| match st {
            Ok(st) => same_rate(i, o, |i, o| DcBlock::<CQ15>::dc_block(&Sdrk, st, i, o)),
            Err(e) => Err(e.clone()),
        },
    )?;
    Ok(())
}

// ------------------------------------------------------------------- window

pub(super) fn window_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "window_vs_reference";
    const N: usize = 512;
    let mut c = arena.carver();
    let (wf, wq) = (c.take::<f32>(N)?, c.take::<Q15>(N)?);
    let (xf, fb, rf, fac) = (
        c.take::<C32>(N)?,
        c.take::<C32>(N)?,
        c.take::<C32>(N)?,
        c.take::<C32>(N)?,
    );
    let (xq, qb, qr, qfac) = (
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
        c.take::<CQ15>(N)?,
    );
    let mut rng = Pcg32::new(651, 1);
    for sym in [Symmetry::Periodic, Symmetry::Symmetric] {
        kaiser(wf, 6.0, sym)?;
        kaiser(wq, 6.0, sym)?;
        for s in xf.iter_mut() {
            *s = random_c32(&mut rng, 1.0);
        }
        for s in xq.iter_mut() {
            *s = random_cq15(&mut rng, 1.0);
        }
        Window::<C32>::window(&Fallback, xf, fb, wf)?;
        Window::<C32>::window(&Reference, xf, rf, wf)?;
        Window::<C32>::window(&Sdrk, xf, fac, wf)?;
        ensure(
            bit_exact(fb, rf).is_none() && bit_exact(fb, fac).is_none(),
            || Failure::new(CASE, "c32 window is not bit-exact"),
        )?;
        Window::<CQ15>::window(&Fallback, xq, qb, wq)?;
        Window::<CQ15>::window(&Reference, xq, qr, wq)?;
        Window::<CQ15>::window(&Sdrk, xq, qfac, wq)?;
        ensure(
            bit_exact(qb, qr).is_none() && bit_exact(qb, qfac).is_none(),
            || Failure::new(CASE, "cq15 window is not bit-exact"),
        )?;
        // Spot check against the definition.
        let first_ok = fb
            .first()
            .zip(xf.first())
            .zip(wf.first())
            .is_some_and(|((o, x), w)| o.re == x.re * w && o.im == x.im * w);
        let q_ok = qb
            .get(7)
            .zip(xq.get(7))
            .zip(wq.get(7))
            .is_some_and(|((o, x), w)| o.re == q15_mul(x.re, *w) && o.im == q15_mul(x.im, *w));
        ensure(first_ok && q_ok, || {
            Failure::new(CASE, "window differs from its definition")
        })?;
    }
    Ok(())
}

pub(super) fn window_generators(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "window_generators";
    const N: usize = 1025;
    let mut c = arena.carver();
    let (w, p) = (c.take::<f32>(N)?, c.take::<f32>(N - 1)?);
    for which in 0..3 {
        let make = |out: &mut [f32], sym| match which {
            0 => hann(out, sym),
            1 => blackman_harris(out, sym),
            _ => kaiser(out, 8.0, sym),
        };
        make(w, Symmetry::Symmetric)?;
        let symmetric = (0..N / 2)
            .all(|i| w.get(i).map(|v| v.to_bits()) == w.get(N - 1 - i).map(|v| v.to_bits()));
        ensure(symmetric, || {
            Failure::new(CASE, "not symmetric").detail(format_args!("window {which}"))
        })?;
        // The periodic window of length N - 1 is the symmetric window of length N without its last value.
        make(p, Symmetry::Periodic)?;
        let head = w.get(..N - 1).unwrap_or_default();
        ensure(bit_exact(p, head).is_none(), || {
            Failure::new(CASE, "periodic differs from the truncated symmetric window")
                .detail(format_args!("window {which}"))
        })?;
        let mut one = [0f32; 1];
        make(&mut one, Symmetry::Symmetric)?;
        ensure(one == [1.0], || Failure::new(CASE, "length one window"))?;
    }
    // Gain of the periodic Hann window.
    hann(p, Symmetry::Periodic)?;
    let (cg, nebw) = window_gain(p);
    ensure(
        libm::fabs(cg - 0.5) < 1e-6 && libm::fabs(nebw - 1.5) < 1e-3,
        || Failure::new(CASE, "hann gain").detail(format_args!("{cg} {nebw}")),
    )
}

// ------------------------------------------------------------ mag_sq, power_db

pub(super) fn mag_sq_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "mag_sq_vs_reference";
    const N: usize = 1024;
    let mut c = arena.carver();
    let xf = c.take::<C32>(N)?;
    let (fb, rf, fac) = (c.take::<f32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
    let xq = c.take::<CQ15>(N)?;
    let (qb, qr, qfac) = (c.take::<Q15>(N)?, c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    let mut rng = Pcg32::new(661, 1);
    for s in xf.iter_mut() {
        *s = random_c32(&mut rng, 1.0);
    }
    MagSq::<C32>::mag_sq(&Fallback, xf, fb)?;
    MagSq::<C32>::mag_sq(&Reference, xf, rf)?;
    MagSq::<C32>::mag_sq(&Sdrk, xf, fac)?;
    min_snr(CASE, "f32 below 140 dB", rf, fb, 140.0)?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs")
    })?;
    for (pass, extremes) in [false, true].into_iter().enumerate() {
        if extremes {
            r#gen::extreme::full_scale_alternating(xq);
        } else {
            for s in xq.iter_mut() {
                *s = random_cq15(&mut rng, 1.0);
            }
        }
        MagSq::<CQ15>::mag_sq(&Fallback, xq, qb)?;
        MagSq::<CQ15>::mag_sq(&Reference, xq, qr)?;
        MagSq::<CQ15>::mag_sq(&Sdrk, xq, qfac)?;
        ensure(
            bit_exact(qb, qr).is_none() && bit_exact(qb, qfac).is_none(),
            || Failure::new(CASE, "q15 is not bit-exact").detail(format_args!("pass {pass}")),
        )?;
    }
    // Full scale saturates.
    ensure(qb.first() == Some(&Q15::MAX), || {
        Failure::new(CASE, "full scale did not saturate")
    })
}

pub(super) fn power_db_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "power_db_vs_reference";
    const N: usize = 1024;
    let mut c = arena.carver();
    let p = c.take::<f32>(N)?;
    let (fb, rf, fac) = (c.take::<f32>(N)?, c.take::<f32>(N)?, c.take::<f32>(N)?);
    let pq = c.take::<Q15>(N)?;
    let mut rng = Pcg32::new(662, 1);
    for s in p.iter_mut() {
        *s = libm::pow(10.0, rng.next_f64() * 26.0 - 25.0) as f32;
    }
    for (slot, v) in p
        .iter_mut()
        .zip([0.0, -0.0, -1.0, f32::NAN, -1e-25, 1e-20, 1e-30, 1e-40])
    {
        *slot = v;
    }
    PowerDb::<f32>::power_db(&Fallback, p, fb, 1.5)?;
    PowerDb::<f32>::power_db(&Reference, p, rf, 1.5)?;
    PowerDb::<f32>::power_db(&Sdrk, p, fac, 1.5)?;
    let worst = fb
        .iter()
        .zip(rf.iter())
        .map(|(a, b)| libm::fabs(f64::from(a - b)))
        .fold(0.0f64, f64::max);
    ensure(worst <= 1e-4, || {
        Failure::new(CASE, "f32 error above 1e-4 dB").detail(format_args!("{worst}"))
    })?;
    ensure(bit_exact(fb, fac).is_none(), || {
        Failure::new(CASE, "facade differs")
    })?;
    ensure(
        fb.iter()
            .take(8)
            .all(|v| libm::fabs(f64::from(v - (-200.0 + 1.5))) < 1e-4),
        || Failure::new(CASE, "floor not applied"),
    )?;
    for s in pq.iter_mut() {
        *s = Q15((rng.next_u32() >> 16) as i16);
    }
    PowerDb::<Q15>::power_db(&Fallback, pq, fb, 0.0)?;
    PowerDb::<Q15>::power_db(&Reference, pq, rf, 0.0)?;
    let worst = fb
        .iter()
        .zip(rf.iter())
        .map(|(a, b)| libm::fabs(f64::from(a - b)))
        .fold(0.0f64, f64::max);
    ensure(worst <= 1e-4, || {
        Failure::new(CASE, "q15 error above 1e-4 dB").detail(format_args!("{worst}"))
    })
}

// ------------------------------------------------------------------ extremes

pub(super) fn extremes_rx_kernels(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "extremes_rx_kernels";
    const N: usize = 512;
    let mut c = arena.carver();
    let (xf, yf) = (c.take::<C32>(N)?, c.take::<C32>(N * 3)?);
    let (xq, yq) = (c.take::<CQ15>(N)?, c.take::<CQ15>(N * 3)?);
    let (rf, rq, rq2) = (c.take::<f32>(N)?, c.take::<Q15>(N)?, c.take::<Q15>(N)?);
    let (af, aq) = (c.take::<f32>(N)?, c.take::<Q15>(N)?);
    let rate = Rate::hz(48_000)?;
    let taps_f = [0.25f32, 0.5, 0.25];
    let taps_q = [Q15(8192), Q15(16384), Q15(8192)];
    let has_nan = |s: &[C32]| s.iter().any(|v| v.re.is_nan() || v.im.is_nan());

    // f32 specials (NaN, infinities, denormals, f32::MAX) through every f32 kernel: no panic,
    // and NaN reaches the output.
    r#gen::extreme::f32_specials_complex(xf);
    let mut st = FirDecimState::<C32, _, _>::new(taps_f, [C32::ZERO; 2], 2)?;
    FirDecimate::<C32>::fir_decimate(&Sdrk, &mut st, xf, yf)?;
    ensure(has_nan(yf.get(..N / 2).unwrap_or_default()), || {
        Failure::new(CASE, "fir_decimate swallowed NaN")
    })?;
    let mut st = ResampleState::<C32, _, _>::new(taps_f, [C32::ZERO; 1], 3, 2)?;
    let p = Resample::<C32>::resample(&Sdrk, &mut st, xf, yf)?;
    ensure(
        p.consumed == N && has_nan(yf.get(..p.produced).unwrap_or_default()),
        || Failure::new(CASE, "resample swallowed NaN"),
    )?;
    let mut st = QuadDemodState::<C32>::new();
    QuadDemod::<C32>::quad_demod(&Sdrk, &mut st, xf, rf, 1.0)?;
    ensure(rf.iter().any(|v| v.is_nan()), || {
        Failure::new(CASE, "quad_demod swallowed NaN")
    })?;
    r#gen::extreme::f32_specials(af);
    let mut st = DeemphasisState::<f32>::new(75e-6, rate)?;
    Deemphasis::<f32>::deemphasis(&Sdrk, &mut st, af, rf)?;
    ensure(rf.iter().any(|v| v.is_nan()), || {
        Failure::new(CASE, "deemphasis swallowed NaN")
    })?;
    let mut st = DcBlockState::<C32>::new(0.99)?;
    DcBlock::<C32>::dc_block(&Sdrk, &mut st, xf, yf.get_mut(..N).unwrap_or_default())?;
    ensure(has_nan(yf.get(..N).unwrap_or_default()), || {
        Failure::new(CASE, "dc_block swallowed NaN")
    })?;
    MagSq::<C32>::mag_sq(&Sdrk, xf, rf)?;
    ensure(rf.iter().any(|v| v.is_nan()), || {
        Failure::new(CASE, "mag_sq swallowed NaN")
    })?;
    PowerDb::<f32>::power_db(&Sdrk, af, rf, 0.0)?;
    ensure(rf.iter().all(|v| !v.is_nan()), || {
        Failure::new(CASE, "power_db returned NaN")
    })?;
    let mut w = [1.0f32; N];
    Window::<C32>::window(&Sdrk, xf, yf.get_mut(..N).unwrap_or_default(), &w)?;
    w.fill(f32::NAN);
    Window::<C32>::window(&Sdrk, xf, yf.get_mut(..N).unwrap_or_default(), &w)?;

    // Silence stays silent in every Q15 kernel (and with it the state).
    r#gen::extreme::silence(xq);
    let mut st = FirDecimState::<CQ15, _, _>::new(taps_q, [CQ15::ZERO; 2], 2)?;
    FirDecimate::<CQ15>::fir_decimate(&Sdrk, &mut st, xq, yq)?;
    ensure(yq.iter().take(N / 2).all(|s| *s == CQ15::ZERO), || {
        Failure::new(CASE, "fir_decimate: silence")
    })?;
    let mut st = ResampleState::<CQ15, _, _>::new(taps_q, [CQ15::ZERO; 1], 3, 2)?;
    let p = Resample::<CQ15>::resample(&Sdrk, &mut st, xq, yq)?;
    ensure(yq.iter().take(p.produced).all(|s| *s == CQ15::ZERO), || {
        Failure::new(CASE, "resample: silence")
    })?;
    let mut st = QuadDemodState::<CQ15>::new();
    QuadDemod::<CQ15>::quad_demod(&Sdrk, &mut st, xq, rq, Q15(32767))?;
    ensure(rq.iter().all(|s| s.0 == 0), || {
        Failure::new(CASE, "quad_demod: silence")
    })?;
    let mut st = DcBlockState::<CQ15>::new(Q15(32735))?;
    DcBlock::<CQ15>::dc_block(&Sdrk, &mut st, xq, yq.get_mut(..N).unwrap_or_default())?;
    ensure(yq.iter().take(N).all(|s| *s == CQ15::ZERO), || {
        Failure::new(CASE, "dc_block: silence")
    })?;
    r#gen::extreme::silence(aq);
    let mut st = DeemphasisState::<Q15>::new(75e-6, rate)?;
    Deemphasis::<Q15>::deemphasis(&Sdrk, &mut st, aq, rq)?;
    ensure(rq.iter().all(|s| s.0 == 0), || {
        Failure::new(CASE, "deemphasis: silence")
    })?;
    MagSq::<CQ15>::mag_sq(&Sdrk, xq, rq)?;
    ensure(rq.iter().all(|s| s.0 == 0), || {
        Failure::new(CASE, "mag_sq: silence")
    })?;

    // Full scale: saturation instead of wrap-around, always matching the reference.
    r#gen::extreme::full_scale_alternating(xq);
    let mut st = FirDecimState::<CQ15, _, _>::new([Q15(32767); 8], [CQ15::ZERO; 7], 1)?;
    let mut rs = st.clone();
    FirDecimate::<CQ15>::fir_decimate(&Sdrk, &mut st, xq, yq)?;
    let mut expect = [CQ15::ZERO; N];
    FirDecimate::<CQ15>::fir_decimate(&Reference, &mut rs, xq, &mut expect)?;
    ensure(yq.get(..N) == Some(&expect[..]), || {
        Failure::new(CASE, "fir_decimate: full scale")
    })?;
    let mut st = DcBlockState::<CQ15>::new(Q15(32767))?;
    let mut rs = st;
    DcBlock::<CQ15>::dc_block(&Sdrk, &mut st, xq, yq.get_mut(..N).unwrap_or_default())?;
    DcBlock::<CQ15>::dc_block(&Reference, &mut rs, xq, &mut expect)?;
    ensure(yq.get(..N) == Some(&expect[..]) && st == rs, || {
        Failure::new(CASE, "dc_block: full scale")
    })?;
    for (i, s) in aq.iter_mut().enumerate() {
        *s = if i % 3 == 0 { Q15::MIN } else { Q15::MAX };
    }
    let mut st = DeemphasisState::<Q15>::new(1e-6, rate)?;
    let mut rs = st;
    Deemphasis::<Q15>::deemphasis(&Sdrk, &mut st, aq, rq)?;
    Deemphasis::<Q15>::deemphasis(&Reference, &mut rs, aq, rq2)?;
    ensure(bit_exact(rq, rq2).is_none() && st == rs, || {
        Failure::new(CASE, "deemphasis: full scale")
    })?;
    MagSq::<CQ15>::mag_sq(&Sdrk, xq, rq)?;
    ensure(rq.iter().all(|s| *s == Q15::MAX), || {
        Failure::new(CASE, "mag_sq: full scale")
    })?;
    let mut st = QuadDemodState::<CQ15>::new();
    let mut rs = st;
    QuadDemod::<CQ15>::quad_demod(&Sdrk, &mut st, xq, rq, Q15::MAX)?;
    QuadDemod::<CQ15>::quad_demod(&Reference, &mut rs, xq, rq2, Q15::MAX)?;
    ensure(
        rq.iter()
            .zip(rq2.iter())
            .all(|(a, b)| (i32::from(a.0) - i32::from(b.0)).abs() <= 1)
            && st == rs,
        || Failure::new(CASE, "quad_demod: full scale"),
    )?;
    let ones = [Q15::MIN; N];
    let (win, expect_win) = yq.split_at_mut(N);
    let expect_win = expect_win.get_mut(..N).unwrap_or_default();
    Window::<CQ15>::window(&Sdrk, xq, win, &ones)?;
    Window::<CQ15>::window(&Reference, xq, expect_win, &ones)?;
    ensure(bit_exact(win, expect_win).is_none(), || {
        Failure::new(CASE, "window: full scale")
    })?;
    // +32767 * -1 = -32767 and -32767 * -1 = 32767: saturation, not wrap-around.
    ensure(
        win.first().map(|s| s.re.0) == Some(-32767) && win.get(1).map(|s| s.re.0) == Some(32767),
        || Failure::new(CASE, "window: full-scale values"),
    )?;
    let mut st = DeemphasisState::<f32>::new(75e-6, rate)?;
    af.fill(0.5);
    Deemphasis::<f32>::deemphasis(&Sdrk, &mut st, af, rf)?;
    ensure(rf.iter().all(|v| v.is_finite()), || {
        Failure::new(CASE, "deemphasis: state not finite")
    })
}
