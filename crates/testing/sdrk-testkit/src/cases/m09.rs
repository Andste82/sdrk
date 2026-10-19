//! Suite cases of M09: the `dot` kernels and the blocks (`ddc`, `fm_demod`, `spectrum`,
//! `resampler`).
//!
//! The blocks get their memory from the suite arena through `ArenaStorage`, so the cases run on
//! microcontrollers too.

#![allow(clippy::many_single_char_names, clippy::too_many_lines)]

use core::f64::consts::TAU;

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::design::DesignSpec;
use sdrk_blocks::{
    ARENA_ALIGN, ArenaStorage, Block, BlockSample, DdcBlock, DdcConfig, FrameBlock, ResamplerBlock,
    ResamplerConfig, SpectrumBlock, SpectrumConfig, TunedElem,
};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::dot::{Dot, DotReal};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::{C32, CQ15, ComplexSample, Fallback, Q15, Rate, Reference};

use super::ensure;
use super::m05::{random_c32, random_cq15};
use super::m06::{flat, min_snr};
use crate::arena::{Arena, Carver};
use crate::e2e;
use crate::metrics::bit_exact;
use crate::rng::Pcg32;
use crate::suite::Failure;

// ---------------------------------------------------------------------------------------
// dot
// ---------------------------------------------------------------------------------------

/// Lengths of the dot products.
const LENGTHS: [usize; 4] = [1, 7, 64, 255];
/// Number of windows per length.
const WINDOWS: usize = 64;

pub(super) fn dot_c32_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "dot_c32_vs_reference";
    let mut c = arena.carver();
    let x = c.take::<C32>(255 + WINDOWS)?;
    let taps = c.take::<f32>(255)?;
    let (want, got, fast) = (
        c.take::<C32>(WINDOWS)?,
        c.take::<C32>(WINDOWS)?,
        c.take::<C32>(WINDOWS)?,
    );
    let mut rng = Pcg32::new(900, 1);
    for s in x.iter_mut() {
        *s = random_c32(&mut rng, 0.5);
    }
    for t in taps.iter_mut() {
        *t = (rng.next_f64() - 0.5) as f32;
    }
    for n in LENGTHS {
        let t = taps.get(..n).unwrap_or_default();
        for i in 0..WINDOWS {
            let w = x.get(i..i + n).unwrap_or_default();
            *want.get_mut(i).ok_or_else(|| Failure::new(CASE, "index"))? =
                Dot::<C32>::dot(&Reference, w, t)?;
            *got.get_mut(i).ok_or_else(|| Failure::new(CASE, "index"))? =
                Dot::<C32>::dot(&Fallback, w, t)?;
            *fast.get_mut(i).ok_or_else(|| Failure::new(CASE, "index"))? =
                Dot::<C32>::dot(&Sdrk, w, t)?;
        }
        min_snr(
            CASE,
            "fallback differs from the reference",
            flat(want),
            flat(got),
            120.0,
        )?;
        ensure(bit_exact(got, fast).is_none(), || {
            Failure::new(CASE, "facade differs from the fallback")
        })?;
    }
    Ok(())
}

pub(super) fn dot_cq15_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "dot_cq15_vs_reference";
    let mut c = arena.carver();
    let x = c.take::<CQ15>(255 + WINDOWS)?;
    let taps = c.take::<Q15>(255)?;
    let mut rng = Pcg32::new(901, 1);
    for s in x.iter_mut() {
        *s = random_cq15(&mut rng, 1.0);
    }
    for t in taps.iter_mut() {
        *t = Q15::from_f32((rng.next_f64() * 2.0 - 1.0) as f32);
    }
    for n in LENGTHS {
        let t = taps.get(..n).unwrap_or_default();
        for i in 0..WINDOWS {
            let w = x.get(i..i + n).unwrap_or_default();
            let want = Dot::<CQ15>::dot(&Reference, w, t)?;
            let got = Dot::<CQ15>::dot(&Fallback, w, t)?;
            let fast = Dot::<CQ15>::dot(&Sdrk, w, t)?;
            ensure(want == got && got == fast, || {
                Failure::new(CASE, "dot differs from the reference")
                    .detail(format_args!("length {n}, window {i}"))
            })?;
        }
    }
    Ok(())
}

pub(super) fn dot_real_vs_reference(arena: &mut Arena<'_>) -> Result<(), Failure> {
    const CASE: &str = "dot_real_vs_reference";
    let mut c = arena.carver();
    let x = c.take::<f32>(255 + WINDOWS)?;
    let taps = c.take::<f32>(255)?;
    let (want, got) = (c.take::<f32>(WINDOWS)?, c.take::<f32>(WINDOWS)?);
    let (xq, tq) = (c.take::<Q15>(255 + WINDOWS)?, c.take::<Q15>(255)?);
    let mut rng = Pcg32::new(902, 1);
    for (s, q) in x.iter_mut().zip(xq.iter_mut()) {
        *s = (rng.next_f64() - 0.5) as f32;
        *q = Q15::from_f32((rng.next_f64() * 2.0 - 1.0) as f32);
    }
    for (t, q) in taps.iter_mut().zip(tq.iter_mut()) {
        *t = (rng.next_f64() - 0.5) as f32;
        *q = Q15::from_f32((rng.next_f64() * 2.0 - 1.0) as f32);
    }
    for n in LENGTHS {
        let t = taps.get(..n).unwrap_or_default();
        for i in 0..WINDOWS {
            let w = x.get(i..i + n).unwrap_or_default();
            *want.get_mut(i).ok_or_else(|| Failure::new(CASE, "index"))? =
                DotReal::<f32>::dot_real(&Reference, w, t)?;
            *got.get_mut(i).ok_or_else(|| Failure::new(CASE, "index"))? =
                DotReal::<f32>::dot_real(&Sdrk, w, t)?;
            let wq = xq.get(i..i + n).unwrap_or_default();
            let tqs = tq.get(..n).unwrap_or_default();
            let a = DotReal::<Q15>::dot_real(&Reference, wq, tqs)?;
            let b = DotReal::<Q15>::dot_real(&Fallback, wq, tqs)?;
            let d = DotReal::<Q15>::dot_real(&Sdrk, wq, tqs)?;
            ensure(a == b && b == d, || {
                Failure::new(CASE, "Q15 dot differs from the reference")
                    .detail(format_args!("length {n}, window {i}"))
            })?;
        }
        min_snr(CASE, "f32 dot differs from the reference", want, got, 120.0)?;
    }
    Ok(())
}

pub(super) fn extremes_dot(arena: &mut Arena<'_>) -> Result<(), Failure> {
    use crate::r#gen::extreme;
    const CASE: &str = "extremes_dot";
    const N: usize = 33;
    let mut c = arena.carver();
    let q = c.take::<CQ15>(N)?;
    let qr = c.take::<Q15>(N)?;
    let taps_q = c.take::<Q15>(N)?;
    let x = c.take::<C32>(N)?;
    let xr = c.take::<f32>(N)?;
    let taps_f = c.take::<f32>(N)?;
    // Full-scale alternating Q15 data against full-scale taps saturates identically.
    extreme::full_scale_alternating(q);
    for (i, (r, t)) in qr.iter_mut().zip(taps_q.iter_mut()).enumerate() {
        *r = Q15(if i % 2 == 0 { -32768 } else { 32767 });
        *t = Q15(-32768);
    }
    ensure(
        Dot::<CQ15>::dot(&Sdrk, q, taps_q)? == Dot::<CQ15>::dot(&Reference, q, taps_q)?,
        || Failure::new(CASE, "CQ15 dot with full-scale data"),
    )?;
    let sat = DotReal::<Q15>::dot_real(&Sdrk, qr, taps_q)?;
    ensure(
        sat == DotReal::<Q15>::dot_real(&Reference, qr, taps_q)?,
        || Failure::new(CASE, "Q15 dot with full-scale data"),
    )?;
    // Silence gives zero; an empty slice is zero; unequal lengths are an error.
    let zeros = c.take::<CQ15>(N)?;
    ensure(
        Dot::<CQ15>::dot(&Sdrk, zeros, taps_q)? == CQ15::default(),
        || Failure::new(CASE, "silence"),
    )?;
    ensure(
        Dot::<CQ15>::dot(&Sdrk, &[], &[])? == CQ15::default(),
        || Failure::new(CASE, "empty dot"),
    )?;
    ensure(Dot::<C32>::dot(&Sdrk, x, &[0.0; 3]).is_err(), || {
        Failure::new(CASE, "length mismatch accepted")
    })?;
    ensure(
        DotReal::<Q15>::dot_real(&Sdrk, qr, &[Q15(0)]).is_err(),
        || Failure::new(CASE, "length mismatch accepted"),
    )?;
    // NaN and infinity propagate the same way in the reference and the fallback.
    extreme::f32_specials_complex(x);
    extreme::f32_specials(xr);
    for t in taps_f.iter_mut() {
        *t = 0.5;
    }
    let a = Dot::<C32>::dot(&Sdrk, x, taps_f)?;
    let b = Dot::<C32>::dot(&Reference, x, taps_f)?;
    let same = |p: f32, r: f32| (p.is_nan() && r.is_nan()) || p == r;
    ensure(same(a.re, b.re) && same(a.im, b.im), || {
        Failure::new(CASE, "C32 dot with specials").detail(format_args!("{a} vs {b}"))
    })?;
    let a = DotReal::<f32>::dot_real(&Sdrk, xr, taps_f)?;
    let b = DotReal::<f32>::dot_real(&Reference, xr, taps_f)?;
    ensure(same(a, b), || Failure::new(CASE, "f32 dot with specials"))
}

// ---------------------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------------------

const DDC_RATE_HZ: u64 = 2_400_000;
const DDC_FREQ_HZ: f64 = 600_000.0;
const DDC_DECIMATION: usize = 10;
const DDC_INPUT: usize = 1000;

/// Memory for a block: the bytes `storage_bytes` asks for plus the alignment slack.
fn arena_for<'s>(c: &mut Carver<'s>, bytes: usize) -> Result<ArenaStorage<'s>, Failure> {
    Ok(ArenaStorage::new(c.take::<u8>(bytes + ARENA_ALIGN)?))
}

fn ddc_vs_chain_for<T>(arena: &mut Arena<'_>, seed: u64) -> Result<(), Failure>
where
    T: BlockSample + ComplexSample,
    Sdrk: Rotate<T> + FirDecimate<T> + sdrk_core::kernels::fused::Ddc<T>,
{
    const CASE: &str = "block_ddc_vs_chain";
    let rate = Rate::hz(DDC_RATE_HZ)?;
    let cfg = DdcConfig {
        frequency_hz: DDC_FREQ_HZ,
        decimation: DDC_DECIMATION,
        max_chunk: 128,
        ..DdcConfig::default()
    };
    let taps_n = cfg.num_taps(rate)?;
    let mut c = arena.carver();
    let mut storage = arena_for(
        &mut c,
        DdcBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?,
    )?;
    let mut block = DdcBlock::<T, _>::new(&cfg, rate, &mut storage)?;
    // The chain written out: design, convert to the tap type, rotate, decimate.
    let design = c.take::<f64>(taps_n)?;
    DesignSpec {
        cutoff_hz: 0.4 * DDC_RATE_HZ as f64 / DDC_DECIMATION as f64,
        transition_hz: 0.1 * DDC_RATE_HZ as f64 / DDC_DECIMATION as f64,
        attenuation_db: 60.0,
    }
    .design(DDC_RATE_HZ as f64, 1023, design)?;
    let taps = c.take::<T::Real>(taps_n)?;
    for (t, d) in taps.iter_mut().zip(design.iter()) {
        *t = <T as TunedElem>::tap(*d)?;
    }
    let hist = c.take::<T>(taps_n - 1)?;
    let mut nco = Nco::new(-DDC_FREQ_HZ, rate)?;
    let mut fir = FirDecimState::<T, _, _>::new(&*taps, hist, DDC_DECIMATION)?;
    let input = c.take::<T>(DDC_INPUT)?;
    let rotated = c.take::<T>(DDC_INPUT)?;
    let want = c.take::<T>(DDC_INPUT / DDC_DECIMATION)?;
    let got = c.take::<T>(DDC_INPUT / DDC_DECIMATION + 8)?;
    let mut rng = Pcg32::new(seed, 5);
    for s in input.iter_mut() {
        *s = T::from_c64_round(Complex64::new(
            (rng.next_f64() * 2.0 - 1.0) * 0.8,
            (rng.next_f64() * 2.0 - 1.0) * 0.8,
        ));
    }
    Rotate::<T>::rotate(&Sdrk, &mut nco, input, rotated)?;
    let p = FirDecimate::<T>::fir_decimate(&Sdrk, &mut fir, rotated, want)?;
    ensure(p.produced == DDC_INPUT / DDC_DECIMATION, || {
        Failure::new(CASE, "chain output count")
    })?;
    // The block in chunks of random size.
    let (mut produced, mut pos) = (0usize, 0usize);
    for size in crate::invariance::ChunkPlan::new(seed + 1, 77).sizes() {
        if pos >= DDC_INPUT {
            break;
        }
        let end = (pos + size).min(DDC_INPUT);
        let mut rest = input.get(pos..end).unwrap_or_default();
        while !rest.is_empty() {
            let room = got.get_mut(produced..).unwrap_or_default();
            let p = block.process(rest, room)?;
            if p.consumed == 0 && p.produced == 0 {
                return Err(Failure::new(CASE, "block made no progress"));
            }
            produced += p.produced;
            rest = rest.get(p.consumed..).unwrap_or_default();
        }
        pos = end;
    }
    ensure(produced == p.produced, || {
        Failure::new(CASE, "block output count").detail(format_args!("{produced}"))
    })?;
    ensure(
        bit_exact(
            want.get(..produced).unwrap_or_default(),
            got.get(..produced).unwrap_or_default(),
        )
        .is_none(),
        || Failure::new(CASE, "block output differs from the chain of kernels"),
    )
}

pub(super) fn block_ddc_vs_chain(arena: &mut Arena<'_>) -> Result<(), Failure> {
    ddc_vs_chain_for::<C32>(arena, 910)?;
    ddc_vs_chain_for::<CQ15>(arena, 911)
}

/// Thresholds of the block-based receive chain: the same as the kernel-based chain of M08.
const RX_MIN_DB: [f64; 2] = [40.0, 35.0];
/// 0.1 s of signal at 2.4 MHz.
const RX_SAMPLES: usize = 240_000;

fn fm_demod_sinad_for<T>(arena: &mut Arena<'_>, min_db: f64) -> Result<(), Failure>
where
    T: e2e::E2eSample + BlockSample,
    Sdrk: sdrk_core::kernels::convert::Convert<T>
        + sdrk_core::kernels::fused::Ddc<T>
        + sdrk_core::kernels::fused::FmDemod<T>
        + sdrk_core::kernels::demod::QuadDemod<T>
        + sdrk_core::kernels::decim::ResampleReal<<T as ComplexSample>::Real>,
{
    const CASE: &str = "block_fm_demod_sinad";
    let m = e2e::measure_rx_blocks::<T>(arena, RX_SAMPLES)?;
    ensure(m.audio_samples == RX_SAMPLES / 50, || {
        Failure::new(CASE, "wrong audio sample count").detail(format_args!("{}", m.audio_samples))
    })?;
    ensure(m.sinad_db >= min_db, || {
        Failure::new(CASE, "SINAD below the threshold")
            .detail(format_args!("{} dB, need {min_db}", m.sinad_db))
    })?;
    ensure(libm::fabs(m.tone_hz - e2e::AUDIO_HZ) <= 0.1, || {
        Failure::new(CASE, "audio frequency off").detail(format_args!("{} Hz", m.tone_hz))
    })?;
    ensure(m.amplitude > 0.3, || {
        Failure::new(CASE, "audio amplitude too low").detail(format_args!("{}", m.amplitude))
    })
}

pub(super) fn block_fm_demod_sinad(arena: &mut Arena<'_>) -> Result<(), Failure> {
    fm_demod_sinad_for::<C32>(arena, RX_MIN_DB[0])?;
    fm_demod_sinad_for::<CQ15>(arena, RX_MIN_DB[1])
}

const FFT: usize = 1024;

fn spectrum_level_for<T>(arena: &mut Arena<'_>, tolerance_db: f64) -> Result<(), Failure>
where
    T: sdrk_blocks::SpectrumSample,
    Sdrk: sdrk_core::kernels::spectrum::Window<T>
        + sdrk_core::kernels::fft::Fft<T>
        + sdrk_core::kernels::spectrum::PowerDb<f32>,
{
    const CASE: &str = "block_spectrum_tone_level";
    let rate = Rate::hz(1_024_000)?;
    let cfg = SpectrumConfig {
        fft_size: FFT,
        averaging: 2,
        ..SpectrumConfig::default()
    };
    let mut c = arena.carver();
    let mut storage = arena_for(
        &mut c,
        SpectrumBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?,
    )?;
    let mut block = SpectrumBlock::<T, _>::new(&cfg, rate, &mut storage)?;
    let frame = c.take::<f32>(FFT)?;
    let piece = c.take::<T>(128)?;
    let amplitude = 1.0 - 1.0 / 32768.0;
    for bin in [100usize, FFT - 37] {
        block.reset();
        let mut meta = None;
        let mut fed = 0usize;
        while meta.is_none() && fed < 4 * FFT {
            for (i, s) in piece.iter_mut().enumerate() {
                let phase = TAU * (bin * (fed + i)) as f64 / FFT as f64;
                *s = T::from_c64_round(Complex64::new(
                    amplitude * libm::cos(phase),
                    amplitude * libm::sin(phase),
                ));
            }
            let mut rest: &[T] = piece;
            while !rest.is_empty() && meta.is_none() {
                let (used, m) = block.push(rest, frame)?;
                meta = m;
                rest = rest.get(used..).unwrap_or_default();
            }
            fed += 128 - rest.len();
        }
        ensure(meta.is_some(), || Failure::new(CASE, "no frame"))?;
        let index = (FFT / 2 + bin) % FFT;
        let peak =
            frame
                .iter()
                .copied()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, v)| {
                    if v > best.1 { (i, v) } else { best }
                });
        ensure(peak.0 == index, || {
            Failure::new(CASE, "tone in the wrong bin")
                .detail(format_args!("{} vs {index}", peak.0))
        })?;
        ensure(libm::fabs(f64::from(peak.1)) <= tolerance_db, || {
            Failure::new(CASE, "tone level off").detail(format_args!("{} dBFS", peak.1))
        })?;
    }
    Ok(())
}

pub(super) fn block_spectrum_tone_level(arena: &mut Arena<'_>) -> Result<(), Failure> {
    spectrum_level_for::<C32>(arena, 0.05)?;
    spectrum_level_for::<CQ15>(arena, 0.3)
}

fn ppm_count_for<T>(arena: &mut Arena<'_>, seed: u64) -> Result<(), Failure>
where
    T: BlockSample,
    Sdrk: sdrk_core::kernels::decim::Resample<T>,
{
    const CASE: &str = "block_resampler_ppm_count";
    const SAMPLES: usize = 20_000;
    const PIECE: usize = 250;
    let rate = Rate::hz(48_000)?;
    let cfg = ResamplerConfig {
        tunable: true,
        ppm: 1000.0,
        ..ResamplerConfig::default()
    };
    let mut c = arena.carver();
    let mut storage = arena_for(
        &mut c,
        ResamplerBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?,
    )?;
    let mut block = ResamplerBlock::<T, _>::new(&cfg, rate, &mut storage)?;
    let input = c.take::<T>(PIECE)?;
    let out = c.take::<T>(PIECE + 16)?;
    let mut rng = Pcg32::new(seed, 6);
    for s in input.iter_mut() {
        *s = T::from_c64_round(Complex64::new(
            (rng.next_f64() - 0.5) * 0.5,
            (rng.next_f64() - 0.5) * 0.5,
        ));
    }
    let count = |block: &mut ResamplerBlock<T, _>, out: &mut [T]| -> Result<usize, Failure> {
        let mut produced = 0usize;
        for _ in 0..SAMPLES / PIECE {
            let mut rest: &[T] = input;
            while !rest.is_empty() {
                let p = block.process(rest, out)?;
                if p.consumed == 0 && p.produced == 0 {
                    return Err(Failure::new(CASE, "block made no progress"));
                }
                produced += p.produced;
                rest = rest.get(p.consumed..).unwrap_or_default();
            }
        }
        Ok(produced)
    };
    // +1000 ppm: 20 000 samples give 20 020 outputs (+-1).
    let n = count(&mut block, out)?;
    ensure(n.abs_diff(20_020) <= 1, || {
        Failure::new(CASE, "count at +1000 ppm").detail(format_args!("{n}"))
    })?;
    // Back to 0 ppm after a reset: exactly one output per input (+-1).
    block.reset();
    block.set_param("ppm", sdrk_blocks::ParamValue::F64(0.0))?;
    let n = count(&mut block, out)?;
    ensure(n.abs_diff(20_000) <= 1, || {
        Failure::new(CASE, "count at 0 ppm").detail(format_args!("{n}"))
    })?;
    block.reset();
    block.set_param("ppm", sdrk_blocks::ParamValue::F64(-1000.0))?;
    let n = count(&mut block, out)?;
    ensure(n.abs_diff(19_980) <= 1, || {
        Failure::new(CASE, "count at -1000 ppm").detail(format_args!("{n}"))
    })
}

pub(super) fn block_resampler_ppm_count(arena: &mut Arena<'_>) -> Result<(), Failure> {
    ppm_count_for::<C32>(arena, 920)?;
    ppm_count_for::<CQ15>(arena, 921)
}
