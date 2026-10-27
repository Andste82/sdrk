//! End-to-end SSB receive (E2E-SSB of M21), built from blocks.
//!
//! The signal at 2.4 MS/s holds two equally strong tones (amplitude 0.3) on both sides of a
//! suppressed carrier at an offset of +300 kHz: one 1 kHz above the carrier (upper sideband) and
//! one 1.5 kHz below it (lower sideband), plus complex Gaussian noise (sigma 0.003 per
//! component), encoded to `CU8` with the independent testkit encoder. The receiver is the chain
//! of the example flow `ssb_receiver`: `ConvertBlock`, `DdcBlock` (300 kHz, decimation 50,
//! cutoff 4 kHz, transition 2 kHz, up to 8192 taps), `SquelchBlock` (-50 dB), `AgcBlock` (the
//! defaults) and `SsbDemodBlock` (the defaults, with the sideband of the run). Receiving the
//! upper sideband makes the 1 kHz tone the wanted one; receiving the lower sideband swaps the
//! roles (the wanted tone is 1 kHz below the carrier, the rejected one 1.5 kHz above it).
//!
//! [`measure_ssb`] streams the signal through the chain in pieces of 100 samples and keeps no
//! audio buffer: the SINAD at 1 kHz and the amplitudes of the 1 kHz and the 1.5 kHz components
//! are accumulated over the last half of the audio while it is produced.

// See `e2e.rs`: the lint counts every `Result<_, Failure>` temporary separately.
#![allow(
    clippy::large_stack_frames,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::{
    ARENA_ALIGN, AgcBlock, AgcConfig, ArenaStorage, Block, BlockSample, ConvertBlock,
    ConvertConfig, DdcBlock, DdcConfig, SquelchBlock, SquelchConfig, SsbDemodBlock, SsbDemodConfig,
    SsbSideband,
};
use sdrk_core::kernels::agc::Agc;
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::FirDecimate;
use sdrk_core::kernels::fused::Ddc;
use sdrk_core::kernels::mix::Rotate;
use sdrk_core::kernels::squelch::Squelch;
use sdrk_core::{C32, ComplexSample, Rate, RealSample};
use sdrk_device::formats;

use crate::arena::Arena;
use crate::e2e::{E2eSample, IF_RATE_HZ};
use crate::encode::encode;
use crate::r#gen::{Sideband, ssb_tone};
use crate::meters::ToneAccumulator;
use crate::rng::Pcg32;
use crate::suite::Failure;

const CASE: &str = "e2e_ssb_rx";

/// Input samples per step through the chain (small, to keep the arena below 80 KiB).
const PIECE: usize = 100;

/// Offset of the suppressed carrier from the tuner centre in Hz.
pub const SSB_OFFSET_HZ: f64 = 300_000.0;
/// Frequency of the wanted tone from the carrier in Hz.
pub const SSB_WANTED_HZ: f64 = 1_000.0;
/// Frequency of the tone in the opposite sideband from the carrier in Hz.
pub const SSB_OPPOSITE_HZ: f64 = 1_500.0;
/// Amplitude of each tone relative to full scale.
pub const SSB_AMPLITUDE: f64 = 0.3;
/// Standard deviation of the noise per component.
pub const SSB_NOISE_SIGMA: f64 = 0.003;
/// Decimation of the channel filter (2.4 MS/s to 48 kHz).
pub const SSB_DECIMATION: usize = 50;
/// Audio sample rate in Hz.
pub const SSB_AUDIO_RATE_HZ: f64 = 48_000.0;
/// Cutoff of the channel filter in Hz.
pub const SSB_CHANNEL_CUTOFF_HZ: f64 = 4_000.0;
/// Transition width of the channel filter in Hz.
pub const SSB_CHANNEL_TRANSITION_HZ: f64 = 2_000.0;
/// Largest tap count of the channel filter (the design needs about 4400).
pub const SSB_CHANNEL_MAX_TAPS: usize = 8192;
/// Threshold of the squelch in dBFS.
pub const SSB_SQUELCH_DB: f64 = -50.0;
/// Quick variant: 0.2 s at 2.4 MS/s.
pub const SSB_QUICK_SAMPLES: usize = 480_000;
/// Full variant: 1.0 s at 2.4 MS/s.
pub const SSB_FULL_SAMPLES: usize = 2_400_000;
/// SINAD threshold for `C32` in dB.
pub const SSB_MIN_SINAD_F32_DB: f64 = 35.0;
/// SINAD threshold for `CQ15` in dB.
pub const SSB_MIN_SINAD_Q15_DB: f64 = 30.0;
/// Rejection of the opposite sideband for `C32` in dB (the 1.5 kHz component below the 1 kHz tone).
pub const SSB_MIN_REJECTION_F32_DB: f64 = 50.0;
/// Rejection of the opposite sideband for `CQ15` in dB.
pub const SSB_MIN_REJECTION_Q15_DB: f64 = 40.0;

/// What [`measure_ssb`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsbMeasurement {
    /// SINAD at 1 kHz over the last half of the audio in dB.
    pub sinad_db: f64,
    /// Amplitude of the 1 kHz tone over the last half of the audio (full scale is 1.0).
    pub tone_amplitude: f64,
    /// Amplitude of the 1.5 kHz component over the last half of the audio.
    pub opposite_amplitude: f64,
    /// `20 log10(tone_amplitude / opposite_amplitude)` in dB: how far the opposite sideband is
    /// below the wanted tone.
    pub rejection_db: f64,
    /// Audio samples produced in total.
    pub audio_samples: usize,
}

/// The configuration of the channel filter, shared by the example flow `ssb_receiver` and the
/// E2E case.
pub fn ssb_ddc_config(max_chunk: usize) -> DdcConfig {
    DdcConfig {
        frequency_hz: SSB_OFFSET_HZ,
        decimation: SSB_DECIMATION,
        cutoff_hz: Some(SSB_CHANNEL_CUTOFF_HZ),
        transition_hz: Some(SSB_CHANNEL_TRANSITION_HZ),
        max_taps: SSB_CHANNEL_MAX_TAPS,
        max_chunk,
        ..DdcConfig::default()
    }
}

/// Runs E2E-SSB over `samples` input samples (a multiple of 100) with the sample type `T`,
/// receiving `side`.
///
/// # Errors
/// A [`Failure`] if the arena is exhausted, a block fails or `samples` is not a multiple of
/// 100.
pub fn measure_ssb<T>(
    arena: &mut Arena<'_>,
    samples: usize,
    side: SsbSideband,
) -> Result<SsbMeasurement, Failure>
where
    T: E2eSample + BlockSample,
    Sdrk: Convert<T> + Ddc<T> + Squelch<T> + Agc<T> + Rotate<T> + FirDecimate<T>,
{
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    if samples == 0 || !samples.is_multiple_of(PIECE) {
        return Err(fail("sample count must be a multiple of 100"));
    }
    let (wanted_side, opposite_side) = match side {
        SsbSideband::Lsb => (Sideband::Lsb, Sideband::Usb),
        _ => (Sideband::Usb, Sideband::Lsb),
    };
    let if_rate = Rate::hz(IF_RATE_HZ)?;
    let audio_rate = Rate::hz(48_000)?;
    let ddc_cfg = ssb_ddc_config(PIECE);
    let squelch_cfg = SquelchConfig {
        threshold_db: SSB_SQUELCH_DB,
        ..SquelchConfig::default()
    };
    let agc_cfg = AgcConfig::default();
    let ssb_cfg = SsbDemodConfig {
        sideband: side,
        ..SsbDemodConfig::default()
    };
    let mut c = arena.carver();
    let ddc_bytes =
        DdcBlock::<T, ArenaStorage<'_>>::storage_bytes(&ddc_cfg, if_rate)? + ARENA_ALIGN;
    let ssb_bytes =
        SsbDemodBlock::<T, ArenaStorage<'_>>::storage_bytes(&ssb_cfg, audio_rate)? + ARENA_ALIGN;
    let mut ddc_store = ArenaStorage::new(c.take::<u8>(ddc_bytes)?);
    let mut ssb_store = ArenaStorage::new(c.take::<u8>(ssb_bytes)?);
    let mut convert = ConvertBlock::<T, _>::new(
        &ConvertConfig {
            format: formats::CU8,
        },
        if_rate,
        &mut ArenaStorage::new(&mut []),
    )?;
    let mut ddc = DdcBlock::<T, _>::new(&ddc_cfg, if_rate, &mut ddc_store)?;
    let mut gate =
        SquelchBlock::<T, _>::new(&squelch_cfg, audio_rate, &mut ArenaStorage::new(&mut []))?;
    let mut agc = AgcBlock::<T, _>::new(&agc_cfg, audio_rate, &mut ArenaStorage::new(&mut []))?;
    let mut ssb = SsbDemodBlock::<T, _>::new(&ssb_cfg, audio_rate, &mut ssb_store)?;

    let wanted = c.take::<C32>(PIECE)?;
    let opposite = c.take::<C32>(PIECE)?;
    let scratch = c.take::<Complex64>(PIECE)?;
    let bytes = c.take::<u8>(formats::CU8.bytes_for_samples(PIECE)?)?;
    let iq = c.take::<T>(PIECE)?;
    let low = c.take::<T>(PIECE / SSB_DECIMATION + 4)?;
    let gated = c.take::<T>(PIECE / SSB_DECIMATION + 4)?;
    let leveled = c.take::<T>(PIECE / SSB_DECIMATION + 4)?;
    let out = c.take::<<T as ComplexSample>::Real>(PIECE / SSB_DECIMATION + 4)?;

    let audio_len = samples / SSB_DECIMATION;
    let window_start = audio_len / 2;

    let mut rng = Pcg32::new(2100, 1);
    let mut meter = ToneAccumulator::new(SSB_AUDIO_RATE_HZ, SSB_WANTED_HZ);
    let mut other = ToneAccumulator::new(SSB_AUDIO_RATE_HZ, SSB_OPPOSITE_HZ);
    let mut audio_index = 0usize;

    for piece in 0..samples / PIECE {
        // Source: both tones around the suppressed carrier, shifted to the offset, plus noise,
        // encoded to CU8.
        let start = piece * PIECE;
        ssb_tone(
            wanted,
            if_rate,
            SSB_WANTED_HZ,
            wanted_side,
            SSB_AMPLITUDE,
            start as u64,
        );
        ssb_tone(
            opposite,
            if_rate,
            SSB_OPPOSITE_HZ,
            opposite_side,
            SSB_AMPLITUDE,
            start as u64,
        );
        for (i, ((a, b), d)) in wanted
            .iter()
            .zip(opposite.iter())
            .zip(scratch.iter_mut())
            .enumerate()
        {
            let cycles = SSB_OFFSET_HZ * (start + i) as f64 / IF_RATE_HZ as f64;
            let w = 2.0 * PI * (cycles - libm::floor(cycles));
            let sum = Complex64::new(
                f64::from(a.re) + f64::from(b.re),
                f64::from(a.im) + f64::from(b.im),
            );
            *d = sum * Complex64::new(libm::cos(w), libm::sin(w));
            d.re += SSB_NOISE_SIGMA * rng.normal();
            d.im += SSB_NOISE_SIGMA * rng.normal();
        }
        encode(&formats::CU8, scratch, bytes)?;
        let p = convert.process(bytes, iq)?;
        if p.consumed != bytes.len() || p.produced != PIECE {
            return Err(fail("convert did not take the whole piece"));
        }
        let mut rest: &[T] = iq;
        while !rest.is_empty() {
            let p = ddc.process(rest, low)?;
            if p.consumed == 0 && p.produced == 0 {
                return Err(fail("ddc made no progress"));
            }
            rest = rest.get(p.consumed..).unwrap_or_default();
            let produced = low.get(..p.produced).unwrap_or_default();
            let g = gate.process(produced, gated)?;
            if g.consumed != p.produced || g.produced != p.produced {
                return Err(fail("squelch did not take the whole block"));
            }
            let gated_part = gated.get(..g.produced).unwrap_or_default();
            let q = agc.process(gated_part, leveled)?;
            if q.consumed != g.produced || q.produced != g.produced {
                return Err(fail("agc did not take the whole block"));
            }
            let mut a_rest = leveled.get(..q.produced).unwrap_or_default();
            while !a_rest.is_empty() {
                let r = ssb.process(a_rest, out)?;
                if r.consumed == 0 && r.produced == 0 {
                    return Err(fail("ssb_demod made no progress"));
                }
                for v in out.iter().take(r.produced) {
                    if audio_index >= window_start {
                        meter.push(v.to_f64());
                        other.push(v.to_f64());
                    }
                    audio_index += 1;
                }
                a_rest = a_rest.get(r.consumed..).unwrap_or_default();
            }
        }
    }
    let tone_amplitude = meter.amplitude();
    let opposite_amplitude = other.amplitude();
    Ok(SsbMeasurement {
        sinad_db: meter.sinad_db(),
        tone_amplitude,
        opposite_amplitude,
        rejection_db: 20.0 * libm::log10(tone_amplitude / opposite_amplitude),
        audio_samples: audio_index,
    })
}
