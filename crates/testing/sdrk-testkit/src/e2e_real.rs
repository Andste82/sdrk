//! End-to-end FM receive from a real-sampling input (E2E-REAL of M23), built from blocks.
//!
//! The signal of E2E-RX (deviation 75 kHz, audio 1 kHz) sits at a real intermediate frequency of
//! 1.8 MHz at 4.8 MS/s: `Re{fm_baseband(..) e^(j 2 pi 1.8e6 n / fs)}` with the amplitude 0.45,
//! plus real Gaussian noise (sigma 0.005), encoded to `RS16` with the independent testkit
//! encoder. The receiver is the chain of the example flow `real_input_fm_receiver`:
//! `ConvertRealBlock` (`rs16`), `RealToIqBlock` (the defaults: transition 0.1 fs, 60 dB; the
//! output centre is 1.2 MHz of the input), `DdcBlock` (+600 kHz, decimation 10) and
//! `FmDemodBlock` (75 kHz, 50 us, 15 kHz, decimation 5).
//!
//! [`measure_real_rx`] streams the signal in blocks of [`REAL_CHUNK`] real samples and keeps only
//! the last half of the audio (for the SINAD at 1 kHz).

// See `e2e.rs`: the lint counts every `Result<_, Failure>` temporary separately.
#![allow(clippy::large_stack_frames, clippy::too_many_lines)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::{
    ARENA_ALIGN, ArenaStorage, Block, BlockReal, BlockSample, ConvertConfig, ConvertRealBlock,
    DdcBlock, DdcConfig, FmDemodBlock, FmDemodConfig, HalfbandDecimateConfig, RealToIqBlock,
};
use sdrk_core::kernels::convert::ConvertReal;
use sdrk_core::kernels::decim::ResampleReal;
use sdrk_core::kernels::demod::QuadDemod;
use sdrk_core::kernels::fused::{Ddc, FmDemod};
use sdrk_core::kernels::halfband::RealToComplex;
use sdrk_core::{ComplexSample, Rate, RealSample};
use sdrk_device::formats;

use crate::arena::Arena;
use crate::e2e::{AUDIO_HZ, DEEMPHASIS_TAU_S, DEVIATION_HZ, E2eSample, Measurement, measure_audio};
use crate::encode::encode;
use crate::fm::fm_baseband;
use crate::rng::Pcg32;
use crate::suite::Failure;

const CASE: &str = "e2e_real_fm_rx";

/// Real samples per block (a multiple of the total decimation of 100).
pub const REAL_CHUNK: usize = 1_000;
/// Sample rate of the real input in Hz.
pub const REAL_RATE_HZ: u64 = 4_800_000;
/// Real intermediate frequency of the station in Hz.
pub const REAL_IF_HZ: f64 = 1_800_000.0;
/// Amplitude of the received test signal relative to full scale.
pub const REAL_SIGNAL_AMPLITUDE: f64 = 0.45;
/// Standard deviation of the real noise.
pub const REAL_NOISE_SIGMA: f64 = 0.005;
/// Quick variant: 0.1 s at 4.8 MS/s.
pub const REAL_QUICK_SAMPLES: usize = 480_000;
/// Full variant: 1.0 s at 4.8 MS/s.
pub const REAL_FULL_SAMPLES: usize = 4_800_000;
/// SINAD threshold for the `f32` chain in dB.
pub const REAL_MIN_SINAD_F32_DB: f64 = 40.0;
/// SINAD threshold for the Q15 chain in dB.
pub const REAL_MIN_SINAD_Q15_DB: f64 = 35.0;

fn fail(msg: &'static str) -> Failure {
    Failure::new(CASE, msg)
}

/// Runs E2E-REAL over `samples` real input samples (a multiple of [`REAL_CHUNK`]) with the
/// complex sample type `T` (its real type is the type of the converted input) and measures the
/// last half of the audio. The arena needs about 24 KiB for the blocks and buffers plus 8 bytes
/// per audio sample of the second half (`samples / 200`).
///
/// # Errors
/// A [`Failure`] if the arena is exhausted, a block fails or `samples` is not a multiple of
/// [`REAL_CHUNK`].
pub fn measure_real_rx<T>(arena: &mut Arena<'_>, samples: usize) -> Result<Measurement, Failure>
where
    T: E2eSample + BlockSample,
    <T as ComplexSample>::Real: BlockReal,
    Sdrk: ConvertReal<<T as ComplexSample>::Real>
        + RealToComplex<T>
        + Ddc<T>
        + FmDemod<T>
        + QuadDemod<T>
        + ResampleReal<<T as ComplexSample>::Real>,
{
    if samples == 0 || !samples.is_multiple_of(REAL_CHUNK) {
        return Err(fail("sample count must be a multiple of the block size"));
    }
    let rate_real = Rate::hz(REAL_RATE_HZ)?;
    let rate_complex = Rate::hz(REAL_RATE_HZ / 2)?;
    let rate_audio_in = Rate::hz(REAL_RATE_HZ / 20)?;
    let half_cfg = HalfbandDecimateConfig::default();
    // The centre of the complex signal is 1.2 MHz of the input: tune f - 1.2 MHz.
    let ddc_cfg = DdcConfig {
        frequency_hz: REAL_IF_HZ - (REAL_RATE_HZ / 4) as f64,
        decimation: 10,
        // The cutoff of examples/flows/real_input_fm_receiver.{json,yaml}.
        cutoff_hz: Some(100_000.0),
        max_chunk: REAL_CHUNK / 2,
        ..DdcConfig::default()
    };
    let demod_cfg = FmDemodConfig {
        deviation_hz: DEVIATION_HZ,
        deemphasis_us: DEEMPHASIS_TAU_S * 1e6,
        audio_cutoff_hz: 15e3,
        audio_decimation: 5,
    };
    let mut c = arena.carver();
    let half_bytes =
        RealToIqBlock::<T, ArenaStorage<'_>>::storage_bytes(&half_cfg, rate_real)? + ARENA_ALIGN;
    let ddc_bytes =
        DdcBlock::<T, ArenaStorage<'_>>::storage_bytes(&ddc_cfg, rate_complex)? + ARENA_ALIGN;
    let demod_bytes =
        FmDemodBlock::<T, ArenaStorage<'_>>::storage_bytes(&demod_cfg, rate_audio_in)?
            + ARENA_ALIGN;
    let mut half_store = ArenaStorage::new(c.take::<u8>(half_bytes)?);
    let mut ddc_store = ArenaStorage::new(c.take::<u8>(ddc_bytes)?);
    let mut demod_store = ArenaStorage::new(c.take::<u8>(demod_bytes)?);
    let mut convert = ConvertRealBlock::<<T as ComplexSample>::Real, _>::new(
        &ConvertConfig {
            format: formats::RS16,
        },
        rate_real,
        &mut ArenaStorage::new(&mut []),
    )?;
    let mut iq_block = RealToIqBlock::<T, _>::new(&half_cfg, rate_real, &mut half_store)?;
    let mut ddc = DdcBlock::<T, _>::new(&ddc_cfg, rate_complex, &mut ddc_store)?;
    let mut demod = FmDemodBlock::<T, _>::new(&demod_cfg, rate_audio_in, &mut demod_store)?;

    let total_audio = samples / 100;
    let skip = total_audio / 2;
    let audio = c.take::<f64>(total_audio - skip)?;
    let scratch = c.take::<Complex64>(REAL_CHUNK)?;
    let bytes = c.take::<u8>(formats::RS16.bytes_for_samples(REAL_CHUNK)?)?;
    let real = c.take::<<T as ComplexSample>::Real>(REAL_CHUNK)?;
    let iq = c.take::<T>(REAL_CHUNK / 2 + 2)?;
    let low = c.take::<T>(REAL_CHUNK / 20 + 4)?;
    let out = c.take::<<T as ComplexSample>::Real>(REAL_CHUNK / 100 + 4)?;

    let mut rng = Pcg32::new(2300, 1);
    let mut produced_total = 0usize;
    for chunk in 0..samples / REAL_CHUNK {
        // Source: the station at the real IF, plus real noise, encoded to RS16.
        fm_baseband(
            scratch,
            REAL_RATE_HZ as f64,
            REAL_IF_HZ,
            DEVIATION_HZ,
            AUDIO_HZ,
            1.0,
            (chunk * REAL_CHUNK) as u64,
        );
        for s in scratch.iter_mut() {
            *s = Complex64::new(
                REAL_SIGNAL_AMPLITUDE * s.re + REAL_NOISE_SIGMA * rng.normal(),
                0.0,
            );
        }
        encode(&formats::RS16, scratch, bytes)?;
        let p = convert.process(bytes, real)?;
        if p.consumed != bytes.len() || p.produced != REAL_CHUNK {
            return Err(fail("convert did not take the whole block"));
        }
        let mut real_rest: &[<T as ComplexSample>::Real] = real;
        while !real_rest.is_empty() {
            let p = iq_block.process(real_rest, iq)?;
            if p.consumed == 0 && p.produced == 0 {
                return Err(fail("real_to_iq made no progress"));
            }
            real_rest = real_rest.get(p.consumed..).unwrap_or_default();
            let mut iq_rest = iq.get(..p.produced).unwrap_or_default();
            while !iq_rest.is_empty() {
                let q = ddc.process(iq_rest, low)?;
                if q.consumed == 0 && q.produced == 0 {
                    return Err(fail("ddc made no progress"));
                }
                iq_rest = iq_rest.get(q.consumed..).unwrap_or_default();
                let mut low_rest = low.get(..q.produced).unwrap_or_default();
                while !low_rest.is_empty() {
                    let r = demod.process(low_rest, out)?;
                    if r.consumed == 0 && r.produced == 0 {
                        return Err(fail("fm_demod made no progress"));
                    }
                    for v in out.iter().take(r.produced) {
                        let idx = produced_total;
                        produced_total += 1;
                        if let Some(slot) = idx.checked_sub(skip).and_then(|i| audio.get_mut(i)) {
                            *slot = v.to_f64();
                        }
                    }
                    low_rest = low_rest.get(r.consumed..).unwrap_or_default();
                }
            }
        }
    }
    Ok(measure_audio(audio, produced_total))
}
