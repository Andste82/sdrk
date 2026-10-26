//! Suite cases of M21: the `ssb_demod` block in both sidebands, the run-time switch of the
//! sideband and the end-to-end SSB receive.
//!
//! The block cases need no files, so they run on microcontrollers too.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::{
    ArenaStorage, Block, BlockSample, ParamValue, SsbDemodBlock, SsbDemodConfig, SsbSideband,
};
use sdrk_core::kernels::agc::Agc;
use sdrk_core::kernels::convert::Convert;
use sdrk_core::kernels::decim::FirDecimate;
use sdrk_core::kernels::fused::Ddc;
use sdrk_core::kernels::mix::Rotate;
use sdrk_core::kernels::squelch::Squelch;
use sdrk_core::{C32, CQ15, ComplexSample, Rate, RealSample};

use super::ensure;
use crate::arena::Arena;
use crate::e2e::E2eSample;
use crate::e2e_ssb::{
    SSB_FULL_SAMPLES, SSB_MIN_REJECTION_F32_DB, SSB_MIN_REJECTION_Q15_DB, SSB_MIN_SINAD_F32_DB,
    SSB_MIN_SINAD_Q15_DB, SSB_QUICK_SAMPLES, measure_ssb,
};
use crate::r#gen::{Sideband, ssb_tone};
use crate::meters::ToneAccumulator;
use crate::suite::Failure;

const FS: u64 = 48_000;
/// Input samples of the block cases.
const N: usize = 2048;
/// Samples measured at the end of a block: 20 ms, a whole number of periods of every test tone
/// (400, 1000, 1500 and 2500 Hz), so that the fits of the tones do not leak into each other.
const WINDOW: usize = 960;
/// Amplitude of each tone.
const AMPLITUDE: f64 = 0.3;

fn rate() -> Result<Rate, Failure> {
    Ok(Rate::hz(FS)?)
}

/// A tone of `first_hz` on the side `wanted` plus a tone of 1.5 kHz on the other side, both
/// with amplitude 0.3, converted to the sample type.
fn two_sided<T: ComplexSample>(
    x: &mut [T],
    scratch: &mut [C32],
    wanted: Sideband,
    first_hz: f64,
    start: u64,
) -> Result<(), Failure> {
    let other = match wanted {
        Sideband::Usb => Sideband::Lsb,
        Sideband::Lsb => Sideband::Usb,
    };
    let fs = rate()?;
    ssb_tone(scratch, fs, first_hz, wanted, AMPLITUDE, start);
    for (d, s) in x.iter_mut().zip(scratch.iter()) {
        *d = T::from_c64_round(Complex64::new(f64::from(s.re), f64::from(s.im)));
    }
    // The other tone: add in f64 before the single rounding.
    ssb_tone(scratch, fs, 1_500.0, other, AMPLITUDE, start);
    for (d, s) in x.iter_mut().zip(scratch.iter()) {
        let c = d.to_c64();
        *d = T::from_c64_round(Complex64::new(
            c.re + f64::from(s.re),
            c.im + f64::from(s.im),
        ));
    }
    Ok(())
}

/// Tone amplitude in dB re `AMPLITUDE`.
fn level_db(amplitude: f64) -> f64 {
    20.0 * libm::log10(amplitude / AMPLITUDE)
}

fn side_for<T>(
    case: &'static str,
    arena: &mut Arena<'_>,
    sideband: SsbSideband,
    wanted: Sideband,
) -> Result<(), Failure>
where
    T: BlockSample,
    Sdrk: Rotate<T> + FirDecimate<T>,
{
    let fail = |msg: &'static str| Failure::new(case, msg);
    let cfg = SsbDemodConfig {
        sideband,
        ..SsbDemodConfig::default()
    };
    let bytes = SsbDemodBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, rate()?)?
        + sdrk_blocks::ARENA_ALIGN;
    let mut c = arena.carver();
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let input = c.take::<T>(N)?;
    let scratch = c.take::<C32>(N)?;
    let out = c.take::<T::Real>(N)?;
    let mut block = SsbDemodBlock::<T, _>::new(&cfg, rate()?, &mut store)?;
    for hz in [400.0, 1_000.0, 2_500.0] {
        block.reset();
        two_sided(input, scratch, wanted, hz, 0)?;
        let p = block.process(input, out)?;
        ensure(p.consumed == N && p.produced == N, || {
            fail("block did not process the whole input")
        })?;
        // The wanted tone (at the audio frequency `hz`) and the opposite one (1.5 kHz) over the
        // last 20 ms, after the filter has settled.
        let (mut tone, mut opposite) = (
            ToneAccumulator::new(FS as f64, hz),
            ToneAccumulator::new(FS as f64, 1_500.0),
        );
        for v in out.iter().skip(N - WINDOW) {
            tone.push(v.to_f64());
            opposite.push(v.to_f64());
        }
        ensure(level_db(tone.amplitude()).abs() <= 0.5, || {
            fail("the tone level is off by more than 0.5 dB")
                .detail(format_args!("{hz} Hz: {} dB", level_db(tone.amplitude())))
        })?;
        // The tone of the opposite sideband (1.5 kHz) is gone.
        ensure(
            20.0 * libm::log10(tone.amplitude() / opposite.amplitude().max(1e-12)) >= 40.0,
            || {
                fail("the opposite sideband is not rejected by 40 dB")
                    .detail(format_args!("{hz} Hz"))
            },
        )?;
    }
    Ok(())
}

pub(super) fn block_ssb_demod_usb(arena: &mut Arena<'_>) -> Result<(), Failure> {
    side_for::<C32>(
        "block_ssb_demod_usb",
        arena,
        SsbSideband::Usb,
        Sideband::Usb,
    )?;
    side_for::<CQ15>(
        "block_ssb_demod_usb",
        arena,
        SsbSideband::Usb,
        Sideband::Usb,
    )
}

pub(super) fn block_ssb_demod_lsb(arena: &mut Arena<'_>) -> Result<(), Failure> {
    side_for::<C32>(
        "block_ssb_demod_lsb",
        arena,
        SsbSideband::Lsb,
        Sideband::Lsb,
    )?;
    side_for::<CQ15>(
        "block_ssb_demod_lsb",
        arena,
        SsbSideband::Lsb,
        Sideband::Lsb,
    )
}

fn switch_for<T>(arena: &mut Arena<'_>) -> Result<(), Failure>
where
    T: BlockSample,
    Sdrk: Rotate<T> + FirDecimate<T>,
{
    const CASE: &str = "block_ssb_sideband_switch";
    let fail = |msg: &'static str| Failure::new(CASE, msg);
    let cfg = SsbDemodConfig::default();
    let bytes = SsbDemodBlock::<T, ArenaStorage<'_>>::storage_bytes(&cfg, rate()?)?
        + sdrk_blocks::ARENA_ALIGN;
    let mut c = arena.carver();
    let mut store = ArenaStorage::new(c.take::<u8>(bytes)?);
    let input = c.take::<T>(N)?;
    let scratch = c.take::<C32>(N)?;
    let out = c.take::<T::Real>(N)?;
    let mut block = SsbDemodBlock::<T, _>::new(&cfg, rate()?, &mut store)?;
    // One signal with the tone at 1 kHz above the carrier and one at 1.5 kHz below it,
    // continuous over both calls.
    two_sided(input, scratch, Sideband::Usb, 1_000.0, 0)?;
    block.process(input, out)?;
    let mut last = out.last().map_or(0.0, |v| v.to_f64());
    // From the next call on the lower sideband passes.
    block.set_param("sideband", ParamValue::Choice(SsbSideband::Lsb.index()))?;
    ensure(
        block.param("sideband") == Ok(ParamValue::Choice(SsbSideband::Lsb.index())),
        || fail("the sideband is not reported"),
    )?;
    two_sided(input, scratch, Sideband::Usb, 1_000.0, N as u64)?;
    let p = block.process(input, out)?;
    ensure(p.produced == N, || fail("second call produced no output"))?;
    let mut worst = 0.0f64;
    let mut nan = false;
    let mut tone = ToneAccumulator::new(FS as f64, 1_500.0);
    for (i, v) in out.iter().enumerate() {
        let x = v.to_f64();
        nan |= !x.is_finite();
        worst = worst.max(libm::fabs(x - last));
        last = x;
        if i >= N - WINDOW {
            tone.push(x);
        }
    }
    ensure(!nan, || fail("the output contains NaN or infinity"))?;
    ensure(worst <= AMPLITUDE, || {
        fail("the output jumps at the switch").detail(format_args!("{worst}"))
    })?;
    // After the transient the lower-sideband tone passes with its amplitude.
    ensure(level_db(tone.amplitude()).abs() <= 0.5, || {
        fail("the lower sideband tone is not at its level after the switch")
            .detail(format_args!("{} dB", level_db(tone.amplitude())))
    })?;
    // A reset keeps the sideband and starts both phases at 0 again: the next call equals a
    // fresh block's.
    block.reset();
    two_sided(input, scratch, Sideband::Usb, 1_000.0, 0)?;
    block.process(input, out)?;
    ensure(out.iter().all(|v| v.to_f64().is_finite()), || {
        fail("the output after reset is not finite")
    })
}

pub(super) fn block_ssb_sideband_switch(arena: &mut Arena<'_>) -> Result<(), Failure> {
    switch_for::<C32>(arena)?;
    switch_for::<CQ15>(arena)
}

// ---------------------------------------------------------------------------------------
// end to end
// ---------------------------------------------------------------------------------------

/// The upper sideband (the role of the example flow), then the lower one with the tones swapped.
const BOTH: [SsbSideband; 2] = [SsbSideband::Usb, SsbSideband::Lsb];

fn ssb_case<T>(
    case: &'static str,
    arena: &mut Arena<'_>,
    samples: usize,
    min_sinad_db: f64,
    min_rejection_db: f64,
    sides: &[SsbSideband],
) -> Result<(), Failure>
where
    T: E2eSample + BlockSample,
    Sdrk: Convert<T> + Ddc<T> + Squelch<T> + Agc<T> + Rotate<T> + FirDecimate<T>,
{
    for &side in sides {
        let m = measure_ssb::<T>(arena, samples, side)?;
        ensure(m.audio_samples == samples / 50, || {
            Failure::new(case, "wrong audio sample count")
                .detail(format_args!("{}", m.audio_samples))
        })?;
        ensure(m.sinad_db >= min_sinad_db, || {
            Failure::new(case, "SINAD below the threshold")
                .detail(format_args!("{} dB, need {min_sinad_db}", m.sinad_db))
        })?;
        ensure(m.rejection_db >= min_rejection_db, || {
            Failure::new(case, "the opposite sideband is not rejected enough").detail(format_args!(
                "{} dB, need {min_rejection_db}",
                m.rejection_db
            ))
        })?;
    }
    Ok(())
}

pub(super) fn e2e_ssb_rx_f32(arena: &mut Arena<'_>) -> Result<(), Failure> {
    ssb_case::<C32>(
        "e2e_ssb_rx_f32",
        arena,
        SSB_QUICK_SAMPLES,
        SSB_MIN_SINAD_F32_DB,
        SSB_MIN_REJECTION_F32_DB,
        &BOTH,
    )
}

/// The quick Q15 case receives the upper sideband only: on the emulators the double-precision
/// signal generation dominates the run time, and the lower sideband is covered by the full-tier
/// cases and by the host tests.
pub(super) fn e2e_ssb_rx_q15(arena: &mut Arena<'_>) -> Result<(), Failure> {
    ssb_case::<CQ15>(
        "e2e_ssb_rx_q15",
        arena,
        SSB_QUICK_SAMPLES,
        SSB_MIN_SINAD_Q15_DB,
        SSB_MIN_REJECTION_Q15_DB,
        &[SsbSideband::Usb],
    )
}

pub(super) fn e2e_ssb_rx_f32_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    ssb_case::<C32>(
        "e2e_ssb_rx_f32_1s",
        arena,
        SSB_FULL_SAMPLES,
        SSB_MIN_SINAD_F32_DB,
        SSB_MIN_REJECTION_F32_DB,
        &BOTH,
    )
}

pub(super) fn e2e_ssb_rx_q15_1s(arena: &mut Arena<'_>) -> Result<(), Failure> {
    ssb_case::<CQ15>(
        "e2e_ssb_rx_q15_1s",
        arena,
        SSB_FULL_SAMPLES,
        SSB_MIN_SINAD_Q15_DB,
        SSB_MIN_REJECTION_Q15_DB,
        &BOTH,
    )
}
