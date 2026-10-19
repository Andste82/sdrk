//! End-to-end kernel chains: FM receive from raw device bytes and FM transmit.
//!
//! The chains stream in blocks of [`CHUNK`] samples at the intermediate-frequency rate, use only
//! caller-provided memory (a [`Carver`]) and run on the facade or any other kernel
//! implementation. They are used by the host tests (1.0 s of signal), by the suite cases (0.1 s)
//! and by later milestones as the reference for the block-level chains.
//!
//! Receive (`fm_rx`): raw bytes, `convert`, `ddc` (NCO at minus the carrier offset,
//! `fir_decimate` by 10), `fm_demod` at 240 kHz (de-emphasis 50 us), `resample_real` by 1/5 to
//! 48 kHz audio.
//!
//! [`fm_rx_blocks`] is the same receive chain built from the blocks of `sdrk-blocks`
//! (`ConvertBlock`, `DdcBlock`, `FmDemodBlock`).
//!
//! Transmit ([`FmTx`]): 48 kHz audio, `fir_interpolate_real` by 5, `fm_mod` at 240 kHz, `duc`
//! (interpolation by 10, NCO at the carrier offset) and `convert_out` to raw bytes at 2.4 MHz.

// The stack-frame lint counts every `Result<_, Failure>` temporary of an unoptimised build
// separately (a `Failure` is about 200 bytes by specification), so these functions exceed 4 KiB
// in the lint's model although the optimised frames are small. See docs/decisions-log.md (M03).
#![allow(clippy::large_stack_frames)]

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_blocks::{
    ARENA_ALIGN, ArenaStorage, Block, BlockSample, ConvertBlock, ConvertConfig, DdcBlock,
    DdcConfig, FmDemodBlock, FmDemodConfig,
};
use sdrk_core::kernels::convert::{Convert, ConvertOut};
use sdrk_core::kernels::decim::{FirDecimState, ResampleReal, ResampleRealState};
use sdrk_core::kernels::demod::{DeemphasisState, QuadDemod, QuadDemodState};
use sdrk_core::kernels::fused::{Ddc, Duc, FmDemod};
use sdrk_core::kernels::interp::{FirInterpRealState, FirInterpState, FirInterpolateReal};
use sdrk_core::kernels::mix::Nco;
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::{C32, CQ15, ComplexSample, Q15, Rate, RealSample};
use sdrk_device::SampleFormat;

use crate::arena::{Arena, Carver};
use crate::design_helpers::lowpass_hann;
use crate::encode::encode;
use crate::fm::fm_baseband;
use crate::r#gen::real_tone;
use crate::meters::{ToneMeter, fit_frequency, sinad_db};
use crate::suite::Failure;

const CASE: &str = "e2e";

/// Samples per block at the intermediate-frequency rate (a multiple of the total decimation of 50).
pub const CHUNK: usize = 500;
/// Intermediate-frequency sample rate in Hz.
pub const IF_RATE_HZ: u64 = 2_400_000;
/// Rate after the digital down-converter in Hz.
pub const BASEBAND_RATE_HZ: u64 = 240_000;
/// Audio sample rate in Hz.
pub const AUDIO_RATE_HZ: u64 = 48_000;
/// Offset of the carrier from the tuner centre in Hz.
pub const CARRIER_OFFSET_HZ: f64 = 600_000.0;
/// Peak frequency deviation in Hz.
pub const DEVIATION_HZ: f64 = 75_000.0;
/// Frequency of the audio test tone in Hz.
pub const AUDIO_HZ: f64 = 1_000.0;
/// De-emphasis time constant in seconds.
pub const DEEMPHASIS_TAU_S: f64 = 50e-6;
/// Amplitude of the received test signal relative to full scale.
pub const RX_SIGNAL_AMPLITUDE: f64 = 0.9;

/// Sample types the chains run on (`C32` and `CQ15`).
pub trait E2eSample: ComplexSample {
    /// Gain of the discriminator at 240 kHz. `C32`: `fs / (2 pi deviation)`, which maps the full
    /// deviation to 1.0. `CQ15`: raw 32767, the largest possible; the output unit is phase / pi,
    /// so the full deviation gives about 0.625 (SINAD does not depend on the amplitude).
    fn demod_gain() -> Self::Real;
}

impl E2eSample for C32 {
    fn demod_gain() -> f32 {
        (BASEBAND_RATE_HZ as f64 / (2.0 * PI * DEVIATION_HZ)) as f32
    }
}

impl E2eSample for CQ15 {
    fn demod_gain() -> Q15 {
        Q15(32767)
    }
}

/// The kernels of the receive chain.
pub trait RxKernels<T: E2eSample>:
    Convert<T> + Ddc<T> + FmDemod<T> + ResampleReal<T::Real>
{
}

impl<T: E2eSample, K: Convert<T> + Ddc<T> + FmDemod<T> + ResampleReal<T::Real>> RxKernels<T> for K {}

/// The kernels of the transmit chain (and, for the transmit-then-receive test, of the receive chain).
pub trait TxKernels<T: E2eSample>:
    RxKernels<T> + ConvertOut<T> + Duc<T> + FmMod<T> + FirInterpolateReal<T::Real>
{
}

impl<T, K> TxKernels<T> for K
where
    T: E2eSample,
    K: RxKernels<T> + ConvertOut<T> + Duc<T> + FmMod<T> + FirInterpolateReal<T::Real>,
{
}

fn fail(msg: &'static str) -> Failure {
    Failure::new(CASE, msg)
}

/// Designs a `lowpass_hann` filter times `gain` and converts it to the real sample type.
pub(crate) fn design<'s, R: RealSample>(
    c: &mut Carver<'s>,
    cutoff_norm: f64,
    n: usize,
    gain: f64,
) -> Result<&'s [R], Failure> {
    let h = c.take::<f64>(n)?;
    lowpass_hann(cutoff_norm, h);
    let out = c.take::<R>(n)?;
    for (o, v) in out.iter_mut().zip(h.iter()) {
        *o = R::from_f64_round(gain * v);
    }
    Ok(&*out)
}

/// A source of raw receive bytes: `fill(first_sample, bytes)` writes the samples
/// `first_sample ..` of the signal.
pub type ByteSource<'a> = dyn FnMut(u64, &mut [u8]) -> Result<(), Failure> + 'a;

/// Source of the synthetic FM broadcast of the receive test: `fm_baseband` at the receive
/// parameters, scaled by [`RX_SIGNAL_AMPLITUDE`] and encoded with the independent testkit
/// encoder. `scratch` must hold [`CHUNK`] samples.
///
/// ```
/// use num_complex::Complex64;
/// use sdrk_device::formats;
/// let mut scratch = [Complex64::new(0.0, 0.0); sdrk_testkit::e2e::CHUNK];
/// let mut source = sdrk_testkit::e2e::fm_source(&mut scratch, formats::CU8);
/// let mut bytes = [0u8; 2 * sdrk_testkit::e2e::CHUNK];
/// source(0, &mut bytes)?;
/// assert_eq!(&bytes[..2], &[243, 128]); // 0.9 + 0j in CU8
/// # Ok::<(), sdrk_testkit::Failure>(())
/// ```
pub fn fm_source(
    scratch: &mut [Complex64],
    format: SampleFormat,
) -> impl FnMut(u64, &mut [u8]) -> Result<(), Failure> + '_ {
    move |start, bytes| {
        let n = format.samples_in_bytes(bytes.len());
        let block = scratch
            .get_mut(..n)
            .ok_or_else(|| fail("scratch too small"))?;
        fm_baseband(
            block,
            IF_RATE_HZ as f64,
            CARRIER_OFFSET_HZ,
            DEVIATION_HZ,
            AUDIO_HZ,
            1.0,
            start,
        );
        for s in block.iter_mut() {
            *s *= RX_SIGNAL_AMPLITUDE;
        }
        encode(&format, block, bytes)?;
        Ok(())
    }
}

/// Runs the receive chain over `samples` input samples (a multiple of [`CHUNK`]) from `source`
/// and stores the audio samples with index `audio_skip ..` in `audio` (as many as fit).
/// Returns the total number of audio samples produced, `samples / 50`.
///
/// Needs about 14 KiB of the carver.
///
/// # Errors
/// A [`Failure`] if the carver is exhausted, a kernel fails or `samples` is not a multiple of
/// [`CHUNK`].
pub fn fm_rx<T, K>(
    k: &K,
    c: &mut Carver<'_>,
    format: &SampleFormat,
    source: &mut ByteSource<'_>,
    samples: usize,
    audio_skip: usize,
    audio: &mut [f64],
) -> Result<usize, Failure>
where
    T: E2eSample,
    K: RxKernels<T>,
{
    if !samples.is_multiple_of(CHUNK) {
        return Err(fail("sample count is not a multiple of the block size"));
    }
    let if_rate = Rate::hz(IF_RATE_HZ)?;
    let bb_rate = Rate::hz(BASEBAND_RATE_HZ)?;
    let taps_ddc = design::<T::Real>(c, 100e3 / 2.4e6, 127, 1.0)?;
    let taps_audio = design::<T::Real>(c, 15e3 / 240e3, 241, 1.0)?;
    let mut fir = FirDecimState::<T, _, _>::new(taps_ddc, c.take::<T>(126)?, 10)?;
    let mut resampler =
        ResampleRealState::<T::Real, _, _>::new(taps_audio, c.take::<T::Real>(241)?, 1, 5)?;
    let mut nco = Nco::new(-CARRIER_OFFSET_HZ, if_rate)?;
    let mut quad = QuadDemodState::<T>::new();
    let mut deemph = DeemphasisState::<T::Real>::new(DEEMPHASIS_TAU_S, bb_rate)?;
    let gain = T::demod_gain();

    let bytes_per_chunk = format.bytes_for_samples(CHUNK)?;
    let bytes = c.take::<u8>(bytes_per_chunk)?;
    let iq = c.take::<T>(CHUNK)?;
    let scratch = c.take::<T>(CHUNK)?;
    let low = c.take::<T>(CHUNK / 10 + 4)?;
    let demod = c.take::<T::Real>(CHUNK / 10 + 4)?;
    let scratch_real = c.take::<T::Real>(CHUNK / 10 + 4)?;
    let out = c.take::<T::Real>(CHUNK / 50 + 4)?;

    let mut produced_total = 0usize;
    for chunk in 0..samples / CHUNK {
        source((chunk * CHUNK) as u64, bytes)?;
        let n = k.convert(format, bytes, iq)?;
        if n != CHUNK {
            return Err(fail("convert returned the wrong sample count"));
        }
        // Down-convert: the output room always covers a whole block, but stay general.
        let mut pos = 0usize;
        let mut low_count = 0usize;
        while pos < CHUNK {
            let rest = iq.get(pos..).ok_or_else(|| fail("block split"))?;
            let room = low
                .get_mut(low_count..)
                .ok_or_else(|| fail("block split"))?;
            let p = k.ddc(&mut nco, &mut fir, rest, room, scratch)?;
            if p.consumed == 0 && p.produced == 0 {
                return Err(fail("ddc made no progress"));
            }
            pos += p.consumed;
            low_count += p.produced;
        }
        let low_block = low.get(..low_count).ok_or_else(|| fail("block split"))?;
        let demod_block = demod
            .get_mut(..low_count)
            .ok_or_else(|| fail("block split"))?;
        k.fm_demod(
            &mut quad,
            &mut deemph,
            low_block,
            demod_block,
            scratch_real,
            gain,
        )?;
        let mut rest: &[T::Real] = demod_block;
        while !rest.is_empty() {
            let p = k.resample_real(&mut resampler, rest, out)?;
            for r in out.iter().take(p.produced) {
                let idx = produced_total;
                produced_total += 1;
                if let Some(slot) = idx.checked_sub(audio_skip).and_then(|i| audio.get_mut(i)) {
                    *slot = r.to_f64();
                }
            }
            if p.consumed == 0 && p.produced == 0 {
                return Err(fail("resample_real made no progress"));
            }
            rest = rest.get(p.consumed..).unwrap_or_default();
        }
    }
    Ok(produced_total)
}

/// The receive chain of [`fm_rx`] built from blocks: `ConvertBlock`, `DdcBlock` (600 kHz,
/// decimation 10, default channel filter) and `FmDemodBlock` (75 kHz, 50 us, audio filter at
/// 15 kHz, decimation 5). Same contract as [`fm_rx`]; the block storage comes from the carver.
///
/// # Errors
/// A [`Failure`] if the carver is exhausted, a block fails or `samples` is not a multiple of
/// [`CHUNK`].
pub fn fm_rx_blocks<T>(
    c: &mut Carver<'_>,
    format: &SampleFormat,
    source: &mut ByteSource<'_>,
    samples: usize,
    audio_skip: usize,
    audio: &mut [f64],
) -> Result<usize, Failure>
where
    T: E2eSample + BlockSample,
    Sdrk:
        Convert<T> + Ddc<T> + FmDemod<T> + QuadDemod<T> + ResampleReal<<T as ComplexSample>::Real>,
{
    if !samples.is_multiple_of(CHUNK) {
        return Err(fail("sample count is not a multiple of the block size"));
    }
    let if_rate = Rate::hz(IF_RATE_HZ)?;
    let bb_rate = Rate::hz(BASEBAND_RATE_HZ)?;
    let ddc_cfg = DdcConfig {
        frequency_hz: CARRIER_OFFSET_HZ,
        decimation: 10,
        max_chunk: CHUNK,
        ..DdcConfig::default()
    };
    let demod_cfg = FmDemodConfig {
        deviation_hz: DEVIATION_HZ,
        deemphasis_us: DEEMPHASIS_TAU_S * 1e6,
        audio_cutoff_hz: 15e3,
        audio_decimation: 5,
    };
    let ddc_bytes =
        DdcBlock::<T, ArenaStorage<'_>>::storage_bytes(&ddc_cfg, if_rate)? + ARENA_ALIGN;
    let demod_bytes =
        FmDemodBlock::<T, ArenaStorage<'_>>::storage_bytes(&demod_cfg, bb_rate)? + ARENA_ALIGN;
    let mut ddc_store = ArenaStorage::new(c.take::<u8>(ddc_bytes)?);
    let mut demod_store = ArenaStorage::new(c.take::<u8>(demod_bytes)?);
    let mut convert = ConvertBlock::<T, _>::new(
        &ConvertConfig { format: *format },
        if_rate,
        &mut ArenaStorage::new(&mut []),
    )?;
    let mut ddc = DdcBlock::<T, _>::new(&ddc_cfg, if_rate, &mut ddc_store)?;
    let mut demod = FmDemodBlock::<T, _>::new(&demod_cfg, bb_rate, &mut demod_store)?;

    let bytes = c.take::<u8>(format.bytes_for_samples(CHUNK)?)?;
    let iq = c.take::<T>(CHUNK)?;
    let low = c.take::<T>(CHUNK / 10 + 4)?;
    let out = c.take::<T::Real>(CHUNK / 50 + 4)?;
    let mut produced_total = 0usize;
    for chunk in 0..samples / CHUNK {
        source((chunk * CHUNK) as u64, bytes)?;
        let p = convert.process(bytes, iq)?;
        if p.consumed != bytes.len() || p.produced != CHUNK {
            return Err(fail("convert did not take the whole block"));
        }
        let mut rest: &[T] = iq;
        while !rest.is_empty() {
            let p = ddc.process(rest, low)?;
            if p.consumed == 0 && p.produced == 0 {
                return Err(fail("ddc made no progress"));
            }
            rest = rest.get(p.consumed..).unwrap_or_default();
            let mut low_rest = low.get(..p.produced).unwrap_or_default();
            while !low_rest.is_empty() {
                let q = demod.process(low_rest, out)?;
                if q.consumed == 0 && q.produced == 0 {
                    return Err(fail("fm_demod made no progress"));
                }
                for r in out.iter().take(q.produced) {
                    let idx = produced_total;
                    produced_total += 1;
                    if let Some(slot) = idx.checked_sub(audio_skip).and_then(|i| audio.get_mut(i)) {
                        *slot = r.to_f64();
                    }
                }
                low_rest = low_rest.get(q.consumed..).unwrap_or_default();
            }
        }
    }
    Ok(produced_total)
}

/// E2E-RX built from blocks: like [`measure_rx`] with [`fm_rx_blocks`]. Memory: about the same
/// as [`measure_rx`] plus the storage of the blocks (about 10 KiB).
pub fn measure_rx_blocks<T>(arena: &mut Arena<'_>, samples: usize) -> Result<Measurement, Failure>
where
    T: E2eSample + BlockSample,
    Sdrk:
        Convert<T> + Ddc<T> + FmDemod<T> + QuadDemod<T> + ResampleReal<<T as ComplexSample>::Real>,
{
    let mut c = arena.carver();
    let total_audio = samples / 50;
    let skip = total_audio / 2;
    let audio = c.take::<f64>(total_audio - skip)?;
    let scratch = c.take::<Complex64>(CHUNK)?;
    let mut source = fm_source(scratch, sdrk_device::formats::CU8);
    let produced = fm_rx_blocks::<T>(
        &mut c,
        &sdrk_device::formats::CU8,
        &mut source,
        samples,
        skip,
        audio,
    )?;
    Ok(measure_audio(audio, produced))
}

/// The FM transmit chain as a byte source for [`fm_rx`] (or any consumer of raw bytes).
///
/// Create it with [`FmTx::new`] and call [`FmTx::fill`] for consecutive blocks of [`CHUNK`]
/// samples. Needs about 20 KiB of the carver.
pub struct FmTx<'a, T: E2eSample, K> {
    k: &'a K,
    format: SampleFormat,
    amplitude: f64,
    audio_f64: &'a mut [f64],
    audio: &'a mut [T::Real],
    audio_up: &'a mut [T::Real],
    modulated: &'a mut [T],
    scratch: &'a mut [T],
    iq: &'a mut [T],
    interp: FirInterpRealState<T::Real, &'a [T::Real], &'a mut [T::Real]>,
    fm: FmModState,
    duc: FirInterpState<T, &'a [T::Real], &'a mut [T]>,
    nco: Nco,
    next: u64,
}

impl<'a, T, K> FmTx<'a, T, K>
where
    T: E2eSample,
    K: ConvertOut<T> + Duc<T> + FmMod<T> + FirInterpolateReal<T::Real>,
{
    /// Creates the chain: audio tone of [`AUDIO_HZ`] with amplitude `amplitude` (1.0 is the full
    /// deviation), output `format` (interleaved or planar complex).
    pub fn new(
        k: &'a K,
        c: &mut Carver<'a>,
        format: SampleFormat,
        amplitude: f64,
    ) -> Result<Self, Failure> {
        let n_audio = CHUNK / 50;
        let taps_audio = design::<T::Real>(c, 15e3 / 240e3, 241, 5.0)?;
        let taps_duc = design::<T::Real>(c, 100e3 / 2.4e6, 127, 10.0)?;
        Ok(Self {
            k,
            format,
            amplitude,
            audio_f64: c.take::<f64>(n_audio)?,
            audio: c.take::<T::Real>(n_audio)?,
            audio_up: c.take::<T::Real>(CHUNK / 10)?,
            modulated: c.take::<T>(CHUNK / 10)?,
            scratch: c.take::<T>(CHUNK)?,
            iq: c.take::<T>(CHUNK)?,
            interp: FirInterpRealState::new(taps_audio, c.take::<T::Real>(49)?, 5)?,
            fm: FmModState::new(DEVIATION_HZ, Rate::hz(BASEBAND_RATE_HZ)?)?,
            duc: FirInterpState::new(taps_duc, c.take::<T>(13)?, 10)?,
            nco: Nco::new(CARRIER_OFFSET_HZ, Rate::hz(IF_RATE_HZ)?)?,
            next: 0,
        })
    }

    /// Writes the next block of [`CHUNK`] samples as raw bytes. `start` must be the index of
    /// the first sample of the block and follow the previous block directly.
    pub fn fill(&mut self, start: u64, bytes: &mut [u8]) -> Result<(), Failure> {
        if start != self.next || !start.is_multiple_of(CHUNK as u64) {
            return Err(fail("transmit blocks must be consecutive"));
        }
        let first_audio = start / 50;
        real_tone(
            self.audio_f64,
            AUDIO_HZ / AUDIO_RATE_HZ as f64,
            self.amplitude,
            0.0,
            first_audio,
        );
        for (a, v) in self.audio.iter_mut().zip(self.audio_f64.iter()) {
            *a = T::Real::from_f64_round(*v);
        }
        let p = self
            .k
            .fir_interpolate_real(&mut self.interp, self.audio, self.audio_up)?;
        if p.consumed != self.audio.len() || p.produced != self.audio_up.len() {
            return Err(fail("audio interpolation did not process the whole block"));
        }
        self.k.fm_mod(&mut self.fm, self.audio_up, self.modulated)?;
        let p = self.k.duc(
            &mut self.duc,
            &mut self.nco,
            self.modulated,
            self.iq,
            self.scratch,
        )?;
        if p.consumed != self.modulated.len() || p.produced != CHUNK {
            return Err(fail("duc did not process the whole block"));
        }
        let n = self.k.convert_out(&self.format, self.iq, bytes)?;
        if n != self.format.bytes_for_samples(CHUNK)? {
            return Err(fail("convert_out wrote the wrong byte count"));
        }
        self.next += CHUNK as u64;
        Ok(())
    }
}

impl<T: E2eSample, K> core::fmt::Debug for FmTx<'_, T, K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FmTx")
            .field("next", &self.next)
            .field("amplitude", &self.amplitude)
            .finish_non_exhaustive()
    }
}

/// What the end-to-end tests measure on the last half of the audio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measurement {
    /// SINAD of the 1 kHz tone in dB.
    pub sinad_db: f64,
    /// Frequency of the tone found by minimising the fit residual, in Hz.
    pub tone_hz: f64,
    /// Peak amplitude of the fitted tone (the unit depends on the sample type, see
    /// [`E2eSample::demod_gain`]).
    pub amplitude: f64,
    /// Number of audio samples the chain produced in total.
    pub audio_samples: usize,
}

pub(crate) fn measure_audio(audio: &[f64], produced: usize) -> Measurement {
    let rate = AUDIO_RATE_HZ as f64;
    Measurement {
        sinad_db: sinad_db(audio, rate, AUDIO_HZ),
        tone_hz: fit_frequency(audio, rate, AUDIO_HZ, 5.0),
        amplitude: ToneMeter::measure(audio, rate, AUDIO_HZ).amplitude,
        audio_samples: produced,
    }
}

/// E2E-RX: receives `samples` samples (a multiple of [`CHUNK`]) of the synthetic FM broadcast
/// from `CU8` bytes and measures the last half of the audio. Takes its memory from `arena`:
/// about 27 KiB plus 4 bytes per input sample.
pub fn measure_rx<T: E2eSample, K: RxKernels<T>>(
    k: &K,
    arena: &mut Arena<'_>,
    samples: usize,
) -> Result<Measurement, Failure> {
    let mut c = arena.carver();
    let total_audio = samples / 50;
    let skip = total_audio / 2;
    let audio = c.take::<f64>(total_audio - skip)?;
    let scratch = c.take::<Complex64>(CHUNK)?;
    let mut source = fm_source(scratch, sdrk_device::formats::CU8);
    let produced = fm_rx::<T, K>(
        k,
        &mut c,
        &sdrk_device::formats::CU8,
        &mut source,
        samples,
        skip,
        audio,
    )?;
    Ok(measure_audio(audio, produced))
}

/// E2E-TX-RX: transmits a 1 kHz tone to `CS8` bytes with [`FmTx`] and receives it with
/// [`fm_rx`]; measures the last half of the audio. Memory as [`measure_rx`] plus about 20 KiB.
pub fn measure_txrx<T: E2eSample, K: TxKernels<T>>(
    k: &K,
    arena: &mut Arena<'_>,
    samples: usize,
) -> Result<Measurement, Failure> {
    let mut c = arena.carver();
    let total_audio = samples / 50;
    let skip = total_audio / 2;
    let audio = c.take::<f64>(total_audio - skip)?;
    let mut tx = FmTx::<T, K>::new(k, &mut c, sdrk_device::formats::CS8, 1.0)?;
    let mut source = |start: u64, bytes: &mut [u8]| tx.fill(start, bytes);
    let produced = fm_rx::<T, K>(
        k,
        &mut c,
        &sdrk_device::formats::CS8,
        &mut source,
        samples,
        skip,
        audio,
    )?;
    Ok(measure_audio(audio, produced))
}
