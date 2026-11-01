//! `ssb_demod`: single-sideband demodulator (shift, low-pass with decimation, shift back).

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::timing::Timing;

use super::fir::MAX_TAPS_LIMIT;
use super::{HZ_LIMIT, check_count, divide_rate, head, head_mut, hz, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::DesignSpec;
use crate::error::{Error, Result};
use crate::sample::{BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// Samples per piece the block processes at a time (the size of its two work buffers).
pub const SSB_DEMOD_CHUNK: usize = 256;

/// The sideband an [`SsbDemodBlock`] passes.
///
/// ```
/// use sdrk_blocks::SsbSideband;
/// assert_eq!(SsbSideband::from_index(1), Some(SsbSideband::Lsb));
/// assert_eq!(SsbSideband::Usb.index(), 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SsbSideband {
    /// Upper sideband: the audio lies above the carrier at 0 Hz.
    Usb,
    /// Lower sideband: the audio lies below the carrier at 0 Hz.
    Lsb,
}

impl SsbSideband {
    /// Index into the options of the `sideband` parameter (`usb` is 0, `lsb` is 1).
    pub const fn index(self) -> u16 {
        match self {
            Self::Usb => 0,
            Self::Lsb => 1,
        }
    }

    /// The sideband for an index of the `sideband` parameter, or `None` if out of range.
    pub const fn from_index(i: u16) -> Option<Self> {
        match i {
            0 => Some(Self::Usb),
            1 => Some(Self::Lsb),
            _ => None,
        }
    }

    /// `+1.0` for USB and `-1.0` for LSB.
    const fn sign(self) -> f64 {
        match self {
            Self::Usb => 1.0,
            Self::Lsb => -1.0,
        }
    }
}

/// Description of the `ssb_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "ssb_demod",
    summary: "Demodulates a single-sideband signal (USB or LSB) to audio.",
    details: "The suppressed carrier must be at 0 Hz of the input (tune the ddc so). The block \
        shifts the wanted sideband [low_hz, high_hz] to a band centred on 0 Hz with a numerically \
        controlled oscillator, removes everything else with a Kaiser low-pass (cutoff \
        (high_hz - low_hz) / 2, transition_hz, attenuation_db) that also decimates by \
        audio_decimation, shifts the band back to low_hz ... high_hz and outputs the real part. \
        A tone at the carrier plus f (USB) or minus f (LSB) comes out as an audio tone at f with \
        the input amplitude; the opposite sideband is rejected by the filter. The sideband can \
        be switched while the block runs; both oscillator phases stay continuous. By convention \
        amateur radio uses LSB below 10 MHz and USB above, marine and aviation HF use USB. A \
        mistuning of df Hz shifts the audio by df Hz. Put an agc in front (the example flow \
        does; its defaults suit SSB). Pitfall: the output rate (input rate divided by \
        audio_decimation) must be at least twice high_hz plus transition_hz.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex baseband samples with the suppressed carrier at 0 Hz.",
    }],
    outputs: &[PortDesc {
        name: "audio",
        types: &[PortType::F32, PortType::Q15],
        description: "Demodulated audio (f32 for C32 input, Q15 for CQ15 input).",
    }],
    params: &[
        ParamDesc {
            name: "sideband",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: &["usb", "lsb"],
                default: 0,
            },
            typical: "usb; lsb for amateur radio below 10 MHz",
            mutability: Mutability::Runtime,
            description: "Sideband to receive: usb (above the carrier) or lsb (below).",
        },
        ParamDesc {
            name: "low_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 0.0,
                max: HZ_LIMIT,
                default: Some(300.0),
            },
            typical: "300 for voice",
            mutability: Mutability::Build,
            description: "Lower edge of the audio band in Hz, from the carrier.",
        },
        ParamDesc {
            name: "high_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 0.0,
                max: HZ_LIMIT,
                default: Some(3_000.0),
            },
            typical: "3000 for voice",
            mutability: Mutability::Build,
            description: "Upper edge of the audio band in Hz, from the carrier; above low_hz.",
        },
        ParamDesc {
            name: "transition_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: Some(200.0),
            },
            typical: "200",
            mutability: Mutability::Build,
            description: "Width of the transition band of the filter in Hz.",
        },
        ParamDesc {
            name: "attenuation_db",
            unit: Unit::Decibel,
            kind: ParamKind::F64 {
                min: 8.0,
                max: 200.0,
                default: Some(60.0),
            },
            typical: "60",
            mutability: Mutability::Build,
            description: "Stop-band attenuation in dB (at least 8).",
        },
        ParamDesc {
            name: "max_taps",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_TAPS_LIMIT,
                default: Some(1023),
            },
            typical: "1023",
            mutability: Mutability::Build,
            description: "Upper limit of the number of taps; a design that needs more fails.",
        },
        ParamDesc {
            name: "audio_decimation",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 64,
                default: Some(1),
            },
            typical: "1 at 48 kHz, 6 to go to 8 kHz",
            mutability: Mutability::Build,
            description: "Decimation factor of the audio, 1 to 64.",
        },
    ],
    required: &[],
};

/// Configuration of [`SsbDemodBlock`].
///
/// ```
/// use sdrk_blocks::SsbDemodConfig;
/// let cfg = SsbDemodConfig::default();
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// // The band must fit below half the output rate.
/// assert!(SsbDemodConfig { high_hz: 23_950.0, ..cfg }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsbDemodConfig {
    /// The sideband to receive (can change while the block runs).
    pub sideband: SsbSideband,
    /// Lower edge of the audio band in Hz.
    pub low_hz: f64,
    /// Upper edge of the audio band in Hz.
    pub high_hz: f64,
    /// Width of the transition band in Hz.
    pub transition_hz: f64,
    /// Stop-band attenuation in dB.
    pub attenuation_db: f64,
    /// Upper limit of the number of taps.
    pub max_taps: usize,
    /// Decimation factor of the audio, 1 to 64.
    pub audio_decimation: usize,
}

impl Default for SsbDemodConfig {
    fn default() -> Self {
        Self {
            sideband: SsbSideband::Usb,
            low_hz: 300.0,
            high_hz: 3_000.0,
            transition_hz: 200.0,
            attenuation_db: 60.0,
            max_taps: 1023,
            audio_decimation: 1,
        }
    }
}

impl SsbDemodConfig {
    /// Centre of the audio band in Hz.
    fn centre_hz(&self) -> f64 {
        f64::midpoint(self.low_hz, self.high_hz)
    }

    /// The low-pass specification (half the band width as cutoff, designed at the input rate).
    ///
    /// ```
    /// use sdrk_blocks::SsbDemodConfig;
    /// let spec = SsbDemodConfig::default().design_spec();
    /// assert_eq!((spec.cutoff_hz, spec.transition_hz), (1_350.0, 200.0));
    /// ```
    pub fn design_spec(&self) -> DesignSpec {
        DesignSpec {
            cutoff_hz: (self.high_hz - self.low_hz) / 2.0,
            transition_hz: self.transition_hz,
            attenuation_db: self.attenuation_db,
        }
    }

    /// Number of taps for the input rate (also checks the whole configuration).
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        check_count("audio_decimation", self.audio_decimation, 1, 64)?;
        if !(self.low_hz >= 0.0 && self.low_hz < self.high_hz) {
            return Err(Error::OutOfRange {
                name: "low_hz",
                min: 0.0,
                max: self.high_hz,
                got: self.low_hz,
            });
        }
        let fs = hz(rate_in);
        let out_half = fs / self.audio_decimation as f64 / 2.0;
        let max = out_half - self.transition_hz / 2.0;
        if self.high_hz.is_nan() || self.high_hz > max {
            return Err(Error::OutOfRange {
                name: "high_hz",
                min: self.low_hz,
                max,
                got: self.high_hz,
            });
        }
        self.design_spec().num_taps(fs, self.max_taps)
    }

    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.num_taps(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// ```
    /// use sdrk_blocks::{Error, ParamValue, SsbDemodConfig, SsbSideband};
    /// let mut cfg = SsbDemodConfig::default();
    /// cfg.set("sideband", ParamValue::Choice(1))?;
    /// assert_eq!(cfg.sideband, SsbSideband::Lsb);
    /// assert!(matches!(cfg.set("sideband", ParamValue::Choice(2)), Err(Error::OutOfRange { .. })));
    /// assert_eq!(cfg.set("nope", ParamValue::I64(0)), Err(Error::UnknownParameter));
    /// # Ok::<(), Error>(())
    /// ```
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "sideband" => {
                self.sideband =
                    SsbSideband::from_index(value.as_choice()?).ok_or(Error::WrongParamType)?;
            }
            "low_hz" => self.low_hz = value.as_f64()?,
            "high_hz" => self.high_hz = value.as_f64()?,
            "transition_hz" => self.transition_hz = value.as_f64()?,
            "attenuation_db" => self.attenuation_db = value.as_f64()?,
            "max_taps" => self.max_taps = value.as_usize()?,
            "audio_decimation" => self.audio_decimation = value.as_usize()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "sideband" => Ok(ParamValue::Choice(self.sideband.index())),
            "low_hz" => Ok(ParamValue::F64(self.low_hz)),
            "high_hz" => Ok(ParamValue::F64(self.high_hz)),
            "transition_hz" => Ok(ParamValue::F64(self.transition_hz)),
            "attenuation_db" => Ok(ParamValue::F64(self.attenuation_db)),
            "max_taps" => Ok(ParamValue::I64(self.max_taps as i64)),
            "audio_decimation" => Ok(ParamValue::I64(self.audio_decimation as i64)),
            _ => unknown(),
        }
    }
}

/// SSB demodulator: `C32` to `f32` or `CQ15` to `Q15` (`T` is the complex input type).
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, SsbDemodBlock, SsbDemodConfig};
/// use sdrk::{Rate, C32};
/// let cfg = SsbDemodConfig::default();
/// let rate = Rate::hz(48_000)?;
/// let bytes = SsbDemodBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = SsbDemodBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// // A USB tone 1 kHz above the carrier.
/// let input: Vec<C32> = (0..2000)
///     .map(|n| {
///         let p = 2.0 * std::f64::consts::PI * 1000.0 * n as f64 / 48_000.0;
///         C32::new(p.cos() as f32, p.sin() as f32)
///     })
///     .collect();
/// let mut audio = vec![0.0f32; 2000];
/// let p = block.process(&input, &mut audio)?;
/// assert_eq!((p.consumed, p.produced), (2000, 2000));
/// // After the filter has settled the audio is a 1 kHz cosine of amplitude 1.
/// let peak = audio[1500..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
/// assert!((peak - 1.0).abs() < 0.01);
/// # let _ = AlignedBytes([0u8; 1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SsbDemodBlock<T: BlockSample, S: Storage> {
    config: SsbDemodConfig,
    rate_in: Rate,
    rate_out: Rate,
    centre_hz: f64,
    nco_in: Nco,
    nco_out: Nco,
    fir: FirDecimState<T, S::Buf<T::Real>, S::Buf<T>>,
    a: S::Buf<T>,
    b: S::Buf<T>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for SsbDemodBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SsbDemodBlock")
            .field("config", &self.config)
            .field("rate_in", &self.rate_in)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> SsbDemodBlock<T, S> {
    /// Designs the filter and creates the block; taps, history and work buffers come from
    /// `storage`. Both oscillators start at phase 0.
    pub fn new(config: &SsbDemodConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let lp = config.design_spec().lowpass(hz(rate_in), config.max_taps)?;
        let n = lp.len();
        let mut taps = storage.alloc::<T::Real>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <T as TunedElem>::tap(lp.tap(i))?;
        }
        let hist = storage.alloc::<T>(n - 1)?;
        let a = storage.alloc::<T>(SSB_DEMOD_CHUNK)?;
        let b = storage.alloc::<T>(SSB_DEMOD_CHUNK)?;
        let rate_out = divide_rate(rate_in, config.audio_decimation)?;
        let centre_hz = config.centre_hz();
        let s = config.sideband.sign();
        Ok(Self {
            config: *config,
            rate_in,
            rate_out,
            centre_hz,
            nco_in: Nco::new(-s * centre_hz, rate_in)?,
            nco_out: Nco::new(s * centre_hz, rate_out)?,
            fir: FirDecimState::new(taps, hist, config.audio_decimation)?,
            a,
            b,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &SsbDemodConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<T::Real>(n)?;
        plan.add::<T>(n - 1)?;
        plan.add::<T>(SSB_DEMOD_CHUNK)?;
        plan.add::<T>(SSB_DEMOD_CHUNK)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate divided by `audio_decimation`.
    pub fn rate_out(config: &SsbDemodConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        divide_rate(rate_in, config.audio_decimation)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }
}

impl<T: BlockSample, S: Storage> Block for SsbDemodBlock<T, S>
where
    Sdrk: Rotate<T> + FirDecimate<T>,
{
    type In = T;
    type Out = T::Real;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T::Real]) -> Result<Progress> {
        let (mut consumed, mut produced) = (0usize, 0usize);
        loop {
            let room = (output.len() - produced).min(SSB_DEMOD_CHUNK);
            // The piece never holds an input sample the filter would leave unconsumed, so the
            // first oscillator stays exact.
            let m = (input.len() - consumed)
                .min(SSB_DEMOD_CHUNK)
                .min(self.fir.max_input_for_output(room));
            if m == 0 {
                break;
            }
            let piece = head(tail(input, consumed)?, m)?;
            // Shift the wanted band to 0 Hz: one rotate call for the whole piece.
            <Sdrk as Rotate<T>>::rotate(
                &Sdrk,
                &mut self.nco_in,
                piece,
                head_mut(self.a.as_mut(), m)?,
            )?;
            // Low-pass and decimate: one fir_decimate call.
            let p = <Sdrk as FirDecimate<T>>::fir_decimate(
                &Sdrk,
                &mut self.fir,
                head(self.a.as_ref(), m)?,
                head_mut(self.b.as_mut(), room)?,
            )?;
            if p.consumed != m {
                return Err(Error::Core(sdrk_core::Error::Internal(
                    "ssb filter did not take the whole piece",
                )));
            }
            let k = p.produced;
            // Shift back: one rotate call at the output rate.
            <Sdrk as Rotate<T>>::rotate(
                &Sdrk,
                &mut self.nco_out,
                head(self.b.as_ref(), k)?,
                head_mut(self.a.as_mut(), k)?,
            )?;
            let out = head_mut(tail_mut(output, produced)?, k)?;
            for (o, s) in out.iter_mut().zip(head(self.a.as_ref(), k)?) {
                *o = s.re();
            }
            consumed += m;
            produced += k;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.fir.ratio(),
            kind: self.fir.ratio_kind(),
            delay: self.fir.group_delay(),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
        self.nco_in.reset();
        self.nco_out.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "sideband" => {
                let sb =
                    SsbSideband::from_index(value.as_choice()?).ok_or(Error::WrongParamType)?;
                let s = sb.sign();
                self.nco_in
                    .set_frequency(-s * self.centre_hz, self.rate_in)?;
                self.nco_out
                    .set_frequency(s * self.centre_hz, self.rate_out)?;
                self.config.sideband = sb;
                Ok(())
            }
            _ => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        self.config.audio_decimation
    }
}
