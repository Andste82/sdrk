//! `ssb_mod`: single-sideband modulator (the Weaver chain of `ssb_demod` mirrored).

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::math::Interleave;
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::timing::Timing;

use super::fir::MAX_TAPS_LIMIT;
use super::ssb_demod::{band_centre_hz, band_spec, check_band};
use super::{HZ_LIMIT, head, head_mut, hz, tail, tail_mut, unknown};
use crate::SsbSideband;
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::DesignSpec;
use crate::error::{Error, Result};
use crate::sample::{BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// Samples per piece the block processes at a time (the size of its work buffer).
pub const SSB_MOD_CHUNK: usize = 256;

/// Gain of the taps: the complex low-pass passes one of the two halves of a real tone, so the
/// taps are doubled before quantization and a real tone of amplitude A comes out with
/// amplitude A.
const TAP_GAIN: f64 = 2.0;

/// Description of the `ssb_mod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "ssb_mod",
    summary: "Modulates real audio to a single-sideband signal (USB or LSB) at 0 Hz.",
    details: "The mirror of ssb_demod (the Weaver method). The block takes the real audio as a \
        complex signal, shifts the band [low_hz, high_hz] to a band centred on 0 Hz with a \
        numerically controlled oscillator, removes everything else with a complex Kaiser \
        low-pass (cutoff (high_hz - low_hz) / 2, transition_hz, attenuation_db, taps doubled so \
        that the level is kept) and shifts the band to [low_hz, high_hz] (USB) or [-high_hz, \
        -low_hz] (LSB) with a second oscillator. An audio tone at f comes out at +f (USB) or -f \
        (LSB) of the suppressed carrier at 0 Hz with the input amplitude; the opposite sideband \
        is suppressed by the filter. The sideband can be switched while the block runs; both \
        oscillators change sign and their phases stay continuous. ssb_mod followed by \
        ssb_demod with the same sideband and band returns the audio. Put a duc after the block \
        to reach the transmit frequency. Pitfalls: audio outside [low_hz, high_hz] is removed; \
        the output peak can exceed the input peak by the ripple of the filter and by the \
        envelope of several tones (the sum of their amplitudes), so keep the audio below full \
        scale for CQ15; the output is delayed by (taps - 1) / 2 samples.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32, PortType::Q15],
        description: "Real modulating signal (audio).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex SSB baseband with the suppressed carrier at 0 Hz, at the same \
            rate (f32 gives C32, Q15 gives CQ15).",
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
            description: "Sideband to send: usb (above the carrier) or lsb (below).",
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
            description: "Lower edge of the audio band in Hz.",
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
            description: "Upper edge of the audio band in Hz; above low_hz, and high_hz plus \
                half of transition_hz at most half the sample rate.",
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
            description: "Stop-band attenuation in dB (at least 8); also the suppression of \
                the opposite sideband.",
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
    ],
    required: &[],
};

/// Configuration of [`SsbModBlock`].
///
/// ```
/// use sdrk_blocks::SsbModConfig;
/// let cfg = SsbModConfig::default();
/// assert!(cfg.validate(sdrk::Rate::hz(48_000)?).is_ok());
/// // The band must fit below half the sample rate.
/// assert!(SsbModConfig { high_hz: 23_950.0, ..cfg }.validate(sdrk::Rate::hz(48_000)?).is_err());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsbModConfig {
    /// The sideband to send (can change while the block runs).
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
}

impl Default for SsbModConfig {
    fn default() -> Self {
        Self {
            sideband: SsbSideband::Usb,
            low_hz: 300.0,
            high_hz: 3_000.0,
            transition_hz: 200.0,
            attenuation_db: 60.0,
            max_taps: 1023,
        }
    }
}

impl SsbModConfig {
    /// The low-pass specification (half the band width as cutoff), as in `ssb_demod`.
    ///
    /// ```
    /// use sdrk_blocks::SsbModConfig;
    /// let spec = SsbModConfig::default().design_spec();
    /// assert_eq!((spec.cutoff_hz, spec.transition_hz), (1_350.0, 200.0));
    /// ```
    pub fn design_spec(&self) -> DesignSpec {
        band_spec(
            self.low_hz,
            self.high_hz,
            self.transition_hz,
            self.attenuation_db,
        )
    }

    /// Number of taps for the input rate (also checks the whole configuration).
    ///
    /// ```
    /// use sdrk_blocks::SsbModConfig;
    /// assert_eq!(SsbModConfig::default().num_taps(sdrk::Rate::hz(48_000)?)?, 873);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        check_band(
            self.low_hz,
            self.high_hz,
            self.transition_hz,
            hz(rate_in) / 2.0,
        )?;
        self.design_spec().num_taps(hz(rate_in), self.max_taps)
    }

    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.num_taps(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// ```
    /// use sdrk_blocks::{Error, ParamValue, SsbModConfig, SsbSideband};
    /// let mut cfg = SsbModConfig::default();
    /// cfg.set("sideband", ParamValue::Choice(1))?;
    /// assert_eq!(cfg.sideband, SsbSideband::Lsb);
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
            _ => unknown(),
        }
    }
}

/// SSB modulator: `f32` to `C32` or `Q15` to `CQ15` (`T` is the complex output type).
///
/// ```
/// use sdrk_blocks::{ArenaStorage, Block, SsbModBlock, SsbModConfig};
/// use sdrk::{Rate, C32};
/// let cfg = SsbModConfig::default();
/// let rate = Rate::hz(48_000)?;
/// let bytes = SsbModBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = vec![0u8; bytes + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut block = SsbModBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// // A 1 kHz audio tone of amplitude 0.5.
/// let audio: Vec<f32> = (0..2000)
///     .map(|n| 0.5 * (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / 48_000.0).cos() as f32)
///     .collect();
/// let mut out = vec![C32::default(); 2000];
/// let p = block.process(&audio, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (2000, 2000));
/// // After the filter has settled the output is a complex tone of magnitude 0.5 at +1 kHz.
/// assert!(out[1500..].iter().all(|s| (s.re.hypot(s.im) - 0.5).abs() < 0.01));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SsbModBlock<T: BlockSample, S: Storage> {
    config: SsbModConfig,
    rate_in: Rate,
    centre_hz: f64,
    nco_in: Nco,
    nco_out: Nco,
    fir: FirState<T, S::Buf<T::Real>, S::Buf<T>>,
    work: S::Buf<T>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for SsbModBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SsbModBlock")
            .field("config", &self.config)
            .field("rate_in", &self.rate_in)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> SsbModBlock<T, S> {
    /// Designs the filter and creates the block; taps, history and the work buffer come from
    /// `storage`. Both oscillators start at phase 0.
    pub fn new(config: &SsbModConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let lp = config.design_spec().lowpass(hz(rate_in), config.max_taps)?;
        let n = lp.len();
        let mut taps = storage.alloc::<T::Real>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <T as TunedElem>::tap(TAP_GAIN * lp.tap(i))?;
        }
        let hist = storage.alloc::<T>(n - 1)?;
        let work = storage.alloc::<T>(SSB_MOD_CHUNK)?;
        let centre_hz = band_centre_hz(config.low_hz, config.high_hz);
        let s = config.sideband.sign();
        Ok(Self {
            config: *config,
            rate_in,
            centre_hz,
            nco_in: Nco::new(-s * centre_hz, rate_in)?,
            nco_out: Nco::new(s * centre_hz, rate_in)?,
            fir: FirState::new(taps, hist)?,
            work,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &SsbModConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<T::Real>(n)?;
        plan.add::<T>(n - 1)?;
        plan.add::<T>(SSB_MOD_CHUNK)?;
        Ok(plan.bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &SsbModConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }
}

impl<T: BlockSample, S: Storage> Block for SsbModBlock<T, S>
where
    Sdrk: Interleave<T> + Rotate<T> + Fir<T>,
{
    type In = T::Real;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T::Real], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        let mut done = 0usize;
        while done < n {
            let m = (n - done).min(SSB_MOD_CHUNK);
            let x = head(tail(input, done)?, m)?;
            let out = head_mut(tail_mut(output, done)?, m)?;
            let work = head_mut(self.work.as_mut(), m)?;
            // The audio as a complex signal, then shift the band to 0 Hz: the output piece is
            // the scratch for the first two stages, so one work buffer suffices.
            <Sdrk as Interleave<T>>::interleave(&Sdrk, x, None, work)?;
            <Sdrk as Rotate<T>>::rotate(&Sdrk, &mut self.nco_in, work, out)?;
            // Complex low-pass around 0 Hz.
            <Sdrk as Fir<T>>::fir(&Sdrk, &mut self.fir, out, work)?;
            // Shift the band to the wanted sideband.
            <Sdrk as Rotate<T>>::rotate(&Sdrk, &mut self.nco_out, work, out)?;
            done += m;
        }
        Ok(Progress {
            consumed: n,
            produced: n,
        })
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
                // Both oscillators change sign (decision D431); their phases continue.
                self.nco_in
                    .set_frequency(-s * self.centre_hz, self.rate_in)?;
                self.nco_out
                    .set_frequency(s * self.centre_hz, self.rate_in)?;
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
        1
    }
}
