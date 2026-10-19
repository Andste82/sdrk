//! `duc`: digital up-converter (interpolation, low-pass and mixer in one block).

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::fused::Duc;
use sdrk_core::kernels::interp::FirInterpState;
use sdrk_core::kernels::mix::Nco;
use sdrk_core::timing::Timing;

use super::fir::MAX_TAPS_LIMIT;
use super::{HZ_LIMIT, check_count, check_symmetric, multiply_rate, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::DesignSpec;
use crate::error::{Error, Result};
use crate::sample::{BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// Largest `max_chunk`.
const MAX_CHUNK_LIMIT: i64 = 1 << 20;

/// Description of the `duc` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "duc",
    summary: "Interpolates a signal, removes the images and shifts it up to a carrier frequency.",
    details: "A digital up-converter: the signal is interpolated by the given factor with a \
        Kaiser-window low-pass (the taps carry the gain of the interpolation factor) and then \
        mixed up to +frequency_hz by a numerically controlled oscillator running at the output \
        rate. The output rate is the input rate times interpolation. Unless given, the cutoff \
        is 0.4 times the input rate and the transition width 0.1 times the input rate. \
        frequency_hz can change while the block runs and the oscillator phase stays \
        continuous. The block needs a scratch buffer of max_chunk samples at the output rate, \
        which must hold at least one interpolated sample group.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples at the input rate.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples at the input rate times interpolation.",
    }],
    params: &[
        ParamDesc {
            name: "frequency_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: -HZ_LIMIT,
                max: HZ_LIMIT,
                default: Some(0.0),
            },
            typical: "600e3 to transmit 600 kHz above the centre",
            mutability: Mutability::Runtime,
            description: "Carrier frequency relative to the centre in Hz, within plus or minus \
                half the output rate.",
        },
        ParamDesc {
            name: "interpolation",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 1024,
                default: None,
            },
            typical: "10 to go from 240 kHz to 2.4 MHz",
            mutability: Mutability::Build,
            description: "Interpolation factor, 1 to 1024. Required.",
        },
        ParamDesc {
            name: "cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "100e3 for FM broadcast; default 0.4 times the input rate",
            mutability: Mutability::Build,
            description: "Cutoff (-6 dB point) of the image filter in Hz. The default is \
                0.4 * input rate.",
        },
        ParamDesc {
            name: "transition_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "default 0.1 times the input rate",
            mutability: Mutability::Build,
            description: "Width of the transition band in Hz. The default is 0.1 * input rate.",
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
            description: "Image attenuation in dB (at least 8).",
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
            name: "max_chunk",
            unit: Unit::Samples,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_CHUNK_LIMIT,
                default: Some(4096),
            },
            typical: "4096",
            mutability: Mutability::Build,
            description: "Size of the scratch buffer in output samples; it must be at least \
                the interpolation factor.",
        },
    ],
    required: &["interpolation"],
};

/// Configuration of [`DucBlock`].
///
/// ```
/// use sdrk_blocks::DucConfig;
/// let cfg = DucConfig { frequency_hz: 600e3, interpolation: 10, ..DucConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(240_000)?).is_ok());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DucConfig {
    /// Carrier frequency in Hz (within plus or minus half the output rate).
    pub frequency_hz: f64,
    /// Interpolation factor, 1 to 1024.
    pub interpolation: usize,
    /// Cutoff in Hz; `None` means `0.4 * input rate`.
    pub cutoff_hz: Option<f64>,
    /// Transition width in Hz; `None` means `0.1 * input rate`.
    pub transition_hz: Option<f64>,
    /// Image attenuation in dB.
    pub attenuation_db: f64,
    /// Upper limit of the number of taps.
    pub max_taps: usize,
    /// Size of the scratch buffer in output samples (at least `interpolation`).
    pub max_chunk: usize,
}

impl Default for DucConfig {
    fn default() -> Self {
        Self {
            frequency_hz: 0.0,
            interpolation: 10,
            cutoff_hz: None,
            transition_hz: None,
            attenuation_db: 60.0,
            max_taps: 1023,
            max_chunk: 4096,
        }
    }
}

impl DucConfig {
    /// The filter specification (at the output rate) with the defaults filled in.
    ///
    /// ```
    /// use sdrk_blocks::DucConfig;
    /// let spec = DucConfig::default().design_spec(sdrk::Rate::hz(240_000)?);
    /// assert_eq!((spec.cutoff_hz, spec.transition_hz), (96_000.0, 24_000.0));
    /// # Ok::<(), sdrk::Error>(())
    /// ```
    pub fn design_spec(&self, rate_in: Rate) -> DesignSpec {
        let fin = super::hz(rate_in);
        DesignSpec {
            cutoff_hz: self.cutoff_hz.unwrap_or(0.4 * fin),
            transition_hz: self.transition_hz.unwrap_or(0.1 * fin),
            attenuation_db: self.attenuation_db,
        }
    }

    /// Number of taps for the input rate (also checks the whole configuration).
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        check_count("interpolation", self.interpolation, 1, 1024)?;
        check_count(
            "max_chunk",
            self.max_chunk,
            self.interpolation,
            MAX_CHUNK_LIMIT as usize,
        )?;
        let fs_out = super::hz(rate_in) * self.interpolation as f64;
        check_symmetric("frequency_hz", self.frequency_hz, fs_out / 2.0)?;
        self.design_spec(rate_in).num_taps(fs_out, self.max_taps)
    }

    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.num_taps(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "frequency_hz" => self.frequency_hz = value.as_f64()?,
            "interpolation" => self.interpolation = value.as_usize()?,
            "cutoff_hz" => self.cutoff_hz = Some(value.as_f64()?),
            "transition_hz" => self.transition_hz = Some(value.as_f64()?),
            "attenuation_db" => self.attenuation_db = value.as_f64()?,
            "max_taps" => self.max_taps = value.as_usize()?,
            "max_chunk" => self.max_chunk = value.as_usize()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter. `cutoff_hz` and `transition_hz` are an error while they
    /// are not set (their default depends on the sample rate; the block reports the value in
    /// use).
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        let unset = || Error::InvalidParameter {
            name: if name == "cutoff_hz" {
                "cutoff_hz"
            } else {
                "transition_hz"
            },
            reason: "not set; the default depends on the sample rate",
        };
        match name {
            "frequency_hz" => Ok(ParamValue::F64(self.frequency_hz)),
            "interpolation" => Ok(ParamValue::I64(self.interpolation as i64)),
            "cutoff_hz" => self.cutoff_hz.map(ParamValue::F64).ok_or_else(unset),
            "transition_hz" => self.transition_hz.map(ParamValue::F64).ok_or_else(unset),
            "attenuation_db" => Ok(ParamValue::F64(self.attenuation_db)),
            "max_taps" => Ok(ParamValue::I64(self.max_taps as i64)),
            "max_chunk" => Ok(ParamValue::I64(self.max_chunk as i64)),
            _ => unknown(),
        }
    }
}

/// Digital up-converter for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, DucBlock, DucConfig};
/// use sdrk::{Rate, C32};
/// let cfg = DucConfig { interpolation: 4, max_chunk: 64, ..DucConfig::default() };
/// let rate = Rate::hz(12_000)?;
/// let bytes = DucBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = AlignedBytes([0u8; 8192]);
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = DucBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 64];
/// let p = block.process(&[C32::new(1.0, 0.0); 16], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (16, 64));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct DucBlock<T: BlockSample, S: Storage> {
    config: DucConfig,
    rate_out: Rate,
    nco: Nco,
    fir: FirInterpState<T, S::Buf<T::Real>, S::Buf<T>>,
    scratch: S::Buf<T>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for DucBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DucBlock")
            .field("config", &self.config)
            .field("rate_out", &self.rate_out)
            .field("nco", &self.nco)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> DucBlock<T, S> {
    /// Designs the filter and creates the block; taps, history and scratch come from `storage`.
    pub fn new(config: &DucConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let l = config.interpolation;
        let rate_out = multiply_rate(rate_in, l)?;
        let spec = config.design_spec(rate_in);
        let lp = spec.lowpass(super::hz(rate_out), config.max_taps)?;
        let n = lp.len();
        let mut taps = storage.alloc::<T::Real>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <T as TunedElem>::tap(lp.tap(i) * l as f64)?;
        }
        let hist = storage.alloc::<T>(n.div_ceil(l))?;
        let scratch = storage.alloc::<T>(config.max_chunk)?;
        let mut resolved = *config;
        resolved.cutoff_hz = Some(spec.cutoff_hz);
        resolved.transition_hz = Some(spec.transition_hz);
        Ok(Self {
            config: resolved,
            rate_out,
            nco: Nco::new(config.frequency_hz, rate_out)?,
            fir: FirInterpState::new(taps, hist, l)?,
            scratch,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &DucConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<T::Real>(n)?;
        plan.add::<T>(n.div_ceil(config.interpolation))?;
        plan.add::<T>(config.max_chunk)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate times the interpolation.
    pub fn rate_out(config: &DucConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        multiply_rate(rate_in, config.interpolation)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }
}

impl<T: BlockSample, S: Storage> Block for DucBlock<T, S>
where
    Sdrk: Duc<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let (mut consumed, mut produced) = (0usize, 0usize);
        while consumed < input.len() {
            let p = <Sdrk as Duc<T>>::duc(
                &Sdrk,
                &mut self.fir,
                &mut self.nco,
                tail(input, consumed)?,
                tail_mut(output, produced)?,
                self.scratch.as_mut(),
            )?;
            if p.consumed == 0 {
                break;
            }
            consumed += p.consumed;
            produced += p.produced;
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
        self.nco.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "frequency_hz" => {
                let f = value.as_f64()?;
                check_symmetric("frequency_hz", f, super::hz(self.rate_out) / 2.0)?;
                self.nco.set_frequency(f, self.rate_out)?;
                self.config.frequency_hz = f;
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
