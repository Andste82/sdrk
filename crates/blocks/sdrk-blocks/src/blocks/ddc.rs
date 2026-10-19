//! `ddc`: digital down-converter (mixer, low-pass and decimation in one block).

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::decim::FirDecimState;
use sdrk_core::kernels::fused::Ddc;
use sdrk_core::kernels::mix::Nco;
use sdrk_core::timing::Timing;

use super::fir::MAX_TAPS_LIMIT;
use super::{HZ_LIMIT, check_count, check_symmetric, divide_rate, tail, tail_mut, unknown};
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

/// Description of the `ddc` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "ddc",
    summary: "Shifts a signal to 0 Hz, low-pass filters it and decimates it.",
    details: "A digital down-converter: a numerically controlled oscillator mixes the signal at \
        +frequency_hz down to 0 Hz, a Kaiser-window low-pass removes everything outside the \
        wanted channel and the result is decimated by the given factor. The output rate is \
        the input rate divided by decimation. Unless given, the cutoff is 0.4 times the output \
        rate and the transition width 0.1 times the output rate, so the stop band starts at \
        half the output rate and nothing aliases. frequency_hz can change while the block runs \
        and the oscillator phase stays continuous. The block needs a scratch buffer of \
        max_chunk samples; a larger chunk means fewer kernel calls.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples at the input rate.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples at the input rate divided by decimation.",
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
            typical: "600e3 for a carrier 600 kHz above the tuner centre",
            mutability: Mutability::Runtime,
            description: "Frequency of the wanted signal relative to the centre in Hz; it is \
                shifted to 0 Hz. Within plus or minus half the input rate.",
        },
        ParamDesc {
            name: "decimation",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: 1024,
                default: None,
            },
            typical: "10 to go from 2.4 MHz to 240 kHz",
            mutability: Mutability::Build,
            description: "Decimation factor, 1 to 1024. Required.",
        },
        ParamDesc {
            name: "cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "100e3 for FM broadcast; default 0.4 times the output rate",
            mutability: Mutability::Build,
            description: "Cutoff (-6 dB point) of the channel filter in Hz. The default is \
                0.4 * fs / decimation.",
        },
        ParamDesc {
            name: "transition_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "default 0.1 times the output rate",
            mutability: Mutability::Build,
            description: "Width of the transition band in Hz. The default is 0.1 * fs / decimation.",
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
            name: "max_chunk",
            unit: Unit::Samples,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_CHUNK_LIMIT,
                default: Some(4096),
            },
            typical: "4096",
            mutability: Mutability::Build,
            description: "Size of the scratch buffer in samples; larger input is processed in \
                several steps.",
        },
    ],
    required: &["decimation"],
};

/// Configuration of [`DdcBlock`].
///
/// ```
/// use sdrk_blocks::DdcConfig;
/// let cfg = DdcConfig { frequency_hz: 600e3, decimation: 10, ..DdcConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(2_400_000)?).is_ok());
/// assert_eq!(cfg.num_taps(sdrk::Rate::hz(2_400_000)?)?, 365);
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DdcConfig {
    /// Frequency of the wanted signal in Hz (within plus or minus half the input rate).
    pub frequency_hz: f64,
    /// Decimation factor, 1 to 1024.
    pub decimation: usize,
    /// Cutoff in Hz; `None` means `0.4 * fs / decimation`.
    pub cutoff_hz: Option<f64>,
    /// Transition width in Hz; `None` means `0.1 * fs / decimation`.
    pub transition_hz: Option<f64>,
    /// Stop-band attenuation in dB.
    pub attenuation_db: f64,
    /// Upper limit of the number of taps.
    pub max_taps: usize,
    /// Size of the scratch buffer in samples.
    pub max_chunk: usize,
}

impl Default for DdcConfig {
    fn default() -> Self {
        Self {
            frequency_hz: 0.0,
            decimation: 10,
            cutoff_hz: None,
            transition_hz: None,
            attenuation_db: 60.0,
            max_taps: 1023,
            max_chunk: 4096,
        }
    }
}

impl DdcConfig {
    /// The filter specification for the input rate with the defaults filled in.
    ///
    /// ```
    /// use sdrk_blocks::DdcConfig;
    /// let spec = DdcConfig::default().design_spec(sdrk::Rate::hz(2_400_000)?);
    /// assert_eq!((spec.cutoff_hz, spec.transition_hz), (96_000.0, 24_000.0));
    /// # Ok::<(), sdrk::Error>(())
    /// ```
    pub fn design_spec(&self, rate_in: Rate) -> DesignSpec {
        let out = super::hz(rate_in) / self.decimation.max(1) as f64;
        DesignSpec {
            cutoff_hz: self.cutoff_hz.unwrap_or(0.4 * out),
            transition_hz: self.transition_hz.unwrap_or(0.1 * out),
            attenuation_db: self.attenuation_db,
        }
    }

    /// Number of taps for the input rate (also checks the whole configuration).
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        check_count("decimation", self.decimation, 1, 1024)?;
        check_count("max_chunk", self.max_chunk, 1, MAX_CHUNK_LIMIT as usize)?;
        let half = super::hz(rate_in) / 2.0;
        check_symmetric("frequency_hz", self.frequency_hz, half)?;
        self.design_spec(rate_in)
            .num_taps(super::hz(rate_in), self.max_taps)
    }

    /// Checks the configuration against the input sample rate.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.num_taps(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// ```
    /// use sdrk_blocks::{DdcConfig, Error, ParamValue};
    /// let mut cfg = DdcConfig::default();
    /// cfg.set("decimation", ParamValue::I64(8))?;
    /// assert_eq!(cfg.decimation, 8);
    /// assert!(matches!(cfg.set("decimation", ParamValue::I64(0)), Err(Error::OutOfRange { .. })));
    /// assert_eq!(cfg.set("nope", ParamValue::I64(0)), Err(Error::UnknownParameter));
    /// # Ok::<(), Error>(())
    /// ```
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "frequency_hz" => self.frequency_hz = value.as_f64()?,
            "decimation" => self.decimation = value.as_usize()?,
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
    ///
    /// ```
    /// use sdrk_blocks::{DdcConfig, ParamValue};
    /// let cfg = DdcConfig::default();
    /// assert_eq!(cfg.get("decimation"), Ok(ParamValue::I64(10)));
    /// assert!(cfg.get("cutoff_hz").is_err()); // 0.4 * fs / decimation, known only with the rate
    /// ```
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
            "decimation" => Ok(ParamValue::I64(self.decimation as i64)),
            "cutoff_hz" => self.cutoff_hz.map(ParamValue::F64).ok_or_else(unset),
            "transition_hz" => self.transition_hz.map(ParamValue::F64).ok_or_else(unset),
            "attenuation_db" => Ok(ParamValue::F64(self.attenuation_db)),
            "max_taps" => Ok(ParamValue::I64(self.max_taps as i64)),
            "max_chunk" => Ok(ParamValue::I64(self.max_chunk as i64)),
            _ => unknown(),
        }
    }
}

/// Digital down-converter for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, DdcBlock, DdcConfig};
/// use sdrk::{Rate, C32};
/// let cfg = DdcConfig { decimation: 4, max_chunk: 64, ..DdcConfig::default() };
/// let rate = Rate::hz(48_000)?;
/// let bytes = DdcBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut mem = AlignedBytes([0u8; 4096]);
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = DdcBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 16];
/// let p = block.process(&[C32::new(1.0, 0.0); 64], &mut out)?;
/// assert_eq!((p.consumed, p.produced), (64, 16));
/// assert_eq!(DdcBlock::<C32, ArenaStorage<'_>>::rate_out(&cfg, rate)?, Rate::hz(12_000)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct DdcBlock<T: BlockSample, S: Storage> {
    config: DdcConfig,
    rate_in: Rate,
    nco: Nco,
    fir: FirDecimState<T, S::Buf<T::Real>, S::Buf<T>>,
    scratch: S::Buf<T>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for DdcBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DdcBlock")
            .field("config", &self.config)
            .field("rate_in", &self.rate_in)
            .field("nco", &self.nco)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> DdcBlock<T, S> {
    /// Designs the filter and creates the block; taps, history and scratch come from `storage`.
    pub fn new(config: &DdcConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let fs = super::hz(rate_in);
        let spec = config.design_spec(rate_in);
        let lp = spec.lowpass(fs, config.max_taps)?;
        let n = lp.len();
        let mut taps = storage.alloc::<T::Real>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <T as TunedElem>::tap(lp.tap(i))?;
        }
        let hist = storage.alloc::<T>(n - 1)?;
        let scratch = storage.alloc::<T>(config.max_chunk)?;
        let mut resolved = *config;
        resolved.cutoff_hz = Some(spec.cutoff_hz);
        resolved.transition_hz = Some(spec.transition_hz);
        Ok(Self {
            config: resolved,
            rate_in,
            nco: Nco::new(-config.frequency_hz, rate_in)?,
            fir: FirDecimState::new(taps, hist, config.decimation)?,
            scratch,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    pub fn storage_bytes(config: &DdcConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<T::Real>(n)?;
        plan.add::<T>(n - 1)?;
        plan.add::<T>(config.max_chunk)?;
        Ok(plan.bytes())
    }

    /// Output rate: the input rate divided by the decimation.
    pub fn rate_out(config: &DdcConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        divide_rate(rate_in, config.decimation)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }
}

impl<T: BlockSample, S: Storage> Block for DdcBlock<T, S>
where
    Sdrk: Ddc<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let (mut consumed, mut produced) = (0usize, 0usize);
        while consumed < input.len() {
            let p = <Sdrk as Ddc<T>>::ddc(
                &Sdrk,
                &mut self.nco,
                &mut self.fir,
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
                check_symmetric("frequency_hz", f, super::hz(self.rate_in) / 2.0)?;
                self.nco.set_frequency(-f, self.rate_in)?;
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
        self.config.decimation
    }
}
