//! `fir`: low-pass FIR filter designed from a cutoff, a transition width and an attenuation.

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::timing::Timing;

use super::{HZ_LIMIT, head, head_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::DesignSpec;
use crate::error::{Error, Result};
use crate::sample::{BlockSample, TunedElem};
use crate::storage::{Plan, Storage};

/// Largest value of the `max_taps` parameter.
pub(crate) const MAX_TAPS_LIMIT: i64 = 65_536;

/// Description of the `fir` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "fir",
    summary: "Low-pass FIR filter designed from cutoff, transition width and attenuation.",
    details: "Designs a linear-phase low-pass with a Kaiser window (kaiserord and firwin) at the \
        input sample rate and applies it to a complex signal. The number of taps follows from \
        the transition width and the attenuation; the design fails if it exceeds max_taps. The \
        stop band starts at cutoff plus half the transition width, which must not exceed half \
        the sample rate. The group delay is (taps - 1) / 2 samples. Pitfall: narrow transition \
        bands at high sample rates need many taps and a lot of storage.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::CQ15],
        description: "Complex samples.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::CQ15],
        description: "The filtered samples, same type and rate as the input.",
    }],
    params: &[
        ParamDesc {
            name: "cutoff_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "15e3 for an audio-bandwidth filter at 48 kHz",
            mutability: Mutability::Build,
            description: "Cutoff (-6 dB point) in Hz; cutoff plus half the transition width must \
                not exceed half the sample rate. Required.",
        },
        ParamDesc {
            name: "transition_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: 1e-6,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "0.1 times the cutoff",
            mutability: Mutability::Build,
            description: "Width of the transition band in Hz, centred on the cutoff. Required.",
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
    ],
    required: &["cutoff_hz", "transition_hz"],
};

/// Configuration of [`FirBlock`].
///
/// ```
/// use sdrk_blocks::FirConfig;
/// let cfg = FirConfig { cutoff_hz: 15_000.0, transition_hz: 3_000.0, ..FirConfig::default() };
/// assert!(cfg.validate(sdrk::Rate::hz(240_000)?).is_ok());
/// # Ok::<(), sdrk::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirConfig {
    /// Cutoff in Hz.
    pub cutoff_hz: f64,
    /// Width of the transition band in Hz.
    pub transition_hz: f64,
    /// Stop-band attenuation in dB.
    pub attenuation_db: f64,
    /// Upper limit of the number of taps.
    pub max_taps: usize,
}

impl Default for FirConfig {
    fn default() -> Self {
        Self {
            cutoff_hz: 10_000.0,
            transition_hz: 2_000.0,
            attenuation_db: 60.0,
            max_taps: 1023,
        }
    }
}

impl FirConfig {
    fn spec(&self) -> DesignSpec {
        DesignSpec {
            cutoff_hz: self.cutoff_hz,
            transition_hz: self.transition_hz,
            attenuation_db: self.attenuation_db,
        }
    }

    /// Number of taps for the input rate (also checks the whole configuration).
    ///
    /// ```
    /// use sdrk_blocks::FirConfig;
    /// let cfg = FirConfig { cutoff_hz: 4_000.0, transition_hz: 2_000.0, ..FirConfig::default() };
    /// let taps = cfg.num_taps(sdrk::Rate::hz(48_000)?)?;
    /// assert_eq!(taps % 2, 1);
    /// assert!(FirConfig { max_taps: 10, ..cfg }.num_taps(sdrk::Rate::hz(48_000)?).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        self.spec().num_taps(super::hz(rate_in), self.max_taps)
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
            "cutoff_hz" => self.cutoff_hz = value.as_f64()?,
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
            "cutoff_hz" => Ok(ParamValue::F64(self.cutoff_hz)),
            "transition_hz" => Ok(ParamValue::F64(self.transition_hz)),
            "attenuation_db" => Ok(ParamValue::F64(self.attenuation_db)),
            "max_taps" => Ok(ParamValue::I64(self.max_taps as i64)),
            _ => unknown(),
        }
    }
}

/// Low-pass FIR filter for `C32` and `CQ15`.
///
/// ```
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, FirBlock, FirConfig};
/// use sdrk::{Rate, C32};
/// let cfg = FirConfig { cutoff_hz: 4_000.0, transition_hz: 2_000.0, ..FirConfig::default() };
/// let rate = Rate::hz(48_000)?;
/// let mut mem = AlignedBytes([0u8; 2048]);
/// let bytes = FirBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut block = FirBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
/// let mut out = [C32::default(); 1];
/// block.process(&[C32::new(1.0, 0.0)], &mut out)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FirBlock<T: BlockSample, S: Storage> {
    config: FirConfig,
    state: FirState<T, S::Buf<T::Real>, S::Buf<T>>,
}

impl<T: BlockSample, S: Storage> core::fmt::Debug for FirBlock<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FirBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<T: BlockSample, S: Storage> FirBlock<T, S> {
    /// Designs the taps and creates the block; the taps and the history come from `storage`.
    pub fn new(config: &FirConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let fs = super::hz(rate_in);
        let lp = config.spec().lowpass(fs, config.max_taps)?;
        let n = lp.len();
        let mut taps = storage.alloc::<T::Real>(n)?;
        for (i, t) in taps.as_mut().iter_mut().enumerate() {
            *t = <T as TunedElem>::tap(lp.tap(i))?;
        }
        let hist = storage.alloc::<T>(n - 1)?;
        Ok(Self {
            config: *config,
            state: FirState::new(taps, hist)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// ```
    /// use sdrk_blocks::{ArenaStorage, FirBlock, FirConfig};
    /// use sdrk::{Rate, C32, CQ15};
    /// let cfg = FirConfig { cutoff_hz: 4_000.0, transition_hz: 2_000.0, ..FirConfig::default() };
    /// let rate = Rate::hz(48_000)?;
    /// let c32 = FirBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
    /// let cq15 = FirBlock::<CQ15, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
    /// assert!(cq15 < c32); // 16-bit samples need half the memory
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn storage_bytes(config: &FirConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<T::Real>(n)?;
        plan.add::<T>(n - 1)?;
        Ok(plan.bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    pub fn rate_out(config: &FirConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        Ok(rate_in)
    }

    /// Number of taps of the designed filter.
    pub fn num_taps(&self) -> usize {
        self.state.taps().len()
    }
}

impl<T: BlockSample, S: Storage> Block for FirBlock<T, S>
where
    Sdrk: Fir<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Fir<T>>::fir(
            &Sdrk,
            &mut self.state,
            head(input, n)?,
            head_mut(output, n)?,
        )?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.state.ratio(),
            kind: self.state.ratio_kind(),
            delay: self.state.group_delay(),
        }
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        if DESCRIPTION.param(name).is_some() {
            Err(Error::ReadOnlyParameter)
        } else {
            unknown()
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        1
    }
}
