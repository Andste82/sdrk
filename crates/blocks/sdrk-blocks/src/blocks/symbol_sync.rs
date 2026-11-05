//! `symbol_sync`: symbol timing recovery (Gardner or Mueller–Müller detector, cubic
//! interpolator), one output sample per symbol.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::Sample;
use sdrk_core::kernels::symbol_sync::{MAX_SPS, MIN_SPS, SymbolSync, SymbolSyncState, Ted};

use super::{HZ_LIMIT, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::StreamElem;
use crate::storage::{Plan, Storage};

/// The names of the `ted` choice, in index order.
const TED_OPTIONS: &[&str] = &["gardner", "mm"];

/// Smallest `symbol_rate_hz` (one step of the 1/1000 Hz grid).
const MIN_SYMBOL_RATE_HZ: f64 = 0.001;

/// Default damping: the value 0.7071 of the specification (§2 of M38), which the golden cases
/// use too, not the f64 nearest to 1/√2.
#[allow(clippy::approx_constant)]
const DEFAULT_DAMPING: f64 = 0.7071;

/// Smallest `ted_gain` (D549).
const MIN_TED_GAIN: f64 = 1e-6;
/// Largest `ted_gain` (D549).
const MAX_TED_GAIN: f64 = 1e6;

/// Description of the `symbol_sync` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "symbol_sync",
    summary: "Recovers the symbol clock and outputs one interpolated sample per symbol.",
    details: "A timing error detector (ted gardner, which needs no decisions and no carrier \
        lock, or mm for Mueller-Mueller, which needs symbols of about +1/-1 per component) \
        drives a proportional-integral loop with the bandwidth loop_bw (relative to the symbol \
        rate) and the damping; a cubic Lagrange interpolator computes the sample at each \
        strobe. The output has one sample per symbol: its nominal rate is symbol_rate_hz \
        (rounded to 1/1000 Hz), the actual rate follows the clock of the signal within \
        max_deviation (a fraction of the samples per symbol), so the ratio kind is variable. \
        Place it after the matched filter and an agc: the loop gain is proportional to the \
        signal power, so the input amplitude should be about 1 (ted_gain corrects other levels \
        and the detector slope). The sample rate must be at least twice the symbol rate. \
        Works on C32 (PSK) and f32 (for example an FSK discriminator output).",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32, PortType::F32],
        description: "Matched-filtered samples, at least 2 per symbol, amplitude about 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32, PortType::F32],
        description: "One sample per symbol at the recovered symbol instants; same type as \
            the input.",
    }],
    params: &[
        ParamDesc {
            name: "symbol_rate_hz",
            unit: Unit::Hz,
            kind: ParamKind::F64 {
                min: MIN_SYMBOL_RATE_HZ,
                max: HZ_LIMIT,
                default: None,
            },
            typical: "1187.5 for RDS, 9600 for packet radio",
            mutability: Mutability::Build,
            description: "Nominal symbol rate in Hz, rounded to 1/1000 Hz; the sample rate \
                divided by it is the samples per symbol (2 to 1000000).",
        },
        ParamDesc {
            name: "ted",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: TED_OPTIONS,
                default: 0,
            },
            typical: "gardner; mm at 2 samples per symbol with a locked carrier",
            mutability: Mutability::Build,
            description: "Timing error detector: gardner (two interpolations per symbol) or mm \
                (Mueller-Mueller, decisions sign(y)).",
        },
        ParamDesc {
            name: "loop_bw",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0001,
                max: 0.1,
                default: Some(0.01),
            },
            typical: "0.01; 0.005 for weak signals, 0.02 for fast acquisition",
            mutability: Mutability::Build,
            description: "Loop bandwidth normalized to the symbol rate.",
        },
        ParamDesc {
            name: "damping",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.1,
                max: 4.0,
                default: Some(DEFAULT_DAMPING),
            },
            typical: "0.7071",
            mutability: Mutability::Build,
            description: "Damping factor of the loop.",
        },
        ParamDesc {
            name: "max_deviation",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: 0.05,
                default: Some(0.005),
            },
            typical: "0.005 (5000 ppm)",
            mutability: Mutability::Build,
            description: "Largest deviation of the symbol period from its nominal value, as a \
                fraction.",
        },
        ParamDesc {
            name: "ted_gain",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: MIN_TED_GAIN,
                max: MAX_TED_GAIN,
                default: Some(1.0),
            },
            typical: "1.0 (the detector slope per symbol: about 1.08 for gardner and 1.78 for \
                mm on a raised-cosine 0.35 input of amplitude 1)",
            mutability: Mutability::Build,
            description: "Slope of the detector; the loop gains are divided by it.",
        },
    ],
    required: &["symbol_rate_hz"],
};

/// Configuration of [`SymbolSyncBlock`].
///
/// ```
/// use sdrk_blocks::{SymbolSyncConfig, Ted};
/// let cfg = SymbolSyncConfig { symbol_rate_hz: 2375.0, ..SymbolSyncConfig::default() };
/// assert_eq!(cfg.ted, Ted::Gardner);
/// assert!(cfg.validate(sdrk::Rate::hz(19_000)?).is_ok()); // 8 samples per symbol
/// // fewer than 2 samples per symbol are rejected
/// assert!(cfg.validate(sdrk::Rate::hz(4_000)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymbolSyncConfig {
    /// Nominal symbol rate in Hz (required; rounded to 1/1000 Hz).
    pub symbol_rate_hz: f64,
    /// The timing error detector.
    pub ted: Ted,
    /// Loop bandwidth normalized to the symbol rate (0.0001 to 0.1).
    pub loop_bw: f64,
    /// Damping of the loop (0.1 to 4).
    pub damping: f64,
    /// Largest deviation of the symbol period, as a fraction (0 to 0.05).
    pub max_deviation: f64,
    /// Slope of the detector the loop gains are divided by (1e-6 to 1e6).
    pub ted_gain: f64,
}

impl Default for SymbolSyncConfig {
    /// `symbol_rate_hz` 1200 (a placeholder: the pipeline requires the parameter), Gardner,
    /// `loop_bw` 0.01, `damping` 0.7071, `max_deviation` 0.005, `ted_gain` 1.
    fn default() -> Self {
        Self {
            symbol_rate_hz: 1200.0,
            ted: Ted::Gardner,
            loop_bw: 0.01,
            damping: DEFAULT_DAMPING,
            max_deviation: 0.005,
            ted_gain: 1.0,
        }
    }
}

/// The exact figures of a configuration for an input rate.
#[derive(Debug, Clone, Copy)]
struct Derived {
    /// Output rate / input rate, `symbol_rate / fs`, reduced.
    ratio: Ratio,
    /// The nominal symbol rate on the 1/1000 Hz grid.
    rate_out: Rate,
    /// Samples per symbol, `fs / symbol_rate` rounded once to f64.
    sps: f64,
}

fn check_range(name: &'static str, v: f64, min: f64, max: f64) -> Result<()> {
    if v >= min && v <= max {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name,
            min,
            max,
            got: v,
        })
    }
}

impl SymbolSyncConfig {
    /// The ranges of the description and the samples per symbol: `fs / symbol_rate_hz`
    /// (with the rate rounded to 1/1000 Hz) must lie between 2 and 1000000, otherwise
    /// `OutOfRange` for `symbol_rate_hz`.
    fn derive(&self, rate_in: Rate) -> Result<Derived> {
        check_range(
            "symbol_rate_hz",
            self.symbol_rate_hz,
            MIN_SYMBOL_RATE_HZ,
            HZ_LIMIT,
        )?;
        check_range("loop_bw", self.loop_bw, 0.0001, 0.1)?;
        check_range("damping", self.damping, 0.1, 4.0)?;
        check_range("max_deviation", self.max_deviation, 0.0, 0.05)?;
        check_range("ted_gain", self.ted_gain, MIN_TED_GAIN, MAX_TED_GAIN)?;
        let fs = rate_in.as_f64();
        let sps_range = Error::OutOfRange {
            name: "symbol_rate_hz",
            min: fs / MAX_SPS,
            max: fs / MIN_SPS,
            got: self.symbol_rate_hz,
        };
        // In mHz: at most 1e15, exact in u64.
        let mhz = libm::floor(self.symbol_rate_hz * 1000.0 + 0.5) as u64;
        let rate_out = Rate::new(mhz, 1000)?;
        // symbol_rate / fs = (mhz / 1000) / (num / den) = mhz den / (1000 num).
        let num = u128::from(mhz) * u128::from(rate_in.den());
        let den = 1000u128 * u128::from(rate_in.num());
        // sps = den / num must lie in [2, 1e6]; compared exactly in integers.
        if den < 2 * num || den > 1_000_000 * num {
            return Err(sps_range);
        }
        let g = gcd(num, den);
        let (num, den) = (num / g, den / g);
        let (Ok(n64), Ok(d64)) = (u64::try_from(num), u64::try_from(den)) else {
            return Err(sps_range);
        };
        let ratio = Ratio::new(n64, d64)?;
        Ok(Derived {
            ratio,
            rate_out,
            sps: d64 as f64 / n64 as f64,
        })
    }

    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// `OutOfRange` for a parameter outside its range or a symbol rate that gives fewer than
    /// 2 or more than 1000000 samples per symbol.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.derive(rate_in).map(|_| ())
    }

    /// The samples per symbol the kernel gets: `fs / symbol_rate_hz` with the symbol rate on
    /// the 1/1000 Hz grid, as the exact reduced fraction rounded once to f64.
    ///
    /// ```
    /// use sdrk_blocks::SymbolSyncConfig;
    /// let cfg = SymbolSyncConfig { symbol_rate_hz: 1187.5, ..SymbolSyncConfig::default() };
    /// assert_eq!(cfg.sps(sdrk::Rate::hz(19_000)?)?, 16.0);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    ///
    /// # Errors
    /// As [`SymbolSyncConfig::validate`].
    pub fn sps(&self, rate_in: Rate) -> Result<f64> {
        Ok(self.derive(rate_in)?.sps)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "symbol_rate_hz" => self.symbol_rate_hz = value.as_f64()?,
            "ted" => {
                self.ted = if value.as_choice()? == 0 {
                    Ted::Gardner
                } else {
                    Ted::MuellerMuller
                };
            }
            "loop_bw" => self.loop_bw = value.as_f64()?,
            "damping" => self.damping = value.as_f64()?,
            "max_deviation" => self.max_deviation = value.as_f64()?,
            "ted_gain" => self.ted_gain = value.as_f64()?,
            _ => return unknown(),
        }
        Ok(())
    }

    /// Current value of a parameter.
    ///
    /// # Errors
    /// `UnknownParameter`.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match name {
            "symbol_rate_hz" => Ok(ParamValue::F64(self.symbol_rate_hz)),
            "ted" => Ok(ParamValue::Choice(match self.ted {
                Ted::MuellerMuller => 1,
                _ => 0,
            })),
            "loop_bw" => Ok(ParamValue::F64(self.loop_bw)),
            "damping" => Ok(ParamValue::F64(self.damping)),
            "max_deviation" => Ok(ParamValue::F64(self.max_deviation)),
            "ted_gain" => Ok(ParamValue::F64(self.ted_gain)),
            _ => unknown(),
        }
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}

/// Symbol timing recovery for `C32` and `f32` (no Q15 variants, D529).
///
/// The block calls the kernel `symbol_sync` through the facade with `sps = fs /
/// symbol_rate_hz` and the loop parameters of the configuration. Its timing is
/// `RatioKind::Variable` with the exact nominal ratio `symbol_rate / fs` and a delay of 1
/// input sample. A call consumes all input when the output has room (the last 3 samples stay
/// in the kernel's history); when the output is full it stops, and the caller resubmits the
/// unconsumed input. The block needs no storage.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, SymbolSyncBlock, SymbolSyncConfig};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = SymbolSyncConfig { symbol_rate_hz: 1000.0, loop_bw: 0.05, ..SymbolSyncConfig::default() };
/// let mut sync = SymbolSyncBlock::<f32, _>::new(&cfg, Rate::hz(4000)?, &mut arena)?;
/// // Alternating symbols +1, -1 with 4 samples each.
/// let input: Vec<f32> = (0..4000).map(|n| if (n / 4) % 2 == 0 { 1.0 } else { -1.0 }).collect();
/// let mut out = vec![0.0f32; 1100];
/// let p = sync.process(&input, &mut out)?;
/// assert_eq!(p.consumed, 4000);
/// assert!((995..=1005).contains(&p.produced));
/// assert!(out[p.produced - 100..p.produced].iter().all(|y| y.abs() == 1.0));
/// assert_eq!(sync.timing().ratio, sdrk::Ratio::new(1, 4)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct SymbolSyncBlock<T: Sample, S: Storage> {
    config: SymbolSyncConfig,
    state: SymbolSyncState<T>,
    ratio: Ratio,
    min_input: usize,
    _storage: PhantomData<fn() -> S>,
}

impl<T: Sample, S: Storage> SymbolSyncBlock<T, S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration (see [`SymbolSyncConfig::validate`]);
    /// `Core(Unsupported)` for a sample type other than `C32` and `f32`.
    pub fn new(config: &SymbolSyncConfig, rate_in: Rate, _storage: &mut S) -> Result<Self> {
        let d = config.derive(rate_in)?;
        let state = SymbolSyncState::new(
            d.sps,
            config.ted,
            config.loop_bw,
            config.damping,
            config.max_deviation,
            config.ted_gain,
        )?;
        Ok(Self {
            config: *config,
            state,
            ratio: d.ratio,
            min_input: min_input(d.sps, config.max_deviation),
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// As [`SymbolSyncConfig::validate`].
    pub fn storage_bytes(config: &SymbolSyncConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        Ok(Plan::default().bytes())
    }

    /// The nominal output rate: `symbol_rate_hz` on the 1/1000 Hz grid.
    ///
    /// # Errors
    /// As [`SymbolSyncConfig::validate`].
    pub fn rate_out(config: &SymbolSyncConfig, rate_in: Rate) -> Result<Rate> {
        Ok(config.derive(rate_in)?.rate_out)
    }

    /// The current estimate of the samples per symbol.
    pub fn sps_estimate(&self) -> f64 {
        self.state.sps_estimate()
    }

    /// The kernel state (for inspection).
    pub fn state(&self) -> &SymbolSyncState<T> {
        &self.state
    }
}

/// Input that surely yields an output: the largest strobe step `1.25 omega` (D534) with
/// `omega` at its upper limit, plus the interpolation window.
fn min_input(sps: f64, max_deviation: f64) -> usize {
    let step = libm::ceil(sps * (1.0 + max_deviation) * 1.25);
    (step as usize).saturating_add(3)
}

impl<T: Sample + StreamElem, S: Storage> Block for SymbolSyncBlock<T, S>
where
    Sdrk: SymbolSync<T>,
{
    type In = T;
    type Out = T;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[T], output: &mut [T]) -> Result<Progress> {
        Ok(<Sdrk as SymbolSync<T>>::symbol_sync(
            &Sdrk,
            &mut self.state,
            input,
            output,
        )?)
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.ratio,
            kind: RatioKind::Variable,
            delay: Delay::new(1, 1).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        match DESCRIPTION.param(name) {
            Some(_) => Err(Error::ReadOnlyParameter),
            None => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config.get(name)
    }

    fn min_input(&self) -> usize {
        self.min_input
    }
}
