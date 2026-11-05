//! `costas`: carrier recovery for BPSK and QPSK at one sample per symbol (a Costas loop).

use core::marker::PhantomData;

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::costas::{Costas, CostasState};

use super::{unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Default damping of the carrier loops of M40: the value 0.7071 that `symbol_sync` uses too
/// (D550), not the f64 nearest to 1/√2.
#[allow(clippy::approx_constant)]
pub(crate) const DEFAULT_DAMPING: f64 = 0.7071;
/// Default loop bandwidth of the carrier loop (D604).
pub(crate) const DEFAULT_LOOP_BW: f64 = 0.02;
/// Default limit of the frequency integrator in cycles per symbol (D604).
pub(crate) const DEFAULT_MAX_OFFSET: f64 = 0.05;
/// Smallest loop bandwidth of the carrier loop.
pub(crate) const MIN_LOOP_BW: f64 = 0.0001;
/// Largest loop bandwidth of the carrier loop.
pub(crate) const MAX_LOOP_BW: f64 = 0.2;
/// Largest `max_offset` in cycles per symbol.
pub(crate) const MAX_OFFSET_LIMIT: f64 = 0.25;

/// Description of the `costas` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "costas",
    summary: "Recovers the carrier of BPSK or QPSK symbols: removes phase and frequency offset.",
    details: "A Costas loop at one sample per symbol: every symbol is turned by the phase of \
        an oscillator, a decision-directed detector (order 2: re im; order 4: sign(re) im - \
        sign(im) re) measures the remaining phase error and a proportional-integral loop with \
        the bandwidth loop_bw (relative to the symbol rate) and the damping corrects phase and \
        frequency; the frequency integrator is limited to max_offset cycles per symbol. A lock \
        detector reports lock above 0.5 of its smoothed metric and loses it below 0.3. Place \
        it after symbol_sync (one sample per symbol, amplitude about 1, after an agc). The \
        output is the derotated symbol; BPSK settles on the real axis, QPSK on the diagonals \
        (the Gray points of psk_mod). The phase ambiguity (180 degrees for BPSK, 90 degrees \
        for QPSK) is not resolved here: use framing with both polarities or differential \
        coding. The order-4 detector has the slope sqrt(2) at amplitude 1, so a QPSK loop is \
        about 1.4 times wider than loop_bw. No delay; one output per input.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Symbols, one sample per symbol, amplitude about 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The symbols with the carrier phase and frequency removed.",
    }],
    params: &[
        ParamDesc {
            name: "order",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 2,
                max: 4,
                default: Some(2),
            },
            typical: "2 for BPSK, 4 for QPSK",
            mutability: Mutability::Build,
            description: "Order of the phase-shift keying: 2 (BPSK) or 4 (QPSK).",
        },
        ParamDesc {
            name: "loop_bw",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: MIN_LOOP_BW,
                max: MAX_LOOP_BW,
                default: Some(DEFAULT_LOOP_BW),
            },
            typical: "0.02; 0.005 for weak signals, 0.05 for fast acquisition",
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
            name: "max_offset",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: MAX_OFFSET_LIMIT,
                default: Some(DEFAULT_MAX_OFFSET),
            },
            typical: "0.05 (5 % of the symbol rate)",
            mutability: Mutability::Build,
            description: "Limit of the frequency integrator in cycles per symbol.",
        },
    ],
    required: &[],
};

/// Configuration of [`CostasBlock`].
///
/// ```
/// use sdrk_blocks::CostasConfig;
/// let cfg = CostasConfig::default();
/// assert_eq!((cfg.order, cfg.loop_bw, cfg.max_offset), (2, 0.02, 0.05));
/// assert!(cfg.validate().is_ok());
/// assert!(CostasConfig { order: 3, ..cfg }.validate().is_err());
/// assert!(CostasConfig { loop_bw: 0.5, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostasConfig {
    /// Order: 2 (BPSK) or 4 (QPSK).
    pub order: u32,
    /// Loop bandwidth normalized to the symbol rate (0.0001 to 0.2).
    pub loop_bw: f64,
    /// Damping of the loop (0.1 to 4).
    pub damping: f64,
    /// Limit of the frequency integrator in cycles per symbol (0 to 0.25).
    pub max_offset: f64,
}

impl Default for CostasConfig {
    /// Order 2, `loop_bw` 0.02, `damping` 0.7071, `max_offset` 0.05 (D604).
    fn default() -> Self {
        Self {
            order: 2,
            loop_bw: DEFAULT_LOOP_BW,
            damping: DEFAULT_DAMPING,
            max_offset: DEFAULT_MAX_OFFSET,
        }
    }
}

/// `min <= v <= max`, otherwise `OutOfRange` (NaN included).
pub(crate) fn check_f64(name: &'static str, v: f64, min: f64, max: f64) -> Result<()> {
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

/// The order of a PSK block: 2 or 4, otherwise `InvalidParameter` (3 is inside the range of
/// the description but not a supported order).
pub(crate) fn check_order(order: u32) -> Result<()> {
    match order {
        2 | 4 => Ok(()),
        o if (2..=4).contains(&o) => Err(Error::InvalidParameter {
            name: "order",
            reason: "the order must be 2 (bpsk) or 4 (qpsk)",
        }),
        o => Err(Error::OutOfRange {
            name: "order",
            min: 2.0,
            max: 4.0,
            got: f64::from(o),
        }),
    }
}

impl CostasConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `InvalidParameter` for the order 3, `OutOfRange` for any other value outside its range.
    pub fn validate(&self) -> Result<()> {
        check_order(self.order)?;
        check_f64("loop_bw", self.loop_bw, MIN_LOOP_BW, MAX_LOOP_BW)?;
        check_f64("damping", self.damping, 0.1, 4.0)?;
        check_f64("max_offset", self.max_offset, 0.0, MAX_OFFSET_LIMIT)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for the order 3.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "order" => {
                let o = u32::try_from(value.as_i64()?).unwrap_or(0);
                check_order(o)?;
                self.order = o;
            }
            "loop_bw" => self.loop_bw = value.as_f64()?,
            "damping" => self.damping = value.as_f64()?,
            "max_offset" => self.max_offset = value.as_f64()?,
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
            "order" => Ok(ParamValue::I64(i64::from(self.order))),
            "loop_bw" => Ok(ParamValue::F64(self.loop_bw)),
            "damping" => Ok(ParamValue::F64(self.damping)),
            "max_offset" => Ok(ParamValue::F64(self.max_offset)),
            _ => unknown(),
        }
    }

    /// The kernel state of the configuration.
    fn state(&self) -> Result<CostasState> {
        self.validate()?;
        Ok(CostasState::new(
            self.order,
            self.loop_bw,
            self.damping,
            self.max_offset,
        )?)
    }
}

/// Carrier recovery for `C32` symbols at one sample per symbol: the kernel `costas` through
/// the facade. Timing: ratio 1, delay 0. The block needs no storage.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, CostasBlock, CostasConfig};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = CostasConfig { order: 2, ..CostasConfig::default() };
/// let mut costas = CostasBlock::new(&cfg, Rate::hz(1000)?, &mut arena)?;
/// // BPSK symbols turned by 0.5 rad.
/// let (s, c) = (0.5f32.sin(), 0.5f32.cos());
/// let input: Vec<C32> = (0..2000)
///     .map(|k| if (k * 7) % 3 == 0 { C32::new(c, s) } else { C32::new(-c, -s) })
///     .collect();
/// let mut out = vec![C32::default(); input.len()];
/// let p = costas.process(&input, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (2000, 2000));
/// assert!(costas.locked());
/// assert!(out[1999].im.abs() < 1e-3 && (out[1999].re.abs() - 1.0).abs() < 1e-3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct CostasBlock<S: Storage> {
    config: CostasConfig,
    state: CostasState,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> CostasBlock<S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// As [`CostasConfig::validate`].
    pub fn new(config: &CostasConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        Ok(Self {
            config: *config,
            state: config.state()?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// As [`CostasConfig::validate`].
    pub fn storage_bytes(config: &CostasConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        Ok(Plan::default().bytes())
    }

    /// Whether the lock detector reports lock.
    pub fn locked(&self) -> bool {
        self.state.locked()
    }

    /// The frequency integrator: the estimated carrier offset in cycles per symbol.
    pub fn frequency(&self) -> f64 {
        self.state.frequency()
    }

    /// The kernel state (for inspection).
    pub fn state(&self) -> &CostasState {
        &self.state
    }
}

impl<S: Storage> Block for CostasBlock<S> {
    type In = C32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        let (Some(i), Some(o)) = (input.get(..n), output.get_mut(..n)) else {
            return Err(Error::Core(sdrk_core::Error::Internal("costas split")));
        };
        <Sdrk as Costas>::costas(&Sdrk, &mut self.state, i, o)?;
        Ok(Progress {
            consumed: n,
            produced: n,
        })
    }

    fn timing(&self) -> BlockTiming {
        unit_timing()
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
        1
    }
}
