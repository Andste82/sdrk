//! `equalize`: adaptive equalizer on PSK symbols (CMA, then decision-directed, normalized
//! LMS; optional decision feedback; a decision-directed phase tracker on the output).

use sdrk::{Progress, Rate, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::equalize::{EqConstellation, EqMode, EqParams, Equalize, EqualizeState};
use sdrk_core::timing::Timing;

use super::{check_count, head, head_mut, unit_timing, unknown};
use crate::block::{Block, BlockTiming, ParamValue, runtime_param};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};

/// Smallest and largest number of forward taps (odd only).
pub(crate) const EQ_TAPS: (usize, usize) = (3, 63);
/// Largest number of feedback taps.
pub(crate) const EQ_FEEDBACK_MAX: usize = 63;
/// Smallest and largest (normalized) step size.
const MU: (f64, f64) = (0.000_01, 1.0);
/// Smallest non-zero and largest bandwidth of the phase tracker.
const PHASE_BW: (f64, f64) = (0.0001, 0.2);
/// Default number of taps.
pub(crate) const DEFAULT_TAPS: usize = 11;
/// Default CMA step size (normalized, D828).
pub(crate) const DEFAULT_MU_CMA: f64 = 0.05;
/// Default decision-mode step size (normalized, D828).
pub(crate) const DEFAULT_MU_DD: f64 = 0.02;
/// Default number of symbols in CMA mode (D828).
pub(crate) const DEFAULT_CMA_SYMBOLS: u32 = 1000;
/// Default bandwidth of the phase tracker (that of `costas`).
pub(crate) const DEFAULT_PHASE_BW: f64 = 0.02;
/// The names of the constellations, in the order of the choice.
const CONSTELLATIONS: &[&str] = &["bpsk", "qpsk", "8psk"];

/// Description of the `equalize` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "equalize",
    summary: "Adaptive equalizer: removes the inter-symbol interference of multipath and filters \
        from PSK symbols.",
    details: "An adaptive filter at one sample per symbol: y = sum w[i] x[i] over the joint \
        vector x of the last taps symbols (newest first) and, with feedback_taps above 0, the \
        last feedback_taps decisions (a decision feedback equalizer). A decision-directed \
        phase tracker (bandwidth phase_bw, 0 = off) turns the output back, z = y e^(-j theta), \
        and the decision d is the nearest point of the constellation to z. The weights start \
        with the centre tap 1 and adapt with the normalized LMS step mu / (sum |x|^2 + 1e-6): \
        blind with the constant-modulus algorithm (e = y (|y|^2 - 1), each part clipped to \
        [-1, 1], step mu_cma) for the first cma_symbols symbols, then decision-directed \
        (e = (z - d) e^(j theta), step mu_dd); w[i] = w[i] - mu e conj(x[i]). The decisions \
        are fed back (as d e^(j theta)) only in decision mode. Chain position: after \
        symbol_sync and before costas (as in GNU Radio's PSK receiver): CMA needs no carrier \
        phase, the tracker gives the decision mode its phase, and costas after it removes what \
        remains. The input amplitude should be about 1 (an agc before symbol_sync). \
        cma_symbols 0 starts in decision mode, 4294967295 never leaves CMA (for differential \
        detection). A larger mu adapts faster but leaves more noise in the weights. Weights \
        and history need 2 (taps + feedback_taps) c32 values of storage. Delay (taps - 1) / 2 \
        symbols; one output per input; a non-finite input gives an output of zero and leaves \
        the weights unchanged.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Symbols, one sample per symbol, amplitude about 1 (after symbol_sync).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The equalized symbols, turned back by the phase tracker; same rate as \
            the input.",
    }],
    params: &[
        ParamDesc {
            name: "taps",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: EQ_TAPS.0 as i64,
                max: EQ_TAPS.1 as i64,
                default: Some(DEFAULT_TAPS as i64),
            },
            typical: "11; 15 to 21 for long echoes",
            mutability: Mutability::Build,
            description: "Number of forward weights, odd, 3 to 63; it must span the echoes of \
                the channel on both sides of the main path.",
        },
        ParamDesc {
            name: "feedback_taps",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: EQ_FEEDBACK_MAX as i64,
                default: Some(0),
            },
            typical: "0; 3 to 5 for channels with deep spectral nulls",
            mutability: Mutability::Build,
            description: "Number of feedback weights of the decision feedback equalizer (0 to \
                63, 0 = linear); they act in decision mode only.",
        },
        ParamDesc {
            name: "constellation",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: CONSTELLATIONS,
                default: 1,
            },
            typical: "qpsk; the constellation of the transmitter",
            mutability: Mutability::Build,
            description: "The constellation of the decisions: bpsk, qpsk or 8psk (the points \
                of psk_mod).",
        },
        ParamDesc {
            name: "mu_cma",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: MU.0,
                max: MU.1,
                default: Some(DEFAULT_MU_CMA),
            },
            typical: "0.05",
            mutability: Mutability::Runtime,
            description: "Normalized step size of the CMA update (used as f32); a change \
                applies from the next symbol.",
        },
        ParamDesc {
            name: "mu_dd",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: MU.0,
                max: MU.1,
                default: Some(DEFAULT_MU_DD),
            },
            typical: "0.02",
            mutability: Mutability::Runtime,
            description: "Normalized step size of the decision-directed update (used as f32); a \
                change applies from the next symbol.",
        },
        ParamDesc {
            name: "cma_symbols",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: u32::MAX as i64,
                default: Some(DEFAULT_CMA_SYMBOLS as i64),
            },
            typical: "1000; 0 for decision mode only, 4294967295 for CMA only",
            mutability: Mutability::Build,
            description: "Symbols in CMA mode before the switch to decision mode (0: decision \
                mode from the start, 4294967295: never).",
        },
        ParamDesc {
            name: "phase_bw",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0,
                max: PHASE_BW.1,
                default: Some(DEFAULT_PHASE_BW),
            },
            typical: "0.02; 0 without a carrier loop after it or for differential detection",
            mutability: Mutability::Build,
            description: "Loop bandwidth of the decision-directed phase tracker normalized to \
                the symbol rate: 0 (off) or 0.0001 to 0.2 (damping 0.7071).",
        },
    ],
    required: &[],
};

/// The constellation of a choice index (0 `bpsk`, 1 `qpsk`, 2 `8psk`).
const fn constellation_of(i: u16) -> EqConstellation {
    match i {
        0 => EqConstellation::Bpsk,
        2 => EqConstellation::Psk8,
        _ => EqConstellation::Qpsk,
    }
}

/// The choice index of a constellation.
fn constellation_index(c: EqConstellation) -> u16 {
    match c {
        EqConstellation::Bpsk => 0,
        EqConstellation::Psk8 => 2,
        _ => 1,
    }
}

/// The constellation of the PSK order (2, 4, 8) that `psk_demod` gives its equalizer.
pub(crate) fn constellation_of_order(order: u32) -> EqConstellation {
    match order {
        2 => EqConstellation::Bpsk,
        8 => EqConstellation::Psk8,
        _ => EqConstellation::Qpsk,
    }
}

/// An odd number of taps from `min` to 63, otherwise `OutOfRange` (outside) or
/// `InvalidParameter` (even).
pub(crate) fn check_taps(name: &'static str, taps: usize, min: usize) -> Result<()> {
    check_count(name, taps, min, EQ_TAPS.1)?;
    if taps.is_multiple_of(2) {
        return Err(Error::InvalidParameter {
            name,
            reason: "the number of equalizer taps must be odd",
        });
    }
    Ok(())
}

/// `phase_bw`: 0 or 1e-4 to 0.2; `OutOfRange` outside 0 to 0.2, `InvalidParameter` between 0
/// and 1e-4.
fn check_phase_bw(bw: f64) -> Result<()> {
    if !(0.0..=PHASE_BW.1).contains(&bw) {
        return Err(Error::OutOfRange {
            name: "phase_bw",
            min: 0.0,
            max: PHASE_BW.1,
            got: bw,
        });
    }
    if bw != 0.0 && bw < PHASE_BW.0 {
        return Err(Error::InvalidParameter {
            name: "phase_bw",
            reason: "the phase tracker bandwidth must be 0 (off) or at least 0.0001",
        });
    }
    Ok(())
}

fn check_mu(name: &'static str, mu: f64) -> Result<()> {
    if mu >= MU.0 && mu <= MU.1 {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name,
            min: MU.0,
            max: MU.1,
            got: mu,
        })
    }
}

/// Configuration of [`EqualizeBlock`].
///
/// ```
/// use sdrk_blocks::{EqConstellation, EqualizeConfig};
/// let cfg = EqualizeConfig::default();
/// assert_eq!((cfg.taps, cfg.feedback_taps, cfg.constellation), (11, 0, EqConstellation::Qpsk));
/// assert_eq!((cfg.mu_cma, cfg.mu_dd, cfg.cma_symbols, cfg.phase_bw), (0.05, 0.02, 1000, 0.02));
/// assert!(cfg.validate().is_ok());
/// assert!(EqualizeConfig { taps: 12, ..cfg }.validate().is_err());
/// assert!(EqualizeConfig { mu_dd: 1.5, ..cfg }.validate().is_err());
/// assert!(EqualizeConfig { feedback_taps: 64, ..cfg }.validate().is_err());
/// assert!(EqualizeConfig { phase_bw: 0.00005, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqualizeConfig {
    /// Number of forward weights (odd, 3 to 63).
    pub taps: usize,
    /// Number of feedback weights (0 to 63; 0 = linear equalizer).
    pub feedback_taps: usize,
    /// The constellation of the decisions.
    pub constellation: EqConstellation,
    /// Normalized step size of the CMA update (1e-5 to 1); the kernel uses it as `f32`.
    pub mu_cma: f64,
    /// Normalized step size of the decision-directed update (1e-5 to 1); the kernel uses it
    /// as `f32`.
    pub mu_dd: f64,
    /// Symbols in CMA mode before decision mode (0: decision mode at once, `u32::MAX`:
    /// never).
    pub cma_symbols: u32,
    /// Bandwidth of the phase tracker (0 = off, else 1e-4 to 0.2).
    pub phase_bw: f64,
}

impl Default for EqualizeConfig {
    /// 11 taps, no feedback, QPSK, `mu_cma` 0.05, `mu_dd` 0.02, 1000 CMA symbols, `phase_bw`
    /// 0.02 (M43 §2 as amended, D828).
    fn default() -> Self {
        Self {
            taps: DEFAULT_TAPS,
            feedback_taps: 0,
            constellation: EqConstellation::Qpsk,
            mu_cma: DEFAULT_MU_CMA,
            mu_dd: DEFAULT_MU_DD,
            cma_symbols: DEFAULT_CMA_SYMBOLS,
            phase_bw: DEFAULT_PHASE_BW,
        }
    }
}

impl EqualizeConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range, `InvalidParameter` for an even number of
    /// taps.
    pub fn validate(&self) -> Result<()> {
        check_taps("taps", self.taps, EQ_TAPS.0)?;
        check_count("feedback_taps", self.feedback_taps, 0, EQ_FEEDBACK_MAX)?;
        check_mu("mu_cma", self.mu_cma)?;
        check_mu("mu_dd", self.mu_dd)?;
        check_phase_bw(self.phase_bw)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for an even
    /// number of taps and a `phase_bw` between 0 and 1e-4.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "taps" => {
                let t = value.as_usize()?;
                check_taps("taps", t, EQ_TAPS.0)?;
                self.taps = t;
            }
            "feedback_taps" => self.feedback_taps = value.as_usize()?,
            "constellation" => self.constellation = constellation_of(value.as_choice()?),
            "mu_cma" => self.mu_cma = value.as_f64()?,
            "mu_dd" => self.mu_dd = value.as_f64()?,
            "cma_symbols" => self.cma_symbols = u32::try_from(value.as_i64()?).unwrap_or(0),
            "phase_bw" => {
                let v = value.as_f64()?;
                check_phase_bw(v)?;
                self.phase_bw = v;
            }
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
            "taps" => Ok(ParamValue::I64(self.taps as i64)),
            "feedback_taps" => Ok(ParamValue::I64(self.feedback_taps as i64)),
            "constellation" => Ok(ParamValue::Choice(constellation_index(self.constellation))),
            "mu_cma" => Ok(ParamValue::F64(self.mu_cma)),
            "mu_dd" => Ok(ParamValue::F64(self.mu_dd)),
            "cma_symbols" => Ok(ParamValue::I64(i64::from(self.cma_symbols))),
            "phase_bw" => Ok(ParamValue::F64(self.phase_bw)),
            _ => unknown(),
        }
    }

    /// The length of the weights and of the history: `taps + feedback_taps`.
    fn len(&self) -> usize {
        self.taps + self.feedback_taps
    }

    /// Exact bytes of storage: `taps + feedback_taps` weights and as many history values of
    /// `C32`.
    pub(crate) fn plan(&self, plan: &mut Plan) -> Result<()> {
        plan.add::<C32>(self.len())?;
        plan.add::<C32>(self.len())
    }

    /// The kernel parameters.
    fn params(&self) -> EqParams {
        EqParams {
            constellation: self.constellation,
            feedback_taps: self.feedback_taps,
            mu_cma: self.mu_cma as f32,
            mu_dd: self.mu_dd as f32,
            cma_symbols: self.cma_symbols,
            phase_bw: self.phase_bw,
        }
    }

    /// The kernel state with the weights, then the history, from `storage`.
    pub(crate) fn state<S: Storage>(
        &self,
        storage: &mut S,
    ) -> Result<EqualizeState<S::Buf<C32>, S::Buf<C32>>> {
        self.validate()?;
        let weights = storage.alloc::<C32>(self.len())?;
        let history = storage.alloc::<C32>(self.len())?;
        Ok(EqualizeState::new(weights, history, self.params())?)
    }
}

/// Adaptive equalizer for `C32` symbols at one sample per symbol: the kernel `equalize`
/// through the facade. Timing: ratio 1, delay `(taps - 1) / 2` symbols. Storage: the
/// `taps + feedback_taps` weights, then the joint history of as many values (`C32`).
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, EqConstellation, EqualizeBlock,
///     EqualizeConfig};
/// let cfg = EqualizeConfig { constellation: EqConstellation::Bpsk, ..EqualizeConfig::default() };
/// let rate = Rate::hz(1000)?;
/// let bytes = EqualizeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
/// assert_eq!(bytes, 2 * 11 * 8);
/// let mut mem = AlignedBytes([0u8; 176]);
/// let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
/// let mut eq = EqualizeBlock::new(&cfg, rate, &mut arena)?;
/// // BPSK through the echo 1 + 0.3 z^-1: the equalizer learns to remove it.
/// let sym: Vec<f32> = (0u32..8000)
///     .map(|k| if k.wrapping_mul(2_654_435_761) >> 31 == 0 { 1.0 } else { -1.0 })
///     .collect();
/// let input: Vec<C32> = (0..sym.len())
///     .map(|k| C32::new(sym[k] + if k > 0 { 0.3 * sym[k - 1] } else { 0.0 }, 0.0))
///     .collect();
/// let mut out = vec![C32::default(); input.len()];
/// let p = eq.process(&input, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (8000, 8000));
/// assert!(eq.mse() < 0.01);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct EqualizeBlock<S: Storage> {
    config: EqualizeConfig,
    state: EqualizeState<S::Buf<C32>, S::Buf<C32>>,
}

impl<S: Storage> core::fmt::Debug for EqualizeBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EqualizeBlock")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> EqualizeBlock<S> {
    /// Creates the block; the weights and the history come from `storage`, in this order.
    ///
    /// # Errors
    /// As [`EqualizeConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &EqualizeConfig, _rate_in: Rate, storage: &mut S) -> Result<Self> {
        Ok(Self {
            config: *config,
            state: config.state(storage)?,
        })
    }

    /// Exact bytes of storage the block needs: `2 taps` values of `C32`.
    ///
    /// # Errors
    /// As [`EqualizeConfig::validate`].
    pub fn storage_bytes(config: &EqualizeConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        let mut plan = Plan::default();
        config.plan(&mut plan)?;
        Ok(plan.bytes())
    }

    /// Output rate for the configuration (equal to the input rate).
    ///
    /// # Errors
    /// As [`EqualizeConfig::validate`].
    pub fn rate_out(config: &EqualizeConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate_in)
    }

    /// The adaptation mode.
    pub fn mode(&self) -> EqMode {
        self.state.mode()
    }

    /// The smoothed decision error power (linear; see the kernel's `mse`).
    pub fn mse(&self) -> f32 {
        self.state.mse()
    }

    /// The weights: `taps` forward weights (`w[i]` multiplies the symbol `i` symbols back),
    /// then the `feedback_taps` feedback weights.
    pub fn weights(&self) -> &[C32] {
        self.state.weights()
    }

    /// The kernel state (for inspection).
    pub fn state(&self) -> &EqualizeState<S::Buf<C32>, S::Buf<C32>> {
        &self.state
    }
}

impl<S: Storage> Block for EqualizeBlock<S> {
    type In = C32;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> Result<Progress> {
        let n = input.len().min(output.len());
        <Sdrk as Equalize>::equalize(
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
            delay: self.state.group_delay(),
            ..unit_timing()
        }
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match runtime_param(&DESCRIPTION, name, &value)? {
            "mu_cma" => {
                let v = value.as_f64()?;
                self.state.set_mu_cma(v as f32)?;
                self.config.mu_cma = v;
                Ok(())
            }
            "mu_dd" => {
                let v = value.as_f64()?;
                self.state.set_mu_dd(v as f32)?;
                self.config.mu_dd = v;
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
