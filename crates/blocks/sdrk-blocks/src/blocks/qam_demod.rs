//! `qam_demod`: 16-QAM and 64-QAM demodulator (matched filter, symbol timing, frequency
//! acquisition, equalizer, decision-directed carrier recovery, soft bits), a composite of the
//! building blocks `fir`, `symbol_sync`, `qam_acquire`, `equalize`, `costas` and `symbol_demap`
//! with the lock state machine `LockWatch` (M44.1 §5).

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::costas::CostasConstellation;
use sdrk_core::kernels::equalize::{EqConstellation, EqualizeState};
use sdrk_core::kernels::filter::{Fir, FirState};

use super::costas::{
    CostasBlock, CostasConfig, DEFAULT_DAMPING, DEFAULT_MAX_OFFSET, MAX_LOOP_BW, MIN_LOOP_BW,
    check_f64,
};
use super::equalize::{
    DEFAULT_CMA_SYMBOLS, DEFAULT_MU_CMA, DEFAULT_TAPS, EQ_TAPS, EqualizeBlock, EqualizeConfig,
};
use super::psk_demod::{
    MIN_SYMBOL_RATE_HZ, SPS_TOLERANCE, SYNC_MAX_DEVIATION, check_equalizer_taps, split_work,
    work_len,
};
use super::qam_acquire::{AcquireWindow, QamAcquireBlock, QamAcquireConfig};
use super::qam_mod::{ORDER_PARAM, order_of_value, qam_of_order};
use super::symbol_demap::{SymbolDemapBlock, SymbolDemapConfig, SymbolDemapConstellation};
use super::symbol_sync::SymbolSyncBlock;
use super::{HZ_LIMIT, check_count, head, head_mut, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MAX_SPS, RRC_MIN_ROLLOFF, RRC_MIN_SPAN, Rrc};
use crate::error::{Error, Result};
use crate::lock_watch::{LockAction, LockCheck, LockPolicy, LockState, LockTransition, LockWatch};
use crate::storage::{Plan, Storage};
use crate::{SymbolSyncConfig, Ted};

/// The state of the equalizer of `qam_demod` on the storage `S`.
type EqState<S> = EqualizeState<<S as Storage>::Buf<C32>, <S as Storage>::Buf<C32>>;

/// Input samples per piece the block processes at a time (the size of its work buffers).
pub const QAM_DEMOD_CHUNK: usize = 256;

/// The default loop bandwidth of the carrier loop of `qam_demod` (M44 §4; 0.0025 since the
/// maintainer's decision D1618: 0.005 slipped the carrier phase at 16-QAM Es/N0 10 dB and 64-QAM
/// 16 dB).
const DEFAULT_CARRIER_BW: f64 = 0.0025;

/// The step size of the decision-directed mode of the equalizer of `qam_demod` (D1618; the
/// `equalize` default is 0.02).
const QAM_MU_DD: f64 = 0.005;

/// The bandwidth of the equalizer's phase tracker of `qam_demod` from the start, a `reset` or a
/// restart until the first check after an estimate (the pull-in, D1622): the former tracker of
/// D829 at the former `carrier_bw` 0.005; it takes the error of the estimate (up to 5.6e-5
/// cycles per symbol, D1406) out of the equalizer's input.
pub const QAM_TRACKER_PULL_BW: f64 = 0.005;

/// The bandwidth of the equalizer's phase tracker of `qam_demod` after the pull-in (from the first
/// check after an estimate on, D1622): narrow, so that `costas` does the phase tracking (D1618),
/// but not 0: the decision-directed equalizer slips on a frequency error of its input of 2e-6
/// cycles per symbol already, which no estimate reaches, and the carrier loop after it cannot
/// take that error out of its input. Measured after the same pull-in (D1626): 0 (the frequency
/// the tracker pulled in, then held) fails the restart tests and the flow test 7 (442 of 1000
/// packets on seed 606); 0.0005 and 0.001 pass every test but lose the signal more often than
/// 0.0015 on the signals of the comparison of D1618 (alignment changes per 1e5 symbols over
/// three sets 1513.65 and 1365.34 against 1146.68).
pub const QAM_TRACKER_HOLD_BW: f64 = 0.0015;

/// The default loop bandwidth of the symbol timing of `qam_demod` (half that of `psk_demod`:
/// the self-noise of the Gardner detector on 64-QAM costs 0.9 dB at 0.01, D1215).
const DEFAULT_SYNC_BW: f64 = 0.005;

/// The reason of the rate error (a hint to resample).
const RATE_REASON: &str = "the sample rate must be within 1 % of an integer multiple (2 to \
    1024) of the symbol rate and at least twice it; put a resample before qam_demod";

/// The `symbol_rate_hz` parameter of `qam_demod` and `qam_frame_decode`.
pub(crate) const SYMBOL_RATE_PARAM: ParamDesc = ParamDesc {
    name: "symbol_rate_hz",
    unit: Unit::Hz,
    kind: ParamKind::F64 {
        min: MIN_SYMBOL_RATE_HZ,
        max: HZ_LIMIT,
        default: None,
    },
    typical: "25000 for 100 kbit/s 16-QAM",
    mutability: Mutability::Build,
    description: "Symbol rate in Hz; the sample rate must be within 1 % of an integer \
            multiple (2 to 1024) of it and at least twice it. Required.",
};

/// The `rolloff` parameter of `qam_demod` and `qam_frame_decode`.
pub(crate) const DEMOD_ROLLOFF_PARAM: ParamDesc = ParamDesc {
    name: "rolloff",
    unit: Unit::None,
    kind: ParamKind::F64 {
        min: RRC_MIN_ROLLOFF,
        max: 1.0,
        default: Some(0.35),
    },
    typical: "0.35 (the roll-off of the transmitter)",
    mutability: Mutability::Build,
    description: "Roll-off of the matched root-raised-cosine filter.",
};

/// The `span` parameter of `qam_demod` and `qam_frame_decode`.
pub(crate) const DEMOD_SPAN_PARAM: ParamDesc = ParamDesc {
    name: "span",
    unit: Unit::None,
    kind: ParamKind::I64 {
        min: RRC_MIN_SPAN as i64,
        max: RRC_MAX_SPAN as i64,
        default: Some(10),
    },
    typical: "10",
    mutability: Mutability::Build,
    description: "Length of the matched filter in symbols.",
};

/// The `equalizer_taps` parameter of `qam_demod` and `qam_frame_decode`.
pub(crate) const EQUALIZER_TAPS_PARAM: ParamDesc = ParamDesc {
    name: "equalizer_taps",
    unit: Unit::None,
    kind: ParamKind::I64 {
        min: 0,
        max: EQ_TAPS.1 as i64,
        default: Some(DEFAULT_TAPS as i64),
    },
    typical: "11; 15 against stronger multipath",
    mutability: Mutability::Build,
    description: "Forward taps of the adaptive equalizer between the timing loop and \
            the carrier loop: 0 (off) or odd from 3 to 63.",
};

/// The `sync_bw` parameter of `qam_demod` and `qam_frame_decode`.
pub(crate) const SYNC_BW_PARAM: ParamDesc = ParamDesc {
    name: "sync_bw",
    unit: Unit::None,
    kind: ParamKind::F64 {
        min: 0.0001,
        max: 0.1,
        default: Some(DEFAULT_SYNC_BW),
    },
    typical: "0.005; up to 0.01 for a faster acquisition",
    mutability: Mutability::Build,
    description: "Loop bandwidth of the symbol timing, normalized to the symbol rate.",
};

/// The `carrier_bw` parameter of `qam_demod` and `qam_frame_decode`.
pub(crate) const CARRIER_BW_PARAM: ParamDesc = ParamDesc {
    name: "carrier_bw",
    unit: Unit::None,
    kind: ParamKind::F64 {
        min: MIN_LOOP_BW,
        max: MAX_LOOP_BW,
        default: Some(DEFAULT_CARRIER_BW),
    },
    typical: "0.0025; wider for phase noise",
    mutability: Mutability::Build,
    description: "Loop bandwidth of the carrier loop, normalized to the symbol rate.",
};

/// Description of the `qam_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "qam_demod",
    summary: "Demodulates 16-QAM or 64-QAM to soft bits: matched filter, symbol timing, \
        equalizer and carrier recovery.",
    details: "The input is the complex baseband at the nominal level of unit average power \
        (an agc before it) with fs / symbol_rate_hz within 1 % of an integer and at least 2 \
        (otherwise the rate is unsupported: resample first). A matched root-raised-cosine \
        filter (rolloff, span symbols, normalized so that the pulse of qam_mod gives the \
        points at the symbol centres) is followed by symbol_sync (Gardner, loop bandwidth \
        sync_bw, default 0.005, damping 0.7071, max_deviation 0.005, ted_gain 1), a frequency \
        acquisition (the carrier offset from the spectrum of the fourth powers of 512 symbols \
        from symbol 256 on, range +-0.125 cycles per symbol, repeated on the next 512 symbols \
        until its peak stands clear of the noise; the offset then goes to the phase tracker of \
        the equalizer, or to the carrier loop without the equalizer; every 4096 \
        symbols after that the lock of the carrier loop is checked, and after a check without \
        lock a new window of the fourth powers that shows a clear carrier more than 0.0002 \
        cycles per symbol away from the loops starts the equalizer again and the acquisition \
        searches again; so does an energy of the equalizer weights (times the power of its input over the \
        pull-in, the energy at unit input power) below 0.75 for 16-QAM or 0.85 for 64-QAM at two \
        checks in a row, and a clear window after six or more windows \
        without a clear peak, a signal back after a gap, starts the equalizer again with the \
        new estimate), the adaptive equalizer \
        equalize (constellation qam16 or qam64 of the order, equalizer_taps taps, no feedback \
        taps, mu_cma 0.05, mu_dd 0.005, 1000 CMA symbols, then decision-directed; its phase \
        tracker at 0.005 until the first check after the estimate, then at 0.0015, a narrow \
        frequency tracker; at the switch to the decision mode the tracker turns by the phase \
        of the fourth power of the equalizer's output over its last 256 blind symbols after the \
        estimate; equalizer_taps 0 leaves it out), the decision-directed carrier \
        loop costas, which tracks the phase (constellation qam16 or qam64, carrier_bw, \
        damping 0.7071, max_offset 0.05 cycles per symbol) and the soft bits of symbol_demap \
        qam16 or \
        qam64 (scale 1): above 0 is a 1. This is the order of psk_demod (equalizer before the \
        carrier loop). The 90 degree ambiguity of the square constellation is not resolved: use \
        frame_decode with rotations four and bits_per_symbol 4 or 6. The output rate is the \
        symbol rate times the bits per symbol and follows the clock of the signal (variable \
        ratio); the equalizer adds (equalizer_taps - 1) / 2 symbols of delay.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Complex baseband, an integer number (2 or more) of samples per symbol, \
            unit average power.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "Soft bits, bits per symbol samples per symbol; above 0 is a 1.",
    }],
    params: &[
        ORDER_PARAM,
        SYMBOL_RATE_PARAM,
        DEMOD_ROLLOFF_PARAM,
        DEMOD_SPAN_PARAM,
        EQUALIZER_TAPS_PARAM,
        SYNC_BW_PARAM,
        CARRIER_BW_PARAM,
    ],
    required: &["symbol_rate_hz"],
};

/// Configuration of [`QamDemodBlock`].
///
/// ```
/// use sdrk_blocks::QamDemodConfig;
/// let cfg = QamDemodConfig { order: 16, symbol_rate_hz: 25_000.0, ..QamDemodConfig::default() };
/// assert_eq!(cfg.samples_per_symbol(sdrk::Rate::hz(200_000)?)?, 8);
/// assert_eq!((cfg.equalizer_taps, cfg.carrier_bw), (11, 0.0025));
/// // 7.5 samples per symbol is not within 1 % of an integer
/// assert!(cfg.validate(sdrk::Rate::hz(187_500)?).is_err());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QamDemodConfig {
    /// Order: 16 (16-QAM) or 64 (64-QAM).
    pub order: u32,
    /// Symbol rate in Hz (required).
    pub symbol_rate_hz: f64,
    /// Roll-off of the matched filter (0.05 to 1).
    pub rolloff: f64,
    /// Length of the matched filter in symbols (4 to 32).
    pub span: usize,
    /// Forward taps of the equalizer: 0 (off) or odd from 3 to 63.
    pub equalizer_taps: usize,
    /// Loop bandwidth of the symbol timing (0.0001 to 0.1).
    pub sync_bw: f64,
    /// Loop bandwidth of the carrier loop (0.0001 to 0.2).
    pub carrier_bw: f64,
}

impl Default for QamDemodConfig {
    /// Order 16, `symbol_rate_hz` 1200 (a placeholder: the pipeline requires the parameter),
    /// `rolloff` 0.35, `span` 10, `equalizer_taps` 11, `sync_bw` 0.005, `carrier_bw` 0.0025.
    fn default() -> Self {
        Self {
            order: 16,
            symbol_rate_hz: 1200.0,
            rolloff: 0.35,
            span: 10,
            equalizer_taps: DEFAULT_TAPS,
            sync_bw: DEFAULT_SYNC_BW,
            carrier_bw: DEFAULT_CARRIER_BW,
        }
    }
}

impl QamDemodConfig {
    /// Bits per symbol: 4 for order 16, 6 for order 64 (4 for an invalid order).
    pub fn bits_per_symbol(&self) -> usize {
        qam_of_order(self.order).map_or(4, Qam::bits_per_symbol)
    }

    /// The constellation of the order (16-QAM for an invalid order).
    fn qam(&self) -> Qam {
        qam_of_order(self.order).unwrap_or(Qam::Qam16)
    }

    /// The configuration of the timing loop.
    fn sync(&self) -> SymbolSyncConfig {
        SymbolSyncConfig {
            symbol_rate_hz: self.symbol_rate_hz,
            ted: Ted::Gardner,
            loop_bw: self.sync_bw,
            damping: DEFAULT_DAMPING,
            max_deviation: SYNC_MAX_DEVIATION,
            ted_gain: 1.0,
        }
    }

    /// The configuration of the equalizer (`None` for `equalizer_taps` 0): the forward taps,
    /// no feedback, the QAM constellation of the order, `mu_cma` and the CMA symbols of the
    /// `equalize` defaults, `mu_dd` 0.005 (D1618) and the phase tracker at
    /// [`QAM_TRACKER_PULL_BW`] (the block narrows it to [`QAM_TRACKER_HOLD_BW`] at the first check
    /// after an estimate, D1622).
    pub fn equalizer(&self) -> Option<EqualizeConfig> {
        (self.equalizer_taps != 0).then(|| EqualizeConfig {
            taps: self.equalizer_taps,
            feedback_taps: 0,
            constellation: match self.qam() {
                Qam::Qam64 => EqConstellation::Qam64,
                _ => EqConstellation::Qam16,
            },
            mu_cma: DEFAULT_MU_CMA,
            mu_dd: QAM_MU_DD,
            cma_symbols: DEFAULT_CMA_SYMBOLS,
            phase_bw: QAM_TRACKER_PULL_BW,
        })
    }

    /// The ranges, and the integer samples per symbol of the matched filter.
    fn derive(&self, rate_in: Rate) -> Result<usize> {
        qam_of_order(self.order)?;
        check_equalizer_taps(self.equalizer_taps)?;
        check_f64(
            "symbol_rate_hz",
            self.symbol_rate_hz,
            MIN_SYMBOL_RATE_HZ,
            HZ_LIMIT,
        )?;
        check_f64("rolloff", self.rolloff, RRC_MIN_ROLLOFF, 1.0)?;
        check_count("span", self.span, RRC_MIN_SPAN, RRC_MAX_SPAN)?;
        check_f64("sync_bw", self.sync_bw, 0.0001, 0.1)?;
        check_f64("carrier_bw", self.carrier_bw, MIN_LOOP_BW, MAX_LOOP_BW)?;
        let sps = rate_in.as_f64() / self.symbol_rate_hz;
        let n = libm::floor(sps + 0.5);
        let near = libm::fabs(sps - n) <= SPS_TOLERANCE * n;
        if !(near && n >= 2.0 && n <= RRC_MAX_SPS as f64) {
            return Err(Error::InvalidParameter {
                name: "symbol_rate_hz",
                reason: RATE_REASON,
            });
        }
        // The exact checks of the timing loop, reported with the hint to resample (as in
        // `psk_demod`).
        match self.sync().validate(rate_in) {
            Err(Error::OutOfRange {
                name: "symbol_rate_hz",
                ..
            }) => Err(Error::InvalidParameter {
                name: "symbol_rate_hz",
                reason: RATE_REASON,
            }),
            other => other,
        }?;
        Ok(n as usize)
    }

    /// The integer samples per symbol of the matched filter: `fs / symbol_rate_hz` rounded.
    ///
    /// # Errors
    /// As [`QamDemodConfig::validate`].
    pub fn samples_per_symbol(&self, rate_in: Rate) -> Result<usize> {
        self.derive(rate_in)
    }

    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// `OutOfRange` for a value outside its range; `InvalidParameter` for the orders 17 to 63,
    /// for an `equalizer_taps` of 1 or an even count and for a symbol rate whose samples per
    /// symbol are not within 1 % of an integer from 2 to 1024 or are below 2 (the pipeline
    /// reports it as `rate_unsupported` with the hint to resample).
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.derive(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for the orders
    /// 17 to 63 and for an `equalizer_taps` of 1 or an even count.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "order" => self.order = order_of_value(&value)?,
            "symbol_rate_hz" => self.symbol_rate_hz = value.as_f64()?,
            "rolloff" => self.rolloff = value.as_f64()?,
            "span" => self.span = value.as_usize()?,
            "equalizer_taps" => {
                let t = value.as_usize()?;
                check_equalizer_taps(t)?;
                self.equalizer_taps = t;
            }
            "sync_bw" => self.sync_bw = value.as_f64()?,
            "carrier_bw" => self.carrier_bw = value.as_f64()?,
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
            "symbol_rate_hz" => Ok(ParamValue::F64(self.symbol_rate_hz)),
            "rolloff" => Ok(ParamValue::F64(self.rolloff)),
            "span" => Ok(ParamValue::I64(self.span as i64)),
            "equalizer_taps" => Ok(ParamValue::I64(self.equalizer_taps as i64)),
            "sync_bw" => Ok(ParamValue::F64(self.sync_bw)),
            "carrier_bw" => Ok(ParamValue::F64(self.carrier_bw)),
            _ => unknown(),
        }
    }
}

/// Symbols after a used estimate, and then between two checks, at which the controller of
/// `qam_demod` checks the carrier loop (D1421, decided by frequency since D1618): a check at which
/// the loop reports lock keeps the estimate; after a check without lock a new window of the
/// fourth powers (the verify window) is taken, and when it shows a clear carrier away from the
/// frequency of the loops the estimate counts as wrong or out of date (a false alarm on noise, a
/// narrowband interferer before the signal, a new transmission on another frequency, or loops
/// that did not converge on a correct estimate and wander about the carrier, D1623): the
/// equalizer starts again from its start weights and blind symbols, the frequencies go to 0, and
/// the acquisition searches again from the start of its window. A verify window without a clear
/// peak (noise, or a weak signal whose lock level noise also reaches) keeps the estimate.
pub const QAM_ACQUISITION_CHECK: usize = 4096;
/// The largest distance in cycles per symbol between a clear estimate of the verify window and
/// the frequency of the loops for which the estimate is kept; beyond it the loops sit on a wrong
/// frequency or have not converged on the signal (D1605, D1618; 2e-4 since D1623: the narrow
/// loops of D1622 that hold a signal stay within 1.1e-4 of the carrier, the estimate within
/// 7e-5 of it, while loops that did not converge wander up to 3e-4 and more).
pub const QAM_ACQUISITION_VERIFY_TOLERANCE: f64 = 2e-4;
/// The blind (CMA) symbols of the equalizer, at most, before its switch to the decision mode over
/// which the hand-over measures the phase of its output (I44-17, [`QamDemodBlock`]).
pub const QAM_HANDOVER_SYMBOLS: usize = 256;

/// The weight watch of `qam_demod` (M44.1 §6, condition 1) for 16-QAM: a check in `Track` at
/// which the energy of the equalizer's weights (`sum |w|^2`) is below this value counts as low,
/// and [`QAM_WATCH_CHECKS`] low checks in a row start the receiver again (the equalizer from its
/// start weights, the loops at frequency 0, a new search; `LockTransition::Shrink`). The energy of
/// correct weights falls with the signal-to-noise ratio (as the Wiener weights, about
/// `(snr / (1 + snr))^2`): measured at the checks of the 120 receivers with the equalizer of the
/// comparison of D1702, at least 0.808 in every 16-QAM receiver that keeps one alignment (11 dB)
/// and 0.757 at 10 dB, while the false states of I44-17 shrink to 0.50 to 0.71 (D1567).
pub const QAM_WATCH_WEIGHT_ENERGY_16: f32 = 0.75;
/// The weight watch of `qam_demod` for 64-QAM (as [`QAM_WATCH_WEIGHT_ENERGY_16`]): every 64-QAM
/// receiver of the comparison that keeps one alignment stays at 0.874 or above (16 dB), and the
/// false state of D1700 falls below 0.85 within about 6000 symbols (D1567).
pub const QAM_WATCH_WEIGHT_ENERGY_64: f32 = 0.85;
/// The low checks in a row of the weight watch of `qam_demod` (M44.1 §6, condition 1): two, so
/// that a single dip of the energy (multipath, a burst of noise) restarts nothing (D1567).
pub const QAM_WATCH_CHECKS: u8 = 2;
/// The signal back after a gap (M44.1 §6, condition 2): a clear verify window after at least this
/// many acquisition windows in a row without a clear peak starts the equalizer again and hands it
/// the new estimate (`LockTransition::Return`), a gap of the signal of about six checks (24576
/// symbols) or more. A hysteresis: the verify windows of weak signals alternate between clear and
/// not clear (D1619); over the 180 receivers of the comparison of D1702 at most 5 windows in a row
/// had no clear peak before a clear one, at most 2 in a receiver that keeps one alignment (D1568).
pub const QAM_WATCH_GAP_WINDOWS: u8 = 6;

/// The thresholds and time limits of the controller of `qam_demod` ([`QamDemodBlock::watch`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QamDemodLock;

impl LockPolicy for QamDemodLock {
    const CHECK: u16 = QAM_ACQUISITION_CHECK as u16;
    const TOLERANCE: f64 = QAM_ACQUISITION_VERIFY_TOLERANCE;
    const WEIGHT_CHECKS: u8 = QAM_WATCH_CHECKS;
    const GAP_WINDOWS: u8 = QAM_WATCH_GAP_WINDOWS;
}

// The check fits the counter of `LockWatch`, and the switch of the equalizer to the decision
// mode comes before the first check after an estimate (the hand-over runs in `PullIn`).
const _: () = assert!(QAM_ACQUISITION_CHECK <= u16::MAX as usize);
const _: () = assert!(DEFAULT_CMA_SYMBOLS as usize <= QAM_ACQUISITION_CHECK);

/// The hand-over of the equalizer from its blind (CMA) symbols to the decision mode (I44-17,
/// D1700): the action of the state `PullIn` at the switch.
///
/// The CMA is blind to the phase, and the decision-directed phase tracker that runs during it, at
/// [`QAM_TRACKER_PULL_BW`] on decisions of an output that has not converged yet, leaves the output
/// at the switch at any phase within the quarter turn. Started there 10 to 40 degrees off, the
/// decision mode of 64-QAM at Es/N0 18 and 19 dB pulled the tracker's frequency away from a correct
/// estimate (5.08e-4 to 6.3e-4 cycles per symbol against a carrier at 5.0e-4) into a false state in
/// which the output turns, the decisions are wrong, the weights shrink (energy 0.98 -> 0.50) and
/// `costas` behind it follows the turn, so that the sum of the frequencies that the verify window
/// checks stays within its tolerance. So at the switch the tracker turns by the phase of the fourth
/// power of the output, the feed-forward estimate of the carrier phase of square QAM modulo the
/// quarter turn (`E[s^4]` is real and negative: the mean of `z^4` is `E[s^4] e^(j 4 phi)` for an
/// output `z = s e^(j phi)`), over the last [`QAM_HANDOVER_SYMBOLS`] blind symbols after an
/// estimate (before it the tracker turns at frequency 0 and the phase of the output runs).
#[derive(Clone, Copy, Debug)]
struct HandOver {
    /// The sum of the fourth powers of the output.
    sum: C32,
    /// The blind symbols of the equalizer left before its switch to the decision mode; 0 when it
    /// is past (the steady state) or without the equalizer (at most `DEFAULT_CMA_SYMBOLS`).
    blind_left: u16,
    /// Whether an estimate went to the equalizer's tracker since its start (only then the phase of
    /// its output stands still and is measured).
    armed: bool,
}

impl HandOver {
    /// The equalizer (`with_eq`) starts (again) from its start weights and blind symbols: the
    /// hand-over waits for its blind symbols and an estimate.
    fn start(with_eq: bool) -> Self {
        Self {
            sum: C32::new(0.0, 0.0),
            blind_left: if with_eq {
                DEFAULT_CMA_SYMBOLS as u16
            } else {
                0
            },
            armed: false,
        }
    }

    /// An estimate went to the equalizer's tracker: the sum starts at 0.
    fn arm(&mut self) {
        self.armed = true;
        self.sum = C32::new(0.0, 0.0);
    }

    /// The hand-over after a run `z` of outputs of the equalizer `eq` (at most `blind_left`
    /// symbols, the runs end at the switch): when armed, the fourth powers `(z^2)^2` of those
    /// among the last [`QAM_HANDOVER_SYMBOLS`] blind symbols into the sum (in `f32`, in their
    /// order, finite ones only), and at the switch, when the sum is not 0, the tracker's phase
    /// grows by `round(atan2(-sum.im, -sum.re) / (8 pi) 2^32)` units of `2^-32` turn: the mean of
    /// `z^4` is `E[s^4] e^(j 4 phi)` with `E[s^4]` real and negative, so `arg(-sum) / 4` is the
    /// phase `phi` of the output in `(-1/8, 1/8]` turn, which the tracker turns back by its phase
    /// (out of line: only the runs of the blind symbols get here).
    #[inline(never)]
    #[cold]
    fn run<S: Storage>(&mut self, eq: &mut EqualizeBlock<S>, z: &[C32]) {
        let left = usize::from(self.blind_left);
        if self.armed {
            for (i, v) in z.iter().enumerate() {
                // The blind symbols left after this one.
                if left.saturating_sub(i + 1) < QAM_HANDOVER_SYMBOLS {
                    let a = C32::new(v.re * v.re - v.im * v.im, 2.0 * v.re * v.im);
                    let q = C32::new(a.re * a.re - a.im * a.im, 2.0 * a.re * a.im);
                    if q.re.is_finite() && q.im.is_finite() {
                        self.sum = C32::new(self.sum.re + q.re, self.sum.im + q.im);
                    }
                }
            }
        }
        // `z.len() <= left`.
        self.blind_left = left.saturating_sub(z.len()) as u16;
        if self.blind_left == 0 {
            let sum = self.sum;
            if self.armed && (sum.re != 0.0 || sum.im != 0.0) {
                let phi = libm::atan2(-f64::from(sum.im), -f64::from(sum.re))
                    / (4.0 * core::f64::consts::TAU);
                let step = libm::round(phi * 4_294_967_296.0) as i64 as u32;
                eq.set_phase(eq.state().phase().wrapping_add(step));
            }
            self.armed = false;
        }
    }
}

/// 16-QAM and 64-QAM demodulator: `C32` in, `f32` soft bits out (`bits_per_symbol` per symbol,
/// above 0 is a 1), a composite of building blocks with a controller (M44.1 §5).
///
/// The building blocks, per piece of at most [`QAM_DEMOD_CHUNK`] input samples: the kernel `fir`
/// (the matched filter, taps `rrc(rolloff, n, span) / sqrt(n)` as `f32` with `n` the rounded
/// samples per symbol), [`SymbolSyncBlock`] (Gardner at the exact `fs / symbol_rate_hz`),
/// [`QamAcquireBlock`] (`correct` false: it only estimates), [`EqualizeBlock`] (when
/// `equalizer_taps` is above 0), [`CostasBlock`] with the constellation `qam16` or `qam64`, and
/// [`SymbolDemapBlock`] `qam16` or `qam64` (scale 1). Every sample the block writes comes from
/// them; it only holds them, one work buffer and the controller.
///
/// The controller is the state machine [`LockWatch`] (the states `Search`, `PullIn`, `Track` and
/// `Verify`, [`state`](QamDemodBlock::state)) and the hand-over of the equalizer, between the
/// calls of the building blocks: the symbols of the timing loop of a piece run through the
/// acquisition, the equalizer and the carrier loop in runs that end where an acquisition window
/// completes, where a check falls (every [`QAM_ACQUISITION_CHECK`] symbols after an estimate) and
/// at the equalizer's switch to the decision mode; after each run the controller carries out the
/// named transition ([`LockTransition`]): in `Search` a clear window goes to the frequency
/// integrator of the equalizer's phase tracker (the carrier loop's frequency to 0; the tracker at
/// [`QAM_TRACKER_PULL_BW`] until the first check after the estimate, then at
/// [`QAM_TRACKER_HOLD_BW`], so that `costas` does the phase tracking, D1618, D1622), or, without
/// the equalizer, to the carrier loop (after windows without a clear peak the equalizer first
/// starts again from its start weights and blind symbols, D1619); in `PullIn` the tracker turns at
/// the switch by the fourth-power phase of the equalizer's output over the last
/// [`QAM_HANDOVER_SYMBOLS`] blind symbols after the estimate (I44-17, D1700); a check without lock
/// takes a verify window, and when it shows the loops on a wrong frequency or not converged, the
/// equalizer starts again from its start weights and blind symbols, the frequencies go to 0 and
/// the acquisition searches again (D1421, D1605, D1618, D1623). The recovery of M44.1 §6 adds two
/// conditions (D1566 to D1569): the weight watch (the energy of the equalizer's weights below
/// [`QAM_WATCH_WEIGHT_ENERGY_16`] or [`QAM_WATCH_WEIGHT_ENERGY_64`] at [`QAM_WATCH_CHECKS`] checks
/// in a row in `Track` starts the receiver again as a verify window away from the loops does) and
/// the signal back after a gap (a clear verify window within the tolerance after at least
/// [`QAM_WATCH_GAP_WINDOWS`] windows without a clear peak starts the equalizer again and hands it
/// the new estimate). With the recovery off ([`set_recovery`](QamDemodBlock::set_recovery)) the
/// output is bit for bit that of the block of M44 with the fix of I44-17 (M44.1 test 5). Soft bits that do not fit into the
/// output wait in a buffer and are written first by the next call (which then consumes no input).
/// Timing: variable ratio `symbol_rate bits / fs`, delay `(span + equalizer_taps - 1) n / 2 + 1`
/// input samples. Storage: the taps, the filter history, one work buffer of `512 + 128 bits` `C32`
/// (filter output, symbols, waiting soft bits as `f32`), for `equalizer_taps` `t > 0` the `t`
/// weights and the `t` history values of the equalizer, and the 1024 `C32` of the acquisition's
/// FFT and its 512 twiddles.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, LockState, QamDemodBlock, QamDemodConfig, QamModBlock,
///     QamModConfig};
/// let rate = Rate::hz(4000)?; // 1000 symbols per second, 4 samples per symbol
/// let tx = QamModConfig { order: 16, samples_per_symbol: 4, ..QamModConfig::default() };
/// let rx = QamDemodConfig { order: 16, symbol_rate_hz: 1000.0, ..QamDemodConfig::default() };
/// let mut mem = vec![0u8; 1 << 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut m = QamModBlock::new(&tx, Rate::hz(4000)?, &mut arena)?;
/// let mut d = QamDemodBlock::new(&rx, rate, &mut arena)?;
/// assert_eq!(d.state(), LockState::Search);
/// let bits: Vec<f32> = (0..16_000u32).map(|k| if (k * 7 + k / 5) % 3 == 0 { 1.0 } else { -1.0 }).collect();
/// let mut sig = vec![C32::default(); 16_000];
/// let p = m.process(&bits, &mut sig)?;
/// assert_eq!(p.produced, 16_000);
/// let mut soft = vec![0.0f32; 16_400];
/// let q = d.process(&sig, &mut soft)?;
/// assert_eq!(q.consumed, 16_000);
/// assert!((15_900..=16_000).contains(&q.produced));
/// assert_eq!(d.state(), LockState::PullIn);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QamDemodBlock<S: Storage> {
    /// The matched filter (the kernel `fir` with given taps).
    fir: FirState<C32, S::Buf<f32>, S::Buf<C32>>,
    sync: SymbolSyncBlock<C32, S>,
    /// The acquisition, the equalizer and the carrier loop with the controller.
    rx: Loops<S>,
    demap: SymbolDemapBlock<C32, S>,
    /// One work buffer `[a | b | pending]` (`split_work` of `psk_demod`): `a` the
    /// matched-filter output, then the equalized symbols; `b` the symbols of `symbol_sync`,
    /// then those of the carrier loop; `pending` (as `f32`) the soft bits that did not fit into
    /// the output, `pending[at..len]`.
    work: S::Buf<C32>,
    /// The roll-off of the matched filter (the only parameter no building block holds).
    rolloff: f64,
    at: u16,
    len: u16,
}

impl<S: Storage> core::fmt::Debug for QamDemodBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QamDemodBlock")
            .field("config", &self.config())
            .field("state", &self.rx.watch.state())
            .field("costas", self.rx.costas.state())
            .finish_non_exhaustive()
    }
}

impl<S: Storage> QamDemodBlock<S> {
    /// Designs the matched filter and creates the block of its building blocks; taps, history,
    /// the work buffer, the equalizer and the acquisition come from `storage`, in this order.
    ///
    /// # Errors
    /// As [`QamDemodConfig::validate`]; `Storage` when the storage is too small.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    pub fn new(config: &QamDemodConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let n = config.derive(rate_in)?;
        let qam = config.qam();
        let rrc = Rrc::new(config.rolloff, n, config.span)?;
        let taps_len = rrc.len();
        let gain = 1.0 / libm::sqrt(n as f64);
        let mut taps = storage.alloc::<f32>(taps_len)?;
        for (k, t) in taps.as_mut().iter_mut().enumerate() {
            *t = (rrc.tap(k) * gain) as f32;
        }
        let hist = storage.alloc::<C32>(taps_len - 1)?;
        let work = storage.alloc::<C32>(work_len(qam.bits_per_symbol()))?;
        let eq = match config.equalizer() {
            Some(e) => Some(EqualizeBlock::new(&e, rate_in, storage)?),
            None => None,
        };
        let acquire = QamAcquireBlock::new(&QamAcquireConfig { correct: false }, rate_in, storage)?;
        let sync = SymbolSyncBlock::<C32, S>::new(&config.sync(), rate_in, storage)?;
        let (costas_c, demap_c) = match qam {
            Qam::Qam64 => (CostasConstellation::Qam64, SymbolDemapConstellation::Qam64),
            _ => (CostasConstellation::Qam16, SymbolDemapConstellation::Qam16),
        };
        let costas = CostasBlock::new(
            &CostasConfig {
                constellation: costas_c,
                loop_bw: config.carrier_bw,
                damping: DEFAULT_DAMPING,
                max_offset: DEFAULT_MAX_OFFSET,
            },
            rate_in,
            storage,
        )?;
        let demap = SymbolDemapBlock::new(
            &SymbolDemapConfig {
                constellation: demap_c,
                scale: 1.0,
            },
            rate_in,
            storage,
        )?;
        let hand = HandOver::start(eq.is_some());
        Ok(Self {
            fir: FirState::new(taps, hist)?,
            sync,
            rx: Loops {
                acquire,
                eq,
                costas,
                watch: LockWatch::new(),
                hand,
                level: Level::default(),
                reference: 1.0,
                symbols: 0,
                restart_at: NO_RESTART,
            },
            demap,
            work,
            rolloff: config.rolloff,
            at: 0,
            len: 0,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`QamDemodConfig::validate`].
    pub fn storage_bytes(config: &QamDemodConfig, rate_in: Rate) -> Result<usize> {
        let mut plan = Plan::default();
        Self::plan(config, rate_in, &mut plan)?;
        Ok(plan.bytes())
    }

    /// Adds the buffers of the block to `plan`, in the order `new` takes them.
    pub(crate) fn plan(config: &QamDemodConfig, rate_in: Rate, plan: &mut Plan) -> Result<()> {
        let n = config.derive(rate_in)?;
        let taps = config.span * n + 1;
        plan.add::<f32>(taps)?;
        plan.add::<C32>(taps - 1)?;
        plan.add::<C32>(work_len(config.bits_per_symbol()))?;
        if let Some(e) = config.equalizer() {
            e.plan(plan)?;
        }
        QamAcquireBlock::<S>::plan(plan)
    }

    /// Output rate: the symbol rate (on the 1/1000 Hz grid) times the bits per symbol.
    ///
    /// # Errors
    /// As [`QamDemodConfig::validate`].
    pub fn rate_out(config: &QamDemodConfig, rate_in: Rate) -> Result<Rate> {
        config.validate(rate_in)?;
        let sym = SymbolSyncBlock::<C32, S>::rate_out(&config.sync(), rate_in)?;
        Ok(sym.scale(Ratio::new(config.bits_per_symbol() as u64, 1)?)?)
    }

    /// The configuration of the block, from its building blocks (the roll-off from the block).
    pub fn config(&self) -> QamDemodConfig {
        let sync = self.sync.config();
        let n = self.samples_per_symbol();
        QamDemodConfig {
            order: if self.bits() == 6 { 64 } else { 16 },
            symbol_rate_hz: sync.symbol_rate_hz,
            rolloff: self.rolloff,
            span: (self.num_taps() - 1) / n.max(1),
            equalizer_taps: self.rx.eq.as_ref().map_or(0, |e| e.state().taps()),
            sync_bw: sync.loop_bw,
            carrier_bw: self.rx.costas.state().loop_bw(),
        }
    }

    /// Bits per symbol (4 or 6).
    fn bits(&self) -> usize {
        self.demap.config().constellation.bits_per_symbol()
    }

    /// Number of taps of the matched filter (`span n + 1`).
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }

    /// The integer samples per symbol of the matched filter.
    pub fn samples_per_symbol(&self) -> usize {
        // `sps` within 1 % of the integer `n` (2 to 1024), as `QamDemodConfig::derive`.
        libm::floor(self.sync.state().sps() + 0.5) as usize
    }

    /// The state of the controller (M44.1 §5).
    pub fn state(&self) -> LockState {
        self.rx.watch.state()
    }

    /// Whether the carrier loop reports lock (the lock detector of `costas`: the decision error
    /// against the QAM grid, D1400).
    pub fn locked(&self) -> bool {
        self.rx.costas.locked()
    }

    /// The smoothed lock metric of the carrier loop (`CostasState::lock_level`: about 1 at
    /// lock, about 0 and below on noise).
    pub fn lock_level(&self) -> f32 {
        self.rx.costas.lock_level()
    }

    /// The estimated carrier offset in cycles per symbol: the frequency of the carrier loop plus
    /// that of the equalizer's phase tracker (which holds the acquired frequency).
    pub fn frequency(&self) -> f64 {
        loops_frequency(&self.rx.costas, self.rx.eq.as_ref())
    }

    /// The carrier offset in cycles per symbol last found by the frequency acquisition (the
    /// fourth-power estimate over windows of 512 symbols from symbol 256 after a start, a
    /// `reset` or a restart, the first whose spectral peak stands clear of the noise; kept over a
    /// restart until the next); `None` before the first.
    pub fn acquired_frequency(&self) -> Option<f64> {
        self.rx.watch.acquired()
    }

    /// How often the controller restarted the receiver since its start or `reset` (a verify
    /// window after a check without lock that found the loops on a wrong frequency, D1421,
    /// D1605, D1618; the weight watch and the signal back after a gap of M44.1 §6, D1569).
    pub fn restarts(&self) -> u32 {
        self.rx.watch.restarts()
    }

    /// The state of the equalizer (`None` for `equalizer_taps` 0).
    pub fn equalizer(&self) -> Option<&EqState<S>> {
        self.rx.eq.as_ref().map(EqualizeBlock::state)
    }

    /// The equalizer as a building block with its control interface (`None` for
    /// `equalizer_taps` 0), for tests and experiments that put the receiver into a state (M44.1
    /// test 7): the controller is not told of a call and reacts to it only through what it
    /// watches.
    pub fn equalizer_mut(&mut self) -> Option<&mut EqualizeBlock<S>> {
        self.rx.eq.as_mut()
    }

    /// The controller: the lock state machine with its counters (M44.1 §2.3, §6).
    pub fn watch(&self) -> &LockWatch<QamDemodLock> {
        &self.rx.watch
    }

    /// Turns the recovery of M44.1 §6 (the weight watch and the signal back after a gap) on (the
    /// default) or off. Off, the controller takes the transitions of §5 only and the soft bits are
    /// those of the block before the recovery, bit for bit (the baselines of M44.1 test 5); kept
    /// over `reset`.
    ///
    /// ```
    /// use sdrk::Rate;
    /// use sdrk_blocks::{ArenaStorage, QamDemodBlock, QamDemodConfig};
    /// let cfg = QamDemodConfig { symbol_rate_hz: 1000.0, ..QamDemodConfig::default() };
    /// let mut mem = vec![0u8; 1 << 16];
    /// let mut d = QamDemodBlock::new(&cfg, Rate::hz(4000)?, &mut ArenaStorage::new(&mut mem))?;
    /// assert!(d.recovery());
    /// d.set_recovery(false);
    /// assert!(!d.watch().recovery());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn set_recovery(&mut self, on: bool) {
        self.rx.watch.set_recovery(on);
    }

    /// Whether the recovery of M44.1 §6 is on.
    pub fn recovery(&self) -> bool {
        self.rx.watch.recovery()
    }

    /// Symbols of the loops (the timing loop's output) since the start or `reset`; the soft bits
    /// of symbol `k` are the outputs `k bits .. (k + 1) bits`.
    pub fn symbols(&self) -> u64 {
        self.rx.symbols
    }

    /// Schedules a restart of the whole receiver by a watch outside the controller at the symbol
    /// `at` of the loops ([`symbols`](QamDemodBlock::symbols)): the transition `Silence` of the
    /// controller (the action `RestartAll`, counted in [`restarts`](QamDemodBlock::restarts)),
    /// at that symbol however the input is cut. `None` cancels; a symbol already reached
    /// restarts at once. Kept until it happens, replaced by the next call; `reset` cancels it.
    /// The frame watch of `qam_frame_decode` (M44.1 §7) uses it.
    ///
    /// ```
    /// use sdrk::Rate;
    /// use sdrk_blocks::{ArenaStorage, LockState, QamDemodBlock, QamDemodConfig};
    /// let cfg = QamDemodConfig { symbol_rate_hz: 1000.0, ..QamDemodConfig::default() };
    /// let mut mem = vec![0u8; 1 << 16];
    /// let mut d = QamDemodBlock::new(&cfg, Rate::hz(4000)?, &mut ArenaStorage::new(&mut mem))?;
    /// d.restart_at(Some(0));
    /// assert_eq!((d.state(), d.restarts()), (LockState::Search, 1));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn restart_at(&mut self, at: Option<u64>) {
        match at {
            Some(s) if s <= self.rx.symbols => {
                self.rx.restart_at = NO_RESTART;
                let t = self.rx.watch.silence();
                self.rx.transition(t);
            }
            Some(s) => self.rx.restart_at = s,
            None => self.rx.restart_at = NO_RESTART,
        }
    }

    /// The current estimate of the samples per symbol of the timing loop.
    pub fn sps_estimate(&self) -> f64 {
        self.sync.sps_estimate()
    }

    /// Writes waiting soft bits to `output`; returns how many.
    fn flush(&mut self, output: &mut [f32]) -> Result<usize> {
        let (at, len) = (usize::from(self.at), usize::from(self.len));
        let (_, _, pending) = split_work(self.work.as_mut())?;
        let waiting = head(tail(pending, at)?, len - at)?;
        let k = waiting.len().min(output.len());
        head_mut(output, k)?.copy_from_slice(head(waiting, k)?);
        // `k <= len <= QAM_DEMOD_CHUNK bits`.
        self.at += k as u16;
        if self.at == self.len {
            self.at = 0;
            self.len = 0;
        }
        Ok(k)
    }

    /// One piece of input samples: the symbols after the carrier loop in `b[..ns]`; returns
    /// `ns`.
    fn piece_symbols(&mut self, piece: &[C32]) -> Result<usize> {
        let m = piece.len();
        let (a, b, _) = split_work(self.work.as_mut())?;
        <Sdrk as Fir<C32>>::fir(&Sdrk, &mut self.fir, piece, head_mut(a, m)?)?;
        let p = self.sync.process(head(a, m)?, b)?;
        if p.consumed != m {
            return Err(Error::Core(sdrk_core::Error::Internal(
                "qam_demod timing loop did not take the whole piece",
            )));
        }
        let ns = p.produced;
        // `b` -> the acquisition (in place, unchanged) -> `a` (the equalizer, or a copy without
        // it) -> `b` (the carrier loop), in runs that end where the controller acts.
        let mut at = 0;
        while at < ns {
            let end = ns.min(at + self.rx.remaining());
            self.rx.run(
                head_mut(tail_mut(b, at)?, end - at)?,
                head_mut(tail_mut(a, at)?, end - at)?,
            )?;
            at = end;
        }
        Ok(ns)
    }
}

/// The building blocks that the controller drives (the acquisition, the equalizer, the carrier
/// loop) with the controller itself (the lock state machine and the hand-over).
struct Loops<S: Storage> {
    acquire: QamAcquireBlock<S>,
    /// The equalizer before the carrier loop; `None` for `equalizer_taps` 0.
    eq: Option<EqualizeBlock<S>>,
    costas: CostasBlock<S>,
    watch: LockWatch<QamDemodLock>,
    hand: HandOver,
    /// The power of the symbols into the equalizer over the pull-in (from the estimate to the
    /// first check after it).
    level: Level,
    /// The power of the symbols into the equalizer over the last pull-in (1 before the first): the
    /// scale of the weight watch.
    reference: f32,
    /// Symbols of the loops since the start or `reset`.
    symbols: u64,
    /// The symbol at which a watch outside the controller restarts the receiver
    /// ([`NO_RESTART`]: none).
    restart_at: u64,
}

/// No restart scheduled by a watch outside the controller.
const NO_RESTART: u64 = u64::MAX;

/// The mean power of the symbols into the equalizer over the pull-in, the symbols from an
/// estimate to the first check after it (the signal that was acquired): the weight watch compares
/// the energy of the weights at unit input power, `sum |w|^2` times this power, so that its floor
/// holds at any level of the input (the AGC of the flow `qam16_receiver` holds the power at about
/// 1.9), while noise after the end of a signal, whose power is lower, does not count as a
/// shrinking of the weights (D1574).
#[derive(Clone, Copy, Debug, Default)]
struct Level {
    /// The sum of `|x|^2` of the finite symbols.
    sum: f64,
    /// Those symbols.
    count: u32,
}

impl Level {
    /// Adds the symbols of a run (finite ones only).
    #[inline]
    fn add(&mut self, run: &[C32]) {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for v in run {
            let p = v.re * v.re + v.im * v.im;
            if p.is_finite() {
                sum += p;
                count += 1;
            }
        }
        self.sum += f64::from(sum);
        self.count = self.count.saturating_add(count);
    }

    /// The mean power, 1 without a finite symbol.
    fn mean(&self) -> f32 {
        if self.count == 0 {
            1.0
        } else {
            (self.sum / f64::from(self.count)) as f32
        }
    }
}

impl<S: Storage> Loops<S> {
    /// Symbols until the controller acts next: the end of the acquisition window, the check, or
    /// the switch of the equalizer to the decision mode when that comes first (the hand-over).
    fn remaining(&self) -> usize {
        // `restart_at > symbols` (`restart_at` restarts at once otherwise).
        let to_restart =
            usize::try_from(self.restart_at.saturating_sub(self.symbols)).unwrap_or(usize::MAX);
        let r = self
            .acquire
            .remaining()
            .min(self.watch.remaining())
            .min(to_restart);
        match self.hand.blind_left {
            0 => r,
            b => r.min(usize::from(b)),
        }
    }

    /// A run of symbols `src` (the symbols of the timing loop, at most
    /// [`remaining`](Loops::remaining)): the acquisition in place (unchanged), the equalizer into
    /// `dst` (or a copy without it), the carrier loop back into `src`, then the controller.
    #[inline]
    fn run(&mut self, src: &mut [C32], dst: &mut [C32]) -> Result<()> {
        let window = self.acquire.process_in_place(src)?;
        match self.eq.as_mut() {
            Some(eq) => {
                if self.watch.state() == LockState::PullIn {
                    self.level.add(src);
                }
                eq.process(src, dst)?;
                if self.hand.blind_left != 0 {
                    self.hand.run(eq, dst);
                }
            }
            None => dst.copy_from_slice(src),
        }
        self.costas.process(dst, src)?;
        self.control(src.len(), window);
        Ok(())
    }

    /// The controller after a run of `n` symbols: the window or the check that ended the run, as
    /// the transition of the state machine (nothing between these points).
    #[inline]
    fn control(&mut self, n: usize, window: Option<AcquireWindow>) {
        self.symbols = self.symbols.saturating_add(n as u64);
        let t = if let Some(w) = window {
            let loops = loops_frequency(&self.costas, self.eq.as_ref());
            // No check falls at the end of a window (the windows are shorter than a check).
            self.watch.advance(n, || LockCheck {
                locked: false,
                weight_energy: None,
                weight_floor: 0.0,
            });
            Some(self.watch.window(w.estimate, loops))
        } else {
            let (costas, eq, reference) = (&self.costas, self.eq.as_ref(), self.reference);
            let t = self.watch.advance(n, || LockCheck {
                locked: costas.locked(),
                // The energy at unit input power.
                weight_energy: eq.map(|e| e.weight_energy() * reference),
                weight_floor: match costas.state().constellation() {
                    CostasConstellation::Qam64 => QAM_WATCH_WEIGHT_ENERGY_64,
                    _ => QAM_WATCH_WEIGHT_ENERGY_16,
                },
            });
            if matches!(
                t,
                Some(LockTransition::Settle | LockTransition::SettleUnlocked)
            ) {
                self.reference = self.level.mean();
            }
            t
        };
        if t.is_some_and(|t| t.frequency().is_some()) {
            // A new pull-in.
            self.level = Level::default();
        }
        if let Some(t) = t {
            self.transition(t);
        }
        if self.symbols == self.restart_at {
            self.restart_at = NO_RESTART;
            let t = self.watch.silence();
            self.transition(t);
        }
    }

    /// Carries out a transition of the state machine by its action (`LockTransition::action`,
    /// the one mapping of the transitions to the outputs of M44.1 §2.3; out of line: once per
    /// window or check).
    #[inline(never)]
    #[cold]
    fn transition(&mut self, t: LockTransition) {
        match t.action() {
            // `Acquire` hands the estimate to the loops, `Settle` ends the pull-in; `Miss` (the
            // acquisition takes the next window itself), `Hold` and `Confirm` change nothing.
            LockAction::Keep => {
                if let Some(f) = t.frequency() {
                    self.pull_in(f);
                } else if t == LockTransition::Settle {
                    self.settle();
                }
            }
            // `SettleUnlocked` ends the pull-in too; `Doubt` only takes the window.
            LockAction::Verify => {
                if t == LockTransition::SettleUnlocked {
                    self.settle();
                }
                self.acquire.verify();
            }
            // `AcquireAfterMiss` and `Return`: the equalizer adapted to the noise of the windows
            // without a clear peak (in the search, or in a gap of the signal, M44.1 §6) starts
            // again before the estimate goes to the loops.
            LockAction::RestartEqualizer => {
                self.restart_equalizer();
                if let Some(f) = t.frequency() {
                    self.pull_in(f);
                }
            }
            // `Restart`, `Shrink` and `Silence`: the whole receiver, the acquisition searches.
            LockAction::RestartAll => {
                self.restart_equalizer();
                self.costas.set_frequency(0.0);
                self.acquire.search();
            }
        }
    }

    /// Starts the equalizer again (its start weights, the tracker at [`QAM_TRACKER_PULL_BW`])
    /// and arms the hand-over of a new pull-in.
    fn restart_equalizer(&mut self) {
        if let Some(eq) = self.eq.as_mut() {
            eq.restart();
            // A valid bandwidth: the call cannot fail.
            let _ = eq.set_phase_bw(QAM_TRACKER_PULL_BW);
        }
        self.hand = HandOver::start(self.eq.is_some());
    }

    /// Hands the acquired frequency `f` (cycles per symbol) to the loops: with the equalizer the
    /// frequency integrator of its phase tracker takes `f` (the output is turned back at that rate,
    /// and the tracker, at [`QAM_TRACKER_PULL_BW`] until the next check, takes out the error of the
    /// estimate), the carrier loop starts again from frequency 0 (it sees the turned-back
    /// equalizer output) and the hand-over is armed; without it the carrier loop takes `f`
    /// (within its `max_offset`).
    fn pull_in(&mut self, f: f64) {
        match self.eq.as_mut() {
            Some(eq) => {
                self.hand.arm();
                eq.set_frequency(f);
                self.costas.set_frequency(0.0);
            }
            None => self.costas.set_frequency(f),
        }
    }

    /// The end of the pull-in of the equalizer's phase tracker at the first check (D1622): the
    /// tracker at [`QAM_TRACKER_HOLD_BW`], its phase and frequency kept (`costas` then does the
    /// phase tracking).
    fn settle(&mut self) {
        if let Some(eq) = self.eq.as_mut() {
            // A valid bandwidth: the call cannot fail.
            let _ = eq.set_phase_bw(QAM_TRACKER_HOLD_BW);
        }
    }
}

/// The frequency of the loops in cycles per symbol: the carrier loop's plus that of the
/// equalizer's phase tracker.
fn loops_frequency<S: Storage>(costas: &CostasBlock<S>, eq: Option<&EqualizeBlock<S>>) -> f64 {
    costas.frequency() + eq.map_or(0.0, |e| f64::from(e.state().frequency()))
}

impl<S: Storage> Block for QamDemodBlock<S> {
    type In = C32;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [f32]) -> Result<Progress> {
        let mut produced = self.flush(output)?;
        let mut consumed = 0usize;
        while self.len == 0 && produced < output.len() {
            let m = (input.len() - consumed).min(QAM_DEMOD_CHUNK);
            if m == 0 {
                break;
            }
            let ns = self.piece_symbols(head(tail(input, consumed)?, m)?)?;
            consumed += m;
            let (_, b, pending) = split_work(self.work.as_mut())?;
            let syms = head(b, ns)?;
            let direct = self.demap.process(syms, tail_mut(output, produced)?)?;
            produced += direct.produced;
            let rest = tail(syms, direct.consumed)?;
            if !rest.is_empty() {
                let n = self.demap.process(rest, pending)?.produced;
                self.at = 0;
                self.len = u16::try_from(n)
                    .map_err(|_| Error::Core(sdrk_core::Error::Internal("qam_demod count")))?;
                produced += self.flush(tail_mut(output, produced)?)?;
            }
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let n = self.samples_per_symbol();
        let c = self.config();
        let bits = Ratio::new(self.bits() as u64, 1).unwrap_or(Ratio::ONE);
        BlockTiming {
            ratio: self.sync.timing().ratio.mul(bits).unwrap_or(Ratio::ONE),
            kind: RatioKind::Variable,
            // The matched filter `span n / 2` plus one sample of the timing loop, and the
            // equalizer `(taps - 1) / 2` symbols of `n` input samples (as `psk_demod`).
            delay: Delay::new(
                (c.span * n + 2 + c.equalizer_taps.saturating_sub(1) * n) as u64,
                2,
            )
            .unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
        self.sync.reset();
        self.rx.costas.reset();
        self.at = 0;
        self.len = 0;
        if let Some(eq) = self.rx.eq.as_mut() {
            eq.reset();
            // A valid bandwidth: the call cannot fail.
            let _ = eq.set_phase_bw(QAM_TRACKER_PULL_BW);
        }
        self.rx.acquire.reset();
        self.rx.watch.reset();
        self.rx.hand = HandOver::start(self.rx.eq.is_some());
        self.rx.level = Level::default();
        self.rx.reference = 1.0;
        self.rx.symbols = 0;
        self.rx.restart_at = NO_RESTART;
    }

    fn set_param(&mut self, name: &str, _value: ParamValue) -> Result<()> {
        match DESCRIPTION.param(name) {
            Some(_) => Err(Error::ReadOnlyParameter),
            None => unknown(),
        }
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        self.config().get(name)
    }

    fn min_input(&self) -> usize {
        self.sync.min_input()
    }
}
