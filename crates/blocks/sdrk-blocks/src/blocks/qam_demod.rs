//! `qam_demod`: 16-QAM and 64-QAM demodulator (matched filter, symbol timing, frequency
//! acquisition, equalizer, decision-directed carrier recovery, soft bits).

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::constellation::Qam;
use sdrk_core::kernels::costas::{Costas, CostasConstellation, CostasState};
use sdrk_core::kernels::equalize::{EqConstellation, Equalize, EqualizeState};
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::kernels::filter::{Fir, FirState};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState};
use sdrk_core::num::flush_denormal;

use super::costas::{DEFAULT_DAMPING, DEFAULT_MAX_OFFSET, MAX_LOOP_BW, MIN_LOOP_BW, check_f64};
use super::equalize::{
    DEFAULT_CMA_SYMBOLS, DEFAULT_MU_CMA, DEFAULT_MU_DD, DEFAULT_TAPS, EQ_TAPS, EqualizeConfig,
};
use super::psk_demod::{
    MIN_SYMBOL_RATE_HZ, SPS_TOLERANCE, SYNC_MAX_DEVIATION, check_equalizer_taps, split_work,
    to_u32, work_len,
};
use super::qam_mod::{ORDER_PARAM, order_of_value, qam_of_order};
use super::slice::{qam16_soft_bits, qam64_soft_bits};
use super::symbol_sync::{SymbolSyncBlock, min_input as sync_min_input};
use super::{HZ_LIMIT, check_count, head, head_mut, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::{RRC_MAX_SPAN, RRC_MAX_SPS, RRC_MIN_ROLLOFF, RRC_MIN_SPAN, Rrc};
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};
use crate::{SymbolSyncConfig, Ted};

/// Input samples per piece the block processes at a time (the size of its work buffers).
pub const QAM_DEMOD_CHUNK: usize = 256;

/// The default loop bandwidth of the carrier loop and the phase tracker of `qam_demod` (M44 §4).
const DEFAULT_CARRIER_BW: f64 = 0.005;

/// The default loop bandwidth of the symbol timing of `qam_demod` (half that of `psk_demod`:
/// the self-noise of the Gardner detector on 64-QAM costs 0.9 dB at 0.01, D1215).
const DEFAULT_SYNC_BW: f64 = 0.005;

/// The reason of the rate error (a hint to resample).
const RATE_REASON: &str = "the sample rate must be within 1 % of an integer multiple (2 to \
    1024) of the symbol rate and at least twice it; put a resample before qam_demod";

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
        the equalizer, or to the carrier loop without it), the adaptive equalizer \
        equalize (constellation qam16 or qam64 of the order, equalizer_taps taps, no feedback \
        taps, mu_cma 0.05, mu_dd 0.02, 1000 CMA symbols, then decision-directed with its \
        phase tracker at carrier_bw; equalizer_taps 0 leaves it out), the decision-directed \
        carrier loop costas (constellation qam16 or qam64, carrier_bw, damping 0.7071, \
        max_offset 0.05 cycles per symbol) and the soft bits of slice qam16 or qam64 (scale \
        1): above 0 is a 1. This is the order of psk_demod (equalizer before the carrier \
        loop). The 90 degree ambiguity of the square constellation is not resolved: use \
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
        ParamDesc {
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
        },
        ParamDesc {
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
        },
        ParamDesc {
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
        },
        ParamDesc {
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
        },
        ParamDesc {
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
        },
        ParamDesc {
            name: "carrier_bw",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: MIN_LOOP_BW,
                max: MAX_LOOP_BW,
                default: Some(DEFAULT_CARRIER_BW),
            },
            typical: "0.005; wider for phase noise",
            mutability: Mutability::Build,
            description: "Loop bandwidth of the carrier loop and of the equalizer's phase \
                tracker, normalized to the symbol rate.",
        },
    ],
    required: &["symbol_rate_hz"],
};

/// Configuration of [`QamDemodBlock`].
///
/// ```
/// use sdrk_blocks::QamDemodConfig;
/// let cfg = QamDemodConfig { order: 16, symbol_rate_hz: 25_000.0, ..QamDemodConfig::default() };
/// assert_eq!(cfg.samples_per_symbol(sdrk::Rate::hz(200_000)?)?, 8);
/// assert_eq!((cfg.equalizer_taps, cfg.carrier_bw), (11, 0.005));
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
    /// Loop bandwidth of the carrier loop and of the equalizer's phase tracker (0.0001 to
    /// 0.2).
    pub carrier_bw: f64,
}

impl Default for QamDemodConfig {
    /// Order 16, `symbol_rate_hz` 1200 (a placeholder: the pipeline requires the parameter),
    /// `rolloff` 0.35, `span` 10, `equalizer_taps` 11, `sync_bw` 0.005, `carrier_bw` 0.005.
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
    /// no feedback, the QAM constellation of the order, the step sizes and CMA symbols of the
    /// `equalize` defaults and the phase tracker at `carrier_bw` (M44 §4, D829).
    pub fn equalizer(&self) -> Option<EqualizeConfig> {
        (self.equalizer_taps != 0).then(|| EqualizeConfig {
            taps: self.equalizer_taps,
            feedback_taps: 0,
            constellation: match self.qam() {
                Qam::Qam64 => EqConstellation::Qam64,
                _ => EqConstellation::Qam16,
            },
            mu_cma: DEFAULT_MU_CMA,
            mu_dd: DEFAULT_MU_DD,
            cma_symbols: DEFAULT_CMA_SYMBOLS,
            phase_bw: self.carrier_bw,
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

/// Symbols skipped after a start or `reset` before the frequency acquisition collects its
/// first window (the pull-in of the timing loop).
pub const QAM_ACQUISITION_SKIP: usize = 256;
/// Symbols of the fourth-power frequency estimate of `qam_demod`.
pub const QAM_ACQUISITION_SYMBOLS: usize = 512;
/// The FFT size of the estimate: the symbols zero-padded to twice their number.
const ACQUISITION_FFT: usize = 2 * QAM_ACQUISITION_SYMBOLS;
/// The smallest peak of the periodogram of the fourth powers, relative to their energy
/// (`|X_k|^2 >= ACQUISITION_MIN_PEAK sum |q_n|^2`), for which the estimate is used. Gaussian
/// noise reaches it in 2 of 10000 windows; QAM gives 13 to 46 in the tests (D1403).
const ACQUISITION_MIN_PEAK: f32 = 12.0;
/// The smoothing factor of the symbol power `p` by which the fourth powers are normalized
/// (`p = p + ACQUISITION_POWER_ALPHA (|x|^2 - p)`, a time constant of 64 symbols), so that a
/// level that still changes (an AGC that settles) does not weight the end of the window more.
const ACQUISITION_POWER_ALPHA: f32 = 1.0 / 64.0;

/// The frequency acquisition of `qam_demod` (I44-3, I44-4, D1403): a feed-forward estimate of
/// the carrier offset from the fourth powers of the symbols of the timing loop, made during the
/// blind (CMA) symbols of the equalizer when the signal is there from the start.
///
/// Per symbol `x` the smoothed power `p = flush(p + (|x|^2 - p) / 64)` (finite values only,
/// from 0 at the start). From symbol [`QAM_ACQUISITION_SKIP`] on, windows of
/// [`QAM_ACQUISITION_SYMBOLS`] symbols of `q = x^4 / (p p)` (0 when not finite or `p` is not above 0;
/// the normalization keeps a level that still settles, as behind an AGC, from weighting the
/// end of the window more). For square QAM the mean of `x^4` is `E[s^4] e^(j 2 pi 4 f n)` with
/// the offset `f` in cycles per symbol (`E[s^4]` = -0.68 and -0.62 at unit power, not 0). The
/// periodogram of `q` zero-padded to `ACQUISITION_FFT` points (the kernel `fft`) has its
/// largest bin `k` near `4 f ACQUISITION_FFT`; the parabola through the magnitudes of the bins
/// `k - 1`, `k`, `k + 1` refines it, and `f = ((k + delta) / ACQUISITION_FFT wrapped into
/// [-0.5, 0.5)) / 4`, the range `[-0.125, 0.125)` cycles per symbol of the fourth power. The
/// estimate is used when the peak stands clear of the noise (`|X_k|^2 >= ACQUISITION_MIN_PEAK
/// sum |q_n|^2`); otherwise the next window follows. After a used estimate nothing more is
/// computed until `reset`.
struct Acquisition<S: Storage> {
    /// The twiddles of the FFT (the plan is made from them for each estimate, so that the block
    /// stays small).
    twiddles: S::Buf<C32>,
    /// The fourth powers, then the spectrum.
    buf: S::Buf<C32>,
    /// Symbols seen since the start or the last window without a clear peak (counted from
    /// [`QAM_ACQUISITION_SKIP`] then), up to the end of the window.
    seen: u32,
    /// The estimate, once one was used.
    estimate: Option<f64>,
    /// The smoothed symbol power.
    power: f32,
}

impl<S: Storage> Acquisition<S> {
    fn new(storage: &mut S) -> Result<Self> {
        let mut twiddles = storage.alloc::<C32>(ACQUISITION_FFT / 2)?;
        FftPlan::new(ACQUISITION_FFT, twiddles.as_mut())?;
        Ok(Self {
            twiddles,
            buf: storage.alloc::<C32>(ACQUISITION_FFT)?,
            seen: 0,
            estimate: None,
            power: 0.0,
        })
    }

    fn plan(plan: &mut Plan) -> Result<()> {
        plan.add::<C32>(ACQUISITION_FFT / 2)?;
        plan.add::<C32>(ACQUISITION_FFT)?;
        Ok(())
    }

    fn reset(&mut self) {
        self.seen = 0;
        self.estimate = None;
        self.power = 0.0;
    }

    /// Symbols until the window is complete; `None` once an estimate is used.
    fn remaining(&self) -> Option<usize> {
        match self.estimate {
            Some(_) => None,
            None => Some(QAM_ACQUISITION_SKIP + QAM_ACQUISITION_SYMBOLS - self.seen as usize),
        }
    }

    /// Takes the next symbols (at most [`Acquisition::remaining`]).
    fn collect(&mut self, x: &[C32]) {
        if self.estimate.is_some() {
            return;
        }
        let buf = self.buf.as_mut();
        for &v in x {
            let p = v.re * v.re + v.im * v.im;
            if p.is_finite() {
                self.power =
                    flush_denormal(self.power + ACQUISITION_POWER_ALPHA * (p - self.power));
            }
            let n = self.seen as usize;
            if let Some(slot) = n
                .checked_sub(QAM_ACQUISITION_SKIP)
                .and_then(|i| buf.get_mut(i))
            {
                *slot = normalized_fourth_power(v, self.power);
            }
            self.seen += 1;
        }
    }

    /// The estimate in cycles per symbol when a window is complete and its peak is clear (then
    /// the acquisition ends); a window without a clear peak starts the next one.
    fn take(&mut self) -> Result<Option<f64>> {
        if self.remaining() != Some(0) {
            return Ok(None);
        }
        let f = self.estimate_now()?;
        match f {
            Some(_) => self.estimate = f,
            None => self.seen = QAM_ACQUISITION_SKIP as u32,
        }
        Ok(f)
    }

    fn estimate_now(&mut self) -> Result<Option<f64>> {
        let buf = self.buf.as_mut();
        let (fourth, pad) = buf.split_at_mut(QAM_ACQUISITION_SYMBOLS);
        let energy: f32 = fourth.iter().map(|v| v.re * v.re + v.im * v.im).sum();
        pad.fill(C32::new(0.0, 0.0));
        let plan = FftPlan::from_twiddles(ACQUISITION_FFT, self.twiddles.as_ref())?;
        <Sdrk as Fft<C32>>::fft(&Sdrk, &plan, buf, Direction::Forward, Scaling::Default)?;
        let power = |v: &C32| v.re * v.re + v.im * v.im;
        let mut bin = 0usize;
        let mut peak = -1.0f32;
        for (i, v) in buf.iter().enumerate() {
            let p = power(v);
            if p > peak {
                peak = p;
                bin = i;
            }
        }
        if !(energy.is_finite() && peak.is_finite() && energy > 0.0)
            || peak < ACQUISITION_MIN_PEAK * energy
        {
            return Ok(None);
        }
        let size = ACQUISITION_FFT;
        let mag = |i: usize| {
            buf.get(i % size)
                .map_or(0.0, |v| libm::sqrt(f64::from(power(v))))
        };
        let (below, top, above) = (mag(bin + size - 1), mag(bin), mag(bin + 1));
        let den = below - 2.0 * top + above;
        let delta = if den < 0.0 {
            0.5 * (below - above) / den
        } else {
            0.0
        };
        let mut cycles = (bin as f64 + delta) / size as f64;
        if cycles >= 0.5 {
            cycles -= 1.0;
        }
        Ok(Some(cycles / 4.0))
    }

    /// The estimate, if one was made and used.
    fn frequency(&self) -> Option<f64> {
        self.estimate
    }
}

/// `x^4 / (p p)` in `f32`: `a = x^2 = (x.re x.re - x.im x.im, 2 x.re x.im)`,
/// `q = a^2 = (a.re a.re - a.im a.im, 2 a.re a.im)`, each part divided by `p p`; 0 when `p` is
/// not above 0 or the result is not finite.
#[inline]
fn normalized_fourth_power(x: C32, p: f32) -> C32 {
    let a = C32::new(x.re * x.re - x.im * x.im, 2.0 * x.re * x.im);
    let pp = p * p;
    let q = C32::new((a.re * a.re - a.im * a.im) / pp, (2.0 * a.re * a.im) / pp);
    if p > 0.0 && q.re.is_finite() && q.im.is_finite() {
        q
    } else {
        C32::new(0.0, 0.0)
    }
}

/// The state of the equalizer of `qam_demod` on the storage `S`.
type EqState<S> = EqualizeState<<S as Storage>::Buf<C32>, <S as Storage>::Buf<C32>>;

/// 16-QAM and 64-QAM demodulator: `C32` in, `f32` soft bits out (`bits_per_symbol` per symbol,
/// above 0 is a 1).
///
/// Per piece of at most [`QAM_DEMOD_CHUNK`] input samples: the kernel `fir` (matched filter,
/// taps `rrc(rolloff, n, span) / sqrt(n)` as `f32` with `n` the rounded samples per symbol),
/// the kernel `symbol_sync` (Gardner at the exact `fs / symbol_rate_hz`), the kernel
/// `equalize` (when `equalizer_taps` is above 0), the kernel `costas` with the constellation
/// `qam16` or `qam64`, and the soft bits of `slice` `qam16` or `qam64` (scale 1), the same
/// functions as the blocks. Until it has found the carrier offset, the frequency acquisition
/// (I44-3, I44-4) collects the normalized fourth powers of the symbols of the timing loop in
/// windows of [`QAM_ACQUISITION_SYMBOLS`] from symbol [`QAM_ACQUISITION_SKIP`] on and estimates the
/// offset from their spectrum (the kernel `fft`); the equalizer and the carrier loop run in two
/// calls split after the symbol that completes a window, and a clear estimate goes to the
/// equalizer's phase tracker (the carrier loop's frequency to 0) or, without the equalizer, to
/// the carrier loop. Soft bits that do not fit into the output wait in a buffer and are
/// written first by the next call (which then consumes no input). Timing: variable ratio
/// `symbol_rate bits / fs`, delay `(span + equalizer_taps - 1) n / 2 + 1` input samples.
/// Storage: the taps, the filter history, one work buffer of `512 + 128 bits` `C32` (filter
/// output, symbols, waiting soft bits as `f32`), the 1024 `C32` of the acquisition's FFT and
/// its 512 twiddles, and for `equalizer_taps` `t > 0` the `t` weights and the `t` history
/// values of the equalizer.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig};
/// let rate = Rate::hz(4000)?; // 1000 symbols per second, 4 samples per symbol
/// let tx = QamModConfig { order: 16, samples_per_symbol: 4, ..QamModConfig::default() };
/// let rx = QamDemodConfig { order: 16, symbol_rate_hz: 1000.0, ..QamDemodConfig::default() };
/// let mut mem = vec![0u8; 1 << 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut m = QamModBlock::new(&tx, Rate::hz(4000)?, &mut arena)?;
/// let mut d = QamDemodBlock::new(&rx, rate, &mut arena)?;
/// let bits: Vec<f32> = (0..16_000u32).map(|k| if (k * 7 + k / 5) % 3 == 0 { 1.0 } else { -1.0 }).collect();
/// let mut sig = vec![C32::default(); 16_000];
/// let p = m.process(&bits, &mut sig)?;
/// assert_eq!(p.produced, 16_000);
/// let mut soft = vec![0.0f32; 16_400];
/// let q = d.process(&sig, &mut soft)?;
/// assert_eq!(q.consumed, 16_000);
/// assert!((15_900..=16_000).contains(&q.produced));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QamDemodBlock<S: Storage> {
    config: QamDemodConfig,
    qam: Qam,
    /// Bits per symbol and the rounded samples per symbol.
    bps: u32,
    sps: u32,
    ratio: Ratio,
    fir: FirState<C32, S::Buf<f32>, S::Buf<C32>>,
    sync: SymbolSyncState<C32>,
    costas: CostasState,
    /// One work buffer `[a | b | pending]` (`split_work` of `psk_demod`): `a` the
    /// matched-filter output, then the equalized symbols; `b` the symbols of `symbol_sync`,
    /// then those of the carrier loop; `pending` (as `f32`) the soft bits that did not fit into
    /// the output, `pending[at..len]`.
    work: S::Buf<C32>,
    at: u32,
    len: u32,
    /// The equalizer before the carrier loop; `None` for `equalizer_taps` 0.
    eq: Option<EqState<S>>,
    /// The frequency acquisition.
    acq: Acquisition<S>,
}

impl<S: Storage> core::fmt::Debug for QamDemodBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QamDemodBlock")
            .field("config", &self.config)
            .field("costas", &self.costas)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> QamDemodBlock<S> {
    /// Designs the matched filter and creates the block; taps, history and work buffers come
    /// from `storage`.
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
        let bps = qam.bits_per_symbol();
        let work = storage.alloc::<C32>(work_len(bps))?;
        let eq = match config.equalizer() {
            Some(e) => Some(e.state(storage)?),
            None => None,
        };
        let acq = Acquisition::new(storage)?;
        let block = SymbolSyncBlock::<C32, S>::new(&config.sync(), rate_in, storage)?;
        let ratio = block.timing().ratio.mul(Ratio::new(bps as u64, 1)?)?;
        let costas = CostasState::new(
            match qam {
                Qam::Qam64 => CostasConstellation::Qam64,
                _ => CostasConstellation::Qam16,
            },
            config.carrier_bw,
            DEFAULT_DAMPING,
            DEFAULT_MAX_OFFSET,
        )?;
        Ok(Self {
            config: *config,
            qam,
            bps: to_u32(bps)?,
            sps: to_u32(n)?,
            ratio,
            fir: FirState::new(taps, hist)?,
            sync: *block.state(),
            costas,
            work,
            at: 0,
            len: 0,
            eq,
            acq,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`QamDemodConfig::validate`].
    pub fn storage_bytes(config: &QamDemodConfig, rate_in: Rate) -> Result<usize> {
        let n = config.derive(rate_in)?;
        let taps = config.span * n + 1;
        let mut plan = Plan::default();
        plan.add::<f32>(taps)?;
        plan.add::<C32>(taps - 1)?;
        plan.add::<C32>(work_len(config.bits_per_symbol()))?;
        if let Some(e) = config.equalizer() {
            e.plan(&mut plan)?;
        }
        Acquisition::<S>::plan(&mut plan)?;
        Ok(plan.bytes())
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

    /// Number of taps of the matched filter (`span n + 1`).
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }

    /// The integer samples per symbol of the matched filter.
    pub fn samples_per_symbol(&self) -> usize {
        self.sps as usize
    }

    /// Whether the carrier loop reports lock (the lock detector of `costas`: the decision error
    /// against the QAM grid, D1400).
    pub fn locked(&self) -> bool {
        self.costas.locked()
    }

    /// The estimated carrier offset in cycles per symbol: the frequency of the carrier loop plus
    /// that of the equalizer's phase tracker (which holds the acquired frequency).
    pub fn frequency(&self) -> f64 {
        self.costas.frequency() + self.eq.as_ref().map_or(0.0, |e| f64::from(e.frequency()))
    }

    /// The carrier offset in cycles per symbol found by the frequency acquisition (the
    /// fourth-power estimate over windows of 512 symbols from symbol 256 after a start or
    /// `reset`, the first whose spectral peak stands clear of the noise); `None` before.
    pub fn acquired_frequency(&self) -> Option<f64> {
        self.acq.frequency()
    }

    /// The state of the equalizer (`None` for `equalizer_taps` 0).
    pub fn equalizer(&self) -> Option<&EqState<S>> {
        self.eq.as_ref()
    }

    /// The current estimate of the samples per symbol of the timing loop.
    pub fn sps_estimate(&self) -> f64 {
        self.sync.sps_estimate()
    }

    /// Writes waiting soft bits to `output`; returns how many.
    fn flush(&mut self, output: &mut [f32]) -> Result<usize> {
        let (at, len) = (self.at as usize, self.len as usize);
        let (_, _, pending) = split_work(self.work.as_mut())?;
        let waiting = head(tail(pending, at)?, len - at)?;
        let k = waiting.len().min(output.len());
        head_mut(output, k)?.copy_from_slice(head(waiting, k)?);
        // `k <= len <= QAM_DEMOD_CHUNK bits`.
        self.at += k as u32;
        if self.at == self.len {
            self.at = 0;
            self.len = 0;
        }
        Ok(k)
    }

    /// One piece of input samples: the symbols after the carrier loop in `b[..ns]`; returns
    /// `ns`.
    fn symbols(&mut self, piece: &[C32]) -> Result<usize> {
        let m = piece.len();
        let (a, b, _) = split_work(self.work.as_mut())?;
        <Sdrk as Fir<C32>>::fir(&Sdrk, &mut self.fir, piece, head_mut(a, m)?)?;
        let p = <Sdrk as SymbolSync<C32>>::symbol_sync(&Sdrk, &mut self.sync, head(a, m)?, b)?;
        if p.consumed != m {
            return Err(Error::Core(sdrk_core::Error::Internal(
                "qam_demod timing loop did not take the whole piece",
            )));
        }
        let ns = p.produced;
        // `b` -> `a` (the equalizer, or a copy without it) -> `b` (the carrier loop), in runs
        // that end where the frequency acquisition completes its window.
        let mut at = 0;
        while at < ns {
            let end = self.acq.remaining().map_or(ns, |r| ns.min(at + r));
            let src = head(tail(b, at)?, end - at)?;
            self.acq.collect(src);
            let dst = head_mut(tail_mut(a, at)?, end - at)?;
            match self.eq.as_mut() {
                Some(eq) => <Sdrk as Equalize>::equalize(&Sdrk, eq, src, dst)?,
                None => dst.copy_from_slice(src),
            }
            let out = head_mut(tail_mut(b, at)?, end - at)?;
            <Sdrk as Costas>::costas(&Sdrk, &mut self.costas, dst, out)?;
            if let Some(f) = self.acq.take()? {
                apply::<S>(&mut self.eq, &mut self.costas, f);
            }
            at = end;
        }
        Ok(ns)
    }
}

/// Hands the acquired frequency `f` (cycles per symbol) to the loops: with the equalizer its
/// phase tracker takes `f` and the carrier loop starts again from frequency 0 (it sees the
/// equalizer output turned back by the tracker); without it the carrier loop takes `f` (within
/// its `max_offset`).
fn apply<S: Storage>(eq: &mut Option<EqState<S>>, costas: &mut CostasState, f: f64) {
    match eq.as_mut() {
        Some(eq) => {
            eq.set_frequency(f as f32);
            costas.set_frequency(0.0);
        }
        None => costas.set_frequency(f),
    }
}

/// The soft bits of the symbols in `input` (scale 1); takes `min(input.len(), output.len() /
/// bits)` symbols and returns that count.
fn demod_bits(qam: Qam, input: &[C32], output: &mut [f32]) -> usize {
    match qam {
        Qam::Qam64 => qam64_soft_bits(1.0, input, output),
        _ => qam16_soft_bits(1.0, input, output),
    }
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
        let qam = self.qam;
        while self.len == 0 && produced < output.len() {
            let m = (input.len() - consumed).min(QAM_DEMOD_CHUNK);
            if m == 0 {
                break;
            }
            let ns = self.symbols(head(tail(input, consumed)?, m)?)?;
            consumed += m;
            let bps = self.bps as usize;
            let (_, b, pending) = split_work(self.work.as_mut())?;
            let syms = head(b, ns)?;
            let direct = demod_bits(qam, syms, tail_mut(output, produced)?);
            produced += direct * bps;
            let rest = tail(syms, direct)?;
            if !rest.is_empty() {
                let n = demod_bits(qam, rest, pending);
                self.at = 0;
                self.len = to_u32(n * bps)?;
                produced += self.flush(tail_mut(output, produced)?)?;
            }
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let n = self.sps as usize;
        BlockTiming {
            ratio: self.ratio,
            kind: RatioKind::Variable,
            // The matched filter `span n / 2` plus one sample of the timing loop, and the
            // equalizer `(taps - 1) / 2` symbols of `n` input samples (as `psk_demod`).
            delay: Delay::new(
                (self.config.span * n + 2 + self.config.equalizer_taps.saturating_sub(1) * n)
                    as u64,
                2,
            )
            .unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.fir.reset();
        self.sync.reset();
        self.costas.reset();
        self.at = 0;
        self.len = 0;
        if let Some(eq) = self.eq.as_mut() {
            eq.reset();
        }
        self.acq.reset();
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
        sync_min_input(self.sync.sps(), SYNC_MAX_DEVIATION)
    }
}
