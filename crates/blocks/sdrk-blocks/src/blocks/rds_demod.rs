//! `rds_demod`: the 57 kHz RDS subcarrier of an FM multiplex to soft data bits (mixing,
//! filtering, chip timing, BPSK carrier loop, biphase and differential decoding).

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind, Sdrk};
use sdrk_core::C32;
use sdrk_core::kernels::costas::{Costas, CostasState};
use sdrk_core::kernels::decim::{FirDecimState, FirDecimate};
use sdrk_core::kernels::math::Interleave;
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::symbol_sync::{SymbolSync, SymbolSyncState};

use super::costas::{DEFAULT_DAMPING, DEFAULT_MAX_OFFSET, MAX_LOOP_BW, MIN_LOOP_BW, check_f64};
use super::fir::MAX_TAPS_LIMIT;
use super::symbol_sync::SymbolSyncBlock;
use super::{head, head_mut, hz, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::design::DesignSpec;
use crate::error::{Error, Result};
use crate::storage::{Plan, Storage};
use crate::{SymbolSyncConfig, Ted};

/// Input samples per piece the block processes at a time (the size of its work buffers).
pub const RDS_DEMOD_CHUNK: usize = 1024;

/// Smallest input sample rate in Hz: the subcarrier band reaches 59.4 kHz, and the
/// decimation by `floor(fs / 19000)` needs at least 6 to keep the chip rate below a ninth of
/// the decimated rate.
pub const RDS_DEMOD_MIN_RATE_HZ: f64 = 128_000.0;

/// Frequency of the RDS subcarrier (three times the pilot) in Hz.
const SUBCARRIER_HZ: f64 = 57_000.0;
/// The decimation is `floor(fs / DECIMATION_HZ)`.
const DECIMATION_HZ: u128 = 19_000;
/// The chip rate of the biphase signal (twice the bit rate 1187.5 bit/s) in Hz.
const CHIP_RATE_HZ: f64 = 2375.0;
/// The channel filter of §2 of M41: cutoff, transition, stop-band attenuation.
const FILTER: DesignSpec = DesignSpec {
    cutoff_hz: 2400.0,
    transition_hz: 1500.0,
    attenuation_db: 60.0,
};
/// DC gain of the channel filter: the multiplex `0.04 b sin(3 phi)` of a 4 % subcarrier gives
/// the complex baseband `0.02 b` after the mixer, so the chips of the nominal level reach
/// about 1, the level the loops of `symbol_sync` and `costas` are designed for (D670).
const FILTER_GAIN: f64 = 50.0;
/// The largest deviation of the chip clock the timing loop follows.
const SYNC_MAX_DEVIATION: f64 = 0.005;
/// The rate of the `|u|` averages of the biphase pairings (time constant 32 bits).
const PAIRING_RATE: f32 = 1.0 / 32.0;
/// The other pairing is taken when its average exceeds the current one by this factor.
const PAIRING_HYSTERESIS: f32 = 1.2;
/// Default loop bandwidth of the timing and the carrier loop (§2 of M41).
const DEFAULT_BW: f64 = 0.01;

/// Description of the `rds_demod` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "rds_demod",
    summary: "Demodulates the 57 kHz RDS subcarrier of an FM multiplex to soft data bits at \
        1187.5 bit/s.",
    details: "The input is the FM multiplex (the f32 output of quad_demod, 1.0 at the full \
        deviation) at 128 kHz or more. The block mixes the subcarrier at 57 kHz to 0 Hz \
        (interleave, rotate by -57000 Hz), low-pass filters and decimates it by d = floor(fs / \
        19000) (Kaiser design, cutoff 2400 Hz, transition 1500 Hz, 60 dB, DC gain 50, so a \
        subcarrier at 4 % of the multiplex gives chips of about 1), recovers the chip clock at \
        2375 Hz (symbol_sync, Gardner, sync_bw, damping 0.7071, max_deviation 0.005) and the \
        carrier (costas order 2, carrier_bw), which puts the BPSK chips on the real axis. \
        Biphase decoding forms u = (c0.re - c1.re) / 2 from two chips; both pairings (even and \
        odd chips first) run, each keeps the average of |u| over 32 bits, and the block \
        outputs the pairing with the larger average, switching only when the other is larger \
        by a factor 1.2. Differential decoding gives soft = -u_k u_k-1 (above 0 is a 1); it \
        cancels the 180 degree ambiguity of the carrier loop. A switch of the pairing may drop \
        or repeat half a bit; rds_decode then finds the blocks again. The output rate follows \
        the clock of the signal (variable ratio 1187.5 / fs). Pitfall: the loops assume about \
        the nominal level; a subcarrier much weaker than 2 % slows them down.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "The FM multiplex at 128 kHz or more, 1.0 at the full deviation.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "Soft data bits at 1187.5 bit/s; above 0 is a 1.",
    }],
    params: &[
        ParamDesc {
            name: "sync_bw",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: 0.0001,
                max: 0.1,
                default: Some(DEFAULT_BW),
            },
            typical: "0.01",
            mutability: Mutability::Build,
            description: "Loop bandwidth of the chip timing, normalized to the chip rate \
                2375 Hz.",
        },
        ParamDesc {
            name: "carrier_bw",
            unit: Unit::None,
            kind: ParamKind::F64 {
                min: MIN_LOOP_BW,
                max: MAX_LOOP_BW,
                default: Some(DEFAULT_BW),
            },
            typical: "0.01",
            mutability: Mutability::Build,
            description: "Loop bandwidth of the carrier loop, normalized to the chip rate \
                2375 Hz.",
        },
    ],
    required: &[],
};

/// Configuration of [`RdsDemodBlock`].
///
/// ```
/// use sdrk_blocks::RdsDemodConfig;
/// let cfg = RdsDemodConfig::default();
/// assert_eq!((cfg.sync_bw, cfg.carrier_bw), (0.01, 0.01));
/// assert_eq!(cfg.decimation(sdrk::Rate::hz(240_000)?)?, 12);
/// assert!(cfg.validate(sdrk::Rate::hz(96_000)?).is_err()); // below 128 kHz
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdsDemodConfig {
    /// Loop bandwidth of the chip timing (0.0001 to 0.1).
    pub sync_bw: f64,
    /// Loop bandwidth of the carrier loop (0.0001 to 0.2).
    pub carrier_bw: f64,
}

impl Default for RdsDemodConfig {
    /// `sync_bw` 0.01, `carrier_bw` 0.01.
    fn default() -> Self {
        Self {
            sync_bw: DEFAULT_BW,
            carrier_bw: DEFAULT_BW,
        }
    }
}

impl RdsDemodConfig {
    /// The configuration of the chip timing loop.
    fn sync(&self) -> SymbolSyncConfig {
        SymbolSyncConfig {
            symbol_rate_hz: CHIP_RATE_HZ,
            ted: Ted::Gardner,
            loop_bw: self.sync_bw,
            damping: DEFAULT_DAMPING,
            max_deviation: SYNC_MAX_DEVIATION,
            ted_gain: 1.0,
        }
    }

    /// The decimation `floor(fs / 19000)` (from the exact rate), after the checks.
    ///
    /// # Errors
    /// As [`RdsDemodConfig::validate`].
    pub fn decimation(&self, rate_in: Rate) -> Result<usize> {
        check_f64("sync_bw", self.sync_bw, 0.0001, 0.1)?;
        check_f64("carrier_bw", self.carrier_bw, MIN_LOOP_BW, MAX_LOOP_BW)?;
        let fs = hz(rate_in);
        if fs.is_nan() || fs < RDS_DEMOD_MIN_RATE_HZ {
            return Err(Error::OutOfRange {
                name: "sample_rate",
                min: RDS_DEMOD_MIN_RATE_HZ,
                max: f64::INFINITY,
                got: fs,
            });
        }
        let d = u128::from(rate_in.num()) / (u128::from(rate_in.den()) * DECIMATION_HZ);
        usize::try_from(d).map_err(|_| Error::OutOfRange {
            name: "sample_rate",
            min: RDS_DEMOD_MIN_RATE_HZ,
            max: f64::INFINITY,
            got: fs,
        })
    }

    /// The number of taps of the channel filter at the input rate (also checks the whole
    /// configuration).
    ///
    /// ```
    /// use sdrk_blocks::RdsDemodConfig;
    /// assert_eq!(RdsDemodConfig::default().num_taps(sdrk::Rate::hz(240_000)?)?, 583);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    ///
    /// # Errors
    /// As [`RdsDemodConfig::validate`].
    pub fn num_taps(&self, rate_in: Rate) -> Result<usize> {
        let d = self.decimation(rate_in)?;
        self.sync().validate(divide(rate_in, d)?)?;
        FILTER.num_taps(hz(rate_in), MAX_TAPS_LIMIT as usize)
    }

    /// Checks the configuration against the input sample rate.
    ///
    /// # Errors
    /// `OutOfRange` for a loop bandwidth outside its range, and with the name `sample_rate`
    /// for an input rate below 128 kHz (the pipeline reports it as `rate_unsupported`) or one
    /// whose channel filter would need more than 65536 taps.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.num_taps(rate_in).map(|_| ())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// ```
    /// use sdrk_blocks::{ParamValue, RdsDemodConfig};
    /// let mut cfg = RdsDemodConfig::default();
    /// cfg.set("carrier_bw", ParamValue::F64(0.02))?;
    /// assert_eq!(cfg.carrier_bw, 0.02);
    /// assert!(cfg.set("sync_bw", ParamValue::F64(0.5)).is_err());
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
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
            "sync_bw" => Ok(ParamValue::F64(self.sync_bw)),
            "carrier_bw" => Ok(ParamValue::F64(self.carrier_bw)),
            _ => unknown(),
        }
    }
}

/// `rate / d` as an exact rate.
fn divide(rate: Rate, d: usize) -> Result<Rate> {
    Ok(rate.scale(Ratio::new(1, d as u64)?)?)
}

/// Samples of the decimated and the chip buffers: the most outputs of one piece.
fn dec_len(d: usize) -> usize {
    RDS_DEMOD_CHUNK.div_ceil(d.max(1)) + 1
}

/// One biphase pairing: the previous `u`, the average of `|u|`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Pairing {
    prev_u: f32,
    level: f32,
}

/// The biphase and differential decoder of §2 of M41 (steps 5 and 6) at the chip rate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Biphase {
    /// The real part of the previous chip.
    prev: f32,
    /// The pairing that completes a pair with the next chip (0 or 1).
    next: usize,
    /// The pairings, index 0 for "even chips first".
    pairs: [Pairing; 2],
    /// The selected pairing.
    selected: usize,
    /// Whether a chip arrived since construction or reset.
    started: bool,
}

impl Biphase {
    /// Takes one chip; returns the soft bit when the selected pairing completed a pair.
    fn push(&mut self, chip: f32) -> Option<f32> {
        if !self.started {
            // The first chip starts the even pairing; the odd one completes with the next.
            self.started = true;
            self.prev = chip;
            self.next = 0;
            return None;
        }
        let k = self.next;
        let u = (self.prev - chip) / 2.0;
        self.prev = chip;
        self.next = 1 - k;
        let [p0, p1] = &mut self.pairs;
        let p = if k == 0 { p0 } else { p1 };
        let mag = libm::fabsf(u);
        // A non-finite chip must not poison the average for good.
        let step = if mag.is_finite() { mag - p.level } else { 0.0 };
        p.level = sdrk_core::num::flush_denormal(p.level + step * PAIRING_RATE);
        let soft = -u * p.prev_u;
        p.prev_u = if u.is_finite() { u } else { 0.0 };
        let out = (k == self.selected).then_some(soft);
        let [a0, a1] = self.pairs.map(|p| p.level);
        let (cur, other) = if self.selected == 0 {
            (a0, a1)
        } else {
            (a1, a0)
        };
        if other > cur * PAIRING_HYSTERESIS {
            self.selected = 1 - self.selected;
        }
        out
    }
}

/// RDS demodulator: `f32` multiplex in, `f32` soft data bits out (above 0 is a 1).
///
/// Per piece of at most [`RDS_DEMOD_CHUNK`] input samples, each step a kernel call through
/// the facade: `interleave` (the multiplex as the real part), `rotate` by -57000 Hz,
/// `fir_decimate` by `d = floor(fs / 19000)` with the channel filter (Kaiser, cutoff 2400 Hz,
/// transition 1500 Hz, 60 dB; the taps of the design times 50 as `f32`), `symbol_sync`
/// (Gardner at the exact `(fs / d) / 2375` samples per chip) and `costas` order 2. Then, in
/// scalar code per chip `c` (the real part after the carrier loop): with `p` the previous chip,
/// `u = (p - c) / 2` completes a pair of the pairing that alternates with every chip (the
/// first chip after construction or reset only starts the even pairing). That pairing updates
/// `level += (|u| - level) / 32` and, if it is the selected one, outputs
/// `soft = -u u_prev` with its own previous `u` (0 at the start); then the selection changes
/// when the other level exceeds 1.2 times the selected one. Soft bits that do not fit into
/// the output wait in a buffer and are written first by the next call (which then consumes no
/// input). Timing: variable ratio `1187.5 / fs`, delay `(taps - 1) / 2 + d` input samples (the
/// channel filter and the one-sample delay of `symbol_sync`). Storage: the taps, the filter
/// history, two complex work buffers of [`RDS_DEMOD_CHUNK`] samples and the waiting soft bits.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, RdsDemodBlock, RdsDemodConfig};
/// let rate = Rate::hz(240_000)?;
/// let cfg = RdsDemodConfig::default();
/// let mut mem = vec![0u8; RdsDemodBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut d = RdsDemodBlock::new(&cfg, rate, &mut arena)?;
/// assert_eq!((d.decimation(), d.num_taps()), (12, 583));
/// // A multiplex without subcarrier: the block runs and outputs soft bits at 1187.5 bit/s.
/// let mpx: Vec<f32> = (0..48_000).map(|n| 0.1 * (n as f32 * 0.4974).sin()).collect();
/// let mut soft = vec![0.0f32; 400];
/// let p = d.process(&mpx, &mut soft)?;
/// assert_eq!(p.consumed, 48_000);
/// assert!((200..=280).contains(&p.produced)); // 0.2 s
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct RdsDemodBlock<S: Storage> {
    config: RdsDemodConfig,
    d: usize,
    ratio: Ratio,
    min_input: usize,
    nco: Nco,
    fir: FirDecimState<C32, S::Buf<f32>, S::Buf<C32>>,
    sync: SymbolSyncState<C32>,
    costas: CostasState,
    biphase: Biphase,
    /// The complex multiplex, then the decimated samples, then the chips after the carrier
    /// loop.
    a: S::Buf<C32>,
    /// The mixed multiplex, then the chips of `symbol_sync`.
    b: S::Buf<C32>,
    /// Soft bits that did not fit into the output, `pending[at..len]`.
    pending: S::Buf<f32>,
    at: usize,
    len: usize,
}

impl<S: Storage> core::fmt::Debug for RdsDemodBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RdsDemodBlock")
            .field("config", &self.config)
            .field("decimation", &self.d)
            .field("costas", &self.costas)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> RdsDemodBlock<S> {
    /// Designs the channel filter and creates the block; taps, history and work buffers come
    /// from `storage`.
    ///
    /// # Errors
    /// As [`RdsDemodConfig::validate`]; `Storage` when the storage is too small.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    pub fn new(config: &RdsDemodConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        let n = config.num_taps(rate_in)?;
        let d = config.decimation(rate_in)?;
        let fs = hz(rate_in);
        let lp = FILTER.lowpass(fs, MAX_TAPS_LIMIT as usize)?;
        let mut taps = storage.alloc::<f32>(n)?;
        for (k, t) in taps.as_mut().iter_mut().enumerate() {
            *t = (lp.tap(k) * FILTER_GAIN) as f32;
        }
        let hist = storage.alloc::<C32>(n - 1)?;
        let a = storage.alloc::<C32>(RDS_DEMOD_CHUNK)?;
        let b = storage.alloc::<C32>(RDS_DEMOD_CHUNK)?;
        let pending = storage.alloc::<f32>(dec_len(d))?;
        let rate_dec = divide(rate_in, d)?;
        let block = SymbolSyncBlock::<C32, S>::new(&config.sync(), rate_dec, storage)?;
        let ratio = block.timing().ratio.mul(Ratio::new(1, 2 * d as u64)?)?;
        // Two chips of the timing loop, in input samples.
        let min_input = block.min_input().saturating_mul(2).saturating_mul(d);
        let sync = *block.state();
        let costas = CostasState::new(2, config.carrier_bw, DEFAULT_DAMPING, DEFAULT_MAX_OFFSET)?;
        Ok(Self {
            config: *config,
            d,
            ratio,
            min_input,
            nco: Nco::new(-SUBCARRIER_HZ, rate_in)?,
            fir: FirDecimState::new(taps, hist, d)?,
            sync,
            costas,
            biphase: Biphase::default(),
            a,
            b,
            pending,
            at: 0,
            len: 0,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`RdsDemodConfig::validate`].
    pub fn storage_bytes(config: &RdsDemodConfig, rate_in: Rate) -> Result<usize> {
        let n = config.num_taps(rate_in)?;
        let d = config.decimation(rate_in)?;
        let mut plan = Plan::default();
        plan.add::<f32>(n)?;
        plan.add::<C32>(n - 1)?;
        plan.add::<C32>(RDS_DEMOD_CHUNK)?;
        plan.add::<C32>(RDS_DEMOD_CHUNK)?;
        plan.add::<f32>(dec_len(d))?;
        Ok(plan.bytes())
    }

    /// Output rate: 1187.5 Hz (half the chip rate 2375 Hz).
    ///
    /// # Errors
    /// As [`RdsDemodConfig::validate`].
    pub fn rate_out(config: &RdsDemodConfig, rate_in: Rate) -> Result<Rate> {
        let d = config.decimation(rate_in)?;
        config.validate(rate_in)?;
        let chips = SymbolSyncBlock::<C32, S>::rate_out(&config.sync(), divide(rate_in, d)?)?;
        Ok(chips.scale(Ratio::new(1, 2)?)?)
    }

    /// Number of taps of the channel filter.
    pub fn num_taps(&self) -> usize {
        self.fir.taps().len()
    }

    /// The decimation `floor(fs / 19000)`.
    pub fn decimation(&self) -> usize {
        self.d
    }

    /// Whether the carrier loop reports lock.
    pub fn locked(&self) -> bool {
        self.costas.locked()
    }

    /// The biphase pairing in use: 0 when the pairs start with the even chips (counted from
    /// the first chip after construction or reset), 1 for the odd chips.
    pub fn pairing(&self) -> usize {
        self.biphase.selected
    }

    /// The current estimate of the samples per chip of the timing loop (at the decimated
    /// rate).
    pub fn sps_estimate(&self) -> f64 {
        self.sync.sps_estimate()
    }

    /// Writes waiting soft bits to `output`; returns how many.
    fn flush(&mut self, output: &mut [f32]) -> Result<usize> {
        let waiting = head(tail(self.pending.as_ref(), self.at)?, self.len - self.at)?;
        let k = waiting.len().min(output.len());
        head_mut(output, k)?.copy_from_slice(head(waiting, k)?);
        self.at += k;
        if self.at == self.len {
            self.at = 0;
            self.len = 0;
        }
        Ok(k)
    }

    /// One piece of input: the soft bits go to `pending[..len]`.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    fn piece(&mut self, piece: &[f32]) -> Result<()> {
        let len = piece.len();
        let complex = head_mut(self.a.as_mut(), len)?;
        <Sdrk as Interleave<C32>>::interleave(&Sdrk, piece, None, complex)?;
        <Sdrk as Rotate<C32>>::rotate(
            &Sdrk,
            &mut self.nco,
            head(self.a.as_ref(), len)?,
            head_mut(self.b.as_mut(), len)?,
        )?;
        let cap = dec_len(self.d).min(RDS_DEMOD_CHUNK);
        let dec = <Sdrk as FirDecimate<C32>>::fir_decimate(
            &Sdrk,
            &mut self.fir,
            head(self.b.as_ref(), len)?,
            head_mut(self.a.as_mut(), cap)?,
        )?;
        if dec.consumed != len {
            return Err(Error::Core(sdrk_core::Error::Internal(
                "rds_demod channel filter did not take the whole piece",
            )));
        }
        let nd = dec.produced;
        let chips = <Sdrk as SymbolSync<C32>>::symbol_sync(
            &Sdrk,
            &mut self.sync,
            head(self.a.as_ref(), nd)?,
            head_mut(self.b.as_mut(), cap)?,
        )?;
        if chips.consumed != nd {
            return Err(Error::Core(sdrk_core::Error::Internal(
                "rds_demod timing loop did not take the whole piece",
            )));
        }
        let nc = chips.produced;
        <Sdrk as Costas>::costas(
            &Sdrk,
            &mut self.costas,
            head(self.b.as_ref(), nc)?,
            head_mut(self.a.as_mut(), nc)?,
        )?;
        let mut slots = self.pending.as_mut().iter_mut();
        let mut n = 0usize;
        for c in head(self.a.as_ref(), nc)? {
            if let Some(soft) = self.biphase.push(c.re) {
                // At most one soft bit per chip, and `pending` holds one per decimated sample.
                let Some(slot) = slots.next() else {
                    return Err(Error::Core(sdrk_core::Error::Internal(
                        "rds_demod soft-bit buffer too small",
                    )));
                };
                *slot = soft;
                n += 1;
            }
        }
        self.at = 0;
        self.len = n;
        Ok(())
    }
}

impl<S: Storage> Block for RdsDemodBlock<S> {
    type In = f32;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<Progress> {
        let mut produced = self.flush(output)?;
        let mut consumed = 0usize;
        while self.len == 0 && produced < output.len() {
            let m = (input.len() - consumed).min(RDS_DEMOD_CHUNK);
            if m == 0 {
                break;
            }
            self.piece(head(tail(input, consumed)?, m)?)?;
            consumed += m;
            produced += self.flush(tail_mut(output, produced)?)?;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let taps = self.fir.taps().len() as u64;
        BlockTiming {
            ratio: self.ratio,
            kind: RatioKind::Variable,
            delay: Delay::new(taps - 1 + 2 * self.d as u64, 2).unwrap_or(Delay::ZERO),
        }
    }

    fn reset(&mut self) {
        self.nco.reset();
        self.fir.reset();
        self.sync.reset();
        self.costas.reset();
        self.biphase = Biphase::default();
        self.at = 0;
        self.len = 0;
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
