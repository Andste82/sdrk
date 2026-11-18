//! `qam_frame_decode`: square QAM baseband (4-QAM to 1024-QAM) to the packets of the test
//! framing, the composite `qam_demod` -> `bit_descramble` -> `frame_decode` with a watch at frame
//! level (M44.1 §7; the descrambler since M44.2, the maintainer's decision on I44.2-6).

use sdrk::{Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;
use sdrk_proto::frame::frame_bits;
use sdrk_proto::{Packet, PacketFlags};

use super::bit_descramble::BitDescrambleBlock;
use super::bit_scramble::{BitScrambleConfig, POLYNOMIAL_PARAM, SCRAMBLER_PARAM, Scrambler};
use super::frame_decode::{
    DECODE_WHITENING_PARAM, EMIT_BAD_PARAM, FrameDecodeBlock, FrameDecodeConfig, FrameRotations,
    MAX_SYNC_ERRORS_PARAM,
};
use super::pulse_match::{MATCH_ROLLOFF_PARAM, MATCH_SPAN_PARAM, SYMBOL_RATE_PARAM};
use super::qam_demod::{
    CARRIER_BW_PARAM, EQUALIZER_TAPS_PARAM, QAM_DEMOD_CHUNK, QamDemodBlock, QamDemodConfig,
    SYNC_BW_PARAM,
};
use super::qam_mod::QAM_CONSTELLATION_PARAM;
use super::{check_count, head, head_mut, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue, check_runtime};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::lock_watch::FrameWatch;
use crate::param_map::{
    ParamMap, Stages, check_composite_param, composite_param, composite_set_param, stage_of,
};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// The default `frame_timeout` of `qam_frame_decode` in symbols: four checks of `qam_demod`
/// (D1575).
pub const QAM_FRAME_TIMEOUT: usize = 16_384;

/// The largest `frame_timeout` in symbols.
const MAX_FRAME_TIMEOUT: usize = 1 << 20;

/// Symbols after the end of `frame_timeout` at which the frame watch starts the demodulator
/// again: the most symbols the loops of `qam_demod` can run ahead of the frame decoder (one
/// piece of [`QAM_DEMOD_CHUNK`] input samples yields at most about 129 symbols at 2 samples per
/// symbol), so that every frame that ends within the timeout is decoded before the restart,
/// however the input is cut (D1576).
pub const QAM_FRAME_WATCH_LAG: u64 = QAM_DEMOD_CHUNK as u64;

/// The `frame_timeout` parameter.
pub(crate) const FRAME_TIMEOUT_PARAM: ParamDesc = ParamDesc {
    name: "frame_timeout",
    unit: Unit::None,
    kind: ParamKind::I64 {
        min: 0,
        max: MAX_FRAME_TIMEOUT as i64,
        default: Some(QAM_FRAME_TIMEOUT as i64),
    },
    typical: "16384 (four checks of the demodulator); above the longest pause of the sender \
        between frames; 0 for off",
    mutability: Mutability::Build,
    description: "Symbols without a frame with CRC_OK, after one since the start, a reset or the \
        last restart by the watch, after which the demodulator starts again (the equalizer, the \
        loops at frequency 0 and a new search of the acquisition); 0 turns the watch off.",
};

/// The flat parameter names of `qam_frame_decode` and the stages they stand for: `0` is
/// `qam_demod`, `1` is `frame_decode` (its `rotations` are `four`, fixed by the composite; its
/// bits per symbol follow from its `constellation`), `2` the frame watch (`FrameWatch`, a
/// crate-internal building block), `3` is `bit_descramble` (left out for `scrambler` `none`; its
/// values are then those of the configuration) and `4` the structure of the composite, the
/// choice `scrambler` (D1823). `constellation` reaches `qam_demod`, `frame_decode` and
/// `bit_descramble`.
pub const QAM_FRAME_DECODE_PARAMS: ParamMap = ParamMap::new(&[
    ("constellation", "0.constellation"),
    ("constellation", "1.constellation"),
    ("constellation", "3.constellation"),
    ("symbol_rate_hz", "0.symbol_rate_hz"),
    ("rolloff", "0.rolloff"),
    ("span", "0.span"),
    ("equalizer_taps", "0.equalizer_taps"),
    ("sync_bw", "0.sync_bw"),
    ("carrier_bw", "0.carrier_bw"),
    ("max_sync_errors", "1.max_sync_errors"),
    ("emit_bad", "1.emit_bad"),
    ("whitening", "1.whitening"),
    ("frame_timeout", "2.frame_timeout"),
    ("scrambler", "4.scrambler"),
    ("polynomial", "3.polynomial"),
]);

/// Description of the `qam_frame_decode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "qam_frame_decode",
    summary: "Demodulates square QAM (4-QAM to 1024-QAM) and decodes the packets of the test \
        framing, with a watch at frame level.",
    details: "The receive side of qam_frame_encode, a composite of qam_demod (matched filter, \
        symbol timing, frequency acquisition, equalizer, carrier loop and soft bits, with its \
        parameters), the self-synchronizing descrambler bit_descramble (scrambler \
        multiplicative, the default, with polynomial and the constellation; none leaves it \
        out; it locks by itself after a symbols of the polynomial, and a quarter turn of the \
        constellation commutes with it) and frame_decode with rotations four and the same \
        constellation (2, 4, 6, 8 or 10 bits per symbol), which resolves the 90 degree ambiguity of the square constellation and reports the \
        quarter turns in bits 16 and 17 of the packet flags (max_sync_errors, emit_bad and \
        whitening as frame_decode). The watch at frame level: after a first frame with CRC_OK \
        since the start, a reset or its last restart, frames are expected, and when no frame \
        with CRC_OK ends within frame_timeout symbols of the last one, the demodulator starts \
        again 256 symbols later (the equalizer from its start weights, the loops at frequency \
        0 and a new search of the acquisition) and the watch waits for the next frame with \
        CRC_OK; frame_timeout 0 turns it off, and the block then gives the packets of the pair \
        qam_demod -> bit_descramble -> frame_decode (qam_demod -> frame_decode with scrambler \
        none). The packet index is the input sample of the first symbol of \
        the sync word at the nominal symbol rate after the delay of the demodulator, counted \
        since construction or reset. Parameters are the flat names of the three blocks, \
        frame_timeout and scrambler.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::C32],
        description: "Complex baseband, an integer number (2 or more) of samples per symbol, \
            unit average power.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "The decoded packets with their rotation in the flags; with emit_bad also \
            the frames whose CRC does not match.",
    }],
    params: &[
        QAM_CONSTELLATION_PARAM,
        SYMBOL_RATE_PARAM,
        MATCH_ROLLOFF_PARAM,
        MATCH_SPAN_PARAM,
        EQUALIZER_TAPS_PARAM,
        SYNC_BW_PARAM,
        CARRIER_BW_PARAM,
        MAX_SYNC_ERRORS_PARAM,
        EMIT_BAD_PARAM,
        DECODE_WHITENING_PARAM,
        FRAME_TIMEOUT_PARAM,
        SCRAMBLER_PARAM,
        POLYNOMIAL_PARAM,
    ],
    required: &["symbol_rate_hz"],
};

/// Configuration of [`QamFrameDecodeBlock`]: the demodulator, the decoder (its `rotations` are
/// `four` and its `constellation` that of the demodulator) and the frame watch.
///
/// ```
/// use sdrk_blocks::{ParamValue, QamFrameDecodeConfig};
/// let mut cfg = QamFrameDecodeConfig::default();
/// cfg.set("symbol_rate_hz", ParamValue::F64(25_000.0))?;
/// cfg.set("max_sync_errors", ParamValue::I64(2))?;
/// cfg.set("frame_timeout", ParamValue::I64(0))?;
/// assert_eq!((cfg.demod.symbol_rate_hz, cfg.frame.max_sync_errors), (25_000.0, 2));
/// assert_eq!(cfg.get("frame_timeout")?, ParamValue::I64(0));
/// assert!(cfg.validate(sdrk::Rate::hz(200_000)?).is_ok());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QamFrameDecodeConfig {
    /// The demodulator.
    pub demod: QamDemodConfig,
    /// The decoder: `max_sync_errors`, `emit_bad` and `whitening` (`rotations` is replaced by
    /// `four` and `constellation` by the one field `demod.constellation`).
    pub frame: FrameDecodeConfig,
    /// Symbols without a frame with `CRC_OK` after which the demodulator starts again (0 to
    /// 1048576; 0 = off).
    pub frame_timeout: usize,
    /// Whether the stage `bit_descramble` is there (`scrambler`, default `multiplicative`).
    pub scrambler: Scrambler,
    /// The descrambler: `polynomial` (its `constellation` is replaced by that of the
    /// demodulator, [`QamFrameDecodeConfig::descramble`]).
    pub scramble: BitScrambleConfig,
}

impl Default for QamFrameDecodeConfig {
    /// The defaults of `qam_demod` and `frame_decode`, `frame_timeout` [`QAM_FRAME_TIMEOUT`], the
    /// descrambler on with its default polynomial over the constellation of `qam_demod`.
    fn default() -> Self {
        let demod = QamDemodConfig::default();
        Self {
            demod,
            frame: FrameDecodeConfig::default(),
            frame_timeout: QAM_FRAME_TIMEOUT,
            scrambler: Scrambler::default(),
            scramble: BitScrambleConfig {
                constellation: demod.constellation,
                ..BitScrambleConfig::default()
            },
        }
    }
}

impl QamFrameDecodeConfig {
    /// The configuration of the descrambler: the `polynomial`, the constellation of the
    /// demodulator.
    pub fn descramble(&self) -> BitScrambleConfig {
        BitScrambleConfig {
            constellation: self.demod.constellation,
            ..self.scramble
        }
    }

    /// The configuration of the decoder: `rotations` `four`, the constellation of the
    /// demodulator.
    pub fn decoder(&self) -> FrameDecodeConfig {
        FrameDecodeConfig {
            rotations: FrameRotations::Four,
            constellation: self.demod.constellation,
            ..self.frame
        }
    }

    /// Checks both stages and `frame_timeout` against the input sample rate.
    ///
    /// # Errors
    /// As [`QamDemodConfig::validate`] and [`FrameDecodeConfig::validate`]; `OutOfRange` for
    /// `frame_timeout` above 1048576.
    pub fn validate(&self, rate_in: Rate) -> Result<()> {
        self.demod.validate(rate_in)?;
        self.decoder().validate()?;
        check_count("frame_timeout", self.frame_timeout, 0, MAX_FRAME_TIMEOUT)
    }

    /// Sets a parameter by its flat name ([`QAM_FRAME_DECODE_PARAMS`]);
    /// checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` as
    /// [`QamDemodConfig::set`].
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        // Every path on a copy, kept only if all of them took the value (all or none, §4).
        let mut next = *self;
        let mut any = false;
        for path in QAM_FRAME_DECODE_PARAMS.paths(name) {
            match path.split_once('.') {
                Some(("0", n)) => next.demod.set(n, value)?,
                Some(("1", n)) => next.frame.set(n, value)?,
                Some(("2", _)) => {
                    FRAME_TIMEOUT_PARAM.check(&value)?;
                    next.frame_timeout = value.as_usize()?;
                }
                Some(("3", n)) => next.scramble.set(n, value)?,
                Some(("4", _)) => {
                    SCRAMBLER_PARAM.check(&value)?;
                    next.scrambler = Scrambler::from_index(value.as_choice()?);
                }
                _ => return unknown(),
            }
            any = true;
        }
        if !any {
            return unknown();
        }
        *self = next;
        Ok(())
    }

    /// Current value of a parameter by its flat name.
    ///
    /// # Errors
    /// `UnknownParameter`.
    pub fn get(&self, name: &str) -> Result<ParamValue> {
        match QAM_FRAME_DECODE_PARAMS
            .path(name)
            .and_then(|p| p.split_once('.'))
        {
            Some(("0", n)) => self.demod.get(n),
            Some(("1", n)) => self.frame.get(n),
            Some(("2", _)) => Ok(ParamValue::I64(self.frame_timeout as i64)),
            Some(("3", n)) => self.descramble().get(n),
            Some(("4", _)) => Ok(ParamValue::Choice(self.scrambler.index())),
            _ => unknown(),
        }
    }
}

/// The soft bits of the buffer between the stages: the symbols of one piece of `qam_demod`.
fn buffer_len(bps: usize) -> usize {
    QAM_DEMOD_CHUNK * bps
}

/// Square QAM receiver (`qpsk` to `qam1024`) of the test framing: `C32` in, [`Packets`] out (M44.1 §7).
///
/// A composite block: [`QamDemodBlock`] writes the soft bits of each piece of at most
/// [`QAM_DEMOD_CHUNK`] input samples into a buffer, [`BitDescrambleBlock`] descrambles them there
/// in place (for `scrambler` `multiplicative`; with the constellation of the demodulator, so that
/// a quarter turn commutes with it), and [`FrameDecodeBlock`] (`rotations` `four`, the same
/// constellation) reads them before the next piece; a packet output that is full
/// keeps the rest of the buffer for the next call. The watch at frame level ([`FrameWatch`]):
/// after a frame with `CRC_OK` (since the start, a reset or its last restart) frames are
/// expected; when none ends within `frame_timeout` symbols after the last one, the demodulator
/// starts again at the symbol [`QAM_FRAME_WATCH_LAG`] after the end of the timeout
/// ([`QamDemodBlock::restart_at`], the transition `Silence` of its controller, the action
/// `RestartAll`), at the same symbol however the input is cut. `frame_timeout` 0 turns the watch
/// off: the packets are then those of the chain `qam_demod` -> `bit_descramble` -> `frame_decode`
/// (without `bit_descramble` for `scrambler` `none`). The packet index is
/// the input sample `floor(k n + d)` of the first symbol `k` of the sync word, with `n` the exact
/// samples per symbol and `d` the delay of the demodulator. Timing: packets, at most one per 56
/// soft bits at the nominal symbol rate; the delay of `qam_demod`. Storage: that of `qam_demod`,
/// the ring of `bit_descramble` (`23 bits` `f32` for the default polynomial) and the buffer of
/// `256 bits` `f32`.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{
///     ArenaStorage, Block, Packet, QamFrameDecodeBlock, QamFrameDecodeConfig,
///     QamFrameEncodeBlock, QamFrameEncodeConfig, FrameIdle,
/// };
/// use sdrk_proto::PacketFlags;
/// let mut tx_cfg = QamFrameEncodeConfig::default();
/// tx_cfg.qam.samples_per_symbol = 4;
/// tx_cfg.frame.idle = FrameIdle::Pn9;
/// let mut rx_cfg = QamFrameDecodeConfig::default();
/// rx_cfg.demod.symbol_rate_hz = 1000.0;
/// let mut mem = vec![0u8; 1 << 17];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut tx = QamFrameEncodeBlock::new(&tx_cfg, Rate::hz(4000)?, &mut arena)?;
/// let mut rx = QamFrameDecodeBlock::new(&rx_cfg, Rate::hz(4000)?, &mut arena)?;
/// // idle symbols for the acquisition and the equalizer, then three packets
/// let mut sig = vec![C32::default(); 40_000];
/// tx.process(&[], &mut sig[..32_000])?;
/// let sent: Vec<Packet> = (0..3u8).map(|k| Packet::new(0, PacketFlags::NONE, &[k; 20])).collect::<Result<_, _>>()?;
/// let p = tx.process(&sent, &mut sig[32_000..])?;
/// assert_eq!(p.consumed, 3);
/// tx.process(&[], &mut sig[32_000 + p.produced..])?;
/// let mut packets = [Packet::default(); 8];
/// let q = rx.process(&sig, &mut packets)?;
/// assert_eq!(q.consumed, sig.len());
/// let ok: Vec<&Packet> = packets[..q.produced].iter().filter(|p| p.flags.contains(PacketFlags::CRC_OK)).collect();
/// assert_eq!(ok.len(), 3);
/// assert_eq!(ok[2].data(), &[2u8; 20]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QamFrameDecodeBlock<S: Storage> {
    demod: QamDemodBlock<S>,
    /// The descrambler (`None` for `scrambler` `none`).
    descramble: Option<BitDescrambleBlock<S>>,
    /// The configuration of the descrambler, also when it is left out (its parameters).
    scramble_cfg: BitScrambleConfig,
    decode: FrameDecodeBlock<S>,
    /// The soft bits between the stages, `bits[at..len]` not decoded yet.
    bits: S::Buf<f32>,
    at: u16,
    len: u16,
    watch: FrameWatch,
    /// The exact input samples per symbol and the delay of the demodulator in input samples (the
    /// packet index).
    sps: f64,
    delay: f64,
}

impl<S: Storage> core::fmt::Debug for QamFrameDecodeBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QamFrameDecodeBlock")
            .field("demod", &self.demod)
            .field("watch", &self.watch)
            .finish_non_exhaustive()
    }
}

impl<S: Storage> QamFrameDecodeBlock<S> {
    /// Creates the block: `qam_demod`, then the buffer between the stages, from `storage`.
    ///
    /// # Errors
    /// As [`QamFrameDecodeConfig::validate`]; `Storage` when the storage is too small.
    pub fn new(config: &QamFrameDecodeConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate(rate_in)?;
        let demod = QamDemodBlock::new(&config.demod, rate_in, storage)?;
        let scramble_cfg = config.descramble();
        let descramble = match config.scrambler {
            Scrambler::Multiplicative => {
                Some(BitDescrambleBlock::new(&scramble_cfg, rate_in, storage)?)
            }
            Scrambler::Off => None,
        };
        let decode = FrameDecodeBlock::new(&config.decoder(), rate_in, storage)?;
        let bits = storage.alloc::<f32>(buffer_len(config.demod.bits_per_symbol()))?;
        let d = demod.timing().delay;
        Ok(Self {
            delay: d.num() as f64 / d.den() as f64,
            sps: rate_in.as_f64() / config.demod.symbol_rate_hz,
            demod,
            descramble,
            scramble_cfg,
            decode,
            bits,
            at: 0,
            len: 0,
            watch: FrameWatch::new(u32::try_from(config.frame_timeout).unwrap_or(u32::MAX)),
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`QamFrameDecodeConfig::validate`].
    pub fn storage_bytes(config: &QamFrameDecodeConfig, rate_in: Rate) -> Result<usize> {
        config.validate(rate_in)?;
        // `frame_decode` needs no storage; `qam_demod`, then the ring of `bit_descramble`.
        let mut plan = Plan::default();
        QamDemodBlock::<S>::plan(&config.demod, rate_in, &mut plan)?;
        if config.scrambler == Scrambler::Multiplicative {
            plan.add::<f32>(config.descramble().delays().0)?;
        }
        plan.add::<f32>(buffer_len(config.demod.bits_per_symbol()))?;
        Ok(plan.bytes())
    }

    /// The configuration of the block.
    pub fn config(&self) -> QamFrameDecodeConfig {
        QamFrameDecodeConfig {
            demod: self.demod.config(),
            frame: self.decode.config(),
            frame_timeout: self.watch.timeout() as usize,
            scrambler: if self.descramble.is_some() {
                Scrambler::Multiplicative
            } else {
                Scrambler::Off
            },
            scramble: self.scramble_cfg,
        }
    }

    /// The descrambler stage (`None` for `scrambler` `none`).
    pub fn descrambler(&self) -> Option<&BitDescrambleBlock<S>> {
        self.descramble.as_ref()
    }

    /// The demodulator stage (its state, lock and restarts).
    pub fn demodulator(&self) -> &QamDemodBlock<S> {
        &self.demod
    }

    /// The demodulator stage with its control interface, for tests and experiments that put the
    /// receiver into a state.
    pub fn demodulator_mut(&mut self) -> &mut QamDemodBlock<S> {
        &mut self.demod
    }

    /// The decoder stage.
    pub fn frame_decoder(&self) -> &FrameDecodeBlock<S> {
        &self.decode
    }

    /// The watch at frame level.
    pub fn frame_watch(&self) -> &FrameWatch {
        &self.watch
    }

    /// A packet of the decoder: a frame with `CRC_OK` goes to the watch (the symbol of its last
    /// bit), and the index moves from the soft bits of the decoder to the input samples.
    fn packet_out(&mut self, packet: &mut Packet) {
        let bps = self.demod.config().bits_per_symbol() as u64;
        let bit = packet.index;
        if packet.flags.contains(PacketFlags::CRC_OK) {
            let bits = frame_bits(packet.len(), 0) as u64;
            self.watch
                .frame_ok(bit.saturating_add(bits).saturating_sub(1) / bps);
        }
        packet.index = ((bit / bps) as f64 * self.sps + self.delay) as u64;
    }
}

impl<S: Storage> Block for QamFrameDecodeBlock<S> {
    type In = C32;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[C32], output: &mut [Packet]) -> Result<Progress> {
        let mut consumed = 0;
        let mut produced = 0;
        loop {
            // 1. The waiting soft bits into the decoder.
            if self.at < self.len {
                let room = tail_mut(output, produced)?;
                if room.is_empty() {
                    break;
                }
                let (at, len) = (usize::from(self.at), usize::from(self.len));
                let waiting = head(tail(self.bits.as_ref(), at)?, len - at)?;
                let p = self.decode.process(waiting, room)?;
                // `p.consumed <= len - at`.
                self.at += p.consumed as u16;
                for packet in head_mut(room, p.produced)? {
                    self.packet_out(packet);
                }
                produced += p.produced;
                if self.at < self.len {
                    break; // the output is full
                }
            }
            // 2. The next piece through the demodulator, with the restart of the watch.
            let m = (input.len() - consumed).min(QAM_DEMOD_CHUNK);
            if m == 0 {
                break;
            }
            let restart = self
                .watch
                .deadline()
                .map(|d| d.saturating_add(QAM_FRAME_WATCH_LAG));
            self.demod.restart_at(restart);
            let q = self
                .demod
                .process(head(tail(input, consumed)?, m)?, self.bits.as_mut())?;
            if let Some(r) = restart
                && self.demod.symbols() >= r
            {
                self.watch.restarted(r);
            }
            if q.consumed != m {
                return Err(Error::Core(sdrk_core::Error::Internal(
                    "qam_frame_decode demodulator did not take the whole piece",
                )));
            }
            consumed += m;
            if let Some(d) = self.descramble.as_mut() {
                d.process_in_place(head_mut(self.bits.as_mut(), q.produced)?);
            }
            self.at = 0;
            self.len = u16::try_from(q.produced)
                .map_err(|_| Error::Core(sdrk_core::Error::Internal("qam_frame_decode count")))?;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        let demod = self.demod.timing();
        let per_input = demod
            .ratio
            .mul(self.decode.timing().ratio)
            .unwrap_or(Ratio::ONE);
        BlockTiming {
            ratio: per_input,
            kind: RatioKind::Packets {
                max_per_input: per_input,
            },
            delay: demod.delay,
        }
    }

    fn reset(&mut self) {
        self.demod.reset();
        if let Some(d) = self.descramble.as_mut() {
            d.reset();
        }
        self.decode.reset();
        self.at = 0;
        self.len = 0;
        self.watch.reset();
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        composite_set_param(self, name, value)
    }

    fn param(&self, name: &str) -> Result<ParamValue> {
        composite_param(self, name)
    }

    fn check_param(&self, name: &str, value: &ParamValue) -> Result<()> {
        check_composite_param(self, name, value)
    }

    fn min_input(&self) -> usize {
        self.demod.min_input()
    }
}

const _: () = assert!(QAM_DEMOD_CHUNK * 6 <= u16::MAX as usize);

impl<S: Storage> Stages for QamFrameDecodeBlock<S> {
    const PARAMS: ParamMap = QAM_FRAME_DECODE_PARAMS;

    fn stage_param(&self, path: &str) -> Result<ParamValue> {
        match stage_of(path)? {
            ("0", n) => self.demod.param(n),
            ("1", n) => self.decode.param(n),
            ("2", _) => Ok(ParamValue::I64(i64::from(self.watch.timeout()))),
            ("3", n) => match self.descramble.as_ref() {
                Some(d) => d.param(n),
                None => self.scramble_cfg.get(n),
            },
            ("4", "scrambler") => Ok(ParamValue::Choice(self.config().scrambler.index())),
            _ => unknown(),
        }
    }

    fn check_stage_param(&self, path: &str, value: &ParamValue) -> Result<()> {
        match stage_of(path)? {
            ("0", n) => self.demod.check_param(n, value),
            ("1", n) => self.decode.check_param(n, value),
            ("2", _) => check_runtime(&FRAME_TIMEOUT_PARAM, value),
            ("3", n) => check_runtime(
                super::bit_descramble::DESCRIPTION
                    .param(n)
                    .ok_or(Error::UnknownParameter)?,
                value,
            ),
            ("4", "scrambler") => check_runtime(&SCRAMBLER_PARAM, value),
            _ => unknown(),
        }
    }

    fn set_stage_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        match stage_of(path)? {
            ("0", n) => self.demod.set_param(n, value),
            ("1", n) => self.decode.set_param(n, value),
            ("2" | "3" | "4", _) => Err(Error::ReadOnlyParameter),
            _ => unknown(),
        }
    }
}
