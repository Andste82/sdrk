//! `qam_frame_encode`: packets to square QAM baseband (4-QAM to 1024-QAM), the composite
//! `frame_encode` -> `bit_scramble` -> `qam_mod` (M44.1 §7; the scrambler since M44.2, the
//! maintainer's decision on I44.2-6).

use sdrk::{Progress, Rate, Ratio, RatioKind};
use sdrk_core::C32;
use sdrk_proto::Packet;
use sdrk_proto::frame::{MAX_PAYLOAD, frame_bits};

use super::bit_scramble::{
    BitScrambleBlock, BitScrambleConfig, POLYNOMIAL_PARAM, SCRAMBLER_PARAM, Scrambler,
};
use super::frame_encode::{
    ENCODE_WHITENING_PARAM, FrameEncodeBlock, FrameEncodeConfig, IDLE_PARAM, PREAMBLE_BYTES_PARAM,
};
use super::pulse_shape::{ROLLOFF_PARAM, SAMPLES_PER_SYMBOL_PARAM, SPAN_PARAM};
use super::qam_mod::{QAM_CONSTELLATION_PARAM, QAM_MOD_CHUNK, QamModBlock, QamModConfig};
use super::{head, head_mut, tail, tail_mut, unknown};
use crate::block::{Block, BlockTiming, ParamValue, check_runtime};
use crate::describe::{BlockDescription, PortDesc, PortType};
use crate::error::{Error, Result};
use crate::param_map::{
    ParamMap, Stages, check_composite_param, composite_param, composite_set_param, stage_of,
};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// The flat parameter names of `qam_frame_encode` and the stages they stand for: `0` is
/// `frame_encode`, `1` is `qam_mod`, `2` is `bit_scramble` (left out for `scrambler` `none`; its
/// values are then those of the configuration) and `3` the structure of the composite, the
/// choice `scrambler` (D1823). `constellation` reaches `qam_mod` and `bit_scramble`.
pub const QAM_FRAME_ENCODE_PARAMS: ParamMap = ParamMap::new(&[
    ("preamble_bytes", "0.preamble_bytes"),
    ("idle", "0.idle"),
    ("whitening", "0.whitening"),
    ("constellation", "1.constellation"),
    ("constellation", "2.constellation"),
    ("samples_per_symbol", "1.samples_per_symbol"),
    ("rolloff", "1.rolloff"),
    ("span", "1.span"),
    ("scrambler", "3.scrambler"),
    ("polynomial", "2.polynomial"),
]);

/// Description of the `qam_frame_encode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "qam_frame_encode",
    summary: "Frames packets, scrambles their bits and modulates them as square QAM (4-QAM to \
        1024-QAM).",
    details: "The transmit side of qam_frame_decode, a composite of frame_encode, bit_scramble \
        and qam_mod: every packet is sent as one frame of the test framing (preamble_bytes \
        bytes 0x55, the sync word 0x1ACFFC1D, length, payload, CRC-16/CCITT-FALSE; whitening \
        pn9 XORs length, payload and CRC with the PN9 sequence), the whole bit stream (frames \
        and idle signal) goes through the self-synchronizing scrambler of bit_scramble \
        (scrambler multiplicative, the default, with polynomial and the constellation; none \
        leaves it out), and its bits are mapped to the square QAM points of qam_mod \
        (constellation qpsk, qam16, qam64, qam256 or qam1024, the first bit the most \
        significant) and interpolated by samples_per_symbol with the root-raised-cosine pulse \
        (rolloff, span). Between packets the block sends the idle signal of frame_encode (idle \
        zero, preamble or pn9; pn9 keeps the timing loop and the blind equalizer of the \
        receiver training). The scrambler makes the fixed preamble, sync word and idle pattern \
        look random on the air, which the blind equalizer of qam256 and qam1024 needs. The \
        output equals the chain frame_encode -> bit_scramble -> qam_mod (frame_encode -> \
        qam_mod with scrambler none) bit for bit. It is a source: the \
        output rate is the rate_hz of the packet input (the bit rate) times \
        samples_per_symbol / bits per symbol. The bits of a frame that do not fill a whole \
        symbol wait for the next bits (the next frame or the idle signal). Parameters are the \
        flat names of the three blocks and scrambler.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Packets],
        description: "The packets to send (at most 253 bytes each); index and flags are ignored.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::C32],
        description: "The shaped baseband signal, samples_per_symbol samples per symbol, unit \
            average power at the symbol centres.",
    }],
    params: &[
        PREAMBLE_BYTES_PARAM,
        IDLE_PARAM,
        ENCODE_WHITENING_PARAM,
        QAM_CONSTELLATION_PARAM,
        SAMPLES_PER_SYMBOL_PARAM,
        ROLLOFF_PARAM,
        SPAN_PARAM,
        SCRAMBLER_PARAM,
        POLYNOMIAL_PARAM,
    ],
    required: &["samples_per_symbol"],
};

/// Configuration of [`QamFrameEncodeBlock`]: the configurations of its stages and the scrambler
/// choice.
///
/// ```
/// use sdrk_blocks::{ParamValue, QamFrameEncodeConfig, SymbolMapConstellation};
/// let mut cfg = QamFrameEncodeConfig::default();
/// cfg.set("constellation", ParamValue::Choice(4))?; // qam64
/// cfg.set("preamble_bytes", ParamValue::I64(8))?;
/// assert_eq!(cfg.qam.constellation, SymbolMapConstellation::Qam64);
/// assert_eq!(cfg.frame.preamble_bytes, 8);
/// assert_eq!(cfg.get("constellation")?, ParamValue::Choice(4));
/// assert!(cfg.validate().is_ok());
/// # Ok::<(), sdrk_blocks::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QamFrameEncodeConfig {
    /// The framing (`preamble_bytes`, `idle`, `whitening`).
    pub frame: FrameEncodeConfig,
    /// The modulator (`constellation`, `samples_per_symbol`, `rolloff`, `span`).
    pub qam: QamModConfig,
    /// Whether the stage `bit_scramble` is there (`scrambler`, default `multiplicative`).
    pub scrambler: Scrambler,
    /// The scrambler: `polynomial` (its `constellation` is replaced by that of `qam`,
    /// [`QamFrameEncodeConfig::scramble`]).
    pub scramble: BitScrambleConfig,
}

impl Default for QamFrameEncodeConfig {
    /// The defaults of `frame_encode` and `qam_mod`, the scrambler on with its default
    /// polynomial over the constellation of `qam_mod`.
    fn default() -> Self {
        let qam = QamModConfig::default();
        Self {
            frame: FrameEncodeConfig::default(),
            qam,
            scrambler: Scrambler::default(),
            scramble: BitScrambleConfig {
                constellation: qam.constellation,
                ..BitScrambleConfig::default()
            },
        }
    }
}

impl QamFrameEncodeConfig {
    /// The configuration of the scrambler: the `polynomial`, the constellation of the
    /// modulator.
    pub fn scramble(&self) -> BitScrambleConfig {
        BitScrambleConfig {
            constellation: self.qam.constellation,
            ..self.scramble
        }
    }

    /// Checks both stages.
    ///
    /// # Errors
    /// As [`FrameEncodeConfig::validate`] and [`QamModConfig::validate`].
    pub fn validate(&self) -> Result<()> {
        self.frame.validate()?;
        self.qam.validate()
    }

    /// Sets a parameter by its flat name ([`QAM_FRAME_ENCODE_PARAMS`]); checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`; `InvalidParameter` for a
    /// constellation that is not a square QAM (`bpsk`, `8psk`).
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        // Every path on a copy, kept only if all of them took the value (all or none, §4).
        let mut next = *self;
        let mut any = false;
        for path in QAM_FRAME_ENCODE_PARAMS.paths(name) {
            match path.split_once('.') {
                Some(("0", n)) => next.frame.set(n, value)?,
                Some(("1", n)) => next.qam.set(n, value)?,
                Some(("2", n)) => next.scramble.set(n, value)?,
                Some(("3", _)) => {
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
        match QAM_FRAME_ENCODE_PARAMS
            .path(name)
            .and_then(|p| p.split_once('.'))
        {
            Some(("0", n)) => self.frame.get(n),
            Some(("1", n)) => self.qam.get(n),
            Some(("2", n)) => self.scramble().get(n),
            Some(("3", _)) => Ok(ParamValue::Choice(self.scrambler.index())),
            _ => unknown(),
        }
    }
}

/// The soft bits of the buffer between the stages: `QAM_MOD_CHUNK` symbols.
fn buffer_len(bps: usize) -> usize {
    QAM_MOD_CHUNK * bps
}

/// Packets to square QAM baseband (`qpsk` to `qam1024`): [`Packets`] in, `C32` out at `samples_per_symbol`
/// samples per symbol (M44.1 §7).
///
/// A composite block: [`FrameEncodeBlock`] writes the bits of the frames (and the idle signal)
/// into a buffer of [`QAM_MOD_CHUNK`] symbols, [`BitScrambleBlock`] scrambles them there in place
/// as they are written (for `scrambler` `multiplicative`), and [`QamModBlock`] takes whole symbols;
/// the framing stage gets room for the bits of the whole symbols that still fit the output. The
/// output is that of the chain `frame_encode` -> `bit_scramble` -> `qam_mod` (without
/// `bit_scramble` for `scrambler` `none`), bit for bit, for the same packets and idle calls. A packet counts as consumed when its frame is written up to its last whole symbol
/// (as `frame_encode`, a frame that does not fit is continued by the next call, which must start
/// with the same packet); the bits of a symbol that it shares with what follows wait. A call with an
/// empty input is idle, as for `frame_encode`: the bits that wait are written, then idle bits for
/// the whole symbols that still fit the output. Timing: a source, at most
/// `ceil(frame_bits / bits) sps` output samples per packet; delay `span bits / 2` samples of the
/// bit rate (that of `qam_mod`). Storage: that of `qam_mod` and the buffer of `256 bits` `f32`.
///
/// ```
/// use sdrk::{Rate, C32};
/// use sdrk_blocks::{ArenaStorage, Block, Packet, QamFrameEncodeBlock, QamFrameEncodeConfig};
/// use sdrk_proto::PacketFlags;
/// let mut cfg = QamFrameEncodeConfig::default();
/// cfg.qam.samples_per_symbol = 4;
/// let rate = Rate::hz(4000)?; // bits per second
/// let mut mem = vec![0u8; QamFrameEncodeBlock::<ArenaStorage<'_>>::storage_bytes(&cfg, rate)? + 16];
/// let mut arena = ArenaStorage::new(&mut mem);
/// let mut tx = QamFrameEncodeBlock::new(&cfg, rate, &mut arena)?;
/// let p = Packet::new(0, PacketFlags::NONE, b"hi")?;
/// let mut out = [C32::default(); 256];
/// // 4 preamble bytes, sync word, length, 2 bytes, CRC: 104 bits, 26 symbols of 4 samples
/// let q = tx.process(&[p], &mut out)?;
/// assert_eq!((q.consumed, q.produced), (1, 104));
/// assert_eq!(QamFrameEncodeBlock::<ArenaStorage<'_>>::rate_out(&cfg, rate)?, Rate::hz(4000)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QamFrameEncodeBlock<S: Storage> {
    enc: FrameEncodeBlock<S>,
    /// The scrambler (`None` for `scrambler` `none`).
    scramble: Option<BitScrambleBlock<S>>,
    /// The configuration of the scrambler, also when it is left out (its parameters).
    scramble_cfg: BitScrambleConfig,
    qam: QamModBlock<S>,
    /// The bits between the stages, `bits[at..len]` waiting.
    bits: S::Buf<f32>,
    at: u16,
    len: u16,
    /// Bits per symbol and samples per symbol.
    bps: u8,
    sps: u8,
    /// The most output samples of one packet.
    max_samples: Ratio,
}

impl<S: Storage> core::fmt::Debug for QamFrameEncodeBlock<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QamFrameEncodeBlock")
            .field("config", &self.config())
            .field("waiting", &(self.len - self.at))
            .finish_non_exhaustive()
    }
}

impl<S: Storage> QamFrameEncodeBlock<S> {
    /// Creates the block: `qam_mod`, then the buffer between the stages, from `storage`.
    ///
    /// # Errors
    /// As [`QamFrameEncodeConfig::validate`]; `Storage` when the storage is too small.
    #[allow(
        clippy::large_stack_frames,
        reason = "clippy counts the temporary of every `?` separately; the real frame is small"
    )]
    pub fn new(config: &QamFrameEncodeConfig, rate_in: Rate, storage: &mut S) -> Result<Self> {
        config.validate()?;
        let bps = config.qam.bits_per_symbol();
        let sps = config.qam.samples_per_symbol;
        let enc = FrameEncodeBlock::new(&config.frame, rate_in, storage)?;
        let scramble_cfg = config.scramble();
        let scramble = match config.scrambler {
            Scrambler::Multiplicative => {
                Some(BitScrambleBlock::new(&scramble_cfg, rate_in, storage)?)
            }
            Scrambler::Off => None,
        };
        let qam = QamModBlock::new(&config.qam, rate_in, storage)?;
        let bits = storage.alloc::<f32>(buffer_len(bps))?;
        let max_bits = frame_bits(MAX_PAYLOAD, config.frame.preamble_bytes) + bps - 1;
        let narrow = |v: usize| {
            u8::try_from(v).map_err(|_| Error::Core(sdrk_core::Error::Internal("qam_frame_encode")))
        };
        Ok(Self {
            enc,
            scramble,
            scramble_cfg,
            qam,
            bits,
            at: 0,
            len: 0,
            bps: narrow(bps)?,
            sps: narrow(sps)?,
            max_samples: Ratio::new((max_bits / bps * sps) as u64, 1)?,
        })
    }

    /// Exact bytes of storage the block needs for an arena that starts at a 16-byte aligned
    /// address (an unaligned slice may need up to 15 bytes more).
    ///
    /// # Errors
    /// As [`QamFrameEncodeConfig::validate`].
    pub fn storage_bytes(config: &QamFrameEncodeConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        // `frame_encode` and `bit_scramble` need no storage.
        let mut plan = Plan::default();
        QamModBlock::<S>::plan(&config.qam, &mut plan)?;
        plan.add::<f32>(buffer_len(config.qam.bits_per_symbol()))?;
        Ok(plan.bytes())
    }

    /// Output rate: the bit rate (the `rate_hz` of the packet input) times
    /// `samples_per_symbol / bits_per_symbol`.
    ///
    /// # Errors
    /// As [`QamFrameEncodeConfig::validate`]; `OutOfRange` when the rate overflows.
    pub fn rate_out(config: &QamFrameEncodeConfig, rate_in: Rate) -> Result<Rate> {
        config.validate()?;
        QamModBlock::<S>::rate_out(&config.qam, rate_in)
    }

    /// The configuration of the block.
    pub fn config(&self) -> QamFrameEncodeConfig {
        QamFrameEncodeConfig {
            frame: self.enc.config(),
            qam: self.qam.config(),
            scrambler: if self.scramble.is_some() {
                Scrambler::Multiplicative
            } else {
                Scrambler::Off
            },
            scramble: self.scramble_cfg,
        }
    }

    /// The scrambler stage (`None` for `scrambler` `none`).
    pub fn scrambler(&self) -> Option<&BitScrambleBlock<S>> {
        self.scramble.as_ref()
    }

    /// The framing stage.
    pub fn frame_encoder(&self) -> &FrameEncodeBlock<S> {
        &self.enc
    }

    /// The modulator stage.
    pub fn modulator(&self) -> &QamModBlock<S> {
        &self.qam
    }

    /// Bits that wait in the buffer between the stages (fewer than one symbol after a call that
    /// had room for all of them).
    pub fn waiting(&self) -> usize {
        usize::from(self.len - self.at)
    }

    /// Writes the whole symbols of the waiting bits that fit `output`; returns the samples.
    fn drain(&mut self, output: &mut [C32]) -> Result<usize> {
        let (at, len) = (usize::from(self.at), usize::from(self.len));
        if len - at < usize::from(self.bps) {
            return Ok(0);
        }
        let waiting = head(tail(self.bits.as_ref(), at)?, len - at)?;
        let p = self.qam.process(waiting, output)?;
        // `p.consumed <= len - at`.
        self.at += p.consumed as u16;
        Ok(p.produced)
    }

    /// Moves the waiting bits to the front of the buffer.
    fn compact(&mut self) {
        let (at, len) = (usize::from(self.at), usize::from(self.len));
        if at > 0 {
            self.bits.as_mut().copy_within(at..len, 0);
            self.len -= self.at;
            self.at = 0;
        }
    }
}

impl<S: Storage> Block for QamFrameEncodeBlock<S> {
    type In = Packets;
    type Out = C32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[Packet], output: &mut [C32]) -> Result<Progress> {
        let (bps, sps) = (usize::from(self.bps), usize::from(self.sps));
        let idle = input.is_empty();
        let mut consumed = 0;
        let mut produced = 0;
        loop {
            produced += self.drain(tail_mut(output, produced)?)?;
            let room = output.len() - produced;
            if room < sps {
                break;
            }
            // Fewer than `bps` bits wait: the next bits from the framing stage.
            self.compact();
            let len = usize::from(self.len);
            let free = tail_mut(self.bits.as_mut(), len)?;
            // The bits of the whole symbols that still fit the output: a packet counts as
            // consumed only when its frame is written up to its last whole symbol (the bits of a
            // symbol it shares with the next frame or the idle signal wait), as for the pair.
            let want = ((room / sps) * bps).saturating_sub(len).min(free.len());
            let p = if idle {
                self.enc.process(&[], head_mut(free, want)?)?
            } else if consumed < input.len() {
                self.enc
                    .process(tail(input, consumed)?, head_mut(free, want)?)?
            } else {
                break; // every packet is in the buffer (an empty input would be idle)
            };
            consumed += p.consumed;
            if let Some(s) = self.scramble.as_mut() {
                s.process_in_place(head_mut(free, p.produced)?);
            }
            // `p.produced <= free.len()`.
            self.len += p.produced as u16;
            if p.produced == 0 {
                break;
            }
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.max_samples,
            kind: RatioKind::Source,
            // The delay of `qam_mod`, `span / 2` symbols in samples of the bit rate.
            delay: self.qam.timing().delay,
        }
    }

    fn reset(&mut self) {
        self.enc.reset();
        if let Some(s) = self.scramble.as_mut() {
            s.reset();
        }
        self.qam.reset();
        self.at = 0;
        self.len = 0;
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
        1
    }
}

impl<S: Storage> Stages for QamFrameEncodeBlock<S> {
    const PARAMS: ParamMap = QAM_FRAME_ENCODE_PARAMS;

    fn stage_param(&self, path: &str) -> Result<ParamValue> {
        match stage_of(path)? {
            ("0", n) => self.enc.param(n),
            ("1", n) => self.qam.param(n),
            ("2", n) => match self.scramble.as_ref() {
                Some(s) => s.param(n),
                None => self.scramble_cfg.get(n),
            },
            ("3", "scrambler") => Ok(ParamValue::Choice(self.config().scrambler.index())),
            _ => unknown(),
        }
    }

    fn check_stage_param(&self, path: &str, value: &ParamValue) -> Result<()> {
        match stage_of(path)? {
            ("0", n) => self.enc.check_param(n, value),
            ("1", n) => self.qam.check_param(n, value),
            ("2", n) => check_runtime(
                super::bit_scramble::DESCRIPTION
                    .param(n)
                    .ok_or(Error::UnknownParameter)?,
                value,
            ),
            ("3", "scrambler") => check_runtime(&SCRAMBLER_PARAM, value),
            _ => unknown(),
        }
    }

    fn set_stage_param(&mut self, path: &str, value: ParamValue) -> Result<()> {
        match stage_of(path)? {
            ("0", n) => self.enc.set_param(n, value),
            ("1", n) => self.qam.set_param(n, value),
            ("2" | "3", _) => Err(Error::ReadOnlyParameter),
            _ => unknown(),
        }
    }
}

const _: () = assert!(QAM_MOD_CHUNK * 6 <= u16::MAX as usize);
