//! `frame_decode`: soft bits to the packets of the test framing of `sdrk_proto::frame`.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_proto::Packet;
use sdrk_proto::frame::{
    Decoder, MAX_BITS_PER_SYMBOL, MAX_SYNC_ERRORS, RotatingDecoder, Whitening, frame_bits,
};

use super::{WHITENING_OPTIONS, check_count, unknown, whitening_choice, whitening_from_choice};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// Bits of the shortest frame without preamble: sync word, length byte and CRC (56).
pub(crate) const MIN_FRAME_BITS: usize = frame_bits(0, 0);

/// The names of the `rotations` choice, in index order.
const ROTATIONS_OPTIONS: &[&str] = &["normal", "both", "four"];

/// The largest `bits_per_symbol`.
const MAX_BPS: usize = MAX_BITS_PER_SYMBOL as usize;

/// The sync-word rotations a [`FrameDecodeBlock`] accepts (parameter `rotations`, M44; `Normal`
/// and `Both` are the former `polarity` of M40).
///
/// ```
/// use sdrk_blocks::FrameRotations;
/// assert_eq!(FrameRotations::default(), FrameRotations::Normal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum FrameRotations {
    /// Only the sync word itself.
    #[default]
    Normal,
    /// The sync word and its inverse; the bits of a frame found by the inverse are inverted
    /// (`sdrk_proto::frame::Decoder::with_both_polarities`).
    Both,
    /// The four rotations by 90 degrees of symbols of `bits_per_symbol` bits; the rotation is
    /// reported in bits 16 and 17 of the packet flags (`sdrk_proto::frame::RotatingDecoder`).
    Four,
}

/// The rotation rule of the choice index `i` (0 `normal`, 1 `both`, 2 `four`).
const fn rotations_of(i: u16) -> FrameRotations {
    match i {
        1 => FrameRotations::Both,
        2 => FrameRotations::Four,
        _ => FrameRotations::Normal,
    }
}

/// The choice index of a rotation rule.
fn rotations_index(r: FrameRotations) -> u16 {
    match r {
        FrameRotations::Both => 1,
        FrameRotations::Four => 2,
        _ => 0,
    }
}

/// Description of the `frame_decode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "frame_decode",
    summary: "Finds frames of the test framing in a stream of soft bits and outputs their \
        packets.",
    details: "Reads one soft bit per input sample (a value above 0 is a 1, anything else, NaN \
        included, a 0) and hunts for the sync word 0x1ACFFC1D with at most max_sync_errors \
        differing bits. It then reads the length byte, the payload and the CRC-16/CCITT-FALSE \
        and outputs the packet with the flag CRC_OK; a frame with a wrong CRC is output with \
        CRC_FAIL only when emit_bad is set. The packet index is the input sample index of \
        the first sync word bit, counted since construction or reset. After every frame the \
        hunt starts again with the next bit. With whitening pn9 every byte after the sync \
        word is XORed with the PN9 sequence before it is read, as frame_encode with the same \
        setting sends it. With rotations both the hunt also accepts the inverted sync word and \
        then inverts the bits of the frame, which resolves the 180 degree ambiguity of a \
        coherent psk_demod (not the 90 degree one of QPSK). With rotations four the soft bits \
        are taken in symbols of bits_per_symbol bits (counted from the first input sample \
        since construction or reset; the in-phase half first), and the hunt tests the sync \
        word in the four rotations by 90 degrees of a square constellation with Gray coded \
        axes (one quarter turn maps the bit halves (I, Q) to (Q with its first bit inverted, \
        I)), which resolves the ambiguity of QPSK (bits_per_symbol 2) and of qam_demod (4 for \
        16-QAM, 6 for 64-QAM); the frame is read under the inverse rotation, and the packet \
        flags carry the quarter turns 0 to 3 in bits 16 and 17. Pitfall: max_sync_errors above 0 \
        finds frames with damaged sync words but also false syncs in noise; those mostly fail \
        the CRC.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Soft bits, one sample per bit; above 0 is a 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "The decoded packets, at most one per 56 input samples.",
    }],
    params: &[
        ParamDesc {
            name: "max_sync_errors",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: 4,
                default: Some(0),
            },
            typical: "0 for clean links, 2 for noisy ones",
            mutability: Mutability::Build,
            description: "Number of sync word bits that may differ.",
        },
        ParamDesc {
            name: "emit_bad",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false; true to count or inspect damaged frames",
            mutability: Mutability::Build,
            description: "Also output frames whose CRC does not match, with the flag CRC_FAIL.",
        },
        ParamDesc {
            name: "whitening",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: WHITENING_OPTIONS,
                default: 0,
            },
            typical: "none; pn9 for links whose data has long runs of equal bits",
            mutability: Mutability::Build,
            description: "Whitening the sender applied to length, payload and CRC: none, or pn9 (the \
                PN9 sequence is removed).",
        },
        ParamDesc {
            name: "rotations",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: ROTATIONS_OPTIONS,
                default: 0,
            },
            typical: "normal; both behind a coherent BPSK psk_demod, four behind QPSK or \
                qam_demod",
            mutability: Mutability::Build,
            description: "normal accepts only the sync word; both also accepts the inverted \
                sync word and inverts the bits of that frame; four accepts the sync word in \
                the four 90 degree rotations of symbols of bits_per_symbol bits and reports the \
                rotation in the flags (bits 16 and 17).",
        },
        ParamDesc {
            name: "bits_per_symbol",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_BPS as i64,
                default: Some(2),
            },
            typical: "2 for QPSK, 4 for 16-QAM, 6 for 64-QAM",
            mutability: Mutability::Build,
            description: "Bits per symbol of the rotations four (even, 2 to 6); not used by \
                normal and both.",
        },
    ],
    required: &[],
};

/// Configuration of [`FrameDecodeBlock`].
///
/// ```
/// use sdrk_blocks::{FrameDecodeConfig, FrameRotations};
/// let cfg = FrameDecodeConfig::default();
/// assert_eq!((cfg.max_sync_errors, cfg.emit_bad), (0, false));
/// assert_eq!((cfg.rotations, cfg.bits_per_symbol), (FrameRotations::Normal, 2));
/// assert!(cfg.validate().is_ok());
/// assert!(FrameDecodeConfig { max_sync_errors: 5, ..cfg }.validate().is_err());
/// // four rotations need an even number of bits per symbol
/// let four = FrameDecodeConfig { rotations: FrameRotations::Four, bits_per_symbol: 4, ..cfg };
/// assert!(four.validate().is_ok());
/// assert!(FrameDecodeConfig { bits_per_symbol: 3, ..four }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDecodeConfig {
    /// Sync word bits that may differ (0 to 4).
    pub max_sync_errors: usize,
    /// Output frames with a wrong CRC (flag `CRC_FAIL`).
    pub emit_bad: bool,
    /// The whitening of length, payload and CRC that the sender applied.
    pub whitening: Whitening,
    /// The accepted sync-word rotations (M44; `Both` is the former polarity `both` of M40).
    pub rotations: FrameRotations,
    /// Bits per symbol of [`FrameRotations::Four`] (1 to 6, even for `Four`; M44).
    pub bits_per_symbol: usize,
}

impl Default for FrameDecodeConfig {
    /// No sync errors, no bad frames, no whitening, rotations `Normal`, 2 bits per symbol.
    fn default() -> Self {
        Self {
            max_sync_errors: 0,
            emit_bad: false,
            whitening: Whitening::None,
            rotations: FrameRotations::Normal,
            bits_per_symbol: 2,
        }
    }
}

impl FrameDecodeConfig {
    /// Checks the ranges and the combination of `rotations` and `bits_per_symbol`.
    ///
    /// # Errors
    /// `OutOfRange` for `max_sync_errors` above 4 and `bits_per_symbol` outside 1 to 6;
    /// `InvalidParameter` of `bits_per_symbol` for an odd count with `rotations` `four`.
    pub fn validate(&self) -> Result<()> {
        check_count(
            "max_sync_errors",
            self.max_sync_errors,
            0,
            MAX_SYNC_ERRORS as usize,
        )?;
        check_count("bits_per_symbol", self.bits_per_symbol, 1, MAX_BPS)?;
        if self.rotations == FrameRotations::Four && !self.bits_per_symbol.is_multiple_of(2) {
            return Err(Error::InvalidParameter {
                name: "bits_per_symbol",
                reason: "the rotations four need an even number of bits per symbol",
            });
        }
        Ok(())
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange` (the combination of the parameters
    /// is checked by [`FrameDecodeConfig::validate`]).
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "max_sync_errors" => self.max_sync_errors = value.as_usize()?,
            "emit_bad" => self.emit_bad = value.as_bool()?,
            "whitening" => self.whitening = whitening_from_choice(value.as_choice()?),
            "rotations" => self.rotations = rotations_of(value.as_choice()?),
            "bits_per_symbol" => self.bits_per_symbol = value.as_usize()?,
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
            "max_sync_errors" => Ok(ParamValue::I64(self.max_sync_errors as i64)),
            "emit_bad" => Ok(ParamValue::Bool(self.emit_bad)),
            "whitening" => Ok(ParamValue::Choice(whitening_choice(self.whitening))),
            "rotations" => Ok(ParamValue::Choice(rotations_index(self.rotations))),
            "bits_per_symbol" => Ok(ParamValue::I64(self.bits_per_symbol as i64)),
            _ => unknown(),
        }
    }
}

/// The decoder of a [`FrameDecodeBlock`]: the plain one for `normal` and `both`, the rotating
/// one for `four`.
#[derive(Debug)]
enum Hunt {
    Plain(Decoder),
    Four(RotatingDecoder),
}

impl Hunt {
    /// The decoder of `config` with `max` sync-word errors. The two kinds are built out of line,
    /// so that the decoders of a few hundred bytes each do not add up on one stack frame.
    fn new(config: &FrameDecodeConfig, max: u32) -> Result<Self> {
        if config.rotations == FrameRotations::Four {
            Self::four(config, max)
        } else {
            Self::plain(config, max)
        }
    }

    #[inline(never)]
    fn plain(config: &FrameDecodeConfig, max: u32) -> Result<Self> {
        let d = Decoder::new(max, config.emit_bad, config.whitening)?;
        Ok(Self::Plain(d.with_both_polarities(
            config.rotations == FrameRotations::Both,
        )))
    }

    #[inline(never)]
    fn four(config: &FrameDecodeConfig, max: u32) -> Result<Self> {
        let d = Decoder::new(max, config.emit_bad, config.whitening)?;
        let k = u32::try_from(config.bits_per_symbol).unwrap_or(0);
        Ok(Self::Four(RotatingDecoder::new(d, k)?))
    }
}

/// Decoder of the test framing: `f32` soft bits in (one per sample, `> 0` is 1), [`Packets`]
/// out. The block needs no storage.
///
/// The `index` of a packet is the input index of the first bit of its sync word, counted
/// since construction or the last [`Block::reset`]. When a call fills the last free output
/// slot it stops right after the bit that completed that packet, so no packet is ever lost
/// for want of room; a call with no output room consumes nothing.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, FrameDecodeBlock, FrameDecodeConfig, Packet};
/// use sdrk_proto::{frame, PacketFlags};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut dec = FrameDecodeBlock::new(&FrameDecodeConfig::default(), Rate::hz(9600)?, &mut arena)?;
/// let sent = Packet::new(0, PacketFlags::NONE, b"hi")?;
/// let mut bits = [0u8; frame::frame_bits(2, 4)];
/// frame::encode(&sent, 4, frame::Whitening::None, &mut bits)?;
/// let soft: Vec<f32> = bits.iter().map(|&b| if b == 1 { 1.0 } else { -1.0 }).collect();
/// let mut out = [Packet::default(); 2];
/// let p = dec.process(&soft, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (soft.len(), 1));
/// assert_eq!(out[0].data(), b"hi");
/// assert_eq!(out[0].flags, PacketFlags::CRC_OK);
/// assert_eq!(out[0].index, 32); // after the 4 preamble bytes
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FrameDecodeBlock<S: Storage> {
    config: FrameDecodeConfig,
    decoder: Hunt,
    /// Input samples since construction or reset.
    count: u64,
    /// At most one packet per [`MIN_FRAME_BITS`] input samples.
    max_per_input: Ratio,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> FrameDecodeBlock<S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// As [`FrameDecodeConfig::validate`].
    pub fn new(config: &FrameDecodeConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate()?;
        let max = u32::try_from(config.max_sync_errors).unwrap_or(u32::MAX);
        Ok(Self {
            config: *config,
            decoder: Hunt::new(config, max)?,
            count: 0,
            max_per_input: Ratio::new(1, MIN_FRAME_BITS as u64)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// As [`FrameDecodeConfig::validate`].
    pub fn storage_bytes(config: &FrameDecodeConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        Ok(Plan::default().bytes())
    }

    /// Input samples consumed since construction or the last reset.
    pub fn position(&self) -> u64 {
        self.count
    }
}

impl<S: Storage> Block for FrameDecodeBlock<S> {
    type In = f32;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [Packet]) -> Result<Progress> {
        let mut slots = output.iter_mut();
        let Some(mut slot) = slots.next() else {
            return Ok(Progress::default());
        };
        // The index count stays in a local; the block state is written once at the end.
        let start = self.count;
        let mut consumed = input.len();
        let mut produced = 0;
        // One loop per decoder type, so that the bit loop of the plain decoder stays free of the
        // symbol logic of the rotating one.
        macro_rules! decode {
            ($dec:expr) => {
                for (i, &x) in input.iter().enumerate() {
                    let index = start.wrapping_add(i as u64);
                    if let Some(packet) = $dec.push_bit(x > 0.0, index) {
                        *slot = packet;
                        produced += 1;
                        let Some(next) = slots.next() else {
                            consumed = i + 1;
                            break;
                        };
                        slot = next;
                    }
                }
            };
        }
        match &mut self.decoder {
            Hunt::Plain(d) => decode!(d),
            Hunt::Four(d) => decode!(d),
        }
        self.count = start.wrapping_add(consumed as u64);
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.max_per_input,
            kind: RatioKind::Packets {
                max_per_input: self.max_per_input,
            },
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        match &mut self.decoder {
            Hunt::Plain(d) => d.reset(),
            Hunt::Four(d) => d.reset(),
        }
        self.count = 0;
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
