//! `frame_decode`: soft bits to the packets of the test framing of `sdrk_proto::frame`.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_proto::Packet;
use sdrk_proto::frame::{Decoder, MAX_SYNC_ERRORS, Whitening, frame_bits};

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
        setting sends it. Pitfall: max_sync_errors above 0 finds frames \
        with damaged sync words but also false syncs in noise; those mostly fail the CRC.",
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
    ],
    required: &[],
};

/// Configuration of [`FrameDecodeBlock`].
///
/// ```
/// use sdrk_blocks::FrameDecodeConfig;
/// let cfg = FrameDecodeConfig::default();
/// assert_eq!((cfg.max_sync_errors, cfg.emit_bad), (0, false));
/// assert!(cfg.validate().is_ok());
/// assert!(FrameDecodeConfig { max_sync_errors: 5, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameDecodeConfig {
    /// Sync word bits that may differ (0 to 4).
    pub max_sync_errors: usize,
    /// Output frames with a wrong CRC (flag `CRC_FAIL`).
    pub emit_bad: bool,
    /// The whitening of length, payload and CRC that the sender applied.
    pub whitening: Whitening,
}

impl FrameDecodeConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for `max_sync_errors` above 4.
    pub fn validate(&self) -> Result<()> {
        check_count(
            "max_sync_errors",
            self.max_sync_errors,
            0,
            MAX_SYNC_ERRORS as usize,
        )
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "max_sync_errors" => self.max_sync_errors = value.as_usize()?,
            "emit_bad" => self.emit_bad = value.as_bool()?,
            "whitening" => self.whitening = whitening_from_choice(value.as_choice()?),
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
            _ => unknown(),
        }
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
    decoder: Decoder,
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
    /// `OutOfRange` for an invalid configuration.
    pub fn new(config: &FrameDecodeConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate()?;
        let max = u32::try_from(config.max_sync_errors).unwrap_or(u32::MAX);
        Ok(Self {
            config: *config,
            decoder: Decoder::new(max, config.emit_bad, config.whitening)?,
            count: 0,
            max_per_input: Ratio::new(1, MIN_FRAME_BITS as u64)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
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
        for (i, &x) in input.iter().enumerate() {
            let index = start.wrapping_add(i as u64);
            if let Some(packet) = self.decoder.push_bit(x > 0.0, index) {
                *slot = packet;
                produced += 1;
                let Some(next) = slots.next() else {
                    consumed = i + 1;
                    break;
                };
                slot = next;
            }
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
        self.decoder.reset();
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
