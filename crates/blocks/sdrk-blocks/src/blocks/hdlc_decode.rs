//! `hdlc_decode`: soft bits to the packets of HDLC frames (AX.25), with NRZI decoding inside.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_proto::Packet;
use sdrk_proto::hdlc::{self, Decoder};
use sdrk_proto::nrzi;

use super::unknown;
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// Input bits of the shortest frame with a shared flag: 8 flag bits and 4 bytes (2 payload
/// bytes and the FCS), so at most one packet per 40 input samples.
pub(crate) const MIN_FRAME_BITS: usize = 8 + 8 * (hdlc::MIN_PAYLOAD + 2);

/// Description of the `hdlc_decode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "hdlc_decode",
    summary: "Finds HDLC frames (AX.25, APRS) in a stream of soft bits and outputs their \
        packets.",
    details: "Reads one soft bit per input sample (a value above 0 is a 1, anything else, NaN \
        included, a 0). With nrzi set (the AX.25 line code) the bits are NRZI decoded first: \
        an unchanged level is a 1, a change a 0. The decoder waits for a flag 0x7E, collects \
        the bits up to the next flag (which also opens the next frame), removes the 0 after \
        every five 1s and reads the bytes least significant bit first. A frame of 2 to 256 \
        payload bytes whose FCS (CRC-16/X-25, sent low byte first) matches is output with the \
        flag CRC_OK and without the FCS; a frame with a wrong FCS only when emit_bad is set, \
        with CRC_FAIL. Seven 1s in a row abort a frame; frames of other lengths are dropped. \
        The packet index is the input sample index of the first bit after the opening flag, \
        counted since construction or reset. Pitfall: the soft bits must come one per bit, \
        for example from fsk_demod; the polarity does not matter with nrzi.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Soft bits, one sample per bit; above 0 is a 1 (before NRZI decoding).",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "The payloads of the decoded frames (without FCS), at most one per 40 \
            input samples.",
    }],
    params: &[
        ParamDesc {
            name: "nrzi",
            unit: Unit::None,
            kind: ParamKind::Bool { default: true },
            typical: "true for AX.25 and APRS",
            mutability: Mutability::Build,
            description: "NRZI decode the bits before the HDLC decoder (a change of level is a \
                0).",
        },
        ParamDesc {
            name: "emit_bad",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false; true to count or inspect damaged frames",
            mutability: Mutability::Build,
            description: "Also output frames whose FCS does not match, with the flag CRC_FAIL.",
        },
    ],
    required: &[],
};

/// Configuration of [`HdlcDecodeBlock`].
///
/// ```
/// use sdrk_blocks::HdlcDecodeConfig;
/// let cfg = HdlcDecodeConfig::default();
/// assert_eq!((cfg.nrzi, cfg.emit_bad), (true, false));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdlcDecodeConfig {
    /// NRZI decode the bits first.
    pub nrzi: bool,
    /// Output frames with a wrong FCS (flag `CRC_FAIL`).
    pub emit_bad: bool,
}

impl Default for HdlcDecodeConfig {
    fn default() -> Self {
        Self {
            nrzi: true,
            emit_bad: false,
        }
    }
}

impl HdlcDecodeConfig {
    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// ```
    /// use sdrk_blocks::{HdlcDecodeConfig, ParamValue};
    /// let mut cfg = HdlcDecodeConfig::default();
    /// cfg.set("nrzi", ParamValue::Bool(false))?;
    /// assert!(!cfg.nrzi);
    /// # Ok::<(), sdrk_blocks::Error>(())
    /// ```
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "nrzi" => self.nrzi = value.as_bool()?,
            "emit_bad" => self.emit_bad = value.as_bool()?,
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
            "nrzi" => Ok(ParamValue::Bool(self.nrzi)),
            "emit_bad" => Ok(ParamValue::Bool(self.emit_bad)),
            _ => unknown(),
        }
    }
}

/// HDLC decoder: `f32` soft bits in (one per sample, `> 0` is 1), [`Packets`] out. The block
/// needs no storage.
///
/// Per input sample the block NRZI decodes the bit (when `nrzi` is set) and feeds it to
/// [`sdrk_proto::hdlc::Decoder`]; the FCS is checked once per frame. The `index` of a packet
/// is the input index of the first bit after its opening flag, counted since construction or
/// the last [`Block::reset`]. When a call fills the last free output slot it stops right
/// after the bit that completed that packet, so no packet is lost for want of room; a call
/// with no output room consumes nothing.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, HdlcDecodeBlock, HdlcDecodeConfig, Packet};
/// use sdrk_proto::{hdlc, PacketFlags};
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = HdlcDecodeConfig { nrzi: false, ..HdlcDecodeConfig::default() };
/// let mut dec = HdlcDecodeBlock::new(&cfg, Rate::hz(1200)?, &mut arena)?;
/// let sent = Packet::new(0, PacketFlags::NONE, b"hi")?;
/// let mut bits = [0u8; 128];
/// let n = hdlc::encode(&sent, 2, 1, &mut bits)?;
/// let soft: Vec<f32> = bits[..n].iter().map(|&b| if b == 1 { 1.0 } else { -1.0 }).collect();
/// let mut out = [Packet::default(); 2];
/// let p = dec.process(&soft, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (n, 1));
/// assert_eq!(out[0].data(), b"hi");
/// assert_eq!(out[0].flags, PacketFlags::CRC_OK);
/// assert_eq!(out[0].index, 16); // after the two opening flags
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct HdlcDecodeBlock<S: Storage> {
    config: HdlcDecodeConfig,
    nrzi: nrzi::Decoder,
    decoder: Decoder,
    /// Input samples since construction or reset.
    count: u64,
    /// At most one packet per [`MIN_FRAME_BITS`] input samples.
    max_per_input: Ratio,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> HdlcDecodeBlock<S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// None in practice (the signature matches the other blocks).
    pub fn new(config: &HdlcDecodeConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        Ok(Self {
            config: *config,
            nrzi: nrzi::Decoder::new(false),
            decoder: Decoder::new(config.emit_bad),
            count: 0,
            max_per_input: Ratio::new(1, MIN_FRAME_BITS as u64)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// None in practice.
    pub fn storage_bytes(_config: &HdlcDecodeConfig, _rate_in: Rate) -> Result<usize> {
        Ok(Plan::default().bytes())
    }

    /// Input samples consumed since construction or the last reset.
    pub fn position(&self) -> u64 {
        self.count
    }

    /// The per-sample loop; `NRZI` selects the line decoding at compile time.
    // `always`: both instances are the whole of `process`.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn run<const NRZI: bool>(&mut self, input: &[f32], output: &mut [Packet]) -> Progress {
        let mut slots = output.iter_mut();
        let Some(mut slot) = slots.next() else {
            return Progress::default();
        };
        let start = self.count;
        let mut consumed = input.len();
        let mut produced = 0;
        for (i, &x) in input.iter().enumerate() {
            let level = x > 0.0;
            let bit = if NRZI { self.nrzi.decode(level) } else { level };
            let index = start.wrapping_add(i as u64);
            if let Some(packet) = self.decoder.push_bit(bit, index) {
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
        Progress { consumed, produced }
    }
}

impl<S: Storage> Block for HdlcDecodeBlock<S> {
    type In = f32;
    type Out = Packets;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[f32], output: &mut [Packet]) -> Result<Progress> {
        Ok(if self.config.nrzi {
            self.run::<true>(input, output)
        } else {
            self.run::<false>(input, output)
        })
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
        self.nrzi.reset(false);
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
