//! `hdlc_encode`: packets to the bits of HDLC frames (AX.25), one `±1.0` sample per bit, with
//! NRZI coding inside.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_proto::hdlc::{self, Encoder};
use sdrk_proto::{Packet, nrzi};

use super::{check_count, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// Largest `preamble_flags` (D637).
const MAX_PREAMBLE_FLAGS: usize = 1024;

/// Largest `tail_flags` (D637).
const MAX_TAIL_FLAGS: usize = 64;

/// Description of the `hdlc_encode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "hdlc_encode",
    summary: "Frames packets as HDLC (AX.25, APRS) and outputs one +1/-1 sample per bit.",
    details: "Sends every packet as one HDLC frame: preamble_flags flags 0x7E, the payload and \
        its FCS (CRC-16/X-25, low byte first) with a 0 stuffed after every five 1s, then \
        tail_flags flags; bytes go least significant bit first. With nrzi set (the AX.25 line \
        code) a 0 bit changes the level and a 1 bit keeps it; the level is +1.0 or -1.0. \
        Without nrzi a 1 bit is +1.0 and a 0 bit -1.0. Between packets the block sends flags, \
        which keep the receiver in sync. It is a source: the output rate is the rate_hz of \
        the packet input of the pipeline (1200 for APRS), and in push mode the block only \
        produces the samples of the packets it has. A packet is consumed when its last \
        sample is written. Pitfall: a packet needs at least 2 bytes; a shorter one is \
        rejected with an error.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Packets],
        description: "The packets to send (2 to 256 bytes each); index and flags are ignored.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "One sample per bit, +1.0 or -1.0; flags between frames.",
    }],
    params: &[
        ParamDesc {
            name: "nrzi",
            unit: Unit::None,
            kind: ParamKind::Bool { default: true },
            typical: "true for AX.25 and APRS",
            mutability: Mutability::Build,
            description: "NRZI code the bits (a 0 changes the level, a 1 keeps it).",
        },
        ParamDesc {
            name: "preamble_flags",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_PREAMBLE_FLAGS as i64,
                default: Some(32),
            },
            typical: "32 (about 210 ms at 1200 Bd); more for slow receivers",
            mutability: Mutability::Build,
            description: "Flags sent before every frame, 1 to 1024.",
        },
        ParamDesc {
            name: "tail_flags",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 1,
                max: MAX_TAIL_FLAGS as i64,
                default: Some(4),
            },
            typical: "4",
            mutability: Mutability::Build,
            description: "Flags sent after every frame, 1 to 64.",
        },
    ],
    required: &[],
};

/// Configuration of [`HdlcEncodeBlock`].
///
/// ```
/// use sdrk_blocks::HdlcEncodeConfig;
/// let cfg = HdlcEncodeConfig::default();
/// assert_eq!((cfg.nrzi, cfg.preamble_flags, cfg.tail_flags), (true, 32, 4));
/// assert!(cfg.validate().is_ok());
/// assert!(HdlcEncodeConfig { tail_flags: 0, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdlcEncodeConfig {
    /// NRZI code the bits.
    pub nrzi: bool,
    /// Flags before every frame (1 to 1024).
    pub preamble_flags: usize,
    /// Flags after every frame (1 to 64).
    pub tail_flags: usize,
}

impl Default for HdlcEncodeConfig {
    fn default() -> Self {
        Self {
            nrzi: true,
            preamble_flags: 32,
            tail_flags: 4,
        }
    }
}

impl HdlcEncodeConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for `preamble_flags` outside 1 to 1024 or `tail_flags` outside 1 to 64.
    pub fn validate(&self) -> Result<()> {
        check_count("preamble_flags", self.preamble_flags, 1, MAX_PREAMBLE_FLAGS)?;
        check_count("tail_flags", self.tail_flags, 1, MAX_TAIL_FLAGS)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "nrzi" => self.nrzi = value.as_bool()?,
            "preamble_flags" => self.preamble_flags = value.as_usize()?,
            "tail_flags" => self.tail_flags = value.as_usize()?,
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
            "preamble_flags" => Ok(ParamValue::I64(self.preamble_flags as i64)),
            "tail_flags" => Ok(ParamValue::I64(self.tail_flags as i64)),
            _ => unknown(),
        }
    }
}

/// HDLC encoder: [`Packets`] in, one `f32` sample per bit out (`+1.0` or `-1.0`). The block
/// needs no storage.
///
/// `process` writes the frames of the packets in order, bit by bit from
/// [`sdrk_proto::hdlc::Encoder`] (the FCS is computed once per frame). A packet counts as
/// consumed only when the last sample of its frame is written; a frame that does not fit the
/// output is continued in the next call, which must start with the same packet (the caller
/// resubmits the unconsumed input). A call with an empty input is idle: it drops a frame in
/// progress (its packet was not resubmitted) and fills the whole output with flags. The NRZI
/// level runs on across frames and flags. The pipeline makes idle calls only in pull mode.
///
/// A packet with fewer than 2 bytes is an error when it is the first packet of a call; after
/// other packets, the call stops before it (and returns them as consumed).
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, HdlcEncodeBlock, HdlcEncodeConfig, Packet};
/// use sdrk_proto::PacketFlags;
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = HdlcEncodeConfig { nrzi: false, preamble_flags: 1, tail_flags: 1 };
/// let mut enc = HdlcEncodeBlock::new(&cfg, Rate::hz(1200)?, &mut arena)?;
/// let p = Packet::new(0, PacketFlags::NONE, b"hi")?;
/// let mut out = [0.0f32; 128];
/// let progress = enc.process(&[p], &mut out)?;
/// // Two flags and four bytes (payload and FCS); "hi" and its FCS need no stuffed bit.
/// assert_eq!((progress.consumed, progress.produced), (1, 48));
/// assert_eq!(&out[..8], &[-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]); // 0x7E
/// // Idle: an empty input fills the output with flags.
/// let idle = enc.process(&[], &mut out)?;
/// assert_eq!((idle.consumed, idle.produced), (0, 128));
/// assert_eq!(&out[8..16], &[-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct HdlcEncodeBlock<S: Storage> {
    config: HdlcEncodeConfig,
    /// The frame of the first input packet, while it is being written.
    frame: Option<Encoder>,
    /// The NRZI line level (true: +1.0).
    level: bool,
    /// The next bit of the idle flag (0 to 7, least significant bit first).
    idle_pos: u8,
    /// Largest number of output samples of one packet.
    max_samples: Ratio,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> HdlcEncodeBlock<S> {
    /// Creates the block; it needs no storage. The NRZI level starts at `-1.0`.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn new(config: &HdlcEncodeConfig, _rate_out: Rate, _storage: &mut S) -> Result<Self> {
        config.validate()?;
        let max = hdlc::max_bits(hdlc::MAX_PAYLOAD, config.preamble_flags, config.tail_flags);
        Ok(Self {
            config: *config,
            frame: None,
            level: false,
            idle_pos: 0,
            max_samples: Ratio::new(max as u64, 1)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn storage_bytes(config: &HdlcEncodeConfig, _rate_out: Rate) -> Result<usize> {
        config.validate()?;
        Ok(Plan::default().bytes())
    }

    /// Output rate: the rate the block is built with (the `rate_hz` of its packet input).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn rate_out(config: &HdlcEncodeConfig, rate: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate)
    }

    /// Whether a frame is being written (its packet is the first of the next call).
    pub fn in_frame(&self) -> bool {
        self.frame.is_some()
    }

    /// The sample of `bit` (`0` or `1`); with NRZI it moves the line level.
    #[inline]
    fn sample(&mut self, bit: bool) -> f32 {
        let one = if self.config.nrzi {
            self.level = nrzi::encode(self.level, bit);
            self.level
        } else {
            bit
        };
        if one { 1.0 } else { -1.0 }
    }

    /// Writes the next samples of the frame in progress to `out`; returns the number written.
    /// The frame ends when its last bit is written.
    fn write_frame(&mut self, out: &mut [f32]) -> usize {
        let Some(mut enc) = self.frame.take() else {
            return 0;
        };
        let mut n = 0;
        for o in out.iter_mut() {
            let Some(bit) = enc.next() else {
                break;
            };
            *o = self.sample(bit != 0);
            n += 1;
        }
        // The upper bound of the remaining bits is 0 exactly when the frame is complete.
        if enc.size_hint().1 != Some(0) {
            self.frame = Some(enc);
        }
        n
    }

    /// Fills `out` with idle flags.
    fn fill_idle(&mut self, out: &mut [f32]) {
        for o in out {
            let bit = (hdlc::FLAG >> self.idle_pos) & 1 != 0;
            self.idle_pos = (self.idle_pos + 1) & 7;
            *o = self.sample(bit);
        }
    }
}

impl<S: Storage> Block for HdlcEncodeBlock<S> {
    type In = Packets;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[Packet], output: &mut [f32]) -> Result<Progress> {
        if input.is_empty() {
            self.frame = None;
            self.fill_idle(output);
            return Ok(Progress {
                consumed: 0,
                produced: output.len(),
            });
        }
        let mut consumed = 0;
        let mut produced = 0;
        while let Some(packet) = input.get(consumed) {
            let room = output.get_mut(produced..).unwrap_or_default();
            if room.is_empty() {
                break;
            }
            if self.frame.is_none() {
                let cfg = self.config;
                match Encoder::new(packet, cfg.preamble_flags, cfg.tail_flags) {
                    Ok(enc) => self.frame = Some(enc),
                    // Report the packets before the bad one first.
                    Err(_) if consumed > 0 => break,
                    Err(e) => return Err(Error::Proto(e)),
                }
            }
            produced += self.write_frame(room);
            if self.in_frame() {
                break; // the output is full
            }
            consumed += 1;
        }
        Ok(Progress { consumed, produced })
    }

    fn timing(&self) -> BlockTiming {
        BlockTiming {
            ratio: self.max_samples,
            kind: RatioKind::Source,
            delay: Delay::ZERO,
        }
    }

    fn reset(&mut self) {
        self.frame = None;
        self.level = false;
        self.idle_pos = 0;
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
