//! `frame_encode`: packets to the bits of the test framing of `sdrk_proto::frame`, one `±1.0`
//! sample per bit.

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_proto::Packet;
use sdrk_proto::frame::{MAX_PAYLOAD, Whitening, encode_bytes, frame_bits};
use sdrk_proto::whiten::{PN9_PERIOD, pn9_byte};

use super::{WHITENING_OPTIONS, check_count, unknown, whitening_choice, whitening_from_choice};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// Largest `preamble_bytes`.
const MAX_PREAMBLE: usize = 32;

/// Bytes of the longest frame: 32 preamble bytes, sync word, length, 253 payload bytes, CRC.
const MAX_FRAME_BYTES: usize = frame_bits(MAX_PAYLOAD, MAX_PREAMBLE) / 8;

/// The four samples of every nibble value, most significant bit first (`+1.0` for 1, `-1.0`
/// for 0): a whole byte of a frame is two copies of 16 bytes.
static NIBBLE_SAMPLES: [[f32; 4]; 16] = nibble_samples();

const fn nibble_samples() -> [[f32; 4]; 16] {
    let mut table = [[0.0f32; 4]; 16];
    let mut b = 0;
    while b < 16 {
        let mut k = 0;
        while k < 4 {
            #[allow(clippy::indexing_slicing)] // const fn: `b < 16`, `k < 4`
            {
                table[b][k] = if (b >> (3 - k)) & 1 == 1 { 1.0 } else { -1.0 };
            }
            k += 1;
        }
        b += 1;
    }
    table
}

/// The sample of bit `pos` (counted from the first bit of the frame) of `frame`.
fn bit_sample(frame: &[u8], pos: usize) -> f32 {
    let byte = frame.get(pos / 8).copied().unwrap_or(0);
    if (byte >> (7 - pos % 8)) & 1 == 1 {
        1.0
    } else {
        -1.0
    }
}

/// The names of the `idle` choice, in index order.
const IDLE_OPTIONS: &[&str] = &["zero", "preamble", "pn9"];

/// Bits of one period of the `pn9` idle signal (511 bytes).
const PN9_IDLE_BITS: usize = 8 * PN9_PERIOD;

/// Description of the `frame_encode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "frame_encode",
    summary: "Frames packets with preamble, sync word, length and CRC and outputs one +1/-1 \
        sample per bit.",
    details: "Sends every packet as one frame of the test framing of sdrk_proto::frame: \
        preamble_bytes bytes 0x55, the sync word 0x1ACFFC1D, a length byte, the payload and a \
        CRC-16/CCITT-FALSE over length and payload, most significant bit first. A 1 bit is \
        +1.0, a 0 bit -1.0. Between packets the block sends idle samples: 0.0 (idle zero), \
        the alternating bits of 0x55 (idle preamble) or the bits of the PN9 sequence of \
        sdrk_proto::whiten, most significant bit of each byte first, continued from gap to \
        gap and restarted only by reset (idle pn9: random-looking symbols on which the timing \
        loop and the blind equalizer of a QAM or PSK receiver keep training). It is a source: the output rate is the \
        rate_hz of the packet input of the pipeline, and in push mode the block only produces \
        the samples of the packets it has. A packet is consumed when its last sample is \
        written. With whitening pn9 the length byte, the payload and the CRC are XORed with \
        the PN9 sequence (x^9 + x^5 + 1, seed 0x1FF) of sdrk_proto::whiten; the receiver must \
        use the same setting. Pitfall: payloads above 253 bytes do not fit the length byte \
        and are rejected with an error.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::Packets],
        description: "The packets to send (at most 253 bytes each); index and flags are ignored.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::F32],
        description: "One sample per bit: +1.0 for 1, -1.0 for 0; idle samples between frames.",
    }],
    params: &[
        ParamDesc {
            name: "preamble_bytes",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: 32,
                default: Some(4),
            },
            typical: "4; more for receivers that need time to settle",
            mutability: Mutability::Build,
            description: "Bytes 0x55 sent before the sync word of every frame.",
        },
        ParamDesc {
            name: "idle",
            unit: Unit::None,
            kind: ParamKind::Choice {
                options: IDLE_OPTIONS,
                default: 0,
            },
            typical: "zero; preamble to keep a receiver's clock recovery busy; pn9 for QAM \
                and PSK receivers",
            mutability: Mutability::Build,
            description: "What is sent between frames: zero sends 0.0, preamble the bits of \
                0x55 as +1/-1, pn9 the bits of the PN9 sequence as +1/-1.",
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
            description: "Whitening of length, payload and CRC: none, or pn9 (XOR with the PN9 \
                sequence).",
        },
    ],
    required: &[],
};

/// The signal between frames of [`FrameEncodeBlock`].
///
/// ```
/// use sdrk_blocks::{FrameEncodeConfig, FrameIdle};
/// assert_eq!(FrameEncodeConfig::default().idle, FrameIdle::Zero);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrameIdle {
    /// `0.0` samples.
    Zero,
    /// The alternating bits of `0x55` (`-1.0`, `+1.0`, ...).
    Preamble,
    /// The bits of the PN9 sequence of [`sdrk_proto::whiten`] (seed `0x1FF`, first bytes
    /// `FF E1 1D 9A`), most significant bit of each byte first, as `+1.0` for 1 and `-1.0`
    /// for 0. The sequence continues from one idle gap to the next (period 4088 bits) and
    /// starts again only after `reset`.
    Pn9,
}

/// Configuration of [`FrameEncodeBlock`].
///
/// ```
/// use sdrk_blocks::FrameEncodeConfig;
/// let cfg = FrameEncodeConfig::default();
/// assert_eq!(cfg.preamble_bytes, 4);
/// assert!(cfg.validate().is_ok());
/// assert!(FrameEncodeConfig { preamble_bytes: 33, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameEncodeConfig {
    /// Preamble bytes `0x55` before every sync word (0 to 32).
    pub preamble_bytes: usize,
    /// The signal between frames.
    pub idle: FrameIdle,
    /// The whitening of length, payload and CRC.
    pub whitening: Whitening,
}

impl Default for FrameEncodeConfig {
    fn default() -> Self {
        Self {
            preamble_bytes: 4,
            idle: FrameIdle::Zero,
            whitening: Whitening::None,
        }
    }
}

impl FrameEncodeConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for `preamble_bytes` above 32.
    pub fn validate(&self) -> Result<()> {
        check_count("preamble_bytes", self.preamble_bytes, 0, MAX_PREAMBLE)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "preamble_bytes" => self.preamble_bytes = value.as_usize()?,
            "idle" => {
                self.idle = match value.as_choice()? {
                    0 => FrameIdle::Zero,
                    1 => FrameIdle::Preamble,
                    _ => FrameIdle::Pn9,
                }
            }
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
            "preamble_bytes" => Ok(ParamValue::I64(self.preamble_bytes as i64)),
            "idle" => Ok(ParamValue::Choice(match self.idle {
                FrameIdle::Zero => 0,
                FrameIdle::Preamble => 1,
                FrameIdle::Pn9 => 2,
            })),
            "whitening" => Ok(ParamValue::Choice(whitening_choice(self.whitening))),
            _ => unknown(),
        }
    }
}

/// Encoder of the test framing: [`Packets`] in, one `f32` sample per bit out (`+1.0` for 1,
/// `-1.0` for 0). The block needs no storage.
///
/// `process` writes the frames of the packets in order. A packet counts as consumed only
/// when the last sample of its frame is written; a frame that does not fit the output is
/// continued in the next call, which must start with the same packet (the caller resubmits
/// the unconsumed input). A call with an empty input is idle: it drops a frame in progress
/// (its packet was not resubmitted) and fills the whole output with idle samples. The
/// pipeline makes such calls only in pull mode, where the output has to be filled.
///
/// A packet with more than [`MAX_PAYLOAD`] bytes is an error when it is the first packet of
/// a call; after other packets, the call stops before it (and returns them as consumed).
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, FrameEncodeBlock, FrameEncodeConfig, Packet};
/// use sdrk_proto::PacketFlags;
/// let mut arena = ArenaStorage::new(&mut []);
/// let cfg = FrameEncodeConfig { preamble_bytes: 1, ..FrameEncodeConfig::default() };
/// let mut enc = FrameEncodeBlock::new(&cfg, Rate::hz(48_000)?, &mut arena)?;
/// let p = Packet::new(0, PacketFlags::NONE, b"hi")?;
/// let mut out = [0.0f32; 128];
/// let progress = enc.process(&[p], &mut out)?;
/// // 1 preamble byte, 4 sync bytes, length, 2 payload bytes, 2 CRC bytes.
/// assert_eq!((progress.consumed, progress.produced), (1, 80));
/// assert_eq!(&out[..4], &[-1.0, 1.0, -1.0, 1.0]); // 0x55
/// // Idle: an empty input fills the output.
/// let idle = enc.process(&[], &mut out)?;
/// assert_eq!((idle.consumed, idle.produced), (0, 128));
/// assert!(out.iter().all(|&x| x == 0.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FrameEncodeBlock<S: Storage> {
    config: FrameEncodeConfig,
    /// The frame of the first input packet as packed bytes, while it is being written.
    frame: [u8; MAX_FRAME_BYTES],
    /// Bits of that frame; 0 when no frame is in progress.
    frame_len: usize,
    /// Bits of that frame already written.
    pos: usize,
    /// Next idle bit of the `preamble` idle signal (true: +1.0).
    idle_bit: bool,
    /// Next bit of the `pn9` idle signal, `0 .. PN9_IDLE_BITS`.
    idle_pos: usize,
    /// Largest number of output samples of one packet.
    max_samples: Ratio,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> FrameEncodeBlock<S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn new(config: &FrameEncodeConfig, _rate_out: Rate, _storage: &mut S) -> Result<Self> {
        config.validate()?;
        let max = frame_bits(MAX_PAYLOAD, config.preamble_bytes);
        Ok(Self {
            config: *config,
            frame: [0; MAX_FRAME_BYTES],
            frame_len: 0,
            pos: 0,
            idle_bit: false,
            idle_pos: 0,
            max_samples: Ratio::new(max as u64, 1)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn storage_bytes(config: &FrameEncodeConfig, _rate_out: Rate) -> Result<usize> {
        config.validate()?;
        Ok(Plan::default().bytes())
    }

    /// Output rate: the rate the block is built with (the `rate_hz` of its packet input).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn rate_out(config: &FrameEncodeConfig, rate: Rate) -> Result<Rate> {
        config.validate()?;
        Ok(rate)
    }

    /// The number of output samples of the frame of a packet with `payload_len` bytes.
    ///
    /// ```
    /// use sdrk_blocks::{ArenaStorage, FrameEncodeBlock, FrameEncodeConfig};
    /// let mut arena = ArenaStorage::new(&mut []);
    /// let enc = FrameEncodeBlock::new(&FrameEncodeConfig::default(), sdrk::Rate::hz(9600)?, &mut arena)?;
    /// assert_eq!(enc.frame_samples(0), 88);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn frame_samples(&self, payload_len: usize) -> usize {
        frame_bits(payload_len, self.config.preamble_bytes)
    }

    /// Whether a frame is being written (its packet is the first of the next call).
    pub fn in_frame(&self) -> bool {
        self.frame_len != 0
    }

    /// Starts the frame of `packet`.
    fn start(&mut self, packet: &Packet) -> Result<()> {
        let bytes = encode_bytes(
            packet,
            self.config.preamble_bytes,
            self.config.whitening,
            &mut self.frame,
        )?;
        self.frame_len = bytes * 8;
        self.pos = 0;
        Ok(())
    }

    /// Writes the next samples of the frame in progress to `out`; returns the number written.
    fn write_frame(&mut self, out: &mut [f32]) -> usize {
        let n = out.len().min(self.frame_len.saturating_sub(self.pos));
        let Some(out) = out.get_mut(..n) else {
            return 0;
        };
        // Up to the next byte boundary bit by bit, then whole bytes from the table, then the
        // bits of a last partial byte.
        let lead = ((8 - self.pos % 8) % 8).min(n);
        let (head, rest) = out.split_at_mut(lead);
        for (k, o) in head.iter_mut().enumerate() {
            *o = bit_sample(&self.frame, self.pos + k);
        }
        let first = (self.pos + lead) / 8;
        let mut chunks = rest.chunks_exact_mut(8);
        let bytes = self.frame.get(first..).unwrap_or_default();
        for (chunk, &byte) in (&mut chunks).zip(bytes) {
            let (hi, lo) = chunk.split_at_mut(4);
            let nibble = |n: u8| NIBBLE_SAMPLES.get(usize::from(n & 15)).copied();
            if let (Some(h), Some(l)) = (nibble(byte >> 4), nibble(byte)) {
                hi.copy_from_slice(&h);
                lo.copy_from_slice(&l);
            }
        }
        let tail = chunks.into_remainder();
        let tail_pos = self.pos + n - tail.len();
        for (k, o) in tail.iter_mut().enumerate() {
            *o = bit_sample(&self.frame, tail_pos + k);
        }
        self.pos += n;
        if self.pos == self.frame_len {
            self.frame_len = 0;
            self.pos = 0;
        }
        n
    }

    /// Fills `out` with idle samples.
    fn fill_idle(&mut self, out: &mut [f32]) {
        match self.config.idle {
            FrameIdle::Zero => out.fill(0.0),
            FrameIdle::Preamble => {
                for o in out {
                    *o = if self.idle_bit { 1.0 } else { -1.0 };
                    self.idle_bit = !self.idle_bit;
                }
            }
            FrameIdle::Pn9 => self.fill_pn9(out),
        }
    }

    /// Fills `out` with the next bits of the PN9 sequence: bit by bit up to the next byte
    /// boundary, then whole bytes from the nibble table, then the bits of a partial byte.
    fn fill_pn9(&mut self, out: &mut [f32]) {
        let pn9_bit = |pos: usize| {
            if (pn9_byte(pos / 8) >> (7 - pos % 8)) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        };
        let lead = ((8 - self.idle_pos % 8) % 8).min(out.len());
        let (head, rest) = out.split_at_mut(lead);
        for o in head {
            *o = pn9_bit(self.idle_pos);
            self.idle_pos += 1;
        }
        self.idle_pos %= PN9_IDLE_BITS;
        if rest.is_empty() {
            return; // `idle_pos` may be inside a byte
        }
        let mut byte = self.idle_pos / 8;
        let mut chunks = rest.chunks_exact_mut(8);
        for chunk in &mut chunks {
            let b = pn9_byte(byte);
            let (hi, lo) = chunk.split_at_mut(4);
            let nibble = |n: u8| NIBBLE_SAMPLES.get(usize::from(n & 15)).copied();
            if let (Some(h), Some(l)) = (nibble(b >> 4), nibble(b)) {
                hi.copy_from_slice(&h);
                lo.copy_from_slice(&l);
            }
            byte = (byte + 1) % PN9_PERIOD;
        }
        self.idle_pos = byte * 8;
        for o in chunks.into_remainder() {
            *o = pn9_bit(self.idle_pos);
            self.idle_pos += 1;
        }
        self.idle_pos %= PN9_IDLE_BITS;
    }
}

impl<S: Storage> Block for FrameEncodeBlock<S> {
    type In = Packets;
    type Out = f32;

    fn description(&self) -> &'static BlockDescription {
        &DESCRIPTION
    }

    fn process(&mut self, input: &[Packet], output: &mut [f32]) -> Result<Progress> {
        if input.is_empty() {
            self.frame_len = 0;
            self.pos = 0;
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
            if !self.in_frame() {
                match self.start(packet) {
                    Ok(()) => {}
                    // Report the packets before the bad one first.
                    Err(_) if consumed > 0 => break,
                    Err(e) => return Err(e),
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
        self.frame_len = 0;
        self.pos = 0;
        self.idle_bit = false;
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
