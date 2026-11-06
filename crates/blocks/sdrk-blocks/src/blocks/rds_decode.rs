//! `rds_decode`: soft data bits to RDS groups (block synchronization with the offset words).

use core::marker::PhantomData;

use sdrk::{Delay, Progress, Rate, Ratio, RatioKind};
use sdrk_proto::rds::{BLOCK_BITS, GENERATOR, GROUP_BITS, Offset, VALID_SHIFT, VERSION_B};
use sdrk_proto::{Packet, PacketFlags};

use super::{check_count, unknown};
use crate::block::{Block, BlockTiming, ParamValue};
use crate::describe::{
    BlockDescription, Mutability, ParamDesc, ParamKind, PortDesc, PortType, Unit,
};
use crate::error::{Error, Result};
use crate::ports::Packets;
use crate::storage::{Plan, Storage};

/// The blocks whose validity the synchronized decoder counts.
const WINDOW: usize = 50;
/// The largest `max_bad` (all blocks of the window).
const MAX_BAD_LIMIT: usize = WINDOW;
/// The mask of the 26 bits of a block.
const BLOCK_MASK: u32 = (1 << BLOCK_BITS) - 1;
/// The mask of the validity window.
const WINDOW_MASK: u64 = (1 << WINDOW) - 1;
/// `x^325 mod g(x)`: the syndrome of a single 1 in the last bit of a block (the last row of
/// the parity-check matrix).
const SYNDROME_IN: u16 = 0x31B;
/// `x^351 = x^10 mod g(x)`: the syndrome a 1 contributes when it leaves the 26-bit window.
const SYNDROME_OUT: u16 = GENERATOR & 0x3FF;

/// Description of the `rds_decode` block.
pub static DESCRIPTION: BlockDescription = BlockDescription {
    kind: "rds_decode",
    summary: "Finds the blocks of RDS groups in a stream of soft data bits and outputs the \
        groups as packets.",
    details: "Reads one soft bit per input sample (above 0 is a 1, anything else, NaN \
        included, a 0) into a register of the last 26 bits and computes its syndrome at every \
        bit. Search: a valid block for any offset (A, B, C, C', D) is remembered with its \
        position; the decoder is synchronized when a second valid block follows exactly 26 \
        bits later with the next offset (A B C-or-C' D A). Synchronized: every 26 bits a block \
        is read at the expected position and is valid if its syndrome matches the expected \
        offset (C or C' for the third block). After block D a packet is output: 8 bytes, the \
        information words A, B, C, D big-endian, with CRC_OK when all four blocks are valid, \
        else CRC_FAIL (only with emit_bad); the flag bits 16 to 19 are the validity of the \
        blocks A to D, bit 20 tells that the third block had the offset C' (version B). A \
        group whose block A was not read since the synchronization (the first, partial group) \
        is not output. More than max_bad invalid blocks among the last 50 since the \
        synchronization end it, and the search starts again. The packet index is the input \
        sample index of the first bit of block A, counted since construction or reset. No \
        error correction: a block with a wrong syndrome is only marked invalid.",
    inputs: &[PortDesc {
        name: "in",
        types: &[PortType::F32],
        description: "Soft data bits, one sample per bit (1187.5 bit/s from rds_demod); above \
            0 is a 1.",
    }],
    outputs: &[PortDesc {
        name: "out",
        types: &[PortType::Packets],
        description: "The RDS groups, 8 bytes each, at most one per 104 input samples.",
    }],
    params: &[
        ParamDesc {
            name: "emit_bad",
            unit: Unit::None,
            kind: ParamKind::Bool { default: false },
            typical: "false; true to see damaged groups",
            mutability: Mutability::Build,
            description: "Also output groups with an invalid block, with the flag CRC_FAIL.",
        },
        ParamDesc {
            name: "max_bad",
            unit: Unit::None,
            kind: ParamKind::I64 {
                min: 0,
                max: MAX_BAD_LIMIT as i64,
                default: Some(10),
            },
            typical: "10",
            mutability: Mutability::Build,
            description: "The synchronization ends when more than this many of the last 50 \
                blocks were invalid.",
        },
    ],
    required: &[],
};

/// Configuration of [`RdsDecodeBlock`].
///
/// ```
/// use sdrk_blocks::RdsDecodeConfig;
/// let cfg = RdsDecodeConfig::default();
/// assert_eq!((cfg.emit_bad, cfg.max_bad), (false, 10));
/// assert!(cfg.validate().is_ok());
/// assert!(RdsDecodeConfig { max_bad: 51, ..cfg }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RdsDecodeConfig {
    /// Output groups with an invalid block (flag `CRC_FAIL`).
    pub emit_bad: bool,
    /// Invalid blocks among the last 50 that end the synchronization when exceeded (0 to 50).
    pub max_bad: usize,
}

impl Default for RdsDecodeConfig {
    /// `emit_bad` false, `max_bad` 10.
    fn default() -> Self {
        Self {
            emit_bad: false,
            max_bad: 10,
        }
    }
}

impl RdsDecodeConfig {
    /// Checks the ranges.
    ///
    /// # Errors
    /// `OutOfRange` for `max_bad` above 50.
    pub fn validate(&self) -> Result<()> {
        check_count("max_bad", self.max_bad, 0, MAX_BAD_LIMIT)
    }

    /// Sets a parameter by its description name; checks kind and range.
    ///
    /// # Errors
    /// `UnknownParameter`, `WrongParamType`, `OutOfRange`.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()> {
        let p = DESCRIPTION.param(name).ok_or(Error::UnknownParameter)?;
        p.check(&value)?;
        match p.name {
            "emit_bad" => self.emit_bad = value.as_bool()?,
            "max_bad" => self.max_bad = value.as_usize()?,
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
            "emit_bad" => Ok(ParamValue::Bool(self.emit_bad)),
            "max_bad" => Ok(ParamValue::I64(self.max_bad as i64)),
            _ => unknown(),
        }
    }
}

/// The syndrome of the register after shifting in the bit `bit`, from the syndrome `s` of the
/// register before and the bit `out` that leaves it (bit 25 before the shift): with
/// `S(v) = v(x) x^325 mod g(x)`, `S(v x + bit - out x^26) = x S(v) + bit x^325 - out x^351`.
// Inlined into the per-bit path of the block loop (a call per bit otherwise).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) const fn next_syndrome(s: u16, bit: bool, out: bool) -> u16 {
    let mut t = s << 1;
    if t & 0x400 != 0 {
        t ^= GENERATOR;
    }
    if bit {
        t ^= SYNDROME_IN;
    }
    if out {
        t ^= SYNDROME_OUT;
    }
    t & 0x3FF
}

/// The position of a block in a group (0 to 3) for the expected offsets.
const fn offset_at(position: usize) -> Offset {
    match position {
        0 => Offset::A,
        1 => Offset::B,
        2 => Offset::C,
        _ => Offset::D,
    }
}

/// The group being collected while synchronized.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Collect {
    words: [u16; 4],
    valid: [bool; 4],
    /// Block C' was seen.
    c_prime: bool,
    /// Block A was read (at its position) since the synchronization.
    has_a: bool,
    /// The input index of the first bit of block A.
    index: u64,
}

impl Collect {
    fn set(&mut self, position: usize, word: u16, valid: bool) {
        if let (Some(w), Some(v)) = (self.words.get_mut(position), self.valid.get_mut(position)) {
            *w = word;
            *v = valid;
        }
    }

    fn packet(&self) -> Result<Packet> {
        let mut flags = self
            .valid
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v)
            .fold(0u32, |f, (i, _)| f | (1 << (VALID_SHIFT + i as u32)));
        if self.c_prime {
            flags |= VERSION_B;
        }
        let crc = if self.valid == [true; 4] {
            PacketFlags::CRC_OK
        } else {
            PacketFlags::CRC_FAIL
        };
        let [[a0, a1], [b0, b1], [c0, c1], [d0, d1]] = self.words.map(u16::to_be_bytes);
        Ok(Packet::new(
            self.index,
            PacketFlags(flags) | crc,
            &[a0, a1, b0, b1, c0, c1, d0, d1],
        )?)
    }
}

/// The block synchronization of §3 of M41.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Decoder {
    emit_bad: bool,
    max_bad: u32,
    /// The last 26 bits, the newest in bit 0.
    reg: u32,
    /// The syndrome of `reg`.
    syndrome: u16,
    /// Bits read, up to 26.
    filled: usize,
    /// Search: the valid block (offset and information word) that ended at each of the last
    /// 26 bit positions, indexed by `phase`.
    ring: [Option<(Offset, u16)>; BLOCK_BITS],
    phase: usize,
    /// Synchronized: bits to the end of the next block, its position in the group, the
    /// invalid blocks of the window (bit set = invalid), the group being collected.
    synced: bool,
    countdown: usize,
    position: usize,
    window: u64,
    group: Collect,
}

impl Decoder {
    fn new(config: &RdsDecodeConfig) -> Self {
        Self {
            emit_bad: config.emit_bad,
            max_bad: u32::try_from(config.max_bad).unwrap_or(u32::MAX),
            reg: 0,
            syndrome: 0,
            filled: 0,
            ring: [None; BLOCK_BITS],
            phase: 0,
            synced: false,
            countdown: 0,
            position: 0,
            window: 0,
            group: Collect::default(),
        }
    }

    fn reset(&mut self) {
        *self = Self::new(&RdsDecodeConfig {
            emit_bad: self.emit_bad,
            max_bad: self.max_bad as usize,
        });
    }

    /// Takes the block of offset position `position` that ended with the bit at `index`.
    fn take_block(&mut self, position: usize, word: u16, valid: bool, index: u64) {
        if position == 0 {
            self.group = Collect {
                has_a: true,
                index: index.wrapping_sub(BLOCK_BITS as u64 - 1),
                ..Collect::default()
            };
        }
        self.group.set(position, word, valid);
    }

    /// The packet of a finished group, if it is output.
    fn finish(&mut self) -> Option<Collect> {
        let g = self.group;
        self.group.has_a = false;
        (g.has_a && (self.emit_bad || g.valid == [true; 4])).then_some(g)
    }

    /// One bit at the input index `index`; returns a finished group. The per-bit part is
    /// inlined into the loop of the block; the block ends and the search are not.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn push_bit(&mut self, bit: bool, index: u64) -> Option<Collect> {
        let out = (self.reg >> (BLOCK_BITS - 1)) & 1 != 0;
        self.reg = ((self.reg << 1) | u32::from(bit)) & BLOCK_MASK;
        self.syndrome = next_syndrome(self.syndrome, bit, out);
        if self.synced {
            self.countdown -= 1;
            if self.countdown > 0 {
                return None;
            }
            return self.block_end(index);
        }
        self.search(index)
    }

    /// Synchronized: the block that ended with the bit at `index`.
    #[inline(never)]
    fn block_end(&mut self, index: u64) -> Option<Collect> {
        self.countdown = BLOCK_BITS;
        let word = (self.reg >> 10) as u16;
        let position = self.position;
        let valid = if position == 2 {
            let c_prime = self.syndrome == Offset::CPrime.syndrome();
            self.group.c_prime = c_prime;
            c_prime || self.syndrome == Offset::C.syndrome()
        } else {
            self.syndrome == offset_at(position).syndrome()
        };
        self.take_block(position, word, valid, index);
        self.window = ((self.window << 1) | u64::from(!valid)) & WINDOW_MASK;
        self.position = (position + 1) % 4;
        let done = if position == 3 { self.finish() } else { None };
        if self.window.count_ones() > self.max_bad {
            self.synced = false;
            self.ring = [None; BLOCK_BITS];
            self.group = Collect::default();
        }
        done
    }

    /// Searching: the bit at `index` ended a 26-bit window.
    #[inline(never)]
    fn search(&mut self, index: u64) -> Option<Collect> {
        if self.filled < BLOCK_BITS {
            self.filled += 1;
            if self.filled < BLOCK_BITS {
                self.advance_phase(None);
                return None;
            }
        }
        let word = (self.reg >> 10) as u16;
        let found = Offset::from_syndrome(self.syndrome);
        let before = self.ring.get(self.phase).copied().flatten();
        if let (Some(cur), Some((prev, prev_word))) = (found, before)
            && cur.may_follow(prev)
        {
            self.synced = true;
            self.countdown = BLOCK_BITS;
            self.window = 0;
            self.group = Collect::default();
            if prev == Offset::A {
                self.take_block(0, prev_word, true, index.wrapping_sub(BLOCK_BITS as u64));
            }
            let position = cur.position();
            self.take_block(position, word, true, index);
            if cur == Offset::CPrime {
                self.group.c_prime = true;
            }
            self.position = (position + 1) % 4;
            self.ring = [None; BLOCK_BITS];
            self.phase = 0;
            return if position == 3 { self.finish() } else { None };
        }
        self.advance_phase(found.map(|o| (o, word)));
        None
    }

    fn advance_phase(&mut self, entry: Option<(Offset, u16)>) {
        if let Some(slot) = self.ring.get_mut(self.phase) {
            *slot = entry;
        }
        self.phase = (self.phase + 1) % BLOCK_BITS;
    }
}

/// RDS group decoder: `f32` soft bits in (one per sample, `> 0` is 1), [`Packets`] out (one
/// per group, 8 bytes A B C D big-endian, flags as `sdrk_proto::rds` describes them). The
/// block needs no storage.
///
/// The syndrome of the 26-bit register is updated per bit with
/// `S' = x S + b x^325 - v25 x^351 (mod g)` (`v25` the bit that leaves the register), the
/// same value as `sdrk_proto::rds::syndrome` of the register. The search keeps the valid
/// block of each of the last 26 bit positions; synchronization needs a second valid block 26
/// bits after the first with the next offset. While synchronized a window of the validity of
/// the last 50 blocks (cleared at every synchronization, so the invalid blocks before it do
/// not count) decides when to search again. The `index` of a packet is the input index of the
/// first bit of block A, counted since construction or the last [`Block::reset`]. When a call
/// fills the last free output slot it stops right after the bit that completed that packet;
/// a call with no output room consumes nothing.
///
/// ```
/// use sdrk::Rate;
/// use sdrk_blocks::{ArenaStorage, Block, Packet, RdsDecodeBlock, RdsDecodeConfig};
/// use sdrk_proto::PacketFlags;
/// use sdrk_proto::rds::{self, Group, Offset};
/// let mut arena = ArenaStorage::new(&mut []);
/// let mut dec = RdsDecodeBlock::new(&RdsDecodeConfig::default(), Rate::hz(1187)?, &mut arena)?;
/// // Two groups 0A, one bit per sample.
/// let words = [0xD123, 0x0148, 0xE0E0, u16::from_be_bytes(*b"SD")];
/// let mut soft = Vec::new();
/// for _ in 0..2 {
///     for (w, o) in words.iter().zip([Offset::A, Offset::B, Offset::C, Offset::D]) {
///         let block = rds::block26(*w, o);
///         soft.extend((0..26).rev().map(|i| if (block >> i) & 1 == 1 { 1.0f32 } else { -1.0 }));
///     }
/// }
/// let mut out = [Packet::default(); 4];
/// let p = dec.process(&soft, &mut out)?;
/// assert_eq!((p.consumed, p.produced), (208, 2));
/// assert_eq!(Group::from_packet(&out[0]), Some(Group::new(words)));
/// assert!(out[0].flags.contains(PacketFlags::CRC_OK));
/// assert_eq!((out[0].index, out[1].index), (0, 104));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct RdsDecodeBlock<S: Storage> {
    config: RdsDecodeConfig,
    decoder: Decoder,
    /// Input samples since construction or reset.
    count: u64,
    /// At most one packet per group of 104 input samples.
    max_per_input: Ratio,
    _storage: PhantomData<fn() -> S>,
}

impl<S: Storage> RdsDecodeBlock<S> {
    /// Creates the block; it needs no storage.
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn new(config: &RdsDecodeConfig, _rate_in: Rate, _storage: &mut S) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config: *config,
            decoder: Decoder::new(config),
            count: 0,
            max_per_input: Ratio::new(1, GROUP_BITS as u64)?,
            _storage: PhantomData,
        })
    }

    /// Exact bytes of storage the block needs (zero).
    ///
    /// # Errors
    /// `OutOfRange` for an invalid configuration.
    pub fn storage_bytes(config: &RdsDecodeConfig, _rate_in: Rate) -> Result<usize> {
        config.validate()?;
        Ok(Plan::default().bytes())
    }

    /// Input samples consumed since construction or the last reset.
    pub fn position(&self) -> u64 {
        self.count
    }

    /// Whether the decoder is synchronized to the blocks.
    pub fn synchronized(&self) -> bool {
        self.decoder.synced
    }
}

impl<S: Storage> Block for RdsDecodeBlock<S> {
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
        let start = self.count;
        let mut consumed = input.len();
        let mut produced = 0;
        for (i, &x) in input.iter().enumerate() {
            let index = start.wrapping_add(i as u64);
            if let Some(group) = self.decoder.push_bit(x > 0.0, index) {
                *slot = group.packet()?;
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

#[cfg(test)]
mod tests {
    use super::next_syndrome;
    use sdrk_proto::rds::syndrome;

    /// The per-bit update equals the syndrome of the register (the matrix product) for a
    /// pseudo-random bit stream.
    #[test]
    fn the_running_syndrome_equals_the_matrix_product() {
        let (mut reg, mut s, mut x) = (0u32, 0u16, 0x1234_5678u32);
        for _ in 0..5000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let bit = x & 1 != 0;
            let out = (reg >> 25) & 1 != 0;
            reg = ((reg << 1) | u32::from(bit)) & 0x3FF_FFFF;
            s = next_syndrome(s, bit, out);
            assert_eq!(s, syndrome(reg));
        }
    }
}
