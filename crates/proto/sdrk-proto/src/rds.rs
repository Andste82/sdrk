//! The Radio Data System of FM broadcast (IEC 62106): block checkwords and syndromes, groups
//! and the assembly of the programme service name and the radiotext.
//!
//! An RDS group is four blocks A, B, C (or C') and D. Each block is a 16-bit information word
//! followed by a 10-bit checkword, sent most significant bit first: 26 bits per block, 104 bits
//! per group at 1187.5 bit/s. The checkword is the remainder of `info(x) x^10` modulo the
//! generator `g(x) = x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1` ([`GENERATOR`]) plus the offset
//! word of the block position ([`Offset::word`]), so a receiver also learns where a group
//! starts.
//!
//! [`syndrome`] multiplies a received 26-bit block with the parity-check matrix `H` of the
//! standard. A block is valid for the offset `o` when its syndrome equals [`Offset::syndrome`],
//! the syndrome of `o`'s offset word alone. The syndrome is not the plain remainder of the
//! block modulo `g` (that remainder of a valid block is the offset word itself): the rows of
//! `H` are `x^(25 - i) x^325 mod g`, so `syndrome(v) = v(x) x^325 mod g`, where `x^325` is the
//! inverse of `x^16` modulo `g` (`g` has the order 341).
//!
//! The decoder `rds_decode` of `sdrk-blocks` emits one [`Packet`] per group: 8 bytes, the four
//! information words big-endian, the validity of blocks A to D in the flag bits 16 to 19
//! ([`VALID_SHIFT`]) and [`VERSION_B`] when block C' was seen. [`Group::from_packet`] reads
//! such a packet back and [`RdsState`] collects the text of the groups 0A, 0B, 2A and 2B.
//!
//! ```
//! use sdrk_proto::rds::{self, Offset};
//!
//! let block = rds::block26(0xD123, Offset::A);
//! assert_eq!(block >> 10, 0xD123);
//! assert_eq!(rds::syndrome(block), Offset::A.syndrome());
//! assert_eq!(Offset::from_syndrome(rds::syndrome(block)), Some(Offset::A));
//! // One wrong bit makes the block invalid.
//! assert_eq!(Offset::from_syndrome(rds::syndrome(block ^ (1 << 20))), None);
//! ```

use crate::packet::{Packet, PacketFlags};

/// The generator polynomial `x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1` (bit `k` = `x^k`).
pub const GENERATOR: u16 = 0x5B9;
/// Bits of one block (16 information bits and the 10-bit checkword).
pub const BLOCK_BITS: usize = 26;
/// Bits of one group (four blocks).
pub const GROUP_BITS: usize = 104;
/// Bytes of a group packet: the four information words, big-endian.
pub const GROUP_BYTES: usize = 8;
/// The bit rate of RDS in bit/s.
pub const BIT_RATE_HZ: f64 = 1187.5;
/// The first flag bit of the block validity in a group packet: bit `VALID_SHIFT + i` is set
/// when block `i` (A = 0 to D = 3) was valid.
pub const VALID_SHIFT: u32 = 16;
/// The flag bit of a group packet that tells that the third block had the offset C'.
pub const VERSION_B: u32 = 1 << 20;
/// Characters of the programme service name.
pub const PS_LEN: usize = 8;
/// Characters of the radiotext of group 2A (group 2B carries at most 32).
pub const RT_LEN: usize = 64;

/// The rows of the parity-check matrix `H` of IEC 62106, row `i` for bit `25 - i` of the block.
const H: [u16; BLOCK_BITS] = [
    0x200, 0x100, 0x080, 0x040, 0x020, 0x010, 0x008, 0x004, 0x002, 0x001, 0x2DC, 0x16E, 0x0B7,
    0x287, 0x39F, 0x313, 0x355, 0x376, 0x1BB, 0x201, 0x3DC, 0x1EE, 0x0F7, 0x2A7, 0x38F, 0x31B,
];

/// The position of a block in a group, given by its offset word.
///
/// The set is fixed by the standard (the offset E of the North American variant is not part
/// of RDS), so the enum is exhaustive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Offset {
    /// Block 1 (programme identification).
    A,
    /// Block 2 (group type and the data of the group).
    B,
    /// Block 3 of a version A group.
    C,
    /// Block 3 of a version B group (it repeats the programme identification).
    CPrime,
    /// Block 4.
    D,
}

impl Offset {
    /// The five offsets in the order A, B, C, C', D.
    pub const ALL: [Self; 5] = [Self::A, Self::B, Self::C, Self::CPrime, Self::D];

    /// The 10-bit offset word.
    ///
    /// ```
    /// use sdrk_proto::rds::Offset;
    /// assert_eq!(Offset::A.word(), 0x0FC);
    /// assert_eq!(Offset::CPrime.word(), 0x350);
    /// ```
    pub const fn word(self) -> u16 {
        match self {
            Self::A => 0x0FC,
            Self::B => 0x198,
            Self::C => 0x168,
            Self::CPrime => 0x350,
            Self::D => 0x1B4,
        }
    }

    /// The syndrome of a valid block with this offset (the syndrome of the offset word).
    ///
    /// ```
    /// use sdrk_proto::rds::{self, Offset};
    /// assert_eq!(Offset::A.syndrome(), 0x3D8);
    /// assert_eq!(Offset::D.syndrome(), rds::syndrome(u32::from(Offset::D.word())));
    /// ```
    pub const fn syndrome(self) -> u16 {
        match self {
            Self::A => 0x3D8,
            Self::B => 0x3D4,
            Self::C => 0x25C,
            Self::CPrime => 0x3CC,
            Self::D => 0x258,
        }
    }

    /// The offset with the syndrome `s`, if any.
    ///
    /// ```
    /// use sdrk_proto::rds::Offset;
    /// assert_eq!(Offset::from_syndrome(0x25C), Some(Offset::C));
    /// assert_eq!(Offset::from_syndrome(0), None);
    /// ```
    pub fn from_syndrome(s: u16) -> Option<Self> {
        Self::ALL.into_iter().find(|o| o.syndrome() == s)
    }

    /// The offset of the next block: A to B, B to C (a version B group has C' instead, see
    /// [`Offset::may_follow`]), C and C' to D, D to A.
    ///
    /// ```
    /// use sdrk_proto::rds::Offset;
    /// assert_eq!(Offset::B.next(), Offset::C);
    /// assert_eq!(Offset::CPrime.next(), Offset::D);
    /// assert_eq!(Offset::D.next(), Offset::A);
    /// ```
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::C,
            Self::C | Self::CPrime => Self::D,
            Self::D => Self::A,
        }
    }

    /// Whether a block with this offset may follow a block with the offset `prev`.
    ///
    /// ```
    /// use sdrk_proto::rds::Offset;
    /// assert!(Offset::CPrime.may_follow(Offset::B));
    /// assert!(Offset::C.may_follow(Offset::B));
    /// assert!(!Offset::D.may_follow(Offset::B));
    /// ```
    pub const fn may_follow(self, prev: Self) -> bool {
        matches!(
            (prev, self),
            (Self::A, Self::B)
                | (Self::B, Self::C | Self::CPrime)
                | (Self::C | Self::CPrime, Self::D)
                | (Self::D, Self::A)
        )
    }

    /// The block number in the group: A 0, B 1, C and C' 2, D 3.
    ///
    /// ```
    /// use sdrk_proto::rds::Offset;
    /// assert_eq!(Offset::CPrime.position(), 2);
    /// ```
    pub const fn position(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C | Self::CPrime => 2,
            Self::D => 3,
        }
    }
}

/// The remainder of `v(x)` modulo `g(x)`, for `v` of at most 32 bits.
const fn remainder(v: u32) -> u16 {
    let mut r = v;
    let mut bit = 31u32;
    loop {
        if (r >> bit) & 1 != 0 {
            r ^= (GENERATOR as u32) << (bit - 10);
        }
        if bit == 10 {
            break;
        }
        bit -= 1;
    }
    r as u16
}

/// The 10-bit checkword of the information word `info` at the position `offset`: the remainder
/// of `info(x) x^10` modulo `g(x)`, plus the offset word.
///
/// ```
/// use sdrk_proto::rds::{self, Offset};
/// // A zero information word has the offset word as its checkword.
/// assert_eq!(rds::checkword(0, Offset::B), Offset::B.word());
/// assert_eq!(rds::checkword(0x0001, Offset::A), 0x1B9 ^ 0x0FC);
/// ```
pub const fn checkword(info: u16, offset: Offset) -> u16 {
    remainder((info as u32) << 10) ^ offset.word()
}

/// The 26-bit block of `info` at the position `offset`: `info << 10 | checkword`.
///
/// ```
/// use sdrk_proto::rds::{self, Offset};
/// assert_eq!(rds::block26(0, Offset::D), 0x1B4);
/// ```
pub const fn block26(info: u16, offset: Offset) -> u32 {
    ((info as u32) << 10) | checkword(info, offset) as u32
}

/// The syndrome of a received 26-bit block (bits above bit 25 are ignored): the product with
/// the parity-check matrix `H` of IEC 62106, equal to `block(x) x^325 mod g(x)`.
///
/// ```
/// use sdrk_proto::rds::{self, Offset};
/// for o in Offset::ALL {
///     assert_eq!(rds::syndrome(rds::block26(0x1234, o)), o.syndrome());
/// }
/// ```
pub fn syndrome(block26: u32) -> u16 {
    H.iter()
        .enumerate()
        .filter(|&(i, _)| (block26 >> (BLOCK_BITS - 1 - i)) & 1 != 0)
        .fold(0, |s, (_, row)| s ^ row)
}

/// The version of a group (bit 11 of block B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Version {
    /// Version A: block 3 carries data (offset C).
    A,
    /// Version B: block 3 repeats the programme identification (offset C').
    B,
}

/// One RDS group: the four information words and whether each block was valid.
///
/// The accessors read the raw words; whether a word may be trusted is in `valid`.
///
/// ```
/// use sdrk_proto::rds::{Group, Version};
///
/// // Group 0A of PI 0xD123, PTY 10, segment 0 "SD".
/// let g = Group::new([0xD123, 0x0148, 0xE0E0, u16::from_be_bytes(*b"SD")]);
/// assert_eq!(g.pi(), 0xD123);
/// assert_eq!(g.group_type(), (0, Version::A));
/// assert_eq!(g.pty(), 10);
/// assert!(!g.tp());
/// assert_eq!(Group::from_packet(&g.to_packet(0)?), Some(g));
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Group {
    /// The information words of blocks A, B, C (or C') and D.
    pub blocks: [u16; 4],
    /// Whether each block had a valid syndrome.
    pub valid: [bool; 4],
}

impl Group {
    /// A group of four valid blocks.
    pub const fn new(blocks: [u16; 4]) -> Self {
        Self {
            blocks,
            valid: [true; 4],
        }
    }

    const fn block(&self, i: usize) -> u16 {
        let [w0, w1, w2, w3] = self.blocks;
        match i {
            0 => w0,
            1 => w1,
            2 => w2,
            _ => w3,
        }
    }

    /// The programme identification (block A).
    pub const fn pi(&self) -> u16 {
        self.block(0)
    }

    /// The group type 0 to 15 and the version (bits 15 to 11 of block B).
    pub const fn group_type(&self) -> (u8, Version) {
        let b = self.block(1);
        let version = if b & 0x0800 != 0 {
            Version::B
        } else {
            Version::A
        };
        ((b >> 12) as u8, version)
    }

    /// The traffic programme flag (bit 10 of block B).
    pub const fn tp(&self) -> bool {
        self.block(1) & 0x0400 != 0
    }

    /// The programme type 0 to 31 (bits 9 to 5 of block B).
    pub const fn pty(&self) -> u8 {
        ((self.block(1) >> 5) & 0x1F) as u8
    }

    /// The validity flag bits of a group packet (bit `VALID_SHIFT + i` for a valid block `i`).
    ///
    /// ```
    /// use sdrk_proto::rds::Group;
    /// let mut g = Group::new([0; 4]);
    /// g.valid[2] = false;
    /// assert_eq!(g.validity_bits(), 0b1011 << 16);
    /// ```
    pub fn validity_bits(&self) -> u32 {
        self.valid
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v)
            .fold(0, |f, (i, _)| f | (1 << (VALID_SHIFT + i as u32)))
    }

    /// The group of a packet of `rds_decode`: 8 bytes (the four words big-endian), the
    /// validity from the flag bits 16 to 19. `None` if the payload is not 8 bytes long.
    pub fn from_packet(packet: &Packet) -> Option<Self> {
        let d: [u8; GROUP_BYTES] = packet.data().try_into().ok()?;
        let [a0, a1, b0, b1, c0, c1, d0, d1] = d;
        let flags = packet.flags.bits();
        let valid = |i: u32| (flags >> (VALID_SHIFT + i)) & 1 != 0;
        Some(Self {
            blocks: [
                u16::from_be_bytes([a0, a1]),
                u16::from_be_bytes([b0, b1]),
                u16::from_be_bytes([c0, c1]),
                u16::from_be_bytes([d0, d1]),
            ],
            valid: [valid(0), valid(1), valid(2), valid(3)],
        })
    }

    /// The four words big-endian, the payload of a group packet.
    ///
    /// ```
    /// use sdrk_proto::rds::Group;
    /// let g = Group::new([0x0102, 0x0304, 0x0506, 0x0708]);
    /// assert_eq!(g.to_bytes(), [1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    pub fn to_bytes(&self) -> [u8; GROUP_BYTES] {
        let [[a0, a1], [b0, b1], [c0, c1], [d0, d1]] = self.blocks.map(u16::to_be_bytes);
        [a0, a1, b0, b1, c0, c1, d0, d1]
    }

    /// A group packet with the index `index`: the bytes of [`Group::to_bytes`], the validity
    /// bits, `CRC_OK` when all four blocks are valid (else `CRC_FAIL`) and [`VERSION_B`] for a
    /// version B group whose block B is valid.
    ///
    /// # Errors
    /// None in practice (8 bytes always fit a packet); the `Result` is that of [`Packet::new`].
    pub fn to_packet(&self, index: u64) -> crate::Result<Packet> {
        let mut flags = PacketFlags(self.validity_bits());
        flags |= if self.valid == [true; 4] {
            PacketFlags::CRC_OK
        } else {
            PacketFlags::CRC_FAIL
        };
        let [_, b_ok, _, _] = self.valid;
        if b_ok && self.group_type().1 == Version::B {
            flags |= PacketFlags(VERSION_B);
        }
        Packet::new(index, flags, &self.to_bytes())
    }
}

/// The text character of an RDS byte: 0x20 to 0x7E are ASCII, everything else is `?` (the
/// full RDS character table is not implemented).
///
/// ```
/// use sdrk_proto::rds::map_char;
/// assert_eq!(map_char(b'A'), b'A');
/// assert_eq!(map_char(0x8E), b'?');
/// ```
pub const fn map_char(b: u8) -> u8 {
    if b >= 0x20 && b <= 0x7E { b } else { b'?' }
}

/// The carriage return that ends a radiotext.
const CR: u8 = 0x0D;

/// Segments of the radiotext (both versions).
const RT_SEGMENTS: usize = 16;

/// The programme service name and the radiotext of one station, assembled from groups.
///
/// [`RdsState::update`] takes the groups in the order they arrive:
///
/// - The programme identification of a group is block A when it is valid, else block C' of a
///   version B group with valid blocks B and C'. A group without a readable identification is
///   ignored; a new identification clears the state (another station).
/// - Groups 0A and 0B (blocks B and D valid) carry two characters of the name at the address
///   in the low two bits of block B. [`RdsState::ps`] gives the 8 characters once all four
///   segments arrived.
/// - Groups 2A (blocks B, C and D valid) carry four characters of a 64-character radiotext,
///   groups 2B (blocks B and D valid) two characters of a 32-character one, at the address in
///   the low four bits of block B. A change of the A/B flag (bit 4 of block B) or of the version
///   clears the text. The text ends at the first carriage return (0x0D); [`RdsState::rt`]
///   gives it once every segment up to the one with the carriage return arrived (or all 16
///   segments when there is none).
/// - Characters 0x20 to 0x7E are kept, others become `?` ([`map_char`]).
///
/// ```
/// use sdrk_proto::rds::{Group, RdsState};
///
/// let mut s = RdsState::new();
/// for (a, pair) in [*b"SD", *b"RK", *b" T", *b"ST"].into_iter().enumerate() {
///     assert_eq!(s.ps(), "");
///     s.update(&Group::new([0xD123, 0x0148 | a as u16, 0xE0E0, u16::from_be_bytes(pair)]));
/// }
/// assert_eq!(s.ps(), "SDRK TST");
/// assert_eq!(s.pi(), Some(0xD123));
/// assert_eq!(s.pty(), Some(10));
/// // 2A segment 0 with the carriage return completes a short text.
/// s.update(&Group::new([0xD123, 0x2140, u16::from_be_bytes(*b"Hi"), 0x0D20]));
/// assert_eq!(s.rt(), "Hi");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdsState {
    pi: Option<u16>,
    pty: Option<u8>,
    tp: Option<bool>,
    ps: [u8; PS_LEN],
    ps_seen: u8,
    rt: [u8; RT_LEN],
    rt_seen: u16,
    /// Version and A/B flag of the text in `rt`.
    rt_kind: Option<(Version, bool)>,
}

impl Default for RdsState {
    fn default() -> Self {
        Self::new()
    }
}

impl RdsState {
    /// An empty state.
    pub const fn new() -> Self {
        Self {
            pi: None,
            pty: None,
            tp: None,
            ps: [b' '; PS_LEN],
            ps_seen: 0,
            rt: [b' '; RT_LEN],
            rt_seen: 0,
            rt_kind: None,
        }
    }

    /// Forgets everything.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// The programme identification of the station, once a group was read.
    pub const fn pi(&self) -> Option<u16> {
        self.pi
    }

    /// The programme type of the last group with a valid block B.
    pub const fn pty(&self) -> Option<u8> {
        self.pty
    }

    /// The traffic programme flag of the last group with a valid block B.
    pub const fn tp(&self) -> Option<bool> {
        self.tp
    }

    /// Whether all four segments of the programme service name arrived.
    pub const fn ps_ready(&self) -> bool {
        self.ps_seen == 0x0F
    }

    /// The programme service name (8 characters), or `""` until all four segments arrived.
    pub fn ps(&self) -> &str {
        if self.ps_ready() {
            core::str::from_utf8(&self.ps).unwrap_or("")
        } else {
            ""
        }
    }

    /// The characters per radiotext segment of the current text.
    fn rt_chars(&self) -> usize {
        match self.rt_kind {
            Some((Version::B, _)) => 2,
            _ => 4,
        }
    }

    /// The length of the radiotext when it is complete.
    fn rt_len(&self) -> Option<usize> {
        self.rt_kind?;
        let n = self.rt_chars();
        for seg in 0..RT_SEGMENTS {
            if (self.rt_seen >> seg) & 1 == 0 {
                return None;
            }
            let start = seg * n;
            let chars = self.rt.get(start..start + n)?;
            if let Some(p) = chars.iter().position(|&c| c == CR) {
                return Some(start + p);
            }
        }
        Some(RT_SEGMENTS * n)
    }

    /// Whether the radiotext is complete (see [`RdsState`]).
    pub fn rt_ready(&self) -> bool {
        self.rt_len().is_some()
    }

    /// The radiotext up to its carriage return, or `""` until it is complete.
    pub fn rt(&self) -> &str {
        self.rt_len()
            .and_then(|n| self.rt.get(..n))
            .and_then(|t| core::str::from_utf8(t).ok())
            .unwrap_or("")
    }

    /// The programme identification of `g`, if it can be read.
    fn pi_of(g: &Group) -> Option<u16> {
        let [a_ok, b_ok, c_ok, _] = g.valid;
        if a_ok {
            Some(g.pi())
        } else if b_ok && c_ok && g.group_type().1 == Version::B {
            Some(g.block(2))
        } else {
            None
        }
    }

    /// Takes one group (see [`RdsState`] for the rules).
    pub fn update(&mut self, group: &Group) {
        let Some(pi) = Self::pi_of(group) else {
            return;
        };
        if self.pi.is_some_and(|p| p != pi) {
            self.reset();
        }
        self.pi = Some(pi);
        let [_, b_ok, c_ok, d_ok] = group.valid;
        if !b_ok {
            return;
        }
        self.pty = Some(group.pty());
        self.tp = Some(group.tp());
        let b = group.block(1);
        let [d0, d1] = group.block(3).to_be_bytes();
        match group.group_type() {
            (0, _) if d_ok => {
                let at = usize::from(b & 0x3) * 2;
                if let Some(dst) = self.ps.get_mut(at..at + 2) {
                    dst.copy_from_slice(&[map_char(d0), map_char(d1)]);
                    self.ps_seen |= 1 << (b & 0x3);
                }
            }
            (2, version) if d_ok && (version == Version::B || c_ok) => {
                let kind = (version, b & 0x10 != 0);
                if self.rt_kind != Some(kind) {
                    self.rt = [b' '; RT_LEN];
                    self.rt_seen = 0;
                    self.rt_kind = Some(kind);
                }
                let seg = usize::from(b & 0xF);
                let [c0, c1] = group.block(2).to_be_bytes();
                let text = |c: u8| if c == CR { CR } else { map_char(c) };
                if version == Version::A {
                    if let Some(dst) = self.rt.get_mut(seg * 4..seg * 4 + 4) {
                        dst.copy_from_slice(&[text(c0), text(c1), text(d0), text(d1)]);
                    }
                } else if let Some(dst) = self.rt.get_mut(seg * 2..seg * 2 + 2) {
                    dst.copy_from_slice(&[text(d0), text(d1)]);
                }
                self.rt_seen |= 1 << seg;
            }
            _ => {}
        }
    }
}
