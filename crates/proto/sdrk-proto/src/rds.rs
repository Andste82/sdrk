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
//! The text is in the basic RDS character set of IEC 62106, Annex E (formerly EN 50067,
//! Annex E): ASCII with a few exceptions below 0x80, accented letters and symbols above.
//! [`decode_char`] and [`encode_char`] map one byte, [`RdsText`] reads a whole name or text
//! (in the RDS set, in UTF-8 for stations that send it, or detected with [`Charset::Auto`])
//! into ASCII or UTF-8 in caller storage, and [`encode_text`] turns a string into RDS bytes.
//! The table is 256 bytes of flash (the code points of 0x80 to 0xFF as `u16`).
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

use core::fmt::{self, Write as _};

use crate::packet::{Packet, PacketFlags};
use crate::{Error, Result};

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

/// The plain ASCII view of an RDS byte used by [`RdsState::ps`] and [`RdsState::rt`]: 0x20 to
/// 0x7E are kept, everything else is `?`.
///
/// This is not the RDS character set: four of the kept bytes are other characters there (0x24
/// `¤`, 0x5E `―`, 0x60 `║`, 0x7E `¯`) and the bytes from 0x80 on are accented letters and
/// symbols. [`decode_char`] and [`RdsText`] give the real characters.
///
/// ```
/// use sdrk_proto::rds::map_char;
/// assert_eq!(map_char(b'A'), b'A');
/// assert_eq!(map_char(0x8E), b'?');
/// ```
pub const fn map_char(b: u8) -> u8 {
    if b >= 0x20 && b <= 0x7E { b } else { b'?' }
}

/// The code points of the bytes 0x80 to 0xFF of the basic RDS character set (IEC 62106,
/// Annex E, Figure E.1), one row of the figure per line; 0xFF is no character and decodes to a
/// space. The bytes below 0x80 are ASCII with five exceptions, handled in [`decode_char`].
#[rustfmt::skip]
static HIGH: [u16; 128] = [
    // 0x80: á à é è í ì ó ò ú ù Ñ Ç Ş ß ¡ Ĳ
    0x00E1, 0x00E0, 0x00E9, 0x00E8, 0x00ED, 0x00EC, 0x00F3, 0x00F2,
    0x00FA, 0x00F9, 0x00D1, 0x00C7, 0x015E, 0x00DF, 0x00A1, 0x0132,
    // 0x90: â ä ê ë î ï ô ö û ü ñ ç ş ğ ı ĳ
    0x00E2, 0x00E4, 0x00EA, 0x00EB, 0x00EE, 0x00EF, 0x00F4, 0x00F6,
    0x00FB, 0x00FC, 0x00F1, 0x00E7, 0x015F, 0x011F, 0x0131, 0x0133,
    // 0xA0: ª α © ‰ Ğ ě ň ő π € £ $ ← ↑ → ↓
    0x00AA, 0x03B1, 0x00A9, 0x2030, 0x011E, 0x011B, 0x0148, 0x0151,
    0x03C0, 0x20AC, 0x00A3, 0x0024, 0x2190, 0x2191, 0x2192, 0x2193,
    // 0xB0: º ¹ ² ³ ± İ ń ű µ ¿ ÷ ° ¼ ½ ¾ §
    0x00BA, 0x00B9, 0x00B2, 0x00B3, 0x00B1, 0x0130, 0x0144, 0x0171,
    0x00B5, 0x00BF, 0x00F7, 0x00B0, 0x00BC, 0x00BD, 0x00BE, 0x00A7,
    // 0xC0: Á À É È Í Ì Ó Ò Ú Ù Ř Č Š Ž Ð Ŀ
    0x00C1, 0x00C0, 0x00C9, 0x00C8, 0x00CD, 0x00CC, 0x00D3, 0x00D2,
    0x00DA, 0x00D9, 0x0158, 0x010C, 0x0160, 0x017D, 0x00D0, 0x013F,
    // 0xD0: Â Ä Ê Ë Î Ï Ô Ö Û Ü ř č š ž đ ŀ
    0x00C2, 0x00C4, 0x00CA, 0x00CB, 0x00CE, 0x00CF, 0x00D4, 0x00D6,
    0x00DB, 0x00DC, 0x0159, 0x010D, 0x0161, 0x017E, 0x0111, 0x0140,
    // 0xE0: Ã Å Æ Œ ŷ Ý Õ Ø Þ Ŋ Ŕ Ć Ś Ź Ŧ ð
    0x00C3, 0x00C5, 0x00C6, 0x0152, 0x0177, 0x00DD, 0x00D5, 0x00D8,
    0x00DE, 0x014A, 0x0154, 0x0106, 0x015A, 0x0179, 0x0166, 0x00F0,
    // 0xF0: ã å æ œ ŵ ý õ ø þ ŋ ŕ ć ś ź ŧ, 0xFF space
    0x00E3, 0x00E5, 0x00E6, 0x0153, 0x0175, 0x00FD, 0x00F5, 0x00F8,
    0x00FE, 0x014B, 0x0155, 0x0107, 0x015B, 0x017A, 0x0167, 0x0020,
];

/// The character of a byte in the basic RDS character set of IEC 62106, Annex E, Figure E.1
/// (formerly EN 50067, Annex E), the set of the programme service name and the radiotext.
///
/// - 0x20 to 0x7D are ASCII except 0x24, the currency sign `¤` (U+00A4; the dollar sign is
///   0xAB), 0x5E, the horizontal bar `―` (U+2015), and 0x60, the double vertical line `║`
///   (U+2551); 0x7E is the macron `¯` (U+00AF).
/// - 0x80 to 0xFE are the accented letters, the Greek letters, the symbols and the arrows of
///   the figure.
/// - The control codes 0x00 to 0x1F and 0x7F and the unused 0xFF decode to a space. Four of
///   them mean something in a radiotext: 0x0A a preferred line break, 0x0B the end of a
///   headline, 0x0D the end of the text (where [`RdsState`] ends the radiotext) and 0x1F a
///   soft hyphen. A caller that wants them reads the raw bytes ([`RdsText::bytes`]).
///
/// ```
/// use sdrk_proto::rds::decode_char;
/// assert_eq!(decode_char(b'A'), 'A');
/// assert_eq!(decode_char(0x99), 'ü');
/// assert_eq!(decode_char(0x8D), 'ß');
/// assert_eq!(decode_char(0xAB), '$');
/// assert_eq!(decode_char(0x24), '¤');
/// assert_eq!(decode_char(0x0D), ' ');
/// ```
pub const fn decode_char(b: u8) -> char {
    let code = match b {
        0x24 => 0x00A4,
        0x5E => 0x2015,
        0x60 => 0x2551,
        0x7E => 0x00AF,
        0x20..=0x7D => b as u32,
        #[allow(clippy::indexing_slicing)] // `b - 0x80 < 128`
        0x80..=0xFF => HIGH[(b - 0x80) as usize] as u32,
        _ => 0x20,
    };
    match char::from_u32(code) {
        Some(c) => c,
        None => ' ',
    }
}

/// The byte of `c` in the basic RDS character set ([`decode_char`]), `None` if the set has no
/// such character.
///
/// The space is 0x20 (the control codes that also decode to a space are never chosen), the
/// dollar sign `$` is 0xAB, and the ASCII characters `^`, `` ` `` and `~` have no code (their
/// places hold `―`, `║` and `¯`). Every byte except the control codes and 0xFF is the code
/// of exactly one character, so `encode_char(decode_char(b)) == Some(b)` for them.
///
/// ```
/// use sdrk_proto::rds::{decode_char, encode_char};
/// assert_eq!(encode_char('ü'), Some(0x99));
/// assert_eq!(encode_char('$'), Some(0xAB));
/// assert_eq!(encode_char(' '), Some(0x20));
/// assert_eq!(encode_char('~'), None);
/// assert_eq!(decode_char(encode_char('€').unwrap_or(b'?')), '€');
/// ```
pub fn encode_char(c: char) -> Option<u8> {
    match c {
        '$' => Some(0xAB),
        '^' | '`' | '~' => None,
        ' '..='}' => Some(c as u8),
        '\u{A4}' => Some(0x24),
        '\u{2015}' => Some(0x5E),
        '\u{2551}' => Some(0x60),
        '\u{AF}' => Some(0x7E),
        _ => {
            let code = u32::from(c);
            HIGH.iter()
                .take(127)
                .position(|&u| u32::from(u) == code)
                .map(|i| 0x80 + i as u8)
        }
    }
}

/// How the bytes of an [`RdsText`] are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Charset {
    /// The basic RDS character set of the standard ([`decode_char`]).
    #[default]
    Rds,
    /// UTF-8, which some stations send against the standard. An invalid sequence becomes the
    /// replacement character U+FFFD and the ASCII control codes become spaces, as in the RDS
    /// table.
    Utf8,
    /// UTF-8 when the bytes contain a multi-byte sequence (a byte of 0x80 or more) and are
    /// valid UTF-8 as a whole, else the RDS character set. A text of ASCII bytes only is read
    /// with the RDS table.
    Auto,
}

/// A programme service name or a radiotext as raw bytes, with the character set to read them
/// in ([`Charset::Rds`] unless [`RdsText::with_charset`] chose another).
///
/// [`RdsState::ps_text`] and [`RdsState::rt_text`] give one; [`RdsText::new`] wraps any bytes.
/// Nothing is copied: the conversions write into caller storage.
///
/// ```
/// use sdrk_proto::rds::{Charset, RdsText};
///
/// // "Württemberg" in the RDS character set (0x99 is ü).
/// let raw = b"W\x99rttemberg";
/// let text = RdsText::new(raw);
/// assert_eq!(text.chars().nth(1), Some('ü'));
/// let mut buf = [0u8; 32];
/// assert_eq!(text.to_utf8(&mut buf)?, "Württemberg");
/// assert_eq!(text.to_ascii(&mut buf)?, "W?rttemberg");
/// // A station that sends UTF-8: `Auto` notices it.
/// let utf8 = RdsText::new("Württemberg".as_bytes()).with_charset(Charset::Auto);
/// assert_eq!(utf8.charset(), Charset::Utf8);
/// assert_eq!(utf8.to_utf8(&mut buf)?, "Württemberg");
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RdsText<'a> {
    bytes: &'a [u8],
    utf8: bool,
}

impl<'a> RdsText<'a> {
    /// The text of `bytes` in the RDS character set.
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, utf8: false }
    }

    /// The same bytes read with `charset`; [`Charset::Auto`] decides here, from the bytes.
    ///
    /// ```
    /// use sdrk_proto::rds::{Charset, RdsText};
    /// // A lone 0xC3 is no valid UTF-8, so `Auto` keeps the RDS table (0xC3 is È).
    /// let t = RdsText::new(b"\xC3 ok").with_charset(Charset::Auto);
    /// assert_eq!(t.charset(), Charset::Rds);
    /// assert_eq!(t.chars().next(), Some('È'));
    /// ```
    #[must_use]
    pub fn with_charset(self, charset: Charset) -> Self {
        let utf8 = match charset {
            Charset::Utf8 => true,
            Charset::Auto => {
                self.bytes.iter().any(|&b| b >= 0x80) && core::str::from_utf8(self.bytes).is_ok()
            }
            _ => false,
        };
        Self { utf8, ..self }
    }

    /// The character set the text is read in: [`Charset::Rds`] or [`Charset::Utf8`], never
    /// [`Charset::Auto`].
    pub const fn charset(&self) -> Charset {
        if self.utf8 {
            Charset::Utf8
        } else {
            Charset::Rds
        }
    }

    /// The raw bytes as received.
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// The characters of the text.
    pub fn chars(&self) -> RdsChars<'a> {
        let inner = if self.utf8 {
            CharsInner::Utf8 {
                chunks: self.bytes.utf8_chunks(),
                valid: "".chars(),
                invalid: false,
            }
        } else {
            CharsInner::Rds(self.bytes.iter())
        };
        RdsChars { inner }
    }

    /// The number of characters, the room [`RdsText::to_ascii`] needs (in the RDS character
    /// set the number of bytes).
    pub fn char_count(&self) -> usize {
        if self.utf8 {
            self.chars().count()
        } else {
            self.bytes.len()
        }
    }

    /// The length of the text in UTF-8, the room [`RdsText::to_utf8`] needs.
    pub fn utf8_len(&self) -> usize {
        self.chars().map(char::len_utf8).sum()
    }

    /// The text in ASCII, one byte per character: characters outside ASCII become `?`. Fails
    /// with `BufferTooSmall` when `out` is shorter than [`RdsText::char_count`] (nothing is
    /// truncated).
    ///
    /// ```
    /// use sdrk_proto::{Error, rds::RdsText};
    /// let text = RdsText::new(b"gr\x97\x8Dten");
    /// let mut buf = [0u8; 8];
    /// assert_eq!(text.to_ascii(&mut buf)?, "gr??ten");
    /// assert_eq!(text.to_ascii(&mut buf[..6]), Err(Error::BufferTooSmall { needed: 7, got: 6 }));
    /// # Ok::<(), Error>(())
    /// ```
    pub fn to_ascii<'b>(&self, out: &'b mut [u8]) -> Result<&'b str> {
        let needed = self.char_count();
        let got = out.len();
        let dst = out
            .get_mut(..needed)
            .ok_or(Error::BufferTooSmall { needed, got })?;
        for (o, c) in dst.iter_mut().zip(self.chars()) {
            *o = if c.is_ascii() { c as u8 } else { b'?' };
        }
        core::str::from_utf8(dst).map_err(|_| Error::InvalidArgument("text is not ascii"))
    }

    /// The text in UTF-8. Fails with `BufferTooSmall` when `out` is shorter than
    /// [`RdsText::utf8_len`] (nothing is truncated).
    ///
    /// ```
    /// use sdrk_proto::{Error, rds::RdsText};
    /// let text = RdsText::new(b"gr\x97\x8Dten");
    /// let mut buf = [0u8; 16];
    /// assert_eq!(text.to_utf8(&mut buf)?, "größten");
    /// assert_eq!(text.utf8_len(), 9);
    /// assert_eq!(text.to_utf8(&mut buf[..8]), Err(Error::BufferTooSmall { needed: 9, got: 8 }));
    /// # Ok::<(), Error>(())
    /// ```
    pub fn to_utf8<'b>(&self, out: &'b mut [u8]) -> Result<&'b str> {
        let needed = self.utf8_len();
        let got = out.len();
        let dst = out
            .get_mut(..needed)
            .ok_or(Error::BufferTooSmall { needed, got })?;
        let mut at = 0;
        for c in self.chars() {
            let len = c.len_utf8();
            let slot = dst
                .get_mut(at..at + len)
                .ok_or(Error::BufferTooSmall { needed, got })?;
            c.encode_utf8(slot);
            at += len;
        }
        core::str::from_utf8(dst).map_err(|_| Error::InvalidArgument("text is not utf-8"))
    }
}

/// Writes the characters of the text (as [`RdsText::chars`]).
///
/// ```
/// use sdrk_proto::rds::RdsText;
/// assert_eq!(format!("{}", RdsText::new(b"Caf\x82")), "Café");
/// ```
impl fmt::Display for RdsText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.chars().try_for_each(|c| f.write_char(c))
    }
}

/// The characters of an [`RdsText`], from [`RdsText::chars`].
#[derive(Debug, Clone)]
pub struct RdsChars<'a> {
    inner: CharsInner<'a>,
}

#[derive(Debug, Clone)]
enum CharsInner<'a> {
    Rds(core::slice::Iter<'a, u8>),
    Utf8 {
        chunks: core::str::Utf8Chunks<'a>,
        valid: core::str::Chars<'a>,
        /// Whether the current chunk ends with an invalid sequence.
        invalid: bool,
    },
}

impl Iterator for RdsChars<'_> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        match &mut self.inner {
            CharsInner::Rds(bytes) => bytes.next().map(|&b| decode_char(b)),
            CharsInner::Utf8 {
                chunks,
                valid,
                invalid,
            } => loop {
                if let Some(c) = valid.next() {
                    return Some(if c.is_ascii_control() { ' ' } else { c });
                }
                if *invalid {
                    *invalid = false;
                    return Some(char::REPLACEMENT_CHARACTER);
                }
                let chunk = chunks.next()?;
                *valid = chunk.valid().chars();
                *invalid = !chunk.invalid().is_empty();
            },
        }
    }
}

impl core::iter::FusedIterator for RdsChars<'_> {}

/// Encodes `text` in the basic RDS character set ([`encode_char`]) into `out` and returns the
/// number of bytes written, one per character.
///
/// # Errors
/// `BufferTooSmall` when `out` has fewer bytes than `text` has characters (nothing is
/// written); `InvalidArgument` for a character the set does not have (`^`, `` ` ``, `~`,
/// most of Unicode; the bytes before it are written). A caller that prefers a replacement
/// maps the characters with [`encode_char`] itself.
///
/// ```
/// use sdrk_proto::{Error, rds::{self, RdsText}};
/// let mut buf = [0u8; 8];
/// let n = rds::encode_text("Grüße", &mut buf)?;
/// assert_eq!(&buf[..n], b"Gr\x99\x8De");
/// assert_eq!(RdsText::new(&buf[..n]).to_string(), "Grüße");
/// assert!(rds::encode_text("a~b", &mut buf).is_err());
/// assert_eq!(rds::encode_text("123456789", &mut buf), Err(Error::BufferTooSmall { needed: 9, got: 8 }));
/// # Ok::<(), Error>(())
/// ```
pub fn encode_text(text: &str, out: &mut [u8]) -> Result<usize> {
    let needed = text.chars().count();
    let got = out.len();
    let dst = out
        .get_mut(..needed)
        .ok_or(Error::BufferTooSmall { needed, got })?;
    for (o, c) in dst.iter_mut().zip(text.chars()) {
        *o = encode_char(c).ok_or(Error::InvalidArgument(
            "character not in the rds character set",
        ))?;
    }
    Ok(needed)
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
/// - The raw bytes are kept. [`RdsState::ps`] and [`RdsState::rt`] give the plain ASCII view
///   (0x20 to 0x7E kept, others `?`, [`map_char`]); [`RdsState::ps_text`] and
///   [`RdsState::rt_text`] give the bytes as an [`RdsText`] for the real characters of the RDS
///   character set ([`decode_char`]) or UTF-8.
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
///
/// // The characters of the RDS character set: 0x99 is ü.
/// s.update(&Group::new([0xD123, 0x2150, u16::from_be_bytes(*b"Gr"), 0x990D]));
/// assert_eq!(s.rt(), "Gr?");
/// let mut buf = [0u8; 2 * sdrk_proto::rds::RT_LEN];
/// assert_eq!(s.rt_text().map(|t| t.to_utf8(&mut buf)), Some(Ok("Grü")));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdsState {
    pi: Option<u16>,
    pty: Option<u8>,
    tp: Option<bool>,
    /// The name as received.
    ps_raw: [u8; PS_LEN],
    /// The name through [`map_char`], for [`RdsState::ps`].
    ps: [u8; PS_LEN],
    ps_seen: u8,
    /// The radiotext as received.
    rt_raw: [u8; RT_LEN],
    /// The radiotext through [`map_char`] (the carriage return kept), for [`RdsState::rt`].
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
            ps_raw: [b' '; PS_LEN],
            ps: [b' '; PS_LEN],
            ps_seen: 0,
            rt_raw: [b' '; RT_LEN],
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
    /// Bytes outside 0x20 to 0x7E are `?` ([`map_char`]); [`RdsState::ps_text`] has the real
    /// characters.
    pub fn ps(&self) -> &str {
        if self.ps_ready() {
            core::str::from_utf8(&self.ps).unwrap_or("")
        } else {
            ""
        }
    }

    /// The programme service name as received (8 bytes), once all four segments arrived.
    ///
    /// ```
    /// use sdrk_proto::rds::{Group, RdsState};
    /// let mut s = RdsState::new();
    /// for (a, pair) in [*b"SW", *b"R1", [b' ', 0x99], *b"rt"].into_iter().enumerate() {
    ///     assert!(s.ps_text().is_none());
    ///     s.update(&Group::new([0xD3C1, 0x0048 | a as u16, 0xE0E0, u16::from_be_bytes(pair)]));
    /// }
    /// assert_eq!(s.ps(), "SWR1 ?rt");
    /// let ps = s.ps_text().unwrap_or(sdrk_proto::rds::RdsText::new(b""));
    /// assert_eq!(ps.to_string(), "SWR1 ürt");
    /// assert_eq!(ps.bytes(), b"SWR1 \x99rt");
    /// ```
    pub fn ps_text(&self) -> Option<RdsText<'_>> {
        self.ps_ready().then(|| RdsText::new(&self.ps_raw))
    }

    /// The programme service name as far as it arrived, for a display that fills while the
    /// segments come in: the 8 raw bytes with spaces for segments not received yet; `None`
    /// before the first segment. Equal to [`RdsState::ps_text`] once complete.
    ///
    /// ```
    /// use sdrk_proto::rds::{Group, RdsState};
    ///
    /// let mut s = RdsState::new();
    /// assert!(s.ps_partial().is_none());
    /// s.update(&Group::new([0xD123, 0x0141, 0xE0E0, u16::from_be_bytes(*b"RK")]));
    /// assert_eq!(s.ps_partial().map(|t| t.to_string()).as_deref(), Some("  RK    "));
    /// assert!(s.ps_text().is_none());
    /// ```
    #[must_use]
    pub fn ps_partial(&self) -> Option<RdsText<'_>> {
        (self.ps_seen != 0).then(|| RdsText::new(&self.ps_raw))
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

    /// The radiotext up to its carriage return, or `""` until it is complete. Bytes outside
    /// 0x20 to 0x7E are `?` ([`map_char`]); [`RdsState::rt_text`] has the real characters.
    pub fn rt(&self) -> &str {
        self.rt_len()
            .and_then(|n| self.rt.get(..n))
            .and_then(|t| core::str::from_utf8(t).ok())
            .unwrap_or("")
    }

    /// The radiotext as received, up to its carriage return (not included), once it is
    /// complete (see [`RdsState`]).
    pub fn rt_text(&self) -> Option<RdsText<'_>> {
        self.rt_len()
            .and_then(|n| self.rt_raw.get(..n))
            .map(RdsText::new)
    }

    /// The radiotext as far as it arrived, for a display that fills while the segments come
    /// in: the raw bytes up to the first carriage return in a received segment (else the full
    /// 64 or 32 characters of the version), with spaces for segments not received yet; `None`
    /// before the first segment of the current text. Equal to [`RdsState::rt_text`] once
    /// complete.
    ///
    /// ```
    /// use sdrk_proto::rds::{Group, RdsState};
    ///
    /// let mut s = RdsState::new();
    /// // 2A segment 1 ("o fr") before segment 0 ("Hell").
    /// s.update(&Group::new([0xD123, 0x2141, u16::from_be_bytes(*b"o "), u16::from_be_bytes(*b"fr")]));
    /// let partial = s.rt_partial().map(|t| t.to_string()).unwrap_or_default();
    /// assert!(partial.starts_with("    o fr"));
    /// assert!(s.rt_text().is_none());
    /// ```
    #[must_use]
    pub fn rt_partial(&self) -> Option<RdsText<'_>> {
        self.rt_kind?;
        if self.rt_seen == 0 {
            return None;
        }
        let n = self.rt_chars();
        let end = (0..RT_SEGMENTS)
            .filter(|seg| (self.rt_seen >> seg) & 1 == 1)
            .find_map(|seg| {
                let start = seg * n;
                self.rt
                    .get(start..start + n)?
                    .iter()
                    .position(|&c| c == CR)
                    .map(|p| start + p)
            })
            .unwrap_or(RT_SEGMENTS * n);
        self.rt_raw.get(..end).map(RdsText::new)
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
                if let (Some(raw), Some(dst)) =
                    (self.ps_raw.get_mut(at..at + 2), self.ps.get_mut(at..at + 2))
                {
                    raw.copy_from_slice(&[d0, d1]);
                    dst.copy_from_slice(&[map_char(d0), map_char(d1)]);
                    self.ps_seen |= 1 << (b & 0x3);
                }
            }
            (2, version) if d_ok && (version == Version::B || c_ok) => {
                let kind = (version, b & 0x10 != 0);
                if self.rt_kind != Some(kind) {
                    self.rt_raw = [b' '; RT_LEN];
                    self.rt = [b' '; RT_LEN];
                    self.rt_seen = 0;
                    self.rt_kind = Some(kind);
                }
                let seg = usize::from(b & 0xF);
                let [c0, c1] = group.block(2).to_be_bytes();
                let text = |c: u8| if c == CR { CR } else { map_char(c) };
                let (at, raw): (usize, &[u8]) = if version == Version::A {
                    (seg * 4, &[c0, c1, d0, d1])
                } else {
                    (seg * 2, &[d0, d1])
                };
                let n = raw.len();
                if let (Some(r), Some(dst)) =
                    (self.rt_raw.get_mut(at..at + n), self.rt.get_mut(at..at + n))
                {
                    r.copy_from_slice(raw);
                    for (d, &c) in dst.iter_mut().zip(raw) {
                        *d = text(c);
                    }
                }
                self.rt_seen |= 1 << seg;
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Carousel (M42)
// ---------------------------------------------------------------------------------------------

/// Raw groups that [`Carousel::push_raw`] can queue.
pub const RAW_QUEUE: usize = 16;

/// Block C of group 0A: the "no alternative frequencies" filler (code 224, twice).
pub const NO_AF: u16 = 0xE0E0;

/// The 26-bit blocks of a group: each information word followed by its checkword with the offset
/// A, B, C (C' when bit 11 of block B marks a version B group) and D, as sent most significant
/// bit first.
///
/// ```
/// use sdrk_proto::rds::{self, group_blocks26, Offset};
/// let b = group_blocks26([0xD123, 0x0148, 0xE0E0, 0x5344]);
/// assert_eq!(b[0], rds::block26(0xD123, Offset::A));
/// assert_eq!(b[2], rds::block26(0xE0E0, Offset::C));
/// // Version B: the third block carries C'.
/// let b = group_blocks26([0xD123, 0x0948, 0xD123, 0x5344]);
/// assert_eq!(b[2], rds::block26(0xD123, Offset::CPrime));
/// ```
pub const fn group_blocks26(words: [u16; 4]) -> [u32; 4] {
    let [a, b, c, d] = words;
    let third = if b & 0x0800 != 0 {
        Offset::CPrime
    } else {
        Offset::C
    };
    [
        block26(a, Offset::A),
        block26(b, Offset::B),
        block26(c, third),
        block26(d, Offset::D),
    ]
}

/// Whether a radiotext is sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RtState {
    /// No text was ever set (the first text keeps the A/B flag).
    Never,
    /// A text was set, then the empty text (no groups 2A).
    Off,
    /// Groups 2A carry the text.
    On,
}

/// The group generator of an RDS transmitter: the programme service name in groups 0A and the
/// radiotext in groups 2A, plus raw groups from the caller.
///
/// [`Carousel::next_group`] returns the queued raw groups first (first in, first out, at most
/// [`RAW_QUEUE`]); otherwise it alternates one group 0A (the four name segments in turn) and one
/// group 2A (the 16 radiotext segments in turn), starting with 0A segment 0. While the radiotext
/// is empty only groups 0A are sent. Raw groups do not advance the alternation.
///
/// - Group 0A: block B carries the group type 0, version A, TP, PTY, TA = 0, M/S = 1 (music),
///   the decoder-identification bit 0 and the segment address; block C is [`NO_AF`]; block D two
///   characters of the name.
/// - Group 2A: block B carries the group type 2, version A, TP, PTY, the text A/B flag and the
///   segment address; blocks C and D four characters of the 64-byte text.
///
/// The name starts as 8 spaces. [`Carousel::set_rt`] pads a text shorter than 64 bytes with a
/// carriage return (0x0D) and spaces; a text that differs from the last one sent toggles the A/B
/// flag (the first text keeps flag 0) and restarts the radiotext at segment 0.
///
/// ```
/// use sdrk_proto::rds::{Carousel, Group, RdsState};
///
/// let mut c = Carousel::new(0xD123, 10, false)?;
/// c.set_ps(b"SDRK TST");
/// c.set_rt(b"Hello from sdrk!")?;
/// let mut rx = RdsState::new();
/// for _ in 0..32 {
///     rx.update(&Group::new(c.next_group()));
/// }
/// assert_eq!(rx.ps(), "SDRK TST");
/// assert_eq!(rx.rt(), "Hello from sdrk!");
/// // A raw group goes out before the next carousel group.
/// c.push_raw(Group::new([0xD123, 0x4000, 1, 2]))?;
/// assert_eq!(c.next_group(), [0xD123, 0x4000, 1, 2]);
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Carousel {
    pi: u16,
    pty: u8,
    tp: bool,
    ps: [u8; PS_LEN],
    rt: [u8; RT_LEN],
    rt_state: RtState,
    ab: bool,
    ps_seg: u8,
    rt_seg: u8,
    /// Whether the last carousel group was a 0A (the next one is then a 2A if there is a text).
    last_ps: bool,
    raw: [[u16; 4]; RAW_QUEUE],
    raw_head: u8,
    raw_len: u8,
}

impl Carousel {
    /// A carousel for the station `pi` with the programme type `pty` (0 to 31) and the traffic
    /// programme flag `tp`; the name is 8 spaces, the radiotext empty, no raw group queued.
    ///
    /// # Errors
    /// `InvalidArgument` when `pty` is above 31.
    ///
    /// ```
    /// use sdrk_proto::rds::Carousel;
    /// let mut c = Carousel::new(0xD123, 10, true)?;
    /// // Group 0A segment 0 of the name "        ": TP in bit 10, PTY 10, M/S = 1.
    /// assert_eq!(c.next_group(), [0xD123, 0x0548, 0xE0E0, 0x2020]);
    /// assert!(Carousel::new(0xD123, 32, false).is_err());
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn new(pi: u16, pty: u8, tp: bool) -> crate::Result<Self> {
        if pty > 31 {
            return Err(crate::Error::InvalidArgument(
                "rds programme type must be 0 to 31",
            ));
        }
        Ok(Self {
            pi,
            pty,
            tp,
            ps: [b' '; PS_LEN],
            rt: [b' '; RT_LEN],
            rt_state: RtState::Never,
            ab: false,
            ps_seg: 0,
            rt_seg: 0,
            last_ps: false,
            raw: [[0; 4]; RAW_QUEUE],
            raw_head: 0,
            raw_len: 0,
        })
    }

    /// The programme identification.
    pub const fn pi(&self) -> u16 {
        self.pi
    }

    /// The programme type.
    pub const fn pty(&self) -> u8 {
        self.pty
    }

    /// The traffic programme flag.
    pub const fn tp(&self) -> bool {
        self.tp
    }

    /// Sets the programme service name (8 bytes, sent as they are) from the next group 0A on.
    pub fn set_ps(&mut self, ps: &[u8; PS_LEN]) {
        self.ps = *ps;
    }

    /// The programme service name.
    pub const fn ps(&self) -> &[u8; PS_LEN] {
        &self.ps
    }

    /// Sets the radiotext: at most 64 bytes, a shorter text is followed by a carriage return
    /// and spaces. A text that differs from the last one set toggles the A/B flag (not for the
    /// first text) and restarts the radiotext at segment 0; the same text again changes
    /// nothing. An empty text stops the groups 2A (the flag and the last text are kept, so the
    /// next text compares with it).
    ///
    /// # Errors
    /// `TooLong` for more than 64 bytes; nothing changes then.
    ///
    /// ```
    /// use sdrk_proto::rds::Carousel;
    /// let mut c = Carousel::new(0xD123, 10, false)?;
    /// c.set_rt(b"Hi")?;
    /// assert_eq!(&c.rt()[..4], b"Hi\r ");
    /// assert!(!c.ab_flag());
    /// c.set_rt(b"Ho")?;
    /// assert!(c.ab_flag());
    /// c.set_rt(b"Ho")?;
    /// assert!(c.ab_flag());
    /// assert!(c.set_rt(&[b'x'; 65]).is_err());
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn set_rt(&mut self, text: &[u8]) -> crate::Result<()> {
        if text.len() > RT_LEN {
            return Err(crate::Error::TooLong {
                len: text.len(),
                max: RT_LEN,
            });
        }
        if text.is_empty() {
            if self.rt_state == RtState::On {
                self.rt_state = RtState::Off;
            }
            return Ok(());
        }
        let mut padded = [b' '; RT_LEN];
        for (d, &s) in padded.iter_mut().zip(text) {
            *d = s;
        }
        if let Some(cr) = padded.get_mut(text.len()) {
            *cr = CR;
        }
        let seen = self.rt_state != RtState::Never;
        if seen && padded == self.rt {
            self.rt_state = RtState::On;
            return Ok(());
        }
        if seen {
            self.ab = !self.ab;
        }
        self.rt = padded;
        self.rt_state = RtState::On;
        self.rt_seg = 0;
        Ok(())
    }

    /// The 64 radiotext bytes as sent (with the carriage return and the padding); spaces before
    /// the first text.
    pub const fn rt(&self) -> &[u8; RT_LEN] {
        &self.rt
    }

    /// Whether groups 2A are sent (a non-empty radiotext is set).
    pub fn rt_active(&self) -> bool {
        self.rt_state == RtState::On
    }

    /// The text A/B flag of the groups 2A.
    pub const fn ab_flag(&self) -> bool {
        self.ab
    }

    /// Queues a raw group; its four words are sent unchanged (the validity flags are ignored)
    /// before the next carousel group.
    ///
    /// # Errors
    /// `BufferTooSmall` when [`RAW_QUEUE`] groups are already queued; nothing changes then.
    ///
    /// ```
    /// use sdrk_proto::rds::{Carousel, Group, RAW_QUEUE};
    /// let mut c = Carousel::new(0xD123, 0, false)?;
    /// for i in 0..RAW_QUEUE as u16 {
    ///     c.push_raw(Group::new([0xD123, 0x4000, i, 0]))?;
    /// }
    /// assert!(c.is_raw_full());
    /// assert!(c.push_raw(Group::new([0; 4])).is_err());
    /// assert_eq!(c.next_group()[2], 0);
    /// assert_eq!(c.raw_len(), RAW_QUEUE - 1);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn push_raw(&mut self, group: Group) -> crate::Result<()> {
        if self.is_raw_full() {
            return Err(crate::Error::BufferTooSmall {
                needed: RAW_QUEUE + 1,
                got: RAW_QUEUE,
            });
        }
        let at = (usize::from(self.raw_head) + usize::from(self.raw_len)) % RAW_QUEUE;
        if let Some(slot) = self.raw.get_mut(at) {
            *slot = group.blocks;
            self.raw_len += 1;
        }
        Ok(())
    }

    /// Restarts the alternation at group 0A segment 0 (the next carousel group is the first one
    /// a new carousel would send); the name, the radiotext, the A/B flag and the queued raw
    /// groups are kept. The encoder block calls it on `reset`.
    ///
    /// ```
    /// use sdrk_proto::rds::Carousel;
    /// let mut c = Carousel::new(0xD123, 10, false)?;
    /// c.set_rt(b"Hi")?;
    /// let first = c.next_group();
    /// c.next_group();
    /// c.restart();
    /// assert_eq!(c.next_group(), first);
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    pub fn restart(&mut self) {
        self.ps_seg = 0;
        self.rt_seg = 0;
        self.last_ps = false;
    }

    /// The raw groups queued.
    pub const fn raw_len(&self) -> usize {
        self.raw_len as usize
    }

    /// Whether the raw queue is full.
    pub const fn is_raw_full(&self) -> bool {
        self.raw_len as usize >= RAW_QUEUE
    }

    /// Block B common to the groups 0A and 2A: group type, version A, TP and PTY.
    const fn block_b(&self, group_type: u16) -> u16 {
        (group_type << 12) | ((self.tp as u16) << 10) | ((self.pty as u16) << 5)
    }

    fn pair(bytes: &[u8], at: usize) -> u16 {
        match bytes.get(at..at + 2) {
            Some(&[a, b]) => u16::from_be_bytes([a, b]),
            _ => 0x2020,
        }
    }

    /// The next group: a queued raw group, else the next group 0A or 2A (see [`Carousel`]).
    pub fn next_group(&mut self) -> [u16; 4] {
        if self.raw_len > 0 {
            let g = self
                .raw
                .get(usize::from(self.raw_head))
                .copied()
                .unwrap_or([0; 4]);
            self.raw_head = ((usize::from(self.raw_head) + 1) % RAW_QUEUE) as u8;
            self.raw_len -= 1;
            return g;
        }
        if self.last_ps && self.rt_state == RtState::On {
            self.last_ps = false;
            let seg = usize::from(self.rt_seg);
            self.rt_seg = ((seg + 1) % RT_SEGMENTS) as u8;
            let b = self.block_b(2) | (u16::from(self.ab) << 4) | seg as u16;
            [
                self.pi,
                b,
                Self::pair(&self.rt, 4 * seg),
                Self::pair(&self.rt, 4 * seg + 2),
            ]
        } else {
            self.last_ps = true;
            let seg = usize::from(self.ps_seg);
            self.ps_seg = ((seg + 1) % 4) as u8;
            // TA = 0, M/S = 1, DI bit 0 for every segment.
            let b = self.block_b(0) | (1 << 3) | seg as u16;
            [self.pi, b, NO_AF, Self::pair(&self.ps, 2 * seg)]
        }
    }
}
