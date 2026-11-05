//! The packet type and its flags.

use core::fmt;
use core::ops::{BitOr, BitOrAssign};

use crate::error::{Error, Result};

/// The largest payload of a [`Packet`], in bytes.
pub const MAX_PACKET_BYTES: usize = 256;

/// Flags of a [`Packet`].
///
/// Bits 0 to 15 are defined by SDRk (the constants below; bits 4 to 15 are reserved), bits 16
/// to 31 are free for a protocol and are documented by the decoder that sets them.
///
/// ```
/// use sdrk_proto::PacketFlags;
///
/// let f = PacketFlags::CRC_OK | PacketFlags::CORRECTED;
/// assert!(f.contains(PacketFlags::CRC_OK));
/// assert!(!f.contains(PacketFlags::CRC_FAIL));
/// assert_eq!(f.bits(), 5);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PacketFlags(pub u32);

impl PacketFlags {
    /// No flag set.
    pub const NONE: Self = Self(0);
    /// The checksum of the packet matched.
    pub const CRC_OK: Self = Self(1);
    /// The checksum of the packet did not match (only delivered when the decoder is asked to).
    pub const CRC_FAIL: Self = Self(2);
    /// Bits of the packet were corrected (by forward error correction or a checksum repair).
    pub const CORRECTED: Self = Self(4);
    /// The packet was cut short.
    pub const TRUNCATED: Self = Self(8);

    /// The raw bits.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// The flags with the raw bits `bits` (every bit is kept).
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Whether every bit of `other` is set in `self`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no bit is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for PacketFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PacketFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl fmt::Debug for PacketFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PacketFlags({:#x})", self.0)
    }
}

/// A byte message with a sample index and flags.
///
/// The payload is stored inline (at most [`MAX_PACKET_BYTES`]), so a packet is `Copy` and
/// needs no heap. The bytes after the payload are always zero, so two packets compare equal
/// exactly when index, flags and payload are equal.
///
/// ```
/// use sdrk_proto::{Packet, PacketFlags};
///
/// let p = Packet::new(1200, PacketFlags::CRC_OK, b"hello")?;
/// assert_eq!(p.data(), b"hello");
/// assert_eq!(p.len(), 5);
/// assert_eq!(p.index, 1200);
/// assert!(Packet::new(0, PacketFlags::NONE, &[0u8; 257]).is_err());
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Packet {
    /// Sample index at the decoder's input where the packet starts (decoders), or 0 (sources);
    /// a transmitting encoder ignores it.
    pub index: u64,
    /// Flags of the packet.
    pub flags: PacketFlags,
    len: u16,
    data: [u8; MAX_PACKET_BYTES],
}

impl Packet {
    /// The empty packet: index 0, no flags, no payload.
    pub const EMPTY: Self = Self {
        index: 0,
        flags: PacketFlags::NONE,
        len: 0,
        data: [0; MAX_PACKET_BYTES],
    };

    /// A packet with a copy of `data`.
    ///
    /// # Errors
    /// [`Error::TooLong`] when `data` has more than [`MAX_PACKET_BYTES`] bytes.
    pub fn new(index: u64, flags: PacketFlags, data: &[u8]) -> Result<Self> {
        let mut p = Self::EMPTY;
        let len = u16::try_from(data.len()).map_err(|_| too_long(data.len()))?;
        p.data
            .get_mut(..data.len())
            .ok_or_else(|| too_long(data.len()))?
            .copy_from_slice(data);
        p.len = len;
        p.index = index;
        p.flags = flags;
        Ok(p)
    }

    /// The payload.
    pub fn data(&self) -> &[u8] {
        self.data.get(..usize::from(self.len)).unwrap_or(&[])
    }

    /// The payload length in bytes.
    pub fn len(&self) -> usize {
        usize::from(self.len)
    }

    /// Whether the payload is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for Packet {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl fmt::Debug for Packet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Packet")
            .field("index", &self.index)
            .field("flags", &self.flags)
            .field("data", &self.data())
            .finish_non_exhaustive()
    }
}

fn too_long(len: usize) -> Error {
    Error::TooLong {
        len,
        max: MAX_PACKET_BYTES,
    }
}
