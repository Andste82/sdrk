//! A simple framing for tests and custom links.
//!
//! | Field | Bits | Content |
//! | --- | --- | --- |
//! | preamble | `preamble_bytes`·8 | `0x55` repeated |
//! | sync word | 32 | [`SYNC_WORD`] `0x1ACFFC1D` (the CCSDS attached sync marker) |
//! | length | 8 | payload length `L` (0 to [`MAX_PAYLOAD`]) |
//! | payload | `8·L` | the packet bytes |
//! | CRC | 16 | [`crc16_ccitt_false`] over length and payload, big-endian |
//!
//! With [`Whitening::Pn9`] the length byte, the payload and the CRC (computed over the plain
//! length and payload) are combined by XOR with the PN9 sequence of [`crate::whiten`], starting with its
//! first byte at the length byte; preamble and sync word are sent as they are. The decoder
//! removes the whitening before it reads the length and checks the CRC. Both sides must use
//! the same setting.
//!
//! Bits are sent most significant bit first. [`encode`] writes a whole frame as bytes `0`/`1`,
//! [`FrameBits`] yields the same bits one at a time (for encoders that cannot hold a frame),
//! and [`Decoder`] is a bit-by-bit state machine that finds frames in a bit stream.
//!
//! ```
//! use sdrk_proto::frame::{self, Decoder, Whitening};
//! use sdrk_proto::{Packet, PacketFlags};
//!
//! let p = Packet::new(0, PacketFlags::NONE, b"abc")?;
//! let mut bits = [0u8; frame::frame_bits(3, 4)];
//! let n = frame::encode(&p, 4, Whitening::Pn9, &mut bits)?;
//! assert_eq!(n, bits.len());
//!
//! let mut dec = Decoder::new(0, false, Whitening::Pn9)?;
//! let mut got = None;
//! for (i, &b) in bits.iter().enumerate() {
//!     if let Some(packet) = dec.push_bit(b != 0, 100 + i as u64) {
//!         got = Some(packet);
//!     }
//! }
//! let got = got.expect("one frame");
//! assert_eq!(got.data(), b"abc");
//! assert_eq!(got.flags, PacketFlags::CRC_OK);
//! assert_eq!(got.index, 100 + 32); // the first bit of the sync word
//! # Ok::<(), sdrk_proto::Error>(())
//! ```

use crate::crc::crc16_ccitt_false;
use crate::error::{Error, Result};
use crate::packet::{MAX_PACKET_BYTES, Packet, PacketFlags};
use crate::whiten::{Pn9, pn9_byte};

/// The whitening of a frame (see the module documentation).
///
/// ```
/// use sdrk_proto::frame::Whitening;
/// assert_eq!(Whitening::default(), Whitening::None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Whitening {
    /// The bytes are sent as they are.
    #[default]
    None,
    /// Length, payload and CRC are combined by XOR with the PN9 sequence ([`crate::whiten::Pn9`]).
    Pn9,
}

impl Whitening {
    /// The byte the `k`-th byte after the sync word is combined with by XOR (0 without whitening).
    #[inline]
    fn mask(self, k: usize) -> u8 {
        match self {
            Self::None => 0,
            Self::Pn9 => pn9_byte(k),
        }
    }
}

/// The sync word (CCSDS attached sync marker), sent most significant bit first.
pub const SYNC_WORD: u32 = 0x1ACF_FC1D;

/// The byte of the preamble (alternating bits, starting with 0).
pub const PREAMBLE_BYTE: u8 = 0x55;

/// The largest payload of a frame: length, payload and CRC fit in [`MAX_PACKET_BYTES`].
pub const MAX_PAYLOAD: usize = MAX_PACKET_BYTES - 3;

/// The largest number of sync-word bit errors a [`Decoder`] accepts.
pub const MAX_SYNC_ERRORS: u32 = 4;

/// Bytes of a frame after the preamble that are not payload: sync word, length and CRC.
const OVERHEAD_BYTES: usize = 4 + 1 + 2;

/// The number of bits of a frame with `payload_len` payload bytes and `preamble_bytes`
/// preamble bytes (saturating; a frame never comes close).
///
/// ```
/// assert_eq!(sdrk_proto::frame::frame_bits(0, 4), 88);
/// assert_eq!(sdrk_proto::frame::frame_bits(253, 4), 2112);
/// ```
pub const fn frame_bits(payload_len: usize, preamble_bytes: usize) -> usize {
    preamble_bytes
        .saturating_add(OVERHEAD_BYTES)
        .saturating_add(payload_len)
        .saturating_mul(8)
}

/// The bits of one frame, most significant bit of every byte first, as an iterator of `0`/`1`.
///
/// The iterator owns a copy of the packet, so an encoder can keep it across calls and resume
/// where its output buffer ended.
///
/// ```
/// use sdrk_proto::frame::{FrameBits, Whitening};
/// use sdrk_proto::{Packet, PacketFlags};
///
/// let p = Packet::new(0, PacketFlags::NONE, &[])?;
/// let bits: Vec<u8> = FrameBits::new(&p, 1, Whitening::None)?.collect();
/// assert_eq!(bits.len(), 8 * (1 + 4 + 1 + 2));
/// assert_eq!(&bits[..8], &[0, 1, 0, 1, 0, 1, 0, 1]); // the preamble 0x55
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameBits {
    packet: Packet,
    preamble_bytes: usize,
    whitening: Whitening,
    crc: u16,
    pos: usize,
    total: usize,
}

impl FrameBits {
    /// The bits of the frame of `packet` with `preamble_bytes` preamble bytes and the given
    /// whitening.
    ///
    /// # Errors
    /// [`Error::TooLong`] when the payload has more than [`MAX_PAYLOAD`] bytes;
    /// [`Error::InvalidArgument`] when the frame length overflows `usize`.
    pub fn new(packet: &Packet, preamble_bytes: usize, whitening: Whitening) -> Result<Self> {
        let len = packet.len();
        if len > MAX_PAYLOAD {
            return Err(Error::TooLong {
                len,
                max: MAX_PAYLOAD,
            });
        }
        let total = preamble_bytes
            .checked_add(OVERHEAD_BYTES + len)
            .and_then(|b| b.checked_mul(8))
            .ok_or(Error::InvalidArgument("preamble_bytes is too large"))?;
        // CRC over the length byte and the payload.
        let mut head = [0u8; MAX_PACKET_BYTES];
        let body = head.get_mut(..=len).ok_or(Error::TooLong {
            len,
            max: MAX_PAYLOAD,
        })?;
        if let Some((first, rest)) = body.split_first_mut() {
            *first = len as u8;
            rest.copy_from_slice(packet.data());
        }
        let crc = crc16_ccitt_false(body);
        Ok(Self {
            packet: *packet,
            preamble_bytes,
            whitening,
            crc,
            pos: 0,
            total,
        })
    }

    /// The total number of bits of the frame.
    pub fn total(&self) -> usize {
        self.total
    }

    /// The number of bits already returned.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The `k`-th byte of the frame (`k` < `total / 8`).
    fn byte(&self, k: usize) -> u8 {
        let pre = self.preamble_bytes;
        let len = self.packet.len();
        match k.checked_sub(pre) {
            None => PREAMBLE_BYTE,
            Some(j @ 0..4) => SYNC_WORD.to_be_bytes().get(j).copied().unwrap_or(0),
            Some(j) => {
                // `b`: the byte index after the sync word (0 is the length byte).
                let b = j - 4;
                let plain = match b.checked_sub(1) {
                    None => len as u8,
                    Some(p) => match p.cmp(&len) {
                        core::cmp::Ordering::Less => {
                            self.packet.data().get(p).copied().unwrap_or(0)
                        }
                        core::cmp::Ordering::Equal => (self.crc >> 8) as u8,
                        core::cmp::Ordering::Greater => self.crc as u8,
                    },
                };
                plain ^ self.whitening.mask(b)
            }
        }
    }
}

impl Iterator for FrameBits {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.pos >= self.total {
            return None;
        }
        let byte = self.byte(self.pos / 8);
        let bit = (byte >> (7 - (self.pos % 8))) & 1;
        self.pos += 1;
        Some(bit)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.total - self.pos;
        (left, Some(left))
    }
}

impl ExactSizeIterator for FrameBits {}

/// Writes the frame of `packet` with `preamble_bytes` preamble bytes and the given whitening as
/// bits (`0`/`1`, one per byte) to the start of `out` and returns the number of bits,
/// [`frame_bits`]`(packet.len(), preamble_bytes)`.
///
/// ```
/// use sdrk_proto::frame::{self, Whitening};
/// use sdrk_proto::{Packet, PacketFlags};
///
/// let p = Packet::new(0, PacketFlags::NONE, &[0xA5])?;
/// let mut out = [9u8; 200];
/// let n = frame::encode(&p, 0, Whitening::None, &mut out)?;
/// assert_eq!(n, 64);
/// assert_eq!(&out[40..48], &[1, 0, 1, 0, 0, 1, 0, 1]); // the payload byte 0xA5
/// assert_eq!(out[64], 9); // untouched
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
///
/// # Errors
/// [`Error::TooLong`] when the payload has more than [`MAX_PAYLOAD`] bytes;
/// [`Error::BufferTooSmall`] when `out` is shorter than the frame.
pub fn encode(
    packet: &Packet,
    preamble_bytes: usize,
    whitening: Whitening,
    out: &mut [u8],
) -> Result<usize> {
    let bits = FrameBits::new(packet, preamble_bytes, whitening)?;
    let total = bits.total();
    let got = out.len();
    let dst = out
        .get_mut(..total)
        .ok_or(Error::BufferTooSmall { needed: total, got })?;
    for (d, b) in dst.iter_mut().zip(bits) {
        *d = b;
    }
    Ok(total)
}

/// Writes the frame of `packet` with `preamble_bytes` preamble bytes and the given whitening as
/// packed bytes (the bits of [`encode`], eight per byte, most significant bit first) to the
/// start of `out` and returns the number of bytes, [`frame_bits`]`(packet.len(),
/// preamble_bytes) / 8`.
///
/// ```
/// use sdrk_proto::frame::{self, Whitening};
/// use sdrk_proto::{Packet, PacketFlags};
///
/// let p = Packet::new(0, PacketFlags::NONE, &[0xA5])?;
/// let mut out = [0u8; 16];
/// let n = frame::encode_bytes(&p, 1, Whitening::None, &mut out)?;
/// assert_eq!(n, 1 + 4 + 1 + 1 + 2);
/// assert_eq!(&out[..7], &[0x55, 0x1A, 0xCF, 0xFC, 0x1D, 1, 0xA5]);
/// // PN9: the length byte and the payload are XORed with FF E1.
/// frame::encode_bytes(&p, 1, Whitening::Pn9, &mut out)?;
/// assert_eq!(&out[5..7], &[1 ^ 0xFF, 0xA5 ^ 0xE1]);
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
///
/// # Errors
/// [`Error::TooLong`] when the payload has more than [`MAX_PAYLOAD`] bytes;
/// [`Error::BufferTooSmall`] when `out` is shorter than the frame.
pub fn encode_bytes(
    packet: &Packet,
    preamble_bytes: usize,
    whitening: Whitening,
    out: &mut [u8],
) -> Result<usize> {
    let len = packet.len();
    if len > MAX_PAYLOAD {
        return Err(Error::TooLong {
            len,
            max: MAX_PAYLOAD,
        });
    }
    let total = preamble_bytes
        .checked_add(OVERHEAD_BYTES + len)
        .filter(|t| t.checked_mul(8).is_some())
        .ok_or(Error::InvalidArgument("preamble_bytes is too large"))?;
    let got = out.len();
    let too_small = Error::BufferTooSmall { needed: total, got };
    let dst = out.get_mut(..total).ok_or(too_small.clone())?;
    let (preamble, rest) = dst
        .split_at_mut_checked(preamble_bytes)
        .ok_or(too_small.clone())?;
    preamble.fill(PREAMBLE_BYTE);
    let (sync, rest) = rest.split_at_mut_checked(4).ok_or(too_small.clone())?;
    sync.copy_from_slice(&SYNC_WORD.to_be_bytes());
    let (body, crc) = rest
        .split_at_mut_checked(len + 1)
        .ok_or(too_small.clone())?;
    let (len_byte, payload) = body.split_first_mut().ok_or(too_small.clone())?;
    *len_byte = len as u8;
    payload.copy_from_slice(packet.data());
    crc.copy_from_slice(&crc16_ccitt_false(body).to_be_bytes());
    if whitening == Whitening::Pn9 {
        // Length, payload and CRC: everything after the sync word.
        Pn9::new().apply(rest);
    }
    Ok(total)
}

/// The sync words a [`Decoder`] hunts for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Polarity {
    /// Only [`SYNC_WORD`].
    Normal,
    /// [`SYNC_WORD`] and its inverse.
    Both,
}

/// The bit-by-bit frame decoder.
///
/// It hunts for the sync word with at most `max_sync_errors` differing bits, then reads the
/// length, the payload and the CRC. A frame with a matching CRC is returned with
/// [`PacketFlags::CRC_OK`]; one with a wrong CRC only when `emit_bad` is set, with
/// [`PacketFlags::CRC_FAIL`]. A length above [`MAX_PAYLOAD`] ends the frame at once (nothing
/// is returned). After every frame, delivered or not, the hunt starts afresh with the next bit:
/// the bits of the frame are never searched for another sync word. With [`Whitening::Pn9`]
/// every byte after the sync word is de-whitened as soon as it is complete.
///
/// The `index` of a returned packet is the index of the first bit of its sync word, computed
/// from the index the caller passes with the last sync-word bit (indices must be consecutive).
///
/// With [`Decoder::with_both_polarities`] the hunt also accepts the inverted sync word (with
/// the same number of differing bits); the bits of such a frame are inverted before they are
/// read. This resolves the 180 degree ambiguity of a coherent PSK receiver.
#[derive(Clone, Debug)]
pub struct Decoder {
    max_sync_errors: u32,
    emit_bad: bool,
    whitening: Whitening,
    /// Which sync words the hunt accepts.
    polarity: Polarity,
    /// Whether the frame being read was found by the inverted sync word.
    invert: bool,
    /// The last bits while hunting, newest in bit 0.
    shift: u32,
    /// Bits still missing before `shift` holds 32 bits of this hunt.
    pending: u8,
    /// Whether a frame body is being read (false: hunting).
    in_frame: bool,
    /// Length byte, payload and CRC of the frame being read.
    body: [u8; MAX_PACKET_BYTES],
    /// The bits of the body byte being read, newest in bit 0.
    cur: u8,
    /// Body bits read so far.
    body_bits: usize,
    /// Body bits of the frame (8 until the length byte is known).
    body_need: usize,
    /// Index of the first sync-word bit of the frame being read.
    start: u64,
}

impl Decoder {
    /// A decoder in the hunting state for frames with the given whitening.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] when `max_sync_errors` is above [`MAX_SYNC_ERRORS`].
    pub fn new(max_sync_errors: u32, emit_bad: bool, whitening: Whitening) -> Result<Self> {
        if max_sync_errors > MAX_SYNC_ERRORS {
            return Err(Error::InvalidArgument("max_sync_errors is above 4"));
        }
        Ok(Self {
            max_sync_errors,
            emit_bad,
            whitening,
            polarity: Polarity::Normal,
            invert: false,
            shift: 0,
            pending: 31,
            in_frame: false,
            body: [0; MAX_PACKET_BYTES],
            cur: 0,
            body_bits: 0,
            body_need: 8,
            start: 0,
        })
    }

    /// The decoder with the polarity rule `both`: when set, the hunt also accepts the inverted
    /// sync word `!SYNC_WORD` (at most `max_sync_errors` differing bits, the normal word is
    /// tried first) and inverts the bits of the frame that follows (M40, for coherent PSK
    /// receivers whose phase may be off by 180 degrees). [`Decoder::new`] gives `false`.
    ///
    /// ```
    /// use sdrk_proto::frame::{self, Decoder, Whitening};
    /// use sdrk_proto::{Packet, PacketFlags};
    ///
    /// let p = Packet::new(0, PacketFlags::NONE, b"ok")?;
    /// let mut bits = [0u8; frame::frame_bits(2, 2)];
    /// frame::encode(&p, 2, Whitening::None, &mut bits)?;
    /// let mut dec = Decoder::new(0, false, Whitening::None)?.with_both_polarities(true);
    /// assert!(dec.both_polarities());
    /// let mut got = None;
    /// for (i, &b) in bits.iter().enumerate() {
    ///     // every bit inverted
    ///     if let Some(packet) = dec.push_bit(b == 0, i as u64) {
    ///         got = Some(packet);
    ///     }
    /// }
    /// let got = got.expect("one frame");
    /// assert_eq!((got.data(), got.flags), (&b"ok"[..], PacketFlags::CRC_OK));
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    #[must_use]
    pub fn with_both_polarities(mut self, both: bool) -> Self {
        self.polarity = if both {
            Polarity::Both
        } else {
            Polarity::Normal
        };
        self.hunt();
        self
    }

    /// Whether the inverted sync word is accepted too (see [`Decoder::with_both_polarities`]).
    pub fn both_polarities(&self) -> bool {
        self.polarity == Polarity::Both
    }

    /// The accepted number of sync-word bit errors.
    pub fn max_sync_errors(&self) -> u32 {
        self.max_sync_errors
    }

    /// Whether frames with a wrong CRC are returned.
    pub fn emit_bad(&self) -> bool {
        self.emit_bad
    }

    /// The whitening the decoder removes.
    pub fn whitening(&self) -> Whitening {
        self.whitening
    }

    /// Whether the decoder is hunting for a sync word (not inside a frame).
    pub fn is_hunting(&self) -> bool {
        !self.in_frame
    }

    /// Back to the state after [`Decoder::new`].
    pub fn reset(&mut self) {
        self.hunt();
    }

    fn hunt(&mut self) {
        self.shift = 0;
        self.pending = 31;
        self.in_frame = false;
        self.invert = false;
        self.body_bits = 0;
        self.body_need = 8;
    }

    /// Feeds one bit with its sample index; returns a packet when the bit completes a frame
    /// that is delivered.
    ///
    /// ```
    /// use sdrk_proto::frame::{Decoder, Whitening};
    /// let mut dec = Decoder::new(0, false, Whitening::None)?;
    /// assert!(dec.push_bit(true, 0).is_none());
    /// # Ok::<(), sdrk_proto::Error>(())
    /// ```
    // `always`: with more than one caller in a binary (the bench links the decoder and the
    // block `frame_decode`) LLVM otherwise keeps one out-of-line copy and pays a call per bit.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn push_bit(&mut self, bit: bool, index: u64) -> Option<Packet> {
        if !self.in_frame {
            self.shift = (self.shift << 1) | u32::from(bit);
            if self.pending != 0 {
                self.pending -= 1;
                return None;
            }
            let diff = self.shift ^ SYNC_WORD;
            // The exact match first: no population count for the default `max_sync_errors` 0.
            if diff == 0 || (self.max_sync_errors != 0 && diff.count_ones() <= self.max_sync_errors)
            {
                self.in_frame = true;
                self.body_bits = 0;
                self.body_need = 8;
                self.start = index.wrapping_sub(31);
            } else if self.polarity == Polarity::Both {
                // `shift ^ !SYNC_WORD` is `!diff`.
                let inv = !diff;
                if inv == 0
                    || (self.max_sync_errors != 0 && inv.count_ones() <= self.max_sync_errors)
                {
                    self.in_frame = true;
                    self.invert = true;
                    self.body_bits = 0;
                    self.body_need = 8;
                    self.start = index.wrapping_sub(31);
                }
            }
            return None;
        }
        self.cur = (self.cur << 1) | u8::from(bit != self.invert);
        self.body_bits += 1;
        if !self.body_bits.is_multiple_of(8) {
            return None;
        }
        // A whole byte: de-whiten and store it.
        let k = self.body_bits / 8 - 1;
        let byte = self.cur ^ self.whitening.mask(k);
        let Some(slot) = self.body.get_mut(k) else {
            // Unreachable: `body_need` never exceeds the body. Start over to stay safe.
            self.hunt();
            return None;
        };
        *slot = byte;
        if self.body_bits == 8 {
            let len = usize::from(byte);
            if len > MAX_PAYLOAD {
                self.hunt();
                return None;
            }
            self.body_need = (len + 3) * 8;
        }
        if self.body_bits < self.body_need {
            return None;
        }
        let out = self.finish();
        self.hunt();
        out
    }

    /// Checks the CRC of the complete body and builds the packet.
    #[cold]
    #[inline(never)]
    fn finish(&self) -> Option<Packet> {
        let len = self.body_need / 8 - 3;
        let (head, crc) = self.body.split_at_checked(len + 1)?;
        let sent = u16::from_be_bytes([*crc.first()?, *crc.get(1)?]);
        let ok = crc16_ccitt_false(head) == sent;
        if !ok && !self.emit_bad {
            return None;
        }
        let flags = if ok {
            PacketFlags::CRC_OK
        } else {
            PacketFlags::CRC_FAIL
        };
        Packet::new(self.start, flags, head.get(1..)?).ok()
    }
}
