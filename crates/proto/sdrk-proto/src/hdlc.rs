//! HDLC framing as used by AX.25 (APRS): flags, bit stuffing and the CRC-16/X-25 frame check
//! sequence.
//!
//! | Part | Content |
//! | --- | --- |
//! | flags | [`FLAG`] `0x7E` (`01111110`), sent before and after a frame and between frames |
//! | payload | the packet bytes, each least significant bit first |
//! | FCS | [`crc16_x25`] of the payload, low byte first (each byte least significant bit first) |
//!
//! Between the flags (over payload and FCS) a 0 bit is inserted after every five 1 bits in a
//! row (bit stuffing), so the data never shows six 1s; six 1s between two 0s are a flag, seven
//! or more 1s abort the frame. A frame has [`MIN_PAYLOAD`] to [`MAX_PAYLOAD`] payload bytes
//! (2 to 256, plus the two FCS bytes).
//!
//! [`Encoder`] yields the bits of one frame with its opening and closing flags one at a time
//! (so an encoder block can keep it across calls), [`encode`] writes them to a buffer, and
//! [`Decoder`] is a bit-by-bit state machine that finds frames in a bit stream. NRZI line
//! coding is separate ([`crate::nrzi`]).
//!
//! ```
//! use sdrk_proto::hdlc::{self, Decoder};
//! use sdrk_proto::{Packet, PacketFlags};
//!
//! let p = Packet::new(0, PacketFlags::NONE, &[0xFF, 0x7E, 0x01])?;
//! let mut bits = [0u8; hdlc::max_bits(3, 2, 1)];
//! let n = hdlc::encode(&p, 2, 1, &mut bits)?;
//!
//! let mut dec = Decoder::new(false);
//! let mut got = None;
//! for (i, &b) in bits[..n].iter().enumerate() {
//!     if let Some(packet) = dec.push_bit(b != 0, i as u64) {
//!         got = Some(packet);
//!     }
//! }
//! let got = got.expect("one frame");
//! assert_eq!(got.data(), &[0xFF, 0x7E, 0x01]);
//! assert_eq!(got.flags, PacketFlags::CRC_OK);
//! assert_eq!(got.index, 16); // the first bit after the second (opening) flag
//! # Ok::<(), sdrk_proto::Error>(())
//! ```

use crate::crc::crc16_x25;
use crate::error::{Error, Result};
use crate::packet::{MAX_PACKET_BYTES, Packet, PacketFlags};

/// The HDLC flag `01111110`.
pub const FLAG: u8 = 0x7E;

/// The smallest payload of a frame, in bytes (without the FCS).
pub const MIN_PAYLOAD: usize = 2;

/// The largest payload of a frame, in bytes (without the FCS): a whole [`Packet`].
pub const MAX_PAYLOAD: usize = MAX_PACKET_BYTES;

/// The bytes of the frame check sequence.
const FCS_BYTES: usize = 2;

/// Payload and FCS of the largest frame.
const MAX_BODY: usize = MAX_PAYLOAD + FCS_BYTES;

/// The largest number of bits of a frame with `payload_len` payload bytes, `preamble_flags`
/// opening and `tail_flags` closing flags: the flags, payload and FCS, and one stuffed bit per
/// five data bits (saturating; a frame never comes close). The bits [`Encoder`] yields never
/// exceed it.
///
/// ```
/// // 2 flags, 4 bytes (two payload, two FCS) = 32 bits with at most 6 stuffed bits.
/// assert_eq!(sdrk_proto::hdlc::max_bits(2, 1, 1), 16 + 32 + 6);
/// ```
pub const fn max_bits(payload_len: usize, preamble_flags: usize, tail_flags: usize) -> usize {
    let body = payload_len.saturating_add(FCS_BYTES).saturating_mul(8);
    preamble_flags
        .saturating_add(tail_flags)
        .saturating_mul(8)
        .saturating_add(body)
        .saturating_add(body / 5)
}

/// The part of the frame an [`Encoder`] is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Preamble,
    Body,
    Tail,
    Done,
}

/// The bits of one HDLC frame (opening flags, stuffed payload and FCS, closing flags) as an
/// iterator of `0`/`1`.
///
/// The iterator owns a copy of the packet, so an encoder can keep it across calls and resume
/// where its output buffer ended. The FCS is computed once, in [`Encoder::new`].
///
/// ```
/// use sdrk_proto::hdlc::Encoder;
/// use sdrk_proto::{Packet, PacketFlags};
///
/// let p = Packet::new(0, PacketFlags::NONE, &[0xFF, 0xFF])?;
/// let bits: Vec<u8> = Encoder::new(&p, 1, 1)?.collect();
/// assert_eq!(&bits[..8], &[0, 1, 1, 1, 1, 1, 1, 0]); // the flag
/// assert_eq!(&bits[8..14], &[1, 1, 1, 1, 1, 0]); // five 1s, then a stuffed 0
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoder {
    packet: Packet,
    fcs: u16,
    preamble_bits: usize,
    tail_bits: usize,
    body_bits: usize,
    phase: Phase,
    /// The next bit of the current phase.
    pos: usize,
    /// The 1 bits in a row sent in the body (a 0 is stuffed when it reaches 5).
    ones: u8,
}

impl Encoder {
    /// The bits of the frame of `packet` with `preamble_flags` opening and `tail_flags`
    /// closing flags.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] when the payload has fewer than [`MIN_PAYLOAD`] bytes or the
    /// number of flags overflows `usize`.
    pub fn new(packet: &Packet, preamble_flags: usize, tail_flags: usize) -> Result<Self> {
        let len = packet.len();
        if len < MIN_PAYLOAD {
            return Err(Error::InvalidArgument(
                "an hdlc frame needs at least 2 payload bytes",
            ));
        }
        let too_many = Error::InvalidArgument("too many hdlc flags");
        let preamble_bits = preamble_flags.checked_mul(8).ok_or(too_many.clone())?;
        let tail_bits = tail_flags.checked_mul(8).ok_or(too_many)?;
        Ok(Self {
            packet: *packet,
            fcs: crc16_x25(packet.data()),
            preamble_bits,
            tail_bits,
            body_bits: (len + FCS_BYTES) * 8,
            phase: Phase::Preamble,
            pos: 0,
            ones: 0,
        })
    }

    /// The frame check sequence of the frame ([`crc16_x25`] of the payload).
    pub fn fcs(&self) -> u16 {
        self.fcs
    }

    /// Whether every bit has been returned.
    pub fn is_done(&self) -> bool {
        self.phase == Phase::Done
    }

    /// The byte `k` of the body: payload, then the FCS low byte first.
    #[inline]
    fn body_byte(&self, k: usize) -> u8 {
        let len = self.packet.len();
        match k.checked_sub(len) {
            None => self.packet.data().get(k).copied().unwrap_or(0),
            Some(0) => self.fcs as u8,
            Some(_) => (self.fcs >> 8) as u8,
        }
    }
}

impl Iterator for Encoder {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        loop {
            match self.phase {
                Phase::Preamble => {
                    if self.pos < self.preamble_bits {
                        let bit = (FLAG >> (self.pos % 8)) & 1;
                        self.pos += 1;
                        return Some(bit);
                    }
                    self.phase = Phase::Body;
                    self.pos = 0;
                }
                Phase::Body => {
                    if self.ones == 5 {
                        self.ones = 0;
                        return Some(0);
                    }
                    if self.pos < self.body_bits {
                        let bit = (self.body_byte(self.pos / 8) >> (self.pos % 8)) & 1;
                        self.ones = if bit != 0 { self.ones + 1 } else { 0 };
                        self.pos += 1;
                        return Some(bit);
                    }
                    self.phase = Phase::Tail;
                    self.pos = 0;
                }
                Phase::Tail => {
                    if self.pos < self.tail_bits {
                        let bit = (FLAG >> (self.pos % 8)) & 1;
                        self.pos += 1;
                        return Some(bit);
                    }
                    self.phase = Phase::Done;
                }
                Phase::Done => return None,
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Remaining flag and body bits; the stuffed bits are not known in advance.
        let (pre, body, pending) = match self.phase {
            Phase::Preamble => (self.preamble_bits - self.pos, self.body_bits, 0),
            Phase::Body => (0, self.body_bits - self.pos, usize::from(self.ones == 5)),
            Phase::Tail | Phase::Done => (0, 0, 0),
        };
        let tail = match self.phase {
            Phase::Tail => self.tail_bits - self.pos,
            Phase::Done => 0,
            Phase::Preamble | Phase::Body => self.tail_bits,
        };
        let low = pre
            .saturating_add(body)
            .saturating_add(tail)
            .saturating_add(pending);
        // Five 1s in a row are needed for each further stuffed bit.
        let high = low.checked_add((usize::from(self.ones) + body) / 5);
        (low, high)
    }
}

/// Writes the frame of `packet` with `preamble_flags` opening and `tail_flags` closing flags
/// as bits (`0`/`1`, one per byte) to the start of `out` and returns the number of bits (at
/// most [`max_bits`]`(packet.len(), preamble_flags, tail_flags)`).
///
/// ```
/// use sdrk_proto::hdlc;
/// use sdrk_proto::{Packet, PacketFlags};
///
/// let p = Packet::new(0, PacketFlags::NONE, &[0x01, 0x02])?;
/// let mut out = [9u8; 64];
/// let n = hdlc::encode(&p, 1, 1, &mut out)?;
/// assert_eq!(&out[8..16], &[1, 0, 0, 0, 0, 0, 0, 0]); // 0x01, least significant bit first
/// assert_eq!(out[n], 9); // untouched
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
///
/// # Errors
/// [`Error::InvalidArgument`] as [`Encoder::new`]; [`Error::BufferTooSmall`] when `out` is
/// shorter than the frame.
pub fn encode(
    packet: &Packet,
    preamble_flags: usize,
    tail_flags: usize,
    out: &mut [u8],
) -> Result<usize> {
    let mut bits = Encoder::new(packet, preamble_flags, tail_flags)?;
    let mut n = 0;
    for (d, b) in out.iter_mut().zip(bits.by_ref()) {
        *d = b;
        n += 1;
    }
    let rest = bits.count();
    if rest == 0 {
        return Ok(n);
    }
    Err(Error::BufferTooSmall {
        needed: n.saturating_add(rest),
        got: out.len(),
    })
}

/// The bit-by-bit HDLC decoder.
///
/// It waits for a flag, then collects the bits up to the next flag, removing every 0 that
/// follows five 1s. A frame ends at a flag, which also opens the next frame (frames may share
/// a flag). A frame whose bits fill whole bytes, with [`MIN_PAYLOAD`] to [`MAX_PAYLOAD`]
/// payload bytes and a matching FCS is returned with [`PacketFlags::CRC_OK`]; one with a wrong
/// FCS only when `emit_bad` is set, with [`PacketFlags::CRC_FAIL`]. Anything else between two
/// flags (shorter or longer frames, a bit count that is not a multiple of 8) is dropped. Seven
/// 1s in a row abort the frame; the decoder then waits for the next flag.
///
/// The `index` of a returned packet is the index of the first bit after its opening flag,
/// computed from the index the caller passes with the last bit of that flag (indices must be
/// consecutive).
///
/// ```
/// use sdrk_proto::hdlc::Decoder;
/// let mut dec = Decoder::new(true);
/// assert!(dec.is_hunting());
/// for b in [0, 1, 1, 1, 1, 1, 1, 0] {
///     assert!(dec.push_bit(b != 0, 0).is_none());
/// }
/// assert!(!dec.is_hunting()); // a flag opened a frame
/// ```
#[derive(Clone, Debug)]
pub struct Decoder {
    emit_bad: bool,
    /// The 1 bits in a row received (saturating).
    ones: u8,
    /// Whether a frame is being read (false: waiting for a flag).
    in_frame: bool,
    /// Payload and FCS of the frame being read.
    body: [u8; MAX_BODY],
    /// The bits of the byte being read, the newest in bit 7.
    cur: u8,
    /// Data bits of the frame read so far.
    bits: usize,
    /// Index of the first bit after the opening flag.
    start: u64,
}

impl Decoder {
    /// A decoder that waits for a flag; frames with a wrong FCS are returned when `emit_bad`
    /// is set.
    pub fn new(emit_bad: bool) -> Self {
        Self {
            emit_bad,
            ones: 0,
            in_frame: false,
            body: [0; MAX_BODY],
            cur: 0,
            bits: 0,
            start: 0,
        }
    }

    /// Whether frames with a wrong FCS are returned.
    pub fn emit_bad(&self) -> bool {
        self.emit_bad
    }

    /// Whether the decoder waits for a flag (not inside a frame).
    pub fn is_hunting(&self) -> bool {
        !self.in_frame
    }

    /// Back to the state after [`Decoder::new`].
    pub fn reset(&mut self) {
        self.ones = 0;
        self.in_frame = false;
        self.bits = 0;
        self.cur = 0;
    }

    /// Feeds one bit with its sample index; returns a packet when the bit completes the
    /// closing flag of a frame that is delivered.
    ///
    /// ```
    /// use sdrk_proto::hdlc::Decoder;
    /// let mut dec = Decoder::new(false);
    /// assert!(dec.push_bit(true, 0).is_none());
    /// ```
    // `always`: as `frame::Decoder::push_bit`, so the per-bit loop of a block has no call.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn push_bit(&mut self, bit: bool, index: u64) -> Option<Packet> {
        if bit {
            self.ones = self.ones.saturating_add(1);
            if self.ones >= 7 {
                // Abort: wait for the next flag.
                self.in_frame = false;
            } else if self.in_frame {
                self.append(0x80);
            }
            return None;
        }
        let ones = self.ones;
        self.ones = 0;
        match ones {
            6 => {
                // A flag: it closes the frame being read and opens the next one. The first
                // seven bits of the flag were collected as data; the frame is whole bytes
                // exactly when they are the only bits of the incomplete byte.
                let out = if self.in_frame && self.bits % 8 == 7 {
                    self.finish(self.bits / 8)
                } else {
                    None
                };
                self.in_frame = true;
                self.bits = 0;
                self.start = index.wrapping_add(1);
                out
            }
            // A stuffed 0.
            5 => None,
            _ => {
                if self.in_frame {
                    self.append(0);
                }
                None
            }
        }
    }

    /// Appends one data bit (`0x80` for a 1, `0` for a 0), least significant bit first.
    #[inline]
    fn append(&mut self, high: u8) {
        self.cur = (self.cur >> 1) | high;
        self.bits += 1;
        if !self.bits.is_multiple_of(8) {
            return;
        }
        match self.body.get_mut(self.bits / 8 - 1) {
            Some(slot) => *slot = self.cur,
            // Too long for a frame: drop it and wait for the next flag.
            None => self.in_frame = false,
        }
    }

    /// Checks the FCS of a frame of `n` bytes and builds the packet.
    #[cold]
    #[inline(never)]
    fn finish(&self, n: usize) -> Option<Packet> {
        if n < MIN_PAYLOAD + FCS_BYTES {
            return None;
        }
        let (payload, fcs) = self.body.get(..n)?.split_at_checked(n - FCS_BYTES)?;
        let sent = u16::from_le_bytes([*fcs.first()?, *fcs.get(1)?]);
        let ok = crc16_x25(payload) == sent;
        if !ok && !self.emit_bad {
            return None;
        }
        let flags = if ok {
            PacketFlags::CRC_OK
        } else {
            PacketFlags::CRC_FAIL
        };
        Packet::new(self.start, flags, payload).ok()
    }
}

impl Default for Decoder {
    /// A decoder without `emit_bad`.
    fn default() -> Self {
        Self::new(false)
    }
}
