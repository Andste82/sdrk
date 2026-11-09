//! Pulse trains (M45.1): the packet payload of `band_pulse_detect`, one detected short-range
//! transmission as its pulse and gap widths.
//!
//! # Payload
//!
//! A payload is a header of [`HEADER_BYTES`] bytes followed by the widths as pairs of
//! [`PAIR_BYTES`] bytes, everything little endian:
//!
//! | Offset | Type | Field |
//! | --- | --- | --- |
//! | 0 | `u8` | version, [`VERSION`] |
//! | 1 | `u8` | modulation: 0 [`Modulation::Ook`], 1 [`Modulation::Fsk`] |
//! | 2 | `i16` | signed channel number `c` of the channelizer (−M/2 … M/2 − 1, 0 the centre) |
//! | 4 | `i32` | frequency offset of the burst from the band centre in Hz |
//! | 8 | `i16` | level of the burst in dB × 10 |
//! | 10 | `i16` | noise level in dB × 10 |
//! | 12 | `u32` | sample rate of the widths in Hz |
//! | 16 + 8 k | `u32`, `u32` | pair `k`: pulse width, gap width in samples of that rate |
//!
//! For on-off keying a pulse is the time with the carrier on and a gap the time without it; for
//! FSK the pulse is the time at the upper frequency and the gap the time at the lower one. The
//! [`Packet::index`] of a pulse train is the burst start, counted in time steps of the input of
//! the detector (whose step rate is the header's sample rate).
//!
//! # Long trains
//!
//! A packet holds at most 256 bytes, that is [`MAX_PAIRS`] (30) pairs. A longer train is sent
//! as several packets with the same index and the same header, in order; every packet but the
//! last has the flag [`CONTINUED`]. [`packets`] cuts a train this way and [`Assembler`] joins
//! the packets again into caller storage.
//!
//! # JSON
//!
//! [`write_json`] writes a train as one JSON object, the human-readable form next to the
//! packet files of `sdrk_io` (whose lines carry the binary payload as hex):
//! `{"mod":"ook","freq_hz":433920000,"rssi_db":-12.5,"noise_db":-40.0,"channel":0,"rate_hz":250000,"pulses":[[p,g],…]}`.
//!
//! ```
//! use sdrk_proto::pulses::{self, Header, Modulation, Pair};
//! use sdrk_proto::PacketFlags;
//!
//! let header = Header {
//!     modulation: Modulation::Ook,
//!     channel: -3,
//!     freq_offset_hz: -372_500,
//!     level_db10: -125,
//!     noise_db10: -400,
//!     rate_hz: 250_000,
//! };
//! let pairs = [Pair { pulse: 125, gap: 250 }, Pair { pulse: 250, gap: 125 }];
//! let packet = pulses::encode_packet(1000, PacketFlags::NONE, &header, &pairs)?;
//! let train = pulses::decode(packet.data())?;
//! assert_eq!(train.header(), header);
//! assert!(train.pairs().eq(pairs.iter().copied()));
//!
//! let mut json = String::new();
//! pulses::write_json(&mut json, &header, train.pairs(), 433_920_000)?;
//! assert_eq!(
//!     json,
//!     r#"{"mod":"ook","freq_hz":433547500,"rssi_db":-12.5,"noise_db":-40.0,"channel":-3,"rate_hz":250000,"pulses":[[125,250],[250,125]]}"#
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::fmt;

use crate::error::{Error, Result};
use crate::packet::{MAX_PACKET_BYTES, Packet, PacketFlags};

/// The version byte this module writes and the only one it reads.
pub const VERSION: u8 = 1;

/// Bytes of the header.
pub const HEADER_BYTES: usize = 16;

/// Bytes of one pair (pulse and gap, `u32` each).
pub const PAIR_BYTES: usize = 8;

/// The most pairs one packet holds: `(256 - 16) / 8` = 30.
pub const MAX_PAIRS: usize = (MAX_PACKET_BYTES - HEADER_BYTES) / PAIR_BYTES;

/// Flag of a packet whose train continues in the next packet (bit 16, a protocol bit of
/// [`PacketFlags`]).
pub const CONTINUED: PacketFlags = PacketFlags(1 << 16);

/// The modulation of a pulse train.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Modulation {
    /// On-off keying: a pulse is carrier on, a gap carrier off.
    Ook,
    /// Two-level FSK: a pulse is the upper frequency, a gap the lower one.
    Fsk,
}

impl Modulation {
    /// The code in the header (0 for OOK, 1 for FSK).
    pub const fn code(self) -> u8 {
        match self {
            Self::Ook => 0,
            Self::Fsk => 1,
        }
    }

    /// The modulation of a header code.
    ///
    /// # Errors
    /// [`Error::Parse`] for a code other than 0 and 1.
    pub const fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Ook),
            1 => Ok(Self::Fsk),
            _ => Err(Error::Parse("unknown pulse train modulation")),
        }
    }

    /// The name in the JSON form: `"ook"` or `"fsk"`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ook => "ook",
            Self::Fsk => "fsk",
        }
    }
}

/// The header of a pulse train (see the [module documentation](self); the version is not a
/// field: [`encode`] writes [`VERSION`] and [`decode`] accepts only it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Header {
    /// The modulation.
    pub modulation: Modulation,
    /// The signed channel number of the channelizer (0 the centre).
    pub channel: i16,
    /// The frequency offset of the burst from the band centre in Hz.
    pub freq_offset_hz: i32,
    /// The level of the burst in dB × 10.
    pub level_db10: i16,
    /// The noise level in dB × 10.
    pub noise_db10: i16,
    /// The sample rate of the widths in Hz.
    pub rate_hz: u32,
}

/// One pulse and the gap after it, in samples of the header's rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Pair {
    /// Width of the pulse.
    pub pulse: u32,
    /// Width of the gap that follows it.
    pub gap: u32,
}

/// The payload length of a train of `pairs` pairs in one packet.
pub const fn payload_len(pairs: usize) -> usize {
    HEADER_BYTES + pairs * PAIR_BYTES
}

/// Writes the payload of `header` and `pairs` to the front of `out` and returns its length.
///
/// # Errors
/// [`Error::TooLong`] for more than [`MAX_PAIRS`] pairs (the payload would not fit a packet),
/// [`Error::BufferTooSmall`] when `out` is shorter than the payload.
pub fn encode(header: &Header, pairs: &[Pair], out: &mut [u8]) -> Result<usize> {
    if pairs.len() > MAX_PAIRS {
        return Err(Error::TooLong {
            len: HEADER_BYTES.saturating_add(pairs.len().saturating_mul(PAIR_BYTES)),
            max: MAX_PACKET_BYTES,
        });
    }
    encode_joined(header, pairs, out)
}

/// Writes the payload of a whole train of any number of pairs (an [`Assembled`] train, which
/// no longer fits one packet) to the front of `out` and returns its length; [`decode`] reads
/// it, so the slicers and the sensor decoders work on joined trains. The result is not a packet
/// payload when it has more than [`MAX_PAIRS`] pairs.
///
/// ```
/// use sdrk_proto::pulses::{self, Header, Modulation, Pair};
/// let header = Header { modulation: Modulation::Ook, channel: 0, freq_offset_hz: 0,
///     level_db10: 0, noise_db10: 0, rate_hz: 250_000 };
/// let pairs = [Pair { pulse: 5, gap: 9 }; 100];
/// let mut buf = [0u8; 1024];
/// let n = pulses::encode_joined(&header, &pairs, &mut buf)?;
/// assert_eq!(pulses::decode(&buf[..n])?.len(), 100);
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
///
/// # Errors
/// [`Error::BufferTooSmall`] when `out` is shorter than the payload.
pub fn encode_joined(header: &Header, pairs: &[Pair], out: &mut [u8]) -> Result<usize> {
    let len = HEADER_BYTES.saturating_add(pairs.len().saturating_mul(PAIR_BYTES));
    let got = out.len();
    let out = out
        .get_mut(..len)
        .ok_or(Error::BufferTooSmall { needed: len, got })?;
    let (head, body) = out.split_at_mut(HEADER_BYTES);
    let mut h = [0u8; HEADER_BYTES];
    h[0] = VERSION;
    h[1] = header.modulation.code();
    h[2..4].copy_from_slice(&header.channel.to_le_bytes());
    h[4..8].copy_from_slice(&header.freq_offset_hz.to_le_bytes());
    h[8..10].copy_from_slice(&header.level_db10.to_le_bytes());
    h[10..12].copy_from_slice(&header.noise_db10.to_le_bytes());
    h[12..16].copy_from_slice(&header.rate_hz.to_le_bytes());
    head.copy_from_slice(&h);
    for (chunk, pair) in body.chunks_exact_mut(PAIR_BYTES).zip(pairs) {
        let (p, g) = chunk.split_at_mut(4);
        p.copy_from_slice(&pair.pulse.to_le_bytes());
        g.copy_from_slice(&pair.gap.to_le_bytes());
    }
    Ok(len)
}

/// A packet with the payload of `header` and `pairs`.
///
/// # Errors
/// [`Error::TooLong`] for more than [`MAX_PAIRS`] pairs; use [`packets`] for longer trains.
pub fn encode_packet(
    index: u64,
    flags: PacketFlags,
    header: &Header,
    pairs: &[Pair],
) -> Result<Packet> {
    let mut buf = [0u8; MAX_PACKET_BYTES];
    let len = encode(header, pairs, &mut buf)?;
    Packet::new(index, flags, buf.get(..len).unwrap_or(&[]))
}

/// A decoded payload: the header and a view of the pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PulseTrain<'a> {
    header: Header,
    body: &'a [u8],
}

impl<'a> PulseTrain<'a> {
    /// The header.
    pub fn header(&self) -> Header {
        self.header
    }

    /// The number of pairs.
    pub fn len(&self) -> usize {
        self.body.len() / PAIR_BYTES
    }

    /// Whether the payload has no pair.
    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    /// The pairs in order.
    pub fn pairs(&self) -> Pairs<'a> {
        Pairs {
            chunks: self.body.chunks_exact(PAIR_BYTES),
        }
    }
}

/// The pairs of a [`PulseTrain`].
#[derive(Debug, Clone)]
pub struct Pairs<'a> {
    chunks: core::slice::ChunksExact<'a, u8>,
}

fn le_u32(b: &[u8]) -> u32 {
    let mut w = [0u8; 4];
    w.copy_from_slice(b.get(..4).unwrap_or(&[0; 4]));
    u32::from_le_bytes(w)
}

impl Iterator for Pairs<'_> {
    type Item = Pair;

    fn next(&mut self) -> Option<Pair> {
        let c = self.chunks.next()?;
        let (p, g) = c.split_at(4);
        Some(Pair {
            pulse: le_u32(p),
            gap: le_u32(g),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.chunks.size_hint()
    }
}

impl ExactSizeIterator for Pairs<'_> {}

/// Reads a payload.
///
/// # Errors
/// [`Error::Parse`] when `data` is shorter than the header, its version is not [`VERSION`], its
/// modulation code is unknown or the bytes after the header are not a whole number of pairs.
pub fn decode(data: &[u8]) -> Result<PulseTrain<'_>> {
    if data.len() < HEADER_BYTES {
        return Err(Error::Parse("pulse train shorter than its header"));
    }
    let (h, body) = data.split_at(HEADER_BYTES);
    let mut head = [0u8; HEADER_BYTES];
    head.copy_from_slice(h);
    if head[0] != VERSION {
        return Err(Error::Parse("unknown pulse train version"));
    }
    if body.len() % PAIR_BYTES != 0 {
        return Err(Error::Parse("pulse train is not a header and whole pairs"));
    }
    let header = Header {
        modulation: Modulation::from_code(head[1])?,
        channel: i16::from_le_bytes([head[2], head[3]]),
        freq_offset_hz: i32::from_le_bytes([head[4], head[5], head[6], head[7]]),
        level_db10: i16::from_le_bytes([head[8], head[9]]),
        noise_db10: i16::from_le_bytes([head[10], head[11]]),
        rate_hz: u32::from_le_bytes([head[12], head[13], head[14], head[15]]),
    };
    Ok(PulseTrain { header, body })
}

/// Cuts a train into packets of at most [`MAX_PAIRS`] pairs (see the
/// [module documentation](self)): every packet has `index`, `header` and `flags`, every packet
/// but the last also [`CONTINUED`]. A train without pairs is one packet.
///
/// ```
/// use sdrk_proto::pulses::{self, Header, Modulation, Pair, CONTINUED, MAX_PAIRS};
/// use sdrk_proto::PacketFlags;
///
/// let header = Header { modulation: Modulation::Fsk, channel: 1, freq_offset_hz: 250_000,
///     level_db10: -200, noise_db10: -450, rate_hz: 500_000 };
/// let pairs = [Pair { pulse: 10, gap: 20 }; 70];
/// let parts: Vec<_> = pulses::packets(7, PacketFlags::NONE, &header, &pairs).collect::<Result<_, _>>()?;
/// assert_eq!(parts.len(), 3);
/// assert!(parts[0].flags.contains(CONTINUED) && !parts[2].flags.contains(CONTINUED));
/// assert_eq!(pulses::decode(parts[2].data())?.len(), 70 - 2 * MAX_PAIRS);
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
pub fn packets<'a>(
    index: u64,
    flags: PacketFlags,
    header: &Header,
    pairs: &'a [Pair],
) -> Packets<'a> {
    Packets {
        index,
        flags,
        header: *header,
        rest: pairs,
        first: true,
    }
}

/// The packets of a train, from [`packets`].
#[derive(Debug, Clone)]
pub struct Packets<'a> {
    index: u64,
    flags: PacketFlags,
    header: Header,
    rest: &'a [Pair],
    first: bool,
}

impl Iterator for Packets<'_> {
    type Item = Result<Packet>;

    fn next(&mut self) -> Option<Result<Packet>> {
        if self.rest.is_empty() && !self.first {
            return None;
        }
        self.first = false;
        let (part, rest) = self.rest.split_at(self.rest.len().min(MAX_PAIRS));
        self.rest = rest;
        let mut flags = self.flags;
        if !rest.is_empty() {
            flags |= CONTINUED;
        }
        Some(encode_packet(self.index, flags, &self.header, part))
    }
}

/// A train joined from its packets by an [`Assembler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Assembled<'a> {
    /// The index of the packets.
    pub index: u64,
    /// The flags of the last packet without [`CONTINUED`]; [`PacketFlags::TRUNCATED`] is added
    /// when the storage was too small for every pair.
    pub flags: PacketFlags,
    /// The header of the packets.
    pub header: Header,
    /// The pairs of all packets in order.
    pub pairs: &'a [Pair],
}

/// Joins the packets of [`packets`] into whole trains, in caller storage (no heap).
///
/// ```
/// use sdrk_proto::pulses::{self, Assembler, Header, Modulation, Pair};
/// use sdrk_proto::PacketFlags;
///
/// let header = Header { modulation: Modulation::Ook, channel: 0, freq_offset_hz: 0,
///     level_db10: 0, noise_db10: -300, rate_hz: 250_000 };
/// let pairs: Vec<Pair> = (0..45).map(|k| Pair { pulse: k, gap: 2 * k }).collect();
/// let mut asm = Assembler::new([Pair::default(); 64]);
/// let mut done = None;
/// for p in pulses::packets(9, PacketFlags::NONE, &header, &pairs) {
///     if let Some(t) = asm.push(&p?)? {
///         done = Some((t.index, t.pairs.to_vec()));
///     }
/// }
/// assert_eq!(done, Some((9, pairs)));
/// # Ok::<(), sdrk_proto::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Assembler<S> {
    storage: S,
    len: usize,
    truncated: bool,
    open: Option<(u64, Header)>,
}

impl<S: AsRef<[Pair]> + AsMut<[Pair]>> Assembler<S> {
    /// An assembler that holds up to `storage.len()` pairs of one train.
    pub fn new(storage: S) -> Self {
        Self {
            storage,
            len: 0,
            truncated: false,
            open: None,
        }
    }

    /// Drops a partly received train.
    pub fn reset(&mut self) {
        self.len = 0;
        self.truncated = false;
        self.open = None;
    }

    /// Takes the next packet; returns the whole train when `packet` is its last part.
    ///
    /// A packet with another index or header than the train in progress starts a new train
    /// (the old one is dropped: one of its packets was lost). Pairs beyond the storage are
    /// dropped and the train is marked [`PacketFlags::TRUNCATED`].
    ///
    /// # Errors
    /// The errors of [`decode`]; the train in progress is dropped.
    pub fn push(&mut self, packet: &Packet) -> Result<Option<Assembled<'_>>> {
        let train = match decode(packet.data()) {
            Ok(t) => t,
            Err(e) => {
                self.reset();
                return Err(e);
            }
        };
        let key = (packet.index, train.header());
        if self.open != Some(key) {
            self.reset();
            self.open = Some(key);
        }
        let store = self.storage.as_mut();
        for pair in train.pairs() {
            match store.get_mut(self.len) {
                Some(slot) => {
                    *slot = pair;
                    self.len += 1;
                }
                None => self.truncated = true,
            }
        }
        if packet.flags.contains(CONTINUED) {
            return Ok(None);
        }
        let mut flags = PacketFlags(packet.flags.bits() & !CONTINUED.bits());
        if self.truncated {
            flags |= PacketFlags::TRUNCATED;
        }
        let len = self.len;
        self.len = 0;
        self.truncated = false;
        self.open = None;
        Ok(Some(Assembled {
            index: packet.index,
            flags,
            header: train.header(),
            pairs: self.storage.as_ref().get(..len).unwrap_or(&[]),
        }))
    }
}

/// Writes a value of dB × 10 with one decimal (`-125` as `-12.5`).
fn write_db10<W: fmt::Write>(w: &mut W, v: i16) -> fmt::Result {
    let v = i32::from(v);
    let sign = if v < 0 { "-" } else { "" };
    let a = v.unsigned_abs();
    write!(w, "{sign}{}.{}", a / 10, a % 10)
}

/// Writes a train as one JSON object (see the [module documentation](self)): `mod`, `freq_hz`
/// (`center_hz` plus the header's offset: the absolute frequency when `center_hz` is the RF
/// centre of the band, the offset when it is 0), `rssi_db` (the level), `noise_db`, `channel`,
/// `rate_hz` and `pulses`, in this order, without spaces.
///
/// # Errors
/// The errors of `w`.
pub fn write_json<W: fmt::Write>(
    w: &mut W,
    header: &Header,
    pairs: impl IntoIterator<Item = Pair>,
    center_hz: i64,
) -> fmt::Result {
    let freq = center_hz.saturating_add(i64::from(header.freq_offset_hz));
    write!(
        w,
        "{{\"mod\":\"{}\",\"freq_hz\":{freq},\"rssi_db\":",
        header.modulation.as_str()
    )?;
    write_db10(w, header.level_db10)?;
    w.write_str(",\"noise_db\":")?;
    write_db10(w, header.noise_db10)?;
    write!(
        w,
        ",\"channel\":{},\"rate_hz\":{},\"pulses\":[",
        header.channel, header.rate_hz
    )?;
    for (k, p) in pairs.into_iter().enumerate() {
        if k > 0 {
            w.write_char(',')?;
        }
        write!(w, "[{},{}]", p.pulse, p.gap)?;
    }
    w.write_str("]}")
}
