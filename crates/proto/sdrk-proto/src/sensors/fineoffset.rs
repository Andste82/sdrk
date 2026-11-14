//! The shared parts of the Fine Offset / Ecowitt FSK devices (decision D1152).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433`
//! decoders (written by a third party, nothing copied). The devices send NRZ bits at 58 us (the
//! `WH55` at 60 us, the `WH31E` at 56 us) with the upper frequency as the pulse: a preamble of
//! bytes 0xAA, the sync word 0x2D 0xD4 and a message that starts with a family byte. The
//! message ends with a CRC-8 (polynomial 0x31, init 0, MSB first) over the bytes before it and,
//! for most devices, a last byte that is the sum of all bytes before it modulo 256.
//!
//! The decoder side finds the pattern 0xAA2DD4 followed by a known family byte in a row of the
//! [`Nrz`] slicer, reads the message and checks both bytes; the encoder side writes the
//! preamble, the sync word and the message as the runs of an FSK burst. Zeros at the end of a
//! message are not seen as part of the burst: up to seven missing bits are read as zeros, a
//! message with more of them is [`EncodeError::Unrepresentable`] (as for `lacrosse_tx29`).

use super::encoder::{EncodeError, TrainSink};
use super::util::crc8_msb;
use super::{DecodeResult, Reading, ReadingSink};
use crate::pulses::{Header, PulseTrain};
use crate::slicers::{Nrz, Row, Rows, Slicer};

/// A gap at least this long ends a row: a message holds up to 80 zero bits in a row.
pub(super) const RESET_US: f32 = 5000.0;
/// The pause between two copies of a message (above the reset time of the devices).
pub(super) const REPEAT_GAP_US: u32 = 12_000;
/// The largest message of the family in bytes (the `WS90` has 32).
pub(super) const MAX_BYTES: usize = 32;

/// The layout of one kind of message: the family byte, the length in bytes, the index of the
/// CRC byte and whether a sum byte follows it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Frame {
    pub family: u8,
    pub len: usize,
    pub crc_at: usize,
    pub sum: bool,
}

/// What the closure of a device gets: the message bytes and the header of the train.
pub(super) type Build = fn(&[u8], &Header) -> Result<Reading, DecodeResult>;

/// Sum of bytes modulo 256.
pub(super) fn sum8(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0u8, |a, &b| a.wrapping_add(b))
}

/// Decodes `train` for a device of `frames` at `bit_us` per bit: every row is searched for the
/// pattern `AA 2D D4` and a family byte of `frames`.
pub(super) fn decode(
    train: &PulseTrain<'_>,
    out: &mut ReadingSink,
    bit_us: f32,
    frames: &[Frame],
    build: Build,
) -> DecodeResult {
    let Ok(slicer) = Nrz::new(bit_us, RESET_US) else {
        return DecodeResult::NoMatch;
    };
    let mut rows = Rows::new();
    slicer.slice(train, &mut rows);
    let header = train.header();
    super::util::collect(&rows, out, |row| {
        decode_row(row, &header, frames, build)
    })
}

fn decode_row(
    row: &Row,
    header: &Header,
    frames: &[Frame],
    build: Build,
) -> Result<Reading, DecodeResult> {
    let mut res = DecodeResult::NoMatch;
    let Some(last) = row.len().checked_sub(24) else {
        return Err(res);
    };
    for p in 0..=last {
        if row.field(p, 24) != Some(0x00AA_2DD4) {
            continue;
        }
        let first = p + 24;
        let Some(fam) = row.field(first, 8) else {
            res = res.best(DecodeResult::Truncated);
            continue;
        };
        let Some(f) = frames.iter().find(|f| u32::from(f.family) == fam) else {
            continue;
        };
        match decode_frame(row, first, f, header, build) {
            Ok(r) => return Ok(r),
            Err(e) => res = res.best(e),
        }
    }
    Err(res)
}

fn decode_frame(
    row: &Row,
    first: usize,
    f: &Frame,
    header: &Header,
    build: Build,
) -> Result<Reading, DecodeResult> {
    // up to seven missing bits at the end are zeros (the burst ends with the last 1); rows cut
    // by an error keep only a prefix
    let missing = (first + 8 * f.len).saturating_sub(row.len());
    if missing > 7 || (missing > 0 && row.end().is_error()) {
        return Err(DecodeResult::Truncated);
    }
    let mut b = [0u8; MAX_BYTES];
    let bytes = b.get_mut(..f.len).ok_or(DecodeResult::NoMatch)?;
    for (i, v) in bytes.iter_mut().enumerate() {
        let mut x = 0u8;
        for j in 0..8 {
            x = (x << 1) | u8::from(row.bit(first + 8 * i + j).unwrap_or(false));
        }
        *v = x;
    }
    let bytes: &[u8] = bytes;
    let crc = bytes.get(f.crc_at).copied().ok_or(DecodeResult::NoMatch)?;
    let covered = bytes.get(..f.crc_at).ok_or(DecodeResult::NoMatch)?;
    let mut ok = crc8_msb(0, covered.iter().copied()) == crc;
    if f.sum {
        let upto = bytes.get(..=f.crc_at).ok_or(DecodeResult::NoMatch)?;
        ok = ok && bytes.get(f.crc_at + 1).copied() == Some(sum8(upto));
    }
    if !ok {
        return Err(DecodeResult::BadIntegrity);
    }
    build(bytes, header)
}

/// Writes `repeats` copies of a message as the runs of an FSK burst: `preamble` bytes 0xAA, the
/// sync word, `msg` (the family byte first; the CRC and sum bytes are set here from the layout
/// `f`), `bit_us` per bit, the upper frequency as the pulse.
pub(super) fn encode(
    msg: &mut [u8],
    f: &Frame,
    preamble: usize,
    bit_us: u32,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    let crc = crc8_msb(0, msg.iter().take(f.crc_at).copied());
    if let Some(c) = msg.get_mut(f.crc_at) {
        *c = crc;
    }
    if f.sum {
        let s = sum8(msg.get(..=f.crc_at).unwrap_or(&[]));
        if let Some(c) = msg.get_mut(f.crc_at + 1) {
            *c = s;
        }
    }
    let mut bytes = [0u8; MAX_BYTES + 12];
    let head = preamble + 2;
    let total = head + msg.len();
    let dst = bytes.get_mut(..total).ok_or(EncodeError::BufferTooSmall)?;
    for (i, v) in dst.iter_mut().enumerate() {
        *v = match i {
            _ if i < preamble => 0xAA,
            _ if i == preamble => 0x2D,
            _ if i == preamble + 1 => 0xD4,
            _ => msg.get(i - head).copied().unwrap_or(0),
        };
    }
    let bit = |k: usize| bytes.get(k / 8).is_some_and(|b| b >> (7 - k % 8) & 1 == 1);
    let mut n = total * 8;
    while n > 0 && !bit(n - 1) {
        n -= 1;
    }
    if total * 8 - n > 7 {
        return Err(EncodeError::Unrepresentable);
    }
    out.atomic(|o| {
        o.copies(repeats, REPEAT_GAP_US, |o| {
            let mut k = 0;
            while k < n {
                // the burst starts with a 1: a pulse, then a gap, alternately
                let mut pulse = 0;
                while k < n && bit(k) {
                    pulse += 1;
                    k += 1;
                }
                let mut gap = 0;
                while k < n && !bit(k) {
                    gap += 1;
                    k += 1;
                }
                o.push(pulse * bit_us, gap * bit_us)?;
            }
            Ok(())
        })
    })
}

/// `num / den` rounded to three decimals, as the `f32` that the three-decimal JSON text of the
/// value reads back to (decision D1152: a level that the JSON form carries must come back
/// equal).
pub(super) fn milli(num: i32, den: i32) -> f32 {
    let n = i64::from(num) * 1000;
    let d = i64::from(den);
    // half away from zero
    let k = if n >= 0 {
        (2 * n + d) / (2 * d)
    } else {
        -((-2 * n + d) / (2 * d))
    };
    (k as i32) as f32 / 1000.0
}

/// Implements the decoder and the encoder trait for a device type of this family from a decode
/// function `fn(&PulseTrain, &mut ReadingSink) -> DecodeResult` and an encode function
/// `fn(&[&str], &Reading, u32, &mut TrainSink) -> Result<(), EncodeError>` (the models are the
/// first argument). The device sends two copies by default.
macro_rules! fsk_device {
    ($ty:ident, $name:literal, $models:expr, $decode:path, $encode:path) => {
        impl $crate::sensors::SensorDecoder for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn modulation(&self) -> $crate::pulses::Modulation {
                $crate::pulses::Modulation::Fsk
            }
            fn decode(
                &self,
                train: &$crate::pulses::PulseTrain<'_>,
                out: &mut $crate::sensors::ReadingSink,
            ) -> $crate::sensors::DecodeResult {
                $decode(train, out)
            }
        }
        impl $crate::sensors::SensorEncoder for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn models(&self) -> &'static [&'static str] {
                &$models
            }
            fn modulation(&self) -> $crate::pulses::Modulation {
                $crate::pulses::Modulation::Fsk
            }
            fn mic(&self) -> $crate::sensors::Mic {
                $crate::sensors::Mic::Crc
            }
            fn default_repeats(&self) -> u32 {
                2
            }
            fn repeat_gap_us(&self) -> u32 {
                $crate::sensors::fineoffset::REPEAT_GAP_US
            }
            fn encode(
                &self,
                r: &$crate::sensors::Reading,
                repeats: u32,
                out: &mut $crate::sensors::TrainSink<'_>,
            ) -> Result<(), $crate::sensors::EncodeError> {
                $encode(self.models(), r, repeats, out)
            }
        }
    };
}
pub(super) use fsk_device;
