//! Oregon Scientific v2.1 sensors THN132N / THR238NF (id EC40, temperature) and THGR122NX /
//! THGN123N (id 1D20, temperature and humidity), 433 MHz, on-off keying.
//!
//! Source: "Oregon Scientific RF Protocol Description (Versions 1.0, 2.1, 3.0)", June 2011;
//! see decision D1108. Manchester coding at 1024 bit/s (half bit 488 us), a 0 is a rising and a
//! 1 a falling edge in the middle of the bit (IEEE 802.3), and in version 2.1 every data bit is
//! sent twice: first inverted, then as it is (the decoder keeps the second bit of each pair and
//! checks that the two differ). The data bits are: a preamble of sixteen 1 bits, the sync nibble
//! 0xA, the payload, a checksum and a post-amble of two nibbles that is ignored. Every nibble is
//! sent least significant bit first. Payload nibbles: 0 to 3 the sensor id, 4 the channel
//! (1, 2 or 4 for the channels 1 to 3), 5 and 6 the rolling code (low nibble first, used as the
//! device id), 7 the flags (bit 0x4: battery low), 8 to 10 the temperature as BCD digits
//! (tenths first), 11 the sign (non-zero: negative); for the id 1D20 also 12 and 13 the
//! humidity (ones first) and 14 an unknown nibble. The checksum is the sum of the nibbles from
//! the id up to the checksum, sent as two nibbles, low nibble first; the message has 14 nibbles
//! for EC40 and 17 for 1D20 including the checksum.

use super::encoder::{
    self, BATTERY, CHANNEL, Carries, EncodeError, HUMIDITY, SensorEncoder, TrainSink,
};
use super::util::collect;
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Header, Modulation, PulseTrain};
use crate::slicers::{Convention, Manchester, Row, Rows, Slicer};

/// The decoder of the Oregon Scientific v2.1 sensors.
#[derive(Debug, Clone, Copy, Default)]
pub struct OregonV21;

/// The most preamble bits that are skipped before the sync nibble.
const MAX_PREAMBLE: usize = 24;
/// The least preamble bits (a detector may lose the first ones).
const MIN_PREAMBLE: usize = 8;

impl SensorDecoder for OregonV21 {
    fn name(&self) -> &'static str {
        "oregon_v21"
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
        let Ok(slicer) = Manchester::new(1.0e6 / 2048.0, 3000.0, Convention::Ieee8023) else {
            return DecodeResult::NoMatch;
        };
        let slicer = slicer.with_virtual_lead(true);
        let mut rows = Rows::new();
        slicer.slice(train, &mut rows);
        let header = train.header();
        collect(&rows, out, |row| decode_row(row, &header))
    }
}

/// Data bit `k`: the second bit of the pair `k`; `Err(Truncated)` past the end of the row and
/// `Err(NoMatch)` for a pair of equal bits.
fn data_bit(row: &Row, k: usize) -> Result<bool, DecodeResult> {
    let (Some(a), Some(b)) = (row.bit(2 * k), row.bit(2 * k + 1)) else {
        return Err(DecodeResult::Truncated);
    };
    if a == b {
        Err(DecodeResult::NoMatch)
    } else {
        Ok(b)
    }
}

fn decode_row(row: &Row, header: &Header) -> Result<Reading, DecodeResult> {
    let mut ones = 0;
    while ones <= MAX_PREAMBLE && data_bit(row, ones) == Ok(true) {
        ones += 1;
    }
    if !(MIN_PREAMBLE..=MAX_PREAMBLE).contains(&ones) {
        return Err(DecodeResult::NoMatch);
    }
    for (j, want) in [false, true, false, true].into_iter().enumerate() {
        if data_bit(row, ones + j)? != want {
            return Err(DecodeResult::NoMatch);
        }
    }
    let base = ones + 4;
    let nibble = |i: usize| -> Result<u8, DecodeResult> {
        let mut v = 0u8;
        for j in 0..4 {
            v |= u8::from(data_bit(row, base + 4 * i + j)?) << j;
        }
        Ok(v)
    };
    let mut id = 0u32;
    for i in 0..4 {
        id |= u32::from(nibble(i)?) << (4 * (3 - i));
    }
    let (n, model, has_humidity) = match id {
        0xEC40 => (14, "oregon_thn132n", false),
        0x1D20 => (17, "oregon_thgr122nx", true),
        _ => return Err(DecodeResult::NoMatch),
    };
    let mut nib = [0u8; 17];
    for (i, slot) in nib.iter_mut().enumerate().take(n) {
        *slot = nibble(i)?;
    }
    let sum: u32 = nib.iter().take(n - 2).map(|&v| u32::from(v)).sum();
    let (Some(lo), Some(hi)) = (nib.get(n - 2), nib.get(n - 1)) else {
        return Err(DecodeResult::Truncated);
    };
    let check = u32::from(*lo) | (u32::from(*hi) << 4);
    if sum & 0xFF != check {
        return Err(DecodeResult::BadIntegrity);
    }
    let channel = match nib[4] {
        1 => 1,
        2 => 2,
        4 => 3,
        _ => return Err(DecodeResult::OutOfRange),
    };
    let (t0, t1, t2) = (nib[8], nib[9], nib[10]);
    if t0 > 9 || t1 > 9 || t2 > 9 {
        return Err(DecodeResult::OutOfRange);
    }
    let mag = i32::from(t2) * 100 + i32::from(t1) * 10 + i32::from(t0);
    if mag > 700 {
        return Err(DecodeResult::OutOfRange);
    }
    let tenths = if nib[11] != 0 { -mag } else { mag };
    let mut r = Reading::new(
        model,
        u32::from(nib[5]) | (u32::from(nib[6]) << 4),
        Mic::Checksum,
        header,
    );
    r.channel = Some(channel);
    r.battery_ok = Some(nib[7] & 0x4 == 0);
    r.temperature_c = Some(tenths as f32 / 10.0);
    if has_humidity {
        let (h0, h1) = (nib[12], nib[13]);
        if h0 > 9 || h1 > 9 || h1 * 10 + h0 > 100 {
            return Err(DecodeResult::OutOfRange);
        }
        r.humidity = Some(f32::from(h1 * 10 + h0));
    }
    Ok(r)
}

/// The half bit in microseconds: 1 / 2048 s = 488.28 us, rounded to a whole microsecond
/// (0.06 percent short).
const HALF_BIT_US: u32 = 488;
/// The pause between two copies (the repeat gap, decision D1124).
const REPEAT_GAP_US: u32 = 8000;
/// The levels of a message: every data bit is sent twice and every bit has two half bits.
const MAX_LEVELS: usize = 4 * (16 + 4 + 4 * 19);

fn run_len(levels: &[bool], from: usize) -> usize {
    let first = levels.get(from).copied();
    levels
        .iter()
        .skip(from)
        .take_while(|&&l| Some(l) == first)
        .count()
}

/// Writes the Manchester coded message of `nib` (the nibbles from the sensor id to the
/// checksum): preamble of sixteen 1 bits, the sync nibble 0xA, the nibbles and the post-amble of
/// two nibbles, every nibble least significant bit first and every data bit twice, first
/// inverted. A 1 is on-off, a 0 off-on; the idle half bit before a first 0 and after a last 1
/// is not sent.
fn write_message(out: &mut TrainSink<'_>, nib: &[u8]) -> Result<(), EncodeError> {
    let mut levels = [false; MAX_LEVELS];
    let mut n = 0usize;
    let mut put = |bit: bool| {
        // a 1 bit is high then low, a 0 bit low then high; leading low half bits are not sent
        for level in [bit, !bit] {
            if n == 0 && !level {
                continue;
            }
            if let Some(slot) = levels.get_mut(n) {
                *slot = level;
                n += 1;
            }
        }
    };
    let mut data = |bit: bool| {
        put(!bit);
        put(bit);
    };
    for _ in 0..16 {
        data(true);
    }
    for j in 0..4 {
        data(0xA >> j & 1 == 1);
    }
    for &v in nib.iter().chain(&[0u8, 0]) {
        for j in 0..4 {
            data(v >> j & 1 == 1);
        }
    }
    // a trailing low half bit is the end of the train
    while n > 0 && levels.get(n - 1) == Some(&false) {
        n -= 1;
    }
    let levels = levels.get(..n).unwrap_or(&[]);
    let mut i = 0;
    while i < levels.len() {
        let pulse = run_len(levels, i);
        i += pulse;
        let gap = if i < levels.len() {
            let g = run_len(levels, i);
            i += g;
            g
        } else {
            0
        };
        out.push(pulse as u32 * HALF_BIT_US, gap as u32 * HALF_BIT_US)?;
    }
    Ok(())
}

impl SensorEncoder for OregonV21 {
    fn name(&self) -> &'static str {
        "oregon_v21"
    }

    fn models(&self) -> &'static [&'static str] {
        &["oregon_thn132n", "oregon_thgr122nx"]
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn mic(&self) -> Mic {
        Mic::Checksum
    }

    fn default_repeats(&self) -> u32 {
        2
    }

    fn repeat_gap_us(&self) -> u32 {
        REPEAT_GAP_US
    }

    /// The message of the decoder for `oregon_thn132n` (id EC40, 14 nibbles) or
    /// `oregon_thgr122nx` (id 1D20, with humidity, 17 nibbles): the channel 1 to 3 as the nibble
    /// 1, 2 or 4, the rolling code (the reading's `id`, below 256), the battery flag 0x4, the
    /// temperature from -70.0 to 70.0 as BCD digits and the nibble sum as checksum; the
    /// unknown nibble of the THGR122NX is sent as 0; its humidity is two BCD digits (0 to 99). Half bits of 488 us, copies 8 ms apart.
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encoder::model(r, self.models())?;
        let thgr = r.model == "oregon_thgr122nx";
        Carries(CHANNEL | BATTERY | (if thgr { HUMIDITY } else { 0 })).check(r)?;
        let rolling = encoder::id(r, 256)?;
        let ch = encoder::channel(r, 3)?;
        let t = encoder::temperature(r, 10.0, -700, 700)?;
        let mut nib = [0u8; 17];
        let dev: u16 = if thgr { 0x1D20 } else { 0xEC40 };
        let count = if thgr { 17 } else { 14 };
        let mag = t.unsigned_abs();
        let digits = [
            (dev >> 12) as u8 & 0xF,
            (dev >> 8) as u8 & 0xF,
            (dev >> 4) as u8 & 0xF,
            dev as u8 & 0xF,
            1 << (ch - 1),
            (rolling & 0xF) as u8,
            (rolling >> 4) as u8,
            if r.battery_ok == Some(false) { 0x4 } else { 0 },
            (mag % 10) as u8,
            (mag / 10 % 10) as u8,
            (mag / 100) as u8,
            if t < 0 { 8 } else { 0 },
        ];
        for (slot, v) in nib.iter_mut().zip(digits) {
            *slot = v;
        }
        if thgr {
            // two BCD digits: the decoder reads 0 to 99
            let h = encoder::humidity(r)?;
            if h > 99 {
                return Err(EncodeError::OutOfRange("humidity"));
            }
            for (slot, v) in nib.iter_mut().skip(12).zip([h % 10, h / 10, 0]) {
                *slot = v;
            }
        }
        let body = count - 2;
        let sum: u32 = nib.iter().take(body).map(|&v| u32::from(v)).sum();
        for (slot, v) in nib.iter_mut().skip(body).zip([sum & 0xF, (sum >> 4) & 0xF]) {
            *slot = v as u8;
        }
        let msg = nib.get(..count).unwrap_or(&[]);
        out.atomic(|o| o.copies(repeats, REPEAT_GAP_US, |o| write_message(o, msg)))
    }
}
