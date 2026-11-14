//! Ecowitt `WH53` / `WH0280` / `WH0281A` outdoor thermometer (433 MHz, on-off keying).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433`
//! decoders (written by a third party, nothing copied; decision D1151). Pulse-width coding: a 1
//! is a pulse of 500 us, a 0 one of 1480 us, every pulse is followed by a gap of 1000 us (the document names 1500 us as the longest gap inside a row,
//! not the nominal one: an assumption, decision D1151). A
//! message has 55 bits, MSB first: seven 1 bits (the head, unstable on the air), the model code
//! 0x53, an 8 bit id, two 0 bits, the channel minus one in two bits, two 0 bits, the
//! temperature as ten bits, the byte 0xFF and a CRC-8 (polynomial 0x31, init 0, MSB first) over
//! the five bytes from the model code to the 0xFF. The temperature is `(raw - 400) / 10`
//! degrees.

use super::encoder::{self, CHANNEL, Carries, EncodeError, SensorEncoder, TrainSink};
use super::util::{collect, crc8_msb, prefix_result};
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Header, Modulation, PulseTrain};
use crate::slicers::{Pwm, Row, Rows, Slicer};

/// The decoder and encoder of the WH53.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ecowitt;

/// The model string (the name `rtl_433` prints).
const MODEL: &str = "Ecowitt-WH53";
/// The message length in bits.
const BITS: usize = 55;
/// The gap after every pulse: the document gives 1500 us as the longest gap inside a row (the
/// row end is 2000 us) and no nominal gap, so the 1000 us of the other devices of the maker is
/// used (decision D1151).
const GAP_US: u32 = 1000;
/// The pause between two copies of a message in a train.
const REPEAT_GAP_US: u32 = 8000;

impl SensorDecoder for Ecowitt {
    fn name(&self) -> &'static str {
        "ecowitt"
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
        let Ok(slicer) = Pwm::new(500.0, 1480.0, 1000.0, 2500.0, true) else {
            return DecodeResult::NoMatch;
        };
        let mut rows = Rows::new();
        slicer.slice(train, &mut rows);
        let header = train.header();
        collect(&rows, out, |row| decode_row(row, &header))
    }
}

fn decode_row(row: &Row, header: &Header) -> Result<Reading, DecodeResult> {
    // the head has four stable bits (the 12 bit pattern 0xF53 starts at bit 3)
    let start = row.field(3, 4) == Some(0xF);
    if row.end().is_error() {
        return Err(prefix_result(row, start));
    }
    if !start {
        return Err(DecodeResult::NoMatch);
    }
    if row.len() < BITS {
        return Err(DecodeResult::Truncated);
    }
    if row.len() > BITS || row.field(7, 8) != Some(0x53) {
        return Err(DecodeResult::NoMatch);
    }
    let mut b = [0u8; 6];
    for (i, v) in b.iter_mut().enumerate() {
        let mut x = 0u8;
        for j in 0..8 {
            x = (x << 1) | u8::from(row.bit(7 + 8 * i + j).unwrap_or(false));
        }
        *v = x;
    }
    let [m, id, b2, b3, ff, crc] = b;
    if b2 & 0xCC != 0 || ff != 0xFF {
        return Err(DecodeResult::NoMatch);
    }
    if crc8_msb(0, [m, id, b2, b3, ff]) != crc {
        return Err(DecodeResult::BadIntegrity);
    }
    let channel = (b2 >> 4) & 3;
    if channel > 2 {
        return Err(DecodeResult::OutOfRange);
    }
    let raw = (u32::from(b2 & 3) << 8) | u32::from(b3);
    let mut r = Reading::new(MODEL, u32::from(id), Mic::Crc, header);
    r.channel = Some(channel + 1);
    r.temperature_c = Some((raw as f32 - 400.0) / 10.0);
    Ok(r)
}

impl SensorEncoder for Ecowitt {
    fn name(&self) -> &'static str {
        "ecowitt"
    }

    fn models(&self) -> &'static [&'static str] {
        &[MODEL]
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn mic(&self) -> Mic {
        Mic::Crc
    }

    fn default_repeats(&self) -> u32 {
        3
    }

    fn repeat_gap_us(&self) -> u32 {
        REPEAT_GAP_US
    }

    /// The 55 bits of the decoder (seven 1 bits, 0x53, the id below 256, the channel 1 to 3,
    /// the temperature from -40.0 to 62.3, 0xFF, the CRC-8) in the pulse widths 500 us (1) and
    /// 1480 us (0) with gaps of 1000 us, copies 8 ms apart. `battery_ok` and `humidity` are not
    /// carried.
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encoder::model(r, self.models())?;
        Carries(CHANNEL).check(r)?;
        let id = encoder::id(r, 256)?;
        let ch = encoder::channel(r, 3)? - 1;
        let raw = encoder::temperature(r, 10.0, -400, 623)? + 400;
        let b = [
            0x53,
            id as u8,
            (ch << 4) | (raw >> 8) as u8,
            raw as u8,
            0xFF,
        ];
        let crc = crc8_msb(0, b);
        // the seven head bits and the six bytes: 55 bits
        let mut bits = [true; BITS];
        for (k, bit) in bits.iter_mut().enumerate().skip(7) {
            let byte = if k < 47 {
                b.get((k - 7) / 8).copied().unwrap_or(0)
            } else {
                crc
            };
            let n = if k < 47 { (k - 7) % 8 } else { k - 47 };
            *bit = byte >> (7 - n) & 1 == 1;
        }
        out.atomic(|o| {
            o.copies(repeats, REPEAT_GAP_US, |o| {
                for &bit in &bits {
                    o.push(if bit { 500 } else { 1480 }, GAP_US)?;
                }
                Ok(())
            })
        })
    }
}
