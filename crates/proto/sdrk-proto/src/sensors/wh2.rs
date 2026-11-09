//! Fine Offset / Ecowitt `WH2` temperature and humidity sensor (433 MHz, on-off keying).
//!
//! Source: the public protocol description of the `WH2` in the blog "Weather station hacking,
//! part 2" (L. Small) and the README of the `BetterWH2` project (public domain); see decision
//! D1106 of the decisions log. Pulse-width coding: a 1 is a pulse of 500 us, a 0 one of
//! 1500 us, every pulse is followed by a gap of 1000 us. A message has 48 bits, MSB first:
//! a preamble of eight 1 bits, 12 bits of type and id (the first one always 0), a 12 bit
//! temperature in tenths of a degree (sign and magnitude), an 8 bit humidity and an 8 bit
//! CRC-8 of the polynomial 0x31 over the four payload bytes. The CRC variant (reflected,
//! init 0, as in the `OneWire` library the description names) is an inference not yet
//! confirmed on a real capture.

use super::encoder::{self, Carries, EncodeError, HUMIDITY, SensorEncoder, TrainSink};
use super::util::{byte_at, collect, crc8_reflected, prefix_result};
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Header, Modulation, PulseTrain};
use crate::slicers::{Pwm, Row, Rows, Slicer};

/// The decoder of the WH2.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wh2;

/// The largest temperature magnitude accepted, tenths of a degree.
const MAX_TEMP_TENTHS: u32 = 800;

impl SensorDecoder for Wh2 {
    fn name(&self) -> &'static str {
        "wh2"
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
        let Ok(slicer) = Pwm::new(500.0, 1500.0, 1000.0, 2500.0, true) else {
            return DecodeResult::NoMatch;
        };
        let mut rows = Rows::new();
        slicer.slice(train, &mut rows);
        let header = train.header();
        collect(&rows, out, |row| decode_row(row, &header))
    }
}

fn decode_row(row: &Row, header: &Header) -> Result<Reading, DecodeResult> {
    let start = row.field(0, 8) == Some(0xFF);
    if row.end().is_error() {
        return Err(prefix_result(row, start));
    }
    if !start {
        return Err(DecodeResult::NoMatch);
    }
    if row.len() < 48 {
        return Err(DecodeResult::Truncated);
    }
    if row.len() > 48 {
        return Err(DecodeResult::NoMatch);
    }
    let mut b = [0u8; 5];
    for (i, v) in b.iter_mut().enumerate() {
        *v = byte_at(row, i + 1).ok_or(DecodeResult::Truncated)?;
    }
    let [b0, b1, b2, b3, crc] = b;
    if b0 & 0x80 != 0 {
        return Err(DecodeResult::NoMatch);
    }
    if crc8_reflected([b0, b1, b2, b3]) != crc {
        return Err(DecodeResult::BadIntegrity);
    }
    let raw = (u32::from(b1 & 0x0F) << 8) | u32::from(b2);
    let mag = raw & 0x7FF;
    if mag > MAX_TEMP_TENTHS || b3 > 100 {
        return Err(DecodeResult::OutOfRange);
    }
    let t = mag as f32 / 10.0;
    let id = (u32::from(b0) << 4) | u32::from(b1 >> 4);
    let mut r = Reading::new("wh2", id, Mic::Crc, header);
    r.temperature_c = Some(if raw & 0x800 != 0 { -t } else { t });
    r.humidity = Some(f32::from(b3));
    Ok(r)
}

/// The pause between two copies of a message in a train (the repeat gap, decision D1124).
const REPEAT_GAP_US: u32 = 8000;

impl SensorEncoder for Wh2 {
    fn name(&self) -> &'static str {
        "wh2"
    }

    fn models(&self) -> &'static [&'static str] {
        &["wh2"]
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

    /// The 48 bits of the decoder (eight 1 bits, the 12 bit id, the temperature as sign and
    /// magnitude in tenths up to 80.0, the humidity and the CRC-8 of the decoder), every copy
    /// in the pulse widths 500 us (1) and 1500 us (0) with gaps of 1000 us, copies 8 ms apart.
    /// The id is below 0x800; `channel` and `battery_ok` are not carried.
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encoder::model(r, self.models())?;
        Carries(HUMIDITY).check(r)?;
        let id = encoder::id(r, 0x800)?;
        let max = MAX_TEMP_TENTHS as i32;
        let t = encoder::temperature(r, 10.0, -max, max)?;
        let h = encoder::humidity(r)?;
        let raw = (if t < 0 { 0x800 } else { 0 }) | t.unsigned_abs();
        let b = [
            (id >> 4) as u8,
            (((id & 0xF) << 4) | (raw >> 8)) as u8,
            raw as u8,
            h,
        ];
        let crc = crc8_reflected(b);
        let msg = [0xFF, b[0], b[1], b[2], b[3], crc];
        out.atomic(|o| {
            o.copies(repeats, REPEAT_GAP_US, |o| {
                o.pwm_bytes(&msg, 500, 1500, 1000)
            })
        })
    }
}
