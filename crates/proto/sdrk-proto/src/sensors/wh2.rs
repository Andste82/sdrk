//! Fine Offset / Ecowitt `WH2` temperature and humidity sensor (433 MHz, on-off keying).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433` decoders
//! (written by a third party, nothing copied; decision D1137) over the earlier basis of
//! decision D1106 (the blog "Weather station hacking, part 2" and the README of the `BetterWH2`
//! project). Pulse-width coding: a 1 is a pulse of 500 us, a 0 one of 1500 us, every pulse is
//! followed by a gap of 1000 us. A message has 48 bits, MSB first: a preamble of eight 1 bits,
//! the type nibble 4, an 8 bit id, a 12 bit temperature in tenths of a degree (sign and
//! magnitude), an 8 bit humidity (0xFF: no humidity sensor) and an 8 bit CRC-8 (polynomial
//! 0x31, init 0, MSB first) over the four payload bytes. The variants of the family with other
//! lengths (`WH2A`, `WH5`, Proove) are not decoded.

use super::encoder::{self, Carries, EncodeError, HUMIDITY, SensorEncoder, TrainSink};
use super::util::{byte_at, collect, crc8_msb, prefix_result};
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Header, Modulation, PulseTrain};
use crate::slicers::{Pwm, Row, Rows, Slicer};

/// The decoder of the WH2.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wh2;

/// The largest temperature magnitude accepted, tenths of a degree.
const MAX_TEMP_TENTHS: u32 = 800;
/// The humidity byte of a sensor without a humidity sensor.
const NO_HUMIDITY: u8 = 0xFF;

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
    if b0 >> 4 != 4 {
        return Err(DecodeResult::NoMatch);
    }
    if crc8_msb(0, [b0, b1, b2, b3]) != crc {
        return Err(DecodeResult::BadIntegrity);
    }
    let raw = (u32::from(b1 & 0x0F) << 8) | u32::from(b2);
    let mag = raw & 0x7FF;
    if mag > MAX_TEMP_TENTHS || (b3 > 100 && b3 != NO_HUMIDITY) {
        return Err(DecodeResult::OutOfRange);
    }
    let t = mag as f32 / 10.0;
    let id = (u32::from(b0 & 0x0F) << 4) | u32::from(b1 >> 4);
    let mut r = Reading::new("wh2", id, Mic::Crc, header);
    r.temperature_c = Some(if raw & 0x800 != 0 { -t } else { t });
    if b3 != NO_HUMIDITY {
        r.humidity = Some(f32::from(b3));
    }
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

    /// The 48 bits of the decoder (eight 1 bits, the type nibble 4, the 8 bit id, the
    /// temperature as sign and magnitude in tenths up to 80.0, the humidity or 0xFF when the
    /// reading has none, the CRC-8), every copy in the pulse widths 500 us (1) and 1500 us (0)
    /// with gaps of 1000 us, copies 8 ms apart. The id is below 256; `channel` and `battery_ok`
    /// are not carried.
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encoder::model(r, self.models())?;
        Carries(HUMIDITY).check(r)?;
        let id = encoder::id(r, 256)?;
        let max = MAX_TEMP_TENTHS as i32;
        let t = encoder::temperature(r, 10.0, -max, max)?;
        let h = encoder::humidity_opt(r)?.unwrap_or(NO_HUMIDITY);
        let raw = (if t < 0 { 0x800 } else { 0 }) | t.unsigned_abs();
        let b = [
            0x40 | (id >> 4) as u8,
            (((id & 0xF) << 4) | (raw >> 8)) as u8,
            raw as u8,
            h,
        ];
        let crc = crc8_msb(0, b);
        let msg = [0xFF, b[0], b[1], b[2], b[3], crc];
        out.atomic(|o| {
            o.copies(repeats, REPEAT_GAP_US, |o| {
                o.pwm_bytes(&msg, 500, 1500, 1000)
            })
        })
    }
}
