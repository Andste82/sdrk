//! Fine Offset WH1080 / WH3080 weather station, on-off keying variant.
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433` decoders
//! (written by a third party, nothing copied; decision D1140) over the earlier basis of decision
//! D1109 (a summary of the MIT licensed ook-decoder project for the layout and the CRC). The
//! coding is pulse-width: a 1 is a pulse of 544 us, a 0 one of 1524 us; the gap after a pulse is
//! assumed to be 1000 us (no source states it). A row has 88 to 99 bits, MSB first, eleven bytes
//! B0 to B10: B0 = 0xFF (the preamble), B1 the message type in the high nibble (0xA: weather) and
//! the high nibble of the id, the low nibble of B1 and the high nibble of B2 the 8 bit id, the
//! low two bits of B2 and B3 the temperature ((raw - 400) / 10 degrees), B4 the humidity, B5 the
//! average wind speed and B6 the gust (0.34 m/s per count, written in km/h), the low nibble of B7
//! and B8 the rain counter (0.3 mm per count), the high nibble of B9 the status (1: battery
//! low), its low nibble the wind direction (a table of 16 whole degrees) and B10 a CRC-8
//! (polynomial 0x31, MSB first, init 0xFF over all eleven bytes, which is init 0 over B1 to B9
//! compared with B10). The time message (type 0xB), the UV message (0x7) and the shorter and
//! FSK variants of the family are not decoded.

use super::encoder::{
    self, BATTERY, Carries, EncodeError, HUMIDITY, RAIN, SensorEncoder, TrainSink, WIND,
};
use super::util::{byte_at, collect, crc8_msb, prefix_result};
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Header, Modulation, PulseTrain};
use crate::slicers::{Pwm, Row, Rows, Slicer};

/// The decoder of the WH1080.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wh1080;

/// The wind direction in degrees for the 16 values of the low nibble of B9.
const DIRECTIONS: [u16; 16] = [
    0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338,
];

/// The nibble of the direction nearest to `deg` (circular distance; the first on a tie).
fn direction_nibble(deg: f32) -> u8 {
    let mut best = (f32::MAX, 0u8);
    for (i, &d) in DIRECTIONS.iter().enumerate() {
        let diff = (deg - f32::from(d)).abs();
        let dist = diff.min(360.0 - diff);
        if dist < best.0 {
            best = (dist, i as u8);
        }
    }
    best.1
}

/// The shortest and the longest row (the preamble byte and ten more, with up to 11 stray bits).
const MIN_BITS: usize = 88;
const MAX_BITS: usize = 99;

/// km/h per count of the wind speed bytes: 0.34 m/s times 3.6.
const KMH_PER_COUNT: f32 = 1.224;

impl SensorDecoder for Wh1080 {
    fn name(&self) -> &'static str {
        "wh1080"
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
        let Ok(slicer) = Pwm::new(544.0, 1524.0, 1000.0, 2500.0, true) else {
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
        return Err(prefix_result(row, start && row.len() > 48));
    }
    if !start || row.len() <= 48 {
        return Err(DecodeResult::NoMatch);
    }
    if row.len() < MIN_BITS {
        return Err(DecodeResult::Truncated);
    }
    if row.len() > MAX_BITS {
        return Err(DecodeResult::NoMatch);
    }
    let mut b = [0u8; 11];
    for (i, v) in b.iter_mut().enumerate() {
        *v = byte_at(row, i).ok_or(DecodeResult::Truncated)?;
    }
    if b[1] >> 4 != 0xA {
        return Err(DecodeResult::NoMatch);
    }
    if crc8_msb(0, b[1..10].iter().copied()) != b[10] {
        return Err(DecodeResult::BadIntegrity);
    }
    let temp_raw = (u32::from(b[2] & 0x03) << 8) | u32::from(b[3]);
    if b[4] > 100 {
        return Err(DecodeResult::OutOfRange);
    }
    let id = (u32::from(b[1] & 0x0F) << 4) | u32::from(b[2] >> 4);
    let rain = (u32::from(b[7] & 0x0F) << 8) | u32::from(b[8]);
    let mut r = Reading::new("wh1080", id, Mic::Crc, header);
    r.temperature_c = Some((temp_raw as f32 - 400.0) / 10.0);
    r.humidity = Some(f32::from(b[4]));
    r.wind_avg_km_h = Some(f32::from(b[5]) * KMH_PER_COUNT);
    r.wind_max_km_h = Some(f32::from(b[6]) * KMH_PER_COUNT);
    let dir = DIRECTIONS
        .get(usize::from(b[9] & 0x0F))
        .copied()
        .unwrap_or(0);
    r.wind_dir_deg = Some(f32::from(dir));
    r.battery_ok = Some(b[9] >> 4 != 1);
    r.rain_mm = Some(rain as f32 * 0.3);
    Ok(r)
}

/// The pause between two copies (the repeat gap, decision D1124).
const REPEAT_GAP_US: u32 = 8000;

impl SensorEncoder for Wh1080 {
    fn name(&self) -> &'static str {
        "wh1080"
    }

    fn models(&self) -> &'static [&'static str] {
        &["wh1080"]
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn mic(&self) -> Mic {
        Mic::Crc
    }

    fn default_repeats(&self) -> u32 {
        2
    }

    fn repeat_gap_us(&self) -> u32 {
        REPEAT_GAP_US
    }

    /// The 88 bits of the decoder (B0 = 0xFF, the type nibble 0xA, the 8 bit id, the temperature
    /// from -40.0 to 62.3 as `(t + 40) * 10`, humidity, wind speed and gust as counts of
    /// 0.34 m/s, the rain counter of 0.3 mm, the status nibble 1 for a low battery and 0
    /// otherwise, the wind direction as the nearest of the 16 table values, the CRC-8) with the
    /// pulse widths 544 / 1524 us (gaps 1000 us, an assumption), copies 8 ms apart. The id is
    /// below 256. Wind, gust, direction and rain default to 0 when absent; `channel` is not
    /// carried. The band and the gap are assumptions of the decoder (D1109).
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encoder::model(r, self.models())?;
        Carries(BATTERY | HUMIDITY | WIND | RAIN).check(r)?;
        let id = encoder::id(r, 256)?;
        let raw = encoder::temperature(r, 10.0, -400, 623)? + 400;
        let h = encoder::humidity(r)?;
        let count = |v: Option<f32>, scale: f32, hi: i32, name: &'static str| {
            encoder::scaled_in(v.unwrap_or(0.0), scale, 0, hi, name)
        };
        let avg = count(r.wind_avg_km_h, 1.0 / KMH_PER_COUNT, 255, "wind_avg_km_h")?;
        let gust = count(r.wind_max_km_h, 1.0 / KMH_PER_COUNT, 255, "wind_max_km_h")?;
        let rain = count(r.rain_mm, 1.0 / 0.3, 4095, "rain_mm")?;
        let dir = match r.wind_dir_deg {
            None => 0,
            Some(d) if d.is_finite() && (0.0..=360.0).contains(&d) => direction_nibble(d),
            Some(_) => return Err(EncodeError::OutOfRange("wind_dir_deg")),
        };
        let status = u8::from(r.battery_ok == Some(false));
        let mut b = [
            0xFF,
            0xA0 | (id >> 4) as u8,
            (((id & 0xF) << 4) as u8) | (raw >> 8) as u8,
            raw as u8,
            h,
            avg as u8,
            gust as u8,
            (rain >> 8) as u8,
            rain as u8,
            (status << 4) | dir,
            0,
        ];
        let crc = crc8_msb(0, b.iter().skip(1).take(9).copied());
        if let Some(last) = b.last_mut() {
            *last = crc;
        }
        out.atomic(|o| o.copies(repeats, REPEAT_GAP_US, |o| o.pwm_bytes(&b, 544, 1524, 1000)))
    }
}
