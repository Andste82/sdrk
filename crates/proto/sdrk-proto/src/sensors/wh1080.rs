//! Fine Offset WH1080 / WH3080 weather station, on-off keying variant.
//!
//! Source: a summary of the MIT licensed ook-decoder project (file wh1080.c) for the layout and
//! the CRC; the 433 MHz band and the timings are not confirmed by any public source, see decision
//! D1109. The decoder assumes the pulse-width coding of the WH2 (500 us and 1500 us pulses,
//! 1000 us gaps). A message has 88 bits, MSB first, eleven bytes B0 to B10: B0 = 0xFF, B1 and the
//! high nibble of B2 the id, the low nibble of B2 and B3 the temperature ((raw - 400) / 10
//! degrees), B4 the humidity, B5 the average wind speed and B6 the gust (0.34 m/s per count,
//! written in km/h), the low nibble of B7 and B8 the rain counter (0.3 mm per count), the low
//! nibble of B9 the wind direction (16 points of 22.5 degrees) and B10 a CRC-8 (polynomial 0x31,
//! init 0, MSB first) over B1 to B9. The status nibble of B9 is not interpreted.

use super::encoder::{self, Carries, EncodeError, HUMIDITY, RAIN, SensorEncoder, TrainSink, WIND};
use super::util::{byte_at, collect, crc8_msb, prefix_result};
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Header, Modulation, PulseTrain};
use crate::slicers::{Pwm, Row, Rows, Slicer};

/// The decoder of the WH1080.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wh1080;

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
        return Err(prefix_result(row, start && row.len() > 48));
    }
    if !start || row.len() <= 48 {
        return Err(DecodeResult::NoMatch);
    }
    if row.len() < 88 {
        return Err(DecodeResult::Truncated);
    }
    if row.len() > 88 {
        return Err(DecodeResult::NoMatch);
    }
    let mut b = [0u8; 11];
    for (i, v) in b.iter_mut().enumerate() {
        *v = byte_at(row, i).ok_or(DecodeResult::Truncated)?;
    }
    if crc8_msb(0, b[1..10].iter().copied()) != b[10] {
        return Err(DecodeResult::BadIntegrity);
    }
    let temp_raw = (u32::from(b[2] & 0x0F) << 8) | u32::from(b[3]);
    if temp_raw > 1000 || b[4] > 100 {
        return Err(DecodeResult::OutOfRange);
    }
    let id = (u32::from(b[1]) << 4) | u32::from(b[2] >> 4);
    let rain = (u32::from(b[7] & 0x0F) << 8) | u32::from(b[8]);
    let mut r = Reading::new("wh1080", id, Mic::Crc, header);
    r.temperature_c = Some((temp_raw as f32 - 400.0) / 10.0);
    r.humidity = Some(f32::from(b[4]));
    r.wind_avg_km_h = Some(f32::from(b[5]) * KMH_PER_COUNT);
    r.wind_max_km_h = Some(f32::from(b[6]) * KMH_PER_COUNT);
    r.wind_dir_deg = Some(f32::from(b[9] & 0x0F) * 22.5);
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

    /// The 88 bits of the decoder (B0 = 0xFF, the 12 bit id, the temperature from -40.0 to 60.0
    /// as `(t + 40) * 10`, humidity, wind speed and gust as counts of 0.34 m/s, the rain counter
    /// of 0.3 mm, the wind direction as a multiple of 22.5 degrees, the CRC-8) with the pulse
    /// widths of the WH2 (500 / 1500 us, gaps 1000 us), copies 8 ms apart. The status nibble of
    /// B9 is sent as 0. Wind, gust, direction and rain default to 0 when absent; `channel` and
    /// `battery_ok` are not carried. The band and the timings are the unconfirmed assumptions of
    /// the decoder (D1109).
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encoder::model(r, self.models())?;
        Carries(HUMIDITY | WIND | RAIN).check(r)?;
        let id = encoder::id(r, 0x1000)?;
        let raw = encoder::temperature(r, 10.0, -400, 600)? + 400;
        let h = encoder::humidity(r)?;
        let count = |v: Option<f32>, scale: f32, hi: i32, name: &'static str| {
            encoder::scaled_in(v.unwrap_or(0.0), scale, 0, hi, name)
        };
        let avg = count(r.wind_avg_km_h, 1.0 / KMH_PER_COUNT, 255, "wind_avg_km_h")?;
        let gust = count(r.wind_max_km_h, 1.0 / KMH_PER_COUNT, 255, "wind_max_km_h")?;
        let rain = count(r.rain_mm, 1.0 / 0.3, 4095, "rain_mm")?;
        let dir = count(r.wind_dir_deg, 1.0 / 22.5, 16, "wind_dir_deg")? & 0xF;
        let mut b = [
            0xFF,
            (id >> 4) as u8,
            (((id & 0xF) << 4) as u8) | (raw >> 8) as u8,
            raw as u8,
            h,
            avg as u8,
            gust as u8,
            (rain >> 8) as u8,
            rain as u8,
            dir as u8,
            0,
        ];
        let crc = crc8_msb(0, b.iter().skip(1).take(9).copied());
        if let Some(last) = b.last_mut() {
            *last = crc;
        }
        out.atomic(|o| o.copies(repeats, REPEAT_GAP_US, |o| o.pwm_bytes(&b, 500, 1500, 1000)))
    }
}
