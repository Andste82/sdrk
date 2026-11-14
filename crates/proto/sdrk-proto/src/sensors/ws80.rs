//! Fine Offset / Ecowitt `WS80` ultrasonic weather station (FSK, 58 us per bit).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433`
//! decoders (written by a third party, nothing copied; decision D1158), checked against what
//! `rtl_433` prints. After six bytes 0xAA and the sync word the message has 18 bytes: the
//! family byte 0x80, a 24 bit id, the light in units of 10 lux (16 bit), the battery in 20 mV,
//! a flag byte (0x88 and the high bits: bits 1 and 0 of the temperature, bit 4 of the wind,
//! bit 5 of the direction, bit 6 of the gust), the low temperature byte (ten bits,
//! `(raw - 400) / 10` degrees), the humidity, the low bytes of the wind (0.1 m/s), the direction
//! (degrees) and the gust (0.1 m/s), the UV index in 0.1, two unknown bytes (0x3F 0xFF), the
//! CRC-8 and the sum byte. A value the sensor does not have is all ones (temperature 0x3FF,
//! humidity 0xFF, wind, direction and gust 0x1FF, UV 0xFF, light 0xFFFF) and is left out of the
//! reading, as `rtl_433` does. The battery level is `(mV - 1400) / 1600` from 0 up.

use super::encoder::{self, Carries, EncodeError, HUMIDITY, TrainSink};
use super::fineoffset::{self, Frame, milli};
use super::{DecodeResult, Extra, Mic, Reading, ReadingSink};
use crate::pulses::{Header, PulseTrain};

/// The decoder and encoder of the WS80.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ws80;

const MODEL: &str = "Fineoffset-WS80";
const BIT_US: u32 = 58;
const FRAME: Frame = Frame {
    family: 0x80,
    len: 18,
    crc_at: 16,
    sum: true,
};

fn decode_train(train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
    fineoffset::decode(train, out, BIT_US as f32, &[FRAME], build)
}

fn build(b: &[u8], header: &Header) -> Result<Reading, DecodeResult> {
    let [_, i2, i1, i0, l1, l0, bat, fl, t0, hum, w0, d0, g0, uv, _, _, _, _] = b else {
        return Err(DecodeResult::NoMatch);
    };
    let nine = |bit: u8, low: u8| u32::from(fl >> bit & 1) << 8 | u32::from(low);
    let temp = i32::from(fl & 3) << 8 | i32::from(*t0);
    let (wind, dir, gust) = (nine(4, *w0), nine(5, *d0), nine(6, *g0));
    if (*hum > 100 && *hum != 0xFF) || (dir > 359 && dir != 0x1FF) {
        return Err(DecodeResult::OutOfRange);
    }
    let id = u32::from(*i2) << 16 | u32::from(*i1) << 8 | u32::from(*i0);
    let mut r = Reading::new(MODEL, id, Mic::Crc, header);
    let mv = i32::from(*bat) * 20;
    r.set_extra(Extra::BatteryOk, milli((mv - 1400).max(0), 1600));
    r.set_extra(Extra::BatteryMv, mv as f32);
    if temp != 0x3FF {
        r.temperature_c = Some((temp - 400) as f32 / 10.0);
    }
    if *hum != 0xFF {
        r.humidity = Some(f32::from(*hum));
    }
    if wind != 0x1FF {
        r.set_extra(Extra::WindAvgMs, wind as f32 / 10.0);
    }
    if dir != 0x1FF {
        r.wind_dir_deg = Some(dir as f32);
    }
    if gust != 0x1FF {
        r.set_extra(Extra::WindMaxMs, gust as f32 / 10.0);
    }
    if *uv != 0xFF {
        r.set_extra(Extra::Uvi, f32::from(*uv) / 10.0);
    }
    let light = u32::from(*l1) << 8 | u32::from(*l0);
    if light != 0xFFFF {
        r.set_extra(Extra::LightLux, (light * 10) as f32);
    }
    Ok(r)
}

/// The message of `r`: the id below 2^24, the battery in mV (0 to 5100 in steps of 20) with its
/// level, and any of the values (a missing one is sent as "not available"): temperature -40.0
/// to 62.2, humidity 0 to 100 (0xFF is "none"), wind and gust 0.0 to 51.0 m/s, direction 0 to
/// 359 degrees (whole), UV index 0.0 to 25.4, light 0 to 655340 lux in steps of 10.
fn encode_reading(
    models: &[&str],
    r: &Reading,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    encoder::model(r, models)?;
    Carries(HUMIDITY | encoder::WIND).check_extras(
        r,
        &[
            Extra::BatteryOk,
            Extra::BatteryMv,
            Extra::WindAvgMs,
            Extra::WindMaxMs,
            Extra::Uvi,
            Extra::LightLux,
        ],
    )?;
    // the km/h wind fields of other devices are not carried
    if r.wind_avg_km_h.is_some() {
        return Err(EncodeError::UnsupportedField("wind_avg_km_h"));
    }
    if r.wind_max_km_h.is_some() {
        return Err(EncodeError::UnsupportedField("wind_max_km_h"));
    }
    let id = encoder::id(r, 1 << 24)?;
    let bat = encoder::extra(r, Extra::BatteryMv, 0.05, 0, 255)?;
    if r.extra(Extra::BatteryOk) != Some(milli((bat * 20 - 1400).max(0), 1600)) {
        return Err(EncodeError::OutOfRange("battery_ok"));
    }
    let temp = match r.temperature_c {
        Some(_) => encoder::temperature(r, 10.0, -400, 622)? + 400,
        None => 0x3FF,
    };
    let hum = encoder::humidity_opt(r)?.map_or(0xFF, i32::from);
    let opt = |e: Extra, scale: f32, hi: i32| -> Result<i32, EncodeError> {
        match r.extra(e) {
            Some(_) => encoder::extra(r, e, scale, 0, hi),
            None => Ok(-1),
        }
    };
    let wind = opt(Extra::WindAvgMs, 10.0, 510)?;
    let gust = opt(Extra::WindMaxMs, 10.0, 510)?;
    let uv = opt(Extra::Uvi, 10.0, 254)?;
    let light = opt(Extra::LightLux, 0.1, 65534)?;
    let dir = match r.wind_dir_deg {
        Some(d) => encoder::scaled_in(d, 1.0, 0, 359, "wind_dir_deg")?,
        None => -1,
    };
    let nine = |v: i32| if v < 0 { 0x1FF } else { v };
    let (wind, gust, dir) = (nine(wind), nine(gust), nine(dir));
    let flags = 0x88 | (temp >> 8) | (wind >> 8) << 4 | (dir >> 8) << 5 | (gust >> 8) << 6;
    let light = if light < 0 { 0xFFFF } else { light };
    let mut msg = [
        0x80,
        (id >> 16) as u8,
        (id >> 8) as u8,
        id as u8,
        (light >> 8) as u8,
        light as u8,
        bat as u8,
        flags as u8,
        temp as u8,
        hum as u8,
        wind as u8,
        dir as u8,
        gust as u8,
        if uv < 0 { 0xFF } else { uv as u8 },
        0x3F,
        0xFF,
        0,
        0,
    ];
    fineoffset::encode(&mut msg, &FRAME, 6, BIT_US, repeats, out)
}

fineoffset::fsk_device!(Ws80, "ws80", [MODEL], decode_train, encode_reading);
