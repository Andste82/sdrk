//! Fine Offset / Ecowitt `WS90` ultrasonic weather station with a rain gauge (FSK, 58 us per
//! bit).
//!
//! Source: protocol facts from the maintainer-provided fact sheet about the `rtl_433` decoders
//! (written by a third party, nothing copied; decision D1161), checked against what `rtl_433`
//! prints. After the bytes 0xAA (at least two) and the sync word the message has 32 bytes: the
//! bytes 0 to 13 as of the `WS80` (family byte 0x90 instead of 0x80), the pressure word
//! (bytes 14 and 15, sent as 0x3FFF, "none"; `rtl_433` 25.12 prints it only in its debug string,
//! so there is no pressure field), the rain flag (bit 4 of byte 16 is `rain_start`), the rain total in
//! 0.1 mm (bytes 19 and 20), the supercapacitor voltage in 0.1 V (low six bits of byte 21), the
//! firmware byte 29, the CRC-8 over the bytes 0 to 29 (byte 30) and the sum of the bytes 0 to
//! 30 (byte 31). `rtl_433` needs 256 bits after the sync word, so a byte 0xFF follows the burst
//! (the sum may end in zeros that the end of the burst hides). The battery level is
//! `min((mV - 1400) / 16, 100) / 100`, 0 below 1400 mV. Assumptions: the unit of the pressure
//! word is unknown, so it is not decoded; the unknown bits and bytes are sent as ones (a
//! long run of zeros would end the row at the reset gap of 3000 us) and the flag byte is sent as
//! 0x88 and not carried by the reading.

use super::encoder::{self, Carries, EncodeError, HUMIDITY, RAIN, TrainSink, WIND};
use super::fineoffset::{self, Frame, milli};
use super::{DecodeResult, Extra, Mic, Reading, ReadingSink};
use crate::pulses::{Header, PulseTrain};

/// The decoder and encoder of the WS90.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ws90;

const MODEL: &str = "Fineoffset-WS90";
const BIT_US: u32 = 58;
const FRAME: Frame = Frame {
    family: 0x90,
    len: 32,
    crc_at: 30,
    sum: true,
};

fn decode_train(train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
    fineoffset::decode(train, out, BIT_US as f32, &[FRAME], build)
}

/// The battery level in whole percent of a voltage in mV.
pub(super) fn level(mv: i32) -> i32 {
    if mv < 1400 {
        0
    } else {
        ((mv - 1400) / 16).min(100)
    }
}

fn build(b: &[u8], header: &Header) -> Result<Reading, DecodeResult> {
    let at = |i: usize| b.get(i).copied().unwrap_or(0);
    let be = |i: usize| u32::from(at(i)) << 8 | u32::from(at(i + 1));
    let fl = at(7);
    let nine = |bit: u8, i: usize| u32::from(fl >> bit & 1) << 8 | u32::from(at(i));
    let temp = i32::from(fl & 3) << 8 | i32::from(at(8));
    let (wind, dir, gust) = (nine(4, 10), nine(5, 11), nine(6, 12));
    let hum = at(9);
    if (hum > 100 && hum != 0xFF) || (dir > 359 && dir != 0x1FF) {
        return Err(DecodeResult::OutOfRange);
    }
    let id = u32::from(at(1)) << 16 | be(2);
    let mut r = Reading::new(MODEL, id, Mic::Crc, header);
    let mv = i32::from(at(6)) * 20;
    r.set_extra(Extra::BatteryOk, milli(level(mv), 100));
    r.set_extra(Extra::BatteryMv, mv as f32);
    if temp != 0x3FF {
        r.temperature_c = Some((temp - 400) as f32 / 10.0);
    }
    if hum != 0xFF {
        r.humidity = Some(f32::from(hum));
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
    if at(13) != 0xFF {
        r.set_extra(Extra::Uvi, f32::from(at(13)) / 10.0);
    }
    if be(4) != 0xFFFF {
        r.set_extra(Extra::LightLux, (be(4) * 10) as f32);
    }
    r.set_extra(Extra::RainStart, f32::from(at(16) >> 4 & 1));
    r.rain_mm = Some(be(19) as f32 / 10.0);
    r.set_extra(Extra::SupercapV, f32::from(at(21) & 0x3F) / 10.0);
    r.set_extra(Extra::Firmware, f32::from(at(29)));
    Ok(r)
}

/// The message of `r`: the id below 2^24, the battery in mV (0 to 5100 in steps of 20) with its
/// level, the rain total 0.0 to 6553.5 mm, `rain_start` 0 or 1, the supercapacitor 0.0 to
/// 6.3 V and the firmware byte (every one is required) and any of the values (a missing one is
/// sent as "not available"): temperature -40.0 to 62.2, humidity 0 to 100, wind and gust 0.0 to
/// 51.0 m/s, direction 0 to 359 degrees (whole), UV index 0.0 to 25.4, light 0 to 655340 lux in
/// steps of 10.
fn encode_reading(
    models: &[&str],
    r: &Reading,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    encoder::model(r, models)?;
    Carries(HUMIDITY | WIND | RAIN).check_extras(
        r,
        &[
            Extra::BatteryOk,
            Extra::BatteryMv,
            Extra::WindAvgMs,
            Extra::WindMaxMs,
            Extra::Uvi,
            Extra::LightLux,
            Extra::RainStart,
            Extra::SupercapV,
            Extra::Firmware,
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
    if r.extra(Extra::BatteryOk) != Some(milli(level(bat * 20), 100)) {
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
    let rain = r.rain_mm.ok_or(EncodeError::MissingField("rain_mm"))?;
    let rain = encoder::scaled_in(rain, 10.0, 0, 65535, "rain_mm")?;
    let rain_start = encoder::extra(r, Extra::RainStart, 1.0, 0, 1)?;
    let cap = encoder::extra(r, Extra::SupercapV, 10.0, 0, 63)?;
    let fw = encoder::extra(r, Extra::Firmware, 1.0, 0, 255)?;
    let nine = |v: i32| if v < 0 { 0x1FF } else { v };
    let (wind, gust, dir) = (nine(wind), nine(gust), nine(dir));
    let flags = 0x88 | (temp >> 8) | (wind >> 8) << 4 | (dir >> 8) << 5 | (gust >> 8) << 6;
    let light = if light < 0 { 0xFFFF } else { light };
    let mut msg = [
        0x90,
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
        0xEF | (rain_start as u8) << 4,
        0xFF,
        0xFF,
        (rain >> 8) as u8,
        rain as u8,
        0xC0 | cap as u8,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        fw as u8,
        0,
        0,
        0xFF,
    ];
    fineoffset::encode(&mut msg, &FRAME, 6, BIT_US, repeats, out)
}

fineoffset::fsk_device!(Ws90, "ws90", [MODEL], decode_train, encode_reading);
