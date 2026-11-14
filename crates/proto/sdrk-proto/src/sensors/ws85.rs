//! Fine Offset / Ecowitt `WS85` wind and rain station (FSK, 58 us per bit).
//!
//! Source: protocol facts from the maintainer-provided fact sheet about the `rtl_433` decoders
//! (written by a third party, nothing copied; decision D1162), checked against what `rtl_433`
//! prints. After the bytes 0xAA (at least two) and the sync word the message has 32 bytes of
//! which 28 are used: the family byte 0x85, a 24 bit id, the battery in 20 mV (byte 4), a flag
//! byte (0x88 and the high bits: bit 4 of the wind, bit 5 of the direction, bit 6 of the gust),
//! an unknown byte, the low bytes of the wind (0.1 m/s), the direction (degrees) and the gust
//! (0.1 m/s) in the bytes 7 to 9 (0x1FF is "none"), `rain_start` as bit 4 of byte 12, the rain
//! total in 0.1 mm (bytes 15 and 16), the supercapacitor voltage in 0.1 V (low six bits of byte
//! 17), the firmware byte 25, the CRC-8 over the bytes 0 to 25 (byte 26) and the sum of the
//! bytes 0 to 26 (byte 27). `rtl_433` needs 32 bytes after the sync word, so the bytes 28 to 31
//! are sent as 0xFF. The `battery_ok` printed is 1 above 2400 mV (an integer, as for no other
//! device of the family), `battery_pct` is `min((mV - 1400) / 16, 100)`, 0 below 1400 mV.
//! Assumptions: the unknown bits and bytes are sent as ones (a long run of zeros would end the
//! row at the reset gap of 3000 us) and the flag byte is sent as 0x88 and not carried.

use super::encoder::{self, Carries, EncodeError, RAIN, TrainSink, WIND};
use super::fineoffset::{self, Frame};
use super::ws90::level;
use super::{DecodeResult, Extra, Mic, Reading, ReadingSink};
use crate::pulses::{Header, PulseTrain};

/// The decoder and encoder of the WS85.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ws85;

const MODEL: &str = "Fineoffset-WS85";
const BIT_US: u32 = 58;
const FRAME: Frame = Frame {
    family: 0x85,
    len: 32,
    crc_at: 26,
    sum: true,
};

fn decode_train(train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
    fineoffset::decode(train, out, BIT_US as f32, &[FRAME], build)
}

fn build(b: &[u8], header: &Header) -> Result<Reading, DecodeResult> {
    let at = |i: usize| b.get(i).copied().unwrap_or(0);
    let fl = at(5);
    let nine = |bit: u8, i: usize| u32::from(fl >> bit & 1) << 8 | u32::from(at(i));
    let (wind, dir, gust) = (nine(4, 7), nine(5, 8), nine(6, 9));
    if dir > 359 && dir != 0x1FF {
        return Err(DecodeResult::OutOfRange);
    }
    let id = u32::from(at(1)) << 16 | u32::from(at(2)) << 8 | u32::from(at(3));
    let mut r = Reading::new(MODEL, id, Mic::Crc, header);
    let mv = i32::from(at(4)) * 20;
    r.set_extra(Extra::BatteryOk, f32::from(u8::from(mv > 2400)));
    r.set_extra(Extra::BatteryPct, level(mv) as f32);
    r.set_extra(Extra::BatteryMv, mv as f32);
    if wind != 0x1FF {
        r.set_extra(Extra::WindAvgMs, wind as f32 / 10.0);
    }
    if dir != 0x1FF {
        r.wind_dir_deg = Some(dir as f32);
    }
    if gust != 0x1FF {
        r.set_extra(Extra::WindMaxMs, gust as f32 / 10.0);
    }
    r.set_extra(Extra::RainStart, f32::from(at(12) >> 4 & 1));
    r.rain_mm = Some((u32::from(at(15)) << 8 | u32::from(at(16))) as f32 / 10.0);
    r.set_extra(Extra::SupercapV, f32::from(at(17) & 0x3F) / 10.0);
    r.set_extra(Extra::Firmware, f32::from(at(25)));
    Ok(r)
}

/// The message of `r`: the id below 2^24, the battery in mV (0 to 5100 in steps of 20) with its
/// level and percent, the rain total 0.0 to 6553.5 mm, `rain_start` 0 or 1, the supercapacitor
/// 0.0 to 6.3 V and the firmware byte (every one is required) and any of the values (a missing
/// one is sent as "not available"): wind and gust 0.0 to 51.0 m/s, direction 0 to 359 degrees
/// (whole).
fn encode_reading(
    models: &[&str],
    r: &Reading,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    encoder::model(r, models)?;
    Carries(WIND | RAIN).check_extras(
        r,
        &[
            Extra::BatteryOk,
            Extra::BatteryPct,
            Extra::BatteryMv,
            Extra::WindAvgMs,
            Extra::WindMaxMs,
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
    let mv = bat * 20;
    if r.extra(Extra::BatteryOk) != Some(f32::from(u8::from(mv > 2400))) {
        return Err(EncodeError::OutOfRange("battery_ok"));
    }
    if r.extra(Extra::BatteryPct) != Some(level(mv) as f32) {
        return Err(EncodeError::OutOfRange("battery_pct"));
    }
    let opt = |e: Extra| -> Result<i32, EncodeError> {
        match r.extra(e) {
            Some(_) => encoder::extra(r, e, 10.0, 0, 510),
            None => Ok(0x1FF),
        }
    };
    let (wind, gust) = (opt(Extra::WindAvgMs)?, opt(Extra::WindMaxMs)?);
    let dir = match r.wind_dir_deg {
        Some(d) => encoder::scaled_in(d, 1.0, 0, 359, "wind_dir_deg")?,
        None => 0x1FF,
    };
    let rain = r.rain_mm.ok_or(EncodeError::MissingField("rain_mm"))?;
    let rain = encoder::scaled_in(rain, 10.0, 0, 65535, "rain_mm")?;
    let rain_start = encoder::extra(r, Extra::RainStart, 1.0, 0, 1)?;
    let cap = encoder::extra(r, Extra::SupercapV, 10.0, 0, 63)?;
    let fw = encoder::extra(r, Extra::Firmware, 1.0, 0, 255)?;
    let flags = 0x88 | (wind >> 8) << 4 | (dir >> 8) << 5 | (gust >> 8) << 6;
    let mut msg = [
        0x85,
        (id >> 16) as u8,
        (id >> 8) as u8,
        id as u8,
        bat as u8,
        flags as u8,
        0xFF,
        wind as u8,
        dir as u8,
        gust as u8,
        0xFF,
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
        0xFF,
        0xFF,
        0xFF,
    ];
    fineoffset::encode(&mut msg, &FRAME, 6, BIT_US, repeats, out)
}

fineoffset::fsk_device!(Ws85, "ws85", [MODEL], decode_train, encode_reading);
