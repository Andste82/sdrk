//! Fine Offset / Ecowitt `WH45` air quality sensor (also `WH0295`, Froggit `DP250`, Ambient
//! Weather `AQIN`; FSK, 58 us per bit).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433`
//! decoders (written by a third party, nothing copied; decision D1157), checked against what
//! `rtl_433` prints. After four bytes 0xAA and the sync word the message has 15 bytes: the
//! family byte 0x45, a 24 bit id, an 11 bit temperature (`(raw - 400) / 10` degrees, in the low
//! three bits of a byte and the next byte), the humidity, PM2.5 and PM10 as 14 bit values in
//! 0.1 microgram per cubic metre (the high bits of the PM2.5 and PM10 words carry the battery:
//! bit 6 of the PM2.5 byte is the high bit and the top two bits of the PM10 byte the low two
//! bits of a battery count of 0 to 6, 6 meaning an external supply), the CO2 in ppm as 16 bit,
//! the CRC-8 and the sum byte. The battery level printed is `min(count, 5) / 5` and
//! `ext_power` is 1 for a count of 6. The `WH46` (family byte 0x46) is not decoded:
//! `rtl_433` 25.12 prints no valid record for the documented frame.

use super::encoder::{self, Carries, EncodeError, HUMIDITY, TrainSink};
use super::fineoffset::{self, Frame};
use super::{DecodeResult, Extra, Mic, Reading, ReadingSink};
use crate::pulses::{Header, PulseTrain};

/// The decoder and encoder of the WH45.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wh45;

const MODEL: &str = "Fineoffset-WH45";
const BIT_US: u32 = 58;
const FRAME: Frame = Frame {
    family: 0x45,
    len: 15,
    crc_at: 13,
    sum: true,
};

fn decode_train(train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
    fineoffset::decode(train, out, BIT_US as f32, &[FRAME], build)
}

fn build(b: &[u8], header: &Header) -> Result<Reading, DecodeResult> {
    let [_, i2, i1, i0, th, tl, hum, p1, p0, q1, q0, c1, c0, _, _] = b else {
        return Err(DecodeResult::NoMatch);
    };
    let count = (p1 >> 6 & 1) << 2 | q1 >> 6;
    if *hum > 100 || count > 6 {
        return Err(DecodeResult::OutOfRange);
    }
    let raw = i32::from(th & 7) << 8 | i32::from(*tl);
    let id = u32::from(*i2) << 16 | u32::from(*i1) << 8 | u32::from(*i0);
    let mut r = Reading::new(MODEL, id, Mic::Crc, header);
    r.temperature_c = Some((raw - 400) as f32 / 10.0);
    r.humidity = Some(f32::from(*hum));
    r.set_extra(Extra::BatteryOk, f32::from(count.min(5)) / 5.0);
    r.set_extra(Extra::ExtPower, f32::from(u8::from(count == 6)));
    let pm25 = u32::from(p1 & 0x3F) << 8 | u32::from(*p0);
    let pm10 = u32::from(q1 & 0x3F) << 8 | u32::from(*q0);
    r.set_extra(Extra::Pm25, pm25 as f32 / 10.0);
    r.set_extra(Extra::Pm10, pm10 as f32 / 10.0);
    r.set_extra(Extra::Co2, (u32::from(*c1) << 8 | u32::from(*c0)) as f32);
    Ok(r)
}

/// The message of `r`: the id below 2^24, the temperature -40.0 to 164.7, the humidity 0 to 100,
/// PM2.5 and PM10 0.0 to 1638.3, the CO2 0 to 65535, the battery level in steps of 0.2 and
/// `ext_power`; every one is required. With `ext_power` 1 the battery level must be 1.
fn encode_reading(
    models: &[&str],
    r: &Reading,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    encoder::model(r, models)?;
    Carries(HUMIDITY).check_extras(
        r,
        &[
            Extra::BatteryOk,
            Extra::ExtPower,
            Extra::Pm25,
            Extra::Pm10,
            Extra::Co2,
        ],
    )?;
    let id = encoder::id(r, 1 << 24)?;
    let raw = encoder::temperature(r, 10.0, -400, 1647)? + 400;
    let hum = encoder::humidity(r)?;
    let pm25 = encoder::extra(r, Extra::Pm25, 10.0, 0, 16383)?;
    let pm10 = encoder::extra(r, Extra::Pm10, 10.0, 0, 16383)?;
    let co2 = encoder::extra(r, Extra::Co2, 1.0, 0, 65535)?;
    let level = encoder::extra(r, Extra::BatteryOk, 5.0, 0, 5)?;
    let count = if encoder::extra(r, Extra::ExtPower, 1.0, 0, 1)? == 1 {
        if level != 5 {
            return Err(EncodeError::OutOfRange("battery_ok"));
        }
        6
    } else {
        level
    };
    let mut msg = [
        0x45,
        (id >> 16) as u8,
        (id >> 8) as u8,
        id as u8,
        (raw >> 8) as u8,
        raw as u8,
        hum,
        ((count >> 2) << 6 | pm25 >> 8) as u8,
        pm25 as u8,
        ((count & 3) << 6 | pm10 >> 8) as u8,
        pm10 as u8,
        (co2 >> 8) as u8,
        co2 as u8,
        0,
        0,
    ];
    fineoffset::encode(&mut msg, &FRAME, 4, BIT_US, repeats, out)
}

fineoffset::fsk_device!(Wh45, "wh45", [MODEL], decode_train, encode_reading);
