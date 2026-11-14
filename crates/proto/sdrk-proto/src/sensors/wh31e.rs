//! Ambient Weather `WH31E` thermo-hygrometer (also `WS-3000-X5`), Ecowitt `WH31B` and the
//! Ecowitt `WH40` rain gauge (FSK, 56 us per bit).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433`
//! decoders (written by a third party, nothing copied; decision D1156), checked against what
//! `rtl_433` prints. After six bytes 0xAA and the sync word the message starts with a type byte
//! and ends with a CRC-8 and the sum byte. The `WH31E` (type 0x30) and the `WH31B` (0x37) have
//! seven bytes: the id, a byte with the channel minus one (bits 6 to 4), the low battery flag
//! (bit 2) and the high two bits of a ten bit temperature (`(raw - 400) / 10` degrees), the low
//! temperature byte and the humidity. The `WH40` (0x40) has nine bytes: a 16 bit id (the two bytes after
//! the first; the document names 20 bits with the low nibble of the first, `rtl_433` prints 16), the battery voltage in 0.1 V in the low five bits
//! of a byte (0: not reported), the rain in 0.1 mm as 16 bit and the check bytes; its battery
//! level is `(v - 9) / 6` limited to 0 to 1. The wake-up carrier that precedes the preamble on
//! the air is not sent or needed. The types 0x20 (`WN20`) and 0x68 (`WS68`) of the document are
//! not decoded: `rtl_433` 25.12 does not decode them from frames of the documented layout.

use super::encoder::{self, BATTERY, CHANNEL, Carries, EncodeError, HUMIDITY, RAIN, TrainSink};
use super::fineoffset::{self, Frame, milli};
use super::{DecodeResult, Extra, Mic, Reading, ReadingSink};
use crate::pulses::{Header, PulseTrain};

/// The decoder and encoder of the `WH31E`, `WH31B` and `WH40`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wh31e;

const E: &str = "AmbientWeather-WH31E";
const B: &str = "AmbientWeather-WH31B";
const RAIN_MODEL: &str = "EcoWitt-WH40";
const BIT_US: u32 = 56;
const FRAMES: [Frame; 3] = [
    Frame {
        family: 0x30,
        len: 7,
        crc_at: 5,
        sum: true,
    },
    Frame {
        family: 0x37,
        len: 7,
        crc_at: 5,
        sum: true,
    },
    Frame {
        family: 0x40,
        len: 9,
        crc_at: 7,
        sum: true,
    },
];

fn decode_train(train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
    fineoffset::decode(train, out, BIT_US as f32, &FRAMES, build)
}

fn build(b: &[u8], header: &Header) -> Result<Reading, DecodeResult> {
    match b {
        [fam @ (0x30 | 0x37), id, ct, t0, hum, _, _] => {
            if *hum > 100 {
                return Err(DecodeResult::OutOfRange);
            }
            let raw = i32::from(ct & 3) << 8 | i32::from(*t0);
            let model = if *fam == 0x30 { E } else { B };
            let mut r = Reading::new(model, u32::from(*id), Mic::Crc, header);
            r.channel = Some(((ct >> 4) & 7) + 1);
            r.battery_ok = Some(ct & 4 == 0);
            r.temperature_c = Some((raw - 400) as f32 / 10.0);
            r.humidity = Some(f32::from(*hum));
            Ok(r)
        }
        [0x40, _, i1, i0, fv, r1, r0, _, _] => {
            let id = u32::from(*i1) << 8 | u32::from(*i0);
            let mut r = Reading::new(RAIN_MODEL, id, Mic::Crc, header);
            let v = i32::from(fv & 0x1F);
            if v > 0 {
                r.set_extra(Extra::BatteryV, v as f32 / 10.0);
                r.set_extra(Extra::BatteryOk, level(v));
            }
            r.rain_mm = Some((u32::from(*r1) << 8 | u32::from(*r0)) as f32 / 10.0);
            Ok(r)
        }
        _ => Err(DecodeResult::NoMatch),
    }
}

/// The battery level of the `WH40` for the raw voltage `v` (0.1 V): `(v - 9) / 6` from 0 to 1.
fn level(v: i32) -> f32 {
    milli((v - 9).clamp(0, 6), 6)
}

/// The message of `r` for its model. `WH31E` and `WH31B`: the id below 256, the channel 1 to 8,
/// the battery flag, the temperature -40.0 to 62.3 and the humidity 0 to 100. `WH40`: the id
/// below 2^20, the rain 0.0 to 6553.5 mm and, optionally, the battery voltage 0.1 to 3.1 V with
/// its level.
fn encode_reading(
    models: &[&str],
    r: &Reading,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    encoder::model(r, models)?;
    if r.model == RAIN_MODEL {
        Carries(RAIN).check_extras(r, &[Extra::BatteryOk, Extra::BatteryV])?;
        let id = encoder::id(r, 1 << 16)?;
        let rain = encoder::scaled_in(
            r.rain_mm.ok_or(EncodeError::MissingField("rain_mm"))?,
            10.0,
            0,
            65535,
            "rain_mm",
        )?;
        let v = match r.extra(Extra::BatteryV) {
            Some(_) => encoder::extra(r, Extra::BatteryV, 10.0, 1, 31)?,
            None => 0,
        };
        let want = (v > 0).then(|| level(v));
        if r.extra(Extra::BatteryOk) != want {
            return Err(EncodeError::OutOfRange("battery_ok"));
        }
        let mut msg = [
            0x40,
            0,
            (id >> 8) as u8,
            id as u8,
            v as u8,
            (rain >> 8) as u8,
            rain as u8,
            0,
            0,
        ];
        return fineoffset::encode(&mut msg, &FRAMES[2], 6, BIT_US, repeats, out);
    }
    Carries(BATTERY | CHANNEL | HUMIDITY).check(r)?;
    let id = encoder::id(r, 256)?;
    let ch = encoder::channel(r, 8)? - 1;
    let raw = encoder::temperature(r, 10.0, -400, 623)? + 400;
    let hum = encoder::humidity(r)?;
    let low = u8::from(r.battery_ok == Some(false));
    let (family, frame) = if r.model == E {
        (0x30, &FRAMES[0])
    } else {
        (0x37, &FRAMES[1])
    };
    let mut msg = [
        family,
        id as u8,
        ch << 4 | low << 2 | (raw >> 8) as u8,
        raw as u8,
        hum,
        0,
        0,
    ];
    fineoffset::encode(&mut msg, frame, 6, BIT_US, repeats, out)
}

fineoffset::fsk_device!(Wh31e, "wh31e", [E, B, RAIN_MODEL], decode_train, encode_reading);
