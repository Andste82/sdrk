//! Fine Offset / Ecowitt `WH51` soil moisture sensor, also the `WN31` (FSK, 58 us per bit).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433`
//! decoders (written by a third party, nothing copied; decision D1154), checked against what
//! `rtl_433` prints. The message has 14 bytes after the sync word: the family byte 0x51, a
//! 24 bit id (`rtl_433` prints it as a hexadecimal string, here it is a number), a byte with
//! the send counter `boost` in bits 7 to 5 and the battery in 100 mV in bits 4 to 0, an unused
//! byte, the moisture in percent, a byte and a byte with the raw A/D value (nine bits), three
//! unused bytes, a CRC-8 and the sum byte. The battery level that `rtl_433` prints is
//! `(mV - 700) / 900`, not limited to 1 (the document's table of steps differs; the output of
//! `rtl_433` is followed, decision D1154).

use super::encoder::{self, Carries, EncodeError, TrainSink};
use super::fineoffset::{self, Frame, milli};
use super::{DecodeResult, Extra, Mic, Reading, ReadingSink};
use crate::pulses::{Header, PulseTrain};

/// The decoder and encoder of the WH51.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wh51;

const MODEL: &str = "Fineoffset-WH51";
const BIT_US: u32 = 58;
const FRAME: Frame = Frame {
    family: 0x51,
    len: 14,
    crc_at: 12,
    sum: true,
};

fn decode_train(train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
    fineoffset::decode(train, out, BIT_US as f32, &[FRAME], build)
}

fn build(b: &[u8], header: &Header) -> Result<Reading, DecodeResult> {
    let [_, i2, i1, i0, bb, _, moisture, a1, a0, _, _, _, _, _] = b else {
        return Err(DecodeResult::NoMatch);
    };
    if *moisture > 100 {
        return Err(DecodeResult::OutOfRange);
    }
    let id = u32::from(*i2) << 16 | u32::from(*i1) << 8 | u32::from(*i0);
    let mv = i32::from(bb & 0x1F) * 100;
    let mut r = Reading::new(MODEL, id, Mic::Crc, header);
    r.set_extra(Extra::BatteryOk, milli(mv - 700, 900));
    r.set_extra(Extra::BatteryMv, mv as f32);
    r.set_extra(Extra::Moisture, f32::from(*moisture));
    r.set_extra(Extra::Boost, f32::from(bb >> 5));
    r.set_extra(Extra::AdRaw, f32::from(u16::from(a1 & 1) << 8 | u16::from(*a0)));
    Ok(r)
}

/// The message of `r`: the id below 2^24, the battery in mV (0 to 3100 in steps of 100) with
/// the level that goes with it, the moisture 0 to 100, `boost` 0 to 7 and the raw value 0 to
/// 511; every one is required.
fn encode_reading(
    models: &[&str],
    r: &Reading,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    encoder::model(r, models)?;
    Carries(0).check_extras(
        r,
        &[
            Extra::BatteryOk,
            Extra::BatteryMv,
            Extra::Moisture,
            Extra::Boost,
            Extra::AdRaw,
        ],
    )?;
    let id = encoder::id(r, 1 << 24)?;
    let bat = encoder::extra(r, Extra::BatteryMv, 0.01, 0, 31)?;
    if r.extra(Extra::BatteryOk) != Some(milli(bat * 100 - 700, 900)) {
        return Err(EncodeError::OutOfRange("battery_ok"));
    }
    let moisture = encoder::extra(r, Extra::Moisture, 1.0, 0, 100)?;
    let boost = encoder::extra(r, Extra::Boost, 1.0, 0, 7)?;
    let ad = encoder::extra(r, Extra::AdRaw, 1.0, 0, 511)?;
    let mut msg = [
        0x51,
        (id >> 16) as u8,
        (id >> 8) as u8,
        id as u8,
        (boost << 5 | bat) as u8,
        0,
        moisture as u8,
        (ad >> 8) as u8,
        ad as u8,
        0,
        0,
        0,
        0,
        0,
    ];
    fineoffset::encode(&mut msg, &FRAME, 4, BIT_US, repeats, out)
}

fineoffset::fsk_device!(Wh51, "wh51", [MODEL], decode_train, encode_reading);
