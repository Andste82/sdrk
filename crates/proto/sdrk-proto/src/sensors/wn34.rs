//! Fine Offset / Ecowitt `WN34` temperature sensor (`WN34S`, `WN34L`, `WN34D`; also sold as
//! Froggit `DP150` and `DP35`; FSK, 58 us per bit).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433`
//! decoders (written by a third party, nothing copied; decision D1155), checked against what
//! `rtl_433` prints. The message has nine bytes after the sync word: the family byte 0x34, a
//! 24 bit id, a byte with the subtype in the high nibble (4: the `WN34D`) and the high four bits
//! of a 12 bit two's complement temperature in the low nibble, the low byte of the temperature
//! (steps of 0.1 degree; the other subtypes add an offset of -40 degrees), the battery in 20 mV
//! in the low seven bits of a byte, the CRC-8 and the sum byte. `rtl_433` prints the model
//! `Fineoffset-WN34D` for the subtype 4 and `Fineoffset-WN34` for every other subtype. The
//! decoder takes the ranges of the devices: -40.0 to 60.0 degrees (`WN34`) and -55.0 to 125.0
//! degrees (`WN34D`). The battery level is `(bars - 1) / 4` with the bars 5 above 1440 mV, 4
//! above 1380, 3 above 1300, 2 above 1200 and 1 otherwise. The `WN38` (family byte 0x38) is not
//! decoded: `rtl_433` does not decode it from a frame of this layout.

use super::encoder::{self, Carries, EncodeError, TrainSink};
use super::fineoffset::{self, Frame};
use super::{DecodeResult, Extra, Mic, Reading, ReadingSink};
use crate::pulses::{Header, PulseTrain};

/// The decoder and encoder of the WN34.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wn34;

const MODEL: &str = "Fineoffset-WN34";
const MODEL_D: &str = "Fineoffset-WN34D";
const BIT_US: u32 = 58;
const FRAME: Frame = Frame {
    family: 0x34,
    len: 9,
    crc_at: 7,
    sum: true,
};

fn decode_train(train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
    fineoffset::decode(train, out, BIT_US as f32, &[FRAME], build)
}

/// The bars of a battery voltage in millivolt (1 to 5).
fn bars(mv: i32) -> i32 {
    match mv {
        1441.. => 5,
        1381.. => 4,
        1301.. => 3,
        1201.. => 2,
        _ => 1,
    }
}

fn build(b: &[u8], header: &Header) -> Result<Reading, DecodeResult> {
    let [_, i2, i1, i0, st, t0, bat, _, _] = b else {
        return Err(DecodeResult::NoMatch);
    };
    let raw = i32::from(st & 0x0F) << 8 | i32::from(*t0);
    let signed = if raw >= 0x800 { raw - 0x1000 } else { raw };
    let d = st >> 4 == 4;
    let (model, tenths) = if d {
        (MODEL_D, signed)
    } else {
        (MODEL, signed - 400)
    };
    let (lo, hi) = if d { (-550, 1250) } else { (-400, 600) };
    if !(lo..=hi).contains(&tenths) {
        return Err(DecodeResult::OutOfRange);
    }
    let id = u32::from(*i2) << 16 | u32::from(*i1) << 8 | u32::from(*i0);
    let mv = i32::from(bat & 0x7F) * 20;
    let mut r = Reading::new(model, id, Mic::Crc, header);
    r.temperature_c = Some(tenths as f32 / 10.0);
    r.set_extra(Extra::BatteryOk, (bars(mv) - 1) as f32 / 4.0);
    r.set_extra(Extra::BatteryMv, mv as f32);
    Ok(r)
}

/// The message of `r`: the id below 2^24, the temperature within the range of the model, the
/// battery in mV (0 to 2540 in steps of 20) with the level that goes with it; every one is
/// required.
fn encode_reading(
    models: &[&str],
    r: &Reading,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    encoder::model(r, models)?;
    Carries(0).check_extras(r, &[Extra::BatteryOk, Extra::BatteryMv])?;
    let d = r.model == MODEL_D;
    let id = encoder::id(r, 1 << 24)?;
    let (lo, hi, off) = if d { (-550, 1250, 0) } else { (-400, 600, 400) };
    let raw = (encoder::temperature(r, 10.0, lo, hi)? + off) & 0xFFF;
    let bat = encoder::extra(r, Extra::BatteryMv, 0.05, 0, 127)?;
    if r.extra(Extra::BatteryOk) != Some((bars(bat * 20) - 1) as f32 / 4.0) {
        return Err(EncodeError::OutOfRange("battery_ok"));
    }
    let mut msg = [
        0x34,
        (id >> 16) as u8,
        (id >> 8) as u8,
        id as u8,
        (u8::from(d) * 0x40) | (raw >> 8) as u8,
        raw as u8,
        bat as u8,
        0,
        0,
    ];
    fineoffset::encode(&mut msg, &FRAME, 4, BIT_US, repeats, out)
}

fineoffset::fsk_device!(Wn34, "wn34", [MODEL, MODEL_D], decode_train, encode_reading);
