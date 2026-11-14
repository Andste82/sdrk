//! Fine Offset / Ecowitt `WH55` water leak sensor (FSK, 60 us per bit).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433`
//! decoders (written by a third party, nothing copied; decision D1153), checked against what
//! `rtl_433` prints. The shared framing is in [`super::fineoffset`] (three bytes 0xAA, the sync
//! word, then the message). The message has the family byte 0x55, the channel minus one in the
//! high nibble of the next byte (the low nibble belongs to a 20 bit id that is not reported),
//! the 16 bit id, the battery in steps of 20 percent (0 to 5), the raw sensor value (16 bit),
//! a flag byte (bit 7 high sensitivity, bit 6 alarm) and a CRC-8; three further bytes of the
//! device are unknown and not used.

use super::encoder::{self, CHANNEL, Carries, EncodeError, TrainSink};
use super::fineoffset::{self, Frame};
use super::{DecodeResult, Extra, Mic, Reading, ReadingSink};
use crate::pulses::{Header, PulseTrain};

/// The decoder and encoder of the WH55.
#[derive(Debug, Clone, Copy, Default)]
pub struct Wh55;

const MODEL: &str = "Fineoffset-WH55";
const BIT_US: u32 = 60;
const FRAME: Frame = Frame {
    family: 0x55,
    len: 9,
    crc_at: 8,
    sum: false,
};

fn decode_train(train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
    fineoffset::decode(train, out, BIT_US as f32, &[FRAME], build)
}

fn build(b: &[u8], header: &Header) -> Result<Reading, DecodeResult> {
    let [_, ch, i1, i0, bat, v1, v0, flags, _] = b else {
        return Err(DecodeResult::NoMatch);
    };
    if ch >> 4 > 3 || *bat > 5 {
        return Err(DecodeResult::OutOfRange);
    }
    let mut r = Reading::new(
        MODEL,
        u32::from(*i1) << 8 | u32::from(*i0),
        Mic::Crc,
        header,
    );
    r.channel = Some((ch >> 4) + 1);
    r.set_extra(Extra::BatteryOk, f32::from(*bat) / 5.0);
    r.set_extra(
        Extra::RawValue,
        (u32::from(*v1) << 8 | u32::from(*v0)) as f32,
    );
    r.set_extra(Extra::Sensitivity, f32::from(flags >> 7));
    r.set_extra(Extra::Alarm, f32::from(flags >> 6 & 1));
    Ok(r)
}

/// The message of `r`: the id below 65536, the channel 1 to 4, the battery level in steps of
/// 0.2, the raw value 0 to 65535 and the two flags; every one is required.
fn encode_reading(
    models: &[&str],
    r: &Reading,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    encoder::model(r, models)?;
    Carries(CHANNEL).check_extras(
        r,
        &[
            Extra::BatteryOk,
            Extra::RawValue,
            Extra::Sensitivity,
            Extra::Alarm,
        ],
    )?;
    let id = encoder::id(r, 1 << 16)?;
    let ch = encoder::channel(r, 4)? - 1;
    let bat = encoder::extra(r, Extra::BatteryOk, 5.0, 0, 5)? as u8;
    let raw = encoder::extra(r, Extra::RawValue, 1.0, 0, 65535)?;
    let sens = encoder::extra(r, Extra::Sensitivity, 1.0, 0, 1)? as u8;
    let alarm = encoder::extra(r, Extra::Alarm, 1.0, 0, 1)? as u8;
    let mut msg = [
        0x55,
        ch << 4,
        (id >> 8) as u8,
        id as u8,
        bat,
        (raw >> 8) as u8,
        raw as u8,
        sens << 7 | alarm << 6,
        0,
    ];
    fineoffset::encode(&mut msg, &FRAME, 3, BIT_US, repeats, out)
}

fineoffset::fsk_device!(Wh55, "wh55", [MODEL], decode_train, encode_reading);
