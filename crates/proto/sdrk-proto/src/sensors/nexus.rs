//! Nexus style 36-bit temperature and humidity sensors (433 MHz, on-off keying; sold under
//! several names).
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433` decoders
//! (written by a third party, nothing copied; decision D1138) over the earlier basis of decision
//! D1107 (the prose of the README of the nexus433 project and the bit table of an openHAB
//! community post). Pulse-position coding: every bit is a pulse of 500 us, the gap after it is
//! 1000 us for a 0 and 2000 us for a 1; frames are separated by a gap of 4000 us and repeated
//! (about twelve times). A frame has 36 bits, MSB first: an 8 bit id, a battery bit (1 good), a
//! test-button bit (not reported), a 2 bit channel (shown as 1 to 3, the value 3 is invalid), a
//! 12 bit two's complement temperature in tenths of a degree, the fixed nibble 1111 and an 8 bit
//! humidity (0: the device has no humidity sensor). The device has no checksum: a reading is
//! reported only when three frames of the train are equal (`"mic":"repeat"`, `repeats` the
//! number of equal frames), and a frame that passes the check of the Rubicson protocol (CRC-8,
//! polynomial 0x31, init 0x6C) is not a Nexus frame.

use super::encoder::{
    self, BATTERY, CHANNEL, Carries, EncodeError, HUMIDITY, SensorEncoder, TrainSink,
};
use super::util::{crc8_msb, prefix_result};
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Modulation, PulseTrain};
use crate::slicers::{MAX_ROWS, Ppm, Row, Rows, Slicer};

/// The decoder of the Nexus sensors.
#[derive(Debug, Clone, Copy, Default)]
pub struct Nexus;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Frame {
    id: u8,
    battery_ok: bool,
    channel: u8,
    temp_tenths: i32,
    /// 0: the device has no humidity sensor.
    humidity: u8,
}

/// The equal frames of a train that make a reading (the device has no check).
const MIN_FRAMES: usize = 3;

/// The CRC init of the Rubicson protocol, which frames of this layout must not pass.
const RUBICSON_INIT: u8 = 0x6C;

/// Whether the frame bytes pass the Rubicson check: CRC-8 (0x31, init 0x6C) over the id, the two
/// temperature bytes, the nibble 1111 (as the high nibble) and the humidity as a byte is 0.
fn is_rubicson(id: u8, b1: u8, b2: u8, hum: u8) -> bool {
    crc8_msb(RUBICSON_INIT, [id, b1, b2, 0xF0, hum]) == 0
}

fn parse(row: &Row) -> Result<Frame, DecodeResult> {
    if row.end().is_error() {
        return Err(prefix_result(row, row.len() >= 24));
    }
    if row.len() < 36 {
        return Err(if row.len() >= 24 {
            DecodeResult::Truncated
        } else {
            DecodeResult::NoMatch
        });
    }
    if row.len() > 36 {
        return Err(DecodeResult::NoMatch);
    }
    let (Some(id), Some(bat), Some(ch), Some(t), Some(ff), Some(h), Some(b1)) = (
        row.field(0, 8),
        row.field(8, 1),
        row.field(10, 2),
        row.field(12, 12),
        row.field(24, 4),
        row.field(28, 8),
        row.field(8, 8),
    ) else {
        return Err(DecodeResult::Truncated);
    };
    // bit 6 of the second byte is the test button: any value is a frame
    if ff != 0xF || ch == 3 {
        return Err(DecodeResult::NoMatch);
    }
    let t = if t & 0x800 != 0 {
        t as i32 - 4096
    } else {
        t as i32
    };
    if !(-400..=800).contains(&t) || h > 100 {
        return Err(DecodeResult::OutOfRange);
    }
    if is_rubicson(id as u8, b1 as u8, (t & 0xFF) as u8, h as u8) {
        return Err(DecodeResult::NoMatch);
    }
    Ok(Frame {
        id: id as u8,
        battery_ok: bat == 1,
        channel: ch as u8 + 1,
        temp_tenths: t,
        humidity: h as u8,
    })
}

impl SensorDecoder for Nexus {
    fn name(&self) -> &'static str {
        "nexus"
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
        let Ok(slicer) = Ppm::new(1000.0, 2000.0, 3000.0) else {
            return DecodeResult::NoMatch;
        };
        let mut rows = Rows::new();
        slicer.slice(train, &mut rows);
        let header = train.header();
        let mut frames: [Option<Frame>; MAX_ROWS] = [None; MAX_ROWS];
        let mut res = DecodeResult::NoMatch;
        for (slot, row) in frames.iter_mut().zip(rows.iter()) {
            match parse(row) {
                Ok(f) => *slot = Some(f),
                Err(e) => res = res.best(e),
            }
        }
        for (i, f) in frames.iter().enumerate() {
            let Some(f) = f else { continue };
            if frames.iter().take(i).any(|g| g.as_ref() == Some(f)) {
                continue;
            }
            let n = frames
                .iter()
                .skip(i)
                .filter(|g| g.as_ref() == Some(f))
                .count();
            if n < MIN_FRAMES {
                // too few frames for a device without a check
                res = res.best(DecodeResult::BadIntegrity);
                continue;
            }
            let mut r = Reading::new("nexus", u32::from(f.id), Mic::Repeat, &header);
            r.channel = Some(f.channel);
            r.battery_ok = Some(f.battery_ok);
            r.temperature_c = Some(f.temp_tenths as f32 / 10.0);
            if f.humidity != 0 {
                r.humidity = Some(f32::from(f.humidity));
            }
            r.repeats = n as u32;
            if out.push(r) {
                res = res.best(DecodeResult::Decoded(1));
            }
        }
        res
    }
}

/// The gap between two frames (the end gap of a frame, 4000 us, as the description says).
const FRAME_GAP_US: u32 = 4000;

impl SensorEncoder for Nexus {
    fn name(&self) -> &'static str {
        "nexus"
    }

    fn models(&self) -> &'static [&'static str] {
        &["nexus"]
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn mic(&self) -> Mic {
        Mic::Repeat
    }

    fn default_repeats(&self) -> u32 {
        4
    }

    fn repeat_gap_us(&self) -> u32 {
        FRAME_GAP_US
    }

    /// The 36 bits of the decoder (id, battery, test button 0, channel - 1, 12 bit two's
    /// complement temperature in tenths from -40.0 to 80.0, 1111, humidity or 0 when the reading
    /// has none) as pulses of 500 us with gaps of 1000 us (0) and 2000 us (1), one end pulse and
    /// a gap of 4000 us after every frame. The decoder needs three equal frames, so `repeats`
    /// below 3 is a train it does not read. The id is below 256, the channel 1 to 3, the
    /// humidity 1 to 100. A frame that passes the check of the Rubicson protocol is
    /// [`EncodeError::Unrepresentable`] (about one frame in 256): the decoder takes it for
    /// another device.
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encoder::model(r, self.models())?;
        Carries(CHANNEL | BATTERY | HUMIDITY).check(r)?;
        let id = encoder::id(r, 256)?;
        let ch = encoder::channel(r, 3)?;
        let t = encoder::temperature(r, 10.0, -400, 800)?;
        let h = match encoder::humidity_opt(r)? {
            Some(0) => return Err(EncodeError::OutOfRange("humidity")),
            Some(h) => h,
            None => 0,
        };
        let bat = u64::from(r.battery_ok.unwrap_or(true));
        let b1 = ((bat << 7) | (u64::from(ch - 1) << 4) | ((t as u64 >> 8) & 0xF)) as u8;
        if is_rubicson(id as u8, b1, t as u8, h) {
            return Err(EncodeError::Unrepresentable);
        }
        let bits: u64 = (u64::from(id) << 28)
            | (bat << 27)
            | (u64::from(ch - 1) << 24)
            | (((t & 0xFFF) as u64) << 12)
            | (0xF << 8)
            | u64::from(h);
        out.atomic(|o| {
            o.copies(repeats, FRAME_GAP_US, |o| {
                for k in (0..36).rev() {
                    o.push(500, if bits >> k & 1 == 1 { 2000 } else { 1000 })?;
                }
                o.push(500, FRAME_GAP_US)
            })
        })
    }
}
