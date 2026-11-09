//! Nexus style 36-bit temperature and humidity sensors (433 MHz, on-off keying; sold under
//! several names).
//!
//! Source: the prose description of the Nexus protocol in the README of the nexus433 project
//! and the bit table of an openHAB community post (facts only, timings chosen independently;
//! see decision D1107). Pulse-position coding: every bit is a pulse of 500 us, the gap after
//! it is 1000 us for a 0 and 2000 us for a 1; frames are separated by a gap of 4000 us and
//! repeated (about twelve times). A frame has 36 bits, MSB first: an 8 bit id, a battery bit
//! (1 good), a fixed 0, a 2 bit channel (shown as 1 to 4), a 12 bit two's complement
//! temperature in tenths of a degree, the fixed nibble 1111 and an 8 bit humidity. The device
//! has no checksum: a reading is reported only when two frames of the train are equal
//! (`"mic":"repeat"`, `repeats` the number of equal frames).

use super::encoder::{
    self, BATTERY, CHANNEL, Carries, EncodeError, HUMIDITY, SensorEncoder, TrainSink,
};
use super::util::prefix_result;
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
    humidity: u8,
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
    let (Some(id), Some(bat), Some(fixed0), Some(ch), Some(t), Some(ff), Some(h)) = (
        row.field(0, 8),
        row.field(8, 1),
        row.field(9, 1),
        row.field(10, 2),
        row.field(12, 12),
        row.field(24, 4),
        row.field(28, 8),
    ) else {
        return Err(DecodeResult::Truncated);
    };
    if fixed0 != 0 || ff != 0xF {
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
            if n < 2 {
                // a single frame is not enough for a device without a check
                res = res.best(DecodeResult::BadIntegrity);
                continue;
            }
            let mut r = Reading::new("nexus", u32::from(f.id), Mic::Repeat, &header);
            r.channel = Some(f.channel);
            r.battery_ok = Some(f.battery_ok);
            r.temperature_c = Some(f.temp_tenths as f32 / 10.0);
            r.humidity = Some(f32::from(f.humidity));
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

    /// The 36 bits of the decoder (id, battery, 0, channel - 1, 12 bit two's complement
    /// temperature in tenths from -40.0 to 80.0, 1111, humidity) as pulses of 500 us with gaps
    /// of 1000 us (0) and 2000 us (1), one end pulse and a gap of 4000 us after every frame.
    /// The decoder needs two equal frames, so `repeats` of 1 is a train it does not read. The
    /// id is below 256, the channel 1 to 4.
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encoder::model(r, self.models())?;
        Carries(CHANNEL | BATTERY | HUMIDITY).check(r)?;
        let id = encoder::id(r, 256)?;
        let ch = encoder::channel(r, 4)?;
        let t = encoder::temperature(r, 10.0, -400, 800)?;
        let h = encoder::humidity(r)?;
        let bat = u64::from(r.battery_ok.unwrap_or(true));
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
