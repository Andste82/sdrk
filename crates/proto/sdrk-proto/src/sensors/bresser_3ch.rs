//! Bresser 3CH thermometer and hygrometer sensors (433 MHz, on-off keying; also sold under
//! other names, among them the sensor labelled "AS2-WS-6B, type 60109").
//!
//! Source: own analysis of three recordings of the maintainer's sensor (2026-11-09), which found
//! the layout independently (including the Fahrenheit scale from two points); a reader pass over
//! the `rtl_433` device description (facts only, nothing copied) then identified the protocol as
//! the Bresser 3CH family and confirmed the temperature scale, the flag bits of byte 1, the id
//! behaviour and the sanity limits (decision D1119). The product name of the maintainer's sensor
//! is not confirmed, only its protocol.
//!
//! A frame is a preamble of 3 to 5 pairs of a pulse and a gap of about 735 us each followed by 40
//! data bits with a fixed bit period of 750 us: a 1 is a pulse of 500 us and a gap of 250 us, a 0
//! a pulse of 250 us and a gap of 500 us. Frames follow each other back to back (the gap of the
//! last bit runs into the preamble of the next frame; the pause after the last frame is the end
//! of the train); a button press of the maintainer's sensor sent 22 to 24 of them, one every
//! 36 ms. The 40 bits are five bytes, MSB first:
//!
//! | Byte | Content |
//! | --- | --- |
//! | 0 | the id, random, new whenever the sensor starts (battery insertion); reported as `id` |
//! | 1 | bit 7 battery low (`battery_ok` is false when set), bit 6 the test button flag (set when the TX button was pressed; not reported, the reading has no field for it), bits 5 and 4 the channel 1 to 3 (0 is not a frame of the device), bits 3 to 0 the high part of the temperature |
//! | 2 | the low byte of the temperature `T` (12 bits with the low nibble of byte 1) |
//! | 3 | the relative humidity in percent, a plain binary number |
//! | 4 | the sum of bytes 0 to 3 modulo 256 (reported as `"mic":"checksum"`) |
//!
//! The temperature is in tenths of degrees Fahrenheit plus 900: `F = (T - 900) / 10`, so
//! `C = (T - 1220) / 18`, written with one decimal. A humidity above 100 or a temperature below
//! -20 or above 160 degrees Fahrenheit (-28.9 to 71.1 C) rejects the frame. A frame with a valid
//! sum is a reading (the sum is the integrity check, as for the other checksum decoders); equal
//! frames of a train are one reading with `repeats` set to their number.
//!
//! The coding does not fit a slicer (the preamble and the bits have different pair shapes and
//! the frames follow each other without a reset gap), so the decoder classifies every pair of the
//! train by its two widths within the timing tolerance of the slicers (20 percent, at least 2
//! samples) and cuts the frames itself: at least 2 preamble pairs (the detector may lose the
//! first pulse of the first frame) then exactly 40 data pairs. A pair that is neither a preamble
//! nor a data pair drops the frame in progress and waits for the next preamble.
//!
//! ```
//! use sdrk_proto::pulses::{self, Header, Modulation, Pair};
//! use sdrk_proto::sensors::{Bresser3ch, DecodeResult, ReadingSink, SensorDecoder};
//!
//! // 250 kHz: 735 us = 184 samples, 500 us = 125, 250 us = 62
//! let frame = [0xD8u8, 0x66, 0x52, 0x40, 0xD0]; // channel 2, 22.1 C, 64 %
//! let mut pairs = vec![Pair { pulse: 184, gap: 184 }; 4];
//! for b in frame {
//!     for i in (0..8).rev() {
//!         pairs.push(if b >> i & 1 == 1 {
//!             Pair { pulse: 125, gap: 62 }
//!         } else {
//!             Pair { pulse: 62, gap: 125 }
//!         });
//!     }
//! }
//! pairs.extend_from_within(..); // a second frame
//! if let Some(p) = pairs.last_mut() {
//!     p.gap = 0;
//! }
//! let header = Header { modulation: Modulation::Ook, channel: 0, freq_offset_hz: 0,
//!     level_db10: -250, noise_db10: -600, rate_hz: 250_000 };
//! let mut buf = vec![0u8; 16 + 8 * pairs.len()];
//! let n = pulses::encode_joined(&header, &pairs, &mut buf)?;
//! let train = pulses::decode(&buf[..n])?;
//! let mut sink = ReadingSink::new();
//! assert_eq!(Bresser3ch.decode(&train, &mut sink), DecodeResult::Decoded(1));
//! let r = sink.get(0).ok_or("no reading")?;
//! assert_eq!((r.id, r.channel, r.battery_ok), (0xD8, Some(2), Some(true)));
//! assert_eq!((r.temperature_c, r.humidity, r.repeats), (Some(22.1), Some(64.0), 2));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::{DecodeResult, MAX_READINGS, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Modulation, PulseTrain};
use crate::slicers::Tolerance;

/// The decoder of the Bresser 3CH sensors.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bresser3ch;

/// Nominal widths in microseconds.
const PREAMBLE_US: f32 = 735.0;
const LONG_US: f32 = 500.0;
const SHORT_US: f32 = 250.0;

/// Data bits of a frame.
const BITS: u32 = 40;
/// Preamble pairs a frame needs before its data (3 to 5 are sent, the first pulse may be lost).
const MIN_PREAMBLE: u32 = 2;

/// The raw temperature of 0 degrees Celsius (`C = (T - 1220) / 18`).
const T_ZERO: i32 = 1220;
/// The range of the device as raw values: -20 F and 160 F.
const MIN_RAW: i32 = 700;
const MAX_RAW: i32 = 2500;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Preamble,
    One,
    Zero,
}

/// The nominal widths of the three pair shapes in samples, `(pulse, gap)`.
struct Shapes {
    preamble: (f32, f32),
    one: (f32, f32),
    zero: (f32, f32),
    tol: Tolerance,
}

fn dist(w: f32, nominal: f32) -> f32 {
    if w > nominal {
        w - nominal
    } else {
        nominal - w
    }
}

impl Shapes {
    fn new(rate_hz: u32) -> Self {
        let s = |us: f32| us * (rate_hz as f32) / 1.0e6;
        Self {
            preamble: (s(PREAMBLE_US), s(PREAMBLE_US)),
            one: (s(LONG_US), s(SHORT_US)),
            zero: (s(SHORT_US), s(LONG_US)),
            tol: Tolerance::DEFAULT,
        }
    }

    /// The error of a width against a nominal one in samples, `None` outside the tolerance.
    fn err(&self, w: f32, nominal: f32) -> Option<f32> {
        let e = dist(w, nominal);
        (e <= (self.tol.rel() * nominal).max(self.tol.min_samples())).then_some(e)
    }

    /// The shape of a pair (the nearest of those within the tolerance); the gap of the last
    /// pair of a train is 0 and not checked.
    fn classify(&self, pulse: u32, gap: u32, last: bool) -> Option<Class> {
        let (p, g) = (pulse as f32, gap as f32);
        let mut best: Option<(f32, Class)> = None;
        for (class, (np, ng)) in [
            (Class::Preamble, self.preamble),
            (Class::One, self.one),
            (Class::Zero, self.zero),
        ] {
            let Some(ep) = self.err(p, np) else { continue };
            let eg = if last {
                0.0
            } else if let Some(e) = self.err(g, ng) {
                e
            } else {
                continue;
            };
            let e = ep + eg;
            if best.is_none_or(|(b, _)| e < b) {
                best = Some((e, class));
            }
        }
        best.map(|(_, c)| c)
    }
}

/// The checked contents of a frame.
struct Frame {
    id: u8,
    battery_ok: bool,
    channel: u8,
    tenths: i32,
    humidity: u8,
}

/// The five bytes of a frame (40 bits, byte 0 in the highest bits), or why they are no reading.
fn parse(bits: u64) -> Result<Frame, DecodeResult> {
    let byte = |i: u32| ((bits >> (8 * (4 - i))) & 0xFF) as u8;
    let (b0, b1, b2, b3, sum) = (byte(0), byte(1), byte(2), byte(3), byte(4));
    if b0.wrapping_add(b1).wrapping_add(b2).wrapping_add(b3) != sum {
        return Err(DecodeResult::BadIntegrity);
    }
    let channel = (b1 >> 4) & 3;
    if channel == 0 {
        return Err(DecodeResult::NoMatch);
    }
    let raw = (i32::from(b1 & 0x0F) << 8) | i32::from(b2);
    // tenths of a degree = round((raw - 1220) * 10 / 18), half away from zero (no tie occurs)
    let n = (raw - T_ZERO) * 5;
    let tenths = (2 * n + if n < 0 { -9 } else { 9 }) / 18;
    if !(MIN_RAW..=MAX_RAW).contains(&raw) || b3 > 100 {
        return Err(DecodeResult::OutOfRange);
    }
    Ok(Frame {
        id: b0,
        battery_ok: b1 & 0x80 == 0,
        channel,
        tenths,
        humidity: b3,
    })
}

impl SensorDecoder for Bresser3ch {
    fn name(&self) -> &'static str {
        "bresser_3ch"
    }

    fn modulation(&self) -> Modulation {
        Modulation::Ook
    }

    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
        let header = train.header();
        if header.rate_hz == 0 {
            return DecodeResult::NoMatch;
        }
        let shapes = Shapes::new(header.rate_hz);
        // the distinct valid frames of the train with their counts
        let mut found: [Option<(u64, u32)>; MAX_READINGS] = [None; MAX_READINGS];
        let mut res = DecodeResult::NoMatch;
        let (mut preamble, mut n_bits, mut bits) = (0u32, 0u32, 0u64);
        let n = train.len();
        for (i, p) in train.pairs().enumerate() {
            let class = shapes.classify(p.pulse, p.gap, i + 1 == n);
            let bit = match class {
                Some(Class::One) => Some(1u64),
                Some(Class::Zero) => Some(0),
                _ => None,
            };
            match bit {
                // data after a preamble
                Some(bit) if preamble >= MIN_PREAMBLE => {
                    bits = (bits << 1) | bit;
                    n_bits += 1;
                    if n_bits == BITS {
                        match parse(bits) {
                            Ok(_) => {
                                let slot =
                                    found.iter_mut().find(|s| s.is_none_or(|(b, _)| b == bits));
                                if let Some(slot) = slot {
                                    let count = slot.map_or(0, |(_, c)| c);
                                    *slot = Some((bits, count + 1));
                                }
                            }
                            Err(e) => res = res.best(e),
                        }
                        (preamble, n_bits, bits) = (0, 0, 0);
                    }
                }
                // data without a preamble: the tail of a frame cut at the start of the train
                Some(_) => {}
                None => {
                    if n_bits > 0 {
                        // a frame cut short: its preamble does not count for the next one
                        res = res.best(DecodeResult::Truncated);
                        preamble = 0;
                    }
                    n_bits = 0;
                    bits = 0;
                    preamble = if class == Some(Class::Preamble) {
                        preamble + 1
                    } else {
                        0
                    };
                }
            }
        }
        if n_bits > 0 {
            res = res.best(DecodeResult::Truncated);
        }
        let mut count = 0;
        for (bits, repeats) in found.iter().flatten() {
            let Ok(f) = parse(*bits) else { continue };
            let mut r = Reading::new("Bresser-3CH", u32::from(f.id), Mic::Checksum, &header);
            r.channel = Some(f.channel);
            r.battery_ok = Some(f.battery_ok);
            r.temperature_c = Some(f.tenths as f32 / 10.0);
            r.humidity = Some(f32::from(f.humidity));
            r.repeats = *repeats;
            if out.push(r) {
                count += 1;
            }
        }
        if count > 0 {
            DecodeResult::Decoded(count)
        } else {
            res
        }
    }
}
