//! `LaCrosse` IT+ `TX29` temperature (and humidity) sensor (868 MHz, FSK); also the frame core of the `TX35`.
//!
//! Source: protocol facts from the maintainer-provided fact document about the `rtl_433` decoders
//! (written by a third party, nothing copied; decision D1141) over the earlier basis of decision
//! D1110 (the frame description and the two worked examples of the article "`SensorLogger`" of
//! the German wiki `mikrocontroller.net`). The line coding is NRZ at 17.241 kbit/s (58 us per
//! bit; the document names 55 us and 17.24 kbit/s, which are 58 us, and the decoder accepts
//! both), the frame is found by the 24 bit pattern 0xA2DD49 (the last nibble of the preamble,
//! the sync word 0x2DD4 and the model nibble 9) and has five bytes B0 to B4, MSB first: the high
//! nibble of B0 is the model (9), the low nibble of B0 and the high two bits of B1 the 6 bit id,
//! bit 5 of B1 the new-battery flag (not reported), the low nibble of B1 and B2 the temperature
//! as three decimal digits d2 d1 d0 with t = (d2 d1 d0) / 10 - 40 degrees, the low seven bits of
//! B3 the humidity (106 = no humidity sensor, 125 = the probe of the TX25U, which is rejected),
//! bit 7 of B3 the weak battery flag and B4 a CRC-8 (polynomial 0x31, init 0, MSB first) over B0
//! to B3.

use super::encoder::{self, BATTERY, Carries, EncodeError, HUMIDITY, SensorEncoder, TrainSink};
use super::util::{collect, crc8_msb};
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Header, Modulation, PulseTrain};
use crate::slicers::{Nrz, Row, Rows, Slicer};

/// The decoder and encoder of the TX29.
#[derive(Debug, Clone, Copy, Default)]
pub struct LacrosseTx29;

/// The bit times the decoder tries, in microseconds.
const BIT_TIMES_US: [f32; 2] = [58.0, 55.0];
/// The pattern a frame starts with: the last preamble nibble 0xA, the sync word 0x2DD4 and the
/// model nibble 9.
const PATTERN: u32 = 0xA2_DD49;
/// The bits of the pattern before the five bytes of the frame (the model nibble belongs to B0).
const PATTERN_BITS: usize = 24;

impl SensorDecoder for LacrosseTx29 {
    fn name(&self) -> &'static str {
        "lacrosse_tx29"
    }

    fn modulation(&self) -> Modulation {
        Modulation::Fsk
    }

    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
        // the bit time is 58 us (17.241 kbit/s) or 55 us (the figure of the fact document): the
        // tolerance of a run is 20 % of a bit, so a long run needs the right one of the two
        decode_frames(train, out, "lacrosse_tx29", &BIT_TIMES_US, RESET_US)
    }
}

/// The decoder core shared by the devices of this frame layout (`lacrosse_tx29` and
/// `lacrosse_tx35`, which differ in the bit time and the model name): tries every bit time and
/// stops at the first that decodes.
pub(super) fn decode_frames(
    train: &PulseTrain<'_>,
    out: &mut ReadingSink,
    model: &'static str,
    bit_times_us: &[f32],
    reset_us: f32,
) -> DecodeResult {
    let header = train.header();
    let mut res = DecodeResult::NoMatch;
    for &bit_us in bit_times_us {
        let Ok(slicer) = Nrz::new(bit_us, reset_us) else {
            return DecodeResult::NoMatch;
        };
        let mut rows = Rows::new();
        slicer.slice(train, &mut rows);
        res = res.best(collect(&rows, out, |row| decode_row(row, &header, model)));
        if matches!(res, DecodeResult::Decoded(_)) {
            break;
        }
    }
    res
}

fn find_sync(row: &Row) -> Option<usize> {
    let last = row.len().checked_sub(PATTERN_BITS)?;
    (0..=last).find(|&p| row.field(p, PATTERN_BITS) == Some(PATTERN))
}

fn decode_row(row: &Row, header: &Header, model: &'static str) -> Result<Reading, DecodeResult> {
    let p = find_sync(row).ok_or(DecodeResult::NoMatch)?;
    // the frame starts with the model nibble, the last four bits of the pattern
    let first = p + PATTERN_BITS - 4;
    // The last bits of the frame may be zeros that the detector sees as the end of the burst
    // (the last gap of a train is 0): up to seven missing bits are read as 0, the CRC decides.
    // Rows cut by an error keep only a prefix: the zero-fill is for Reset and TrainEnd only.
    let missing = (first + 40).saturating_sub(row.len());
    if missing > 7 || (missing > 0 && row.end().is_error()) {
        return Err(DecodeResult::Truncated);
    }
    let mut b = [0u8; 5];
    for (i, v) in b.iter_mut().enumerate() {
        let mut x = 0u8;
        for j in 0..8 {
            x = (x << 1) | u8::from(row.bit(first + 8 * i + j).unwrap_or(false));
        }
        *v = x;
    }
    if b[0] >> 4 != 9 {
        return Err(DecodeResult::NoMatch);
    }
    if crc8_msb(0, b[..4].iter().copied()) != b[4] {
        return Err(DecodeResult::BadIntegrity);
    }
    // bit 5 of B1 is the new-battery flag and bit 4 is unused
    let d2 = b[1] & 0x0F;
    let d1 = b[2] >> 4;
    let d0 = b[2] & 0x0F;
    let hum = b[3] & 0x7F;
    if d2 > 9 || d1 > 9 || d0 > 9 || (hum > 100 && hum != 106) {
        return Err(DecodeResult::OutOfRange);
    }
    let digits = u32::from(d2) * 100 + u32::from(d1) * 10 + u32::from(d0);
    let id = (u32::from(b[0] & 0x0F) << 2) | u32::from(b[1] >> 6);
    let mut r = Reading::new(model, id, Mic::Crc, header);
    r.temperature_c = Some(digits as f32 / 10.0 - 40.0);
    if hum != 106 {
        r.humidity = Some(f32::from(hum));
    }
    r.battery_ok = Some(b[3] & 0x80 == 0);
    Ok(r)
}

/// The bit time in microseconds, 17.241 kbit/s rounded to a whole microsecond (the decoder's
/// 58 us).
const BIT_US: u32 = 58;
/// The pause between two copies (the repeat gap, decision D1124).
pub(super) const REPEAT_GAP_US: u32 = 4000;
/// A gap at least this long ends a row: a frame holds up to 35 zero bits in a row (id 0,
/// temperature -40.0, humidity 0, a CRC with leading zeros: 2030 us at 58 us per bit, 3675 us at
/// 105 us per bit); the pause between two copies of the encoder is 4000 us (the fact document
/// has a row end of 4000 us for both devices).
pub(super) const RESET_US: f32 = 3900.0;

impl SensorEncoder for LacrosseTx29 {
    fn name(&self) -> &'static str {
        "lacrosse_tx29"
    }

    fn models(&self) -> &'static [&'static str] {
        &["lacrosse_tx29"]
    }

    fn modulation(&self) -> Modulation {
        Modulation::Fsk
    }

    fn mic(&self) -> Mic {
        Mic::Crc
    }

    fn default_repeats(&self) -> u32 {
        2
    }

    fn repeat_gap_us(&self) -> u32 {
        REPEAT_GAP_US
    }

    /// The frame of the decoder as the pulse train of an FSK burst: four bytes 0xAA, the sync
    /// word 0x2DD4 and the five bytes of the frame (model nibble 9, the 6 bit id, the
    /// new-battery flag 0, the temperature from -40.0 to 59.9 as three decimal digits of
    /// `t + 40`, the humidity or 106 when absent, the weak battery bit, the CRC-8) as NRZ bits of
    /// 58 us: a run of ones is a
    /// pulse (the upper frequency), a run of zeros a gap, copies 4 ms apart. Zeros at the end of
    /// a frame are not part of the train (the burst ends); a frame with more than seven of
    /// them (a CRC byte of 0x00) is [`EncodeError::Unrepresentable`]. The device uses two
    /// frequencies, so the train is sent by the FSK chain (`pulse_key` in mode `fsk`, then
    /// `fsk_mod`, flow `lacrosse_868_tx.json`, D1146 and D1147), not by the OOK chain.
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encode_frame(r, self.models(), BIT_US, repeats, out)
    }
}

/// The encoder core shared by the devices of this frame layout: `r` must be of one of `models`,
/// the bits last `bit_us` microseconds.
pub(super) fn encode_frame(
    r: &Reading,
    models: &[&str],
    bit_us: u32,
    repeats: u32,
    out: &mut TrainSink<'_>,
) -> Result<(), EncodeError> {
    encoder::model(r, models)?;
    Carries(BATTERY | HUMIDITY).check(r)?;
    let id = encoder::id(r, 64)?;
    let digits = encoder::temperature(r, 10.0, -400, 599)? + 400;
    let hum = match r.humidity {
        Some(_) => encoder::humidity(r)?,
        None => 106,
    };
    let weak = u8::from(r.battery_ok == Some(false));
    let mut frame = [
        0x90 | (id >> 2) as u8,
        (((id & 0x3) << 6) as u8) | (digits / 100) as u8,
        ((digits / 10 % 10) << 4) as u8 | (digits % 10) as u8,
        (weak << 7) | hum,
        0,
    ];
    let crc = crc8_msb(0, frame.iter().take(4).copied());
    if let Some(last) = frame.last_mut() {
        *last = crc;
    }
    let mut bytes = [0xAA, 0xAA, 0xAA, 0xAA, 0x2D, 0xD4, 0, 0, 0, 0, 0];
    for (slot, v) in bytes.iter_mut().skip(6).zip(frame) {
        *slot = v;
    }
    let mut n = bytes.len() * 8;
    let bit = |k: usize| bytes.get(k / 8).is_some_and(|b| b >> (7 - k % 8) & 1 == 1);
    while n > 0 && !bit(n - 1) {
        n -= 1;
    }
    if bytes.len() * 8 - n > 7 {
        return Err(EncodeError::Unrepresentable);
    }
    out.atomic(|o| {
        o.copies(repeats, REPEAT_GAP_US, |o| {
            let mut k = 0;
            while k < n {
                // the frame starts with a 1: a pulse, then a gap, alternately
                let mut pulse = 0;
                while k < n && bit(k) {
                    pulse += 1;
                    k += 1;
                }
                let mut gap = 0;
                while k < n && !bit(k) {
                    gap += 1;
                    k += 1;
                }
                o.push(pulse * bit_us, gap * bit_us)?;
            }
            Ok(())
        })
    })
}
