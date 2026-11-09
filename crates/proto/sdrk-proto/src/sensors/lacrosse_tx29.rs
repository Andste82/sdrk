//! `LaCrosse` IT+ `TX29` / `TX35` temperature and humidity sensor (868 MHz, FSK).
//!
//! Source: the frame description and the two worked examples of the article "`SensorLogger`" of
//! the German wiki `mikrocontroller.net`; see decision D1110. The line coding is taken as NRZ at
//! 17.241 kbit/s (58 us per bit, the data rate of the receiver set-up in the article), the
//! frame is found by the sync word 0x2DD4 and has five bytes B0 to B4, MSB first: the high
//! nibble of B0 is the length (9), the low nibble of B0 and the high nibble of B1 are the id,
//! the low nibble of B1 and B2 the temperature as three decimal digits d2 d1 d0 with
//! t = (d2 d1 d0) / 10 - 40 degrees, the low seven bits of B3 the humidity (106 = no humidity
//! sensor), bit 7 of B3 the weak battery flag and B4 a CRC-8 (polynomial 0x31, init 0, MSB
//! first) over B0 to B3. The CRC parameters are consistent with both examples of the article
//! but not confirmed by a source.

use super::encoder::{self, BATTERY, Carries, EncodeError, HUMIDITY, SensorEncoder, TrainSink};
use super::util::{collect, crc8_msb};
use super::{DecodeResult, Mic, Reading, ReadingSink, SensorDecoder};
use crate::pulses::{Header, Modulation, PulseTrain};
use crate::slicers::{Nrz, Row, Rows, Slicer};

/// The decoder of the TX29 / TX35.
#[derive(Debug, Clone, Copy, Default)]
pub struct LacrosseTx29;

const SYNC: u32 = 0x2DD4;

impl SensorDecoder for LacrosseTx29 {
    fn name(&self) -> &'static str {
        "lacrosse_tx29"
    }

    fn modulation(&self) -> Modulation {
        Modulation::Fsk
    }

    fn decode(&self, train: &PulseTrain<'_>, out: &mut ReadingSink) -> DecodeResult {
        let Ok(slicer) = Nrz::new(58.0, 2000.0) else {
            return DecodeResult::NoMatch;
        };
        let mut rows = Rows::new();
        slicer.slice(train, &mut rows);
        let header = train.header();
        collect(&rows, out, |row| decode_row(row, &header))
    }
}

fn find_sync(row: &Row) -> Option<usize> {
    let last = row.len().checked_sub(16)?;
    (0..=last).find(|&p| row.field(p, 16) == Some(SYNC))
}

fn decode_row(row: &Row, header: &Header) -> Result<Reading, DecodeResult> {
    let p = find_sync(row).ok_or(DecodeResult::NoMatch)?;
    let first = p + 16;
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
    let d2 = b[1] & 0x0F;
    let d1 = b[2] >> 4;
    let d0 = b[2] & 0x0F;
    let hum = b[3] & 0x7F;
    if d2 > 9 || d1 > 9 || d0 > 9 || (hum > 100 && hum != 106) {
        return Err(DecodeResult::OutOfRange);
    }
    let digits = u32::from(d2) * 100 + u32::from(d1) * 10 + u32::from(d0);
    let id = (u32::from(b[0] & 0x0F) << 4) | u32::from(b[1] >> 4);
    let mut r = Reading::new("lacrosse_tx29", id, Mic::Crc, header);
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
const REPEAT_GAP_US: u32 = 4000;

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
    /// word 0x2DD4 and the five bytes of the frame (length nibble 9, the 8 bit id, the
    /// temperature from -40.0 to 59.9 as three decimal digits of `t + 40`, the humidity or 106
    /// when absent, the weak battery bit, the CRC-8) as NRZ bits of 58 us: a run of ones is a
    /// pulse (the upper frequency), a run of zeros a gap, copies 4 ms apart. Zeros at the end of
    /// a frame are not part of the train (the burst ends); a frame with more than seven of
    /// them (a CRC byte of 0x00) is [`EncodeError::Unrepresentable`]. The OOK transmit chain
    /// cannot send this train (it keys the carrier on and off, the device uses two
    /// frequencies); the encoder exists for the symmetry of the registry and for tests.
    fn encode(
        &self,
        r: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError> {
        encoder::model(r, self.models())?;
        Carries(BATTERY | HUMIDITY).check(r)?;
        let id = encoder::id(r, 256)?;
        let digits = encoder::temperature(r, 10.0, -400, 599)? + 400;
        let hum = match r.humidity {
            Some(_) => encoder::humidity(r)?,
            None => 106,
        };
        let weak = u8::from(r.battery_ok == Some(false));
        let mut frame = [
            0x90 | (id >> 4) as u8,
            (((id & 0xF) << 4) as u8) | (digits / 100) as u8,
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
                    o.push(pulse * BIT_US, gap * BIT_US)?;
                }
                Ok(())
            })
        })
    }
}
