//! The encoder side of the sensors (M46 extension, decision D1123): the mirror image of the
//! decoders. A [`SensorEncoder`] turns a [`Reading`] into the pulse train of the device (a
//! header and pairs of a pulse and a gap), with the timings of its decoder, a number of repeats
//! and the repeat gap of the device. The widths are written in microseconds at the sample rate
//! of 1 MHz ([`TRAIN_RATE_HZ`]), so that every nominal timing is an integer; a consumer that
//! needs another rate converts the edges (the block `pulse_key` does, without drift).
//!
//! An encoder holds no state and never allocates: it writes the pairs into a [`TrainSink`] on
//! storage of the caller (at most [`MAX_TRAIN_PAIRS`] pairs, the capacity of `sensor_decode`).
//! A reading that cannot be sent is an [`EncodeError`] and nothing is written (the sink is
//! cut back). The registry [`ENCODERS`] has the order of
//! [`DECODERS`](super::DECODERS).
//!
//! A reading is encoded from the fields the device reports: a field the device does not carry
//! ([`EncodeError::UnsupportedField`]), a missing required field
//! ([`EncodeError::MissingField`]) and a value outside the range of the device
//! ([`EncodeError::OutOfRange`], the same limits as the decoder) are errors; the optional
//! `channel` and `battery_ok` default to channel 1 and a good battery. A value is rounded to the
//! resolution of the device (half away from zero), so `decode(encode(x)) == x` holds for every
//! reading a decoder can produce (tested over the whole range of every device).
//!
//! The encoders inherit what the decoders assume: the gap between the pulses of the `wh1080`, the
//! band of the `wh1080` and the repeat gaps rest on assumptions that no source states
//! (decisions D1109, D1124 and D1136 to D1146), and so do the transmitted trains. The flags a
//! [`Reading`] has no field for (the test button of the `bresser_3ch` and the `nexus`, the
//! new-battery flag of the `lacrosse_tx29`) are sent as 0.
//!
//! ```
//! use sdrk_proto::pulses::{self, Header, Modulation};
//! use sdrk_proto::sensors::{self, Mic, Reading, ReadingSink, SensorEncoder, TrainSink};
//!
//! let header = Header { modulation: Modulation::Ook, channel: 0, freq_offset_hz: 0,
//!     level_db10: 0, noise_db10: 0, rate_hz: 1_000_000 };
//! let mut r = Reading::new("nexus", 0x5C, Mic::Repeat, &header);
//! r.channel = Some(2);
//! r.battery_ok = Some(true);
//! r.temperature_c = Some(-3.5);
//! r.humidity = Some(71.0);
//!
//! let enc = sensors::encoder_by_model("nexus").ok_or("no encoder")?;
//! let mut storage = [pulses::Pair::default(); sensors::MAX_TRAIN_PAIRS];
//! let mut sink = TrainSink::new(&mut storage);
//! enc.encode(&r, 3, &mut sink)?; // three frames in one train
//! assert_eq!(sink.len(), 3 * 37);
//!
//! // the train decodes to the reading, with `repeats` = 3
//! let mut buf = vec![0u8; 16 + 8 * sink.len()];
//! let n = pulses::encode_joined(&enc.header(), sink.pairs(), &mut buf)?;
//! let train = pulses::decode(&buf[..n])?;
//! let mut readings = ReadingSink::new();
//! let dec = sensors::by_name("nexus").ok_or("no decoder")?;
//! assert_eq!(dec.decode(&train, &mut readings).count(), 1);
//! let got = readings.get(0).ok_or("no reading")?;
//! assert!(got.same_content(&r));
//! assert_eq!(got.repeats, 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::fmt;

use super::{Mic, Reading};
use crate::pulses::{Header, Modulation, Pair};

/// The sample rate of the pulse trains of the encoders: 1 MHz, one sample per microsecond.
pub const TRAIN_RATE_HZ: u32 = 1_000_000;

/// The most pairs a train can hold: the capacity of `sensor_decode` (decision D1121).
pub const MAX_TRAIN_PAIRS: usize = 1200;

/// Why a reading cannot be encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncodeError {
    /// The model of the reading is not one of the encoder.
    UnknownModel,
    /// A field the device needs is missing (the JSON name).
    MissingField(&'static str),
    /// A field the device does not carry is set (the JSON name).
    UnsupportedField(&'static str),
    /// A value outside the range of the device or not finite (the JSON name).
    OutOfRange(&'static str),
    /// `repeats` is 0.
    InvalidRepeats,
    /// The train does not fit the storage of the sink.
    BufferTooSmall,
    /// The message cannot be seen by the decoder: its last bits are zeros of an FSK frame that
    /// the burst end hides (more than seven, see `lacrosse_tx29`), or it is a frame that the
    /// decoder takes for another device (a `nexus` frame that passes the check of the Rubicson
    /// protocol).
    Unrepresentable,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownModel => f.write_str("the model is not one of this encoder"),
            Self::MissingField(n) => write!(f, "the field {n} is missing"),
            Self::UnsupportedField(n) => write!(f, "the device does not carry the field {n}"),
            Self::OutOfRange(n) => write!(f, "the value of {n} is outside the range of the device"),
            Self::InvalidRepeats => f.write_str("repeats must be at least 1"),
            Self::BufferTooSmall => f.write_str("the train does not fit the buffer"),
            Self::Unrepresentable => {
                f.write_str("the message cannot be told from a message the decoder rejects")
            }
        }
    }
}

impl core::error::Error for EncodeError {}

/// Pairs of a train on storage of the caller (see the [module documentation](crate::sensors)).
#[derive(Debug)]
pub struct TrainSink<'a> {
    pairs: &'a mut [Pair],
    len: usize,
}

impl<'a> TrainSink<'a> {
    /// An empty sink over `storage` (the train can have at most `storage.len()` pairs).
    pub fn new(storage: &'a mut [Pair]) -> Self {
        Self {
            pairs: storage,
            len: 0,
        }
    }

    /// Removes every pair.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// The number of pairs.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sink is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The capacity in pairs.
    pub fn capacity(&self) -> usize {
        self.pairs.len()
    }

    /// The pairs written.
    pub fn pairs(&self) -> &[Pair] {
        self.pairs.get(..self.len).unwrap_or(&[])
    }

    /// Cuts the train back to `len` pairs (no effect when it holds fewer).
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// Adds a pair.
    ///
    /// # Errors
    /// [`EncodeError::BufferTooSmall`] when the storage is full (nothing is added).
    pub fn push(&mut self, pulse: u32, gap: u32) -> Result<(), EncodeError> {
        let slot = self
            .pairs
            .get_mut(self.len)
            .ok_or(EncodeError::BufferTooSmall)?;
        *slot = Pair { pulse, gap };
        self.len += 1;
        Ok(())
    }

    /// Sets the gap of the last pair (the repeat gap, or 0 at the end of the train).
    pub(crate) fn set_last_gap(&mut self, gap: u32) {
        if let Some(i) = self.len.checked_sub(1)
            && let Some(p) = self.pairs.get_mut(i)
        {
            p.gap = gap;
        }
    }

    /// Runs `f`; when it fails, the pairs it wrote are removed again.
    pub(crate) fn atomic<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, EncodeError>,
    ) -> Result<T, EncodeError> {
        let mark = self.len;
        let res = f(self);
        if res.is_err() {
            self.len = mark;
        }
        res
    }

    /// Writes `repeats` copies made by `copy`, `gap_us` apart (the gap of the last pair of a
    /// copy is replaced; 0 keeps the gap the copy ended with), and ends the train with a gap of 0
    /// (the pause after the last copy is the end of the train).
    pub(crate) fn copies(
        &mut self,
        repeats: u32,
        gap_us: u32,
        mut copy: impl FnMut(&mut Self) -> Result<(), EncodeError>,
    ) -> Result<(), EncodeError> {
        if repeats == 0 {
            return Err(EncodeError::InvalidRepeats);
        }
        for i in 0..repeats {
            if i > 0 && gap_us > 0 {
                self.set_last_gap(gap_us);
            }
            copy(self)?;
        }
        self.set_last_gap(0);
        Ok(())
    }

    /// Pulse-width coded bits, MSB first: a 1 is `one_us`, a 0 `zero_us`, every pulse is
    /// followed by `gap_us`.
    pub(crate) fn pwm_bytes(
        &mut self,
        bytes: &[u8],
        one_us: u32,
        zero_us: u32,
        gap_us: u32,
    ) -> Result<(), EncodeError> {
        for &b in bytes {
            for k in (0..8).rev() {
                self.push(if b >> k & 1 == 1 { one_us } else { zero_us }, gap_us)?;
            }
        }
        Ok(())
    }
}

/// A device encoder (see the [module documentation](crate::sensors)).
pub trait SensorEncoder: Sync {
    /// The name in the registry, the same as the decoder's.
    fn name(&self) -> &'static str;

    /// The model strings of the readings this encoder takes (`Reading::model`): the name of
    /// the decoder, except for `bresser_3ch` (`"Bresser-3CH"`) and `oregon_v21` (two models).
    fn models(&self) -> &'static [&'static str];

    /// The modulation of the device.
    fn modulation(&self) -> Modulation;

    /// How a reading of this device is checked (the `mic` of the decoder's readings).
    fn mic(&self) -> Mic;

    /// The decimals of `temperature_C` in the JSON form of the device (1; 2 for `bresser_3ch`).
    fn temp_decimals(&self) -> u8 {
        1
    }

    /// The number of repeats in a train when the caller does not choose one (the device sends
    /// several copies; the figures are the choices of decision D1124).
    fn default_repeats(&self) -> u32;

    /// The gap between two repeats in microseconds (0 for a device whose frames follow each
    /// other without a pause, such as `bresser_3ch`).
    fn repeat_gap_us(&self) -> u32;

    /// The header of the trains of this encoder: the modulation of the device, the rate
    /// [`TRAIN_RATE_HZ`], channel, offset and levels 0.
    fn header(&self) -> Header {
        Header {
            modulation: self.modulation(),
            channel: 0,
            freq_offset_hz: 0,
            level_db10: 0,
            noise_db10: 0,
            rate_hz: TRAIN_RATE_HZ,
        }
    }

    /// Writes `repeats` copies of the message of `reading` to `out` (appended; the last gap of
    /// the train is 0).
    ///
    /// # Errors
    /// See [`EncodeError`]; on an error `out` is unchanged.
    fn encode(
        &self,
        reading: &Reading,
        repeats: u32,
        out: &mut TrainSink<'_>,
    ) -> Result<(), EncodeError>;
}

/// Field bits of [`Carries`].
pub(crate) const CHANNEL: u8 = 1;
pub(crate) const BATTERY: u8 = 2;
pub(crate) const HUMIDITY: u8 = 4;
pub(crate) const WIND: u8 = 8;
pub(crate) const RAIN: u8 = 16;

/// The fields a device carries, as a set of the bits above (the others are rejected).
#[derive(Clone, Copy)]
pub(crate) struct Carries(pub u8);

impl Carries {
    /// Checks that `r` has no field outside the set.
    pub(crate) fn check(self, r: &Reading) -> Result<(), EncodeError> {
        let bad = |bit: u8, set: bool, name: &'static str| {
            if set && self.0 & bit == 0 {
                Err(EncodeError::UnsupportedField(name))
            } else {
                Ok(())
            }
        };
        bad(CHANNEL, r.channel.is_some(), "channel")?;
        bad(BATTERY, r.battery_ok.is_some(), "battery_ok")?;
        bad(HUMIDITY, r.humidity.is_some(), "humidity")?;
        bad(WIND, r.wind_avg_km_h.is_some(), "wind_avg_km_h")?;
        bad(WIND, r.wind_max_km_h.is_some(), "wind_max_km_h")?;
        bad(WIND, r.wind_dir_deg.is_some(), "wind_dir_deg")?;
        bad(RAIN, r.rain_mm.is_some(), "rain_mm")?;
        bad(0, r.pressure_hpa.is_some(), "pressure_hPa")
    }
}

/// `v * scale` rounded half away from zero, or `OutOfRange(name)` for a value that is not
/// finite or beyond a million units.
pub(crate) fn scaled(v: f32, scale: f32, name: &'static str) -> Result<i32, EncodeError> {
    let x = v * scale;
    if !x.is_finite() || x.abs() > 1.0e6 {
        return Err(EncodeError::OutOfRange(name));
    }
    // the cast truncates toward zero
    Ok(if x < 0.0 {
        (x - 0.5) as i32
    } else {
        (x + 0.5) as i32
    })
}

/// A scaled value within `lo..=hi`.
pub(crate) fn scaled_in(
    v: f32,
    scale: f32,
    lo: i32,
    hi: i32,
    name: &'static str,
) -> Result<i32, EncodeError> {
    let n = scaled(v, scale, name)?;
    if (lo..=hi).contains(&n) {
        Ok(n)
    } else {
        Err(EncodeError::OutOfRange(name))
    }
}

/// The required temperature in units of `1 / scale` degrees within `lo..=hi`.
pub(crate) fn temperature(r: &Reading, scale: f32, lo: i32, hi: i32) -> Result<i32, EncodeError> {
    let t = r
        .temperature_c
        .ok_or(EncodeError::MissingField("temperature_C"))?;
    scaled_in(t, scale, lo, hi, "temperature_C")
}

/// The required humidity, a whole percent within 0 to 100.
pub(crate) fn humidity(r: &Reading) -> Result<u8, EncodeError> {
    let h = r.humidity.ok_or(EncodeError::MissingField("humidity"))?;
    Ok(scaled_in(h, 1.0, 0, 100, "humidity")? as u8)
}

/// The optional humidity, a whole percent within 0 to 100 (`None` when the reading has none).
pub(crate) fn humidity_opt(r: &Reading) -> Result<Option<u8>, EncodeError> {
    r.humidity
        .map(|h| scaled_in(h, 1.0, 0, 100, "humidity").map(|v| v as u8))
        .transpose()
}

/// The channel (default 1) within `1..=max`.
pub(crate) fn channel(r: &Reading, max: u8) -> Result<u8, EncodeError> {
    let c = r.channel.unwrap_or(1);
    if (1..=max).contains(&c) {
        Ok(c)
    } else {
        Err(EncodeError::OutOfRange("channel"))
    }
}

/// The device id, below `limit`.
pub(crate) fn id(r: &Reading, limit: u32) -> Result<u32, EncodeError> {
    if r.id < limit {
        Ok(r.id)
    } else {
        Err(EncodeError::OutOfRange("id"))
    }
}

/// The model of `r` is one of `models`.
pub(crate) fn model(r: &Reading, models: &[&str]) -> Result<(), EncodeError> {
    if models.contains(&r.model) {
        Ok(())
    } else {
        Err(EncodeError::UnknownModel)
    }
}
