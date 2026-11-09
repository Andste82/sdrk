//! The reading of a sensor and its JSON form.

use core::fmt;

use crate::pulses::Header;

/// How a reading was checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Mic {
    /// A CRC over the message.
    Crc,
    /// A simple checksum (a sum of nibbles or bytes).
    Checksum,
    /// The device has no check: two equal repeats were required.
    Repeat,
}

impl Mic {
    /// The name in the JSON form: `"crc"`, `"checksum"` or `"repeat"`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Crc => "crc",
            Self::Checksum => "checksum",
            Self::Repeat => "repeat",
        }
    }
}

/// One decoded reading. The values are `None` for what the device does not report. Units are
/// metric: degrees Celsius, percent, km/h, degrees, millimetres, hectopascal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    /// Time of the reading in seconds since the start of the stream (set by the block; written
    /// first when present).
    pub time_s: Option<f64>,
    /// The decoder's model name, for example `"nexus"`.
    pub model: &'static str,
    /// The device id as the device sends it.
    pub id: u32,
    /// The channel switch of the device, when it has one (the number the device shows).
    pub channel: Option<u8>,
    /// Battery state: `Some(true)` for good, `Some(false)` for low.
    pub battery_ok: Option<bool>,
    /// Temperature in degrees Celsius, written with [`temp_decimals`](Self::temp_decimals)
    /// decimals.
    pub temperature_c: Option<f32>,
    /// The decimals of `temperature_C` in the JSON form: 1 for every device except
    /// `Bresser-3CH`, whose steps of 0.1 degree Fahrenheit (0.0556 degrees Celsius) need 2
    /// (decision D1123). It is a function of the model, not part of the content.
    pub temp_decimals: u8,
    /// Relative humidity in percent (written as an integer).
    pub humidity: Option<f32>,
    /// Average wind speed in km/h (one decimal).
    pub wind_avg_km_h: Option<f32>,
    /// Gust speed in km/h (one decimal).
    pub wind_max_km_h: Option<f32>,
    /// Wind direction in degrees (one decimal).
    pub wind_dir_deg: Option<f32>,
    /// Accumulated rain in millimetres (one decimal).
    pub rain_mm: Option<f32>,
    /// Air pressure in hectopascal (one decimal).
    pub pressure_hpa: Option<f32>,
    /// How the message was checked.
    pub mic: Mic,
    /// The frequency offset of the burst from the band centre in Hz (from the pulse train).
    pub freq_offset_hz: i32,
    /// The centre of the band in Hz; 0 unless the caller sets it (the offset is then written).
    pub center_hz: u32,
    /// The level of the burst in dB (from the pulse train).
    pub rssi_db: f32,
    /// How many copies were merged into this reading (1 for a single message).
    pub repeats: u32,
}

impl Reading {
    /// A reading of `model` and `id` with no values, `mic`, and the radio fields of `header`.
    ///
    /// ```
    /// use sdrk_proto::pulses::{Header, Modulation};
    /// use sdrk_proto::sensors::{Mic, Reading};
    /// let h = Header { modulation: Modulation::Ook, channel: 0, freq_offset_hz: -2000,
    ///     level_db10: -125, noise_db10: -400, rate_hz: 250_000 };
    /// let mut r = Reading::new("demo", 7, Mic::Crc, &h);
    /// r.temperature_c = Some(21.5);
    /// r.center_hz = 433_920_000;
    /// let mut s = String::new();
    /// r.to_json(&mut s)?;
    /// assert_eq!(s, r#"{"model":"demo","id":7,"temperature_C":21.5,"mic":"crc","freq_MHz":433.9180,"rssi_dB":-12.5,"repeats":1}"#);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(model: &'static str, id: u32, mic: Mic, header: &Header) -> Self {
        Self {
            time_s: None,
            model,
            id,
            channel: None,
            battery_ok: None,
            temperature_c: None,
            temp_decimals: 1,
            humidity: None,
            wind_avg_km_h: None,
            wind_max_km_h: None,
            wind_dir_deg: None,
            rain_mm: None,
            pressure_hpa: None,
            mic,
            freq_offset_hz: header.freq_offset_hz,
            center_hz: 0,
            rssi_db: f32::from(header.level_db10) / 10.0,
            repeats: 1,
        }
    }

    /// Whether two readings are the same message content: model, id, channel, battery and
    /// every value (the radio fields, time and repeat count are not compared).
    pub fn same_content(&self, other: &Self) -> bool {
        self.model == other.model
            && self.id == other.id
            && self.channel == other.channel
            && self.battery_ok == other.battery_ok
            && self.temperature_c == other.temperature_c
            && self.humidity == other.humidity
            && self.wind_avg_km_h == other.wind_avg_km_h
            && self.wind_max_km_h == other.wind_max_km_h
            && self.wind_dir_deg == other.wind_dir_deg
            && self.rain_mm == other.rain_mm
            && self.pressure_hpa == other.pressure_hpa
    }

    /// A 64-bit FNV-1a hash of the fields [`same_content`](Self::same_content) compares: equal
    /// content gives an equal hash (the block `sensor_decode` keeps these instead of readings).
    ///
    /// ```
    /// use sdrk_proto::pulses::{Header, Modulation};
    /// use sdrk_proto::sensors::{Mic, Reading};
    /// let h = Header { modulation: Modulation::Ook, channel: 0, freq_offset_hz: 0,
    ///     level_db10: 0, noise_db10: 0, rate_hz: 250_000 };
    /// let mut a = Reading::new("demo", 7, Mic::Crc, &h);
    /// let mut b = a;
    /// b.rssi_db = -30.0;
    /// assert_eq!(a.content_hash(), b.content_hash());
    /// a.temperature_c = Some(1.5);
    /// assert_ne!(a.content_hash(), b.content_hash());
    /// ```
    pub fn content_hash(&self) -> u64 {
        fn add(h: &mut u64, bytes: &[u8]) {
            for &b in bytes {
                *h ^= u64::from(b);
                *h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        fn opt(h: &mut u64, v: Option<f32>) {
            match v {
                Some(x) => {
                    add(h, &[1]);
                    add(h, &x.to_bits().to_le_bytes());
                }
                None => add(h, &[0]),
            }
        }
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        add(&mut h, self.model.as_bytes());
        add(&mut h, &self.id.to_le_bytes());
        add(
            &mut h,
            &[
                self.channel.unwrap_or(0xff),
                u8::from(self.channel.is_some()),
            ],
        );
        add(&mut h, &[self.battery_ok.map_or(2, u8::from)]);
        for v in [
            self.temperature_c,
            self.humidity,
            self.wind_avg_km_h,
            self.wind_max_km_h,
            self.wind_dir_deg,
            self.rain_mm,
            self.pressure_hpa,
        ] {
            opt(&mut h, v);
        }
        h
    }

    /// Writes the reading as one JSON object without a trailing newline. The fields come in
    /// a fixed order and only when present: `time`, `model`, `id`, `channel`, `battery_ok`,
    /// `temperature_C` (one decimal, two for `Bresser-3CH`), `humidity`, `wind_avg_km_h`,
    /// `wind_max_km_h`, `wind_dir_deg`, `rain_mm`, `pressure_hPa`, `mic`, `freq_MHz` (four decimals), `rssi_dB` (one decimal),
    /// `repeats`.
    ///
    /// # Errors
    /// `fmt::Error` when the writer fails or a value is not finite.
    pub fn to_json<W: fmt::Write>(&self, w: &mut W) -> fmt::Result {
        w.write_char('{')?;
        if let Some(t) = self.time_s {
            check(t)?;
            write!(w, "\"time\":{t:.3},")?;
        }
        w.write_str("\"model\":\"")?;
        write_escaped(w, self.model)?;
        write!(w, "\",\"id\":{}", self.id)?;
        if let Some(c) = self.channel {
            write!(w, ",\"channel\":{c}")?;
        }
        if let Some(b) = self.battery_ok {
            write!(w, ",\"battery_ok\":{}", if b { "true" } else { "false" })?;
        }
        value(
            w,
            "temperature_C",
            self.temperature_c,
            usize::from(self.temp_decimals),
        )?;
        value(w, "humidity", self.humidity, 0)?;
        value(w, "wind_avg_km_h", self.wind_avg_km_h, 1)?;
        value(w, "wind_max_km_h", self.wind_max_km_h, 1)?;
        value(w, "wind_dir_deg", self.wind_dir_deg, 1)?;
        value(w, "rain_mm", self.rain_mm, 1)?;
        value(w, "pressure_hPa", self.pressure_hpa, 1)?;
        write!(w, ",\"mic\":\"{}\"", self.mic.as_str())?;
        let f = (f64::from(self.center_hz) + f64::from(self.freq_offset_hz)) / 1.0e6;
        write!(w, ",\"freq_MHz\":{f:.4}")?;
        let rssi = self.rssi_db;
        check(f64::from(rssi))?;
        // adding 0.0 turns -0.0 into 0.0
        write!(w, ",\"rssi_dB\":{:.1}", rssi + 0.0)?;
        write!(w, ",\"repeats\":{}", self.repeats)?;
        w.write_char('}')
    }
}

fn check(v: f64) -> fmt::Result {
    if v.is_finite() {
        Ok(())
    } else {
        Err(fmt::Error)
    }
}

fn value<W: fmt::Write>(w: &mut W, name: &str, v: Option<f32>, decimals: usize) -> fmt::Result {
    let Some(v) = v else { return Ok(()) };
    check(f64::from(v))?;
    // adding 0.0 turns -0.0 into 0.0
    write!(w, ",\"{name}\":{:.decimals$}", v + 0.0)
}

fn write_escaped<W: fmt::Write>(w: &mut W, s: &str) -> fmt::Result {
    for c in s.chars() {
        match c {
            '"' => w.write_str("\\\"")?,
            '\\' => w.write_str("\\\\")?,
            c if (c as u32) < 0x20 => write!(w, "\\u{:04x}", c as u32)?,
            c => w.write_char(c)?,
        }
    }
    Ok(())
}

/// A fixed buffer for the JSON of a reading (256 bytes, the size of a packet payload); a
/// reading that does not fit is an error, never truncated.
#[derive(Debug, Clone)]
pub struct JsonBuf {
    bytes: [u8; Self::CAPACITY],
    len: usize,
}

impl Default for JsonBuf {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonBuf {
    /// The capacity in bytes.
    pub const CAPACITY: usize = 256;

    /// An empty buffer.
    pub const fn new() -> Self {
        Self {
            bytes: [0; Self::CAPACITY],
            len: 0,
        }
    }

    /// Empties the buffer.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// The number of bytes written.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Cuts the buffer back to `len` bytes (no effect when it holds fewer); a writer uses it
    /// to take back a partial write.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// The bytes written.
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.get(..self.len).unwrap_or(&[])
    }
}

impl fmt::Write for JsonBuf {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len.checked_add(s.len()).ok_or(fmt::Error)?;
        let dst = self.bytes.get_mut(self.len..end).ok_or(fmt::Error)?;
        dst.copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}
