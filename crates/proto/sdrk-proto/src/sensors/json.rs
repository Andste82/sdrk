//! The JSON form of a reading read back: [`Reading::from_json`], the inverse of
//! [`Reading::to_json`] for the fields an encoder needs.

use core::fmt;

use super::{Mic, Reading, encoder_by_model};
use crate::pulses::{Header, Modulation};

/// Why a text is not a reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum JsonError {
    /// The text is not a JSON object (or nested deeper than eight levels).
    Syntax,
    /// A field the reading needs is missing (`model` or `id`).
    Missing(&'static str),
    /// A field has a value of the wrong type or outside its type (the JSON name).
    BadValue(&'static str),
    /// The model is not one of the registry [`ENCODERS`](super::ENCODERS).
    UnknownModel,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax => f.write_str("not a JSON object"),
            Self::Missing(n) => write!(f, "the field {n} is missing"),
            Self::BadValue(n) => write!(f, "the value of {n} is not valid"),
            Self::UnknownModel => f.write_str("the model is not in the registry"),
        }
    }
}

impl core::error::Error for JsonError {}

/// The longest key or model string that is kept (longer ones are not known names).
const KEEP: usize = 32;
/// The deepest nesting of a skipped value.
const MAX_DEPTH: u32 = 8;

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

/// A string read into a buffer: its length, or too long to keep.
struct Text {
    buf: [u8; KEEP],
    len: Option<usize>,
}

impl Text {
    fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(self.buf.get(..self.len?)?).ok()
    }
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), JsonError> {
        self.ws();
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(JsonError::Syntax)
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let d = self.peek().ok_or(JsonError::Syntax)?;
            v = v * 16 + char::from(d).to_digit(16).ok_or(JsonError::Syntax)?;
            self.i += 1;
        }
        Ok(v)
    }

    /// The character of a `\u` escape (the `\u` is consumed). A surrogate pair is combined; a
    /// lone surrogate, which JSON allows syntactically, becomes U+FFFD.
    fn unicode_escape(&mut self) -> Result<char, JsonError> {
        let first = self.hex4()?;
        if (0xD800..0xDC00).contains(&first) {
            let at = self.i;
            if self.b.get(at..at + 2) == Some(b"\\u".as_slice()) {
                self.i += 2;
                let low = self.hex4()?;
                if (0xDC00..0xE000).contains(&low) {
                    let cp = 0x10000 + ((first - 0xD800) << 10) + (low - 0xDC00);
                    return Ok(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                }
                self.i = at;
            }
        }
        Ok(char::from_u32(first).unwrap_or('\u{FFFD}'))
    }

    /// A string (the opening quote is at the position); the unescaped bytes are kept up to
    /// [`KEEP`].
    #[inline(never)]
    fn string(&mut self) -> Result<Text, JsonError> {
        self.eat(b'"')?;
        let mut t = Text {
            buf: [0; KEEP],
            len: Some(0),
        };
        let mut n = 0usize;
        let mut keep = true;
        let mut put = |t: &mut Text, bytes: &[u8]| {
            for &c in bytes {
                match t.buf.get_mut(n) {
                    Some(slot) if keep => {
                        *slot = c;
                        n += 1;
                    }
                    _ => keep = false,
                }
            }
        };
        loop {
            let c = self.peek().ok_or(JsonError::Syntax)?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.peek().ok_or(JsonError::Syntax)?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => self.unicode_escape()?,
                        _ => return Err(JsonError::Syntax),
                    };
                    let mut enc = [0u8; 4];
                    put(&mut t, ch.encode_utf8(&mut enc).as_bytes());
                }
                0..=0x1F => return Err(JsonError::Syntax),
                c => put(&mut t, &[c]),
            }
        }
        t.len = keep.then_some(n);
        Ok(t)
    }

    /// The bare token of a number or literal.
    #[inline(never)]
    fn token(&mut self) -> Result<&str, JsonError> {
        self.ws();
        let start = self.i;
        while matches!(
            self.peek(),
            Some(b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'+' | b'-' | b'.')
        ) {
            self.i += 1;
        }
        let t = self.b.get(start..self.i).ok_or(JsonError::Syntax)?;
        if t.is_empty() {
            return Err(JsonError::Syntax);
        }
        core::str::from_utf8(t).map_err(|_| JsonError::Syntax)
    }

    /// Skips one value of any kind.
    #[inline(never)]
    fn skip(&mut self, depth: u32) -> Result<(), JsonError> {
        self.ws();
        match self.peek() {
            Some(b'"') => self.string().map(|_| ()),
            Some(open @ (b'{' | b'[')) => {
                if depth >= MAX_DEPTH {
                    return Err(JsonError::Syntax);
                }
                self.i += 1;
                let close = if open == b'{' { b'}' } else { b']' };
                self.ws();
                if self.peek() == Some(close) {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    if open == b'{' {
                        self.string()?;
                        self.eat(b':')?;
                    }
                    self.skip(depth + 1)?;
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(c) if c == close => {
                            self.i += 1;
                            return Ok(());
                        }
                        _ => return Err(JsonError::Syntax),
                    }
                }
            }
            _ => self.token().map(|_| ()),
        }
    }

    /// The token of a scalar value; a string, an object or an array is a bad value.
    fn scalar(&mut self, name: &'static str) -> Result<&str, JsonError> {
        self.ws();
        if matches!(self.peek(), Some(b'"' | b'{' | b'[')) {
            return Err(JsonError::BadValue(name));
        }
        self.token()
    }

    /// A number, or `None` for `null`.
    fn number(&mut self, name: &'static str) -> Result<Option<f64>, JsonError> {
        let t = self.scalar(name)?;
        if t == "null" {
            return Ok(None);
        }
        let v: f64 = t.parse().map_err(|_| JsonError::BadValue(name))?;
        if v.is_finite() {
            Ok(Some(v))
        } else {
            Err(JsonError::BadValue(name))
        }
    }

    fn float(&mut self, name: &'static str) -> Result<Option<f32>, JsonError> {
        let t = self.scalar(name)?;
        if t == "null" {
            return Ok(None);
        }
        let v: f32 = t.parse().map_err(|_| JsonError::BadValue(name))?;
        if v.is_finite() {
            Ok(Some(v))
        } else {
            Err(JsonError::BadValue(name))
        }
    }

    fn integer(&mut self, name: &'static str) -> Result<Option<u32>, JsonError> {
        let t = self.scalar(name)?;
        if t == "null" {
            return Ok(None);
        }
        t.parse::<u32>()
            .map(Some)
            .map_err(|_| JsonError::BadValue(name))
    }

    fn boolean(&mut self, name: &'static str) -> Result<Option<bool>, JsonError> {
        match self.scalar(name)? {
            "true" => Ok(Some(true)),
            "false" => Ok(Some(false)),
            "null" => Ok(None),
            _ => Err(JsonError::BadValue(name)),
        }
    }
}

impl Reading {
    /// Reads the JSON object that [`to_json`](Reading::to_json) writes (no heap, no panic).
    ///
    /// The `model` is mapped to a model string of the registry
    /// [`ENCODERS`](super::ENCODERS) (an unknown one is [`JsonError::UnknownModel`]); `model`
    /// and `id` are required, every value field is optional (`null` is absent). The radio
    /// fields `freq_MHz`, `rssi_dB` and `repeats` and `mic` are read and ignored (the radio
    /// fields keep the values of [`Reading::new`], `mic` and the decimals of the temperature
    /// are those of the device), unknown fields of any type are skipped, `time` is kept.
    /// The result has the same content as the reading that was written
    /// ([`same_content`](Reading::same_content)), up to the printed resolution.
    ///
    /// ```
    /// use sdrk_proto::sensors::Reading;
    /// let r = Reading::from_json(
    ///     r#"{"model":"Bresser-3CH","id":66,"channel":1,"temperature_C":22.56,"humidity":66,"x":[1,{"y":2}]}"#,
    /// )?;
    /// assert_eq!((r.model, r.id, r.channel), ("Bresser-3CH", 66, Some(1)));
    /// assert_eq!((r.temperature_c, r.humidity, r.temp_decimals), (Some(22.56), Some(66.0), 2));
    /// assert!(Reading::from_json(r#"{"model":"nobody","id":1}"#).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// [`JsonError`] for text that is not an object, has no `model` or `id`, has a value of
    /// the wrong type, or names a model that has no encoder.
    pub fn from_json(text: &str) -> Result<Self, JsonError> {
        let mut p = Parser {
            b: text.as_bytes(),
            i: 0,
        };
        let mut r = Reading::new(
            "",
            0,
            Mic::Crc,
            &Header {
                modulation: Modulation::Ook,
                channel: 0,
                freq_offset_hz: 0,
                level_db10: 0,
                noise_db10: 0,
                rate_hz: 0,
            },
        );
        let (model, id) = p.object(&mut r)?;
        r.model = model.ok_or(JsonError::Missing("model"))?;
        r.id = id.ok_or(JsonError::Missing("id"))?;
        Ok(r)
    }
}

type Found = (Option<&'static str>, Option<u32>);

impl Parser<'_> {
    /// The whole object: the values go to `r`, the model and the id are returned.
    #[inline(never)]
    fn object(&mut self, r: &mut Reading) -> Result<Found, JsonError> {
        self.eat(b'{')?;
        let mut found: Found = (None, None);
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
        } else {
            loop {
                let key = self.string()?;
                self.eat(b':')?;
                self.field(key.as_str(), r, &mut found)?;
                self.ws();
                match self.peek() {
                    Some(b',') => self.i += 1,
                    Some(b'}') => {
                        self.i += 1;
                        break;
                    }
                    _ => return Err(JsonError::Syntax),
                }
            }
        }
        self.ws();
        if self.i != self.b.len() {
            return Err(JsonError::Syntax);
        }
        Ok(found)
    }

    /// One field: the value of `key` goes to `r` (or `found`), an unknown key is skipped.
    #[inline(never)]
    fn field(
        &mut self,
        key: Option<&str>,
        r: &mut Reading,
        found: &mut Found,
    ) -> Result<(), JsonError> {
        match key {
            Some("model") => {
                let t = self.string()?;
                let m = t.as_str().ok_or(JsonError::UnknownModel)?;
                let enc = encoder_by_model(m).ok_or(JsonError::UnknownModel)?;
                found.0 = enc.models().iter().copied().find(|&x| x == m);
                r.mic = enc.mic();
                r.temp_decimals = enc.temp_decimals();
            }
            Some("time") => r.time_s = self.number("time")?,
            Some("id") => found.1 = self.integer("id")?,
            Some("channel") => {
                r.channel = match self.integer("channel")? {
                    Some(c) => Some(u8::try_from(c).map_err(|_| JsonError::BadValue("channel"))?),
                    None => None,
                }
            }
            Some("battery_ok") => r.battery_ok = self.boolean("battery_ok")?,
            Some("temperature_C") => r.temperature_c = self.float("temperature_C")?,
            Some("humidity") => r.humidity = self.float("humidity")?,
            Some("wind_avg_km_h") => r.wind_avg_km_h = self.float("wind_avg_km_h")?,
            Some("wind_max_km_h") => r.wind_max_km_h = self.float("wind_max_km_h")?,
            Some("wind_dir_deg") => r.wind_dir_deg = self.float("wind_dir_deg")?,
            Some("rain_mm") => r.rain_mm = self.float("rain_mm")?,
            Some("pressure_hPa") => r.pressure_hpa = self.float("pressure_hPa")?,
            _ => self.skip(0)?,
        }
        Ok(())
    }
}
