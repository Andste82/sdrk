//! Packet files in the JSON Lines format: one packet per line.
//!
//! A line is a JSON object with exactly the fields `index` (sample index, an unsigned 64-bit
//! integer), `flags` (the flag bits of the packet, an unsigned 32-bit integer) and `data` (the
//! payload as lower-case hex without spaces, at most [`MAX_DATA_BYTES`] bytes):
//!
//! ```text
//! {"index":123,"flags":1,"data":"1acf"}
//! ```
//!
//! The writer emits the fields in this order with LF line ends. The reader skips empty lines
//! (and lines of only white space); every other line must be such an object. An unknown or
//! missing field, a value of the wrong type or range, hex that is not lower case or has an odd
//! number of digits, or a payload longer than [`MAX_DATA_BYTES`] bytes is a
//! [`Error::Parse`] that names the line number (counted from 1).
//!
//! The module works on its own record type [`PacketRecord`], so `sdrk-io` stays on the core
//! types (layer rule R2): the packet type `Packet` of `sdrk-proto` is converted by the caller,
//! for example `Packet::new(r.index, PacketFlags::from_bits(r.flags), &r.data)`.

use std::io::{BufRead, Write};

use serde_json::Value;

use crate::error::{Error, Result};

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Largest payload of one line in bytes; equal to `MAX_PACKET_BYTES` of `sdrk-proto`.
pub const MAX_DATA_BYTES: usize = 256;

/// One packet of a JSON Lines file: the fields of a line.
///
/// ```
/// use sdrk_io::packets::PacketRecord;
/// let r = PacketRecord::new(123, 1, vec![0x1a, 0xcf])?;
/// assert_eq!((r.index(), r.flags(), r.data()), (123, 1, &[0x1a, 0xcf][..]));
/// assert!(PacketRecord::new(0, 0, vec![0; 257]).is_err());
/// # Ok::<(), sdrk_io::Error>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PacketRecord {
    index: u64,
    flags: u32,
    data: Vec<u8>,
}

impl PacketRecord {
    /// A record from its sample index, flag bits and payload.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] if `data` is longer than [`MAX_DATA_BYTES`].
    pub fn new(index: u64, flags: u32, data: Vec<u8>) -> Result<Self> {
        if data.len() > MAX_DATA_BYTES {
            return Err(too_long(data.len()));
        }
        Ok(Self { index, flags, data })
    }

    /// The sample index.
    #[must_use]
    pub fn index(&self) -> u64 {
        self.index
    }

    /// The flag bits.
    #[must_use]
    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// The payload.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

fn too_long(len: usize) -> Error {
    Error::invalid(format!(
        "data has {len} bytes, at most {MAX_DATA_BYTES} are allowed"
    ))
}

/// The payload `data` as lower-case hex, two digits per byte.
///
/// ```
/// assert_eq!(sdrk_io::packets::to_hex(&[0x1a, 0xcf, 0x00]), "1acf00");
/// assert_eq!(sdrk_io::packets::to_hex(&[]), "");
/// ```
#[must_use]
pub fn to_hex(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for &b in data {
        for nibble in [b >> 4, b & 0x0f] {
            if let Some(&c) = HEX.get(usize::from(nibble)) {
                s.push(char::from(c));
            }
        }
    }
    s
}

fn hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

/// The bytes of a payload in lower-case hex (the representation of the JSON Lines format and
/// of the WebAssembly binding).
///
/// ```
/// use sdrk_io::packets::from_hex;
/// assert_eq!(from_hex("1acf")?, [0x1a, 0xcf]);
/// assert!(from_hex("1ACF").is_err()); // upper case
/// assert!(from_hex("1ac").is_err()); // odd number of digits
/// assert!(from_hex(&"00".repeat(257)).is_err()); // 257 bytes
/// # Ok::<(), sdrk_io::Error>(())
/// ```
///
/// # Errors
/// [`Error::InvalidArgument`] for a character other than `0-9a-f`, an odd number of digits or
/// more than [`MAX_DATA_BYTES`] bytes.
pub fn from_hex(hex: &str) -> Result<Vec<u8>> {
    let digits = hex.as_bytes();
    if !digits.len().is_multiple_of(2) {
        return Err(Error::invalid(format!(
            "data has an odd number of hex digits ({})",
            digits.len()
        )));
    }
    let len = digits.len() / 2;
    if len > MAX_DATA_BYTES {
        return Err(too_long(len));
    }
    digits
        .chunks_exact(2)
        .map(|pair| match pair {
            [hi, lo] => hex_digit(*hi)
                .zip(hex_digit(*lo))
                .map(|(hi, lo)| (hi << 4) | lo),
            _ => None,
        })
        .collect::<Option<Vec<u8>>>()
        .ok_or_else(|| {
            Error::invalid("data must be lower-case hex digits (0-9, a-f) without spaces")
        })
}

/// Writes `packets` as JSON Lines, one line per packet.
///
/// ```
/// use sdrk_io::packets::PacketRecord;
/// let packets = [PacketRecord::new(123, 1, vec![0x1a, 0xcf])?];
/// let mut out = Vec::new();
/// sdrk_io::packets::write_jsonl(&mut out, &packets)?;
/// assert_eq!(out, b"{\"index\":123,\"flags\":1,\"data\":\"1acf\"}\n");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
/// [`Error::Io`] if writing fails.
pub fn write_jsonl<W: Write>(mut w: W, packets: &[PacketRecord]) -> Result<()> {
    for p in packets {
        writeln!(
            w,
            "{{\"index\":{},\"flags\":{},\"data\":\"{}\"}}",
            p.index,
            p.flags,
            to_hex(&p.data)
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Reads every packet of a JSON Lines input.
///
/// ```
/// let text = "{\"index\":123,\"flags\":1,\"data\":\"1acf\"}\n\n{\"index\":9,\"flags\":0,\"data\":\"\"}\n";
/// let packets = sdrk_io::packets::read_jsonl(text.as_bytes())?;
/// assert_eq!(packets.len(), 2);
/// assert_eq!((packets[0].index(), packets[0].data()), (123, &[0x1a, 0xcf][..]));
/// let bad = sdrk_io::packets::read_jsonl("{\"index\":1,\"flags\":0,\"data\":\"\",\"x\":1}".as_bytes());
/// assert_eq!(bad.unwrap_err().to_string(), "cannot parse line 1: unknown field `x`");
/// # Ok::<(), sdrk_io::Error>(())
/// ```
///
/// # Errors
/// [`Error::Parse`] naming the line (`line <n>`) for a malformed line (see the module
/// documentation); [`Error::Io`] if reading fails or the input is not UTF-8.
pub fn read_jsonl<R: BufRead>(r: R) -> Result<Vec<PacketRecord>> {
    let mut packets = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let n = i + 1;
        packets
            .push(parse_line(&line).map_err(|reason| Error::parse(format!("line {n}"), reason))?);
    }
    Ok(packets)
}

fn parse_line(line: &str) -> core::result::Result<PacketRecord, String> {
    let value: Value = serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let Value::Object(fields) = value else {
        return Err("a line must be a JSON object".to_string());
    };
    if let Some(unknown) = fields
        .keys()
        .find(|k| !matches!(k.as_str(), "index" | "flags" | "data"))
    {
        return Err(format!("unknown field `{unknown}`"));
    }
    let field = |name: &str| {
        fields
            .get(name)
            .ok_or_else(|| format!("missing field `{name}`"))
    };
    let index = field("index")?
        .as_u64()
        .ok_or("`index` must be an unsigned 64-bit integer")?;
    let flags = field("flags")?
        .as_u64()
        .and_then(|f| u32::try_from(f).ok())
        .ok_or("`flags` must be an unsigned 32-bit integer")?;
    let data = field("data")?.as_str().ok_or("`data` must be a string")?;
    let data = from_hex(data).map_err(|e| match e {
        Error::InvalidArgument(reason) => reason,
        other => other.to_string(),
    })?;
    Ok(PacketRecord { index, flags, data })
}
