//! Packet files in the JSON Lines format: one packet per line.
//!
//! A line is a JSON object with exactly the fields `index` (sample index, an unsigned 64-bit
//! integer), `flags` (the bits of [`PacketFlags`], an unsigned 32-bit integer) and `data` (the
//! payload as lower-case hex without spaces, at most [`MAX_PACKET_BYTES`] bytes):
//!
//! ```text
//! {"index":123,"flags":1,"data":"1acf"}
//! ```
//!
//! The writer emits the fields in this order with LF line ends. The reader skips empty lines
//! (and lines of only white space); every other line must be such an object. An unknown or
//! missing field, a value of the wrong type or range, hex that is not lower case or has an odd
//! number of digits, or a payload longer than [`MAX_PACKET_BYTES`] bytes is a
//! [`Error::Parse`] that names the line number (counted from 1).

use std::io::{BufRead, Write};

use sdrk_proto::{MAX_PACKET_BYTES, Packet, PacketFlags};
use serde_json::Value;

use crate::error::{Error, Result};

const HEX: &[u8; 16] = b"0123456789abcdef";

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

/// A packet from its index, flag bits and payload as lower-case hex (the representation of the
/// JSON Lines format and of the WebAssembly binding). Does not allocate.
///
/// ```
/// use sdrk_io::packets::packet_from_hex;
/// use sdrk_proto::PacketFlags;
/// let p = packet_from_hex(7, 1, "1acf")?;
/// assert_eq!((p.index, p.flags, p.data()), (7, PacketFlags::CRC_OK, &[0x1a, 0xcf][..]));
/// assert!(packet_from_hex(0, 0, "1ACF").is_err()); // upper case
/// assert!(packet_from_hex(0, 0, "1ac").is_err()); // odd number of digits
/// assert!(packet_from_hex(0, 0, &"00".repeat(257)).is_err()); // 257 bytes
/// # Ok::<(), sdrk_io::Error>(())
/// ```
///
/// # Errors
/// [`Error::InvalidArgument`] for a character other than `0-9a-f`, an odd number of digits or
/// more than [`MAX_PACKET_BYTES`] bytes.
pub fn packet_from_hex(index: u64, flags: u32, hex: &str) -> Result<Packet> {
    let digits = hex.as_bytes();
    if !digits.len().is_multiple_of(2) {
        return Err(Error::invalid(format!(
            "data has an odd number of hex digits ({})",
            digits.len()
        )));
    }
    let len = digits.len() / 2;
    if len > MAX_PACKET_BYTES {
        return Err(Error::invalid(format!(
            "data has {len} bytes, at most {MAX_PACKET_BYTES} are allowed"
        )));
    }
    let mut bytes = [0u8; MAX_PACKET_BYTES];
    for (dst, pair) in bytes.iter_mut().zip(digits.chunks_exact(2)) {
        let (hi, lo) = match pair {
            [hi, lo] => (hex_digit(*hi), hex_digit(*lo)),
            _ => (None, None),
        };
        let (Some(hi), Some(lo)) = (hi, lo) else {
            return Err(Error::invalid(
                "data must be lower-case hex digits (0-9, a-f) without spaces",
            ));
        };
        *dst = (hi << 4) | lo;
    }
    let data = bytes.get(..len).unwrap_or_default();
    Packet::new(index, PacketFlags::from_bits(flags), data)
        .map_err(|e| Error::invalid(e.to_string()))
}

/// Writes `packets` as JSON Lines, one line per packet.
///
/// ```
/// use sdrk_proto::{Packet, PacketFlags};
/// let packets = [Packet::new(123, PacketFlags::CRC_OK, &[0x1a, 0xcf])?];
/// let mut out = Vec::new();
/// sdrk_io::packets::write_jsonl(&mut out, &packets)?;
/// assert_eq!(out, b"{\"index\":123,\"flags\":1,\"data\":\"1acf\"}\n");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
/// [`Error::Io`] if writing fails.
pub fn write_jsonl<W: Write>(mut w: W, packets: &[Packet]) -> Result<()> {
    for p in packets {
        writeln!(
            w,
            "{{\"index\":{},\"flags\":{},\"data\":\"{}\"}}",
            p.index,
            p.flags.bits(),
            to_hex(p.data())
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
/// assert_eq!((packets[0].index, packets[0].data()), (123, &[0x1a, 0xcf][..]));
/// let bad = sdrk_io::packets::read_jsonl("{\"index\":1,\"flags\":0,\"data\":\"\",\"x\":1}".as_bytes());
/// assert_eq!(bad.unwrap_err().to_string(), "cannot parse line 1: unknown field `x`");
/// # Ok::<(), sdrk_io::Error>(())
/// ```
///
/// # Errors
/// [`Error::Parse`] naming the line (`line <n>`) for a malformed line (see the module
/// documentation); [`Error::Io`] if reading fails or the input is not UTF-8.
pub fn read_jsonl<R: BufRead>(r: R) -> Result<Vec<Packet>> {
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

fn parse_line(line: &str) -> core::result::Result<Packet, String> {
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
    packet_from_hex(index, flags, data).map_err(|e| match e {
        Error::InvalidArgument(reason) => reason,
        other => other.to_string(),
    })
}
