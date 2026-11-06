//! Fields and parity of Mode S downlink frames (ADS-B).
//!
//! A Mode S frame has 56 bits (7 bytes, downlink formats below 16) or 112 bits (14 bytes,
//! formats 16 and above), sent most significant bit first. The first five bits are the
//! downlink format (DF); in the all-call reply (DF 11) and the extended squitter (DF 17, and
//! DF 18 of non-transponder devices) the next three bits are the capability and bytes 1 to 3
//! the 24-bit ICAO aircraft address. The last 24 bits are the parity: for DF 17 and 18 the
//! [`crc24_adsb`] of the message, so the CRC over the whole frame is 0 when it has no error.
//! In the other formats the parity is combined with the address (or an interrogator code)
//! and cannot be checked without it.
//!
//! ```
//! use sdrk_proto::adsb;
//!
//! let frame = [0x8D, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0, 0x57, 0x60, 0x98];
//! assert!(adsb::crc_ok(&frame));
//! assert_eq!(adsb::df(&frame), 17);
//! assert_eq!(adsb::icao(&frame), 0x48_40D6);
//! ```

use crate::crc::crc24_adsb;

/// The bytes of a long (112-bit) frame.
pub const LONG_FRAME_BYTES: usize = 14;

/// The bytes of a short (56-bit) frame.
pub const SHORT_FRAME_BYTES: usize = 7;

/// Whether the CRC-24 residue of the long frame `frame` (message and parity) is 0, the parity
/// check of DF 17 and DF 18.
///
/// ```
/// let mut frame = [0x8D, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0, 0x57, 0x60, 0x98];
/// assert!(sdrk_proto::adsb::crc_ok(&frame));
/// frame[5] ^= 0x10;
/// assert!(!sdrk_proto::adsb::crc_ok(&frame));
/// ```
pub fn crc_ok(frame: &[u8; LONG_FRAME_BYTES]) -> bool {
    crc24_adsb(frame) == 0
}

/// The downlink format of `frame`: its first five bits (0 for an empty slice).
///
/// ```
/// assert_eq!(sdrk_proto::adsb::df(&[0x8D]), 17);
/// assert_eq!(sdrk_proto::adsb::df(&[0x5D, 0, 0, 0, 0, 0, 0]), 11);
/// ```
pub fn df(frame: &[u8]) -> u8 {
    frame.first().map_or(0, |b| b >> 3)
}

/// The 24-bit address in bytes 1 to 3 of `frame` (the ICAO aircraft address of DF 11, 17 and
/// 18); 0 when the slice has fewer than 4 bytes.
///
/// ```
/// assert_eq!(sdrk_proto::adsb::icao(&[0x8D, 0x48, 0x40, 0xD6, 0x20]), 0x48_40D6);
/// assert_eq!(sdrk_proto::adsb::icao(&[0x8D, 0x48]), 0);
/// ```
pub fn icao(frame: &[u8]) -> u32 {
    match frame.get(1..4) {
        Some(&[a, b, c]) => u32::from_be_bytes([0, a, b, c]),
        _ => 0,
    }
}
