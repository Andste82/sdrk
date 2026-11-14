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
//!
//! [`parse`] reads the 56-bit message field (ME) of an extended squitter: the aircraft
//! identification (type codes 1 to 4), the airborne position (9 to 18 with the barometric
//! altitude, 20 to 22 with the GNSS height) and the airborne velocity (19). The position holds
//! the 17-bit CPR coordinates; [`cpr_global`] turns an even and an odd one into a latitude and
//! longitude, [`cpr_local`] one of them and a reference position within about 180 NM. Surface
//! positions (5 to 8), status and the other type codes are reported as [`Body::NotDecoded`].
//!
//! ```
//! use sdrk_proto::adsb::{self, Body};
//!
//! let frame = [0x8D, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0, 0x57, 0x60, 0x98];
//! let msg = adsb::parse(&frame).unwrap();
//! let Body::Identification(id) = msg.body else { panic!() };
//! assert_eq!(id.callsign_str(), "KLM1023");
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

/// The kinds of message [`parse`] reads.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Body {
    /// Aircraft identification (type codes 1 to 4).
    Identification(Identification),
    /// Airborne position (type codes 9 to 18 and 20 to 22).
    Position(Position),
    /// Airborne velocity (type code 19, subtypes 1 to 4).
    Velocity(Velocity),
    /// A valid frame with another type code (surface position, status, ...): the type code.
    NotDecoded(u8),
}

/// A decoded extended squitter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Message {
    /// The downlink format, 17 or 18.
    pub df: u8,
    /// The 24-bit ICAO aircraft address.
    pub icao: u32,
    /// The type code, the first five bits of the message field.
    pub type_code: u8,
    /// The content.
    pub body: Body,
}

/// Aircraft identification: the callsign and the emitter category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identification {
    /// The category as a letter and the digit of the 3-bit category field: `A` for type code
    /// 4, `B` for 3, `C` for 2, `D` for 1 (`A3` is a large aircraft).
    pub category: (u8, u8),
    /// The 8 characters of the callsign as ASCII; a space for the code 32 and `#` for a code
    /// with no character.
    pub callsign: [u8; 8],
}

impl Identification {
    /// The callsign without trailing spaces.
    pub fn callsign_str(&self) -> &str {
        let n = self
            .callsign
            .iter()
            .rposition(|&c| c != b' ')
            .map_or(0, |p| p + 1);
        self.callsign
            .get(..n)
            .and_then(|b| core::str::from_utf8(b).ok())
            .unwrap_or("")
    }
}

/// An airborne position message: the altitude and the CPR coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// Surveillance status, 0 to 3.
    pub surveillance_status: u8,
    /// True for the GNSS height (type codes 20 to 22, in metres), false for the barometric
    /// altitude (type codes 9 to 18, in feet).
    pub gnss: bool,
    /// The altitude in feet (metres when `gnss`); `None` when the field is 0 or an invalid
    /// Gillham code.
    pub altitude: Option<i32>,
    /// True for an odd CPR frame, false for an even one.
    pub odd: bool,
    /// The 17-bit CPR latitude.
    pub cpr_lat: u32,
    /// The 17-bit CPR longitude.
    pub cpr_lon: u32,
}

/// Airborne velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Velocity {
    /// The subtype, 1 to 4 (2 and 4 are supersonic: speeds in steps of 4 kt).
    pub subtype: u8,
    /// The horizontal velocity.
    pub horizontal: Horizontal,
    /// The vertical rate in ft/min (positive up); `None` when not available.
    pub vertical_rate: Option<i32>,
    /// True when the vertical rate comes from the barometric, false from the GNSS source.
    pub vertical_rate_baro: bool,
    /// GNSS height minus barometric altitude in feet (3150 in magnitude means 3137.5 or more);
    /// `None` when not available.
    pub gnss_baro_diff: Option<i32>,
}

/// The horizontal part of [`Velocity`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Horizontal {
    /// Ground speed (subtypes 1 and 2): the components in knots; `None` when not available.
    Ground {
        /// East (positive) or west velocity.
        east_kt: Option<i32>,
        /// North (positive) or south velocity.
        north_kt: Option<i32>,
    },
    /// Heading and airspeed (subtypes 3 and 4).
    Air {
        /// The magnetic heading in degrees, `None` when not available.
        heading_deg: Option<f64>,
        /// The airspeed in knots, `None` when not available.
        airspeed_kt: Option<i32>,
        /// True for the true airspeed, false for the indicated airspeed.
        true_airspeed: bool,
    },
}

impl Horizontal {
    /// The ground speed in knots when both components are known.
    pub fn ground_speed_kt(&self) -> Option<f64> {
        match *self {
            Self::Ground {
                east_kt: Some(e),
                north_kt: Some(n),
            } => Some(libm::sqrt(
                f64::from(e) * f64::from(e) + f64::from(n) * f64::from(n),
            )),
            _ => None,
        }
    }

    /// The track angle in degrees from north (0 up to 360) when both components are known.
    pub fn track_deg(&self) -> Option<f64> {
        match *self {
            Self::Ground {
                east_kt: Some(e),
                north_kt: Some(n),
            } => {
                let t = libm::atan2(f64::from(e), f64::from(n)).to_degrees();
                Some(if t < 0.0 { t + 360.0 } else { t })
            }
            _ => None,
        }
    }
}

/// The 6-bit characters of the callsign.
const CHARSET: &[u8; 64] = b"#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######";

/// The `len` bits that start at the 1-based bit `from` of the 56-bit message field `me`.
fn bits(me: u64, from: u32, len: u32) -> u32 {
    ((me >> (57 - from - len)) & ((1 << len) - 1)) as u32
}

/// Parses the extended squitter `frame` (DF 17, or DF 18 with control field 0 or 6, both with
/// an ICAO address): `None` when it is not 14 bytes, has another format or a wrong CRC-24.
///
/// ```
/// use sdrk_proto::adsb::{self, Body};
/// let frame = [0x8D, 0x48, 0x50, 0x20, 0x99, 0x44, 0x09, 0x94, 0x08, 0x38, 0x17, 0x5B, 0x28, 0x4F];
/// let msg = adsb::parse(&frame).unwrap();
/// let Body::Velocity(v) = msg.body else { panic!() };
/// assert_eq!(v.vertical_rate, Some(-832));
/// assert_eq!(v.horizontal.ground_speed_kt().unwrap().round(), 159.0);
/// assert!(adsb::parse(&frame[..13]).is_none());
/// ```
pub fn parse(frame: &[u8]) -> Option<Message> {
    let frame: &[u8; LONG_FRAME_BYTES] = frame.try_into().ok()?;
    let format = df(frame);
    let cf = frame[0] & 7;
    if !(format == 17 || (format == 18 && (cf == 0 || cf == 6))) || !crc_ok(frame) {
        return None;
    }
    let mut me = [0u8; 8];
    me[1..].copy_from_slice(&frame[4..11]);
    let me = u64::from_be_bytes(me);
    let tc = bits(me, 1, 5) as u8;
    let body = match tc {
        1..=4 => Body::Identification(identification(me, tc)),
        9..=18 | 20..=22 => Body::Position(position(me, tc >= 20)),
        19 => velocity(me).map_or(Body::NotDecoded(tc), Body::Velocity),
        _ => Body::NotDecoded(tc),
    };
    Some(Message {
        df: format,
        icao: icao(frame),
        type_code: tc,
        body,
    })
}

fn identification(me: u64, tc: u8) -> Identification {
    let mut callsign = [b' '; 8];
    for (i, c) in callsign.iter_mut().enumerate() {
        *c = CHARSET[bits(me, 9 + 6 * i as u32, 6) as usize];
    }
    Identification {
        category: (b'A' + (4 - tc), bits(me, 6, 3) as u8),
        callsign,
    }
}

fn position(me: u64, gnss: bool) -> Position {
    let field = bits(me, 9, 12);
    let altitude = if gnss {
        (field != 0).then_some(field as i32)
    } else {
        altitude_baro(field)
    };
    Position {
        surveillance_status: bits(me, 6, 2) as u8,
        gnss,
        altitude,
        odd: bits(me, 22, 1) == 1,
        cpr_lat: bits(me, 23, 17),
        cpr_lon: bits(me, 40, 17),
    }
}

/// The barometric altitude in feet of the 12-bit field: 25 ft steps with the Q bit (the 8th),
/// else the 100 ft steps of the Gillham code; `None` for 0 and an invalid code.
fn altitude_baro(field: u32) -> Option<i32> {
    if field == 0 {
        return None;
    }
    if field & 0x10 != 0 {
        let n = ((field & 0xFE0) >> 1) | (field & 0xF);
        return Some(n as i32 * 25 - 1000);
    }
    gillham(field)
}

/// The Gillham (Mode C) altitude in feet of the 12 bits `C1 A1 C2 A2 C4 A4 B1 Q B2 D2 B4 D4`.
fn gillham(field: u32) -> Option<i32> {
    let b = |i: u32| (field >> (11 - i)) & 1;
    let (c1, a1, c2, a2, c4, a4) = (b(0), b(1), b(2), b(3), b(4), b(5));
    let (b1, b2, d2, b4, d4) = (b(6), b(8), b(9), b(10), b(11));
    // 500 ft steps: the Gray code D2 D4 A1 A2 A4 B1 B2 B4 (D1 is 0 here).
    let mut g = d2 << 7 | d4 << 6 | a1 << 5 | a2 << 4 | a4 << 3 | b1 << 2 | b2 << 1 | b4;
    let mut n = 0;
    while g != 0 {
        n ^= g;
        g >>= 1;
    }
    // 100 ft steps: C1 C2 C4 runs 001, 011, 010, 110, 100.
    let mut c = match c1 << 2 | c2 << 1 | c4 {
        1 => 1,
        3 => 2,
        2 => 3,
        6 => 4,
        4 => 5,
        _ => return None,
    };
    if n & 1 == 1 {
        c = 6 - c;
    }
    Some((n as i32 * 5 + c) * 100 - 1300)
}

fn velocity(me: u64) -> Option<Velocity> {
    let subtype = bits(me, 6, 3) as u8;
    let sign = |s: u32, v: i32| if s == 1 { -v } else { v };
    // Speeds are the field minus 1 (0: no information), times 4 for the supersonic subtypes.
    let speed = |v: u32| (v != 0).then(|| (v as i32 - 1) * if subtype & 1 == 0 { 4 } else { 1 });
    let horizontal = match subtype {
        1 | 2 => Horizontal::Ground {
            east_kt: speed(bits(me, 15, 10)).map(|v| sign(bits(me, 14, 1), v)),
            north_kt: speed(bits(me, 26, 10)).map(|v| sign(bits(me, 25, 1), v)),
        },
        3 | 4 => Horizontal::Air {
            heading_deg: (bits(me, 14, 1) == 1)
                .then(|| f64::from(bits(me, 15, 10)) * 360.0 / 1024.0),
            airspeed_kt: speed(bits(me, 26, 10)),
            true_airspeed: bits(me, 25, 1) == 1,
        },
        _ => return None,
    };
    let vr = bits(me, 38, 9) as i32;
    let dif = bits(me, 50, 7) as i32;
    Some(Velocity {
        subtype,
        horizontal,
        vertical_rate: (vr != 0).then(|| sign(bits(me, 37, 1), (vr - 1) * 64)),
        vertical_rate_baro: bits(me, 36, 1) == 1,
        gnss_baro_diff: (dif != 0).then(|| sign(bits(me, 49, 1), (dif - 1) * 25)),
    })
}

/// The number of longitude zones `NL(lat)` (1 to 59) of the CPR coding with 15 latitude zones.
///
/// ```
/// assert_eq!(sdrk_proto::adsb::nl(0.0), 59);
/// assert_eq!(sdrk_proto::adsb::nl(87.0), 2);
/// assert_eq!(sdrk_proto::adsb::nl(-88.0), 1);
/// ```
pub fn nl(lat: f64) -> u8 {
    let lat = libm::fabs(lat);
    if lat >= 87.0 {
        return if lat > 87.0 { 1 } else { 2 };
    }
    if lat == 0.0 {
        return 59;
    }
    let a = 1.0 - libm::cos(core::f64::consts::PI / 30.0);
    let c = libm::cos(lat.to_radians());
    let n = libm::floor(2.0 * core::f64::consts::PI / libm::acos(1.0 - a / (c * c)));
    n.clamp(1.0, 59.0) as u8
}

const CPR_SCALE: f64 = 131_072.0;

fn modulo(x: f64, y: f64) -> f64 {
    x - y * libm::floor(x / y)
}

/// Local CPR decoding: the latitude and longitude in degrees of one frame (`odd` its format,
/// the 17-bit `lat` and `lon`) that lies within about 180 NM of the reference position
/// (`ref_lat`, `ref_lon`). `None` for CPR values of 2^17 or more.
///
/// ```
/// let (lat, lon) = sdrk_proto::adsb::cpr_local(93000, 51372, false, 52.258, 3.918).unwrap();
/// assert!((lat - 52.2572).abs() < 1e-4 && (lon - 3.91937).abs() < 1e-4);
/// ```
pub fn cpr_local(lat: u32, lon: u32, odd: bool, ref_lat: f64, ref_lon: f64) -> Option<(f64, f64)> {
    if lat >= 1 << 17 || lon >= 1 << 17 {
        return None;
    }
    let (la, lo) = (f64::from(lat) / CPR_SCALE, f64::from(lon) / CPR_SCALE);
    let i = f64::from(u8::from(odd));
    let dlat = 360.0 / (60.0 - i);
    let j = libm::floor(ref_lat / dlat) + libm::floor(modulo(ref_lat, dlat) / dlat - la + 0.5);
    let rlat = dlat * (j + la);
    let dlon = 360.0 / (f64::from(nl(rlat)) - i).max(1.0);
    let m = libm::floor(ref_lon / dlon) + libm::floor(modulo(ref_lon, dlon) / dlon - lo + 0.5);
    let mut rlon = dlon * (m + lo);
    if rlon >= 180.0 {
        rlon -= 360.0;
    } else if rlon < -180.0 {
        rlon += 360.0;
    }
    Some((rlat, rlon))
}

/// Global CPR decoding from an even and an odd frame, each as (17-bit latitude, 17-bit
/// longitude); `odd_latest` tells which of them came last (its zone gives the result). `None`
/// when the two latitudes lie in different longitude zone counts (the aircraft crossed a zone
/// border between the frames), a latitude is out of range or a value is 2^17 or more.
///
/// ```
/// let (lat, lon) =
///     sdrk_proto::adsb::cpr_global((93000, 51372), (74158, 50194), false).unwrap();
/// assert!((lat - 52.2572).abs() < 1e-4 && (lon - 3.91937).abs() < 1e-4);
/// ```
pub fn cpr_global(even: (u32, u32), odd: (u32, u32), odd_latest: bool) -> Option<(f64, f64)> {
    if even.0 >= 1 << 17 || even.1 >= 1 << 17 || odd.0 >= 1 << 17 || odd.1 >= 1 << 17 {
        return None;
    }
    let f = |v: u32| f64::from(v) / CPR_SCALE;
    let (la0, lo0, la1, lo1) = (f(even.0), f(even.1), f(odd.0), f(odd.1));
    let j = libm::floor(59.0 * la0 - 60.0 * la1 + 0.5);
    let fix = |l: f64| if l >= 270.0 { l - 360.0 } else { l };
    let lat_e = fix(360.0 / 60.0 * (modulo(j, 60.0) + la0));
    let lat_o = fix(360.0 / 59.0 * (modulo(j, 59.0) + la1));
    if !(-90.0..=90.0).contains(&lat_e) || !(-90.0..=90.0).contains(&lat_o) {
        return None;
    }
    if nl(lat_e) != nl(lat_o) {
        return None;
    }
    let nlf = f64::from(nl(lat_e));
    let m = libm::floor(lo0 * (nlf - 1.0) - lo1 * nlf + 0.5);
    let (lat, ni, lo) = if odd_latest {
        (lat_o, (nlf - 1.0).max(1.0), lo1)
    } else {
        (lat_e, nlf.max(1.0), lo0)
    };
    let mut lon = 360.0 / ni * (modulo(m, ni) + lo);
    if lon >= 180.0 {
        lon -= 360.0;
    }
    Some((lat, lon))
}
