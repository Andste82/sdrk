//! The message field of the extended squitter: `adsb::parse`, the CPR decoding (M39.1 extension).
//!
//! The vectors are the frames printed in "The 1090 MHz Riddle" (J. Sun, mode-s.org) with the
//! values the book gives; the CPR grid test encodes with a small encoder written from the
//! description of the coding.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unreadable_literal,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use sdrk_proto::adsb::{self, Body, Horizontal};
use sdrk_proto::crc::crc24_adsb;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

const IDENT: &str = "8D4840D6202CC371C32CE0576098";
const EVEN: &str = "8D40621D58C382D690C8AC2863A7";
const ODD: &str = "8D40621D58C386435CC412692AD6";
const VEL1: &str = "8D485020994409940838175B284F";
const VEL3: &str = "8DA05F219B06B6AF189400CBC33F";

/// A frame with the given first byte, ICAO address and message field, with its parity.
fn frame(first: u8, icao: u32, me: u64) -> [u8; 14] {
    let mut f = [0u8; 14];
    f[0] = first;
    f[1..4].copy_from_slice(&icao.to_be_bytes()[1..]);
    f[4..11].copy_from_slice(&me.to_be_bytes()[1..]);
    let p = crc24_adsb(&f[..11]);
    f[11..].copy_from_slice(&p.to_be_bytes()[1..]);
    assert!(adsb::crc_ok(&f));
    f
}

#[test]
fn identification_of_the_book() {
    let m = adsb::parse(&hex(IDENT)).unwrap();
    assert_eq!((m.df, m.icao, m.type_code), (17, 0x4840D6, 4));
    let Body::Identification(id) = m.body else {
        panic!("{m:?}")
    };
    assert_eq!(&id.callsign, b"KLM1023 ");
    assert_eq!(id.callsign_str(), "KLM1023");
    assert_eq!(id.category, (b'A', 0));
}

#[test]
fn position_of_the_book() {
    let even = adsb::parse(&hex(EVEN)).unwrap();
    let odd = adsb::parse(&hex(ODD)).unwrap();
    let (Body::Position(e), Body::Position(o)) = (even.body, odd.body) else {
        panic!()
    };
    assert_eq!(even.icao, 0x40621D);
    assert_eq!(even.type_code, 11);
    assert_eq!((e.altitude, e.gnss, e.odd), (Some(38000), false, false));
    assert_eq!((e.cpr_lat, e.cpr_lon), (93000, 51372));
    assert_eq!((o.altitude, o.odd), (Some(38000), true));
    assert_eq!((o.cpr_lat, o.cpr_lon), (74158, 50194));
    // The book: j = 8, both NL 36, latitude 52.2572 and longitude 3.91937 (even frame latest).
    let (lat, lon) = adsb::cpr_global((93000, 51372), (74158, 50194), false).unwrap();
    assert!((lat - 52.2572021484375).abs() < 1e-12, "{lat}");
    assert!((lon - 3.91937255859375).abs() < 1e-12, "{lon}");
    assert_eq!(adsb::nl(lat), 36);
    // The odd frame as the latest one gives a point within the resolution.
    let (lat_o, lon_o) = adsb::cpr_global((93000, 51372), (74158, 50194), true).unwrap();
    assert!((lat_o - 52.26578017412606).abs() < 1e-12, "{lat_o}");
    assert!((lon_o - lon).abs() < 0.05, "{lon_o}");
    // The book's local example: reference 52.258, 3.918, the even frame.
    let (lat, lon) = adsb::cpr_local(93000, 51372, false, 52.258, 3.918).unwrap();
    assert!((lat - 52.2572021484375).abs() < 1e-12 && (lon - 3.91937255859375).abs() < 1e-12);
    assert!(adsb::cpr_local(1 << 17, 0, false, 0.0, 0.0).is_none());
    assert!(adsb::cpr_global((0, 1 << 17), (0, 0), false).is_none());
}

#[test]
fn velocity_of_the_book() {
    let m = adsb::parse(&hex(VEL1)).unwrap();
    let Body::Velocity(v) = m.body else { panic!() };
    assert_eq!((m.type_code, v.subtype), (19, 1));
    assert_eq!(v.vertical_rate, Some(-832));
    assert!(!v.vertical_rate_baro); // bit 36 is 0: the GNSS source
    // Sign bit 0: GNSS above barometric (pyModeS 3.7.0 agrees; the book text says "below").
    assert_eq!(v.gnss_baro_diff, Some(550));
    let g = v.horizontal.ground_speed_kt().unwrap();
    let t = v.horizontal.track_deg().unwrap();
    assert!((g - 159.2).abs() < 0.01, "{g}");
    assert!((t - 182.88).abs() < 0.01, "{t}");

    let m = adsb::parse(&hex(VEL3)).unwrap();
    let Body::Velocity(v) = m.body else { panic!() };
    assert_eq!(v.subtype, 3);
    assert_eq!(v.vertical_rate, Some(-2304));
    assert_eq!(v.gnss_baro_diff, None);
    let Horizontal::Air {
        heading_deg,
        airspeed_kt,
        true_airspeed,
    } = v.horizontal
    else {
        panic!()
    };
    assert!((heading_deg.unwrap() - 243.98).abs() < 0.01);
    assert_eq!((airspeed_kt, true_airspeed), (Some(375), true));
}

#[test]
fn velocity_corner_cases() {
    // Subtype 2 (supersonic): east 11 -> 40 kt west, north 3 -> 8 kt south; no vertical rate.
    let me = (19u64 << 51) | (2 << 48) | (1 << 42) | (11 << 32) | (1 << 31) | (3 << 21);
    let Body::Velocity(v) = adsb::parse(&frame(0x8D, 1, me)).unwrap().body else {
        panic!()
    };
    assert_eq!(
        v.horizontal,
        Horizontal::Ground {
            east_kt: Some(-40),
            north_kt: Some(-8)
        }
    );
    assert_eq!((v.vertical_rate, v.gnss_baro_diff), (None, None));
    // Components 0 mean no information.
    let Body::Velocity(v) = adsb::parse(&frame(0x8D, 1, (19u64 << 51) | (1 << 48)))
        .unwrap()
        .body
    else {
        panic!()
    };
    assert_eq!(v.horizontal.ground_speed_kt(), None);
    // Subtype 5 is not decoded.
    let m = adsb::parse(&frame(0x8D, 1, (19u64 << 51) | (5 << 48))).unwrap();
    assert_eq!(m.body, Body::NotDecoded(19));
}

/// The Gillham field of an altitude (100 ft steps) from the code table.
fn gillham_field(alt: i32) -> u32 {
    let n = (alt + 1300) / 100 - 1;
    let (n500, c) = ((n / 5) as u32, (n % 5) as u32 + 1);
    let c = if n500 & 1 == 1 { 6 - c } else { c };
    let cbits = [0, 0b001, 0b011, 0b010, 0b110, 0b100][c as usize];
    let g = n500 ^ (n500 >> 1); // D2 D4 A1 A2 A4 B1 B2 B4
    let (d2, d4, a1, a2, a4, b1, b2, b4) = (
        g >> 7 & 1,
        g >> 6 & 1,
        g >> 5 & 1,
        g >> 4 & 1,
        g >> 3 & 1,
        g >> 2 & 1,
        g >> 1 & 1,
        g & 1,
    );
    let (c1, c2, c4) = (cbits >> 2 & 1, cbits >> 1 & 1, cbits & 1);
    // C1 A1 C2 A2 C4 A4 B1 Q B2 D2 B4 D4 with Q = 0
    c1 << 11
        | a1 << 10
        | c2 << 9
        | a2 << 8
        | c4 << 7
        | a4 << 6
        | b1 << 5
        | b2 << 3
        | d2 << 2
        | b4 << 1
        | d4
}

fn position_frame(tc: u64, alt_field: u32, odd: bool, lat: u32, lon: u32) -> [u8; 14] {
    let me = (tc << 51)
        | (u64::from(alt_field) << 36)
        | (u64::from(odd) << 34)
        | (u64::from(lat) << 17)
        | u64::from(lon);
    frame(0x8D, 0xABCDEF, me)
}

#[test]
fn altitude_q_bit_gillham_and_gnss() {
    let alt = |tc, field| {
        let Body::Position(p) = adsb::parse(&position_frame(tc, field, false, 0, 0))
            .unwrap()
            .body
        else {
            panic!()
        };
        p.altitude
    };
    // Q = 1: 25 ft steps from -1000 ft.
    assert_eq!(alt(11, 0b0000_0001_0000), Some(-1000));
    assert_eq!(alt(11, 0b1100_0011_1000), Some(38000));
    assert_eq!(alt(11, 0), None);
    // Gillham: every step of 100 ft from -1200 to 126700 ft, and one bit changes per step.
    let mut last: Option<u32> = None;
    for a in (-1200..=126_700).step_by(100) {
        let f = gillham_field(a);
        assert_eq!(alt(12, f), Some(a), "alt {a} field {f:012b}");
        if let Some(l) = last {
            assert_eq!((f ^ l).count_ones(), 1, "alt {a}");
        }
        last = Some(f);
    }
    // Invalid C bits (000, 101, 111).
    assert_eq!(alt(12, 0b0100_0000_0000), None);
    // GNSS height in metres.
    let Body::Position(p) = adsb::parse(&position_frame(21, 1234, true, 5, 6))
        .unwrap()
        .body
    else {
        panic!()
    };
    assert_eq!(
        (p.gnss, p.altitude, p.odd, p.cpr_lat, p.cpr_lon),
        (true, Some(1234), true, 5, 6)
    );
}

#[test]
fn rejected_and_undecoded_frames() {
    let good = hex(IDENT);
    for len in 0..14 {
        assert!(adsb::parse(&good[..len]).is_none());
    }
    assert!(adsb::parse(&[good.clone(), vec![0]].concat()).is_none());
    for bit in 0..112 {
        let mut f = good.clone();
        f[bit / 8] ^= 0x80 >> (bit % 8);
        assert!(adsb::parse(&f).is_none(), "bit {bit}");
    }
    // Other formats with a valid parity.
    assert!(adsb::parse(&frame(0x80, 1, 0)).is_none());
    for cf in [2u8, 3, 4, 7] {
        assert!(
            adsb::parse(&frame(0x90 | cf, 1, 4u64 << 51)).is_none(),
            "cf {cf}"
        );
    }
    // DF 18: the control fields 1 and 5 carry a non-ICAO address, 0 and 6 an ICAO address.
    for (cf, non_icao) in [(0u8, false), (1, true), (5, true), (6, false)] {
        let m = adsb::parse(&frame(0x90 | cf, 0x123456, 4u64 << 51)).unwrap();
        assert_eq!(
            (m.df, m.non_icao, m.icao),
            (18, non_icao, 0x123456),
            "cf {cf}"
        );
    }
    assert!(!adsb::parse(&frame(0x8D, 1, 4u64 << 51)).unwrap().non_icao);
    // Type codes and subtypes that are not decoded keep their type code.
    for me in [
        23u64 << 51,
        24 << 51,
        27 << 51,
        30 << 51,
        (28 << 51) | (2 << 48),
        (29 << 51) | (2 << 49),
        (31 << 51) | (2 << 48),
    ] {
        let m = adsb::parse(&frame(0x8D, 1, me)).unwrap();
        assert_eq!(m.body, Body::NotDecoded((me >> 51) as u8), "{me:x}");
    }
    // Type code 0: the altitude field without a position.
    let m = adsb::parse(&frame(0x8D, 1, 0xC38u64 << 36)).unwrap();
    assert_eq!(m.body, Body::AltitudeOnly(Some(38000)));
    assert_eq!(
        adsb::parse(&frame(0x8D, 1, 0)).unwrap().body,
        Body::AltitudeOnly(None)
    );
}

#[test]
fn random_frames_never_panic() {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut decoded = 0;
    for _ in 0..20_000 {
        let mut f = [0u8; 14];
        for b in &mut f {
            *b = next() as u8;
        }
        // Mostly with a correct parity, so that the fields are reached.
        if next() % 4 != 0 {
            f[0] = 0x88 | ((next() % 2) as u8 * 8) | (next() % 8) as u8;
            let p = crc24_adsb(&f[..11]);
            f[11..].copy_from_slice(&p.to_be_bytes()[1..]);
        }
        let n = (next() % 20) as usize;
        let _ = adsb::parse(&f[..n.min(14)]);
        decoded += usize::from(adsb::parse(&f).is_some());
        let _ = adsb::cpr_global(
            (next() as u32 & 0x1FFFF, next() as u32 & 0x1FFFF),
            (next() as u32 & 0x1FFFF, next() as u32 & 0x1FFFF),
            next() % 2 == 0,
        );
        let _ = adsb::cpr_local(
            next() as u32,
            next() as u32,
            false,
            (next() % 400) as f64 - 200.0,
            f64::NAN,
        );
    }
    assert!(decoded > 5000, "{decoded}");
}

/// The CPR coordinates of a position for the format `odd`.
fn cpr_encode(lat: f64, lon: f64, odd: bool) -> (u32, u32) {
    let scale = 131_072.0;
    let i = if odd { 1.0 } else { 0.0 };
    let dlat = 360.0 / (60.0 - i);
    let m = |x: f64, y: f64| x - y * (x / y).floor();
    let yz = (scale * m(lat, dlat) / dlat + 0.5).floor();
    let rlat = dlat * (yz / scale + (lat / dlat).floor());
    let dlon = 360.0 / (f64::from(adsb::nl(rlat)) - i).max(1.0);
    let xz = (scale * m(lon, dlon) / dlon + 0.5).floor();
    (yz as u32 & 0x1FFFF, xz as u32 & 0x1FFFF)
}

#[test]
fn nl_at_the_zone_borders() {
    assert_eq!(adsb::nl(0.0), 59);
    assert_eq!(adsb::nl(10.4704), 59);
    assert_eq!(adsb::nl(10.4705), 58);
    assert_eq!(adsb::nl(-10.4705), 58);
    assert_eq!(adsb::nl(86.9), 2);
    assert_eq!(adsb::nl(87.0), 2);
    assert_eq!(adsb::nl(87.1), 1);
    assert_eq!(adsb::nl(-87.1), 1);
    assert_eq!(adsb::nl(90.0), 1);
    let mut last = 59;
    for k in 0..=8700 {
        let n = adsb::nl(f64::from(k) / 100.0);
        assert!(n <= last && n >= 2);
        last = n;
    }
}

/// The difference of two longitudes in degrees, folded to -180 to 180.
fn dlon(a: f64, b: f64) -> f64 {
    let d = (a - b) % 360.0;
    if d > 180.0 {
        d - 360.0
    } else if d < -180.0 {
        d + 360.0
    } else {
        d
    }
}

#[test]
fn cpr_over_a_grid_of_positions() {
    let mut lats = vec![
        0.0, 1e-4, -1e-4, 86.9, 87.0, 87.0001, -86.9, -87.0, -87.0001, 89.9, -89.9,
    ];
    for b in [
        10.47047130,
        14.82817437,
        18.18626357,
        25.82924707,
        52.0,
        86.5,
    ] {
        lats.extend([b - 0.01, b - 1e-4, b, b + 1e-4, b + 0.01, -b, -b - 1e-4]);
    }
    let mut a = -88.0;
    while a <= 88.0 {
        lats.push(a);
        a += 1.37;
    }
    let lons = [
        -179.99, -170.3, -90.0, -1e-3, 0.0, 1e-3, 3.919, 45.5, 119.7, 179.99,
    ];
    let (mut global_none, mut total) = (0, 0);
    for &lat in &lats {
        for &lon in &lons {
            let e = cpr_encode(lat, lon, false);
            let o = cpr_encode(lat, lon, true);
            for odd_latest in [false, true] {
                total += 1;
                // The latitude steps are 6/2^17 and 360/59/2^17 degrees; the longitude step
                // is a zone width over 2^17, wide near the poles.
                let ni = (f64::from(adsb::nl(lat)) - f64::from(u8::from(odd_latest))).max(1.0);
                let tol_lon = 360.0 / ni / 131_072.0 * 1.01;
                match adsb::cpr_global(e, o, odd_latest) {
                    Some((la, lo)) => {
                        assert!(
                            (la - lat).abs() < 6.0 / 131_072.0 * 1.01,
                            "{lat} {lon}: {la}"
                        );
                        // Within a step of the grid, also when the longitude is 180 degrees away.
                        assert!(dlon(lo, lon).abs() <= tol_lon + 1e-9, "{lat} {lon}: {lo}");
                    }
                    None => global_none += 1,
                }
                let f = if odd_latest { o } else { e };
                let (la, lo) = adsb::cpr_local(f.0, f.1, odd_latest, lat + 0.9, lon - 0.9).unwrap();
                assert!(
                    (la - lat).abs() < 6.0 / 131_072.0 * 1.01,
                    "local {lat} {lon}: {la}"
                );
                assert!(
                    dlon(lo, lon).abs() <= tol_lon + 1e-9,
                    "local {lat} {lon}: {lo}"
                );
            }
        }
    }
    // Only the pairs that straddle a zone border (different NL) are refused.
    assert!(global_none * 20 < total, "{global_none} of {total}");
}

const SURF_EVEN: &str = "8C4841753AAB238733C8CD4020B1";
const SURF_ODD: &str = "8C4841753A8A35323FAEBDAC702D";
const SURF_LOCAL: &str = "8C4841753A9A153237AEF0F275BE";

#[test]
fn surface_position_of_the_book() {
    let m = adsb::parse(&hex(SURF_LOCAL)).unwrap();
    let Body::Surface(s) = m.body else { panic!() };
    // The book: MOV 41 = 17 kt, S = 1, TRK 33 = 92.8125 deg, F = 1.
    assert_eq!(s.speed_kt, Some(17.0));
    assert_eq!(s.track_deg, Some(92.8125));
    assert!(s.odd);
    let (lat, lon) = adsb::cpr_surface_local(s.cpr_lat, s.cpr_lon, s.odd, 51.99, 4.375).unwrap();
    assert!(
        // The hand calculation of the book for the single odd frame: 52.320561, 4.735735.
        (lat - 52.320561).abs() < 1e-5 && (lon - 4.735735).abs() < 1e-5,
        "{lat} {lon}"
    );

    let (Body::Surface(e), Body::Surface(o)) = (
        adsb::parse(&hex(SURF_EVEN)).unwrap().body,
        adsb::parse(&hex(SURF_ODD)).unwrap().body,
    ) else {
        panic!()
    };
    assert_eq!((e.cpr_lat, e.cpr_lon, e.odd), (115609, 116941, false));
    assert_eq!((o.cpr_lat, o.cpr_lon, o.odd), (39199, 110269, true));
    // The book: reference 51.990, 4.375 gives 52.320607, 4.734735 (the odd frame is the latest).
    let (lat, lon) = adsb::cpr_surface_global(
        (e.cpr_lat, e.cpr_lon),
        (o.cpr_lat, o.cpr_lon),
        true,
        51.99,
        4.375,
    )
    .unwrap();
    assert!(
        (lat - 52.320607).abs() < 1e-5 && (lon - 4.734735).abs() < 1e-5,
        "{lat} {lon}"
    );
    // Another quadrant of the reference selects the other solution: the longitude repeats every
    // 90 degrees and the latitude has a southern twin 90 degrees away.
    let (lat2, lon2) = adsb::cpr_surface_global(
        (e.cpr_lat, e.cpr_lon),
        (o.cpr_lat, o.cpr_lon),
        true,
        -38.0,
        94.4,
    )
    .unwrap();
    assert!((lat2 - (lat - 90.0)).abs() < 1e-9, "{lat2}");
    assert!(dlon(lon2, 94.4).abs() < 45.0, "{lon2}");
}

#[test]
fn surface_movement_and_track_fields() {
    let speed = |mov: u64, status: u64, trk: u64| {
        let me = (5u64 << 51) | (mov << 44) | (status << 43) | (trk << 36);
        let Body::Surface(s) = adsb::parse(&frame(0x8D, 1, me)).unwrap().body else {
            panic!()
        };
        (s.speed_kt, s.track_deg)
    };
    // The ends of every range of the table of the book.
    for (mov, kt) in [
        (1, 0.0),
        (2, 0.125),
        (8, 0.875),
        (9, 1.0),
        (12, 1.75),
        (13, 2.0),
        (38, 14.5),
        (39, 15.0),
        (93, 69.0),
        (94, 70.0),
        (108, 98.0),
        (109, 100.0),
        (123, 170.0),
        (124, 175.0),
    ] {
        assert_eq!(speed(mov, 0, 0).0, Some(kt), "mov {mov}");
    }
    for mov in [0, 125, 127] {
        assert_eq!(speed(mov, 0, 0).0, None);
    }
    assert_eq!(speed(10, 0, 64).1, None);
    assert_eq!(speed(10, 1, 64).1, Some(180.0));
    assert_eq!(speed(10, 1, 127).1, Some(127.0 * 2.8125));
}

/// The CPR coordinates of the surface coding (zone of 90 degrees).
fn surface_encode(lat: f64, lon: f64, odd: bool) -> (u32, u32) {
    let scale = 131_072.0;
    let i = if odd { 1.0 } else { 0.0 };
    let dlat = 90.0 / (60.0 - i);
    let m = |x: f64, y: f64| x - y * (x / y).floor();
    let yz = (scale * m(lat, dlat) / dlat + 0.5).floor();
    let rlat = dlat * (yz / scale + (lat / dlat).floor());
    let dlon = 90.0 / (f64::from(adsb::nl(rlat)) - i).max(1.0);
    let xz = (scale * m(lon, dlon) / dlon + 0.5).floor();
    (yz as u32 & 0x1FFFF, xz as u32 & 0x1FFFF)
}

#[test]
fn surface_cpr_over_a_grid_of_positions() {
    let mut lats = vec![
        0.0, 1e-4, -1e-4, 86.9, 87.0, 87.0001, -86.9, -87.0, -87.0001, 89.5,
    ];
    for b in [10.47047130, 25.82924707, 52.0, 86.5] {
        lats.extend([b - 1e-3, b, b + 1e-3, -b, -b - 1e-3]);
    }
    let mut a = -85.0;
    while a <= 85.0 {
        lats.push(a);
        a += 2.9;
    }
    let lons = [
        -179.9, -90.2, -45.0, -1e-3, 0.0, 1e-3, 4.7, 89.99, 90.01, 133.3, 179.9,
    ];
    let (mut none, mut total) = (0, 0);
    for &lat in &lats {
        for &lon in &lons {
            let (e, o) = (
                surface_encode(lat, lon, false),
                surface_encode(lat, lon, true),
            );
            for odd_latest in [false, true] {
                total += 1;
                let ni = (f64::from(adsb::nl(lat)) - f64::from(u8::from(odd_latest))).max(1.0);
                let (tol_lat, tol_lon) =
                    (90.0 / 59.0 / 131_072.0 * 1.01, 90.0 / ni / 131_072.0 * 1.01);
                // A reference 0.1 degrees away, within 45 NM.
                let (rlat, rlon) = (lat + 0.1, lon - 0.1);
                match adsb::cpr_surface_global(e, o, odd_latest, rlat, rlon) {
                    Some((la, lo)) => {
                        assert!((la - lat).abs() < tol_lat, "{lat} {lon}: {la}");
                        assert!(dlon(lo, lon).abs() <= tol_lon + 1e-9, "{lat} {lon}: {lo}");
                    }
                    None => none += 1,
                }
                let f = if odd_latest { o } else { e };
                let (la, lo) = adsb::cpr_surface_local(f.0, f.1, odd_latest, rlat, rlon).unwrap();
                assert!((la - lat).abs() < tol_lat, "local {lat} {lon}: {la}");
                assert!(
                    dlon(lo, lon).abs() <= tol_lon + 1e-9,
                    "local {lat} {lon}: {lo}"
                );
            }
        }
    }
    assert!(none * 10 < total, "{none} of {total}");
}

/// The 13 Mode A identity bits of a squawk written as four octal digits (C1 A1 C2 A2 C4 A4 X B1
/// D1 B2 D2 B4 D4, the order of the identity field).
fn mode_a_bits(squawk: u16) -> u64 {
    let (a, b, c, d) = (
        squawk / 1000,
        squawk / 100 % 10,
        squawk / 10 % 10,
        squawk % 10,
    );
    let bit = |v: u16, k: u16| u64::from(v >> k & 1);
    bit(c, 0) << 12
        | bit(a, 0) << 11
        | bit(c, 1) << 10
        | bit(a, 1) << 9
        | bit(c, 2) << 8
        | bit(a, 2) << 7
        | bit(b, 0) << 5
        | bit(d, 0) << 4
        | bit(b, 1) << 3
        | bit(d, 1) << 2
        | bit(b, 2) << 1
        | bit(d, 2)
}

#[test]
fn aircraft_status_emergency_and_squawk() {
    for (state, squawk, name) in [
        (0u64, 1200u16, "none"),
        (1, 7700, "general"),
        (4, 7600, "nordo"),
        (5, 7500, "unlawful"),
        (2, 0o0, "medical"),
        (7, 7777, "reserved"),
    ] {
        let squawk_dec = squawk;
        let me = (28u64 << 51) | (1 << 48) | (state << 45) | (mode_a_bits(squawk_dec) << 32);
        let Body::Status(s) = adsb::parse(&frame(0x8D, 1, me)).unwrap().body else {
            panic!()
        };
        assert_eq!(
            (s.emergency, s.squawk, s.emergency_name()),
            (state as u8, squawk, name)
        );
    }
}

#[test]
fn target_state_fields() {
    // Selected altitude 35008 ft from the FMS, raw baro code 267, heading 90 and 270 degrees,
    // NACp 9, NICbaro 1, SIL 3, autopilot, LNAV and TCAS (ME bit n is bit 56 - n of the word).
    let word = |alt: u64, qnh: u64, hs: u64, hsign: u64, hdg: u64, modes: u64| {
        (29u64 << 51)
            | (1 << 49)
            | (1 << 47) // selected altitude from the FMS
            | (alt << 36)
            | (qnh << 27)
            | (hs << 26)
            | (hsign << 25)
            | (hdg << 17)
            | (9 << 13) // NACp
            | (1 << 12) // NICbaro
            | (3 << 10) // SIL
            | modes
    };
    // modes: ME bits 47 (valid) 48 autopilot, 49 VNAV, 50 alt hold, 52 approach, 53 TCAS, 54 LNAV
    let modes = (1 << 9) | (1 << 8) | (1 << 3) | (1 << 2);
    let me = word(35008 / 32 + 1, 267, 1, 0, 128, modes);
    let Body::Target(t) = adsb::parse(&frame(0x8D, 1, me)).unwrap().body else {
        panic!("{me:x}")
    };
    assert!(t.altitude_fms);
    assert_eq!(t.selected_altitude, Some(35008));
    assert_eq!(t.baro_setting, Some(f64::from(267 - 1) * 0.8 + 800.0));
    assert_eq!(t.selected_heading, Some(90.0));
    assert_eq!((t.nac_p, t.nic_baro, t.sil), (9, true, 3));
    let m = t.modes.unwrap();
    assert!(m.autopilot && m.lnav && m.tcas && !m.vnav && !m.alt_hold && !m.approach);
    // A heading with the sign bit, absent values and invalid modes.
    let me = word(0, 0, 1, 1, 128, 0);
    let Body::Target(t) = adsb::parse(&frame(0x8D, 1, me)).unwrap().body else {
        panic!()
    };
    assert_eq!(t.selected_heading, Some(270.0));
    assert_eq!(
        (t.selected_altitude, t.baro_setting, t.modes),
        (None, None, None)
    );
    let me = word(1, 1, 0, 0, 200, 0);
    let Body::Target(t) = adsb::parse(&frame(0x8D, 1, me)).unwrap().body else {
        panic!()
    };
    assert_eq!(
        (t.selected_altitude, t.baro_setting, t.selected_heading),
        (Some(0), Some(800.0), None)
    );
}

#[test]
fn operational_status_fields() {
    // Version 2, NACp 10, SIL 3, GVA 2, NICa 1, HRD 1, SILs 1; capability and mode are raw bits.
    let me = |st: u64| {
        (31u64 << 51)
            | (st << 48)
            | (0xA5C3 << 32)
            | (0x1234 << 16)
            | (2 << 13)
            | (1 << 12)
            | (10 << 8)
            | (2 << 6)
            | (3 << 4)
            | (1 << 3)
            | (1 << 2)
            | (1 << 1)
    };
    let Body::Operational(o) = adsb::parse(&frame(0x8D, 1, me(0))).unwrap().body else {
        panic!()
    };
    assert!(o.airborne);
    assert_eq!((o.capability, o.mode, o.version), (0xA5C3, 0x1234, 2));
    assert_eq!(
        (o.nic_supplement, o.nac_p, o.gva, o.sil),
        (true, 10, Some(2), 3)
    );
    assert!(o.baro_or_track_bit && o.hrd && o.sil_supplement);
    let Body::Operational(o) = adsb::parse(&frame(0x8D, 1, me(1))).unwrap().body else {
        panic!()
    };
    assert!(!o.airborne);
    assert_eq!(o.gva, None);
}
