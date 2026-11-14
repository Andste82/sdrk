//! The block `adsb_parse`: messages as JSON, the aircraft table and the CPR pairing.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unreadable_literal,
    clippy::format_collect,
    clippy::cast_possible_truncation,
    clippy::precedence
)]

use sdrk_blocks::{
    ADSB_PARSE_AIRCRAFT, AdsbParseBlock, AdsbParseConfig, ArenaStorage, Block, Packet,
};
use sdrk_proto::PacketFlags;
use sdrk_proto::crc::crc24_adsb;
use sdrk_testkit::VecStorage;

const IDENT: &str = "8D4840D6202CC371C32CE0576098";
const EVEN: &str = "8D40621D58C382D690C8AC2863A7";
const ODD: &str = "8D40621D58C386435CC412692AD6";
const VEL: &str = "8D485020994409940838175B284F";
const MHZ2: u64 = 2_000_000;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn pkt(s: &str, index: u64) -> Packet {
    Packet::new(index, PacketFlags::CRC_OK, &hex(s)).unwrap()
}

/// Runs the packets through a block and returns the JSON lines.
fn run(cfg: &AdsbParseConfig, input: &[Packet]) -> Vec<String> {
    let mut b = AdsbParseBlock::new(cfg, &mut VecStorage).unwrap();
    let mut out = vec![Packet::default(); input.len()];
    let p = b.process(input, &mut out).unwrap();
    assert_eq!(p.consumed, input.len());
    out[..p.produced]
        .iter()
        .map(|p| String::from_utf8(p.data().to_vec()).unwrap())
        .collect()
}

#[test]
fn identification_position_and_velocity_lines() {
    let cfg = AdsbParseConfig::default();
    let lines = run(
        &cfg,
        &[
            pkt(IDENT, MHZ2),
            pkt(EVEN, 2 * MHZ2),
            pkt(ODD, 2 * MHZ2 + 100_000),
            pkt(VEL, 3 * MHZ2),
        ],
    );
    assert_eq!(lines.len(), 4);
    assert_eq!(
        lines[0],
        r#"{"time":1.000,"hex":"4840d6","icao":4735190,"df":17,"tc":4,"flight":"KLM1023","category":"A0"}"#
    );
    // The even frame alone has no position without a reference.
    assert_eq!(
        lines[1],
        r#"{"time":2.000,"hex":"40621d","icao":4219421,"df":17,"tc":11,"alt_baro":38000}"#
    );
    // With the odd frame the pair decodes (odd latest: the odd latitude and its zone).
    assert!(
        lines[2].contains(r#""alt_baro":38000,"lat":52.26578,"lon":3.9"#),
        "{}",
        lines[2]
    );
    assert_eq!(
        lines[3],
        r#"{"time":3.000,"hex":"485020","icao":4739104,"df":17,"tc":19,"gs":159.2,"track":182.9,"geom_rate":-832,"geom_delta":550}"#
    );
    assert!(lines.iter().all(|l| l.len() < 256));
}

#[test]
fn local_decoding_with_a_reference() {
    let cfg = AdsbParseConfig {
        ref_lat: 52.258,
        ref_lon: 3.918,
        ..AdsbParseConfig::default()
    };
    let lines = run(&cfg, &[pkt(EVEN, 0)]);
    assert!(
        lines[0].ends_with(r#""lat":52.25720,"lon":3.91937}"#),
        "{}",
        lines[0]
    );
    // The pair wins over the reference: the odd frame gives the odd latitude.
    let lines = run(&cfg, &[pkt(EVEN, 0), pkt(ODD, 1000)]);
    assert!(lines[1].contains(r#""lat":52.26578"#), "{}", lines[1]);
}

#[test]
fn the_pair_must_be_young_enough() {
    let cfg = AdsbParseConfig {
        max_age_s: 1.0,
        ..AdsbParseConfig::default()
    };
    let lines = run(&cfg, &[pkt(EVEN, 0), pkt(ODD, MHZ2 + 1)]);
    assert!(!lines[1].contains("lat"), "{}", lines[1]);
    let lines = run(&cfg, &[pkt(EVEN, 0), pkt(ODD, MHZ2)]);
    assert!(lines[1].contains("lat"), "{}", lines[1]);
}

#[test]
fn other_packets_give_no_line() {
    let mut damaged = hex(IDENT);
    damaged[6] ^= 1;
    let short = [0x5D, 0x48, 0x40, 0xD6, 0, 0, 0];
    // Surface position (type code 5) with a correct parity.
    let mut surface = [0u8; 14];
    surface[0] = 0x8D;
    surface[4] = 27 << 3;
    let p = crc24_adsb(&surface[..11]);
    surface[11..].copy_from_slice(&p.to_be_bytes()[1..]);
    let input = [
        Packet::new(0, PacketFlags::NONE, &damaged).unwrap(),
        Packet::new(1, PacketFlags::NONE, &short).unwrap(),
        Packet::new(2, PacketFlags::CRC_OK, &surface).unwrap(),
    ];
    let mut b = AdsbParseBlock::new(&AdsbParseConfig::default(), &mut VecStorage).unwrap();
    let mut out = [Packet::default(); 3];
    let p = b.process(&input, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (3, 0));
    assert_eq!((b.ignored(), b.undecoded(), b.parsed()), (2, 1, 0));
}

#[test]
fn the_table_replaces_the_oldest_aircraft() {
    // The even frame of aircraft 0 is forgotten after 64 other aircraft were seen.
    let cfg = AdsbParseConfig::default();
    let mut input = vec![pkt(EVEN, 0)];
    for k in 0..ADSB_PARSE_AIRCRAFT as u32 {
        let mut f = hex(EVEN);
        f[1..4].copy_from_slice(&(0x0010_0000 + k).to_be_bytes()[1..]);
        let p = crc24_adsb(&f[..11]);
        f[11..].copy_from_slice(&p.to_be_bytes()[1..]);
        input.push(Packet::new(10 + u64::from(k), PacketFlags::CRC_OK, &f).unwrap());
    }
    input.push(pkt(ODD, 1000));
    let lines = run(&cfg, &input);
    assert!(!lines.last().unwrap().contains("lat"));
    // Without the others the pair is found.
    assert!(run(&cfg, &[pkt(EVEN, 0), pkt(ODD, 1000)])[1].contains("lat"));
}

#[test]
fn output_room_and_reset() {
    let cfg = AdsbParseConfig::default();
    let mut mem = vec![0u8; AdsbParseBlock::<ArenaStorage<'_>>::storage_bytes(&cfg).unwrap() + 16];
    let mut arena = ArenaStorage::new(&mut mem);
    let mut b = AdsbParseBlock::new(&cfg, &mut arena).unwrap();
    let input = [pkt(EVEN, 0), pkt(ODD, 10), pkt(IDENT, 20)];
    let mut out = [Packet::default(); 2];
    let p = b.process(&input, &mut out).unwrap();
    assert_eq!((p.consumed, p.produced), (2, 2));
    assert_eq!(b.positions(), 1);
    b.reset();
    let mut out = [Packet::default(); 1];
    b.process(&input[1..2], &mut out).unwrap();
    // After the reset the even frame is forgotten.
    assert!(!core::str::from_utf8(out[0].data()).unwrap().contains("lat"));
    assert_eq!(b.parsed(), 1);
    assert!(
        AdsbParseBlock::new(
            &AdsbParseConfig {
                max_age_s: 0.0,
                ..cfg
            },
            &mut VecStorage
        )
        .is_err()
    );
}

const SURF_EVEN: &str = "8C4841753AAB238733C8CD4020B1";
const SURF_ODD: &str = "8C4841753A8A35323FAEBDAC702D";

fn mkframe(first: u8, icao: u32, me: u64) -> String {
    let mut f = [0u8; 14];
    f[0] = first;
    f[1..4].copy_from_slice(&icao.to_be_bytes()[1..]);
    f[4..11].copy_from_slice(&me.to_be_bytes()[1..]);
    let p = crc24_adsb(&f[..11]);
    f[11..].copy_from_slice(&p.to_be_bytes()[1..]);
    f.iter().map(|b| format!("{b:02X}")).collect()
}

#[test]
fn surface_positions_need_the_reference() {
    let none = run(
        &AdsbParseConfig::default(),
        &[pkt(SURF_EVEN, 0), pkt(SURF_ODD, 1000)],
    );
    assert_eq!(none.len(), 2);
    assert!(
        none[1].contains(r#""ground":true,"gs":16,"track":98.4"#),
        "{}",
        none[1]
    );
    assert!(!none[1].contains("lat"), "{}", none[1]);
    let cfg = AdsbParseConfig {
        ref_lat: 51.99,
        ref_lon: 4.375,
        ..AdsbParseConfig::default()
    };
    let lines = run(&cfg, &[pkt(SURF_EVEN, 0), pkt(SURF_ODD, 1000)]);
    // The pair (the odd frame latest) gives the book's 52.320607, 4.734735.
    assert!(
        lines[1].contains(r#""lat":52.32061,"lon":4.7347"#),
        "{}",
        lines[1]
    );
    // The first frame alone is decoded locally; a surface frame does not pair with an airborne
    // one of the same aircraft.
    assert!(lines[0].contains("\"lat\":52.3"), "{}", lines[0]);
    let mixed = run(&cfg, &[pkt(EVEN, 0), pkt(SURF_ODD, 10)]);
    assert!(mixed[1].contains("ground"), "{}", mixed[1]);
}

#[test]
fn other_messages_and_non_icao_addresses() {
    let cfg = AdsbParseConfig::default();
    // DF 18 with the control field 1: a non-ICAO address; the same number as an ICAO address
    // is another aircraft in the table.
    let id = (4u64 << 51) | 0x0820_C30C_30C3;
    let lines = run(
        &cfg,
        &[
            pkt(&mkframe(0x91, 0x123456, id), 0),
            pkt(&mkframe(0x8D, 0x123456, id), 1),
            pkt(&mkframe(0x92, 0x123456, id), 2),
        ],
    );
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(
        lines[0].contains(r#""df":18,"tc":4,"non_icao":true,"#),
        "{}",
        lines[0]
    );
    assert!(!lines[1].contains("non_icao"));
    // The aircraft status of the general emergency: squawk 7700 (A 7, B 7, C 0, D 0).
    let a = 7u64;
    let b = 7u64;
    let id13 = ((a & 1) << 11)
        | ((a >> 1 & 1) << 9)
        | ((a >> 2 & 1) << 7)
        | ((b & 1) << 5)
        | ((b >> 1 & 1) << 3)
        | ((b >> 2 & 1) << 1);
    let me = (28u64 << 51) | (1 << 48) | (1 << 45) | (id13 << 32);
    let lines = run(&cfg, &[pkt(&mkframe(0x8D, 0xABCDEF, me), 0)]);
    assert!(
        lines[0].ends_with(r#""emergency":"general","squawk":"7700"}"#),
        "{}",
        lines[0]
    );
    // Target state with every field and the longest list of modes stays under 256 bytes and
    // closes the object.
    let me = (29u64 << 51)
        | (1 << 49)
        | (1 << 47)
        | (1400 << 36)
        | (500 << 27)
        | (1 << 26)
        | (1 << 25)
        | (255 << 17)
        | (15 << 13)
        | (1 << 12)
        | (3 << 10)
        | (1 << 9 | 1 << 8 | 1 << 7 | 1 << 6 | 1 << 4 | 1 << 3 | 1 << 2);
    let lines = run(
        &cfg,
        &[pkt(&mkframe(0x95, 0xFFFFFF, me), 4_000_000_000_000)],
    );
    let l = &lines[0];
    assert!(l.len() < 256 && l.ends_with('}'), "{} {l}", l.len());
    assert!(
        l.contains(r#""nav_modes":["autopilot","vnav","althold","approach","lnav","tcas"]"#),
        "{l}"
    );
    // The operational status and the not decoded type codes.
    let me = (31u64 << 51) | (2 << 13) | (9 << 8) | (3 << 4);
    let lines = run(
        &cfg,
        &[
            pkt(&mkframe(0x8D, 1, me), 0),
            pkt(&mkframe(0x8D, 1, 27 << 51), 1),
        ],
    );
    assert_eq!(lines.len(), 1);
    assert!(
        lines[0].ends_with(r#""version":2,"nac_p":9,"sil":3,"nic_a":0,"gva":0}"#),
        "{}",
        lines[0]
    );
}
