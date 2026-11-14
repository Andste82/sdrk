//! The block `adsb_parse`: messages as JSON, the aircraft table and the CPR pairing.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
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
    surface[4] = 5 << 3;
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
        f[1..4].copy_from_slice(&(0x100000 + k).to_be_bytes()[1..]);
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
