//! The flow `adsb_receiver` (M39.1 §5) and test 5.
//!
//! - The flow validates without warnings, builds and has the parameters of §5; a rate other
//!   than 2 MHz is `rate_unsupported`.
//! - **Test 5:** the golden case `flows/adsb_df17_200_frames` (seed 544: 200 DF 17 frames at
//!   random positions, 10 to 30 dB above the noise, with overlapping noise bursts) through the
//!   flow, with the criterion of the maintainer (D708): at least 99 % of the frames whose
//!   effective SNR (D696: the strongest overlapping burst added to the noise) is above 10 dB
//!   come out with `CRC_OK`, their 14 bytes and the index of their preamble, every frame at an
//!   effective SNR of 13 dB or more comes out, and no other `CRC_OK` frame and no
//!   address/parity packet comes out (every frame of the case is a DF 17 frame, so such a
//!   packet would come from noise). The literal count (the drawn level) is printed too. The case `flows/adsb_short_frames` (seed 545)
//!   gives its 6 DF 17 frames and its DF 11 frame with residue 0 with `CRC_OK`, its 20 short
//!   frames whose address was seen before as 7 bytes with flag bit 16, and drops its 4 short
//!   frames of unknown addresses (D702, D704).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::too_many_lines
)]

mod common;

use sdrk_pipeline::{BuildOptions, InputData, Packet, PacketFlags, Pipeline, Registry};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use serde_json::Value;

use common::{build, flow};

/// Flag bit 16 of an address/parity frame (`sdrk_blocks::ADSB_ADDRESS_PARITY`).
const ADDRESS_PARITY: u32 = 1 << 16;
/// Samples per push (`max_chunk` of the flow).
const CHUNK: usize = 4096;

#[test]
fn the_flow_validates_without_warnings() {
    let report =
        Registry::standard().validate_json(&flow("adsb_receiver"), &BuildOptions::default());
    assert!(report.is_ok(), "{}", report.to_json());
    assert!(report.warnings.is_empty(), "{}", report.to_json());
    assert!(report.memory_bytes > 0);
    build(&flow("adsb_receiver"));
}

#[test]
fn the_flow_has_the_parameters_of_the_spec() {
    let spec: Value = serde_json::from_str(&flow("adsb_receiver")).unwrap();
    assert_eq!(spec["inputs"]["rf"]["format"], "cu8");
    assert_eq!(spec["inputs"]["rf"]["rate_hz"], 2_000_000);
    assert_eq!(spec["nodes"]["dec"]["kind"], "adsb_decode");
    assert_eq!(spec["outputs"]["packets"]["type"], "packets");
    // Any other rate is `rate_unsupported`.
    let other = flow("adsb_receiver").replace("2000000", "2400000");
    let report = Registry::standard().validate_json(&other, &BuildOptions::default());
    assert!(!report.is_ok());
    assert!(
        report.to_json().contains("rate_unsupported"),
        "{}",
        report.to_json()
    );
}

/// The rf bytes of a case through the flow in pushes of `CHUNK` samples.
fn run(case: &GoldenCase) -> Vec<Packet> {
    let Array::U8(rf) = case.array("rf").unwrap() else {
        panic!("rf is u8")
    };
    let mut p = build(&flow("adsb_receiver"));
    let mut got = Vec::new();
    for piece in rf.chunks(2 * CHUNK) {
        p.push("rf", InputData::Raw(piece)).unwrap();
        read_packets(&mut p, &mut got);
    }
    got
}

fn read_packets(p: &mut Pipeline, into: &mut Vec<Packet>) {
    let mut buf = [Packet::default(); 64];
    loop {
        let n = p.read_packets("packets", &mut buf).unwrap();
        if n == 0 {
            return;
        }
        into.extend_from_slice(&buf[..n]);
    }
}

/// The bytes of a hex string.
fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// The downlink format (the first 5 bits) and the ICAO address (bytes 1 to 3) of a frame, as
/// `examples/README.md` prints them.
fn df_icao(data: &[u8]) -> (u8, u32) {
    (
        data[0] >> 3,
        u32::from(data[1]) << 16 | u32::from(data[2]) << 8 | u32::from(data[3]),
    )
}

/// Effective SNR in dB from which every frame of test 5 must decode (D708): two-sample PPM
/// without error correction has a bit error rate of about `0.5 exp(-SNR / 2)`, so a 112-bit
/// frame is lost with about 1 % from 13 dB on.
const ALL_FROM_DB: f64 = 13.0;
/// Smallest share of the frames above `min_snr_db` that must decode (D708).
const MIN_SHARE: f64 = 0.99;

/// Test 5 as measured.
struct Test5 {
    /// Frames with an effective SNR above `min_snr_db`.
    required: usize,
    /// Of them, the frames missed as (index, effective SNR, level).
    missed: Vec<(u64, f64, f64)>,
    /// `CRC_OK` packets that are not one of the frames at its index.
    false_frames: Vec<u64>,
    /// Address/parity packets (flag bit 16) as indices: all from noise.
    address_parity: Vec<u64>,
}

fn test5() -> Test5 {
    let case = golden::load("flows", "adsb_df17_200_frames").unwrap();
    let min_snr = case.param_f64("min_snr_db").unwrap();
    let frames = case.params["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 200);
    let got = run(&case);

    let ok: Vec<&Packet> = got
        .iter()
        .filter(|p| p.flags == PacketFlags::CRC_OK)
        .collect();
    let short_at: Vec<u64> = got
        .iter()
        .filter(|p| p.flags.0 & ADDRESS_PARITY != 0)
        .map(|p| p.index)
        .collect();
    let short = short_at.len();
    let mut strong = 0;
    let mut required = 0;
    let mut missed = Vec::new();
    let mut decoded_below = 0;
    // The literal reading of §5: the level above the noise without the bursts.
    let (mut literal, mut literal_missed, mut literal_missed_burst) = (0, 0, 0);
    for f in frames {
        let index = f["index"].as_u64().unwrap();
        let bytes = unhex(f["hex"].as_str().unwrap());
        let snr = f["snr_db"].as_f64().unwrap();
        let level = f["level_db"].as_f64().unwrap();
        let found = ok.iter().any(|p| p.index == index && p.data() == bytes);
        if level > min_snr {
            literal += 1;
            if !found {
                literal_missed += 1;
                if snr < level {
                    literal_missed_burst += 1;
                }
            }
        }
        if snr >= ALL_FROM_DB {
            strong += 1;
        }
        if snr > min_snr {
            required += 1;
            if !found {
                missed.push((index, snr, f["level_db"].as_f64().unwrap()));
            }
        } else if found {
            decoded_below += 1;
        }
        if found {
            let (df, icao) = df_icao(&bytes);
            assert_eq!(df, 17);
            assert_eq!(u64::from(icao), f["icao"].as_u64().unwrap());
        }
    }
    // No false frames: every CRC_OK packet is one of the frames, at its index.
    let false_frames: Vec<u64> = ok
        .iter()
        .filter(|p| {
            !frames.iter().any(|f| {
                f["index"].as_u64().unwrap() == p.index
                    && unhex(f["hex"].as_str().unwrap()) == p.data()
            })
        })
        .map(|p| p.index)
        .collect();
    eprintln!(
        "ADS-B test 5: {} frames, {required} above {min_snr} dB effective SNR, {} of them \
         decoded (missed: {missed:?} as (index, effective SNR dB, level dB)); {strong} at \
         {ALL_FROM_DB} dB or more, {} of them decoded; {decoded_below} of \
         the {} frames at or below {min_snr} dB decoded too; {} CRC_OK packets, {} false; {short} \
         address/parity packets (flag bit 16) from noise at {short_at:?}; \
         literal reading (level above {min_snr} dB without the bursts): {literal} frames, {} \
         decoded, {literal_missed} missed ({literal_missed_burst} of them under a burst)",
        frames.len(),
        required - missed.len(),
        strong - missed.iter().filter(|m| m.1 >= ALL_FROM_DB).count(),
        frames.len() - required,
        ok.len(),
        false_frames.len(),
        literal - literal_missed,
    );
    Test5 {
        required,
        missed,
        false_frames,
        address_parity: short_at,
    }
}

/// Test 5 with the criterion of D708: at least 99 % of the frames above 10 dB effective SNR
/// decode (184 of 185 measured; the one missed, 129916 at 10.08 dB, has two bit errors, which
/// only error correction could repair), every frame at 13 dB or more decodes, and noise gives
/// neither a false `CRC_OK` frame nor an address/parity packet.
#[test]
fn test5_frames_above_10_db_decode_and_no_false_frames() {
    let t = test5();
    let decoded = t.required - t.missed.len();
    eprintln!(
        "ADS-B test 5 criterion (D708): {decoded} of {} frames above 10 dB decoded (at least \
         {:.0} % required), {} missed at {ALL_FROM_DB} dB or more, {} false CRC_OK frames, {} \
         address/parity packets from noise",
        t.required,
        100.0 * MIN_SHARE,
        t.missed.iter().filter(|m| m.1 >= ALL_FROM_DB).count(),
        t.false_frames.len(),
        t.address_parity.len(),
    );
    assert!(t.required > 0);
    assert!(
        decoded as f64 >= MIN_SHARE * t.required as f64,
        "{decoded} of {} decoded, missed {:?}",
        t.required,
        t.missed
    );
    let strong_missed: Vec<_> = t.missed.iter().filter(|m| m.1 >= ALL_FROM_DB).collect();
    assert!(
        strong_missed.is_empty(),
        "missed at 13 dB or more: {strong_missed:?}"
    );
    assert!(
        t.false_frames.is_empty(),
        "false frames at {:?}",
        t.false_frames
    );
    assert!(
        t.address_parity.is_empty(),
        "address/parity packets from noise at {:?}",
        t.address_parity
    );
}

#[test]
fn test5_short_frames_come_out_with_flag_bit_16() {
    let case = golden::load("flows", "adsb_short_frames").unwrap();
    let frames = case.params["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 31);
    let got = run(&case);
    // (index, bytes, flags) of every frame the case expects out, in order.
    let want: Vec<(u64, Vec<u8>, u32)> = frames
        .iter()
        .filter_map(|f| {
            let flags = match f["out"].as_str().unwrap() {
                "crc_ok" => PacketFlags::CRC_OK.0,
                "address_parity" => ADDRESS_PARITY,
                "dropped" => return None,
                other => panic!("out {other}"),
            };
            Some((
                f["index"].as_u64().unwrap(),
                unhex(f["hex"].as_str().unwrap()),
                flags,
            ))
        })
        .collect();
    let count = |out: &str| frames.iter().filter(|f| f["out"] == out).count();
    assert_eq!(
        (count("crc_ok"), count("address_parity"), count("dropped")),
        (7, 20, 4)
    );
    let have: Vec<(u64, Vec<u8>, u32)> = got
        .iter()
        .map(|p| (p.index, p.data().to_vec(), p.flags.0))
        .collect();
    eprintln!(
        "ADS-B short frames: {} frames sent, {} expected out (7 checked, 20 with a known \
         address), {} packets out",
        frames.len(),
        want.len(),
        got.len()
    );
    assert_eq!(have, want);
    for p in got.iter().filter(|p| p.flags.0 == ADDRESS_PARITY) {
        assert_eq!(p.data().len(), 7);
        assert!([4, 5, 11].contains(&df_icao(p.data()).0));
    }
}
