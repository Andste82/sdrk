//! The CRCs of `sdrk-proto`: catalogue check values, the Mode S residue and the golden cases
//! of `tests/golden/generate.py` (M37 Tests 1).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use sdrk_proto::crc::{crc16_ccitt_false, crc16_x25, crc24_adsb, crc32};
use sdrk_testkit::golden::{self, Array, Tolerance};

const CHECK: &[u8] = b"123456789";
const MODE_S: [u8; 14] = [
    0x8D, 0x48, 0x40, 0xD6, 0x20, 0x2C, 0xC3, 0x71, 0xC3, 0x2C, 0xE0, 0x57, 0x60, 0x98,
];

#[test]
fn catalogue_check_values() {
    assert_eq!(crc16_ccitt_false(CHECK), 0x29B1);
    assert_eq!(crc16_x25(CHECK), 0x906E);
    assert_eq!(crc32(CHECK), 0xCBF4_3926);
}

#[test]
fn empty_input_gives_init_xor_out() {
    assert_eq!(crc16_ccitt_false(&[]), 0xFFFF);
    assert_eq!(crc16_x25(&[]), 0x0000);
    assert_eq!(crc32(&[]), 0x0000_0000);
    assert_eq!(crc24_adsb(&[]), 0);
}

#[test]
fn mode_s_frame_has_residue_zero() {
    assert_eq!(crc24_adsb(&MODE_S), 0);
    // The parity field is the CRC of the message part.
    assert_eq!(crc24_adsb(&MODE_S[..11]), 0x57_6098);
    // Any single bit error gives a non-zero residue.
    for bit in 0..MODE_S.len() * 8 {
        let mut f = MODE_S;
        f[bit / 8] ^= 0x80 >> (bit % 8);
        assert_ne!(crc24_adsb(&f), 0, "bit {bit}");
    }
    // The result always fits in 24 bits.
    assert!(crc24_adsb(&[0xFF; 300]) <= 0x00FF_FFFF);
}

#[test]
fn appending_the_crc_gives_the_residue() {
    // CCITT-FALSE (no final xor, MSB first): the CRC appended big-endian gives 0.
    let mut m = CHECK.to_vec();
    m.extend_from_slice(&crc16_ccitt_false(CHECK).to_be_bytes());
    assert_eq!(crc16_ccitt_false(&m), 0);
    // X-25 (reflected, final xor 0xFFFF): appended low byte first gives the good residue
    // 0xF0B8 before the final xor, 0x0F47 after it.
    let mut m = CHECK.to_vec();
    m.extend_from_slice(&crc16_x25(CHECK).to_le_bytes());
    assert_eq!(crc16_x25(&m), 0x0F47);
    // CRC-32: the residue 0xDEBB20E3 before the final xor, 0x2144DF1C after it.
    let mut m = CHECK.to_vec();
    m.extend_from_slice(&crc32(CHECK).to_le_bytes());
    assert_eq!(crc32(&m), 0x2144_DF1C);
}

fn i32s(a: &Array) -> Vec<i32> {
    match a {
        Array::I32(v) => v.clone(),
        other => panic!("expected i32, got {}", other.dtype()),
    }
}

fn u8s(a: &Array) -> Vec<u8> {
    match a {
        Array::U8(v) => v.clone(),
        other => panic!("expected u8, got {}", other.dtype()),
    }
}

/// Runs `f` over every message of the golden case `name` and compares bit for bit.
fn golden_crc(name: &str, f: fn(&[u8]) -> u32) {
    let case = golden::load("proto", name).unwrap();
    assert_eq!(case.tolerance, Tolerance::BitExact);
    let data = u8s(case.array("data").unwrap());
    let lengths = i32s(case.array("lengths").unwrap());
    let expected = i32s(case.array("expected").unwrap());
    assert_eq!(lengths.len(), expected.len());
    assert_eq!(case.param_u64("messages").unwrap() as usize, lengths.len());
    let mut pos = 0usize;
    for (k, (&len, &want)) in lengths.iter().zip(&expected).enumerate() {
        let len = usize::try_from(len).unwrap();
        let msg = &data[pos..pos + len];
        pos += len;
        assert_eq!(f(msg), want as u32, "{name}: message {k} of {len} bytes");
    }
    assert_eq!(pos, data.len());
}

#[test]
fn golden_crc16_ccitt_false() {
    golden_crc("crc16_ccitt_false", |d| u32::from(crc16_ccitt_false(d)));
}

#[test]
fn golden_crc16_x25() {
    golden_crc("crc16_x25", |d| u32::from(crc16_x25(d)));
}

#[test]
fn golden_crc32() {
    golden_crc("crc32", crc32);
}

#[test]
fn golden_crc24_adsb() {
    golden_crc("crc24_adsb", crc24_adsb);
}
