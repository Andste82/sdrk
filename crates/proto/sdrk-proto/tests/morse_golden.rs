//! The Morse encoder of `sdrk-proto` against the golden cases `morse/encode_units` (seed 650)
//! and `morse/encode_units_farnsworth` (seed 651) of `tests/golden/sdrk_golden/cases/morse.py`
//! (M47 Test 1): the Python side has its own table, unit expansion and gap rounding, written
//! from ITU-R M.1677-1 and the Farnsworth formula, so a mistake in either side shows as a
//! difference.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_sign_loss
)]

use sdrk_proto::morse::Encoder;
use sdrk_testkit::golden::{self, Array, GoldenCase};

fn i32s(c: &GoldenCase, name: &str) -> Vec<i32> {
    match c.array(name).unwrap() {
        Array::I32(v) => v.clone(),
        other => panic!("{name}: i32 expected, got {}", other.dtype()),
    }
}

fn u8s(c: &GoldenCase, name: &str) -> Vec<u8> {
    match c.array(name).unwrap() {
        Array::U8(v) => v.clone(),
        other => panic!("{name}: u8 expected, got {}", other.dtype()),
    }
}

fn f32s(c: &GoldenCase, name: &str) -> Vec<f32> {
    match c.array(name).unwrap() {
        Array::F32(v) => v.clone(),
        other => panic!("{name}: f32 expected, got {}", other.dtype()),
    }
}

/// The units of `text` through the encoder, one byte at a time.
fn units(enc: &mut Encoder, text: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for &c in text {
        enc.push(c).unwrap();
        while let Some(k) = enc.next_unit() {
            out.push(u8::from(k));
        }
    }
    out
}

/// The texts and the units of a case, cut at their lengths.
fn split(c: &GoldenCase) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let data = u8s(c, "data");
    let units = u8s(c, "units");
    let lengths = i32s(c, "lengths");
    let counts = i32s(c, "unit_counts");
    assert_eq!(lengths.len(), counts.len());
    let (mut d, mut u) = (0, 0);
    let mut texts = Vec::new();
    let mut want = Vec::new();
    for (&l, &n) in lengths.iter().zip(&counts) {
        texts.push(data[d..d + l as usize].to_vec());
        want.push(units[u..u + n as usize].to_vec());
        d += l as usize;
        u += n as usize;
    }
    assert_eq!((d, u), (data.len(), units.len()));
    (texts, want)
}

#[test]
fn the_units_equal_those_of_the_python_reference() {
    let c = golden::load("morse", "encode_units").unwrap();
    let (texts, want) = split(&c);
    assert_eq!(texts.len(), 40);
    for (i, (t, w)) in texts.iter().zip(&want).enumerate() {
        let mut enc = Encoder::new(20.0, 20.0).unwrap();
        assert_eq!(
            &units(&mut enc, t),
            w,
            "text {i}: {:?}",
            String::from_utf8_lossy(t)
        );
    }
}

#[test]
fn farnsworth_gaps_equal_those_of_the_python_reference() {
    let c = golden::load("morse", "encode_units_farnsworth").unwrap();
    let (texts, want) = split(&c);
    let (wpm, char_wpm) = (f32s(&c, "wpm"), f32s(&c, "char_wpm"));
    let (cg, wg) = (i32s(&c, "char_gap"), i32s(&c, "word_gap"));
    assert_eq!(texts.len(), 6);
    for i in 0..texts.len() {
        let mut enc = Encoder::new(wpm[i], char_wpm[i]).unwrap();
        assert_eq!(
            (enc.char_gap(), enc.word_gap()),
            (cg[i] as u32, wg[i] as u32),
            "{} / {}",
            wpm[i],
            char_wpm[i]
        );
        assert_eq!(units(&mut enc, &texts[i]), want[i], "pair {i}");
    }
}
