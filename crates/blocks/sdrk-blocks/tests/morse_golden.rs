//! The Morse blocks against the golden cases of `tests/golden/sdrk_golden/cases/morse.py`
//! (M47 Tests 1 and 2): `morse/encode_envelope` (seed 652: the key envelope of three packets
//! at 17 words per minute and 1000 Hz, built by the Python reference with its own table, units
//! and unit boundaries, equal bit for bit to the output of `morse_encode`) and
//! `morse_decode/words` (seed 653: three transmissions at 6, 12 and 25 words per minute in
//! Gaussian noise, decoded by `morse_decode`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

mod common;

use common::{rate, same_bits};
use sdrk_blocks::{
    ArenaStorage, Block, MorseDecodeBlock, MorseDecodeConfig, MorseEncodeBlock, MorseEncodeConfig,
    Packet, PacketFlags,
};
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

/// `data` cut at `lengths`.
fn cut(data: &[u8], lengths: &[i32]) -> Vec<Vec<u8>> {
    let mut at = 0;
    lengths
        .iter()
        .map(|&l| {
            let part = data[at..at + l as usize].to_vec();
            at += l as usize;
            part
        })
        .collect()
}

#[test]
fn the_envelope_equals_the_python_reference() {
    let c = golden::load("morse", "encode_envelope").unwrap();
    let texts = cut(&u8s(&c, "data"), &i32s(&c, "lengths"));
    let want = f32s(&c, "envelope");
    let cfg = MorseEncodeConfig {
        wpm: 17.0,
        rise_ms: 0.0,
        ..MorseEncodeConfig::default()
    };
    let mut enc = MorseEncodeBlock::new(&cfg, rate(1000), &mut ArenaStorage::new(&mut [])).unwrap();
    let packets: Vec<Packet> = texts
        .iter()
        .map(|t| Packet::new(0, PacketFlags::NONE, t).unwrap())
        .collect();
    let mut got = Vec::new();
    let mut room = vec![0.0f32; 100_000];
    let p = enc.process(&packets, &mut room).unwrap();
    assert_eq!(p.consumed, packets.len());
    got.extend_from_slice(&room[..p.produced]);
    same_bits(&got, &want, "key envelope");
}

fn words_of(env: &[f32], wpm: f64, wpm_max: f64) -> Vec<String> {
    let cfg = MorseDecodeConfig {
        wpm,
        wpm_max,
        ..MorseDecodeConfig::default()
    };
    let mut dec = MorseDecodeBlock::new(&cfg, rate(1000), &mut ArenaStorage::new(&mut [])).unwrap();
    let mut room = vec![Packet::default(); 4096];
    let p = dec.process(env, &mut room).unwrap();
    assert_eq!(p.consumed, env.len());
    room[..p.produced]
        .iter()
        .map(|w| String::from_utf8(w.data().to_vec()).unwrap())
        .collect()
}

#[test]
fn the_words_of_noisy_transmissions_are_decoded() {
    let c = golden::load("morse_decode", "words").unwrap();
    let env = f32s(&c, "envelope");
    let starts: Vec<usize> = i32s(&c, "starts").iter().map(|&s| s as usize).collect();
    let speeds = f32s(&c, "wpm");
    let words = cut(&u8s(&c, "words"), &i32s(&c, "word_lengths"));
    let words: Vec<String> = words
        .into_iter()
        .map(|w| String::from_utf8(w).unwrap())
        .collect();
    // Each transmission alone, with the speed as the hint: every word comes out.
    let per = [6usize, 4, 7];
    let mut at = 0;
    for (i, &n) in per.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(env.len());
        let got = words_of(&env[starts[i]..end], f64::from(speeds[i]), 40.0);
        assert_eq!(
            got,
            words[at..at + n],
            "transmission {i} ({} wpm)",
            speeds[i]
        );
        at += n;
    }
    assert_eq!(at, words.len());
    // The three in a row with the default hint of 20: the preamble VVV settles the estimate,
    // so everything after it is exact, and the silence between them (22 word gaps) makes the
    // decoder forget the speed.
    let got = words_of(&env, 20.0, 40.0).join(" ");
    let mut at = 0;
    for &n in &per {
        let tail = words[at + 1..at + n].join(" ");
        assert!(got.contains(&tail), "{tail:?} not in {got:?}");
        at += n;
    }
}
