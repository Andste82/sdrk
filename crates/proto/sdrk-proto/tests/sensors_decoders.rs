//! The device decoders on arbitrary input (M46, seed 648): no panic, no reading for noise.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use sdrk_proto::pulses::{self, Header, Modulation, Pair};
use sdrk_proto::sensors::{self, DecodeResult, ReadingSink};

const SEED: u64 = 648;

/// A xorshift64 generator (the proto crate has no random number generator).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn every_decoder_survives_random_trains() {
    let mut rng = Rng(SEED.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for round in 0..600 {
        let modulation = if round % 2 == 0 {
            Modulation::Ook
        } else {
            Modulation::Fsk
        };
        let rate = [0u32, 1, 250_000, 2_000_000][rng.below(4) as usize];
        let n = rng.below(400) as usize;
        // widths around the nominal ones of the devices, with outliers and zeros
        let pairs: Vec<Pair> = (0..n)
            .map(|_| {
                let long_gap = rng.below(10) == 0;
                let mut w = |scale: u64| {
                    let base = [0, 100, 125, 250, 375, 500, 14, 29, 1000][rng.below(9) as usize];
                    (base + rng.below(scale)) as u32
                };
                let pulse = w(40);
                Pair {
                    pulse,
                    gap: w(if long_gap { 4000 } else { 40 }),
                }
            })
            .collect();
        let header = Header {
            modulation,
            channel: 0,
            freq_offset_hz: 0,
            level_db10: 0,
            noise_db10: 0,
            rate_hz: rate,
        };
        let mut buf = vec![0u8; 16 + 8 * n];
        let len = pulses::encode_joined(&header, &pairs, &mut buf).unwrap();
        let train = pulses::decode(&buf[..len]).unwrap();
        for d in sensors::DECODERS {
            let mut sink = ReadingSink::new();
            let res = d.decode(&train, &mut sink);
            // random widths are not a message of any device
            assert!(
                !matches!(res, DecodeResult::Decoded(_)) && sink.is_empty(),
                "{} read noise (round {round})",
                d.name()
            );
        }
    }
}

#[test]
fn the_registry_lists_the_five_decoders() {
    let names: Vec<_> = sensors::DECODERS.iter().map(|d| d.name()).collect();
    assert_eq!(
        names,
        ["wh2", "nexus", "oregon_v21", "wh1080", "lacrosse_tx29"]
    );
    for n in names {
        assert_eq!(sensors::by_name(n).unwrap().name(), n);
    }
    assert!(sensors::by_name("nope").is_none());
    let ook = sensors::DECODERS
        .iter()
        .filter(|d| d.modulation() == Modulation::Ook)
        .count();
    assert_eq!(ook, 4);
}

/// The pairs (pulse, gap) of the bits `bits` as NRZ with 58 samples per bit (rate 1 MHz); the
/// last gap is `last_gap`.
fn nrz_pairs(bits: &[bool], last_gap: u32) -> Vec<Pair> {
    let mut runs: Vec<(bool, u32)> = Vec::new();
    for &b in bits {
        match runs.last_mut() {
            Some((v, n)) if *v == b => *n += 58,
            _ => runs.push((b, 58)),
        }
    }
    let mut pairs = Vec::new();
    let mut i = 0;
    if runs.first().is_some_and(|r| !r.0) {
        i = 1;
    }
    while i < runs.len() {
        let pulse = runs[i].1;
        let gap = runs.get(i + 1).map_or(last_gap, |r| r.1);
        pairs.push(Pair { pulse, gap });
        i += 2;
    }
    pairs
}

fn tx29_decode(bits: &[bool], last_gap: u32, bad_tail: bool) -> DecodeResult {
    let mut pairs = nrz_pairs(bits, last_gap);
    if bad_tail {
        pairs.push(Pair { pulse: 25, gap: 0 });
    }
    let header = Header {
        modulation: Modulation::Fsk,
        channel: 0,
        freq_offset_hz: 0,
        level_db10: 0,
        noise_db10: 0,
        rate_hz: 1_000_000,
    };
    let mut buf = vec![0u8; 16 + 8 * pairs.len()];
    let len = pulses::encode_joined(&header, &pairs, &mut buf).unwrap();
    let train = pulses::decode(&buf[..len]).unwrap();
    let mut sink = ReadingSink::new();
    sensors::by_name("lacrosse_tx29")
        .unwrap()
        .decode(&train, &mut sink)
}

#[test]
fn lacrosse_zero_fill_is_for_clean_row_ends_only() {
    // Preamble, sync and the frame 92 03 94 59 3C; the last two bits of the CRC are zeros.
    let bytes = [
        0xAAu8, 0xAA, 0xAA, 0xAA, 0x2D, 0xD4, 0x92, 0x03, 0x94, 0x59, 0x3C,
    ];
    let mut bits: Vec<bool> = bytes
        .iter()
        .flat_map(|b| (0..8).rev().map(move |i| b >> i & 1 == 1))
        .collect();
    bits.truncate(bits.len() - 2);
    // A burst that ends cleanly: the missing zeros are filled in and the CRC decides.
    assert!(matches!(
        tx29_decode(&bits, 0, false),
        DecodeResult::Decoded(_)
    ));
    // The same bits cut by a width that is no multiple of a bit: a prefix, never a reading.
    assert!(!matches!(
        tx29_decode(&bits, 58, true),
        DecodeResult::Decoded(_)
    ));
}
