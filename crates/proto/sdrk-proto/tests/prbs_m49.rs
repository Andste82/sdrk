//! The polynomials `Prbs5`, `Prbs7`, `Prbs10` and `Prbs11` of `sdrk_proto::prbs` (M49 §6): the
//! full period `2^n − 1` (every non-zero state once), the first bits from seed 1 and from a
//! random seed against the golden case `proto/prbs_first_bits` (seed 696) of the independent
//! Python model `tests/golden/sdrk_golden/prbs.py`, and the word-parallel step
//! [`Prbs::next_bits`] against [`Prbs::next_bit`] over the full period of every polynomial.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_sign_loss
)]

use std::collections::BTreeSet;

use sdrk_proto::prbs::{Poly, Prbs};
use sdrk_testkit::golden::{self, Array, GoldenCase};

/// Every polynomial, in the order of the golden case (`polys`).
const ALL: [(Poly, &str); 6] = [
    (Poly::Prbs5, "prbs5"),
    (Poly::Prbs7, "prbs7"),
    (Poly::Prbs9, "prbs9"),
    (Poly::Prbs10, "prbs10"),
    (Poly::Prbs11, "prbs11"),
    (Poly::Prbs15, "prbs15"),
];

/// The four polynomials of M49.
const NEW: [Poly; 4] = [Poly::Prbs5, Poly::Prbs7, Poly::Prbs10, Poly::Prbs11];

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

fn first(poly: Poly, seed: u32, n: usize) -> Vec<u8> {
    let mut p = Prbs::new(poly, seed).unwrap();
    (0..n).map(|_| p.next_bit()).collect()
}

#[test]
fn degrees_taps_and_periods() {
    let want = [
        (Poly::Prbs5, 5, 3, 31),
        (Poly::Prbs7, 7, 6, 127),
        (Poly::Prbs10, 10, 7, 1023),
        (Poly::Prbs11, 11, 9, 2047),
    ];
    for (poly, n, t, period) in want {
        assert_eq!(poly.degree(), n, "{poly:?}");
        assert_eq!(poly.second_tap(), t, "{poly:?}");
        assert_eq!(poly.period(), period, "{poly:?}");
    }
}

#[test]
fn every_new_polynomial_has_the_full_period() {
    for poly in NEW {
        let period = poly.period() as usize;
        let mut p = Prbs::new(poly, 1).unwrap();
        let mut seen = BTreeSet::new();
        let mut ones = 0usize;
        for _ in 0..period {
            assert!(seen.insert(p.state()), "{poly:?}: a state repeats early");
            ones += usize::from(p.next_bit());
        }
        // Every non-zero state once, and back to the start.
        assert_eq!(seen.len(), period, "{poly:?}");
        assert!(!seen.contains(&0), "{poly:?}");
        assert_eq!(p.state(), 1, "{poly:?}: period");
        assert_eq!(ones, period.div_ceil(2), "{poly:?}: ones per period");
    }
}

#[test]
fn seeds_that_mask_to_zero_are_rejected() {
    for poly in NEW {
        assert!(Prbs::new(poly, 0).is_err(), "{poly:?}");
        assert!(Prbs::new(poly, 1 << poly.degree()).is_err(), "{poly:?}");
        assert_eq!(
            Prbs::new(poly, (1 << poly.degree()) | 3).unwrap().state(),
            3,
            "{poly:?}"
        );
    }
}

#[test]
fn the_first_bits_equal_the_python_model() {
    let c = golden::load("proto", "prbs_first_bits").unwrap();
    assert_eq!(c.seed, 696);
    let bits_per_seed = c.params["bits_per_seed"].as_u64().unwrap() as usize;
    let seeds_per_poly = c.params["seeds_per_poly"].as_u64().unwrap() as usize;
    let names: Vec<&str> = c.params["polys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(names, ALL.map(|(_, n)| n));
    let degree = i32s(&c, "degree");
    let tap = i32s(&c, "tap");
    let seeds = i32s(&c, "seed");
    let bits = u8s(&c, "bits");
    assert_eq!(seeds.len(), ALL.len() * seeds_per_poly);
    assert_eq!(bits.len(), seeds.len() * bits_per_seed);
    for (i, (poly, name)) in ALL.into_iter().enumerate() {
        assert_eq!(degree[i] as u32, poly.degree(), "{name}");
        assert_eq!(tap[i] as u32, poly.second_tap(), "{name}");
        for s in 0..seeds_per_poly {
            let k = i * seeds_per_poly + s;
            let seed = seeds[k] as u32;
            let want = &bits[k * bits_per_seed..(k + 1) * bits_per_seed];
            if s == 0 {
                assert_eq!(seed, 1, "{name}: the first seed is 1");
                // The first 16 bits from seed 1 (the spec's check), then all of them.
                assert_eq!(first(poly, 1, 16), want[..16], "{name}: first 16 bits");
            }
            assert_eq!(first(poly, seed, bits_per_seed), want, "{name} seed {seed}");
        }
    }
}

#[test]
fn the_word_step_equals_next_bit_over_the_full_period() {
    for (poly, name) in ALL {
        for count in 0..=32u32 {
            let mut word = Prbs::new(poly, 1).unwrap();
            let mut serial = word.clone();
            // At least one full period of bits, so every register state is a start state of
            // some word.
            let period = u64::from(poly.period());
            let words = if count == 0 {
                1
            } else {
                period.div_ceil(u64::from(count)) + 1
            };
            for _ in 0..words {
                let got = word.next_bits(count);
                let want = (0..count).fold(0u32, |w, _| {
                    w.wrapping_shl(1) | u32::from(serial.next_bit())
                });
                assert_eq!(got, want, "{name} count {count}");
                assert_eq!(word, serial, "{name} count {count}: state");
            }
        }
        // Every start state and every count 1 … 8 of the block (the full period of states).
        let mut p = Prbs::new(poly, 1).unwrap();
        for _ in 0..poly.period() {
            for count in 1..=8 {
                let mut a = p.clone();
                let mut b = p.clone();
                let got = a.next_bits(count);
                let want = (0..count).fold(0u32, |w, _| (w << 1) | u32::from(b.next_bit()));
                assert_eq!((got, &a), (want, &b), "{name} state {:#x}", p.state());
            }
            p.next_bit();
        }
    }
}

#[test]
fn a_count_above_32_is_32() {
    let mut a = Prbs::new(Poly::Prbs15, 0x1234).unwrap();
    let mut b = a.clone();
    assert_eq!(a.next_bits(40), b.next_bits(32));
    assert_eq!(a, b);
}
