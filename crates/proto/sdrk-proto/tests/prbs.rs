//! `sdrk_proto::prbs` (M38 §4, test 5): the first 32 bits of each sequence from the all-ones
//! seed, the period, the seed validation and the `Send`/`Sync` assertions.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use sdrk_device::{ErrorKind, SdrkError};
use sdrk_proto::Error;
use sdrk_proto::prbs::{Poly, Prbs};

/// Computed independently with a list model of the shift register (Python, stages 9 and 5).
const PRBS9_FIRST_32: [u8; 32] = [
    0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,
];

/// Computed independently with a list model of the shift register (Python, stages 15 and 14).
const PRBS15_FIRST_32: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
];

fn first(poly: Poly, seed: u32, n: usize) -> Vec<u8> {
    let mut p = Prbs::new(poly, seed).unwrap();
    (0..n).map(|_| p.next_bit()).collect()
}

#[test]
fn prbs9_first_32_bits_from_all_ones() {
    assert_eq!(first(Poly::Prbs9, 0x1ff, 32), PRBS9_FIRST_32);
}

#[test]
fn prbs15_first_32_bits_from_all_ones() {
    assert_eq!(first(Poly::Prbs15, 0x7fff, 32), PRBS15_FIRST_32);
}

#[test]
fn sequences_are_maximal_length() {
    for poly in [Poly::Prbs9, Poly::Prbs15] {
        let period = poly.period() as usize;
        let mut p = Prbs::new(poly, poly.period()).unwrap();
        let start = p.state();
        let mut ones = 0usize;
        for k in 1..=period {
            ones += usize::from(p.next_bit());
            if k < period {
                assert_ne!(p.state(), start, "{poly:?} repeats after {k} bits");
            }
        }
        assert_eq!(p.state(), start, "{poly:?} period");
        assert_eq!(ones, period.div_ceil(2), "{poly:?} ones per period");
    }
}

#[test]
fn zero_seed_is_rejected() {
    for (poly, seed) in [
        (Poly::Prbs9, 0),
        (Poly::Prbs9, 0x200),
        (Poly::Prbs15, 0x8000),
    ] {
        let err = Prbs::new(poly, seed).unwrap_err();
        assert_eq!(err, Error::InvalidArgument("prbs seed must be non-zero"));
        assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    }
}

#[test]
fn bits_are_zero_or_one() {
    let bits = first(Poly::Prbs15, 0x1234, 1000);
    assert!(bits.iter().all(|&b| b <= 1));
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Prbs>();
    assert_send_sync::<Poly>();
}
