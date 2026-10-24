//! `rotate` `C32` with the block oscillator: accuracy over many short and long calls, and the
//! independence of the output from the split into calls for every kind of increment.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::print_stderr
)]

mod common;

use common::{Rng, bits, c32_noise, chunks, db, powers};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::{C32, Rate, Reference};

/// `inc` with exactly `t` trailing zero bits (any odd part from `rng`).
fn inc_with_zeros(rng: &mut Rng, t: u32) -> u32 {
    if t >= 32 {
        0
    } else {
        (rng.next_u32() | 1) << t
    }
}

/// Rotates `x` from `phase` with `inc` in one call: the result of `Simd` and of `Reference`.
fn both(x: &[C32], phase: u32, inc: u32) -> (Vec<C32>, Vec<C32>) {
    let mut a = Nco::from_increment(inc);
    a.set_phase(phase);
    let mut b = a;
    let (mut want, mut got) = (vec![C32::default(); x.len()], vec![C32::default(); x.len()]);
    Reference.rotate(&mut a, x, &mut want).unwrap();
    Simd.rotate(&mut b, x, &mut got).unwrap();
    assert_eq!(a, b, "the oscillator advances by the number of samples");
    (want, got)
}

#[test]
fn every_call_stays_above_130_db() {
    // Short calls start anywhere in a block; long ones cover many blocks. Every call is above
    // the floor of the backend tests (the contract is 120 dB).
    let mut rng = Rng::new(40);
    let (mut sig, mut err, mut worst) = (0f64, 0f64, f64::INFINITY);
    for trial in 0..3000 {
        let n = if trial % 10 == 0 {
            1 + rng.below(5000)
        } else {
            1 + rng.below(200)
        };
        let inc = match trial % 7 {
            0 => {
                let t = rng.below(33) as u32;
                inc_with_zeros(&mut rng, t)
            }
            _ => rng.next_u32(),
        };
        let phase = rng.next_u32();
        let x = c32_noise(&mut rng, n, 1.0);
        let (want, got) = both(&x, phase, inc);
        let (s, e) = powers(&want, &got);
        let snr = db(s, e);
        assert!(snr > 130.0, "n={n} phase={phase:#x} inc={inc:#x} snr={snr}");
        worst = worst.min(snr);
        sig += s;
        err += e;
    }
    eprintln!("worst {worst:.2} dB, aggregate {:.2} dB", db(sig, err));
}

#[test]
fn the_output_does_not_depend_on_the_split_for_any_increment() {
    // Increments with every number of trailing zeros: the block length is the period of the
    // phase sequence when that is shorter than the longest block, and inc = 0 is constant.
    let mut rng = Rng::new(41);
    let n = 2500;
    for t in 0..=32 {
        for _ in 0..3 {
            let inc = inc_with_zeros(&mut rng, t);
            let x = c32_noise(&mut rng, n, 1.0);
            let start = {
                let mut nco = Nco::from_increment(inc);
                nco.set_phase(rng.next_u32());
                nco
            };
            let mut nco = start;
            let mut whole = vec![C32::default(); n];
            Simd.rotate(&mut nco, &x, &mut whole).unwrap();
            let mut parts = vec![C32::default(); n];
            let mut nco2 = start;
            let mut at = 0;
            for c in chunks(&mut rng, n) {
                Simd.rotate(&mut nco2, &x[at..at + c], &mut parts[at..at + c])
                    .unwrap();
                at += c;
            }
            assert_eq!(bits(&whole), bits(&parts), "inc={inc:#x}");
            assert_eq!(nco, nco2);
        }
    }
}

#[test]
fn a_split_at_every_offset_gives_the_same_bits() {
    // Every split point of a signal longer than the longest block (2048 samples at AVX-512):
    // every position within a block and within a vector starts a call once.
    let mut rng = Rng::new(42);
    let n = 2100;
    for inc in [0x0123_4567u32, 0xFEDC_BA99, 0x0000_0100, 0x8000_0000] {
        let x = c32_noise(&mut rng, n, 1.0);
        let mut start = Nco::from_increment(inc);
        start.set_phase(rng.next_u32());
        let mut whole = vec![C32::default(); n];
        Simd.rotate(&mut { start }, &x, &mut whole).unwrap();
        let want = bits(&whole);
        let mut parts = vec![C32::default(); n];
        for at in 0..=n {
            let mut nco = start;
            Simd.rotate(&mut nco, &x[..at], &mut parts[..at]).unwrap();
            Simd.rotate(&mut nco, &x[at..], &mut parts[at..]).unwrap();
            assert!(bits(&parts) == want, "inc={inc:#x} split at {at}");
        }
    }
}

#[test]
fn a_retune_continues_from_the_current_phase_with_the_new_increment() {
    // After `set_frequency` the output equals that of an oscillator started at the current
    // phase with the new increment: the old increment leaves no trace in the bits.
    let mut rng = Rng::new(43);
    let rate = Rate::hz(2_400_000).unwrap();
    let x = c32_noise(&mut rng, 3000, 1.0);
    let mut nco = Nco::new(123_456.7, rate).unwrap();
    nco.set_phase(0x89AB_CDEF);
    let mut y = vec![C32::default(); 3000];
    Simd.rotate(&mut nco, &x[..1234], &mut y[..1234]).unwrap();
    nco.set_frequency(-654_321.0, rate).unwrap();
    let mut fresh = Nco::new(-654_321.0, rate).unwrap();
    fresh.set_phase(nco.phase());
    Simd.rotate(&mut nco, &x[1234..], &mut y[1234..]).unwrap();
    let mut z = vec![C32::default(); 3000 - 1234];
    Simd.rotate(&mut fresh, &x[1234..], &mut z).unwrap();
    assert_eq!(bits(&y[1234..]), bits(&z));
    assert_eq!(nco, fresh);
}
