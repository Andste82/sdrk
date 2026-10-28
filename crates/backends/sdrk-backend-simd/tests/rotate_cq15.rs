//! `rotate` `CQ15`: bit-exact with the fallback for every length, including the zero-padded
//! last vector, and for every kind of increment.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{Rng, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::{CQ15, Fallback, Q15};

#[test]
fn every_length_is_bit_exact_with_the_fallback() {
    let mut rng = Rng::new(50);
    for n in 0..=70 {
        for inc in [0u32, 1, 3 << 30, 0x0123_4567, rng.next_u32()] {
            let x = cq15_noise(&mut rng, n);
            let mut a = Nco::from_increment(inc);
            a.set_phase(rng.next_u32());
            let mut b = a;
            let (mut want, mut got) = (vec![CQ15::default(); n], vec![CQ15::default(); n]);
            Fallback.rotate(&mut a, &x, &mut want).unwrap();
            Simd.rotate(&mut b, &x, &mut got).unwrap();
            assert_eq!(want, got, "n={n} inc={inc:#x}");
            assert_eq!(a, b);
        }
    }
}

#[test]
fn extreme_samples_and_phases_are_bit_exact_with_the_fallback() {
    // Full-scale components (products and sums at the edge of `i32`, saturation of both
    // components) against the oscillator at and next to the quarter turns, where the table
    // gives -32768 (`cos(pi)`) and the negation of the sine must not overflow.
    let values = [-32768i16, -32767, -16384, -1, 0, 1, 16384, 32766, 32767];
    let x: Vec<CQ15> = values
        .iter()
        .flat_map(|&re| {
            values.iter().map(move |&im| CQ15 {
                re: Q15(re),
                im: Q15(im),
            })
        })
        .collect();
    let mut phases = Vec::new();
    for q in 0..4u32 {
        for d in [
            0u32,
            1,
            31,
            32,
            0x10_0000,
            0xFFFF_FFFF,
            0xFFFF_FFE0,
            0xFFF0_0000,
        ] {
            phases.push((q << 30).wrapping_add(d));
        }
    }
    for &phase in &phases {
        for inc in [0u32, 0x0001_0000, 0x0123_4567, 0x8000_0000, 0xFFFF_FFFF] {
            for n in [x.len(), 37, 8, 1] {
                let x = &x[..n];
                let mut a = Nco::from_increment(inc);
                a.set_phase(phase);
                let mut b = a;
                let (mut want, mut got) = (vec![CQ15::default(); n], vec![CQ15::default(); n]);
                Fallback.rotate(&mut a, x, &mut want).unwrap();
                Simd.rotate(&mut b, x, &mut got).unwrap();
                assert_eq!(want, got, "phase={phase:#x} inc={inc:#x} n={n}");
                assert_eq!(a, b);
            }
        }
    }
}

#[test]
fn the_output_does_not_depend_on_the_split_into_calls() {
    let mut rng = Rng::new(51);
    let x = cq15_noise(&mut rng, 300);
    for inc in [0x0123_4567u32, 0xFEDC_BA98, 3] {
        let mut whole = Nco::from_increment(inc);
        whole.set_phase(0x8765_4321);
        let mut parts = whole;
        let (mut want, mut got) = (vec![CQ15::default(); 300], vec![CQ15::default(); 300]);
        Simd.rotate(&mut whole, &x, &mut want).unwrap();
        let mut at = 0;
        for len in [1usize, 7, 16, 33, 2, 64, 100, 77] {
            Simd.rotate(&mut parts, &x[at..at + len], &mut got[at..at + len])
                .unwrap();
            at += len;
        }
        assert_eq!(at, 300);
        assert_eq!(want, got, "inc={inc:#x}");
        assert_eq!(whole, parts);
    }
}
