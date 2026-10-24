//! `fm_mod` of the SIMD backend against the fallback at the level of the build: every length
//! and block position, inputs at the limit of the vector path's input test, extreme scales,
//! and every entry of the interpolation tables (also through the `CQ15` `rotate`, which shares
//! the Q15 lookup).
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]

mod common;

use common::{Rng, bits, chunks, cq15_noise};
use sdrk_backend_simd::{Simd, level_name};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::{C32, CQ15, Fallback, Q15, Rate};

/// The bound `2^50` on `|x * k|` of the vector path.
const STEP_BOUND: f64 = 1_125_899_906_842_624.0;

/// A state with the scale `k` (exact for normal `k`: the rate is `2^32`).
fn state(k: f64, phase: u32) -> FmModState {
    let mut s = FmModState::new(k, Rate::hz(1 << 32).unwrap()).unwrap();
    s.set_phase(phase);
    s
}

/// `fm_mod` of the backend equals the fallback bit for bit, phase included.
fn check_c32(s: FmModState, x: &[f32], what: &str) {
    let (mut a, mut b) = (s, s);
    let (mut want, mut got) = (vec![C32::default(); x.len()], vec![C32::default(); x.len()]);
    Fallback.fm_mod(&mut a, x, &mut want).unwrap();
    Simd.fm_mod(&mut b, x, &mut got).unwrap();
    assert_eq!(bits(&want), bits(&got), "{} {what}", level_name());
    assert_eq!(a.phase(), b.phase(), "{} {what}", level_name());
}

/// The `CQ15` variant of [`check_c32`].
fn check_q15(s: FmModState, x: &[Q15], what: &str) {
    let (mut a, mut b) = (s, s);
    let (mut want, mut got) = (
        vec![CQ15::default(); x.len()],
        vec![CQ15::default(); x.len()],
    );
    Fallback.fm_mod(&mut a, x, &mut want).unwrap();
    Simd.fm_mod(&mut b, x, &mut got).unwrap();
    assert_eq!(want, got, "{} {what}", level_name());
    assert_eq!(a.phase(), b.phase(), "{} {what}", level_name());
}

/// The scale of a 75 kHz deviation at 240 kHz.
fn broadcast_k() -> f64 {
    75_000.0 / 240_000.0 * 4_294_967_296.0
}

#[test]
fn every_length_and_block_position_is_bit_exact() {
    let mut rng = Rng::new(31);
    let lengths = (0..=300).chain([511, 512, 513, 1031, 2048 + 129]);
    for n in lengths {
        let amp = [1.0f32, 1.0e-3, 4.0e3][n % 3];
        let x: Vec<f32> = (0..n).map(|_| rng.unit() * amp).collect();
        check_c32(state(broadcast_k(), rng.next_u32()), &x, &format!("n={n}"));
        let q: Vec<Q15> = (0..n).map(|_| Q15(rng.next_u32() as i16)).collect();
        check_q15(state(broadcast_k(), rng.next_u32()), &q, &format!("n={n}"));
    }
}

#[test]
fn inputs_at_the_limit_of_the_input_test_are_bit_exact() {
    let mut rng = Rng::new(32);
    for k in [
        1.0,
        3.7e3,
        broadcast_k(),
        1.0e9,
        1_099_511_627_776.0,
        STEP_BOUND / 2.0,
        STEP_BOUND,
        2.0 * STEP_BOUND,
        1.0e17,
    ] {
        // Around |x * k| = 2^50 (the limit of the test), 2^51 (the limit of the exact steps)
        // and 2^52, with both signs and the neighbouring f32 values.
        let mut values = Vec::new();
        for bound in [STEP_BOUND, 2.0 * STEP_BOUND, 4.0 * STEP_BOUND] {
            let v = (bound / k) as f32;
            for w in [v.next_down().next_down(), v.next_down(), v, v.next_up()] {
                values.extend([w, -w]);
            }
        }
        for (i, &v) in values.iter().enumerate() {
            for at in [0, 3, 4, 31, 32, 127, 128, 200] {
                let mut x: Vec<f32> = (0..300).map(|_| rng.unit()).collect();
                x[at] = v;
                check_c32(
                    state(k, rng.next_u32()),
                    &x,
                    &format!("k={k} value {i} = {v} at {at}"),
                );
            }
        }
    }
}

#[test]
fn non_finite_and_extreme_inputs_are_bit_exact() {
    let mut rng = Rng::new(33);
    let specials = [
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1.0e-45,
        -0.0,
        0.0,
    ];
    for k in [broadcast_k(), 1.0e-30, 0.0, 1.0e300] {
        for (i, &v) in specials.iter().enumerate() {
            for at in [0, 1, 15, 16, 64, 129, 299] {
                let mut x: Vec<f32> = (0..300).map(|_| rng.unit() * 2.0).collect();
                x[at] = v;
                check_c32(state(k, 7), &x, &format!("k={k} special {i} at {at}"));
            }
        }
    }
    // Every sample out of range, and out-of-range samples in every block.
    let x = vec![f32::NAN; 300];
    check_c32(state(broadcast_k(), 1), &x, "all NaN");
    let x: Vec<f32> = (0..1000)
        .map(|i| if i % 37 == 0 { 1.0e30 } else { rng.unit() })
        .collect();
    check_c32(state(broadcast_k(), 2), &x, "sparse out of range");
}

#[test]
fn extreme_scales_are_bit_exact() {
    let mut rng = Rng::new(34);
    let x: Vec<f32> = (0..700).map(|_| rng.unit()).collect();
    let q: Vec<Q15> = (0..700).map(|_| Q15(rng.next_u32() as i16)).collect();
    let full: Vec<Q15> = (0..700)
        .map(|i| Q15(if i % 2 == 0 { -32768 } else { 32767 }))
        .collect();
    for k in [
        0.0,
        5.0e-324,
        1.0e-310,
        2.0e-308,
        1.0e-300,
        STEP_BOUND,
        STEP_BOUND.next_up(),
        1.0e300,
    ] {
        check_c32(state(k, 3), &x, &format!("k={k}"));
        check_q15(state(k, 3), &q, &format!("k={k}"));
        check_q15(state(k, 3), &full, &format!("k={k} full scale"));
    }
    // A scale that overflows to infinity.
    let s = FmModState::new(1.0e300, Rate::new(1, 1 << 40).unwrap()).unwrap();
    assert!(s.scale().is_infinite());
    check_c32(s, &x, "k=inf");
    check_q15(s, &q, "k=inf");
}

#[test]
fn phases_cover_every_table_entry_and_fraction_bit() {
    // A constant input with an odd step walks the phase through every table index with
    // varying fractions.
    let n = 1 << 16;
    for step in [0x0001_0F01u32, 0x9E37_79B9, 0xFFF0_0001] {
        let k = f64::from(step);
        check_c32(state(k, 0), &vec![1.0f32; n], &format!("step {step:#x}"));
        let q = vec![Q15(-32768); n];
        check_q15(state(k, 0), &q, &format!("step {step:#x}"));
    }
}

#[test]
fn chunking_does_not_change_the_result() {
    let mut rng = Rng::new(35);
    let n = 3000;
    let x: Vec<f32> = (0..n)
        .map(|i| if i % 211 == 5 { f32::NAN } else { rng.unit() })
        .collect();
    let start = state(broadcast_k(), 99);
    let mut s1 = start;
    let mut whole = vec![C32::default(); n];
    Simd.fm_mod(&mut s1, &x, &mut whole).unwrap();
    let mut s2 = start;
    let mut parts = vec![C32::default(); n];
    let mut at = 0;
    for c in chunks(&mut rng, n) {
        Simd.fm_mod(&mut s2, &x[at..at + c], &mut parts[at..at + c])
            .unwrap();
        at += c;
    }
    assert_eq!(bits(&whole), bits(&parts));
    assert_eq!(s1.phase(), s2.phase());
}

#[test]
fn cq15_rotate_through_every_table_entry_is_bit_exact() {
    let mut rng = Rng::new(36);
    let n = 1 << 14;
    let x = cq15_noise(&mut rng, n);
    for inc in [0x0004_0F03u32, 0x9E37_79B9] {
        let mut a = Nco::new(f64::from(inc), Rate::hz(1 << 32).unwrap()).unwrap();
        a.set_phase(12345);
        let mut b = a;
        let (mut want, mut got) = (vec![CQ15::default(); n], vec![CQ15::default(); n]);
        Fallback.rotate(&mut a, &x, &mut want).unwrap();
        Simd.rotate(&mut b, &x, &mut got).unwrap();
        assert_eq!(want, got, "{} inc {inc:#x}", level_name());
    }
}
