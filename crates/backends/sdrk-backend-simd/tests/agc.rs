//! `agc` of `C32` against the fallback: bit-exact output and state at every length from 0 to
//! 1100 (more than four blocks of the backend), at every start offset within a vector, for
//! signals that cross every branch of the update (attack above and below the divisor limit,
//! release above and below it, silence), with zeros, NaN, infinities and subnormal samples in
//! blocks of their own and mixed into others, for several parameter sets, and in random
//! chunks. The tests hold in every build: the kernel uses no multiply-add.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_precision_loss
)]

mod common;

use common::{Rng, c32_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::agc::{Agc, AgcState};
use sdrk_core::{C32, Fallback, Rate};

/// The longest length; every length up to it is tested.
const MAX_LEN: usize = 1100;

/// Start offsets in samples: the kernel must not depend on the alignment of the buffers.
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

/// Equal bits, or both NaN (the payload of a NaN result is not specified).
fn same(want: &[C32], got: &[C32]) -> bool {
    let bits = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    want.iter()
        .zip(got)
        .all(|(w, g)| bits(w.re, g.re) && bits(w.im, g.im))
}

fn state(params: (f64, f64, f64, f64, f64, f64)) -> AgcState<C32> {
    let (reference, attack, release, min, max, init) = params;
    AgcState::new(
        reference,
        attack,
        release,
        min,
        max,
        init,
        Rate::hz(48_000).unwrap(),
    )
    .unwrap()
}

/// The parameter sets: the one of the benchmark, fast loops with narrow limits, slow loops
/// with wide limits.
const PARAMS: [(f64, f64, f64, f64, f64, f64); 3] = [
    (0.5, 0.005, 0.05, -20.0, 60.0, 0.0),
    (0.25, 0.0002, 0.001, -6.0, 12.0, 6.0),
    (1.0, 0.1, 0.5, -40.0, 90.0, 40.0),
];

/// Runs both implementations from equal states over `x` and compares output and state.
fn check(params: (f64, f64, f64, f64, f64, f64), x: &[C32], what: &str) {
    let (mut sw, mut sg) = (state(params), state(params));
    let (mut want, mut got) = (vec![C32::default(); x.len()], vec![C32::default(); x.len()]);
    Fallback.agc(&mut sw, x, &mut want).unwrap();
    Simd.agc(&mut sg, x, &mut got).unwrap();
    assert!(same(&want, &got), "{what}: output differs");
    assert_eq!(sw, sg, "{what}: state differs");
}

/// Noise whose amplitude steps by 30 dB up and down every 97 samples, so that the loop runs
/// through attack and release on both sides of the divisor limits.
fn steps(rng: &mut Rng, n: usize) -> Vec<C32> {
    let amps = [0.01_f32, 0.3, 3.0, 0.003, 30.0, 0.1];
    let mut x = c32_noise(rng, n, 1.0);
    for (i, s) in x.iter_mut().enumerate() {
        *s *= amps[(i / 97) % amps.len()];
    }
    x
}

#[test]
fn c32_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(2001);
    let x = steps(&mut rng, MAX_LEN + OFFSETS.len());
    for (k, &params) in PARAMS.iter().enumerate() {
        for n in 0..=MAX_LEN {
            let off = OFFSETS[(n + k) % OFFSETS.len()];
            check(
                params,
                &x[off..off + n],
                &format!("params {k} n={n} offset={off}"),
            );
        }
    }
}

#[test]
fn c32_is_bit_exact_with_zeros_nan_infinity_and_subnormals() {
    let mut rng = Rng::new(2002);
    let base = steps(&mut rng, 2048);
    let specials = [
        C32::new(0.0, 0.0),
        C32::new(-0.0, 0.0),
        C32::new(f32::NAN, 0.1),
        C32::new(0.1, f32::NAN),
        C32::new(f32::INFINITY, 0.0),
        C32::new(0.0, f32::NEG_INFINITY),
        C32::new(1e-40, 0.0),
        C32::new(1e-45, -1e-45),
        C32::new(1e-23, 0.0),
        C32::new(3e19, 3e19),
        C32::new(f32::MAX, f32::MAX),
    ];
    for &special in &specials {
        // One special sample in the middle of a block, at the start and at the end of one, a
        // whole block of it, and every fourth sample.
        let patterns: [&dyn Fn(usize) -> bool; 5] = [
            &|i| i == 300,
            &|i| i == 256,
            &|i| i == 511,
            &|i| (512..768).contains(&i),
            &|i| i % 4 == 1,
        ];
        for (p, pattern) in patterns.iter().enumerate() {
            let x: Vec<C32> = base
                .iter()
                .enumerate()
                .map(|(i, &s)| if pattern(i) { special } else { s })
                .collect();
            for (k, &params) in PARAMS.iter().enumerate() {
                check(params, &x, &format!("{special:?} pattern {p} params {k}"));
            }
        }
    }
    // Silence alone, and a faint signal alone.
    for (k, &params) in PARAMS.iter().enumerate() {
        check(
            params,
            &vec![C32::new(0.0, 0.0); 700],
            &format!("silence params {k}"),
        );
        check(
            params,
            &vec![C32::new(1e-6, 0.0); 700],
            &format!("faint params {k}"),
        );
    }
}

#[test]
fn c32_is_bit_exact_in_random_chunks() {
    let mut rng = Rng::new(2003);
    let x = steps(&mut rng, 6000);
    for &params in &PARAMS {
        let (mut sw, mut sg) = (state(params), state(params));
        let (mut want, mut got) = (vec![C32::default(); x.len()], vec![C32::default(); x.len()]);
        Fallback.agc(&mut sw, &x, &mut want).unwrap();
        let mut done = 0;
        while done < x.len() {
            let n = (1 + rng.below(700)).min(x.len() - done);
            Simd.agc(&mut sg, &x[done..done + n], &mut got[done..done + n])
                .unwrap();
            done += n;
        }
        assert!(same(&want, &got));
        assert_eq!(sw, sg);
    }
}

#[test]
fn the_recursion_alone_gives_the_gains_of_the_fallback() {
    let mut rng = Rng::new(2004);
    let x = steps(&mut rng, 1000);
    let mag: Vec<f32> = x
        .iter()
        .map(|s| sdrk_core::num::sqrt_f32(s.re * s.re + s.im * s.im))
        .collect();
    for &params in &PARAMS {
        let (mut sw, mut sg) = (state(params), state(params));
        let mut want = vec![C32::default(); x.len()];
        Fallback.agc(&mut sw, &x, &mut want).unwrap();
        let mut gains = vec![0.0; x.len()];
        sg.gains_with(&mag, &mut gains, true, f32::min, f32::max)
            .unwrap();
        assert_eq!(sw, sg);
        for ((w, s), g) in want.iter().zip(&x).zip(&gains) {
            assert_eq!(*w, *s * *g);
        }
    }
}
