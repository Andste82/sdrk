//! `agc` of `C32` against the fallback: bit-exact output and state at every length from 0 to
//! 1100 (more than four blocks of the backend), at every start offset within a vector, for
//! signals that cross every branch of the update (attack above and below the divisor limit,
//! release above and below it, silence), with zeros, NaN, infinities and subnormal samples in
//! blocks of their own and mixed into others, for several parameter sets, and in random
//! chunks, and for signals whose every output level lies at or next to a bound of the branches
//! (`ref`, `ref / 4`, `1.5 ref`); a long random run is `#[ignore]`d. The tests hold in every
//! build: the kernel uses no multiply-add.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
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

/// The gain that the next sample is scaled with (a zero magnitude holds it).
fn gain_of(st: &AgcState<C32>) -> f32 {
    let mut probe = *st;
    let mut g = [0.0];
    probe.gains(&[0.0], &mut g).unwrap();
    g[0]
}

/// A signal whose every sample puts the output level at or next to `ref`, `lo` or `hi` of the
/// gain it meets (the fallback follows it while it is built), on and off the axes, with tiny,
/// huge and special samples among them.
fn bounds(params: (f64, f64, f64, f64, f64, f64), n: usize, rng: &mut Rng) -> Vec<C32> {
    let reference = params.0 as f32;
    let targets = [reference, reference * 0.25, reference * 1.5];
    let specials = [
        C32::new(0.0, 0.0),
        C32::new(f32::NAN, 0.0),
        C32::new(f32::INFINITY, 1.0),
        C32::new(1e-23, 0.0),
        C32::new(5e-23, 5e-23),
        C32::new(3e19, 3e19),
        C32::new(1e-40, 0.0),
    ];
    let mut st = state(params);
    let mut x = Vec::with_capacity(n);
    let mut y = [C32::default()];
    while x.len() < n {
        let s = if rng.below(50) == 0 {
            specials[rng.below(specials.len())]
        } else {
            let m = targets[rng.below(3)] / gain_of(&st);
            let k = rng.below(7) as i32 - 3;
            let m = f32::from_bits(m.to_bits().wrapping_add_signed(k));
            match rng.below(3) {
                0 => C32::new(m, 0.0),
                1 => C32::new(0.0, -m),
                _ => C32::new(m * 0.6, -m * 0.8),
            }
        };
        Fallback.agc(&mut st, &[s], &mut y).unwrap();
        x.push(s);
    }
    x
}

#[test]
fn c32_is_bit_exact_at_the_branch_bounds() {
    let mut rng = Rng::new(2005);
    for (k, &params) in PARAMS.iter().enumerate() {
        let x = bounds(params, 3000, &mut rng);
        check(params, &x, &format!("bounds params {k}"));
        for off in OFFSETS {
            check(
                params,
                &x[off..off + 777],
                &format!("bounds params {k} offset {off}"),
            );
        }
    }
}

/// About 16 million samples per parameter set, in random chunks; run in release:
/// `cargo test -p sdrk-backend-simd --release --test agc -- --ignored`.
#[test]
#[ignore = "long; run in release"]
fn c32_is_bit_exact_on_long_random_signals() {
    let mut rng = Rng::new(2006);
    for &params in &PARAMS {
        let (mut sw, mut sg) = (state(params), state(params));
        for round in 0..64 {
            let mut x = if round % 4 == 0 {
                bounds(params, 1 << 18, &mut rng)
            } else {
                steps(&mut rng, 1 << 18)
            };
            // Amplitudes over many decades, and zeros and NaN mixed in.
            let scale = 10f32.powi(rng.below(13) as i32 - 6);
            for s in &mut x {
                *s *= scale;
                match rng.below(1000) {
                    0 => *s = C32::new(f32::NAN, 0.0),
                    1 => *s = C32::new(0.0, 0.0),
                    _ => {}
                }
            }
            let (mut want, mut got) =
                (vec![C32::default(); x.len()], vec![C32::default(); x.len()]);
            Fallback.agc(&mut sw, &x, &mut want).unwrap();
            let mut done = 0;
            while done < x.len() {
                let n = (1 + rng.below(5000)).min(x.len() - done);
                Simd.agc(&mut sg, &x[done..done + n], &mut got[done..done + n])
                    .unwrap();
                done += n;
            }
            assert!(same(&want, &got), "round {round}");
            assert_eq!(sw, sg, "round {round}");
        }
    }
}
