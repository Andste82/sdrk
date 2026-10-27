//! `biquad` of `C32` and `CQ15` against the fallback: bit-exact output and history at every
//! length from 0 to 300 (all tails of the four-sample steps), at every start offset within a
//! vector, for one to five sections, for stable and for unstable and resonant coefficients
//! (overflow to infinity and NaN in `C32`, saturation in `CQ15`), with zeros, NaN, infinities
//! and subnormal samples, in random chunks, and for the corners of the `CQ15` coefficient
//! range. The tests hold in every build: the kernels use no multiply-add.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

mod common;

use common::{Rng, c32_noise, cq, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::biquad::{Biquad, BiquadState};
use sdrk_core::{C32, CQ15, Fallback};

const MAX_LEN: usize = 300;
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

/// Equal bits, or both NaN (the payload of a NaN result is not specified).
fn same_c32(want: &[C32], got: &[C32]) -> bool {
    let bits = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    want.len() == got.len()
        && want
            .iter()
            .zip(got)
            .all(|(w, g)| bits(w.re, g.re) && bits(w.im, g.im))
}

/// A stable section: the second-order low-pass at `w` (fraction of the sample rate) with
/// damping `d`.
fn stable(w: f32, d: f32) -> [f32; 5] {
    let k = (w * core::f32::consts::PI).tan();
    let n = 1.0 / (1.0 + d * k + k * k);
    [
        k * k * n,
        2.0 * k * k * n,
        k * k * n,
        2.0 * (k * k - 1.0) * n,
        (1.0 - d * k + k * k) * n,
    ]
}

fn coefs_c32(rng: &mut Rng, sections: usize, wild: bool) -> Vec<[f32; 5]> {
    (0..sections)
        .map(|_| {
            if wild {
                [(); 5].map(|()| rng.unit() * 2.0)
            } else {
                stable(
                    0.02 + 0.2 * (rng.unit() * 0.5 + 0.5),
                    0.3 + rng.unit().abs(),
                )
            }
        })
        .collect()
}

/// `mode` 0: the corners of the range, 1: any `i16`, 2: stable Q2.14 low-passes.
fn coefs_cq15(rng: &mut Rng, sections: usize, mode: usize) -> Vec<[i16; 5]> {
    const EDGE: [i16; 5] = [i16::MIN, i16::MAX, 0, 1, -1];
    (0..sections)
        .map(|_| match mode {
            0 => [(); 5].map(|()| EDGE[rng.below(EDGE.len())]),
            1 => [(); 5].map(|()| rng.next_u32() as i16),
            _ => stable(0.02 + 0.2 * (rng.unit() * 0.5 + 0.5), 1.4)
                .map(|v| (v * 16384.0).round() as i16),
        })
        .collect()
}

fn check_c32(coefs: &[[f32; 5]], x: &[C32], what: &str) {
    let n = coefs.len();
    let new = || BiquadState::<C32, _, _>::new(coefs.to_vec(), vec![C32::default(); 4 * n]);
    let (mut sw, mut sg) = (new().unwrap(), new().unwrap());
    let (mut want, mut got) = (vec![C32::default(); x.len()], vec![C32::default(); x.len()]);
    Fallback.biquad(&mut sw, x, &mut want).unwrap();
    Simd.biquad(&mut sg, x, &mut got).unwrap();
    assert!(same_c32(&want, &got), "{what}: output differs");
    assert!(
        same_c32(sw.history(), sg.history()),
        "{what}: history differs"
    );
}

fn check_cq15(coefs: &[[i16; 5]], x: &[CQ15], what: &str) {
    let n = coefs.len();
    let new = || BiquadState::<CQ15, _, _>::new(coefs.to_vec(), vec![CQ15::default(); 4 * n]);
    let (mut sw, mut sg) = (new().unwrap(), new().unwrap());
    let (mut want, mut got) = (
        vec![CQ15::default(); x.len()],
        vec![CQ15::default(); x.len()],
    );
    Fallback.biquad(&mut sw, x, &mut want).unwrap();
    Simd.biquad(&mut sg, x, &mut got).unwrap();
    assert_eq!(want, got, "{what}: output differs");
    assert_eq!(sw.history(), sg.history(), "{what}: history differs");
}

#[test]
fn c32_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(2201);
    let x = c32_noise(&mut rng, MAX_LEN + OFFSETS.len(), 0.5);
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let sections = 1 + n % 5;
        let coefs = coefs_c32(&mut rng, sections, false);
        check_c32(
            &coefs,
            &x[off..off + n],
            &format!("n={n} offset={off} sections={sections}"),
        );
    }
}

#[test]
fn c32_is_bit_exact_for_wild_coefficients() {
    let mut rng = Rng::new(2202);
    let x = c32_noise(&mut rng, 600, 1.0);
    for round in 0..40 {
        let sections = 1 + round % 4;
        let coefs = coefs_c32(&mut rng, sections, true);
        check_c32(&coefs, &x[..(round * 15) % 600], &format!("round {round}"));
        check_c32(&coefs, &x, &format!("round {round} full"));
    }
}

#[test]
fn c32_is_bit_exact_with_zeros_nan_infinity_and_subnormals() {
    let mut rng = Rng::new(2203);
    let base = c32_noise(&mut rng, 301, 0.5);
    let specials = [
        C32::new(0.0, 0.0),
        C32::new(-0.0, 0.0),
        C32::new(f32::NAN, 0.1),
        C32::new(0.1, f32::NAN),
        C32::new(f32::INFINITY, 0.0),
        C32::new(0.0, f32::NEG_INFINITY),
        C32::new(1e-40, 0.0),
        C32::new(1e-45, -1e-45),
        C32::new(1e-31, 1e-29),
        C32::new(3e19, 3e19),
        C32::new(f32::MAX, f32::MAX),
    ];
    for &special in &specials {
        let patterns: [&dyn Fn(usize) -> bool; 4] = [
            &|i| i == 100,
            &|i| i == 300,
            &|i| (64..128).contains(&i),
            &|i| i % 4 == 1,
        ];
        for (p, pattern) in patterns.iter().enumerate() {
            let x: Vec<C32> = base
                .iter()
                .enumerate()
                .map(|(i, &s)| if pattern(i) { special } else { s })
                .collect();
            for sections in [1, 2, 3] {
                let coefs = coefs_c32(&mut rng, sections, false);
                check_c32(
                    &coefs,
                    &x,
                    &format!("{special:?} pattern {p} sections {sections}"),
                );
            }
        }
    }
    let coefs = coefs_c32(&mut rng, 2, false);
    check_c32(&coefs, &[C32::new(0.0, 0.0); 200], "silence");
    check_c32(&coefs, &[C32::new(1e-33, -1e-33); 200], "faint");
}

#[test]
fn c32_is_bit_exact_in_random_chunks() {
    let mut rng = Rng::new(2204);
    let x = c32_noise(&mut rng, 3000, 0.5);
    for sections in 1..=4 {
        let coefs = coefs_c32(&mut rng, sections, false);
        let new = || {
            BiquadState::<C32, _, _>::new(coefs.clone(), vec![C32::default(); 4 * sections])
                .unwrap()
        };
        let (mut sw, mut sg) = (new(), new());
        let (mut want, mut got) = (vec![C32::default(); x.len()], vec![C32::default(); x.len()]);
        Fallback.biquad(&mut sw, &x, &mut want).unwrap();
        let mut done = 0;
        while done < x.len() {
            let c = (1 + rng.below(70)).min(x.len() - done);
            Simd.biquad(&mut sg, &x[done..done + c], &mut got[done..done + c])
                .unwrap();
            done += c;
        }
        assert!(same_c32(&want, &got));
        assert!(same_c32(sw.history(), sg.history()));
    }
}

#[test]
fn cq15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(2205);
    let x = cq15_noise(&mut rng, MAX_LEN + OFFSETS.len());
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let sections = 1 + n % 5;
        let coefs = coefs_cq15(&mut rng, sections, 2);
        check_cq15(
            &coefs,
            &x[off..off + n],
            &format!("n={n} offset={off} sections={sections}"),
        );
    }
}

#[test]
fn cq15_is_bit_exact_for_every_coefficient_range() {
    let mut rng = Rng::new(2206);
    let noise = cq15_noise(&mut rng, 700);
    let corners: Vec<CQ15> = (0..700)
        .map(|i| match i % 4 {
            0 => cq(i16::MAX, i16::MIN),
            1 => cq(i16::MIN, i16::MIN),
            2 => cq(i16::MAX, i16::MAX),
            _ => cq(0, if i % 3 == 0 { 1 } else { -1 }),
        })
        .collect();
    for mode in 0..3 {
        for round in 0..30 {
            let sections = 1 + round % 4;
            let coefs = coefs_cq15(&mut rng, sections, mode);
            check_cq15(&coefs, &noise, &format!("mode {mode} round {round} noise"));
            check_cq15(
                &coefs,
                &corners,
                &format!("mode {mode} round {round} corners"),
            );
            check_cq15(
                &coefs,
                &noise[..(round * 23) % 700],
                &format!("mode {mode} round {round} short"),
            );
        }
    }
}

#[test]
fn cq15_is_bit_exact_on_silence_dc_and_the_largest_terms() {
    let mut rng = Rng::new(2207);
    let coefs = coefs_cq15(&mut rng, 2, 2);
    check_cq15(&coefs, &[CQ15::default(); 300], "silence");
    check_cq15(&coefs, &[cq(20000, -20000); 300], "dc");
    check_cq15(&coefs, &[cq(-32768, -32768); 300], "negative full scale");
    // Five products of the largest magnitude add up to the largest accumulator.
    let largest = [[i16::MIN, i16::MIN, i16::MIN, i16::MAX, i16::MAX]];
    let x: Vec<CQ15> = (0..200)
        .map(|i| cq(if i % 2 == 0 { i16::MIN } else { i16::MAX }, i16::MIN))
        .collect();
    check_cq15(&largest, &x, "largest terms");
}

#[test]
fn cq15_is_bit_exact_in_random_chunks() {
    let mut rng = Rng::new(2208);
    let x = cq15_noise(&mut rng, 3000);
    for sections in 1..=4 {
        let coefs = coefs_cq15(&mut rng, sections, 2);
        let new = || {
            BiquadState::<CQ15, _, _>::new(coefs.clone(), vec![CQ15::default(); 4 * sections])
                .unwrap()
        };
        let (mut sw, mut sg) = (new(), new());
        let (mut want, mut got) = (
            vec![CQ15::default(); x.len()],
            vec![CQ15::default(); x.len()],
        );
        Fallback.biquad(&mut sw, &x, &mut want).unwrap();
        let mut done = 0;
        while done < x.len() {
            let c = (1 + rng.below(70)).min(x.len() - done);
            Simd.biquad(&mut sg, &x[done..done + c], &mut got[done..done + c])
                .unwrap();
            done += c;
        }
        assert_eq!(want, got);
        assert_eq!(sw.history(), sg.history());
    }
}

/// The rounding and saturation of the `f64` arithmetic against the integers: one section with
/// `y = round_shift(b0 x, 14)` for every input value (re and im together, the im part runs
/// through the values in the opposite order) and a set of `b0` that puts the products on and
/// next to every rounding tie (`b0 x = 2^13 mod 2^14`) and beyond the saturation limits.
/// `cq15_sweeps_every_coefficient_of_one_tap` (ignored, nightly) runs all 65536 values of `b0`.
fn sweep_one_tap(b0_values: impl Iterator<Item = i16>) {
    let x: Vec<CQ15> = (0..=u16::MAX)
        .map(|v| cq(v as i16, (u16::MAX - v) as i16))
        .collect();
    for b0 in b0_values {
        check_cq15(&[[b0, 0, 0, 0, 0]], &x, &format!("b0 = {b0}"));
    }
}

#[test]
fn cq15_sweeps_the_rounding_ties_and_the_saturation_of_one_tap() {
    let mut b0s: Vec<i16> = vec![
        i16::MIN,
        i16::MIN + 1,
        -16385,
        -16384,
        -16383,
        -1,
        0,
        1,
        2,
        3,
    ];
    b0s.extend([
        8191,
        8192,
        8193,
        16383,
        16384,
        16385,
        24575,
        i16::MAX - 1,
        i16::MAX,
    ]);
    for shift in 0..15 {
        let p = 1_i16 << shift;
        b0s.extend([p, p - 1, p + 1, -p]);
    }
    b0s.sort_unstable();
    b0s.dedup();
    sweep_one_tap(b0s.into_iter());
}

/// The part `k` of `n` of the exhaustive test (`SDRK_SWEEP_PART=k/n`, default `0/1`, all of
/// it): the coefficients whose index is `k` modulo `n`; `SDRK_SWEEP_STRIDE=s` (default 1) takes
/// only every `s`-th of them, for the emulators.
fn sweep_part() -> (usize, usize) {
    let Ok(part) = std::env::var("SDRK_SWEEP_PART") else {
        return (0, 1);
    };
    let parsed = part
        .split_once('/')
        .and_then(|(k, n)| Some((k.parse::<usize>().ok()?, n.parse::<usize>().ok()?)))
        .filter(|&(k, n)| k < n);
    parsed.unwrap_or_else(|| panic!("SDRK_SWEEP_PART must be k/n with k < n, not {part:?}"))
}

/// About 4.3 * 10^9 samples; run in release:
/// `cargo test -p sdrk-backend-simd --release --test biquad -- --ignored`.
#[test]
#[ignore = "long; run in release"]
fn cq15_sweeps_every_coefficient_of_one_tap() {
    let (part, parts) = sweep_part();
    let stride = std::env::var("SDRK_SWEEP_STRIDE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1);
    sweep_one_tap((i16::MIN..=i16::MAX).skip(part).step_by(parts * stride));
}
