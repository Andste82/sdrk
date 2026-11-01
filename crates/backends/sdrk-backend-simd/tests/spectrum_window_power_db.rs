//! `window` of `C32` and `CQ15` and `power_db` of `f32` and `Q15` against the fallback: bit-exact
//! at every length from 0 to 2048 (all vector tails of every level), at every start offset
//! within a vector, for the extreme values and for every raw value where the domain is small
//! (all `Q15` powers, all `Q15` products with a set of weights, `f32` powers over every binade).
//! The tests hold in every build: the kernels use no multiply-add.
#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

mod common;

use common::{Rng, c32_noise, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::spectrum::{PowerDb, Window};
use sdrk_core::{C32, CQ15, Error, Fallback, Q15};

/// The longest length; every length up to it is tested.
const MAX_LEN: usize = 2048;

/// Start offsets in samples: the kernels must not depend on the alignment of the buffers.
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

/// Equal bits, or both NaN (the payload of a NaN result is not specified).
fn same_f32(want: &[f32], got: &[f32]) -> bool {
    want.iter()
        .zip(got)
        .all(|(w, g)| w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()))
}

fn check_window_c32(x: &[C32], w: &[f32], what: &str) {
    let (mut want, mut got) = (
        vec![C32::new(0.0, 0.0); x.len()],
        vec![C32::new(0.0, 0.0); x.len()],
    );
    Fallback.window(x, &mut want, w).unwrap();
    Simd.window(x, &mut got, w).unwrap();
    let flat = |v: &[C32]| -> Vec<f32> { v.iter().flat_map(|c| [c.re, c.im]).collect() };
    assert!(same_f32(&flat(&want), &flat(&got)), "{what}");
}

fn check_window_cq15(x: &[CQ15], w: &[Q15], what: &str) {
    let zero = CQ15 {
        re: Q15(0),
        im: Q15(0),
    };
    let (mut want, mut got) = (vec![zero; x.len()], vec![zero; x.len()]);
    Fallback.window(x, &mut want, w).unwrap();
    Simd.window(x, &mut got, w).unwrap();
    assert_eq!(want, got, "{what}");
}

fn check_power_db_f32(p: &[f32], offset: f32, what: &str) {
    let (mut want, mut got) = (vec![0f32; p.len()], vec![0f32; p.len()]);
    Fallback.power_db(p, &mut want, offset).unwrap();
    Simd.power_db(p, &mut got, offset).unwrap();
    let bad = (0..p.len()).find(|&i| !same_f32(&want[i..=i], &got[i..=i]));
    assert!(
        bad.is_none(),
        "{what}: {:?}",
        bad.map(|i| (p[i], want[i], got[i]))
    );
}

fn check_power_db_q15(p: &[Q15], offset: f32, what: &str) {
    let (mut want, mut got) = (vec![0f32; p.len()], vec![0f32; p.len()]);
    Fallback.power_db(p, &mut want, offset).unwrap();
    Simd.power_db(p, &mut got, offset).unwrap();
    assert!(same_f32(&want, &got), "{what}");
}

#[test]
fn window_c32_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(150);
    let x = c32_noise(&mut rng, MAX_LEN + OFFSETS.len(), 4.0);
    let w: Vec<f32> = (0..MAX_LEN + OFFSETS.len())
        .map(|_| rng.unit() * 2.0 - 0.5)
        .collect();
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let woff = OFFSETS[(n / 4) % OFFSETS.len()];
        check_window_c32(
            &x[off..off + n],
            &w[woff..woff + n],
            &format!("n={n} offset={off}"),
        );
    }
}

#[test]
fn window_cq15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(151);
    let x = cq15_noise(&mut rng, MAX_LEN + OFFSETS.len());
    let w: Vec<Q15> = cq15_noise(&mut rng, MAX_LEN + OFFSETS.len())
        .iter()
        .map(|c| c.re)
        .collect();
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        let woff = OFFSETS[(n / 4) % OFFSETS.len()];
        check_window_cq15(
            &x[off..off + n],
            &w[woff..woff + n],
            &format!("n={n} offset={off}"),
        );
    }
}

#[test]
fn window_c32_extremes_are_bit_exact() {
    let v = [
        0.0,
        -0.0,
        1e-45,
        -1e-40,
        1.5,
        -3.0,
        3e38,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    // Every pair of values with every weight, long enough for the vector loops and the tail.
    let mut x = Vec::new();
    let mut w = Vec::new();
    for &re in &v {
        for &im in &v {
            for &g in &v {
                x.push(C32::new(re, im));
                w.push(g);
            }
        }
    }
    check_window_c32(&x, &w, "extremes");
}

#[test]
fn window_cq15_every_value_with_extreme_weights_is_bit_exact() {
    // Every raw value as real and imaginary part (reversed), against the weights that reach
    // the saturation, the rounding ties and the sign changes.
    let weights = [i16::MIN, -32767, -16384, -1, 0, 1, 2, 16384, 32766, 32767];
    let x: Vec<CQ15> = (i16::MIN..=i16::MAX)
        .map(|r| CQ15 {
            re: Q15(r),
            im: Q15(r.wrapping_neg().wrapping_sub(1)),
        })
        .collect();
    for &g in &weights {
        let w = vec![Q15(g); x.len()];
        check_window_cq15(&x, &w, &format!("weight {g}"));
        // The weight varying per sample: the duplication of the weights for both parts.
        let w: Vec<Q15> = (0..x.len())
            .map(|i| Q15(g.wrapping_add(i as i16)))
            .collect();
        check_window_cq15(&x, &w, &format!("weights from {g}"));
    }
}

#[test]
fn window_length_errors_match_the_fallback() {
    let x = [C32::new(1.0, 2.0); 20];
    let mut out = [C32::new(0.0, 0.0); 20];
    let w = [0.5f32; 19];
    assert_eq!(
        Simd.window(&x, &mut out[..19], &w[..19]),
        Err(Error::LengthMismatch {
            expected: 20,
            got: 19
        })
    );
    assert_eq!(
        Simd.window(&x, &mut out, &w),
        Fallback.window(&x, &mut out, &w)
    );
    let xq = [CQ15 {
        re: Q15(1),
        im: Q15(2),
    }; 20];
    let mut outq = [CQ15 {
        re: Q15(0),
        im: Q15(0),
    }; 20];
    let wq = [Q15(7); 21];
    assert_eq!(
        Simd.window(&xq, &mut outq, &wq),
        Fallback.window(&xq, &mut outq, &wq)
    );
}

#[test]
fn power_db_f32_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(152);
    // Powers over many decades, zeros and negative values among them.
    let p: Vec<f32> = (0..MAX_LEN + OFFSETS.len())
        .map(|i| match i % 17 {
            0 => 0.0,
            1 => -rng.unit(),
            _ => rng.unit() * 10f32.powi(rng.below(60) as i32 - 30),
        })
        .collect();
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        check_power_db_f32(&p[off..off + n], -3.5, &format!("n={n} offset={off}"));
    }
}

#[test]
fn power_db_f32_every_binade_is_bit_exact() {
    // A sweep of the mantissa (every 509th pattern) in every binade from below the floor to the
    // largest, plus every pattern near 1.0 (where `f` and the polynomial are smallest).
    let mut p: Vec<f32> = Vec::new();
    for e in 0u32..=255 {
        for m in (0u32..1 << 23).step_by(509) {
            p.push(f32::from_bits(e << 23 | m));
        }
    }
    for b in 0x3f7f_0000u32..=0x3f81_0000 {
        p.push(f32::from_bits(b));
    }
    for offset in [0.0, 30.0, -174.0] {
        check_power_db_f32(&p, offset, &format!("offset {offset}"));
    }
}

#[test]
fn power_db_f32_extremes_are_bit_exact() {
    let mut p = vec![
        0.0,
        -0.0,
        1e-45,
        1e-38,
        9.99e-21,
        1e-20,
        1.000_001e-20,
        1.0,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        f32::NAN,
        -f32::NAN,
    ];
    // Repeated so that each value passes through the vector loop as well as the tail.
    let x: Vec<f32> = p.iter().cycle().take(p.len() * 7).copied().collect();
    p.extend(x);
    for offset in [
        0.0,
        6.02,
        -200.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ] {
        check_power_db_f32(&p, offset, &format!("offset {offset}"));
    }
}

#[test]
fn power_db_f32_every_reduced_mantissa_is_bit_exact() {
    // The reduction depends on the mantissa only, and the binades [sqrt(2)/2, 1) and
    // [1, sqrt(2)) hold every reduced value `m` once: every pattern of [0.5, 2) covers all of
    // them (and the shift of `k` at sqrt(2)) for both exponents.
    let p: Vec<f32> = (0x3f00_0000u32..0x4000_0000).map(f32::from_bits).collect();
    check_power_db_f32(&p, -1.25, "mantissas of [0.5, 2)");
}

#[test]
fn power_db_f32_special_value_at_every_position_is_bit_exact() {
    // One special value at each position of each length up to 80 samples: the prologue, the
    // steady loop and the drain of the pipelined and blocked forms at every level, and the tail.
    let base: Vec<f32> = (0..80).map(|i| 10f32.powf(i as f32 / 7.0 - 5.0)).collect();
    for special in [
        f32::INFINITY,
        f32::NAN,
        0.0,
        -1.0,
        f32::MAX,
        f32::MIN_POSITIVE,
    ] {
        for n in 1..=base.len() {
            for pos in 0..n {
                let mut p = base[..n].to_vec();
                p[pos] = special;
                check_power_db_f32(&p, 3.0, &format!("{special} at {pos} of {n}"));
            }
        }
    }
}

#[test]
fn power_db_f32_is_chunk_invariant() {
    // A stateless kernel: the output of a buffer is the output of its pieces, cut anywhere.
    let mut rng = Rng::new(154);
    let p: Vec<f32> = (0..1000)
        .map(|_| rng.unit() * 10f32.powi(rng.below(40) as i32 - 20))
        .collect();
    let mut whole = vec![0f32; p.len()];
    Simd.power_db(&p, &mut whole, 7.5).unwrap();
    for cut in [1, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 999] {
        let mut pieces = vec![0f32; p.len()];
        for (i, o) in p.chunks(cut).zip(pieces.chunks_mut(cut)) {
            Simd.power_db(i, o, 7.5).unwrap();
        }
        assert!(same_f32(&whole, &pieces), "cut {cut}");
    }
}

#[test]
fn power_db_q15_every_value_is_bit_exact() {
    // Every raw value, shifted by one per round so that each passes every lane and the tail.
    let all: Vec<Q15> = (i16::MIN..=i16::MAX).map(Q15).collect();
    for shift in 0..3 {
        check_power_db_q15(&all[shift..], 12.5, &format!("shift {shift}"));
    }
    let mut rng = Rng::new(153);
    let p: Vec<Q15> = cq15_noise(&mut rng, MAX_LEN + OFFSETS.len())
        .iter()
        .map(|c| c.im)
        .collect();
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        check_power_db_q15(&p[off..off + n], 0.0, &format!("n={n} offset={off}"));
    }
}
