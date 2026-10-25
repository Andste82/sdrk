//! `mag` of `C32` and `CQ15` against the fallback: bit-exact at every length from 0 to 4096, at
//! every start offset within a vector, for the extreme values, and for the sums of squares
//! around every perfect square and every rounding boundary `r (r + 1)` of the integer square
//! root. The exhaustive sweep over all 2^32 raw `CQ15` pairs is ignored (run with
//! `--release -- --ignored`; nightly in CI). The tests hold in every build, the relaxed-SIMD
//! build of wasm32 included: `mag` uses no multiply-add and the square root is correctly rounded.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation
)]

mod common;

use common::{Rng, c32_noise, cq15_noise};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::spectrum::Mag;
use sdrk_core::num::isqrt_u64;
use sdrk_core::{C32, CQ15, Fallback, Q15};

/// The longest length; every length up to it is tested.
const MAX_LEN: usize = 4096;

/// Start offsets in samples: the kernels must not depend on the alignment of the buffers.
const OFFSETS: [usize; 4] = [0, 1, 2, 3];

/// Equal bits, or both NaN (the payload of a NaN result is not specified).
fn same_f32(want: &[f32], got: &[f32]) -> bool {
    want.iter()
        .zip(got)
        .all(|(w, g)| w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()))
}

fn check_c32(x: &[C32], what: &str) {
    let (mut want, mut got) = (vec![0f32; x.len()], vec![0f32; x.len()]);
    Fallback.mag(x, &mut want).unwrap();
    Simd.mag(x, &mut got).unwrap();
    assert!(same_f32(&want, &got), "{what}: {want:?} {got:?}");
}

fn check_cq15(x: &[CQ15], what: &str) {
    let (mut want, mut got) = (vec![Q15(0); x.len()], vec![Q15(0); x.len()]);
    Fallback.mag(x, &mut want).unwrap();
    Simd.mag(x, &mut got).unwrap();
    assert_eq!(want, got, "{what}");
}

/// `round(sqrt(re^2 + im^2))` by the rule of the specification, saturated.
fn rule(re: i16, im: i16) -> i16 {
    let s = (i64::from(re) * i64::from(re) + i64::from(im) * i64::from(im)) as u64;
    let r = isqrt_u64(s);
    (r + u64::from(s - r * r > r)).min(32767) as i16
}

fn cq15(re: i16, im: i16) -> CQ15 {
    CQ15 {
        re: Q15(re),
        im: Q15(im),
    }
}

#[test]
fn c32_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(142);
    let x = c32_noise(&mut rng, MAX_LEN + OFFSETS.len(), 4.0);
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        check_c32(&x[off..off + n], &format!("n={n} offset={off}"));
    }
}

#[test]
fn cq15_is_bit_exact_at_every_length_and_offset() {
    let mut rng = Rng::new(143);
    let x = cq15_noise(&mut rng, MAX_LEN + OFFSETS.len());
    for n in 0..=MAX_LEN {
        let off = OFFSETS[n % OFFSETS.len()];
        check_cq15(&x[off..off + n], &format!("n={n} offset={off}"));
    }
}

#[test]
fn c32_extremes_are_bit_exact() {
    let v = [
        0.0,
        -0.0,
        1e-45,
        -1e-40,
        1e-20,
        1.5,
        -3.0,
        1e19,
        1.8e19,
        1.9e19,
        -3e38,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    // Every pair, repeated so that it passes through the vector loops as well as the tail.
    let x: Vec<C32> = v
        .iter()
        .flat_map(|&re| v.iter().map(move |&im| C32::new(re, im)))
        .collect();
    let x: Vec<C32> = x.iter().chain(&x).chain(&x).copied().collect();
    for off in OFFSETS {
        check_c32(&x[off..], &format!("offset={off}"));
    }
}

#[test]
fn c32_is_bit_exact_for_random_bit_patterns() {
    let mut rng = Rng::new(144);
    let x: Vec<C32> = (0..MAX_LEN + 3)
        .map(|_| {
            C32::new(
                f32::from_bits(rng.next_u32()),
                f32::from_bits(rng.next_u32()),
            )
        })
        .collect();
    for off in OFFSETS {
        check_c32(&x[off..], &format!("offset={off}"));
    }
}

#[test]
fn cq15_is_bit_exact_for_every_real_part() {
    // Every real part against the imaginary parts at the ends of the range and around the
    // saturation of the sum.
    for im in [-32768, -32767, -23171, -1, 0, 1, 181, 23170, 32766, 32767] {
        let x: Vec<CQ15> = (i16::MIN..=i16::MAX).map(|re| cq15(re, im)).collect();
        check_cq15(&x, &format!("im={im}"));
        check_cq15(&x[1..], &format!("im={im} offset=1"));
    }
}

#[test]
fn cq15_equals_the_rule_for_every_real_part_and_a_strided_imaginary_part() {
    // The rule itself, not only the fallback: 65536 real parts against 257 imaginary parts.
    let mut got = vec![Q15(0); 65536];
    for im in (i16::MIN..=i16::MAX).step_by(255) {
        let x: Vec<CQ15> = (i16::MIN..=i16::MAX).map(|re| cq15(re, im)).collect();
        Simd.mag(&x, &mut got).unwrap();
        for (c, g) in x.iter().zip(&got) {
            assert_eq!(g.0, rule(c.re.0, c.im.0), "{c:?}");
        }
    }
}

#[test]
fn cq15_rounds_correctly_around_every_boundary_of_the_root() {
    // Samples whose sum of squares `a^2 + b^2` is just below, at and just above `r^2` and
    // `r (r + 1)` (where the rounded root steps), for every root up to the saturation: `a` is
    // the root of the sum and `b` the root of the rest, kept where the pair is exact.
    let mut x = Vec::new();
    for r in 0..=46_341u64 {
        for s in [r * r, r * r + r, r * r + r + 1] {
            for s in [s.saturating_sub(1), s, s + 1] {
                for a in [isqrt_u64(s), isqrt_u64(s) / 2] {
                    let rest = s - a * a;
                    let b = isqrt_u64(rest);
                    if b * b == rest && a <= 32768 && b <= 32768 {
                        x.push(cq15(-(a as i32).min(32768) as i16, b.min(32767) as i16));
                        x.push(cq15(a.min(32767) as i16, -(b as i32) as i16));
                    }
                }
            }
        }
    }
    assert!(x.len() > 10_000, "{}", x.len());
    let mut got = vec![Q15(0); x.len()];
    Simd.mag(&x, &mut got).unwrap();
    for (c, g) in x.iter().zip(&got) {
        assert_eq!(g.0, rule(c.re.0, c.im.0), "{c:?}");
    }
    check_cq15(&x, "boundaries");
    check_cq15(&x[1..], "boundaries, offset 1");
}

#[test]
fn cq15_corners_saturate() {
    for (re, im) in [
        (-32768, -32768),
        (-32768, 32767),
        (32767, 32767),
        (32767, -32768),
        (23170, 23170),
        (0, -32768),
    ] {
        let x = vec![cq15(re, im); 67];
        let mut got = vec![Q15(0); x.len()];
        Simd.mag(&x, &mut got).unwrap();
        assert!(
            got.iter().all(|q| q.0 == rule(re, im)),
            "{re} {im}: {:?}",
            got[0]
        );
        check_cq15(&x, "corner");
    }
}

/// The part `k` of `n` of the exhaustive test (`SDRK_SWEEP_PART=k/n`, default `0/1`, all of
/// it): the chunks whose index is `k` modulo `n`.
fn sweep_part() -> (u32, u32) {
    let Ok(part) = std::env::var("SDRK_SWEEP_PART") else {
        return (0, 1);
    };
    let parsed = part
        .split_once('/')
        .and_then(|(k, n)| Some((k.parse::<u32>().ok()?, n.parse::<u32>().ok()?)))
        .filter(|&(k, n)| k < n);
    parsed.unwrap_or_else(|| panic!("SDRK_SWEEP_PART must be k/n with k < n, not {part:?}"))
}

#[test]
#[ignore = "all 2^32 raw CQ15 pairs: minutes; run with --release -- --ignored"]
fn mag_cq15_exhaustive_every_raw_pair() {
    // The sample with raw bits `b` has `re = b & 0xffff` and `im = b >> 16`; a chunk is 2^20
    // consecutive values of `b`, so every chunk holds 16 values of `im` against all `re`.
    const CHUNK: u32 = 1 << 20;
    let (part, parts) = sweep_part();
    let mut input = vec![cq15(0, 0); CHUNK as usize];
    let (mut want, mut got) = (vec![Q15(0); CHUNK as usize], vec![Q15(0); CHUNK as usize]);
    for index in (part..=u32::MAX / CHUNK).step_by(parts as usize) {
        let start = index * CHUNK;
        for (k, c) in input.iter_mut().enumerate() {
            let b = start + k as u32;
            *c = cq15(b as u16 as i16, (b >> 16) as u16 as i16);
        }
        Fallback.mag(&input, &mut want).unwrap();
        Simd.mag(&input, &mut got).unwrap();
        if let Some(k) = want.iter().zip(&got).position(|(w, g)| w != g) {
            panic!("{:?}: fallback {:?}, simd {:?}", input[k], want[k], got[k]);
        }
        // The rule itself on a stride, so that an error common to both would show.
        for (c, g) in input.iter().zip(&got).step_by(251) {
            assert_eq!(g.0, rule(c.re.0, c.im.0), "{c:?}");
        }
    }
}
