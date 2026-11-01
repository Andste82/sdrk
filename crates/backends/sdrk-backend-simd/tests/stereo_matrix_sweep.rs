//! `stereo_matrix` `CQ15` over every `(s, d)` pair of 16-bit values (2^32 pairs, ignored, run
//! with `--release -- --ignored`; nightly in CI), against the integer rule `sat16(s +- 2 d)`
//! itself, and a fast test of the boundary values in the normal run. The level paths (the
//! exact 32-bit sums of x86 and WASM, the saturating chains of NEON) equal the rule for every
//! pair, which is the proof of the argument in `stereo.rs`.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation
)]

use sdrk_backend_simd::Simd;
use sdrk_core::kernels::stereo::StereoMatrix;
use sdrk_core::{CQ15, Fallback, Q15};

fn rule(s: i16, d: i16) -> (i16, i16) {
    let (s, d2) = (i64::from(s), 2 * i64::from(d));
    (
        (s + d2).clamp(-32768, 32767) as i16,
        (s - d2).clamp(-32768, 32767) as i16,
    )
}

fn sample(s: i16, d: i16) -> CQ15 {
    CQ15 {
        re: Q15(s),
        im: Q15(d),
    }
}

/// The same split as `convert_out_sweep`: `SDRK_SWEEP_PART=k/n` takes the values of `s` that
/// are `k` modulo `n`; `SDRK_SWEEP_STRIDE=m` every `m`-th of them (for the emulators).
fn sweep_part() -> (u32, u32, u32) {
    let (k, n) = std::env::var("SDRK_SWEEP_PART")
        .ok()
        .and_then(|p| {
            let (k, n) = p.split_once('/')?;
            Some((k.parse::<u32>().ok()?, n.parse::<u32>().ok()?))
        })
        .filter(|&(k, n)| k < n)
        .unwrap_or((0, 1));
    let stride = std::env::var("SDRK_SWEEP_STRIDE")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(1);
    (k, n, stride)
}

#[test]
fn boundary_values_equal_the_rule_at_every_alignment() {
    let v = [
        i16::MIN,
        i16::MIN + 1,
        -16385,
        -16384,
        -16383,
        -2,
        -1,
        0,
        1,
        2,
        16383,
        16384,
        16385,
        i16::MAX - 1,
        i16::MAX,
    ];
    let mut x = Vec::new();
    for &s in &v {
        for &d in &v {
            x.push(sample(s, d));
        }
    }
    for off in 0..4 {
        for len in [x.len() - off, 33, 64, 65] {
            let x = &x[off..off + len.min(x.len() - off)];
            let mut got = vec![CQ15::default(); x.len()];
            Simd.stereo_matrix(x, &mut got).unwrap();
            for (i, g) in x.iter().zip(&got) {
                let (l, r) = rule(i.re.0, i.im.0);
                assert_eq!((g.re.0, g.im.0), (l, r), "s={} d={}", i.re.0, i.im.0);
            }
        }
    }
}

#[test]
#[ignore = "all 2^32 pairs: minutes; run with --release -- --ignored"]
fn every_pair_equals_the_rule() {
    let (part, parts, stride) = sweep_part();
    // One block per `s`: all 65536 values of `d`.
    let d: Vec<i16> = (0..=u16::MAX).map(|v| v as i16).collect();
    let mut input = vec![CQ15::default(); d.len()];
    let mut got = vec![CQ15::default(); d.len()];
    let mut want = vec![CQ15::default(); d.len()];
    for k in (part..=u32::from(u16::MAX)).step_by((parts * stride) as usize) {
        let s = k as u16 as i16;
        for (i, &d) in input.iter_mut().zip(&d) {
            *i = sample(s, d);
        }
        Simd.stereo_matrix(&input, &mut got).unwrap();
        Fallback.stereo_matrix(&input, &mut want).unwrap();
        for ((g, w), &d) in got.iter().zip(&want).zip(&d) {
            let (l, r) = rule(s, d);
            assert!(
                g == w && (g.re.0, g.im.0) == (l, r),
                "s={s} d={d}: got ({}, {}), rule ({l}, {r})",
                g.re.0,
                g.im.0
            );
        }
    }
}
