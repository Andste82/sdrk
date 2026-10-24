//! The generic `CQ15` FFT of the SIMD backend (the levels without a level-specific path, and the
//! sizes below it on the levels with one) against the fallback, bit for bit and exponent
//! included: every size, both directions and scalings, and inputs that drive every
//! block-floating-point shift, saturate the outputs and use the extreme twiddles. The
//! level-specific paths are within the tolerance of the contract instead
//! (`fft_q15_accuracy.rs`).
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]

mod common;

use common::{Rng, cq, cq15_noise, cq15_plan as plan};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::{CQ15, Fallback};

/// The sizes below this one run the generic code at the level of this build: the level-specific
/// paths need a tile of `V x V` samples and four vectors (`V` samples per vector) on x86 and
/// wasm, and 64 samples on NEON.
fn generic_below() -> usize {
    match sdrk_backend_simd::level_name() {
        "scalar" => usize::MAX,
        "avx2" | "neon" => 64,
        "avx512" => 256,
        _ => 16,
    }
}

/// The sizes `2^log` of `logs` that run the generic code.
fn generic_sizes(logs: impl Iterator<Item = u32>) -> impl Iterator<Item = usize> {
    logs.map(|log| 1usize << log)
        .filter(|&n| n < generic_below())
}

/// Inputs of size `n`: full-scale noise, the four full-scale corners, a full-scale tone (whose
/// energy collects in one bin, so that block floating point shifts by 1 almost every stage), an
/// impulse, small noise (no shift until late), and noise of every magnitude class of the
/// block-floating-point thresholds.
fn inputs(rng: &mut Rng, n: usize) -> Vec<Vec<CQ15>> {
    let mut v = vec![
        cq15_noise(rng, n),
        vec![cq(-32768, -32768); n],
        vec![cq(32767, 32767); n],
        vec![cq(-32768, 32767); n],
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    cq(32767, -32768)
                } else {
                    cq(-32768, 32767)
                }
            })
            .collect(),
        (0..n)
            .map(|i| {
                let th = 2.0 * std::f64::consts::PI * ((7 * i) % n) as f64 / n as f64;
                cq((32767.0 * th.cos()) as i16, (32767.0 * th.sin()) as i16)
            })
            .collect(),
        (0..n)
            .map(|i| if i == 1 { cq(-32768, 32767) } else { cq(0, 0) })
            .collect(),
        (0..n)
            .map(|_| {
                cq(
                    (rng.next_u32() % 64) as i16 - 32,
                    (rng.next_u32() % 64) as i16 - 32,
                )
            })
            .collect(),
        vec![CQ15::default(); n],
    ];
    for amp in [13_572i32, 13_573, 23_170, 27_145, 27_146] {
        v.push(
            (0..n)
                .map(|_| {
                    let r = |rng: &mut Rng| (rng.next_u32() % (2 * amp as u32 + 1)) as i32 - amp;
                    cq(r(rng) as i16, r(rng) as i16)
                })
                .collect(),
        );
    }
    v
}

fn check(plan: &FftPlan<CQ15, Vec<CQ15>>, x: &[CQ15], what: &str) {
    for dir in [Direction::Forward, Direction::Inverse] {
        for scaling in [Scaling::Default, Scaling::BlockFloat] {
            let (mut want, mut got) = (x.to_vec(), x.to_vec());
            let sw = Fallback.fft(plan, &mut want, dir, scaling).unwrap();
            let sg = Simd.fft(plan, &mut got, dir, scaling).unwrap();
            let n = x.len();
            assert_eq!(sw, sg, "{what} n={n} {dir:?} {scaling:?}");
            let first = (0..n).find(|&i| want[i] != got[i]);
            assert!(
                first.is_none(),
                "{what} n={n} {dir:?} {scaling:?}: first difference at {first:?}"
            );
        }
    }
}

#[test]
fn fft_cq15_is_bit_exact_at_every_size() {
    let mut rng = Rng::new(1501);
    for n in generic_sizes(4..=14) {
        let p = plan(n);
        for (i, x) in inputs(&mut rng, n).iter().enumerate() {
            check(&p, x, &format!("input {i}"));
        }
    }
}

#[test]
fn fft_cq15_is_bit_exact_at_the_largest_sizes() {
    let mut rng = Rng::new(1502);
    for n in generic_sizes(15..=16) {
        let p = plan(n);
        check(&p, &cq15_noise(&mut rng, n), "noise");
        check(&p, &vec![cq(-32768, 32767); n], "corner");
    }
}

#[test]
fn fft_cq15_repeated_calls_stay_bit_exact() {
    // The benchmark transforms the output again and again; the data then wanders through every
    // magnitude class.
    let mut rng = Rng::new(1503);
    for n in generic_sizes([4, 5, 6, 10].into_iter()) {
        let p = plan(n);
        for scaling in [Scaling::Default, Scaling::BlockFloat] {
            let mut want = cq15_noise(&mut rng, n);
            let mut got = want.clone();
            for call in 0..12 {
                let dir = if call % 3 == 2 {
                    Direction::Inverse
                } else {
                    Direction::Forward
                };
                let sw = Fallback.fft(&p, &mut want, dir, scaling).unwrap();
                let sg = Simd.fft(&p, &mut got, dir, scaling).unwrap();
                assert_eq!(sw, sg, "n={n} {scaling:?} call {call}");
                assert_eq!(want, got, "n={n} {scaling:?} call {call}");
            }
        }
    }
}

#[test]
fn fft_cq15_subsampled_tables_are_bit_exact() {
    // The table of a larger plan, every `stride`-th entry: the values of the static tables.
    let big = plan(4096);
    let mut rng = Rng::new(1504);
    for n in generic_sizes([4, 5, 7, 8, 11].into_iter()) {
        let table: Vec<CQ15> = big.twiddles().iter().step_by(4096 / n).copied().collect();
        let p = FftPlan::<CQ15, _>::from_twiddles(n, table).unwrap();
        check(&p, &cq15_noise(&mut rng, n), "subsampled");
    }
}

#[test]
fn fft_cq15_twiddles_stay_inside_the_unit_octagon() {
    // The generic stages rely on `|wr| + |wi| <= 46341`: the products and sums of a butterfly
    // fit `i32`. No twiddle has both parts at -32768 (but `wr = -32768` occurs from n = 2048 on).
    for log in 4..=16 {
        let n = 1usize << log;
        for w in plan(n).twiddles() {
            let (re, im) = (i32::from(w.re.0), i32::from(w.im.0));
            assert!(re.abs() + im.abs() <= 46_341, "n={n} {w:?}");
            assert!(re > -32768 || im > -32768, "n={n} {w:?}");
        }
    }
}
