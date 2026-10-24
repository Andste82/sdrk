//! The FFT against the fallback and against the exact transform.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

mod common;

use common::{Rng, c32_noise, cq15_noise, cq15_plan, snr_db_exact as snr_db};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::{C32, CQ15, Fallback, Q15};

/// Exact DFT in f64 (direct sum, small sizes only).
fn dft(x: &[C32], inverse: bool) -> Vec<(f64, f64)> {
    let n = x.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0f64, 0f64);
            for (j, v) in x.iter().enumerate() {
                let ang = sign * 2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                let (s, c) = ang.sin_cos();
                re += f64::from(v.re) * c - f64::from(v.im) * s;
                im += f64::from(v.re) * s + f64::from(v.im) * c;
            }
            if inverse {
                (re / n as f64, im / n as f64)
            } else {
                (re, im)
            }
        })
        .collect()
}

fn plan_c32(n: usize) -> FftPlan<C32, Vec<C32>> {
    FftPlan::new(n, vec![C32::default(); n / 2]).unwrap()
}

#[test]
fn fft_c32_matches_the_exact_transform() {
    let mut rng = Rng::new(40);
    for log in 4..=10 {
        let n = 1usize << log;
        let plan = plan_c32(n);
        for dir in [Direction::Forward, Direction::Inverse] {
            let x = c32_noise(&mut rng, n, 1.0);
            let want = dft(&x, dir == Direction::Inverse);
            let mut got = x.clone();
            let scale = Simd.fft(&plan, &mut got, dir, Scaling::Default).unwrap();
            let fb_scale = Fallback
                .fft(&plan, &mut x.clone(), dir, Scaling::Default)
                .unwrap();
            assert_eq!(scale, fb_scale);
            let snr = snr_db(&want, &got);
            assert!(snr > 125.0, "n={n} {dir:?} snr={snr}");
        }
    }
}

#[test]
fn fft_c32_matches_the_fallback_for_large_sizes() {
    let mut rng = Rng::new(41);
    for log in [11, 12, 13, 16] {
        let n = 1usize << log;
        let plan = plan_c32(n);
        for dir in [Direction::Forward, Direction::Inverse] {
            let x = c32_noise(&mut rng, n, 1.0);
            let mut want = x.clone();
            Fallback
                .fft(&plan, &mut want, dir, Scaling::Default)
                .unwrap();
            let mut got = x.clone();
            Simd.fft(&plan, &mut got, dir, Scaling::Default).unwrap();
            let w: Vec<(f64, f64)> = want
                .iter()
                .map(|v| (f64::from(v.re), f64::from(v.im)))
                .collect();
            let snr = snr_db(&w, &got);
            assert!(snr > 120.0, "n={n} {dir:?} snr={snr}");
        }
    }
}

#[test]
fn fft_c32_special_inputs() {
    for n in [16usize, 64, 256] {
        let plan = plan_c32(n);
        // Impulse, DC, a single tone at the Nyquist frequency.
        let mut impulse = vec![C32::default(); n];
        impulse[1] = C32::new(1.0, 0.0);
        let dc = vec![C32::new(1.0, -0.5); n];
        let nyq: Vec<C32> = (0..n)
            .map(|i| C32::new(if i % 2 == 0 { 1.0 } else { -1.0 }, 0.0))
            .collect();
        for x in [impulse, dc, nyq] {
            let want = dft(&x, false);
            let mut got = x.clone();
            Simd.fft(&plan, &mut got, Direction::Forward, Scaling::Default)
                .unwrap();
            for (w, g) in want.iter().zip(&got) {
                assert!(
                    (w.0 - f64::from(g.re)).abs() < 1e-4 && (w.1 - f64::from(g.im)).abs() < 1e-4,
                    "n={n}"
                );
            }
        }
        // Zeros stay zeros.
        let mut z = vec![C32::default(); n];
        Simd.fft(&plan, &mut z, Direction::Forward, Scaling::Default)
            .unwrap();
        assert!(z.iter().all(|v| v.re == 0.0 && v.im == 0.0));
    }
}

#[test]
fn fft_c32_errors_match_the_fallback() {
    let plan = plan_c32(64);
    let mut data = vec![C32::default(); 32];
    assert_eq!(
        Fallback.fft(&plan, &mut data, Direction::Forward, Scaling::Default),
        Simd.fft(&plan, &mut data, Direction::Forward, Scaling::Default)
    );
    let mut data = vec![C32::default(); 64];
    assert_eq!(
        Fallback.fft(&plan, &mut data, Direction::Forward, Scaling::BlockFloat),
        Simd.fft(&plan, &mut data, Direction::Forward, Scaling::BlockFloat)
    );
}

#[test]
fn fft_c32_subsampled_twiddle_table() {
    // A plan over every k-th entry of the 4096-point table, copied into a table of its own
    // (`from_twiddles`, a table stride of 1); `fft_c32_sizes.rs` covers the plans of
    // `from_static_4096`, which read the static table with the stride k.
    let big = plan_c32(4096);
    let table: Vec<C32> = big.twiddles().to_vec();
    let mut rng = Rng::new(42);
    for n in [16usize, 128, 1024] {
        let stride = 4096 / n;
        let small = FftPlan::<C32, _>::from_twiddles(
            n,
            table.iter().step_by(stride).copied().collect::<Vec<_>>(),
        )
        .unwrap();
        let x = c32_noise(&mut rng, n, 1.0);
        let mut got = x.clone();
        Simd.fft(&small, &mut got, Direction::Forward, Scaling::Default)
            .unwrap();
        let snr = snr_db(&dft(&x, false), &got);
        assert!(snr > 125.0, "n={n} snr={snr}");
    }
}

/// The output of a `CQ15` transform times `2^exponent`.
fn scaled(x: &[CQ15], exponent: i32) -> Vec<(f64, f64)> {
    let g = 2f64.powi(exponent);
    x.iter()
        .map(|v| (f64::from(v.re.0) * g, f64::from(v.im.0) * g))
        .collect()
}

/// Bit-exact with the fallback, exponent included, where the generic code runs (the levels without
/// a level-specific path, and the sizes below it: see `fft_q15_generic.rs`); elsewhere (radix 4
/// with the twiddle products rounded per part, a tolerance kernel) the outputs times `2^exponent`
/// stay within 40 dB of the fallback's, except for the weak signal, whose few LSB both round
/// differently (the accuracy of weak signals is in `fft_q15_accuracy.rs`).
#[test]
fn fft_cq15_matches_the_fallback_including_the_exponent() {
    let generic_below = match sdrk_backend_simd::level_name() {
        "scalar" => usize::MAX,
        "avx2" | "neon" => 64,
        "avx512" => 256,
        _ => 16,
    };
    let mut rng = Rng::new(43);
    for log in 4..=13 {
        let n = 1usize << log;
        let generic = n < generic_below;
        let plan = cq15_plan(n);
        for dir in [Direction::Forward, Direction::Inverse] {
            for scaling in [Scaling::Default, Scaling::BlockFloat] {
                let inputs = [
                    cq15_noise(&mut rng, n),
                    vec![
                        CQ15 {
                            re: Q15(-32768),
                            im: Q15(-32768)
                        };
                        n
                    ],
                    vec![
                        CQ15 {
                            re: Q15(32767),
                            im: Q15(32767)
                        };
                        n
                    ],
                    (0..n)
                        .map(|i| CQ15 {
                            re: Q15(if i % 2 == 0 { 32767 } else { -32767 }),
                            im: Q15(0),
                        })
                        .collect(),
                    // Weak signal: block floating point must not shift it away.
                    (0..n)
                        .map(|_| CQ15 {
                            re: Q15((rng.next_u32() % 64) as i16 - 32),
                            im: Q15(0),
                        })
                        .collect(),
                    vec![CQ15::default(); n],
                ];
                for (i, x) in inputs.into_iter().enumerate() {
                    let (mut want, mut got) = (x.clone(), x.clone());
                    let sw = Fallback.fft(&plan, &mut want, dir, scaling).unwrap();
                    let sg = Simd.fft(&plan, &mut got, dir, scaling).unwrap();
                    if generic {
                        assert_eq!(sw, sg, "n={n} {dir:?} {scaling:?}");
                        assert_eq!(want, got, "n={n} {dir:?} {scaling:?}");
                        continue;
                    }
                    if scaling == Scaling::Default {
                        assert_eq!(sw, sg, "n={n} {dir:?} {scaling:?}");
                    }
                    if i == 4 {
                        continue;
                    }
                    let (want, got) = (scaled(&want, sw.exponent), scaled(&got, sg.exponent));
                    let (mut sig, mut err) = (0f64, 0f64);
                    for (w, g) in want.iter().zip(&got) {
                        sig += w.0 * w.0 + w.1 * w.1;
                        err += (w.0 - g.0).powi(2) + (w.1 - g.1).powi(2);
                    }
                    assert!(
                        err == 0.0 || 10.0 * (sig / err).log10() >= 40.0,
                        "input {i} n={n} {dir:?} {scaling:?}: {sig} / {err}"
                    );
                }
            }
        }
    }
}

#[test]
fn fft_cq15_errors_match_the_fallback() {
    let plan = cq15_plan(64);
    let mut data = vec![CQ15::default(); 32];
    assert_eq!(
        Fallback.fft(&plan, &mut data, Direction::Forward, Scaling::Default),
        Simd.fft(&plan, &mut data, Direction::Forward, Scaling::Default)
    );
}
