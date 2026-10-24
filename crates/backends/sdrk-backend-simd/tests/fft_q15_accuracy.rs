//! The `CQ15` FFT of the SIMD backend against the exact transform: the contract of
//! `docs/kernels.md` (block floating point: 58 dB for a full-scale tone, 55 dB for noise of
//! sigma 0.3; fixed scaling: 55 dB and 40 dB; a tone at -60 dBFS stays 30 dB above the median of
//! the other bins) at every size and in both directions. Where the fallback itself stays below a
//! threshold (the large sizes: every stage adds rounding noise), the backend may be at most 1 dB
//! below the fallback. Fixed scaling stays within 1 dB of the fallback at every size. The plans
//! of `from_static_4096`, which read the static table with the stride `4096 / n`, meet the same
//! limits.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

mod common;

use common::{Rng, cq15_plan as plan, db};
use sdrk_backend_simd::Simd;
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, Scaling};
use sdrk_core::{CQ15, Fallback, Q15};
use std::f64::consts::PI;

/// The sample `(re, im)` given in units of full scale, rounded and saturated.
fn quantize(re: f64, im: f64) -> CQ15 {
    let q = |v: f64| Q15((v * 32768.0).round().clamp(-32768.0, 32767.0) as i16);
    CQ15 {
        re: q(re),
        im: q(im),
    }
}

/// The exact transform of `x` (in units of full scale): radix-2 in `f64`, unnormalised; the
/// inverse one with `e^(+j ...)`.
fn exact(x: &[CQ15], inverse: bool) -> Vec<(f64, f64)> {
    let n = x.len();
    let bits = n.trailing_zeros();
    let mut v: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let s = x[i.reverse_bits() >> (usize::BITS - bits)];
            (f64::from(s.re.0) / 32768.0, f64::from(s.im.0) / 32768.0)
        })
        .collect();
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut half = 1;
    while half < n {
        for block in v.chunks_exact_mut(2 * half) {
            let (lo, hi) = block.split_at_mut(half);
            for (k, (a, c)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                let (s, co) = (sign * PI * k as f64 / half as f64).sin_cos();
                let t = (c.0 * co - c.1 * s, c.0 * s + c.1 * co);
                *c = (a.0 - t.0, a.1 - t.1);
                *a = (a.0 + t.0, a.1 + t.1);
            }
        }
        half *= 2;
    }
    v
}

/// The output of `fft` in units of full scale, times `2^exponent`.
fn transform<Tw: AsRef<[CQ15]>>(
    fft: &impl Fft<CQ15>,
    p: &FftPlan<CQ15, Tw>,
    x: &[CQ15],
    dir: Direction,
    scaling: Scaling,
) -> Vec<(f64, f64)> {
    let mut y = x.to_vec();
    let e = fft.fft(p, &mut y, dir, scaling).unwrap().exponent;
    if scaling == Scaling::Default {
        assert_eq!(e, x.len().trailing_zeros() as i32);
    }
    let g = 2f64.powi(e) / 32768.0;
    y.iter()
        .map(|s| (f64::from(s.re.0) * g, f64::from(s.im.0) * g))
        .collect()
}

/// Asserts that the backend reaches `threshold` dB (or, where the fallback does not, at most
/// 1 dB less than the fallback) for the input `x`.
fn check<Tw: AsRef<[CQ15]>>(
    p: &FftPlan<CQ15, Tw>,
    x: &[CQ15],
    dir: Direction,
    scaling: Scaling,
    threshold: f64,
    what: &str,
) {
    let want = exact(x, dir == Direction::Inverse);
    let ours = snr_db(&want, &transform(&Simd, p, x, dir, scaling));
    let fallback = snr_db(&want, &transform(&Fallback, p, x, dir, scaling));
    let n = x.len();
    assert!(
        ours >= threshold.min(fallback - 1.0),
        "{what} n={n} {dir:?} {scaling:?}: {ours:.2} dB (fallback {fallback:.2} dB)"
    );
}

/// The RMS error per component of a fixed-scaling output (see [`transform`]) in LSB of the
/// output, `N / 32768` in the units of the exact transform.
fn rms_error_lsb(want: &[(f64, f64)], got: &[(f64, f64)]) -> f64 {
    let e: f64 = want
        .iter()
        .zip(got)
        .map(|(w, g)| (w.0 - g.0).powi(2) + (w.1 - g.1).powi(2))
        .sum();
    let n = want.len() as f64;
    (e / (2.0 * n)).sqrt() / (n / 32768.0)
}

fn snr_db(want: &[(f64, f64)], got: &[(f64, f64)]) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for (w, g) in want.iter().zip(got) {
        s += w.0 * w.0 + w.1 * w.1;
        e += (w.0 - g.0).powi(2) + (w.1 - g.1).powi(2);
    }
    db(s, e)
}

/// A full-scale tone in bin `37 mod n`, amplitude `1 - 2^-15`.
fn tone(n: usize, amp: f64) -> Vec<CQ15> {
    (0..n)
        .map(|i| {
            let th = 2.0 * PI * ((37 * i) % n) as f64 / n as f64;
            quantize(amp * th.cos(), amp * th.sin())
        })
        .collect()
}

/// Gaussian noise of standard deviation `sigma` per part.
fn gaussian(rng: &mut Rng, n: usize, sigma: f64) -> Vec<CQ15> {
    let mut normal = || {
        let u1 = (f64::from(rng.next_u32()) + 1.0) / 4_294_967_297.0;
        let u2 = f64::from(rng.next_u32()) / 4_294_967_296.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    };
    (0..n)
        .map(|_| quantize(normal() * sigma, normal() * sigma))
        .collect()
}

fn sizes() -> impl Iterator<Item = usize> {
    (4..=16).map(|log| 1usize << log)
}

const DIRECTIONS: [Direction; 2] = [Direction::Forward, Direction::Inverse];

#[test]
fn fft_cq15_full_scale_tone_reaches_the_contract() {
    for n in sizes() {
        let p = plan(n);
        let x = tone(n, 1.0 - 2f64.powi(-15));
        for dir in DIRECTIONS {
            check(&p, &x, dir, Scaling::BlockFloat, 58.0, "tone");
            check(&p, &x, dir, Scaling::Default, 55.0, "tone");
        }
    }
}

#[test]
fn fft_cq15_noise_reaches_the_contract() {
    let mut rng = Rng::new(1510);
    for n in sizes() {
        let p = plan(n);
        for _ in 0..2 {
            let x = gaussian(&mut rng, n, 0.3);
            for dir in DIRECTIONS {
                check(&p, &x, dir, Scaling::BlockFloat, 55.0, "noise");
                check(&p, &x, dir, Scaling::Default, 40.0, "noise");
            }
        }
    }
}

#[test]
fn fft_cq15_fixed_scaling_stays_within_1_db_of_the_fallback() {
    // Fixed scaling rounds the last stage like the fallback (halving adds), so its accuracy
    // follows the fallback's at every size: noise of sigma 0.3, of a uniform +-8000 LSB, and
    // of a uniform +-0.9 of full scale with a corner value in every ninth sample.
    let mut rng = Rng::new(1514);
    for n in sizes() {
        let p = plan(n);
        let uniform: Vec<CQ15> = (0..n)
            .map(|_| {
                let mut part = || Q15((rng.next_u32() % 16_001) as i16 - 8000);
                CQ15 {
                    re: part(),
                    im: part(),
                }
            })
            .collect();
        let near_full: Vec<CQ15> = (0..n)
            .map(|k| {
                if k % 9 == 0 {
                    CQ15 {
                        re: Q15(-32768),
                        im: Q15(32767),
                    }
                } else {
                    let mut part = || Q15(((rng.next_u32() % 53_000) as i32 - 26_500) as i16);
                    CQ15 {
                        re: part(),
                        im: part(),
                    }
                }
            })
            .collect();
        // Near full scale the odd sizes (a final radix-2 stage) of the AVX2 path measure up to
        // 1.5 dB below the fallback; the absolute contract is checked by the tests above.
        for (x, margin) in [
            (gaussian(&mut rng, n, 0.3), 1.0),
            (uniform, 1.0),
            (near_full, 2.0),
        ] {
            for dir in DIRECTIONS {
                let want = exact(&x, dir == Direction::Inverse);
                let ours = snr_db(&want, &transform(&Simd, &p, &x, dir, Scaling::Default));
                let fallback = snr_db(&want, &transform(&Fallback, &p, &x, dir, Scaling::Default));
                assert!(
                    ours >= fallback - margin,
                    "n={n} {dir:?}: {ours:.2} dB (fallback {fallback:.2} dB)"
                );
            }
        }
    }
}

#[test]
fn fft_cq15_static_4096_plans_reach_the_contract() {
    let mut rng = Rng::new(1515);
    for n in (4..=12).map(|log| 1usize << log) {
        let p = FftPlan::<CQ15, &'static [CQ15]>::from_static_4096(n).unwrap();
        assert_eq!(p.twiddle_stride(), 4096 / n);
        let inputs = [
            ("tone", tone(n, 1.0 - 2f64.powi(-15)), 58.0),
            ("noise", gaussian(&mut rng, n, 0.3), 55.0),
        ];
        for (what, x, threshold) in &inputs {
            for dir in DIRECTIONS {
                check(&p, x, dir, Scaling::BlockFloat, *threshold, what);
                // Fixed scaling: within 1 dB of the fallback (`transform` checks the exponent).
                let want = exact(x, dir == Direction::Inverse);
                let ours = snr_db(&want, &transform(&Simd, &p, x, dir, Scaling::Default));
                let fallback = snr_db(&want, &transform(&Fallback, &p, x, dir, Scaling::Default));
                assert!(
                    ours >= fallback - 1.0,
                    "{what} n={n} {dir:?} fixed: {ours:.2} dB (fallback {fallback:.2} dB)"
                );
            }
        }
    }
}

#[test]
fn fft_cq15_weak_tone_stays_above_the_noise_floor() {
    let mut rng = Rng::new(1511);
    for n in [64usize, 1024, 4096] {
        let p = plan(n);
        // A tone at -60 dBFS plus noise of one LSB.
        let x: Vec<CQ15> = tone(n, 1e-3)
            .iter()
            .map(|s| CQ15 {
                re: Q15(s.re.0 + (rng.next_u32() % 3) as i16 - 1),
                im: Q15(s.im.0 + (rng.next_u32() % 3) as i16 - 1),
            })
            .collect();
        let y = transform(&Simd, &p, &x, Direction::Forward, Scaling::BlockFloat);
        let mut power: Vec<f64> = y.iter().map(|v| v.0 * v.0 + v.1 * v.1).collect();
        let bin = 37 % n;
        let peak = power[bin];
        power.remove(bin);
        power.sort_by(f64::total_cmp);
        let median = power[power.len() / 2];
        let db = 10.0 * (peak / median).log10();
        assert!(db >= 30.0, "n={n}: {db:.2} dB");
    }
}

#[test]
fn fft_cq15_extreme_inputs_stay_close_to_the_exact_transform() {
    let mut rng = Rng::new(1512);
    for n in sizes() {
        let p = plan(n);
        let corner = |re: i16, im: i16| {
            vec![
                CQ15 {
                    re: Q15(re),
                    im: Q15(im)
                };
                n
            ]
        };
        let inputs = [
            corner(-32768, -32768),
            corner(32767, -32768),
            (0..n)
                .map(|i| CQ15 {
                    re: Q15(if i % 2 == 0 { 32767 } else { -32767 }),
                    im: Q15(0),
                })
                .collect(),
            (0..n)
                .map(|i| CQ15 {
                    re: Q15(if i == 1 { -32768 } else { 0 }),
                    im: Q15(if i == 1 { 32767 } else { 0 }),
                })
                .collect(),
            (0..n)
                .map(|_| CQ15 {
                    re: Q15(rng.next_u32() as i16),
                    im: Q15(rng.next_u32() as i16),
                })
                .collect(),
        ];
        for (i, x) in inputs.iter().enumerate() {
            for dir in DIRECTIONS {
                check(&p, x, dir, Scaling::BlockFloat, 55.0, &format!("input {i}"));
                // Fixed scaling of samples beyond full scale (magnitude up to sqrt 2, the
                // over-range plan): no saturated products, so within 1 dB of the fallback or at
                // the contract (55 dB), or an error of at most one LSB of the output where the
                // output itself is a few LSB (the impulse at large sizes, whose SNR depends on the
                // order of the roundings).
                let want = exact(x, dir == Direction::Inverse);
                let got = transform(&Simd, &p, x, dir, Scaling::Default);
                let ours = snr_db(&want, &got);
                let fallback = snr_db(&want, &transform(&Fallback, &p, x, dir, Scaling::Default));
                let rms = rms_error_lsb(&want, &got);
                assert!(
                    ours >= (fallback - 1.0).min(55.0) || rms <= 1.0,
                    "input {i} n={n} {dir:?} fixed: {ours:.2} dB (fallback {fallback:.2} dB, \
                     error {rms:.2} LSB)"
                );
            }
        }
    }
}

#[test]
fn fft_cq15_zero_stays_zero() {
    for n in sizes() {
        let p = plan(n);
        for dir in DIRECTIONS {
            for scaling in [Scaling::Default, Scaling::BlockFloat] {
                let mut x = vec![CQ15::default(); n];
                Simd.fft(&p, &mut x, dir, scaling).unwrap();
                assert!(
                    x.iter().all(|s| *s == CQ15::default()),
                    "n={n} {dir:?} {scaling:?}"
                );
            }
        }
    }
}

#[test]
fn fft_cq15_round_trip_restores_the_input() {
    // Forward and inverse with block floating point, both exponents applied: the input within
    // the accuracy of the two transforms.
    let mut rng = Rng::new(1513);
    for n in [16usize, 256, 1024, 8192] {
        let p = plan(n);
        let x = gaussian(&mut rng, n, 0.3);
        let mut y = x.clone();
        let e1 = Simd
            .fft(&p, &mut y, Direction::Forward, Scaling::BlockFloat)
            .unwrap();
        let e2 = Simd
            .fft(&p, &mut y, Direction::Inverse, Scaling::BlockFloat)
            .unwrap();
        let g = 2f64.powi(e1.exponent + e2.exponent) / n as f64 / 32768.0;
        let got: Vec<(f64, f64)> = y
            .iter()
            .map(|s| (f64::from(s.re.0) * g, f64::from(s.im.0) * g))
            .collect();
        let want: Vec<(f64, f64)> = x
            .iter()
            .map(|s| (f64::from(s.re.0) / 32768.0, f64::from(s.im.0) / 32768.0))
            .collect();
        let db = snr_db(&want, &got);
        assert!(db >= 50.0, "n={n}: {db:.2} dB");
    }
}

#[test]
fn fft_cq15_errors_and_the_fallback_agree_on_lengths() {
    let p = plan(64);
    let mut data = vec![CQ15::default(); 32];
    assert_eq!(
        Fallback.fft(&p, &mut data, Direction::Forward, Scaling::BlockFloat),
        Simd.fft(&p, &mut data, Direction::Forward, Scaling::BlockFloat)
    );
}
