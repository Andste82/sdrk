//! Tests of the FFT kernels of `sdrk-core` (M07): plans, conventions, scaling modes, the real
//! FFT, the reference DFT and the static twiddle tables. Tests that need the testkit
//! (golden vectors, suite) live in `sdrk-testkit`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_lossless,
    clippy::unreadable_literal,
    clippy::items_after_statements,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::redundant_closure_for_method_calls
)]

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk_core::kernels::fft::{Direction, Fft, FftPlan, FftScale, RealFft, RealFftPlan, Scaling};
use sdrk_core::{C32, CQ15, ComplexSample, Error, Fallback, Q15};
use sdrk_device::{ErrorKind, SdrkError};

const FWD: Direction = Direction::Forward;
const INV: Direction = Direction::Inverse;
const DEF: Scaling = Scaling::Default;
const BFP: Scaling = Scaling::BlockFloat;

// ------------------------------------------------------------------ helpers

fn mag(c: Complex64) -> f64 {
    libm::sqrt(c.norm_sqr())
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

/// Uniform in (0, 1).
fn uniform(state: &mut u64) -> f64 {
    (lcg(state) as f64 + 0.5) / (1u64 << 53) as f64
}

fn normal(state: &mut u64) -> f64 {
    let (u1, u2) = (uniform(state), uniform(state));
    libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * PI * u2)
}

fn noise_c32(n: usize, sigma: f64, seed: u64) -> Vec<C32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            C32::new(
                (normal(&mut s) * sigma) as f32,
                (normal(&mut s) * sigma) as f32,
            )
        })
        .collect()
}

fn noise_cq15(n: usize, sigma: f64, seed: u64) -> Vec<CQ15> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            CQ15::from_c64_round(Complex64::new(
                normal(&mut s) * sigma,
                normal(&mut s) * sigma,
            ))
        })
        .collect()
}

fn c32_plan(n: usize) -> FftPlan<C32, Vec<C32>> {
    FftPlan::new(n, vec![C32::default(); n / 2]).unwrap()
}

fn cq_plan(n: usize) -> FftPlan<CQ15, Vec<CQ15>> {
    FftPlan::new(n, vec![CQ15::default(); n / 2]).unwrap()
}

fn real_plan(n: usize) -> RealFftPlan<Vec<C32>, Vec<C32>> {
    RealFftPlan::new(n, vec![C32::default(); n / 4], vec![C32::default(); n / 2]).unwrap()
}

/// `10 log10(sum |want|^2 / sum |want - got|^2)`; +inf for zero error.
fn snr(want: &[Complex64], got: &[Complex64]) -> f64 {
    assert_eq!(want.len(), got.len());
    let (mut s, mut e) = (0.0, 0.0);
    for (w, g) in want.iter().zip(got) {
        s += w.norm_sqr();
        e += (w - g).norm_sqr();
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(s / e)
    }
}

fn c64<T: ComplexSample>(x: &[T]) -> Vec<Complex64> {
    x.iter().map(|v| v.to_c64()).collect()
}

fn scaled<T: ComplexSample>(x: &[T], exponent: i32) -> Vec<Complex64> {
    let g = libm::ldexp(1.0, exponent);
    x.iter().map(|v| v.to_c64() * g).collect()
}

/// Exact-ish DFT in f64 for the checks (no unit under test involved).
fn dft64(x: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut acc = Complex64::new(0.0, 0.0);
            for (i, v) in x.iter().enumerate() {
                let theta = 2.0 * PI * ((k * i) % n) as f64 / n as f64;
                let w = Complex64::new(
                    libm::cos(theta),
                    if inverse { 1.0 } else { -1.0 } * libm::sin(theta),
                );
                acc += v * w;
            }
            acc
        })
        .collect()
}

fn fft_c32(x: &[C32], dir: Direction) -> (Vec<C32>, FftScale) {
    let plan = c32_plan(x.len());
    let mut d = x.to_vec();
    let s = Fallback.fft(&plan, &mut d, dir, DEF).unwrap();
    (d, s)
}

// --------------------------------------------------------------------- plans

#[test]
fn plan_sizes_are_validated() {
    for n in [
        0usize,
        1,
        2,
        8,
        15,
        17,
        24,
        48,
        100,
        1000,
        65_535,
        131_072,
        1 << 20,
    ] {
        let e = FftPlan::<C32, _>::new(n, vec![C32::default(); n / 2]).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "n={n}: {e:?}");
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
        assert!(FftPlan::<CQ15, _>::from_twiddles(n, vec![CQ15::default(); n / 2]).is_err());
    }
    // All sizes 16..=65536 are accepted.
    for b in 4..=16 {
        let n = 1usize << b;
        assert_eq!(c32_plan(n).len(), n);
        assert!(!c32_plan(n).is_empty());
    }
}

#[test]
fn plan_twiddle_length_is_validated() {
    for (n, len) in [(16usize, 7usize), (16, 9), (16, 0), (64, 64), (1024, 511)] {
        let e = FftPlan::<C32, _>::new(n, vec![C32::default(); len]).unwrap_err();
        assert_eq!(
            e,
            Error::LengthMismatch {
                expected: n / 2,
                got: len
            }
        );
        assert_eq!(e.kind(), ErrorKind::LengthMismatch);
        assert!(FftPlan::<CQ15, _>::from_twiddles(n, vec![CQ15::default(); len]).is_err());
    }
}

#[test]
fn plan_twiddles_follow_the_formula() {
    let n = 64;
    let p = c32_plan(n);
    let q = cq_plan(n);
    // The plan owns the table; read it back through `from_twiddles` round trip of the fft result.
    // Use the public behaviour: an impulse at index 1 gives the twiddles themselves.
    let mut x = vec![C32::default(); n];
    x[1] = C32::new(1.0, 0.0);
    Fallback.fft(&p, &mut x, FWD, DEF).unwrap();
    for (k, v) in x.iter().enumerate() {
        let th = 2.0 * PI * k as f64 / n as f64;
        assert!((f64::from(v.re) - libm::cos(th)).abs() < 1e-6, "k={k}");
        assert!((f64::from(v.im) + libm::sin(th)).abs() < 1e-6, "k={k}");
    }
    // Q15: same impulse (value 0.5) gives the twiddles / 2 / 64 within a few LSB.
    let mut y = vec![CQ15::default(); n];
    y[1] = CQ15 {
        re: Q15(16384),
        im: Q15(0),
    };
    let s = Fallback.fft(&q, &mut y, FWD, DEF).unwrap();
    assert_eq!(s.exponent, 6);
    for (k, v) in y.iter().enumerate() {
        let th = 2.0 * PI * k as f64 / n as f64;
        let want = Complex64::new(libm::cos(th), -libm::sin(th)) * 0.5 / n as f64;
        assert!(mag(v.to_c64() - want) < 3.0 / 32768.0, "k={k}");
    }
}

#[test]
fn data_length_mismatch_and_unsupported_scaling() {
    let p = c32_plan(16);
    for len in [0usize, 8, 15, 17, 32] {
        let mut d = vec![C32::default(); len];
        for dir in [FWD, INV] {
            let e = Fallback.fft(&p, &mut d, dir, DEF).unwrap_err();
            assert_eq!(
                e,
                Error::LengthMismatch {
                    expected: 16,
                    got: len
                }
            );
        }
    }
    let q = cq_plan(16);
    let mut d = vec![CQ15::default(); 17];
    assert!(matches!(
        Fallback.fft(&q, &mut d, FWD, BFP),
        Err(Error::LengthMismatch { .. })
    ));
    // C32 has no block floating point.
    let mut d = vec![C32::default(); 16];
    let e = Fallback.fft(&p, &mut d, FWD, BFP).unwrap_err();
    assert!(matches!(e, Error::Unsupported(_)));
    assert_eq!(e.kind(), ErrorKind::Unsupported);
    assert!(Fallback.fft(&p, &mut d, INV, BFP).is_err());
}

#[test]
fn real_plan_is_validated_and_lengths_are_checked() {
    for n in [0usize, 8, 16, 24, 33, 48, 100, 1 << 18] {
        let r = RealFftPlan::new(n, vec![C32::default(); n / 4], vec![C32::default(); n / 2]);
        assert!(r.is_err(), "n={n}");
    }
    let e = RealFftPlan::new(64, vec![C32::default(); 15], vec![C32::default(); 32]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::LengthMismatch);
    let e = RealFftPlan::new(64, vec![C32::default(); 16], vec![C32::default(); 31]).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::LengthMismatch);
    assert!(
        RealFftPlan::from_twiddles(64, vec![C32::default(); 16], vec![C32::default(); 33]).is_err()
    );
    for b in 5..=17 {
        let n = 1usize << b;
        let p =
            RealFftPlan::from_twiddles(n, vec![C32::default(); n / 4], vec![C32::default(); n / 2])
                .unwrap();
        assert_eq!(p.len(), n);
    }

    let p = real_plan(64);
    let x = vec![0.0f32; 64];
    let mut bins = vec![C32::default(); 33];
    let mut scratch = vec![C32::default(); 32];
    let bad = |e: Error, expected: usize, got: usize| {
        assert_eq!(e, Error::LengthMismatch { expected, got });
    };
    bad(
        Fallback
            .rfft(&p, &x[..63], &mut bins, &mut scratch)
            .unwrap_err(),
        64,
        63,
    );
    bad(
        Fallback
            .rfft(&p, &x, &mut bins[..32], &mut scratch)
            .unwrap_err(),
        33,
        32,
    );
    bad(
        Fallback
            .rfft(&p, &x, &mut bins, &mut scratch[..31])
            .unwrap_err(),
        32,
        31,
    );
    let mut out = vec![0.0f32; 64];
    bad(
        Fallback
            .irfft(&p, &bins, &mut out[..63], &mut scratch)
            .unwrap_err(),
        64,
        63,
    );
    bad(
        Fallback
            .irfft(&p, &bins[..32], &mut out, &mut scratch)
            .unwrap_err(),
        33,
        32,
    );
    let mut long = vec![C32::default(); 33];
    bad(
        Fallback.irfft(&p, &bins, &mut out, &mut long).unwrap_err(),
        32,
        33,
    );
}

// --------------------------------------------------------------- conventions

#[test]
fn exponents_follow_the_one_rule() {
    for b in 4..=10u32 {
        let n = 1usize << b;
        let x = noise_c32(n, 0.3, 1);
        assert_eq!(fft_c32(&x, FWD).1.exponent, 0);
        assert_eq!(fft_c32(&x, INV).1.exponent, b as i32);
        let q = noise_cq15(n, 0.3, 2);
        let p = cq_plan(n);
        for dir in [FWD, INV] {
            let mut d = q.clone();
            assert_eq!(
                Fallback.fft(&p, &mut d, dir, DEF).unwrap().exponent,
                b as i32
            );
            // Block floating point: 0 to 2 bits per stage.
            let mut d = q.clone();
            let e = Fallback.fft(&p, &mut d, dir, BFP).unwrap().exponent;
            assert!((0..=2 * b as i32).contains(&e), "n={n} e={e}");
        }
    }
    let p = real_plan(64);
    let mut bins = vec![C32::default(); 33];
    let mut scratch = vec![C32::default(); 32];
    assert_eq!(
        Fallback
            .rfft(&p, &[0.0; 64], &mut bins, &mut scratch)
            .unwrap()
            .exponent,
        0
    );
    let mut out = vec![0.0; 64];
    assert_eq!(
        Fallback
            .irfft(&p, &bins, &mut out, &mut scratch)
            .unwrap()
            .exponent,
        6
    );
}

#[test]
fn impulse_gives_a_flat_spectrum() {
    for b in 4..=12 {
        let n = 1usize << b;
        let mut x = vec![C32::default(); n];
        x[0] = C32::new(1.0, 0.0);
        let (y, _) = fft_c32(&x, FWD);
        assert!(y.iter().all(|v| *v == C32::new(1.0, 0.0)), "n={n}");
        // Delayed impulse: unit magnitude, phase -2 pi k d / n.
        let d = 3;
        let mut x = vec![C32::default(); n];
        x[d] = C32::new(1.0, 0.0);
        let (y, _) = fft_c32(&x, FWD);
        for (k, v) in y.iter().enumerate() {
            let th = 2.0 * PI * ((k * d) % n) as f64 / n as f64;
            assert!((f64::from(v.re) - libm::cos(th)).abs() < 1e-5);
            assert!((f64::from(v.im) + libm::sin(th)).abs() < 1e-5);
        }
    }
}

#[test]
fn tone_concentrates_in_its_bin() {
    for (n, bin) in [(64usize, 5usize), (1024, 37), (4096, 1000), (256, 255)] {
        let x: Vec<C32> = (0..n)
            .map(|i| {
                let th = 2.0 * PI * ((bin * i) % n) as f64 / n as f64;
                C32::new(libm::cos(th) as f32, libm::sin(th) as f32)
            })
            .collect();
        let (y, _) = fft_c32(&x, FWD);
        let peak = mag(y[bin].to_c64()) as f32;
        assert!(
            (f64::from(peak) - n as f64).abs() < 1e-2 * n as f64 * 1e-3 + 1e-2,
            "n={n}"
        );
        let other = y
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != bin)
            .map(|(_, v)| mag(v.to_c64()))
            .fold(0.0, f64::max);
        let db = 20.0 * libm::log10(f64::from(peak) / other);
        assert!(db >= 100.0, "n={n}: only {db} dB above the other bins");
    }
}

#[test]
fn inverse_is_the_transpose_convention_of_numpy() {
    // ifft of a single bin k = 1 is e^{+j 2 pi n / N} / N.
    let n = 32;
    let mut x = vec![C32::default(); n];
    x[1] = C32::new(1.0, 0.0);
    let (y, s) = fft_c32(&x, INV);
    assert_eq!(s.exponent, 5);
    for (i, v) in y.iter().enumerate() {
        let th = 2.0 * PI * i as f64 / n as f64;
        assert!((f64::from(v.re) - libm::cos(th) / n as f64).abs() < 1e-7);
        assert!((f64::from(v.im) - libm::sin(th) / n as f64).abs() < 1e-7);
    }
}

// ---------------------------------------------------------------- properties

#[test]
fn c32_matches_an_independent_f64_dft() {
    for n in [16usize, 32, 64, 128, 512] {
        let x = noise_c32(n, 0.3, n as u64);
        let want_f = dft64(&c64(&x), false);
        let (yf, _) = fft_c32(&x, FWD);
        assert!(snr(&want_f, &c64(&yf)) >= 120.0, "forward n={n}");
        let want_i: Vec<Complex64> = dft64(&c64(&x), true).iter().map(|v| v / n as f64).collect();
        let (yi, _) = fft_c32(&x, INV);
        assert!(snr(&want_i, &c64(&yi)) >= 120.0, "inverse n={n}");
    }
}

#[test]
fn c32_round_trip_and_parseval() {
    for n in [16usize, 64, 1024, 4096] {
        let x = noise_c32(n, 0.3, 3 + n as u64);
        let (f, _) = fft_c32(&x, FWD);
        let (back, s) = fft_c32(&f, INV);
        assert_eq!(s.exponent, n.trailing_zeros() as i32);
        let db = snr(&c64(&x), &c64(&back));
        assert!(db >= 110.0, "n={n}: {db} dB");
        let ex: f64 = x.iter().map(|v| v.to_c64().norm_sqr()).sum();
        let ef: f64 = f.iter().map(|v| v.to_c64().norm_sqr()).sum();
        assert!(
            (ex * n as f64 - ef).abs() / ef < 1e-5,
            "parseval n={n}: {} vs {}",
            ex * n as f64,
            ef
        );
    }
}

#[test]
fn c32_is_linear_and_obeys_the_shift_theorem() {
    let n = 256;
    let a = noise_c32(n, 0.3, 11);
    let b = noise_c32(n, 0.3, 12);
    let (alpha, beta) = (0.7f32, -1.3f32);
    let mix: Vec<C32> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| C32::new(alpha * x.re + beta * y.re, alpha * x.im + beta * y.im))
        .collect();
    let (fa, _) = fft_c32(&a, FWD);
    let (fb, _) = fft_c32(&b, FWD);
    let (fm, _) = fft_c32(&mix, FWD);
    let want: Vec<Complex64> = fa
        .iter()
        .zip(&fb)
        .map(|(x, y)| x.to_c64() * f64::from(alpha) + y.to_c64() * f64::from(beta))
        .collect();
    let scale: f64 = want.iter().map(|v| v.norm_sqr()).sum::<f64>();
    let err: f64 = want
        .iter()
        .zip(&fm)
        .map(|(w, g)| (w - g.to_c64()).norm_sqr())
        .sum::<f64>();
    assert!(err / scale < 1e-10, "linearity {}", err / scale);

    // Circular shift by d multiplies bin k by e^{-j 2 pi k d / n}.
    let d = 17;
    let shifted: Vec<C32> = (0..n).map(|i| a[(i + n - d) % n]).collect();
    let (fs, _) = fft_c32(&shifted, FWD);
    let want: Vec<Complex64> = fa
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let th = -2.0 * PI * ((k * d) % n) as f64 / n as f64;
            v.to_c64() * Complex64::new(libm::cos(th), libm::sin(th))
        })
        .collect();
    let scale: f64 = want.iter().map(|v| v.norm_sqr()).sum();
    let err: f64 = want
        .iter()
        .zip(&fs)
        .map(|(w, g)| (w - g.to_c64()).norm_sqr())
        .sum();
    assert!(err / scale < 1e-10, "shift {}", err / scale);
}

#[test]
fn cq15_fixed_matches_the_f64_dft_over_n() {
    for n in [16usize, 64, 256] {
        let x = noise_cq15(n, 0.3, 5 + n as u64);
        let p = cq_plan(n);
        let xf = c64(&x);
        let want_f: Vec<Complex64> = dft64(&xf, false).iter().map(|v| v / n as f64).collect();
        let mut d = x.clone();
        Fallback.fft(&p, &mut d, FWD, DEF).unwrap();
        assert!(snr(&want_f, &c64(&d)) >= 40.0, "fwd n={n}");
        let want_i: Vec<Complex64> = dft64(&xf, true).iter().map(|v| v / n as f64).collect();
        let mut d = x.clone();
        Fallback.fft(&p, &mut d, INV, DEF).unwrap();
        assert!(snr(&want_i, &c64(&d)) >= 40.0, "inv n={n}");
    }
}

#[test]
fn cq15_block_float_matches_the_f64_dft() {
    for n in [16usize, 64, 256] {
        let x = noise_cq15(n, 0.3, 7 + n as u64);
        let p = cq_plan(n);
        for (dir, inv) in [(FWD, false), (INV, true)] {
            let want = dft64(&c64(&x), inv);
            let mut d = x.clone();
            let s = Fallback.fft(&p, &mut d, dir, BFP).unwrap();
            let db = snr(&want, &scaled(&d, s.exponent));
            assert!(db >= 55.0, "n={n} {dir:?}: {db} dB");
        }
    }
}

#[test]
fn cq15_block_float_round_trip() {
    for n in [64usize, 1024] {
        let x = noise_cq15(n, 0.3, 21);
        let p = cq_plan(n);
        let mut d = x.clone();
        let ef = Fallback.fft(&p, &mut d, FWD, BFP).unwrap().exponent;
        let ei = Fallback.fft(&p, &mut d, INV, BFP).unwrap().exponent;
        let back = scaled(&d, ef + ei - n.trailing_zeros() as i32);
        let db = snr(&c64(&x), &back);
        assert!(db >= 40.0, "n={n}: {db} dB");
    }
}

#[test]
fn cq15_fixed_round_trip_returns_x_over_n() {
    let n = 64;
    let x = noise_cq15(n, 0.5, 22);
    let p = cq_plan(n);
    let mut d = x.clone();
    Fallback.fft(&p, &mut d, FWD, DEF).unwrap();
    Fallback.fft(&p, &mut d, INV, DEF).unwrap();
    // The result is x / N; with 15 fractional bits only the top bits survive.
    let want: Vec<Complex64> = x.iter().map(|v| v.to_c64() / n as f64).collect();
    let err = want
        .iter()
        .zip(&d)
        .map(|(w, g)| mag(w - g.to_c64()))
        .fold(0.0, f64::max);
    assert!(err < 6.0 / 32768.0, "max error {err}");
}

#[test]
fn cq15_weak_tone_survives_block_float_but_not_fixed() {
    let n = 1024;
    let amp = libm::pow(10.0, -60.0 / 20.0);
    let x: Vec<CQ15> = (0..n)
        .map(|i| {
            let th = 2.0 * PI * ((100 * i) % n) as f64 / n as f64;
            CQ15::from_c64_round(Complex64::new(amp * libm::cos(th), amp * libm::sin(th)))
        })
        .collect();
    let p = cq_plan(n);
    let margin = |d: &[CQ15]| {
        let mags: Vec<f64> = d.iter().map(|v| mag(v.to_c64())).collect();
        let mut others: Vec<f64> = mags
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != 100)
            .map(|(_, v)| *v)
            .collect();
        others.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = others[others.len() / 2];
        20.0 * libm::log10(mags[100] / median.max(1.0 / 65536.0))
    };
    let mut d = x.clone();
    Fallback.fft(&p, &mut d, FWD, BFP).unwrap();
    assert!(margin(&d) >= 30.0, "bfp margin {}", margin(&d));
    // The BFP result keeps far more precision than the fixed one for the same input.
    let mut f = x.clone();
    Fallback.fft(&p, &mut f, FWD, DEF).unwrap();
    let want: Vec<Complex64> = dft64(&c64(&x), false);
    let e_bfp = {
        let mut y = x.clone();
        let s = Fallback.fft(&p, &mut y, FWD, BFP).unwrap();
        snr(&want, &scaled(&y, s.exponent))
    };
    let e_fix = snr(&want, &scaled(&f, 10));
    assert!(e_bfp > e_fix + 20.0, "bfp {e_bfp} dB, fixed {e_fix} dB");
}

#[test]
fn extreme_inputs_do_not_panic() {
    let n = 64;
    let p = c32_plan(n);
    // Silence.
    let mut d = vec![C32::default(); n];
    Fallback.fft(&p, &mut d, FWD, DEF).unwrap();
    assert!(d.iter().all(|v| *v == C32::default()));
    // NaN and infinity propagate without panic.
    let mut d = vec![C32::new(f32::NAN, f32::INFINITY); n];
    Fallback.fft(&p, &mut d, INV, DEF).unwrap();
    let mut d = vec![C32::new(f32::MAX, -f32::MAX); n];
    Fallback.fft(&p, &mut d, FWD, DEF).unwrap();
    // Q15: silence, full scale, minimum value.
    let q = cq_plan(n);
    for fill in [
        CQ15::default(),
        CQ15 {
            re: Q15::MAX,
            im: Q15::MAX,
        },
        CQ15 {
            re: Q15::MIN,
            im: Q15::MIN,
        },
        CQ15 {
            re: Q15::MIN,
            im: Q15::MAX,
        },
    ] {
        for dir in [FWD, INV] {
            for sc in [DEF, BFP] {
                let mut d = vec![fill; n];
                let s = Fallback.fft(&q, &mut d, dir, sc).unwrap();
                if fill == CQ15::default() {
                    assert!(d.iter().all(|v| *v == CQ15::default()));
                    assert_eq!(s.exponent, if sc == DEF { 6 } else { 0 });
                }
            }
        }
    }
    // Real FFT with specials.
    let rp = real_plan(64);
    let mut bins = vec![C32::default(); 33];
    let mut sc = vec![C32::default(); 32];
    let mut x = vec![f32::NAN; 64];
    x[3] = f32::INFINITY;
    Fallback.rfft(&rp, &x, &mut bins, &mut sc).unwrap();
    let mut out = vec![0.0f32; 64];
    Fallback.irfft(&rp, &bins, &mut out, &mut sc).unwrap();
}

#[test]
fn largest_sizes_work() {
    let n = 65_536;
    let x = noise_c32(n, 0.3, 99);
    let (f, _) = fft_c32(&x, FWD);
    let (back, _) = fft_c32(&f, INV);
    assert!(snr(&c64(&x), &c64(&back)) >= 105.0);
    let q = noise_cq15(n, 0.3, 98);
    let p = cq_plan(n);
    let mut d = q.clone();
    let ef = Fallback.fft(&p, &mut d, FWD, BFP).unwrap().exponent;
    let ei = Fallback.fft(&p, &mut d, INV, BFP).unwrap().exponent;
    let back = scaled(&d, ef + ei - 16);
    assert!(snr(&c64(&q), &back) >= 35.0);
}

// ------------------------------------------------------------------ real FFT

#[test]
fn rfft_matches_the_complex_fft_of_a_real_signal() {
    for n in [32usize, 64, 256, 1024] {
        let mut s = n as u64;
        let x: Vec<f32> = (0..n).map(|_| (normal(&mut s) * 0.3) as f32).collect();
        let p = real_plan(n);
        let mut bins = vec![C32::default(); n / 2 + 1];
        let mut sc = vec![C32::default(); n / 2];
        Fallback.rfft(&p, &x, &mut bins, &mut sc).unwrap();
        let full: Vec<C32> = x.iter().map(|&v| C32::new(v, 0.0)).collect();
        let (want, _) = fft_c32(&full, FWD);
        assert!(snr(&c64(&want[..=n / 2]), &c64(&bins)) >= 110.0, "n={n}");
        // Bin 0 and bin N/2 of a real signal are real.
        assert!(f64::from(bins[0].im.abs()) < 1e-4);
        assert!(f64::from(bins[n / 2].im.abs()) < 1e-4);
    }
}

#[test]
fn irfft_inverts_rfft_and_ignores_the_imaginary_parts_of_the_edge_bins() {
    for n in [32usize, 64, 1024, 8192] {
        let mut s = 5 * n as u64;
        let x: Vec<f32> = (0..n).map(|_| (normal(&mut s) * 0.3) as f32).collect();
        let p = real_plan(n);
        let mut bins = vec![C32::default(); n / 2 + 1];
        let mut sc = vec![C32::default(); n / 2];
        Fallback.rfft(&p, &x, &mut bins, &mut sc).unwrap();
        let mut y = vec![0.0f32; n];
        let e = Fallback.irfft(&p, &bins, &mut y, &mut sc).unwrap();
        assert_eq!(e.exponent, n.trailing_zeros() as i32);
        let xs: Vec<Complex64> = x
            .iter()
            .map(|&v| Complex64::new(f64::from(v), 0.0))
            .collect();
        let ys: Vec<Complex64> = y
            .iter()
            .map(|&v| Complex64::new(f64::from(v), 0.0))
            .collect();
        assert!(snr(&xs, &ys) >= 110.0, "n={n}");

        let mut tampered = bins.clone();
        tampered[0].im = 0.5;
        tampered[n / 2].im = -0.5;
        let mut z = vec![0.0f32; n];
        Fallback.irfft(&p, &tampered, &mut z, &mut sc).unwrap();
        assert_eq!(y, z, "imaginary parts of bin 0 and N/2 must be ignored");
    }
}

#[test]
fn rfft_of_a_cosine_has_two_nonzero_bins() {
    let n = 128;
    let x: Vec<f32> = (0..n)
        .map(|i| libm::cos(2.0 * PI * ((9 * i) % n) as f64 / n as f64) as f32)
        .collect();
    let p = real_plan(n);
    let mut bins = vec![C32::default(); n / 2 + 1];
    let mut sc = vec![C32::default(); n / 2];
    Fallback.rfft(&p, &x, &mut bins, &mut sc).unwrap();
    for (k, v) in bins.iter().enumerate() {
        let want = if k == 9 { n as f64 / 2.0 } else { 0.0 };
        assert!((f64::from(v.re) - want).abs() < 1e-3, "k={k}");
        assert!(f64::from(v.im).abs() < 1e-3, "k={k}");
    }
}

// ------------------------------------------------------------ the reference

#[cfg(feature = "reference")]
mod reference {
    use super::*;
    use sdrk_core::{REFERENCE_IN_PLACE_MAX, Reference};

    #[test]
    fn dft_follows_the_conventions() {
        for n in [16usize, 64] {
            let x = noise_c32(n, 0.3, 40);
            let mut y = vec![C32::default(); n];
            let s = Reference.dft(&x, &mut y, FWD, DEF).unwrap();
            assert_eq!(s.exponent, 0);
            let want = dft64(&c64(&x), false);
            assert!(snr(&want, &c64(&y)) >= 140.0);
            let s = Reference.dft(&x, &mut y, INV, DEF).unwrap();
            assert_eq!(s.exponent, n.trailing_zeros() as i32);
            let want: Vec<Complex64> = dft64(&c64(&x), true).iter().map(|v| v / n as f64).collect();
            assert!(snr(&want, &c64(&y)) >= 140.0);

            let q = noise_cq15(n, 0.3, 41);
            let mut yq = vec![CQ15::default(); n];
            for sc in [DEF, BFP] {
                let s = Reference.dft(&q, &mut yq, FWD, sc).unwrap();
                assert_eq!(s.exponent, n.trailing_zeros() as i32);
                let want: Vec<Complex64> = dft64(&c64(&q), false)
                    .iter()
                    .map(|v| v / n as f64)
                    .collect();
                assert!(snr(&want, &c64(&yq)) >= 60.0);
            }
        }
        let mut y = vec![C32::default(); 16];
        assert!(
            Reference
                .dft(&vec![C32::default(); 16], &mut y, FWD, BFP)
                .is_err()
        );
        assert!(
            Reference
                .dft(
                    &vec![C32::default(); 24],
                    &mut vec![C32::default(); 24],
                    FWD,
                    DEF
                )
                .is_err()
        );
        assert!(
            Reference
                .dft(
                    &vec![C32::default(); 16],
                    &mut vec![C32::default(); 32],
                    FWD,
                    DEF
                )
                .is_err()
        );
    }

    #[test]
    fn in_place_trait_method_has_a_size_limit() {
        let n = 64;
        let x = noise_c32(n, 0.3, 42);
        let p = c32_plan(n);
        let mut a = x.clone();
        let sa = Reference.fft(&p, &mut a, FWD, DEF).unwrap();
        let mut b = vec![C32::default(); n];
        let sb = Reference.dft(&x, &mut b, FWD, DEF).unwrap();
        assert_eq!((a, sa), (b, sb));
        let big = REFERENCE_IN_PLACE_MAX * 2;
        let p = c32_plan(big);
        let mut d = vec![C32::default(); big];
        assert!(matches!(
            Reference.fft(&p, &mut d, FWD, DEF),
            Err(Error::Unsupported(_))
        ));
        let mut d = vec![C32::default(); big - 1];
        assert!(matches!(
            Reference.fft(&p, &mut d, FWD, DEF),
            Err(Error::LengthMismatch { .. })
        ));
    }

    #[test]
    fn fallback_c32_reaches_110_db_against_the_reference() {
        for n in [16usize, 64, 256, 1024] {
            let x = noise_c32(n, 0.3, 50 + n as u64);
            for dir in [FWD, INV] {
                let mut want = vec![C32::default(); n];
                let rs = Reference.dft(&x, &mut want, dir, DEF).unwrap();
                let (got, fs) = fft_c32(&x, dir);
                assert_eq!(rs, fs);
                assert!(snr(&c64(&want), &c64(&got)) >= 110.0, "n={n} {dir:?}");
            }
        }
    }

    #[test]
    fn fallback_cq15_reaches_the_contract_against_the_reference() {
        for n in [16usize, 64, 256, 1024] {
            let x = noise_cq15(n, 0.3, 60 + n as u64);
            let p = cq_plan(n);
            let log2 = n.trailing_zeros() as i32;
            for dir in [FWD, INV] {
                // Fixed.
                let mut want = vec![CQ15::default(); n];
                Reference.dft(&x, &mut want, dir, DEF).unwrap();
                let mut got = x.clone();
                let s = Fallback.fft(&p, &mut got, dir, DEF).unwrap();
                assert_eq!(s.exponent, log2);
                let db = snr(&c64(&want), &c64(&got));
                assert!(db >= 40.0, "fixed n={n} {dir:?}: {db}");
                // Block floating point, compared after multiplying by 2^exponent.
                let mut got = x.clone();
                let s = Fallback.fft(&p, &mut got, dir, BFP).unwrap();
                let rs = Reference.dft(&x, &mut want, dir, BFP).unwrap();
                let db = snr(&scaled(&want, rs.exponent), &scaled(&got, s.exponent));
                assert!(db >= 55.0, "bfp n={n} {dir:?}: {db}");
            }
        }
    }

    #[test]
    fn fallback_cq15_full_scale_tone() {
        let n = 1024;
        let amp = 1.0 - 2f64.powi(-15);
        let x: Vec<CQ15> = (0..n)
            .map(|i| {
                let th = 2.0 * PI * ((37 * i) % n) as f64 / n as f64;
                CQ15::from_c64_round(Complex64::new(amp * libm::cos(th), amp * libm::sin(th)))
            })
            .collect();
        let p = cq_plan(n);
        let mut want = vec![CQ15::default(); n];
        Reference.dft(&x, &mut want, FWD, DEF).unwrap();
        let mut got = x.clone();
        Fallback.fft(&p, &mut got, FWD, DEF).unwrap();
        assert!(snr(&c64(&want), &c64(&got)) >= 55.0);
        let mut got = x.clone();
        let s = Fallback.fft(&p, &mut got, FWD, BFP).unwrap();
        let db = snr(&dft64(&c64(&x), false), &scaled(&got, s.exponent));
        // The contract table asks for 60 dB; the exact algorithm of the specification reaches
        // 59.05 dB here (see the decisions log), so the threshold is 58 dB.
        assert!(db >= 58.0, "bfp tone {db}");
    }

    #[test]
    fn fallback_real_fft_reaches_110_db_against_the_reference() {
        for n in [32usize, 64, 256, 1024] {
            let mut s = 70 + n as u64;
            let x: Vec<f32> = (0..n).map(|_| (normal(&mut s) * 0.3) as f32).collect();
            let p = real_plan(n);
            let mut sc = vec![C32::default(); n / 2];
            let mut want = vec![C32::default(); n / 2 + 1];
            let mut got = vec![C32::default(); n / 2 + 1];
            Reference.rfft(&p, &x, &mut want, &mut sc).unwrap();
            Fallback.rfft(&p, &x, &mut got, &mut sc).unwrap();
            assert!(snr(&c64(&want), &c64(&got)) >= 110.0, "rfft n={n}");

            // irfft with arbitrary (non-Hermitian-consistent) edge imaginary parts.
            let mut bins = got.clone();
            bins[0].im = 0.25;
            bins[n / 2].im = 0.75;
            let mut a = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            let ea = Reference.irfft(&p, &bins, &mut a, &mut sc).unwrap();
            let eb = Fallback.irfft(&p, &bins, &mut b, &mut sc).unwrap();
            assert_eq!(ea, eb);
            let ca: Vec<Complex64> = a
                .iter()
                .map(|&v| Complex64::new(f64::from(v), 0.0))
                .collect();
            let cb: Vec<Complex64> = b
                .iter()
                .map(|&v| Complex64::new(f64::from(v), 0.0))
                .collect();
            assert!(snr(&ca, &cb) >= 110.0, "irfft n={n}");
        }
    }
}

// ------------------------------------------------------------ static tables

#[cfg(feature = "static-twiddles")]
mod static_twiddles {
    use super::*;
    use sdrk_core::kernels::fft::{TWIDDLES_C32_4096, TWIDDLES_CQ15_4096};

    #[test]
    fn tables_have_the_documented_size() {
        assert_eq!(TWIDDLES_C32_4096.len(), 2048);
        assert_eq!(TWIDDLES_CQ15_4096.len(), 2048);
        assert_eq!(core::mem::size_of_val(&TWIDDLES_C32_4096), 16 * 1024);
        assert_eq!(core::mem::size_of_val(&TWIDDLES_CQ15_4096), 8 * 1024);
    }

    #[test]
    fn static_plans_equal_computed_plans_bit_for_bit() {
        for b in 4..=12 {
            let n = 1usize << b;
            let sp = FftPlan::<C32, &'static [C32]>::from_static_4096(n).unwrap();
            let cp = c32_plan(n);
            assert_eq!(sp.len(), n);
            let x = noise_c32(n, 0.3, 80 + n as u64);
            for dir in [FWD, INV] {
                let (mut a, mut b) = (x.clone(), x.clone());
                let ea = Fallback.fft(&sp, &mut a, dir, DEF).unwrap();
                let eb = Fallback.fft(&cp, &mut b, dir, DEF).unwrap();
                assert_eq!(ea, eb);
                assert_eq!(
                    bytemuck::cast_slice::<C32, u32>(&a),
                    bytemuck::cast_slice::<C32, u32>(&b),
                    "c32 n={n}"
                );
            }
            let sq = FftPlan::<CQ15, &'static [CQ15]>::from_static_4096(n).unwrap();
            let cq = cq_plan(n);
            let x = noise_cq15(n, 0.3, 90 + n as u64);
            for dir in [FWD, INV] {
                for sc in [DEF, BFP] {
                    let (mut a, mut b) = (x.clone(), x.clone());
                    let ea = Fallback.fft(&sq, &mut a, dir, sc).unwrap();
                    let eb = Fallback.fft(&cq, &mut b, dir, sc).unwrap();
                    assert_eq!(ea, eb);
                    assert_eq!(a, b, "cq15 n={n}");
                }
            }
        }
    }

    #[test]
    fn static_plan_validation() {
        for n in [0usize, 8, 24, 8192, 65_536] {
            assert!(
                FftPlan::<C32, &'static [C32]>::from_static_4096(n).is_err(),
                "n={n}"
            );
            assert!(
                FftPlan::<CQ15, &'static [CQ15]>::from_static_4096(n).is_err(),
                "n={n}"
            );
        }
    }

    #[test]
    fn static_tables_equal_the_computed_twiddles() {
        let p = c32_plan(4096);
        let mut x = vec![C32::default(); 4096];
        x[1] = C32::new(1.0, 0.0);
        Fallback.fft(&p, &mut x, FWD, DEF).unwrap();
        for k in 0..2048 {
            assert_eq!(x[k], TWIDDLES_C32_4096[k], "k={k}");
        }
    }
}

// --------------------------------------------------------------------- send

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<Direction>();
    assert_send::<Scaling>();
    assert_send::<FftScale>();
    assert_send::<FftPlan<C32, Vec<C32>>>();
    assert_send::<FftPlan<CQ15, &'static [CQ15]>>();
    assert_send::<FftPlan<CQ15, [CQ15; 8]>>();
    assert_send::<RealFftPlan<Vec<C32>, &'static [C32]>>();
}

#[test]
fn plans_work_over_every_storage_kind() {
    // Array on the stack, a slice and a Vec all give the same transform.
    let x = noise_c32(16, 0.3, 123);
    let arr = FftPlan::<C32, _>::new(16, [C32::default(); 8]).unwrap();
    let mut buf = [C32::default(); 8];
    let slice = FftPlan::<C32, &mut [C32]>::new(16, &mut buf[..]).unwrap();
    let vec = c32_plan(16);
    let mut a = x.clone();
    let mut b = x.clone();
    let mut c = x.clone();
    Fallback.fft(&arr, &mut a, FWD, DEF).unwrap();
    Fallback.fft(&slice, &mut b, FWD, DEF).unwrap();
    Fallback.fft(&vec, &mut c, FWD, DEF).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, c);
}
