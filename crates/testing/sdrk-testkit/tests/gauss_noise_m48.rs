//! Statistics, the statistical cross-check with the testkit channel model and block-size
//! invariance of the M48 kernel `gauss_noise` (tests 3, 4 and 5 of the milestone).
//!
//! - Test 3 (seed 685, 10^6 values): for `f32` and `C32` (each component on its own) the mean
//!   within +-5 sigma / sqrt(n), the variance within +-1 % of sigma^2, the kurtosis within
//!   3 +- 0.03, the Kolmogorov-Smirnov distance to the normal CDF below 1.63 / sqrt(n) (the 1 %
//!   level), a chi-square test on 64 equiprobable bins of `Φ(z / sigma)` below 92.0 (the 1 %
//!   critical value for 63 degrees of freedom), the count of `|z| > 4 sigma` within
//!   `6.334e-5 n +- 5 sqrt(6.334e-5 n)`, the positive and negative counts within `5 sqrt(n)` of
//!   each other, the normalized autocorrelation at lags 1 to 16 below 4 / sqrt(n), the correlation
//!   of the real and imaginary parts below 4 / sqrt(n), every value finite and within
//!   `6.33796 sigma`. For `Q15` and `CQ15` the mean and the variance at sigma 0.1 and the clipping
//!   rate at sigma 0.6 within 10 % of `erfc(1 / (0.6 sqrt 2))`. The tail reach is checked without
//!   sampling: a seed built so that the first draw is `r = 0` (or `0x8000_0000`) gives exactly
//!   `+-830729 2^-17 sigma`.
//! - Test 4 (seed 686): `channel::Awgn` on a zero input against the kernel with the stream of
//!   `Awgn` (`0x6177_676e`, repeated here; the testkit keeps the constant private), `C32`, sigma
//!   0.7, 10^5 samples: the two-sample KS distance per component below `1.63 sqrt(2 / n)`, the
//!   variance ratio within +-2 %, the mean difference within `5 sigma sqrt(2 / n)` and
//!   `N0 = 2 sigma^2` per complex sample within +-2 % for both. The two are different sequences
//!   with the same statistics. Then the BPSK bit error rate with `f32` noise at Eb/N0 4, 6 and
//!   8 dB (2 10^6 bits each) within +-4 binomial standard errors of `Q(sqrt(2 Eb/N0))`, with at
//!   least 200 errors per point.
//! - Test 5 (seed 687): every type through the facade, `Fallback` and `Reference`, in 20 chunk
//!   plans with the helper of M03, and a `sigma` change between chunks.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::noise::{GaussNoise, GaussNoiseState, PCG32_MULT};
use sdrk_core::{C32, CQ15, Fallback, Progress, Q15, Reference, Sample};
use sdrk_testkit::channel::{Awgn, Impairment};
use sdrk_testkit::rng::Pcg32;
use sdrk_testkit::{ChunkPlan, check_block_invariance};

/// The seed of the statistics test (M48 test 3).
const STATS_SEED: u64 = 685;
/// The seed of the cross-check with `Awgn` (M48 test 4).
const AWGN_SEED: u64 = 686;
/// The PCG32 stream of `channel::Awgn` (private in the testkit, repeated here).
const AWGN_STREAM: u64 = 0x6177_676e;
/// The seed of the chunk plans (M48 test 5); plan `k` uses `PLAN_SEED * 100 + k`.
const PLAN_SEED: u64 = 687;
/// Chunk plans per type and kernel.
const PLANS: u64 = 20;
/// Largest chunk of the plans.
const MAX_CHUNK: usize = 300;
/// Values of the statistics test.
const N_STATS: usize = 1_000_000;
/// The largest magnitude of the generator in units of sigma, rounded up (`830729 2^-17`).
const ZMAX: f64 = 6.337_96;
/// `P(|N(0, 1)| > 4)`.
const P_ABOVE_4: f64 = 6.334e-5;
/// The 1 % critical value of the chi-square distribution with 63 degrees of freedom.
const CHI2_63_1PCT: f64 = 92.0;

/// The standard normal CDF.
fn phi(x: f64) -> f64 {
    0.5 * libm::erfc(-x / core::f64::consts::SQRT_2)
}

/// The mean and the variance (divided by `n`) of `x`.
fn mean_var(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var)
}

/// Checks the mean within 5 sigma / sqrt(n) and the variance within 1 % of sigma^2.
fn check_moments(x: &[f64], sigma: f64, what: &str) {
    let n = x.len() as f64;
    let (mean, var) = mean_var(x);
    assert!(
        mean.abs() <= 5.0 * sigma / n.sqrt(),
        "{what}: mean {mean} beyond 5 sigma / sqrt(n)"
    );
    let s2 = sigma * sigma;
    assert!(
        (var - s2).abs() <= 0.01 * s2,
        "{what}: variance {var}, sigma^2 {s2}"
    );
}

/// Every check of test 3 on one series `x` of standard deviation `sigma`.
fn check_series(x: &[f64], sigma: f64, what: &str) {
    let n = x.len() as f64;
    let bound = 1.0 / n.sqrt();
    check_moments(x, sigma, what);
    let (mean, var) = mean_var(x);
    let m4 = x.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    let kurtosis = m4 / (var * var);
    assert!(
        (kurtosis - 3.0).abs() <= 0.03,
        "{what}: kurtosis {kurtosis}"
    );
    // Finite and within the tail limit.
    for v in x {
        assert!(
            v.is_finite() && v.abs() <= ZMAX * sigma,
            "{what}: value {v} beyond {ZMAX} sigma"
        );
    }
    // Kolmogorov-Smirnov distance to the normal CDF.
    let mut sorted: Vec<f64> = x.iter().map(|v| v / sigma).collect();
    sorted.sort_by(f64::total_cmp);
    let mut d = 0.0f64;
    for (i, v) in sorted.iter().enumerate() {
        let f = phi(*v);
        d = d.max(f - i as f64 / n).max((i + 1) as f64 / n - f);
    }
    assert!(d < 1.63 * bound, "{what}: KS distance {d}");
    // Chi-square on 64 equiprobable bins of Φ(z / sigma).
    let mut bins = [0_u64; 64];
    for v in &sorted {
        let b = ((phi(*v) * 64.0) as usize).min(63);
        bins[b] += 1;
    }
    let e = n / 64.0;
    let chi2: f64 = bins.iter().map(|&o| (o as f64 - e).powi(2) / e).sum();
    assert!(chi2 < CHI2_63_1PCT, "{what}: chi-square {chi2}");
    // The count beyond 4 sigma.
    let above = x.iter().filter(|v| v.abs() > 4.0 * sigma).count() as f64;
    let want = P_ABOVE_4 * n;
    assert!(
        (above - want).abs() <= 5.0 * want.sqrt(),
        "{what}: {above} values beyond 4 sigma, expected {want}"
    );
    // Sign balance.
    let pos = x.iter().filter(|v| **v > 0.0).count() as f64;
    let neg = x.iter().filter(|v| **v < 0.0).count() as f64;
    assert!(
        (pos - neg).abs() <= 5.0 * n.sqrt(),
        "{what}: {pos} positive, {neg} negative"
    );
    // Normalized autocorrelation at lags 1 to 16.
    let m2 = var * n;
    for lag in 1..=16 {
        let r: f64 = x
            .iter()
            .zip(&x[lag..])
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / m2;
        assert!(
            r.abs() < 4.0 * bound,
            "{what}: autocorrelation {r} at lag {lag}"
        );
    }
}

/// The correlation coefficient of `a` and `b`.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        ab += (x - ma) * (y - mb);
        aa += (x - ma) * (x - ma);
        bb += (y - mb) * (y - mb);
    }
    ab / (aa * bb).sqrt()
}

#[test]
fn f32_noise_has_the_statistics_of_a_normal_distribution() {
    let sigma = 0.7_f32;
    let mut st = GaussNoiseState::new(STATS_SEED, 1, sigma).unwrap();
    let mut out = vec![0.0f32; N_STATS];
    Sdrk.gauss_noise(&mut st, &mut out).unwrap();
    let x: Vec<f64> = out.iter().map(|v| f64::from(*v)).collect();
    check_series(&x, f64::from(sigma), "f32");
}

#[test]
fn c32_noise_has_the_statistics_of_a_normal_distribution_per_component() {
    let sigma = 0.25_f32;
    let mut st = GaussNoiseState::new(STATS_SEED, 2, sigma).unwrap();
    let mut out = vec![C32::default(); N_STATS];
    Sdrk.gauss_noise(&mut st, &mut out).unwrap();
    let re: Vec<f64> = out.iter().map(|v| f64::from(v.re)).collect();
    let im: Vec<f64> = out.iter().map(|v| f64::from(v.im)).collect();
    check_series(&re, f64::from(sigma), "C32 re");
    check_series(&im, f64::from(sigma), "C32 im");
    let r = correlation(&re, &im);
    let bound = 4.0 / (N_STATS as f64).sqrt();
    assert!(r.abs() < bound, "correlation of re and im {r}");
    // Total power per complex sample: N0 = 2 sigma^2 (within the 1 % of the components).
    let p = out
        .iter()
        .map(|v| f64::from(v.re).powi(2) + f64::from(v.im).powi(2))
        .sum::<f64>()
        / N_STATS as f64;
    let n0 = 2.0 * f64::from(sigma).powi(2);
    assert!((p - n0).abs() <= 0.01 * n0, "power {p}, 2 sigma^2 {n0}");
}

#[test]
fn q15_noise_has_the_mean_and_variance_of_sigma_0_1() {
    let sigma = 0.1_f32;
    let mut st = GaussNoiseState::new(STATS_SEED, 5, sigma).unwrap();
    let mut out = vec![Q15(0); N_STATS];
    Sdrk.gauss_noise(&mut st, &mut out).unwrap();
    let x: Vec<f64> = out.iter().map(|q| f64::from(q.to_f32())).collect();
    check_moments(&x, f64::from(sigma), "Q15");
    let mut st = GaussNoiseState::new(STATS_SEED, 6, sigma).unwrap();
    let mut out = vec![CQ15::default(); N_STATS / 2];
    Sdrk.gauss_noise(&mut st, &mut out).unwrap();
    let re: Vec<f64> = out.iter().map(|c| f64::from(c.re.to_f32())).collect();
    let im: Vec<f64> = out.iter().map(|c| f64::from(c.im.to_f32())).collect();
    check_moments(&re, f64::from(sigma), "CQ15 re");
    check_moments(&im, f64::from(sigma), "CQ15 im");
}

#[test]
fn q15_clipping_rate_at_sigma_0_6_matches_the_theory() {
    let sigma = 0.6_f32;
    let theory = libm::erfc(1.0 / (f64::from(sigma) * core::f64::consts::SQRT_2));
    assert!((theory - 9.558e-2).abs() < 1e-5);
    let mut st = GaussNoiseState::new(STATS_SEED, 3, sigma).unwrap();
    let mut out = vec![Q15(0); N_STATS];
    Sdrk.gauss_noise(&mut st, &mut out).unwrap();
    let clipped = out
        .iter()
        .filter(|q| q.0 == i16::MAX || q.0 == i16::MIN)
        .count();
    let rate = clipped as f64 / N_STATS as f64;
    assert!(
        (rate - theory).abs() <= 0.1 * theory,
        "clipping rate {rate}, theory {theory}"
    );
    // CQ15 clips each component at the same rate.
    let mut st = GaussNoiseState::new(STATS_SEED, 4, sigma).unwrap();
    let mut out = vec![CQ15::default(); N_STATS / 2];
    Sdrk.gauss_noise(&mut st, &mut out).unwrap();
    let clipped = out
        .iter()
        .flat_map(|c| [c.re, c.im])
        .filter(|q| q.0 == i16::MAX || q.0 == i16::MIN)
        .count();
    let rate = clipped as f64 / N_STATS as f64;
    assert!(
        (rate - theory).abs() <= 0.1 * theory,
        "CQ15 clipping rate {rate}, theory {theory}"
    );
}

/// An LCG state whose PCG32 output is `rotate_right(xs, rot)`: bits 59 to 63 hold `rot`, and
/// bits 27 to 58 are chosen from the top down so that `((old >> 18) ^ old) >> 27` equals `xs`.
fn state_with_output(xs: u32, rot: u32) -> u64 {
    let mut old = u64::from(rot & 31) << 59;
    for k in (27..59).rev() {
        let want = u64::from((xs >> (k - 27)) & 1);
        let above = old.checked_shr(k + 18).unwrap_or(0) & 1;
        old |= (want ^ above) << k;
    }
    old
}

/// The PCG32 seed whose generator for `stream` stands at the LCG state `target` after `new`:
/// `new` gives `(inc + seed) MULT + inc`, so `seed = (target - inc) MULT^-1 - inc`.
fn seed_for_state(target: u64, stream: u64) -> u64 {
    let inc = (stream << 1) | 1;
    // The inverse of the odd multiplier modulo 2^64 by Newton's iteration.
    let mut inv = PCG32_MULT;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2_u64.wrapping_sub(PCG32_MULT.wrapping_mul(inv)));
    }
    assert_eq!(PCG32_MULT.wrapping_mul(inv), 1);
    target.wrapping_sub(inc).wrapping_mul(inv).wrapping_sub(inc)
}

#[test]
fn the_largest_magnitude_is_reached_at_the_extreme_draws() {
    const STREAM: u64 = 1;
    // r = 0 (q = 2^-33, the positive extreme) and r = 0x8000_0000 (the negative one).
    for (xs, rot, r, sign) in [(0_u32, 0_u32, 0_u32, 1.0_f64), (1, 1, 0x8000_0000, -1.0)] {
        let seed = seed_for_state(state_with_output(xs, rot), STREAM);
        // The testkit's own generator confirms the draw.
        assert_eq!(Pcg32::new(seed, STREAM).next_u32(), r);
        for sigma in [1.0_f32, 0.7, 16.0] {
            let zmax = 830_729.0 / 131_072.0 * f64::from(sigma) * sign;
            let mut st = GaussNoiseState::new(seed, STREAM, sigma).unwrap();
            let mut x = [0.0_f32; 1];
            Sdrk.gauss_noise(&mut st, &mut x).unwrap();
            assert_eq!(x[0], zmax as f32, "f32, sigma {sigma}, r {r:#x}");
            let mut st = GaussNoiseState::new(seed, STREAM, sigma).unwrap();
            let mut q = [Q15(0); 1];
            Sdrk.gauss_noise(&mut st, &mut q).unwrap();
            let want = (zmax * 32768.0 + 0.5).floor().clamp(-32768.0, 32767.0) as i16;
            assert_eq!(q[0].0, want, "Q15, sigma {sigma}, r {r:#x}");
        }
    }
}

/// The two-sample Kolmogorov-Smirnov distance of `a` and `b`.
fn ks_two_sample(a: &[f64], b: &[f64]) -> f64 {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j, mut d) = (0, 0, 0.0_f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / na - j as f64 / nb).abs());
    }
    d
}

#[test]
fn c32_noise_has_the_statistics_of_the_testkit_channel_model() {
    const N: usize = 100_000;
    let sigma = 0.7_f32;
    let s = f64::from(sigma);
    let mut awgn = Awgn::new(s, AWGN_SEED).unwrap();
    let mut added = Vec::new();
    awgn.process(&vec![Complex64::new(0.0, 0.0); N], &mut added);
    assert_eq!(added.len(), N);
    let mut st = GaussNoiseState::new(AWGN_SEED, AWGN_STREAM, sigma).unwrap();
    let mut out = vec![C32::default(); N];
    Sdrk.gauss_noise(&mut st, &mut out).unwrap();
    let n = N as f64;
    let parts: [(&str, Vec<f64>, Vec<f64>); 2] = [
        (
            "re",
            added.iter().map(|c| c.re).collect(),
            out.iter().map(|c| f64::from(c.re)).collect(),
        ),
        (
            "im",
            added.iter().map(|c| c.im).collect(),
            out.iter().map(|c| f64::from(c.im)).collect(),
        ),
    ];
    for (what, a, k) in &parts {
        let d = ks_two_sample(a, k);
        assert!(
            d < 1.63 * (2.0 / n).sqrt(),
            "{what}: two-sample KS distance {d}"
        );
        let (ma, va) = mean_var(a);
        let (mk, vk) = mean_var(k);
        assert!(
            (vk / va - 1.0).abs() <= 0.02,
            "{what}: variance ratio {}",
            vk / va
        );
        assert!(
            (mk - ma).abs() <= 5.0 * s * (2.0 / n).sqrt(),
            "{what}: mean difference {}",
            mk - ma
        );
    }
    let n0 = 2.0 * s * s;
    let pa = added.iter().map(Complex64::norm_sqr).sum::<f64>() / n;
    let pk = out
        .iter()
        .map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2))
        .sum::<f64>()
        / n;
    assert!((pa - n0).abs() <= 0.02 * n0, "Awgn N0 {pa}, 2 sigma^2 {n0}");
    assert!(
        (pk - n0).abs() <= 0.02 * n0,
        "kernel N0 {pk}, 2 sigma^2 {n0}"
    );
}

#[test]
fn bpsk_bit_error_rate_with_f32_noise_matches_the_theory() {
    const BITS: usize = 2_000_000;
    const CHUNK: usize = 65_536;
    for (k, ebn0_db) in [4.0_f64, 6.0, 8.0].into_iter().enumerate() {
        let ebn0 = libm::pow(10.0, ebn0_db / 10.0);
        let sigma = (1.0 / (2.0 * ebn0)).sqrt() as f32;
        let mut st = GaussNoiseState::new(AWGN_SEED, 1 + k as u64, sigma).unwrap();
        let mut bits = Pcg32::new(AWGN_SEED, 100 + k as u64);
        let mut noise = vec![0.0f32; CHUNK];
        let (mut errors, mut done) = (0_u64, 0);
        while done < BITS {
            let m = CHUNK.min(BITS - done);
            Sdrk.gauss_noise(&mut st, &mut noise[..m]).unwrap();
            for v in &noise[..m] {
                let one = bits.next_u32() & 1 == 1;
                let s = if one { 1.0_f32 } else { -1.0 };
                let rx = s + v;
                if (rx > 0.0) != one {
                    errors += 1;
                }
            }
            done += m;
        }
        let n = BITS as f64;
        let p = 0.5 * libm::erfc((2.0 * ebn0).sqrt() / core::f64::consts::SQRT_2);
        let se = (p * (1.0 - p) / n).sqrt();
        let ber = errors as f64 / n;
        assert!(errors >= 200, "{ebn0_db} dB: only {errors} errors");
        assert!(
            (ber - p).abs() <= 4.0 * se,
            "{ebn0_db} dB: BER {ber}, theory {p} (standard error {se})"
        );
    }
}

/// Block-size invariance of one type and kernel: 20 plans of the M03 helper, each against the
/// run in one call; the input is only a clock of the output length.
fn invariance<T: Sample, K: GaussNoise<T>>(k: &K, name: &str) {
    const N: usize = 4096;
    let clock = vec![0u8; N];
    for (j, sigma) in [0.0_f32, 0.3, 1.0, 16.0].into_iter().enumerate() {
        for p in 0..PLANS {
            let mut whole = vec![T::ZERO; N];
            let mut chunked = vec![T::ZERO; N];
            let report = check_block_invariance(
                &clock,
                &mut whole,
                &mut chunked,
                ChunkPlan::new(PLAN_SEED * 100 + p, MAX_CHUNK),
                || {
                    let mut st = GaussNoiseState::new(PLAN_SEED, 5 + j as u64, sigma).unwrap();
                    st.seek(12_345);
                    st
                },
                |st, i, o| {
                    let m = i.len().min(o.len());
                    k.gauss_noise(st, &mut o[..m])?;
                    Ok(Progress {
                        consumed: m,
                        produced: m,
                    })
                },
            );
            let report =
                report.unwrap_or_else(|f| panic!("{name}, sigma {sigma}, plan {p}: {f:?}"));
            assert_eq!(report.outputs, N, "{name}");
        }
    }
}

#[test]
fn every_type_is_block_size_invariant() {
    macro_rules! all {
        ($t:ty) => {
            invariance::<$t, _>(&Sdrk, concat!("facade ", stringify!($t)));
            invariance::<$t, _>(&Fallback, concat!("fallback ", stringify!($t)));
            invariance::<$t, _>(&Reference, concat!("reference ", stringify!($t)));
        };
    }
    all!(C32);
    all!(CQ15);
    all!(f32);
    all!(Q15);
}

/// A `sigma` change between chunks takes effect at the next call and keeps the position: the
/// samples after the change equal those of a state created with the new `sigma` and sought to
/// the same index.
#[test]
fn a_sigma_change_between_chunks_keeps_the_position() {
    for p in 0..PLANS {
        let mut st = GaussNoiseState::new(PLAN_SEED, 9, 0.5).unwrap();
        let mut out = vec![C32::default(); 2048];
        let mut pos = 0;
        let mut sizes = ChunkPlan::new(PLAN_SEED * 100 + p, MAX_CHUNK).sizes();
        let mut sigmas = [0.5_f32, 0.0, 2.0, 0.125].into_iter().cycle();
        let mut segments = Vec::new();
        while pos < out.len() {
            let n = sizes.next().unwrap().min(out.len() - pos);
            let sigma = sigmas.next().unwrap();
            st.set_sigma(sigma).unwrap();
            Fallback
                .gauss_noise(&mut st, &mut out[pos..pos + n])
                .unwrap();
            segments.push((pos, n, sigma));
            pos += n;
        }
        for (pos, n, sigma) in segments {
            let mut fresh = GaussNoiseState::new(PLAN_SEED, 9, sigma).unwrap();
            fresh.seek(pos as u64);
            let mut want = vec![C32::default(); n];
            Reference.gauss_noise(&mut fresh, &mut want).unwrap();
            let got = &out[pos..pos + n];
            assert!(
                want.iter()
                    .zip(got)
                    .all(|(a, b)| a.re.to_bits() == b.re.to_bits()
                        && a.im.to_bits() == b.im.to_bits()),
                "plan {p}: segment at {pos} (sigma {sigma}) differs"
            );
        }
    }
}
