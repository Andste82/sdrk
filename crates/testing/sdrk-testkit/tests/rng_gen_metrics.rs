//! Tests of `rng`, `gen` and `metrics` (milestone tests 1 to 3).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unnecessary_wraps,
    clippy::ignored_unit_patterns,
    clippy::items_after_statements,
    clippy::result_large_err,
    clippy::needless_pass_by_value,
    clippy::match_wildcard_for_single_variants,
    clippy::single_match_else
)]

use num_complex::Complex64;
use sdrk_core::{CQ15, Q15};
use sdrk_testkit::r#gen::{self, extreme};
use sdrk_testkit::metrics::{bit_exact, compare_complex, compare_real};
use sdrk_testkit::rng::Pcg32;

// With `num-complex/std` (enabled by the dev-dependencies of other workspace crates through
// feature unification) `Complex64::norm` is an inherent method and this trait is not needed.
#[allow(dead_code)]
trait Norm {
    fn norm(self) -> f64;
}
impl Norm for Complex64 {
    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}
fn polar(r: f64, a: f64) -> Complex64 {
    Complex64::new(r * a.cos(), r * a.sin())
}

const Z: Complex64 = Complex64::new(0.0, 0.0);

#[test]
fn pcg32_known_answer() {
    let mut rng = Pcg32::new(42, 54);
    let got: Vec<u32> = (0..6).map(|_| rng.next_u32()).collect();
    assert_eq!(
        got,
        [
            0xa15c_02b7,
            0x7b47_f409,
            0xba1d_3330,
            0x83d2_f293,
            0xbfa4_784b,
            0xcbed_606e
        ]
    );
}

#[test]
fn pcg32_u64_takes_the_high_half_first_and_f64_is_in_range() {
    let mut a = Pcg32::new(1, 2);
    let mut b = a.clone();
    let v = a.next_u64();
    let hi = u64::from(b.next_u32());
    let lo = u64::from(b.next_u32());
    assert_eq!(v, (hi << 32) | lo);
    let mut rng = Pcg32::new(3, 4);
    for _ in 0..100_000 {
        let x = rng.next_f64();
        assert!((0.0..1.0).contains(&x));
    }
}

#[test]
fn below_stays_in_range_and_is_roughly_uniform() {
    let mut rng = Pcg32::new(9, 9);
    let mut counts = [0u32; 10];
    for _ in 0..1_000_000 {
        let v = rng.below(10);
        assert!(v < 10);
        counts[v as usize] += 1;
    }
    for c in counts {
        assert!((99_000..=101_000).contains(&c), "count {c}");
    }
    assert_eq!(rng.below(0), 0);
    assert_eq!(rng.below(1), 0);
    for n in [3u32, 1 << 31, u32::MAX] {
        assert!((0..1000).all(|_| rng.below(n) < n));
    }
}

#[test]
fn normal_has_zero_mean_and_unit_variance() {
    let mut rng = Pcg32::new(2024, 1);
    let n = 1_000_000;
    let (mut sum, mut sq) = (0.0f64, 0.0f64);
    for _ in 0..n {
        let x = rng.normal();
        sum += x;
        sq += x * x;
    }
    let mean = sum / f64::from(n);
    let var = sq / f64::from(n) - mean * mean;
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!((var - 1.0).abs() < 0.01, "variance {var}");
}

#[test]
fn advance_matches_stepping() {
    for delta in [0u64, 1, 2, 3, 100, 12_345] {
        let mut a = Pcg32::new(8, 8);
        let mut b = a.clone();
        for _ in 0..delta {
            a.next_u32();
        }
        b.advance(delta);
        assert_eq!(a.next_u32(), b.next_u32());
    }
}

#[test]
fn tone_in_chunks_equals_tone_at_once_bit_for_bit() {
    let freq = 0.123_456_789;
    let mut whole = vec![Z; 1000];
    r#gen::tone(&mut whole, freq, 0.7, 0.3, 5);
    let mut rng = Pcg32::new(5, 5);
    let mut chunked = vec![Z; 1000];
    let mut pos = 0;
    let mut chunks = 0;
    while pos < 1000 {
        let len = (1 + rng.below(300) as usize).min(1000 - pos);
        r#gen::tone(&mut chunked[pos..pos + len], freq, 0.7, 0.3, 5 + pos as u64);
        pos += len;
        chunks += 1;
    }
    assert!(chunks >= 4);
    let w: Vec<u64> = whole
        .iter()
        .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
        .collect();
    let c: Vec<u64> = chunked
        .iter()
        .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
        .collect();
    assert_eq!(w, c);
}

#[test]
fn tone_has_its_peak_at_bin_37_with_100_db_dynamic_range() {
    const N: usize = 1024;
    let mut x = vec![Z; N];
    r#gen::tone(&mut x, 37.0 / N as f64, 1.0, 0.0, 0);
    let mags: Vec<f64> = (0..N)
        .map(|k| {
            let mut acc = Z;
            for (n, s) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * ((k * n) % N) as f64 / N as f64;
                acc += s * Complex64::new(a.cos(), a.sin());
            }
            acc.norm()
        })
        .collect();
    let peak = mags[37];
    for (k, m) in mags.iter().enumerate().filter(|(k, _)| *k != 37) {
        assert!(20.0 * (peak / m).log10() >= 100.0, "bin {k}");
    }
}

#[test]
fn tone_values_and_negative_frequency() {
    let mut x = [Z; 4];
    r#gen::tone(&mut x, -0.25, 2.0, 0.0, 0);
    assert!((x[0] - Complex64::new(2.0, 0.0)).norm() < 1e-12);
    assert!((x[1] - Complex64::new(0.0, -2.0)).norm() < 1e-12);
    assert!((x[2] - Complex64::new(-2.0, 0.0)).norm() < 1e-12);
    let mut r = [0.0; 4];
    r#gen::real_tone(&mut r, 0.25, 1.0, 0.0, 0);
    assert!((r[0] - 1.0).abs() < 1e-12 && r[1].abs() < 1e-12 && (r[2] + 1.0).abs() < 1e-12);
}

#[test]
fn multitone_is_the_sum_of_tones() {
    let tones = [(0.1, 0.5, 0.0), (-0.2, 0.25, 1.0)];
    let mut sum = vec![Z; 64];
    r#gen::multitone(&mut sum, &tones, 3);
    let (mut a, mut b) = (vec![Z; 64], vec![Z; 64]);
    r#gen::tone(&mut a, 0.1, 0.5, 0.0, 3);
    r#gen::tone(&mut b, -0.2, 0.25, 1.0, 3);
    for i in 0..64 {
        assert!((sum[i] - (a[i] + b[i])).norm() < 1e-12);
    }
}

#[test]
fn noise_power_is_within_one_percent() {
    let mut x = vec![Z; 100_000];
    r#gen::noise(&mut x, &mut Pcg32::new(1, 1), 0.5);
    let p = x
        .iter()
        .map(|c| {
            let n = c.norm();
            n * n
        })
        .sum::<f64>()
        / x.len() as f64;
    assert!((p / 0.5 - 1.0).abs() < 0.01, "power {p}");
    let mut r = vec![0.0; 100_000];
    r#gen::real_noise(&mut r, &mut Pcg32::new(2, 1), 4.0);
    let p = r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64;
    assert!((p / 4.0 - 1.0).abs() < 0.01, "power {p}");
}

#[test]
fn chirp_impulse_step_and_chunking() {
    let mut whole = vec![Z; 200];
    r#gen::chirp(&mut whole, 0.0, 0.25, 200, 1.0, 0);
    let mut parts = vec![Z; 200];
    for (i, piece) in parts.chunks_mut(33).enumerate() {
        r#gen::chirp(piece, 0.0, 0.25, 200, 1.0, (i * 33) as u64);
    }
    assert_eq!(
        compare_complex(&whole, parts.iter().copied(), 0).first_mismatch,
        None
    );
    // phase of sample n: 2 pi (f0 n + (f1 - f0) n^2 / (2 len))
    let n = 100.0f64;
    let want = 2.0 * std::f64::consts::PI * (0.25 * n * n / 400.0);
    assert!((whole[100] - polar(1.0, want)).norm() < 1e-9);
    let mut imp = [Z; 8];
    r#gen::impulse(&mut imp, 10, 0.5, 8);
    assert_eq!(imp[2], Complex64::new(0.5, 0.0));
    assert_eq!(imp.iter().filter(|c| c.re != 0.0).count(), 1);
    let mut st = [Z; 8];
    r#gen::step(&mut st, 3, 1.0, 0);
    assert_eq!(st.map(|c| c.re), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn extreme_signals() {
    let mut q = [CQ15 {
        re: Q15(5),
        im: Q15(5),
    }; 3];
    extreme::silence(&mut q);
    assert!(q.iter().all(|s| s.re == Q15(0) && s.im == Q15(0)));
    extreme::full_scale_alternating(&mut q);
    assert_eq!((q[0].re, q[0].im), (Q15(32767), Q15(32767)));
    assert_eq!((q[1].re, q[1].im), (Q15(-32767), Q15(-32767)));
    assert_eq!(q[2].re, Q15(32767));
    let mut f = [0.0f32; 9];
    extreme::f32_specials(&mut f);
    assert!(f[0] == 0.0 && f[0].is_sign_positive());
    assert!(f[1] == 0.0 && f[1].is_sign_negative());
    assert!(f[2].is_nan());
    assert_eq!((f[3], f[4]), (f32::INFINITY, f32::NEG_INFINITY));
    assert!(f[5] > 0.0 && f[5] < f32::MIN_POSITIVE);
    assert_eq!(f[6], f32::MAX);
    assert!(f[7] == 0.0 && f[7].is_sign_positive());
    let mut c = [sdrk_core::C32::new(0.0, 0.0); 4];
    extreme::f32_specials_complex(&mut c);
    assert!(c[0].re == 0.0 && c[0].im == f32::INFINITY);
}

#[test]
fn write_as_rounds_into_the_sample_type() {
    let src = [Complex64::new(1.0, -1.0), Complex64::new(0.5, f64::NAN)];
    let mut dst = [CQ15::default(); 2];
    r#gen::write_as(&src, &mut dst);
    assert_eq!(
        dst[0],
        CQ15 {
            re: Q15(32767),
            im: Q15(-32768)
        }
    );
    assert_eq!(
        dst[1],
        CQ15 {
            re: Q15(16384),
            im: Q15(0)
        }
    );
}

#[test]
fn compare_complex_metrics() {
    let e = vec![Complex64::new(1.0, 0.0); 100];
    let same = compare_complex(&e, e.iter().copied(), 0);
    assert_eq!(same.snr_db, f64::INFINITY);
    assert_eq!(
        (same.max_abs_err, same.rms_err, same.first_mismatch),
        (0.0, 0.0, None)
    );
    assert_eq!(same.compared, 100);
    // error 0.1 on every sample: SNR = 20 dB
    let a: Vec<_> = e.iter().map(|c| c + Complex64::new(0.0, 0.1)).collect();
    let c = compare_complex(&e, a.iter().copied(), 0);
    assert!((c.snr_db - 20.0).abs() < 1e-9);
    assert!((c.max_abs_err - 0.1).abs() < 1e-12 && (c.rms_err - 0.1).abs() < 1e-12);
    assert_eq!(c.first_mismatch, Some(0));
    // skip ignores the leading samples
    let mut b = e.clone();
    b[3] = Complex64::new(2.0, 0.0);
    let c = compare_complex(&e, b.iter().copied(), 4);
    assert_eq!(
        (c.snr_db, c.compared, c.first_mismatch),
        (f64::INFINITY, 96, None)
    );
    let c = compare_complex(&e, b.iter().copied(), 0);
    assert_eq!(c.first_mismatch, Some(3));
    // zero signal with error
    let z = vec![Z; 4];
    let c = compare_complex(&z, e.iter().copied(), 0);
    assert_eq!(c.snr_db, f64::NEG_INFINITY);
    // NaN is a mismatch
    let c = compare_complex(&e, std::iter::repeat(Complex64::new(f64::NAN, 0.0)), 0);
    assert_eq!(c.first_mismatch, Some(0));
    assert!(c.max_abs_err.is_nan());
    // shorter actual: only the common part is compared
    assert_eq!(
        compare_complex(&e, e.iter().copied().take(10), 0).compared,
        10
    );
}

#[test]
fn compare_real_metrics() {
    let e = [1.0, -1.0, 1.0, -1.0];
    let c = compare_real(&e, [1.0, -1.0, 1.0, -0.9].into_iter(), 0);
    assert_eq!(c.first_mismatch, Some(3));
    assert!((c.max_abs_err - 0.1).abs() < 1e-12);
    assert!((c.snr_db - 10.0 * (4.0f64 / 0.01).log10()).abs() < 1e-9);
    let c = compare_real(&e, e.into_iter(), 2);
    assert_eq!((c.snr_db, c.compared), (f64::INFINITY, 2));
}

#[test]
fn bit_exact_compares_raw_bits() {
    assert_eq!(bit_exact(&[Q15(1), Q15(2)], &[Q15(1), Q15(2)]), None);
    assert_eq!(bit_exact(&[Q15(1), Q15(2)], &[Q15(1), Q15(3)]), Some(1));
    assert_eq!(bit_exact(&[0.0f32, 1.0], &[0.0, -0.0]), Some(1));
    assert_eq!(bit_exact(&[f32::NAN], &[f32::NAN]), None);
    assert_eq!(bit_exact(&[1.0f32], &[1.0, 2.0]), Some(1));
    assert_eq!(bit_exact::<f32>(&[], &[]), None);
}
