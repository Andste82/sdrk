//! Tests of the M44.2 constellation kernels of `sdrk-core` for every square order (`qpsk`,
//! `qam16`, `qam64`, `qam256`, `qam1024`), M44.2 test 1: the soft bits of `qam_soft_bits` (one
//! generic implementation over the bits per axis): their signs at every point, derived from the
//! mapping, their magnitudes on the grid, the `f32` recursion of M44.2 §2 bit for bit (for 16-QAM
//! and 64-QAM also the written-out form of M44), the extremes and the length check; with the
//! feature `reference` the `Reference` against the `f64` definition and the fallback against
//! that optimum. The points, the decision at every boundary, `R2` and the rotation map of every
//! symbol are tested in the unit tests of `kernels::constellation`; the bit identity of 16-QAM
//! and 64-QAM with M44 in the baselines of `sdrk-testkit` (`qam_kernels_baseline_m44_2`). Seed
//! 820 of M44.2 (streams 0 and up).

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]

use sdrk_core::kernels::constellation::{Qam, QamSoftBits};
use sdrk_core::{C32, Fallback};

const SEED: u64 = 820;
const SCALES: [f32; 5] = [1.0, 0.25, 5.3, 1e-9, 1e9];

/// A small xorshift generator (deterministic, no dependency).
struct Rng(u64);

impl Rng {
    fn new(stream: u64) -> Self {
        Rng((SEED << 32) ^ (stream.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

/// Random points of `q` with uniform noise of `noise` level spacings (`2 / √E`) per component.
fn noisy_points(q: Qam, stream: u64, n: usize, noise: f32) -> Vec<C32> {
    let mut rng = Rng::new(stream);
    let d = noise * 2.0 / q.sqrt_energy();
    (0..n)
        .map(|_| {
            let p = q.point(rng.next() as u32);
            C32::new(p.re + d * rng.unit(), p.im + d * rng.unit())
        })
        .collect()
}

/// The `f32` recursion of M44.2 §2 written out per component: `t_0 = x √E`,
/// `t_i = 2^(b - i) - |t_(i - 1)|`, each times `s`.
fn soft_bits_f32(q: Qam, x: &[C32], s: f32) -> Vec<f32> {
    let c = q.sqrt_energy();
    let b = q.bits_per_axis();
    let mut out = Vec::new();
    for x in x.iter().flat_map(|x| [x.re, x.im]) {
        let mut t = x * c;
        out.push(s * t);
        for i in 1..b {
            t = (1u32 << (b - i)) as f32 - t.abs();
            out.push(s * t);
        }
    }
    out
}

/// The form of M44 §2 for 16-QAM and 64-QAM: `v`, `2 - |v|` or `v`, `4 - |v|`, `2 - ||v| - 4|`.
fn soft_bits_m44(q: Qam, x: &[C32], s: f32) -> Vec<f32> {
    let c = q.sqrt_energy();
    let mut out = Vec::new();
    for x in x.iter().flat_map(|x| [x.re, x.im]) {
        let v = x * c;
        let a = v.abs();
        out.push(s * v);
        if q == Qam::Qam16 {
            out.push(s * (2.0 - a));
        } else {
            out.push(s * (4.0 - a));
            out.push(s * (2.0 - (a - 4.0).abs()));
        }
    }
    out
}

fn soft_bits<K: QamSoftBits<C32>>(k: &K, q: Qam, x: &[C32], s: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len() * q.bits_per_symbol()];
    k.qam_soft_bits(q, x, &mut out, s).unwrap();
    out
}

/// The bits of a soft bit, every NaN folded to one value (sign and payload are not part of the
/// contract).
fn f32_bits(x: &[f32]) -> Vec<u32> {
    x.iter()
        .map(|&v| if v.is_nan() { u32::MAX } else { v.to_bits() })
        .collect()
}

#[test]
fn the_soft_bits_of_every_point_follow_the_mapping() {
    for q in Qam::ALL {
        let k = q.bits_per_symbol();
        let b = q.bits_per_axis();
        let m = q.levels_per_axis() as u32;
        let out = soft_bits(&Fallback, q, q.point_table(), 1.0);
        for (s, bits) in out.chunks_exact(k).enumerate() {
            // The level indices of both axes, found by searching the Gray codes.
            let level = |g: u32| (0..m).find(|&i| (i ^ (i >> 1)) == g).unwrap();
            let (li, lq) = (level(s as u32 >> b), level(s as u32 & (m - 1)));
            for (i, &v) in bits.iter().enumerate() {
                let bit = (s >> (k - 1 - i)) & 1 == 1;
                assert_eq!(v > 0.0, bit, "{q:?} symbol {s} bit {i}: {v}");
                // On the grid every soft bit is an odd integer, `t_i` of the integer level.
                let l = if i < b { li } else { lq };
                let mut t = 2 * l as i32 - (m as i32 - 1);
                for j in 1..=i % b {
                    t = (1 << (b - j)) - t.abs();
                }
                assert!(
                    (v - t as f32).abs() < 2e-5 * m as f32,
                    "{q:?} {s} bit {i}: {v} {t}"
                );
            }
        }
    }
}

#[test]
fn the_soft_bits_are_the_f32_recursion() {
    for (stream, q) in Qam::ALL.into_iter().enumerate() {
        let x = noisy_points(q, stream as u64, 3001, 0.7);
        for s in SCALES {
            let got = f32_bits(&soft_bits(&Fallback, q, &x, s));
            assert_eq!(got, f32_bits(&soft_bits_f32(q, &x, s)), "{q:?} scale {s}");
            if matches!(q, Qam::Qam16 | Qam::Qam64) {
                assert_eq!(got, f32_bits(&soft_bits_m44(q, &x, s)), "{q:?} scale {s}");
            }
        }
    }
}

#[test]
fn the_soft_bits_handle_the_extremes_and_check_the_length() {
    for q in Qam::ALL {
        let k = q.bits_per_symbol();
        let b = q.bits_per_axis();
        let x = [
            C32::new(0.0, -0.0),
            C32::new(f32::NAN, f32::INFINITY),
            C32::new(f32::NEG_INFINITY, f32::MAX),
        ];
        let out = soft_bits(&Fallback, q, &x, 1.0);
        assert_eq!(f32_bits(&out), f32_bits(&soft_bits_f32(q, &x, 1.0)));
        // Zero: the first soft bit keeps the sign of the zero, the others are finite.
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[b].to_bits(), (-0.0f32).to_bits());
        assert!(out[..k].iter().all(|v| v.is_finite()));
        // NaN gives NaN soft bits; an infinite component infinite ones.
        assert!(out[k..k + b].iter().all(|v| v.is_nan()));
        assert_eq!(out[k + b], f32::INFINITY);
        assert!(
            out[k + b + 1..2 * k]
                .iter()
                .all(|&v| v == f32::NEG_INFINITY)
        );
        assert_eq!(out[2 * k], f32::NEG_INFINITY);
        for got in [0, 3 * k - 1, 3 * k + 1] {
            assert_eq!(
                Fallback.qam_soft_bits(q, &x, &mut vec![0.0; got], 1.0),
                Err(sdrk_core::Error::LengthMismatch {
                    expected: 3 * k,
                    got
                })
            );
        }
        assert_eq!(Fallback.qam_soft_bits(q, &[], &mut [], 1.0), Ok(()));
    }
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    /// The `f64` definition with the exact `√E`, each soft bit times the scale.
    fn definition_f64(q: Qam, x: &[C32], s: f32) -> Vec<f64> {
        let c = libm::sqrt(q.energy());
        let b = q.bits_per_axis();
        let mut out = Vec::new();
        for x in x.iter().flat_map(|x| [x.re, x.im]) {
            let mut t = f64::from(x) * c;
            out.push(f64::from(s) * t);
            for i in 1..b {
                t = f64::from(1u32 << (b - i)) - t.abs();
                out.push(f64::from(s) * t);
            }
        }
        out
    }

    fn snr_db(got: &[f32], want: &[f64]) -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for (g, w) in got.iter().zip(want) {
            num += w * w;
            den += (w - f64::from(*g)) * (w - f64::from(*g));
        }
        10.0 * libm::log10(num / den)
    }

    /// The reference is the `f64` definition rounded once; the fallback stays above the contract
    /// floor of 140 dB (D1413) and within 9 dB of that optimum (measured 1.5 to 8.5 dB, the
    /// largest for 256-QAM: the steps of the recursion subtract from the rounded `v`).
    #[test]
    fn the_fallback_against_the_reference() {
        for (stream, q) in Qam::ALL.into_iter().enumerate() {
            let x = noisy_points(q, 10 + stream as u64, 4096, 0.7);
            for s in [1.0f32, 0.25, 5.3, 0.7] {
                let want = definition_f64(q, &x, s);
                let r = soft_bits(&Reference, q, &x, s);
                for (r, w) in r.iter().zip(&want) {
                    assert_eq!(r.to_bits(), (*w as f32).to_bits(), "{q:?}");
                }
                let (fs, opt) = (
                    snr_db(&soft_bits(&Fallback, q, &x, s), &want),
                    snr_db(&r, &want),
                );
                assert!(
                    fs >= 140.0 && opt - fs < 9.0,
                    "{q:?} scale {s}: {fs:.1} dB, optimum {opt:.1}"
                );
            }
        }
    }
}
