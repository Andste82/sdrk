//! Tests of the M30 kernels of `sdrk-core`: `add`, `add_real`, `mul`, `mul_real`, `oscillate`,
//! `interleave` and `deinterleave` (fallback against reference, corners, length errors,
//! oscillator contract).

#![cfg(feature = "reference")]
#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::items_after_statements,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

use sdrk_core::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use sdrk_core::kernels::mix::{Nco, Oscillate, Rotate};
use sdrk_core::{C32, CQ15, ComplexSample, Error, Fallback, Q15, Reference, Sample};
use sdrk_device::{ErrorKind, SdrkError};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn q15(&mut self) -> Q15 {
        // A third of the values at the corners, the rest uniform.
        let v = self.next();
        match v % 6 {
            0 => Q15(i16::MIN),
            1 => Q15(i16::MAX),
            _ => Q15((v >> 16) as i16),
        }
    }

    fn cq15(&mut self) -> CQ15 {
        CQ15 {
            re: self.q15(),
            im: self.q15(),
        }
    }

    fn f32(&mut self) -> f32 {
        let v = self.next();
        match v % 16 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::MAX,
            3 => f32::MIN_POSITIVE / 4.0,
            _ => ((v >> 8) as i32 as f32) / 1_000_000.0,
        }
    }

    fn c32(&mut self) -> C32 {
        C32::new(self.f32(), self.f32())
    }
}

/// Bit patterns of the f32 components, so that NaN and signed zeros compare.
fn bits(x: &[C32]) -> Vec<u32> {
    x.iter()
        .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
        .collect()
}

fn fbits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

const LENS: [usize; 8] = [0, 1, 2, 3, 7, 16, 33, 1000];

#[test]
fn add_and_mul_equal_the_reference() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for n in LENS {
        let a: Vec<C32> = (0..n).map(|_| rng.c32()).collect();
        let b: Vec<C32> = (0..n).map(|_| rng.c32()).collect();
        let (mut f, mut r) = (vec![C32::ZERO; n], vec![C32::ZERO; n]);
        Fallback.add(&a, &b, &mut f).unwrap();
        Reference.add(&a, &b, &mut r).unwrap();
        assert_eq!(bits(&f), bits(&r), "add c32 n={n}");
        Fallback.mul(&a, &b, &mut f).unwrap();
        Reference.mul(&a, &b, &mut r).unwrap();
        assert_eq!(bits(&f), bits(&r), "mul c32 n={n}");

        let (x, y): (Vec<f32>, Vec<f32>) = (
            (0..n).map(|_| rng.f32()).collect(),
            (0..n).map(|_| rng.f32()).collect(),
        );
        let (mut f, mut r) = (vec![0.0f32; n], vec![0.0f32; n]);
        Fallback.add_real(&x, &y, &mut f).unwrap();
        Reference.add_real(&x, &y, &mut r).unwrap();
        assert_eq!(fbits(&f), fbits(&r), "add_real f32 n={n}");
        Fallback.mul_real(&x, &y, &mut f).unwrap();
        Reference.mul_real(&x, &y, &mut r).unwrap();
        assert_eq!(fbits(&f), fbits(&r), "mul_real f32 n={n}");

        let a: Vec<CQ15> = (0..n).map(|_| rng.cq15()).collect();
        let b: Vec<CQ15> = (0..n).map(|_| rng.cq15()).collect();
        let (mut f, mut r) = (vec![CQ15::ZERO; n], vec![CQ15::ZERO; n]);
        Fallback.add(&a, &b, &mut f).unwrap();
        Reference.add(&a, &b, &mut r).unwrap();
        assert_eq!(f, r, "add cq15 n={n}");
        Fallback.mul(&a, &b, &mut f).unwrap();
        Reference.mul(&a, &b, &mut r).unwrap();
        assert_eq!(f, r, "mul cq15 n={n}");

        let x: Vec<Q15> = (0..n).map(|_| rng.q15()).collect();
        let y: Vec<Q15> = (0..n).map(|_| rng.q15()).collect();
        let (mut f, mut r) = (vec![Q15(0); n], vec![Q15(0); n]);
        Fallback.add_real(&x, &y, &mut f).unwrap();
        Reference.add_real(&x, &y, &mut r).unwrap();
        assert_eq!(f, r, "add_real q15 n={n}");
        Fallback.mul_real(&x, &y, &mut f).unwrap();
        Reference.mul_real(&x, &y, &mut r).unwrap();
        assert_eq!(f, r, "mul_real q15 n={n}");
    }
}

#[test]
fn q15_corners_follow_the_numeric_conventions() {
    let min = Q15(i16::MIN);
    let max = Q15(i16::MAX);
    let mut o = [Q15(0)];
    Fallback.mul_real(&[min], &[min], &mut o).unwrap();
    assert_eq!(o[0], max, "-1 * -1 saturates");
    Fallback.mul_real(&[min], &[max], &mut o).unwrap();
    assert_eq!(o[0], Q15(-32767));
    Fallback.add_real(&[min], &[min], &mut o).unwrap();
    assert_eq!(o[0], min);
    Fallback.add_real(&[max], &[max], &mut o).unwrap();
    assert_eq!(o[0], max);
    // (-1 - j) * (-1 - j) = (1 - 1) + j(1 + 1): re 0, im saturates
    let m = CQ15 { re: min, im: min };
    let mut c = [CQ15::ZERO];
    Fallback.mul(&[m], &[m], &mut c).unwrap();
    assert_eq!(c[0].re, Q15(0));
    assert_eq!(c[0].im, max);
    // (-1) * (-1) in the real part: 2^30 rounds to 32768, saturates
    let r = CQ15 {
        re: min,
        im: Q15(0),
    };
    Fallback.mul(&[r], &[r], &mut c).unwrap();
    assert_eq!(
        c[0],
        CQ15 {
            re: max,
            im: Q15(0)
        }
    );
    // the sum of two products reaches 2^31: -1 * (-1 + j*-1)... with both terms positive
    let a = CQ15 { re: min, im: max };
    let b = CQ15 { re: min, im: min };
    Fallback.mul(&[a], &[b], &mut c).unwrap();
    let mut want = [CQ15::ZERO];
    Reference.mul(&[a], &[b], &mut want).unwrap();
    assert_eq!(c, want);
}

#[test]
fn cq15_mul_equals_the_reference_on_every_corner_combination() {
    // The fallback sums in `i32` with the imaginary part negated (D292): every combination of
    // ten values at and next to the ends, zero and the rounding ties, in one call per `a`
    // (so the vector loops and the rest of the fallback both see them).
    let edge = [
        -32768i16, -32767, -32766, -16384, -1, 0, 1, 16384, 32766, 32767,
    ]
    .map(Q15);
    let mut b = Vec::new();
    for &br in &edge {
        for &bi in &edge {
            b.push(CQ15 { re: br, im: bi });
        }
    }
    for &ar in &edge {
        for &ai in &edge {
            let a = vec![CQ15 { re: ar, im: ai }; b.len()];
            let (mut f, mut r) = (vec![CQ15::ZERO; b.len()], vec![CQ15::ZERO; b.len()]);
            Fallback.mul(&a, &b, &mut f).unwrap();
            Reference.mul(&a, &b, &mut r).unwrap();
            assert_eq!(f, r, "a = ({}, {})", ar.0, ai.0);
        }
    }
}

#[test]
fn f32_extremes_propagate_like_ieee() {
    let a = [C32::new(f32::NAN, 1.0), C32::new(f32::INFINITY, 0.0)];
    let b = [C32::new(1.0, 1.0), C32::new(0.0, 1.0)];
    let (mut f, mut r) = ([C32::ZERO; 2], [C32::ZERO; 2]);
    Fallback.mul(&a, &b, &mut f).unwrap();
    Reference.mul(&a, &b, &mut r).unwrap();
    // NaN lanes agree by class, every other lane by bits: the sign and payload of a NaN are not
    // part of the contract (D295; a constant-folded `inf * 0` gives `+NaN`, the hardware `-NaN`).
    let lanes = |x: &[C32]| x.iter().flat_map(|v| [v.re, v.im]).collect::<Vec<f32>>();
    for (i, (x, y)) in lanes(&f).into_iter().zip(lanes(&r)).enumerate() {
        if x.is_nan() || y.is_nan() {
            assert!(x.is_nan() && y.is_nan(), "lane {i}: {x} against {y}");
        } else {
            assert_eq!(x.to_bits(), y.to_bits(), "lane {i}");
        }
    }
    assert!(f[0].re.is_nan());
    Fallback.add(&a, &b, &mut f).unwrap();
    assert!(f[0].re.is_nan());
    assert_eq!(f[1].re, f32::INFINITY);
}

#[test]
fn unequal_lengths_are_rejected_and_change_nothing() {
    let a2 = [C32::new(1.0, 2.0); 2];
    let a3 = [C32::new(1.0, 2.0); 3];
    let sentinel = C32::new(7.0, 7.0);
    for imp in 0..2 {
        let mut out = [sentinel; 3];
        let results = [
            if imp == 0 {
                Fallback.add(&a2, &a3, &mut out)
            } else {
                Reference.add(&a2, &a3, &mut out)
            },
            if imp == 0 {
                Fallback.mul(&a3, &a3, &mut out[..2])
            } else {
                Reference.mul(&a3, &a3, &mut out[..2])
            },
        ];
        for res in results {
            let e = res.unwrap_err();
            assert!(matches!(e, Error::LengthMismatch { .. }));
            assert_eq!(e.kind(), ErrorKind::LengthMismatch);
        }
        assert_eq!(out, [sentinel; 3]);
    }
    let mut q = [Q15(5); 2];
    assert!(
        Fallback
            .add_real(&[Q15(1); 3], &[Q15(1); 3], &mut q)
            .is_err()
    );
    assert!(
        Fallback
            .mul_real(&[Q15(1); 2], &[Q15(1); 1], &mut q)
            .is_err()
    );
    assert_eq!(q, [Q15(5); 2]);
}

fn interleave_roundtrip<T: ComplexSample, I: Interleave<T>>(
    imp: &I,
    re: &[T::Real],
    im: &[T::Real],
) {
    let n = re.len();
    let mut out = vec![T::ZERO; n];
    imp.interleave(re, Some(im), &mut out).unwrap();
    for i in 0..n {
        assert_eq!(out[i].re(), re[i]);
        assert_eq!(out[i].im(), im[i]);
    }
    let (mut r, mut m) = (vec![re[0]; n], vec![re[0]; n]);
    imp.deinterleave(&out, Some(&mut r), Some(&mut m)).unwrap();
    assert_eq!((r, m), (re.to_vec(), im.to_vec()));
    // only one output
    let mut only_im = vec![re[0]; n];
    imp.deinterleave(&out, None, Some(&mut only_im)).unwrap();
    assert_eq!(only_im, im);
    imp.deinterleave(&out, None, None).unwrap();
    // im = None gives zero
    imp.interleave(re, None, &mut out).unwrap();
    for i in 0..n {
        assert_eq!(out[i].re(), re[i]);
        assert_eq!(out[i].im(), <T::Real as sdrk_core::Sample>::ZERO);
    }
    // mismatches change nothing
    let before = out.clone();
    assert!(imp.interleave(re, Some(&im[1..]), &mut out).is_err());
    assert!(imp.interleave(&re[1..], None, &mut out).is_err());
    assert_eq!(out, before);
    let mut short = vec![re[0]; n - 1];
    let mut full = vec![im[0]; n];
    assert!(
        imp.deinterleave(&before, Some(&mut full), Some(&mut short))
            .is_err()
    );
    assert!(full.iter().all(|v| *v == im[0]), "no partial write");
}

#[test]
fn interleave_and_deinterleave_copy() {
    let mut rng = Rng(99);
    let re: Vec<f32> = (0..37).map(|_| rng.f32()).collect();
    let im: Vec<f32> = (0..37).map(|_| rng.f32()).collect();
    interleave_roundtrip::<C32, _>(&Fallback, &re, &im);
    interleave_roundtrip::<C32, _>(&Reference, &re, &im);
    let re: Vec<Q15> = (0..37).map(|_| rng.q15()).collect();
    let im: Vec<Q15> = (0..37).map(|_| rng.q15()).collect();
    interleave_roundtrip::<CQ15, _>(&Fallback, &re, &im);
    interleave_roundtrip::<CQ15, _>(&Reference, &re, &im);
}

fn rate() -> sdrk_core::Rate {
    sdrk_core::Rate::hz(192_000).unwrap()
}

#[test]
fn oscillate_rejects_a_bad_harmonic_and_changes_nothing() {
    let mut nco = Nco::new(19_000.0, rate()).unwrap();
    nco.set_phase(0x1234_5678);
    let before = nco;
    for k in [0u32, 9, 100, u32::MAX] {
        let mut out = [C32::new(3.0, 3.0); 4];
        let e = Fallback.oscillate(&mut nco, k, &mut out).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)));
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
        assert_eq!(out, [C32::new(3.0, 3.0); 4]);
        assert_eq!(nco, before);
        assert!(Reference.oscillate(&mut nco, k, &mut out).is_err());
        let mut q = [CQ15::ZERO; 4];
        assert!(Fallback.oscillate(&mut nco, k, &mut q).is_err());
        assert_eq!(nco, before);
    }
    for k in 1..=8 {
        let mut out = [C32::ZERO; 4];
        Fallback.oscillate(&mut nco, k, &mut out).unwrap();
    }
}

#[test]
fn oscillate_advances_the_oscillator_like_rotate() {
    let mut a = Nco::new(-3_333.0, rate()).unwrap();
    a.set_phase(77);
    let mut b = a;
    let mut out = vec![C32::ZERO; 1001];
    Fallback.oscillate(&mut a, 3, &mut out).unwrap();
    let ones = vec![C32::new(1.0, 0.0); 1001];
    Fallback.rotate(&mut b, &ones, &mut out).unwrap();
    assert_eq!(a, b);
    let mut empty: [C32; 0] = [];
    Fallback.oscillate(&mut a, 2, &mut empty).unwrap();
    assert_eq!(a, b);
}

#[test]
fn oscillate_harmonic_one_equals_rotate_of_one() {
    for start in [0u32, 0x1234_5678, 0xFFFF_FF00, 1 << 30] {
        let mut a = Nco::new(19_000.0, rate()).unwrap();
        a.set_phase(start);
        let mut b = a;
        let n = 100_000;
        let mut osc = vec![C32::ZERO; n];
        let mut rot = vec![C32::ZERO; n];
        Fallback.oscillate(&mut a, 1, &mut osc).unwrap();
        let ones = vec![C32::new(1.0, 0.0); n];
        Fallback.rotate(&mut b, &ones, &mut rot).unwrap();
        assert_eq!(bits(&osc), bits(&rot), "start {start:#x}");
    }
    // Q15: the same values (rotate multiplies by 1 - 2^-15 at most, so compare to the table).
    let mut a = Nco::new(19_000.0, rate()).unwrap();
    let mut b = a;
    let mut f = vec![CQ15::ZERO; 4096];
    let mut r = vec![CQ15::ZERO; 4096];
    Fallback.oscillate(&mut a, 1, &mut f).unwrap();
    Reference.oscillate(&mut b, 1, &mut r).unwrap();
    assert_eq!(f, r);
}

#[test]
fn oscillate_harmonic_k_is_the_phase_multiplied() {
    let n = 100_000;
    for k in 1..=8u32 {
        let mut nco = Nco::new(19_000.0, rate()).unwrap();
        nco.set_phase(0x1234_5678);
        let (p0, inc) = (nco.phase(), nco.increment());
        let mut osc = vec![C32::ZERO; n];
        Fallback.oscillate(&mut nco, k, &mut osc).unwrap();
        assert_eq!(nco.phase(), p0.wrapping_add(inc.wrapping_mul(n as u32)));
        // the same as the harmonic 1 oscillator started at the multiplied phase and increment
        let mut one = Nco::from_increment(inc.wrapping_mul(k));
        one.set_phase(p0.wrapping_mul(k));
        let mut want = vec![C32::ZERO; n];
        Fallback.oscillate(&mut one, 1, &mut want).unwrap();
        assert_eq!(bits(&osc), bits(&want), "k={k}");

        let mut q = vec![CQ15::ZERO; n];
        let mut nco = Nco::new(19_000.0, rate()).unwrap();
        nco.set_phase(0x1234_5678);
        Fallback.oscillate(&mut nco, k, &mut q).unwrap();
        let mut one = Nco::from_increment(inc.wrapping_mul(k));
        one.set_phase(p0.wrapping_mul(k));
        let mut wq = vec![CQ15::ZERO; n];
        Fallback.oscillate(&mut one, 1, &mut wq).unwrap();
        assert_eq!(q, wq, "k={k}");
    }
}

fn snr_db(want: &[(f64, f64)], got: &[C32]) -> f64 {
    let (mut s, mut e) = (0.0f64, 0.0f64);
    for (w, g) in want.iter().zip(got) {
        s += w.0 * w.0 + w.1 * w.1;
        e += (w.0 - f64::from(g.re)).powi(2) + (w.1 - f64::from(g.im)).powi(2);
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s / e).log10()
    }
}

#[test]
fn oscillate_c32_reaches_120_db_against_exp() {
    for k in [1u32, 2, 3, 8] {
        let mut nco = Nco::new(19_000.0, rate()).unwrap();
        nco.set_phase(0x1234_5678);
        let inc = nco.increment();
        let n = 8192;
        let mut out = vec![C32::ZERO; n];
        Fallback.oscillate(&mut nco, k, &mut out).unwrap();
        let want: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let p = 0x1234_5678u32
                    .wrapping_add(inc.wrapping_mul(i as u32))
                    .wrapping_mul(k);
                let t = 2.0 * std::f64::consts::PI * f64::from(p) / 4_294_967_296.0;
                (t.cos(), t.sin())
            })
            .collect();
        let snr = snr_db(&want, &out);
        assert!(snr >= 120.0, "k={k} snr={snr}");
        // the reference is within one rounding of the exponential
        let mut r = vec![C32::ZERO; n];
        let mut nco = Nco::new(19_000.0, rate()).unwrap();
        nco.set_phase(0x1234_5678);
        Reference.oscillate(&mut nco, k, &mut r).unwrap();
        assert!(snr_db(&want, &r) > 140.0);
    }
}

#[test]
fn oscillate_equals_the_reference_for_q15_and_is_block_size_invariant() {
    let n = 20_000;
    for k in [1u32, 2, 3, 7, 8] {
        let mk = || {
            let mut nco = Nco::new(-19_000.0, rate()).unwrap();
            nco.set_phase(0xDEAD_BEEF);
            nco
        };
        let mut whole = vec![CQ15::ZERO; n];
        let mut r = vec![CQ15::ZERO; n];
        Fallback.oscillate(&mut mk(), k, &mut whole).unwrap();
        Reference.oscillate(&mut mk(), k, &mut r).unwrap();
        assert_eq!(whole, r, "k={k}");

        let mut nco = mk();
        let mut rng = Rng(5);
        let mut chunked = vec![CQ15::ZERO; n];
        let mut pos = 0;
        while pos < n {
            let len = ((rng.next() % 300) as usize + 1).min(n - pos);
            Fallback
                .oscillate(&mut nco, k, &mut chunked[pos..pos + len])
                .unwrap();
            pos += len;
        }
        assert_eq!(whole, chunked, "chunked k={k}");

        let mut wc = vec![C32::ZERO; n];
        Fallback.oscillate(&mut mk(), k, &mut wc).unwrap();
        let mut nco = mk();
        let mut cc = vec![C32::ZERO; n];
        let mut pos = 0;
        while pos < n {
            let len = ((rng.next() % 300) as usize + 1).min(n - pos);
            Fallback
                .oscillate(&mut nco, k, &mut cc[pos..pos + len])
                .unwrap();
            pos += len;
        }
        assert_eq!(bits(&wc), bits(&cc));
    }
}

#[test]
fn the_public_types_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Fallback>();
    assert_send::<Reference>();
}
