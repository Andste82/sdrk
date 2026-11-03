//! Golden tests of the M30 kernels `add`, `add_real`, `mul`, `mul_real`, `oscillate` and
//! `interleave`/`deinterleave` through the facade `Sdrk`, `Fallback` and `Reference`.
//!
//! Every case is run in one piece and in chunks of random size (the plan seed is the case seed);
//! both runs must match the golden file, so they match each other as well. The oscillator cases
//! also check that the NCO advanced by the number of samples and that the cosine tables of
//! `sdrk-core` are the ones of the generator.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]

use sdrk::Sdrk;
use sdrk_core::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use sdrk_core::kernels::mix::{Nco, Oscillate};
use sdrk_core::tables::{COS_F32, COS_Q15};
use sdrk_core::{C32, CQ15, ComplexSample, Fallback, Q15, Rate, Reference, Sample};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        write!(s, "{b:02x}").unwrap();
        s
    })
}

/// The chunk sizes of a run: the whole signal at once, or the sizes of the plan of the case.
fn chunks(len: usize, plan: Option<ChunkPlan>) -> Vec<(usize, usize)> {
    let mut sizes = plan.map(|p| p.sizes());
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < len {
        let n = sizes
            .as_mut()
            .map_or(len - pos, |s| s.next().unwrap())
            .min(len - pos);
        out.push((pos, pos + n));
        pos += n;
    }
    out
}

fn plan_of(case: &GoldenCase) -> ChunkPlan {
    ChunkPlan::new(case.seed, 300)
}

fn raw_f32(a: &Array) -> Vec<f32> {
    match a {
        Array::F32(v) => v.clone(),
        other => panic!("f32 expected, got {}", other.dtype()),
    }
}

fn raw_c32(a: &Array) -> Vec<C32> {
    match a {
        Array::CF32(v) => v.clone(),
        other => panic!("cf32 expected, got {}", other.dtype()),
    }
}

fn raw_cq15(a: &Array) -> Vec<CQ15> {
    match a {
        Array::CI16(v) => v.clone(),
        other => panic!("ci16 expected, got {}", other.dtype()),
    }
}

/// Runs `kernel` over `a` and `b` whole and chunked and compares both with `expected`.
fn element_wise<T: Sample>(
    case: &GoldenCase,
    a: &[T],
    b: &[T],
    kernel: impl Fn(&[T], &[T], &mut [T]),
) {
    let want = case.array("expected").unwrap();
    for plan in [None, Some(plan_of(case))] {
        let mut out = vec![T::ZERO; a.len()];
        for (s, e) in chunks(a.len(), plan) {
            kernel(&a[s..e], &b[s..e], &mut out[s..e]);
        }
        golden::assert_matches(case, want, &out).unwrap();
    }
}

fn add_c32<K: Add<C32>>(k: &K, case: &GoldenCase) {
    let (a, b) = (
        raw_c32(case.array("a").unwrap()),
        raw_c32(case.array("b").unwrap()),
    );
    element_wise(case, &a, &b, |a, b, o| k.add(a, b, o).unwrap());
}

fn add_cq15<K: Add<CQ15>>(k: &K, case: &GoldenCase) {
    let (a, b) = (
        raw_cq15(case.array("a").unwrap()),
        raw_cq15(case.array("b").unwrap()),
    );
    element_wise(case, &a, &b, |a, b, o| k.add(a, b, o).unwrap());
}

fn mul_c32<K: Mul<C32>>(k: &K, case: &GoldenCase) {
    let (a, b) = (
        raw_c32(case.array("a").unwrap()),
        raw_c32(case.array("b").unwrap()),
    );
    element_wise(case, &a, &b, |a, b, o| k.mul(a, b, o).unwrap());
}

fn mul_cq15<K: Mul<CQ15>>(k: &K, case: &GoldenCase) {
    let (a, b) = (
        raw_cq15(case.array("a").unwrap()),
        raw_cq15(case.array("b").unwrap()),
    );
    element_wise(case, &a, &b, |a, b, o| k.mul(a, b, o).unwrap());
}

fn add_real_f32<K: AddReal<f32>>(k: &K, case: &GoldenCase) {
    let (a, b) = (
        raw_f32(case.array("a").unwrap()),
        raw_f32(case.array("b").unwrap()),
    );
    element_wise(case, &a, &b, |a, b, o| k.add_real(a, b, o).unwrap());
}

fn add_real_q15<K: AddReal<Q15>>(k: &K, case: &GoldenCase) {
    let a = case.array("a").unwrap().as_q15().unwrap();
    let b = case.array("b").unwrap().as_q15().unwrap();
    element_wise(case, &a, &b, |a, b, o| k.add_real(a, b, o).unwrap());
}

fn mul_real_f32<K: MulReal<f32>>(k: &K, case: &GoldenCase) {
    let (a, b) = (
        raw_f32(case.array("a").unwrap()),
        raw_f32(case.array("b").unwrap()),
    );
    element_wise(case, &a, &b, |a, b, o| k.mul_real(a, b, o).unwrap());
}

fn mul_real_q15<K: MulReal<Q15>>(k: &K, case: &GoldenCase) {
    let a = case.array("a").unwrap().as_q15().unwrap();
    let b = case.array("b").unwrap().as_q15().unwrap();
    element_wise(case, &a, &b, |a, b, o| k.mul_real(a, b, o).unwrap());
}

macro_rules! for_each_kernel {
    ($f:ident, $case:expr) => {
        $f(&Sdrk, $case);
        $f(&Fallback, $case);
        $f(&Reference, $case);
    };
}

#[test]
fn the_cases_of_m30_are_present() {
    let list = |k: &str| golden::list(k).unwrap();
    assert_eq!(list("add"), ["add_c32_4096", "add_cq15_4096"]);
    assert_eq!(list("add_real"), ["add_f32_4096", "add_q15_4096"]);
    assert_eq!(list("mul"), ["mul_c32_4096", "mul_cq15_4096"]);
    assert_eq!(list("mul_real"), ["mul_f32_4096", "mul_q15_4096"]);
    assert_eq!(
        list("oscillate"),
        [
            "osc_c32_k1_8192",
            "osc_c32_k2_8192",
            "osc_cq15_k1_8192",
            "osc_cq15_k3_8192"
        ]
    );
    assert_eq!(
        list("interleave"),
        ["interleave_c32_1024", "interleave_cq15_1024"]
    );
}

#[test]
fn add_golden() {
    let case = golden::load("add", "add_c32_4096").unwrap();
    for_each_kernel!(add_c32, &case);
    let case = golden::load("add", "add_cq15_4096").unwrap();
    for_each_kernel!(add_cq15, &case);
}

#[test]
fn add_real_golden() {
    let case = golden::load("add_real", "add_f32_4096").unwrap();
    for_each_kernel!(add_real_f32, &case);
    let case = golden::load("add_real", "add_q15_4096").unwrap();
    for_each_kernel!(add_real_q15, &case);
}

#[test]
fn mul_golden() {
    let case = golden::load("mul", "mul_c32_4096").unwrap();
    for_each_kernel!(mul_c32, &case);
    let case = golden::load("mul", "mul_cq15_4096").unwrap();
    for_each_kernel!(mul_cq15, &case);
}

#[test]
fn mul_real_golden() {
    let case = golden::load("mul_real", "mul_f32_4096").unwrap();
    for_each_kernel!(mul_real_f32, &case);
    let case = golden::load("mul_real", "mul_q15_4096").unwrap();
    for_each_kernel!(mul_real_q15, &case);
}

#[test]
fn the_saturating_and_corner_cases_exercise_the_limits() {
    // The generator puts full-scale pairs and (-1)*(-1) at the start; make sure they are there.
    let case = golden::load("mul", "mul_cq15_4096").unwrap();
    let want = raw_cq15(case.array("expected").unwrap());
    assert_eq!((want[0].re, want[0].im), (Q15(i16::MAX), Q15(0)));
    let case = golden::load("mul_real", "mul_q15_4096").unwrap();
    assert_eq!(
        case.array("expected").unwrap().as_q15().unwrap()[0],
        Q15(i16::MAX)
    );
    let case = golden::load("add_real", "add_q15_4096").unwrap();
    let sums = case.array("expected").unwrap().as_q15().unwrap();
    assert!(sums.iter().take(64).any(|q| q.0 == i16::MAX));
    assert!(sums.iter().take(64).any(|q| q.0 == i16::MIN));
}

// --------------------------------------------------------------------------------------------
// oscillate
// --------------------------------------------------------------------------------------------

fn oscillate_case<T: ComplexSample, K: Oscillate<T>>(k: &K, case: &GoldenCase) {
    let inc = u32::try_from(case.param_u64("inc").unwrap()).unwrap();
    let phase0 = u32::try_from(case.param_u64("phase0").unwrap()).unwrap();
    let harmonic = u32::try_from(case.param_u64("harmonic").unwrap()).unwrap();
    // The increment computed from the frequency equals the Python model.
    let from_freq = Nco::new(case.param_f64("freq").unwrap(), Rate::hz(1).unwrap()).unwrap();
    assert_eq!(
        u64::from(from_freq.increment()),
        u64::from(inc),
        "{}",
        case.case
    );
    let want = case.array("expected").unwrap();
    let n = want.len();
    for plan in [None, Some(plan_of(case))] {
        let mut nco = Nco::from_increment(inc);
        nco.set_phase(phase0);
        let mut out = vec![T::ZERO; n];
        for (s, e) in chunks(n, plan) {
            k.oscillate(&mut nco, harmonic, &mut out[s..e]).unwrap();
        }
        golden::assert_matches(case, want, &out).unwrap();
        let expected_phase = phase0.wrapping_add(inc.wrapping_mul(n as u32));
        assert_eq!(nco.phase(), expected_phase, "{}: final phase", case.case);
    }
}

fn oscillate_c32<K: Oscillate<C32>>(k: &K, case: &GoldenCase) {
    oscillate_case::<C32, K>(k, case);
}

fn oscillate_cq15<K: Oscillate<CQ15>>(k: &K, case: &GoldenCase) {
    oscillate_case::<CQ15, K>(k, case);
}

#[test]
fn oscillate_golden() {
    for name in ["osc_c32_k1_8192", "osc_c32_k2_8192"] {
        let case = golden::load("oscillate", name).unwrap();
        for_each_kernel!(oscillate_c32, &case);
    }
    for name in ["osc_cq15_k1_8192", "osc_cq15_k3_8192"] {
        let case = golden::load("oscillate", name).unwrap();
        for_each_kernel!(oscillate_cq15, &case);
    }
}

#[test]
fn the_cosine_tables_are_the_ones_of_the_generator() {
    let f32_hash = hex(&Sha256::digest(bytemuck::cast_slice::<f32, u8>(&COS_F32)));
    let q15_hash = hex(&Sha256::digest(bytemuck::cast_slice::<i16, u8>(&COS_Q15)));
    for name in golden::list("oscillate").unwrap() {
        let case = golden::load("oscillate", &name).unwrap();
        assert_eq!(
            case.param_str("cos_f32_sha256").unwrap(),
            f32_hash,
            "{name}"
        );
        assert_eq!(
            case.param_str("cos_q15_sha256").unwrap(),
            q15_hash,
            "{name}"
        );
    }
}

// --------------------------------------------------------------------------------------------
// interleave and deinterleave
// --------------------------------------------------------------------------------------------

fn interleave_case<T: ComplexSample, K: Interleave<T>>(
    k: &K,
    case: &GoldenCase,
    re: &[T::Real],
    im: &[T::Real],
) where
    T::Real: Sample,
{
    let n = re.len();
    for plan in [None, Some(plan_of(case))] {
        for (name, with_im) in [("expected", true), ("expected_no_im", false)] {
            let mut out = vec![T::ZERO; n];
            for (s, e) in chunks(n, plan) {
                let im_part = with_im.then(|| &im[s..e]);
                k.interleave(&re[s..e], im_part, &mut out[s..e]).unwrap();
            }
            golden::assert_matches(case, case.array(name).unwrap(), &out).unwrap();
        }
        // deinterleave of the expected stream gives the inputs back; either output may be left out.
        let mut both = (vec![T::Real::ZERO; n], vec![T::Real::ZERO; n]);
        let mut only_re = vec![T::Real::ZERO; n];
        let mut only_im = vec![T::Real::ZERO; n];
        let joined = {
            let mut o = vec![T::ZERO; n];
            k.interleave(re, Some(im), &mut o).unwrap();
            o
        };
        for (s, e) in chunks(n, plan) {
            k.deinterleave(
                &joined[s..e],
                Some(&mut both.0[s..e]),
                Some(&mut both.1[s..e]),
            )
            .unwrap();
            k.deinterleave(&joined[s..e], Some(&mut only_re[s..e]), None)
                .unwrap();
            k.deinterleave(&joined[s..e], None, Some(&mut only_im[s..e]))
                .unwrap();
        }
        assert_eq!(bit_exact(&both.0, re), None);
        assert_eq!(bit_exact(&both.1, im), None);
        assert_eq!(bit_exact(&only_re, re), None);
        assert_eq!(bit_exact(&only_im, im), None);
    }
}

fn interleave_c32<K: Interleave<C32>>(k: &K, case: &GoldenCase) {
    let (re, im) = (
        raw_f32(case.array("re").unwrap()),
        raw_f32(case.array("im").unwrap()),
    );
    interleave_case::<C32, K>(k, case, &re, &im);
}

fn interleave_cq15<K: Interleave<CQ15>>(k: &K, case: &GoldenCase) {
    let re = case.array("re").unwrap().as_q15().unwrap();
    let im = case.array("im").unwrap().as_q15().unwrap();
    interleave_case::<CQ15, K>(k, case, &re, &im);
}

#[test]
fn interleave_golden() {
    let case = golden::load("interleave", "interleave_c32_1024").unwrap();
    for_each_kernel!(interleave_c32, &case);
    let case = golden::load("interleave", "interleave_cq15_1024").unwrap();
    for_each_kernel!(interleave_cq15, &case);
}
