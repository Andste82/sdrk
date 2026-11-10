//! Golden tests of the M44 kernel `qam_soft_bits` (open item I44-1, decisions D1412 to D1414)
//! through the facade `Sdrk`, `Fallback` and `Reference`.
//!
//! Every case must match the golden file bit for bit on the facade (the SIMD backend where the
//! build has it) and on the fallback, whole and in 20 cut plans of seed 777 (the kernel is
//! stateless, so a plan cuts the input into pieces of whole symbols); the `f64` definition of the
//! `Reference` checks the stored `snr_db_guard` (D111) and `snr_db_optimum`.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]

use sdrk::Sdrk;
use sdrk_core::kernels::constellation::{Qam, QamSoftBits};
use sdrk_core::{C32, Fallback, Reference};
use sdrk_testkit::ChunkPlan;
use sdrk_testkit::golden::{self, Array, GoldenCase};

const CASES: [&str; 2] = ["qam16_c32_4096", "qam64_c32_4096"];
/// The seed of the cut plans; plan `k` uses `PLAN_SEED * 100 + k`.
const PLAN_SEED: u64 = 777;
/// Cut plans per case.
const PLANS: u64 = 20;
/// The largest piece of the plans, in symbols.
const MAX_CHUNK: usize = 300;

#[test]
fn the_cases_of_m44_are_present() {
    assert_eq!(golden::list("qam_soft_bits").unwrap(), CASES);
}

fn qam_of(case: &GoldenCase) -> Qam {
    match case.param_str("constellation").unwrap() {
        "qam16" => Qam::Qam16,
        "qam64" => Qam::Qam64,
        other => panic!("{}: unknown constellation {other}", case.case),
    }
}

fn scale_of(case: &GoldenCase) -> f32 {
    f32::from_bits(u32::try_from(case.param_u64("scale_bits").unwrap()).unwrap())
}

fn input_of(case: &GoldenCase) -> &[C32] {
    let Array::CF32(v) = case.array("input").unwrap() else {
        panic!("{}: cf32 input expected", case.case);
    };
    v
}

/// The soft bits of `input`: whole, or in the pieces of plan `k`.
fn run<K: QamSoftBits<C32>>(
    kernel: &K,
    qam: Qam,
    input: &[C32],
    scale: f32,
    plan: Option<u64>,
) -> Vec<f32> {
    let bits = qam.bits_per_symbol();
    let mut out = vec![0.0f32; input.len() * bits];
    match plan {
        None => kernel.qam_soft_bits(qam, input, &mut out, scale).unwrap(),
        Some(k) => {
            let mut sizes = ChunkPlan::new(PLAN_SEED * 100 + k, MAX_CHUNK).sizes();
            let mut pos = 0;
            while pos < input.len() {
                let a = sizes.next().unwrap().min(input.len() - pos);
                kernel
                    .qam_soft_bits(
                        qam,
                        &input[pos..pos + a],
                        &mut out[pos * bits..(pos + a) * bits],
                        scale,
                    )
                    .unwrap();
                pos += a;
            }
        }
    }
    out
}

/// The SNR of `got` and of the `f64` values rounded once to `f32` (the optimum) against the
/// `f64` values.
fn snr_db(got: &[f32], f64_of: &[f64]) -> (f64, f64) {
    let (mut num, mut den, mut den_opt) = (0.0, 0.0, 0.0);
    for (&g, &r) in got.iter().zip(f64_of) {
        num += r * r;
        den += (r - f64::from(g)) * (r - f64::from(g));
        den_opt += (r - f64::from(r as f32)) * (r - f64::from(r as f32));
    }
    (
        10.0 * libm::log10(num / den),
        10.0 * libm::log10(num / den_opt),
    )
}

/// The `f64` definition of M44 §2 with the exact `√E` (the `Reference` before its rounding).
fn definition_f64(qam: Qam, input: &[C32], scale: f32) -> Vec<f64> {
    let c = libm::sqrt(qam.energy());
    let s = f64::from(scale);
    let mut out = Vec::with_capacity(input.len() * qam.bits_per_symbol());
    for x in input.iter().flat_map(|x| [x.re, x.im]) {
        let v = f64::from(x) * c;
        let a = v.abs();
        out.push(s * v);
        if qam == Qam::Qam16 {
            out.push(s * (2.0 - a));
        } else {
            out.push(s * (4.0 - a));
            out.push(s * (2.0 - (a - 4.0).abs()));
        }
    }
    out
}

#[test]
fn qam_soft_bits_golden() {
    for name in CASES {
        let case = golden::load("qam_soft_bits", name).unwrap();
        let (qam, scale, input) = (qam_of(&case), scale_of(&case), input_of(&case));
        let expected = case.array("expected").unwrap();
        let Array::F32(want) = expected else {
            panic!("{name}: f32 expected");
        };
        assert_eq!(want.len(), input.len() * qam.bits_per_symbol());
        let facade = run(&Sdrk, qam, input, scale, None);
        golden::assert_matches(&case, expected, &facade).unwrap();
        golden::assert_matches(&case, expected, &run(&Fallback, qam, input, scale, None)).unwrap();
        for k in 0..PLANS {
            for (what, got) in [
                ("facade", run(&Sdrk, qam, input, scale, Some(k))),
                ("fallback", run(&Fallback, qam, input, scale, Some(k))),
            ] {
                let same = got
                    .iter()
                    .zip(want)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{name}: {what} differs in cut plan {k}");
            }
        }
        // The reference is the f64 definition rounded once: the optimum of an f32 output.
        let f64_of = definition_f64(qam, input, scale);
        let reference = run(&Reference, qam, input, scale, None);
        for (r, d) in reference.iter().zip(&f64_of) {
            assert_eq!(r.to_bits(), (*d as f32).to_bits(), "{name}");
        }
        let (snr, optimum) = snr_db(&facade, &f64_of);
        let guard = case.param_f64("snr_db_guard").unwrap();
        assert!(
            snr >= guard && snr < guard + 3.0,
            "{name}: SNR {snr:.2} dB against the float64 definition, guard {guard} dB"
        );
        let stored = case.param_f64("snr_db_optimum").unwrap();
        assert!(
            (optimum - stored).abs() < 0.06,
            "{name}: optimum {optimum:.2} dB, stored {stored}"
        );
    }
}
