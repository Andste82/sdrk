//! Golden tests of the M36 kernels `preemph_limit_stereo` and `stereo_encode` through the facade
//! `Sdrk`, `Fallback` and `Reference`.
//!
//! Every case is run in one piece and in 20 chunk plans of random size; every run must match the
//! golden file bit for bit and leave the same final state. The state is built from the values
//! stored in the case (the limiter coefficients as IEEE bits). Every case checks its
//! `snr_db_guard` against its own float64 loop on the stored inputs (the regression guard of
//! decision D111 for the limiter, the accuracy guard of D34 for the encoder).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::manual_midpoint
)]

use num_complex::Complex64;
use sdrk::Sdrk;
use sdrk_core::kernels::preemph_limit::{PreemphLimitStereo, PreemphLimitStereoState};
use sdrk_core::kernels::stereo::{StereoEncode, StereoEncodeState};
use sdrk_core::tables::COS_F32;
use sdrk_core::{C32, Fallback, Rate, Reference};
use sdrk_testkit::golden::{self, Array, GoldenCase};
use sdrk_testkit::{ChunkPlan, bit_exact};
use sha2::{Digest, Sha256};

/// Chunk plans per kernel and case.
const PLANS: u64 = 20;

/// The chunk bounds of a run: the whole signal at once, or the sizes of a plan.
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

/// The plans of a case: the whole signal, then 20 plans with chunks of 1 to 700 samples.
fn plans(case: &GoldenCase) -> Vec<Option<ChunkPlan>> {
    let mut out = vec![None];
    out.extend((0..PLANS).map(|k| Some(ChunkPlan::new(case.seed * 100 + k, 700))));
    out
}

fn rate_of(case: &GoldenCase) -> Rate {
    Rate::hz(case.param_u64("rate_hz").unwrap()).unwrap()
}

fn c32_input(case: &GoldenCase) -> &[C32] {
    let Array::CF32(input) = case.array("input").unwrap() else {
        panic!("{}: cf32 input expected", case.case);
    };
    input
}

fn snr_db(reference: &[Complex64], test: &[Complex64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (r, t) in reference.iter().zip(test) {
        num += r.norm_sqr();
        den += (r - t).norm_sqr();
    }
    if den == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(num / den)
    }
}

/// The SNR of `got` against the float64 output `want` from `guard_skip` (0 when absent) on must
/// reach the stored guard and stay below the guard plus 3 dB (so the guard is not vacuous).
fn check_guard(case: &GoldenCase, want: &[Complex64], got: &[Complex64]) {
    let skip = case
        .param_u64("guard_skip")
        .map_or(0, |s| usize::try_from(s).unwrap());
    let guard = case.param_f64("snr_db_guard").unwrap();
    let snr = snr_db(&want[skip..], &got[skip..]);
    assert!(
        snr >= guard && snr < guard + 3.0,
        "{}: SNR {snr:.2} dB against the float64 loop, guard {guard} dB",
        case.case
    );
}

#[test]
fn the_cases_of_m36_are_present() {
    let stereo: Vec<String> = golden::list("preemph_limit")
        .unwrap()
        .into_iter()
        .filter(|n| n.starts_with("stereo_"))
        .collect();
    assert_eq!(stereo, ["stereo_equal_c32_8192", "stereo_limit_c32_16384"]);
    assert_eq!(golden::list("stereo_encode").unwrap(), ["c32_16384"]);
}

// --------------------------------------------------------------------------------------------
// preemph_limit_stereo
// --------------------------------------------------------------------------------------------

fn coefs_of(case: &GoldenCase) -> [f64; 3] {
    let bits = |key: &str| f64::from_bits(case.param_u64(key).unwrap());
    [bits("b0_bits"), bits("b1_bits"), bits("a1_bits")]
}

fn limiter_state(case: &GoldenCase) -> PreemphLimitStereoState {
    PreemphLimitStereoState::new(
        coefs_of(case),
        case.param_f64("limit").unwrap(),
        case.param_f64("hold_s").unwrap(),
        case.param_f64("release_s").unwrap(),
        rate_of(case),
    )
    .unwrap()
}

/// Runs `kernel` whole and in the 20 plans; every run must match `expected` and end in the same
/// state.
fn limiter_check<K: PreemphLimitStereo>(kernel: &K, case: &GoldenCase) -> Vec<C32> {
    let input = c32_input(case);
    let want = case.array("expected").unwrap();
    let mut first: Option<(Vec<C32>, PreemphLimitStereoState)> = None;
    for plan in plans(case) {
        let mut state = limiter_state(case);
        let mut out = vec![C32::new(0.0, 0.0); input.len()];
        for (a, b) in chunks(input.len(), plan) {
            kernel
                .preemph_limit_stereo(&mut state, &input[a..b], &mut out[a..b])
                .unwrap();
        }
        golden::assert_matches(case, want, &out).unwrap();
        match &first {
            None => first = Some((out, state)),
            Some((whole, end)) => {
                assert_eq!(
                    bit_exact(whole, &out),
                    None,
                    "{}: block size changed the output",
                    case.case
                );
                assert_eq!(*end, state, "{}: block size changed the state", case.case);
            }
        }
    }
    first.unwrap().0
}

/// The limiter in float64 (float64 coefficients, release factor and limit, no flush), the
/// regression guard of D111; the model of `preemph_stereo_f64` of the generator.
fn limiter_f64(case: &GoldenCase, input: &[C32]) -> Vec<Complex64> {
    let [b0, b1, a1] = coefs_of(case);
    let fs = case.param_f64("rate_hz").unwrap();
    let t = case.param_f64("limit").unwrap();
    let hold_len = libm::floor(case.param_f64("hold_s").unwrap() * fs + 0.5) as u64;
    let r = libm::exp(-1.0 / (fs * case.param_f64("release_s").unwrap()));
    let sanitize = |x: f32| {
        if x.is_finite() {
            f64::from(x).clamp(-1024.0, 1024.0)
        } else {
            0.0
        }
    };
    let (mut x1, mut v1) = ([0.0f64; 2], [0.0f64; 2]);
    let (mut env, mut hold) = (0.0f64, 0u64);
    let clamp = |y: f64| {
        if y > t {
            t
        } else if y < -t {
            -t
        } else {
            y
        }
    };
    input
        .iter()
        .map(|x| {
            let mut v = [0.0f64; 2];
            for (ch, s) in [x.re, x.im].into_iter().enumerate() {
                let xs = sanitize(s);
                v[ch] = (b0 * xs + b1 * x1[ch]) - a1 * v1[ch];
                x1[ch] = xs;
                v1[ch] = v[ch];
            }
            let (ml, mr) = (v[0].abs(), v[1].abs());
            let m = if ml > mr { ml } else { mr };
            if m >= env {
                env = m;
                hold = hold_len;
            } else if hold > 0 {
                hold -= 1;
            } else {
                env *= r;
            }
            let g = if env > t { t / env } else { 1.0 };
            Complex64::new(clamp(v[0] * g), clamp(v[1] * g))
        })
        .collect()
}

fn limiter_case(name: &str) {
    let case = golden::load("preemph_limit", name).unwrap();
    let facade = limiter_check(&Sdrk, &case);
    let fallback = limiter_check(&Fallback, &case);
    let reference = limiter_check(&Reference, &case);
    assert_eq!(bit_exact(&facade, &fallback), None);
    assert_eq!(bit_exact(&facade, &reference), None);
    let want = limiter_f64(&case, c32_input(&case));
    let got: Vec<Complex64> = facade
        .iter()
        .map(|y| Complex64::new(f64::from(y.re), f64::from(y.im)))
        .collect();
    check_guard(&case, &want, &got);
}

#[test]
fn stereo_limit_golden() {
    limiter_case("stereo_limit_c32_16384");
}

#[test]
fn stereo_equal_golden() {
    limiter_case("stereo_equal_c32_8192");
}

/// `stereo_equal_c32_8192` holds the expected output of the mono case `limit_f32_8192` in both
/// channels, and its input is the mono input in both channels.
#[test]
fn the_equal_case_is_the_mono_case() {
    let stereo = golden::load("preemph_limit", "stereo_equal_c32_8192").unwrap();
    let mono = golden::load("preemph_limit", "limit_f32_8192").unwrap();
    for name in ["input", "expected"] {
        let Array::CF32(s) = stereo.array(name).unwrap() else {
            panic!("cf32 expected");
        };
        let Array::F32(m) = mono.array(name).unwrap() else {
            panic!("f32 expected");
        };
        assert_eq!(s.len(), m.len());
        for (x, y) in s.iter().zip(m) {
            assert_eq!(x.re.to_bits(), y.to_bits(), "{name}");
            assert_eq!(x.im.to_bits(), y.to_bits(), "{name}");
        }
    }
    for key in ["b0_bits", "b1_bits", "a1_bits"] {
        assert_eq!(stereo.param_u64(key).unwrap(), mono.param_u64(key).unwrap());
    }
    assert_eq!(
        stereo.param_f64("limit").unwrap(),
        mono.param_f64("limit").unwrap()
    );
}

// --------------------------------------------------------------------------------------------
// stereo_encode
// --------------------------------------------------------------------------------------------

fn encoder_state(case: &GoldenCase) -> StereoEncodeState {
    let mut state = StereoEncodeState::new(
        case.param_f64("pilot_hz").unwrap(),
        case.param_f64("main").unwrap() as f32,
        case.param_f64("pilot").unwrap() as f32,
        rate_of(case),
    )
    .unwrap();
    assert_eq!(
        u64::from(state.increment()),
        case.param_u64("inc").unwrap(),
        "{}: phase increment",
        case.case
    );
    state.set_phase(u32::try_from(case.param_u64("phase0").unwrap()).unwrap());
    state
}

fn encoder_check<K: StereoEncode>(kernel: &K, case: &GoldenCase) -> Vec<f32> {
    let input = c32_input(case);
    let want = case.array("expected").unwrap();
    let phase_end = case.param_u64("phase_end").unwrap();
    let mut first: Option<Vec<f32>> = None;
    for plan in plans(case) {
        let mut state = encoder_state(case);
        let mut out = vec![0.0f32; input.len()];
        for (a, b) in chunks(input.len(), plan) {
            kernel
                .stereo_encode(&mut state, &input[a..b], &mut out[a..b])
                .unwrap();
        }
        golden::assert_matches(case, want, &out).unwrap();
        assert_eq!(u64::from(state.phase()), phase_end, "{}: phase", case.case);
        match &first {
            None => first = Some(out),
            Some(whole) => assert_eq!(
                bit_exact(whole, &out),
                None,
                "{}: block size changed the output",
                case.case
            ),
        }
    }
    first.unwrap()
}

#[test]
fn stereo_encode_golden() {
    let case = golden::load("stereo_encode", "c32_16384").unwrap();
    let facade = encoder_check(&Sdrk, &case);
    let fallback = encoder_check(&Fallback, &case);
    let reference = encoder_check(&Reference, &case);
    assert_eq!(bit_exact(&facade, &fallback), None);
    assert_eq!(bit_exact(&facade, &reference), None);
    // The accuracy guard (D34): the float64 formula with exact sines of the exact phase.
    let main = f64::from(case.param_f64("main").unwrap() as f32);
    let pilot = f64::from(case.param_f64("pilot").unwrap() as f32);
    let inc = case.param_u64("inc").unwrap();
    let phase0 = case.param_u64("phase0").unwrap();
    let want: Vec<Complex64> = c32_input(&case)
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let phase = (phase0 + i as u64 * inc) % (1u64 << 32);
            let phi = 2.0 * core::f64::consts::PI * (phase as f64 / 4_294_967_296.0);
            let (l, r) = (f64::from(x.re), f64::from(x.im));
            let (m, d) = ((l + r) * 0.5, (l - r) * 0.5);
            Complex64::new(
                main * m + main * d * libm::sin(2.0 * phi) + pilot * libm::sin(phi),
                0.0,
            )
        })
        .collect();
    let got: Vec<Complex64> = facade
        .iter()
        .map(|&y| Complex64::new(f64::from(y), 0.0))
        .collect();
    check_guard(&case, &want, &got);
}

#[test]
fn the_cosine_table_is_the_one_of_the_generator() {
    let hash = Sha256::digest(bytemuck::cast_slice::<f32, u8>(&COS_F32));
    let hex = hash.iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
        s
    });
    let case = golden::load("stereo_encode", "c32_16384").unwrap();
    assert_eq!(case.param_str("cos_f32_sha256").unwrap(), hex);
}
