//! Golden tests of the FM stereo kernels (M24) through the facade `Sdrk`, `Fallback` and
//! `Reference`.
//!
//! Every case is run in one piece and in chunks of random size (the plan seed is the case seed);
//! both runs must match the golden file and each other bit for bit. `fade_f32_16384` also checks
//! `is_stereo()` after every chunk against the stored flags, `disable_f32_8192` calls
//! `set_enabled(false)` before sample 4096 in both modes. The cosine tables of `sdrk-core` must
//! match the SHA-256 stored in the case, and the lock detector of the model under test must stay
//! within the regression guard against the float64 loop (decision D111).
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use sdrk::Sdrk;
use sdrk_core::kernels::stereo::{StereoMatrix, StereoPll, StereoPllState};
use sdrk_core::tables::{COS_F32, COS_Q15};
use sdrk_core::{C32, CQ15, Fallback, Q15, Rate, RealSample, Reference};
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

fn state_of<T: RealSample>(case: &GoldenCase) -> StereoPllState<T> {
    let rate = Rate::hz(case.param_u64("rate_hz").unwrap()).unwrap();
    StereoPllState::new(
        case.param_f64("pilot_hz").unwrap(),
        case.param_f64("natural_hz").unwrap(),
        case.param_f64("pilot_level").unwrap(),
        rate,
    )
    .unwrap()
}

/// Everything a run needs: where the chunks end, whether to check the flags, when to disable.
struct Run<'a> {
    plan: Option<ChunkPlan>,
    changes: &'a [(usize, bool)],
    flags: Option<&'a [u8]>,
}

fn run<T: RealSample, K: StereoPll<T>>(
    kernel: &K,
    mut state: StereoPllState<T>,
    mpx: &[T],
    pilot: &[T],
    cfg: &Run<'_>,
) -> Vec<T::Complex> {
    let mut out = vec![<T::Complex as sdrk_core::Sample>::ZERO; mpx.len()];
    let mut cuts = vec![0];
    cuts.extend(cfg.changes.iter().map(|c| c.0));
    cuts.push(mpx.len());
    let mut sizes = cfg.plan.map(|p| p.sizes());
    for seg in cuts.windows(2) {
        let (mut pos, end) = (seg[0], seg[1]);
        if let Some(&(_, enabled)) = cfg.changes.iter().find(|c| c.0 == pos) {
            state.set_enabled(enabled);
        }
        while pos < end {
            let n = sizes
                .as_mut()
                .map_or(end - pos, |s| s.next().unwrap().min(end - pos));
            kernel
                .stereo_pll(
                    &mut state,
                    &mpx[pos..pos + n],
                    &pilot[pos..pos + n],
                    &mut out[pos..pos + n],
                )
                .unwrap();
            pos += n;
            if let Some(flags) = cfg.flags {
                assert_eq!(
                    state.is_stereo(),
                    flags[pos - 1] == 1,
                    "is_stereo() after sample {}",
                    pos - 1
                );
            }
        }
    }
    out
}

fn f32_inputs(case: &GoldenCase) -> (Vec<f32>, Vec<f32>) {
    let (Array::F32(m), Array::F32(p)) = (case.array("mpx").unwrap(), case.array("pilot").unwrap())
    else {
        panic!("f32 inputs expected");
    };
    (m.clone(), p.clone())
}

fn q15_inputs(case: &GoldenCase) -> (Vec<Q15>, Vec<Q15>) {
    (
        case.array("mpx").unwrap().as_q15().unwrap(),
        case.array("pilot").unwrap().as_q15().unwrap(),
    )
}

fn check_all<T: RealSample, K: StereoPll<T>>(
    kernel: &K,
    case: &GoldenCase,
    mpx: &[T],
    pilot: &[T],
) -> Vec<T::Complex> {
    let want = case.array("expected").unwrap();
    let flags = match case.array("stereo") {
        Ok(Array::U8(f)) => Some(f.as_slice()),
        _ => None,
    };
    let mut changes = Vec::new();
    for (key, enabled) in [("disable_at", false), ("enable_at", true)] {
        if let Ok(at) = case.param_u64(key) {
            changes.push((usize::try_from(at).unwrap(), enabled));
        }
    }
    let whole = run(
        kernel,
        state_of::<T>(case),
        mpx,
        pilot,
        &Run {
            plan: None,
            changes: &changes,
            flags: None,
        },
    );
    let plan = ChunkPlan::new(case.seed, 700);
    let chunked = run(
        kernel,
        state_of::<T>(case),
        mpx,
        pilot,
        &Run {
            plan: Some(plan),
            changes: &changes,
            flags,
        },
    );
    golden::assert_matches(case, want, &whole).unwrap();
    assert_eq!(
        bit_exact(&whole, &chunked),
        None,
        "{}: block size changed the output",
        case.case
    );
    whole
}

#[test]
fn all_cases_are_present() {
    assert_eq!(
        golden::list("stereo_pll").unwrap(),
        [
            "disable_enable_f32_16384",
            "disable_f32_8192",
            "fade_f32_16384",
            "lock_f32_16384",
            "lock_q15_16384"
        ]
    );
    assert_eq!(
        golden::list("stereo_matrix").unwrap(),
        ["c32_4096", "cq15_4096"]
    );
}

#[test]
fn the_cosine_tables_are_the_ones_of_the_generator() {
    let f32_hash = hex(&Sha256::digest(bytemuck::cast_slice::<f32, u8>(&COS_F32)));
    let q15_hash = hex(&Sha256::digest(bytemuck::cast_slice::<i16, u8>(&COS_Q15)));
    for name in golden::list("stereo_pll").unwrap() {
        let case = golden::load("stereo_pll", &name).unwrap();
        assert_eq!(
            case.param_str("cos_f32_sha256").unwrap(),
            f32_hash,
            "{name}: COS_F32 of sdrk-core differs from the table of the golden generator"
        );
        assert_eq!(
            case.param_str("cos_q15_sha256").unwrap(),
            q15_hash,
            "{name}: COS_Q15 of sdrk-core differs from the table of the golden generator"
        );
    }
}

#[test]
fn the_coefficients_equal_the_ones_of_the_generator() {
    for name in golden::list("stereo_pll").unwrap() {
        let case = golden::load("stereo_pll", &name).unwrap();
        let c = if name.contains("q15") {
            state_of::<Q15>(&case).coefficients()
        } else {
            state_of::<f32>(&case).coefficients()
        };
        for (what, got, key) in [
            ("kp", c.kp, "kp_bits"),
            ("ki", c.ki, "ki_bits"),
            ("imax", c.imax, "imax_bits"),
            ("beta", c.beta, "beta_bits"),
            ("on", c.on, "on_bits"),
            ("off", c.off, "off_bits"),
        ] {
            assert_eq!(
                got.to_bits(),
                case.param_u64(key).unwrap(),
                "{name}: coefficient {what} differs in the last bits (libm exp or sqrt?)"
            );
        }
        assert_eq!(u64::from(c.step0), case.param_u64("step0").unwrap());
    }
}

#[test]
fn stereo_pll_f32_golden() {
    for name in [
        "lock_f32_16384",
        "fade_f32_16384",
        "disable_f32_8192",
        "disable_enable_f32_16384",
    ] {
        let case = golden::load("stereo_pll", name).unwrap();
        let (mpx, pilot) = f32_inputs(&case);
        let facade = check_all::<f32, _>(&Sdrk, &case, &mpx, &pilot);
        let fallback = check_all::<f32, _>(&Fallback, &case, &mpx, &pilot);
        let reference = check_all::<f32, _>(&Reference, &case, &mpx, &pilot);
        assert_eq!(bit_exact(&facade, &fallback), None);
        assert_eq!(bit_exact(&facade, &reference), None);
    }
}

#[test]
fn stereo_pll_q15_golden() {
    let case = golden::load("stereo_pll", "lock_q15_16384").unwrap();
    let (mpx, pilot) = q15_inputs(&case);
    let facade = check_all::<Q15, _>(&Sdrk, &case, &mpx, &pilot);
    let fallback = check_all::<Q15, _>(&Fallback, &case, &mpx, &pilot);
    let reference = check_all::<Q15, _>(&Reference, &case, &mpx, &pilot);
    assert_eq!(bit_exact(&facade, &fallback), None);
    assert_eq!(bit_exact(&facade, &reference), None);
}

#[test]
fn the_lock_follows_the_documented_positions() {
    let case = golden::load("stereo_pll", "fade_f32_16384").unwrap();
    let Array::U8(flags) = case.array("stereo").unwrap() else {
        panic!("u8 flags expected");
    };
    // Off at the start, on after the 50 ms detector has passed 0.4 kd (about 5000 samples),
    // and still on at the end: the pilot has vanished only 4096 samples before (see D112).
    assert!(flags[..4000].iter().all(|&f| f == 0));
    assert!(flags[6000..12_000].iter().all(|&f| f == 1));
    let first_on = flags.iter().position(|&f| f == 1).unwrap();
    assert!((4500..5500).contains(&first_on), "locked at {first_on}");
    // The difference channel is zero exactly while the flag is off.
    let Array::CF32(out) = case.array("expected").unwrap() else {
        panic!("cf32 expected");
    };
    assert!(
        out[..first_on]
            .iter()
            .all(|s| s.im.to_bits() == 0.0f32.to_bits())
    );
}

#[test]
fn disabling_zeroes_the_difference_channel_from_the_listed_sample() {
    // The case of the milestone disables before the lock (about sample 5000): the difference
    // channel is zero throughout.
    let case = golden::load("stereo_pll", "disable_f32_8192").unwrap();
    let at = usize::try_from(case.param_u64("disable_at").unwrap()).unwrap();
    let Array::CF32(out) = case.array("expected").unwrap() else {
        panic!("cf32 expected");
    };
    assert!(out[at..].iter().all(|s| s.im.to_bits() == 0.0f32.to_bits()));
    // The additional case disables and enables a locked loop.
    let case = golden::load("stereo_pll", "disable_enable_f32_16384").unwrap();
    let off = usize::try_from(case.param_u64("disable_at").unwrap()).unwrap();
    let on = usize::try_from(case.param_u64("enable_at").unwrap()).unwrap();
    let Array::CF32(out) = case.array("expected").unwrap() else {
        panic!("cf32 expected");
    };
    assert!(out[off - 200..off].iter().any(|s| s.im != 0.0));
    assert!(
        out[off..on]
            .iter()
            .all(|s| s.im.to_bits() == 0.0f32.to_bits())
    );
    assert!(out[on..on + 200].iter().all(|s| s.im != 0.0));
}

// --------------------------------------------------------------------------------------------
// The float64 regression guard (D111)
// --------------------------------------------------------------------------------------------

/// The loop of section 3 of the milestone in f64 with exact sine and cosine (same equations as
/// the generator; no tables, no flush); returns the difference channel.
fn loop_f64(case: &GoldenCase, mpx: &[f64], pilot: &[f64]) -> Vec<f64> {
    let c = state_of::<f32>(case).coefficients();
    let (mut phase, mut integ, mut level, mut stereo) = (0u32, 0.0f64, 0.0f64, false);
    let mut out = Vec::with_capacity(mpx.len());
    for (&x, &p) in mpx.iter().zip(pilot) {
        let theta = 2.0 * core::f64::consts::PI * (f64::from(phase) / 4_294_967_296.0);
        let s2 = libm::sin(2.0 * theta);
        let e = if p.is_finite() {
            let q = p * libm::sin(theta);
            level += c.beta * (q - level);
            if level >= c.on {
                stereo = true;
            } else if level < c.off {
                stereo = false;
            }
            p * libm::cos(theta)
        } else {
            0.0
        };
        out.push(if stereo { x * s2 } else { 0.0 });
        integ = (integ + c.ki * e).clamp(-c.imax, c.imax);
        let f = c.kp * e + integ;
        let corr = libm::floor(f * 4_294_967_296.0 + 0.5) as i64 as u32;
        phase = phase.wrapping_add(c.step0).wrapping_add(corr);
    }
    out
}

fn snr_db(reference: &[f64], test: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (r, t) in reference.iter().zip(test) {
        num += r * r;
        den += (r - t) * (r - t);
    }
    10.0 * libm::log10(num / den)
}

#[test]
fn the_difference_channel_stays_within_the_f64_guard() {
    for name in ["lock_f32_16384", "fade_f32_16384", "lock_q15_16384"] {
        let case = golden::load("stereo_pll", name).unwrap();
        let skip = usize::try_from(case.param_u64("guard_skip").unwrap()).unwrap();
        let guard = case.param_f64("snr_db_guard").unwrap();
        let (got, want) = if name.contains("q15") {
            let (mpx, pilot) = q15_inputs(&case);
            let mut out = vec![CQ15::default(); mpx.len()];
            Sdrk.stereo_pll(&mut state_of::<Q15>(&case), &mpx, &pilot, &mut out)
                .unwrap();
            let to_f = |v: &[Q15]| v.iter().map(|q| q.to_f64()).collect::<Vec<_>>();
            (
                out.iter().map(|s| s.im.to_f64()).collect::<Vec<_>>(),
                loop_f64(&case, &to_f(&mpx), &to_f(&pilot)),
            )
        } else {
            let (mpx, pilot) = f32_inputs(&case);
            let mut out = vec![C32::default(); mpx.len()];
            Sdrk.stereo_pll(&mut state_of::<f32>(&case), &mpx, &pilot, &mut out)
                .unwrap();
            let to_f = |v: &[f32]| v.iter().map(|&q| f64::from(q)).collect::<Vec<_>>();
            (
                out.iter().map(|s| f64::from(s.im)).collect::<Vec<_>>(),
                loop_f64(&case, &to_f(&mpx), &to_f(&pilot)),
            )
        };
        let snr = snr_db(&want[skip..], &got[skip..]);
        assert!(
            snr >= guard,
            "{name}: difference channel SNR {snr:.2} dB against the f64 loop is below the guard {guard} dB"
        );
        // The guard is the measured value minus 1 dB, so it is meaningful, not vacuous.
        assert!(snr < guard + 3.0, "{name}: SNR {snr:.2}, guard {guard}");
    }
}

// --------------------------------------------------------------------------------------------
// stereo_matrix
// --------------------------------------------------------------------------------------------

fn matrix_run<T: sdrk_core::ComplexSample, K: StereoMatrix<T>>(
    kernel: &K,
    input: &[T],
    plan: Option<ChunkPlan>,
) -> Vec<T> {
    let mut out = vec![T::ZERO; input.len()];
    let mut sizes = plan.map(|p| p.sizes());
    let mut pos = 0;
    while pos < input.len() {
        let n = sizes
            .as_mut()
            .map_or(input.len() - pos, |s| s.next().unwrap())
            .min(input.len() - pos);
        kernel
            .stereo_matrix(&input[pos..pos + n], &mut out[pos..pos + n])
            .unwrap();
        pos += n;
    }
    out
}

fn matrix_check<T: sdrk_core::ComplexSample, K: StereoMatrix<T>>(
    kernel: &K,
    case: &GoldenCase,
    input: &[T],
) {
    let want = case.array("expected").unwrap();
    let whole = matrix_run(kernel, input, None);
    let chunked = matrix_run(kernel, input, Some(ChunkPlan::new(case.seed, 300)));
    golden::assert_matches(case, want, &whole).unwrap();
    golden::assert_matches(case, want, &chunked).unwrap();
}

#[test]
fn stereo_matrix_golden() {
    let case = golden::load("stereo_matrix", "c32_4096").unwrap();
    let Array::CF32(x) = case.array("input").unwrap() else {
        panic!("cf32 expected");
    };
    matrix_check::<C32, _>(&Sdrk, &case, x);
    matrix_check::<C32, _>(&Fallback, &case, x);
    matrix_check::<C32, _>(&Reference, &case, x);
    let case = golden::load("stereo_matrix", "cq15_4096").unwrap();
    let Array::CI16(x) = case.array("input").unwrap() else {
        panic!("ci16 expected");
    };
    matrix_check::<CQ15, _>(&Sdrk, &case, x);
    matrix_check::<CQ15, _>(&Fallback, &case, x);
    matrix_check::<CQ15, _>(&Reference, &case, x);
}
