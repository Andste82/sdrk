//! Tests of the M25 kernel `preemph_limit` of `sdrk-core`: the construction rules (every row of
//! the validation list), timing, the limiter properties (no output above the limit for input up
//! to 20 times full scale, hold and release at exact sample positions, `set_limit` from the next
//! sample), a tone below the limit, the corners of both types (silence, full scale, NaN and
//! infinities, the largest pre-emphasis gain in `Q15`, unstable coefficients) and the
//! thread-safety of the public types. The comparison of `Fallback` with `Reference` runs only
//! with the feature `reference`.

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

use sdrk_core::kernels::preemph_limit::{PreemphLimit, PreemphLimitState};
use sdrk_core::{Delay, Error, Fallback, Q15, Rate, Ratio, RealSample, Timing};
use sdrk_device::{ErrorKind, SdrkError};

const PASS: [f64; 3] = [1.0, 0.0, 0.0];

fn rate(hz: u64) -> Rate {
    Rate::hz(hz).unwrap()
}

/// The design of `sdrk_blocks::design::preemphasis_coefs`, written out (the core crate cannot
/// depend on the blocks; the generator and the blocks tests check the real function).
fn preemph(tau_s: f64, fs: f64) -> [f64; 3] {
    let tau_p = 1.0 / (2.0 * core::f64::consts::PI * (fs / 2.4));
    let k = 2.0 * fs;
    let den = 1.0 + k * tau_p;
    [
        (1.0 + k * tau_s) / den,
        (1.0 - k * tau_s) / den,
        (1.0 - k * tau_p) / den,
    ]
}

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        v as f64 / (1u64 << 52) as f64 - 1.0
    }
}

fn from_f64<T: RealSample>(x: f64) -> T {
    T::from_f64_round(x)
}

fn run<T: RealSample, K: PreemphLimit<T>>(
    k: &K,
    s: &mut PreemphLimitState<T>,
    input: &[T],
) -> Vec<T> {
    let mut out = vec![T::ZERO; input.len()];
    k.preemph_limit(s, input, &mut out).unwrap();
    out
}

fn new_state<T: RealSample>(
    coefs: [f64; 3],
    limit: f64,
    hold_s: f64,
    release_s: f64,
    fs: u64,
) -> PreemphLimitState<T> {
    PreemphLimitState::new(coefs, limit, hold_s, release_s, rate(fs)).unwrap()
}

#[allow(clippy::needless_pass_by_value)]
fn invalid(r: sdrk_core::Result<impl Sized>) -> bool {
    matches!(r, Err(Error::InvalidParameter(_)))
}

#[test]
fn construction_rules() {
    fn all<T: RealSample>() {
        let ok = |c: [f64; 3], l: f64, h: f64, r: f64| {
            PreemphLimitState::<T>::new(c, l, h, r, rate(48_000))
        };
        assert!(ok(PASS, 1.0, 0.0, 0.2).is_ok());
        assert!(ok(preemph(530e-6, 48_000.0), 0.1, 0.02, 0.001).is_ok());
        // limit in (0, 1]
        for l in [0.0, -0.5, 1.000_000_1, f64::NAN, f64::INFINITY] {
            assert!(invalid(ok(PASS, l, 0.02, 0.2)), "limit {l}");
        }
        // release_s > 0 and finite
        for r in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(invalid(ok(PASS, 1.0, 0.02, r)), "release {r}");
        }
        // hold_s >= 0 and finite
        for h in [-0.001, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(invalid(ok(PASS, 1.0, h, 0.2)), "hold {h}");
        }
        // H <= 2^31: 2^31 samples are accepted, one more is not
        let max_hold = 2_147_483_648.0 / 48_000.0;
        assert!(ok(PASS, 1.0, max_hold, 0.2).is_ok());
        assert!(invalid(ok(PASS, 1.0, max_hold + 0.001, 0.2)));
        assert!(invalid(ok(PASS, 1.0, 1e300, 0.2)));
        // every coefficient finite with |c| < 2^14
        for (i, bad) in [16_384.0, -16_384.0, f64::NAN, f64::INFINITY, 1e300]
            .into_iter()
            .enumerate()
        {
            for pos in 0..3 {
                let mut c = PASS;
                c[pos] = bad;
                assert!(
                    invalid(ok(c, 1.0, 0.02, 0.2)),
                    "coefficient {pos} = {bad} ({i})"
                );
            }
        }
        assert!(ok([16_383.99, -16_383.99, 16_383.99], 1.0, 0.02, 0.2).is_ok());
    }
    all::<f32>();
    all::<Q15>();
    // A release that never completes: the factor per sample rounds to 1.
    assert!(invalid(PreemphLimitState::<f32>::new(
        PASS,
        1.0,
        0.0,
        10.0,
        rate(48_000_000)
    )));
    assert!(PreemphLimitState::<Q15>::new(PASS, 1.0, 0.0, 10.0, rate(48_000_000)).is_ok());
    assert!(invalid(PreemphLimitState::<Q15>::new(
        PASS,
        1.0,
        0.0,
        10.0,
        rate(1_000_000_000)
    )));
    let e = PreemphLimitState::<f32>::new(PASS, 2.0, 0.0, 0.2, rate(48_000)).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidArgument);
}

#[test]
fn timing_and_accessors() {
    let mut s = new_state::<f32>(PASS, 0.5, 0.02, 0.2, 48_000);
    assert_eq!(s.ratio(), Ratio::ONE);
    assert_eq!(s.group_delay(), Delay::ZERO);
    assert_eq!(s.limit(), 0.5);
    assert_eq!(s.hold_samples(), 960);
    assert_eq!(s.gain(), 1.0);
    // The hold time is rounded to a whole number of samples.
    assert_eq!(
        new_state::<Q15>(PASS, 1.0, 0.000_010_4, 0.2, 48_000).hold_samples(),
        0
    );
    assert_eq!(
        new_state::<Q15>(PASS, 1.0, 0.000_010_5, 0.2, 48_000).hold_samples(),
        1
    );
    // reset restores the state at rest and keeps the configuration.
    let fresh = s;
    let out = run(&Fallback, &mut s, &[3.0, -2.0]);
    assert_eq!(out[0], 0.5);
    assert!(s.gain() < 1.0);
    assert_ne!(s, fresh);
    s.reset();
    assert_eq!(s, fresh);
    s.set_limit(0.25).unwrap();
    s.reset();
    assert_eq!(s.limit(), 0.25);
}

#[test]
fn lengths_must_match() {
    let mut s = new_state::<f32>(PASS, 1.0, 0.0, 0.2, 48_000);
    let mut out = [0.0f32; 3];
    assert!(matches!(
        Fallback.preemph_limit(&mut s, &[0.0; 4], &mut out),
        Err(Error::LengthMismatch { .. })
    ));
}

#[test]
fn a_tone_below_the_limit_gets_only_the_pre_emphasis() {
    // Without pre-emphasis nothing changes, bit for bit.
    let input: Vec<f32> = (0..2000).map(|n| 0.9 * (0.07 * n as f32).sin()).collect();
    let mut s = new_state::<f32>(PASS, 1.0, 0.02, 0.2, 48_000);
    assert_eq!(run(&Fallback, &mut s, &input), input);
    let q: Vec<Q15> = input.iter().map(|&x| from_f64(f64::from(x))).collect();
    let mut s = new_state::<Q15>(PASS, 1.0, 0.02, 0.2, 48_000);
    assert_eq!(run(&Fallback, &mut s, &q), q);
    // With the pre-emphasis the output is the direct form recursion.
    let c = preemph(50e-6, 48_000.0);
    let input: Vec<f32> = (0..4000).map(|n| 0.1 * (0.13 * n as f32).sin()).collect();
    let mut s = new_state::<f32>(c, 1.0, 0.02, 0.2, 48_000);
    let got = run(&Fallback, &mut s, &input);
    let (b0, b1, a1) = (c[0] as f32, c[1] as f32, c[2] as f32);
    let (mut x1, mut v1) = (0.0f32, 0.0f32);
    for (&x, &y) in input.iter().zip(&got) {
        let v = (b0 * x + b1 * x1) - a1 * v1;
        assert_eq!(y, v);
        x1 = x;
        v1 = v;
    }
    assert_eq!(s.gain(), 1.0);
    // The same in f64 within the rounding of f32.
    let (mut x1, mut v1) = (0.0f64, 0.0f64);
    for (&x, &y) in input.iter().zip(&got) {
        let x = f64::from(x);
        let v = (c[0] * x + c[1] * x1) - c[2] * v1;
        assert!((f64::from(y) - v).abs() < 1e-5);
        x1 = x;
        v1 = v;
    }
    // Q15: within a few LSB of the f64 recursion.
    let q: Vec<Q15> = input.iter().map(|&x| from_f64(f64::from(x))).collect();
    let mut s = new_state::<Q15>(c, 1.0, 0.02, 0.2, 48_000);
    let got = run(&Fallback, &mut s, &q);
    let (mut x1, mut v1) = (0.0f64, 0.0f64);
    for (x, y) in q.iter().zip(&got) {
        let x = x.to_f64();
        let v = (c[0] * x + c[1] * x1) - c[2] * v1;
        assert!((y.to_f64() - v).abs() < 2e-4, "{} against {v}", y.to_f64());
        x1 = x;
        v1 = v;
    }
}

#[test]
fn hold_then_release_at_exact_positions_f32() {
    // 10 kHz, hold 100 samples, release 1 ms (r = exp(-0.1)), limit 0.5.
    let fs = 10_000;
    let (hold, limit) = (100usize, 0.5f32);
    let mut s = new_state::<f32>(PASS, f64::from(limit), 0.01, 0.001, fs);
    assert_eq!(s.hold_samples() as usize, hold);
    let mut input = vec![1.0f32; 400];
    input[0] = 2.0;
    let out = run(&Fallback, &mut s, &input);
    assert_eq!(out[0], 0.5); // the peak itself is limited to the limit
    let r = (-0.1f64).exp();
    // Samples 1 ..= hold: the envelope stays at 2.0, so the gain is exactly 0.25.
    for (n, &y) in out.iter().enumerate().take(hold + 1).skip(1) {
        assert_eq!(y, 0.25, "sample {n}");
    }
    // From sample hold + 1 the envelope falls by r per sample while it is above the limit.
    let mut env = 2.0f64;
    let mut env32 = 2.0f32;
    let rf = r as f32;
    #[allow(clippy::needless_range_loop)]
    for n in hold + 1..input.len() {
        env *= r;
        env32 *= rf;
        let want = if env32 > limit {
            1.0 * (limit / env32)
        } else {
            1.0
        };
        assert_eq!(out[n], want.min(limit), "sample {n}");
        // The closed form 2 r^k agrees with the iterated product within f32 rounding.
        assert!((f64::from(env32) / env - 1.0).abs() < 1e-5);
    }
    // After about 1.4 ms * ln(4) the limiter is idle again and passes 1.0 clamped to 0.5.
    assert_eq!(*out.last().unwrap(), 0.5);
    assert!(out[hold] < out[hold + 40]);
}

#[test]
fn hold_then_release_at_exact_positions_q15() {
    let (hold, fs) = (100usize, 10_000);
    let mut s = new_state::<Q15>(PASS, 0.5, 0.01, 0.001, fs);
    let mut input = vec![Q15(16_000); 600];
    input[0] = Q15(32_767);
    let out = run(&Fallback, &mut s, &input);
    assert_eq!(out[0], Q15(16_384));
    // Samples 1 ..= hold: the envelope is 32767 * 2^15, the gain 0.5 / 0.99997.
    let flat = out[1].0;
    assert!((8_000..=8_001).contains(&flat));
    for (n, y) in out.iter().enumerate().take(hold + 1).skip(1) {
        assert_eq!(y.0, flat, "sample {n}");
    }
    // Then the gain rises with r per sample until the limiter is idle and the clamp holds the
    // sample at 16000 (below the limit of 16384).
    assert!(out[hold + 1].0 > flat);
    for w in out[hold..].windows(2) {
        assert!(w[1].0 >= w[0].0);
    }
    assert_eq!(out.last().unwrap().0, 16_000);
    let gain_after_release = s.gain();
    assert_eq!(gain_after_release, 1.0);
}

#[test]
fn set_limit_applies_from_the_next_sample_and_keeps_the_envelope() {
    fn check<T: RealSample>()
    where
        Fallback: PreemphLimit<T>,
    {
        let input = [from_f64::<T>(0.8), from_f64::<T>(0.8), from_f64::<T>(0.1)];
        let mut a = new_state::<T>(PASS, 1.0, 0.02, 0.2, 48_000);
        let first = run(&Fallback, &mut a, &input[..1]);
        assert_eq!(first[0], input[0]);
        assert_eq!(a.gain(), 1.0);
        a.set_limit(0.5).unwrap();
        let rest = run(&Fallback, &mut a, &input[1..]);
        // The envelope 0.8 of the first sample is kept, the new limit acts on the next one.
        assert!((rest[0].to_f64() - 0.5).abs() < 1e-3);
        assert!((rest[1].to_f64() - 0.1 * 0.5 / 0.8).abs() < 1e-3);
        assert!((a.gain() - 0.5 / 0.8).abs() < 1e-3);
        // An invalid limit changes nothing.
        let before = a;
        assert!(invalid(a.set_limit(0.0)));
        assert!(invalid(a.set_limit(1.5)));
        assert!(invalid(a.set_limit(f64::NAN)));
        assert_eq!(a, before);
    }
    check::<f32>();
    check::<Q15>();
}

#[test]
fn the_output_never_exceeds_the_limit() {
    fn check<T: RealSample>(seed: u64)
    where
        Fallback: PreemphLimit<T>,
    {
        let mut rng = Rng(seed);
        for (tau, fs, limit, hold, release) in [
            (50e-6, 48_000u64, 1.0, 0.02, 0.2),
            (530e-6, 48_000, 0.85, 0.005, 0.01),
            (75e-6, 96_000, 0.3, 0.0, 0.001),
            (1000e-6, 240_000, 0.1, 0.001, 0.05),
            (0.0, 48_000, 0.5, 0.02, 0.2),
        ] {
            let c = if tau == 0.0 {
                PASS
            } else {
                preemph(tau, fs as f64)
            };
            let mut s = new_state::<T>(c, limit, hold, release, fs);
            // Input up to 20 times full scale in the f32 type, saturated in Q15.
            let input: Vec<T> = (0..20_000)
                .map(|n| {
                    let scale = if n % 5000 < 3000 { 20.0 } else { 0.05 };
                    from_f64(rng.next() * scale)
                })
                .collect();
            let out = run(&Fallback, &mut s, &input);
            let max = out.iter().map(|y| y.to_f64().abs()).fold(0.0, f64::max);
            let t15 = (limit * 32768.0 + 0.5).floor().min(32767.0) / 32768.0;
            let t = if T::TYPE == sdrk_core::SampleType::Q15 {
                t15
            } else {
                f64::from(limit as f32)
            };
            assert!(max <= t, "max {max} limit {t} tau {tau}");
            // And the limiter does work: the loud part reaches the limit.
            assert!(max > 0.9 * t);
        }
    }
    check::<f32>(1);
    check::<Q15>(2);
}

#[test]
fn silence_full_scale_and_non_finite_input() {
    let mut s = new_state::<f32>(preemph(50e-6, 48_000.0), 1.0, 0.02, 0.2, 48_000);
    assert!(
        run(&Fallback, &mut s, &[0.0; 100])
            .iter()
            .all(|&y| y == 0.0)
    );
    let out = run(&Fallback, &mut s, &[1.0; 100]);
    assert!(out.iter().all(|&y| y.abs() <= 1.0));
    // Non-finite input counts as 0.0: the output equals that of an input with zeros there.
    let mut rng = Rng(5);
    let clean: Vec<f32> = (0..500).map(|_| (rng.next() * 0.3) as f32).collect();
    let mut dirty = clean.clone();
    let mut zeroed = clean.clone();
    for (i, v) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -f32::NAN]
        .into_iter()
        .cycle()
        .take(40)
        .enumerate()
    {
        dirty[i * 12] = v;
        zeroed[i * 12] = 0.0;
    }
    let c = preemph(50e-6, 48_000.0);
    let mut a = new_state::<f32>(c, 0.5, 0.02, 0.2, 48_000);
    let mut b = new_state::<f32>(c, 0.5, 0.02, 0.2, 48_000);
    let out_dirty = run(&Fallback, &mut a, &dirty);
    let out_zeroed = run(&Fallback, &mut b, &zeroed);
    assert_eq!(out_dirty, out_zeroed);
    assert!(out_dirty.iter().all(|y| y.is_finite()));
    assert_eq!(a, b);
}

#[test]
fn q15_corners() {
    // The idle limiter at limit 1.0 passes everything except -32768, which the clamp to
    // +-T_q15 (T_q15 = 32767) turns into -32767.
    let mut s = new_state::<Q15>(PASS, 1.0, 0.02, 0.2, 48_000);
    let input = [
        Q15(-32_768),
        Q15(32_767),
        Q15(-32_767),
        Q15(0),
        Q15(1),
        Q15(-1),
    ];
    let out = run(&Fallback, &mut s, &input);
    assert_eq!(
        out,
        [
            Q15(-32_767),
            Q15(32_767),
            Q15(-32_767),
            Q15(0),
            Q15(1),
            Q15(-1)
        ]
    );
    // Silence stays silent, also after a loud passage has decayed far enough.
    let mut s = new_state::<Q15>(preemph(530e-6, 48_000.0), 0.5, 0.0, 0.001, 48_000);
    run(&Fallback, &mut s, &[Q15(-32_768); 200]);
    let out = run(&Fallback, &mut s, &vec![Q15(0); 5000]);
    assert!(
        out[4000..].iter().all(|y| y.0 == 0),
        "{:?}",
        &out[4000..4010]
    );
    // A tiny limit gives a zero clamp and a zero T_q30 without a panic.
    let mut s = new_state::<Q15>(PASS, 1e-12, 0.0, 0.2, 48_000);
    assert!(
        run(&Fallback, &mut s, &[Q15(32_767); 10])
            .iter()
            .all(|y| y.0 == 0)
    );
}

#[test]
fn the_largest_gain_in_q15_does_not_overflow() {
    // 1000 us at 240 kHz: b0 is about 272 and the value in Q30 reaches 2^39.
    let c = preemph(1000e-6, 240_000.0);
    assert!(c[0] > 270.0 && c[0] < 275.0, "{c:?}");
    for limit in [1.0, 0.5, 0.1] {
        let mut s = new_state::<Q15>(c, limit, 0.001, 0.05, 240_000);
        // Full-scale square wave with a period of 48 samples (5 kHz), then the Nyquist rate.
        let mut input: Vec<Q15> = (0..4800)
            .map(|n| Q15(if (n / 24) % 2 == 0 { 32_767 } else { -32_768 }))
            .collect();
        input.extend((0..4800).map(|n| Q15(if n % 2 == 0 { 32_767 } else { -32_768 })));
        let out = run(&Fallback, &mut s, &input);
        let t15 = (limit * 32768.0 + 0.5f64).floor().min(32767.0);
        assert!(out.iter().all(|y| f64::from(y.0).abs() <= t15));
        assert!(s.gain() < 0.05, "gain {}", s.gain());
    }
}

#[test]
fn unstable_and_extreme_coefficients_do_not_panic() {
    let big = [16_383.0, -16_383.0, 16_383.0];
    for c in [big, [16_383.0, 16_383.0, -16_383.0], [1.0, 1.0, 1.0001]] {
        let mut s = new_state::<Q15>(c, 1.0, 0.0, 0.2, 48_000);
        let input: Vec<Q15> = (0..3000)
            .map(|n| Q15(if n % 3 == 0 { 32_767 } else { -32_768 }))
            .collect();
        let out = run(&Fallback, &mut s, &input);
        assert!(out.iter().all(|y| y.0 >= -32_767));
        let mut s = new_state::<f32>(c, 1.0, 0.0, 0.2, 48_000);
        let input: Vec<f32> = (0..3000)
            .map(|n| if n % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        // The f32 state may become NaN for an unstable filter (documented); it must not panic.
        let _ = run(&Fallback, &mut s, &input);
    }
}

#[test]
fn the_public_types_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<PreemphLimitState<f32>>();
    assert_send::<PreemphLimitState<Q15>>();
}

#[test]
fn the_state_is_small() {
    assert!(core::mem::size_of::<PreemphLimitState<f32>>() <= 256);
    assert!(core::mem::size_of::<PreemphLimitState<Q15>>() <= 256);
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    fn compare<T: RealSample>(seed: u64)
    where
        Reference: PreemphLimit<T>,
        Fallback: PreemphLimit<T>,
    {
        let mut rng = Rng(seed);
        for round in 0..300 {
            let fs = [40_000u64, 44_100, 48_000, 96_000, 240_000][round % 5];
            let tau = [0.0, 50e-6, 75e-6, 530e-6, 1000e-6][(round / 5) % 5];
            let c = if tau == 0.0 {
                PASS
            } else {
                preemph(tau, fs as f64)
            };
            let limit = [1.0, 0.85, 0.5, 0.1][round % 4];
            let hold = [0.0, 0.0001, 0.02][round % 3];
            let release = [0.0005, 0.01, 0.2][(round / 3) % 3];
            let n = 1 + (rng.next().abs() * 3000.0) as usize;
            let scale = [0.05, 1.0, 20.0][(round / 7) % 3];
            let input: Vec<T> = (0..n)
                .map(|i| {
                    let tone = 0.5 * (0.31 * i as f64).sin();
                    from_f64((rng.next() * 0.5 + tone) * scale)
                })
                .collect();
            let mut a = new_state::<T>(c, limit, hold, release, fs);
            let mut b = a;
            let out_a = run(&Fallback, &mut a, &input);
            let out_b = run(&Reference, &mut b, &input);
            assert_eq!(out_a, out_b, "round {round}");
            assert_eq!(a, b, "round {round}: state");
            // Split into two calls with a limit change in between.
            let cut = n / 2;
            let mut a = new_state::<T>(c, limit, hold, release, fs);
            let mut b = a;
            let mut ra = run(&Fallback, &mut a, &input[..cut]);
            let mut rb = run(&Reference, &mut b, &input[..cut]);
            a.set_limit(limit * 0.5).unwrap();
            b.set_limit(limit * 0.5).unwrap();
            ra.extend(run(&Fallback, &mut a, &input[cut..]));
            rb.extend(run(&Reference, &mut b, &input[cut..]));
            assert_eq!(ra, rb, "round {round}: split");
            assert_eq!(a, b);
        }
    }

    #[test]
    fn fallback_equals_reference_f32() {
        compare::<f32>(11);
    }

    #[test]
    fn fallback_equals_reference_q15() {
        compare::<Q15>(12);
    }

    #[test]
    fn fallback_equals_reference_at_the_corners() {
        let specials = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
            0.0,
            -0.0,
        ];
        let input: Vec<f32> = (0..700)
            .map(|i| specials[i % specials.len()] * (1.0 / (1 + i % 3) as f32))
            .collect();
        let mut a = new_state::<f32>(PASS, 0.5, 0.0, 0.2, 48_000);
        let mut b = a;
        let (oa, ob) = (
            run(&Fallback, &mut a, &input),
            run(&Reference, &mut b, &input),
        );
        assert!(oa.iter().zip(&ob).all(|(x, y)| x.to_bits() == y.to_bits()));
        let q = [Q15(-32_768), Q15(32_767), Q15(0), Q15(-1), Q15(1)];
        let input: Vec<Q15> = (0..700).map(|i| q[(i * 7) % 5]).collect();
        for coefs in [
            PASS,
            preemph(1000e-6, 240_000.0),
            [16_383.0, -16_383.0, 16_383.0],
        ] {
            let mut a = new_state::<Q15>(coefs, 0.7, 0.0001, 0.001, 240_000);
            let mut b = a;
            assert_eq!(
                run(&Fallback, &mut a, &input),
                run(&Reference, &mut b, &input)
            );
            assert_eq!(a, b);
        }
    }
}
