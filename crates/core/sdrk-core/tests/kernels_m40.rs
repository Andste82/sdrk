//! Tests of the M40 kernel of `sdrk-core`: `costas`. The construction rules, the loop constants,
//! timing, the length check, lock and frequency tracking on clean BPSK and QPSK, the lock
//! detector on and off, block-size invariance (pieces of 1 to N symbols), `reset`, silence, full
//! scale and beyond, non-finite input and the thread-safety of the public types. The comparison
//! of `Fallback` with `Reference` runs only with the feature `reference`.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]

use sdrk_core::kernels::costas::{Costas, CostasGains, CostasState};
use sdrk_core::{C32, Delay, Error, Fallback, Ratio, RatioKind, Timing};
use sdrk_device::{ErrorKind, SdrkError};

#[allow(
    clippy::approx_constant,
    reason = "the damping of the block defaults, written as in the spec"
)]
const DAMP: f64 = 0.7071;
/// The seed of the randomized tests (M40 seeds 550 to 559); every signal mixes in a stream.
const SEED: u64 = 559;

/// Deterministic generator (xorshift64*), uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (v as f64 / (1u64 << 52) as f64 - 1.0) as f32
    }

    fn bit(&mut self) -> bool {
        self.next() >= 0.0
    }
}

/// Symbols of `order` (BPSK `+-1`, QPSK `(+-1 +-j) / sqrt 2`) rotated by
/// `phase0 + 2 pi offset k` (radians, cycles per symbol), plus uniform noise of `noise`.
fn symbols(seed: u64, n: usize, order: u32, phase0: f64, offset: f64, noise: f32) -> Vec<C32> {
    let mut rng = Rng(SEED * 1000 + seed);
    let r = core::f32::consts::FRAC_1_SQRT_2;
    (0..n)
        .map(|k| {
            let s = if order == 2 {
                C32::new(if rng.bit() { 1.0 } else { -1.0 }, 0.0)
            } else {
                C32::new(
                    if rng.bit() { r } else { -r },
                    if rng.bit() { r } else { -r },
                )
            };
            let turns = (phase0 / core::f64::consts::TAU + offset * k as f64).rem_euclid(1.0);
            let (sn, c) = (core::f64::consts::TAU * turns).sin_cos();
            let (sn, c) = (sn as f32, c as f32);
            C32::new(
                s.re * c - s.im * sn + noise * rng.next(),
                s.re * sn + s.im * c + noise * rng.next(),
            )
        })
        .collect()
}

fn run(st: &mut CostasState, x: &[C32]) -> Vec<C32> {
    let mut y = vec![C32::default(); x.len()];
    Fallback.costas(st, x, &mut y).unwrap();
    y
}

fn bits(x: &[C32]) -> Vec<(u32, u32)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

#[test]
fn construction_rules() {
    assert!(CostasState::new(2, 0.02, DAMP, 0.05).is_ok());
    assert!(CostasState::new(4, 0.02, DAMP, 0.05).is_ok());
    // Order 8 is M40.1: unsupported, not invalid.
    let e = CostasState::new(8, 0.02, DAMP, 0.05).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Unsupported);
    for order in [0, 1, 3, 5, 16] {
        assert!(
            matches!(
                CostasState::new(order, 0.02, DAMP, 0.05),
                Err(Error::InvalidParameter(_))
            ),
            "{order}"
        );
    }
    for bw in [0.0, 0.000_09, 0.200_001, f64::NAN, f64::INFINITY] {
        assert!(CostasState::new(4, bw, DAMP, 0.05).is_err(), "{bw}");
    }
    for d in [0.09, 4.01, f64::NAN] {
        assert!(CostasState::new(4, 0.02, d, 0.05).is_err(), "{d}");
    }
    for m in [-0.001, 0.2501, f64::NAN] {
        assert!(CostasState::new(4, 0.02, DAMP, m).is_err(), "{m}");
    }
    // The bounds themselves are allowed.
    for (bw, d, m) in [(0.0001, 0.1, 0.0), (0.2, 4.0, 0.25)] {
        let st = CostasState::new(2, bw, d, m).unwrap();
        assert_eq!((st.loop_bw(), st.damping(), st.max_offset()), (bw, d, m));
    }
}

#[test]
fn loop_constants_follow_the_formula() {
    for (bw, d) in [(0.02, DAMP), (0.0001, 0.1), (0.2, 4.0), (0.05, 1.0)] {
        let st = CostasState::new(4, bw, d, 0.05).unwrap();
        let theta = bw / (d + 1.0 / (4.0 * d));
        let den = (1.0 + (2.0 * d) * theta) + theta * theta;
        let kp = (((4.0 * d) * theta) / den) / core::f64::consts::TAU;
        let ki = ((4.0 * (theta * theta)) / den) / core::f64::consts::TAU;
        assert_eq!(
            st.gains(),
            CostasGains {
                kp: kp as f32,
                ki: ki as f32,
                max_offset: 0.05
            }
        );
    }
}

#[test]
fn timing_is_one_to_one_without_delay() {
    let st = CostasState::new(2, 0.02, DAMP, 0.05).unwrap();
    assert_eq!(st.ratio(), Ratio::ONE);
    assert_eq!(st.ratio_kind(), RatioKind::Constant);
    assert_eq!(st.group_delay(), Delay::ZERO);
}

#[test]
fn unequal_lengths_change_nothing() {
    let mut st = CostasState::new(4, 0.02, DAMP, 0.05).unwrap();
    let before = st;
    let mut y = [C32::default(); 3];
    let e = Fallback
        .costas(&mut st, &[C32::new(0.6, 0.8); 4], &mut y)
        .unwrap_err();
    assert!(matches!(
        e,
        Error::LengthMismatch {
            expected: 4,
            got: 3
        }
    ));
    assert_eq!(st, before);
}

#[test]
fn bpsk_locks_onto_the_real_axis() {
    for (k, phase0) in [0.3, 1.2, -0.7, 2.9].into_iter().enumerate() {
        let x = symbols(10 + k as u64, 3000, 2, phase0, 0.0, 0.0);
        let mut st = CostasState::new(2, 0.02, DAMP, 0.05).unwrap();
        let y = run(&mut st, &x);
        assert!(st.locked(), "{phase0}");
        for v in &y[2000..] {
            assert!(
                (v.re.abs() - 1.0).abs() < 1e-3 && v.im.abs() < 1e-3,
                "{phase0}: {v}"
            );
        }
        assert!(st.lock_level() > 0.99);
    }
}

#[test]
fn qpsk_locks_onto_the_diagonals_and_follows_an_offset() {
    let r = core::f32::consts::FRAC_1_SQRT_2;
    for (k, offset) in [0.0, 0.004, -0.01, 0.02].into_iter().enumerate() {
        let x = symbols(20 + k as u64, 4000, 4, 0.5 + k as f64, offset, 0.0);
        let mut st = CostasState::new(4, 0.02, DAMP, 0.05).unwrap();
        let y = run(&mut st, &x);
        assert!(st.locked(), "{offset}");
        assert!(
            (st.frequency() - offset).abs() < 1e-4,
            "{offset}: {}",
            st.frequency()
        );
        for v in &y[3000..] {
            assert!(
                (v.re.abs() - r).abs() < 2e-3 && (v.im.abs() - r).abs() < 2e-3,
                "{offset}: {v}"
            );
        }
    }
}

#[test]
fn the_frequency_integrator_stops_at_its_limit() {
    // An offset of 0.04 cycles per symbol against a limit of 0.01.
    let x = symbols(30, 3000, 2, 0.0, 0.04, 0.0);
    let mut st = CostasState::new(2, 0.02, DAMP, 0.01).unwrap();
    run(&mut st, &x);
    assert!(st.frequency().abs() <= f64::from(0.01f32));
    // max_offset 0: the integrator never moves.
    let mut st = CostasState::new(2, 0.02, DAMP, 0.0).unwrap();
    run(&mut st, &x);
    assert_eq!(st.frequency(), 0.0);
}

#[test]
fn a_tiny_max_offset_is_flushed_and_the_integrator_never_holds_a_subnormal() {
    // R6.3: a limit below 1e-30 (subnormal as f32, or normal but below the flush level) is 0.
    let x = symbols(31, 500, 2, 0.0, 0.04, 0.0);
    for m in [1e-40, 1e-38, 1e-31, f64::MIN_POSITIVE] {
        let mut st = CostasState::new(2, 0.02, DAMP, m).unwrap();
        assert_eq!(st.gains().max_offset.to_bits(), 0, "{m}");
        assert_eq!(st.max_offset(), m);
        run(&mut st, &x);
        assert_eq!(st.frequency(), 0.0, "{m}");
    }
    // A normal limit at the flush level is kept and the integrator ends on it.
    let mut st = CostasState::new(2, 0.02, DAMP, 2e-30).unwrap();
    assert_eq!(st.gains().max_offset, 2e-30f32);
    run(&mut st, &x);
    assert!(
        st.frequency().abs() == f64::from(2e-30f32),
        "{}",
        st.frequency()
    );
    assert!((st.frequency() as f32).is_normal());
}

#[test]
fn the_lock_detector_turns_off_on_noise() {
    let mut st = CostasState::new(4, 0.02, DAMP, 0.05).unwrap();
    run(&mut st, &symbols(40, 2000, 4, 0.4, 0.0, 0.05));
    assert!(st.locked());
    // Noise alone: the metric averages to about 0, below the off level within a few hundred
    // symbols.
    let mut rng = Rng(SEED * 1000 + 41);
    let noise: Vec<C32> = (0..1500)
        .map(|_| C32::new(rng.next(), rng.next()))
        .collect();
    run(&mut st, &noise);
    assert!(!st.locked(), "level {}", st.lock_level());
    assert!(st.lock_level().abs() < 0.3);
    // A fresh loop on noise never locks.
    let mut st = CostasState::new(2, 0.02, DAMP, 0.05).unwrap();
    let mut y = [C32::default(); 1];
    for x in &noise {
        Fallback
            .costas(&mut st, core::slice::from_ref(x), &mut y)
            .unwrap();
        assert!(!st.locked());
    }
}

#[test]
fn output_does_not_depend_on_the_block_size() {
    for order in [2, 4] {
        let x = symbols(50 + u64::from(order), 1500, order, 2.0, 0.006, 0.2);
        let mut whole = CostasState::new(order, 0.03, DAMP, 0.05).unwrap();
        let want = run(&mut whole, &x);
        for size in [1usize, 2, 3, 7, 64, 499, 1500] {
            let mut st = CostasState::new(order, 0.03, DAMP, 0.05).unwrap();
            let mut y = vec![C32::default(); x.len()];
            for (xi, yi) in x.chunks(size).zip(y.chunks_mut(size)) {
                Fallback.costas(&mut st, xi, yi).unwrap();
            }
            assert_eq!(bits(&y), bits(&want), "order {order}, size {size}");
            assert_eq!(st, whole);
        }
        // Empty calls change nothing.
        let mut st = whole;
        Fallback.costas(&mut st, &[], &mut []).unwrap();
        assert_eq!(st, whole);
    }
}

#[test]
fn reset_returns_to_the_new_state() {
    let mut st = CostasState::new(4, 0.02, DAMP, 0.05).unwrap();
    let x = symbols(60, 1000, 4, 1.0, 0.01, 0.05);
    let first = run(&mut st, &x);
    st.reset();
    assert_eq!(st, CostasState::new(4, 0.02, DAMP, 0.05).unwrap());
    assert_eq!(bits(&run(&mut st, &x)), bits(&first));
    st.set_phase(123);
    assert_eq!(st.phase(), 123);
}

#[test]
fn silence_gives_zeros_and_a_resting_loop() {
    for order in [2, 4] {
        let mut st = CostasState::new(order, 0.02, DAMP, 0.05).unwrap();
        st.set_phase(0x1234_5678);
        let y = run(&mut st, &[C32::new(0.0, 0.0); 500]);
        assert!(y.iter().all(|v| v.re == 0.0 && v.im == 0.0));
        // e = 0 and the metric 0 / 0 is skipped: nothing moves.
        assert_eq!(st.phase(), 0x1234_5678);
        assert_eq!(
            (st.frequency(), st.lock_level(), st.locked()),
            (0.0, 0.0, false)
        );
    }
}

#[test]
fn full_scale_and_beyond_keep_the_loop_bounded() {
    for order in [2, 4] {
        for amp in [1.0f32, 1e18, 1e30, f32::MAX] {
            let x: Vec<C32> = symbols(70, 600, order, 0.8, 0.003, 0.0)
                .iter()
                .map(|v| C32::new(v.re * amp, v.im * amp))
                .collect();
            let mut st = CostasState::new(order, 0.02, DAMP, 0.05).unwrap();
            run(&mut st, &x);
            assert!(st.frequency().abs() <= 0.05 + 1e-9, "{order} {amp}");
            assert!(st.lock_level().is_finite() && st.lock_level().abs() <= 1.0 + 1e-6);
        }
    }
}

#[test]
fn non_finite_input_stays_in_its_own_symbols() {
    for order in [2, 4] {
        let mut x = symbols(80, 2000, order, 0.4, 0.0, 0.02);
        for (i, v) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            x[1000 + i] = C32::new(v, 0.5);
            x[1010 + i] = C32::new(0.5, v);
        }
        let mut st = CostasState::new(order, 0.02, DAMP, 0.05).unwrap();
        let y = run(&mut st, &x);
        // NaN and infinity reach the output of their own symbol only.
        assert!(
            y[..1000]
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
        );
        assert!(!y[1000].re.is_finite() && !y[1010].im.is_finite());
        // The bad symbols give no error to the loop: the lock holds and the output is finite
        // again right after them.
        assert!(
            y[1013..]
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
        );
        assert!(st.locked() && st.frequency().is_finite());
    }
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<CostasState>();
    assert_send::<CostasGains>();
}

#[cfg(feature = "reference")]
mod against_the_reference {
    use super::*;
    use sdrk_core::Reference;

    const SPECIALS: [f32; 8] = [
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE / 2.0,
        f32::MAX,
        1.0,
    ];

    #[test]
    fn fallback_equals_reference() {
        for order in [2, 4] {
            for (k, offset) in [0.0, 0.01, -0.03].into_iter().enumerate() {
                let mut x = symbols(90 + k as u64, 3000, order, 1.0, offset, 0.3);
                for (j, s) in SPECIALS.iter().enumerate() {
                    x[500 + 7 * j] = C32::new(*s, 0.3);
                    x[900 + 7 * j] = C32::new(-0.3, *s);
                }
                let mut a = CostasState::new(order, 0.05, DAMP, 0.05).unwrap();
                let mut b = a;
                let mut ya = vec![C32::default(); x.len()];
                let mut yb = vec![C32::default(); x.len()];
                Fallback.costas(&mut a, &x, &mut ya).unwrap();
                Reference.costas(&mut b, &x, &mut yb).unwrap();
                let same = ya.iter().zip(&yb).all(|(p, q)| {
                    (p.re.to_bits() == q.re.to_bits() || (p.re.is_nan() && q.re.is_nan()))
                        && (p.im.to_bits() == q.im.to_bits() || (p.im.is_nan() && q.im.is_nan()))
                });
                assert!(same, "order {order}, offset {offset}");
                assert_eq!(a, b);
            }
            // Every pair of special values.
            let x: Vec<C32> = SPECIALS
                .iter()
                .flat_map(|&a| SPECIALS.iter().map(move |&b| C32::new(a, b)))
                .collect();
            let mut a = CostasState::new(order, 0.2, 4.0, 0.25).unwrap();
            let mut b = a;
            let mut ya = vec![C32::default(); x.len()];
            let mut yb = vec![C32::default(); x.len()];
            Fallback.costas(&mut a, &x, &mut ya).unwrap();
            Reference.costas(&mut b, &x, &mut yb).unwrap();
            assert_eq!(a, b, "order {order}");
        }
    }
}
