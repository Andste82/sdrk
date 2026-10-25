//! Tests of the M20 items of `sdrk-core`: `isqrt_u64`, the construction rules and timing of
//! `AgcState` and `DcBlockRealState`, and the thread-safety of the public types.

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use sdrk_core::kernels::agc::AgcState;
use sdrk_core::kernels::demod::DcBlockRealState;
use sdrk_core::num::isqrt_u64;
use sdrk_core::{C32, CQ15, Delay, Error, Q15, Rate, Ratio, Timing};
use sdrk_device::{ErrorKind, SdrkError};

fn rate() -> Rate {
    Rate::hz(48_000).unwrap()
}

#[test]
fn isqrt_is_exact_at_the_boundaries() {
    assert_eq!(isqrt_u64(0), 0);
    assert_eq!(isqrt_u64(1), 1);
    assert_eq!(isqrt_u64(2), 1);
    assert_eq!(isqrt_u64(3), 1);
    assert_eq!(isqrt_u64(4), 2);
    assert_eq!(isqrt_u64(u64::MAX), u64::from(u32::MAX));
    assert_eq!(isqrt_u64(u64::MAX - 1), u64::from(u32::MAX));
    // r^2 - 1, r^2 and (r + 1)^2 - 1 for the small roots, the powers of two and the top.
    let mut roots: Vec<u64> = (1..=2000).collect();
    for k in 1..32 {
        roots.extend([(1u64 << k) - 1, 1 << k, (1 << k) + 1]);
    }
    roots.extend([u64::from(u32::MAX), u64::from(u32::MAX) - 1, 3_037_000_499]);
    for r in roots {
        let sq = u128::from(r) * u128::from(r);
        let check = |x: u128, want: u64| {
            if x <= u128::from(u64::MAX) {
                assert_eq!(isqrt_u64(x as u64), want, "x = {x}");
            }
        };
        check(sq - 1, r - 1);
        check(sq, r);
        check(sq + 2 * u128::from(r), r); // (r + 1)^2 - 1
    }
}

#[test]
fn isqrt_matches_u128_squaring_on_random_values() {
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for i in 0..200_000u32 {
        // Mix magnitudes: full range and values with fewer significant bits.
        let x = next() >> (i % 64);
        let r = isqrt_u64(x);
        assert!(u128::from(r) * u128::from(r) <= u128::from(x), "x = {x}");
        assert!(
            u128::from(r + 1) * u128::from(r + 1) > u128::from(x),
            "x = {x}"
        );
    }
}

#[test]
fn agc_construction_validates_every_parameter() {
    let ok = |r: f64, a: f64, rel: f64, lo: f64, hi: f64, init: f64| {
        AgcState::<C32>::new(r, a, rel, lo, hi, init, rate())
    };
    assert!(ok(0.5, 0.005, 0.05, -20.0, 60.0, 0.0).is_ok());
    assert!(ok(1.0 / 32.0, 0.005, 0.05, -40.0, 90.0, 90.0).is_ok());
    assert!(ok(1.0, 0.005, 0.05, -40.0, -40.0, -40.0).is_ok());
    // A time constant of 1.45 samples is the shortest accepted (rate 0.5 at 1 / ln 2).
    assert!(ok(0.5, 1.45 / 48_000.0, 1.45 / 48_000.0, -20.0, 60.0, 0.0).is_ok());
    let bad = [
        (0.03, 0.005, 0.05, -20.0, 60.0, 0.0),
        (1.01, 0.005, 0.05, -20.0, 60.0, 0.0),
        (f64::NAN, 0.005, 0.05, -20.0, 60.0, 0.0),
        (0.5, 0.0, 0.05, -20.0, 60.0, 0.0),
        (0.5, -1.0, 0.05, -20.0, 60.0, 0.0),
        (0.5, f64::NAN, 0.05, -20.0, 60.0, 0.0),
        (0.5, f64::INFINITY, 0.05, -20.0, 60.0, 0.0),
        (0.5, 0.005, 0.0, -20.0, 60.0, 0.0),
        (0.5, 0.005, f64::NAN, -20.0, 60.0, 0.0),
        // rate = 1 - exp(-1 / 1.0) = 0.63 > 0.5
        (0.5, 1.0 / 48_000.0, 0.05, -20.0, 60.0, 0.0),
        (0.5, 0.005, 1.0 / 48_000.0, -20.0, 60.0, 0.0),
        // the rate underflows to zero
        (0.5, 0.005, 1e30, -20.0, 60.0, 0.0),
        (0.5, 0.005, 0.05, -40.5, 60.0, 0.0),
        (0.5, 0.005, 0.05, -20.0, 90.5, 0.0),
        (0.5, 0.005, 0.05, 10.0, 5.0, 7.0),
        (0.5, 0.005, 0.05, -20.0, 60.0, -21.0),
        (0.5, 0.005, 0.05, -20.0, 60.0, 61.0),
        (0.5, 0.005, 0.05, -20.0, 60.0, f64::NAN),
        (0.5, 0.005, 0.05, f64::NAN, 60.0, 0.0),
        (0.5, 0.005, 0.05, -20.0, f64::INFINITY, 0.0),
    ];
    for (r, a, rel, lo, hi, init) in bad {
        let e = ok(r, a, rel, lo, hi, init).unwrap_err();
        assert!(
            matches!(e, Error::InvalidParameter(_)),
            "{r} {a} {rel} {lo} {hi} {init}"
        );
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
        assert!(AgcState::<CQ15>::new(r, a, rel, lo, hi, init, rate()).is_err());
    }
}

#[test]
fn agc_state_starts_at_the_initial_gain() {
    let c = AgcState::<C32>::new(0.5, 0.005, 0.05, -20.0, 60.0, 12.0, rate()).unwrap();
    assert!((c.gain_db() - 12.0).abs() < 1e-5);
    let q = AgcState::<CQ15>::new(0.5, 0.005, 0.05, -20.0, 60.0, 12.0, rate()).unwrap();
    assert!((q.gain_db() - 12.0).abs() < 1e-8);
}

#[test]
fn timing_is_one_to_one_without_delay() {
    let a = AgcState::<C32>::new(0.5, 0.005, 0.05, -20.0, 60.0, 0.0, rate()).unwrap();
    let b = AgcState::<CQ15>::new(0.5, 0.005, 0.05, -20.0, 60.0, 0.0, rate()).unwrap();
    let c = DcBlockRealState::<f32>::new(0.999).unwrap();
    let d = DcBlockRealState::<Q15>::new(Q15(32700)).unwrap();
    assert_eq!(
        (a.ratio(), b.ratio(), c.ratio(), d.ratio()),
        (Ratio::ONE, Ratio::ONE, Ratio::ONE, Ratio::ONE)
    );
    assert_eq!(a.group_delay(), Delay::ZERO);
    assert_eq!(b.group_delay(), Delay::ZERO);
    assert_eq!(c.group_delay(), Delay::ZERO);
    assert_eq!(d.group_delay(), Delay::ZERO);
}

#[test]
fn dc_block_real_construction_rules() {
    assert!(DcBlockRealState::<f32>::new(0.999).is_ok());
    for r in [0.0, 1.0, -0.5, 1.5, f32::NAN, f32::INFINITY] {
        let e = DcBlockRealState::<f32>::new(r).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "{r}");
    }
    assert!(DcBlockRealState::<Q15>::new(Q15(1)).is_ok());
    assert!(DcBlockRealState::<Q15>::new(Q15(32767)).is_ok());
    assert!(DcBlockRealState::<Q15>::new(Q15(0)).is_err());
    assert!(DcBlockRealState::<Q15>::new(Q15(-5)).is_err());
    let mut st = DcBlockRealState::<f32>::new(0.9).unwrap();
    assert_eq!((st.r(), st.x_prev(), st.y_prev()), (0.9, 0.0, 0.0));
    st.reset();
    assert_eq!(st.r(), 0.9);
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<AgcState<C32>>();
    assert_send::<AgcState<CQ15>>();
    assert_send::<DcBlockRealState<f32>>();
    assert_send::<DcBlockRealState<Q15>>();
    assert_sync::<AgcState<C32>>();
    assert_sync::<AgcState<CQ15>>();
    assert_sync::<DcBlockRealState<f32>>();
    assert_sync::<DcBlockRealState<Q15>>();
}
