//! Tests of the `squelch` kernel of `sdrk-core` (M19): construction rules, timing
//! declaration and the thread-safety of the public types.

#![allow(clippy::unwrap_used, clippy::float_cmp)]

use sdrk_core::kernels::squelch::SquelchState;
use sdrk_core::{C32, CQ15, Delay, Error, Rate, Ratio, Timing};
use sdrk_device::{ErrorKind, SdrkError};

fn rate() -> Rate {
    Rate::hz(48_000).unwrap()
}

#[test]
fn construction_validates_every_parameter() {
    assert!(SquelchState::<C32>::new(-40.0, 3.0, 0.01, rate()).is_ok());
    assert!(SquelchState::<CQ15>::new(0.0, 0.0, 0.01, rate()).is_ok());
    assert!(SquelchState::<C32>::new(-150.0, 40.0, 0.01, rate()).is_ok());
    let bad = [
        (f64::NAN, 3.0, 0.01),
        (-150.5, 3.0, 0.01),
        (0.1, 3.0, 0.01),
        (f64::NEG_INFINITY, 3.0, 0.01),
        (-40.0, -0.1, 0.01),
        (-40.0, f64::NAN, 0.01),
        (-40.0, f64::INFINITY, 0.01),
        (-40.0, 3.0, 0.0),
        (-40.0, 3.0, -1.0),
        (-40.0, 3.0, f64::NAN),
        (-40.0, 3.0, f64::INFINITY),
        // alpha = 1 - exp(-1 / 0.5) = 0.63 > 0.5: fewer than 1.44 samples per time constant.
        (-40.0, 3.0, 1.0 / 48_000.0),
        // alpha underflows to zero.
        (-40.0, 3.0, 1e30),
    ];
    for (t, h, tau) in bad {
        let e = SquelchState::<C32>::new(t, h, tau, rate()).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "{t} {h} {tau}");
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
        assert!(SquelchState::<CQ15>::new(t, h, tau, rate()).is_err());
    }
    // The largest accepted coefficient: alpha <= 0.5 needs tau >= 1 / (fs ln 2).
    assert!(SquelchState::<C32>::new(-40.0, 3.0, 1.45 / 48_000.0, rate()).is_ok());
    // CQ15 only: the close level must be at least -90 dB, and the Q31 coefficient must not
    // round to zero (fs * tau above about 4.3e9 samples; C32 still accepts it).
    for (t, h, tau) in [(-91.0, 0.0, 0.01), (-80.0, 10.5, 0.01), (-40.0, 3.0, 1e5)] {
        let e = SquelchState::<CQ15>::new(t, h, tau, rate()).unwrap_err();
        assert!(matches!(e, Error::InvalidParameter(_)), "{t} {h} {tau}");
        assert!(SquelchState::<C32>::new(t, h, tau, rate()).is_ok());
    }
    assert!(SquelchState::<CQ15>::new(-80.0, 10.0, 0.01, rate()).is_ok());
    assert!(SquelchState::<CQ15>::new(-40.0, 3.0, 4e4, rate()).is_ok());
}

#[test]
fn a_new_state_is_closed_with_zero_power_and_reports_its_levels() {
    let st = SquelchState::<CQ15>::new(-40.0, 3.0, 0.01, rate()).unwrap();
    assert!(!st.is_open());
    assert_eq!(st.average_power(), 0.0);
    assert_eq!((st.threshold_db(), st.hysteresis_db()), (-40.0, 3.0));
}

#[test]
fn timing_is_one_to_one_without_delay() {
    let c = SquelchState::<C32>::new(-40.0, 3.0, 0.01, rate()).unwrap();
    let q = SquelchState::<CQ15>::new(-40.0, 3.0, 0.01, rate()).unwrap();
    assert_eq!(c.ratio(), Ratio::ONE);
    assert_eq!(q.ratio(), Ratio::ONE);
    assert_eq!(c.group_delay(), Delay::ZERO);
    assert_eq!(q.group_delay(), Delay::ZERO);
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<SquelchState<C32>>();
    assert_send::<SquelchState<CQ15>>();
    assert_sync::<SquelchState<C32>>();
    assert_sync::<SquelchState<CQ15>>();
}
