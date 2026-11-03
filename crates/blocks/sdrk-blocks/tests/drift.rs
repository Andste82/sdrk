//! `DriftController`: validation, the seven-step update, edge cases.
#![allow(clippy::unwrap_used, clippy::float_cmp)]

use sdrk_blocks::{DriftConfig, DriftController, Error};

fn cfg() -> DriftConfig {
    DriftConfig::new(48_000.0, 4_800.0)
}

#[test]
fn defaults() {
    let c = cfg();
    assert_eq!(c.time_constant_s, 30.0);
    assert_eq!(c.max_ppm, 500.0);
    assert_eq!(c.slew_ppm_per_s, 20.0);
    assert_eq!(c.smoothing_s, 1.0);
}

#[test]
fn invalid_configs_are_invalid_parameter() {
    let edits: [fn(&mut DriftConfig); 12] = [
        |c| c.rate_hz = 0.0,
        |c| c.rate_hz = f64::NAN,
        |c| c.target = 0.0,
        |c| c.time_constant_s = 0.5,
        |c| c.time_constant_s = 601.0,
        |c| c.max_ppm = 0.0,
        |c| c.max_ppm = 1000.5,
        |c| c.max_ppm = f64::NAN,
        |c| c.slew_ppm_per_s = 0.0,
        |c| c.smoothing_s = 0.0,
        |c| c.smoothing_s = f64::INFINITY,
        |c| c.target = -1.0,
    ];
    for edit in edits {
        let mut c = cfg();
        edit(&mut c);
        assert!(matches!(
            DriftController::new(c),
            Err(Error::InvalidParameter { .. })
        ));
    }
    let mut ok = cfg();
    ok.time_constant_s = 1.0;
    ok.max_ppm = 1000.0;
    assert!(DriftController::new(ok).is_ok());
}

#[test]
fn first_call_returns_initial_ppm() {
    let mut c = DriftController::new(cfg()).unwrap();
    c.set_initial_ppm(25.0);
    assert_eq!(c.update(3.0, 9000.0), 25.0);
    assert_eq!(c.ppm(), 25.0);
}

#[test]
fn initial_ppm_is_clamped() {
    let mut c = DriftController::new(cfg()).unwrap();
    c.set_initial_ppm(-1e9);
    assert_eq!(c.ppm(), -500.0);
    c.set_initial_ppm(f64::NAN);
    assert_eq!(c.ppm(), 0.0);
}

#[test]
fn non_positive_dt_changes_nothing() {
    let mut c = DriftController::new(cfg()).unwrap();
    c.update(0.0, 4800.0);
    let a = c.update(1.0, 5000.0);
    assert_eq!(c.update(1.0, 9000.0), a);
    assert_eq!(c.update(0.5, 9000.0), a);
    let mut d = DriftController::new(cfg()).unwrap();
    d.update(0.0, 4800.0);
    d.update(1.0, 5000.0);
    assert_eq!(c.update(2.0, 5000.0), d.update(2.0, 5000.0));
}

#[test]
fn first_step_matches_the_formulas() {
    let mut c = DriftController::new(cfg()).unwrap();
    c.update(0.0, 4800.0);
    // Error of 480 samples = 10 ms after the first call; es starts at 0 error (first fill).
    let out = c.update(0.1, 5280.0);
    let e = 480.0 / 48_000.0;
    let a = 1.0 - libm_exp(-0.1);
    let es = a * e;
    let wn = 1.0 / 30.0;
    let integ = wn * wn / 1e-6 * es * 0.1;
    let u = -(2.0 * wn / 1e-6 * es + integ);
    let expect = u.max(-2.0); // slew 20 ppm/s * 0.1 s
    assert!((out - expect).abs() < 1e-9, "{out} vs {expect}");
}

fn libm_exp(x: f64) -> f64 {
    x.exp()
}

#[test]
fn slew_and_limit_hold() {
    let mut c = DriftController::new(cfg()).unwrap();
    c.update(0.0, 4800.0);
    let mut prev = c.ppm();
    for i in 1..2000 {
        let t = f64::from(i) * 0.1;
        let out = c.update(t, 9600.0);
        assert!((out - prev).abs() <= 20.0 * 0.1 + 1e-9);
        assert!(out >= -500.0);
        prev = out;
    }
    assert_eq!(prev, -500.0);
}

#[test]
fn reset_restarts_from_initial_ppm() {
    let mut c = DriftController::new(cfg()).unwrap();
    c.set_initial_ppm(12.0);
    c.update(0.0, 4800.0);
    c.update(10.0, 6000.0);
    assert_ne!(c.ppm(), 12.0);
    c.reset();
    assert_eq!(c.ppm(), 12.0);
    assert_eq!(c.update(100.0, 1.0), 12.0);
}

#[test]
fn non_finite_input_is_ignored() {
    let mut c = DriftController::new(cfg()).unwrap();
    c.update(0.0, 4800.0);
    let a = c.update(1.0, 5000.0);
    assert_eq!(c.update(2.0, f64::NAN), a);
    assert_eq!(c.update(f64::INFINITY, 5000.0), a);
    assert!(c.update(2.0, 5000.0).is_finite());
}

#[test]
fn types_are_send_and_sync() {
    fn assert_send<T: Send + Sync>() {}
    assert_send::<DriftConfig>();
    assert_send::<DriftController>();
}
