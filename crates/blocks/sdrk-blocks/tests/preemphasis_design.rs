//! Tests of `preemphasis_coefs` (M25 test 2): the magnitude against the ideal pre-emphasis, the
//! DC gain, the Nyquist gain, the pole position and the rejections.
#![allow(clippy::unwrap_used, clippy::float_cmp, clippy::cast_precision_loss)]

use core::f64::consts::PI;

use sdrk_blocks::Error;
use sdrk_blocks::design::preemphasis_coefs;

/// `|H(e^{j w})|` of `(b0 + b1 z^-1) / (1 + a1 z^-1)` at `f` hertz.
fn magnitude(c: [f64; 3], f: f64, fs: f64) -> f64 {
    let w = 2.0 * PI * f / fs;
    let (cos, sin) = (w.cos(), w.sin());
    let num = ((c[0] + c[1] * cos).powi(2) + (c[1] * sin).powi(2)).sqrt();
    let den = ((1.0 + c[2] * cos).powi(2) + (c[2] * sin).powi(2)).sqrt();
    num / den
}

#[test]
fn the_magnitude_follows_the_ideal_pre_emphasis_within_0_2_db() {
    let mut worst = 0.0f64;
    for fs in [44_100.0, 48_000.0, 96_000.0, 240_000.0] {
        for tau_us in [50.0, 75.0, 530.0] {
            let tau = tau_us * 1e-6;
            let c = preemphasis_coefs(tau, fs).unwrap();
            let top = 15_000.0f64.min(0.45 * fs);
            let mut f = 30.0;
            while f <= top {
                let ideal = (1.0 + (2.0 * PI * f * tau).powi(2)).sqrt();
                let err = 20.0 * (magnitude(c, f, fs) / ideal).log10();
                worst = worst.max(err.abs());
                assert!(
                    err.abs() <= 0.2,
                    "fs {fs} tau {tau_us} us f {f}: {err:.3} dB"
                );
                f *= 1.02;
            }
        }
    }
    // The measured worst case, for the record in docs/kernels.md.
    assert!(worst > 0.0);
}

#[test]
fn dc_gain_is_one_and_the_nyquist_gain_is_tau_over_tau_p() {
    for fs in [40_000.0, 48_000.0, 240_000.0] {
        for tau in [1e-6, 50e-6, 530e-6, 1e-3] {
            let c = preemphasis_coefs(tau, fs).unwrap();
            assert!(((c[0] + c[1]) / (1.0 + c[2]) - 1.0).abs() < 1e-12);
            let tau_p = 1.0 / (2.0 * PI * fs / 2.4);
            // At Nyquist z = -1: (b0 - b1) / (1 - a1) = K tau * 2 / (K tau_p * 2) = tau / tau_p.
            let nyquist = (c[0] - c[1]) / (1.0 - c[2]);
            assert!((nyquist / (tau / tau_p) - 1.0).abs() < 1e-12);
        }
    }
    // 16 dB for 50 us at 48 kHz, 36 dB for 530 us at 48 kHz, 56 dB for 1000 us at 240 kHz.
    let db = |tau: f64, fs: f64| {
        let c = preemphasis_coefs(tau, fs).unwrap();
        20.0 * ((c[0] - c[1]) / (1.0 - c[2])).log10()
    };
    assert!((db(50e-6, 48_000.0) - 16.0).abs() < 0.5);
    assert!((db(530e-6, 48_000.0) - 36.0).abs() < 0.5);
    assert!((db(1e-3, 240_000.0) - 56.0).abs() < 0.5);
}

#[test]
fn the_pole_is_the_one_of_the_bilinear_transform_of_fs_over_2_4() {
    // a1 = (1 - K tau_p) / (1 + K tau_p) with K tau_p = 2.4 / pi, for every rate.
    for fs in [40_000.0, 48_000.0, 240_000.0] {
        let c = preemphasis_coefs(50e-6, fs).unwrap();
        let k_tau_p = 2.4 / PI;
        assert!((c[2] - (1.0 - k_tau_p) / (1.0 + k_tau_p)).abs() < 1e-12);
    }
}

#[test]
fn rejections() {
    let out_of_range = |r: sdrk_blocks::Result<[f64; 3]>, which: &str| match r {
        Err(Error::OutOfRange { name, .. }) => assert_eq!(name, which),
        other => panic!("{other:?}"),
    };
    for tau in [0.0, -1e-6, 1.0001e-3, 1e-2, f64::NAN, f64::INFINITY] {
        out_of_range(preemphasis_coefs(tau, 48_000.0), "tau_s");
    }
    for fs in [39_999.9, 32_000.0, 0.0, -48_000.0, f64::NAN, f64::INFINITY] {
        out_of_range(preemphasis_coefs(50e-6, fs), "fs_hz");
    }
    // The limits themselves are accepted.
    assert!(preemphasis_coefs(1e-3, 40_000.0).is_ok());
    assert!(preemphasis_coefs(1e-12, 1e9).is_ok());
}
