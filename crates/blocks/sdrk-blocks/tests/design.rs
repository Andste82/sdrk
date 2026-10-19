//! Filter design: properties, closed forms, error cases (the SciPy comparison is a golden test
//! in `sdrk-testkit`).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_lossless,
    clippy::items_after_statements,
    clippy::useless_vec,
    clippy::assert_is_empty,
    clippy::range_plus_one,
    clippy::large_stack_arrays,
    clippy::doc_markdown,
    clippy::multiple_bound_locations,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation
)]

use core::f64::consts::PI;

use sdrk_blocks::Error;
use sdrk_blocks::design::{
    DesignSpec, DesignWindow, firwin_lowpass, kaiser_beta, kaiserord, to_f32, to_q15,
};
use sdrk_core::Q15;

fn taps(cutoff: f64, fs: f64, window: DesignWindow, n: usize) -> Vec<f64> {
    let mut h = vec![0.0; n];
    firwin_lowpass(cutoff, fs, window, &mut h).unwrap();
    h
}

#[test]
fn a_single_tap_is_one() {
    for w in [
        DesignWindow::Hann,
        DesignWindow::Hamming,
        DesignWindow::Blackman,
        DesignWindow::Kaiser { beta: 5.0 },
    ] {
        assert_eq!(taps(0.1, 1.0, w, 1), [1.0]);
    }
}

#[test]
fn every_window_gives_a_symmetric_filter_with_unit_dc_gain() {
    for w in [
        DesignWindow::Hann,
        DesignWindow::Hamming,
        DesignWindow::Blackman,
        DesignWindow::Kaiser { beta: 8.6 },
    ] {
        for n in [3, 31, 64, 201] {
            let h = taps(1_000.0, 8_000.0, w, n);
            assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{w:?} {n}");
            for i in 0..n / 2 {
                assert!((h[i] - h[n - 1 - i]).abs() < 1e-15, "{w:?} {n} {i}");
            }
        }
    }
}

#[test]
fn hann_matches_the_closed_form() {
    // N = 5, fc = 0.2: h[n] = 2 fc sinc(2 fc (n - 2)) w[n] / sum.
    let n = 5;
    let fc = 0.2;
    let raw: Vec<f64> = (0..n)
        .map(|i| {
            let x = 2.0 * fc * (i as f64 - 2.0);
            let sinc = if x == 0.0 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / 4.0).cos();
            2.0 * fc * sinc * w
        })
        .collect();
    let sum: f64 = raw.iter().sum();
    let h = taps(0.2, 1.0, DesignWindow::Hann, n);
    for (a, b) in h.iter().zip(&raw) {
        assert!((a - b / sum).abs() < 1e-15);
    }
    // The end points of a symmetric Hann window are zero.
    assert!(h[0].abs() < 1e-15 && h[4].abs() < 1e-15);
}

#[test]
fn the_filter_passes_dc_and_stops_far_above_the_cutoff() {
    let beta = kaiser_beta(60.0);
    let (n, _) = kaiserord(60.0, 500.0, 8_000.0).unwrap();
    let h = taps(1_000.0, 8_000.0, DesignWindow::Kaiser { beta }, n);
    let response = |f: f64| {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, v) in h.iter().enumerate() {
            let p = -2.0 * PI * f * i as f64 / 8_000.0;
            re += v * p.cos();
            im += v * p.sin();
        }
        (re * re + im * im).sqrt()
    };
    assert!((response(0.0) - 1.0).abs() < 1e-9);
    assert!((response(1_000.0) - 0.5).abs() < 0.02); // the -6 dB point
    assert!(response(1_300.0) < 1e-3, "stop band at -60 dB");
    assert!(response(3_500.0) < 1e-3);
}

#[test]
fn a_window_without_dc_gain_is_an_error_not_nan() {
    // A symmetric Hann window of two points is [0, 0].
    let mut h = [0.0; 2];
    let e = firwin_lowpass(0.1, 1.0, DesignWindow::Hann, &mut h).unwrap_err();
    assert!(matches!(e, Error::InvalidParameter { name: "window", .. }));
    assert!(firwin_lowpass(0.1, 1.0, DesignWindow::Hamming, &mut h).is_ok());
}

#[test]
fn the_cutoff_must_be_between_zero_and_half_the_rate() {
    let mut h = [0.0; 9];
    for cutoff in [0.0, -1.0, 500.0, 600.0, f64::NAN, f64::INFINITY] {
        let e = firwin_lowpass(cutoff, 1_000.0, DesignWindow::Hann, &mut h).unwrap_err();
        assert!(
            matches!(
                e,
                Error::OutOfRange {
                    name: "cutoff_hz",
                    ..
                }
            ),
            "{cutoff}: {e}"
        );
    }
    assert!(firwin_lowpass(100.0, 0.0, DesignWindow::Hann, &mut h).is_err());
    assert!(firwin_lowpass(100.0, f64::NAN, DesignWindow::Hann, &mut h).is_err());
    assert!(matches!(
        firwin_lowpass(100.0, 1_000.0, DesignWindow::Hann, &mut []),
        Err(Error::InvalidParameter { .. })
    ));
    for beta in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            firwin_lowpass(100.0, 1_000.0, DesignWindow::Kaiser { beta }, &mut h),
            Err(Error::InvalidParameter { name: "beta", .. })
        ));
    }
}

#[test]
fn kaiser_beta_follows_the_three_branches() {
    assert_eq!(kaiser_beta(8.0), 0.0);
    assert_eq!(kaiser_beta(21.0), 0.0);
    assert!((kaiser_beta(30.0) - (0.5842 * 9.0f64.powf(0.4) + 0.07886 * 9.0)).abs() < 1e-12);
    assert!((kaiser_beta(50.0) - (0.5842 * 29.0f64.powf(0.4) + 0.07886 * 29.0)).abs() < 1e-12);
    assert!((kaiser_beta(60.0) - 0.1102 * 51.3).abs() < 1e-12);
    // The two formulas almost meet at 50 dB (SciPy has the same small step of 0.02).
    assert!((kaiser_beta(50.0001) - kaiser_beta(50.0)).abs() < 0.05);
}

#[test]
fn kaiserord_follows_the_formula_and_is_odd() {
    for (a, t, fs) in [
        (60.0, 1_000.0, 48_000.0),
        (80.0, 100.0, 10_000.0),
        (40.0, 2_000.0, 8_000.0),
        (-60.0, 1_000.0, 48_000.0),
    ] {
        let (n, beta) = kaiserord(a, t, fs).unwrap();
        let a = f64::abs(a);
        let width = 2.0 * t / fs;
        let want = ((a - 7.95) / (2.285 * PI * width) + 1.0).ceil() as usize;
        assert!(n == want || n == want + 1, "{n} vs {want}");
        assert_eq!(n % 2, 1);
        assert_eq!(beta, kaiser_beta(a));
    }
    assert!(matches!(
        kaiserord(7.9, 100.0, 1_000.0),
        Err(Error::InvalidParameter { .. })
    ));
    assert!(kaiserord(f64::NAN, 100.0, 1_000.0).is_err());
    assert!(kaiserord(60.0, 0.0, 1_000.0).is_err());
    assert!(kaiserord(60.0, -5.0, 1_000.0).is_err());
    assert!(kaiserord(60.0, 5.0, 0.0).is_err());
    assert!(kaiserord(60.0, 1e-12, 1e12).is_err()); // absurdly many taps
}

#[test]
fn conversion_to_f32_and_q15() {
    let src = [0.5, -0.25, 1.0, -1.0, 1.0 / 65536.0];
    let mut f = [0.0f32; 5];
    to_f32(&src, &mut f).unwrap();
    assert_eq!(f, [0.5, -0.25, 1.0, -1.0, 1.0 / 65536.0]);
    let mut q = [Q15(0); 5];
    to_q15(&src, &mut q).unwrap();
    // Round half up: 1 / 65536 is half an LSB, so it rounds to 1; +1.0 saturates.
    assert_eq!(q, [Q15(16384), Q15(-8192), Q15(32767), Q15(-32768), Q15(1)]);
    assert!(to_f32(&src, &mut [0.0; 4]).is_err());
    assert!(to_q15(&src, &mut [Q15(0); 6]).is_err());
}

#[test]
fn q15_conversion_rejects_taps_that_would_saturate_badly() {
    let mut q = [Q15(0); 1];
    assert!(to_q15(&[1.009], &mut q).is_ok());
    for bad in [1.011, -1.011, 2.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                to_q15(&[bad], &mut q),
                Err(Error::OutOfRange { name: "tap", .. })
            ),
            "{bad}"
        );
    }
}

#[test]
fn design_spec_counts_designs_and_caps_the_taps() {
    let spec = DesignSpec {
        cutoff_hz: 4_000.0,
        transition_hz: 1_000.0,
        attenuation_db: 60.0,
    };
    let n = spec.num_taps(48_000.0, 1023).unwrap();
    let (want, _) = kaiserord(60.0, 1_000.0, 48_000.0).unwrap();
    assert_eq!(n, want);
    let mut h = vec![0.0; n];
    spec.design(48_000.0, 1023, &mut h).unwrap();
    assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // The cap.
    let e = spec.num_taps(48_000.0, n - 1).unwrap_err();
    assert!(matches!(
        e,
        Error::OutOfRange {
            name: "max_taps",
            ..
        }
    ));
    assert!(spec.num_taps(48_000.0, n).is_ok());
    // The slice must have the right length.
    assert!(spec.design(48_000.0, 1023, &mut h[1..]).is_err());
    // The stop band must stay below half the rate.
    let wide = DesignSpec {
        cutoff_hz: 23_800.0,
        transition_hz: 1_000.0,
        attenuation_db: 60.0,
    };
    assert!(matches!(
        wide.num_taps(48_000.0, 1023),
        Err(Error::OutOfRange {
            name: "cutoff_hz",
            ..
        })
    ));
    for bad in [
        DesignSpec {
            cutoff_hz: 0.0,
            ..spec
        },
        DesignSpec {
            cutoff_hz: -1.0,
            ..spec
        },
        DesignSpec {
            cutoff_hz: f64::NAN,
            ..spec
        },
        DesignSpec {
            transition_hz: 0.0,
            ..spec
        },
        DesignSpec {
            transition_hz: f64::NAN,
            ..spec
        },
        DesignSpec {
            attenuation_db: 7.0,
            ..spec
        },
    ] {
        assert!(bad.num_taps(48_000.0, 1023).is_err(), "{bad:?}");
    }
    assert!(spec.num_taps(0.0, 1023).is_err());
}
