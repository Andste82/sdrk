//! Tests of `design::gaussian` (M39 §1): the formula, the normalisation, the symmetry, the
//! 3 dB bandwidth and the rejections.
#![allow(clippy::unwrap_used, clippy::float_cmp, clippy::cast_precision_loss)]

use core::f64::consts::{LN_2, PI};

use sdrk_blocks::Error;
use sdrk_blocks::design::gaussian;

fn taps(bt: f64, sps: usize, span: usize) -> Vec<f64> {
    let mut h = vec![0.0; span * sps + 1];
    gaussian(bt, sps, span, &mut h).unwrap();
    h
}

/// `|H(e^{j w})|` of the taps at `f` cycles per sample.
fn magnitude(h: &[f64], f: f64) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for (n, t) in h.iter().enumerate() {
        let w = 2.0 * PI * f * n as f64;
        re += t * w.cos();
        im -= t * w.sin();
    }
    re.hypot(im)
}

#[test]
fn the_taps_follow_the_formula_and_sum_to_one() {
    for bt in [0.2, 0.3, 0.5, 1.0] {
        for sps in [2usize, 3, 4, 8, 16, 64] {
            for span in [2usize, 3, 4, 8] {
                let h = taps(bt, sps, span);
                let len = span * sps + 1;
                assert_eq!(h.len(), len);
                let c = 2.0 * PI * PI * bt * bt / LN_2;
                let raw: Vec<f64> = (0..len)
                    .map(|n| {
                        let x = (n as f64 - (len - 1) as f64 / 2.0) / sps as f64;
                        (-c * x * x).exp()
                    })
                    .collect();
                let sum: f64 = raw.iter().sum();
                for (got, r) in h.iter().zip(&raw) {
                    assert!(
                        (got - r / sum).abs() <= 1e-15,
                        "bt {bt} sps {sps} span {span}"
                    );
                }
                assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-13);
                for (a, b) in h.iter().zip(h.iter().rev()) {
                    assert_eq!(a, b, "symmetric: bt {bt} sps {sps} span {span}");
                }
                assert!(h.iter().all(|&t| t > 0.0 && t.is_finite()));
            }
        }
    }
}

#[test]
fn the_peak_is_at_the_centre() {
    let h = taps(0.5, 8, 4);
    let peak = h.iter().copied().fold(0.0, f64::max);
    assert_eq!(h[16], peak);
    assert!(h.windows(2).take(16).all(|w| w[0] < w[1]));
    // An odd length in samples (`span sps` odd): the two centre taps share the peak.
    let h = taps(0.5, 3, 3);
    assert_eq!(h.len(), 10);
    assert_eq!(h[4], h[5]);
    assert_eq!(h[4], h.iter().copied().fold(0.0, f64::max));
}

#[test]
fn the_3_db_bandwidth_is_bt_per_symbol() {
    // `bt` is the -3 dB bandwidth times the symbol period: |H| = 1/sqrt(2) at `bt / sps`
    // cycles per sample (span 8 keeps the truncation negligible).
    for bt in [0.3, 0.5, 1.0] {
        let sps = 16;
        let h = taps(bt, sps, 8);
        assert!((magnitude(&h, 0.0) - 1.0).abs() < 1e-12);
        let m = magnitude(&h, bt / sps as f64);
        assert!((m - 0.5f64.sqrt()).abs() < 1e-3, "bt {bt}: {m}");
    }
}

#[test]
fn smaller_bt_gives_a_wider_pulse() {
    let narrow = taps(1.0, 8, 4);
    let wide = taps(0.3, 8, 4);
    assert!(wide[16] < narrow[16]);
    assert!(wide[0] > narrow[0]);
}

#[test]
fn the_limits_are_accepted() {
    for (bt, sps, span) in [(0.2, 2, 2), (1.0, 2, 8), (0.2, 64, 8), (1.0, 1000, 2)] {
        let h = taps(bt, sps, span);
        assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn invalid_parameters_are_rejected() {
    let mut h = vec![0.0; 33];
    for bt in [0.19, 1.01, 0.0, -0.5, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                gaussian(bt, 8, 4, &mut h),
                Err(Error::OutOfRange { name: "bt", .. })
            ),
            "bt {bt}"
        );
    }
    for sps in [0usize, 1, usize::MAX] {
        assert!(
            matches!(
                gaussian(0.5, sps, 4, &mut h),
                Err(Error::OutOfRange { name: "sps", .. })
            ),
            "sps {sps}"
        );
    }
    for span in [0usize, 1, 9] {
        assert!(
            matches!(
                gaussian(0.5, 8, span, &mut h),
                Err(Error::OutOfRange { name: "span", .. })
            ),
            "span {span}"
        );
    }
    for len in [0usize, 32, 34] {
        assert!(matches!(
            gaussian(0.5, 8, 4, &mut vec![0.0; len]),
            Err(Error::Core(sdrk_core::Error::LengthMismatch {
                expected: 33,
                ..
            }))
        ));
    }
}
