//! M43 section 4: the multipath channel `channel::Multipath` (a single tap at delay 0 is the
//! identity, whole delays shift exactly, a fractional delay of a tone gives the analytic phase,
//! the presets, the processing lag and chunk invariance). The signals are fixed tones; no
//! random numbers are drawn.
#![cfg(feature = "std")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

use std::f64::consts::PI;

use num_complex::Complex64;
use sdrk_testkit::channel::{Impairment, MULTIPATH_MAX_DELAY, Multipath};

/// `m e^(j ph)` (with the `libm` functions of the channel).
fn polar(m: f64, ph: f64) -> Complex64 {
    Complex64::new(m * libm::cos(ph), m * libm::sin(ph))
}

fn tone(f: f64, phase: f64, n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| polar(1.0, 2.0 * PI * f * i as f64 + phase))
        .collect()
}

fn run(ch: &mut Multipath, x: &[Complex64]) -> Vec<Complex64> {
    let mut y = Vec::new();
    ch.process(x, &mut y);
    y
}

/// The same input in chunks of 1, 2, 3, ... 97 samples (cycling).
fn run_chunked(ch: &mut Multipath, x: &[Complex64]) -> Vec<Complex64> {
    let mut y = Vec::new();
    let (mut pos, mut size) = (0, 1);
    while pos < x.len() {
        let end = (pos + size).min(x.len());
        ch.process(&x[pos..end], &mut y);
        pos = end;
        size = size % 97 + 1;
    }
    y
}

fn assert_send<T: Send>() {}

#[test]
fn multipath_is_send() {
    assert_send::<Multipath>();
}

#[test]
fn a_single_tap_at_delay_zero_is_the_identity() {
    let x = tone(0.137, 0.4, 1000);
    let mut ch = Multipath::new(&[(0.0, Complex64::new(1.0, 0.0))]).unwrap();
    assert_eq!(ch.lag(), 0);
    assert_eq!(ch.response(), [Complex64::new(1.0, 0.0)]);
    assert_eq!(run(&mut ch, &x), x);
}

#[test]
fn whole_delays_shift_exactly() {
    let x = tone(0.21, -1.0, 500);
    let g1 = Complex64::new(0.5, -0.25);
    let g2 = polar(0.3, 2.0);
    let mut ch = Multipath::new(&[(0.0, Complex64::new(1.0, 0.0)), (3.0, g1), (7.0, g2)]).unwrap();
    assert_eq!(ch.lag(), 0);
    let y = run(&mut ch, &x);
    assert_eq!(y.len(), x.len());
    for (n, v) in y.iter().enumerate() {
        let mut want = x[n];
        if n >= 3 {
            want += g1 * x[n - 3];
        }
        if n >= 7 {
            want += g2 * x[n - 7];
        }
        assert!((*v - want).norm_sqr().sqrt() < 1e-15, "{n}: {v} {want}");
    }
}

#[test]
fn a_fractional_delay_of_a_tone_gives_the_analytic_phase() {
    let n = 4000;
    for (f, d) in [(0.05, 3.37), (0.12, 0.5), (-0.2, 10.81), (0.3, 1.3)] {
        let phase = 0.3;
        let g = polar(0.7, -0.9);
        let x = tone(f, phase, n);
        let mut ch = Multipath::new(&[(d, g)]).unwrap();
        let lag = ch.lag();
        assert_eq!(lag, 31 - libm::floor(d) as usize, "delay {d}");
        let y = run(&mut ch, &x);
        assert_eq!(y.len(), n - lag);
        let (mut worst, mut worst_phase) = (0.0f64, 0.0f64);
        // From output 32 + d on, the window of the interpolator lies entirely in the input.
        for (m, v) in y.iter().enumerate().skip(32 + d as usize + 1) {
            let want = g * polar(1.0, 2.0 * PI * f * (m as f64 - d) + phase);
            worst = worst.max((*v - want).norm_sqr().sqrt());
            // The phase of y against the undelayed tone: arg g - 2 pi f d.
            let q = *v / x[m];
            let rel = q.im.atan2(q.re);
            let target = g.im.atan2(g.re) - 2.0 * PI * f * d;
            let diff = (rel - target + PI).rem_euclid(2.0 * PI) - PI;
            worst_phase = worst_phase.max(diff.abs());
        }
        // The windowed sinc of M38 (32 taps per side, 4-term Blackman-Harris): below 1e-5 up to
        // 0.3 cycles per sample, as for ClockDrift.
        assert!(worst < 1e-5, "f {f}, d {d}: max error {worst}");
        assert!(
            worst_phase < 2e-5,
            "f {f}, d {d}: phase error {worst_phase}"
        );
    }
}

#[test]
fn the_output_does_not_depend_on_the_chunks() {
    let x = tone(0.08, 1.1, 3000);
    for sps in [1.0, 2.5, 4.0] {
        for preset in [Multipath::mild, Multipath::severe] {
            let whole = run(&mut preset(sps).unwrap(), &x);
            let chunked = run_chunked(&mut preset(sps).unwrap(), &x);
            assert_eq!(whole, chunked, "sps {sps}");
            assert_eq!(whole.len(), x.len() - preset(sps).unwrap().lag());
        }
    }
}

#[test]
fn the_presets_scale_the_delays_by_sps() {
    // At 10 samples per symbol every delay of `mild` and `severe` is a whole number of samples
    // (13, 27; 8, 21, 40): the response holds exactly the gains of the spec.
    let mild = Multipath::mild(10.0).unwrap();
    assert_eq!(mild.lag(), 0);
    let h = mild.response();
    assert_eq!(h.len(), 28);
    assert_eq!(h[0], Complex64::new(1.0, 0.0));
    assert!((h[13] - polar(0.3, 1.0)).norm_sqr().sqrt() < 1e-16);
    assert!((h[27] - polar(0.1, -2.0)).norm_sqr().sqrt() < 1e-16);
    let nonzero = h.iter().filter(|v| v.norm_sqr().sqrt() != 0.0).count();
    assert_eq!(nonzero, 3);

    let severe = Multipath::severe(10.0).unwrap();
    let h = severe.response();
    assert_eq!((severe.lag(), h.len()), (0, 41));
    assert!((h[8] - polar(0.6, 0.5)).norm_sqr().sqrt() < 1e-16);
    assert!((h[21] - polar(0.4, -1.5)).norm_sqr().sqrt() < 1e-16);
    assert!((h[40] - Complex64::new(0.2, 0.0)).norm_sqr().sqrt() < 1e-16);
    assert_eq!(h.iter().filter(|v| v.norm_sqr().sqrt() != 0.0).count(), 4);

    // On the symbol stream (sps 1) the fractional delays are interpolated: the direct path
    // dominates the response and the lag is 31 (the smallest fractional delay, 0.8 symbols).
    let severe1 = Multipath::severe(1.0).unwrap();
    assert_eq!(severe1.lag(), 31);
    let peak = severe1.response()[severe1.lag()];
    assert!(
        (peak - Complex64::new(1.0, 0.0)).norm_sqr().sqrt() < 0.3,
        "{peak}"
    );
    assert_eq!(Multipath::MILD.len(), 3);
    assert_eq!(Multipath::SEVERE.len(), 4);
}

#[test]
fn invalid_taps_and_sps_are_rejected() {
    let one = Complex64::new(1.0, 0.0);
    assert!(Multipath::new(&[]).is_err());
    assert!(Multipath::new(&[(-0.5, one)]).is_err());
    assert!(Multipath::new(&[(f64::NAN, one)]).is_err());
    assert!(Multipath::new(&[(MULTIPATH_MAX_DELAY + 1.0, one)]).is_err());
    assert!(Multipath::new(&[(1.0, Complex64::new(0.0, f64::INFINITY))]).is_err());
    assert!(Multipath::new(&[(MULTIPATH_MAX_DELAY, one)]).is_ok());
    for sps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(Multipath::mild(sps).is_err(), "{sps}");
        assert!(Multipath::severe(sps).is_err(), "{sps}");
    }
    assert!(Multipath::severe(MULTIPATH_MAX_DELAY).is_err());
}
