//! The biquad design functions: closed forms, the Butterworth gain at the cutoff, and the
//! error cases (the `SciPy` comparison is a golden test in `sdrk-testkit`).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use core::f64::consts::PI;

use sdrk_blocks::Error;
use sdrk_blocks::design::{BiquadResponse, biquad_coefs, butterworth_sos, to_f32_sos, to_q14};

/// `|H(e^{jw})|` of one section.
fn gain(c: &[f64; 5], w: f64) -> f64 {
    let e = |k: f64| (f64::cos(k * w), -f64::sin(k * w));
    let (e1, e2) = (e(1.0), e(2.0));
    let num = (c[0] + c[1] * e1.0 + c[2] * e2.0, c[1] * e1.1 + c[2] * e2.1);
    let den = (1.0 + c[3] * e1.0 + c[4] * e2.0, c[3] * e1.1 + c[4] * e2.1);
    f64::hypot(num.0, num.1) / f64::hypot(den.0, den.1)
}

fn cascade_db(sos: &[[f64; 5]], f_hz: f64, fs_hz: f64) -> f64 {
    let w = 2.0 * PI * f_hz / fs_hz;
    20.0 * sos.iter().map(|c| gain(c, w)).product::<f64>().log10()
}

fn butter(response: BiquadResponse, order: usize, cutoff: f64, fs: f64) -> Vec<[f64; 5]> {
    let mut sos = vec![[0.0; 5]; order / 2];
    butterworth_sos(response, order, cutoff, fs, &mut sos).unwrap();
    sos
}

#[test]
fn butterworth_is_minus_3_0103_db_at_the_cutoff_for_every_order() {
    for order in [2, 4, 6, 8] {
        for (cutoff, fs) in [
            (0.05, 1.0),
            (0.2, 1.0),
            (1_000.0, 48_000.0),
            (4_000.0, 48_000.0),
        ] {
            for response in [BiquadResponse::Lowpass, BiquadResponse::Highpass] {
                let sos = butter(response, order, cutoff, fs);
                let db = cascade_db(&sos, cutoff, fs);
                assert!(
                    (db + 3.0103).abs() <= 0.001,
                    "{response:?} order {order} cutoff {cutoff}/{fs}: {db} dB"
                );
            }
        }
    }
}

#[test]
fn butterworth_passband_and_stopband_slopes() {
    // 4th order low-pass at 1 kHz of 48 kHz: flat at DC, 24 dB per octave beyond the cutoff.
    let sos = butter(BiquadResponse::Lowpass, 4, 1_000.0, 48_000.0);
    assert!(cascade_db(&sos, 1.0, 48_000.0).abs() < 1e-6);
    let (a, b) = (
        cascade_db(&sos, 2_000.0, 48_000.0),
        cascade_db(&sos, 4_000.0, 48_000.0),
    );
    assert!((a - b - 24.0).abs() < 2.0, "{a} {b}");
    // The high-pass is flat towards Nyquist.
    let sos = butter(BiquadResponse::Highpass, 6, 2_000.0, 48_000.0);
    assert!(cascade_db(&sos, 23_900.0, 48_000.0).abs() < 0.01);
}

#[test]
fn butterworth_sections_have_the_lowest_q_first() {
    // The pole radius sqrt(a2) grows with Q.
    let sos = butter(BiquadResponse::Lowpass, 8, 0.05, 1.0);
    assert_eq!(sos.len(), 4);
    for pair in sos.windows(2) {
        assert!(pair[0][4] < pair[1][4]);
    }
}

#[test]
fn cookbook_sections_have_their_defining_responses() {
    let (f0, fs) = (1_000.0, 48_000.0);
    let w0 = 2.0 * PI * f0 / fs;
    let lp = biquad_coefs(
        BiquadResponse::Lowpass,
        f0,
        core::f64::consts::FRAC_1_SQRT_2,
        fs,
    )
    .unwrap();
    assert!((gain(&lp, 0.0) - 1.0).abs() < 1e-12);
    let hp = biquad_coefs(
        BiquadResponse::Highpass,
        f0,
        core::f64::consts::FRAC_1_SQRT_2,
        fs,
    )
    .unwrap();
    assert!((gain(&hp, PI) - 1.0).abs() < 1e-12);
    // The band-pass has 0 dB peak gain at the centre frequency, the notch a zero there.
    let bp = biquad_coefs(BiquadResponse::Bandpass, f0, 2.0, fs).unwrap();
    assert!((gain(&bp, w0) - 1.0).abs() < 1e-12);
    let notch = biquad_coefs(BiquadResponse::Notch, f0, 10.0, fs).unwrap();
    assert!(gain(&notch, w0) < 1e-12);
    assert!((gain(&notch, 0.0) - 1.0).abs() < 1e-9);
    // The cookbook low-pass with q = 1/sqrt(2) is the second-order Butterworth section.
    let b2 = butter(BiquadResponse::Lowpass, 2, f0, fs);
    let q = 1.0 / (2.0 * (PI / 4.0).cos());
    let lp = biquad_coefs(BiquadResponse::Lowpass, f0, q, fs).unwrap();
    assert_eq!(b2, [lp]);
}

#[test]
fn biquad_coefs_closed_form() {
    // f0 = fs / 4: c = 0, alpha = 1 / (2 q): lowpass b = [1/2, 1, 1/2] / (1 + alpha).
    let c = biquad_coefs(BiquadResponse::Lowpass, 0.25, 1.0, 1.0).unwrap();
    let a0 = 1.5;
    let want = [0.5 / a0, 1.0 / a0, 0.5 / a0, 0.0, 0.5 / a0];
    for (x, y) in c.iter().zip(want) {
        assert!((x - y).abs() < 1e-15, "{c:?}");
    }
}

#[test]
fn biquad_coefs_validation() {
    let ok = |f0, q, fs| biquad_coefs(BiquadResponse::Lowpass, f0, q, fs);
    assert!(ok(0.1, 0.7, 1.0).is_ok());
    for (f0, q, fs) in [
        (0.0, 0.7, 1.0),
        (-0.1, 0.7, 1.0),
        (0.5, 0.7, 1.0),
        (0.7, 0.7, 1.0),
        (f64::NAN, 0.7, 1.0),
        (0.1, 0.0, 1.0),
        (0.1, -1.0, 1.0),
        (0.1, f64::NAN, 1.0),
        (0.1, 0.7, 0.0),
        (0.1, 0.7, f64::NAN),
    ] {
        assert!(
            matches!(ok(f0, q, fs), Err(Error::OutOfRange { .. })),
            "{f0} {q} {fs}"
        );
    }
}

#[test]
fn butterworth_validation() {
    let mut sos = [[0.0; 5]; 5];
    for order in [0, 1, 3, 5, 7, 9, 10] {
        let r = butterworth_sos(
            BiquadResponse::Lowpass,
            order,
            0.1,
            1.0,
            &mut sos[..order / 2],
        );
        assert!(
            matches!(r, Err(Error::InvalidParameter { .. })),
            "order {order}"
        );
    }
    for response in [BiquadResponse::Bandpass, BiquadResponse::Notch] {
        let r = butterworth_sos(response, 2, 0.1, 1.0, &mut sos[..1]);
        assert!(matches!(r, Err(Error::InvalidParameter { .. })));
    }
    // The slice must hold order / 2 sections.
    for len in [0, 1, 3, 4] {
        assert!(butterworth_sos(BiquadResponse::Lowpass, 4, 0.1, 1.0, &mut sos[..len]).is_err());
    }
    assert!(butterworth_sos(BiquadResponse::Lowpass, 4, 0.5, 1.0, &mut sos[..2]).is_err());
    assert!(butterworth_sos(BiquadResponse::Lowpass, 4, 0.0, 1.0, &mut sos[..2]).is_err());
}

#[test]
fn to_q14_rounds_half_up_and_rejects_values_outside_the_range() {
    let mut dst = [[0i16; 5]; 2];
    let src = [
        [
            0.5,
            -2.0,
            1.999_938_964_843_75,
            1.0 / 32768.0,
            -1.0 / 32768.0,
        ],
        [0.0, 1.0, -1.0, 0.25, -0.25],
    ];
    to_q14(&src, &mut dst).unwrap();
    // 1/32768 is half an LSB: it rounds up, -1/32768 rounds to zero.
    assert_eq!(dst[0], [8192, -32768, 32767, 1, 0]);
    assert_eq!(dst[1], [0, 16384, -16384, 4096, -4096]);
    for bad in [2.0, -2.0001, f64::NAN, f64::INFINITY] {
        let r = to_q14(&[[0.0, 0.0, 0.0, bad, 0.0]], &mut dst[..1]);
        assert!(matches!(r, Err(Error::OutOfRange { .. })), "{bad}");
    }
    assert!(to_q14(&src, &mut dst[..1]).is_err());
}

#[test]
fn to_f32_sos_rounds_to_nearest() {
    let mut dst = [[0.0f32; 5]; 1];
    to_f32_sos(&[[0.1, -1.9, 1e-40, 3.0, 0.0]], &mut dst).unwrap();
    assert_eq!(dst[0], [0.1f32, -1.9f32, 1e-40f32, 3.0, 0.0]);
    assert!(to_f32_sos(&[[0.0; 5]; 2], &mut dst).is_err());
}

#[test]
fn a_butterworth_design_fits_q14_down_to_fs_over_500() {
    // The coefficient a1 stays above -2 for any cutoff, so the Q2.14 conversion succeeds; the
    // quality of the filter is the business of the block (cutoff >= fs / 500).
    for cutoff in [0.002, 0.01, 0.1, 0.4] {
        let sos = butter(BiquadResponse::Lowpass, 8, cutoff, 1.0);
        let mut q = vec![[0i16; 5]; sos.len()];
        to_q14(&sos, &mut q).unwrap();
    }
}
