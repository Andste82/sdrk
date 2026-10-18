//! Tests of the M08 testkit additions: the FM generator, the meters and the low-pass helper.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::cast_lossless
)]

use core::f64::consts::PI;

use num_complex::Complex64;
use sdrk_testkit::Pcg32;
use sdrk_testkit::design_helpers::lowpass_hann;
use sdrk_testkit::fm::fm_baseband;
use sdrk_testkit::meters::{
    ESTIMATE_MAX_SAMPLES, MASK_SEGMENT, ToneMeter, estimate_frequency, fit_frequency, sinad_db,
    spectral_mask,
};

fn polar(r: f64, theta: f64) -> Complex64 {
    Complex64::new(r * theta.cos(), r * theta.sin())
}

fn cosine(n: usize, rate: f64, freq: f64, amp: f64, phase: f64, dc: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (2.0 * PI * freq * i as f64 / rate + phase).cos() + dc)
        .collect()
}

// --- fm_baseband ------------------------------------------------------------------------

fn fm(n: usize, start: u64) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    fm_baseband(&mut out, 2.4e6, 600e3, 75e3, 1e3, 1.0, start);
    out
}

#[test]
fn fm_baseband_has_unit_amplitude_and_starts_at_phase_zero() {
    let x = fm(5000, 0);
    assert_eq!(x[0], Complex64::new(1.0, 0.0));
    for s in &x {
        assert!((s.norm_sqr().sqrt() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn fm_baseband_instantaneous_frequency_follows_the_audio() {
    let (rate, offset, dev, audio_hz, amp) = (2.4e6, 600e3, 75e3, 1e3, 0.8);
    let mut x = vec![Complex64::new(0.0, 0.0); 4800];
    fm_baseband(&mut x, rate, offset, dev, audio_hz, amp, 0);
    for n in 1..x.len() {
        // The phase step between samples n - 1 and n is the integral of the frequency over one
        // sample; the midpoint frequency approximates it to second order.
        let d = x[n] * x[n - 1].conj();
        let step = d.im.atan2(d.re);
        let t_mid = (n as f64 - 0.5) / rate;
        let f_mid = offset + dev * amp * (2.0 * PI * audio_hz * t_mid).sin();
        let want = 2.0 * PI * f_mid / rate;
        let want = (want + PI).rem_euclid(2.0 * PI) - PI;
        assert!((step - want).abs() < 1e-5, "n = {n}: {step} vs {want}");
    }
}

#[test]
fn fm_baseband_chunked_generation_is_exact() {
    let whole = fm(3000, 0);
    let mut chunked = vec![Complex64::new(0.0, 0.0); 3000];
    let mut pos = 0;
    for size in [1usize, 7, 499, 33, 1000, 600].iter().cycle() {
        if pos >= 3000 {
            break;
        }
        let end = (pos + size).min(3000);
        fm_baseband(
            &mut chunked[pos..end],
            2.4e6,
            600e3,
            75e3,
            1e3,
            1.0,
            pos as u64,
        );
        pos = end;
    }
    assert_eq!(whole, chunked);
    // A late start equals the same samples of a long run.
    let late = fm(10, 2_399_990);
    let all = fm(2_400_000, 0);
    assert_eq!(&all[2_399_990..], &late[..]);
}

#[test]
fn fm_baseband_without_modulation_is_a_carrier() {
    let mut x = vec![Complex64::new(0.0, 0.0); 8];
    fm_baseband(&mut x, 1000.0, 125.0, 75.0, 1.0, 0.0, 0);
    for (n, s) in x.iter().enumerate() {
        let a = 2.0 * PI * n as f64 / 8.0;
        assert!((s.re - a.cos()).abs() < 1e-12 && (s.im - a.sin()).abs() < 1e-12);
    }
}

// --- lowpass_hann -----------------------------------------------------------------------

#[test]
fn lowpass_hann_follows_the_specified_formula() {
    for (c, n) in [
        (100e3 / 2.4e6, 127usize),
        (15e3 / 240e3, 241),
        (0.2, 10),
        (0.25, 3),
    ] {
        let mut h = vec![0.0; n];
        lowpass_hann(c, &mut h);
        let m = (n as f64 - 1.0) / 2.0;
        let raw: Vec<f64> = (0..n)
            .map(|i| {
                let x = 2.0 * c * (i as f64 - m);
                let sinc = if x == 0.0 {
                    1.0
                } else {
                    (PI * x).sin() / (PI * x)
                };
                let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos();
                2.0 * c * sinc * w
            })
            .collect();
        let sum: f64 = raw.iter().sum();
        for (a, b) in h.iter().zip(&raw) {
            assert!((a - b / sum).abs() < 1e-14);
        }
        assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-13);
        for i in 0..n / 2 {
            assert!((h[i] - h[n - 1 - i]).abs() < 1e-15, "symmetry");
        }
    }
}

#[test]
fn lowpass_hann_response() {
    let mut h = vec![0.0; 127];
    lowpass_hann(100e3 / 2.4e6, &mut h);
    let gain_db = |f: f64| {
        let w = 2.0 * PI * f;
        let (re, im) = h.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, v)| {
            (re + v * (w * n as f64).cos(), im - v * (w * n as f64).sin())
        });
        20.0 * (re * re + im * im).sqrt().log10()
    };
    assert!(gain_db(0.0).abs() < 1e-9);
    assert!(gain_db(50e3 / 2.4e6).abs() < 0.1, "passband");
    assert!(gain_db(200e3 / 2.4e6) < -40.0, "stopband");
    assert!(gain_db(1.0e6 / 2.4e6) < -50.0);
}

#[test]
fn lowpass_hann_edge_cases() {
    let mut none: [f64; 0] = [];
    lowpass_hann(0.1, &mut none);
    let mut one = [0.0];
    lowpass_hann(0.1, &mut one);
    assert_eq!(one, [1.0]);
    let mut two = [0.0; 2];
    lowpass_hann(0.1, &mut two);
    // Hann is zero at both ends of two taps; the sum is zero and the taps stay finite.
    assert!(two.iter().all(|v| v.is_finite()));
}

// --- ToneMeter and SINAD ----------------------------------------------------------------

#[test]
fn tone_meter_recovers_amplitude_phase_and_offset() {
    let rate = 48_000.0;
    for (freq, amp, phase, dc) in [
        (1000.0, 0.5, 0.0, 0.0),
        (1000.0, 0.7, 1.0, 0.1),
        (3333.3, 0.2, -2.5, -0.05),
        (23_000.0, 1.0, 3.0, 0.3),
    ] {
        let x = cosine(4800, rate, freq, amp, phase, dc);
        let t = ToneMeter::measure(&x, rate, freq);
        assert!((t.amplitude - amp).abs() < 1e-10, "{freq}");
        assert!(
            (t.phase - phase).abs() < 1e-9,
            "{freq}: {} vs {phase}",
            t.phase
        );
        assert!((t.dc - dc).abs() < 1e-10);
        assert!(t.residual_power < 1e-20);
    }
}

#[test]
fn tone_meter_measures_known_noise_power() {
    let rate = 48_000.0;
    let mut rng = Pcg32::new(7, 7);
    let sigma = 0.01;
    let n = 24_000;
    let x: Vec<f64> = cosine(n, rate, 1000.0, 0.5, 0.3, 0.02)
        .iter()
        .map(|v| v + sigma * rng.normal())
        .collect();
    let t = ToneMeter::measure(&x, rate, 1000.0);
    // The fit removes three degrees of freedom out of N.
    let want = sigma * sigma * (1.0 - 3.0 / n as f64);
    assert!(
        (t.residual_power / want - 1.0).abs() < 0.03,
        "{}",
        t.residual_power
    );
    assert!((t.amplitude - 0.5).abs() < 1e-3);
}

#[test]
fn sinad_matches_a_known_signal_to_noise_ratio() {
    let rate = 48_000.0;
    let n = 48_000;
    for snr_db in [20.0f64, 35.0, 50.0, 70.0] {
        let amp = 0.5;
        let sigma = (amp * amp / 2.0 / 10f64.powf(snr_db / 10.0)).sqrt();
        let mut rng = Pcg32::new(snr_db as u64, 11);
        let x: Vec<f64> = cosine(n, rate, 1000.0, amp, 0.7, 0.1)
            .iter()
            .map(|v| v + sigma * rng.normal())
            .collect();
        let measured = sinad_db(&x, rate, 1000.0);
        assert!((measured - snr_db).abs() < 0.5, "{snr_db}: {measured}");
    }
}

#[test]
fn sinad_special_cases() {
    let rate = 8000.0;
    let pure = cosine(800, rate, 500.0, 1.0, 0.0, 0.0);
    assert!(sinad_db(&pure, rate, 500.0) > 250.0);
    // The offset does not count as signal.
    let with_dc = cosine(800, rate, 500.0, 1.0, 0.0, 5.0);
    assert!(sinad_db(&with_dc, rate, 500.0) > 250.0);
    // Measuring at the wrong frequency gives a bad value.
    assert!(sinad_db(&pure, rate, 700.0) < 5.0);
    // Empty and degenerate inputs do not panic.
    assert_eq!(ToneMeter::measure(&[], rate, 500.0).amplitude, 0.0);
    let t = ToneMeter::measure(&[1.0, 1.0, 1.0, 1.0], rate, 0.0);
    assert_eq!((t.amplitude, t.dc), (0.0, 1.0));
    let t = ToneMeter::measure(&[1.0, 2.0], rate, 500.0);
    assert!(t.dc.is_finite() && t.amplitude.is_finite());
}

#[test]
fn fit_frequency_finds_small_offsets() {
    let rate = 48_000.0;
    let mut rng = Pcg32::new(3, 3);
    for off in [-4.0, -0.3, 0.0, 0.07, 2.5] {
        let x: Vec<f64> = cosine(9600, rate, 1000.0 + off, 0.8, 0.4, 0.0)
            .iter()
            .map(|v| v + 0.001 * rng.normal())
            .collect();
        let f = fit_frequency(&x, rate, 1000.0, 5.0);
        assert!((f - (1000.0 + off)).abs() < 0.01, "{off}: {f}");
    }
}

// --- estimate_frequency -----------------------------------------------------------------

fn complex_tone(n: usize, rate: f64, freq: f64) -> Vec<Complex64> {
    (0..n)
        .map(|i| polar(0.7, 2.0 * PI * freq * i as f64 / rate + 0.3))
        .collect()
}

#[test]
fn estimate_frequency_is_accurate_for_positive_and_negative_tones() {
    let rate = 48_000.0;
    for freq in [1234.5, -1234.5, 0.0, 17_000.3, -23_000.0, 0.4, 100.0] {
        let x = complex_tone(1024, rate, freq);
        let f = estimate_frequency(&x, rate);
        // One bin of the zero-padded transform is rate / (4 N) = 11.7 Hz.
        assert!((f - freq).abs() < 1.5, "{freq}: {f}");
    }
    // With noise.
    let mut rng = Pcg32::new(1, 1);
    let mut x = complex_tone(2048, rate, 5000.0);
    for s in &mut x {
        *s += Complex64::new(rng.normal(), rng.normal()) * 0.1;
    }
    assert!((estimate_frequency(&x, rate) - 5000.0).abs() < 1.0);
    // Only the first ESTIMATE_MAX_SAMPLES samples count; a longer signal is accepted.
    let long = complex_tone(ESTIMATE_MAX_SAMPLES * 2, rate, 3000.0);
    assert!((estimate_frequency(&long, rate) - 3000.0).abs() < 1.0);
    assert_eq!(estimate_frequency(&[], rate), 0.0);
}

#[test]
fn estimate_frequency_finds_the_fm_carrier() {
    // Without modulation the carrier is at the offset; with modulation the strongest line is
    // within the deviation.
    let mut x = vec![Complex64::new(0.0, 0.0); 2048];
    fm_baseband(&mut x, 2.4e6, 600e3, 75e3, 1e3, 0.0, 0);
    assert!((estimate_frequency(&x, 2.4e6) - 600e3).abs() < 200.0);
    fm_baseband(&mut x, 2.4e6, 600e3, 75e3, 1e3, 1.0, 0);
    assert!((estimate_frequency(&x, 2.4e6) - 600e3).abs() < 80e3);
}

// --- spectral_mask ----------------------------------------------------------------------

#[test]
fn spectral_mask_accepts_and_rejects() {
    let rate = 16_000.0;
    // A tone at +2000 Hz (bin 128 of 1024) and a weaker one at -3000 Hz, 50 dB down.
    let n = 4 * MASK_SEGMENT;
    let x: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 / rate;
            polar(1.0, 2.0 * PI * 2000.0 * t)
                + polar(10f64.powf(-50.0 / 20.0), -2.0 * PI * 3000.0 * t)
        })
        .collect();
    // Everything outside +-100 Hz around the strong tone must be 40 dB down: the weak tone
    // is 50 dB down, so this passes.
    let ok = [(-8000.0, 1900.0, -40.0), (2100.0, 7999.0, -40.0)];
    assert_eq!(spectral_mask(&x, rate, &ok), Ok(()));
    // 60 dB down is too strict for the weak tone, and the report names its bin.
    let tight = [(-8000.0, 1900.0, -60.0), (2100.0, 7999.0, -60.0)];
    let v = spectral_mask(&x, rate, &tight).unwrap_err();
    assert!(
        (v.freq_hz + 3000.0).abs() <= rate / MASK_SEGMENT as f64,
        "{v}"
    );
    assert!(v.level_db > -60.0 && v.level_db < -45.0, "{v}");
    assert_eq!(v.limit_db, -60.0);
    assert!(format!("{v}").contains("mask"));
    // The strong tone itself violates a mask that covers it.
    assert!(spectral_mask(&x, rate, &[(1900.0, 2100.0, -10.0)]).is_err());
    // An empty mask, silence and a signal shorter than one segment are accepted or handled.
    assert_eq!(spectral_mask(&x, rate, &[]), Ok(()));
    assert_eq!(
        spectral_mask(&vec![Complex64::new(0.0, 0.0); 2048], rate, &tight),
        Ok(())
    );
    assert!(spectral_mask(&x[..100], rate, &[(-8000.0, 7999.0, -3.0)]).is_err());
    assert_eq!(spectral_mask(&[], rate, &tight), Ok(()));
}

#[test]
fn spectral_mask_uses_the_lowest_limit_of_overlapping_entries() {
    let rate = 16_000.0;
    let x: Vec<Complex64> = (0..MASK_SEGMENT)
        .map(|i| polar(1.0, 2.0 * PI * 2000.0 * i as f64 / rate))
        .collect();
    // Far from the tone both entries allow -30 dB or more; the stricter one decides.
    let far = (-8000.0, -4000.0, -20.0);
    assert_eq!(spectral_mask(&x, rate, &[far]), Ok(()));
    let overlapping = [far, (-6000.0, -5000.0, -200.0)];
    assert!(spectral_mask(&x, rate, &overlapping).is_err());
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<ToneMeter>();
    assert_send::<sdrk_testkit::meters::Tone>();
    assert_send::<sdrk_testkit::meters::MaskViolation>();
    assert_sync::<sdrk_testkit::meters::MaskViolation>();
    assert_send::<sdrk_testkit::e2e::Measurement>();
    assert_send::<sdrk_testkit::e2e::FmTx<'static, sdrk_core::C32, sdrk::Sdrk>>();
    assert_send::<sdrk_testkit::e2e::FmTx<'static, sdrk_core::CQ15, sdrk::Sdrk>>();
}
