//! Loop tests of M25: the transmitter flows into the receiver flows without a channel (the
//! `cs8` bytes of `pull` are pushed into the receiver). They check the frequency response of
//! the pre-emphasis (TX-FLAT), the peak deviation of the limiter (TX-PEAK) and the narrow-band
//! chain (NFM), in `f32` and Q15 precision.
#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::cast_precision_loss
)]

use sdrk_device::formats;
use sdrk_pipeline::{FormatName, Precision};
use sdrk_system_tests::{
    AUDIO_RATE, band_limited_noise, build, flow_value, peak_deviation, receive_bytes, remove_node,
    set_input_format, set_precision, spec_from_value, transmit_bytes,
};
use sdrk_testkit::meters::{ToneMeter, sinad_db};
use serde_json::json;

/// Seconds of one tone of TX-FLAT.
const TONE_SECONDS: f64 = 0.3;
/// Frequencies of TX-FLAT in hertz.
const FLAT_TONES: [f64; 5] = [100.0, 1000.0, 5000.0, 10_000.0, 12_000.0];
/// Seed of the noise of TX-PEAK and of the NFM peak test (milestone M25, section 6).
const NOISE_SEED: u64 = 450;
/// The largest allowed instantaneous frequency of TX-PEAK (75 kHz plus 2 percent).
const PEAK_LIMIT: f64 = 1.02;

fn db(a: f64, b: f64) -> f64 {
    20.0 * (a / b).log10()
}

/// The tones one after the other: `TONE_SECONDS` each, amplitude 0.1.
fn flat_audio() -> Vec<f32> {
    let per = (AUDIO_RATE * TONE_SECONDS) as usize;
    let mut audio = Vec::with_capacity(per * FLAT_TONES.len());
    for f in FLAT_TONES {
        for n in 0..per {
            let t = n as f64 * f / AUDIO_RATE;
            audio.push((0.1 * (2.0 * std::f64::consts::PI * (t - t.floor())).sin()) as f32);
        }
    }
    audio
}

/// Tone levels (amplitudes) of the received audio, one per tone of `FLAT_TONES`.
fn flat_levels(tx: &mut serde_json::Value, precision: Precision) -> Vec<f64> {
    set_precision(tx, precision);
    let bytes = transmit_bytes(&spec_from_value(tx), flat_audio(), precision);
    let mut rx = flow_value("fm_receiver");
    rx["settings"]["precision"] = json!(if precision == Precision::Q15 {
        "q15"
    } else {
        "f32"
    });
    let mut spec = spec_from_value(&rx);
    set_input_format(&mut spec, "rf", FormatName::Cs8);
    let mut p = build(&spec);
    p.set_param("tune", "frequency_hz", json!(600_000)).unwrap();
    let audio = receive_bytes(&mut p, &bytes, precision);
    let per = (AUDIO_RATE * TONE_SECONDS) as usize;
    assert!(
        audio.len() >= per * FLAT_TONES.len() - 64,
        "{}",
        audio.len()
    );
    FLAT_TONES
        .iter()
        .enumerate()
        .map(|(i, f)| {
            // Skip the settling of the filters (and of the previous tone), measure 0.15 s.
            let seg: Vec<f64> = audio[i * per + 4800..i * per + 12_000]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            ToneMeter::measure(&seg, AUDIO_RATE, *f).amplitude
        })
        .collect()
}

fn loop_fm_flat(precision: Precision, tolerance_db: f64) {
    let mut tx = flow_value("fm_transmitter");
    let levels = flat_levels(&mut tx, precision);
    let reference = levels[1];
    let deviations: Vec<f64> = levels.iter().map(|l| db(*l, reference)).collect();
    println!("{precision:?} TX-FLAT levels re 1 kHz in dB: {deviations:.2?}");
    for (f, d) in FLAT_TONES.iter().zip(&deviations) {
        assert!(
            d.abs() <= tolerance_db,
            "{precision:?}: {f} Hz is {d:.2} dB from the 1 kHz level (allowed {tolerance_db} dB)"
        );
    }
    // The same test without the shape node fails: the treble arrives about 10 dB low.
    let mut plain = flow_value("fm_transmitter");
    remove_node(&mut plain, "shape");
    let levels = flat_levels(&mut plain, precision);
    let without: Vec<f64> = levels.iter().map(|l| db(*l, levels[1])).collect();
    println!("{precision:?} TX-FLAT without shape: {without:.2?}");
    assert!(
        without.iter().any(|d| d.abs() > tolerance_db),
        "{precision:?}: the test must fail without the shape node: {without:.2?}"
    );
    assert!(
        without[3] < -5.0,
        "10 kHz without shape: {:.2} dB",
        without[3]
    );
}

#[test]
fn loop_fm_flat_f32() {
    loop_fm_flat(Precision::F32, 0.5);
}

#[test]
fn loop_fm_flat_q15() {
    loop_fm_flat(Precision::Q15, 1.0);
}

fn noise() -> Vec<f32> {
    band_limited_noise(NOISE_SEED, 48_000, AUDIO_RATE, 15_000.0, 3.0)
}

fn loop_fm_peak(precision: Precision) {
    let mut tx = flow_value("fm_transmitter");
    set_precision(&mut tx, precision);
    let bytes = transmit_bytes(&spec_from_value(&tx), noise(), precision);
    let peak = peak_deviation(&bytes, 600_000.0, 100_000.0, 40_000.0, 75_000.0, 2000);
    println!("{precision:?} TX-PEAK with shape: {peak:.4}");
    assert!(peak <= PEAK_LIMIT, "{precision:?}: peak {peak:.4}");
    assert!(
        peak > 0.9,
        "{precision:?}: the limiter must be working, peak {peak:.4}"
    );

    let mut plain = flow_value("fm_transmitter");
    remove_node(&mut plain, "shape");
    set_precision(&mut plain, precision);
    let bytes = transmit_bytes(&spec_from_value(&plain), noise(), precision);
    let without = peak_deviation(&bytes, 600_000.0, 100_000.0, 40_000.0, 75_000.0, 2000);
    println!("{precision:?} TX-PEAK without shape: {without:.4}");
    if precision == Precision::F32 {
        assert!(without > 1.5, "{precision:?}: without shape {without:.4}");
    } else {
        // The Q15 audio input saturates at full scale, so without the shape node the deviation
        // cannot exceed 1.0 plus the overshoot of the filters: the control of the milestone
        // (more than 1.5) cannot hold in Q15. The measured value is recorded (1.043) and the
        // threshold is not changed for f32; open item I25-2, decision D183.
        assert!(
            (0.9..1.1).contains(&without),
            "{precision:?}: without shape {without:.4}"
        );
    }
}

#[test]
fn loop_fm_peak_f32() {
    loop_fm_peak(Precision::F32);
}

#[test]
fn loop_fm_peak_q15() {
    loop_fm_peak(Precision::Q15);
}

fn loop_nfm(precision: Precision, min_sinad_db: f64) {
    let mut tx = flow_value("nfm_transmitter");
    set_precision(&mut tx, precision);
    let tx = spec_from_value(&tx);
    // A 1 kHz tone at amplitude 0.3 (0.5 s) through the receiver.
    let n = (AUDIO_RATE * 0.5) as usize;
    let tone: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 * 1000.0 / AUDIO_RATE;
            (0.3 * (2.0 * std::f64::consts::PI * (t - t.floor())).sin()) as f32
        })
        .collect();
    let bytes = transmit_bytes(&tx, tone, precision);
    let mut rx = flow_value("nfm_receiver");
    rx["settings"]["precision"] = json!(if precision == Precision::Q15 {
        "q15"
    } else {
        "f32"
    });
    let mut spec = spec_from_value(&rx);
    set_input_format(&mut spec, "rf", FormatName::Cs8);
    let mut p = build(&spec);
    p.set_param("tune", "frequency_hz", json!(200_000)).unwrap();
    let audio = receive_bytes(&mut p, &bytes, precision);
    // Skip the opening of the squelch and the settling of the filters.
    let seg: Vec<f64> = audio[audio.len() / 2..]
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    assert!(seg.iter().any(|v| *v != 0.0), "the squelch stayed closed");
    let sinad = sinad_db(&seg, AUDIO_RATE, 1000.0);
    println!("{precision:?} NFM SINAD {sinad:.1} dB");
    assert!(sinad >= min_sinad_db, "{precision:?}: SINAD {sinad:.1} dB");

    // The noise of TX-PEAK: the deviation stays below 1.0 x 5 kHz.
    let bytes = transmit_bytes(&tx, noise(), precision);
    let peak = peak_deviation(&bytes, 200_000.0, 25_000.0, 15_000.0, 5_000.0, 2000);
    println!("{precision:?} NFM peak deviation {peak:.4}");
    assert!(peak < 1.0, "{precision:?}: NFM peak {peak:.4}");
    let _ = formats::CS8;
}

#[test]
fn loop_nfm_f32() {
    loop_nfm(Precision::F32, 30.0);
}

#[test]
fn loop_nfm_q15() {
    loop_nfm(Precision::Q15, 25.0);
}
