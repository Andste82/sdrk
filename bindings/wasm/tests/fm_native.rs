//! The receive path of the bindings on the host: the same engine the JavaScript API calls.
//!
//! The test `fixture_matches_the_fallback_build` runs only in the CI job that builds with
//! `RUSTFLAGS="--cfg sdrk_fallback_only"`; it pins the audio of the fallback to the committed
//! `tests/fixtures/fm_receiver.json`, which the scalar WASM build must reproduce bit for bit.
//! `SDRK_UPDATE_FIXTURE=1` rewrites the fixture instead of comparing.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::missing_panics_doc
)]

mod common;

use common::{
    FM_RECEIVER, PUSH_SAMPLES, SIGNAL_SECONDS, audio_sinad, fixture, fm_rx_cu8, sha256_hex,
};
use sdrk_wasm::engine::{Engine, Lanes};

/// Pushes the CU8 signal through `fm_receiver.json` and returns the audio.
fn receive(cu8: &[u8]) -> Vec<f32> {
    let mut engine = Engine::from_json(FM_RECEIVER).unwrap();
    let mut audio = Vec::new();
    for chunk in cu8.chunks(2 * PUSH_SAMPLES) {
        engine.push_u8("rf", chunk, None).unwrap();
        while let Lanes::F32(samples) = engine.read_chunk("audio", 65_536).unwrap() {
            if samples.is_empty() {
                break;
            }
            audio.extend_from_slice(samples);
        }
    }
    audio
}

#[test]
fn sha256_matches_the_standard_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(&[b'a'; 1000]),
        "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
    );
}

#[test]
fn the_receive_signal_is_the_one_of_the_fixture() {
    let input = fm_rx_cu8();
    assert_eq!(sha256_hex(&input), fixture().input_sha256);
}

#[test]
fn fm_receive_meets_the_sinad_requirement() {
    let audio = receive(&fm_rx_cu8());
    let sinad = audio_sinad(&audio);
    assert!(sinad >= 40.0, "SINAD {sinad} dB");
    assert!(audio.len() > (SIGNAL_SECONDS * 48_000.0) as usize - 100);
}

#[cfg(not(sdrk_fallback_only))]
#[test]
fn the_accelerated_backend_stays_within_half_a_db_of_the_fixture() {
    let audio = receive(&fm_rx_cu8());
    let sinad = audio_sinad(&audio);
    let reference = fixture().sinad_db;
    assert!(
        sinad >= reference - 0.5,
        "SINAD {sinad} dB against the fixture {reference} dB"
    );
}

#[cfg(sdrk_fallback_only)]
#[test]
fn fixture_matches_the_fallback_build() {
    use common::f32_bytes;

    let input = fm_rx_cu8();
    let audio = receive(&input);
    let measured = audio_sinad(&audio);
    if std::env::var_os("SDRK_UPDATE_FIXTURE").is_some() {
        let text = format!(
            "{{\n  \"description\": \"Audio of examples/flows/fm_receiver.json for the E2E-RX signal ({SIGNAL_SECONDS} s, CU8, 2.4 MS/s) on the fallback backend. Written by tests/fm_native.rs with SDRK_UPDATE_FIXTURE=1 and RUSTFLAGS=--cfg sdrk_fallback_only; the scalar WASM build must reproduce audio_sha256 bit for bit.\",\n  \"input_sha256\": \"{}\",\n  \"audio_samples\": {},\n  \"audio_sha256\": \"{}\",\n  \"sinad_db\": {measured:.3}\n}}\n",
            sha256_hex(&input),
            audio.len(),
            sha256_hex(&f32_bytes(&audio)),
        );
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("fm_receiver.json");
        std::fs::write(path, text).unwrap();
        return;
    }
    let fixture = fixture();
    assert_eq!(sha256_hex(&input), fixture.input_sha256, "the input signal");
    assert_eq!(audio.len(), fixture.audio_samples);
    assert_eq!(
        sha256_hex(&f32_bytes(&audio)),
        fixture.audio_sha256,
        "the audio"
    );
    assert!((measured - fixture.sinad_db).abs() < 0.001, "{measured}");
}
