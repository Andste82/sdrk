//! Shared helpers of the integration tests: the receive signal, the meter, SHA-256 and the
//! fixture.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::fmt::Write;

use num_complex::Complex64;
use sdrk_device::formats;
use sdrk_testkit::e2e::{CHUNK, fm_source};
use sdrk_testkit::meters::sinad_db;

/// The receiver flow of the examples.
pub const FM_RECEIVER: &str = include_str!("../../../../examples/flows/fm_receiver.json");
/// The committed reproducibility fixture.
pub const FIXTURE: &str = include_str!("../fixtures/fm_receiver.json");
/// Seconds of signal of the receive test.
pub const SIGNAL_SECONDS: f64 = 0.25;
/// Audio sample rate in Hz.
pub const AUDIO_RATE: f64 = 48_000.0;
/// Audio samples skipped before the SINAD measurement (the filters settle).
pub const SINAD_SKIP: usize = 1200;
/// Samples pushed per call.
pub const PUSH_SAMPLES: usize = 4096;

/// The E2E-RX signal as CU8 bytes: an FM broadcast at +600 kHz with a 1 kHz tone, 2.4 MS/s.
pub fn fm_rx_cu8() -> Vec<u8> {
    let samples = (2_400_000.0 * SIGNAL_SECONDS) as usize;
    assert_eq!(samples % CHUNK, 0);
    let mut bytes = vec![0u8; 2 * samples];
    let mut scratch = vec![Complex64::new(0.0, 0.0); CHUNK];
    let mut source = fm_source(&mut scratch, formats::CU8);
    for (i, block) in bytes.chunks_exact_mut(2 * CHUNK).enumerate() {
        source((i * CHUNK) as u64, block).unwrap();
    }
    bytes
}

/// SINAD of the 1 kHz tone in the audio, in dB.
pub fn audio_sinad(audio: &[f32]) -> f64 {
    let x: Vec<f64> = audio
        .iter()
        .skip(SINAD_SKIP)
        .map(|&v| f64::from(v))
        .collect();
    sinad_db(&x, AUDIO_RATE, 1000.0)
}

/// The audio samples as little-endian bytes.
pub fn f32_bytes(audio: &[f32]) -> Vec<u8> {
    audio.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The values of the fixture.
pub struct Fixture {
    pub input_sha256: String,
    pub audio_samples: usize,
    pub audio_sha256: String,
    pub sinad_db: f64,
}

pub fn fixture() -> Fixture {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    Fixture {
        input_sha256: v["input_sha256"].as_str().unwrap().to_string(),
        audio_samples: v["audio_samples"].as_u64().unwrap() as usize,
        audio_sha256: v["audio_sha256"].as_str().unwrap().to_string(),
        sinad_db: v["sinad_db"].as_f64().unwrap(),
    }
}

// ---------------------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), so the tests need no extra dependency.

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// SHA-256 of `data` as lowercase hex.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    h.iter().fold(String::new(), |mut hex, x| {
        write!(hex, "{x:08x}").unwrap();
        hex
    })
}
