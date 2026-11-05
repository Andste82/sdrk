//! M33: `noise_blank` and `auto_notch` are not accelerated (per-sample recursions). The three
//! entries (`noise_blank` `C32`, `noise_blank_cq15` and `auto_notch` `f32`) are counted through
//! the facade, which is the fallback, against our own fallback as information at every level,
//! including `avx2-runtime`.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

const KERNELS: [&str; 3] = ["noise_blank", "noise_blank_cq15", "auto_notch"];

const LEVELS: [&str; 7] = [
    "sse2",
    "sse4.2",
    "avx2",
    "avx2-runtime",
    "avx512",
    "neon",
    "simd128",
];

fn limits() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("benches")
        .join("limits.json");
    let text = fs::read_to_string(&path).expect("read the limits file");
    serde_json::from_str(&text).expect("parse the limits file")
}

#[test]
fn the_limits_file_lists_the_noise_reduction_kernels_as_information_at_every_level() {
    let limits = limits();
    for kernel in KERNELS {
        for level in LEVELS {
            let entry = &limits[kernel][level];
            assert_eq!(entry["status"].as_str(), Some("info"), "{kernel}/{level}");
            assert!(
                entry["waiver"]
                    .as_str()
                    .is_some_and(|w| w.contains("not accelerated")),
                "{kernel}/{level} has no waiver text"
            );
        }
    }
}
