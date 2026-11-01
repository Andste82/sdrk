//! M25: `preemph_limit` is not accelerated (both recursions run per sample). Both variants
//! (`f32` and `Q15`) are counted through the facade, which is the fallback, against our own
//! fallback as information at every level, including `avx2-runtime`; no library has a peak
//! limiter of this kind. The committed result files must agree with `benches/limits.json`.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

const KERNELS: [&str; 2] = ["preemph_limit", "preemph_limit_q15"];

const LEVELS: [&str; 7] = [
    "sse2",
    "sse4.2",
    "avx2",
    "avx2-runtime",
    "avx512",
    "neon",
    "simd128",
];

fn bench_file(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("benches")
        .join(name);
    let text = fs::read_to_string(&path).expect("read the bench file");
    serde_json::from_str(&text).expect("parse the bench file")
}

#[test]
fn the_limits_file_lists_preemph_limit_as_information_at_every_level() {
    let limits = bench_file("limits.json");
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

#[test]
fn every_platform_reports_preemph_limit_against_the_fallback() {
    let expected = [
        ("x86_64", &["sse2", "sse4.2", "avx2", "avx2-runtime"][..]),
        ("x86_64-avx512", &["avx512"][..]),
        ("aarch64", &["neon"][..]),
        ("wasm32", &["simd128"][..]),
    ];
    for (platform, levels) in expected {
        let result = bench_file(&format!("results/{platform}.json"));
        for level in levels {
            for kernel in KERNELS {
                let entry = &result["levels"][*level]["kernels"][kernel];
                assert_eq!(
                    entry["status"].as_str(),
                    Some("info"),
                    "{platform}/{level}/{kernel}: {entry}"
                );
                for side in ["ours_instr_per_sample", "other_instr_per_sample"] {
                    assert!(
                        entry[side].as_f64().is_some_and(|v| v > 0.0),
                        "{platform}/{level}/{kernel}: {entry}"
                    );
                }
                // `avx2-runtime` is judged against the fixed `avx2` build of ours.
                let want = if *level == "avx2-runtime" {
                    "ours at avx2"
                } else {
                    "sdrk fallback"
                };
                assert_eq!(
                    entry["compared_with"].as_str(),
                    Some(want),
                    "{platform}/{level}/{kernel}"
                );
            }
        }
    }
}
