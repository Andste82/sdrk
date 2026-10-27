//! M22: `biquad` (`C32`, `CQ15`) has the two-lane path of the SIMD backend and is counted
//! against our fallback; `biquad_real` (`f32`, `Q15`) is a recursion per section, is not
//! accelerated and is counted as information. The committed result files and
//! `benches/limits.json` must carry the entries on the platforms that were measured.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

const ACCELERATED: [&str; 2] = ["biquad", "biquad_cq15"];
const INFO_KERNELS: [&str; 2] = ["biquad_real", "biquad_real_q15"];
const LEVELS: [&str; 6] = ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"];

fn bench_file(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("benches")
        .join(name);
    let text = fs::read_to_string(&path).expect("read the bench file");
    serde_json::from_str(&text).expect("parse the bench file")
}

#[test]
fn the_limits_file_has_an_entry_per_level_for_every_biquad_kernel() {
    let limits = bench_file("limits.json");
    for kernel in ACCELERATED.into_iter().chain(INFO_KERNELS) {
        for level in LEVELS {
            let status = limits[kernel][level]["status"].as_str();
            assert!(
                matches!(status, Some("dev" | "info" | "release")),
                "{kernel}/{level}: {status:?}"
            );
        }
    }
}

#[test]
fn the_unaccelerated_biquad_kernels_are_information_at_every_level() {
    let limits = bench_file("limits.json");
    for kernel in INFO_KERNELS {
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
fn a_level_without_the_two_lane_path_is_information_and_names_the_numbers() {
    let limits = bench_file("limits.json");
    for kernel in ACCELERATED {
        for level in LEVELS {
            let entry = &limits[kernel][level];
            if entry["status"].as_str() == Some("info") {
                assert!(
                    entry["waiver"]
                        .as_str()
                        .is_some_and(|w| w.contains("fallback")),
                    "{kernel}/{level} is information without a reason"
                );
            }
        }
    }
}

#[test]
fn the_measured_platforms_report_every_biquad_kernel() {
    for (platform, levels) in [
        ("x86_64", &["sse2", "sse4.2", "avx2"][..]),
        ("x86_64-avx512", &["avx512"][..]),
        ("aarch64", &["neon"][..]),
        ("wasm32", &["simd128"][..]),
    ] {
        let result = bench_file(&format!("results/{platform}.json"));
        for level in levels {
            for kernel in ACCELERATED.into_iter().chain(INFO_KERNELS) {
                let entry = &result["levels"][*level]["kernels"][kernel];
                assert!(
                    entry["ours_instr_per_sample"]
                        .as_f64()
                        .is_some_and(|v| v > 0.0),
                    "{platform}/{level}/{kernel}: {entry}"
                );
                assert!(
                    entry["compared_with"]
                        .as_str()
                        .is_some_and(|c| c.contains("fallback")),
                    "{platform}/{level}/{kernel} is not compared with the fallback"
                );
            }
        }
    }
}
