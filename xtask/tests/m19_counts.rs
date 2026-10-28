//! M19: `squelch` is not accelerated. The `C32` variant is gated against the `pwr_squelch_cc`
//! loop of GNU Radio (`release` at every level but `avx2-runtime`, which runs the baseline
//! build of the same fallback and stays `info`); `squelch_cq15` has no counterpart and is
//! counted as information at every level. The committed result files must agree with
//! `benches/limits.json`.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

const KERNELS: [&str; 2] = ["squelch", "squelch_cq15"];

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

/// The status `benches/limits.json` gives an entry.
fn expected_status(kernel: &str, level: &str) -> &'static str {
    if kernel == "squelch" && level != "avx2-runtime" {
        "release"
    } else {
        "info"
    }
}

#[test]
fn every_platform_reports_squelch_with_the_status_of_the_limits() {
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
                let status = entry["status"].as_str();
                // Result files written before GNU Radio's loop became the reference of the
                // `C32` entry compare with our fallback as information.
                let before_reference = status == Some("info")
                    && entry["compared_with"].as_str() == Some("sdrk fallback");
                assert!(
                    status == Some(expected_status(kernel, level)) || before_reference,
                    "{platform}/{level}/{kernel}: {status:?}"
                );
            }
        }
    }
}

#[test]
fn the_limits_file_gates_squelch_c32_against_gnu_radio() {
    let limits = bench_file("limits.json");
    for kernel in KERNELS {
        for level in LEVELS {
            let entry = &limits[kernel][level];
            let status = entry["status"].as_str();
            assert_eq!(
                status,
                Some(expected_status(kernel, level)),
                "{kernel}/{level}: {status:?}"
            );
            if status == Some("info") {
                assert!(
                    entry["waiver"]
                        .as_str()
                        .is_some_and(|w| w.contains("not accelerated")),
                    "{kernel}/{level} has no waiver text"
                );
            }
        }
    }
}
