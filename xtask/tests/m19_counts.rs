//! M19: `squelch` is not accelerated and is counted as information on every platform. The
//! committed result files and `benches/limits.json` must carry an `info` entry for `squelch`
//! and `squelch_cq15` at every level.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

const KERNELS: [&str; 2] = ["squelch", "squelch_cq15"];

fn bench_file(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("benches")
        .join(name);
    let text = fs::read_to_string(&path).expect("read the bench file");
    serde_json::from_str(&text).expect("parse the bench file")
}

#[test]
fn every_platform_reports_squelch_as_information() {
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
                let status = &result["levels"][*level]["kernels"][kernel]["status"];
                assert_eq!(
                    status.as_str(),
                    Some("info"),
                    "{platform}/{level}/{kernel}: {status}"
                );
            }
        }
    }
}

#[test]
fn the_limits_file_lists_squelch_as_information_at_every_level() {
    let limits = bench_file("limits.json");
    for kernel in KERNELS {
        for level in [
            "sse2",
            "sse4.2",
            "avx2",
            "avx2-runtime",
            "avx512",
            "neon",
            "simd128",
        ] {
            let status = &limits[kernel][level]["status"];
            assert_eq!(status.as_str(), Some("info"), "{kernel}/{level}: {status}");
            assert!(
                limits[kernel][level]["waiver"]
                    .as_str()
                    .is_some_and(|w| w.contains("not accelerated")),
                "{kernel}/{level} has no waiver text"
            );
        }
    }
}
