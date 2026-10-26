//! M20: `mag` is accelerated and counted against VOLK, the `C32` `agc` is counted against the
//! `agc_cc` loop of GNU Radio, and `agc_cq15` and `dc_block_real` are not accelerated and
//! counted as information on every platform. The committed result files and
//! `benches/limits.json` must carry the entries at every level, and the Cortex-M4 report (no
//! limits) counts all four kernels of the fallback.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

const INFO_KERNELS: [&str; 3] = ["agc_cq15", "dc_block_real", "dc_block_real_q15"];

const PLATFORMS: [(&str, &[&str]); 4] = [
    ("x86_64", &["sse2", "sse4.2", "avx2", "avx2-runtime"]),
    ("x86_64-avx512", &["avx512"]),
    ("aarch64", &["neon"]),
    ("wasm32", &["simd128"]),
];

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
fn every_platform_reports_the_unaccelerated_kernels_as_information() {
    for (platform, levels) in PLATFORMS {
        let result = bench_file(&format!("results/{platform}.json"));
        for level in levels {
            for kernel in INFO_KERNELS {
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
fn every_platform_counts_mag_against_volk() {
    for (platform, levels) in PLATFORMS {
        let result = bench_file(&format!("results/{platform}.json"));
        for level in levels {
            for (kernel, volk) in [
                ("mag", "volk_32fc_magnitude_32f"),
                ("mag_q15", "volk_16ic_magnitude_16i"),
            ] {
                let entry = &result["levels"][*level]["kernels"][kernel];
                let status = entry["status"].as_str();
                if *level == "avx2-runtime" {
                    // Judged against our own numbers at `avx2`.
                    assert_eq!(status, Some("release"), "{platform}/{level}/{kernel}");
                    continue;
                }
                assert_eq!(
                    status,
                    Some("release"),
                    "{platform}/{level}/{kernel}: {entry}"
                );
                assert!(
                    entry.to_string().contains(volk),
                    "{platform}/{level}/{kernel} is not compared with {volk}"
                );
            }
        }
    }
}

#[test]
fn every_platform_counts_agc_against_gnu_radio() {
    for (platform, levels) in PLATFORMS {
        let result = bench_file(&format!("results/{platform}.json"));
        for level in levels {
            let entry = &result["levels"][*level]["kernels"]["agc"];
            if *level == "avx2-runtime" {
                // Judged against our own numbers at `avx2` (the backend detects the level).
                assert_eq!(
                    entry["status"].as_str(),
                    Some("release"),
                    "{platform}/{level}"
                );
                continue;
            }
            assert_eq!(
                entry["status"].as_str(),
                Some("release"),
                "{platform}/{level}: {entry}"
            );
            assert!(
                entry.to_string().contains("GNU Radio agc_cc"),
                "{platform}/{level}/agc is not compared with agc_cc"
            );
        }
    }
}

#[test]
fn the_limits_file_judges_agc_against_gnu_radio_at_every_level() {
    let limits = bench_file("limits.json");
    for level in ["sse2", "sse4.2", "avx2", "avx512", "neon", "simd128"] {
        assert_eq!(limits["agc"][level]["status"].as_str(), Some("release"));
    }
    // The run-time detection is judged with the default limit against `avx2`.
    assert!(limits["agc"]["avx2-runtime"].is_null());
}

#[test]
fn the_limits_file_lists_the_unaccelerated_kernels_as_information_at_every_level() {
    let limits = bench_file("limits.json");
    for kernel in INFO_KERNELS {
        for level in LEVELS {
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

#[test]
fn the_limits_file_judges_mag_at_every_level() {
    let limits = bench_file("limits.json");
    for level in &LEVELS[..4] {
        if *level == "avx2-runtime" {
            continue;
        }
        assert_eq!(limits["mag"][*level]["status"].as_str(), Some("release"));
        assert_eq!(
            limits["mag_q15"][*level]["status"].as_str(),
            Some("release")
        );
    }
    for level in ["avx512", "neon", "simd128"] {
        assert_eq!(limits["mag"][level]["status"].as_str(), Some("release"));
        assert_eq!(limits["mag_q15"][level]["status"].as_str(), Some("release"));
    }
}

#[test]
fn the_cortex_m4_report_counts_the_m20_kernels_of_the_fallback() {
    let result = bench_file("results/cortex-m4.json");
    for kernel in ["agc", "agc_cq15", "dc_block_real", "dc_block_real_q15"] {
        let entry = &result["kernels"][kernel];
        assert_eq!(
            entry["implementation"].as_str(),
            Some("fallback"),
            "cortex-m4/{kernel}: {entry}"
        );
        assert!(
            entry["instructions_per_sample"]
                .as_f64()
                .is_some_and(|v| v > 0.0),
            "cortex-m4/{kernel} has no count"
        );
    }
}
