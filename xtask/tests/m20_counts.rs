//! M20: `mag` is accelerated and counted against VOLK, the `C32` `agc` is counted against the
//! `agc_cc` loop of GNU Radio (information, D2768, D2769), and `agc_cq15` and `dc_block_real` are counted as information on
//! every platform (`agc_cq15` and `dc_block_real` `Q15` are not accelerated); `dc_block_real`
//! `f32`, the block scan of the SIMD backend, is judged against KFR's one-section `iir` (D2902).
//! `benches/limits.json`
//! must carry the entries at every level, the committed `simd128` counts must agree with it (the
//! other levels are measured in cycles, D2700), and the Cortex-M4 report (no limits) counts all
//! four kernels of the fallback.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

const INFO_KERNELS: [&str; 2] = ["agc_cq15", "dc_block_real_q15"];

/// The platforms with instruction counts and their levels.
const PLATFORMS: [(&str, &[&str]); 1] = [("wasm32", &["simd128"])];

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
fn dc_block_real_is_judged_against_kfr_at_every_level() {
    // D2902: KFR's `iir` with one section, a library comparison at every level.
    let limits = bench_file("limits.json");
    for level in LEVELS {
        assert_eq!(
            limits["dc_block_real"][level]["status"].as_str(),
            Some("release"),
            "dc_block_real/{level}"
        );
    }
    for (platform, levels) in PLATFORMS {
        let result = bench_file(&format!("results/{platform}.json"));
        for level in levels {
            let entry = &result["levels"][*level]["kernels"]["dc_block_real"];
            assert!(
                entry
                    .to_string()
                    .contains("KFR 7.2.0 kfr::iir, iir_state<float, 1>"),
                "{platform}/{level}/dc_block_real is not compared with KFR"
            );
        }
    }
}

#[test]
fn every_platform_counts_mag_against_volk() {
    for (platform, levels) in PLATFORMS {
        let result = bench_file(&format!("results/{platform}.json"));
        for level in levels {
            // `mag` is judged against KFR's `cabs` at `simd128`, VOLK's generic build is
            // information (D2850); `mag_q15` stays information against VOLK (D2768).
            for (kernel, volk, want) in [
                ("mag", "volk_32fc_magnitude_32f", "release"),
                ("mag_q15", "volk_16ic_magnitude_16i", "info"),
            ] {
                let entry = &result["levels"][*level]["kernels"][kernel];
                let status = entry["status"].as_str();
                assert_eq!(status, Some(want), "{platform}/{level}/{kernel}: {entry}");
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
            assert_eq!(
                entry["status"].as_str(),
                Some("info"),
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
fn the_limits_file_reports_agc_against_gnu_radio_as_information_at_every_level() {
    // GNU Radio's `agc_cc` is a scalar loop with the generic flags: no like-with-like gate of the
    // SIMD path (D2769) and no SIMD reference at `simd128` (D2768), so `agc` is `info` with the
    // efficiency verdict at every level; its fallback is judged by `agc_fallback`.
    let limits = bench_file("limits.json");
    for level in LEVELS {
        assert_eq!(
            limits["agc"][level]["status"].as_str(),
            Some("info"),
            "{level}"
        );
        assert!(
            limits["agc"][level]["waiver"]
                .as_str()
                .is_some_and(|w| w.contains("D2768") || w.contains("D2769")),
            "{level}"
        );
    }
}

#[test]
fn the_limits_file_lists_the_kernels_without_a_reference_as_information_at_every_level() {
    let limits = bench_file("limits.json");
    for kernel in INFO_KERNELS {
        for level in LEVELS {
            let status = &limits[kernel][level]["status"];
            assert_eq!(status.as_str(), Some("info"), "{kernel}/{level}: {status}");
            assert!(
                limits[kernel][level]["waiver"]
                    .as_str()
                    .is_some_and(|w| w.contains("not accelerated") || w.contains("block scan")),
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
    for level in ["avx512", "neon"] {
        assert_eq!(limits["mag"][level]["status"].as_str(), Some("release"));
        assert_eq!(limits["mag_q15"][level]["status"].as_str(), Some("release"));
    }
    // VOLK's wasm32 build has only its `generic` protokernels (D2768): `mag` is judged
    // against KFR's `cabs` there (D2850), `mag_q15` is information.
    assert_eq!(limits["mag"]["simd128"]["status"].as_str(), Some("release"));
    assert_eq!(
        limits["mag_q15"]["simd128"]["status"].as_str(),
        Some("info")
    );
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
