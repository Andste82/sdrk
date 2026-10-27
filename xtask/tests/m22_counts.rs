//! M22 with the references of D15 and D16: `biquad` C32 and `biquad_real` f32 are counted
//! against KFR (CMSIS-DSP at `simd128`, where KFR is not built), `biquad` CQ15 and
//! `biquad_real` Q15 against our fallback (no SIMD Q15 biquad exists on our levels), and four
//! fallback entries count our fallback against CMSIS-DSP's generic C. Every entry is `release`.
//! The committed result files and `benches/limits.json` must carry the entries on the platforms
//! that were measured, and every committed record must pass.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

const SIMD: [&str; 4] = ["biquad", "biquad_cq15", "biquad_real", "biquad_real_q15"];
const FALLBACK: [&str; 4] = [
    "biquad_fallback",
    "biquad_cq15_fallback",
    "biquad_real_fallback",
    "biquad_real_q15_fallback",
];
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
fn every_biquad_entry_is_release_at_every_level() {
    let limits = bench_file("limits.json");
    for kernel in SIMD.into_iter().chain(FALLBACK) {
        for level in LEVELS {
            let status = limits[kernel][level]["status"].as_str();
            assert_eq!(status, Some("release"), "{kernel}/{level}");
        }
        // avx2-runtime follows the run-time rule (no entry); the fallback entries are not
        // measured there.
        assert!(limits[kernel]["avx2-runtime"].is_null(), "{kernel}");
    }
}

/// The comparison an entry is judged against at a level.
fn expected(kernel: &str, level: &str) -> &'static str {
    match kernel {
        "biquad" | "biquad_real" if level == "simd128" => "CMSIS-DSP",
        "biquad" | "biquad_real" => "KFR",
        "biquad_cq15" | "biquad_real_q15" => "fallback",
        _ => "CMSIS-DSP",
    }
}

#[test]
fn the_measured_platforms_report_every_biquad_entry() {
    let limits = bench_file("limits.json");
    let release = limits["_limits"]["release"]
        .as_f64()
        .expect("the release limit");
    for (platform, levels) in [
        ("x86_64", &["sse2", "sse4.2", "avx2"][..]),
        ("x86_64-avx512", &["avx512"][..]),
        ("aarch64", &["neon"][..]),
        ("wasm32", &["simd128"][..]),
    ] {
        let result = bench_file(&format!("results/{platform}.json"));
        for level in levels {
            for kernel in SIMD.into_iter().chain(FALLBACK) {
                let entry = &result["levels"][*level]["kernels"][kernel];
                for side in ["ours_instr_per_sample", "other_instr_per_sample"] {
                    assert!(
                        entry[side].as_f64().is_some_and(|v| v > 0.0),
                        "{platform}/{level}/{kernel}: {entry}"
                    );
                }
                let want = expected(kernel, level);
                assert!(
                    entry["compared_with"]
                        .as_str()
                        .is_some_and(|c| c.contains(want)),
                    "{platform}/{level}/{kernel} is not compared with {want}"
                );
                // The committed record passes as a release entry: the verdict of
                // `cargo xtask count` and, independently of it, the ratio of the two counts
                // within the release limit.
                assert_eq!(
                    entry["status"].as_str(),
                    Some("release"),
                    "{platform}/{level}/{kernel}"
                );
                assert_eq!(
                    entry["verdict"].as_str(),
                    Some("pass"),
                    "{platform}/{level}/{kernel}: {entry}"
                );
                let ours = entry["ours_instr_per_sample"].as_f64().unwrap_or(f64::NAN);
                let other = entry["other_instr_per_sample"].as_f64().unwrap_or(f64::NAN);
                assert!(
                    ours <= release * other,
                    "{platform}/{level}/{kernel}: {ours} > {release} x {other}"
                );
            }
        }
    }
}
