//! M22 with the references of D15 and D16: `biquad` C32 and `biquad_real` f32 are counted
//! against KFR (CMSIS-DSP at `simd128`, where KFR is not built), `biquad` CQ15 and
//! `biquad_real` Q15 against our fallback (no SIMD Q15 biquad exists on our levels), and four
//! fallback entries count our fallback against CMSIS-DSP's generic C. The entries against a
//! reference library are `release`; the two Q15 SIMD entries are `info`, because our own fallback
//! is no fair gate for a SIMD branch (D21). The committed result files and `benches/limits.json`
//! must carry the entries on the platforms that were measured, and every committed record must
//! pass or be information.

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

/// The status of an entry and the verdict its committed record carries.
fn expected_status(kernel: &str) -> (&'static str, &'static str) {
    match kernel {
        "biquad_cq15" | "biquad_real_q15" => ("info", "info"),
        _ => ("release", "pass"),
    }
}

#[test]
fn every_biquad_entry_has_its_status_at_every_level() {
    let limits = bench_file("limits.json");
    for kernel in SIMD.into_iter().chain(FALLBACK) {
        let (status, _) = expected_status(kernel);
        for level in LEVELS {
            let entry = &limits[kernel][level];
            assert_eq!(entry["status"].as_str(), Some(status), "{kernel}/{level}");
            // An `info` entry states why it is not judged.
            if status == "info" {
                assert!(entry["waiver"].is_string(), "{kernel}/{level}");
            }
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
                // The committed record has its status and verdict (`pass` for a release
                // entry, `info` for the Q15 SIMD entries), and independently of it the ratio of
                // the two counts is within the release limit, which also guards the `info`
                // entries against a regression of the SIMD branch.
                let (status, verdict) = expected_status(kernel);
                assert_eq!(
                    entry["status"].as_str(),
                    Some(status),
                    "{platform}/{level}/{kernel}"
                );
                assert_eq!(
                    entry["verdict"].as_str(),
                    Some(verdict),
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

/// The Cortex-M4 report (open item M22-10, closed) has the four biquad kernels of our fallback
/// with CMSIS-DSP built for the M4F next to them and both SNRs on the same input, and no limit
/// or verdict (report only, like every entry of that report).
#[test]
fn the_cortex_m4_report_compares_the_biquad_fallback_with_cmsis_dsp() {
    let result = bench_file("results/cortex-m4.json");
    assert_eq!(result["status"].as_str(), Some("report only, no limits"));
    let libraries = &result["references"]["libraries"];
    assert_eq!(libraries["cmsis_dsp"]["version"].as_str(), Some("1.18.0"));
    assert_eq!(libraries["cmsis_core"]["version"].as_str(), Some("6.3.0"));
    for kernel in SIMD {
        let entry = &result["kernels"][kernel];
        assert_eq!(
            entry["implementation"].as_str(),
            Some("fallback"),
            "{kernel}"
        );
        assert_eq!(entry["samples_per_call"].as_u64(), Some(4096), "{kernel}");
        let ours = entry["instructions_per_sample"].as_f64().unwrap_or(0.0);
        let other = entry["other_instr_per_sample"].as_f64().unwrap_or(0.0);
        assert!(ours > 0.0 && other > 0.0, "cortex-m4/{kernel}: {entry}");
        let ratio = entry["ratio"].as_f64().unwrap_or(f64::NAN);
        assert!(
            (ratio - ours / other).abs() < 0.01,
            "cortex-m4/{kernel}: {entry}"
        );
        assert!(
            entry["compared_with"]
                .as_str()
                .is_some_and(|c| c.contains("CMSIS-DSP 1.18.0") && c.contains("Cortex-M4")),
            "cortex-m4/{kernel}: {entry}"
        );
        for side in ["ours", "other"] {
            assert!(
                entry["snr_db"][side].as_f64().is_some_and(|v| v > 40.0),
                "cortex-m4/{kernel}: {entry}"
            );
        }
        assert!(
            entry["status"].is_null() && entry["verdict"].is_null(),
            "{kernel}"
        );
    }
}
