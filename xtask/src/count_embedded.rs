//! `cargo xtask count --platform cortex-m4`: instructions per sample of the portable fallback on
//! a Cortex-M4F, counted in `qemu-system-arm` with the `insn` plugin. The numbers are reported
//! to `benches/results/cortex-m4.json`; there are no limits yet (the comparison with CMSIS-DSP
//! comes with the `arm-dsp` backend).
//!
//! The method is the one of the other platforms: the workload binary
//! (`targets/qemu-cortex-m4/src/bin/count.rs`) runs a kernel once as a warm-up and then `k`
//! times; the instructions of `k = 16` minus those of `k = 0`, divided by the samples of the 16
//! calls, are the instructions per sample.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::large_stack_frames,
    reason = "xtask is a host tool; std::process::Command is large"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, anyhow, bail};
use serde_json::json;

use crate::count::{K, build_insn_plugin, parse, round2};
use crate::emu::{Tool, find_tool};

/// Samples per call of the workload binary (a kernel of `n` samples, or an FFT of `n` points).
pub const SAMPLES_PER_CALL: u64 = 1024;

/// The kernels the workload binary knows, in report order.
pub const KERNELS: [&str; 11] = [
    "convert_cs8",
    "convert_cs16",
    "convert_out_cs16",
    "rotate",
    "fir",
    "fir_decimate",
    "mag_sq",
    "quad_demod",
    "fm_mod",
    "fft_1024",
    "fft_cq15_1024",
];

/// Instructions per sample from the counts of the runs with `k = 0` and `k = calls`.
///
/// # Errors
/// The longer run counted fewer instructions than the shorter one.
pub fn per_sample(base: u64, full: u64, calls: u64, samples: u64) -> anyhow::Result<f64> {
    let diff = full
        .checked_sub(base)
        .ok_or_else(|| anyhow!("the run with k = {calls} counted fewer instructions than k = 0"))?;
    Ok(diff as f64 / (calls * samples) as f64)
}

fn build(root: &Path) -> anyhow::Result<PathBuf> {
    let dir = root.join("targets").join("qemu-cortex-m4");
    let build_dir = root
        .join("target")
        .join("emu")
        .join("build")
        .join("qemu-cortex-m4");
    println!("==> building the workload of targets/qemu-cortex-m4");
    let status = Command::new("cargo")
        .current_dir(&dir)
        .args([
            "build",
            "--profile",
            "emu",
            "--bin",
            "count",
            "--target-dir",
        ])
        .arg(&build_dir)
        .env_remove("RUSTUP_TOOLCHAIN")
        .env_remove("CARGO")
        .status()
        .context("cannot run cargo")?;
    if !status.success() {
        bail!("building the Cortex-M4 workload failed");
    }
    Ok(build_dir
        .join("thumbv7em-none-eabihf")
        .join("emu")
        .join("count"))
}

fn count_run(
    qemu: &Path,
    plugin: &Path,
    binary: &Path,
    kernel: &str,
    k: u64,
) -> anyhow::Result<u64> {
    let out = Command::new(qemu)
        .args(["-machine", "mps2-an386", "-cpu", "cortex-m4", "-nographic"])
        .arg("-semihosting-config")
        .arg(format!("enable=on,target=native,arg={kernel},arg={k}"))
        .arg("-plugin")
        .arg(plugin)
        .args(["-d", "plugin", "-kernel"])
        .arg(binary)
        .output()
        .context("cannot run qemu-system-arm")?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stdout.contains("ok=true") || !out.status.success() {
        bail!("the workload failed for {kernel} k={k}: {stdout}{stderr}");
    }
    parse::qemu_insn(&stderr)
}

/// Runs the count.
///
/// # Errors
/// A tool is missing or a run fails.
pub fn run(root: &Path, only: &[String], write: bool) -> anyhow::Result<bool> {
    let qemu = find_tool(root, Tool::QemuArm)
        .ok_or_else(|| anyhow!("qemu-system-arm is missing; {}", Tool::QemuArm.hint()))?;
    let tools = root.join("target").join("tools");
    let plugin = tools.join("libinsn.so");
    if !plugin.exists() {
        build_insn_plugin(&tools, &plugin)?;
    }
    let binary = build(root)?;
    let mut rows = serde_json::Map::new();
    println!("{:<18} {:>14}", "kernel", "instr/sample");
    for kernel in KERNELS {
        if !only.is_empty() && !only.iter().any(|k| k == kernel) {
            continue;
        }
        let base = count_run(&qemu, &plugin, &binary, kernel, 0)?;
        let full = count_run(&qemu, &plugin, &binary, kernel, K)?;
        let v = per_sample(base, full, K, SAMPLES_PER_CALL)?;
        println!("{kernel:<18} {v:>14.2}");
        rows.insert(
            kernel.to_string(),
            json!({ "implementation": "fallback", "instructions_per_sample": round2(v) }),
        );
    }
    if write && only.is_empty() {
        let dir = root.join("benches").join("results");
        std::fs::create_dir_all(&dir)?;
        let doc = json!({
            "platform": "cortex-m4",
            "machine": "qemu-system-arm -machine mps2-an386 -cpu cortex-m4",
            "method": {
                "k": K,
                "samples_per_call": SAMPLES_PER_CALL,
                "counted": "(count(k = K) - count(k = 0)) / (K * samples_per_call)",
                "counter": "QEMU insn plugin",
            },
            "kernels": rows,
        });
        let mut text = serde_json::to_string_pretty(&doc)?;
        text.push('\n');
        std::fs::write(dir.join("cortex-m4.json"), text)?;
    }
    println!("count cortex-m4: ok (report only, no limits)");
    Ok(true)
}
