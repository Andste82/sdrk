//! `cargo xtask count --platform cortex-m4`: instructions per sample of the portable fallback on
//! a Cortex-M4F, counted in `qemu-system-arm` with the `insn` plugin. The numbers are reported
//! to `benches/results/cortex-m4.json`; there are no limits (report only).
//!
//! The method is the one of the other platforms: the workload binary
//! (`targets/qemu-cortex-m4/src/bin/count.rs`) runs a kernel once as a warm-up and then `k`
//! times; the instructions of `k = 16` minus those of `k = 0`, divided by the samples of the 16
//! calls, are the instructions per sample.
//!
//! The four biquad kernels (M22) are also counted with CMSIS-DSP 1.18.0 built for the M4F (the
//! DSP extension and the FPU; `benches/refs/build.sh cortex-m4`), on the input and with the
//! samples per call of the host counts, and both sides get the SNR against the f64 model (the
//! workload prints the energies, see [`snr_db`]).
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::large_stack_frames,
    reason = "xtask is a host tool; std::process::Command is large"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, anyhow, bail};
use serde_json::{Value, json};

use crate::count::{K, SNR_MARGIN_DB, build_insn_plugin, parse, references, round2};
use crate::emu::{Tool, find_tool};

/// Samples per call of the workload binary (a kernel of `n` samples, or an FFT of `n` points).
pub const SAMPLES_PER_CALL: u64 = 1024;

/// The CPU the workload is compiled for (`-C target-cpu`, set in
/// `targets/qemu-cortex-m4/.cargo/config.toml`): the CPU of the QEMU machine and the
/// `-mcpu` of the CMSIS-DSP reference, so that both sides are tuned for the same core (D30).
pub const TARGET_CPU: &str = "cortex-m4";

/// Samples per call of the biquad kernels: those of the host counts (`benches/kernels`).
pub const BIQUAD_SAMPLES_PER_CALL: u64 = 4096;

/// The kernels the workload binary knows, in report order.
pub const KERNELS: [&str; 19] = [
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
    "agc",
    "agc_cq15",
    "dc_block_real",
    "dc_block_real_q15",
    "biquad",
    "biquad_cq15",
    "biquad_real",
    "biquad_real_q15",
];

/// What the CMSIS-DSP side of a biquad kernel calls (the text of `compared_with`), or `None`
/// for a kernel without a reference.
#[must_use]
pub fn cmsis_reference(kernel: &str) -> Option<&'static str> {
    Some(match kernel {
        "biquad" => {
            "CMSIS-DSP 1.18.0 arm_biquad_cascade_stereo_df2T_f32 (re and im as the two channels), Cortex-M4 with the FPU (-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -O3 -ffp-contract=off, LOOPUNROLL, CMSIS-Core 6.3.0)"
        }
        "biquad_cq15" => {
            "CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), two calls on planar re and im (deinterleaving not counted), Cortex-M4 with the DSP extension (ARM_MATH_DSP; -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -O3 -ffp-contract=off, LOOPUNROLL, CMSIS-Core 6.3.0)"
        }
        "biquad_real" => {
            "CMSIS-DSP 1.18.0 arm_biquad_cascade_df2T_f32, Cortex-M4 with the FPU (-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -O3 -ffp-contract=off, LOOPUNROLL, CMSIS-Core 6.3.0)"
        }
        "biquad_real_q15" => {
            "CMSIS-DSP 1.18.0 arm_biquad_cascade_df1_q15 (postShift 1, truncating), Cortex-M4 with the DSP extension (ARM_MATH_DSP; -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -O3 -ffp-contract=off, LOOPUNROLL, CMSIS-Core 6.3.0)"
        }
        _ => return None,
    })
}

/// Samples per call of `kernel` in the workload binary.
#[must_use]
pub fn samples_per_call(kernel: &str) -> u64 {
    if cmsis_reference(kernel).is_some() {
        BIQUAD_SAMPLES_PER_CALL
    } else {
        SAMPLES_PER_CALL
    }
}

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

/// The SNR in dB from the line `sdrk-snr <name> sig=<f64> err=<f64>` of the workload: the
/// formula of `benches/kernels` (`10 log10(sig / err)`, 999 for an exact output).
///
/// # Errors
/// The line is missing or malformed.
pub fn snr_db(stdout: &str) -> anyhow::Result<f64> {
    let line = stdout
        .lines()
        .find(|l| l.starts_with("sdrk-snr "))
        .ok_or_else(|| anyhow!("no sdrk-snr line in the output: {stdout}"))?;
    let value = |key: &str| -> anyhow::Result<f64> {
        line.split_whitespace()
            .find_map(|w| w.strip_prefix(key))
            .ok_or_else(|| anyhow!("no {key} in {line}"))?
            .parse::<f64>()
            .with_context(|| format!("{key} in {line}"))
    };
    let (sig, err) = (value("sig=")?, value("err=")?);
    Ok(if err == 0.0 {
        999.0
    } else {
        10.0 * (sig / err).log10()
    })
}

fn build(root: &Path, refs: &Path) -> anyhow::Result<PathBuf> {
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
            "--features",
            "cmsis-dsp",
            "--target-dir",
        ])
        .arg(&build_dir)
        .env("SDRK_REFS_DIR", refs)
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

/// One run of the workload: `kernel` with `arg` (`k`, or `snr`); the standard output and the
/// instruction count.
fn qemu_run(
    qemu: &Path,
    plugin: &Path,
    binary: &Path,
    kernel: &str,
    arg: &str,
) -> anyhow::Result<(String, u64)> {
    let out = Command::new(qemu)
        .args(["-machine", "mps2-an386", "-cpu", "cortex-m4", "-nographic"])
        .arg("-semihosting-config")
        .arg(format!("enable=on,target=native,arg={kernel},arg={arg}"))
        .arg("-plugin")
        .arg(plugin)
        .args(["-d", "plugin", "-kernel"])
        .arg(binary)
        .output()
        .context("cannot run qemu-system-arm")?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stdout.contains("ok=true") || !out.status.success() {
        bail!("the workload failed for {kernel} {arg}: {stdout}{stderr}");
    }
    let count = parse::qemu_insn(&stderr)?;
    Ok((stdout, count))
}

/// Instructions per sample of `kernel` (`<kernel>` or `<kernel>@cmsis`).
fn count(
    qemu: &Path,
    plugin: &Path,
    binary: &Path,
    kernel: &str,
    samples: u64,
) -> anyhow::Result<f64> {
    let (_, base) = qemu_run(qemu, plugin, binary, kernel, "0")?;
    let (_, full) = qemu_run(qemu, plugin, binary, kernel, &K.to_string())?;
    per_sample(base, full, K, samples)
}

/// The SNR of one call of `kernel` (`<kernel>` or `<kernel>@cmsis`) against the f64 model.
fn accuracy(qemu: &Path, plugin: &Path, binary: &Path, kernel: &str) -> anyhow::Result<f64> {
    let (stdout, _) = qemu_run(qemu, plugin, binary, kernel, "snr")?;
    snr_db(&stdout)
}

/// The record of a biquad kernel: our fallback, CMSIS-DSP on the same input, their ratio and
/// both SNRs (report only: no limit, no verdict).
fn biquad_row(
    qemu: &Path,
    plugin: &Path,
    binary: &Path,
    kernel: &str,
    ours: f64,
    compared_with: &str,
) -> anyhow::Result<Value> {
    let samples = samples_per_call(kernel);
    let cmsis = format!("{kernel}@cmsis");
    let other = count(qemu, plugin, binary, &cmsis, samples)?;
    let snr_ours = accuracy(qemu, plugin, binary, kernel)?;
    let snr_other = accuracy(qemu, plugin, binary, &cmsis)?;
    let ratio = round2(ours / other);
    println!(
        "{:<18} {:>14.2}   CMSIS-DSP {other:.2}, ratio {ratio:.2}, SNR {snr_ours:.2} dB against {snr_other:.2} dB",
        "", ours
    );
    if snr_other - snr_ours > SNR_MARGIN_DB {
        println!(
            "  note: {kernel} is more than {SNR_MARGIN_DB} dB less accurate than CMSIS-DSP on the same input"
        );
    }
    Ok(json!({
        "implementation": "fallback",
        "instructions_per_sample": round2(ours),
        "samples_per_call": samples,
        "compared_with": compared_with,
        "other_instr_per_sample": round2(other),
        "ratio": ratio,
        "snr_db": { "ours": round2(snr_ours), "other": round2(snr_other) },
    }))
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
    let refs = references(root, "cortex-m4")?;
    let binary = build(root, &refs)?;
    let mut rows = serde_json::Map::new();
    println!("{:<18} {:>14}", "kernel", "instr/sample");
    for kernel in KERNELS {
        if !only.is_empty() && !only.iter().any(|k| k == kernel) {
            continue;
        }
        let v = count(&qemu, &plugin, &binary, kernel, samples_per_call(kernel))?;
        println!("{kernel:<18} {v:>14.2}");
        let row = match cmsis_reference(kernel) {
            Some(compared_with) => biquad_row(&qemu, &plugin, &binary, kernel, v, compared_with)?,
            None => json!({ "implementation": "fallback", "instructions_per_sample": round2(v) }),
        };
        rows.insert(kernel.to_string(), row);
    }
    if write && only.is_empty() {
        let manifest = std::fs::read_to_string(refs.join("refs.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .unwrap_or(Value::Null);
        let dir = root.join("benches").join("results");
        std::fs::create_dir_all(&dir)?;
        let doc = json!({
            "platform": "cortex-m4",
            "machine": "qemu-system-arm -machine mps2-an386 -cpu cortex-m4",
            "status": "report only, no limits",
            "build": {
                "target": "thumbv7em-none-eabihf",
                "target_cpu": TARGET_CPU,
                "profile": "emu",
            },
            "method": {
                "k": K,
                "samples_per_call": SAMPLES_PER_CALL,
                "samples_per_call_biquad": BIQUAD_SAMPLES_PER_CALL,
                "counted": "(count(k = K) - count(k = 0)) / (K * samples_per_call)",
                "counter": "QEMU insn plugin",
                "accuracy": "SNR in dB of one call from a fresh state against the f64 model of benches/kernels, on the same input for both sides",
            },
            "references": manifest,
            "kernels": rows,
        });
        let mut text = serde_json::to_string_pretty(&doc)?;
        text.push('\n');
        std::fs::write(dir.join("cortex-m4.json"), text)?;
    }
    println!("count cortex-m4: ok (report only, no limits)");
    Ok(true)
}
