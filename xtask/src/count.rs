//! `cargo xtask count`: instruction counts of the SDRk kernels against VOLK, rustfft and our own
//! fallback.
//!
//! The method is differential. The measurement binary of `benches/kernels` (`count run <kernel>
//! <impl> <k>`) performs one warm-up call and then `k` calls on the same buffers. It is started
//! twice under an instruction counter, with `k = 0` and with `k = K`; the instructions per
//! sample are `(count(K) - count(0)) / (K * samples)`. Everything outside the `K` calls (start
//! of the process, allocation, set-up, the warm-up call) cancels.
//!
//! Counters per platform:
//!
//! | Platform | Counter |
//! | --- | --- |
//! | `x86_64` (`sse2`, `avx2`) | Valgrind `callgrind` |
//! | `x86_64-avx512` | Intel SDE (`sde64 -spr -mix`) |
//! | `aarch64` | QEMU user mode with the `insn` plugin |
//! | `wasm32` | wasmtime fuel (found by bisection) |
//!
//! The limits are in `benches/limits.json` and the results in `benches/results/<platform>.json`.
//! See `docs/performance.md`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, anyhow, bail};
use serde_json::{Value, json};

/// Calls per measurement in the differential method.
pub const K: u64 = 16;
/// Samples per call unless a kernel fixes its own size.
pub const N: u64 = 4096;
/// Hard limit of a `release` kernel: instructions per sample against the comparison.
pub const RELEASE_LIMIT: f64 = 1.15;
/// Limit of a kernel that is still `dev`.
pub const DEV_LIMIT: f64 = 1.50;
/// From this ratio on, a passing kernel gets a warning.
pub const WARN_FROM: f64 = 1.10;
/// How much worse than the comparison the accuracy may be, in dB.
pub const SNR_MARGIN_DB: f64 = 1.0;

/// The release state of a kernel on one level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Must stay within [`RELEASE_LIMIT`].
    Release,
    /// Under development: within [`DEV_LIMIT`].
    Dev,
}

/// One entry of `benches/limits.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limit {
    /// The state.
    pub status: Status,
    /// If present, a failing verdict is reported as waived: the text names the open item in
    /// `docs/open-items.md` that tracks the exceedance. A waiver never hides the numbers.
    pub waiver: Option<String>,
}

/// The parsed `benches/limits.json`: kernel, then level.
#[derive(Clone, Debug, Default)]
pub struct Limits(BTreeMap<String, BTreeMap<String, Limit>>);

impl Limits {
    /// Parses the file: `{ "<kernel>": { "<level>": { "status": "release" | "dev",
    /// "waiver": "<text>" } } }`.
    ///
    /// # Errors
    /// Invalid JSON, or an unknown status.
    pub fn parse(text: &str) -> anyhow::Result<Limits> {
        let value: Value = serde_json::from_str(text).context("limits.json is not valid JSON")?;
        let Some(kernels) = value.as_object() else {
            bail!("limits.json must be an object");
        };
        let mut out = BTreeMap::new();
        for (kernel, levels) in kernels {
            if kernel.starts_with('_') {
                continue;
            }
            let Some(levels) = levels.as_object() else {
                bail!("limits.json: `{kernel}` must be an object");
            };
            let mut per_level = BTreeMap::new();
            for (level, entry) in levels {
                let status = match entry.get("status").and_then(Value::as_str) {
                    Some("release") => Status::Release,
                    Some("dev") => Status::Dev,
                    other => bail!("limits.json: `{kernel}.{level}`: unknown status {other:?}"),
                };
                let waiver = entry
                    .get("waiver")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                per_level.insert(level.clone(), Limit { status, waiver });
            }
            out.insert(kernel.clone(), per_level);
        }
        Ok(Limits(out))
    }

    /// The entry of a kernel on a level; a kernel that is not listed is `dev`.
    #[must_use]
    pub fn get(&self, kernel: &str, level: &str) -> Limit {
        self.0
            .get(kernel)
            .and_then(|l| l.get(level))
            .cloned()
            .unwrap_or(Limit {
                status: Status::Dev,
                waiver: None,
            })
    }
}

/// What the comparison of one kernel on one level measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// Kernel name.
    pub kernel: String,
    /// Level name (`sse2`, `avx2`, ...).
    pub level: String,
    /// Instructions per sample of our kernel.
    pub ours: f64,
    /// Instructions per sample of the comparison.
    pub other: f64,
    /// Name of the comparison.
    pub compared_with: String,
    /// SNR in dB of our kernel against the f64 reference.
    pub ours_snr_db: f64,
    /// SNR in dB of the comparison against the f64 reference.
    pub other_snr_db: f64,
}

impl Measurement {
    /// Instructions of ours divided by instructions of the comparison.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.other > 0.0 {
            self.ours / self.other
        } else {
            f64::INFINITY
        }
    }
}

/// The judgement of one measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Within the limit.
    Pass,
    /// Within the limit, but at or above [`WARN_FROM`].
    Warn(String),
    /// Above the limit, or less accurate than the comparison.
    Fail(String),
    /// Above the limit, with a waiver in `limits.json`.
    Waived(String),
}

/// Judges a measurement against its entry.
#[must_use]
pub fn judge(limit: &Limit, m: &Measurement) -> Verdict {
    let ratio = m.ratio();
    let cap = match limit.status {
        Status::Release => RELEASE_LIMIT,
        Status::Dev => DEV_LIMIT,
    };
    let mut problem = None;
    if ratio > cap {
        problem = Some(format!(
            "{ratio:.2}x instructions per sample against {} exceeds the {} limit of {cap:.2}x",
            m.compared_with,
            match limit.status {
                Status::Release => "release",
                Status::Dev => "dev",
            }
        ));
    } else if m.other_snr_db - m.ours_snr_db > SNR_MARGIN_DB {
        problem = Some(format!(
            "accuracy {:.1} dB is more than {SNR_MARGIN_DB:.0} dB below the {:.1} dB of {}",
            m.ours_snr_db, m.other_snr_db, m.compared_with
        ));
    }
    match (problem, &limit.waiver) {
        (Some(p), Some(w)) => Verdict::Waived(format!("{p} (waiver: {w})")),
        (Some(p), None) => Verdict::Fail(p),
        (None, _) if ratio > WARN_FROM => Verdict::Warn(format!(
            "{ratio:.2}x is above {WARN_FROM:.2}x (hard limit {RELEASE_LIMIT:.2}x)"
        )),
        (None, _) => Verdict::Pass,
    }
}

/// The outcome of a whole level.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// One line per kernel for the console.
    pub lines: Vec<String>,
    /// `false` if any measurement failed.
    pub ok: bool,
    /// The per-kernel records for the results file.
    pub records: BTreeMap<String, Value>,
}

/// Judges all measurements of one level against the limits.
#[must_use]
pub fn evaluate(limits: &Limits, level: &str, measurements: &[Measurement]) -> Report {
    let mut report = Report {
        ok: true,
        ..Report::default()
    };
    for m in measurements {
        let limit = limits.get(&m.kernel, level);
        let verdict = judge(&limit, m);
        let (word, detail) = match &verdict {
            Verdict::Pass => ("pass", String::new()),
            Verdict::Warn(d) => ("warn", d.clone()),
            Verdict::Fail(d) => {
                report.ok = false;
                ("FAIL", d.clone())
            }
            Verdict::Waived(d) => ("waived", d.clone()),
        };
        let status = match limit.status {
            Status::Release => "release",
            Status::Dev => "dev",
        };
        let mut line = format!(
            "{:<18} {:<8} {:>8.2} vs {:>8.2} instr/sample  ratio {:>5.2}  [{status}]  {word}",
            m.kernel,
            m.level,
            m.ours,
            m.other,
            m.ratio()
        );
        if !detail.is_empty() {
            let _ = write!(line, "  {detail}");
        }
        report.lines.push(line);
        report.records.insert(
            m.kernel.clone(),
            json!({
                "ours_instr_per_sample": round2(m.ours),
                "other_instr_per_sample": round2(m.other),
                "ratio": round2(m.ratio()),
                "compared_with": m.compared_with,
                "status": status,
                "verdict": word.to_lowercase(),
                "snr_db": { "ours": round2(m.ours_snr_db), "other": round2(m.other_snr_db) },
            }),
        );
    }
    report
}

/// A level that is only reported (no limit applies): the numbers and the comparison, no verdict.
#[must_use]
pub fn informational(measurements: &[Measurement]) -> Report {
    let mut report = Report {
        ok: true,
        ..Report::default()
    };
    for m in measurements {
        report.lines.push(format!(
            "{:<18} {:<12} {:>8.2} vs {:>8.2} instr/sample  ratio {:>5.2}  [informational]",
            m.kernel,
            m.level,
            m.ours,
            m.other,
            m.ratio()
        ));
        report.records.insert(
            m.kernel.clone(),
            json!({
                "ours_instr_per_sample": round2(m.ours),
                "other_instr_per_sample": round2(m.other),
                "ratio": round2(m.ratio()),
                "compared_with": m.compared_with,
                "verdict": "info",
                "snr_db": { "ours": round2(m.ours_snr_db), "other": round2(m.other_snr_db) },
            }),
        );
    }
    report
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

// ---------------------------------------------------------------------------------------------
// The measurement plan

/// How a kernel is compared on a platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    /// The system VOLK where the platform has it, else our fallback.
    Volk,
    /// `rustfft`.
    RustFft,
    /// Our fallback.
    Fallback,
}

/// The kernels of the benchmark table: name, comparison, samples per call.
pub const KERNELS: [(&str, Compare, u64); 13] = [
    ("convert_cs8", Compare::Volk, N),
    ("convert_cs16", Compare::Volk, N),
    ("convert_out_cs16", Compare::Volk, N),
    ("rotate", Compare::Volk, N),
    ("fir", Compare::Volk, N),
    ("fir_decimate", Compare::Volk, N),
    ("mag_sq", Compare::Volk, N),
    ("quad_demod", Compare::Volk, N),
    ("fm_mod", Compare::Fallback, N),
    ("fft_64", Compare::RustFft, 64),
    ("fft_1024", Compare::RustFft, 1024),
    ("fft_4096", Compare::RustFft, 4096),
    ("fft_cq15_1024", Compare::Fallback, 1024),
];

/// How the measurement binary is run and counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counter {
    /// Valgrind callgrind.
    Callgrind,
    /// Intel SDE mix histogram.
    Sde,
    /// QEMU user mode with the `insn` plugin.
    Qemu,
    /// wasmtime with fuel.
    Wasmtime,
}

/// Which VOLK implementations the dispatcher may choose, so that the comparison is made at the
/// same instruction set as our kernels and does not depend on the CPU of the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolkPin {
    /// VOLK picks its best implementation itself.
    Free,
    /// Implementations that need no more than SSE3.
    Sse,
    /// Implementations that need no more than AVX2 and FMA (no AVX-512).
    Avx2,
}

impl VolkPin {
    /// The `volk_config` that pins the dispatcher, and the directory name below `benches/target`.
    #[must_use]
    pub const fn config(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::Free => None,
            Self::Sse => Some(("volk-sse", VOLK_SSE_CONFIG)),
            Self::Avx2 => Some(("volk-avx2", VOLK_AVX2_CONFIG)),
        }
    }
}

/// A level of a platform: how to build the binary and how to count it.
#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct LevelPlan {
    /// Platform name given to `--platform`.
    pub platform: &'static str,
    /// Level name (key of `limits.json`).
    pub level: &'static str,
    /// Only reported, never judged.
    pub info_only: bool,
    /// `cargo --target`, if cross-compiling.
    pub target: Option<&'static str>,
    /// `RUSTFLAGS`.
    pub rustflags: &'static str,
    /// Feature `std` of the kernels crate (run-time detection of the level).
    pub runtime_detect: bool,
    /// The counter.
    pub counter: Counter,
    /// VOLK exists on this platform (it is the system library of the host).
    pub volk: bool,
    /// The VOLK implementations the dispatcher may choose.
    pub volk_pin: VolkPin,
}

/// All levels, in the order they run.
pub const PLANS: [LevelPlan; 6] = [
    LevelPlan {
        platform: "x86_64",
        level: "sse2",
        info_only: false,
        target: None,
        rustflags: "",
        runtime_detect: false,
        counter: Counter::Callgrind,
        volk: true,
        volk_pin: VolkPin::Sse,
    },
    LevelPlan {
        platform: "x86_64",
        level: "avx2",
        info_only: false,
        target: None,
        rustflags: "-C target-cpu=x86-64-v3",
        runtime_detect: false,
        counter: Counter::Callgrind,
        volk: true,
        volk_pin: VolkPin::Avx2,
    },
    LevelPlan {
        platform: "x86_64",
        level: "avx2-runtime",
        info_only: true,
        target: None,
        rustflags: "",
        runtime_detect: true,
        counter: Counter::Callgrind,
        volk: true,
        volk_pin: VolkPin::Avx2,
    },
    LevelPlan {
        platform: "x86_64-avx512",
        level: "avx512",
        info_only: false,
        // An explicit target keeps `RUSTFLAGS` away from the build scripts, which run on the
        // host (without AVX-512) while the measurement binary runs under SDE.
        target: Some("x86_64-unknown-linux-gnu"),
        rustflags: "-C target-cpu=sapphirerapids",
        runtime_detect: false,
        counter: Counter::Sde,
        volk: true,
        volk_pin: VolkPin::Free,
    },
    LevelPlan {
        platform: "aarch64",
        level: "neon",
        info_only: false,
        target: Some("aarch64-unknown-linux-gnu"),
        rustflags: "",
        runtime_detect: false,
        counter: Counter::Qemu,
        volk: false,
        volk_pin: VolkPin::Free,
    },
    LevelPlan {
        platform: "wasm32",
        level: "simd128",
        info_only: false,
        target: Some("wasm32-wasip1"),
        rustflags: "-C target-feature=+simd128",
        runtime_detect: false,
        counter: Counter::Wasmtime,
        volk: false,
        volk_pin: VolkPin::Free,
    },
];

/// The implementation name and the label of the comparison of a kernel on a level.
#[must_use]
pub fn comparison(plan: &LevelPlan, kernel: &str, compare: Compare) -> (&'static str, String) {
    match compare {
        Compare::Volk if plan.volk => ("library", volk_name(kernel).to_string()),
        Compare::RustFft => ("library", "rustfft".to_string()),
        _ => ("fallback", "sdrk fallback".to_string()),
    }
}

/// The VOLK function(s) a kernel is compared with.
#[must_use]
pub fn volk_name(kernel: &str) -> &'static str {
    match kernel {
        "convert_cs8" => "volk_8i_s32f_convert_32f",
        "convert_cs16" => "volk_16i_s32f_convert_32f",
        "convert_out_cs16" => "volk_32f_s32f_convert_16i",
        "rotate" => "volk_32fc_s32fc_x2_rotator2_32fc",
        "fir" | "fir_decimate" => "volk_32fc_32f_dot_prod_32fc",
        "mag_sq" => "volk_32fc_magnitude_squared_32f",
        "quad_demod" => "volk_32fc_x2_multiply_conjugate_32fc + volk_32fc_s32f_atan2_32f",
        _ => "volk",
    }
}

/// The VOLK implementations that need no more than SSE3, for the `sse2` level. VOLK has no
/// `sse2` variant of most kernels; a kernel without one runs the portable `generic` code.
const VOLK_SSE_CONFIG: &str = "\
volk_8i_s32f_convert_32f generic generic
volk_16i_s32f_convert_32f a_sse u_sse
volk_32f_s32f_convert_16i a_sse2 u_sse2
volk_32fc_s32fc_x2_rotator2_32fc generic generic
volk_32fc_32f_dot_prod_32fc a_sse u_sse
volk_32fc_magnitude_squared_32f a_sse3 u_sse3
volk_32fc_x2_multiply_conjugate_32fc a_sse3 u_sse3
volk_32fc_s32f_atan2_32f generic generic
";

/// The VOLK implementations that need no more than AVX2 and FMA, for the levels `avx2` and
/// `avx2-runtime`. Without this, a runner with AVX-512 makes VOLK dispatch to its AVX-512
/// implementations (`rotator2`, `atan2`, `dot_prod` have them) and the AVX2 build of our kernels
/// would be compared with a different instruction set than on a machine without AVX-512.
const VOLK_AVX2_CONFIG: &str = "\
volk_8i_s32f_convert_32f a_avx2 u_avx2
volk_16i_s32f_convert_32f a_avx2 u_avx2
volk_32f_s32f_convert_16i a_avx2 u_avx2
volk_32fc_s32fc_x2_rotator2_32fc a_avx u_avx
volk_32fc_32f_dot_prod_32fc a_avx2_fma u_avx2_fma
volk_32fc_magnitude_squared_32f a_avx u_avx
volk_32fc_x2_multiply_conjugate_32fc a_avx u_avx
volk_32fc_s32f_atan2_32f a_avx2_fma u_avx2_fma
";

// ---------------------------------------------------------------------------------------------
// Counters

/// Extracts the instruction count from the output of a counter.
pub mod parse {
    use anyhow::{anyhow, bail};

    /// The total of a callgrind run: the line `==pid== Collected : N ...` on standard error.
    ///
    /// # Errors
    /// No such line.
    pub fn callgrind(stderr: &str) -> anyhow::Result<u64> {
        for line in stderr.lines() {
            if let Some(rest) = line.split("Collected :").nth(1) {
                let first = rest
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| anyhow!("empty callgrind total"))?;
                return Ok(first.parse()?);
            }
        }
        bail!("no `Collected :` line in the callgrind output")
    }

    /// The total of an Intel SDE mix report: the line `*total  N`.
    ///
    /// # Errors
    /// No such line.
    pub fn sde_mix(report: &str) -> anyhow::Result<u64> {
        for line in report.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("*total") {
                let first = rest
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| anyhow!("empty total in the mix report"))?;
                return Ok(first.parse()?);
            }
        }
        bail!("no `*total` line in the SDE mix report")
    }

    /// The total of the QEMU `insn` plugin: `total insns: N`, or the sum of the lines
    /// `cpu K insns: N`.
    ///
    /// # Errors
    /// Neither form is present.
    pub fn qemu_insn(stderr: &str) -> anyhow::Result<u64> {
        let mut sum = 0u64;
        let mut seen = false;
        for line in stderr.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("total insns:") {
                return Ok(rest.trim().parse()?);
            }
            if line.starts_with("cpu ")
                && let Some(n) = line.split("insns:").nth(1)
            {
                sum += n.trim().parse::<u64>()?;
                seen = true;
            }
        }
        if seen {
            Ok(sum)
        } else {
            bail!("no instruction count in the QEMU plugin output")
        }
    }
}

/// The smallest fuel with which `run(fuel)` succeeds, found by bisection. `run` returns `true`
/// when the program completes.
///
/// # Errors
/// The program does not complete with the largest fuel that is tried.
pub fn least_fuel(mut run: impl FnMut(u64) -> anyhow::Result<bool>) -> anyhow::Result<u64> {
    let mut hi = 1u64 << 20;
    while !run(hi)? {
        hi = hi
            .checked_mul(4)
            .filter(|h| *h < 1 << 50)
            .ok_or_else(|| anyhow!("the program needs more than 2^50 fuel"))?;
    }
    let mut lo = 0u64; // known to fail
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if run(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(hi)
}

/// Environment and tools of a counting session.
struct Session {
    root: PathBuf,
    plan: LevelPlan,
    binary: PathBuf,
    volk_config: Option<PathBuf>,
    qemu: Option<(PathBuf, PathBuf, PathBuf)>,
}

impl Session {
    fn command(&self, program: impl AsRef<std::ffi::OsStr>) -> Command {
        let mut cmd = Command::new(program);
        if let Some(dir) = &self.volk_config {
            cmd.env("VOLK_CONFIGPATH", dir);
        }
        cmd
    }

    /// Runs the measurement binary without a counter and returns its standard output.
    fn plain(&self, args: &[String]) -> anyhow::Result<String> {
        let mut cmd = match self.plan.counter {
            Counter::Callgrind => self.command(&self.binary),
            Counter::Sde => {
                let mut c = self.command(sde());
                c.args(["-spr", "--"]).arg(&self.binary);
                c
            }
            Counter::Qemu => {
                let (qemu, _, sysroot) = self.qemu.clone().ok_or_else(|| anyhow!("qemu"))?;
                let mut c = self.command(qemu);
                c.arg("-L").arg(sysroot).arg(&self.binary);
                c
            }
            Counter::Wasmtime => {
                let mut c = self.command("wasmtime");
                c.arg("run").arg(&self.binary);
                c
            }
        };
        let out = cmd
            .args(args)
            .stderr(Stdio::inherit())
            .output()
            .context("cannot run the measurement binary")?;
        if !out.status.success() {
            bail!("the measurement binary failed: {args:?}");
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// Instructions of one run with the given arguments.
    fn count(&self, args: &[String]) -> anyhow::Result<u64> {
        match self.plan.counter {
            Counter::Callgrind => {
                let mut cmd = self.command("valgrind");
                cmd.args(["--tool=callgrind", "--callgrind-out-file=/dev/null"])
                    .arg(&self.binary)
                    .args(args);
                let out = cmd.output().context("cannot run valgrind")?;
                if !out.status.success() {
                    bail!("valgrind failed: {}", String::from_utf8_lossy(&out.stderr));
                }
                parse::callgrind(&String::from_utf8_lossy(&out.stderr))
            }
            Counter::Sde => {
                let report = self.root.join("benches/target/sde-mix.txt");
                let mut cmd = self.command(sde());
                cmd.args(["-spr", "-mix", "-omix"])
                    .arg(&report)
                    .arg("--")
                    .arg(&self.binary)
                    .args(args)
                    .stdout(Stdio::null());
                let status = cmd.status().context("cannot run sde64")?;
                if !status.success() {
                    bail!("sde64 failed");
                }
                parse::sde_mix(&std::fs::read_to_string(&report)?)
            }
            Counter::Qemu => {
                let (qemu, plugin, sysroot) = self.qemu.clone().ok_or_else(|| anyhow!("qemu"))?;
                let mut cmd = self.command(qemu);
                cmd.arg("-L")
                    .arg(sysroot)
                    .arg("-plugin")
                    .arg(plugin)
                    .args(["-d", "plugin"])
                    .arg(&self.binary)
                    .args(args);
                let out = cmd.output().context("cannot run qemu")?;
                if !out.status.success() {
                    bail!("qemu failed: {}", String::from_utf8_lossy(&out.stderr));
                }
                parse::qemu_insn(&String::from_utf8_lossy(&out.stderr))
            }
            Counter::Wasmtime => least_fuel(|fuel| {
                let status = self
                    .command("wasmtime")
                    .arg("run")
                    .arg("-W")
                    .arg(format!("fuel={fuel}"))
                    .arg(&self.binary)
                    .args(args)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .context("cannot run wasmtime")?;
                Ok(status.success())
            }),
        }
    }

    /// `count(k) - count(0)` for `kernel` and `which`.
    fn differential(&self, kernel: &str, which: &str, calls: u64) -> anyhow::Result<u64> {
        let run = |k: u64| -> Vec<String> {
            ["run", kernel, which, &k.to_string()]
                .iter()
                .map(ToString::to_string)
                .collect()
        };
        let base = self.count(&run(0))?;
        let full = self.count(&run(calls))?;
        full.checked_sub(base).ok_or_else(|| {
            anyhow!(
                "{kernel} {which}: the run with k = {calls} counted fewer instructions than k = 0"
            )
        })
    }

    fn snr(&self, kernel: &str, which: &str) -> anyhow::Result<f64> {
        let args: Vec<String> = ["accuracy", kernel, which]
            .iter()
            .map(ToString::to_string)
            .collect();
        let out = self.plain(&args)?;
        let v: Value = serde_json::from_str(out.trim()).context("accuracy output")?;
        v.get("snr_db")
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow!("no snr_db in {out}"))
    }
}

fn sde() -> PathBuf {
    std::env::var_os("SDRK_SDE").map_or_else(|| PathBuf::from("sde64"), PathBuf::from)
}

/// The tools of the QEMU counter: the QEMU binary with plugins, the built `insn` plugin and the
/// sysroot of the target libc. See `docs/performance.md` for how to set them up.
fn qemu_tools(root: &Path) -> anyhow::Result<(PathBuf, PathBuf, PathBuf)> {
    let qemu = std::env::var_os("SDRK_QEMU_AARCH64")
        .map_or_else(|| PathBuf::from("qemu-aarch64"), PathBuf::from);
    let sysroot = std::env::var_os("SDRK_AARCH64_SYSROOT")
        .map_or_else(|| PathBuf::from("/usr/aarch64-linux-gnu"), PathBuf::from);
    let tools = root.join("target").join("tools");
    let plugin = tools.join("libinsn.so");
    if !plugin.exists() {
        build_insn_plugin(&tools, &plugin)?;
    }
    Ok((qemu, plugin, sysroot))
}

/// Builds the `insn` plugin of the QEMU tree into `target/tools/`. The source is taken from the
/// tree named by `SDRK_QEMU_SRC` (needs `include/qemu/qemu-plugin.h` and
/// `tests/tcg/plugins/insn.c`; the location has moved between QEMU releases) or downloaded
/// from the QEMU repository at the tag `SDRK_QEMU_TAG` (default `v10.2.1`). Needs a C compiler
/// and the glib development files (`libglib2.0-dev`, `pkg-config`).
fn build_insn_plugin(tools: &Path, plugin: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(tools)?;
    let src_dir = if let Some(dir) = std::env::var_os("SDRK_QEMU_SRC") {
        PathBuf::from(dir)
    } else {
        let tag = std::env::var("SDRK_QEMU_TAG").unwrap_or_else(|_| "v10.2.1".to_string());
        let dir = tools.join("qemu-src");
        for file in ["tests/tcg/plugins/insn.c", "include/qemu/qemu-plugin.h"] {
            let dest = dir.join(file);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let url = format!("https://raw.githubusercontent.com/qemu/qemu/{tag}/{file}");
            let status = Command::new("curl")
                .args(["-sfL", &url, "-o"])
                .arg(&dest)
                .status()
                .context("cannot run curl to fetch the QEMU plugin source")?;
            if !status.success() {
                bail!("cannot download {url}; set SDRK_QEMU_SRC to a QEMU source tree");
            }
        }
        dir
    };
    let glib = Command::new("pkg-config")
        .args(["--cflags", "glib-2.0"])
        .output()
        .context("pkg-config is needed to build the QEMU plugin")?;
    if !glib.status.success() {
        bail!("pkg-config cannot find glib-2.0 (install libglib2.0-dev)");
    }
    let cflags = String::from_utf8_lossy(&glib.stdout).into_owned();
    let status = Command::new("gcc")
        .args(["-shared", "-fPIC", "-O2", "-o"])
        .arg(plugin)
        .arg(src_dir.join("tests/tcg/plugins/insn.c"))
        .arg(format!("-I{}", src_dir.join("include/qemu").display()))
        .arg(format!("-I{}", src_dir.join("include").display()))
        .args(cflags.split_whitespace())
        .status()
        .context("cannot run gcc")?;
    if !status.success() {
        bail!("building the QEMU insn plugin failed");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Driver

fn build(root: &Path, plan: &LevelPlan) -> anyhow::Result<PathBuf> {
    let target_dir = root.join("benches").join("target").join(plan.level);
    let mut cmd = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cmd.current_dir(root.join("benches"))
        .args(["build", "--release", "-p", "kernels", "--target-dir"])
        .arg(&target_dir);
    if let Some(target) = plan.target {
        cmd.args(["--target", target]);
    }
    if plan.runtime_detect {
        cmd.args(["--features", "std"]);
    }
    cmd.env("RUSTFLAGS", plan.rustflags);
    if plan.target == Some("aarch64-unknown-linux-gnu") {
        cmd.env(
            "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER",
            "aarch64-linux-gnu-gcc",
        );
    }
    println!(
        "==> building the measurement binary for {}/{}",
        plan.platform, plan.level
    );
    let status = cmd.status().context("cannot run cargo")?;
    if !status.success() {
        bail!("building benches/kernels failed");
    }
    let mut path = target_dir;
    if let Some(target) = plan.target {
        path.push(target);
    }
    path.push("release");
    path.push(if plan.target.is_some_and(|t| t.starts_with("wasm")) {
        "count.wasm"
    } else {
        "count"
    });
    Ok(path)
}

#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the Windows file system calls are large; xtask is a host tool"
)]
fn measure_level(
    root: &Path,
    plan: &LevelPlan,
    only: &[String],
) -> anyhow::Result<Vec<Measurement>> {
    let binary = build(root, plan)?;
    let volk_config = if let Some((name, config)) = plan.volk_pin.config() {
        let dir = root.join("benches").join("target").join(name);
        let file = dir.join("volk").join("volk_config");
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&file, config)?;
        Some(dir)
    } else {
        None
    };
    let qemu = if plan.counter == Counter::Qemu {
        Some(qemu_tools(root)?)
    } else {
        None
    };
    let session = Session {
        root: root.to_path_buf(),
        plan: *plan,
        binary,
        volk_config,
        qemu,
    };
    let mut out = Vec::new();
    for (kernel, compare, samples) in KERNELS {
        if !only.is_empty() && !only.iter().any(|k| k == kernel) {
            continue;
        }
        let (which, label) = comparison(plan, kernel, compare);
        println!("==> {} {} (ours vs {label})", plan.level, kernel);
        // K calls of 4096 samples; a kernel with fewer samples per call gets proportionally more
        // calls, so that every measurement covers the same number of samples and the
        // difference of the two counts stays far above the run-to-run noise of the process.
        let calls = K * (N / samples).max(1);
        let scale = (calls * samples) as f64;
        let ours = session.differential(kernel, "ours", calls)? as f64 / scale;
        let other = session.differential(kernel, which, calls)? as f64 / scale;
        out.push(Measurement {
            kernel: kernel.to_string(),
            level: plan.level.to_string(),
            ours,
            other,
            compared_with: label,
            ours_snr_db: session.snr(kernel, "ours")?,
            other_snr_db: session.snr(kernel, which)?,
        });
    }
    Ok(out)
}

/// `cargo xtask count --pipeline`: instructions per input sample of `fm_receiver.json` at
/// 2.4 MHz (callgrind, x86-64 with AVX2) and the latency of the audio output, written to
/// `benches/results/pipeline.json`.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command is large; xtask is a host tool"
)]
fn pipeline(root: &Path, write: bool) -> anyhow::Result<bool> {
    let target_dir = root.join("benches").join("target").join("pipeline");
    println!("==> building the pipeline benchmark");
    let status = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .current_dir(root.join("benches"))
        .args(["build", "--release", "-p", "pipeline-bench", "--target-dir"])
        .arg(&target_dir)
        .env("RUSTFLAGS", "-C target-cpu=x86-64-v3")
        .status()
        .context("cannot run cargo")?;
    if !status.success() {
        bail!("building benches/pipeline failed");
    }
    let binary = target_dir.join("release").join("pipeline");
    let count = |k: u64| -> anyhow::Result<u64> {
        let out = Command::new("valgrind")
            .args(["--tool=callgrind", "--callgrind-out-file=/dev/null"])
            .arg(&binary)
            .args(["run", &k.to_string()])
            .output()
            .context("cannot run valgrind")?;
        if !out.status.success() {
            bail!("valgrind failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        parse::callgrind(&String::from_utf8_lossy(&out.stderr))
    };
    let blocks = K;
    let diff = count(blocks)?
        .checked_sub(count(0)?)
        .ok_or_else(|| anyhow!("the pipeline run with more blocks counted fewer instructions"))?;
    let per_sample = diff as f64 / (blocks * N) as f64;
    let info = Command::new(&binary)
        .arg("info")
        .output()
        .context("cannot run the pipeline benchmark")?;
    let latency: Value = serde_json::from_slice(&info.stdout).context("latency output")?;
    println!(
        "pipeline fm_receiver.json at 2.4 MHz: {per_sample:.1} instructions per input sample, latency {} samples = {} us",
        latency.get("latency_samples").unwrap_or(&Value::Null),
        latency.get("latency_us").unwrap_or(&Value::Null)
    );
    if write {
        let dir = root.join("benches").join("results");
        std::fs::create_dir_all(&dir)?;
        let doc = json!({
            "flow": "examples/flows/fm_receiver.json",
            "input": { "format": "cu8", "rate_hz": 2_400_000 },
            "platform": "x86_64 (AVX2)",
            "method": { "blocks": blocks, "block_samples": N, "counted": "(count(k blocks) - count(0 blocks)) / (k * block_samples)" },
            "instructions_per_input_sample": round2(per_sample),
            "latency": latency,
        });
        let mut text = serde_json::to_string_pretty(&doc)?;
        text.push('\n');
        std::fs::write(dir.join("pipeline.json"), text)?;
    }
    Ok(true)
}

/// Runs `cargo xtask count ...`.
///
/// ```text
/// cargo xtask count --platform <x86_64|x86_64-avx512|aarch64|wasm32> [--kernel NAME]... [--no-write]
/// cargo xtask count --pipeline [--no-write]
/// ```
///
/// # Errors
/// A tool is missing or fails; the return value is `Ok(false)` if a measurement exceeds its
/// limit.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the plan table are large; xtask is a host tool"
)]
pub fn run(root: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let mut platform = None;
    let mut only = Vec::new();
    let mut write = true;
    let mut pipeline_mode = false;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match *arg {
            "--platform" => platform = it.next().map(|s| (*s).to_string()),
            "--kernel" => {
                if let Some(k) = it.next() {
                    only.push((*k).to_string());
                }
            }
            "--no-write" => write = false,
            "--pipeline" => pipeline_mode = true,
            other => bail!(
                "unknown option `{other}`; usage: cargo xtask count --platform <p> [--kernel NAME] [--no-write]"
            ),
        }
    }
    if pipeline_mode {
        return pipeline(root, write);
    }
    let Some(platform) = platform else {
        bail!(
            "usage: cargo xtask count --platform <x86_64|x86_64-avx512|aarch64|wasm32> [--kernel NAME] [--no-write]"
        );
    };
    let plans: Vec<&LevelPlan> = PLANS.iter().filter(|p| p.platform == platform).collect();
    if plans.is_empty() {
        bail!("unknown platform `{platform}`");
    }
    let limits_path = root.join("benches").join("limits.json");
    let limits = Limits::parse(
        &std::fs::read_to_string(&limits_path)
            .with_context(|| format!("cannot read {}", limits_path.display()))?,
    )?;
    let mut all_ok = true;
    let mut levels = serde_json::Map::new();
    for plan in plans {
        let measurements = measure_level(root, plan, &only)?;
        let judged = if plan.info_only {
            informational(&measurements)
        } else {
            evaluate(&limits, plan.level, &measurements)
        };
        for line in &judged.lines {
            println!("{line}");
        }
        all_ok &= judged.ok;
        levels.insert(
            plan.level.to_string(),
            json!({ "informational": plan.info_only, "kernels": judged.records }),
        );
    }
    if write && only.is_empty() {
        let dir = root.join("benches").join("results");
        std::fs::create_dir_all(&dir)?;
        let doc = json!({
            "platform": platform,
            "method": { "k": K, "samples": N, "counted": "(count(k = K) - count(k = 0)) / (K * samples)" },
            "levels": levels,
        });
        let mut text = serde_json::to_string_pretty(&doc)?;
        text.push('\n');
        std::fs::write(dir.join(format!("{platform}.json")), text)?;
    }
    if all_ok {
        println!("count {platform}: ok");
    } else {
        eprintln!("count {platform}: limits exceeded");
    }
    Ok(all_ok)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn m(kernel: &str, ours: f64, other: f64) -> Measurement {
        Measurement {
            kernel: kernel.into(),
            level: "avx2".into(),
            ours,
            other,
            compared_with: "volk".into(),
            ours_snr_db: 130.0,
            other_snr_db: 130.0,
        }
    }

    fn limits(text: &str) -> Limits {
        Limits::parse(text).unwrap()
    }

    #[test]
    fn a_slow_release_kernel_fails_the_command() {
        let l = limits(r#"{ "dummy_slow": { "avx2": { "status": "release" } } }"#);
        let report = evaluate(&l, "avx2", &[m("dummy_slow", 20.0, 10.0)]);
        assert!(!report.ok);
        assert!(report.lines[0].contains("FAIL"), "{}", report.lines[0]);
        assert!(report.lines[0].contains("2.00x"));
    }

    #[test]
    fn the_limits_are_115_percent_for_release_and_150_for_dev() {
        let rel = Limit {
            status: Status::Release,
            waiver: None,
        };
        let dev = Limit {
            status: Status::Dev,
            waiver: None,
        };
        assert_eq!(
            judge(&rel, &m("k", 11.4, 10.0)),
            Verdict::Warn("1.14x is above 1.10x (hard limit 1.15x)".into())
        );
        assert_eq!(judge(&rel, &m("k", 10.9, 10.0)), Verdict::Pass);
        assert!(matches!(judge(&rel, &m("k", 11.6, 10.0)), Verdict::Fail(_)));
        assert!(matches!(judge(&dev, &m("k", 11.6, 10.0)), Verdict::Warn(_)));
        assert!(matches!(judge(&dev, &m("k", 14.9, 10.0)), Verdict::Warn(_)));
        assert!(matches!(judge(&dev, &m("k", 15.1, 10.0)), Verdict::Fail(_)));
    }

    #[test]
    fn a_kernel_that_is_not_listed_is_dev() {
        let l = limits("{}");
        assert_eq!(l.get("anything", "avx2").status, Status::Dev);
        let report = evaluate(&l, "avx2", &[m("anything", 14.0, 10.0)]);
        assert!(report.ok);
        let report = evaluate(&l, "avx2", &[m("anything", 16.0, 10.0)]);
        assert!(!report.ok);
    }

    #[test]
    fn lower_accuracy_than_the_comparison_fails() {
        let rel = Limit {
            status: Status::Release,
            waiver: None,
        };
        let mut x = m("k", 10.0, 10.0);
        x.ours_snr_db = 128.5;
        assert!(matches!(judge(&rel, &x), Verdict::Fail(_)));
        x.ours_snr_db = 129.5;
        assert_eq!(judge(&rel, &x), Verdict::Pass);
        x.ours_snr_db = 140.0;
        assert_eq!(judge(&rel, &x), Verdict::Pass);
    }

    #[test]
    fn a_waiver_reports_but_does_not_fail() {
        let l = limits(
            r#"{ "fft_4096": { "avx2": { "status": "dev", "waiver": "open item M13-1" } } }"#,
        );
        let report = evaluate(&l, "avx2", &[m("fft_4096", 40.0, 10.0)]);
        assert!(report.ok);
        assert!(report.lines[0].contains("waived") && report.lines[0].contains("M13-1"));
        assert_eq!(report.records["fft_4096"]["verdict"], "waived");
    }

    #[test]
    fn an_informational_level_never_fails() {
        let report = informational(&[m("fft_4096", 400.0, 10.0)]);
        assert!(report.ok);
        assert!(report.lines[0].contains("informational"));
        assert_eq!(report.records["fft_4096"]["verdict"], "info");
    }

    #[test]
    fn limits_json_rejects_an_unknown_status() {
        assert!(Limits::parse(r#"{ "k": { "avx2": { "status": "fast" } } }"#).is_err());
        assert!(Limits::parse("[]").is_err());
        assert!(Limits::parse("{").is_err());
    }

    #[test]
    fn callgrind_output_is_parsed() {
        let text = "==123== Callgrind, a call-graph generating cache profiler\n==123== \n==123== Collected : 108110 2 3\n==123== I   refs: 108,110\n";
        assert_eq!(parse::callgrind(text).unwrap(), 108_110);
        assert!(parse::callgrind("nothing").is_err());
    }

    #[test]
    fn sde_and_qemu_output_is_parsed() {
        let mix = "# mix\n*total                                                  12345678\n*total_other 5\n";
        assert_eq!(parse::sde_mix(mix).unwrap(), 12_345_678);
        assert!(parse::sde_mix("no totals").is_err());
        assert_eq!(
            parse::qemu_insn("cpu 0 insns: 10\ncpu 1 insns: 5\n").unwrap(),
            15
        );
        assert_eq!(parse::qemu_insn("total insns: 99\n").unwrap(), 99);
        assert!(parse::qemu_insn("").is_err());
    }

    #[test]
    fn the_fuel_bisection_finds_the_exact_amount() {
        for need in [1u64, 2, 1000, 1_048_576, 1_048_577, 5_000_000_000] {
            let mut calls = 0;
            let got = least_fuel(|fuel| {
                calls += 1;
                Ok(fuel >= need)
            })
            .unwrap();
            assert_eq!(got, need);
            assert!(calls < 80);
        }
    }

    #[test]
    fn the_plan_covers_the_four_platforms() {
        for p in ["x86_64", "x86_64-avx512", "aarch64", "wasm32"] {
            assert!(PLANS.iter().any(|plan| plan.platform == p), "{p}");
        }
        let avx2 = PLANS.iter().find(|p| p.level == "avx2").unwrap();
        assert_eq!(comparison(avx2, "rotate", Compare::Volk).0, "library");
        let neon = PLANS.iter().find(|p| p.level == "neon").unwrap();
        assert_eq!(comparison(neon, "rotate", Compare::Volk).0, "fallback");
        assert_eq!(comparison(neon, "fft_1024", Compare::RustFft).0, "library");
    }
}
