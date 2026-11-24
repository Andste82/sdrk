//! `cargo xtask count`: instruction counts where no core model exists for the cycles (D68): the
//! `simd128` level of `wasm32` (wasmtime fuel, judged by `benches/limits.json`) and, in
//! [`crate::count_embedded`], the Cortex-M4 (report only). Every level with a core model
//! (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`) is measured in cycles per sample with
//! `cargo measure` ([`crate::measure`]); `count` rejects their platforms.
//!
//! The method is differential (see [`crate::bench`]): the measurement binary of
//! `benches/kernels` runs under wasmtime with `k = 0` and with `k = K` calls; the instructions
//! per sample are `(count(K) - count(0)) / (K * samples)`, where a count is the least fuel with
//! which the run completes (found by bisection). The in-place FFTs restore their input before
//! every call; the count of the restore alone is subtracted from both sides. The C references
//! are built by `benches/refs/build.sh` into `benches/target/refs/wasm32`; the limits are in
//! `benches/limits.json` and the results in `benches/results/wasm32.json`. See
//! `docs/performance.md`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::process::Stdio;

use anyhow::{Context, anyhow, bail};
use serde_json::{Value, json};

use crate::bench::{
    ComparisonKind, K, KERNELS, KernelSpec, LevelPlan, N, PLANS, Reference, Run, Runner, Session,
    Variant, comparisons, references, run_args, session,
};

/// The platforms whose levels have a core model: measured in cycles per sample by
/// `cargo measure`, not counted.
pub const CYCLE_PLATFORMS: [&str; 3] = ["x86_64", "x86_64-avx512", "aarch64"];

/// How much worse than the comparison the accuracy may be, in dB.
pub const SNR_MARGIN_DB: f64 = 1.0;
/// Sample counts of the informational small-block runs.
pub const SMALL_N: [u64; 2] = [64, 100];

/// The ratio limits. The defaults can be overridden by the object `_limits` in
/// `benches/limits.json` (`{"release": 1.10, ...}`), so that the gate is tightened together with
/// the entries it applies to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Caps {
    /// Hard limit of a `release` entry.
    pub release: f64,
    /// Limit of a `dev` entry.
    pub dev: f64,
    /// Up to this ratio a passing `dev` entry is reported as ready to be promoted; above it a
    /// passing entry gets a warning. With the defaults of D14 it equals the release limit, so it
    /// only marks `dev` entries.
    pub warn: f64,
    /// The goal; results record whether an entry reaches it.
    pub goal: f64,
}

impl Default for Caps {
    fn default() -> Self {
        Caps {
            release: 1.10,
            dev: 1.50,
            warn: 1.10,
            goal: 1.00,
        }
    }
}

/// The release state of a kernel on one level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Must stay within [`Caps::release`].
    Release,
    /// Under development: within [`Caps::dev`].
    Dev,
    /// Reported, never judged: the comparison is not valid for this level (the `waiver` text
    /// says why).
    Info,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Status::Release => "release",
            Status::Dev => "dev",
            Status::Info => "info",
        }
    }
}

/// One entry of `benches/limits.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Limit {
    /// The state.
    pub status: Status,
    /// If present, an instruction ratio above the limit is reported as waived: the text names
    /// the open item in `docs/open-items.md` that tracks it. A waiver never hides the numbers,
    /// and it never covers the accuracy.
    pub waiver: Option<String>,
    /// The same for an accuracy more than [`SNR_MARGIN_DB`] below the comparison.
    pub accuracy_waiver: Option<String>,
    /// The accuracy trade of the cycle gate (`cargo xtask check cycles`, D2760): a `release`
    /// entry above the release limit in cycles stays `release` while its SNR is at least
    /// [`crate::cycle_gate::TRADE_DB`] above the comparison's and its ratio within the `dev`
    /// limit; the text names the SNR of both sides. Not used by the count of `simd128`.
    pub accuracy_trade: Option<String>,
}

#[cfg(test)]
impl Limit {
    /// A `release` entry without waivers.
    #[must_use]
    pub const fn release() -> Self {
        Limit {
            status: Status::Release,
            waiver: None,
            accuracy_waiver: None,
            accuracy_trade: None,
        }
    }
}

/// The parsed `benches/limits.json`: the caps, and the entries by kernel, then level.
#[derive(Clone, Debug, Default)]
pub struct Limits {
    /// The ratio limits.
    pub caps: Caps,
    entries: BTreeMap<String, BTreeMap<String, Limit>>,
}

impl Limits {
    /// Parses the file: `{ "_limits": { "release": 1.10, ... }, "<kernel>": { "<level>": {
    /// "status": "release" | "dev" | "info", "waiver": "<text>", "accuracy_waiver": "<text>",
    /// "accuracy_trade": "<text>" } } }`.
    /// Other keys starting with `_` are comments.
    ///
    /// # Errors
    /// Invalid JSON, an unknown status, or a cap that is not a positive number.
    pub fn parse(text: &str) -> anyhow::Result<Limits> {
        let value: Value = serde_json::from_str(text).context("limits.json is not valid JSON")?;
        let Some(kernels) = value.as_object() else {
            bail!("limits.json must be an object");
        };
        let mut caps = Caps::default();
        if let Some(over) = kernels.get("_limits") {
            let Some(over) = over.as_object() else {
                bail!("limits.json: `_limits` must be an object");
            };
            for (name, v) in over {
                let x = v
                    .as_f64()
                    .filter(|x| *x > 0.0)
                    .ok_or_else(|| anyhow!("limits.json: `_limits.{name}` must be > 0"))?;
                match name.as_str() {
                    "release" => caps.release = x,
                    "dev" => caps.dev = x,
                    "warn" => caps.warn = x,
                    "goal" => caps.goal = x,
                    _ => bail!("limits.json: unknown cap `_limits.{name}`"),
                }
            }
        }
        let mut entries = BTreeMap::new();
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
                    Some("info") => Status::Info,
                    other => bail!("limits.json: `{kernel}.{level}`: unknown status {other:?}"),
                };
                // A target on a comparison of our code with our own code is a verdict on our own
                // code, which the measurement order forbids (D2767; D102 is superseded).
                if entry.get("own_target").is_some() {
                    bail!(
                        "limits.json: `{kernel}.{level}`: `own_target` is no longer a field: an own-kernel comparison is information without a target (D2767)"
                    );
                }
                let text = |key: &str| entry.get(key).and_then(Value::as_str).map(str::to_string);
                per_level.insert(
                    level.clone(),
                    Limit {
                        status,
                        waiver: text("waiver"),
                        accuracy_waiver: text("accuracy_waiver"),
                        accuracy_trade: text("accuracy_trade"),
                    },
                );
            }
            entries.insert(kernel.clone(), per_level);
        }
        Ok(Limits { caps, entries })
    }

    /// The entry of a kernel on a level, `None` if it is not listed.
    #[must_use]
    pub fn get(&self, kernel: &str, level: &str) -> Option<&Limit> {
        self.entries.get(kernel).and_then(|l| l.get(level))
    }

    /// Every entry: kernel, level and its limit, sorted by kernel, then level.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str, &Limit)> {
        self.entries.iter().flat_map(|(kernel, levels)| {
            levels
                .iter()
                .map(move |(level, limit)| (kernel.as_str(), level.as_str(), limit))
        })
    }
}

/// An informational comparison next to the judged one: another library, the cheapest VOLK
/// implementation, or a variant of the buffers (misaligned, small blocks).
#[derive(Clone, Debug, PartialEq)]
pub struct Extra {
    /// What differs from the judged comparison (`misaligned`, `n = 64`, the library).
    pub label: String,
    /// Instructions per sample of our kernel in this variant.
    pub ours: f64,
    /// Instructions per sample of the comparison in this variant.
    pub other: f64,
    /// Name of the comparison.
    pub compared_with: String,
    /// What the comparison is (D2767).
    pub comparison_kind: ComparisonKind,
    /// SNR in dB of the comparison, where measured.
    pub other_snr_db: Option<f64>,
}

impl Extra {
    /// `ours / other`.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        ratio(self.ours, self.other)
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
    /// What the comparison is: a foreign library or our own code (D2767).
    pub comparison_kind: ComparisonKind,
    /// SNR in dB of our kernel against the f64 reference.
    pub ours_snr_db: f64,
    /// SNR in dB of the comparison against the f64 reference.
    pub other_snr_db: f64,
    /// Informational comparisons, never judged.
    pub extras: Vec<Extra>,
}

impl Measurement {
    /// Instructions of ours divided by instructions of the comparison.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        ratio(self.ours, self.other)
    }
}

fn ratio(ours: f64, other: f64) -> f64 {
    if other > 0.0 {
        ours / other
    } else {
        f64::INFINITY
    }
}

/// The judgement of one measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Within the limit.
    Pass,
    /// Within the limit, with a remark: at or above [`Caps::warn`], a waiver that no longer
    /// covers anything, or a `dev` entry ready to become `release`.
    Warn(String),
    /// Above the limit or less accurate than the comparison, without a waiver for it, or not
    /// listed in `limits.json`.
    Fail(String),
    /// Above the limit or less accurate, with a waiver for each exceedance.
    Waived(String),
    /// Not judged (`info` entry); carries the reason from `limits.json`.
    Info(String),
}

impl Verdict {
    /// The word of the console line and of the results file.
    #[must_use]
    pub fn word(&self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Warn(_) => "warn",
            Verdict::Fail(_) => "FAIL",
            Verdict::Waived(_) => "waived",
            Verdict::Info(_) => "info",
        }
    }

    /// The explanation, empty for [`Verdict::Pass`].
    #[must_use]
    pub fn detail(&self) -> &str {
        match self {
            Verdict::Pass => "",
            Verdict::Warn(d) | Verdict::Fail(d) | Verdict::Waived(d) | Verdict::Info(d) => d,
        }
    }
}

/// `ratio` (above `cap`) with as many decimals as it takes to print above the cap:
/// `1.0215x` rather than a `1.02x` that "exceeds" a limit of `1.02x`.
fn over(ratio: f64, cap: f64) -> String {
    let digits = (2..6)
        .find(|d| {
            format!("{ratio:.d$}", d = *d)
                .parse::<f64>()
                .is_ok_and(|r| r > cap)
        })
        .unwrap_or(6);
    format!("{ratio:.digits$}x")
}

/// Judges a measurement against its comparison (an established library or our fallback) by
/// its entry of `limits.json`, which must exist. The instruction ratio and the accuracy are
/// checked independently, and each exceedance needs its own waiver (`waiver`,
/// `accuracy_waiver`).
#[must_use]
#[allow(
    clippy::large_stack_frames,
    reason = "the texts of the verdict are built in place; xtask is a host tool"
)]
pub fn judge(caps: &Caps, limit: Option<&Limit>, m: &Measurement) -> Verdict {
    let Some(limit) = limit else {
        return Verdict::Fail(format!(
            "no entry for `{}` at `{}` in benches/limits.json",
            m.kernel, m.level
        ));
    };
    let cap = match limit.status {
        Status::Info => return Verdict::Info(limit.waiver.clone().unwrap_or_default()),
        Status::Release => caps.release,
        Status::Dev => caps.dev,
    };
    let status = limit.status.name();
    if m.comparison_kind.is_own() {
        return Verdict::Fail(format!(
            "{status} judged against our own code (D32, D2764): the comparison `{}` is `{}`; the entry is `info` (judged by its efficiency) until a foreign library is its comparison",
            m.compared_with,
            m.comparison_kind.name()
        ));
    }
    let ratio = m.ratio();
    let slow = ratio > cap;
    let inaccurate = m.other_snr_db - m.ours_snr_db > SNR_MARGIN_DB;
    let mut failing = Vec::new();
    let mut waived = Vec::new();
    if slow {
        let text = format!(
            "{} instructions per sample against {} exceeds the {status} limit of {cap:.2}x",
            over(ratio, cap),
            m.compared_with
        );
        match &limit.waiver {
            Some(w) => match waiver_above_cap(caps, limit.status, ratio, w) {
                Ok(()) => waived.push(format!("{text} (waiver: {w})")),
                Err(why) => failing.push(format!("{text}; {why}")),
            },
            None => failing.push(text),
        }
    }
    if inaccurate {
        let text = format!(
            "accuracy {:.1} dB is more than {SNR_MARGIN_DB:.0} dB below the {:.1} dB of {}",
            m.ours_snr_db, m.other_snr_db, m.compared_with
        );
        match &limit.accuracy_waiver {
            Some(w) => waived.push(format!("{text} (accuracy waiver: {w})")),
            None => failing.push(text),
        }
    }
    if !failing.is_empty() {
        failing.extend(waived);
        return Verdict::Fail(failing.join("; "));
    }
    if !waived.is_empty() {
        return Verdict::Waived(waived.join("; "));
    }
    let mut notes = Vec::new();
    if let Some(w) = &limit.waiver {
        notes.push(format!(
            "{ratio:.2}x is within the {status} limit of {cap:.2}x: the waiver is no longer needed ({w})"
        ));
    }
    if let Some(w) = &limit.accuracy_waiver {
        notes.push(format!(
            "the accuracy is within {SNR_MARGIN_DB:.0} dB: the accuracy waiver is no longer needed ({w})"
        ));
    }
    if limit.status == Status::Dev && ratio <= caps.warn {
        notes.push(format!(
            "{ratio:.2}x: ready to be promoted from dev to release (release limit {:.2}x)",
            caps.release
        ));
    } else if ratio > caps.warn {
        notes.push(format!(
            "{ratio:.2}x is above {:.2}x ({status} limit {cap:.2}x)",
            caps.warn
        ));
    }
    if notes.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Warn(notes.join("; "))
    }
}

/// The phrase of a waiver that keeps a `dev` entry above the `dev` limit while the maintainer
/// decides it (D2779).
pub const AWAITING: &str = "awaiting the maintainer's decision";

/// Whether a plain `waiver` may cover a ratio `worst` above the limit of `status` (D2779, the
/// maintainer's measurement order): never for `release` above [`Caps::release`] (only an
/// `accuracy_trade` keeps `release` there, up to [`crate::cycle_gate::TRADE_CAP`]: the entry is
/// `dev` with its open item), and for `dev` above [`Caps::dev`] only while the waiver names its
/// open item and says that it is [`AWAITING`]. `Err` says why not.
///
/// # Errors
/// The waiver cannot cover the ratio.
pub fn waiver_above_cap(
    caps: &Caps,
    status: Status,
    worst: f64,
    waiver: &str,
) -> Result<(), String> {
    match status {
        Status::Release if worst > caps.release => Err(format!(
            "a waiver cannot keep a `release` entry above the release limit of {:.2}x: only an `accuracy_trade` keeps it `release` (up to {:.2}x, D2760); otherwise it is `dev` with its open item (D2779)",
            caps.release,
            crate::cycle_gate::TRADE_CAP
        )),
        Status::Dev if worst > caps.dev => {
            let names_item = crate::release::references(waiver)
                .iter()
                .any(|r| r.starts_with("open item "));
            if names_item && waiver.contains(AWAITING) {
                Ok(())
            } else {
                Err(format!(
                    "a waiver keeps a `dev` entry above the dev limit of {:.2}x only while it names its open item and says \"{AWAITING}\" (D2779)",
                    caps.dev
                ))
            }
        }
        _ => Ok(()),
    }
}

/// `--strict` (the release gate of open item M13-4): why an entry blocks a release, if it
/// does. Only a `release` entry without waivers that passes (with or without a warning)
/// does not.
#[must_use]
pub fn blocks_release(limit: Option<&Limit>, verdict: &Verdict) -> Option<String> {
    match verdict {
        Verdict::Fail(d) => return Some(d.clone()),
        Verdict::Waived(d) => return Some(format!("waived: {d}")),
        Verdict::Info(d) => return Some(format!("not judged (info): {d}")),
        Verdict::Pass | Verdict::Warn(_) => {}
    }
    let limit = limit?;
    match (limit.status, &limit.waiver, &limit.accuracy_waiver) {
        (Status::Dev, ..) => Some("status `dev`".into()),
        (_, Some(w), _) | (_, None, Some(w)) => Some(format!("waiver: {w}")),
        _ => None,
    }
}

/// The outcome of a whole level.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// One line per kernel for the console (extras indented below it).
    pub lines: Vec<String>,
    /// `false` if any measurement failed.
    pub ok: bool,
    /// Entries that block a release under `--strict`: `kernel level: reason`.
    pub blockers: Vec<String>,
    /// The per-kernel records for the results file.
    pub records: BTreeMap<String, Value>,
}

/// Judges all measurements of one level.
#[must_use]
pub fn evaluate(limits: &Limits, level: &str, measurements: &[Measurement]) -> Report {
    let caps = &limits.caps;
    let mut report = Report {
        ok: true,
        ..Report::default()
    };
    for m in measurements {
        let limit = limits.get(&m.kernel, level);
        let verdict = judge(caps, limit, m);
        if matches!(verdict, Verdict::Fail(_)) {
            report.ok = false;
        }
        if let Some(why) = blocks_release(limit, &verdict) {
            report.blockers.push(format!("{} {level}: {why}", m.kernel));
        }
        let status = limit.map_or("unlisted", |l| l.status.name());
        // The goal is a figure against a library only (D2767).
        let goal = (!m.comparison_kind.is_own()).then(|| m.ratio() <= caps.goal);
        let mut line = format!(
            "{:<22} {:<12} {:>8.2} vs {:>8.2} instr/sample  ratio {:>5.2}{}  [{status}]  {}",
            m.kernel,
            m.level,
            m.ours,
            m.other,
            m.ratio(),
            match goal {
                Some(true) => " goal",
                Some(false) => "     ",
                None => " own ",
            },
            verdict.word()
        );
        if !verdict.detail().is_empty() {
            let _ = write!(line, "  {}", verdict.detail());
        }
        report.lines.push(line);
        for e in &m.extras {
            report.lines.push(format!(
                "    {:<34} {:>8.2} vs {:>8.2} instr/sample  ratio {:>5.2}  ({}{})",
                e.label,
                e.ours,
                e.other,
                e.ratio(),
                e.compared_with,
                if e.comparison_kind.is_own() {
                    ", own code: information"
                } else {
                    ""
                }
            ));
        }
        report
            .records
            .insert(m.kernel.clone(), record(m, status, &verdict, goal));
    }
    report
}

/// The record of one measurement in the results file. `goal` is `None` for a comparison with our
/// own code, which has no goal (D2767): the record then has no `goal`.
fn record(m: &Measurement, status: &str, verdict: &Verdict, goal: Option<bool>) -> Value {
    let extras: Vec<Value> = m
        .extras
        .iter()
        .map(|e| {
            json!({
                "label": e.label,
                "ours_instr_per_sample": round2(e.ours),
                "other_instr_per_sample": round2(e.other),
                "ratio": round2(e.ratio()),
                "compared_with": e.compared_with,
                "comparison_kind": e.comparison_kind.name(),
                "other_snr_db": e.other_snr_db.map(round2),
            })
        })
        .collect();
    let mut r = json!({
        "ours_instr_per_sample": round2(m.ours),
        "other_instr_per_sample": round2(m.other),
        "ratio": round2(m.ratio()),
        "compared_with": m.compared_with,
        "comparison_kind": m.comparison_kind.name(),
        "status": status,
        "verdict": verdict.word().to_lowercase(),
        "snr_db": { "ours": round2(m.ours_snr_db), "other": round2(m.other_snr_db) },
    });
    if let Some(obj) = r.as_object_mut() {
        if let Some(goal) = goal {
            obj.insert("goal".into(), json!(goal));
        }
        if !extras.is_empty() {
            obj.insert("informational".into(), Value::Array(extras));
        }
    }
    r
}

pub(crate) fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Extracts the instruction count from the output of a counter.
pub mod parse {
    use anyhow::bail;

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
    // `lo` is known to fail (0 always does), `hi` to succeed.
    let (mut lo, mut hi) = (0u64, 1u64 << 20);
    while !run(hi)? {
        lo = hi;
        hi = hi
            .checked_mul(4)
            .filter(|h| *h < 1 << 50)
            .ok_or_else(|| anyhow!("the program needs more than 2^50 fuel"))?;
    }
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

/// The instruction counts of a session of the `simd128` level.
impl Session {
    /// Instructions of one run: the least fuel of wasmtime with which it completes.
    fn count(&self, run: &Run<'_>) -> anyhow::Result<u64> {
        if self.plan.runner != Runner::Wasmtime {
            bail!(
                "the level {} has a core model and is measured in cycles (cargo measure)",
                self.plan.level
            );
        }
        least_fuel(|fuel| {
            let status = self
                .command("wasmtime", run)
                .arg("run")
                .arg("-W")
                .arg(format!("fuel={fuel}"))
                .arg(&self.binary)
                .args(&run.args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .context("cannot run wasmtime")?;
            Ok(status.success())
        })
    }

    /// Instructions per sample of `which` on `kernel`: `(count(calls) - count(0)) / (calls *
    /// samples)`, with `calls` such that every measurement covers `K * N` samples.
    fn per_sample(
        &self,
        kernel: &str,
        which: &str,
        variant: &Variant,
        volk_config: &Path,
    ) -> anyhow::Result<f64> {
        let calls = K * (N / variant.samples).max(1);
        let base = self.count(&Run {
            args: run_args(kernel, which, 0, calls, variant),
            volk_config,
        })?;
        let full = self.count(&Run {
            args: run_args(kernel, which, calls, calls, variant),
            volk_config,
        })?;
        let diff = full.checked_sub(base).ok_or_else(|| {
            anyhow!(
                "{kernel} {which}: the run with k = {calls} counted fewer instructions than k = 0"
            )
        })?;
        Ok(diff as f64 / (calls * variant.samples) as f64)
    }
}

/// What one level measured, with the facts about its references for the results file.
struct LevelResult {
    measurements: Vec<Measurement>,
    volk: Value,
}

#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the Windows file system calls are large; xtask is a host tool"
)]
fn measure_level(
    root: &Path,
    plan: &LevelPlan,
    refs: &Path,
    only: &[String],
    variants: bool,
) -> anyhow::Result<LevelResult> {
    let session = session(root, plan, refs)?;
    let volk = session.query(&["volk-info"])?;
    let mut listed: Vec<String> = session
        .plain(&Run {
            args: vec!["list".to_string()],
            volk_config: &session.volk_config,
        })?
        .lines()
        .map(str::to_string)
        .collect();
    listed.sort_unstable();
    let mut known: Vec<&str> = KERNELS.iter().map(|k| k.name).collect();
    known.sort_unstable();
    if listed != known {
        bail!("the measurement binary lists {listed:?}, xtask knows {known:?}");
    }
    let mut out = Vec::new();
    for spec in &KERNELS {
        if !only.is_empty() && !only.iter().any(|k| k == spec.name) {
            continue;
        }
        if !spec.family.measured_at(plan) {
            continue;
        }
        let m = measure_kernel(&session, spec, variants)?;
        out.push(m);
    }
    Ok(LevelResult {
        measurements: out,
        volk,
    })
}

#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command is large; xtask is a host tool"
)]
fn measure_kernel(
    session: &Session,
    spec: &KernelSpec,
    variants: bool,
) -> anyhow::Result<Measurement> {
    let plan = &session.plan;
    let (gate, extras) = comparisons(plan, spec);
    let info = session.query(&["info", spec.name])?;
    let impls = info.get("impls").and_then(Value::as_object);
    let describe = |r: Reference| -> Option<String> {
        impls
            .and_then(|i| i.get(r.implementation()))
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let label = describe(gate).ok_or_else(|| {
        anyhow!(
            "{} at {}: the comparison `{}` is not in the measurement binary (missing C reference?)",
            spec.name,
            plan.level,
            gate.implementation()
        )
    })?;
    println!("==> {} {} (ours vs {label})", plan.level, spec.name);
    let pinned = session.volk_config.as_path();
    let standard = Variant::standard(spec);
    // The in-place FFTs restore their input before every call, on both sides.
    let restore = if spec.family.is_fft() {
        session.per_sample(spec.name, "restore", &standard, pinned)?
    } else {
        0.0
    };
    let count = |which: &str, variant: &Variant, config: &Path| -> anyhow::Result<f64> {
        Ok(session.per_sample(spec.name, which, variant, config)? - restore)
    };
    let measured = spec.family.measured();
    let ours = count(measured, &standard, pinned)?;
    let ours_snr_db = session.snr(spec.name, measured, pinned)?;
    let other = count(gate.implementation(), &standard, pinned)?;
    let other_snr_db = session.snr(spec.name, gate.implementation(), pinned)?;
    let mut m = Measurement {
        kernel: spec.name.to_string(),
        level: plan.level.to_string(),
        ours,
        other,
        compared_with: label,
        comparison_kind: gate.kind(),
        ours_snr_db,
        other_snr_db,
        extras: Vec::new(),
    };
    for r in extras {
        if let Reference::OwnKernel(kernel) = r {
            m.extras
                .push(own_kernel_extra(session, kernel, ours, &standard, pinned)?);
            continue;
        }
        let Some(name) = describe(r) else {
            println!("    ({} is not in this build)", r.implementation());
            continue;
        };
        m.extras.push(Extra {
            label: format!("information: {}", r.implementation()),
            ours,
            other: count(r.implementation(), &standard, pinned)?,
            compared_with: name,
            comparison_kind: r.kind(),
            other_snr_db: Some(session.snr(spec.name, r.implementation(), pinned)?),
        });
    }
    if variants {
        for v in buffer_variants(spec, &standard) {
            m.extras.push(Extra {
                label: v.label(),
                ours: count(measured, &v, pinned)?,
                other: count(gate.implementation(), &v, pinned)?,
                compared_with: m.compared_with.clone(),
                comparison_kind: m.comparison_kind,
                other_snr_db: None,
            });
        }
    }
    Ok(m)
}

/// The informational variants of the buffers of an entry: the small blocks and the misaligned
/// buffers, where the entry has them.
fn buffer_variants(spec: &KernelSpec, standard: &Variant) -> Vec<Variant> {
    let mut list = Vec::new();
    if spec.small_n {
        list.extend(SMALL_N.map(|n| Variant {
            samples: n,
            misaligned: false,
            explicit_n: true,
        }));
    }
    if spec.misaligned {
        list.push(Variant {
            misaligned: true,
            ..standard.clone()
        });
    }
    list
}

/// Our kernel on the workload of another entry of the table (`ours` of `kernel`, per sample of
/// its own input), as information next to `ours` of the entry (D69).
fn own_kernel_extra(
    session: &Session,
    kernel: &str,
    ours: f64,
    variant: &Variant,
    config: &Path,
) -> anyhow::Result<Extra> {
    Ok(Extra {
        label: format!("information: ours:{kernel}"),
        ours,
        other: session.per_sample(kernel, "ours", variant, config)?,
        compared_with: format!("ours:{kernel}"),
        comparison_kind: ComparisonKind::OwnKernel,
        other_snr_db: Some(session.snr(kernel, "ours", config)?),
    })
}

/// The command line of `cargo xtask count`.
#[allow(
    clippy::struct_excessive_bools,
    reason = "one field per command-line flag"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// `--platform`.
    pub platform: Option<String>,
    /// `--level` (repeatable): only these levels of the platform.
    pub levels: Vec<String>,
    /// `--kernel` (repeatable): only these kernels.
    pub kernels: Vec<String>,
    /// Not `--no-write`.
    pub write: bool,
    /// `--strict`: every entry must be `release` without waivers and pass.
    pub strict: bool,
    /// Not `--no-variants`: also the informational misaligned and small-block runs.
    pub variants: bool,
    /// `--merge`: with `--kernel`, write the records of the measured kernels into the existing
    /// results file and keep everything else of it (a full run measures all kernels against
    /// all references, which a new kernel with no library to compare with does not need).
    pub merge: bool,
    /// `--rejudge`: measure nothing; judge the stored numbers of the results file again by
    /// `benches/limits.json` and the plan, and write the derived fields ([`rejudge`]).
    pub rejudge: bool,
}

const USAGE: &str = "usage: cargo xtask count --platform <wasm32|cortex-m4> [--level L]... [--kernel NAME]... [--no-write] [--no-variants] [--merge] [--strict] [--rejudge]";

impl Options {
    /// Parses the arguments after `count`.
    ///
    /// # Errors
    /// An unknown option or a missing value.
    pub fn parse(args: &[&str]) -> anyhow::Result<Options> {
        let mut o = Options {
            write: true,
            variants: true,
            ..Options::default()
        };
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            let mut value = || {
                it.next()
                    .map(|s| (*s).to_string())
                    .ok_or_else(|| anyhow!("`{arg}` needs a value; {USAGE}"))
            };
            match *arg {
                "--platform" => o.platform = Some(value()?),
                "--level" => o.levels.push(value()?),
                "--kernel" => o.kernels.push(value()?),
                "--no-write" => o.write = false,
                "--no-variants" => o.variants = false,
                "--merge" => o.merge = true,
                "--strict" => o.strict = true,
                "--rejudge" => o.rejudge = true,
                other => bail!("unknown option `{other}`; {USAGE}"),
            }
        }
        Ok(o)
    }
}

/// The plans to run for the options: the selected levels of the platform (all without
/// `--level`). Only the levels without a core model are counted.
///
/// # Errors
/// An unknown platform or level, or a platform whose levels are measured in cycles.
pub fn select_plans(platform: &str, levels: &[String]) -> anyhow::Result<Vec<&'static LevelPlan>> {
    if CYCLE_PLATFORMS.contains(&platform) {
        bail!(
            "the levels of `{platform}` are measured in cycles per sample, not counted: use `cargo measure <kernel> --platform {platform}` (D2700)"
        );
    }
    let all: Vec<&'static LevelPlan> = PLANS
        .iter()
        .filter(|p| p.platform == platform && p.runner == Runner::Wasmtime)
        .collect();
    if all.is_empty() {
        bail!("unknown platform `{platform}`; {USAGE}");
    }
    for l in levels {
        if !all.iter().any(|p| p.level == l) {
            bail!("the platform `{platform}` has no level `{l}`");
        }
    }
    Ok(all
        .into_iter()
        .filter(|p| levels.is_empty() || levels.iter().any(|l| l == p.level))
        .collect())
}

/// Measures the levels of `plans` one after the other and returns their results in the same
/// order.
fn measure_levels(
    root: &Path,
    plans: &[&'static LevelPlan],
    options: &Options,
) -> anyhow::Result<Vec<LevelResult>> {
    let mut refs: BTreeMap<&str, std::path::PathBuf> = BTreeMap::new();
    let mut results = Vec::new();
    for plan in plans {
        if !refs.contains_key(plan.refs) {
            refs.insert(plan.refs, references(root, plan.refs)?);
        }
        let refs = refs
            .get(plan.refs)
            .ok_or_else(|| anyhow!("no references for {}", plan.refs))?;
        results.push(measure_level(
            root,
            plan,
            refs,
            &options.kernels,
            options.variants,
        )?);
    }
    Ok(results)
}

/// Runs `cargo xtask count ...`.
///
/// ```text
/// cargo xtask count --platform <wasm32|cortex-m4> [--level L]... [--kernel NAME]... [--no-write] [--no-variants] [--merge] [--strict]
/// ```
///
/// # Errors
/// A tool is missing or fails; the return value is `Ok(false)` if a measurement exceeds its
/// limit (or, with `--strict`, if any entry blocks a release).
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the plan table are large; xtask is a host tool"
)]
pub fn run(root: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let options = Options::parse(args)?;
    if options.platform.as_deref() == Some("cortex-m4") {
        return crate::count_embedded::run(root, &options.kernels, options.write);
    }
    let Some(platform) = options.platform.as_deref() else {
        bail!("{USAGE}");
    };
    let plans = select_plans(platform, &options.levels)?;
    if options.rejudge {
        return rejudge_file(root, platform, options.write);
    }
    for k in &options.kernels {
        if !KERNELS.iter().any(|s| s.name == k) {
            bail!("unknown kernel `{k}`");
        }
    }
    let limits_path = root.join("benches").join("limits.json");
    let limits = Limits::parse(
        &std::fs::read_to_string(&limits_path)
            .with_context(|| format!("cannot read {}", limits_path.display()))?,
    )?;
    let mut all_ok = true;
    let mut blockers = Vec::new();
    let mut levels = serde_json::Map::new();
    let mut volk_flags = Value::Null;
    let results = measure_levels(root, &plans, &options)?;
    for (plan, result) in plans.iter().zip(&results) {
        let judged = evaluate(&limits, plan.level, &result.measurements);
        for line in &judged.lines {
            println!("{line}");
        }
        all_ok &= judged.ok;
        blockers.extend(judged.blockers);
        levels.insert(
            plan.level.to_string(),
            json!({
                "counter": "wasmtime fuel",
                "judged_against": "references",
                "volk": {
                    "machine": result.volk.get("machine"),
                    "alignment": result.volk.get("alignment"),
                    "pinned": plan.volk_pin.config().lines().collect::<Vec<_>>(),
                },
                "kernels": judged.records,
            }),
        );
        if let Some(flags) = result.volk.get("compiler_flags") {
            volk_flags = flags.clone();
        }
    }
    if options.write && options.kernels.is_empty() && options.levels.is_empty() {
        write_results(root, platform, &limits.caps, &levels, &volk_flags)?;
    } else if options.write && options.merge {
        if options.kernels.is_empty() {
            bail!("--merge needs --kernel");
        }
        merge_results(root, platform, &levels)?;
    }
    if options.strict && !blockers.is_empty() {
        eprintln!("--strict: {} entries block a release:", blockers.len());
        for b in &blockers {
            eprintln!("  {b}");
        }
        all_ok = false;
    }
    if all_ok {
        println!("count {platform}: ok");
    } else {
        eprintln!("count {platform}: limits exceeded");
    }
    Ok(all_ok)
}

/// The derived fields of a record that [`rejudge`] writes: everything else is a measured number
/// or a name and stays as it is.
const DERIVED: [&str; 4] = ["status", "verdict", "goal", "comparison_kind"];

/// The outcome of [`rejudge`].
#[derive(Debug, Default)]
pub struct Rejudged {
    /// The document with the derived fields judged again.
    pub doc: Value,
    /// `kernel level: field old -> new` for every derived field that changed.
    pub changes: Vec<String>,
    /// The records that fail their judgement: `kernel level: reason`.
    pub failures: Vec<String>,
}

/// The measurement of a stored record of `kernel` at the level of `plan`, with the comparison
/// kinds of the plan (D2767): the judged comparison's for the record and the variants, and for
/// an informational comparison (`information: <implementation>`) that of its reference.
///
/// # Errors
/// A record without its numbers, or an informational comparison that the plan does not have.
fn stored_measurement(
    plan: &LevelPlan,
    spec: &KernelSpec,
    r: &Value,
) -> anyhow::Result<Measurement> {
    let num = |v: &Value, k: &str| {
        v.get(k)
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow!("`{}` at `{}` has no `{k}`", spec.name, plan.level))
    };
    let text = |v: &Value| {
        v.get("compared_with")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let (gate, extras) = comparisons(plan, spec);
    let snr = r.get("snr_db").unwrap_or(&Value::Null);
    let mut m = Measurement {
        kernel: spec.name.to_string(),
        level: plan.level.to_string(),
        ours: num(r, "ours_instr_per_sample")?,
        other: num(r, "other_instr_per_sample")?,
        compared_with: text(r),
        comparison_kind: gate.kind(),
        ours_snr_db: num(snr, "ours")?,
        other_snr_db: num(snr, "other")?,
        extras: Vec::new(),
    };
    for e in r
        .get("informational")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let label = e
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let comparison_kind = match label.strip_prefix("information: ") {
            None => gate.kind(),
            Some(name) => extras
                .iter()
                .find(|x| match x {
                    Reference::OwnKernel(k) => name.strip_prefix("ours:") == Some(*k),
                    other => other.implementation() == name,
                })
                .map(|x| x.kind())
                .ok_or_else(|| {
                    anyhow!(
                        "`{}` at `{}`: the plan has no informational comparison `{name}`",
                        spec.name,
                        plan.level
                    )
                })?,
        };
        m.extras.push(Extra {
            label,
            ours: num(e, "ours_instr_per_sample")?,
            other: num(e, "other_instr_per_sample")?,
            compared_with: text(e),
            comparison_kind,
            other_snr_db: e.get("other_snr_db").and_then(Value::as_f64),
        });
    }
    Ok(m)
}

/// Judges the stored numbers of a results document of the counted levels again by `limits` and
/// the plan, without a measurement (D2767): per record the status of `limits.json`, the verdict,
/// the goal (a library comparison only) and the comparison kinds are written anew; the numbers,
/// the names and the rest of the document stay as they are. The method's limits are the caps of
/// `limits`.
///
/// # Errors
/// A level that is not counted, a record of an unknown kernel, or a record without its numbers.
#[allow(
    clippy::large_stack_frames,
    reason = "the plan table and serde_json values are large; xtask is a host tool"
)]
pub fn rejudge(doc: &Value, limits: &Limits) -> anyhow::Result<Rejudged> {
    let mut out = Rejudged {
        doc: doc.clone(),
        ..Rejudged::default()
    };
    let caps = limits.caps;
    if let Some(method) = out.doc.get_mut("method").and_then(Value::as_object_mut) {
        let new = json!({ "release": caps.release, "dev": caps.dev, "warn": caps.warn, "goal": caps.goal });
        if method.get("limits") != Some(&new) {
            out.changes.push(format!(
                "method.limits: {} -> {new}",
                method.get("limits").unwrap_or(&Value::Null)
            ));
            method.insert("limits".into(), new);
        }
    }
    let Some(levels) = out.doc.get_mut("levels").and_then(Value::as_object_mut) else {
        bail!("the results document has no `levels`");
    };
    for (level, l) in levels.iter_mut() {
        let plan = PLANS
            .iter()
            .find(|p| p.level == level && p.runner == Runner::Wasmtime)
            .ok_or_else(|| anyhow!("the level `{level}` is not counted"))?;
        let Some(kernels) = l.get_mut("kernels").and_then(Value::as_object_mut) else {
            continue;
        };
        for (kernel, r) in kernels.iter_mut() {
            let spec = KERNELS
                .iter()
                .find(|s| s.name == kernel)
                .ok_or_else(|| anyhow!("`{kernel}` at `{level}` is not an entry of the table"))?;
            let m = stored_measurement(plan, spec, r)?;
            let report = evaluate(limits, level, std::slice::from_ref(&m));
            let new = report
                .records
                .get(kernel.as_str())
                .cloned()
                .unwrap_or(Value::Null);
            if let Verdict::Fail(why) = judge(&caps, limits.get(kernel, level), &m) {
                out.failures.push(format!("{kernel} {level}: {why}"));
            }
            let Some(obj) = r.as_object_mut() else {
                continue;
            };
            for key in DERIVED {
                let (old, fresh) = (obj.get(key).cloned(), new.get(key).cloned());
                if old != fresh {
                    out.changes.push(format!(
                        "{kernel} {level}: {key} {} -> {}",
                        old.unwrap_or(Value::Null),
                        fresh.clone().unwrap_or(Value::Null)
                    ));
                    match fresh {
                        Some(v) => obj.insert(key.into(), v),
                        None => obj.remove(key),
                    };
                }
            }
            let fresh_extras = new.get("informational").and_then(Value::as_array);
            if let (Some(extras), Some(fresh)) = (
                obj.get_mut("informational").and_then(Value::as_array_mut),
                fresh_extras,
            ) {
                for (e, f) in extras.iter_mut().zip(fresh) {
                    let Some(e) = e.as_object_mut() else { continue };
                    for key in ["target", "meets_target"] {
                        if e.remove(key).is_some() {
                            out.changes
                                .push(format!("{kernel} {level}: informational {key} removed"));
                        }
                    }
                    if e.get("comparison_kind") != f.get("comparison_kind") {
                        out.changes.push(format!(
                            "{kernel} {level}: informational comparison_kind of {} -> {}",
                            e.get("label").unwrap_or(&Value::Null),
                            f["comparison_kind"]
                        ));
                        e.insert("comparison_kind".into(), f["comparison_kind"].clone());
                    }
                }
            }
        }
    }
    Ok(out)
}

/// The path of the results file of a counted platform.
fn results_path(root: &Path, platform: &str) -> std::path::PathBuf {
    root.join("benches")
        .join("results")
        .join(format!("{platform}.json"))
}

/// `cargo xtask count --platform <p> --rejudge`: [`rejudge`] on the committed results file,
/// written back with `write`. `Ok(false)` if a record fails its judgement.
#[allow(
    clippy::large_stack_frames,
    reason = "the Windows file system calls and the records are large; xtask is a host tool"
)]
fn rejudge_file(root: &Path, platform: &str, write: bool) -> anyhow::Result<bool> {
    let path = results_path(root, platform);
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?,
    )?;
    let limits = Limits::parse(&std::fs::read_to_string(
        root.join("benches").join("limits.json"),
    )?)?;
    let r = rejudge(&doc, &limits)?;
    for c in &r.changes {
        println!("changed: {c}");
    }
    for f in &r.failures {
        eprintln!("FAIL: {f}");
    }
    if write && !r.changes.is_empty() {
        let mut text = serde_json::to_string_pretty(&r.doc)?;
        text.push('\n');
        std::fs::write(&path, text)?;
        println!("rejudged {}: {} changes", path.display(), r.changes.len());
    }
    Ok(r.failures.is_empty())
}

/// The check of `cargo xtask check` on the committed counts of `wasm32` (D2767): the derived
/// fields of every record are those that [`rejudge`] gives (the status of `limits.json`, the
/// verdict, the goal of a library comparison only, the comparison kinds of the plan), and no
/// record fails its judgement (a `release` or `dev` entry against our own code, a waiver above
/// its cap, D2779).
///
/// # Errors
/// The results file or `benches/limits.json` cannot be read or parsed.
#[allow(
    clippy::large_stack_frames,
    reason = "the Windows file system calls and the records are large; xtask is a host tool"
)]
pub fn check_counts(root: &Path) -> anyhow::Result<Vec<crate::scan::Finding>> {
    let path = results_path(root, "wasm32");
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?,
    )?;
    let limits = Limits::parse(&std::fs::read_to_string(
        root.join("benches").join("limits.json"),
    )?)?;
    let r = rejudge(&doc, &limits)?;
    let finding = |message: String| crate::scan::Finding {
        path: std::path::PathBuf::from("benches/results/wasm32.json"),
        line: 0,
        message,
    };
    let mut out: Vec<_> = r
        .changes
        .iter()
        .map(|c| {
            finding(format!(
                "stale derived field ({c}); run `cargo xtask count --platform wasm32 --rejudge` (D2767)"
            ))
        })
        .collect();
    out.extend(
        r.failures
            .iter()
            .map(|f| finding(format!("{f} (D14, D2767)"))),
    );
    Ok(out)
}

/// The manifest of the C references of `platform` (`benches/target/refs/<refs>/refs.json`,
/// written by `benches/refs/build.sh`), or `null` when there is none.
fn reference_manifest(root: &Path, platform: &str) -> Value {
    let refs_platform = PLANS
        .iter()
        .find(|p| p.platform == platform)
        .map_or(platform, |p| p.refs);
    let manifest = root
        .join("benches")
        .join("target")
        .join("refs")
        .join(refs_platform)
        .join("refs.json");
    std::fs::read_to_string(&manifest)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null)
}

/// Adds to the `references` section of a results file what the manifest of the references has
/// and the section lacks: a library (`libraries.<name>`) or a top-level fact (a compiler such as
/// `kfr_cxx`). An entry the section already has is kept, because the records of the kernels
/// that were not measured again were made with it; the names of the entries that differ from
/// the manifest are returned, so that the caller can say that a full run is due.
fn merge_references(section: &mut Value, manifest: &Value) -> Vec<String> {
    let mut differ = Vec::new();
    let Some(manifest) = manifest.as_object() else {
        return differ;
    };
    if !section.is_object() {
        *section = json!({});
    }
    let Some(section) = section.as_object_mut() else {
        return differ;
    };
    for (key, value) in manifest {
        if key == "libraries" {
            let Some(libraries) = value.as_object() else {
                continue;
            };
            let target = section.entry("libraries").or_insert_with(|| json!({}));
            if !target.is_object() {
                *target = json!({});
            }
            let Some(target) = target.as_object_mut() else {
                continue;
            };
            for (name, library) in libraries {
                match target.get(name) {
                    None => {
                        target.insert(name.clone(), library.clone());
                    }
                    Some(old) if old != library => differ.push(format!("libraries.{name}")),
                    Some(_) => {}
                }
            }
        } else {
            match section.get(key) {
                None => {
                    section.insert(key.clone(), value.clone());
                }
                Some(old) if old != value => differ.push(key.clone()),
                Some(_) => {}
            }
        }
    }
    differ
}

/// Writes the records of the measured kernels into the results file of `platform`: for every
/// measured level, the records of the kernels in `measured` replace or add the entries of the
/// file, and the `references` section gets the libraries and compilers of the manifest of the
/// references that it lacks (a reference that is new since the last full run); every other
/// entry, and the rest of the file, stays as it is.
///
/// # Errors
/// The file cannot be read or written, or it has no entry for a measured level.
fn merge_results(
    root: &Path,
    platform: &str,
    measured: &serde_json::Map<String, Value>,
) -> anyhow::Result<()> {
    let path = root
        .join("benches")
        .join("results")
        .join(format!("{platform}.json"));
    let mut doc: Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?,
    )?;
    for (level, new) in measured {
        let Some(records) = new["kernels"].as_object() else {
            continue;
        };
        let kernels = doc
            .get_mut("levels")
            .and_then(|l| l.get_mut(level.as_str()))
            .and_then(|l| l.get_mut("kernels"))
            .and_then(Value::as_object_mut)
            .ok_or_else(|| anyhow!("{} has no level {level}", path.display()))?;
        for (kernel, record) in records {
            kernels.insert(kernel.clone(), record.clone());
        }
    }
    let manifest = reference_manifest(root, platform);
    if let Some(object) = doc.as_object_mut() {
        let section = object.entry("references").or_insert(Value::Null);
        let differ = merge_references(section, &manifest);
        if !differ.is_empty() {
            println!(
                "note: {} of the references in {} differ from the manifest of the references and are kept until a full run: {}",
                differ.len(),
                path.display(),
                differ.join(", ")
            );
        }
    }
    let mut text = serde_json::to_string_pretty(&doc)?;
    text.push('\n');
    std::fs::write(&path, text)?;
    println!("merged the measured kernels into {}", path.display());
    Ok(())
}

#[allow(
    clippy::large_stack_frames,
    reason = "serde_json values and the file system calls are large; xtask is a host tool"
)]
fn write_results(
    root: &Path,
    platform: &str,
    caps: &Caps,
    levels: &serde_json::Map<String, Value>,
    volk_flags: &Value,
) -> anyhow::Result<()> {
    let references = reference_manifest(root, platform);
    let host_cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_default();
    let doc = json!({
        "platform": platform,
        "method": {
            "k": K,
            "samples": N,
            "counted": "(count(k = K) - count(k = 0)) / (K * samples), minus the restore of the input for the FFTs",
            "limits": { "release": caps.release, "dev": caps.dev, "warn": caps.warn, "goal": caps.goal },
        },
        "host_cpu": host_cpu,
        "references": references,
        "volk_compiler_flags": volk_flags,
        "levels": levels,
    });
    let dir = root.join("benches").join("results");
    std::fs::create_dir_all(&dir)?;
    let mut text = serde_json::to_string_pretty(&doc)?;
    text.push('\n');
    std::fs::write(dir.join(format!("{platform}.json")), text)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn merging_adds_the_new_references_and_keeps_the_measured_ones() {
        let mut section = json!({
            "cc": "gcc 15",
            "libraries": { "volk": { "version": "3.3.0" }, "fftw": { "version": "3.3.10" } },
        });
        let manifest = json!({
            "cc": "gcc 16",
            "kfr_cxx": "g++ 15",
            "libraries": {
                "volk": { "version": "3.3.0" },
                "fftw": { "version": "3.3.11" },
                "kfr": { "version": "7.2.0" },
            },
        });
        let differ = merge_references(&mut section, &manifest);
        assert_eq!(differ, vec!["cc".to_string(), "libraries.fftw".to_string()]);
        assert_eq!(section["cc"], "gcc 15");
        assert_eq!(section["kfr_cxx"], "g++ 15");
        assert_eq!(section["libraries"]["fftw"]["version"], "3.3.10");
        assert_eq!(section["libraries"]["kfr"]["version"], "7.2.0");
        assert_eq!(section["libraries"]["volk"]["version"], "3.3.0");
        let mut empty = Value::Null;
        assert_eq!(
            merge_references(&mut empty, &manifest),
            Vec::<String>::new()
        );
        assert_eq!(empty, manifest);
        let mut kept = json!({ "cc": "gcc 15" });
        assert_eq!(
            merge_references(&mut kept, &Value::Null),
            Vec::<String>::new()
        );
        assert_eq!(kept, json!({ "cc": "gcc 15" }));
    }

    fn m(kernel: &str, ours: f64, other: f64) -> Measurement {
        Measurement {
            kernel: kernel.into(),
            level: "avx2".into(),
            ours,
            other,
            compared_with: "volk".into(),
            comparison_kind: ComparisonKind::Library,
            ours_snr_db: 130.0,
            other_snr_db: 130.0,
            extras: Vec::new(),
        }
    }

    fn limits(text: &str) -> Limits {
        Limits::parse(text).unwrap()
    }

    /// Caps with a warning band below the release limit, to test it.
    const CAPS: Caps = Caps {
        release: 1.15,
        dev: 1.50,
        warn: 1.10,
        goal: 1.00,
    };

    fn limit(status: Status, waiver: Option<&str>, accuracy: Option<&str>) -> Limit {
        Limit {
            status,
            waiver: waiver.map(Into::into),
            accuracy_waiver: accuracy.map(Into::into),
            accuracy_trade: None,
        }
    }

    fn judged(l: &Limit, x: &Measurement) -> Verdict {
        judge(&CAPS, Some(l), x)
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
    fn the_default_caps_are_those_of_d14() {
        let caps = Caps::default();
        assert_eq!((caps.release, caps.dev, caps.goal), (1.10, 1.50, 1.00));
        assert_eq!(caps.warn, caps.release);
    }

    #[test]
    fn the_caps_set_the_release_and_dev_limits_and_the_warning_band() {
        let rel = Limit::release();
        let dev = limit(Status::Dev, None, None);
        assert_eq!(
            judged(&rel, &m("k", 11.4, 10.0)),
            Verdict::Warn("1.14x is above 1.10x (release limit 1.15x)".into())
        );
        assert_eq!(judged(&rel, &m("k", 10.9, 10.0)), Verdict::Pass);
        assert!(matches!(
            judged(&rel, &m("k", 11.6, 10.0)),
            Verdict::Fail(_)
        ));
        assert_eq!(
            judged(&dev, &m("k", 13.0, 10.0)),
            Verdict::Warn("1.30x is above 1.10x (dev limit 1.50x)".into())
        );
        assert!(matches!(
            judged(&dev, &m("k", 14.9, 10.0)),
            Verdict::Warn(_)
        ));
        assert!(matches!(
            judged(&dev, &m("k", 15.1, 10.0)),
            Verdict::Fail(_)
        ));
    }

    #[test]
    fn the_caps_come_from_limits_json() {
        let l = limits(
            r#"{ "_limits": { "release": 1.10 }, "k": { "avx2": { "status": "release" } } }"#,
        );
        assert_eq!(l.caps.release, 1.10);
        assert_eq!(l.caps.dev, 1.50);
        let report = evaluate(&l, "avx2", &[m("k", 11.2, 10.0)]);
        assert!(!report.ok, "{}", report.lines[0]);
        assert!(Limits::parse(r#"{ "_limits": { "release": 0 } }"#).is_err());
        assert!(Limits::parse(r#"{ "_limits": { "fast": 1.0 } }"#).is_err());
    }

    #[test]
    fn a_dev_entry_within_the_warning_threshold_is_ready_for_release() {
        let dev = limit(Status::Dev, None, None);
        assert!(
            matches!(judged(&dev, &m("k", 10.5, 10.0)), Verdict::Warn(w) if w.contains("promoted"))
        );
    }

    #[test]
    fn a_kernel_that_is_not_listed_fails() {
        let l = limits("{}");
        assert_eq!(l.get("anything", "avx2"), None);
        let report = evaluate(&l, "avx2", &[m("anything", 5.0, 10.0)]);
        assert!(!report.ok);
        assert!(report.lines[0].contains("no entry"), "{}", report.lines[0]);
    }

    #[test]
    fn a_failing_ratio_is_printed_above_the_limit() {
        let l = limit(Status::Release, None, None);
        let v = judge(&CAPS, Some(&l), &m("rotate", 7.505, 6.52));
        assert!(
            v.detail().starts_with("1.151x instructions per sample"),
            "{}",
            v.detail()
        );
        assert!(v.detail().contains("limit of 1.15x"), "{}", v.detail());
        assert_eq!(over(1.3, 1.15), "1.30x");
    }

    #[test]
    fn lower_accuracy_than_the_comparison_fails() {
        let rel = Limit::release();
        let mut x = m("k", 10.0, 10.0);
        x.ours_snr_db = 128.5;
        assert!(matches!(judged(&rel, &x), Verdict::Fail(_)));
        x.ours_snr_db = 129.5;
        assert_eq!(judged(&rel, &x), Verdict::Pass);
        x.ours_snr_db = 140.0;
        assert_eq!(judged(&rel, &x), Verdict::Pass);
    }

    #[test]
    fn a_waiver_reports_but_does_not_fail() {
        let l = limits(
            r#"{ "fft_4096": { "avx2": { "status": "dev", "waiver": "open item M13-1, awaiting the maintainer's decision" } } }"#,
        );
        let report = evaluate(&l, "avx2", &[m("fft_4096", 40.0, 10.0)]);
        assert!(report.ok);
        assert!(report.lines[0].contains("waived") && report.lines[0].contains("M13-1"));
        assert_eq!(report.records["fft_4096"]["verdict"], "waived");
    }

    #[test]
    fn a_ratio_waiver_does_not_hide_an_accuracy_regression() {
        let l = limit(
            Status::Dev,
            Some("open item M13-2, awaiting the maintainer's decision"),
            None,
        );
        let mut x = m("k", 40.0, 10.0);
        x.ours_snr_db = 120.0;
        match judged(&l, &x) {
            Verdict::Fail(d) => assert!(d.contains("accuracy") && d.contains("M13-2"), "{d}"),
            v => panic!("{v:?}"),
        }
        let both = limit(
            Status::Dev,
            Some("open item M13-2, awaiting the maintainer's decision"),
            Some("M13-3"),
        );
        assert!(matches!(judged(&both, &x), Verdict::Waived(_)));
        // An accuracy waiver alone does not cover the instruction count either.
        let acc = limit(Status::Dev, None, Some("M13-3"));
        assert!(matches!(judged(&acc, &x), Verdict::Fail(d) if d.contains("exceeds")));
    }

    #[test]
    fn a_waiver_that_covers_nothing_is_reported() {
        let l = limit(Status::Dev, Some("M13-2"), Some("M13-3"));
        match judged(&l, &m("k", 9.0, 10.0)) {
            Verdict::Warn(w) => {
                assert!(w.contains("waiver is no longer needed (M13-2)"), "{w}");
                assert!(
                    w.contains("accuracy waiver is no longer needed (M13-3)"),
                    "{w}"
                );
            }
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn strict_mode_blocks_everything_but_clean_release_entries() {
        let rel = Limit::release();
        assert_eq!(
            blocks_release(Some(&rel), &judged(&rel, &m("k", 9.0, 10.0))),
            None
        );
        assert_eq!(
            blocks_release(Some(&rel), &judged(&rel, &m("k", 11.2, 10.0))),
            None
        );
        let dev = limit(Status::Dev, None, None);
        assert!(blocks_release(Some(&dev), &judged(&dev, &m("k", 9.0, 10.0))).is_some());
        let stale = limit(Status::Release, Some("M13-1"), None);
        assert!(blocks_release(Some(&stale), &judged(&stale, &m("k", 9.0, 10.0))).is_some());
        let info = limit(Status::Info, Some("no SSE code in VOLK"), None);
        assert!(blocks_release(Some(&info), &judged(&info, &m("k", 9.0, 10.0))).is_some());
        assert!(blocks_release(None, &judge(&CAPS, None, &m("k", 1.0, 1.0))).is_some());
        let l = limits(
            r#"{ "k": { "avx2": { "status": "dev" } }, "j": { "avx2": { "status": "release" } } }"#,
        );
        let report = evaluate(&l, "avx2", &[m("k", 9.0, 10.0), m("j", 9.0, 10.0)]);
        assert!(report.ok);
        assert_eq!(report.blockers.len(), 1, "{:?}", report.blockers);
        assert!(report.blockers[0].starts_with("k avx2"));
    }

    #[test]
    fn an_info_entry_is_reported_but_never_judged() {
        let l = limits(
            r#"{ "rotate": { "sse2": { "status": "info", "waiver": "VOLK has no SSE rotator" } } }"#,
        );
        let report = evaluate(&l, "sse2", &[m("rotate", 400.0, 10.0)]);
        assert!(report.ok);
        assert!(report.lines[0].contains("info") && report.lines[0].contains("SSE rotator"));
        assert_eq!(report.records["rotate"]["verdict"], "info");
        assert_eq!(report.records["rotate"]["status"], "info");
    }

    #[test]
    fn an_own_kernel_comparison_is_bare_information_without_a_target() {
        // `own_target` (D102) is no field any more: no target, no verdict on our own code (D2767).
        assert!(
            Limits::parse(r#"{ "k": { "avx2": { "status": "release", "own_target": 0.60 } } }"#)
                .unwrap_err()
                .to_string()
                .contains("own_target")
        );
        let l = limits(r#"{ "k": { "avx2": { "status": "release" } } }"#);
        let mut x = m("k", 9.0, 10.0);
        x.extras.push(Extra {
            label: "information: ours:w".into(),
            ours: 9.0,
            other: 20.0,
            compared_with: "ours:w".into(),
            comparison_kind: ComparisonKind::OwnKernel,
            other_snr_db: None,
        });
        let report = evaluate(&l, "avx2", &[x]);
        let info = &report.records["k"]["informational"][0];
        assert_eq!(info["ratio"], 0.45);
        assert_eq!(info["comparison_kind"], "own_kernel");
        assert!(info.get("target").is_none() && info.get("meets_target").is_none());
        assert!(!report.lines[1].contains("target") && !report.lines[1].contains("met"));
        assert!(report.lines[1].contains("own code: information"));
    }

    #[test]
    fn release_and_dev_against_our_own_code_fail_and_have_no_goal() {
        for (status, kind) in [
            ("release", ComparisonKind::OwnFallback),
            ("dev", ComparisonKind::OwnFallback),
            ("release", ComparisonKind::OwnKernel),
        ] {
            let l = limits(&format!(
                r#"{{ "k": {{ "simd128": {{ "status": "{status}" }} }} }}"#
            ));
            let mut x = m("k", 0.5, 1.0);
            x.level = "simd128".into();
            x.compared_with = "sdrk fallback".into();
            x.comparison_kind = kind;
            let report = evaluate(&l, "simd128", &[x]);
            assert!(!report.ok, "{status} {kind:?}");
            assert!(
                report.lines[0].contains("judged against our own code"),
                "{}",
                report.lines[0]
            );
            assert!(report.records["k"].get("goal").is_none());
            assert_eq!(report.records["k"]["comparison_kind"], kind.name());
        }
        // `info` against our own fallback is information: no failure, no goal.
        let l = limits(r#"{ "k": { "simd128": { "status": "info" } } }"#);
        let mut x = m("k", 0.5, 1.0);
        x.comparison_kind = ComparisonKind::OwnFallback;
        let report = evaluate(&l, "simd128", &[x]);
        assert!(report.ok);
        assert!(report.records["k"].get("goal").is_none());
    }

    #[test]
    fn a_waiver_keeps_no_entry_above_its_cap_except_dev_awaiting_the_maintainer() {
        let caps = Caps::default();
        // A release entry above 1.10 with a plain waiver fails (D2779).
        let w = limit(Status::Release, Some("open item I1-1: slow"), None);
        assert!(matches!(
            judge(&caps, Some(&w), &m("k", 11.5, 10.0)),
            Verdict::Fail(_)
        ));
        // A dev entry up to 1.50 with a waiver is waived; above 1.50 only while it awaits the
        // maintainer's decision and names its open item.
        let d = limit(Status::Dev, Some("open item I1-1: slow"), None);
        assert!(matches!(
            judge(&caps, Some(&d), &m("k", 14.0, 10.0)),
            Verdict::Pass | Verdict::Warn(_)
        ));
        assert!(matches!(
            judge(&caps, Some(&d), &m("k", 16.0, 10.0)),
            Verdict::Fail(_)
        ));
        let awaiting = limit(
            Status::Dev,
            Some("open item I1-1: 1.6 of the library, awaiting the maintainer's decision"),
            None,
        );
        assert!(matches!(
            judge(&caps, Some(&awaiting), &m("k", 16.0, 10.0)),
            Verdict::Waived(_)
        ));
        let no_item = limit(
            Status::Dev,
            Some("awaiting the maintainer's decision"),
            None,
        );
        assert!(matches!(
            judge(&caps, Some(&no_item), &m("k", 16.0, 10.0)),
            Verdict::Fail(_)
        ));
    }

    #[test]
    fn rejudge_writes_the_derived_fields_from_the_stored_numbers() {
        let doc = json!({
            "method": { "limits": { "release": 1.1, "dev": 1.5, "warn": 1.1, "goal": 1.0, "runtime": 1.02 } },
            "levels": { "simd128": { "kernels": {
                "rotate_cq15": {
                    "ours_instr_per_sample": 5.0, "other_instr_per_sample": 10.0, "ratio": 0.5,
                    "compared_with": "sdrk fallback", "status": "release", "verdict": "pass", "goal": true,
                    "snr_db": { "ours": 120.0, "other": 120.0 },
                },
                "halfband_decimate": {
                    "ours_instr_per_sample": 9.0, "other_instr_per_sample": 10.0, "ratio": 0.9,
                    "compared_with": "liquid", "status": "release", "verdict": "pass", "goal": true,
                    "snr_db": { "ours": 120.0, "other": 120.0 },
                    "informational": [{ "label": "information: ours:halfband_ref_fir_decimate",
                        "ours_instr_per_sample": 9.0, "other_instr_per_sample": 10.0, "ratio": 0.9,
                        "compared_with": "ours:halfband_ref_fir_decimate", "target": 0.6, "meets_target": false,
                        "other_snr_db": 120.0 }],
                },
            } } },
        });
        let l = limits(
            r#"{ "rotate_cq15": { "simd128": { "status": "info" } }, "halfband_decimate": { "simd128": { "status": "release" } } }"#,
        );
        let r = rejudge(&doc, &l).unwrap();
        assert_eq!(r.failures, Vec::<String>::new());
        let k = &r.doc["levels"]["simd128"]["kernels"];
        assert_eq!(k["rotate_cq15"]["status"], "info");
        assert_eq!(k["rotate_cq15"]["verdict"], "info");
        assert_eq!(k["rotate_cq15"]["comparison_kind"], "own_fallback");
        assert!(k["rotate_cq15"].get("goal").is_none());
        assert_eq!(k["rotate_cq15"]["ratio"], 0.5, "the numbers stay");
        let h = &k["halfband_decimate"];
        assert_eq!(h["comparison_kind"], "library");
        assert_eq!(h["goal"], true);
        assert_eq!(h["informational"][0]["comparison_kind"], "own_kernel");
        assert!(h["informational"][0].get("meets_target").is_none());
        assert!(r.doc["method"]["limits"].get("runtime").is_none());
        // Judged again, nothing changes.
        assert_eq!(rejudge(&r.doc, &l).unwrap().changes, Vec::<String>::new());
        // A release entry against our own fallback fails the re-judgement.
        let own = limits(
            r#"{ "rotate_cq15": { "simd128": { "status": "release" } }, "halfband_decimate": { "simd128": { "status": "release" } } }"#,
        );
        let r = rejudge(&doc, &own).unwrap();
        assert_eq!(r.failures.len(), 1, "{:?}", r.failures);
        assert!(
            r.failures[0].contains("rotate_cq15 simd128") && r.failures[0].contains("our own code")
        );
    }

    #[test]
    fn the_committed_counts_are_judged_by_the_limits() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        assert_eq!(check_counts(root).unwrap(), Vec::new());
    }

    #[test]
    fn results_record_the_goal_and_the_informational_comparisons() {
        let l = limits(r#"{ "k": { "neon": { "status": "release" } } }"#);
        let mut x = m("k", 9.0, 10.0);
        x.level = "neon".into();
        x.extras.push(Extra {
            label: "goal: cheapest VOLK implementation".into(),
            ours: 9.0,
            other: 6.0,
            compared_with: "volk_x: neon".into(),
            comparison_kind: ComparisonKind::Library,
            other_snr_db: Some(120.0),
        });
        let report = evaluate(&l, "neon", &[x]);
        assert_eq!(report.records["k"]["goal"], true);
        assert_eq!(report.records["k"]["informational"][0]["ratio"], 1.5);
        assert!(report.lines[1].contains("cheapest"), "{:?}", report.lines);
    }

    #[test]
    fn limits_json_rejects_an_unknown_status() {
        assert!(Limits::parse(r#"{ "k": { "avx2": { "status": "fast" } } }"#).is_err());
        assert!(Limits::parse("[]").is_err());
        assert!(Limits::parse("{").is_err());
    }

    #[test]
    fn qemu_output_is_parsed() {
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
            // The bisection starts from the last amount that failed, not from 0.
            assert!(calls <= 2 * 64_u32.saturating_sub(need.leading_zeros()) + 21);
        }
    }

    #[test]
    fn only_the_levels_without_a_core_model_are_counted() {
        let levels = |platform: &str, l: &[&str]| -> Vec<&str> {
            let l: Vec<String> = l.iter().map(ToString::to_string).collect();
            select_plans(platform, &l)
                .unwrap()
                .iter()
                .map(|p| p.level)
                .collect()
        };
        assert_eq!(levels("wasm32", &[]), vec!["simd128"]);
        assert_eq!(levels("wasm32", &["simd128"]), vec!["simd128"]);
        assert!(select_plans("wasm32", &["avx2".to_string()]).is_err());
        assert!(select_plans("riscv", &[]).is_err());
        for platform in ["x86_64", "x86_64-avx512", "aarch64"] {
            let error = select_plans(platform, &[]).unwrap_err().to_string();
            assert!(error.contains("cargo measure"), "{platform}: {error}");
        }
    }

    #[test]
    fn options_are_parsed() {
        let o = Options::parse(&[
            "--platform",
            "wasm32",
            "--kernel",
            "fir",
            "--strict",
            "--no-variants",
        ])
        .unwrap();
        assert_eq!(o.platform.as_deref(), Some("wasm32"));
        assert_eq!(o.kernels, vec!["fir".to_string()]);
        assert!(o.strict && !o.variants && o.write);
        assert!(Options::parse(&["--platform"]).is_err());
        assert!(Options::parse(&["--fast"]).is_err());
        assert!(Options::parse(&["--pipeline"]).is_err());
    }
}
