//! `cargo xtask measure <kernel>` (alias `cargo measure <kernel>`): every measurement of the
//! measurement policy (`docs/performance.md#measurement-policy`) for one kernel, in one table.
//!
//! The kernel name selects every entry whose name is the name or starts with the name and `_`:
//! `biquad_real` selects the count entries `biquad_real`, `biquad_real_q15`,
//! `biquad_real_fallback` and `biquad_real_q15_fallback` and the golden cases of
//! `tests/golden/vectors/biquad_real`; `biquad` also selects those of `biquad_real`.
//!
//! - **Instructions and accuracy of the counts:** the judged records of `cargo xtask count`
//!   for the selected entries on the levels of the platform (ours, the comparison, the ratio,
//!   the status and verdict of `benches/limits.json`, the SNR of both against the f64 model).
//!   Nothing is written to `benches/results`.
//! - **Golden cases:** the golden tests of `sdrk-testkit` run with the report of
//!   `sdrk_testkit::golden::assert_matches`; per case the contract floor, the measured SNR (the
//!   lowest of all comparisons of the case), the achievable optimum (the expected values rounded
//!   once to the sample type) and the distance to it (`docs/kernels.md#accuracy-policy`).
//! - **Simulated cycles (level 2):** for the levels counted with callgrind, the execution counts
//!   per instruction of ours and of the judged comparison, simulated by `llvm-mca` on the
//!   reference model and the counter-model of the level ([`crate::mca`]): cycles per sample,
//!   the ratio, the efficiency (port bound / cycles), the bottleneck class and the coverage.
//!
//! The result is printed as tables and written to `target/measure/<kernel>.json`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};
use serde_json::{Value, json};

use crate::count::{KERNELS, PLANS};

const USAGE: &str = "usage: cargo xtask measure <kernel> [--platform P]... [--level L]... [--no-count] [--no-golden] [--no-cycles] [--json]";

/// The command line of `cargo xtask measure`.
#[allow(
    clippy::struct_excessive_bools,
    reason = "one field per command-line flag"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// The kernel name (a prefix, see the module documentation).
    pub kernel: String,
    /// `--platform` (repeatable); empty: the platform of the host.
    pub platforms: Vec<String>,
    /// `--level` (repeatable): only these levels.
    pub levels: Vec<String>,
    /// Not `--no-count`.
    pub count: bool,
    /// Not `--no-golden`.
    pub golden: bool,
    /// Not `--no-cycles`.
    pub cycles: bool,
    /// `--json`: print the JSON document instead of the tables.
    pub json: bool,
}

impl Options {
    /// Parses the arguments after `measure`.
    ///
    /// # Errors
    /// No kernel, more than one, an unknown option or a missing value.
    pub fn parse(args: &[&str]) -> anyhow::Result<Options> {
        let mut o = Options {
            count: true,
            golden: true,
            cycles: true,
            ..Options::default()
        };
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            let mut value = || {
                it.next()
                    .map(|s| (*s).to_string())
                    .ok_or_else(|| anyhow::anyhow!("`{arg}` needs a value; {USAGE}"))
            };
            match *arg {
                "--platform" => o.platforms.push(value()?),
                "--level" => o.levels.push(value()?),
                "--no-count" => o.count = false,
                "--no-golden" => o.golden = false,
                "--no-cycles" => o.cycles = false,
                "--json" => o.json = true,
                other if other.starts_with("--") => bail!("unknown option `{other}`; {USAGE}"),
                name if o.kernel.is_empty() => o.kernel = name.to_string(),
                extra => bail!("only one kernel, got `{}` and `{extra}`; {USAGE}", o.kernel),
            }
        }
        if o.kernel.is_empty() {
            bail!("{USAGE}");
        }
        Ok(o)
    }
}

/// Whether `name` is selected by `kernel`: equal, or `kernel` followed by `_` and more.
#[must_use]
pub fn selects(kernel: &str, name: &str) -> bool {
    name == kernel
        || name
            .strip_prefix(kernel)
            .is_some_and(|rest| rest.starts_with('_'))
}

/// The count entries selected by `kernel`, in report order.
#[must_use]
pub fn count_entries(kernel: &str) -> Vec<String> {
    KERNELS
        .iter()
        .filter(|s| selects(kernel, s.name))
        .map(|s| s.name.to_string())
        .collect()
}

/// The golden kernels (directories of `tests/golden/vectors`) selected by `kernel`.
///
/// # Errors
/// The directory cannot be read.
pub fn golden_kernels(root: &Path, kernel: &str) -> anyhow::Result<Vec<String>> {
    let dir = root.join("tests").join("golden").join("vectors");
    let mut out = Vec::new();
    for entry in
        std::fs::read_dir(&dir).with_context(|| format!("cannot read {}", dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && let Some(name) = entry.file_name().to_str()
            && selects(kernel, name)
        {
            out.push(name.to_string());
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// The platforms of `cargo xtask count` that a host of `arch` measures by default.
#[must_use]
pub fn default_platforms(arch: &str) -> Vec<String> {
    match arch {
        "aarch64" => vec!["aarch64".to_string()],
        _ => vec!["x86_64".to_string()],
    }
}

/// One golden case, summarised over all comparisons the tests made of it.
#[derive(Clone, Debug, PartialEq)]
pub struct GoldenCase {
    /// `kernel/case`.
    pub case: String,
    /// The sample type of the compared data.
    pub sample_type: String,
    /// `bit_exact`, `tolerance` or `max_abs_err`.
    pub class: String,
    /// The contract floor of a `tolerance` case.
    pub floor_db: Option<f64>,
    /// The lowest SNR of all comparisons; `None` if every comparison was exact.
    pub snr_db: Option<f64>,
    /// Every comparison was exact (zero error).
    pub exact: bool,
    /// The achievable optimum of a `tolerance` case.
    pub optimum_db: Option<f64>,
    /// Every comparison passed.
    pub passed: bool,
    /// Number of comparisons.
    pub comparisons: usize,
    /// The first failure, if any.
    pub detail: Option<String>,
}

impl GoldenCase {
    /// How far the measured SNR is below the optimum (positive: below).
    #[must_use]
    pub fn distance_db(&self) -> Option<f64> {
        match (self.optimum_db, self.snr_db) {
            (Some(o), Some(s)) => Some(o - s),
            _ => None,
        }
    }
}

/// Summarises the report lines of `sdrk_testkit::golden` for the selected golden kernels: one
/// entry per case and sample type, sorted by name.
///
/// # Errors
/// A line is not valid JSON.
pub fn summarise_golden(report: &str, kernels: &[String]) -> anyhow::Result<Vec<GoldenCase>> {
    let mut cases: BTreeMap<(String, String), GoldenCase> = BTreeMap::new();
    for line in report.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).with_context(|| format!("report line {line}"))?;
        let case = field(&v, "case").as_str().unwrap_or_default().to_string();
        let kernel = case.split('/').next().unwrap_or_default();
        if !kernels.iter().any(|k| k == kernel) {
            continue;
        }
        let sample_type = field(&v, "type").as_str().unwrap_or_default().to_string();
        let snr = field(&v, "snr_db").as_f64();
        let exact = field(&v, "exact").as_bool().unwrap_or(false);
        let passed = field(&v, "passed").as_bool().unwrap_or(false);
        let detail = field(&v, "detail").as_str().map(str::to_string);
        let entry = cases
            .entry((case.clone(), sample_type.clone()))
            .or_insert_with(|| GoldenCase {
                case,
                sample_type,
                class: field(&v, "class").as_str().unwrap_or_default().to_string(),
                floor_db: field(&v, "snr_db_min").as_f64(),
                snr_db: None,
                exact: true,
                optimum_db: field(&v, "optimum_db").as_f64(),
                passed: true,
                comparisons: 0,
                detail: None,
            });
        entry.comparisons += 1;
        entry.exact &= exact;
        if !exact {
            entry.snr_db = Some(match (entry.snr_db, snr) {
                (Some(a), Some(b)) => a.min(b),
                (None, Some(b)) => b,
                (a, None) => a.unwrap_or(f64::NAN),
            });
        }
        if !passed {
            entry.passed = false;
            if entry.detail.is_none() {
                entry.detail = detail;
            }
        }
    }
    Ok(cases.into_values().collect())
}

/// The member `key` of a JSON object, `null` if it is missing.
fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap_or(&Value::Null)
}

fn db(x: Option<f64>) -> String {
    x.map_or_else(|| "-".to_string(), |v| format!("{v:.1}"))
}

/// The table of the golden cases.
#[must_use]
pub fn golden_table(cases: &[GoldenCase]) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{:<44} {:<5} {:<11} {:>7} {:>9} {:>9} {:>9}  result",
        "case", "type", "class", "floor", "measured", "optimum", "distance"
    );
    for c in cases {
        let measured = if c.exact {
            "exact".to_string()
        } else {
            db(c.snr_db)
        };
        let result = match (&c.detail, c.passed) {
            (_, true) => format!("pass ({}x)", c.comparisons),
            (Some(d), false) => format!("FAIL: {d}"),
            (None, false) => "FAIL".to_string(),
        };
        let _ = writeln!(
            s,
            "{:<44} {:<5} {:<11} {:>7} {:>9} {:>9} {:>9}  {result}",
            c.case,
            c.sample_type,
            c.class,
            db(c.floor_db),
            measured,
            db(c.optimum_db),
            db(c.distance_db()),
        );
    }
    s
}

/// The table of the count records of one platform: `(level, kernel -> record)`.
#[must_use]
pub fn count_table(levels: &[(String, BTreeMap<String, Value>)]) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{:<26} {:<13} {:>9} {:>9} {:>6}  {:<8} {:<7} {:>15}  comparison",
        "entry", "level", "ours", "other", "ratio", "status", "verdict", "SNR ours/other"
    );
    for (level, records) in levels {
        for (kernel, r) in records {
            let num = |key: &str| {
                field(r, key)
                    .as_f64()
                    .map_or("-".to_string(), |v| format!("{v:.2}"))
            };
            let snr = format!(
                "{}/{}",
                db(field(field(r, "snr_db"), "ours").as_f64()),
                db(field(field(r, "snr_db"), "other").as_f64())
            );
            let _ = writeln!(
                s,
                "{:<26} {:<13} {:>9} {:>9} {:>6}  {:<8} {:<7} {:>15}  {}",
                kernel,
                level,
                num("ours_instr_per_sample"),
                num("other_instr_per_sample"),
                num("ratio"),
                field(r, "status").as_str().unwrap_or("-"),
                field(r, "verdict").as_str().unwrap_or("-"),
                snr,
                field(r, "compared_with").as_str().unwrap_or("-"),
            );
        }
    }
    s
}

/// Runs the golden tests of `sdrk-testkit` with the report written to `report`.
fn run_golden(root: &Path, report: &Path) -> anyhow::Result<String> {
    match std::fs::remove_file(report) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(e).context("cannot remove the old golden report");
        }
        _ => {}
    }
    println!("==> golden tests of sdrk-testkit (run-time detection of the host's level)");
    let out = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args([
            "test",
            "-q",
            "-p",
            "sdrk-testkit",
            "--features",
            "std",
            "--test",
            "golden_*",
        ])
        .env("SDRK_GOLDEN_REPORT", report)
        .current_dir(root)
        .output()
        .context("cannot run cargo test")?;
    // The output of the passing tests is noise here; a failure shows it in full, also for the
    // tests of other kernels, because a build error or a panic outside a comparison would
    // otherwise go unnoticed.
    if !out.status.success() {
        print!("{}", String::from_utf8_lossy(&out.stdout));
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        println!("    (the golden tests failed; the cases of the kernel are marked FAIL below)");
    }
    match std::fs::read_to_string(report) {
        Ok(text) => Ok(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(e).context("cannot read the golden report"),
    }
}

/// One part of the result: its table, its JSON value and whether it passed.
struct Part {
    table: String,
    doc: Value,
    ok: bool,
}

/// The counts of the selected entries on every platform of `o`.
#[allow(
    clippy::large_stack_frames,
    reason = "the plan tables and std::process::Command are large; xtask is a host tool"
)]
fn measure_counts(root: &Path, o: &Options, entries: &[String]) -> Part {
    let mut part = Part {
        table: String::new(),
        doc: Value::Null,
        ok: true,
    };
    let mut doc = serde_json::Map::new();
    let platforms = if o.platforms.is_empty() {
        default_platforms(std::env::consts::ARCH)
    } else {
        o.platforms.clone()
    };
    for platform in &platforms {
        let levels: Vec<String> = o
            .levels
            .iter()
            .filter(|l| {
                PLANS
                    .iter()
                    .any(|p| p.platform == platform && p.level == *l)
            })
            .cloned()
            .collect();
        if !o.levels.is_empty() && levels.is_empty() {
            continue;
        }
        match crate::count::judged_records(root, platform, &levels, entries) {
            Ok(records) => {
                let records: Vec<(String, BTreeMap<String, Value>)> = records
                    .into_iter()
                    .map(|(l, r)| (l.to_string(), r))
                    .collect();
                part.ok &= records.iter().all(|(_, r)| {
                    r.values()
                        .all(|v| field(v, "verdict").as_str() != Some("fail"))
                });
                let _ = writeln!(part.table, "\ncount, platform {platform}:");
                part.table.push_str(&count_table(&records));
                doc.insert(
                    platform.clone(),
                    Value::Object(records.into_iter().map(|(l, r)| (l, json!(r))).collect()),
                );
            }
            Err(e) => {
                part.ok = false;
                let _ = writeln!(
                    part.table,
                    "\ncount, platform {platform}: not measured: {e:#}"
                );
                doc.insert(platform.clone(), json!({ "error": format!("{e:#}") }));
            }
        }
    }
    part.doc = Value::Object(doc);
    part
}

/// The levels of `platform` that `o` selects.
fn selected_levels(o: &Options, platform: &str) -> Vec<&'static str> {
    PLANS
        .iter()
        .filter(|p| p.platform == platform)
        .filter(|p| o.levels.is_empty() || o.levels.iter().any(|l| l == p.level))
        .map(|p| p.level)
        .collect()
}

/// The simulation of one side of an entry on both models of the level.
fn simulate_side(
    counts: &crate::mca::Counts,
    samples: u64,
    models: &crate::mca::Models,
) -> anyhow::Result<[crate::mca::Simulation; 2]> {
    let blocks = crate::mca::profile_blocks(counts)?;
    Ok([
        crate::mca::simulate(&blocks, samples, models.triple, models.reference)?,
        crate::mca::simulate(&blocks, samples, models.triple, models.counter)?,
    ])
}

fn simulation_json(s: &crate::mca::Simulation) -> Value {
    json!({
        "cpu": s.cpu,
        "instructions_per_sample": s.instructions_per_sample,
        "cycles_per_sample": s.cycles_per_sample,
        "port_bound_per_sample": s.port_bound_per_sample,
        "efficiency": s.efficiency,
        "bottleneck": crate::mca::bottleneck(s.efficiency),
        "coverage": s.coverage,
        "blocks": s.blocks,
        "rejected_blocks": s.rejected,
    })
}

/// The simulated cycles of the selected entries (level 2).
#[allow(
    clippy::large_stack_frames,
    reason = "the plan tables and std::process::Command are large; xtask is a host tool"
)]
fn measure_cycles(root: &Path, o: &Options, entries: &[String]) -> Part {
    let mut part = Part {
        table: String::new(),
        doc: Value::Null,
        ok: true,
    };
    let mut doc = serde_json::Map::new();
    let platforms = if o.platforms.is_empty() {
        default_platforms(std::env::consts::ARCH)
    } else {
        o.platforms.clone()
    };
    for platform in &platforms {
        let mut levels_doc = serde_json::Map::new();
        for level in selected_levels(o, platform) {
            let Some(plan) = PLANS
                .iter()
                .find(|p| p.platform == platform && p.level == level)
            else {
                continue;
            };
            let Some(models) = crate::mca::models(level) else {
                let _ = writeln!(
                    part.table,
                    "\ncycles, {platform} {level}: no core model (not simulated)"
                );
                continue;
            };
            if plan.baseline.is_some() {
                continue;
            }
            if plan.counter != crate::count::Counter::Callgrind {
                let note = "not simulated yet: the execution counts come from callgrind, and this level is counted under an emulator";
                let _ = writeln!(part.table, "\ncycles, {platform} {level}: {note}");
                levels_doc.insert(level.to_string(), json!({ "note": note }));
                continue;
            }
            match cycles_of_level(root, platform, level, models, entries) {
                Ok((table, value)) => {
                    let _ = writeln!(
                        part.table,
                        "\ncycles, {platform} {level} (llvm-mca; reference model {}, counter-model {}):",
                        models.reference, models.counter
                    );
                    part.table.push_str(&table);
                    levels_doc.insert(level.to_string(), value);
                }
                Err(e) => {
                    part.ok = false;
                    let _ = writeln!(
                        part.table,
                        "\ncycles, {platform} {level}: not simulated: {e:#}"
                    );
                    levels_doc.insert(level.to_string(), json!({ "error": format!("{e:#}") }));
                }
            }
        }
        doc.insert(platform.clone(), Value::Object(levels_doc));
    }
    part.doc = Value::Object(doc);
    part
}

/// One model of a level: ours and the comparison simulated on it.
fn model_json(ours: &crate::mca::Simulation, other: &crate::mca::Simulation) -> Value {
    json!({
        "ours": simulation_json(ours),
        "other": simulation_json(other),
        "ratio": ours.cycles_per_sample / other.cycles_per_sample,
    })
}

/// The cycles table and JSON of one level.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and serde_json values are large; xtask is a host tool"
)]
fn cycles_of_level(
    root: &Path,
    platform: &str,
    level: &str,
    models: &crate::mca::Models,
    entries: &[String],
) -> anyhow::Result<(String, Value)> {
    let mut table = String::new();
    let _ = writeln!(
        table,
        "{:<26} {:<23} {:>9} {:>9} {:>6} {:>6} {:<10} {:>8}  comparison",
        "entry", "model", "ours", "other", "ratio", "eff.", "bottleneck", "coverage"
    );
    let mut value = serde_json::Map::new();
    for e in crate::count::profiles(root, platform, level, entries)? {
        let [ours_ref, ours_counter] = simulate_side(&e.ours, e.samples, models)?;
        let [other_ref, other_counter] = simulate_side(&e.other, e.samples, models)?;
        let rows = [
            ("reference", &ours_ref, &other_ref),
            ("counter", &ours_counter, &other_counter),
        ];
        for (n, (role, ours, other)) in rows.iter().enumerate() {
            let first = n == 0;
            let _ = writeln!(
                table,
                "{:<26} {:<23} {:>9.2} {:>9.2} {:>6.2} {:>6.2} {:<10} {:>7.1}%  {}",
                if first { e.kernel.as_str() } else { "" },
                format!("{} ({role})", ours.cpu),
                ours.cycles_per_sample,
                other.cycles_per_sample,
                ours.cycles_per_sample / other.cycles_per_sample,
                ours.efficiency,
                crate::mca::bottleneck(ours.efficiency),
                100.0 * ours.coverage.min(other.coverage),
                if first { e.compared_with.as_str() } else { "" },
            );
        }
        let ratio_ref = ours_ref.cycles_per_sample / other_ref.cycles_per_sample;
        let ratio_counter = ours_counter.cycles_per_sample / other_counter.cycles_per_sample;
        value.insert(
            e.kernel.clone(),
            json!({
                "compared_with": e.compared_with,
                "samples": e.samples,
                "reference_model": model_json(&ours_ref, &other_ref),
                "counter_model": model_json(&ours_counter, &other_counter),
                "counter_vs_reference": ratio_counter / ratio_ref,
            }),
        );
    }
    Ok((table, Value::Object(value)))
}

/// The golden cases of the selected golden kernels.
fn measure_golden(root: &Path, report: &Path, golden: &[String]) -> anyhow::Result<Part> {
    let report = run_golden(root, report)?;
    let cases = summarise_golden(&report, golden)?;
    let mut table = String::from(
        "\ngolden cases (dB; distance = optimum - measured; the optimum is the expected value rounded once to the type):\n",
    );
    table.push_str(&golden_table(&cases));
    let doc = json!({
        "build": "host, run-time detection of the level (cargo test -p sdrk-testkit --features std)",
        "cases": cases.iter().map(|c| json!({
            "case": c.case,
            "type": c.sample_type,
            "class": c.class,
            "floor_db": c.floor_db,
            "snr_db": c.snr_db,
            "exact": c.exact,
            "optimum_db": c.optimum_db,
            "distance_db": c.distance_db(),
            "passed": c.passed,
            "comparisons": c.comparisons,
            "detail": c.detail,
        })).collect::<Vec<_>>(),
    });
    Ok(Part {
        table,
        doc,
        ok: cases.iter().all(|c| c.passed),
    })
}

/// Runs `cargo xtask measure ...`.
///
/// # Errors
/// Unknown options, a kernel that selects nothing, or a file that cannot be written. A count
/// that cannot run (a missing counter or reference) is reported in the result, not as an
/// error; the return value is `Ok(false)` if a count failed or was judged `fail`, or a golden
/// case failed.
#[allow(
    clippy::large_stack_frames,
    reason = "the kernel table and serde_json values are large; xtask is a host tool"
)]
pub fn run(root: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let o = Options::parse(args)?;
    let entries = count_entries(&o.kernel);
    let golden = golden_kernels(root, &o.kernel)?;
    if entries.is_empty() && golden.is_empty() {
        let mut names: Vec<&str> = KERNELS.iter().map(|k| k.name).collect();
        names.sort_unstable();
        bail!(
            "`{}` selects no count entry and no golden case; count entries: {}",
            o.kernel,
            names.join(", ")
        );
    }
    let out_dir = root.join("target").join("measure");
    std::fs::create_dir_all(&out_dir)?;
    let count = (o.count && !entries.is_empty()).then(|| measure_counts(root, &o, &entries));
    let cycles = (o.cycles && !entries.is_empty()).then(|| measure_cycles(root, &o, &entries));
    let golden_part = if o.golden && !golden.is_empty() {
        Some(measure_golden(
            root,
            &out_dir.join(format!("{}-golden.jsonl", o.kernel)),
            &golden,
        )?)
    } else {
        None
    };
    let doc = json!({
        "kernel": o.kernel,
        "selection": { "count": entries, "golden": golden },
        "count": count.as_ref().map(|p| &p.doc),
        "golden": golden_part.as_ref().map(|p| &p.doc),
        "cycles": cycles.as_ref().map(|p| &p.doc),
    });
    let path: PathBuf = out_dir.join(format!("{}.json", o.kernel));
    let mut text = serde_json::to_string_pretty(&doc)?;
    text.push('\n');
    std::fs::write(&path, &text)?;
    if o.json {
        print!("{text}");
    } else {
        println!("\nmeasure {}", o.kernel);
        for part in count.iter().chain(&cycles).chain(&golden_part) {
            print!("{}", part.table);
        }
        println!("\nwritten to {}", path.display());
    }
    Ok(count
        .iter()
        .chain(&cycles)
        .chain(&golden_part)
        .all(|p| p.ok))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_selects_itself_and_its_suffixed_entries() {
        assert!(selects("biquad_real", "biquad_real"));
        assert!(selects("biquad_real", "biquad_real_q15_fallback"));
        assert!(selects("biquad", "biquad_real"));
        assert!(!selects("biquad_real", "biquad"));
        assert!(!selects("fir", "fir2"));
        assert_eq!(
            count_entries("biquad_real"),
            [
                "biquad_real",
                "biquad_real_q15",
                "biquad_real_fallback",
                "biquad_real_q15_fallback"
            ]
        );
        assert_eq!(count_entries("fir_decimate"), ["fir_decimate"]);
        assert_eq!(count_entries("nothing"), Vec::<String>::new());
    }

    #[test]
    fn options_need_exactly_one_kernel() {
        assert!(Options::parse(&[]).is_err());
        assert!(Options::parse(&["a", "b"]).is_err());
        assert!(Options::parse(&["a", "--bogus"]).is_err());
        let o = Options::parse(&["fir", "--level", "avx2", "--no-golden", "--json"]).unwrap();
        assert_eq!(o.kernel, "fir");
        assert_eq!(o.levels, ["avx2"]);
        assert!(o.count && !o.golden && o.json);
    }

    #[test]
    fn golden_lines_are_summarised_per_case_with_the_lowest_snr() {
        let report = r#"{"case":"fir/a","type":"C32","class":"tolerance","snr_db_min":110.0,"snr_db":140.0,"exact":false,"optimum_db":151.0,"passed":true,"detail":null}
{"case":"fir/a","type":"C32","class":"tolerance","snr_db_min":110.0,"snr_db":139.0,"exact":false,"optimum_db":151.0,"passed":true,"detail":null}
{"case":"fir/b","type":"CQ15","class":"bit_exact","snr_db_min":null,"snr_db":null,"exact":true,"optimum_db":null,"passed":true,"detail":null}
{"case":"fir/c","type":"C32","class":"tolerance","snr_db_min":110.0,"snr_db":100.0,"exact":false,"optimum_db":151.0,"passed":false,"detail":"snr below"}
{"case":"rotate/x","type":"C32","class":"tolerance","snr_db_min":120.0,"snr_db":142.0,"exact":false,"optimum_db":151.0,"passed":true,"detail":null}
"#;
        let cases = summarise_golden(report, &["fir".to_string()]).unwrap();
        assert_eq!(cases.len(), 3);
        assert_eq!(cases[0].snr_db, Some(139.0));
        assert_eq!(cases[0].comparisons, 2);
        assert_eq!(cases[0].distance_db(), Some(12.0));
        assert!(cases[1].exact && cases[1].snr_db.is_none());
        assert!(!cases[2].passed);
        assert_eq!(cases[2].detail.as_deref(), Some("snr below"));
        assert!(golden_table(&cases).contains("FAIL: snr below"));
    }

    #[test]
    fn the_host_platform_is_the_default() {
        assert_eq!(default_platforms("x86_64"), ["x86_64"]);
        assert_eq!(default_platforms("aarch64"), ["aarch64"]);
    }
}
