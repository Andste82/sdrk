//! `cargo xtask measure <kernel>` (alias `cargo measure <kernel>`): every measurement of the
//! measurement policy (`docs/performance.md#measurement-policy`) for one kernel, in one table.
//!
//! The kernel name selects every entry whose name is the name or starts with the name and `_`:
//! `biquad_real` selects the count entries `biquad_real`, `biquad_real_q15`,
//! `biquad_real_fallback` and `biquad_real_q15_fallback` of the benchmark table
//! ([`crate::bench::KERNELS`]) and the golden cases of
//! `tests/golden/vectors/biquad_real`; `biquad` also selects those of `biquad_real`.
//!
//! - **Golden cases:** the golden tests of `sdrk-testkit` run with the report of
//!   `sdrk_testkit::golden::assert_matches`; per case the contract floor, the measured SNR (the
//!   lowest of all comparisons of the case), the achievable optimum (the expected values rounded
//!   once to the output format: the sample type, or for a block-floating-point `CQ15` output a
//!   Q15 mantissa with the block exponent the expected values need) and the distance to it
//!   (`docs/kernels.md#accuracy-policy`).
//! - **Simulated cycles (level 2):** on every level with a core model (`sse2`, `sse4.2`,
//!   `avx2`, `avx512`, `neon`; D68), the execution counts per instruction of ours and of the
//!   judged comparison, taken by the profiler of the level in the environment its binary runs
//!   in ([`crate::bench`]: callgrind; Intel SDE `-dcfg`, [`crate::blockcount`]; QEMU with the
//!   plugin `xtask/plugins/bbcount.c`), simulated by `llvm-mca` on the reference model and the
//!   counter-model of the level ([`crate::mca`]): cycles per sample, the ratio, the efficiency
//!   (port bound / cycles), the bottleneck class and the coverage. Only the ratio of ours to the
//!   comparison on one model means something; the absolute cycles are no hardware figures.
//!   `avx2-runtime` (the run-time detection of the level, on the models of `avx2`) is compared
//!   with our own entry in the fixed `avx2` build (D2702), as information; `cargo xtask check
//!   cycles` judges it against the library of the `avx2` record (D2764).
//!   Per entry also the accuracy of both sides on the same workload: the SNR against the f64
//!   reference of the measurement binary (`snr_db`, D34, D2749), judged by
//!   `cargo xtask check cycles`.
//!
//! The result is printed as tables and written to `target/measure/<kernel>.json`
//! (`target/measure/all.json` with `--all`, which selects every entry and golden kernel).
//!
//! **Committed results** (D2704): `--merge` writes the records of the measured entries into
//! `benches/results/cycles/<platform>.json` and keeps every other record of the file; `--all`
//! without `--level` writes the files of the measured platforms anew. Both measure the branches
//! too (`--branches`), only in the judged mode `whole-loop` and with the counts of the profiler
//! of each level, and refuse an `llvm-mca` without the corrections of D942
//! ([`crate::cycle_results`]). `benches/cycle_tables.py` writes the cycle tables of
//! `docs/performance.md` from these files.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};
use serde_json::{Value, json};

use crate::bench::{KERNELS, PLANS};

const USAGE: &str = "usage: cargo xtask measure <kernel>|--all [--platform P]... [--level L]... [--no-golden] [--no-cycles] [--cycle-mode whole-loop|per-block] [--cycle-counter sde] [--branches] [--merge] [--results DIR] [--shard I/N] [--json]";

/// The command line of `cargo xtask measure`.
#[allow(
    clippy::struct_excessive_bools,
    reason = "one field per command-line flag"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// The kernel name (a prefix, see the module documentation); `all` with `--all`.
    pub kernel: String,
    /// `--all`: every entry and every golden kernel instead of a kernel name; without
    /// `--level` the results files of the measured platforms are written anew.
    pub all: bool,
    /// `--merge`: write the records of the measured entries into the results files and keep
    /// every other record (implies `--branches`).
    pub merge: bool,
    /// `--results DIR`: the directory of the results files that `--merge` and `--all` write
    /// (default `benches/results/cycles`; the nightly measurement writes each partition into a
    /// directory of its own).
    pub results: Option<PathBuf>,
    /// `--shard I/N`: only the entries whose place in the sorted selection is `I - 1` modulo
    /// `N` (the partitions of the nightly measurement).
    pub shard: Option<(usize, usize)>,
    /// `--platform` (repeatable); empty: the platform of the host.
    pub platforms: Vec<String>,
    /// `--level` (repeatable): only these levels.
    pub levels: Vec<String>,
    /// Not `--no-golden`.
    pub golden: bool,
    /// Not `--no-cycles`.
    pub cycles: bool,
    /// `--cycle-mode`: the mode of the simulated cycles in the table and in the top-level
    /// fields of the JSON (the other mode is reported beside it).
    pub cycle_mode: crate::mca::CycleMode,
    /// `--cycle-counter`: take the execution counts of the cycles from this profiler instead of
    /// the profiler of the level (`sde` on `avx2` validates the SDE path against callgrind).
    pub cycle_counter: Option<crate::bench::Profiler>,
    /// `--branches`: count the branch mispredictions per sample with cachegrind on the x86-64
    /// levels (D1611) and, with the cycles, the cost `cycles + mispredictions × penalty`.
    pub branches: bool,
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
                "--no-golden" => o.golden = false,
                "--no-cycles" => o.cycles = false,
                "--cycle-mode" => {
                    let v = value()?;
                    o.cycle_mode = crate::mca::CycleMode::parse(&v)
                        .ok_or_else(|| anyhow::anyhow!("unknown cycle mode `{v}`; {USAGE}"))?;
                }
                "--cycle-counter" => {
                    let v = value()?;
                    o.cycle_counter =
                        Some(crate::bench::Profiler::parse(&v).ok_or_else(|| {
                            anyhow::anyhow!("unknown cycle counter `{v}`; {USAGE}")
                        })?);
                }
                "--branches" => o.branches = true,
                "--all" => o.all = true,
                "--merge" => o.merge = true,
                "--results" => o.results = Some(PathBuf::from(value()?)),
                "--shard" => {
                    let v = value()?;
                    o.shard = Some(
                        v.split_once('/')
                            .and_then(|(i, n)| Some((i.parse().ok()?, n.parse().ok()?)))
                            .filter(|&(i, n): &(usize, usize)| i >= 1 && i <= n)
                            .ok_or_else(|| {
                                anyhow::anyhow!("`--shard` takes I/N with 1 <= I <= N, got `{v}`")
                            })?,
                    );
                }
                "--json" => o.json = true,
                other if other.starts_with("--") => bail!("unknown option `{other}`; {USAGE}"),
                name if o.kernel.is_empty() => o.kernel = name.to_string(),
                extra => bail!("only one kernel, got `{}` and `{extra}`; {USAGE}", o.kernel),
            }
        }
        if o.all {
            if !o.kernel.is_empty() {
                bail!("`--all` or a kernel name, not both; {USAGE}");
            }
            o.kernel = "all".to_string();
        } else if o.kernel.is_empty() {
            bail!("{USAGE}");
        }
        if o.writes() {
            if !o.cycles {
                bail!("the results are cycles: `--no-cycles` cannot write them");
            }
            if o.cycle_mode != crate::cycle_results::MODE {
                bail!(
                    "the results are judged in the mode {}; `--cycle-mode` does not write them",
                    crate::cycle_results::MODE.name()
                );
            }
            if o.cycle_counter.is_some() {
                bail!(
                    "the results take the counts of the profiler of each level; `--cycle-counter` does not write them"
                );
            }
            if o.all && !o.merge && (!o.levels.is_empty() || o.shard.is_some()) {
                bail!(
                    "`--all` writes every entry of a platform anew; with `--level` or `--shard` use `--merge`"
                );
            }
            o.branches = true;
        }
        Ok(o)
    }
}

impl Options {
    /// The entries of `names` in this run's shard (all without `--shard`), in sorted order.
    #[must_use]
    pub fn shard_of(&self, mut names: Vec<String>) -> Vec<String> {
        names.sort();
        match self.shard {
            None => names,
            Some((i, n)) => names
                .into_iter()
                .enumerate()
                .filter(|(k, _)| k % n == i - 1)
                .map(|(_, name)| name)
                .collect(),
        }
    }

    /// The directory of the results files.
    #[must_use]
    pub fn results_dir(&self, root: &Path) -> PathBuf {
        self.results
            .clone()
            .unwrap_or_else(|| crate::cycle_results::dir(root))
    }

    /// Whether the run writes the committed results (`--merge`, or `--all`).
    #[must_use]
    pub fn writes(&self) -> bool {
        self.merge || self.all
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

/// The golden kernels (directories of `tests/golden/vectors`) selected by `kernel`; every one
/// for `None` (`--all`).
///
/// # Errors
/// The directory cannot be read.
#[allow(
    clippy::large_stack_frames,
    reason = "the Windows file system calls are large; xtask is a host tool"
)]
pub fn golden_kernels(root: &Path, kernel: Option<&str>) -> anyhow::Result<Vec<String>> {
    let dir = root.join("tests").join("golden").join("vectors");
    let mut out = Vec::new();
    for entry in
        std::fs::read_dir(&dir).with_context(|| format!("cannot read {}", dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && let Some(name) = entry.file_name().to_str()
            && kernel.is_none_or(|k| selects(k, name))
        {
            out.push(name.to_string());
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// The platforms that `cargo measure` simulates by default on a host of `arch`.
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

/// The levels of `platform` that `o` selects.
fn selected_levels(o: &Options, platform: &str) -> Vec<&'static str> {
    PLANS
        .iter()
        .filter(|p| p.platform == platform)
        .filter(|p| o.levels.is_empty() || o.levels.iter().any(|l| l == p.level))
        .map(|p| p.level)
        .collect()
}

/// The simulation of one side of an entry on both models of the level, in `mode`.
fn simulate_side(
    counts: &crate::mca::Counts,
    samples: u64,
    models: &crate::mca::Models,
    mode: crate::mca::CycleMode,
) -> anyhow::Result<[crate::mca::Simulation; 2]> {
    let profile = crate::mca::profile(counts, mode)?;
    Ok([
        crate::mca::simulate(&profile, samples, models.triple, models.reference)?,
        crate::mca::simulate(&profile, samples, models.triple, models.counter)?,
    ])
}

fn simulation_json(s: &crate::mca::Simulation) -> Value {
    json!({
        "cpu": s.cpu,
        "cycle_mode": s.mode.name(),
        "instructions_per_sample": s.instructions_per_sample,
        "cycles_per_sample": s.cycles_per_sample,
        "port_bound_per_sample": s.port_bound_per_sample,
        "efficiency": s.efficiency,
        "bottleneck": crate::mca::bottleneck(s.efficiency),
        "coverage": s.coverage,
        "blocks": s.blocks,
        "loops": s.loops,
        "moves_as_lea": s.moves_as_lea,
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
            let Some(profiler) = o.cycle_counter.or_else(|| crate::bench::Profiler::of(plan))
            else {
                let _ = writeln!(
                    part.table,
                    "\ncycles, {platform} {level}: no execution counts per block (not simulated)"
                );
                continue;
            };
            match cycles_of_level(root, plan, models, entries, o.cycle_mode, profiler) {
                Ok((table, value, entries_ok)) => {
                    // An entry that failed fails the run, as in the branches: its note must not
                    // replace a committed record unnoticed.
                    part.ok &= entries_ok;
                    let _ = writeln!(
                        part.table,
                        "\ncycles, {platform} {level} (llvm-mca; reference model {}, counter-model {}; judged in the mode {}; counts: {}; only the ratio means something):",
                        models.reference,
                        models.counter,
                        o.cycle_mode.name(),
                        profiler.describe(plan),
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

/// The cycles table and JSON of one level: per entry both cycle modes, `mode` first (the
/// top-level fields of the entry), the other one under `other_cycle_mode`; false when an entry
/// failed (an entry that is not measured at the level is no failure).
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and serde_json values are large; xtask is a host tool"
)]
fn cycles_of_level(
    root: &Path,
    plan: &crate::bench::LevelPlan,
    models: &crate::mca::Models,
    entries: &[String],
    mode: crate::mca::CycleMode,
    profiler: crate::bench::Profiler,
) -> anyhow::Result<(String, Value, bool)> {
    let mut table = String::new();
    let mut ok = true;
    let _ = writeln!(
        table,
        "{:<26} {:<10} {:<23} {:>9} {:>9} {:>6} {:>6} {:<10} {:>8}  comparison",
        "entry", "mode", "model", "ours", "other", "ratio", "eff.", "bottleneck", "coverage"
    );
    let both: Vec<crate::mca::CycleMode> = std::iter::once(mode)
        .chain(
            crate::mca::CycleMode::ALL
                .into_iter()
                .filter(|&m| m != mode),
        )
        .collect();
    // Every value of the level is an entry (existing readers iterate them); the counter is
    // named in each.
    let counts = profiler.describe(plan);
    let mut value = serde_json::Map::new();
    for (kernel, e) in crate::bench::profiles(root, plan.platform, plan.level, entries, profiler)? {
        match e.and_then(|e| entry_cycles(&e, models, &both, &mut table)) {
            Ok(mut v) => {
                if let Value::Object(m) = &mut v {
                    m.insert("block_counts".into(), json!(counts));
                }
                value.insert(kernel, v);
            }
            Err(err) => {
                ok &= err.to_string().starts_with("not measured at the level");
                let note = format!("not simulated: {err:#}");
                let _ = writeln!(table, "{kernel:<26} {note}");
                value.insert(kernel, json!({ "note": note, "block_counts": counts }));
            }
        }
    }
    table.push_str(&fallback_verdicts(&value));
    Ok((table, Value::Object(value), ok))
}

/// The cycles per sample of ours of a simulated entry on `model` (`reference_model` or
/// `counter_model`).
fn ours_cycles(entry: &Value, model: &str) -> Option<f64> {
    entry
        .get(model)?
        .get("ours")?
        .get("cycles_per_sample")?
        .as_f64()
}

/// The verdict of the gate of D321 (`cargo xtask check cycles`) for every simulated entry
/// `<k>` whose `<k>_fallback` was simulated too: SIMD / fallback on both models, rounded to the
/// two decimals of the published tables as the gate compares them.
fn fallback_verdicts(entries: &serde_json::Map<String, Value>) -> String {
    let mut text = String::new();
    for (kernel, simd) in entries {
        let Some(fallback) = entries.get(&format!("{kernel}_fallback")) else {
            continue;
        };
        let models = ["reference_model", "counter_model"].map(|m| {
            ours_cycles(simd, m)
                .zip(ours_cycles(fallback, m))
                .map(|(s, f)| ((s * 100.0).round(), (f * 100.0).round()))
        });
        let [Some(r), Some(c)] = models else {
            continue;
        };
        let delegated = simd.get("compared_with").and_then(Value::as_str) == Some("sdrk fallback")
            && ["reference_model", "counter_model"].iter().all(|m| {
                simd.get(*m)
                    .and_then(|v| v.get("ratio"))
                    .and_then(Value::as_f64)
                    .is_some_and(|x| (x * 100.0).round() == 100.0)
            });
        let ok = delegated || (r.0 <= r.1 && c.0 <= c.1);
        let _ = writeln!(
            text,
            "SIMD vs fallback {kernel}: reference {:.2} / {:.2} ({:.2}), counter {:.2} / {:.2} ({:.2}): {}",
            r.0 / 100.0,
            r.1 / 100.0,
            r.0 / r.1,
            c.0 / 100.0,
            c.1 / 100.0,
            c.0 / c.1,
            if ok {
                "ok"
            } else {
                "SLOWER than the fallback (gate D321)"
            }
        );
    }
    text
}

/// The cycles of one entry in both modes (`both`, the judged one first); the rows go to
/// `table`.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and serde_json values are large; xtask is a host tool"
)]
fn entry_cycles(
    e: &crate::bench::EntryProfiles,
    models: &crate::mca::Models,
    both: &[crate::mca::CycleMode],
    table: &mut String,
) -> anyhow::Result<Value> {
    // The rows are written only when the whole entry is simulated.
    let mut rows_text = String::new();
    let mut per_mode = Vec::new();
    for (k, &m) in both.iter().enumerate() {
        let [ours_ref, ours_counter] = simulate_side(&e.ours, e.samples, models, m)?;
        let [other_ref, other_counter] = simulate_side(&e.other, e.samples, models, m)?;
        let rows = [
            ("reference", &ours_ref, &other_ref),
            ("counter", &ours_counter, &other_counter),
        ];
        for (n, (role, ours, other)) in rows.iter().enumerate() {
            let first = k == 0 && n == 0;
            let _ = writeln!(
                rows_text,
                "{:<26} {:<10} {:<23} {:>9.2} {:>9.2} {:>6.2} {:>6.2} {:<10} {:>7.1}%  {}",
                if first { e.kernel.as_str() } else { "" },
                if n == 0 { m.name() } else { "" },
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
        per_mode.push(json!({
            "cycle_mode": m.name(),
            "reference_model": model_json(&ours_ref, &other_ref),
            "counter_model": model_json(&ours_counter, &other_counter),
            "counter_vs_reference": ratio_counter / ratio_ref,
        }));
    }
    let mut entry = serde_json::Map::new();
    entry.insert("compared_with".into(), json!(e.compared_with));
    entry.insert("samples".into(), json!(e.samples));
    entry.insert(
        "snr_db".into(),
        json!({ "ours": round2(e.ours_snr_db), "comparison": round2(e.other_snr_db) }),
    );
    let mut per_mode = per_mode.into_iter();
    if let Some(Value::Object(primary)) = per_mode.next() {
        entry.extend(primary);
    }
    if let Some(other) = per_mode.next() {
        entry.insert("other_cycle_mode".into(), other);
    }
    table.push_str(&rows_text);
    let _ = writeln!(
        table,
        "{:<26} accuracy against the f64 reference: ours {} dB, comparison {} dB (D34)",
        "",
        crate::cycle_gate::snr_text(e.ours_snr_db),
        crate::cycle_gate::snr_text(e.other_snr_db),
    );
    Ok(Value::Object(entry))
}

/// `x` rounded to two decimals, as the results files keep the SNR.
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// The mispredictions per sample of the selected entries (D1611) on every x86-64 level that
/// Valgrind runs, and with `cycles` (the document of [`measure_cycles`]) the cost of both sides
/// on both models of the level: `cycles + mispredictions × penalty`, in the judged cycle mode.
#[allow(
    clippy::large_stack_frames,
    reason = "the plan tables and std::process::Command are large; xtask is a host tool"
)]
fn measure_branches(root: &Path, o: &Options, entries: &[String], cycles: Option<&Value>) -> Part {
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
            if plan.baseline.is_some() {
                continue;
            }
            if !crate::branches::simulated_level(level) {
                let why = match level {
                    "neon" => {
                        "no branch simulation (D1611: no penalty if the AArch64 assembly is branch-free on the data path, else the rate of `sse2`)"
                    }
                    "avx512" => "no branch simulation (Valgrind runs no AVX-512; open item I44-16)",
                    _ => "no branch simulation (D1611: instructions only)",
                };
                let _ = writeln!(part.table, "\nbranches, {platform} {level}: {why}");
                continue;
            }
            let models = crate::mca::models(level);
            let level_cycles = cycles
                .and_then(|c| c.get(platform.as_str()))
                .and_then(|c| c.get(level));
            match crate::bench::branch_profiles(root, platform, level, entries) {
                Ok(list) => {
                    let _ = writeln!(
                        part.table,
                        "\nbranches, {platform} {level} (valgrind cachegrind --branch-sim=yes, per sample; cost = cycles + mispredictions x penalty, D1611):"
                    );
                    let _ = writeln!(
                        part.table,
                        "{:<26} {:>10} {:>10} {:>10} {:>10}  {:<16} {:>7} {:>9} {:>9} {:>9} {:>9}  comparison",
                        "entry",
                        "branches",
                        "mispred.",
                        "other br.",
                        "other mis.",
                        "model",
                        "penalty",
                        "cycles",
                        "cost",
                        "other cyc.",
                        "other cost"
                    );
                    let mut value = serde_json::Map::new();
                    for (kernel, e) in list {
                        match e {
                            Ok(e) => {
                                let v = entry_branches(
                                    &kernel,
                                    &e,
                                    models,
                                    level_cycles.and_then(|c| c.get(&kernel)),
                                    &mut part.table,
                                );
                                value.insert(kernel, v);
                            }
                            Err(err) => {
                                // An entry that is not measured at the level is no failure, as
                                // in the cycles.
                                part.ok &= err.to_string().starts_with("not measured at the level");
                                let note = format!("not counted: {err:#}");
                                let _ = writeln!(part.table, "{kernel:<26} {note}");
                                value.insert(kernel, json!({ "note": note }));
                            }
                        }
                    }
                    levels_doc.insert(level.to_string(), Value::Object(value));
                }
                Err(e) => {
                    part.ok = false;
                    let _ = writeln!(
                        part.table,
                        "\nbranches, {platform} {level}: not counted: {e:#}"
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

/// The rows and JSON of one entry of [`measure_branches`]; `cycles` is the entry of the cycles
/// document of the level, if the cycles were simulated.
#[allow(
    clippy::cast_precision_loss,
    reason = "branch counts are far below 2^52"
)]
#[allow(
    clippy::large_stack_frames,
    reason = "serde_json values are large; xtask is a host tool"
)]
fn entry_branches(
    kernel: &str,
    e: &crate::bench::EntryBranches,
    models: Option<&crate::mca::Models>,
    cycles: Option<&Value>,
    table: &mut String,
) -> Value {
    let per = |n: u64| n as f64 / e.samples as f64;
    let side = |b: &crate::branches::Branches| {
        json!({
            "conditional": b.conditional,
            "conditional_mispredicted": b.conditional_mispredicted,
            "indirect": b.indirect,
            "indirect_mispredicted": b.indirect_mispredicted,
            "branches_per_sample": per(b.conditional + b.indirect),
            "mispredictions_per_sample": per(b.mispredicted()),
        })
    };
    let ours_m = per(e.ours.mispredicted());
    let other_m = per(e.other.mispredicted());
    let num = |x: Option<f64>| x.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"));
    let mut costs = serde_json::Map::new();
    let mut rows: Vec<String> = Vec::new();
    if let Some(models) = models {
        for (key, cpu) in [
            ("reference_model", models.reference),
            ("counter_model", models.counter),
        ] {
            let Some(p) = crate::branches::penalty(cpu) else {
                continue;
            };
            let side_cycles = |s: &str| {
                cycles
                    .and_then(|c| c.get(key))
                    .and_then(|m| m.get(s))
                    .and_then(|m| m.get("cycles_per_sample"))
                    .and_then(Value::as_f64)
            };
            let (ours_c, other_c) = (side_cycles("ours"), side_cycles("other"));
            let ours_cost = ours_c.map(|c| crate::branches::cost(c, ours_m, p));
            let other_cost = other_c.map(|c| crate::branches::cost(c, other_m, p));
            costs.insert(
                key.to_string(),
                json!({
                    "cpu": cpu,
                    "penalty": p,
                    "ours": { "cycles_per_sample": ours_c, "cost_per_sample": ours_cost },
                    "other": { "cycles_per_sample": other_c, "cost_per_sample": other_cost },
                }),
            );
            rows.push(format!(
                "{cpu:<16} {p:>7.0} {:>9} {:>9} {:>9} {:>9}",
                num(ours_c),
                num(ours_cost),
                num(other_c),
                num(other_cost)
            ));
        }
    }
    let _ = writeln!(
        table,
        "{kernel:<26} {:>10.4} {:>10.4} {:>10.4} {:>10.4}  {}  {}",
        per(e.ours.conditional + e.ours.indirect),
        ours_m,
        per(e.other.conditional + e.other.indirect),
        other_m,
        rows.first().map_or("", String::as_str),
        e.compared_with
    );
    for r in rows.iter().skip(1) {
        let _ = writeln!(table, "{:<26} {:>43}  {r}", "", "");
    }
    json!({
        "compared_with": e.compared_with,
        "samples": e.samples,
        "counter": "valgrind cachegrind --branch-sim=yes (Bcm + Bim)",
        "ours": side(&e.ours),
        "other": side(&e.other),
        "cost": costs,
    })
}

/// The golden cases of the selected golden kernels.
fn measure_golden(root: &Path, report: &Path, golden: &[String]) -> anyhow::Result<Part> {
    let report = run_golden(root, report)?;
    let cases = summarise_golden(&report, golden)?;
    let mut table = String::from(
        "\ngolden cases (dB; distance = optimum - measured; the optimum is the expected value rounded once to the output format, for block-float CQ15 a Q15 mantissa with the block exponent it needs):\n",
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
/// Unknown options, a kernel that selects nothing, or a file that cannot be written. A level
/// that cannot be simulated (a missing profiler or reference) is reported in the result, not as
/// an error; the return value is `Ok(false)` if a level could not be simulated, or a golden case
/// failed.
#[allow(
    clippy::large_stack_frames,
    reason = "the kernel table and serde_json values are large; xtask is a host tool"
)]
pub fn run(root: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let o = Options::parse(args)?;
    let (entries, golden) = if o.all {
        (
            KERNELS.iter().map(|s| s.name.to_string()).collect(),
            golden_kernels(root, None)?,
        )
    } else {
        (
            count_entries(&o.kernel),
            golden_kernels(root, Some(&o.kernel))?,
        )
    };
    let (entries, golden) = (o.shard_of(entries), o.shard_of(golden));
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
    // Checked before the long runs: an uncorrected llvm-mca fails at once.
    if o.writes() && !entries.is_empty() {
        crate::cycle_results::require_corrected_llvm_mca()?;
    }
    let cycles = (o.cycles && !entries.is_empty()).then(|| measure_cycles(root, &o, &entries));
    let branches = (o.branches && !entries.is_empty())
        .then(|| measure_branches(root, &o, &entries, cycles.as_ref().map(|p| &p.doc)));
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
        "golden": golden_part.as_ref().map(|p| &p.doc),
        "cycles": cycles.as_ref().map(|p| &p.doc),
        "branches": branches.as_ref().map(|p| &p.doc),
    });
    let path: PathBuf = out_dir.join(format!("{}.json", o.kernel));
    let mut text = serde_json::to_string_pretty(&doc)?;
    text.push('\n');
    std::fs::write(&path, &text)?;
    let mut written = Vec::new();
    // A failed entry writes nothing: its note would replace the committed record.
    let unwritten = o.writes() && cycles.as_ref().is_some_and(|c| !c.ok);
    if let Some(cycles) = cycles.as_ref().filter(|c| o.writes() && c.ok) {
        let platforms = if o.platforms.is_empty() {
            default_platforms(std::env::consts::ARCH)
        } else {
            o.platforms.clone()
        };
        for platform in &platforms {
            // After the runs: the manifest of the C references exists once they are built.
            let tools = crate::cycle_results::tools(root, platform)?;
            let measured = crate::cycle_results::levels_of(
                field(&cycles.doc, platform),
                branches.as_ref().map(|b| field(&b.doc, platform)),
            )?;
            written.push(crate::cycle_results::write(
                &o.results_dir(root),
                platform,
                &tools,
                &measured,
                o.all && !o.merge,
            )?);
        }
    }
    if o.json {
        print!("{text}");
    } else {
        println!("\nmeasure {}", o.kernel);
        for part in cycles.iter().chain(&branches).chain(&golden_part) {
            print!("{}", part.table);
        }
        println!("\nwritten to {}", path.display());
        for p in &written {
            println!("results in {}", p.display());
        }
    }
    if unwritten {
        eprintln!("results not written: the cycles of an entry or a level failed");
    }
    Ok(cycles
        .iter()
        .chain(&branches)
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
        assert_eq!(
            count_entries("fir_decimate"),
            [
                "fir_decimate",
                "fir_decimate_cq15",
                "fir_decimate_fallback",
                "fir_decimate_cq15_fallback"
            ]
        );
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
        assert!(o.cycles && !o.golden && o.json);
        assert!(Options::parse(&["fir", "--no-count"]).is_err());
        assert_eq!(o.cycle_mode, crate::mca::CycleMode::WholeLoop);
        let o = Options::parse(&["fir", "--cycle-mode", "per-block"]).unwrap();
        assert_eq!(o.cycle_mode, crate::mca::CycleMode::PerBlock);
        assert!(Options::parse(&["fir", "--cycle-mode", "both"]).is_err());
        let o = Options::parse(&["fir", "--cycle-counter", "sde"]).unwrap();
        assert_eq!(o.cycle_counter, Some(crate::bench::Profiler::Sde));
        assert!(Options::parse(&["fir", "--cycle-counter", "pin"]).is_err());
        assert!(!o.branches);
        assert!(Options::parse(&["fir", "--branches"]).unwrap().branches);
    }

    #[test]
    fn writing_the_results_needs_the_judged_measurement() {
        let o = Options::parse(&["fir", "--merge"]).unwrap();
        assert!(o.writes() && o.branches && !o.all);
        let o = Options::parse(&["--all", "--no-golden"]).unwrap();
        assert!(o.all && o.writes() && o.branches);
        assert_eq!(o.kernel, "all");
        assert!(!Options::parse(&["fir"]).unwrap().writes());
        assert!(Options::parse(&["--all", "fir"]).is_err());
        assert!(Options::parse(&["--all", "--level", "avx2"]).is_err());
        assert!(Options::parse(&["--all", "--level", "avx2", "--merge"]).is_ok());
        assert!(Options::parse(&["fir", "--merge", "--no-cycles"]).is_err());
        assert!(Options::parse(&["fir", "--merge", "--cycle-mode", "per-block"]).is_err());
        assert!(Options::parse(&["fir", "--merge", "--cycle-counter", "sde"]).is_err());
    }

    #[test]
    fn a_shard_takes_every_nth_entry_of_the_sorted_selection_into_its_own_results() {
        let o =
            Options::parse(&["--all", "--merge", "--shard", "2/3", "--results", "out"]).unwrap();
        assert_eq!(o.shard, Some((2, 3)));
        assert_eq!(o.results_dir(Path::new("/r")), PathBuf::from("out"));
        let names = ["e", "a", "d", "b", "c"].map(String::from).to_vec();
        assert_eq!(o.shard_of(names.clone()), ["b", "e"]);
        let all = Options::parse(&["fir"]).unwrap();
        assert_eq!(all.shard_of(names), ["a", "b", "c", "d", "e"]);
        assert_eq!(
            all.results_dir(Path::new("/r")),
            Path::new("/r")
                .join("benches")
                .join("results")
                .join("cycles")
        );
        // A shard of `--all` would write a partial file anew.
        assert!(Options::parse(&["--all", "--shard", "1/2"]).is_err());
        for bad in ["0/2", "3/2", "x", "1/0"] {
            assert!(Options::parse(&["fir", "--shard", bad]).is_err(), "{bad}");
        }
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

    fn simulated(compared_with: &str, reference: (f64, f64), counter: (f64, f64)) -> Value {
        let model = |(ours, other): (f64, f64)| {
            json!({
                "ours": { "cycles_per_sample": ours },
                "other": { "cycles_per_sample": other },
                "ratio": ours / other,
            })
        };
        json!({
            "compared_with": compared_with,
            "reference_model": model(reference),
            "counter_model": model(counter),
        })
    }

    #[test]
    fn the_fallback_verdict_compares_the_published_two_decimals_on_both_models() {
        let mut m = serde_json::Map::new();
        m.insert("k".into(), simulated("volk", (0.1966, 0.2), (0.132, 0.18)));
        m.insert(
            "k_fallback".into(),
            simulated("cmsis", (0.1943, 0.2), (0.1368, 0.18)),
        );
        let text = fallback_verdicts(&m);
        assert!(
            text.contains("SIMD vs fallback k: reference 0.20 / 0.19"),
            "{text}"
        );
        assert!(text.contains("SLOWER"), "{text}");
        m.insert("k".into(), simulated("volk", (0.1949, 0.2), (0.132, 0.18)));
        assert!(fallback_verdicts(&m).ends_with(": ok\n"));
        // Delegated to the fallback: passes whatever the rounding of the two runs.
        m.insert(
            "k".into(),
            simulated("sdrk fallback", (0.1960, 0.1961), (0.14, 0.1401)),
        );
        assert!(fallback_verdicts(&m).ends_with(": ok\n"));
        m.remove("k_fallback");
        assert_eq!(fallback_verdicts(&m), "");
    }
}
