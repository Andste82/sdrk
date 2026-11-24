//! `cargo xtask check cycles`: the speed gates in simulated cycles per sample, judged on the
//! committed cycle results `benches/results/cycles/<platform>.json` (D2704), so CI enforces them
//! without running `llvm-mca`.
//!
//! **SIMD against its own fallback (D321).** For every level and every entry `<k>` whose
//! results also have `<k>_fallback`, the cycles per sample of ours, rounded to the two decimals
//! of the published tables, must satisfy `cycles(<k>) <= cycles(<k>_fallback)` on the reference
//! model **and** on the counter-model of the level. A level delegated to the fallback
//! (comparison `sdrk fallback`, both ratios `1.00`) passes. A violation passes only with a
//! `cycle_waiver` on the SIMD entry and level in `benches/limits.json` that names an open item
//! of `docs/open-items.md`; a `cycle_waiver` where the rule holds, or on an entry and level
//! without such a pair, is a finding as well, so that no waiver outlives its gap.
//!
//! **The release gate (D14 in cycles, M13.1 §3, D2708).** Every simulated entry and level is
//! judged by its entry in `benches/limits.json`: `release` needs `cycles(ours) <=
//! _limits.release × cycles(comparison)` on **both** models of the level, `dev` the same with
//! `_limits.dev`; `info` is not judged. The level `avx2-runtime` is judged against our own
//! `avx2` build with `_limits.runtime` (D2702); an entry for it is needed only to relax that.
//! An exceedance passes only with a `waiver` whose open items exist (a waiver that names none
//! states the reason in its text, as `release-check` accepts); a waiver of an entry within its
//! limit, a `dev` entry within `_limits.warn` (ready for `release`) and a ratio above
//! `_limits.warn` are reported, not failed; `_limits.goal` is reported per entry. An entry
//! without a record of `limits.json` on a level that is simulated fails, and so does a
//! `release` or `dev` entry without a simulated record (except for a selection, `--partial`).
//! The instruction judgement of `simd128` stays in `cargo xtask count`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow, bail};
use serde_json::Value;

use crate::count::{Caps, Limits, Status};
use crate::scan::Finding;

/// The comparison of a SIMD entry that runs our fallback on its level.
const OWN_FALLBACK: &str = "sdrk fallback";
/// The suffix of the fallback entry of a kernel.
const FALLBACK: &str = "_fallback";
/// The level judged against our own build of another level (D2702).
const RUNTIME_LEVEL: &str = "avx2-runtime";
/// The note of an entry that the level does not measure.
const NOT_MEASURED: &str = "not simulated: not measured at";

/// One model of a record: the cycles per sample of ours and of the comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// The core model (`znver3`, ...).
    pub cpu: String,
    /// Cycles per sample of ours.
    pub ours: f64,
    /// Cycles per sample of the comparison.
    pub comparison: f64,
    /// `ours / comparison`.
    pub ratio: f64,
}

/// One simulated entry and level of the results.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    /// The results file it comes from.
    pub file: PathBuf,
    /// The comparison.
    pub compared_with: String,
    /// The reference model.
    pub reference: Model,
    /// The counter-model.
    pub counter: Model,
}

/// The records of the results, by level, then entry: `Some` for a simulated entry, `None` for
/// one that failed (its note says why); entries that the level does not measure are left out.
pub type Results = BTreeMap<String, BTreeMap<String, Option<Record>>>;

/// A number of the record at `path`.
fn number(v: &Value, path: &[&str]) -> Option<f64> {
    path.iter().try_fold(v, |v, k| v.get(*k))?.as_f64()
}

/// One model of a record.
fn model(r: &Value, key: &str) -> Option<Model> {
    let m = r.get(key)?;
    Some(Model {
        cpu: m.get("cpu")?.as_str()?.to_string(),
        ours: number(m, &["ours", "cycles_per_sample"])?,
        comparison: number(m, &["comparison", "cycles_per_sample"])?,
        ratio: m.get("ratio")?.as_f64()?,
    })
}

/// Adds the records of one results document (`file` names it in the messages) to `into`.
///
/// # Errors
/// Invalid JSON, a document without `levels`, a level that is already in `into` (two files
/// with one level), or a simulated record without both models.
pub fn add_results(file: &Path, text: &str, into: &mut Results) -> anyhow::Result<()> {
    let doc: Value =
        serde_json::from_str(text).with_context(|| format!("cannot parse {}", file.display()))?;
    let levels = doc
        .get("levels")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("{} has no `levels`", file.display()))?;
    for (level, l) in levels {
        if into.contains_key(level) {
            bail!("the level {level} is in two results files");
        }
        let mut records = BTreeMap::new();
        for (entry, r) in l
            .get("entries")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            if r.get("reference_model").is_none() {
                let note = r.get("note").and_then(Value::as_str).unwrap_or_default();
                if !note.starts_with(NOT_MEASURED) {
                    records.insert(entry.clone(), None);
                }
                continue;
            }
            let (Some(reference), Some(counter)) =
                (model(r, "reference_model"), model(r, "counter_model"))
            else {
                bail!(
                    "{}: `{entry}` at `{level}` has no cycles of both models",
                    file.display()
                );
            };
            records.insert(
                entry.clone(),
                Some(Record {
                    file: file.to_path_buf(),
                    compared_with: r
                        .get("compared_with")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    reference,
                    counter,
                }),
            );
        }
        into.insert(level.clone(), records);
    }
    Ok(())
}

/// The records of every `*.json` in `dir`.
///
/// # Errors
/// The directory or a file cannot be read or parsed.
#[allow(
    clippy::large_stack_frames,
    reason = "the Windows file system calls are large; xtask is a host tool"
)]
pub fn load(dir: &Path) -> anyhow::Result<Results> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    let mut out = Results::new();
    for f in &files {
        let text =
            std::fs::read_to_string(f).with_context(|| format!("cannot read {}", f.display()))?;
        add_results(f, &text, &mut out)?;
    }
    Ok(out)
}

/// Cycles per sample in hundredths, as the tables publish them.
fn hundredths(x: f64) -> i64 {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "cycles per sample are small numbers"
    )]
    let h = (x * 100.0).round() as i64;
    h
}

/// One SIMD entry and level compared with its fallback (D321).
#[derive(Clone, Debug, PartialEq)]
pub struct Pair {
    /// The SIMD entry.
    pub entry: String,
    /// The level.
    pub level: String,
    /// The SIMD record.
    pub simd: Record,
    /// The record of `<entry>_fallback`.
    pub fallback: Record,
}

impl Pair {
    /// Whether the level runs our fallback (comparison `sdrk fallback`, both ratios 1.00).
    #[must_use]
    pub fn delegated(&self) -> bool {
        self.simd.compared_with == OWN_FALLBACK
            && hundredths(self.simd.reference.ratio) == 100
            && hundredths(self.simd.counter.ratio) == 100
    }

    /// Whether the SIMD entry is at most as slow as its fallback on both models, in the two
    /// decimals of the tables.
    #[must_use]
    pub fn holds(&self) -> bool {
        self.delegated()
            || (hundredths(self.simd.reference.ours) <= hundredths(self.fallback.reference.ours)
                && hundredths(self.simd.counter.ours) <= hundredths(self.fallback.counter.ours))
    }

    /// The cycles of both sides, for the messages.
    #[must_use]
    pub fn describe(&self) -> String {
        let c = |x: f64| format!("{:.2}", hundredths(x) as f64 / 100.0);
        let ratio = |a: f64, b: f64| {
            let (a, b) = (hundredths(a), hundredths(b));
            if b == 0 {
                "-".to_string()
            } else {
                format!("{:.2}", a as f64 / b as f64)
            }
        };
        let (s, f) = (&self.simd, &self.fallback);
        format!(
            "reference model {} / {} ({}), counter-model {} / {} ({}) cycles per sample, SIMD / fallback",
            c(s.reference.ours),
            c(f.reference.ours),
            ratio(s.reference.ours, f.reference.ours),
            c(s.counter.ours),
            c(f.counter.ours),
            ratio(s.counter.ours, f.counter.ours),
        )
    }
}

/// Every simulated SIMD entry with a simulated fallback record on the same level.
#[must_use]
pub fn pairs(results: &Results) -> Vec<Pair> {
    let mut out = Vec::new();
    for (level, rows) in results {
        for (entry, simd) in rows {
            if entry.ends_with(FALLBACK) {
                continue;
            }
            if let (Some(simd), Some(Some(fallback))) =
                (simd, rows.get(&format!("{entry}{FALLBACK}")))
            {
                out.push(Pair {
                    entry: entry.clone(),
                    level: level.clone(),
                    simd: simd.clone(),
                    fallback: fallback.clone(),
                });
            }
        }
    }
    out
}

/// The `cycle_waiver` texts of `benches/limits.json`, by entry and level.
///
/// # Errors
/// Invalid JSON, or a `cycle_waiver` that is not a string.
pub fn cycle_waivers(limits_json: &str) -> anyhow::Result<BTreeMap<(String, String), String>> {
    let limits: Value = serde_json::from_str(limits_json).context("benches/limits.json")?;
    let mut out = BTreeMap::new();
    let Some(kernels) = limits.as_object() else {
        return Ok(out);
    };
    for (kernel, levels) in kernels.iter().filter(|(k, _)| !k.starts_with('_')) {
        let Some(levels) = levels.as_object() else {
            continue;
        };
        for (level, entry) in levels {
            if let Some(w) = entry.get("cycle_waiver") {
                let text = w.as_str().with_context(|| {
                    format!("benches/limits.json: `{kernel}.{level}.cycle_waiver` must be a string")
                })?;
                out.insert((kernel.clone(), level.clone()), text.to_string());
            }
        }
    }
    Ok(out)
}

/// The open items a waiver names (`open item <ID>`) that `docs/open-items.md` does not have.
fn missing_open_items(text: &str, open_items: &str) -> (Vec<String>, Vec<String>) {
    let items: Vec<String> = crate::release::references(text)
        .into_iter()
        .filter_map(|r| r.strip_prefix("open item ").map(str::to_string))
        .collect();
    let missing = items
        .iter()
        .filter(|id| !open_items.contains(&format!("| {id}:")))
        .cloned()
        .collect();
    (items, missing)
}

/// The outcome of the gate of D321.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Violations without a valid waiver, and stale or invalid waivers.
    pub findings: Vec<Finding>,
    /// The waived violations: the pair and the waiver text.
    pub waived: Vec<(Pair, String)>,
    /// The number of pairs judged.
    pub judged: usize,
}

/// Judges every SIMD and fallback pair of `results` (D321), with the waivers of `limits_json`
/// and the open items of `open_items`.
///
/// # Errors
/// `benches/limits.json` cannot be parsed.
#[allow(
    clippy::large_stack_frames,
    reason = "the findings and waived pairs are built in place; xtask is a host tool"
)]
pub fn judge_fallbacks(
    results: &Results,
    limits_json: &str,
    open_items: &str,
    partial: bool,
) -> anyhow::Result<Outcome> {
    let mut waivers = cycle_waivers(limits_json)?;
    let mut out = Outcome::default();
    let limits = PathBuf::from("benches/limits.json");
    for pair in pairs(results) {
        out.judged += 1;
        let waiver = waivers.remove(&(pair.entry.clone(), pair.level.clone()));
        match (pair.holds(), waiver) {
            (true, None) => {}
            (true, Some(_)) => out.findings.push(Finding {
                path: limits.clone(),
                line: 0,
                message: format!(
                    "`{}` at `{}` has a `cycle_waiver` but is not slower than its fallback ({}); remove the waiver",
                    pair.entry,
                    pair.level,
                    pair.describe()
                ),
            }),
            (false, None) => out.findings.push(Finding {
                path: pair.simd.file.clone(),
                line: 0,
                message: format!(
                    "`{}` at `{}` is slower than `{}{FALLBACK}` in simulated cycles: {}; make the SIMD path at least as fast, or add a `cycle_waiver` naming an open item to benches/limits.json (D321)",
                    pair.entry,
                    pair.level,
                    pair.entry,
                    pair.describe()
                ),
            }),
            (false, Some(text)) => {
                let (items, missing) = missing_open_items(&text, open_items);
                if items.is_empty() || !missing.is_empty() {
                    out.findings.push(Finding {
                        path: limits.clone(),
                        line: 0,
                        message: format!(
                            "the `cycle_waiver` of `{}` at `{}` must name an open item of docs/open-items.md (`open item <ID>`); missing: {}",
                            pair.entry,
                            pair.level,
                            if items.is_empty() {
                                "none named".to_string()
                            } else {
                                missing.join(", ")
                            }
                        ),
                    });
                } else {
                    out.waived.push((pair, text));
                }
            }
        }
    }
    for ((entry, level), _) in waivers {
        // A selection (`--partial`) judges only the pairs it has.
        if partial {
            continue;
        }
        out.findings.push(Finding {
            path: limits.clone(),
            line: 0,
            message: format!(
                "`{entry}` at `{level}` has a `cycle_waiver`, but the cycle results have no simulated `{entry}` / `{entry}{FALLBACK}` pair at that level; remove the waiver"
            ),
        });
    }
    Ok(out)
}

/// The verdict of one entry and level under the release gate.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Within its limit on both models.
    Pass,
    /// Within its limit, with remarks (a waiver no longer needed, a `dev` entry ready for
    /// `release`, a ratio above `_limits.warn`).
    Warn(String),
    /// Above its limit on a model, with a waiver.
    Waived(String),
    /// Above its limit without a waiver, a waiver that names a missing open item, or no entry
    /// in `benches/limits.json`.
    Fail(String),
    /// `info`: not judged.
    Info,
}

/// The two ratios with as many decimals as it takes to show the one above `cap`.
fn ratios(r: &Record, cap: f64) -> String {
    let show = |x: f64| {
        let digits = (2..6)
            .find(|d| {
                format!("{x:.d$}", d = *d)
                    .parse::<f64>()
                    .is_ok_and(|v| (v > cap) == (x > cap))
            })
            .unwrap_or(6);
        format!("{x:.digits$}")
    };
    format!(
        "{} / {} ({} / {} cycles per sample on {} / {})",
        show(r.reference.ratio),
        show(r.counter.ratio),
        format_args!("{:.2} vs {:.2}", r.reference.ours, r.reference.comparison),
        format_args!("{:.2} vs {:.2}", r.counter.ours, r.counter.comparison),
        r.reference.cpu,
        r.counter.cpu,
    )
}

/// Judges one simulated entry and level by its limit (`None`: not listed).
#[must_use]
pub fn judge_record(
    caps: &Caps,
    level: &str,
    limit: Option<&crate::count::Limit>,
    open_items: &str,
    r: &Record,
) -> Verdict {
    let runtime = level == RUNTIME_LEVEL;
    let release = crate::count::Limit::release();
    let limit = match (limit, runtime) {
        (Some(l), _) => l,
        (None, true) => &release,
        (None, false) => return Verdict::Fail("no entry in benches/limits.json".into()),
    };
    let cap = match (limit.status, runtime) {
        (Status::Info, _) => return Verdict::Info,
        (Status::Release, false) => caps.release,
        (Status::Release, true) => caps.runtime,
        (Status::Dev, _) => caps.dev,
    };
    let status = match limit.status {
        Status::Release => "release",
        _ => "dev",
    };
    let worst = r.reference.ratio.max(r.counter.ratio);
    let against = if runtime {
        "our avx2 build".to_string()
    } else {
        format!("`{}`", r.compared_with)
    };
    if worst > cap {
        let text = format!(
            "ours / {against} {} exceeds the {status} limit of {cap:.2}x on a model",
            ratios(r, cap)
        );
        return match &limit.waiver {
            None => Verdict::Fail(text),
            Some(w) => {
                let (_, missing) = missing_open_items(w, open_items);
                if missing.is_empty() {
                    Verdict::Waived(format!("{text} (waiver: {w})"))
                } else {
                    Verdict::Fail(format!(
                        "{text}; the waiver names open items that docs/open-items.md does not have: {}",
                        missing.join(", ")
                    ))
                }
            }
        };
    }
    let mut notes = Vec::new();
    if let Some(w) = &limit.waiver {
        notes.push(format!(
            "{} is within the {status} limit of {cap:.2}x: the waiver is no longer needed ({w})",
            ratios(r, cap)
        ));
    }
    if !runtime {
        if limit.status == Status::Dev && worst <= caps.warn {
            notes.push(format!(
                "{}: ready to be promoted from dev to release (release limit {:.2}x)",
                ratios(r, caps.warn),
                caps.release
            ));
        } else if worst > caps.warn {
            notes.push(format!(
                "{} is above {:.2}x ({status} limit {cap:.2}x)",
                ratios(r, caps.warn),
                caps.warn
            ));
        }
    }
    if notes.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Warn(notes.join("; "))
    }
}

/// The outcome of the release gate.
#[derive(Debug, Default)]
pub struct Gate {
    /// Exceedances without a waiver, waivers with missing open items, entries without a limit
    /// and listed entries without a simulated record.
    pub findings: Vec<Finding>,
    /// The waived exceedances: `entry level: text`.
    pub waived: Vec<String>,
    /// The remarks: `entry level: text`.
    pub warnings: Vec<String>,
    /// The number of entries and levels judged (`info` excluded).
    pub judged: usize,
    /// How many of them are at or below `_limits.goal` on both models.
    pub goal: usize,
}

/// Judges every simulated record of `results` by `benches/limits.json` (the release gate of
/// M13.1 §3). With `partial` (a selection of entries) a listed entry without a record is not a
/// finding.
#[must_use]
#[allow(
    clippy::large_stack_frames,
    reason = "the findings are built in place; xtask is a host tool"
)]
pub fn judge_release(results: &Results, limits: &Limits, open_items: &str, partial: bool) -> Gate {
    let caps = &limits.caps;
    let mut out = Gate::default();
    let path = PathBuf::from("benches/limits.json");
    for (level, rows) in results {
        for (entry, r) in rows {
            let Some(r) = r else { continue };
            let verdict = judge_record(caps, level, limits.get(entry, level), open_items, r);
            if verdict != Verdict::Info {
                out.judged += 1;
                if r.reference.ratio.max(r.counter.ratio) <= caps.goal && level != RUNTIME_LEVEL {
                    out.goal += 1;
                }
            }
            match verdict {
                Verdict::Pass | Verdict::Info => {}
                Verdict::Warn(t) => out.warnings.push(format!("`{entry}` at `{level}`: {t}")),
                Verdict::Waived(t) => out.waived.push(format!("`{entry}` at `{level}`: {t}")),
                Verdict::Fail(t) => out.findings.push(Finding {
                    path: r.file.clone(),
                    line: 0,
                    message: format!("`{entry}` at `{level}`: {t} (M13.1 §3, D14)"),
                }),
            }
        }
    }
    for (entry, level, limit) in limits.iter() {
        if limit.status == Status::Info || crate::mca::models(level).is_none() {
            continue;
        }
        // A level with a core model that the results lack as a whole (a lost results file or
        // partition) is as missing as an entry, unless the results are a selection.
        let Some(rows) = results.get(level) else {
            if !partial {
                out.findings.push(Finding {
                    path: path.clone(),
                    line: 0,
                    message: format!(
                        "`{entry}` at `{level}` has an entry but the cycle results have no level `{level}`; measure it (`cargo measure --all --level {level}`)"
                    ),
                });
            }
            continue;
        };
        match rows.get(entry) {
            Some(Some(_)) => {}
            Some(None) => out.findings.push(Finding {
                path: path.clone(),
                line: 0,
                message: format!(
                    "`{entry}` at `{level}` is `{}` but the cycle results have no simulation of it (its note says why)",
                    if limit.status == Status::Release {
                        "release"
                    } else {
                        "dev"
                    }
                ),
            }),
            None if partial => {}
            None => out.findings.push(Finding {
                path: path.clone(),
                line: 0,
                message: format!(
                    "`{entry}` at `{level}` has an entry but no record in the cycle results; measure it (`cargo measure {entry} --merge`) or remove the entry"
                ),
            }),
        }
    }
    out
}

/// The options of `check cycles`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// `--results DIR`: other results than the committed ones (the nightly measurement).
    pub results: Option<PathBuf>,
    /// `--partial`: the results hold a selection of entries.
    pub partial: bool,
}

impl Options {
    /// Parses `[--results DIR] [--partial]`.
    ///
    /// # Errors
    /// An unknown argument.
    pub fn parse(args: &[&str]) -> anyhow::Result<Options> {
        let mut o = Options::default();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match *a {
                "--results" => {
                    o.results = Some(PathBuf::from(
                        it.next()
                            .ok_or_else(|| anyhow!("`--results` needs a directory"))?,
                    ));
                }
                "--partial" => o.partial = true,
                other => bail!(
                    "unknown argument `{other}`; usage: cargo xtask check cycles [--results DIR] [--partial]"
                ),
            }
        }
        Ok(o)
    }
}

/// Runs both gates on the results of `o` (default: the committed results of the repository at
/// `root`).
///
/// # Errors
/// A file cannot be read or parsed.
pub fn check(root: &Path, o: &Options) -> anyhow::Result<Vec<Finding>> {
    let read =
        |p: &str| std::fs::read_to_string(root.join(p)).with_context(|| format!("cannot read {p}"));
    let dir = o
        .results
        .clone()
        .unwrap_or_else(|| crate::cycle_results::dir(root));
    let results = load(&dir)?;
    // Empty results (every partition failed) would pass every gate.
    if results
        .values()
        .flat_map(|rows| rows.values())
        .all(Option::is_none)
    {
        bail!(
            "{} holds no simulated record: nothing to judge",
            dir.display()
        );
    }
    let limits_json = read("benches/limits.json")?;
    let open_items = read("docs/open-items.md")?;
    let fallbacks = judge_fallbacks(&results, &limits_json, &open_items, o.partial)?;
    let limits = Limits::parse(&limits_json)?;
    let gate = judge_release(&results, &limits, &open_items, o.partial);
    println!(
        "check cycles ({}): {} SIMD entries and levels against their fallback, {} waived (D321)",
        dir.display(),
        fallbacks.judged,
        fallbacks.waived.len()
    );
    for (pair, text) in &fallbacks.waived {
        println!(
            "  waived: `{}` at `{}`: {} ({text})",
            pair.entry,
            pair.level,
            pair.describe()
        );
    }
    println!(
        "check cycles: {} entries and levels against their comparison: {} at the goal of {:.2}x, {} waived, {} with remarks, {} over the limit (D14 in cycles)",
        gate.judged,
        gate.goal,
        limits.caps.goal,
        gate.waived.len(),
        gate.warnings.len(),
        gate.findings.len()
    );
    for w in &gate.waived {
        println!("  waived: {w}");
    }
    for w in &gate.warnings {
        println!("  warn: {w}");
    }
    let mut findings = fallbacks.findings;
    findings.extend(gate.findings);
    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: &str = "| M30 | I30-9: a gap | why | what |\n";

    /// A results document of `sse2` (and `avx2-runtime` with `runtime`) with the entries
    /// `(name, ours ref, other ref, ours counter, other counter, comparison)`.
    fn results(rows: &[(&str, f64, f64, f64, f64, &str)], level: &str) -> Results {
        let entries: serde_json::Map<String, Value> = rows
            .iter()
            .map(|(name, or, cr, oc, cc, comp)| {
                let m = |cpu: &str, o: f64, c: f64| {
                    serde_json::json!({
                        "cpu": cpu,
                        "ours": { "cycles_per_sample": o },
                        "comparison": { "cycles_per_sample": c },
                        "ratio": o / c,
                    })
                };
                (
                    (*name).to_string(),
                    serde_json::json!({
                        "compared_with": comp,
                        "reference_model": m("skylake", *or, *cr),
                        "counter_model": m("znver2", *oc, *cc),
                    }),
                )
            })
            .collect();
        let doc = serde_json::json!({ "levels": { level: { "entries": entries } } });
        let mut out = Results::new();
        add_results(Path::new("x86_64.json"), &doc.to_string(), &mut out).unwrap();
        out
    }

    fn limits_with(entry: &str) -> String {
        format!(
            r#"{{"_limits": {{"release": 1.1, "dev": 1.5, "warn": 1.1, "goal": 1.0, "runtime": 1.02}}, {entry}}}"#
        )
    }

    fn gate(r: &Results, limits: &str) -> Gate {
        judge_release(r, &Limits::parse(limits).unwrap(), OPEN, false)
    }

    #[test]
    fn records_are_read_per_level_and_unmeasured_entries_are_left_out() {
        let doc = r#"{"levels": {"sse2": {"entries": {
            "k": {"compared_with": "volk", "reference_model": {"cpu": "skylake", "ours": {"cycles_per_sample": 0.52}, "comparison": {"cycles_per_sample": 0.7}, "ratio": 0.74},
                  "counter_model": {"cpu": "znver2", "ours": {"cycles_per_sample": 0.56}, "comparison": {"cycles_per_sample": 0.79}, "ratio": 0.71}},
            "j": {"note": "not simulated: not measured at the level sse2"},
            "f": {"note": "not simulated: the run failed"}}}}}"#;
        let mut r = Results::new();
        add_results(Path::new("x.json"), doc, &mut r).unwrap();
        let rows = &r["sse2"];
        assert_eq!(rows.len(), 2);
        let k = rows["k"].as_ref().unwrap();
        assert_eq!((k.reference.ours, k.counter.ratio), (0.52, 0.71));
        assert_eq!(k.reference.cpu, "skylake");
        assert!(rows["f"].is_none());
        // A level in two files, a record with one model, no levels.
        assert!(add_results(Path::new("y.json"), doc, &mut r).is_err());
        let one = r#"{"levels": {"avx2": {"entries": {"k": {"reference_model": {}}}}}}"#;
        assert!(add_results(Path::new("z.json"), one, &mut Results::new()).is_err());
        assert!(add_results(Path::new("z.json"), "{}", &mut Results::new()).is_err());
    }

    #[test]
    fn slower_than_the_fallback_on_either_model_is_a_violation_and_equal_passes() {
        let fb = ("k_fallback", 0.51, 0.70, 0.58, 0.91, "CMSIS-DSP");
        let judge = |simd: (&str, f64, f64, f64, f64, &str)| {
            judge_fallbacks(&results(&[simd, fb], "sse2"), "{}", OPEN, false).unwrap()
        };
        let o = judge(("k", 0.50, 0.70, 0.57, 0.79, "volk"));
        assert_eq!((o.judged, o.findings.len()), (1, 0));
        // Equal in the two decimals of the tables.
        let o = judge(("k", 0.5149, 0.70, 0.58, 0.79, "volk"));
        assert_eq!(o.findings.len(), 0);
        let o = judge(("k", 0.52, 0.70, 0.56, 0.79, "volk"));
        assert_eq!(o.findings.len(), 1);
        assert!(o.findings[0].message.contains("0.52 / 0.51 (1.02)"));
        let o = judge(("k", 0.50, 0.70, 0.59, 0.79, "volk"));
        assert_eq!(o.findings.len(), 1);
        // No fallback record: nothing to judge.
        let o = judge_fallbacks(
            &results(&[("k", 9.0, 0.7, 9.0, 0.79, "volk")], "sse2"),
            "{}",
            OPEN,
            false,
        )
        .unwrap();
        assert_eq!((o.judged, o.findings.len()), (0, 0));
    }

    #[test]
    fn a_level_delegated_to_the_fallback_passes() {
        let fb = ("k_fallback", 0.97, 1.21, 1.02, 1.39, "CMSIS-DSP");
        let o = judge_fallbacks(
            &results(
                &[("k", 0.98, 0.98, 1.03, 1.03, "sdrk fallback"), fb],
                "sse2",
            ),
            "{}",
            OPEN,
            false,
        )
        .unwrap();
        assert_eq!(o.findings.len(), 0);
        let o = judge_fallbacks(
            &results(
                &[("k", 0.98, 0.97, 1.03, 1.02, "sdrk fallback"), fb],
                "sse2",
            ),
            "{}",
            OPEN,
            false,
        )
        .unwrap();
        assert_eq!(o.findings.len(), 1);
    }

    #[test]
    fn a_cycle_waiver_must_name_an_existing_open_item_and_must_not_be_stale() {
        let fb = ("k_fallback", 0.51, 0.70, 0.58, 0.91, "CMSIS-DSP");
        let slow = results(&[("k", 0.52, 0.70, 0.56, 0.79, "volk"), fb], "sse2");
        let w = |t: &str| {
            format!(r#"{{"k": {{"sse2": {{"status": "release", "cycle_waiver": "{t}"}}}}}}"#)
        };
        let o = judge_fallbacks(&slow, &w("store-bound, open item I30-9"), OPEN, false).unwrap();
        assert_eq!((o.findings.len(), o.waived.len()), (0, 1));
        let o = judge_fallbacks(&slow, &w("store-bound, open item I30-8"), OPEN, false).unwrap();
        assert!(o.findings[0].message.contains("I30-8"));
        let o = judge_fallbacks(&slow, &w("store-bound"), OPEN, false).unwrap();
        assert_eq!(o.findings.len(), 1);
        let fast = results(&[("k", 0.50, 0.70, 0.56, 0.79, "volk"), fb], "sse2");
        let o = judge_fallbacks(&fast, &w("open item I30-9"), OPEN, false).unwrap();
        assert!(o.findings[0].message.contains("remove the waiver"));
        let none = results(&[("j", 0.50, 0.70, 0.56, 0.79, "volk")], "sse2");
        let o = judge_fallbacks(&none, &w("open item I30-9"), OPEN, false).unwrap();
        assert_eq!(o.findings.len(), 1);
        // A selection judges only the pairs it has.
        let o = judge_fallbacks(&none, &w("open item I30-9"), OPEN, true).unwrap();
        assert_eq!(o.findings.len(), 0);
        assert!(cycle_waivers(r#"{"k": {"sse2": {"cycle_waiver": 1}}}"#).is_err());
    }

    #[test]
    fn release_needs_both_models_within_the_limit() {
        let l = limits_with(r#""k": {"sse2": {"status": "release"}}"#);
        // 1.10 on both: passes (at the limit), with a remark only above warn.
        let g = gate(&results(&[("k", 1.10, 1.0, 1.0, 1.0, "volk")], "sse2"), &l);
        assert_eq!((g.judged, g.findings.len(), g.goal), (1, 0, 0));
        assert_eq!(g.warnings, Vec::<String>::new());
        // Over the cap on the reference model only, then on the counter-model only.
        let g = gate(&results(&[("k", 1.2, 1.0, 0.9, 1.0, "volk")], "sse2"), &l);
        assert_eq!(g.findings.len(), 1);
        assert!(g.findings[0].message.contains("1.20 / 0.90"));
        assert!(g.findings[0].message.contains("release limit of 1.10x"));
        let g = gate(
            &results(&[("k", 0.9, 1.0, 1.1001, 1.0, "volk")], "sse2"),
            &l,
        );
        assert_eq!(g.findings.len(), 1);
        assert!(g.findings[0].message.contains("1.1001"));
        // At the goal on both models.
        let g = gate(&results(&[("k", 0.9, 1.0, 1.0, 1.0, "volk")], "sse2"), &l);
        assert_eq!((g.findings.len(), g.goal), (0, 1));
    }

    #[test]
    fn dev_is_judged_with_the_dev_limit_and_reported_when_ready_for_release() {
        let l = limits_with(r#""k": {"sse2": {"status": "dev"}}"#);
        let g = gate(&results(&[("k", 1.4, 1.0, 1.3, 1.0, "volk")], "sse2"), &l);
        assert_eq!(g.findings.len(), 0);
        assert!(g.warnings[0].contains("above 1.10x (dev limit 1.50x)"));
        let g = gate(&results(&[("k", 1.6, 1.0, 1.3, 1.0, "volk")], "sse2"), &l);
        assert_eq!(g.findings.len(), 1);
        let g = gate(&results(&[("k", 1.05, 1.0, 1.0, 1.0, "volk")], "sse2"), &l);
        assert!(g.warnings[0].contains("ready to be promoted"));
    }

    #[test]
    fn info_is_not_judged_and_a_missing_entry_fails() {
        let l = limits_with(r#""k": {"sse2": {"status": "info", "waiver": "no fair reference"}}"#);
        let g = gate(&results(&[("k", 9.0, 1.0, 9.0, 1.0, "volk")], "sse2"), &l);
        assert_eq!((g.judged, g.findings.len()), (0, 0));
        let g = gate(&results(&[("j", 1.0, 1.0, 1.0, 1.0, "volk")], "sse2"), &l);
        assert_eq!(g.findings.len(), 1);
        assert!(
            g.findings[0]
                .message
                .contains("no entry in benches/limits.json")
        );
    }

    #[test]
    fn a_waiver_covers_an_exceedance_only_with_existing_open_items() {
        let slow = results(&[("k", 1.3, 1.0, 1.0, 1.0, "volk")], "sse2");
        let w = |t: &str| {
            limits_with(&format!(
                r#""k": {{"sse2": {{"status": "release", "waiver": "{t}"}}}}"#
            ))
        };
        let g = gate(&slow, &w("slow loads, open item I30-9"));
        assert_eq!((g.findings.len(), g.waived.len()), (0, 1));
        let g = gate(&slow, &w("slow loads, open item I30-8"));
        assert_eq!(g.findings.len(), 1);
        assert!(g.findings[0].message.contains("I30-8"));
        // A waiver that states its reason without an open item, as release-check accepts.
        let g = gate(&slow, &w("the reference computes in f64"));
        assert_eq!((g.findings.len(), g.waived.len()), (0, 1));
        // A waiver of an entry within its limit is reported, not failed.
        let fast = results(&[("k", 1.0, 1.0, 1.0, 1.0, "volk")], "sse2");
        let g = gate(&fast, &w("open item I30-9"));
        assert_eq!(g.findings.len(), 0);
        assert!(g.warnings[0].contains("no longer needed"));
    }

    #[test]
    fn the_run_time_level_is_judged_against_our_avx2_build_with_the_runtime_limit() {
        let none = limits_with(r#""j": {"sse2": {"status": "info"}}"#);
        let r = results(&[("k", 1.03, 1.0, 1.0, 1.0, "ours at avx2")], RUNTIME_LEVEL);
        let g = gate(&r, &none);
        assert_eq!(g.findings.len(), 1);
        assert!(g.findings[0].message.contains("our avx2 build"));
        assert!(g.findings[0].message.contains("1.02x"));
        let r = results(
            &[("k", 1.02, 1.0, 1.01, 1.0, "ours at avx2")],
            RUNTIME_LEVEL,
        );
        assert_eq!(gate(&r, &none).findings.len(), 0);
        // An entry of its own relaxes it.
        let dev = limits_with(r#""k": {"avx2-runtime": {"status": "dev"}}"#);
        let r = results(&[("k", 1.4, 1.0, 1.0, 1.0, "ours at avx2")], RUNTIME_LEVEL);
        let g = gate(&r, &dev);
        assert_eq!((g.findings.len(), g.warnings.len()), (0, 0));
    }

    #[test]
    fn a_listed_entry_needs_a_simulated_record_unless_the_results_are_a_selection() {
        let l = limits_with(
            r#""k": {"sse2": {"status": "release"}}, "m": {"sse2": {"status": "info"}}"#,
        );
        let r = results(&[("j", 1.0, 1.0, 1.0, 1.0, "volk")], "sse2");
        let limits = Limits::parse(&l).unwrap();
        let g = judge_release(&r, &limits, OPEN, false);
        // `j` has no entry, `k` has no record; `m` is `info`.
        assert_eq!(g.findings.len(), 2);
        assert!(g.findings.iter().any(|f| {
            f.message
                .contains("`k` at `sse2` has an entry but no record")
        }));
        let g = judge_release(&r, &limits, OPEN, true);
        assert_eq!(g.findings.len(), 1);
        // A record that failed is a finding even in a selection.
        let mut failed = r.clone();
        failed.get_mut("sse2").unwrap().insert("k".into(), None);
        let g = judge_release(&failed, &limits, OPEN, true);
        assert!(
            g.findings
                .iter()
                .any(|f| f.message.contains("no simulation"))
        );
        // A level with a core model that the results lack as a whole (a lost partition) is a
        // finding, except in a selection; a level without a core model is not judged here.
        let l = limits_with(r#""k": {"neon": {"status": "release"}}"#);
        let g = gate(&results(&[], "sse2"), &l);
        assert_eq!(g.findings.len(), 1);
        assert!(
            g.findings[0].message.contains("no level `neon`"),
            "{:?}",
            g.findings
        );
        let limits = Limits::parse(&l).unwrap();
        assert_eq!(
            judge_release(&results(&[], "sse2"), &limits, OPEN, true)
                .findings
                .len(),
            0
        );
        let l = limits_with(r#""k": {"simd128": {"status": "release"}}"#);
        assert_eq!(gate(&results(&[], "sse2"), &l).findings.len(), 0);
    }

    #[test]
    fn options_take_a_results_directory_and_partial() {
        assert_eq!(Options::parse(&[]).unwrap(), Options::default());
        let o = Options::parse(&["--results", "out", "--partial"]).unwrap();
        assert_eq!(o.results, Some(PathBuf::from("out")));
        assert!(o.partial);
        assert!(Options::parse(&["--results"]).is_err());
        assert!(Options::parse(&["--x"]).is_err());
    }
}
