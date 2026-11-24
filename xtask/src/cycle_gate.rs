//! `cargo xtask check cycles`: the speed gates in simulated cycles per sample, judged on the
//! committed cycle results `benches/results/cycles/<platform>.json` (D2704), so CI enforces them
//! without running `llvm-mca`.
//!
//! **SIMD against its own fallback (D321).** A check, never a status: a violation is a
//! finding of its own and no entry of `benches/limits.json` is derived from it (D2764).
//! For every level and every entry `<k>` whose
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
//! `_limits.dev`; `info` is not judged. The level `avx2-runtime` is judged like every level,
//! against the reference library (D2764, the maintainer's decision): its record holds our cycles
//! of the run-time-detection build and, as information, those of our fixed `avx2` build; the
//! library's cycles and SNR come from the `avx2` record of the entry (the same workload on the
//! same core models), from the committed results when judged results lack it. An entry whose
//! `avx2` comparison is our own fallback (no library) must be `info` there, as at `avx2`.
//! An exceedance passes only with a `waiver` whose open items exist (a waiver that names none
//! states the reason in its text, as `release-check` accepts); a waiver of an entry within its
//! limit, a `dev` entry within `_limits.warn` (ready for `release`) and a ratio above
//! `_limits.warn` are reported, not failed; `_limits.goal` is reported per entry. An entry
//! without a record of `limits.json` on a level that is simulated fails, and so does a
//! `release` or `dev` entry without a simulated record (except for a selection, `--partial`).
//!
//! **The accuracy (D34, D2749).** Every judged record carries the SNR of ours and of the
//! comparison against the f64 reference of the measurement binary, on the workload of the
//! cycles (`snr_db`, `cargo measure`). Ours must be at least the comparison's minus
//! [`SNR_MARGIN_DB`]; below that it passes only with an `accuracy_waiver` whose open items exist,
//! as the count judged it before M13.1. A judged record without `snr_db` fails.
//!
//! **The accuracy trade (D2760, the maintainer's decision of 2026-11-24).** A `release` entry
//! above the release limit in cycles whose SNR is at least [`TRADE_DB`] above the comparison's
//! stays `release` with an `accuracy_trade` text that names both SNR figures: it passes while
//! the advantage holds, the comparison's SNR is below [`TRADE_SCARCE_DB`] (where accuracy is
//! scarce, typically fixed point) and its ratio stays at most [`TRADE_CAP`] on both models,
//! and fails as soon as one of them does not ("3 dB better must not mean that we may be
//! slow"). A trade on an entry within its limit is a remark.
//!
//! The instruction judgement of `simd128` stays in `cargo xtask count`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow, bail};
use serde_json::Value;

use crate::bench::{ComparisonKind, KERNELS, PLANS, comparisons};
use crate::count::{Caps, Limits, Status};
use crate::scan::Finding;

/// The comparison of a SIMD entry that runs our fallback on its level.
const OWN_FALLBACK: &str = "sdrk fallback";
/// The suffix of the fallback entry of a kernel.
const FALLBACK: &str = "_fallback";
/// The level with run-time detection: its record compares with our own build of
/// [`RUNTIME_LIBRARY_LEVEL`] (information), and the gate judges it against the library of that
/// level's record (D2702, D2764).
const RUNTIME_LEVEL: &str = "avx2-runtime";
/// The level whose record holds the library's cycles and SNR for [`RUNTIME_LEVEL`].
const RUNTIME_LIBRARY_LEVEL: &str = "avx2";
/// The note of an entry that the level does not measure.
const NOT_MEASURED: &str = "not simulated: not measured at";

/// How far the SNR of ours may lie below the comparison's, in dB (D34).
pub use crate::count::SNR_MARGIN_DB;
/// The SNR advantage over the comparison, in dB, that lets a `release` entry above the release
/// limit stay `release` with an `accuracy_trade` (D2760, the maintainer's decision).
pub const TRADE_DB: f64 = 3.0;
/// The highest ratio to the comparison, on each model, that an accuracy trade covers (D2760,
/// the maintainer's decision: "3 dB better must not mean that we may be slow").
pub const TRADE_CAP: f64 = 1.20;
/// The comparison's SNR, in dB, below which an accuracy trade is possible (D2760, the
/// maintainer's decision): above it, as for most `f32` kernels at 137 to 150 dB, a few dB more
/// accuracy have no practical effect and do not justify more cycles.
pub const TRADE_SCARCE_DB: f64 = 100.0;
/// The SNR the measurement binary reports for an exact result.
pub const EXACT_DB: f64 = 999.0;

/// An SNR in dB for the messages and tables: `exact` for [`EXACT_DB`], else one decimal.
#[must_use]
pub fn snr_text(db: f64) -> String {
    if db >= EXACT_DB {
        "exact".to_string()
    } else {
        format!("{db:.1}")
    }
}

/// The accuracy of a record: the SNR in dB of ours and of the comparison against the f64
/// reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snr {
    /// Ours.
    pub ours: f64,
    /// The comparison.
    pub comparison: f64,
}

impl Snr {
    /// How far ours lies above the comparison, in dB (negative: below).
    #[must_use]
    pub fn advantage(self) -> f64 {
        self.ours - self.comparison
    }

    /// `ours X dB against Y dB of the comparison`, for the messages.
    #[must_use]
    pub fn describe(self) -> String {
        let db = |x: f64| {
            if x >= EXACT_DB {
                "exact".to_string()
            } else {
                format!("{x:.1} dB")
            }
        };
        format!(
            "SNR ours {} against {} of the comparison",
            db(self.ours),
            db(self.comparison)
        )
    }
}

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
    /// What the comparison is (`comparison_kind`, D2767; `None`: the record does not say).
    pub comparison_kind: Option<ComparisonKind>,
    /// At `avx2-runtime`: the comparison that the gate judges the record against, the library
    /// of the `avx2` record, and its kind (`judged_against`, D2764, D2767).
    pub judged_against: Option<(String, Option<ComparisonKind>)>,
    /// The reference model.
    pub reference: Model,
    /// The counter-model.
    pub counter: Model,
    /// The accuracy of both sides (`None`: the record has none).
    pub snr: Option<Snr>,
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
#[allow(
    clippy::large_stack_frames,
    reason = "the Windows file system calls and the records are large; xtask is a host tool"
)]
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
            let kind = |v: &Value| -> anyhow::Result<Option<ComparisonKind>> {
                match v.get("comparison_kind") {
                    None => Ok(None),
                    Some(k) => k
                        .as_str()
                        .and_then(ComparisonKind::parse)
                        .map(Some)
                        .ok_or_else(|| {
                            anyhow!(
                                "{}: `{entry}` at `{level}` has an unknown `comparison_kind` {k}",
                                file.display()
                            )
                        }),
                }
            };
            let text = |v: &Value| {
                v.get("compared_with")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string()
            };
            let judged_against = match r.get("judged_against") {
                Some(j) => Some((text(j), kind(j)?)),
                None => None,
            };
            records.insert(
                entry.clone(),
                Some(Record {
                    file: file.to_path_buf(),
                    compared_with: text(r),
                    comparison_kind: kind(r)?,
                    judged_against,
                    reference,
                    counter,
                    snr: r.get("snr_db").and_then(|v| {
                        Some(Snr {
                            ours: v.get("ours")?.as_f64()?,
                            comparison: v.get("comparison")?.as_f64()?,
                        })
                    }),
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
    /// A `release` entry above the release limit, at most [`TRADE_CAP`] of its comparison on
    /// both models and at least [`TRADE_DB`] more accurate, with an `accuracy_trade` (D2760).
    Traded(String),
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

/// Whether a waiver text is valid: every open item it names exists (a text that names none
/// states its reason, as `release-check` accepts); `Err` names the missing items.
fn valid_waiver(text: &str, open_items: &str) -> Result<(), String> {
    let (_, missing) = missing_open_items(text, open_items);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing.join(", "))
    }
}

/// The accuracy part of a judgement (D34): ours at most [`SNR_MARGIN_DB`] below the
/// comparison, else an `accuracy_waiver` with existing open items.
fn judge_accuracy(
    limit: &crate::count::Limit,
    open_items: &str,
    r: &Record,
    failing: &mut Vec<String>,
    waived: &mut Vec<String>,
    notes: &mut Vec<String>,
) {
    let Some(snr) = r.snr else {
        failing.push(
            "the record has no accuracy (`snr_db`); measure it again with `cargo measure` (D34)"
                .into(),
        );
        return;
    };
    if -snr.advantage() > SNR_MARGIN_DB {
        let text = format!(
            "the accuracy is more than {SNR_MARGIN_DB:.0} dB below the comparison: {} (D34)",
            snr.describe()
        );
        match &limit.accuracy_waiver {
            None => failing.push(text),
            Some(w) => match valid_waiver(w, open_items) {
                Ok(()) => waived.push(format!("{text} (accuracy waiver: {w})")),
                Err(missing) => failing.push(format!(
                    "{text}; the accuracy waiver names open items that docs/open-items.md does not have: {missing}"
                )),
            },
        }
    } else if let Some(w) = &limit.accuracy_waiver {
        notes.push(format!(
            "the accuracy is within {SNR_MARGIN_DB:.0} dB of the comparison ({}): the accuracy waiver is no longer needed ({w})",
            snr.describe()
        ));
    }
}

/// Why the judged comparison of `r` is not a foreign library (D32, D2764, D2767): it is our own
/// code, or the record does not say what it is. `None` for a library.
fn own_comparison(r: &Record) -> Option<String> {
    match r.comparison_kind {
        Some(ComparisonKind::Library) => None,
        Some(kind) => Some(format!(
            "the comparison `{}` is `{}`",
            r.compared_with,
            kind.name()
        )),
        None => Some(format!(
            "the record does not say what its comparison `{}` is (no `comparison_kind`)",
            r.compared_with
        )),
    }
}

/// Judges one simulated entry and level by its limit (`None`: not listed): the cycles against
/// the cap of its status, the accuracy (D34) and an accuracy trade (D2760). At `avx2-runtime`
/// `r` is the record against the library ([`against_library`]).
#[must_use]
#[allow(
    clippy::too_many_lines,
    clippy::large_stack_frames,
    reason = "one judgement with its three parts, in the order of the messages; the texts are built in place"
)]
pub fn judge_record(
    caps: &Caps,
    limit: Option<&crate::count::Limit>,
    open_items: &str,
    r: &Record,
) -> Verdict {
    let Some(limit) = limit else {
        return Verdict::Fail("no entry in benches/limits.json".into());
    };
    if let Some(t) = &limit.accuracy_trade {
        if limit.status != Status::Release {
            return Verdict::Fail(format!(
                "an `accuracy_trade` is only for a `release` entry against a comparison, not for a `{}` entry (accuracy trade: {t})",
                status_name(limit.status)
            ));
        }
        if limit.waiver.is_some() {
            return Verdict::Fail(format!(
                "an entry has a `waiver` or an `accuracy_trade`, not both (accuracy trade: {t})"
            ));
        }
    }
    let cap = match limit.status {
        Status::Info => return Verdict::Info,
        Status::Release => caps.release,
        Status::Dev => caps.dev,
    };
    let status = status_name(limit.status);
    if let Some(why) = own_comparison(r) {
        return Verdict::Fail(format!(
            "{status} judged against our own code (D32, D2764): {why}; the entry is `info` (judged by its efficiency) until a foreign library is its comparison"
        ));
    }
    let worst = r.reference.ratio.max(r.counter.ratio);
    let against = format!("`{}`", r.compared_with);
    let (mut failing, mut waived, mut traded, mut notes) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    if worst > cap {
        let text = format!(
            "ours / {against} {} exceeds the {status} limit of {cap:.2}x on a model",
            ratios(r, cap)
        );
        match (&limit.accuracy_trade, &limit.waiver) {
            (Some(t), _) => {
                let advantage = r.snr.map(Snr::advantage);
                if worst > TRADE_CAP {
                    failing.push(format!(
                        "{text}; the accuracy trade holds only up to {TRADE_CAP:.2}x on both models: {} (D2760; accuracy trade: {t})",
                        ratios(r, TRADE_CAP)
                    ));
                } else if r.snr.is_some_and(|s| s.comparison >= TRADE_SCARCE_DB) {
                    failing.push(format!(
                        "{text}; the accuracy trade needs a comparison below {TRADE_SCARCE_DB:.0} dB, where accuracy is scarce: {} (D2760; accuracy trade: {t})",
                        r.snr.map(Snr::describe).unwrap_or_default()
                    ));
                } else if advantage.is_none_or(|a| a < TRADE_DB) {
                    failing.push(format!(
                        "{text}; the accuracy trade needs an SNR at least {TRADE_DB:.0} dB above the comparison: {} (D2760; accuracy trade: {t})",
                        r.snr.map_or_else(|| "no accuracy in the record".to_string(), Snr::describe)
                    ));
                } else {
                    traded.push(format!(
                        "{text}, traded for the accuracy: {} (D2760; accuracy trade: {t})",
                        r.snr.map(Snr::describe).unwrap_or_default()
                    ));
                }
            }
            (None, None) => failing.push(text),
            (None, Some(w)) => match (
                crate::count::waiver_above_cap(caps, limit.status, worst, w),
                valid_waiver(w, open_items),
            ) {
                (Err(why), _) => failing.push(format!("{text}; {why} (waiver: {w})")),
                (Ok(()), Ok(())) => waived.push(format!("{text} (waiver: {w})")),
                (Ok(()), Err(missing)) => failing.push(format!(
                    "{text}; the waiver names open items that docs/open-items.md does not have: {missing}"
                )),
            },
        }
    } else {
        if let Some(w) = &limit.waiver {
            notes.push(format!(
                "{} is within the {status} limit of {cap:.2}x: the waiver is no longer needed ({w})",
                ratios(r, cap)
            ));
        }
        if let Some(t) = &limit.accuracy_trade {
            notes.push(format!(
                "{} is within the {status} limit of {cap:.2}x: the accuracy trade is no longer needed ({t})",
                ratios(r, cap)
            ));
        }
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
    judge_accuracy(limit, open_items, r, &mut failing, &mut waived, &mut notes);
    if !failing.is_empty() {
        failing.extend(traded);
        failing.extend(waived);
        return Verdict::Fail(failing.join("; "));
    }
    if !waived.is_empty() {
        waived.extend(traded);
        return Verdict::Waived(waived.join("; "));
    }
    if !traded.is_empty() {
        return Verdict::Traded(traded.join("; "));
    }
    if notes.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Warn(notes.join("; "))
    }
}

/// `release`, `dev` or `info`.
fn status_name(status: Status) -> &'static str {
    match status {
        Status::Release => "release",
        Status::Dev => "dev",
        Status::Info => "info",
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
    /// The accuracy trades that hold (D2760): `entry level: text`.
    pub traded: Vec<String>,
    /// The remarks: `entry level: text`.
    pub warnings: Vec<String>,
    /// The number of entries and levels judged (`info` excluded).
    pub judged: usize,
    /// How many of them are at or below `_limits.goal` on both models.
    pub goal: usize,
}

/// The record of `avx2-runtime` against the reference library (D2764): our cycles and SNR of the
/// run-time-detection build `rt`, the library's of the `avx2` record `avx2` of the same entry
/// (the same workload on the same core models). The comparison of `rt` with our fixed `avx2`
/// build is information and is not used.
///
/// # Errors
/// The two records are simulated on other core models.
pub fn against_library(rt: &Record, avx2: &Record) -> Result<Record, String> {
    let model = |ours: &Model, lib: &Model| {
        if ours.cpu == lib.cpu {
            Ok(Model {
                cpu: ours.cpu.clone(),
                ours: ours.ours,
                comparison: lib.comparison,
                ratio: ours.ours / lib.comparison,
            })
        } else {
            Err(format!(
                "the `{RUNTIME_LIBRARY_LEVEL}` record with the library's cycles is simulated on `{}`, the `{RUNTIME_LEVEL}` record on `{}`",
                lib.cpu, ours.cpu
            ))
        }
    };
    Ok(Record {
        file: rt.file.clone(),
        compared_with: avx2.compared_with.clone(),
        comparison_kind: avx2.comparison_kind,
        judged_against: None,
        reference: model(&rt.reference, &avx2.reference)?,
        counter: model(&rt.counter, &avx2.counter)?,
        snr: rt.snr.zip(avx2.snr).map(|(ours, lib)| Snr {
            ours: ours.ours,
            comparison: lib.comparison,
        }),
    })
}

/// The record that the release gate judges for `entry` at `level`: `r` itself, or at
/// `avx2-runtime` the record against the library of the `avx2` record (from `results`, else
/// from `committed`), D2764. `Err` says why there is none; `None` for an `info` entry without
/// one (it is not judged). A judged `avx2-runtime` entry needs a library at `avx2` and a judged
/// status there (`info` at `avx2` is `info` at `avx2-runtime`, D2765), and its record must name
/// that library in `judged_against` (D2767).
fn judged_record(
    results: &Results,
    committed: Option<&Results>,
    limits: &Limits,
    entry: &str,
    level: &str,
    r: &Record,
) -> Result<Option<Record>, String> {
    let limit = limits.get(entry, level);
    // Not listed: `judge_record` reports the missing entry.
    let (true, Some(limit)) = (level == RUNTIME_LEVEL, limit) else {
        return Ok(Some(r.clone()));
    };
    let judged = limit.status != Status::Info;
    let avx2 = [Some(results), committed]
        .into_iter()
        .flatten()
        .find_map(|res| res.get(RUNTIME_LIBRARY_LEVEL)?.get(entry)?.as_ref());
    let Some(avx2) = avx2 else {
        return if judged {
            Err(format!(
                "no simulated `{RUNTIME_LIBRARY_LEVEL}` record of the entry, whose library comparison judges `{RUNTIME_LEVEL}` (D2764)"
            ))
        } else {
            Ok(None)
        };
    };
    if judged {
        if let Some(why) = own_comparison(avx2) {
            return Err(format!(
                "{} judged against our own code (D32, D2764): at `{RUNTIME_LIBRARY_LEVEL}` {why}, not a library: the entry is `info` at `{RUNTIME_LEVEL}` as at `{RUNTIME_LIBRARY_LEVEL}`",
                status_name(limit.status)
            ));
        }
        let base = limits.get(entry, RUNTIME_LIBRARY_LEVEL).map(|l| l.status);
        if !matches!(base, Some(Status::Release | Status::Dev)) {
            return Err(format!(
                "{} at `{RUNTIME_LEVEL}` but {} at `{RUNTIME_LIBRARY_LEVEL}`, whose comparison judges it: an entry that is not judged at `{RUNTIME_LIBRARY_LEVEL}` is `info` at `{RUNTIME_LEVEL}` too (D2765)",
                status_name(limit.status),
                base.map_or("not listed", |s| status_name(s))
            ));
        }
        match &r.judged_against {
            Some((name, Some(ComparisonKind::Library))) if *name == avx2.compared_with => {}
            Some((name, kind)) => {
                return Err(format!(
                    "the record names `{name}` ({}) as the comparison it is judged against, the `{RUNTIME_LIBRARY_LEVEL}` record `{}`: measure it again (D2767)",
                    kind.map_or("no `comparison_kind`", ComparisonKind::name),
                    avx2.compared_with
                ));
            }
            None => {
                return Err(format!(
                    "the record does not name the library it is judged against (`judged_against`, the `{RUNTIME_LIBRARY_LEVEL}` record `{}`), so it reads as a comparison with our own build (D2767)",
                    avx2.compared_with
                ));
            }
        }
    }
    against_library(r, avx2).map(Some)
}

/// Judges every simulated record of `results` by `benches/limits.json` (the release gate of
/// M13.1 §3). With `partial` (a selection of entries) a listed entry without a record is not a
/// finding. `committed` gives the `avx2` records with the library's cycles for the records of
/// `avx2-runtime` when `results` lacks them (D2764).
#[must_use]
#[allow(
    clippy::large_stack_frames,
    reason = "the findings are built in place; xtask is a host tool"
)]
pub fn judge_release(
    results: &Results,
    committed: Option<&Results>,
    limits: &Limits,
    open_items: &str,
    partial: bool,
) -> Gate {
    let caps = &limits.caps;
    let mut out = Gate::default();
    let path = PathBuf::from("benches/limits.json");
    for (level, rows) in results {
        for (entry, r) in rows {
            let Some(r) = r else { continue };
            let limit = limits.get(entry, level);
            let (verdict, judged) = match judged_record(results, committed, limits, entry, level, r)
            {
                Ok(Some(j)) => (judge_record(caps, limit, open_items, &j), Some(j)),
                Ok(None) => (Verdict::Info, None),
                Err(t) => (Verdict::Fail(t), None),
            };
            if verdict != Verdict::Info {
                out.judged += 1;
                // The goal is a figure against a library only (D2767).
                if judged.is_some_and(|j| {
                    own_comparison(&j).is_none()
                        && j.reference.ratio.max(j.counter.ratio) <= caps.goal
                }) {
                    out.goal += 1;
                }
            }
            match verdict {
                Verdict::Pass | Verdict::Info => {}
                Verdict::Warn(t) => out.warnings.push(format!("`{entry}` at `{level}`: {t}")),
                Verdict::Waived(t) => out.waived.push(format!("`{entry}` at `{level}`: {t}")),
                Verdict::Traded(t) => out.traded.push(format!("`{entry}` at `{level}`: {t}")),
                Verdict::Fail(t) => out.findings.push(Finding {
                    path: r.file.clone(),
                    line: 0,
                    message: format!("`{entry}` at `{level}`: {t} (M13.1 §3, D14, D34)"),
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

/// The static check of the measurement plan against `benches/limits.json`, independent of any
/// results (D2767): for every level of [`PLANS`] and every entry of [`KERNELS`] measured there,
/// a `release` or `dev` entry whose judged comparison ([`comparisons`]) is our own code (our
/// fallback, our build of another level, our kernel) is a finding. A level with a baseline
/// (`avx2-runtime`) is judged against the library of the baseline level's record (D2764): there
/// the baseline level's judged comparison must be a library and the entry must be judged at
/// the baseline level as well (an entry that is `info` at `avx2` is `info` at `avx2-runtime`,
/// D2765).
#[must_use]
#[allow(
    clippy::large_stack_frames,
    reason = "the plan table and the findings are large; xtask is a host tool"
)]
pub fn check_comparisons(limits: &Limits) -> Vec<Finding> {
    let path = PathBuf::from("benches/limits.json");
    let mut out = Vec::new();
    for plan in &PLANS {
        for spec in KERNELS.iter().filter(|s| s.family.measured_at(plan)) {
            let Some(limit) = limits.get(spec.name, plan.level) else {
                continue;
            };
            if limit.status == Status::Info {
                continue;
            }
            let status = status_name(limit.status);
            let mut fail = |why: String| {
                out.push(Finding {
                    path: path.clone(),
                    line: 0,
                    message: format!(
                        "`{}` at `{}` is `{status}` but {why}: a comparison of our code with our own code is information only and decides no status (D32, D2764, D2767); make the entry `info`",
                        spec.name, plan.level
                    ),
                });
            };
            let Some(base) = plan.baseline else {
                let (gate, _) = comparisons(plan, spec);
                if gate.kind().is_own() {
                    fail(format!(
                        "its judged comparison is our own code (`{}`)",
                        gate.kind().name()
                    ));
                }
                continue;
            };
            let Some(base_plan) = PLANS
                .iter()
                .find(|p| p.platform == plan.platform && p.level == base)
            else {
                fail(format!("the platform has no baseline level `{base}`"));
                continue;
            };
            let (lib, _) = comparisons(base_plan, spec);
            let base_status = limits.get(spec.name, base).map(|l| l.status);
            if lib.kind().is_own() {
                fail(format!(
                    "it is judged against the comparison of `{base}`, which is our own code (`{}`)",
                    lib.kind().name()
                ));
            } else if !matches!(base_status, Some(Status::Release | Status::Dev)) {
                fail(format!(
                    "it is judged against the comparison of `{base}`, where the entry is {} (D2765)",
                    base_status.map_or("not listed", status_name)
                ));
            }
        }
    }
    out
}

/// The comparison kinds of the records of `results` against the measurement plan (D2767): every
/// record of a level of [`PLANS`] must carry the `comparison_kind` of its judged comparison
/// ([`comparisons`]), and a record of a level with a baseline the `judged_against` kind of the
/// baseline level's comparison, so that no record claims a library where the plan compares with
/// our own code (or the reverse). Levels and entries that the plan does not know are findings.
#[must_use]
pub fn check_record_kinds(results: &Results) -> Vec<Finding> {
    let mut out = Vec::new();
    for (level, rows) in results {
        let Some(plan) = PLANS.iter().find(|p| p.level == level) else {
            out.push(Finding {
                path: PathBuf::from("benches/results/cycles"),
                line: 0,
                message: format!("the level `{level}` is not a level of the measurement plan"),
            });
            continue;
        };
        for (entry, r) in rows {
            let Some(r) = r else { continue };
            let mut fail = |message: String| {
                out.push(Finding {
                    path: r.file.clone(),
                    line: 0,
                    message: format!("`{entry}` at `{level}`: {message} (D2767)"),
                });
            };
            let Some(spec) = KERNELS.iter().find(|s| s.name == entry) else {
                fail("not an entry of the benchmark table".into());
                continue;
            };
            let expected = comparisons(plan, spec).0.kind();
            if r.comparison_kind != Some(expected) {
                fail(format!(
                    "`comparison_kind` is {}, the plan compares with `{}` (`{}`)",
                    r.comparison_kind
                        .map_or("missing".to_string(), |k| format!("`{}`", k.name())),
                    r.compared_with,
                    expected.name()
                ));
            }
            let base = plan.baseline.and_then(|base| {
                PLANS
                    .iter()
                    .find(|p| p.platform == plan.platform && p.level == base)
            });
            match (base, &r.judged_against) {
                (None, None) => {}
                (None, Some(_)) => {
                    fail("`judged_against` on a level without a baseline".into());
                }
                (Some(b), j) => {
                    let lib = comparisons(b, spec).0.kind();
                    if j.as_ref().and_then(|j| j.1) != Some(lib) {
                        fail(format!(
                            "`judged_against` must name the comparison of `{}` (`{}`)",
                            b.level,
                            lib.name()
                        ));
                    }
                }
            }
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
#[allow(
    clippy::large_stack_frames,
    reason = "the gates and their messages are built in place; xtask is a host tool"
)]
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
    // No status decided by our own code, from the plan alone and from the records (D2767).
    let plan_findings = check_comparisons(&limits);
    let kind_findings = check_record_kinds(&results);
    println!(
        "check cycles: the judged comparison of every release and dev entry is a foreign library: {} findings; the comparison kinds of the records follow the plan: {} findings (D2767)",
        plan_findings.len(),
        kind_findings.len()
    );
    // Other results (a measurement of a selection of levels) take the library's cycles of
    // `avx2-runtime` from the committed `avx2` records where they lack them (D2764).
    let committed = match &o.results {
        Some(_) => Some(load(&crate::cycle_results::dir(root))?),
        None => None,
    };
    let gate = judge_release(
        &results,
        committed.as_ref(),
        &limits,
        &open_items,
        o.partial,
    );
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
        "check cycles: {} entries and levels against their comparison: {} at the goal of {:.2}x, {} waived, {} accuracy trades, {} with remarks, {} failed (D14 in cycles, the accuracy of D34)",
        gate.judged,
        gate.goal,
        limits.caps.goal,
        gate.waived.len(),
        gate.traded.len(),
        gate.warnings.len(),
        gate.findings.len()
    );
    for w in &gate.waived {
        println!("  waived: {w}");
    }
    for t in &gate.traded {
        println!("  accuracy trade: {t}");
    }
    for w in &gate.warnings {
        println!("  warn: {w}");
    }
    let mut findings = fallbacks.findings;
    findings.extend(plan_findings);
    findings.extend(kind_findings);
    findings.extend(gate.findings);
    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: &str = "| M30 | I30-9: a gap | why | what |\n";

    /// The results of one level with the entries
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
                let kind = match *comp {
                    OWN_FALLBACK => "own_fallback",
                    "ours at avx2" => "own_build",
                    _ => "library",
                };
                (
                    (*name).to_string(),
                    serde_json::json!({
                        "compared_with": comp,
                        "comparison_kind": kind,
                        "reference_model": m("skylake", *or, *cr),
                        "counter_model": m("znver2", *oc, *cc),
                        "snr_db": { "ours": 100.0, "comparison": 100.0 },
                    }),
                )
            })
            .collect();
        let doc = serde_json::json!({ "levels": { level: { "entries": entries } } });
        let mut out = Results::new();
        add_results(Path::new("x86_64.json"), &doc.to_string(), &mut out).unwrap();
        out
    }

    /// `r` with the SNR of `entry` at `level` set (`None`: no `snr_db` in the record).
    fn with_snr(mut r: Results, level: &str, entry: &str, snr: Option<(f64, f64)>) -> Results {
        let rec = r
            .get_mut(level)
            .and_then(|rows| rows.get_mut(entry))
            .and_then(Option::as_mut)
            .unwrap();
        rec.snr = snr.map(|(ours, comparison)| Snr { ours, comparison });
        r
    }

    fn limits_with(entry: &str) -> String {
        format!(
            r#"{{"_limits": {{"release": 1.1, "dev": 1.5, "warn": 1.1, "goal": 1.0}}, {entry}}}"#
        )
    }

    fn gate(r: &Results, limits: &str) -> Gate {
        judge_release(r, None, &Limits::parse(limits).unwrap(), OPEN, false)
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
        // A `dev` entry above the dev limit, awaiting the maintainer's decision (D2779).
        let slow = results(&[("k", 1.6, 1.0, 1.0, 1.0, "volk")], "sse2");
        let w = |t: &str| {
            limits_with(&format!(
                r#""k": {{"sse2": {{"status": "dev", "waiver": "{t}, awaiting the maintainer's decision"}}}}"#
            ))
        };
        let g = gate(&slow, &w("slow loads, open item I30-9"));
        assert_eq!(
            (g.findings.len(), g.waived.len()),
            (0, 1),
            "{:?}",
            g.findings
        );
        let g = gate(&slow, &w("slow loads, open item I30-8"));
        assert_eq!(g.findings.len(), 1);
        assert!(g.findings[0].message.contains("I30-8"));
        // A waiver without an open item keeps no entry above its cap.
        let g = gate(&slow, &w("the reference computes in f64"));
        assert_eq!((g.findings.len(), g.waived.len()), (1, 0));
        // A waiver of an entry within its limit is reported, not failed.
        let fast = results(&[("k", 1.0, 1.0, 1.0, 1.0, "volk")], "sse2");
        let g = gate(&fast, &w("open item I30-9"));
        assert_eq!(g.findings.len(), 0);
        assert!(g.warnings[0].contains("no longer needed"));
    }

    /// `avx2` with `k` against the library `volk` at 1.00 cycles per sample on both models, and
    /// `avx2-runtime` with `k` at `ours` cycles per sample on both models against our `avx2`
    /// build at 0.90.
    fn run_time(ours: (f64, f64), avx2_comparison: &str) -> Results {
        let mut r = results(&[("k", 0.9, 1.0, 0.9, 1.0, avx2_comparison)], "avx2");
        let mut rt = results(
            &[("k", ours.0, 0.9, ours.1, 0.9, "ours at avx2")],
            RUNTIME_LEVEL,
        );
        if let Some(Some(k)) = rt.get_mut(RUNTIME_LEVEL).and_then(|rows| rows.get_mut("k")) {
            let lib = r["avx2"]["k"].as_ref().unwrap();
            k.judged_against = Some((lib.compared_with.clone(), lib.comparison_kind));
        }
        r.extend(rt);
        r
    }

    #[test]
    fn the_run_time_level_is_judged_against_the_library_of_the_avx2_record() {
        let l = limits_with(
            r#""k": {"avx2": {"status": "release"}, "avx2-runtime": {"status": "release"}}"#,
        );
        // 1.20 / 1.15 of our avx2 build is information: 1.08 / 1.035 of the library passes.
        let g = gate(&run_time((1.08, 1.035), "volk"), &l);
        assert_eq!((g.judged, g.findings.len()), (2, 0), "{:?}", g.findings);
        // Above the release limit of the library on one model fails and names the library.
        let g = gate(&run_time((1.12, 1.0), "volk"), &l);
        assert_eq!(g.findings.len(), 1);
        let m = &g.findings[0].message;
        assert!(
            m.contains("`k` at `avx2-runtime`") && m.contains("`volk`"),
            "{m}"
        );
        assert!(
            m.contains("1.12 / 1.00") && m.contains("release limit of 1.10x"),
            "{m}"
        );
        assert!(!m.contains("our avx2 build"), "{m}");
        // `dev` with the dev limit, and the remark of an entry ready for release.
        let dev = limits_with(
            r#""k": {"avx2": {"status": "release"}, "avx2-runtime": {"status": "dev"}}"#,
        );
        let g = gate(&run_time((1.4, 1.0), "volk"), &dev);
        assert_eq!(g.findings.len(), 0);
        let g = gate(&run_time((1.05, 1.0), "volk"), &dev);
        assert!(
            g.warnings[0].contains("ready to be promoted"),
            "{:?}",
            g.warnings
        );
        // At the goal against the library counts, whatever our avx2 build gives.
        let g = gate(&run_time((0.95, 0.95), "volk"), &l);
        assert_eq!(g.goal, 2);
        // No entry of its own fails, as at every level.
        let only_avx2 = limits_with(r#""k": {"avx2": {"status": "release"}}"#);
        let g = gate(&run_time((1.0, 1.0), "volk"), &only_avx2);
        assert!(
            g.findings[0]
                .message
                .contains("no entry in benches/limits.json")
        );
    }

    #[test]
    fn the_run_time_level_takes_the_accuracy_of_the_library_from_the_avx2_record() {
        let l = limits_with(
            r#""k": {"avx2": {"status": "release"}, "avx2-runtime": {"status": "release"}}"#,
        );
        let r = |lib: f64| {
            let r = with_snr(run_time((1.0, 1.0), "volk"), "avx2", "k", Some((lib, lib)));
            with_snr(r, RUNTIME_LEVEL, "k", Some((100.0, 100.0)))
        };
        // Ours 100 dB at the run-time level against 101 dB of the library passes, 101.5 fails
        // (the comparison of the run-time record, our avx2 build, is not used).
        assert_eq!(gate(&r(101.0), &l).findings.len(), 0);
        let g = gate(&r(101.5), &l);
        assert!(g.findings[0].message.contains("below the comparison"));
        assert!(g.findings[0].message.contains("101.5 dB"));
        // An accuracy trade holds at the run-time level as at every level.
        let trade = limits_with(
            r#""k": {"avx2": {"status": "release"}, "avx2-runtime": {"status": "release", "accuracy_trade": "exact against 82.6 dB"}}"#,
        );
        let r = with_snr(
            with_snr(
                run_time((1.15, 1.0), "volk"),
                "avx2",
                "k",
                Some((999.0, 82.6)),
            ),
            RUNTIME_LEVEL,
            "k",
            Some((999.0, 999.0)),
        );
        let g = gate(&r, &trade);
        assert_eq!(
            (g.findings.len(), g.traded.len()),
            (0, 1),
            "{:?}",
            g.findings
        );
    }

    #[test]
    fn the_run_time_level_without_a_library_at_avx2_is_info() {
        let rt = |status: &str| {
            limits_with(&format!(
                r#""k": {{"avx2": {{"status": "info"}}, "avx2-runtime": {{"status": "{status}"}}}}"#
            ))
        };
        let r = run_time((1.0, 1.0), OWN_FALLBACK);
        for status in ["release", "dev"] {
            let g = gate(&r, &rt(status));
            assert_eq!(g.findings.len(), 1, "{status}");
            assert!(
                g.findings[0]
                    .message
                    .contains("judged against our own code"),
                "{}",
                g.findings[0].message
            );
        }
        let g = gate(&r, &rt("info"));
        assert_eq!((g.judged, g.findings.len()), (0, 0));
    }

    #[test]
    fn the_run_time_level_takes_the_avx2_record_from_the_committed_results() {
        let l = limits_with(
            r#""k": {"avx2": {"status": "release"}, "avx2-runtime": {"status": "release"}}"#,
        );
        let both = run_time((1.0, 1.0), "volk");
        let mut alone = both.clone();
        alone.remove("avx2");
        let limits = Limits::parse(&l).unwrap();
        // Without an `avx2` record a judged entry fails, an `info` entry is not judged.
        let g = judge_release(&alone, None, &limits, OPEN, true);
        assert_eq!(g.findings.len(), 1);
        assert!(g.findings[0].message.contains("no simulated `avx2` record"));
        let info =
            Limits::parse(&limits_with(r#""k": {"avx2-runtime": {"status": "info"}}"#)).unwrap();
        assert_eq!(
            judge_release(&alone, None, &info, OPEN, true)
                .findings
                .len(),
            0
        );
        // The committed results give the library's cycles.
        let g = judge_release(&alone, Some(&both), &limits, OPEN, true);
        assert_eq!((g.judged, g.findings.len()), (1, 0), "{:?}", g.findings);
        // Other core models are a finding.
        let mut other = both.clone();
        if let Some(Some(a)) = other.get_mut("avx2").and_then(|rows| rows.get_mut("k")) {
            a.reference.cpu = "znver3".into();
        }
        let g = judge_release(&alone, Some(&other), &limits, OPEN, true);
        assert!(g.findings[0].message.contains("simulated on `znver3`"));
    }

    #[test]
    fn a_listed_entry_needs_a_simulated_record_unless_the_results_are_a_selection() {
        let l = limits_with(
            r#""k": {"sse2": {"status": "release"}}, "m": {"sse2": {"status": "info"}}"#,
        );
        let r = results(&[("j", 1.0, 1.0, 1.0, 1.0, "volk")], "sse2");
        let limits = Limits::parse(&l).unwrap();
        let g = judge_release(&r, None, &limits, OPEN, false);
        // `j` has no entry, `k` has no record; `m` is `info`.
        assert_eq!(g.findings.len(), 2);
        assert!(g.findings.iter().any(|f| {
            f.message
                .contains("`k` at `sse2` has an entry but no record")
        }));
        let g = judge_release(&r, None, &limits, OPEN, true);
        assert_eq!(g.findings.len(), 1);
        // A record that failed is a finding even in a selection.
        let mut failed = r.clone();
        failed.get_mut("sse2").unwrap().insert("k".into(), None);
        let g = judge_release(&failed, None, &limits, OPEN, true);
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
            judge_release(&results(&[], "sse2"), None, &limits, OPEN, true)
                .findings
                .len(),
            0
        );
        let l = limits_with(r#""k": {"simd128": {"status": "release"}}"#);
        assert_eq!(gate(&results(&[], "sse2"), &l).findings.len(), 0);
    }

    #[test]
    fn the_snr_of_both_sides_is_read_from_the_record() {
        let doc = r#"{"levels": {"sse2": {"entries": {
            "k": {"compared_with": "volk", "snr_db": {"ours": 999.0, "comparison": 82.64},
                  "reference_model": {"cpu": "skylake", "ours": {"cycles_per_sample": 1.0}, "comparison": {"cycles_per_sample": 1.0}, "ratio": 1.0},
                  "counter_model": {"cpu": "znver2", "ours": {"cycles_per_sample": 1.0}, "comparison": {"cycles_per_sample": 1.0}, "ratio": 1.0}},
            "j": {"compared_with": "volk",
                  "reference_model": {"cpu": "skylake", "ours": {"cycles_per_sample": 1.0}, "comparison": {"cycles_per_sample": 1.0}, "ratio": 1.0},
                  "counter_model": {"cpu": "znver2", "ours": {"cycles_per_sample": 1.0}, "comparison": {"cycles_per_sample": 1.0}, "ratio": 1.0}}}}}}"#;
        let mut r = Results::new();
        add_results(Path::new("x.json"), doc, &mut r).unwrap();
        let k = r["sse2"]["k"].as_ref().unwrap().snr.unwrap();
        assert_eq!((k.ours, k.comparison), (999.0, 82.64));
        assert!(k.describe().contains("exact against 82.6 dB"));
        assert!(r["sse2"]["j"].as_ref().unwrap().snr.is_none());
        assert_eq!(snr_text(999.0), "exact");
        assert_eq!(snr_text(41.64), "41.6");
    }

    #[test]
    fn the_accuracy_may_be_at_most_one_db_below_the_comparison() {
        let l = limits_with(r#""k": {"sse2": {"status": "release"}}"#);
        let fast = || results(&[("k", 1.0, 1.0, 1.0, 1.0, "volk")], "sse2");
        // 1 dB below passes, more fails; above passes.
        for (ours, ok) in [(99.0, true), (98.99, false), (120.0, true), (60.0, false)] {
            let g = gate(&with_snr(fast(), "sse2", "k", Some((ours, 100.0))), &l);
            assert_eq!(g.findings.is_empty(), ok, "{ours}: {:?}", g.findings);
            if !ok {
                assert!(g.findings[0].message.contains("below the comparison"));
                assert!(g.findings[0].message.contains("D34"));
            }
        }
        // `dev` is judged as well, `info` is not.
        let dev = limits_with(r#""k": {"sse2": {"status": "dev"}}"#);
        let low = || with_snr(fast(), "sse2", "k", Some((90.0, 100.0)));
        assert_eq!(gate(&low(), &dev).findings.len(), 1);
        let info = limits_with(r#""k": {"sse2": {"status": "info"}}"#);
        assert_eq!(gate(&low(), &info).findings.len(), 0);
        // A judged record without the accuracy fails.
        let g = gate(&with_snr(fast(), "sse2", "k", None), &l);
        assert!(g.findings[0].message.contains("no accuracy"));
    }

    #[test]
    fn an_accuracy_waiver_covers_only_the_accuracy_and_needs_existing_open_items() {
        let w = |t: &str| {
            limits_with(&format!(
                r#""k": {{"sse2": {{"status": "release", "accuracy_waiver": "{t}"}}}}"#
            ))
        };
        let low = |ours_ref: f64| {
            with_snr(
                results(&[("k", ours_ref, 1.0, 1.0, 1.0, "volk")], "sse2"),
                "sse2",
                "k",
                Some((90.0, 100.0)),
            )
        };
        let g = gate(&low(1.0), &w("open item I30-9"));
        assert_eq!((g.findings.len(), g.waived.len()), (0, 1));
        assert!(g.waived[0].contains("accuracy waiver: open item I30-9"));
        let g = gate(&low(1.0), &w("open item I30-8"));
        assert!(g.findings[0].message.contains("I30-8"));
        // It does not cover the cycles.
        let g = gate(&low(1.3), &w("open item I30-9"));
        assert_eq!(g.findings.len(), 1);
        assert!(g.findings[0].message.contains("release limit"));
        // Within the margin it is a remark.
        let ok = with_snr(
            results(&[("k", 1.0, 1.0, 1.0, 1.0, "volk")], "sse2"),
            "sse2",
            "k",
            Some((100.0, 100.0)),
        );
        let g = gate(&ok, &w("open item I30-9"));
        assert_eq!(g.findings.len(), 0);
        assert!(g.warnings[0].contains("accuracy waiver is no longer needed"));
    }

    #[test]
    fn an_accuracy_trade_keeps_release_while_three_db_the_scarce_comparison_and_the_cap_hold() {
        let l = |status: &str, extra: &str| {
            limits_with(&format!(
                r#""k": {{"sse2": {{"status": "{status}", "accuracy_trade": "exact against 82.6 dB"{extra}}}}}"#
            ))
        };
        let rec = |ratio: f64, snr: Option<(f64, f64)>| {
            with_snr(
                results(&[("k", ratio, 1.0, 1.0, 1.0, "CMSIS-DSP")], "sse2"),
                "sse2",
                "k",
                snr,
            )
        };
        // Over the release limit, at most 1.20 on both models, 3 dB or more above: traded.
        for (ratio, snr) in [(1.2, (85.6, 82.6)), (1.15, (999.0, 82.64))] {
            let g = gate(&rec(ratio, Some(snr)), &l("release", ""));
            assert_eq!(
                (g.findings.len(), g.traded.len(), g.waived.len()),
                (0, 1, 0),
                "{:?}",
                g.findings
            );
            assert!(g.traded[0].contains("traded for the accuracy"));
        }
        // Less than 3 dB above, no accuracy, or above 1.20 on a model (within dev or not): fails.
        let g = gate(&rec(1.15, Some((85.59, 82.6))), &l("release", ""));
        assert!(g.findings[0].message.contains("at least 3 dB above"));
        let g = gate(&rec(1.15, None), &l("release", ""));
        assert_eq!(g.findings.len(), 1);
        for ratio in [1.2001, 1.3, 1.51] {
            let g = gate(&rec(ratio, Some((999.0, 82.6))), &l("release", ""));
            assert_eq!((g.findings.len(), g.traded.len()), (1, 0), "{ratio}");
            assert!(g.findings[0].message.contains("only up to 1.20x"));
        }
        // The comparison at 100 dB or above: accuracy is not scarce, no trade.
        for snr in [(140.0, 100.0), (999.0, 137.5)] {
            let g = gate(&rec(1.15, Some(snr)), &l("release", ""));
            assert_eq!((g.findings.len(), g.traded.len()), (1, 0), "{snr:?}");
            assert!(g.findings[0].message.contains("below 100 dB"));
        }
        let g = gate(&rec(1.15, Some((103.0, 99.99))), &l("release", ""));
        assert_eq!(g.traded.len(), 1);
        // The counter-model alone above 1.20 fails too.
        let counter = with_snr(
            results(&[("k", 1.15, 1.0, 1.25, 1.0, "CMSIS-DSP")], "sse2"),
            "sse2",
            "k",
            Some((999.0, 82.6)),
        );
        let g = gate(&counter, &l("release", ""));
        assert!(g.findings[0].message.contains("only up to 1.20x"));
        // Within the release limit: a remark.
        let g = gate(&rec(1.05, Some((999.0, 82.6))), &l("release", ""));
        assert_eq!((g.findings.len(), g.traded.len()), (0, 0));
        assert!(g.warnings[0].contains("accuracy trade is no longer needed"));
        // Only on a release entry, not with a waiver.
        let g = gate(&rec(1.3, Some((999.0, 82.6))), &l("dev", ""));
        assert!(g.findings[0].message.contains("only for a `release` entry"));
        let g = gate(&rec(1.3, Some((999.0, 82.6))), &l("info", ""));
        assert_eq!(g.findings.len(), 1);
        let g = gate(
            &rec(1.3, Some((999.0, 82.6))),
            &l("release", r#", "waiver": "open item I30-9""#),
        );
        assert!(g.findings[0].message.contains("not both"));
        // A trade does not cover an accuracy below the comparison of another entry's kind:
        // the accuracy is judged on its own.
        let g = gate(&rec(1.0, Some((90.0, 100.0))), &l("release", ""));
        assert!(
            g.findings
                .iter()
                .any(|f| f.message.contains("below the comparison"))
        );
    }

    #[test]
    fn release_and_dev_against_our_own_fallback_fail_at_every_level() {
        for level in ["sse2", "sse4.2", "avx2", "avx512", "neon"] {
            for status in ["release", "dev"] {
                let l = limits_with(&format!(r#""k": {{"{level}": {{"status": "{status}"}}}}"#));
                // Ours at 1.00 of our own fallback (a delegated level) would pass any cap.
                let r = results(&[("k", 1.0, 1.0, 1.0, 1.0, OWN_FALLBACK)], level);
                let g = gate(&r, &l);
                assert_eq!(g.findings.len(), 1, "{level} {status}");
                let m = &g.findings[0].message;
                assert!(
                    m.contains("judged against our own code") && m.contains("own_fallback"),
                    "{m}"
                );
                assert_eq!(g.goal, 0, "no goal against our own code");
                // `info` against our own fallback is information.
                let info = limits_with(&format!(r#""k": {{"{level}": {{"status": "info"}}}}"#));
                assert_eq!(gate(&r, &info).findings.len(), 0);
            }
        }
        // A record that does not say what its comparison is fails as well.
        let mut r = results(&[("k", 1.0, 1.0, 1.0, 1.0, "volk")], "sse2");
        if let Some(Some(k)) = r.get_mut("sse2").and_then(|rows| rows.get_mut("k")) {
            k.comparison_kind = None;
        }
        let g = gate(&r, &limits_with(r#""k": {"sse2": {"status": "release"}}"#));
        assert!(g.findings[0].message.contains("no `comparison_kind`"));
    }

    #[test]
    fn the_run_time_level_against_our_avx2_build_fails() {
        let l = limits_with(
            r#""k": {"avx2": {"status": "release"}, "avx2-runtime": {"status": "release"}}"#,
        );
        // The record itself compares with our own build: judged directly it fails.
        let r = run_time((1.0, 1.0), "volk");
        let rt = r[RUNTIME_LEVEL]["k"].as_ref().unwrap();
        let limits = Limits::parse(&l).unwrap();
        assert!(matches!(
            judge_record(&limits.caps, limits.get("k", RUNTIME_LEVEL), OPEN, rt),
            Verdict::Fail(t) if t.contains("judged against our own code") && t.contains("own_build")
        ));
        // Without `judged_against`, or naming another comparison than the avx2 record, it fails.
        let mut without = r.clone();
        if let Some(Some(k)) = without
            .get_mut(RUNTIME_LEVEL)
            .and_then(|rows| rows.get_mut("k"))
        {
            k.judged_against = None;
        }
        assert!(
            gate(&without, &l).findings[0]
                .message
                .contains("does not name the library")
        );
        let mut stale = r.clone();
        if let Some(Some(k)) = stale
            .get_mut(RUNTIME_LEVEL)
            .and_then(|rows| rows.get_mut("k"))
        {
            k.judged_against = Some(("ours at avx2".into(), Some(ComparisonKind::OwnBuild)));
        }
        assert!(
            gate(&stale, &l).findings[0]
                .message
                .contains("measure it again")
        );
        // `info` at avx2 (whatever the comparison) makes a judged avx2-runtime entry fail (D2765).
        for avx2 in ["info", "missing"] {
            let text = if avx2 == "missing" {
                limits_with(r#""k": {"avx2-runtime": {"status": "release"}}"#)
            } else {
                limits_with(
                    r#""k": {"avx2": {"status": "info"}, "avx2-runtime": {"status": "release"}}"#,
                )
            };
            let g = gate(&r, &text);
            assert!(
                g.findings
                    .iter()
                    .any(|f| f.message.contains("is `info` at `avx2-runtime` too")),
                "{avx2}: {:?}",
                g.findings
            );
        }
    }

    #[test]
    fn a_waiver_keeps_no_release_entry_above_the_release_limit() {
        let open = "| M1 | I1-1: slow | why | what |\n";
        let r = results(&[("k", 1.2, 1.0, 1.0, 1.0, "volk")], "sse2");
        let release =
            limits_with(r#""k": {"sse2": {"status": "release", "waiver": "open item I1-1"}}"#);
        let g = judge_release(&r, None, &Limits::parse(&release).unwrap(), open, false);
        assert!(
            g.findings[0]
                .message
                .contains("a waiver cannot keep a `release` entry")
        );
        let dev = limits_with(r#""k": {"sse2": {"status": "dev", "waiver": "open item I1-1"}}"#);
        let g = judge_release(&r, None, &Limits::parse(&dev).unwrap(), open, false);
        assert_eq!(g.findings.len(), 0, "{:?}", g.findings);
        // Above the dev limit only while awaiting the maintainer's decision.
        let r = results(&[("k", 1.6, 1.0, 1.0, 1.0, "volk")], "sse2");
        let g = judge_release(&r, None, &Limits::parse(&dev).unwrap(), open, false);
        assert_eq!(g.findings.len(), 1);
        let awaiting = limits_with(
            r#""k": {"sse2": {"status": "dev", "waiver": "open item I1-1: awaiting the maintainer's decision"}}"#,
        );
        let g = judge_release(&r, None, &Limits::parse(&awaiting).unwrap(), open, false);
        assert_eq!(
            (g.findings.len(), g.waived.len()),
            (0, 1),
            "{:?}",
            g.findings
        );
    }

    /// The entry of the table at `level`, with its judged comparison.
    fn plan_entry(
        level: &str,
        pick: impl Fn(&crate::bench::KernelSpec, crate::bench::Reference) -> bool,
    ) -> &'static str {
        let plan = PLANS.iter().find(|p| p.level == level).unwrap();
        KERNELS
            .iter()
            .find(|s| s.family.measured_at(plan) && pick(s, comparisons(plan, s).0))
            .unwrap()
            .name
    }

    #[test]
    fn the_static_check_finds_a_status_judged_against_our_own_code() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let real = std::fs::read_to_string(root.join("benches/limits.json")).unwrap();
        assert_eq!(
            check_comparisons(&Limits::parse(&real).unwrap()),
            Vec::new()
        );
        // An entry whose judged comparison is our own fallback, release at sse2 and dev at simd128.
        let own = plan_entry("sse2", |_, r| r.kind() == ComparisonKind::OwnFallback);
        let own_wasm = plan_entry("simd128", |_, r| r.kind() == ComparisonKind::OwnFallback);
        let lib = plan_entry("avx2", |s, r| {
            r.kind() == ComparisonKind::Library && s.family.measured_at(&PLANS[3])
        });
        let own_avx2 = plan_entry("avx2", |s, r| {
            r.kind() == ComparisonKind::OwnFallback && s.family.measured_at(&PLANS[3])
        });
        let mut doc = serde_json::Map::new();
        for (entry, level, status) in [
            (own, "sse2", "release"),
            (own_wasm, "simd128", "dev"),
            (lib, "avx2", "info"),
            (lib, "avx2-runtime", "release"),
            (own_avx2, "avx2", "info"),
            (own_avx2, "avx2-runtime", "dev"),
        ] {
            doc.entry(entry)
                .or_insert_with(|| serde_json::json!({}))
                .as_object_mut()
                .unwrap()
                .insert(level.into(), serde_json::json!({ "status": status }));
        }
        let text = Value::Object(doc).to_string();
        let f = check_comparisons(&Limits::parse(&text).unwrap());
        let messages: Vec<&str> = f.iter().map(|f| f.message.as_str()).collect();
        assert_eq!(f.len(), 4, "{messages:#?}");
        assert!(
            messages[..]
                .iter()
                .any(|m| m.contains(&format!("`{own}` at `sse2` is `release`"))
                    && m.contains("own_fallback"))
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains(&format!("`{own_wasm}` at `simd128` is `dev`")))
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains(&format!("`{lib}` at `avx2-runtime`"))
                    && m.contains("where the entry is info"))
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains(&format!("`{own_avx2}` at `avx2-runtime`"))
                    && m.contains("our own code"))
        );
        // The same entries as `info` pass.
        let info = text
            .replace("\"release\"", "\"info\"")
            .replace("\"dev\"", "\"info\"");
        assert_eq!(
            check_comparisons(&Limits::parse(&info).unwrap()),
            Vec::new()
        );
    }

    #[test]
    fn the_comparison_kinds_of_the_records_follow_the_plan() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let committed = load(&crate::cycle_results::dir(root)).unwrap();
        assert_eq!(check_record_kinds(&committed), Vec::new());
        // A record that claims a library where the plan compares with our own fallback.
        let own = plan_entry("sse2", |_, r| r.kind() == ComparisonKind::OwnFallback);
        let r = results(&[(own, 1.0, 1.0, 1.0, 1.0, "a library")], "sse2");
        let f = check_record_kinds(&r);
        assert_eq!(f.len(), 1);
        assert!(
            f[0].message.contains("`library`") && f[0].message.contains("own_fallback"),
            "{}",
            f[0].message
        );
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
