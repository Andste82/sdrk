//! `cargo xtask check cycles`: no SIMD entry may be slower than its own fallback in simulated
//! cycles per sample (D321).
//!
//! The source of truth is the committed cycle tables of `docs/performance.md` (between
//! `<!-- cycle-tables:begin -->` and `<!-- cycle-tables:end -->`, written by
//! `benches/cycle_tables.py` from the `cargo measure` documents), so CI enforces the rule
//! without running `llvm-mca`. For every level and every entry `<k>` whose table also has the
//! row `<k>_fallback`, the cycles per sample of ours (the column "ours" as published, two
//! decimals) must satisfy `cycles(<k>) <= cycles(<k>_fallback)` on the reference model **and**
//! on the counter-model of the level. A level delegated to the fallback (comparison
//! `sdrk fallback`, both ratios `1.00`) passes.
//!
//! A violation passes only with a `cycle_waiver` on the SIMD entry and level in
//! `benches/limits.json` (`{"<k>": {"<level>": {"status": ..., "cycle_waiver": "<reason>, open
//! item <ID>"}}}`) that names an open item of `docs/open-items.md`. A `cycle_waiver` where the
//! rule holds, or on an entry and level the tables do not pair, is a finding as well, so that no
//! waiver outlives its gap.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use crate::scan::Finding;

/// The markers of the generated block of `docs/performance.md`.
const BEGIN: &str = "<!-- cycle-tables:begin -->";
const END: &str = "<!-- cycle-tables:end -->";
/// The comparison of a SIMD entry that runs our fallback on its level.
const OWN_FALLBACK: &str = "sdrk fallback";
/// The suffix of the fallback entry of a kernel.
const FALLBACK: &str = "_fallback";

/// One row of a cycle table: cycles per sample in hundredths (as published).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    /// 1-based line of the row in the document.
    pub line: usize,
    /// Ours on the reference model.
    pub ours_reference: i64,
    /// The ratio ours / comparison on the reference model.
    pub ratio_reference: i64,
    /// Ours on the counter-model.
    pub ours_counter: i64,
    /// The ratio ours / comparison on the counter-model.
    pub ratio_counter: i64,
    /// The comparison column.
    pub comparison: String,
}

/// The rows of the cycle tables, by level, then entry.
pub type Tables = BTreeMap<String, BTreeMap<String, Row>>;

/// A decimal number of the table in hundredths (`0.52` is 52).
fn hundredths(cell: &str) -> Option<i64> {
    let x: f64 = cell.trim().parse().ok()?;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the cells are small numbers with two decimals"
    )]
    Some((x * 100.0).round() as i64)
}

/// Parses the cycle tables of `docs/performance.md`: the level from each heading
/// ``#### Simulated cycles at `<level>` ``, the rows `` | `<entry>` | ours ref | other ref |
/// ratio ref | ours counter | other counter | ratio counter | comparison | ``.
///
/// # Errors
/// The markers are missing, or a row of an entry has a cell that is not a number.
pub fn parse_tables(doc: &str) -> anyhow::Result<Tables> {
    let mut tables = Tables::new();
    let mut inside = false;
    let mut found = false;
    let mut level: Option<String> = None;
    for (i, line) in doc.lines().enumerate() {
        let t = line.trim();
        if t == BEGIN {
            inside = true;
            found = true;
            continue;
        }
        if t == END {
            inside = false;
            continue;
        }
        if !inside {
            continue;
        }
        if let Some(rest) = t.strip_prefix("#### Simulated cycles at `") {
            level = rest.split('`').next().map(str::to_string);
            continue;
        }
        let Some(level) = &level else { continue };
        let cells: Vec<&str> = t.split('|').map(str::trim).collect();
        // `| a | b | ... | h |` splits into an empty first and last cell.
        let [
            "",
            name,
            ours_ref,
            _,
            ratio_ref,
            ours_counter,
            _,
            ratio_counter,
            comparison,
            "",
        ] = cells.as_slice()
        else {
            continue;
        };
        if !name.starts_with('`') {
            continue;
        }
        let entry = name.trim_matches('`').to_string();
        let num = |cell: &str| {
            hundredths(cell).with_context(|| {
                format!(
                    "docs/performance.md:{}: `{entry}` at `{level}`: `{cell}` is not a number",
                    i + 1,
                )
            })
        };
        let row = Row {
            line: i + 1,
            ours_reference: num(ours_ref)?,
            ratio_reference: num(ratio_ref)?,
            ours_counter: num(ours_counter)?,
            ratio_counter: num(ratio_counter)?,
            comparison: (*comparison).to_string(),
        };
        tables.entry(level.clone()).or_default().insert(entry, row);
    }
    anyhow::ensure!(
        found,
        "docs/performance.md has no `{BEGIN}` block of cycle tables"
    );
    Ok(tables)
}

/// One SIMD entry and level compared with its fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    /// The SIMD entry.
    pub entry: String,
    /// The level.
    pub level: String,
    /// The SIMD row.
    pub simd: Row,
    /// The row of `<entry>_fallback`.
    pub fallback: Row,
}

impl Pair {
    /// Whether the level runs our fallback (comparison `sdrk fallback`, both ratios 1.00).
    #[must_use]
    pub fn delegated(&self) -> bool {
        self.simd.comparison == OWN_FALLBACK
            && self.simd.ratio_reference == 100
            && self.simd.ratio_counter == 100
    }

    /// Whether the SIMD entry is at most as slow as its fallback on both models.
    #[must_use]
    pub fn holds(&self) -> bool {
        self.delegated()
            || (self.simd.ours_reference <= self.fallback.ours_reference
                && self.simd.ours_counter <= self.fallback.ours_counter)
    }

    /// The cycles of both sides, for the messages.
    #[must_use]
    pub fn describe(&self) -> String {
        let c = |x: i64| format!("{:.2}", x as f64 / 100.0);
        let ratio = |a: i64, b: i64| {
            if b == 0 {
                "-".to_string()
            } else {
                format!("{:.2}", a as f64 / b as f64)
            }
        };
        format!(
            "reference model {} / {} ({}), counter-model {} / {} ({}) cycles per sample, SIMD / fallback",
            c(self.simd.ours_reference),
            c(self.fallback.ours_reference),
            ratio(self.simd.ours_reference, self.fallback.ours_reference),
            c(self.simd.ours_counter),
            c(self.fallback.ours_counter),
            ratio(self.simd.ours_counter, self.fallback.ours_counter),
        )
    }
}

/// Every SIMD entry with a fallback row on the same level.
#[must_use]
pub fn pairs(tables: &Tables) -> Vec<Pair> {
    let mut out = Vec::new();
    for (level, rows) in tables {
        for (entry, simd) in rows {
            if entry.ends_with(FALLBACK) {
                continue;
            }
            if let Some(fallback) = rows.get(&format!("{entry}{FALLBACK}")) {
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

/// The outcome of the gate.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Violations without a valid waiver, and stale or invalid waivers.
    pub findings: Vec<Finding>,
    /// The waived violations: the pair and the waiver text.
    pub waived: Vec<(Pair, String)>,
    /// The number of pairs judged.
    pub judged: usize,
}

/// Judges every pair of the tables of `doc`, with the waivers of `limits_json` and the open
/// items of `open_items`.
///
/// # Errors
/// The tables or `benches/limits.json` cannot be parsed.
#[allow(
    clippy::large_stack_frames,
    reason = "the findings and waived pairs are built in place; xtask is a host tool"
)]
pub fn judge(doc: &str, limits_json: &str, open_items: &str) -> anyhow::Result<Outcome> {
    let tables = parse_tables(doc)?;
    let mut waivers = cycle_waivers(limits_json)?;
    let mut out = Outcome::default();
    let perf = PathBuf::from("docs/performance.md");
    let limits = PathBuf::from("benches/limits.json");
    for pair in pairs(&tables) {
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
                path: perf.clone(),
                line: pair.simd.line,
                message: format!(
                    "`{}` at `{}` is slower than `{}{FALLBACK}` in simulated cycles: {}; make the SIMD path at least as fast, or add a `cycle_waiver` naming an open item to benches/limits.json (D321)",
                    pair.entry,
                    pair.level,
                    pair.entry,
                    pair.describe()
                ),
            }),
            (false, Some(text)) => {
                let items: Vec<String> = crate::release::references(&text)
                    .into_iter()
                    .filter_map(|r| r.strip_prefix("open item ").map(str::to_string))
                    .collect();
                let missing: Vec<&String> = items
                    .iter()
                    .filter(|id| !open_items.contains(&format!("| {id}:")))
                    .collect();
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
                                missing
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
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
        out.findings.push(Finding {
            path: limits.clone(),
            line: 0,
            message: format!(
                "`{entry}` at `{level}` has a `cycle_waiver`, but the cycle tables of docs/performance.md have no `{entry}` / `{entry}{FALLBACK}` pair at that level; remove the waiver"
            ),
        });
    }
    Ok(out)
}

/// Runs the gate on the files of the repository at `root`.
///
/// # Errors
/// A file cannot be read or parsed.
pub fn check(root: &Path) -> anyhow::Result<Vec<Finding>> {
    let read =
        |p: &str| std::fs::read_to_string(root.join(p)).with_context(|| format!("cannot read {p}"));
    let outcome = judge(
        &read("docs/performance.md")?,
        &read("benches/limits.json")?,
        &read("docs/open-items.md")?,
    )?;
    println!(
        "check cycles: {} SIMD entries and levels against their fallback, {} waived",
        outcome.judged,
        outcome.waived.len()
    );
    for (pair, text) in &outcome.waived {
        println!(
            "  waived: `{}` at `{}`: {} ({text})",
            pair.entry,
            pair.level,
            pair.describe()
        );
    }
    Ok(outcome.findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "| Entry | ours skylake (cycles/sample) | comparison skylake (cycles/sample) | ratio skylake (ours/comparison) | ours znver2 (cycles/sample) | comparison znver2 (cycles/sample) | ratio znver2 (ours/comparison) | comparison |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n";

    fn doc(rows: &[&str]) -> String {
        format!(
            "intro\n{BEGIN}\n\n#### Simulated cycles at `sse2` (`x86_64`)\n\nmodels\n\n{HEAD}{}\n{END}\n| `outside` | 9.00 | 1.00 | 9.00 | 9.00 | 1.00 | 9.00 | x |\n",
            rows.join("\n")
        )
    }

    const OPEN: &str = "| M30 | I30-9: a gap | why | what |\n";

    fn limits(waiver: Option<&str>) -> String {
        match waiver {
            None => r#"{"_limits": {"release": 1.1}, "k": {"sse2": {"status": "release"}}}"#.into(),
            Some(w) => {
                format!(r#"{{"k": {{"sse2": {{"status": "release", "cycle_waiver": "{w}"}}}}}}"#)
            }
        }
    }

    #[test]
    fn rows_are_parsed_per_level_inside_the_markers_only() {
        let t = parse_tables(&doc(&[
            "| `k` | 0.52 | 0.70 | 0.73 | 0.56 | 0.79 | 0.71 | volk_x |",
            "| `k_fallback` | 0.51 | 0.70 | 1.00 | 0.58 | 0.91 | 1.00 | CMSIS-DSP |",
        ]))
        .unwrap();
        assert_eq!(t.len(), 1);
        let rows = &t["sse2"];
        assert_eq!(rows.len(), 2);
        assert_eq!(rows["k"].ours_reference, 52);
        assert_eq!(rows["k"].ours_counter, 56);
        assert_eq!(rows["k"].ratio_reference, 73);
        assert_eq!(rows["k"].comparison, "volk_x");
        assert_eq!(rows["k"].line, 10);
        assert!(parse_tables("no block").is_err());
        assert!(
            parse_tables(&doc(&[
                "| `k` | x | 0.70 | 0.73 | 0.56 | 0.79 | 0.71 | v |"
            ]))
            .is_err()
        );
    }

    #[test]
    fn slower_on_either_model_is_a_violation_and_equal_passes() {
        let judge_rows =
            |simd: &str, fb: &str| judge(&doc(&[simd, fb]), &limits(None), OPEN).unwrap();
        let fb = "| `k_fallback` | 0.51 | 0.70 | 0.73 | 0.58 | 0.91 | 0.64 | CMSIS-DSP |";
        // Faster on both, and equal on both.
        let o = judge_rows(
            "| `k` | 0.50 | 0.70 | 0.71 | 0.57 | 0.79 | 0.72 | volk |",
            fb,
        );
        assert_eq!((o.judged, o.findings.len()), (1, 0));
        let o = judge_rows(
            "| `k` | 0.51 | 0.70 | 0.73 | 0.58 | 0.79 | 0.73 | volk |",
            fb,
        );
        assert_eq!(o.findings.len(), 0);
        // Slower on the reference model only, then on the counter-model only.
        let o = judge_rows(
            "| `k` | 0.52 | 0.70 | 0.74 | 0.56 | 0.79 | 0.71 | volk |",
            fb,
        );
        assert_eq!(o.findings.len(), 1);
        assert_eq!(o.findings[0].line, 10);
        assert!(o.findings[0].message.contains("0.52 / 0.51 (1.02)"));
        let o = judge_rows(
            "| `k` | 0.50 | 0.70 | 0.71 | 0.59 | 0.79 | 0.75 | volk |",
            fb,
        );
        assert_eq!(o.findings.len(), 1);
        // No fallback row: nothing to judge.
        let o = judge(
            &doc(&["| `k` | 9.00 | 0.70 | 0.74 | 9.00 | 0.79 | 0.71 | volk |"]),
            &limits(None),
            OPEN,
        )
        .unwrap();
        assert_eq!((o.judged, o.findings.len()), (0, 0));
    }

    #[test]
    fn a_level_delegated_to_the_fallback_passes() {
        let fb = "| `k_fallback` | 0.97 | 1.21 | 0.80 | 1.02 | 1.39 | 0.73 | CMSIS-DSP |";
        let o = judge(
            &doc(&[
                "| `k` | 0.98 | 0.98 | 1.00 | 1.03 | 1.03 | 1.00 | sdrk fallback |",
                fb,
            ]),
            &limits(None),
            OPEN,
        )
        .unwrap();
        assert_eq!(o.findings.len(), 0);
        // Our fallback as the comparison, but a ratio other than 1.00: judged.
        let o = judge(
            &doc(&[
                "| `k` | 0.98 | 0.97 | 1.01 | 1.03 | 1.02 | 1.01 | sdrk fallback |",
                fb,
            ]),
            &limits(None),
            OPEN,
        )
        .unwrap();
        assert_eq!(o.findings.len(), 1);
    }

    #[test]
    fn a_waiver_must_name_an_existing_open_item_and_must_not_be_stale() {
        let slow = doc(&[
            "| `k` | 0.52 | 0.70 | 0.74 | 0.56 | 0.79 | 0.71 | volk |",
            "| `k_fallback` | 0.51 | 0.70 | 0.73 | 0.58 | 0.91 | 0.64 | CMSIS-DSP |",
        ]);
        let o = judge(&slow, &limits(Some("store-bound, open item I30-9")), OPEN).unwrap();
        assert_eq!(o.findings.len(), 0);
        assert_eq!(o.waived.len(), 1);
        let o = judge(&slow, &limits(Some("store-bound, open item I30-8")), OPEN).unwrap();
        assert_eq!(o.findings.len(), 1);
        assert!(o.findings[0].message.contains("I30-8"));
        let o = judge(&slow, &limits(Some("store-bound")), OPEN).unwrap();
        assert_eq!(o.findings.len(), 1);
        // A waiver on a pair that holds, or on a pair that does not exist.
        let fast = doc(&[
            "| `k` | 0.50 | 0.70 | 0.74 | 0.56 | 0.79 | 0.71 | volk |",
            "| `k_fallback` | 0.51 | 0.70 | 0.73 | 0.58 | 0.91 | 0.64 | CMSIS-DSP |",
        ]);
        let o = judge(&fast, &limits(Some("open item I30-9")), OPEN).unwrap();
        assert_eq!(o.findings.len(), 1);
        assert!(o.findings[0].message.contains("remove the waiver"));
        let o = judge(
            &doc(&["| `j` | 0.50 | 0.70 | 0.74 | 0.56 | 0.79 | 0.71 | volk |"]),
            &limits(Some("open item I30-9")),
            OPEN,
        )
        .unwrap();
        assert_eq!(o.findings.len(), 1);
        assert!(cycle_waivers(r#"{"k": {"sse2": {"cycle_waiver": 1}}}"#).is_err());
    }
}
