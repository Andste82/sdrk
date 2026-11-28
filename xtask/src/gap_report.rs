//! The report of `cargo xtask check cycles` on the gaps of the kernels (O2 group 5, D2978, the
//! inventory of issue #203). A report, never a finding: nothing here fails the command or
//! decides a status.
//!
//! **SIMD with little gain (B of issue #203).** Every SIMD entry and level that needs more than
//! [`SIMD_SHARE`] of the cycles per sample of its own `<k>_fallback` on the reference model **or**
//! on the counter-model of the level, in the two decimals of the tables, with the share on both
//! models; a level that runs the fallback (comparison `sdrk fallback`) is marked. At `simd128`,
//! which has no cycles, the same share in instructions per sample of `benches/results/wasm32.json`
//! (marked). D321 fails only a SIMD path that is slower than its fallback; this report names the
//! paths that are faster by less than a fifth.
//!
//! **No foreign reference (A of issue #203).** Every kernel entry and level whose comparison is
//! our own code (`comparison_kind` not `library`), so that the measurement order (D32, D2764)
//! gives it no status. Kernels only ([`crate::bench::is_kernel_entry`]): blocks, composites,
//! protocol coders and the workloads of own-kernel comparisons (`halfband_ref_*`) are left out.
//! `avx2-runtime` is left out as well: it is judged through the library of its `avx2` record.
#![allow(clippy::print_stdout)]

use std::collections::BTreeMap;

use anyhow::{Context, anyhow};
use serde_json::Value;

use crate::bench::{ComparisonKind, PLANS};
use crate::cycle_gate::{Pair, Results};

/// The share of its fallback's cycles (or, at `simd128`, instructions) per sample above which a
/// SIMD entry is reported (O2 group 5, the target of the round).
pub const SIMD_SHARE: f64 = 0.80;

/// The explicit SIMD paths at about 1.00 of their fallback that the maintainer's rule D3019
/// accepts (entry, level): the report names them, marked [`ACCEPTED`], because a path of its own
/// keeps the speed independent of the compiler's auto-vectorisation of the fallback; such a
/// path is preferred over a switch to the fallback and is never deleted as a duplicate. The
/// AVX2 paths of `interleave`, `interleave_cq15` and `deinterleave` and the NEON paths of
/// `stereo_matrix` and `stereo_matrix_cq15` (removed by D2971 and D2972, restored by D3020),
/// `stereo_matrix` at `avx2` (D3011), the SSE2 paths of the two `Q15` biquads (D3015), `mag` at
/// `avx512` and `neon` (IO2-11) and `mul` at `sse2` (IO2-12).
pub const EXPLICIT_FORM: [(&str, &str); 11] = [
    ("biquad_cq15", "sse2"),
    ("biquad_real_q15", "sse2"),
    ("deinterleave", "avx2"),
    ("interleave", "avx2"),
    ("interleave_cq15", "avx2"),
    ("mag", "avx512"),
    ("mag", "neon"),
    ("mul", "sse2"),
    ("stereo_matrix", "avx2"),
    ("stereo_matrix", "neon"),
    ("stereo_matrix_cq15", "neon"),
];

/// The mark of an entry of [`EXPLICIT_FORM`] in the report.
pub const ACCEPTED: &str = "accepted: explicit form, compiler-independent";

/// Whether `entry` at `level` is an accepted explicit form ([`EXPLICIT_FORM`]).
#[must_use]
pub fn accepted(entry: &str, level: &str) -> bool {
    EXPLICIT_FORM.contains(&(entry, level))
}

/// The level that is left out of the inventory without a foreign reference (D2764).
const RUNTIME_LEVEL: &str = "avx2-runtime";
/// The suffix of the fallback entry of a kernel.
const FALLBACK: &str = "_fallback";

/// One counted entry of `benches/results/wasm32.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Counted {
    /// Instructions per sample of ours (`None`: the entry was not counted).
    pub ours: Option<f64>,
    /// What the comparison is (`None`: the record does not say).
    pub kind: Option<ComparisonKind>,
}

/// The counted entries by level, then entry.
pub type Counts = BTreeMap<String, BTreeMap<String, Counted>>;

/// The counted entries of a count results document (`benches/results/wasm32.json`).
///
/// # Errors
/// Invalid JSON, a document without `levels`, or an unknown `comparison_kind`.
pub fn counts(text: &str) -> anyhow::Result<Counts> {
    let doc: Value = serde_json::from_str(text).context("benches/results/wasm32.json")?;
    let levels = doc
        .get("levels")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("benches/results/wasm32.json has no `levels`"))?;
    let mut out = Counts::new();
    for (level, l) in levels {
        let mut rows = BTreeMap::new();
        for (entry, r) in l
            .get("kernels")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            let kind = match r.get("comparison_kind") {
                None => None,
                Some(k) => Some(k.as_str().and_then(ComparisonKind::parse).ok_or_else(|| {
                    anyhow!("benches/results/wasm32.json: `{entry}` at `{level}` has an unknown `comparison_kind` {k}")
                })?),
            };
            rows.insert(
                entry.clone(),
                Counted {
                    ours: r.get("ours_instr_per_sample").and_then(Value::as_f64),
                    kind,
                },
            );
        }
        out.insert(level.clone(), rows);
    }
    Ok(out)
}

/// The position of `level` in the order of the plans (unknown levels last).
fn level_order(level: &str) -> usize {
    PLANS
        .iter()
        .position(|p| p.level == level)
        .unwrap_or(PLANS.len())
}

/// Hundredths of a number, as the tables publish them.
fn hundredths(x: f64) -> i64 {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "cycles and instructions per sample are small numbers"
    )]
    let h = (x * 100.0).round() as i64;
    h
}

/// `a / b` in the two decimals of the tables (`None`: `b` rounds to zero).
fn share(a: f64, b: f64) -> Option<f64> {
    let (a, b) = (hundredths(a), hundredths(b));
    #[allow(
        clippy::cast_precision_loss,
        reason = "hundredths of cycles per sample are small numbers"
    )]
    (b != 0).then(|| a as f64 / b as f64)
}

/// One SIMD entry and level above [`SIMD_SHARE`] of its fallback.
#[derive(Clone, Debug, PartialEq)]
pub struct Share {
    /// The SIMD entry.
    pub entry: String,
    /// The level.
    pub level: String,
    /// The share of the fallback on the reference model (at `simd128`: in instructions).
    pub reference: f64,
    /// The share on the counter-model (`None` at `simd128`, which has one count).
    pub counter: Option<f64>,
    /// The level runs the fallback (comparison `sdrk fallback`, both ratios 1.00).
    pub delegated: bool,
}

impl Share {
    /// `level share` for the report lines.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut s = match self.counter {
            Some(c) => format!("`{}` {:.2} / {:.2}", self.level, self.reference, c),
            None => format!("`{}` {:.2} (instructions)", self.level, self.reference),
        };
        if self.delegated {
            s.push_str(" (runs the fallback)");
        } else if accepted(&self.entry, &self.level) {
            s.push_str(" (");
            s.push_str(ACCEPTED);
            s.push(')');
        }
        s
    }
}

/// Every SIMD entry and level of `pairs` (cycles) and of `counts` (instructions, the levels
/// without cycles) that needs more than [`SIMD_SHARE`] of its fallback on a model, for the
/// entries that `kernel` accepts; by level order within an entry, entries in name order.
#[must_use]
pub fn simd_shares(pairs: &[Pair], counts: &Counts, kernel: &dyn Fn(&str) -> bool) -> Vec<Share> {
    let above = |x: Option<f64>| x.is_some_and(|x| x > SIMD_SHARE);
    let mut out = Vec::new();
    for p in pairs.iter().filter(|p| kernel(&p.entry)) {
        let r = share(p.simd.reference.ours, p.fallback.reference.ours);
        let c = share(p.simd.counter.ours, p.fallback.counter.ours);
        if above(r) || above(c) {
            out.push(Share {
                entry: p.entry.clone(),
                level: p.level.clone(),
                reference: r.unwrap_or(f64::NAN),
                counter: Some(c.unwrap_or(f64::NAN)),
                delegated: p.delegated(),
            });
        }
    }
    for (level, rows) in counts {
        for (entry, simd) in rows {
            if entry.ends_with(FALLBACK) || !kernel(entry) {
                continue;
            }
            let Some(fallback) = rows.get(&format!("{entry}{FALLBACK}")) else {
                continue;
            };
            let (Some(s), Some(f)) = (simd.ours, fallback.ours) else {
                continue;
            };
            let r = share(s, f);
            if above(r) {
                out.push(Share {
                    entry: entry.clone(),
                    level: level.clone(),
                    reference: r.unwrap_or(f64::NAN),
                    counter: None,
                    delegated: simd.kind == Some(ComparisonKind::OwnFallback)
                        && hundredths(s) == hundredths(f),
                });
            }
        }
    }
    out.sort_by(|a, b| {
        (a.entry.as_str(), level_order(&a.level)).cmp(&(b.entry.as_str(), level_order(&b.level)))
    });
    out
}

/// The kernel entries (as `kernel` accepts them) whose comparison is not a foreign library, with
/// their levels in plan order; `avx2-runtime` and records without a simulation are left out.
#[must_use]
pub fn without_library(
    results: &Results,
    counts: &Counts,
    kernel: &dyn Fn(&str) -> bool,
) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut add = |entry: &str, level: &str, kind: Option<ComparisonKind>| {
        if level != RUNTIME_LEVEL && kernel(entry) && kind != Some(ComparisonKind::Library) {
            out.entry(entry.to_string())
                .or_default()
                .push(level.to_string());
        }
    };
    for (level, rows) in results {
        for (entry, r) in rows {
            if let Some(r) = r {
                add(entry, level, r.comparison_kind);
            }
        }
    }
    for (level, rows) in counts {
        for (entry, c) in rows {
            if c.ours.is_some() {
                add(entry, level, c.kind);
            }
        }
    }
    for levels in out.values_mut() {
        levels.sort_by_key(|l| level_order(l));
        levels.dedup();
    }
    out
}

/// Prints both reports (the `check cycles` lines of O2 group 5).
pub fn print(shares: &[Share], unreferenced: &BTreeMap<String, Vec<String>>) {
    let entries = |v: &[Share]| {
        let mut e: Vec<&str> = v.iter().map(|s| s.entry.as_str()).collect();
        e.dedup();
        e.len()
    };
    let delegated = shares.iter().filter(|s| s.delegated).count();
    let explicit = shares
        .iter()
        .filter(|s| !s.delegated && accepted(&s.entry, &s.level))
        .count();
    println!(
        "check cycles (report, no failure): {} SIMD entries at {} levels need more than {SIMD_SHARE:.2} of the cycles per sample of their own fallback on a model ({delegated} of the levels run the fallback, {explicit} are marked `{ACCEPTED}` (D3019); at `simd128` instructions per sample); share of the fallback, reference / counter model (O2 group 5, issue #203 B)",
        entries(shares),
        shares.len(),
    );
    for group in shares.chunk_by(|a, b| a.entry == b.entry) {
        let parts: Vec<String> = group.iter().map(Share::describe).collect();
        let entry = group.first().map_or("", |s| s.entry.as_str());
        println!("  above {SIMD_SHARE:.2}: `{entry}`: {}", parts.join("; "));
    }
    println!(
        "check cycles (report, no failure): {} kernel entries at {} levels have no foreign reference, only our own code as the comparison, so no status (D32, D2764; `avx2-runtime` left out; O2 group 5, issue #203 A)",
        unreferenced.len(),
        unreferenced.values().map(Vec::len).sum::<usize>(),
    );
    for (entry, levels) in unreferenced {
        println!(
            "  no foreign reference: `{entry}`: {}",
            levels
                .iter()
                .map(|l| format!("`{l}`"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cycle_gate::{Model, Record, pairs};
    use std::path::PathBuf;

    fn model(ours: f64, comparison: f64) -> Model {
        Model {
            cpu: "skylake".to_string(),
            ours,
            comparison,
            ratio: ours / comparison,
        }
    }

    /// A record of ours `(ref, counter)` against a comparison of `kind`.
    fn record(
        ours: (f64, f64),
        kind: ComparisonKind,
        compared_with: &str,
        other: (f64, f64),
    ) -> Record {
        Record {
            file: PathBuf::from("x86_64.json"),
            compared_with: compared_with.to_string(),
            comparison_kind: Some(kind),
            judged_against: None,
            reference: model(ours.0, other.0),
            counter: model(ours.1, other.1),
            snr: None,
        }
    }

    #[allow(clippy::unnecessary_wraps, reason = "a simulated row of `Results`")]
    fn lib(ours: (f64, f64)) -> Option<Record> {
        Some(record(ours, ComparisonKind::Library, "volk", (1.0, 1.0)))
    }

    #[allow(clippy::unnecessary_wraps, reason = "a simulated row of `Results`")]
    fn own(ours: (f64, f64)) -> Option<Record> {
        Some(record(
            ours,
            ComparisonKind::OwnFallback,
            "sdrk fallback",
            ours,
        ))
    }

    fn results(level: &str, rows: Vec<(&str, Option<Record>)>) -> Results {
        let mut r = Results::new();
        r.insert(
            level.to_string(),
            rows.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        );
        r
    }

    fn any(_: &str) -> bool {
        true
    }

    #[test]
    fn a_simd_entry_above_the_share_on_one_model_is_reported_with_both_shares() {
        let r = results(
            "avx2",
            vec![
                // 0.80 / 0.80: at the share, not above it.
                ("a", lib((0.80, 1.60))),
                ("a_fallback", lib((1.00, 2.00))),
                // Above on the counter-model only.
                ("b", lib((0.50, 0.90))),
                ("b_fallback", lib((1.00, 1.00))),
                // Faster by more than a fifth on both.
                ("c", lib((0.40, 0.40))),
                ("c_fallback", lib((1.00, 1.00))),
                // Runs the fallback.
                ("d", own((1.00, 1.00))),
                ("d_fallback", lib((1.00, 1.00))),
                // No fallback entry.
                ("e", lib((9.0, 9.0))),
            ],
        );
        let s = simd_shares(&pairs(&r), &Counts::new(), &any);
        assert_eq!(
            s.iter().map(|s| s.entry.as_str()).collect::<Vec<_>>(),
            ["b", "d"]
        );
        assert_eq!((s[0].reference, s[0].counter), (0.5, Some(0.9)));
        assert!(!s[0].delegated);
        assert!(s[1].delegated);
        assert_eq!(s[0].describe(), "`avx2` 0.50 / 0.90");
        assert_eq!(s[1].describe(), "`avx2` 1.00 / 1.00 (runs the fallback)");
    }

    #[test]
    fn an_accepted_explicit_form_is_marked_and_a_delegation_is_not() {
        // D3019: an explicit path at about 1.00 of its fallback is reported and marked; the
        // same entry and level running the fallback is marked as such.
        let r = results(
            "avx2",
            vec![
                ("interleave", lib((1.00, 1.00))),
                ("interleave_fallback", lib((1.00, 1.00))),
                ("deinterleave", own((1.00, 1.00))),
                ("deinterleave_fallback", lib((1.00, 1.00))),
                ("add", lib((1.00, 1.00))),
                ("add_fallback", lib((1.00, 1.00))),
            ],
        );
        let s = simd_shares(&pairs(&r), &Counts::new(), &any);
        let d: Vec<String> = s.iter().map(Share::describe).collect();
        assert_eq!(
            d,
            [
                "`avx2` 1.00 / 1.00",
                "`avx2` 1.00 / 1.00 (runs the fallback)",
                "`avx2` 1.00 / 1.00 (accepted: explicit form, compiler-independent)",
            ]
        );
        assert!(accepted("stereo_matrix_cq15", "neon"));
        assert!(!accepted("stereo_matrix_cq15", "avx2"));
        // Every listed entry is a SIMD entry of the plan at a level it is measured at.
        for (entry, level) in EXPLICIT_FORM {
            let spec = crate::bench::KERNELS.iter().find(|k| k.name == entry);
            let plan = PLANS.iter().find(|p| p.level == level);
            assert!(
                spec.zip(plan).is_some_and(|(k, p)| k.family.measured_at(p)),
                "{entry} at {level}"
            );
        }
    }

    #[test]
    fn the_share_is_taken_in_the_two_decimals_of_the_tables() {
        // 0.804 / 1.00 rounds to 0.80, at the share; 0.806 rounds to 0.81, above it.
        let r = results(
            "neon",
            vec![
                ("a", lib((0.804, 0.1))),
                ("a_fallback", lib((1.0, 1.0))),
                ("b", lib((0.806, 0.1))),
                ("b_fallback", lib((1.0, 1.0))),
            ],
        );
        let s = simd_shares(&pairs(&r), &Counts::new(), &any);
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].entry.as_str(), s[0].reference), ("b", 0.81));
    }

    const WASM: &str = r#"{"levels": {"simd128": {"kernels": {
        "a": {"comparison_kind": "library", "ours_instr_per_sample": 9.0},
        "a_fallback": {"comparison_kind": "library", "ours_instr_per_sample": 10.0},
        "b": {"comparison_kind": "own_fallback", "ours_instr_per_sample": 5.0},
        "b_fallback": {"comparison_kind": "library", "ours_instr_per_sample": 10.0},
        "q": {"comparison_kind": "own_fallback", "ours_instr_per_sample": 4.0},
        "q_fallback": {"comparison_kind": "own_fallback", "ours_instr_per_sample": 4.0},
        "x": {"comparison_kind": "own_fallback", "note": "not counted"}
    }}}}"#;

    #[test]
    fn simd128_is_reported_in_instructions() {
        let c = counts(WASM).unwrap();
        let s = simd_shares(&[], &c, &any);
        assert_eq!(
            s.iter()
                .map(|s| (s.entry.as_str(), s.reference, s.delegated))
                .collect::<Vec<_>>(),
            [("a", 0.9, false), ("q", 1.0, true)]
        );
        assert_eq!(s[0].describe(), "`simd128` 0.90 (instructions)");
        assert!(
            counts(r#"{"levels": {"simd128": {"kernels": {"a": {"comparison_kind": "mine"}}}}}"#)
                .is_err()
        );
        assert!(counts("{}").is_err());
    }

    #[test]
    fn only_kernel_entries_without_a_library_are_listed_without_avx2_runtime() {
        let mut r = results(
            "sse2",
            vec![
                ("k", own((1.0, 1.0))),
                ("k_fallback", lib((1.0, 1.0))),
                ("l", lib((1.0, 1.0))),
                ("m_block", own((1.0, 1.0))),
                ("failed", None),
            ],
        );
        r.extend(results("avx2-runtime", vec![("k", own((1.0, 1.0)))]));
        r.extend(results("neon", vec![("k", own((1.0, 1.0)))]));
        r.extend(results(
            "avx2",
            vec![(
                "k",
                Some(record(
                    (1.0, 1.0),
                    ComparisonKind::OwnBuild,
                    "ours at avx2",
                    (1.0, 1.0),
                )),
            )],
        ));
        let c = counts(WASM).unwrap();
        let kernel = |e: &str| !e.ends_with("_block");
        let u = without_library(&r, &c, &kernel);
        let got: Vec<(&str, Vec<&str>)> = u
            .iter()
            .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
            .collect();
        assert_eq!(
            got,
            [
                ("b", vec!["simd128"]),
                ("k", vec!["sse2", "avx2", "neon"]),
                ("q", vec!["simd128"]),
                ("q_fallback", vec!["simd128"]),
            ]
        );
    }

    #[test]
    fn the_committed_results_give_both_reports() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let results = crate::cycle_gate::load(&crate::cycle_results::dir(root)).unwrap();
        let c = counts(&std::fs::read_to_string(root.join("benches/results/wasm32.json")).unwrap())
            .unwrap();
        let kernel = |e: &str| crate::bench::is_kernel_entry(e);
        let u = without_library(&results, &c, &kernel);
        assert!(!u.is_empty());
        assert!(u.keys().all(|k| crate::bench::is_kernel_entry(k)), "{u:?}");
        assert!(!u.contains_key("halfband_ref_fir_decimate"));
        let s = simd_shares(&pairs(&results), &c, &kernel);
        assert!(
            s.iter()
                .all(|s| s.reference > SIMD_SHARE || s.counter.is_some_and(|c| c > SIMD_SHARE))
        );
    }
}
