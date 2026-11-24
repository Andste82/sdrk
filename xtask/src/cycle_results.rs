//! The committed cycle results `benches/results/cycles/<platform>.json` (M13.1 §2, D2704).
//!
//! `cargo measure` writes them with `--merge` (the records of the measured entries replace or
//! add those of the file, everything else stays as it is) or `--all` (every entry on every
//! level of the platform: the file is written anew). One record per entry and level with a core
//! model: the comparison, the samples of the counted calls and, on the reference model and the
//! counter-model of the level, the cycles per sample of ours and of the comparison (the library
//! reference of D32, `sdrk fallback` where there is none, our `avx2` build for `avx2-runtime`)
//! with the ratio, the efficiency and the coverage, in the judged cycle mode (`whole-loop`); on
//! the levels with a branch simulation (D1611) the branches and mispredictions per sample of
//! both sides and the cost `cycles + mispredictions × penalty` on both models (D1612). An entry
//! that was not simulated keeps its note. The file names the tools that made the records (the
//! `llvm-mca` build, checked for the corrections of D942, the profiler of each level, Valgrind,
//! the Rust compiler); a merge with other tools is refused, because the records of one file
//! must be comparable.
//!
//! `benches/cycle_tables.py` writes the cycle tables of `docs/performance.md` from these files.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, anyhow, bail};
use serde_json::{Map, Value, json};

/// The directory of the committed cycle results.
#[must_use]
pub fn dir(root: &Path) -> PathBuf {
    root.join("benches").join("results").join("cycles")
}

/// The judged cycle mode of the committed records (D33).
pub const MODE: crate::mca::CycleMode = crate::mca::CycleMode::WholeLoop;

/// The method of the records, as written into every file.
fn method() -> Value {
    json!({
        "command": "cargo measure <kernel> --merge, or cargo measure --all (the cycles and, where they are simulated, the branches)",
        "cycles": "llvm-mca on the reference model and the counter-model of the level, the execution counts of the counted calls from the profiler of the level (block_counts), per sample of the counted calls (D33, D68)",
        "cycle_mode": MODE.name(),
        "branches": "valgrind cachegrind --branch-sim=yes, conditional and indirect mispredictions (Bcm + Bim) of the counted calls, per sample (D1611)",
        "cost": "cycles + mispredictions x penalty of the core model (D1611, D1612)",
        "note": "only the ratio of ours to the comparison on one model means something; the cycles are no hardware figures",
    })
}

/// The member `key` of a JSON object, `null` if it is missing.
fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap_or(&Value::Null)
}

/// One side of a model in the record: the cycles per sample, the efficiency and the coverage.
fn side(simulation: &Value) -> Value {
    json!({
        "cycles_per_sample": field(simulation, "cycles_per_sample"),
        "efficiency": field(simulation, "efficiency"),
        "coverage": field(simulation, "coverage"),
    })
}

/// One model of the record from the model of a `cargo measure` entry.
fn model(m: &Value) -> Value {
    json!({
        "cpu": field(field(m, "ours"), "cpu"),
        "ours": side(field(m, "ours")),
        "comparison": side(field(m, "other")),
        "ratio": field(m, "ratio"),
    })
}

/// The record of one entry from its cycles (the entry of the `cycles` section of a
/// `cargo measure` document) and its branches (the entry of the `branches` section, if the
/// level has a branch simulation).
#[must_use]
pub fn record(cycles: &Value, branches: Option<&Value>) -> Value {
    if cycles.get("reference_model").is_none() {
        return json!({ "note": field(cycles, "note") });
    }
    let mut r = Map::new();
    r.insert(
        "compared_with".into(),
        field(cycles, "compared_with").clone(),
    );
    r.insert("samples".into(), field(cycles, "samples").clone());
    r.insert("cycle_mode".into(), field(cycles, "cycle_mode").clone());
    for key in ["reference_model", "counter_model"] {
        r.insert(key.into(), model(field(cycles, key)));
    }
    if let Some(b) = branches {
        if b.get("ours").is_some() {
            let per = |s: &Value| {
                json!({
                    "branches_per_sample": field(s, "branches_per_sample"),
                    "mispredictions_per_sample": field(s, "mispredictions_per_sample"),
                })
            };
            r.insert(
                "branches".into(),
                json!({ "ours": per(field(b, "ours")), "comparison": per(field(b, "other")) }),
            );
            let mut cost = Map::new();
            if let Some(models) = b.get("cost").and_then(Value::as_object) {
                for (key, c) in models {
                    cost.insert(
                        key.clone(),
                        json!({
                            "cpu": field(c, "cpu"),
                            "penalty": field(c, "penalty"),
                            "ours": field(field(c, "ours"), "cost_per_sample"),
                            "comparison": field(field(c, "other"), "cost_per_sample"),
                        }),
                    );
                }
            }
            r.insert("cost".into(), Value::Object(cost));
        } else {
            r.insert(
                "branches".into(),
                json!({ "note": field(b, "note").clone() }),
            );
        }
    }
    Value::Object(r)
}

/// The header of a level: the models, the judged mode, the profiler and the branch method.
fn level_header(level: &str, block_counts: &Value) -> anyhow::Result<Map<String, Value>> {
    let models =
        crate::mca::models(level).ok_or_else(|| anyhow!("the level {level} has no core model"))?;
    let mut h = Map::new();
    h.insert("reference_model".into(), json!(models.reference));
    h.insert("counter_model".into(), json!(models.counter));
    h.insert("cycle_mode".into(), json!(MODE.name()));
    h.insert("block_counts".into(), block_counts.clone());
    h.insert(
        "branches".into(),
        json!(if crate::branches::simulated_level(level) {
            "valgrind cachegrind --branch-sim=yes on the x86-64 build of the level"
        } else if level == "avx2-runtime" {
            "not simulated: judged against our own avx2 build (D2702)"
        } else if level == "avx512" {
            "not simulated: Valgrind runs no AVX-512 (I44-16)"
        } else {
            "not simulated (D1611)"
        }),
    );
    Ok(h)
}

/// The levels of one platform of a `cargo measure` document as records: per simulated level
/// (levels that failed as a whole are left out, so that a merge keeps their records) the header
/// and the record of every entry.
///
/// # Errors
/// A level without a core model.
pub fn levels_of(cycles: &Value, branches: Option<&Value>) -> anyhow::Result<Map<String, Value>> {
    let mut out = Map::new();
    let Some(levels) = cycles.as_object() else {
        return Ok(out);
    };
    for (level, entries) in levels {
        let Some(entries) = entries.as_object() else {
            continue;
        };
        if entries.contains_key("error") && !entries.values().any(Value::is_object) {
            continue;
        }
        let block_counts = entries
            .values()
            .find_map(|e| e.get("block_counts"))
            .cloned()
            .unwrap_or(Value::Null);
        let mut level_doc = level_header(level, &block_counts)?;
        let level_branches = branches.and_then(|b| b.get(level.as_str()));
        let mut records = Map::new();
        for (name, e) in entries {
            records.insert(
                name.clone(),
                record(e, level_branches.and_then(|b| b.get(name.as_str()))),
            );
        }
        level_doc.insert("entries".into(), Value::Object(records));
        out.insert(level.clone(), Value::Object(level_doc));
    }
    Ok(out)
}

/// Merges `measured` (from [`levels_of`]) into the results document `doc` of `platform`
/// (`None`: no file yet) made with `tools`. With `replace` the document is written anew from
/// `measured` alone (`cargo measure --all`); otherwise the records of the measured entries
/// replace or add those of the document and every other record stays as it is. Returns the new
/// document and the number of records replaced and added.
///
/// # Errors
/// The document was made with other tools, or a measured level has another header (models,
/// mode, profiler) than the document: its records would not be comparable, so the whole
/// platform is to be measured again with `--all`.
pub fn merge(
    doc: Option<Value>,
    platform: &str,
    tools: &Value,
    measured: &Map<String, Value>,
    replace: bool,
) -> anyhow::Result<(Value, usize, usize)> {
    let fresh = || {
        json!({
            "platform": platform,
            "method": method(),
            "tools": tools,
            "levels": {},
        })
    };
    let mut doc = match doc {
        Some(d) if !replace => d,
        _ => fresh(),
    };
    if field(&doc, "tools") != tools {
        bail!(
            "the records of {platform} were made with other tools ({}), this run has {}; measure every entry again with `cargo measure --all --platform {platform}`",
            field(&doc, "tools"),
            tools
        );
    }
    if let Some(o) = doc.as_object_mut() {
        o.insert("method".into(), method());
    }
    let levels = doc
        .get_mut("levels")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow!("the results of {platform} have no `levels`"))?;
    let (mut replaced, mut added) = (0, 0);
    for (level, new) in measured {
        let slot = levels.entry(level.clone()).or_insert_with(|| {
            let mut empty = new.clone();
            if let Some(o) = empty.as_object_mut() {
                o.insert("entries".into(), json!({}));
            }
            empty
        });
        for key in [
            "reference_model",
            "counter_model",
            "cycle_mode",
            "block_counts",
        ] {
            if field(slot, key) != field(new, key) {
                bail!(
                    "{platform} {level}: `{key}` is {} in the results and {} in this run; measure every entry again with `cargo measure --all --platform {platform}`",
                    field(slot, key),
                    field(new, key)
                );
            }
        }
        if let Some(o) = slot.as_object_mut() {
            o.insert("branches".into(), field(new, "branches").clone());
        }
        let entries = slot
            .get_mut("entries")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| anyhow!("{platform} {level} has no `entries`"))?;
        if let Some(new_entries) = new.get("entries").and_then(Value::as_object) {
            for (name, r) in new_entries {
                if entries.insert(name.clone(), r.clone()).is_some() {
                    replaced += 1;
                } else {
                    added += 1;
                }
            }
        }
    }
    Ok((doc, replaced, added))
}

/// The first line of the output of `program args` that contains `needle`, trimmed; `None` if it
/// cannot run.
fn version_line(program: &Path, args: &[&str], needle: &str, root: &Path) -> Option<String> {
    let out = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    text.lines()
        .find(|l| l.contains(needle))
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Four independent stores per iteration: one per cycle on `znver2` with the corrections of
/// D942 (about 400 cycles for 100 iterations), three per cycle without them (about 140).
const STORE_PROBE: &str =
    "movq %rax, (%rdi)\nmovq %rax, 8(%rdi)\nmovq %rax, 16(%rdi)\nmovq %rax, 24(%rdi)\n";

/// Whether the `llvm-mca` in use has the corrections of D942, from the stores of `znver2`.
///
/// # Errors
/// `llvm-mca` cannot run or prints no total.
pub fn llvm_mca_corrected() -> anyhow::Result<bool> {
    let mca = crate::mca::llvm_mca();
    let mut child = Command::new(&mca)
        .args([
            "-mtriple=x86_64-unknown-linux-gnu",
            "-mcpu=znver2",
            "-iterations=100",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("cannot run {} (set SDRK_LLVM_MCA)", mca.display()))?;
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write as _;
        stdin.write_all(STORE_PROBE.as_bytes())?;
    }
    let out = child.wait_with_output()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let total: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("Total Cycles:"))
        .and_then(|v| v.trim().parse().ok())
        .ok_or_else(|| anyhow!("{} printed no total for the store probe", mca.display()))?;
    Ok(total >= 300)
}

/// Fails unless the `llvm-mca` in use has the corrections of D942: the committed results are
/// made with the corrected build only.
///
/// # Errors
/// `llvm-mca` cannot run or lacks the corrections.
pub fn require_corrected_llvm_mca() -> anyhow::Result<()> {
    if !llvm_mca_corrected()? {
        bail!(
            "{} lacks the corrections of the scheduling models (D942, assets/llvm-mca-23.1.1-sdrk-sched.patch); the committed results are made with the corrected build (SDRK_LLVM_MCA)",
            crate::mca::llvm_mca().display()
        );
    }
    Ok(())
}

/// The tools of the records of `platform`: the `llvm-mca` build (it must have the corrections of
/// D942), the Rust compiler of the work tree, the compilers of the C references (from their
/// manifest, `benches/target/refs/<platform>/refs.json`) and the profilers of the levels of the
/// platform (Intel SDE, Valgrind, QEMU).
///
/// # Errors
/// `llvm-mca` cannot run or lacks the corrections of D942.
#[allow(
    clippy::large_stack_frames,
    reason = "std::process::Command and the plan table are large; xtask is a host tool"
)]
pub fn tools(root: &Path, platform: &str) -> anyhow::Result<Value> {
    require_corrected_llvm_mca()?;
    let mca = crate::mca::llvm_mca();
    let missing = || "not found".to_string();
    let mut t = Map::new();
    t.insert(
        "llvm-mca".into(),
        json!(format!(
            "{}, with the corrections of D942",
            version_line(&mca, &["--version"], "LLVM version", root).unwrap_or_else(missing)
        )),
    );
    t.insert(
        "rustc".into(),
        json!(version_line(Path::new("rustc"), &["-V"], "rustc", root).unwrap_or_else(missing)),
    );
    let sde = std::env::var_os("SDRK_SDE").map_or_else(|| PathBuf::from("sde64"), PathBuf::from);
    let valgrind = Path::new("valgrind");
    let qemu = std::env::var_os("SDRK_QEMU_AARCH64")
        .map_or_else(|| PathBuf::from("qemu-aarch64"), PathBuf::from);
    match platform {
        "x86_64" => {
            t.insert(
                "sde".into(),
                json!(version_line(&sde, &["-version"], "Version", root).unwrap_or_else(missing)),
            );
            t.insert(
                "valgrind".into(),
                json!(
                    version_line(valgrind, &["--version"], "valgrind", root)
                        .unwrap_or_else(missing)
                ),
            );
        }
        "x86_64-avx512" => {
            t.insert(
                "sde".into(),
                json!(version_line(&sde, &["-version"], "Version", root).unwrap_or_else(missing)),
            );
        }
        "aarch64" => {
            t.insert(
                "qemu".into(),
                json!(version_line(&qemu, &["--version"], "version", root).unwrap_or_else(missing)),
            );
        }
        _ => {}
    }
    let refs = crate::bench::PLANS
        .iter()
        .find(|p| p.platform == platform)
        .map_or(platform, |p| p.refs);
    let manifest: Value = std::fs::read_to_string(
        root.join("benches")
            .join("target")
            .join("refs")
            .join(refs)
            .join("refs.json"),
    )
    .ok()
    .and_then(|text| serde_json::from_str(&text).ok())
    .unwrap_or(Value::Null);
    for key in ["cc", "cxx", "kfr_cxx"] {
        if let Some(v) = manifest
            .get(key)
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
        {
            t.insert(format!("references_{key}"), json!(v));
        }
    }
    Ok(Value::Object(t))
}

/// Writes the records of `measured` for `platform` into its results file (see [`merge`]).
///
/// # Errors
/// The file cannot be read or written, or [`merge`] refuses.
pub fn write(
    root: &Path,
    platform: &str,
    tools: &Value,
    measured: &Map<String, Value>,
    replace: bool,
) -> anyhow::Result<PathBuf> {
    let path = dir(root).join(format!("{platform}.json"));
    let old = match std::fs::read_to_string(&path) {
        Ok(text) => Some(
            serde_json::from_str(&text)
                .with_context(|| format!("cannot parse {}", path.display()))?,
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
    };
    let (doc, replaced, added) = merge(old, platform, tools, measured, replace)?;
    std::fs::create_dir_all(dir(root))?;
    let mut text = serde_json::to_string_pretty(&doc)?;
    text.push('\n');
    std::fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))?;
    println!(
        "{} {}: {replaced} records replaced, {added} added",
        if replace { "wrote" } else { "merged into" },
        path.display()
    );
    Ok(path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn simulated(ours: f64, other: f64) -> Value {
        let m = |cpu: &str| {
            json!({
                "ours": { "cpu": cpu, "cycles_per_sample": ours, "efficiency": 0.5, "coverage": 1.0, "blocks": 3 },
                "other": { "cpu": cpu, "cycles_per_sample": other, "efficiency": 0.4, "coverage": 0.99 },
                "ratio": ours / other,
            })
        };
        json!({
            "compared_with": "a library",
            "samples": 4096,
            "cycle_mode": "whole-loop",
            "reference_model": m("skylake"),
            "counter_model": m("znver2"),
            "counter_vs_reference": 1.0,
            "other_cycle_mode": {},
            "block_counts": "Intel SDE -dcfg",
        })
    }

    fn branches() -> Value {
        json!({
            "ours": { "branches_per_sample": 0.5, "mispredictions_per_sample": 0.25 },
            "other": { "branches_per_sample": 0.75, "mispredictions_per_sample": 0.0 },
            "cost": {
                "reference_model": { "cpu": "skylake", "penalty": 20.0,
                    "ours": { "cycles_per_sample": 1.0, "cost_per_sample": 6.0 },
                    "other": { "cycles_per_sample": 2.0, "cost_per_sample": 2.0 } },
            },
        })
    }

    fn measured(entries: &[(&str, f64)]) -> Map<String, Value> {
        let mut cycles = Map::new();
        let mut b = Map::new();
        for (name, ours) in entries {
            cycles.insert((*name).to_string(), simulated(*ours, 2.0));
            b.insert((*name).to_string(), branches());
        }
        cycles.insert(
            "not_here".into(),
            json!({ "note": "not simulated: not measured at the level sse2", "block_counts": "Intel SDE -dcfg" }),
        );
        levels_of(&json!({ "sse2": cycles }), Some(&json!({ "sse2": b }))).unwrap()
    }

    #[test]
    fn a_record_has_both_models_the_branches_and_the_cost() {
        let r = record(&simulated(1.0, 2.0), Some(&branches()));
        assert_eq!(r["reference_model"]["cpu"], "skylake");
        assert_eq!(r["reference_model"]["ours"]["cycles_per_sample"], 1.0);
        assert_eq!(r["counter_model"]["comparison"]["cycles_per_sample"], 2.0);
        assert_eq!(r["counter_model"]["ratio"], 0.5);
        assert_eq!(r["branches"]["ours"]["mispredictions_per_sample"], 0.25);
        assert_eq!(r["cost"]["reference_model"]["penalty"], 20.0);
        assert_eq!(r["cost"]["reference_model"]["ours"], 6.0);
        assert!(r["reference_model"]["ours"].get("blocks").is_none());
        assert_eq!(
            record(&json!({ "note": "not simulated: x" }), None),
            json!({ "note": "not simulated: x" })
        );
        let r = record(
            &simulated(1.0, 2.0),
            Some(&json!({ "note": "not counted: y" })),
        );
        assert_eq!(r["branches"]["note"], "not counted: y");
        assert!(r.get("cost").is_none());
    }

    #[test]
    fn merge_changes_only_the_given_entries() {
        let tools = json!({ "llvm-mca": "LLVM version 23.1.1, with the corrections of D942" });
        let (doc, r, a) = merge(
            None,
            "x86_64",
            &tools,
            &measured(&[("a", 1.0), ("b", 1.5)]),
            false,
        )
        .unwrap();
        assert_eq!((r, a), (0, 3));
        let before = doc.clone();
        let (doc, r, a) = merge(
            Some(doc),
            "x86_64",
            &tools,
            &measured(&[("b", 1.25), ("c", 0.5)]),
            false,
        )
        .unwrap();
        assert_eq!((r, a), (2, 1));
        let e = &doc["levels"]["sse2"]["entries"];
        assert_eq!(e["a"], before["levels"]["sse2"]["entries"]["a"]);
        assert_eq!(e["b"]["reference_model"]["ours"]["cycles_per_sample"], 1.25);
        assert_eq!(e["c"]["counter_model"]["ours"]["cycles_per_sample"], 0.5);
        assert_eq!(
            e["not_here"]["note"],
            "not simulated: not measured at the level sse2"
        );
        assert_eq!(doc["levels"]["sse2"]["reference_model"], "skylake");
        assert_eq!(doc["levels"]["sse2"]["counter_model"], "znver2");
        // Idempotent.
        let (again, _, _) = merge(
            Some(doc.clone()),
            "x86_64",
            &tools,
            &measured(&[("b", 1.25), ("c", 0.5)]),
            false,
        )
        .unwrap();
        assert_eq!(again, doc);
        // A full run drops what it did not measure.
        let (full, _, _) =
            merge(Some(doc), "x86_64", &tools, &measured(&[("c", 0.5)]), true).unwrap();
        assert!(full["levels"]["sse2"]["entries"].get("a").is_none());
    }

    #[test]
    fn merge_refuses_other_tools_and_another_header() {
        let tools = json!({ "llvm-mca": "23.1.1" });
        let (doc, _, _) = merge(None, "x86_64", &tools, &measured(&[("a", 1.0)]), false).unwrap();
        let other = json!({ "llvm-mca": "24.0.0" });
        let err = merge(
            Some(doc.clone()),
            "x86_64",
            &other,
            &measured(&[("a", 1.0)]),
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("other tools"), "{err}");
        assert!(
            merge(
                Some(doc.clone()),
                "x86_64",
                &other,
                &measured(&[("a", 1.0)]),
                true
            )
            .is_ok()
        );
        let mut changed = measured(&[("a", 1.0)]);
        changed["sse2"]["block_counts"] = json!("callgrind");
        let err = merge(Some(doc), "x86_64", &tools, &changed, false).unwrap_err();
        assert!(err.to_string().contains("block_counts"), "{err}");
    }

    #[test]
    fn a_merge_through_the_file_keeps_the_other_records_bit_for_bit() {
        // Without serde_json's `float_roundtrip` this value comes back one ULP off.
        let x = 0.992_041_588_177_742_7_f64;
        let root = tempfile::tempdir().unwrap();
        let tools = json!({ "llvm-mca": "23.1.1" });
        write(root.path(), "x86_64", &tools, &measured(&[("a", x)]), false).unwrap();
        let before = std::fs::read_to_string(dir(root.path()).join("x86_64.json")).unwrap();
        write(
            root.path(),
            "x86_64",
            &tools,
            &measured(&[("b", 1.0)]),
            false,
        )
        .unwrap();
        let doc: Value = serde_json::from_str(
            &std::fs::read_to_string(dir(root.path()).join("x86_64.json")).unwrap(),
        )
        .unwrap();
        let a = &doc["levels"]["sse2"]["entries"]["a"]["reference_model"]["ours"];
        assert_eq!(
            a["cycles_per_sample"].as_f64().unwrap().to_bits(),
            x.to_bits()
        );
        let old: Value = serde_json::from_str(&before).unwrap();
        assert_eq!(
            old["levels"]["sse2"]["entries"]["a"],
            doc["levels"]["sse2"]["entries"]["a"]
        );
    }

    #[test]
    fn a_level_that_failed_as_a_whole_is_left_out() {
        let l = levels_of(
            &json!({ "sse2": { "error": "no SDE" }, "avx2": { "k": simulated(1.0, 1.0) } }),
            None,
        )
        .unwrap();
        assert!(l.get("sse2").is_none());
        assert_eq!(
            l["avx2"]["branches"],
            "valgrind cachegrind --branch-sim=yes on the x86-64 build of the level"
        );
        assert!(l["avx2"]["entries"]["k"].get("branches").is_none());
    }
}
