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
        "address_space": if crate::bench::fixed_layout() {
            "fixed: the profiled processes run under setarch -R (I13.1-1)"
        } else {
            "randomized: setarch -R is not permitted here (I13.1-1)"
        },
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
/// `method` is that of the measured records (see [`method`]).
///
/// # Errors
/// The document was made with other tools or another method (for example the other address
/// space, D2710), or a measured level has another header (models, mode, profiler) than the
/// document: its records would not be comparable, so the whole platform is to be measured
/// again with `--all`.
pub fn merge(
    doc: Option<Value>,
    platform: &str,
    tools: &Value,
    method: &Value,
    measured: &Map<String, Value>,
    replace: bool,
) -> anyhow::Result<(Value, usize, usize)> {
    let fresh = || {
        json!({
            "platform": platform,
            "method": method,
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
    if field(&doc, "method") != method {
        bail!(
            "the records of {platform} were made with another method ({}), this run has {}; measure every entry again with `cargo measure --all --platform {platform}`",
            field(&doc, "method"),
            method
        );
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
    dir: &Path,
    platform: &str,
    tools: &Value,
    measured: &Map<String, Value>,
    replace: bool,
) -> anyhow::Result<PathBuf> {
    let path = dir.join(format!("{platform}.json"));
    let old = match std::fs::read_to_string(&path) {
        Ok(text) => Some(
            serde_json::from_str(&text)
                .with_context(|| format!("cannot parse {}", path.display()))?,
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
    };
    let (doc, replaced, added) = merge(old, platform, tools, &method(), measured, replace)?;
    std::fs::create_dir_all(dir)?;
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

/// The results documents of the `*.json` files of `dir`, sorted by file name.
///
/// # Errors
/// The directory or a file cannot be read or parsed.
#[allow(
    clippy::large_stack_frames,
    reason = "the Windows file system calls are large; xtask is a host tool"
)]
fn documents(dir: &Path) -> anyhow::Result<Vec<(PathBuf, Value)>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|f| {
            let text = std::fs::read_to_string(&f)
                .with_context(|| format!("cannot read {}", f.display()))?;
            let doc = serde_json::from_str(&text)
                .with_context(|| format!("cannot parse {}", f.display()))?;
            Ok((f, doc))
        })
        .collect()
}

/// `cargo xtask results merge --into DIR SRC...`: merges the results files of the partitions
/// of a measurement (each `SRC` holds `<platform>.json` files written by `cargo measure
/// --merge --results SRC`) into `DIR`, as `--merge` does: the records of a partition replace
/// or add those of `DIR`. The partitions must have the same tools and method (one
/// environment).
///
/// # Errors
/// A source cannot be read, no source holds a results file, a file cannot be parsed or
/// written, or [`merge`] refuses.
pub fn merge_dirs(into: &Path, sources: &[PathBuf]) -> anyhow::Result<()> {
    let sources = sources
        .iter()
        .map(|src| documents(src))
        .collect::<anyhow::Result<Vec<_>>>()?;
    if sources.iter().all(Vec::is_empty) {
        bail!("no source holds a results file (`<platform>.json`): nothing was measured");
    }
    std::fs::create_dir_all(into)?;
    for src in sources {
        for (file, doc) in src {
            let platform = doc
                .get("platform")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("{} has no `platform`", file.display()))?;
            let levels = doc
                .get("levels")
                .and_then(Value::as_object)
                .ok_or_else(|| anyhow!("{} has no `levels`", file.display()))?;
            let path = into.join(format!("{platform}.json"));
            let old: Option<Value> = match std::fs::read_to_string(&path) {
                Ok(text) => Some(serde_json::from_str(&text)?),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
            };
            let merged = merge(
                old,
                platform,
                field(&doc, "tools"),
                field(&doc, "method"),
                levels,
                false,
            );
            let (merged, replaced, added) =
                merged.with_context(|| format!("cannot merge {}", file.display()))?;
            let mut text = serde_json::to_string_pretty(&merged)?;
            text.push('\n');
            std::fs::write(&path, text)
                .with_context(|| format!("cannot write {}", path.display()))?;
            println!(
                "merged {} into {}: {replaced} records replaced, {added} added",
                file.display(),
                path.display()
            );
        }
    }
    Ok(())
}

/// The cycles per sample of a record that a comparison of two results looks at: ours and the
/// comparison on both models.
const COMPARED: [(&str, &str); 4] = [
    ("reference_model", "ours"),
    ("reference_model", "comparison"),
    ("counter_model", "ours"),
    ("counter_model", "comparison"),
];

/// The relative difference of `y` from `x`.
fn relative(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        if y == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (y / x - 1.0).abs()
    }
}

/// The differences of the records of `fresh` from those of `base` (the committed results).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Differences {
    /// One line each: a simulated record whose cycles per sample of ours or of the comparison
    /// differ by more than the tolerance (relative) on either model, a record that `base` does
    /// not have, a record that was simulated in `base` and not in `fresh`, and, for a complete
    /// run, a record of `base` that `fresh` does not have.
    pub drift: Vec<String>,
    /// For information (the gate judges the cycles, D2708): a record whose cost (cycles +
    /// mispredictions x penalty, D1611) differs by more than the tolerance on a model.
    pub cost: Vec<String>,
}

/// The differences of the records of `fresh` from those of `base`. Records of `base` that
/// `fresh` does not have are compared only when `complete` (every entry at every level was
/// measured): then each of them is a difference, as a lost partition would otherwise pass.
#[must_use]
#[allow(
    clippy::large_stack_frames,
    reason = "the serde_json values are built in place; xtask is a host tool"
)]
pub fn differences(base: &[Value], fresh: &[Value], tolerance: f64, complete: bool) -> Differences {
    let record = |docs: &[Value], platform: &str, level: &str, entry: &str| -> Option<Value> {
        docs.iter()
            .find(|d| field(d, "platform").as_str() == Some(platform))
            .and_then(|d| d.get("levels")?.get(level)?.get("entries")?.get(entry))
            .cloned()
    };
    let mut d = Differences::default();
    if complete {
        for doc in base {
            let platform = field(doc, "platform").as_str().unwrap_or_default();
            for (level, l) in field(doc, "levels").as_object().into_iter().flatten() {
                for entry in field(l, "entries")
                    .as_object()
                    .into_iter()
                    .flatten()
                    .map(|e| e.0)
                {
                    if record(fresh, platform, level, entry).is_none() {
                        d.drift.push(format!(
                            "{platform} {level} {entry}: in the committed results, missing in this complete run"
                        ));
                    }
                }
            }
        }
    }
    let out = &mut d.drift;
    for doc in fresh {
        let platform = field(doc, "platform").as_str().unwrap_or_default();
        for (level, l) in field(doc, "levels").as_object().into_iter().flatten() {
            for (entry, r) in field(l, "entries").as_object().into_iter().flatten() {
                let name = format!("{platform} {level} {entry}");
                let Some(b) = record(base, platform, level, entry) else {
                    out.push(format!("{name}: not in the committed results"));
                    continue;
                };
                match (b.get("reference_model"), r.get("reference_model")) {
                    (None, None) => continue,
                    (Some(_), None) => {
                        out.push(format!(
                            "{name}: simulated in the committed results, now {}",
                            field(r, "note")
                        ));
                        continue;
                    }
                    (None, Some(_)) => {
                        out.push(format!(
                            "{name}: not simulated in the committed results ({}), now simulated",
                            field(&b, "note")
                        ));
                        continue;
                    }
                    (Some(_), Some(_)) => {}
                }
                let mut drift = Vec::new();
                for (m, side) in COMPARED {
                    let get = |v: &Value| v.get(m)?.get(side)?.get("cycles_per_sample")?.as_f64();
                    let (Some(x), Some(y)) = (get(&b), get(r)) else {
                        drift.push(format!("{m} {side}: missing"));
                        continue;
                    };
                    if relative(x, y) > tolerance {
                        drift.push(format!(
                            "{} {side} {x:.4} -> {y:.4} ({:+.2} %)",
                            field(&b, m).get("cpu").and_then(Value::as_str).unwrap_or(m),
                            (y / x - 1.0) * 100.0
                        ));
                    }
                }
                if !drift.is_empty() {
                    out.push(format!("{name}: {}", drift.join(", ")));
                }
                let mut cost = Vec::new();
                for (m, c) in field(r, "cost").as_object().into_iter().flatten() {
                    for side in ["ours", "comparison"] {
                        let get = |v: &Value| v.get(side)?.as_f64();
                        if let (Some(x), Some(y)) = (get(field(field(&b, "cost"), m)), get(c))
                            && relative(x, y) > tolerance
                        {
                            cost.push(format!(
                                "{} {side} {x:.4} -> {y:.4} ({:+.2} %)",
                                c.get("cpu").and_then(Value::as_str).unwrap_or(m),
                                (y / x - 1.0) * 100.0
                            ));
                        }
                    }
                }
                if !cost.is_empty() {
                    d.cost.push(format!("{name}: cost {}", cost.join(", ")));
                }
            }
        }
    }
    d
}

/// The reproduction check of the corrected `llvm-mca` (D942): `costas` at `neon`, ours, on the
/// reference and the counter-model, in the two decimals of the tables.
pub const REPRODUCTION: (&str, &str, &str, [f64; 2]) =
    ("aarch64", "neon", "costas", [86.13, 54.09]);

/// Checks the reproduction record in the results of `dir`.
///
/// # Errors
/// The record is missing or its cycles differ from [`REPRODUCTION`].
pub fn check_reproduction(dir: &Path) -> anyhow::Result<()> {
    let (platform, level, entry, want) = REPRODUCTION;
    let docs: Vec<Value> = documents(dir)?.into_iter().map(|(_, d)| d).collect();
    let r = docs
        .iter()
        .find(|d| field(d, "platform").as_str() == Some(platform))
        .and_then(|d| d.get("levels")?.get(level)?.get("entries")?.get(entry))
        .ok_or_else(|| anyhow!("{} has no record `{entry}` at `{level}`", dir.display()))?;
    let got: Vec<String> = ["reference_model", "counter_model"]
        .iter()
        .map(|m| {
            r.get(*m)
                .and_then(|v| v.get("ours")?.get("cycles_per_sample")?.as_f64())
                .map_or_else(|| "-".to_string(), |x| format!("{x:.2}"))
        })
        .collect();
    let want: Vec<String> = want.iter().map(|x| format!("{x:.2}")).collect();
    if got != want {
        bail!(
            "the reproduction check failed: `{entry}` at `{level}` gives {} cycles per sample, the corrected llvm-mca gives {} (D942)",
            got.join(" / "),
            want.join(" / ")
        );
    }
    println!(
        "reproduction check: `{entry}` at `{level}` {} (D942)",
        got.join(" / ")
    );
    Ok(())
}

/// `cargo xtask results <merge --into DIR SRC... | compare BASE FRESH [--tolerance T]
/// [--complete] | repro DIR>`.
///
/// # Errors
/// Wrong arguments, or a file that cannot be read, parsed or written.
pub fn run(args: &[&str]) -> anyhow::Result<bool> {
    const USAGE: &str = "usage: cargo xtask results <merge --into DIR SRC... | compare BASE FRESH [--tolerance T] [--complete] | repro DIR>";
    match args {
        ["merge", "--into", into, sources @ ..] if !sources.is_empty() => {
            let sources: Vec<PathBuf> = sources.iter().map(PathBuf::from).collect();
            merge_dirs(Path::new(into), &sources)?;
            Ok(true)
        }
        ["compare", base, fresh, rest @ ..] => {
            let (mut tolerance, mut complete, mut rest) = (0.01, false, rest);
            loop {
                rest = match rest {
                    [] => break,
                    ["--complete", r @ ..] => {
                        complete = true;
                        r
                    }
                    ["--tolerance", t, r @ ..] => {
                        tolerance = t
                            .parse::<f64>()
                            .ok()
                            .filter(|t| *t >= 0.0)
                            .ok_or_else(|| anyhow!("`--tolerance` takes a number >= 0"))?;
                        r
                    }
                    _ => bail!("{USAGE}"),
                };
            }
            let load = |d: &str| -> anyhow::Result<Vec<Value>> {
                Ok(documents(Path::new(d))?
                    .into_iter()
                    .map(|(_, v)| v)
                    .collect())
            };
            let d = differences(&load(base)?, &load(fresh)?, tolerance, complete);
            for line in &d.cost {
                println!("info: {line}");
            }
            for line in &d.drift {
                println!("{line}");
            }
            println!(
                "results compare: {} records of {fresh} differ from {base} by more than {:.2} % on a model, or are new or missing{}; {} differ in the cost (for information)",
                d.drift.len(),
                tolerance * 100.0,
                if complete { " (a complete run)" } else { "" },
                d.cost.len()
            );
            Ok(d.drift.is_empty())
        }
        ["repro", dir] => {
            check_reproduction(Path::new(dir))?;
            Ok(true)
        }
        _ => bail!("{USAGE}"),
    }
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
            &method(),
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
            &method(),
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
            &method(),
            &measured(&[("b", 1.25), ("c", 0.5)]),
            false,
        )
        .unwrap();
        assert_eq!(again, doc);
        // A full run drops what it did not measure.
        let (full, _, _) = merge(
            Some(doc),
            "x86_64",
            &tools,
            &method(),
            &measured(&[("c", 0.5)]),
            true,
        )
        .unwrap();
        assert!(full["levels"]["sse2"]["entries"].get("a").is_none());
    }

    #[test]
    fn merge_refuses_other_tools_and_another_header() {
        let tools = json!({ "llvm-mca": "23.1.1" });
        let (doc, _, _) = merge(
            None,
            "x86_64",
            &tools,
            &method(),
            &measured(&[("a", 1.0)]),
            false,
        )
        .unwrap();
        let other = json!({ "llvm-mca": "24.0.0" });
        let err = merge(
            Some(doc.clone()),
            "x86_64",
            &other,
            &method(),
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
                &method(),
                &measured(&[("a", 1.0)]),
                true
            )
            .is_ok()
        );
        let mut changed = measured(&[("a", 1.0)]);
        changed["sse2"]["block_counts"] = json!("callgrind");
        let err = merge(
            Some(doc.clone()),
            "x86_64",
            &tools,
            &method(),
            &changed,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("block_counts"), "{err}");
        // Records of another method (another address space, D2710) are not mixed into one file.
        let mut randomized = method();
        randomized["address_space"] = json!("randomized");
        let err = merge(
            Some(doc),
            "x86_64",
            &tools,
            &randomized,
            &measured(&[("a", 1.0)]),
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("another method"), "{err}");
    }

    #[test]
    fn a_merge_through_the_file_keeps_the_other_records_bit_for_bit() {
        // Without serde_json's `float_roundtrip` this value comes back one ULP off.
        let x = 0.992_041_588_177_742_7_f64;
        let root = tempfile::tempdir().unwrap();
        let tools = json!({ "llvm-mca": "23.1.1" });
        let out = dir(root.path());
        write(&out, "x86_64", &tools, &measured(&[("a", x)]), false).unwrap();
        let before = std::fs::read_to_string(out.join("x86_64.json")).unwrap();
        write(&out, "x86_64", &tools, &measured(&[("b", 1.0)]), false).unwrap();
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

    /// A results document of `x86_64` with the given records of `sse2`.
    fn doc_of(entries: &[(&str, f64)]) -> Value {
        let (doc, _, _) = merge(
            None,
            "x86_64",
            &json!({ "llvm-mca": "23.1.1" }),
            &method(),
            &measured(entries),
            true,
        )
        .unwrap();
        doc
    }

    #[test]
    fn differences_name_the_drift_above_the_tolerance_and_new_or_lost_records() {
        let base = doc_of(&[("a", 1.0), ("b", 1.0), ("c", 1.0)]);
        // Within 1 %, 2 % slower on both models, a new record.
        let fresh = doc_of(&[("a", 1.009), ("b", 1.02), ("n", 1.0)]);
        let d = differences(
            std::slice::from_ref(&base),
            std::slice::from_ref(&fresh),
            0.01,
            false,
        )
        .drift;
        assert_eq!(d.len(), 2, "{d:?}");
        assert!(
            d[0].starts_with("x86_64 sse2 b: skylake ours 1.0000 -> 1.0200 (+2.00 %)"),
            "{d:?}"
        );
        assert!(d[1].contains("x86_64 sse2 n: not in the committed results"));
        // A record that is no longer simulated; `c` is not in the selection and not compared.
        let mut lost = doc_of(&[("a", 1.0)]);
        lost["levels"]["sse2"]["entries"]["a"] = json!({ "note": "not simulated: the run failed" });
        let d = differences(std::slice::from_ref(&base), &[lost], 0.01, false).drift;
        assert_eq!(d.len(), 1);
        assert!(
            d[0].contains("now \"not simulated: the run failed\""),
            "{d:?}"
        );
        assert_eq!(
            differences(
                std::slice::from_ref(&base),
                std::slice::from_ref(&base),
                0.0,
                true
            ),
            Differences::default()
        );
        // A complete run names the committed records it lacks (`c`, a lost partition).
        let d = differences(std::slice::from_ref(&base), &[fresh], 0.01, true).drift;
        assert_eq!(d.len(), 3, "{d:?}");
        assert!(
            d[0].contains("x86_64 sse2 c: in the committed results, missing"),
            "{d:?}"
        );
        // So does it for a platform that is missing as a whole.
        let d = differences(std::slice::from_ref(&base), &[], 0.01, true).drift;
        assert_eq!(d.len(), 4, "{d:?}");
    }

    #[test]
    fn partitions_merge_into_one_file_per_platform() {
        let root = tempfile::tempdir().unwrap();
        let (p1, p2, into) = (
            root.path().join("p1"),
            root.path().join("p2"),
            root.path().join("all"),
        );
        let tools = json!({ "llvm-mca": "23.1.1" });
        write(&p1, "x86_64", &tools, &measured(&[("a", 1.0)]), false).unwrap();
        write(&p2, "x86_64", &tools, &measured(&[("b", 2.0)]), false).unwrap();
        merge_dirs(&into, &[p1.clone(), p2]).unwrap();
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(into.join("x86_64.json")).unwrap())
                .unwrap();
        let e = &doc["levels"]["sse2"]["entries"];
        assert!(e.get("a").is_some() && e.get("b").is_some());
        // A partition with other tools is refused.
        let p3 = root.path().join("p3");
        write(
            &p3,
            "x86_64",
            &json!({ "llvm-mca": "22" }),
            &measured(&[("c", 1.0)]),
            false,
        )
        .unwrap();
        assert!(merge_dirs(&into, &[p3]).is_err());
        assert!(run(&["merge", "--into", "x"]).is_err());
        // Nothing to merge (every partition failed): an error, not an empty result.
        let empty = root.path().join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        let err = merge_dirs(&root.path().join("none"), &[empty]).unwrap_err();
        assert!(err.to_string().contains("no source"), "{err}");
        assert!(merge_dirs(&into, &[root.path().join("missing")]).is_err());
    }

    #[test]
    fn the_reproduction_check_reads_costas_at_neon() {
        let root = tempfile::tempdir().unwrap();
        let write_costas = |ours_ref: f64, ours_counter: f64| {
            let m = |o: f64| json!({ "cpu": "x", "ours": { "cycles_per_sample": o }, "comparison": { "cycles_per_sample": 1.0 }, "ratio": o });
            let doc = json!({ "platform": "aarch64", "levels": { "neon": { "entries": {
                "costas": { "reference_model": m(ours_ref), "counter_model": m(ours_counter) } } } } });
            std::fs::write(root.path().join("aarch64.json"), doc.to_string()).unwrap();
        };
        write_costas(86.131_242, 54.094_621);
        check_reproduction(root.path()).unwrap();
        write_costas(86.14, 54.09);
        let err = check_reproduction(root.path()).unwrap_err();
        assert!(err.to_string().contains("86.14 / 54.09"), "{err}");
    }
}
