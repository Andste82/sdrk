//! `cargo xtask release-check <version>` (M25.1 §2): the checks of a release commit.
//!
//! Every check works on text that [`Inputs::read`] collects from the repository, so that the
//! unit tests can run each rule on a prepared bad input.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::Path;
use std::process::Command;

use anyhow::{Context, bail};
use serde_json::Value;

use crate::wasm::workspace_version;

/// Paths (prefixes) whose text may contain the old version: records, lock files, and the
/// fixtures of the `xtask` tests and of this module.
const VERSION_SCAN_SKIP: &[&str] = &[
    "docs/milestones/",
    "CHANGELOG.md",
    "docs/decisions-log.md",
    "xtask/tests/",
    "xtask/src/release.rs",
];

/// The text the release check reads.
#[derive(Debug, Default, Clone)]
pub struct Inputs {
    /// The root `Cargo.toml`.
    pub cargo_toml: String,
    /// The manifests of the members of the root workspace: `(member path, manifest text)`.
    pub members: Vec<(String, String)>,
    /// `CHANGELOG.md`.
    pub changelog: String,
    /// Whether `docs/releases/<version>.md` exists.
    pub release_notes: bool,
    /// `benches/limits.json`.
    pub limits_json: String,
    /// `docs/open-items.md`.
    pub open_items: String,
    /// `bindings/wasm/pkg-template/package.json`.
    pub package_json: String,
    /// Every tracked text file outside the records, lock files and test fixtures: `(path, text)`.
    pub files: Vec<(String, String)>,
}

/// One `dev` or `release` entry of `benches/limits.json` with a waiver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Waiver {
    /// Kernel entry and level, as `fir_decimate_cq15_fallback neon`.
    pub entry: String,
    /// `dev` or `release`.
    pub status: String,
    /// `waiver`, `accuracy_waiver` or both.
    pub field: String,
    /// The open items and issues the text names (empty: the text states why the gap is
    /// accepted).
    pub references: Vec<String>,
}

/// The result of the check.
#[derive(Debug, Default)]
pub struct Outcome {
    /// One line per failed point.
    pub problems: Vec<String>,
    /// The `dev` and `release` entries with a waiver.
    pub waivers: Vec<Waiver>,
}

/// The member paths of the `members = [...]` list of the root manifest.
#[must_use]
pub fn workspace_members(cargo_toml: &str) -> Vec<String> {
    let Some(start) = cargo_toml.find("members = [") else {
        return Vec::new();
    };
    let rest = &cargo_toml[start + "members = [".len()..];
    let list = rest.split(']').next().unwrap_or_default();
    list.split(',')
        .map(|item| item.trim().trim_matches('"').to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

/// The `open item <ID>` and `issue #<N>` references of a waiver text.
#[must_use]
pub fn references(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (marker, prefix) in [("open item ", "open item "), ("issue #", "issue #")] {
        let mut rest = text;
        while let Some(i) = rest.find(marker) {
            let after = &rest[i + marker.len()..];
            let id: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
                .collect();
            let id = id.trim_end_matches('-');
            if !id.is_empty() {
                out.push(format!("{prefix}{id}"));
            }
            rest = after;
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Whether `docs/open-items.md` has the item `id` (a table row with `<ID>:`).
fn open_item_exists(open_items: &str, id: &str) -> bool {
    let needle = format!("| {id}:");
    open_items.lines().any(|line| line.contains(&needle))
}

/// Whether a line names `old` as a version (`version = "0.0.1"`, `"version": "0.0.1"`,
/// `version 0.0.1`).
fn names_version(line: &str, old: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    [
        format!("version = \"{old}\""),
        format!("\"version\": \"{old}\""),
        format!("version {old}"),
        format!("version stays {old}"),
    ]
    .iter()
    .any(|p| lower.contains(p))
}

/// The `dev` and `release` entries of `benches/limits.json` that carry a waiver.
///
/// # Errors
/// If the file is not valid JSON.
pub fn waivers(limits_json: &str) -> anyhow::Result<Vec<Waiver>> {
    let limits: Value = serde_json::from_str(limits_json).context("benches/limits.json")?;
    let mut out = Vec::new();
    let Some(kernels) = limits.as_object() else {
        return Ok(out);
    };
    for (kernel, levels) in kernels {
        let Some(levels) = levels.as_object() else {
            continue;
        };
        for (level, entry) in levels {
            let status = entry
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if status != "dev" && status != "release" {
                continue;
            }
            let texts: Vec<(&str, &str)> = ["waiver", "accuracy_waiver"]
                .iter()
                .filter_map(|f| entry.get(*f).and_then(Value::as_str).map(|t| (*f, t)))
                .collect();
            if texts.is_empty() {
                continue;
            }
            out.push(Waiver {
                entry: format!("{kernel} {level}"),
                status: status.to_string(),
                field: texts.iter().map(|(f, _)| *f).collect::<Vec<_>>().join(", "),
                references: references(
                    &texts.iter().map(|(_, t)| *t).collect::<Vec<_>>().join(" "),
                ),
            });
        }
    }
    Ok(out)
}

/// The problems of the version and changelog points.
fn version_problems(inputs: &Inputs, version: &str, problems: &mut Vec<String>) {
    match workspace_version(&inputs.cargo_toml) {
        Some(v) if v == version => {}
        Some(v) => problems.push(format!(
            "Cargo.toml: the workspace version is {v}, not {version}"
        )),
        None => problems.push("Cargo.toml: no `version` in [workspace.package]".into()),
    }
    for (member, manifest) in &inputs.members {
        if !manifest
            .lines()
            .any(|l| l.split_whitespace().collect::<String>() == "version.workspace=true")
        {
            problems.push(format!(
                "{member}/Cargo.toml does not inherit the version (`version.workspace = true`)"
            ));
        }
    }
    let lines: Vec<&str> = inputs.changelog.lines().collect();
    match lines.iter().position(|l| l.trim() == "## [Unreleased]") {
        None => problems.push("CHANGELOG.md has no `## [Unreleased]` section".into()),
        Some(i) => {
            let next = lines
                .iter()
                .skip(i + 1)
                .find(|l| !l.trim().is_empty())
                .map(|l| l.trim());
            let prefix = format!("## [{version}] - ");
            match next {
                Some(l) if l.starts_with(&prefix) && is_date(&l[prefix.len()..]) => {}
                Some(l) if l.starts_with("## ") => problems.push(format!(
                    "CHANGELOG.md: the section below `[Unreleased]` is `{l}`, not `{prefix}YYYY-MM-DD`"
                )),
                _ => problems.push(format!(
                    "CHANGELOG.md: `[Unreleased]` must be empty and followed by `{prefix}YYYY-MM-DD`"
                )),
            }
        }
    }
    if !inputs.release_notes {
        problems.push(format!("docs/releases/{version}.md does not exist"));
    }
    if !inputs.package_json.contains("\"version\": \"__VERSION__\"") {
        problems.push(
            "bindings/wasm/pkg-template/package.json must keep `\"version\": \"__VERSION__\"`"
                .into(),
        );
    }
}

/// Whether `s` is `YYYY-MM-DD`.
fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10
        && b.iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b'-',
            _ => c.is_ascii_digit(),
        })
}

/// Runs every point of M25.1 §2 on `inputs`; `previous` is the version that must no longer
/// appear as a version.
///
/// # Errors
/// If `benches/limits.json` is not valid JSON.
pub fn check(inputs: &Inputs, version: &str, previous: &str) -> anyhow::Result<Outcome> {
    let mut problems = Vec::new();
    version_problems(inputs, version, &mut problems);
    let waivers = waivers(&inputs.limits_json)?;
    for w in &waivers {
        for r in &w.references {
            if let Some(id) = r.strip_prefix("open item ")
                && !open_item_exists(&inputs.open_items, id)
            {
                problems.push(format!(
                    "benches/limits.json: {} names {r}, which docs/open-items.md does not have",
                    w.entry
                ));
            }
        }
    }
    for (path, text) in &inputs.files {
        for (i, line) in text.lines().enumerate() {
            if names_version(line, previous) {
                problems.push(format!("{path}:{}: names the version {previous}", i + 1));
            }
        }
    }
    Ok(Outcome { problems, waivers })
}

impl Inputs {
    /// Reads the inputs from the repository at `root` (the tracked files through `git ls-files`).
    ///
    /// # Errors
    /// If a file cannot be read or `git` cannot be run.
    pub fn read(root: &Path, version: &str) -> anyhow::Result<Self> {
        let read = |rel: &str| {
            std::fs::read_to_string(root.join(rel)).with_context(|| format!("cannot read {rel}"))
        };
        let cargo_toml = read("Cargo.toml")?;
        let members = workspace_members(&cargo_toml)
            .into_iter()
            .map(|m| {
                let text = read(&format!("{m}/Cargo.toml"))?;
                Ok((m, text))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let listed = Command::new("git")
            .args(["ls-files", "-z"])
            .current_dir(root)
            .output()
            .context("cannot run `git ls-files`")?;
        if !listed.status.success() {
            bail!("`git ls-files` failed");
        }
        let mut files = Vec::new();
        for path in String::from_utf8_lossy(&listed.stdout).split('\0') {
            if path.is_empty()
                || Path::new(path)
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("lock"))
                || VERSION_SCAN_SKIP.iter().any(|s| path.starts_with(s))
            {
                continue;
            }
            // Binary files (golden vectors) are not text and cannot name a version.
            if let Ok(text) = std::fs::read_to_string(root.join(path)) {
                files.push((path.to_string(), text));
            }
        }
        Ok(Self {
            cargo_toml,
            members,
            changelog: read("CHANGELOG.md")?,
            release_notes: root
                .join("docs")
                .join("releases")
                .join(format!("{version}.md"))
                .is_file(),
            limits_json: read("benches/limits.json")?,
            open_items: read("docs/open-items.md")?,
            package_json: read("bindings/wasm/pkg-template/package.json")?,
            files,
        })
    }
}

/// `release-check <version> [--previous <version>]`: prints the report; `Ok(false)` if a
/// point fails.
///
/// # Errors
/// On a wrong command line or if the inputs cannot be read.
pub fn run(root: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let (version, previous) = match args {
        [v] => (*v, "0.0.1"),
        [v, "--previous", p] => (*v, *p),
        _ => bail!("usage: cargo xtask release-check <version> [--previous <version>]"),
    };
    let inputs = Inputs::read(root, version)?;
    let outcome = check(&inputs, version, previous)?;
    println!(
        "release-check {version}: {} dev or release entries carry a waiver",
        outcome.waivers.len()
    );
    for w in &outcome.waivers {
        let refs = if w.references.is_empty() {
            "accepted (no open item or issue)".to_string()
        } else {
            w.references.join(", ")
        };
        println!("  {:<44} {:<7} {:<24} {refs}", w.entry, w.status, w.field);
    }
    for p in &outcome.problems {
        eprintln!("release-check: {p}");
    }
    if outcome.problems.is_empty() {
        println!("release-check {version}: ok");
    } else {
        eprintln!(
            "release-check {version}: {} problem(s)",
            outcome.problems.len()
        );
    }
    Ok(outcome.problems.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: &str = r#"{
      "_limits": { "release": 1.1 },
      "a_fallback": {
        "neon": { "status": "dev", "waiver": "slow; open item I20-1 in docs/open-items.md" },
        "sse2": { "status": "release", "accuracy_waiver": "the reference computes in f64" },
        "avx2": { "status": "info", "waiver": "information, open item X9-9" },
        "simd128": { "status": "release" }
      },
      "b": { "avx2": { "status": "dev", "waiver": "see issue #71" } }
    }"#;

    fn good() -> Inputs {
        Inputs {
            cargo_toml: "[workspace]\nmembers = [\n    \"crates/a\",\n    \"xtask\",\n]\n\n[workspace.package]\nversion = \"0.1.0\"\n".into(),
            members: vec![
                ("crates/a".into(), "[package]\nname = \"a\"\nversion.workspace = true\n".into()),
                ("xtask".into(), "[package]\nname = \"xtask\"\nversion.workspace = true\n".into()),
            ],
            changelog: "# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-11-02\n\n### Added\n\n- x\n".into(),
            release_notes: true,
            limits_json: LIMITS.into(),
            open_items: "| Milestone | Item |\n| --- | --- |\n| #20 | I20-1: FIR fallbacks |\n".into(),
            package_json: "{\n  \"name\": \"sdrk-wasm\",\n  \"version\": \"__VERSION__\"\n}\n".into(),
            files: vec![("README.md".into(), "The workspace version is 0.1.0.\n".into())],
        }
    }

    fn problems(inputs: &Inputs) -> Vec<String> {
        check(inputs, "0.1.0", "0.0.1")
            .expect("valid limits")
            .problems
    }

    #[test]
    fn a_good_tree_passes_and_lists_the_waivers() {
        let outcome = check(&good(), "0.1.0", "0.0.1").expect("valid limits");
        assert_eq!(outcome.problems, Vec::<String>::new());
        let rows: Vec<(String, Vec<String>)> = outcome
            .waivers
            .iter()
            .map(|w| (w.entry.clone(), w.references.clone()))
            .collect();
        assert_eq!(
            rows,
            [
                (
                    "a_fallback neon".to_string(),
                    vec!["open item I20-1".to_string()]
                ),
                ("a_fallback sse2".to_string(), vec![]),
                ("b avx2".to_string(), vec!["issue #71".to_string()]),
            ]
        );
    }

    #[test]
    fn a_wrong_workspace_version_fails() {
        let mut i = good();
        i.cargo_toml = i.cargo_toml.replace("0.1.0", "0.0.1");
        assert!(
            problems(&i)
                .iter()
                .any(|p| p.contains("the workspace version is 0.0.1"))
        );
    }

    #[test]
    fn a_member_with_its_own_version_fails() {
        let mut i = good();
        i.members[0].1 = "[package]\nname = \"a\"\nversion = \"0.1.0\"\n".into();
        assert!(
            problems(&i)
                .iter()
                .any(|p| p.contains("crates/a/Cargo.toml does not inherit"))
        );
    }

    #[test]
    fn a_non_empty_unreleased_section_fails() {
        let mut i = good();
        i.changelog = i
            .changelog
            .replace("## [Unreleased]\n", "## [Unreleased]\n\n- late entry\n");
        assert!(
            problems(&i)
                .iter()
                .any(|p| p.contains("`[Unreleased]` must be empty"))
        );
        let mut i = good();
        i.changelog = i
            .changelog
            .replace("## [0.1.0] - 2026-11-02", "## [0.1.0] - soon");
        assert!(
            problems(&i)
                .iter()
                .any(|p| p.contains("not `## [0.1.0] - YYYY-MM-DD`"))
        );
    }

    #[test]
    fn missing_release_notes_fail() {
        let mut i = good();
        i.release_notes = false;
        assert_eq!(problems(&i), ["docs/releases/0.1.0.md does not exist"]);
    }

    #[test]
    fn a_waiver_naming_a_missing_open_item_fails() {
        let mut i = good();
        i.open_items = String::new();
        assert_eq!(
            problems(&i),
            [
                "benches/limits.json: a_fallback neon names open item I20-1, which docs/open-items.md does not have"
            ]
        );
    }

    #[test]
    fn a_filled_package_version_fails() {
        let mut i = good();
        i.package_json = i.package_json.replace("__VERSION__", "0.1.0");
        assert!(problems(&i).iter().any(|p| p.contains("__VERSION__")));
    }

    #[test]
    fn a_stray_old_version_fails() {
        let mut i = good();
        i.files.push((
            "targets/x/Cargo.toml".into(),
            "[package]\nversion = \"0.0.1\"\n".into(),
        ));
        i.files.push((
            "docs/wasm.md".into(),
            "Sizes (version 0.0.1): 1 MiB\n".into(),
        ));
        let p = problems(&i);
        assert!(
            p.contains(&"targets/x/Cargo.toml:2: names the version 0.0.1".to_string()),
            "{p:?}"
        );
        assert!(
            p.contains(&"docs/wasm.md:1: names the version 0.0.1".to_string()),
            "{p:?}"
        );
    }

    #[test]
    fn references_and_members_are_parsed() {
        assert_eq!(
            references("see open item I20-1 and open item M13-4; issue #71, issue #71."),
            ["issue #71", "open item I20-1", "open item M13-4"]
        );
        assert_eq!(workspace_members(&good().cargo_toml), ["crates/a", "xtask"]);
    }
}
