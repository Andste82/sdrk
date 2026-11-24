//! `cargo xtask check own-code`: the documents must not state a status, a limit or a judgement
//! that compares our code with our own code (the maintainer's measurement order, D32, D2764,
//! D2767). Comparisons of our code with our own code are information only; the text that
//! described the removed own-build judgement of `avx2-runtime` (`_limits.runtime`, 1.02 × our
//! fixed `avx2` build, D2702) is stale wherever it still reads as current.
//!
//! Checked: `docs/performance.md`, `CHANGELOG.md` (a line that says it is superseded, or that the
//! limit was removed, is history) and the status sections of `docs/milestones/*.md` (from a
//! heading with `Status` to the next heading of the same or a higher level). Besides, a row of
//! `docs/open-items.md` that says a waiver kind of `benches/limits.json` names the item (`a
//! cycle_waiver naming this item`) needs such a waiver in `benches/limits.json`.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use crate::scan::Finding;

/// The phrases of the removed own-build judgement, lower case. `1.02` is the removed
/// `_limits.runtime`; a limit or cap of 1.02 is no limit of any level any more.
const STALE: [&str; 15] = [
    "_limits.runtime",
    "1.02 × our own",
    "1.02 x our own",
    "1.02× our own",
    "1.02x our own",
    "1.02 × of our own",
    "judged against our own numbers",
    "judged against our own build",
    "judged against our own fixed",
    "judged against our own `avx2`",
    "judged against `avx2` with",
    "limit 1.02",
    "limit of 1.02",
    "cap 1.02",
    "cap of 1.02",
];

/// Whether a line says that what it describes is history (superseded or removed).
fn history(line: &str) -> bool {
    let l = line.to_lowercase();
    l.contains("superseded") || l.contains("removed") || l.contains("amended by d2764")
}

/// The stale phrases of `text` (`path` names it in the findings), every line, or only the lines
/// of the status sections with `status_only`. A line that is history is skipped.
#[must_use]
pub fn lint_text(path: &Path, text: &str, status_only: bool) -> Vec<Finding> {
    let mut out = Vec::new();
    // The level of the heading of the status section we are in.
    let mut status: Option<usize> = None;
    for (n, line) in text.lines().enumerate() {
        let hashes = line.chars().take_while(|c| *c == '#').count();
        if hashes > 0 && line[hashes..].starts_with(' ') {
            if status.is_some_and(|l| hashes <= l) {
                status = None;
            }
            if line.contains("Status") {
                status = Some(hashes);
            }
        }
        if status_only && status.is_none() {
            continue;
        }
        if history(line) {
            continue;
        }
        let lower = line.to_lowercase();
        for phrase in STALE {
            if lower.contains(&phrase.to_lowercase()) {
                out.push(Finding {
                    path: path.to_path_buf(),
                    line: n + 1,
                    message: format!(
                        "`{phrase}`: the own-build judgement of `avx2-runtime` was removed (D2764); a comparison of our code with our own code is information only (D2767). Rewrite it, or mark the line superseded"
                    ),
                });
            }
        }
    }
    out
}

/// The waiver kinds of `benches/limits.json`, with the phrases by which an open item says that
/// one of them names it.
const KINDS: [(&str, [&str; 2]); 3] = [
    (
        "cycle_waiver",
        [
            "`cycle_waiver` naming this item",
            "cycle waiver naming this item",
        ],
    ),
    (
        "accuracy_waiver",
        [
            "`accuracy_waiver` naming this item",
            "accuracy waiver naming this item",
        ],
    ),
    (
        "waiver",
        ["`waiver` naming this item", "ratio waiver naming this item"],
    ),
];

/// The rows of `open_items` (a table row `| M.. | <ID>: ...`) that say a waiver kind names the
/// item while no entry of `limits_json` has a waiver of that kind naming `open item <ID>`.
///
/// # Errors
/// `benches/limits.json` cannot be parsed.
pub fn lint_open_items(open_items: &str, limits_json: &str) -> anyhow::Result<Vec<Finding>> {
    let limits: Value = serde_json::from_str(limits_json).context("benches/limits.json")?;
    let texts = |kind: &str| -> Vec<String> {
        limits
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(k, _)| !k.starts_with('_'))
            .filter_map(|(_, levels)| levels.as_object())
            .flat_map(|levels| levels.values())
            .filter_map(|e| e.get(kind).and_then(Value::as_str).map(str::to_string))
            .collect()
    };
    let mut out = Vec::new();
    for (n, line) in open_items.lines().enumerate() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        let Some(id) = cells
            .get(2)
            .and_then(|c| c.split_once(':'))
            .map(|(id, _)| id.trim())
            .filter(|id| !id.is_empty() && !id.contains(' '))
        else {
            continue;
        };
        for (kind, phrases) in KINDS {
            if !phrases.iter().any(|p| line.contains(p)) {
                continue;
            }
            let needle = format!("open item {id}");
            if !texts(kind)
                .iter()
                .any(|t| crate::release::references(t).contains(&needle))
            {
                out.push(Finding {
                    path: PathBuf::from("docs/open-items.md"),
                    line: n + 1,
                    message: format!(
                        "{id} says a `{kind}` names it, but no entry of benches/limits.json has a `{kind}` naming `{needle}`: close or correct the item (D2767)"
                    ),
                });
            }
        }
    }
    Ok(out)
}

/// The lines (1-based) of the section `## [Unreleased]` of the changelog: the released sections
/// are history.
#[must_use]
pub fn unreleased(changelog: &str) -> std::ops::Range<usize> {
    let mut start = None;
    for (n, line) in changelog.lines().enumerate() {
        if line.starts_with("## [") {
            if let Some(s) = start {
                return s..n + 1;
            }
            if line.starts_with("## [Unreleased]") {
                start = Some(n + 1);
            }
        }
    }
    start.map_or(0..0, |s| s..usize::MAX)
}

/// The whole check on the repository at `root`.
///
/// # Errors
/// A file cannot be read, or `benches/limits.json` cannot be parsed.
#[allow(
    clippy::large_stack_frames,
    reason = "the Windows file system calls and the records are large; xtask is a host tool"
)]
pub fn check(root: &Path) -> anyhow::Result<Vec<Finding>> {
    let read = |p: &Path| {
        std::fs::read_to_string(root.join(p))
            .with_context(|| format!("cannot read {}", p.display()))
    };
    let mut out = Vec::new();
    let performance = PathBuf::from("docs/performance.md");
    out.extend(lint_text(&performance, &read(&performance)?, false));
    let changelog = PathBuf::from("CHANGELOG.md");
    let text = read(&changelog)?;
    let range = unreleased(&text);
    out.extend(
        lint_text(&changelog, &text, false)
            .into_iter()
            .filter(|f| range.contains(&f.line)),
    );
    let dir = root.join("docs").join("milestones");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .with_context(|| format!("cannot read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    files.sort();
    for f in files {
        let rel = f.strip_prefix(root).unwrap_or(&f).to_path_buf();
        out.extend(lint_text(&rel, &read(&rel)?, true));
    }
    out.extend(lint_open_items(
        &read(Path::new("docs/open-items.md"))?,
        &read(Path::new("benches/limits.json"))?,
    )?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(f: &[Finding]) -> Vec<usize> {
        f.iter().map(|f| f.line).collect()
    }

    #[test]
    fn the_stale_own_build_judgement_fails_and_history_and_information_pass() {
        let text = "\
| Run-time detection (`avx2-runtime`, D2702) | ours ≤ 1.02 × our own fixed `avx2` build |
| `avx2-runtime` | callgrind; judged against our own numbers at `avx2` (at most 1.02×) |
- D2708: `avx2-runtime` against our own `avx2` build with `_limits.runtime` (1.02). Superseded by D2764.
- `_limits.runtime`, `Caps::runtime` and the own-build judgement are removed (D2764).
The column `ours / avx2 build` is ours against our own fixed `avx2` build, information only.
Comparisons of our code with our own code are information and checks, never a status.
`fft_64` 1.07 of the same kernel at compile-time `avx2`, limit 1.02.
";
        let f = lint_text(Path::new("docs/performance.md"), text, false);
        assert_eq!(lines(&f), vec![1, 2, 7], "{f:?}");
        assert!(f[0].message.contains("D2764"));
    }

    #[test]
    fn the_changelog_is_checked_in_its_unreleased_section() {
        let text = "# Changelog\n## [Unreleased]\n- limit 1.02\n## [0.1.0]\n- limit 1.02\n";
        assert_eq!(unreleased(text), 2..4);
        assert_eq!(unreleased("## [0.1.0]\n"), 0..0);
    }

    #[test]
    fn milestones_are_checked_in_their_status_sections_only() {
        let text = "\
# M13
## Status
fft 64: limit 1.02 of the fixed build.
### Status of the acceptance criteria
judged against our own numbers.
## Specification
The level `avx2-runtime` with `_limits.runtime` 1.02.
## Status
Amended by D2764: `_limits.runtime` 1.02 no longer judges.
";
        let f = lint_text(Path::new("docs/milestones/M13.md"), text, true);
        assert_eq!(lines(&f), vec![3, 5], "{f:?}");
    }

    #[test]
    fn an_open_item_that_names_a_missing_waiver_fails() {
        let open = "\
| Milestone | Item | Why | What |
| --- | --- | --- | --- |
| M30 | I30-2: accepted with a ratio waiver naming this item (D339) | x | y |
| M30 | I30-4: each keeps a `cycle_waiver` naming this item | x | y |
| M24 | I24-3: no waiver mentioned | x | y |
";
        let limits = r#"{"_limits": {}, "mag": {"neon": {"status": "release", "cycle_waiver": "a tie (open item I30-4)"}},
            "add_real_q15": {"avx2-runtime": {"status": "release"}}}"#;
        let f = lint_open_items(open, limits).unwrap();
        assert_eq!(lines(&f), vec![3], "{f:?}");
        assert!(f[0].message.contains("I30-2") && f[0].message.contains("`waiver`"));
        // With the waiver in place, the row passes; a waiver of another kind does not count.
        let with = r#"{"add_real_q15": {"avx2-runtime": {"status": "release", "waiver": "open item I30-2"}},
            "mag": {"neon": {"status": "release", "waiver": "open item I30-4"}}}"#;
        let f = lint_open_items(open, with).unwrap();
        assert_eq!(lines(&f), vec![4], "{f:?}");
    }

    #[test]
    fn the_committed_documents_pass() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        assert_eq!(check(root).unwrap(), Vec::<Finding>::new());
    }
}
