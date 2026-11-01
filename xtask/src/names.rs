//! The name check of the block kinds (M24.1, `docs/blocks.md` section "Names").
//!
//! It reads the source files as text:
//! - the kinds of the descriptions (`kind: "<k>",` in `crates/blocks/sdrk-blocks/src/blocks/*.rs`);
//! - the pipeline kinds (`struct <K>Kind` and its `const KIND: &'static str = "<k>";` in
//!   `crates/pipeline/sdrk-pipeline/src/kinds.rs`).
//!
//! For every kind `k`: `k` matches the kind pattern (lowercase words joined by `_`, at most 32
//! characters); its description lives in `blocks/<k>.rs`; that file defines `<K>Block`, either as
//! `pub struct <K>Block` or as the name argument of a macro that defines it (as `biquad.rs`
//! does); `kinds.rs` has `struct <K>Kind` with `KIND = "<k>"`; and the descriptions and the
//! pipeline name the same kinds.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::scan::Finding;

/// Kinds that cannot follow the file and type rules, with the reason. M24.1 needs none.
const EXCEPTIONS: &[(&str, &str)] = &[];

/// The longest kind.
pub const MAX_KIND_LEN: usize = 32;

/// Whether `kind` matches `^[a-z][a-z0-9]*(_[a-z0-9]+)*$` and has at most
/// [`MAX_KIND_LEN`] characters (rule 1).
#[must_use]
pub fn is_valid_kind(kind: &str) -> bool {
    kind.len() <= MAX_KIND_LEN
        && kind.split('_').enumerate().all(|(i, word)| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) if c.is_ascii_lowercase() || (i > 0 && c.is_ascii_digit()) => {
                    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                }
                _ => false,
            }
        })
}

/// The kind in `UpperCamelCase` (rule 9): `fm_stereo_demod` -> `FmStereoDemod`.
#[must_use]
pub fn upper_camel(kind: &str) -> String {
    kind.split('_')
        .map(|word| {
            let mut chars = word.chars();
            chars.next().map_or_else(String::new, |c| {
                c.to_ascii_uppercase().to_string() + chars.as_str()
            })
        })
        .collect()
}

/// The string literal after `prefix` on a line (`kind: "agc",` -> `agc`).
fn literal_after<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = line.trim_start().strip_prefix(prefix)?;
    let rest = rest.strip_prefix('"')?;
    rest.split('"').next()
}

/// The kinds of the descriptions in one block file, with their line numbers.
fn description_kinds(text: &str) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .filter_map(|(i, line)| literal_after(line, "kind: ").map(|k| (i + 1, k.to_string())))
        .collect()
}

/// Whether a block file defines the type `name` (a `pub struct` or a macro's name argument).
fn defines_type(text: &str, name: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim_start();
        line.strip_prefix("pub struct ")
            .is_some_and(|rest| rest.split(['<', ' ', '{', '(', ';']).next() == Some(name))
            || line.split(',').next().map(str::trim) == Some(name) && line.contains(',')
    })
}

/// The pipeline kinds of `kinds.rs`: the value of each `KIND` with the name of the type of
/// the `impl Kind for <T>` block it belongs to, and its line number.
fn pipeline_kinds(text: &str) -> Vec<(usize, String, String)> {
    let mut current: Option<String> = None;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if let Some(rest) = line.trim_start().strip_prefix("impl Kind for ") {
            current = rest.split([' ', '{']).next().map(str::to_string);
        } else if let Some(kind) = literal_after(line, "const KIND: &'static str = ")
            && let Some(ty) = &current
        {
            out.push((i + 1, ty.clone(), kind.to_string()));
        }
    }
    out
}

/// The text inputs of the check, so that it can run on a fake tree.
#[derive(Debug, Default)]
pub struct Sources {
    /// `kinds.rs` of the pipeline.
    pub kinds_rs: String,
    /// The block files by file stem (`resample` for `blocks/resample.rs`).
    pub blocks: BTreeMap<String, String>,
}

fn is_exception(kind: &str) -> bool {
    EXCEPTIONS.iter().any(|(e, _)| *e == kind)
}

/// The problems of one block file, as `(line, message)`.
fn block_problems(stem: &str, text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (line, kind) in description_kinds(text) {
        if !is_valid_kind(&kind) {
            out.push((line, format!(
                "kind `{kind}` breaks rule 1 of the naming scheme: lowercase ASCII words joined by `_`, at most {MAX_KIND_LEN} characters"
            )));
        }
        if is_exception(&kind) {
            continue;
        }
        if kind != stem {
            out.push((line, format!(
                "kind `{kind}` is described in `{stem}.rs`; rule 9 of the naming scheme puts it in `blocks/{kind}.rs`"
            )));
        }
        let block = format!("{}Block", upper_camel(&kind));
        if !defines_type(text, &block) {
            out.push((line, format!(
                "kind `{kind}`: `{stem}.rs` does not define `{block}` (rule 9 of the naming scheme)"
            )));
        }
    }
    out
}

/// The problems of `kinds.rs` against the described kinds, as `(line, message)`.
fn pipeline_problems(kinds_rs: &str, described: &BTreeSet<String>) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut piped = BTreeSet::new();
    for (line, ty, kind) in pipeline_kinds(kinds_rs) {
        let want = format!("{}Kind", upper_camel(&kind));
        if ty != want && !is_exception(&kind) {
            out.push((line, format!(
                "the pipeline kind `{kind}` is the type `{ty}`; rule 9 of the naming scheme names it `{want}`"
            )));
        }
        if !described.contains(&kind) {
            out.push((
                line,
                format!("the pipeline kind `{kind}` has no block description of that kind"),
            ));
        }
        piped.insert(kind);
    }
    for kind in described.difference(&piped) {
        out.push((0, format!(
            "the block kind `{kind}` has no pipeline kind (`struct {}Kind` with `KIND = \"{kind}\"`)",
            upper_camel(kind)
        )));
    }
    out
}

/// The findings of the name check for `sources`; `blocks_dir` and `kinds_path` only name the
/// files in the messages.
#[must_use]
pub fn check_sources(sources: &Sources, blocks_dir: &Path, kinds_path: &Path) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut described = BTreeSet::new();
    for (stem, text) in &sources.blocks {
        described.extend(description_kinds(text).into_iter().map(|(_, k)| k));
        let path = blocks_dir.join(format!("{stem}.rs"));
        findings.extend(
            block_problems(stem, text)
                .into_iter()
                .map(|(line, message)| Finding {
                    path: path.clone(),
                    line,
                    message,
                }),
        );
    }
    findings.extend(
        pipeline_problems(&sources.kinds_rs, &described)
            .into_iter()
            .map(|(line, message)| Finding {
                path: kinds_path.to_path_buf(),
                line,
                message,
            }),
    );
    findings
}

/// `check names`: reads the block files and `kinds.rs` below `root` and checks them.
///
/// # Errors
/// If a file cannot be read.
#[allow(
    clippy::large_stack_frames,
    reason = "the Windows file system calls are large; xtask is a host tool"
)]
pub fn check_names(root: &Path) -> anyhow::Result<Vec<Finding>> {
    let blocks_dir: PathBuf = ["crates", "blocks", "sdrk-blocks", "src", "blocks"]
        .iter()
        .collect();
    let kinds_path: PathBuf = ["crates", "pipeline", "sdrk-pipeline", "src", "kinds.rs"]
        .iter()
        .collect();
    let mut sources = Sources {
        kinds_rs: std::fs::read_to_string(root.join(&kinds_path))
            .with_context(|| format!("cannot read {}", kinds_path.display()))?,
        blocks: BTreeMap::new(),
    };
    let dir = root.join(&blocks_dir);
    for entry in
        std::fs::read_dir(&dir).with_context(|| format!("cannot read {}", dir.display()))?
    {
        let path = entry?.path();
        let (Some(stem), Some("rs")) = (
            path.file_stem().and_then(|s| s.to_str()),
            path.extension().and_then(|e| e.to_str()),
        ) else {
            continue;
        };
        if stem == "mod" {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        sources.blocks.insert(stem.to_string(), text);
    }
    Ok(check_sources(&sources, &blocks_dir, &kinds_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources(kind: &str, file: &str, block: &str, kind_type: &str) -> Sources {
        Sources {
            kinds_rs: format!(
                "pub(crate) struct {kind_type};\n\nimpl Kind for {kind_type} {{\n    const KIND: &'static str = \"{kind}\";\n}}\n"
            ),
            blocks: BTreeMap::from([(
                file.to_string(),
                format!(
                    "static D: BlockDescription = BlockDescription {{\n    kind: \"{kind}\",\n}};\n\npub struct {block}<T, S> {{}}\n"
                ),
            )]),
        }
    }

    fn messages(s: &Sources) -> Vec<String> {
        check_sources(s, Path::new("blocks"), Path::new("kinds.rs"))
            .into_iter()
            .map(|f| f.message)
            .collect()
    }

    #[test]
    fn the_kind_pattern() {
        for ok in ["fm_demod", "real_to_iq", "resample", "a", "fir2", "x_2"] {
            assert!(is_valid_kind(ok), "{ok}");
        }
        for bad in [
            "",
            "Fm",
            "fm__demod",
            "_fm",
            "fm_",
            "2fm",
            "fm-demod",
            "fm demod",
            &"a".repeat(33),
        ] {
            assert!(!is_valid_kind(bad), "{bad}");
        }
        assert_eq!(upper_camel("fm_stereo_demod"), "FmStereoDemod");
        assert_eq!(upper_camel("dc_block"), "DcBlock");
        assert_eq!(upper_camel("real_to_iq"), "RealToIq");
    }

    #[test]
    fn a_consistent_kind_passes() {
        let m = messages(&sources(
            "real_to_iq",
            "real_to_iq",
            "RealToIqBlock",
            "RealToIqKind",
        ));
        assert_eq!(m, Vec::<String>::new());
    }

    #[test]
    fn a_macro_defined_block_passes() {
        let mut s = sources("biquad", "biquad", "Unused", "BiquadKind");
        if let Some(text) = s.blocks.get_mut("biquad") {
            text.push_str("biquad_block! {\n    /// doc\n    BiquadBlock, BlockSample, BiquadState, Biquad, biquad\n}\n");
        }
        assert!(messages(&s).is_empty(), "{:?}", messages(&s));
    }

    #[test]
    fn a_kind_breaking_rule_1_fails() {
        let m = messages(&sources(
            "FmDemod",
            "FmDemod",
            "FmDemodBlock",
            "FmDemodKind",
        ));
        assert!(
            m.iter().any(|m| m.contains("kind `FmDemod` breaks rule 1")),
            "{m:?}"
        );
    }

    #[test]
    fn a_missing_block_type_fails() {
        let m = messages(&sources(
            "resample",
            "resample",
            "ResamplerBlock",
            "ResampleKind",
        ));
        assert_eq!(
            m,
            [
                "kind `resample`: `resample.rs` does not define `ResampleBlock` (rule 9 of the naming scheme)"
            ]
        );
    }

    #[test]
    fn a_wrong_file_or_kind_type_fails() {
        let m = messages(&sources(
            "halfband_decimate",
            "halfband",
            "HalfbandDecimateBlock",
            "HalfbandDecimateKind",
        ));
        assert!(
            m.iter()
                .any(|m| m.contains("is described in `halfband.rs`")),
            "{m:?}"
        );
        let m = messages(&sources(
            "halfband_decimate",
            "halfband_decimate",
            "HalfbandDecimateBlock",
            "HalfbandKind",
        ));
        assert!(
            m.iter().any(|m| m.contains("is the type `HalfbandKind`")),
            "{m:?}"
        );
    }

    #[test]
    fn descriptions_and_pipeline_kinds_must_match() {
        let mut s = sources("scale", "scale", "ScaleBlock", "ScaleKind");
        s.kinds_rs = String::new();
        let m = messages(&s);
        assert!(
            m.iter()
                .any(|m| m.contains("the block kind `scale` has no pipeline kind")),
            "{m:?}"
        );
    }
}
