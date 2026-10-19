//! Layer model of the workspace and the architecture checks built on it.

use std::collections::BTreeMap;
use std::path::{Component, Path};

use anyhow::Context;
use cargo_metadata::{DependencyKind, MetadataCommand, Package};

use crate::scan::{Finding, Scan, StdPolicy, scan_crate};

/// Layer of a workspace crate; the folder of the crate decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer {
    /// `crates/core/sdrk-device` (rank 0).
    Device,
    /// `crates/core/*` except the device crate (rank 1).
    Core,
    /// `crates/backends/*`.
    Backend,
    /// `crates/facade/*`.
    Facade,
    /// `crates/blocks/*`.
    Blocks,
    /// `crates/pipeline/*` (heap allowed).
    Pipeline,
    /// `crates/io/*` (heap allowed).
    Io,
    /// `crates/testing/*`; may depend on any library layer.
    Testing,
    /// Everything else (bindings, examples, targets, benches, xtask, tests).
    App,
}

impl Layer {
    /// Dependency rank; a crate may only depend on strictly lower ranks.
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            Layer::Device => 0,
            Layer::Core => 1,
            Layer::Backend => 2,
            Layer::Facade => 3,
            Layer::Blocks => 4,
            Layer::Pipeline | Layer::Io => 5,
            Layer::Testing => 6,
            Layer::App => 7,
        }
    }

    /// Whether crates of this layer must be `no_std` without heap.
    #[must_use]
    pub fn heap_free(self) -> bool {
        matches!(
            self,
            Layer::Device
                | Layer::Core
                | Layer::Backend
                | Layer::Facade
                | Layer::Blocks
                | Layer::Testing
        )
    }

    /// Policy for `std` inside `#[cfg(feature = "std")]` items of this layer.
    #[must_use]
    pub fn std_policy(self) -> StdPolicy {
        match self {
            Layer::Testing => StdPolicy::Full,
            Layer::Backend => StdPolicy::ArchOnly,
            _ => StdPolicy::Forbidden,
        }
    }

    /// Whether a crate of this layer may use a normal or build dependency on `to`.
    #[must_use]
    pub fn may_depend_on(self, to: Layer) -> bool {
        match self {
            Layer::App => true,
            Layer::Testing => !matches!(to, Layer::App | Layer::Testing),
            _ => !matches!(to, Layer::Testing | Layer::App) && to.rank() < self.rank(),
        }
    }
}

/// Maps a crate directory (relative to the workspace root) and name to its layer.
#[must_use]
pub fn layer_of(rel_dir: &Path, name: &str) -> Layer {
    let parts: Vec<String> = rel_dir
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str().map(str::to_string),
            _ => None,
        })
        .collect();
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    match parts.as_slice() {
        ["crates", "core", ..] if name == "sdrk-device" => Layer::Device,
        ["crates", "core", ..] => Layer::Core,
        ["crates", "backends", ..] => Layer::Backend,
        ["crates", "facade", ..] => Layer::Facade,
        ["crates", "blocks", ..] => Layer::Blocks,
        ["crates", "pipeline", ..] => Layer::Pipeline,
        ["crates", "io", ..] => Layer::Io,
        ["crates", "testing", ..] => Layer::Testing,
        // Everything else is the top layer: `examples/*`, `tests/system` (system tests),
        // `bindings/*`, `targets/*`, `benches/*`, `xtask`.
        _ => Layer::App,
    }
}

/// A workspace crate with its layer.
#[derive(Debug, Clone)]
pub struct Member {
    /// Cargo package.
    pub package: Box<Package>,
    /// Layer derived from the folder.
    pub layer: Layer,
}

/// Reads the workspace members (without resolving dependencies, so it works offline).
///
/// # Errors
/// Fails if `cargo metadata` fails.
#[allow(
    clippy::large_stack_frames,
    reason = "cargo_metadata::Metadata is large; xtask is a host tool"
)]
pub fn members(root: &Path) -> anyhow::Result<(std::path::PathBuf, Vec<Member>)> {
    let metadata = MetadataCommand::new()
        .manifest_path(root.join("Cargo.toml"))
        .no_deps()
        .exec()
        .context("cargo metadata failed")?;
    let ws_root = metadata.workspace_root.as_std_path().to_path_buf();
    let mut out = Vec::new();
    for package in metadata.packages {
        let dir = package
            .manifest_path
            .as_std_path()
            .parent()
            .unwrap_or(Path::new(""))
            .to_path_buf();
        let rel = dir.strip_prefix(&ws_root).unwrap_or(&dir).to_path_buf();
        let layer = layer_of(&rel, &package.name);
        out.push(Member {
            package: Box::new(package),
            layer,
        });
    }
    out.sort_by(|a, b| a.package.name.cmp(&b.package.name));
    Ok((ws_root, out))
}

/// Names of all heap-free crates (derived from the layer table).
///
/// # Errors
/// Fails if `cargo metadata` fails.
pub fn heap_free_crates(root: &Path) -> anyhow::Result<Vec<String>> {
    Ok(members(root)?
        .1
        .into_iter()
        .filter(|m| m.layer.heap_free())
        .map(|m| m.package.name.to_string())
        .collect())
}

/// `check layers`: dependency direction between layers and the `reference` feature.
///
/// # Errors
/// Fails if `cargo metadata` fails.
pub fn check_layers(root: &Path) -> anyhow::Result<Vec<Finding>> {
    let (_, members) = members(root)?;
    let by_name: BTreeMap<String, Layer> = members
        .iter()
        .map(|m| (m.package.name.to_string(), m.layer))
        .collect();
    let mut findings = Vec::new();
    for m in &members {
        let manifest = m.package.manifest_path.as_std_path().to_path_buf();
        let name = &m.package.name;
        for dep in &m.package.dependencies {
            if dep.kind == DependencyKind::Development {
                continue;
            }
            let Some(&to) = by_name.get(&dep.name) else {
                continue;
            };
            if !m.layer.may_depend_on(to) {
                findings.push(Finding {
                    path: manifest.clone(),
                    line: 0,
                    message: format!(
                        "crate `{name}` ({:?}) must not depend on `{}` ({to:?})",
                        m.layer, dep.name
                    ),
                });
            }
            let reference_user_ok = matches!(m.layer, Layer::Testing | Layer::App);
            if dep.name == "sdrk-core"
                && dep.features.iter().any(|f| f == "reference")
                && !reference_user_ok
            {
                findings.push(Finding {
                    path: manifest.clone(),
                    line: 0,
                    message: format!("crate `{name}` enables `sdrk-core/reference`; only the testkit and test code may"),
                });
            }
        }
        if !matches!(m.layer, Layer::Testing | Layer::App) {
            for (feature, values) in &m.package.features {
                if values
                    .iter()
                    .any(|v| v == "sdrk-core/reference" || v == "sdrk-core?/reference")
                {
                    findings.push(Finding {
                        path: manifest.clone(),
                        line: 0,
                        message: format!(
                            "feature `{feature}` of `{name}` enables `sdrk-core/reference`"
                        ),
                    });
                }
            }
        }
    }
    Ok(findings)
}

/// `check heap`: forbidden allocation and `std` use in heap-free crates.
///
/// # Errors
/// Fails if a source file cannot be read.
pub fn check_heap(root: &Path) -> anyhow::Result<Vec<Finding>> {
    let (_, members) = members(root)?;
    let mut findings = Vec::new();
    for m in members.iter().filter(|m| m.layer.heap_free()) {
        let src = m
            .package
            .manifest_path
            .as_std_path()
            .parent()
            .unwrap_or(Path::new(""))
            .join("src");
        if src.is_dir() {
            findings.extend(scan_crate(&src, Scan::Heap(m.layer.std_policy()))?);
        }
    }
    Ok(findings)
}

/// `check global-state`: no global mutable state in library crates.
///
/// # Errors
/// Fails if a source file cannot be read.
pub fn check_global_state(root: &Path) -> anyhow::Result<Vec<Finding>> {
    let (_, members) = members(root)?;
    let mut findings = Vec::new();
    for m in members.iter().filter(|m| m.layer != Layer::App) {
        let is_lib = m.package.targets.iter().any(cargo_metadata::Target::is_lib);
        let src = m
            .package
            .manifest_path
            .as_std_path()
            .parent()
            .unwrap_or(Path::new(""))
            .join("src");
        if is_lib && src.is_dir() {
            findings.extend(scan_crate(&src, Scan::GlobalState)?);
        }
    }
    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folders_map_to_layers() {
        let l = |p: &str, n: &str| layer_of(Path::new(p), n);
        assert_eq!(l("crates/core/sdrk-device", "sdrk-device"), Layer::Device);
        assert_eq!(l("crates/core/sdrk-core", "sdrk-core"), Layer::Core);
        assert_eq!(
            l("crates/backends/sdrk-backend-simd", "sdrk-backend-simd"),
            Layer::Backend
        );
        assert_eq!(l("crates/facade/sdrk", "sdrk"), Layer::Facade);
        assert_eq!(l("crates/blocks/sdrk-blocks", "sdrk-blocks"), Layer::Blocks);
        assert_eq!(
            l("crates/pipeline/sdrk-pipeline", "sdrk-pipeline"),
            Layer::Pipeline
        );
        assert_eq!(l("crates/io/sdrk-io", "sdrk-io"), Layer::Io);
        assert_eq!(
            l("crates/testing/sdrk-testkit", "sdrk-testkit"),
            Layer::Testing
        );
        assert_eq!(l("xtask", "xtask"), Layer::App);
        assert_eq!(l("examples/fm-receiver", "fm-receiver"), Layer::App);
        assert_eq!(l("tests/system", "sdrk-system-tests"), Layer::App);
    }

    #[test]
    fn dependency_rules() {
        assert!(Layer::Core.may_depend_on(Layer::Device));
        assert!(!Layer::Device.may_depend_on(Layer::Core));
        assert!(!Layer::Core.may_depend_on(Layer::Core));
        assert!(!Layer::Backend.may_depend_on(Layer::Backend));
        assert!(Layer::Facade.may_depend_on(Layer::Backend));
        assert!(!Layer::Backend.may_depend_on(Layer::Facade));
        assert!(Layer::Pipeline.may_depend_on(Layer::Blocks));
        assert!(!Layer::Pipeline.may_depend_on(Layer::Io));
        assert!(!Layer::Io.may_depend_on(Layer::Pipeline));
        assert!(Layer::Testing.may_depend_on(Layer::Pipeline));
        assert!(!Layer::Pipeline.may_depend_on(Layer::Testing));
        assert!(Layer::App.may_depend_on(Layer::Testing));
    }

    #[test]
    fn heap_free_layers() {
        assert!(Layer::Blocks.heap_free());
        assert!(Layer::Testing.heap_free());
        assert!(!Layer::Pipeline.heap_free());
        assert!(!Layer::Io.heap_free());
        assert!(!Layer::App.heap_free());
    }
}
