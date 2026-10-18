//! Negative tests: every check must fail on a small fake workspace that breaks its rule,
//! and pass on the same workspace without the violation.

#![allow(clippy::expect_used)]
use std::fs;
use std::path::Path;

use tempfile::TempDir;
use xtask::layers::{check_global_state, check_heap, check_layers};

/// One crate of a fake workspace: (directory, name, dependencies as `name = { path = ... }` lines, lib.rs).
struct Krate<'a> {
    dir: &'a str,
    name: &'a str,
    deps: &'a str,
    lib: &'a str,
}

fn workspace(crates: &[Krate<'_>]) -> TempDir {
    let tmp = TempDir::new().expect("tempdir");
    let members: Vec<String> = crates.iter().map(|c| format!("\"{}\"", c.dir)).collect();
    write(
        tmp.path(),
        "Cargo.toml",
        &format!(
            "[workspace]\nresolver = \"3\"\nmembers = [{}]\n",
            members.join(", ")
        ),
    );
    for c in crates {
        let manifest = format!(
            "[package]\nname = \"{}\"\nversion = \"0.0.1\"\nedition = \"2024\"\n\n[dependencies]\n{}\n",
            c.name, c.deps
        );
        write(tmp.path(), &format!("{}/Cargo.toml", c.dir), &manifest);
        write(tmp.path(), &format!("{}/src/lib.rs", c.dir), c.lib);
    }
    tmp
}

fn write(root: &Path, rel: &str, content: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, content).expect("write");
}

const DEVICE: &str = "crates/core/sdrk-device";
const CORE: &str = "crates/core/sdrk-core";

fn device(deps: &str) -> Krate<'_> {
    Krate {
        dir: DEVICE,
        name: "sdrk-device",
        deps,
        lib: "#![no_std]\n",
    }
}

fn core<'a>(deps: &'a str, lib: &'a str) -> Krate<'a> {
    Krate {
        dir: CORE,
        name: "sdrk-core",
        deps,
        lib,
    }
}

#[test]
fn layers_accepts_correct_direction() {
    let ws = workspace(&[
        device(""),
        core(
            "sdrk-device = { path = \"../sdrk-device\" }",
            "#![no_std]\n",
        ),
    ]);
    assert_eq!(check_layers(ws.path()).expect("run"), []);
}

#[test]
fn layers_rejects_device_depending_on_core() {
    let ws = workspace(&[
        device("sdrk-core = { path = \"../sdrk-core\" }"),
        core("", "#![no_std]\n"),
    ]);
    let findings = check_layers(ws.path()).expect("run");
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0].message.contains("sdrk-device"));
}

#[test]
fn layers_rejects_reference_feature_in_library_crates() {
    let facade = Krate {
        dir: "crates/facade/sdrk",
        name: "sdrk",
        deps: "sdrk-core = { path = \"../../core/sdrk-core\", features = [\"reference\"] }",
        lib: "#![no_std]\n",
    };
    let core_with_feature = Krate {
        dir: CORE,
        name: "sdrk-core",
        deps: "",
        lib: "#![no_std]\n",
    };
    let ws = workspace(&[core_with_feature, facade]);
    // the fake core has no `reference` feature, so cargo metadata would not enforce it; the check must still fail
    let findings = check_layers(ws.path()).expect("run");
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0].message.contains("reference"));
}

#[test]
fn heap_rejects_vec_in_heap_free_crate() {
    let clean = workspace(&[core("", "#![no_std]\npub fn f() {}\n")]);
    assert_eq!(check_heap(clean.path()).expect("run"), []);
    let dirty = workspace(&[core(
        "",
        "#![no_std]\npub fn f() { let v: Vec<u8> = Vec::new(); drop(v); }\n",
    )]);
    assert_ne!(check_heap(dirty.path()).expect("run"), []);
}

#[test]
fn heap_allows_vec_in_pipeline_crate() {
    let pipeline = Krate {
        dir: "crates/pipeline/sdrk-pipeline",
        name: "sdrk-pipeline",
        deps: "",
        lib: "pub fn f() { let v: Vec<u8> = Vec::new(); drop(v); }\n",
    };
    let ws = workspace(&[pipeline]);
    assert_eq!(check_heap(ws.path()).expect("run"), []);
}

#[test]
fn heap_exempts_out_of_line_test_module_but_not_plain_module() {
    let lib = "#![no_std]\n#[cfg(test)]\nmod tests;\n";
    let ws = workspace(&[core("", lib)]);
    write(
        ws.path(),
        "crates/core/sdrk-core/src/tests.rs",
        "fn t() { let _v: Vec<u8> = Vec::new(); }\n",
    );
    assert_eq!(check_heap(ws.path()).expect("run"), []);

    let lib = "#![no_std]\nmod plain;\n";
    let ws = workspace(&[core("", lib)]);
    write(
        ws.path(),
        "crates/core/sdrk-core/src/plain.rs",
        "fn t() { let _v: Vec<u8> = Vec::new(); }\n",
    );
    assert_ne!(check_heap(ws.path()).expect("run"), []);
}

#[test]
fn global_state_rejects_atomic_static() {
    let clean = workspace(&[core(
        "",
        "#![no_std]\npub static TABLE: [u8; 2] = [1, 2];\n",
    )]);
    assert_eq!(check_global_state(clean.path()).expect("run"), []);
    let dirty = workspace(&[core(
        "",
        "#![no_std]\nuse core::sync::atomic::AtomicU32;\nstatic COUNTER: AtomicU32 = AtomicU32::new(0);\n",
    )]);
    assert_ne!(check_global_state(dirty.path()).expect("run"), []);
}

#[test]
fn global_state_ignores_test_code() {
    let lib = "#![no_std]\n#[cfg(test)]\nmod tests {\n    use core::sync::atomic::AtomicU32;\n    static C: AtomicU32 = AtomicU32::new(0);\n}\n";
    let ws = workspace(&[core("", lib)]);
    assert_eq!(check_global_state(ws.path()).expect("run"), []);
}
