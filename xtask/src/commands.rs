//! Command dispatch and the commands that spawn `cargo`.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::Path;
use std::process::Command;

use anyhow::{Context, bail};

use crate::layers;
use crate::scan::Finding;

const USAGE: &str = "usage: cargo xtask <check [layers|heap|global-state|names|docs] | bare-metal | schema [--check] | release-check <version> | test-all | emu [options] | count --platform <p> | measure <kernel> [options] | sde-cpuid <sse2|sse4.2> <file> | wasm-pkg [--relaxed] | wasm-test [--env e] [--variant v]>";

/// The bare-metal target used to prove that heap-free crates need no allocator.
pub const BARE_METAL_TARGET: &str = "thumbv7em-none-eabihf";

/// Runs a command line. Returns `Ok(true)` on success and `Ok(false)` if a check found violations.
///
/// # Errors
/// Fails on unknown commands or if a tool cannot be run.
pub fn run(root: &Path, args: &[String]) -> anyhow::Result<bool> {
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["check"] => check_all(root),
        ["check", "layers"] => Ok(print_findings("layers", &layers::check_layers(root)?)),
        ["check", "heap"] => Ok(print_findings("heap", &layers::check_heap(root)?)),
        ["check", "global-state"] => Ok(print_findings(
            "global-state",
            &layers::check_global_state(root)?,
        )),
        ["check", "names"] => Ok(print_findings("names", &crate::names::check_names(root)?)),
        ["check", "docs"] => check_docs(root),
        ["bare-metal"] => bare_metal(root),
        ["schema"] => schema(root, false),
        ["schema", "--check"] => schema(root, true),
        ["release-check", rest @ ..] => crate::release::run(root, rest),
        ["test-all"] => test_all(root),
        ["emu", rest @ ..] => crate::emu::run(root, rest),
        ["count", rest @ ..] => crate::count::run(root, rest),
        ["measure", rest @ ..] => crate::measure::run(root, rest),
        ["sde-cpuid", level, path] => crate::count::write_cpuid(level, Path::new(path)),
        ["wasm-pkg", rest @ ..] => crate::wasm::wasm_pkg(root, rest),
        ["wasm-test", rest @ ..] => crate::wasm::wasm_test(root, rest),
        _ => bail!("{USAGE}"),
    }
}

/// Prints findings to stderr, one line each; `true` if there are none.
#[must_use]
pub fn print_findings(name: &str, findings: &[Finding]) -> bool {
    for finding in findings {
        eprintln!("{finding}");
    }
    if findings.is_empty() {
        println!("check {name}: ok");
    } else {
        eprintln!("check {name}: {} violation(s)", findings.len());
    }
    findings.is_empty()
}

fn check_all(root: &Path) -> anyhow::Result<bool> {
    let mut ok = print_findings("layers", &layers::check_layers(root)?);
    ok &= print_findings("heap", &layers::check_heap(root)?);
    ok &= print_findings("global-state", &layers::check_global_state(root)?);
    ok &= print_findings("names", &crate::names::check_names(root)?);
    ok &= check_docs(root)?;
    Ok(ok)
}

fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

fn run_step(label: &str, mut cmd: Command, root: &Path) -> anyhow::Result<bool> {
    println!("==> {label}");
    let status = cmd
        .current_dir(root)
        .status()
        .with_context(|| format!("cannot run `{label}`"))?;
    if !status.success() {
        eprintln!("xtask: `{label}` failed");
    }
    Ok(status.success())
}

/// Path of the committed pipeline schema.
fn schema_path(root: &Path) -> std::path::PathBuf {
    root.join("schema").join("sdrk-pipeline-v1.json")
}

/// `schema`: writes `schema/sdrk-pipeline-v1.json` from the block descriptions; with
/// `--check` only compares it with the committed file.
fn schema(root: &Path, check: bool) -> anyhow::Result<bool> {
    let text = sdrk_pipeline::Registry::standard().json_schema_text();
    let path = schema_path(root);
    if check {
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        // The file may have CRLF endings on a checkout without `.gitattributes`.
        let same = committed.replace("\r\n", "\n") == text;
        if same {
            println!("schema: ok");
        } else {
            eprintln!(
                "schema: {} is out of date; run `cargo xtask schema`",
                path.display()
            );
        }
        return Ok(same);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    }
    std::fs::write(&path, &text).with_context(|| format!("cannot write {}", path.display()))?;
    println!("schema: wrote {}", path.display());
    Ok(true)
}

fn check_docs(root: &Path) -> anyhow::Result<bool> {
    let mut cmd = cargo();
    cmd.args(["doc", "--workspace", "--no-deps"])
        .env("RUSTDOCFLAGS", "-D warnings");
    run_step(
        "cargo doc --workspace --no-deps (RUSTDOCFLAGS=-D warnings)",
        cmd,
        root,
    )
}

fn bare_metal(root: &Path) -> anyhow::Result<bool> {
    let crates = layers::heap_free_crates(root)?;
    let mut ok = true;
    for name in crates {
        let mut cmd = cargo();
        cmd.args([
            "build",
            "--target",
            BARE_METAL_TARGET,
            "-p",
            &name,
            "--no-default-features",
        ]);
        let passed = run_step(
            &format!("cargo build --target {BARE_METAL_TARGET} -p {name} --no-default-features"),
            cmd,
            root,
        )?;
        if !passed {
            eprintln!(
                "xtask: if the target is missing, run `rustup target add {BARE_METAL_TARGET}`"
            );
        }
        ok &= passed;
    }
    Ok(ok)
}

/// `test-all`: the commands of R1.2, the tests of the WebAssembly package when Node.js is
/// present, and the tier 1 emulator run (`emu --tier 1`).
fn test_all(root: &Path) -> anyhow::Result<bool> {
    let steps: [(&str, &[&str]); 3] = [
        (
            "cargo fmt --all -- --check",
            &["fmt", "--all", "--", "--check"],
        ),
        (
            "cargo clippy --workspace --all-targets --all-features -- -D warnings",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ],
        ),
        (
            "cargo test --workspace --all-features",
            &["test", "--workspace", "--all-features"],
        ),
    ];
    let mut ok = true;
    for (label, args) in steps {
        let mut cmd = cargo();
        cmd.args(args);
        ok &= run_step(label, cmd, root)?;
    }
    ok &= check_all(root)?;
    ok &= bare_metal(root)?;
    ok &= wasm_package_tests(root)?;
    println!("==> cargo xtask emu --tier 1");
    ok &= crate::emu::run(root, &["--tier", "1"])?;
    Ok(ok)
}

/// The WebAssembly package and its JavaScript tests, if Node.js is installed.
fn wasm_package_tests(root: &Path) -> anyhow::Result<bool> {
    let Some(path) = std::env::var_os("PATH") else {
        return Ok(true);
    };
    let Some(node) = crate::wasm::find_program("node", &path) else {
        println!("==> wasm-pkg tests skipped: node is not in PATH");
        return Ok(true);
    };
    println!("==> cargo xtask wasm-pkg");
    if !crate::wasm::wasm_pkg(root, &[])? {
        return Ok(false);
    }
    let mut cmd = Command::new(node);
    cmd.args(["--test", "bindings/wasm/js-tests/*.test.mjs"]);
    run_step("node --test bindings/wasm/js-tests", cmd, root)
}
