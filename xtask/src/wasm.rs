//! `cargo xtask wasm-pkg` and `cargo xtask wasm-test`: the WebAssembly package of
//! `bindings/wasm` and its tests.
//!
//! No `wasm-pack` (its upstream organisation was archived): the package is built by calling
//! `cargo build --target wasm32-unknown-unknown --release` and `wasm-bindgen --target web`
//! directly, once per variant:
//!
//! | Variant | `RUSTFLAGS` | Cargo features |
//! | --- | --- | --- |
//! | `simd` | `-C target-feature=+simd128` | none |
//! | `scalar` | `--cfg sdrk_fallback_only` | none |
//! | `relaxed` (only with `--relaxed`) | `-C target-feature=+simd128,+relaxed-simd` | `relaxed-simd` |
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};

const USAGE_PKG: &str = "usage: cargo xtask wasm-pkg [--relaxed] [--no-opt] [--no-pack]";
const USAGE_TEST: &str =
    "usage: cargo xtask wasm-test [--env node|chrome|firefox] [--variant simd|scalar|relaxed|all]";

/// The Rust target of the package.
pub const WASM_TARGET: &str = "wasm32-unknown-unknown";

/// Soft size limit of a variant after `wasm-opt`, in bytes.
pub const SOFT_SIZE_LIMIT: u64 = 3 * 512 * 1024;

/// One build of the bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variant {
    /// Name; the files of the package are `sdrk_<name>.js` and `sdrk_<name>.wasm`.
    pub name: &'static str,
    /// Value of `RUSTFLAGS`.
    pub rustflags: &'static str,
    /// Cargo features of `sdrk-wasm`.
    pub features: &'static [&'static str],
}

/// The scalar build: the portable fallback only, no SIMD instructions.
pub const SCALAR: Variant = Variant {
    name: "scalar",
    rustflags: "--cfg sdrk_fallback_only",
    features: &[],
};

/// The SIMD128 build.
pub const SIMD: Variant = Variant {
    name: "simd",
    rustflags: "-C target-feature=+simd128",
    features: &[],
};

/// The SIMD128 build with relaxed SIMD, selectable only explicitly.
pub const RELAXED: Variant = Variant {
    name: "relaxed",
    rustflags: "-C target-feature=+simd128,+relaxed-simd",
    features: &["relaxed-simd"],
};

/// The variants of a package build.
pub fn variants(relaxed: bool) -> Vec<Variant> {
    let mut all = vec![SIMD, SCALAR];
    if relaxed {
        all.push(RELAXED);
    }
    all
}

/// The version `x.y.z` of the line `wasm-bindgen = "=x.y.z"` in the manifest of `sdrk-wasm`.
pub fn pinned_bindgen_version(manifest: &str) -> Option<String> {
    manifest.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("wasm-bindgen")?.trim_start();
        let value = rest.strip_prefix('=')?.trim();
        let value = value.trim_matches('"');
        let version = value.strip_prefix('=')?;
        version
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.')
            .then(|| version.to_string())
    })
}

/// The version in the output of `wasm-bindgen --version` (`wasm-bindgen 0.2.129`).
pub fn cli_version(output: &str) -> Option<String> {
    let mut words = output.split_whitespace();
    (words.next()? == "wasm-bindgen").then(|| words.next().map(str::to_string))?
}

/// The workspace version from the root manifest (`[workspace.package]`).
pub fn workspace_version(manifest: &str) -> Option<String> {
    let mut in_section = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_section = line == "[workspace.package]";
        } else if in_section && let Some(rest) = line.strip_prefix("version") {
            let value = rest.trim_start().strip_prefix('=')?.trim();
            return Some(value.trim_matches('"').to_string());
        }
    }
    None
}

/// Fails if `wasm-bindgen-cli` differs from the pinned library version.
fn check_bindgen_version(root: &Path) -> anyhow::Result<()> {
    let manifest = std::fs::read_to_string(root.join("bindings").join("wasm").join("Cargo.toml"))
        .context("cannot read bindings/wasm/Cargo.toml")?;
    let pinned = pinned_bindgen_version(&manifest)
        .context("bindings/wasm/Cargo.toml must pin wasm-bindgen as `=x.y.z`")?;
    let output = Command::new("wasm-bindgen")
        .arg("--version")
        .output()
        .context("cannot run `wasm-bindgen` (install it with `cargo install wasm-bindgen-cli --version =x.y.z`)")?;
    let text = String::from_utf8_lossy(&output.stdout);
    let installed = cli_version(&text).with_context(|| format!("unexpected output: {text}"))?;
    if installed != pinned {
        bail!(
            "wasm-bindgen-cli is {installed} but the crate pins wasm-bindgen {pinned}; \
             run `cargo install wasm-bindgen-cli --version {pinned} --locked`"
        );
    }
    println!("wasm-bindgen {installed} (matches the pinned library version)");
    Ok(())
}

fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

fn run(label: &str, cmd: &mut Command, root: &Path) -> anyhow::Result<bool> {
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

/// True if `program --version` runs.
fn available(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn npm() -> &'static str {
    if cfg!(windows) { "npm.cmd" } else { "npm" }
}

/// Directory of the assembled package.
pub fn package_dir(root: &Path) -> PathBuf {
    root.join("target").join("wasm-pkg")
}

/// Directory of the tarball made by `npm pack`.
pub fn tarball_dir(root: &Path) -> PathBuf {
    root.join("target").join("wasm-pkg-tgz")
}

/// Builds the module of `variant` with cargo; returns the path of `sdrk_wasm.wasm`.
fn cargo_build(root: &Path, variant: Variant) -> anyhow::Result<Option<PathBuf>> {
    let target_dir = root
        .join("target")
        .join("wasm-pkg-build")
        .join(variant.name);
    let mut build = cargo();
    build
        .args([
            "build",
            "-p",
            "sdrk-wasm",
            "--release",
            "--target",
            WASM_TARGET,
        ])
        .arg("--target-dir")
        .arg(&target_dir)
        .env("RUSTFLAGS", variant.rustflags);
    if !variant.features.is_empty() {
        build.arg("--features").arg(variant.features.join(","));
    }
    let label = format!("cargo build -p sdrk-wasm --release ({})", variant.name);
    if !run(&label, &mut build, root)? {
        return Ok(None);
    }
    Ok(Some(
        target_dir
            .join(WASM_TARGET)
            .join("release")
            .join("sdrk_wasm.wasm"),
    ))
}

/// Runs `wasm-bindgen --target web` on `wasm`, writing `sdrk_<variant>.js` and
/// `sdrk_<variant>_bg.wasm` to `out`.
fn bindgen(root: &Path, variant: Variant, wasm: &Path, out: &Path) -> anyhow::Result<bool> {
    let mut cmd = Command::new("wasm-bindgen");
    cmd.args(["--target", "web", "--no-typescript", "--out-name"])
        .arg(format!("sdrk_{}", variant.name))
        .arg("--out-dir")
        .arg(out)
        .arg(wasm);
    run(&format!("wasm-bindgen ({})", variant.name), &mut cmd, root)
}

/// Optimises `from` into `to` with `wasm-opt -O3` if it is in PATH, else copies it.
fn optimise(root: &Path, name: &str, from: &Path, to: &Path, wanted: bool) -> anyhow::Result<()> {
    if wanted && available("wasm-opt") {
        let mut cmd = Command::new("wasm-opt");
        cmd.args(["-O3", "--all-features"])
            .arg(from)
            .arg("-o")
            .arg(to);
        if run(&format!("wasm-opt -O3 ({name})"), &mut cmd, root)? {
            std::fs::remove_file(from).ok();
            return Ok(());
        }
        eprintln!("xtask: wasm-opt failed; keeping the unoptimised module");
    } else if wanted {
        println!("wasm-opt is not in PATH: the module is not optimised");
    }
    std::fs::rename(from, to).with_context(|| format!("cannot move {}", from.display()))
}

fn build_variant(root: &Path, variant: Variant, out: &Path, wanted: bool) -> anyhow::Result<bool> {
    let Some(wasm) = cargo_build(root, variant)? else {
        return Ok(false);
    };
    if !bindgen(root, variant, &wasm, out)? {
        return Ok(false);
    }
    // The package names the files `sdrk_<variant>.wasm`.
    let name = format!("sdrk_{}", variant.name);
    optimise(
        root,
        variant.name,
        &out.join(format!("{name}_bg.wasm")),
        &out.join(format!("{name}.wasm")),
        wanted,
    )?;
    let js_path = out.join(format!("{name}.js"));
    let js = std::fs::read_to_string(&js_path)
        .with_context(|| format!("cannot read {}", js_path.display()))?;
    std::fs::write(
        &js_path,
        js.replace(&format!("{name}_bg.wasm"), &format!("{name}.wasm")),
    )?;
    Ok(true)
}

fn copy_template(root: &Path, out: &Path) -> anyhow::Result<()> {
    let template = root.join("bindings").join("wasm").join("pkg-template");
    let version = workspace_version(&std::fs::read_to_string(root.join("Cargo.toml"))?)
        .context("no version in [workspace.package]")?;
    for entry in std::fs::read_dir(&template)
        .with_context(|| format!("cannot read {}", template.display()))?
    {
        let entry = entry?;
        let to = out.join(entry.file_name());
        if entry.file_name() == "package.json" {
            let text = std::fs::read_to_string(entry.path())?;
            std::fs::write(&to, text.replace("__VERSION__", &version))?;
        } else {
            std::fs::copy(entry.path(), &to)?;
        }
    }
    for licence in ["LICENSE-MIT", "LICENSE-APACHE"] {
        std::fs::copy(root.join(licence), out.join(licence))
            .with_context(|| format!("cannot copy {licence}"))?;
    }
    Ok(())
}

/// `wasm-pkg`: builds the variants, assembles `target/wasm-pkg` and packs it with `npm pack`.
///
/// # Errors
/// Fails if a tool is missing or the `wasm-bindgen-cli` version differs from the library.
pub fn wasm_pkg(root: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let (mut relaxed, mut optimise, mut pack) = (false, true, true);
    for arg in args {
        match *arg {
            "--relaxed" => relaxed = true,
            "--no-opt" => optimise = false,
            "--no-pack" => pack = false,
            _ => bail!("{USAGE_PKG}"),
        }
    }
    check_bindgen_version(root)?;
    let out = package_dir(root);
    if out.exists() {
        std::fs::remove_dir_all(&out)?;
    }
    std::fs::create_dir_all(&out)?;
    for variant in variants(relaxed) {
        if !build_variant(root, variant, &out, optimise)? {
            return Ok(false);
        }
    }
    copy_template(root, &out)?;
    let mut ok = true;
    println!("\nsizes of the modules:");
    for variant in variants(relaxed) {
        let file = out.join(format!("sdrk_{}.wasm", variant.name));
        let size = std::fs::metadata(&file)?.len();
        let warn = if size > SOFT_SIZE_LIMIT {
            "  (WARNING: above the soft limit of 1.5 MiB)"
        } else {
            ""
        };
        println!("  {}: {size} bytes{warn}", file.display());
    }
    if pack {
        if available(npm()) {
            let dest = tarball_dir(root);
            std::fs::create_dir_all(&dest)?;
            let mut cmd = Command::new(npm());
            cmd.args(["pack", "--pack-destination"])
                .arg(&dest)
                .current_dir(&out);
            println!("==> npm pack");
            let status = cmd.status().context("cannot run `npm pack`")?;
            if !status.success() {
                eprintln!("xtask: `npm pack` failed");
                ok = false;
            }
        } else {
            println!("npm is not in PATH: the tarball is not made");
        }
    }
    println!("wasm-pkg: {}", out.display());
    Ok(ok)
}

/// The browser or runtime of `wasm-test`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestEnv {
    /// Node.js (the default of `wasm-bindgen-test-runner`).
    Node,
    /// Headless Chrome through chromedriver.
    Chrome,
    /// Headless Firefox through geckodriver.
    Firefox,
}

impl TestEnv {
    /// Parses `node`, `chrome` or `firefox`.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "node" => Self::Node,
            "chrome" => Self::Chrome,
            "firefox" => Self::Firefox,
            _ => return None,
        })
    }

    /// Environment variables of `wasm-bindgen-test-runner` for this environment: `(set, unset)`.
    pub fn runner_env(self) -> (Vec<(&'static str, &'static str)>, Vec<&'static str>) {
        match self {
            Self::Node => (
                Vec::new(),
                vec!["WASM_BINDGEN_USE_BROWSER", "CHROMEDRIVER", "GECKODRIVER"],
            ),
            Self::Chrome => (
                vec![("WASM_BINDGEN_USE_BROWSER", "1")],
                vec!["GECKODRIVER", "SAFARIDRIVER", "MSEDGEDRIVER"],
            ),
            Self::Firefox => (
                vec![("WASM_BINDGEN_USE_BROWSER", "1")],
                vec!["CHROMEDRIVER", "SAFARIDRIVER", "MSEDGEDRIVER"],
            ),
        }
    }
}

impl TestEnv {
    /// `(environment variable, program)` of the web driver for a browser run.
    pub fn driver(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::Node => None,
            Self::Chrome => Some(("CHROMEDRIVER", "chromedriver")),
            Self::Firefox => Some(("GECKODRIVER", "geckodriver")),
        }
    }
}

/// The first file called `program` (`program.exe` on Windows) in the directories of `path`.
pub fn find_program(program: &str, path: &std::ffi::OsStr) -> Option<PathBuf> {
    let file = if cfg!(windows) {
        format!("{program}.exe")
    } else {
        program.to_string()
    };
    std::env::split_paths(path)
        .map(|dir| dir.join(&file))
        .find(|candidate| candidate.is_file())
}

/// The variant named `name`.
pub fn variant_named(name: &str) -> Option<Variant> {
    [SIMD, SCALAR, RELAXED].into_iter().find(|v| v.name == name)
}

/// `wasm-test`: runs `tests/wasm.rs` of `sdrk-wasm` under `wasm-bindgen-test-runner`.
///
/// # Errors
/// Fails on bad arguments or if a tool cannot be run.
pub fn wasm_test(root: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let mut env = TestEnv::Node;
    let mut chosen: Vec<Variant> = vec![SIMD, SCALAR];
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match *arg {
            "--env" => {
                let name = iter.next().context(USAGE_TEST)?;
                env = TestEnv::parse(name).with_context(|| USAGE_TEST.to_string())?;
            }
            "--variant" => {
                let name = iter.next().context(USAGE_TEST)?;
                chosen = if *name == "all" {
                    vec![SIMD, SCALAR, RELAXED]
                } else {
                    vec![variant_named(name).with_context(|| USAGE_TEST.to_string())?]
                };
            }
            _ => bail!("{USAGE_TEST}"),
        }
    }
    check_bindgen_version(root)?;
    let mut ok = true;
    for variant in chosen {
        let target_dir = root.join("target").join("wasm-test").join(variant.name);
        let mut cmd = cargo();
        cmd.args([
            "test",
            "-p",
            "sdrk-wasm",
            "--release",
            "--test",
            "wasm",
            "--target",
            WASM_TARGET,
        ])
        .arg("--target-dir")
        .arg(&target_dir)
        .env("RUSTFLAGS", variant.rustflags)
        .env(
            "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER",
            "wasm-bindgen-test-runner",
        );
        if !variant.features.is_empty() {
            cmd.arg("--features").arg(variant.features.join(","));
        }
        // Name the driver explicitly: with both drivers in `PATH` the runner would otherwise
        // pick one on its own.
        if let Some((var, program)) = env.driver()
            && std::env::var_os(var).is_none()
        {
            let path = std::env::var_os("PATH").unwrap_or_default();
            let found = find_program(program, &path)
                .with_context(|| format!("{program} is not in PATH and {var} is not set"))?;
            cmd.env(var, found);
        }
        let (set, unset) = env.runner_env();
        for (key, value) in set {
            cmd.env(key, value);
        }
        for key in unset {
            cmd.env_remove(key);
        }
        ok &= run(
            &format!("wasm-bindgen-test ({} in {env:?})", variant.name),
            &mut cmd,
            root,
        )?;
    }
    Ok(ok)
}
