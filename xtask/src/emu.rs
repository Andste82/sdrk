//! `cargo xtask emu`: builds the embedded test targets in `targets/`, runs them in an emulator
//! and decides pass or fail from the line protocol of the testkit suite.
//!
//! ```text
//! cargo xtask emu [--target <name>]... [--tier 1|2] [--all] [--full]
//!                 [--features a,b] [--build-only] [--timeout <s>] [--skip-missing] [--install] [--list]
//! ```
//!
//! The protocol (see `sdrk_testkit::suite`) is
//!
//! ```text
//! SDRK-SUITE START <n>
//! SDRK-CASE PASS <name>
//! SDRK-CASE FAIL <name> [detail]
//! SDRK-SUITE END passed=<p> failed=<f>
//! SDRK-PANIC <message>
//! ```
//!
//! [`judge`] reads nothing but these lines: a run passes when `END` is present with `failed=0`,
//! and there is no `FAIL` and no `PANIC` line. The exit code of the emulator is never used.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::large_stack_frames,
    reason = "xtask is a host tool; std::process::Command and the target table are large"
)]

use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use sha2::{Digest, Sha256};

const USAGE: &str = "usage: cargo xtask emu [--target <name>]... [--tier 1|2] [--all] [--full] \
[--features a,b] [--build-only] [--timeout <s>] [--skip-missing] [--install] [--list] [--judge <log>]";

/// Default timeout of one target in seconds: quick profile.
pub const QUICK_TIMEOUT_S: u64 = 300;
/// Default timeout of one target in seconds: full profile.
pub const FULL_TIMEOUT_S: u64 = 3600;

// ---------------------------------------------------------------------------------------------
// The protocol

/// What a log says about the run of the suite.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Verdict {
    /// `true` if the run counts as passed.
    pub ok: bool,
    /// The count of `SDRK-SUITE START`, if seen.
    pub started: Option<u32>,
    /// `passed=` of the `END` line.
    pub passed: u32,
    /// `failed=` of the `END` line.
    pub failed: u32,
    /// Number of `SDRK-CASE PASS` lines.
    pub pass_lines: u32,
    /// The `SDRK-CASE FAIL` lines (without the marker).
    pub fail_lines: Vec<String>,
    /// The `SDRK-PANIC` lines (without the marker).
    pub panics: Vec<String>,
    /// An `END` line was seen.
    pub ended: bool,
    /// Why the run does not pass; empty if it does.
    pub reasons: Vec<String>,
}

/// Removes ANSI escape sequences (`ESC [ ... letter`) from a line of console output.
#[must_use]
pub fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for d in chars.by_ref() {
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// The text of `line` from the first protocol marker on (a console may put a prefix in front),
/// with ANSI sequences removed and surrounding white space trimmed. A marker inside quotes is
/// a mention, not a protocol line (`esp-emu` logs `Exit-on string 'SDRK-SUITE END' found`).
fn marker(line: &str) -> Option<String> {
    let clean = strip_ansi(line);
    let mut from = 0;
    while let Some(offset) = clean[from..].find("SDRK-") {
        let at = from + offset;
        let quoted = clean[..at].ends_with(['\'', '"', '`']);
        if !quoted {
            return Some(clean[at..].trim().to_string());
        }
        from = at + "SDRK-".len();
    }
    None
}

/// Whether `m` (a [`marker`]) is a complete `SDRK-SUITE END passed=<p> failed=<f>` line.
fn end_counts(m: &str) -> Option<(u32, u32)> {
    let rest = m.strip_prefix("SDRK-SUITE END")?;
    Some((field(rest, "passed=")?, field(rest, "failed=")?))
}

/// Whether a line of console output is the end line of the suite.
fn is_end_line(line: &str) -> bool {
    marker(line).is_some_and(|m| end_counts(&m).is_some())
}

fn field(text: &str, key: &str) -> Option<u32> {
    text.split_whitespace()
        .find_map(|w| w.strip_prefix(key))
        .and_then(|v| v.parse().ok())
}

/// Judges the log of a run from the protocol lines only.
///
/// ```
/// let log = "boot noise\nSDRK-SUITE START 1\nSDRK-CASE PASS a\nSDRK-SUITE END passed=1 failed=0\n";
/// assert!(xtask::emu::judge(log).ok);
/// assert!(!xtask::emu::judge("SDRK-SUITE START 1\nSDRK-CASE PASS a\n").ok);
/// ```
#[must_use]
pub fn judge(log: &str) -> Verdict {
    let mut v = Verdict::default();
    let mut starts = 0u32;
    let mut ends = 0u32;
    for line in log.lines() {
        let Some(m) = marker(line) else { continue };
        if let Some(rest) = m.strip_prefix("SDRK-SUITE START") {
            starts += 1;
            v.started = rest.trim().parse().ok();
        } else if let Some((passed, failed)) = end_counts(&m) {
            ends += 1;
            v.ended = true;
            v.passed = passed;
            v.failed = failed;
        } else if m.starts_with("SDRK-CASE PASS") {
            v.pass_lines += 1;
        } else if let Some(rest) = m.strip_prefix("SDRK-CASE FAIL") {
            v.fail_lines.push(rest.trim().to_string());
        } else if let Some(rest) = m.strip_prefix("SDRK-PANIC") {
            v.panics.push(rest.trim().to_string());
        }
    }
    if !v.ended {
        v.reasons
            .push("the output has no `SDRK-SUITE END` line".to_string());
    }
    if v.ended && v.failed != 0 {
        v.reasons.push(format!("END reports failed={}", v.failed));
    }
    for f in &v.fail_lines {
        v.reasons.push(format!("FAIL {f}"));
    }
    for p in &v.panics {
        v.reasons.push(format!("PANIC {p}"));
    }
    if starts != 1 {
        v.reasons.push(format!(
            "`SDRK-SUITE START` appears {starts} times, not once"
        ));
    }
    if ends > 1 {
        v.reasons
            .push(format!("`SDRK-SUITE END` appears {ends} times, not once"));
    }
    if v.ended && starts == 1 {
        if v.started != Some(v.passed.saturating_add(v.failed)) {
            v.reasons.push(format!(
                "START announced {:?} cases but END counts passed={} failed={}",
                v.started, v.passed, v.failed
            ));
        }
        if v.pass_lines != v.passed {
            v.reasons.push(format!(
                "END counts passed={} but the output has {} PASS lines",
                v.passed, v.pass_lines
            ));
        }
    }
    v.ok = v.reasons.is_empty();
    v
}

// ---------------------------------------------------------------------------------------------
// The targets

/// How a target is run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runner {
    /// `qemu-system-arm` with a machine and an optional CPU, semihosting for output and exit.
    QemuArm {
        /// `-machine`.
        machine: &'static str,
        /// `-cpu`, if the machine needs one.
        cpu: Option<&'static str>,
    },
    /// `qemu-system-riscv32 -machine virt`, semihosting.
    QemuRiscv32,
    /// `esp-emu` on a merged flash image.
    EspEmu {
        /// `--chip`.
        chip: &'static str,
    },
    /// The Espressif QEMU fork (`qemu-system-xtensa`) on a merged flash image.
    EspQemu {
        /// `-machine`.
        machine: &'static str,
    },
}

/// An emulator target of `targets/`.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    /// Directory name below `targets/` and name for `--target`.
    pub name: &'static str,
    /// 1 runs on every commit, 2 nightly.
    pub tier: u8,
    /// The Rust target triple.
    pub triple: &'static str,
    /// The emulator.
    pub runner: Runner,
    /// Timeout of the quick profile in seconds.
    pub quick_s: u64,
    /// Timeout of the full profile in seconds.
    pub full_s: u64,
}

/// The five targets, in the order they run.
pub const TARGETS: [Target; 5] = [
    Target {
        name: "qemu-cortex-m4",
        tier: 1,
        triple: "thumbv7em-none-eabihf",
        runner: Runner::QemuArm {
            machine: "mps2-an386",
            cpu: Some("cortex-m4"),
        },
        // 110 s measured on a quiet machine, 300 s on a loaded one (software `f64` in the references).
        quick_s: 2 * QUICK_TIMEOUT_S,
        full_s: FULL_TIMEOUT_S,
    },
    Target {
        name: "qemu-cortex-m55",
        tier: 2,
        triple: "thumbv8m.main-none-eabihf",
        runner: Runner::QemuArm {
            machine: "mps3-an547",
            cpu: None,
        },
        // 200 s measured (code in the 512 KiB ITCM needs opt-level `s`).
        quick_s: 3 * QUICK_TIMEOUT_S,
        full_s: FULL_TIMEOUT_S,
    },
    Target {
        name: "qemu-riscv32",
        tier: 2,
        triple: "riscv32imac-unknown-none-elf",
        runner: Runner::QemuRiscv32,
        // Soft-float `C32` kernels: about 200 s measured.
        quick_s: 3 * QUICK_TIMEOUT_S,
        full_s: FULL_TIMEOUT_S,
    },
    Target {
        name: "esp32p4",
        tier: 2,
        triple: "riscv32imafc-unknown-none-elf",
        runner: Runner::EspEmu { chip: "esp32p4" },
        // esp-emu interprets the soft-float code slowly: 1100 s to 1700 s measured for the quick
        // profile (the references run in software `f64`); the full profile adds the one-second
        // end-to-end cases and the 4096-point DFT.
        quick_s: 12 * QUICK_TIMEOUT_S,
        full_s: 4 * FULL_TIMEOUT_S,
    },
    Target {
        name: "esp32s3",
        tier: 2,
        triple: "xtensa-esp32s3-none-elf",
        runner: Runner::EspQemu { machine: "esp32s3" },
        // About 250 s measured for the quick profile (software `f64` in the references).
        quick_s: 3 * QUICK_TIMEOUT_S,
        full_s: 2 * FULL_TIMEOUT_S,
    },
];

/// The target called `name`.
#[must_use]
pub fn target_named(name: &str) -> Option<&'static Target> {
    TARGETS.iter().find(|t| t.name == name)
}

impl Target {
    /// The name of the binary the target crate builds.
    #[must_use]
    pub fn binary(&self) -> String {
        format!("sdrk-target-{}", self.name)
    }

    /// Timeout of the profile.
    #[must_use]
    pub fn timeout(&self, full: bool) -> Duration {
        Duration::from_secs(if full { self.full_s } else { self.quick_s })
    }

    /// Whether the image needs `espflash save-image` (the ESP targets).
    #[must_use]
    pub fn needs_espflash(&self) -> bool {
        matches!(self.runner, Runner::EspEmu { .. } | Runner::EspQemu { .. })
    }
}

/// The options of one `emu` run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct Options {
    /// Names given with `--target`.
    pub targets: Vec<String>,
    /// `--tier`.
    pub tier: Option<u8>,
    /// `--all`.
    pub all: bool,
    /// `--full`.
    pub full: bool,
    /// `--features`, already split.
    pub features: Vec<String>,
    /// `--build-only`: build the target and stop, without an image or an emulator.
    pub build_only: bool,
    /// `--timeout` in seconds.
    pub timeout_s: Option<u64>,
    /// `--skip-missing`.
    pub skip_missing: bool,
    /// `--install`.
    pub install: bool,
    /// `--list`.
    pub list: bool,
    /// `--judge <file>`: judge a saved log instead of running an emulator.
    pub judge: Option<PathBuf>,
}

/// Parses the arguments after `emu`.
///
/// # Errors
/// Unknown option, missing or malformed value.
pub fn parse_args(args: &[&str]) -> anyhow::Result<Options> {
    let mut o = Options::default();
    let mut it = args.iter();
    while let Some(&arg) = it.next() {
        let mut value = |name: &str| -> anyhow::Result<&str> {
            it.next()
                .copied()
                .ok_or_else(|| anyhow!("{name} needs a value; {USAGE}"))
        };
        match arg {
            "--target" => o.targets.push(value("--target")?.to_string()),
            "--tier" => {
                let t: u8 = value("--tier")?.parse().context("--tier is 1 or 2")?;
                if !(1..=2).contains(&t) {
                    bail!("--tier is 1 or 2");
                }
                o.tier = Some(t);
            }
            "--all" => o.all = true,
            "--full" => o.full = true,
            "--features" => o.features.extend(
                value("--features")?
                    .split(',')
                    .map(str::trim)
                    .filter(|f| !f.is_empty())
                    .map(ToString::to_string),
            ),
            "--timeout" => {
                o.timeout_s = Some(
                    value("--timeout")?
                        .parse()
                        .context("--timeout is seconds")?,
                );
            }
            "--build-only" => o.build_only = true,
            "--skip-missing" => o.skip_missing = true,
            "--install" => o.install = true,
            "--list" => o.list = true,
            "--judge" => o.judge = Some(PathBuf::from(value("--judge")?)),
            other => bail!("unknown option `{other}`; {USAGE}"),
        }
    }
    Ok(o)
}

/// The targets an [`Options`] selects; tier 1 if nothing is selected.
///
/// # Errors
/// An unknown target name.
pub fn select(o: &Options) -> anyhow::Result<Vec<&'static Target>> {
    let mut out: Vec<&'static Target> = Vec::new();
    for name in &o.targets {
        out.push(target_named(name).ok_or_else(|| {
            let known: Vec<&str> = TARGETS.iter().map(|t| t.name).collect();
            anyhow!("unknown target `{name}`; known: {}", known.join(", "))
        })?);
    }
    if o.all {
        out.extend(TARGETS.iter());
    } else if let Some(tier) = o.tier {
        out.extend(TARGETS.iter().filter(|t| t.tier == tier));
    } else if out.is_empty() {
        out.extend(TARGETS.iter().filter(|t| t.tier == 1));
    }
    // Keep the table order and drop duplicates.
    Ok(TARGETS
        .iter()
        .filter(|t| out.iter().any(|s| s.name == t.name))
        .collect())
}

// ---------------------------------------------------------------------------------------------
// Tools

/// Versions and checksums of the downloaded tools. A download is only unpacked when its SHA-256
/// matches.
pub mod pins {
    /// Release of `esp-emu` (<https://github.com/espressif/esp-emulator>).
    pub const ESP_EMU_VERSION: &str = "0.49.0";
    /// SHA-256 of the Linux x86-64 archive.
    pub const ESP_EMU_X86_64_LINUX: &str =
        "100d656d33a2c26b4efd38e009b92e68914452105585187d3b8807ee87041ffc";
    /// SHA-256 of the Linux arm64 archive.
    pub const ESP_EMU_AARCH64_LINUX: &str =
        "6b8290b34001898ab0e33d5ac72e04c1d0c76e3369a64d94e1d0a8c0d43912f3";
    /// Release tag of the Espressif QEMU fork (<https://github.com/espressif/qemu>).
    pub const ESP_QEMU_TAG: &str = "esp-develop-9.2.2-20260417";
    /// The build stamp in the file names of that release.
    pub const ESP_QEMU_STAMP: &str = "esp_develop_9.2.2_20260417";
    /// SHA-256 of the Linux x86-64 xtensa archive.
    pub const ESP_QEMU_X86_64_LINUX: &str =
        "0eecb2a34a5586c0e59110f77b9343b7b336e82fdb0e1a30e1dc1bab8a547e35";
    /// SHA-256 of the Linux arm64 xtensa archive.
    pub const ESP_QEMU_AARCH64_LINUX: &str =
        "00de5985094c14e47d1b38464a006b8ed64fd0fa7a289c56da24f3a329f65339";
    /// SHA-256 of the Windows x86-64 xtensa archive.
    pub const ESP_QEMU_X86_64_WINDOWS: &str =
        "3c483d77f5350a568df1faf4d8dbc82c95d6bc2b826d0d4be910485e0a68ca2a";
    /// Version of `espflash` (installed with `cargo install --locked`).
    pub const ESPFLASH_VERSION: &str = "4.6.0";
    /// Version of `espup` (installed with `cargo install --locked`).
    pub const ESPUP_VERSION: &str = "0.18.0";
    /// The Espressif Rust toolchain that `espup` installs.
    pub const ESP_RUST_TOOLCHAIN: &str = "1.95.0.0";
}

/// A host tool that a target needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    /// `qemu-system-arm` (distribution package `qemu-system-arm`).
    QemuArm,
    /// `qemu-system-riscv32` (distribution package `qemu-system-riscv`).
    QemuRiscv32,
    /// `esp-emu`.
    EspEmu,
    /// The Espressif QEMU fork's `qemu-system-xtensa`.
    EspQemu,
    /// `espflash`.
    Espflash,
    /// The Xtensa linker (`xtensa-esp32s3-elf-gcc`) of the Espressif Rust toolchain.
    XtensaGcc,
}

impl Tool {
    /// The executable's name without suffix.
    #[must_use]
    pub const fn program(self) -> &'static str {
        match self {
            Self::QemuArm => "qemu-system-arm",
            Self::QemuRiscv32 => "qemu-system-riscv32",
            Self::EspEmu => "esp-emu",
            Self::EspQemu => "qemu-system-xtensa",
            Self::Espflash => "espflash",
            Self::XtensaGcc => "xtensa-esp32s3-elf-gcc",
        }
    }

    /// The environment variable that overrides the location.
    #[must_use]
    pub const fn env_var(self) -> &'static str {
        match self {
            Self::QemuArm => "SDRK_QEMU_ARM",
            Self::QemuRiscv32 => "SDRK_QEMU_RISCV32",
            Self::EspEmu => "SDRK_ESP_EMU",
            Self::EspQemu => "SDRK_ESP_QEMU_XTENSA",
            Self::Espflash => "SDRK_ESPFLASH",
            Self::XtensaGcc => "SDRK_XTENSA_GCC",
        }
    }

    /// What to do when the tool is missing.
    #[must_use]
    pub const fn hint(self) -> &'static str {
        match self {
            Self::QemuArm => {
                "install QEMU (Ubuntu: `sudo apt-get install qemu-system-arm`; Windows: the installer from qemu.org)"
            }
            Self::QemuRiscv32 => {
                "install QEMU (Ubuntu: `sudo apt-get install qemu-system-riscv`; Windows: the installer from qemu.org)"
            }
            Self::EspEmu => "run `cargo xtask emu --install` (esp-emu has no Windows build)",
            Self::EspQemu | Self::Espflash | Self::XtensaGcc => "run `cargo xtask emu --install`",
        }
    }
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// The host triple name used in the release file names of the tools.
fn host_tag() -> anyhow::Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok("x86_64-linux-gnu"),
        ("linux", "aarch64") => Ok("aarch64-linux-gnu"),
        ("windows", "x86_64") => Ok("x86_64-w64-mingw32"),
        (os, arch) => bail!("no prebuilt emulator tools are pinned for {os}/{arch}"),
    }
}

/// Where `emu --install` puts a tool below `target/tools/`.
#[must_use]
pub fn installed_path(root: &Path, tool: Tool) -> Option<PathBuf> {
    let tools = root.join("target").join("tools");
    match tool {
        Tool::QemuArm | Tool::QemuRiscv32 | Tool::XtensaGcc => None,
        Tool::EspEmu => {
            let v = pins::ESP_EMU_VERSION;
            let host = match host_tag().ok()? {
                "x86_64-linux-gnu" => "x86_64-unknown-linux-gnu",
                "aarch64-linux-gnu" => "aarch64-unknown-linux-gnu",
                _ => return None,
            };
            Some(
                tools
                    .join(format!("esp-emu-{v}"))
                    .join(format!("esp-emu-{v}-{host}"))
                    .join(exe("esp-emu")),
            )
        }
        Tool::EspQemu => Some(
            tools
                .join(format!("espressif-qemu-{}", pins::ESP_QEMU_TAG))
                .join("qemu")
                .join("bin")
                .join(exe("qemu-system-xtensa")),
        ),
        Tool::Espflash => Some(
            tools
                .join(format!("espflash-{}", pins::ESPFLASH_VERSION))
                .join("bin")
                .join(exe("espflash")),
        ),
    }
}

/// The `bin` directory of the Xtensa GCC that `espup` put into the `esp` toolchain.
#[must_use]
pub fn esp_gcc_bin() -> Option<PathBuf> {
    let out = Command::new("rustup")
        .args(["which", "--toolchain", "esp", "rustc"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let rustc = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    // `<toolchain>/bin/rustc`; the compiler is at `<toolchain>/xtensa-esp-elf/<version>/xtensa-esp-elf/bin`.
    let base = rustc.parent()?.parent()?.join("xtensa-esp-elf");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(base)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    versions.sort();
    versions
        .pop()
        .map(|v| v.join("xtensa-esp-elf").join("bin"))
        .filter(|d| d.is_dir())
}

/// Finds a tool: the override variable, then `target/tools/`, then `PATH`. Tools of the
/// Espressif fork are never taken from `PATH` (a distribution's `qemu-system-xtensa` has no
/// ESP32 machines).
#[must_use]
pub fn find_tool(root: &Path, tool: Tool) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(tool.env_var()).map(PathBuf::from) {
        return p.is_file().then_some(p);
    }
    if tool == Tool::XtensaGcc {
        return esp_gcc_bin()
            .map(|d| d.join(exe(tool.program())))
            .filter(|p| p.is_file());
    }
    if let Some(p) = installed_path(root, tool).filter(|p| p.is_file()) {
        return Some(p);
    }
    if tool == Tool::EspQemu {
        return None;
    }
    crate::wasm::find_program(tool.program(), &std::env::var_os("PATH")?)
}

/// The tools a target needs.
#[must_use]
pub fn tools_of(t: &Target) -> Vec<Tool> {
    match t.runner {
        Runner::QemuArm { .. } => vec![Tool::QemuArm],
        Runner::QemuRiscv32 => vec![Tool::QemuRiscv32],
        Runner::EspEmu { .. } => vec![Tool::EspEmu, Tool::Espflash],
        Runner::EspQemu { .. } => vec![Tool::EspQemu, Tool::Espflash, Tool::XtensaGcc],
    }
}

/// The tools that only building `t` needs (`--build-only`): the Xtensa linker for the S3.
#[must_use]
pub fn build_tools_of(t: &Target) -> Vec<Tool> {
    match t.runner {
        Runner::EspQemu { .. } => vec![Tool::XtensaGcc],
        _ => Vec::new(),
    }
}

fn sha256_hex(path: &Path) -> anyhow::Result<String> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    let mut out = String::new();
    for b in hasher.finalize() {
        let _ = write!(out, "{b:02x}");
    }
    Ok(out)
}

/// Downloads `url` into `dest` (with `curl`, which Windows 10 and later ship) and checks the
/// SHA-256 of the file.
fn download(url: &str, dest: &Path, sha256: &str) -> anyhow::Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    println!("==> downloading {url}");
    let status = Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .context("cannot run curl")?;
    if !status.success() {
        bail!("cannot download {url}");
    }
    let got = sha256_hex(dest)?;
    if got != sha256 {
        let _ = std::fs::remove_file(dest);
        bail!("checksum mismatch for {url}: expected {sha256}, got {got}");
    }
    Ok(())
}

fn unpack(archive: &Path, into: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(into)?;
    // `tar` ships with Linux, macOS and Windows 10 and later and reads `.tar.gz` and `.tar.xz`.
    let status = Command::new("tar")
        .arg("-xf")
        .arg(archive)
        .arg("-C")
        .arg(into)
        .status()
        .context("cannot run tar")?;
    if !status.success() {
        bail!("cannot unpack {}", archive.display());
    }
    Ok(())
}

fn install_esp_emu(root: &Path) -> anyhow::Result<()> {
    let Some(target) = installed_path(root, Tool::EspEmu) else {
        println!("esp-emu: no build for this host, skipped");
        return Ok(());
    };
    if target.is_file() {
        println!("esp-emu {}: already installed", pins::ESP_EMU_VERSION);
        return Ok(());
    }
    let (triple, sha) = match host_tag()? {
        "x86_64-linux-gnu" => ("x86_64-unknown-linux-gnu", pins::ESP_EMU_X86_64_LINUX),
        _ => ("aarch64-unknown-linux-gnu", pins::ESP_EMU_AARCH64_LINUX),
    };
    let v = pins::ESP_EMU_VERSION;
    let tools = root.join("target").join("tools");
    let file = format!("esp-emu-{v}-{triple}.tar.gz");
    let archive = tools.join("download").join(&file);
    download(
        &format!("https://github.com/espressif/esp-emulator/releases/download/v{v}/{file}"),
        &archive,
        sha,
    )?;
    unpack(&archive, &tools.join(format!("esp-emu-{v}")))
}

fn install_esp_qemu(root: &Path) -> anyhow::Result<()> {
    let target = installed_path(root, Tool::EspQemu).ok_or_else(|| anyhow!("no path"))?;
    if target.is_file() {
        println!("Espressif QEMU {}: already installed", pins::ESP_QEMU_TAG);
        return Ok(());
    }
    let host = host_tag()?;
    let sha = match host {
        "x86_64-linux-gnu" => pins::ESP_QEMU_X86_64_LINUX,
        "aarch64-linux-gnu" => pins::ESP_QEMU_AARCH64_LINUX,
        _ => pins::ESP_QEMU_X86_64_WINDOWS,
    };
    let file = format!("qemu-xtensa-softmmu-{}-{host}.tar.xz", pins::ESP_QEMU_STAMP);
    let tools = root.join("target").join("tools");
    let archive = tools.join("download").join(&file);
    download(
        &format!(
            "https://github.com/espressif/qemu/releases/download/{}/{file}",
            pins::ESP_QEMU_TAG
        ),
        &archive,
        sha,
    )?;
    unpack(
        &archive,
        &tools.join(format!("espressif-qemu-{}", pins::ESP_QEMU_TAG)),
    )
}

fn cargo_install(root: &Path, krate: &str, version: &str, dir: &str) -> anyhow::Result<()> {
    let dest = root.join("target").join("tools").join(dir);
    let bin = dest.join("bin").join(exe(krate));
    if bin.is_file() {
        println!("{krate} {version}: already installed");
        return Ok(());
    }
    println!("==> cargo install {krate} {version}");
    let status = Command::new("cargo")
        .args(["install", krate, "--locked", "--version", version, "--root"])
        .arg(&dest)
        .env_remove("RUSTUP_TOOLCHAIN")
        .status()
        .context("cannot run cargo install")?;
    if !status.success() {
        bail!("cargo install {krate} failed");
    }
    Ok(())
}

/// Whether the Espressif Rust toolchain (`rustup toolchain list` shows `esp`) is installed.
fn esp_toolchain_installed() -> bool {
    Command::new("rustup")
        .args(["toolchain", "list"])
        .output()
        .is_ok_and(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .any(|l| l.split_whitespace().next() == Some("esp"))
        })
}

fn install_esp_rust(root: &Path) -> anyhow::Result<()> {
    if esp_toolchain_installed() {
        println!("Espressif Rust toolchain: already installed");
        return Ok(());
    }
    cargo_install(
        root,
        "espup",
        pins::ESPUP_VERSION,
        &format!("espup-{}", pins::ESPUP_VERSION),
    )?;
    let espup = root
        .join("target")
        .join("tools")
        .join(format!("espup-{}", pins::ESPUP_VERSION))
        .join("bin")
        .join(exe("espup"));
    println!("==> espup install (toolchain {})", pins::ESP_RUST_TOOLCHAIN);
    let status = Command::new(espup)
        .args([
            "install",
            "--targets",
            "esp32s3",
            "--toolchain-version",
            pins::ESP_RUST_TOOLCHAIN,
            "--export-file",
        ])
        .arg(root.join("target").join("tools").join("export-esp.sh"))
        .status()
        .context("cannot run espup")?;
    if !status.success() {
        bail!("espup install failed");
    }
    Ok(())
}

/// `emu --install`: downloads the pinned emulators into `target/tools/` and installs `espflash`
/// and the Espressif Rust toolchain.
fn install(root: &Path) -> anyhow::Result<()> {
    install_esp_emu(root)?;
    install_esp_qemu(root)?;
    cargo_install(
        root,
        "espflash",
        pins::ESPFLASH_VERSION,
        &format!("espflash-{}", pins::ESPFLASH_VERSION),
    )?;
    install_esp_rust(root)?;
    println!(
        "Arm and RISC-V QEMU come from the distribution (see `docs/embedded.md`); nothing to install here."
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Build, image, run

/// The state of one target after its run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The suite passed.
    Pass,
    /// The suite failed or could not run; the string says why.
    Fail(String),
    /// A tool is missing and `--skip-missing` was given.
    Skipped(String),
}

/// One row of the summary table.
#[derive(Clone, Debug)]
pub struct Row {
    /// Target name.
    pub name: &'static str,
    /// What happened.
    pub outcome: Outcome,
    /// `passed` and `failed` of the `END` line, when there was one.
    pub counts: Option<(u32, u32)>,
    /// Duration of build and run.
    pub elapsed: Duration,
    /// The log file.
    pub log: PathBuf,
}

/// Formats the summary table.
#[must_use]
pub fn summary_table(rows: &[Row]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<18} {:<8} {:>7} {:>7} {:>9}  detail",
        "target", "result", "passed", "failed", "seconds"
    );
    for r in rows {
        let (word, detail) = match &r.outcome {
            Outcome::Pass => ("PASS", String::new()),
            Outcome::Fail(why) => ("FAIL", why.clone()),
            Outcome::Skipped(why) => ("SKIP", why.clone()),
        };
        let (p, f) = r
            .counts
            .map_or(("-".to_string(), "-".to_string()), |(p, f)| {
                (p.to_string(), f.to_string())
            });
        let _ = writeln!(
            out,
            "{:<18} {:<8} {:>7} {:>7} {:>9}  {}",
            r.name,
            word,
            p,
            f,
            r.elapsed.as_secs(),
            detail
        );
    }
    out
}

/// Features given to `cargo build`: the user's list, plus `full` for the full profile.
#[must_use]
pub fn build_features(o: &Options) -> Vec<String> {
    let mut f = o.features.clone();
    if o.full && !f.iter().any(|x| x == "full") {
        f.push("full".to_string());
    }
    f
}

fn target_dir(root: &Path, t: &Target) -> PathBuf {
    root.join("targets").join(t.name)
}

fn build_dir(root: &Path, t: &Target) -> PathBuf {
    root.join("target").join("emu").join("build").join(t.name)
}

fn build(root: &Path, t: &Target, features: &[String]) -> anyhow::Result<PathBuf> {
    let build_dir = build_dir(root, t);
    let mut cmd = Command::new("cargo");
    cmd.current_dir(target_dir(root, t))
        .args([
            "build",
            "--profile",
            "emu",
            "--bin",
            &t.binary(),
            "--target-dir",
        ])
        .arg(&build_dir);
    if !features.is_empty() {
        cmd.args(["--features", &features.join(",")]);
    }
    // Let the `rust-toolchain.toml` of the target decide (the ESP32-S3 needs the `esp` channel);
    // rustup exports the toolchain of the calling `cargo xtask` otherwise.
    cmd.env_remove("RUSTUP_TOOLCHAIN").env_remove("CARGO");
    if let Some(gcc) = find_tool(root, Tool::XtensaGcc).filter(|_| t.triple.starts_with("xtensa"))
        && let Some(dir) = gcc.parent()
    {
        // The linker of the Xtensa target must be on `PATH`.
        let mut paths = vec![dir.to_path_buf()];
        paths.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
        cmd.env("PATH", std::env::join_paths(paths)?);
    }
    println!("==> building targets/{} ({})", t.name, features.join(","));
    let status = cmd.status().context("cannot run cargo")?;
    if !status.success() {
        bail!("the build of targets/{} failed", t.name);
    }
    let elf = build_dir.join(t.triple).join("emu").join(t.binary());
    if !elf.is_file() {
        bail!("the build did not produce {}", elf.display());
    }
    Ok(elf)
}

fn make_image(root: &Path, t: &Target, elf: &Path) -> anyhow::Result<PathBuf> {
    if !t.needs_espflash() {
        return Ok(elf.to_path_buf());
    }
    let espflash = find_tool(root, Tool::Espflash)
        .ok_or_else(|| anyhow!("espflash is missing; {}", Tool::Espflash.hint()))?;
    let chip = match t.runner {
        Runner::EspEmu { chip } => chip,
        Runner::EspQemu { machine } => machine,
        _ => bail!("no image for {}", t.name),
    };
    let image = root
        .join("target")
        .join("emu")
        .join(format!("{}.bin", t.name));
    std::fs::create_dir_all(root.join("target").join("emu"))?;
    let status = Command::new(espflash)
        .args([
            "save-image",
            "--chip",
            chip,
            "--merge",
            "--flash-size",
            "16mb",
        ])
        .arg(elf)
        .arg(&image)
        .status()
        .context("cannot run espflash")?;
    if !status.success() {
        bail!("espflash save-image failed");
    }
    Ok(image)
}

/// The emulator command line for `image`.
fn emulator_command(
    root: &Path,
    t: &Target,
    image: &Path,
    timeout: Duration,
) -> anyhow::Result<Command> {
    let missing = |tool: Tool| anyhow!("{} is missing; {}", tool.program(), tool.hint());
    let semihosting = "enable=on,target=native";
    Ok(match t.runner {
        Runner::QemuArm { machine, cpu } => {
            let mut c =
                Command::new(find_tool(root, Tool::QemuArm).ok_or_else(|| missing(Tool::QemuArm))?);
            c.args(["-machine", machine]);
            if let Some(cpu) = cpu {
                c.args(["-cpu", cpu]);
            }
            c.args(["-nographic", "-semihosting-config", semihosting, "-kernel"])
                .arg(image);
            c
        }
        Runner::QemuRiscv32 => {
            let mut c = Command::new(
                find_tool(root, Tool::QemuRiscv32).ok_or_else(|| missing(Tool::QemuRiscv32))?,
            );
            c.args([
                "-machine",
                "virt",
                "-nographic",
                "-bios",
                "none",
                "-semihosting-config",
                semihosting,
                "-kernel",
            ])
            .arg(image);
            c
        }
        Runner::EspEmu { chip } => {
            let mut c =
                Command::new(find_tool(root, Tool::EspEmu).ok_or_else(|| missing(Tool::EspEmu))?);
            c.args(["--chip", chip, "--firmware"])
                .arg(image)
                .args(["--exit-on", "SDRK-SUITE END", "--timeout"])
                .arg(format!("{}s", timeout.as_secs()));
            c
        }
        Runner::EspQemu { machine } => {
            let mut c =
                Command::new(find_tool(root, Tool::EspQemu).ok_or_else(|| missing(Tool::EspQemu))?);
            c.args(["-nographic", "-machine", machine, "-drive"])
                .arg(format!("file={},if=mtd,format=raw", image.display()));
            c
        }
    })
}

/// One line of output from the emulator, or the end of a stream.
enum Event {
    Line(String),
    Closed,
}

fn pump(stream: impl std::io::Read + Send + 'static, tx: mpsc::Sender<Event>) {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buf)
                        .trim_end_matches(['\r', '\n'])
                        .to_string();
                    if tx.send(Event::Line(line)).is_err() {
                        return;
                    }
                }
            }
        }
        let _ = tx.send(Event::Closed);
    });
}

/// What `run_emulator` saw.
#[derive(Debug)]
pub struct RunLog {
    /// The whole output.
    pub text: String,
    /// The deadline passed before `SDRK-SUITE END` appeared.
    pub timed_out: bool,
}

/// Runs the emulator, streams its output to the console and `log`, and stops it when the
/// suite has ended or the timeout has passed.
pub fn run_emulator(mut cmd: Command, timeout: Duration, log: &Path) -> anyhow::Result<RunLog> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child: Child = cmd.spawn().context("cannot start the emulator")?;
    let (tx, rx) = mpsc::channel();
    if let Some(out) = child.stdout.take() {
        pump(out, tx.clone());
    }
    if let Some(err) = child.stderr.take() {
        pump(err, tx.clone());
    }
    drop(tx);
    let mut file =
        std::fs::File::create(log).with_context(|| format!("cannot create {}", log.display()))?;
    let deadline = Instant::now() + timeout;
    let mut text = String::new();
    let mut ended_at: Option<Instant> = None;
    let mut open_streams = 2;
    let mut timed_out = false;
    while open_streams > 0 {
        let now = Instant::now();
        if let Some(t) = ended_at
            && now.duration_since(t) > Duration::from_secs(5)
        {
            break;
        }
        if now >= deadline {
            timed_out = true;
            break;
        }
        let wait = (deadline - now).min(Duration::from_millis(500));
        match rx.recv_timeout(wait) {
            Ok(Event::Line(line)) => {
                println!("{line}");
                let _ = writeln!(file, "{line}");
                text.push_str(&line);
                text.push('\n');
                if ended_at.is_none() && is_end_line(&line) {
                    ended_at = Some(Instant::now());
                }
            }
            Ok(Event::Closed) => open_streams -= 1,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    // The emulators of the ESP targets do not stop by themselves (or only on request).
    let _ = child.kill();
    let _ = child.wait();
    let _ = file.flush();
    Ok(RunLog { text, timed_out })
}

fn run_one(root: &Path, t: &Target, o: &Options, features: &[String]) -> Row {
    let started = Instant::now();
    let log = root
        .join("target")
        .join("emu")
        .join(format!("{}.log", t.name));
    let row = |outcome: Outcome, counts: Option<(u32, u32)>| Row {
        name: t.name,
        outcome,
        counts,
        elapsed: started.elapsed(),
        log: log.clone(),
    };
    let needed = if o.build_only {
        build_tools_of(t)
    } else {
        tools_of(t)
    };
    let missing: Vec<Tool> = needed
        .into_iter()
        .filter(|tool| find_tool(root, *tool).is_none())
        .collect();
    if !missing.is_empty() {
        let names: Vec<String> = missing
            .iter()
            .map(|m| format!("{} ({})", m.program(), m.hint()))
            .collect();
        let why = format!("missing: {}", names.join("; "));
        return if o.skip_missing {
            row(Outcome::Skipped(why), None)
        } else {
            row(Outcome::Fail(why), None)
        };
    }
    if let Err(e) = std::fs::create_dir_all(root.join("target").join("emu")) {
        return row(Outcome::Fail(format!("{e}")), None);
    }
    let timeout = o
        .timeout_s
        .map_or_else(|| t.timeout(o.full), Duration::from_secs);
    if o.build_only {
        return match build(root, t, features) {
            Ok(_) => row(Outcome::Pass, None),
            Err(e) => row(Outcome::Fail(format!("{e:#}")), None),
        };
    }
    let result = build(root, t, features)
        .and_then(|elf| make_image(root, t, &elf))
        .and_then(|image| {
            let cmd = emulator_command(root, t, &image, timeout)?;
            println!("==> running {} (timeout {} s)", t.name, timeout.as_secs());
            run_emulator(cmd, timeout, &log)
        });
    match result {
        Err(e) => row(Outcome::Fail(format!("{e:#}")), None),
        Ok(run) => {
            let v = judge(&run.text);
            let counts = v.ended.then_some((v.passed, v.failed));
            if v.ok {
                row(Outcome::Pass, counts)
            } else {
                let mut why = v.reasons.join("; ");
                if run.timed_out {
                    let _ = write!(why, "; timeout after {} s", timeout.as_secs());
                }
                row(Outcome::Fail(why), counts)
            }
        }
    }
}

/// Runs `cargo xtask emu`.
///
/// # Errors
/// Bad arguments or an unusable repository; a failing target is reported in the return value
/// (`Ok(false)`), not as an error.
pub fn run(root: &Path, args: &[&str]) -> anyhow::Result<bool> {
    let o = parse_args(args)?;
    if let Some(file) = &o.judge {
        let text = std::fs::read_to_string(file)
            .with_context(|| format!("cannot read {}", file.display()))?;
        let v = judge(&text);
        if v.ok {
            println!(
                "{}: pass (passed={} failed={})",
                file.display(),
                v.passed,
                v.failed
            );
        } else {
            eprintln!("{}: FAIL", file.display());
            for r in &v.reasons {
                eprintln!("  {r}");
            }
        }
        return Ok(v.ok);
    }
    if o.list {
        for t in &TARGETS {
            println!("{:<18} tier {}  {}", t.name, t.tier, t.triple);
        }
        return Ok(true);
    }
    if o.install {
        install(root)?;
        return Ok(true);
    }
    let selected = select(&o)?;
    let features = build_features(&o);
    let mut rows = Vec::new();
    for t in selected {
        rows.push(run_one(root, t, &o, &features));
    }
    println!("\n{}", summary_table(&rows));
    for r in &rows {
        if matches!(r.outcome, Outcome::Fail(_)) {
            eprintln!("xtask: {} failed; log: {}", r.name, r.log.display());
        }
    }
    Ok(rows.iter().all(|r| !matches!(r.outcome, Outcome::Fail(_))))
}
