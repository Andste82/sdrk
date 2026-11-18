//! Digests of block outputs recorded before a refactoring, for the bit-identity tests after it.
//!
//! A refactoring that must not change a single output bit (M44.1: `qam_mod`, `psk_mod` and
//! `qam_demod` rebuilt as composite blocks) cannot compare with the old code once it is gone.
//! The outputs of the old blocks are therefore recorded first, from the unchanged code, as
//! *baselines* in `tests/baseline/<set>/` of the workspace:
//!
//! - `<table>.txt`: one line per case and SIMD level, `<case> <level> <bytes> <digest>`, the
//!   number of bytes of the output and its 64-bit FNV-1a digest (16 hexadecimal digits) over
//!   the little-endian bytes of the values (`f32` and `C32` as their bit patterns); lines
//!   starting with `#` are comments.
//! - `vectors/<case>.<level>.bin`: the full output of a few short cases, the same bytes, so that
//!   a failing bit-identity test can name the first value that differs.
//!
//! The outputs of tolerance kernels depend on the SIMD level (`docs/kernels.md`), so every case
//! has one line per level the test suite runs at (`fallback`, `avx2`, `avx512`, `neon`). The
//! level is the caller's: the facade's backend for the kernels of the block, or `fallback`.
//! The outputs of bit-exact kernels are the same on every level, so their tables can have one
//! line per case at the level `all`, against which every level is checked (the kernels of
//! M44.2 in `tests/baseline/m44_2/`).
//!
//! These files are not golden vectors (those come from the Python generator only, R8): they are
//! the outputs of this repository's own code at one commit, written by the tests themselves in
//! record mode. With the environment variable `SDRK_BASELINE_RECORD` set to a directory,
//! [`Baseline::check`] and [`Baseline::check_vector`] do not compare; they append the line of
//! the case to `<dir>/<table>.txt` and write the vector to `<dir>/vectors/`, from where they are
//! copied into the baseline set. Without the variable they compare and fail on a missing case.
//!
//! ```
//! use sdrk_testkit::baseline::Digest;
//! let mut d = Digest::new();
//! d.f32s(&[1.0, -0.5]);
//! assert_eq!(d.len(), 8);
//! let mut e = Digest::new();
//! e.bytes(&[0, 0, 0x80, 0x3f, 0, 0, 0, 0xbf]);
//! assert_eq!(d, e);
//! ```

// Host-only code: the lint counts the unoptimised `Result` temporaries of the path handling.
#![allow(clippy::large_stack_frames)]

use core::fmt;
use sdrk_core::C32;
use std::collections::BTreeMap;
use std::format;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::vec::Vec;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The environment variable of the record mode (a directory).
pub const RECORD_VAR: &str = "SDRK_BASELINE_RECORD";

/// The 64-bit FNV-1a digest of a byte stream and its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Digest {
    hash: u64,
    len: u64,
}

impl Default for Digest {
    fn default() -> Self {
        Self::new()
    }
}

impl Digest {
    /// The digest of no bytes.
    pub const fn new() -> Self {
        Self {
            hash: FNV_OFFSET,
            len: 0,
        }
    }

    /// Continues the digest with `bytes`.
    pub fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hash = (self.hash ^ u64::from(b)).wrapping_mul(FNV_PRIME);
        }
        self.len = self.len.wrapping_add(bytes.len() as u64);
    }

    /// Continues the digest with the little-endian bit patterns of `values`.
    pub fn f32s(&mut self, values: &[f32]) {
        for v in values {
            self.bytes(&v.to_bits().to_le_bytes());
        }
    }

    /// Continues the digest with the little-endian bit patterns of `values` (real part first).
    pub fn c32s(&mut self, values: &[C32]) {
        for v in values {
            self.bytes(&v.re.to_bits().to_le_bytes());
            self.bytes(&v.im.to_bits().to_le_bytes());
        }
    }

    /// The digest of `values` (see [`Digest::f32s`]).
    pub fn of_f32(values: &[f32]) -> Self {
        let mut d = Self::new();
        d.f32s(values);
        d
    }

    /// The digest of `values` (see [`Digest::c32s`]).
    pub fn of_c32(values: &[C32]) -> Self {
        let mut d = Self::new();
        d.c32s(values);
        d
    }

    /// Number of bytes digested.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether no byte was digested.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The 64-bit digest.
    pub fn value(&self) -> u64 {
        self.hash
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:016x}", self.len, self.hash)
    }
}

/// A failed baseline lookup or comparison (the message names the case, the level and both
/// values).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineError(pub String);

impl fmt::Display for BaselineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for BaselineError {}

/// The level of a baseline line for the kernels of this build: `fallback` when the facade runs
/// the `C32` `fir_interpolate` on the fallback (`--cfg sdrk_fallback_only`, or a target without
/// the SIMD backend), otherwise `simd_level`, the name of the level of the SIMD backend
/// (`sdrk_backend_simd::level_name()`, which the caller passes because only the facade and test
/// code may depend on a backend).
///
/// ```
/// let level = sdrk_testkit::baseline::level("avx2");
/// assert!(level == "avx2" || level == "fallback");
/// ```
pub fn level(simd_level: &'static str) -> &'static str {
    if sdrk::backend_for("fir_interpolate", sdrk::SampleType::C32) == "fallback" {
        "fallback"
    } else {
        simd_level
    }
}

/// The bytes of `values` as [`Digest::f32s`] digests them.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

/// The bytes of `values` as [`Digest::c32s`] digests them.
pub fn c32_bytes(values: &[C32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| {
            let mut b = [0u8; 8];
            let (re, im) = b.split_at_mut(4);
            re.copy_from_slice(&v.re.to_bits().to_le_bytes());
            im.copy_from_slice(&v.im.to_bits().to_le_bytes());
            b
        })
        .collect()
}

/// One table of a baseline set at one SIMD level.
#[derive(Debug, Clone)]
pub struct Baseline {
    dir: PathBuf,
    table: String,
    level: String,
    entries: BTreeMap<String, Digest>,
    record: Option<PathBuf>,
}

/// The directory of the baseline set `set`: `tests/baseline/<set>` of the workspace.
///
/// # Errors
/// When no parent directory of this crate holds `tests/baseline`.
pub fn baseline_dir(set: &str) -> Result<PathBuf, BaselineError> {
    let mut dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("tests").join("baseline").is_dir() {
            return Ok(dir.join("tests").join("baseline").join(set));
        }
        dir = dir.parent().ok_or_else(|| {
            BaselineError("workspace root with tests/baseline not found".to_string())
        })?;
    }
}

impl Baseline {
    /// Loads the lines of `level` from `tests/baseline/<set>/<table>.txt`.
    ///
    /// # Errors
    /// When the file cannot be read, a line is malformed, or a case appears twice at `level`.
    pub fn load(set: &str, table: &str, level: &str) -> Result<Self, BaselineError> {
        let dir = baseline_dir(set)?;
        let path = dir.join(format!("{table}.txt"));
        let text = std::fs::read_to_string(&path)
            .map_err(|e| BaselineError(format!("{}: {e}", path.display())))?;
        let mut entries = BTreeMap::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let bad = || BaselineError(format!("{}:{}: malformed line", path.display(), n + 1));
            let mut it = line.split_whitespace();
            let (Some(case), Some(lv), Some(len), Some(hash), None) =
                (it.next(), it.next(), it.next(), it.next(), it.next())
            else {
                return Err(bad());
            };
            let len: u64 = len.parse().map_err(|_| bad())?;
            let hash = u64::from_str_radix(hash, 16).map_err(|_| bad())?;
            if lv != level {
                continue;
            }
            if entries
                .insert(case.to_string(), Digest { hash, len })
                .is_some()
            {
                return Err(BaselineError(format!(
                    "{}:{}: case {case} twice at level {level}",
                    path.display(),
                    n + 1
                )));
            }
        }
        Ok(Self {
            dir,
            table: table.to_string(),
            level: level.to_string(),
            entries,
            record: std::env::var_os(RECORD_VAR).map(PathBuf::from),
        })
    }

    /// The SIMD level of the table.
    pub fn level(&self) -> &str {
        &self.level
    }

    /// Whether the tests run in record mode (see the module documentation).
    pub fn recording(&self) -> bool {
        self.record.is_some()
    }

    /// Compares `digest` with the baseline of `case` (in record mode: records it).
    ///
    /// # Errors
    /// When the case has no line at the level of the table or its digest differs.
    pub fn check(&self, case: &str, digest: &Digest) -> Result<(), BaselineError> {
        if let Some(dir) = &self.record {
            let line = format!("{case} {} {digest}\n", self.level);
            return append(&dir.join(format!("{}.txt", self.table)), line.as_bytes());
        }
        match self.entries.get(case) {
            Some(want) if want == digest => Ok(()),
            Some(want) => Err(BaselineError(format!(
                "{}: case {case} at level {}: output {digest} (bytes, digest), baseline {want}",
                self.table, self.level
            ))),
            None => Err(BaselineError(format!(
                "{}: case {case} has no baseline at level {} (output {digest})",
                self.table, self.level
            ))),
        }
    }

    /// Compares `bytes` with the full vector `vectors/<case>.<level>.bin` of the set (in record
    /// mode: writes it); the error names the first byte that differs and its 4-byte value.
    ///
    /// # Errors
    /// When the vector cannot be read or differs.
    pub fn check_vector(&self, case: &str, bytes: &[u8]) -> Result<(), BaselineError> {
        let name = format!("{case}.{}.bin", self.level);
        if let Some(dir) = &self.record {
            let vectors = dir.join("vectors");
            std::fs::create_dir_all(&vectors)
                .map_err(|e| BaselineError(format!("{}: {e}", vectors.display())))?;
            let path = vectors.join(name);
            return std::fs::write(&path, bytes)
                .map_err(|e| BaselineError(format!("{}: {e}", path.display())));
        }
        let path = self.dir.join("vectors").join(name);
        let want =
            std::fs::read(&path).map_err(|e| BaselineError(format!("{}: {e}", path.display())))?;
        if want.len() != bytes.len() {
            return Err(BaselineError(format!(
                "{}: {} bytes, the output {}",
                path.display(),
                want.len(),
                bytes.len()
            )));
        }
        match want.iter().zip(bytes).position(|(a, b)| a != b) {
            None => Ok(()),
            Some(i) => {
                let word = |v: &[u8]| {
                    let at = i / 4 * 4;
                    v.get(at..at + 4)
                        .and_then(|w| <[u8; 4]>::try_from(w).ok())
                        .map_or(0, u32::from_le_bytes)
                };
                Err(BaselineError(format!(
                    "{}: first difference at byte {i} (value {}): {:08x} in the baseline, {:08x} \
                     in the output",
                    path.display(),
                    i / 4,
                    word(&want),
                    word(bytes)
                )))
            }
        }
    }
}

fn append(path: &Path, bytes: &[u8]) -> Result<(), BaselineError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| BaselineError(format!("{}: {e}", parent.display())))?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| BaselineError(format!("{}: {e}", path.display())))
}
