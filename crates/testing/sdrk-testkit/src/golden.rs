//! Loader and comparison for the golden vectors in `tests/golden/vectors`.
//!
//! The vectors are produced by `tests/golden/generate.py` (NumPy/SciPy and the
//! independent Python Q15 model) and committed to git. A case is a JSON file
//! plus one binary file per array; see `tests/golden/README.md` for the format.
//! Tolerances live in the JSON, never in Rust code.
//!
//! ```no_run
//! use sdrk_testkit::golden;
//! let case = golden::load("scale", "c32_random_4096")?;
//! let input = case.array("input")?;
//! assert_eq!(input.len(), 4096);
//! # Ok::<(), golden::GoldenError>(())
//! ```

// Host-only code: the lint counts every unoptimised `Result` temporary, so the
// large `GoldenError` and serde types trip it although the real frames are small.
#![allow(clippy::large_stack_frames)]

use core::fmt;
use num_complex::Complex64;
use sdrk_core::{C32, CQ15, Q15, Sample, SampleType};
use sdrk_device::{ErrorKind, SdrkError};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::format;
use std::path::{Path, PathBuf};
use std::string::{String, ToString};
use std::vec::Vec;

use crate::metrics::{Comparison, compare_complex, compare_real};

/// The only supported value of `format_version`.
const FORMAT_VERSION: u32 = 1;

/// Errors of the golden loader and comparison.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GoldenError {
    /// A file or directory could not be read (kind: `Io`).
    Io {
        /// The path that failed.
        path: String,
        /// The operating-system message.
        message: String,
    },
    /// The JSON is malformed or uses an unknown dtype or tolerance class (kind: `Parse`).
    Parse(String),
    /// The content is well formed but inconsistent, or a lookup failed (kind: `Validation`).
    Validation(String),
    /// The actual data does not satisfy the tolerance of the case (kind: `Validation`).
    Mismatch {
        /// Name of the case, `kernel/case`.
        case: String,
        /// Achieved SNR in dB over the compared samples.
        snr_db: f64,
        /// Index of the first differing sample, if any.
        first_mismatch: Option<usize>,
        /// Human-readable description of the failed check.
        detail: String,
    },
}

/// Result alias of the golden module.
pub type Result<T, E = GoldenError> = core::result::Result<T, E>;

impl fmt::Display for GoldenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "i/o error on {path}: {message}"),
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
            Self::Validation(msg) => write!(f, "invalid golden case: {msg}"),
            Self::Mismatch {
                case,
                snr_db,
                first_mismatch,
                detail,
            } => match first_mismatch {
                Some(i) => write!(
                    f,
                    "golden mismatch in {case}: {detail} (snr {snr_db:.2} db, first mismatch at index {i})"
                ),
                None => write!(
                    f,
                    "golden mismatch in {case}: {detail} (snr {snr_db:.2} db)"
                ),
            },
        }
    }
}

impl core::error::Error for GoldenError {}

impl SdrkError for GoldenError {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Io { .. } => ErrorKind::Io,
            Self::Parse(_) => ErrorKind::Parse,
            Self::Validation(_) | Self::Mismatch { .. } => ErrorKind::Validation,
        }
    }
}

/// Unit of the reference values of a `max_abs_err` tolerance with a level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefUnit {
    /// The reference values are already in dB (spectra).
    Db,
    /// The reference values are linear; the level applies to `20 log10 |ref|`.
    Linear,
}

/// Tolerance class of a case.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Tolerance {
    /// Raw bits must be equal.
    BitExact,
    /// Minimum SNR over samples `skip..`, optionally also a largest absolute error.
    Snr {
        /// Minimum SNR in dB.
        min_db: f64,
        /// Number of leading samples that are not compared.
        skip: usize,
        /// Optional limit of the largest absolute error.
        max_abs_err: Option<f64>,
    },
    /// Largest absolute error over samples `skip..`, optionally only where the
    /// reference is at or above a level.
    MaxAbsErr {
        /// Largest allowed absolute error.
        max: f64,
        /// Number of leading samples that are not compared.
        skip: usize,
        /// Level and unit; only samples whose reference is at or above it are checked.
        min_ref: Option<(f64, RefUnit)>,
    },
}

/// A decoded array of a case.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Array {
    /// Unsigned 8-bit.
    U8(Vec<u8>),
    /// Signed 8-bit.
    I8(Vec<i8>),
    /// Signed 16-bit (also Q15 raw values).
    I16(Vec<i16>),
    /// Signed 32-bit.
    I32(Vec<i32>),
    /// 32-bit float.
    F32(Vec<f32>),
    /// 64-bit float.
    F64(Vec<f64>),
    /// Complex Q15 (interleaved i16).
    CI16(Vec<CQ15>),
    /// Complex f32.
    CF32(Vec<C32>),
    /// Complex f64.
    CF64(Vec<Complex64>),
}

impl Array {
    /// Number of elements (a complex sample counts once).
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::CI16(v) => v.len(),
            Self::CF32(v) => v.len(),
            Self::CF64(v) => v.len(),
        }
    }

    /// True when the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The dtype name used in the JSON files.
    pub fn dtype(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::I16(_) => "i16",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::CI16(_) => "ci16",
            Self::CF32(_) => "cf32",
            Self::CF64(_) => "cf64",
        }
    }

    /// The array as `Q15` samples, if it is an `i16` array.
    pub fn as_q15(&self) -> Option<Vec<Q15>> {
        match self {
            Self::I16(v) => Some(v.iter().map(|&r| Q15(r)).collect()),
            _ => None,
        }
    }
}

/// One loaded golden case.
#[derive(Debug, Clone, PartialEq)]
pub struct GoldenCase {
    /// Kernel name (directory).
    pub kernel: String,
    /// Case name.
    pub case: String,
    /// Seed that produced the case.
    pub seed: u64,
    /// Kernel parameters.
    pub params: serde_json::Value,
    /// Tolerance class and limits.
    pub tolerance: Tolerance,
    arrays: BTreeMap<String, Array>,
}

impl GoldenCase {
    /// The array with the given name.
    pub fn array(&self, name: &str) -> Result<&Array> {
        self.arrays.get(name).ok_or_else(|| {
            GoldenError::Validation(format!("{}/{}: no array `{name}`", self.kernel, self.case))
        })
    }

    /// A floating-point parameter.
    pub fn param_f64(&self, name: &str) -> Result<f64> {
        self.param(name)?
            .as_f64()
            .ok_or_else(|| self.bad_param(name, "a number"))
    }

    /// An unsigned integer parameter.
    pub fn param_u64(&self, name: &str) -> Result<u64> {
        self.param(name)?
            .as_u64()
            .ok_or_else(|| self.bad_param(name, "an unsigned integer"))
    }

    /// A string parameter.
    pub fn param_str(&self, name: &str) -> Result<&str> {
        self.param(name)?
            .as_str()
            .ok_or_else(|| self.bad_param(name, "a string"))
    }

    fn param(&self, name: &str) -> Result<&serde_json::Value> {
        self.params.get(name).ok_or_else(|| {
            GoldenError::Validation(format!(
                "{}/{}: no parameter `{name}`",
                self.kernel, self.case
            ))
        })
    }

    fn bad_param(&self, name: &str, wanted: &str) -> GoldenError {
        GoldenError::Validation(format!(
            "{}/{}: parameter `{name}` is not {wanted}",
            self.kernel, self.case
        ))
    }
}

/// The directory that holds the vectors: `SDRK_GOLDEN_DIR` if set, otherwise
/// `tests/golden/vectors` of the workspace.
pub fn golden_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("SDRK_GOLDEN_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let mut dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("tests").join("golden").is_dir() {
            return Ok(dir.join("tests").join("golden").join("vectors"));
        }
        dir = dir.parent().ok_or_else(|| {
            GoldenError::Validation("workspace root with tests/golden not found".to_string())
        })?;
    }
}

/// Loads `<kernel>/<case>` from the default directory (see [`golden_dir`]).
pub fn load(kernel: &str, case: &str) -> Result<GoldenCase> {
    load_from(&golden_dir()?, kernel, case)
}

/// Sorted case names of a kernel in the default directory.
pub fn list(kernel: &str) -> Result<Vec<String>> {
    list_from(&golden_dir()?, kernel)
}

/// Like [`list`], with an explicit root directory.
pub fn list_from(root: &Path, kernel: &str) -> Result<Vec<String>> {
    check_name(kernel)?;
    let dir = root.join(kernel);
    let entries = std::fs::read_dir(&dir).map_err(|e| io_error(&dir, &e))?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| io_error(&dir, &e))?;
        if let Some(name) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.strip_suffix(".json"))
        {
            names.push(name.to_string());
        }
    }
    names.sort();
    Ok(names)
}

#[derive(Deserialize)]
struct RawArray {
    file: String,
    dtype: String,
    len: usize,
}

#[derive(Deserialize)]
struct RawTolerance {
    class: String,
    snr_db_min: Option<f64>,
    skip: Option<usize>,
    max_abs_err: Option<f64>,
    min_ref_db: Option<f64>,
    ref_unit: Option<String>,
}

#[derive(Deserialize)]
struct RawCase {
    format_version: u32,
    kernel: String,
    case: String,
    seed: u64,
    params: serde_json::Value,
    arrays: BTreeMap<String, RawArray>,
    tolerance: RawTolerance,
}

/// Like [`load`], with an explicit root directory.
pub fn load_from(root: &Path, kernel: &str, case: &str) -> Result<GoldenCase> {
    check_name(kernel)?;
    check_name(case)?;
    let dir = root.join(kernel);
    let json_path = dir.join(format!("{case}.json"));
    let text = std::fs::read_to_string(&json_path).map_err(|e| io_error(&json_path, &e))?;
    let raw: RawCase = serde_json::from_str(&text)
        .map_err(|e| GoldenError::Parse(format!("{}: {e}", json_path.display())))?;
    if raw.format_version != FORMAT_VERSION {
        return Err(GoldenError::Validation(format!(
            "{}: unsupported format_version {}",
            json_path.display(),
            raw.format_version
        )));
    }
    if raw.kernel != kernel || raw.case != case {
        return Err(GoldenError::Validation(format!(
            "{}: file describes {}/{}",
            json_path.display(),
            raw.kernel,
            raw.case
        )));
    }
    let tolerance = parse_tolerance(&raw.tolerance)?;
    let mut arrays = BTreeMap::new();
    for (name, meta) in &raw.arrays {
        check_name(&meta.file)?;
        let path = dir.join(&meta.file);
        let bytes = std::fs::read(&path).map_err(|e| io_error(&path, &e))?;
        let array = decode(&meta.dtype, &bytes, meta.len, &path)?;
        arrays.insert(name.clone(), array);
    }
    Ok(GoldenCase {
        kernel: raw.kernel,
        case: raw.case,
        seed: raw.seed,
        params: raw.params,
        tolerance,
        arrays,
    })
}

fn io_error(path: &Path, e: &std::io::Error) -> GoldenError {
    GoldenError::Io {
        path: path.display().to_string(),
        message: e.to_string(),
    }
}

/// Names are single path components; this keeps a hostile JSON inside the vector directory.
fn check_name(name: &str) -> Result<()> {
    if name.is_empty() || name.contains(['/', '\\']) || name == ".." || name == "." {
        return Err(GoldenError::Validation(format!(
            "bad file or case name `{name}`"
        )));
    }
    Ok(())
}

fn parse_tolerance(raw: &RawTolerance) -> Result<Tolerance> {
    let need = |v: Option<f64>, what: &str| {
        v.ok_or_else(|| {
            GoldenError::Parse(format!("tolerance class `{}` needs `{what}`", raw.class))
        })
    };
    match raw.class.as_str() {
        "bit_exact" => Ok(Tolerance::BitExact),
        "tolerance" => Ok(Tolerance::Snr {
            min_db: need(raw.snr_db_min, "snr_db_min")?,
            skip: raw.skip.unwrap_or(0),
            max_abs_err: raw.max_abs_err,
        }),
        "max_abs_err" => {
            let min_ref = match raw.min_ref_db {
                None => None,
                Some(level) => {
                    let unit = match raw.ref_unit.as_deref() {
                        Some("db") => RefUnit::Db,
                        Some("linear") => RefUnit::Linear,
                        Some(other) => {
                            return Err(GoldenError::Parse(format!("unknown ref_unit `{other}`")));
                        }
                        None => {
                            return Err(GoldenError::Parse(
                                "`min_ref_db` needs `ref_unit`".to_string(),
                            ));
                        }
                    };
                    Some((level, unit))
                }
            };
            Ok(Tolerance::MaxAbsErr {
                max: need(raw.max_abs_err, "max_abs_err")?,
                skip: raw.skip.unwrap_or(0),
                min_ref,
            })
        }
        other => Err(GoldenError::Parse(format!(
            "unknown tolerance class `{other}`"
        ))),
    }
}

fn decode(dtype: &str, bytes: &[u8], len: usize, path: &Path) -> Result<Array> {
    let (size, comps): (usize, usize) = match dtype {
        "u8" | "i8" => (1, 1),
        "i16" => (2, 1),
        "i32" | "f32" => (4, 1),
        "f64" => (8, 1),
        "ci16" => (2, 2),
        "cf32" => (4, 2),
        "cf64" => (8, 2),
        other => return Err(GoldenError::Parse(format!("unknown dtype `{other}`"))),
    };
    let expected = len
        .checked_mul(size * comps)
        .ok_or_else(|| GoldenError::Validation("array length overflows".to_string()))?;
    if bytes.len() != expected {
        return Err(GoldenError::Validation(format!(
            "{}: {} bytes, but len {len} of {dtype} needs {expected}",
            path.display(),
            bytes.len()
        )));
    }
    Ok(match dtype {
        "u8" => Array::U8(bytes.to_vec()),
        "i8" => Array::I8(le::<1, _>(bytes, i8::from_le_bytes)),
        "i16" => Array::I16(le::<2, _>(bytes, i16::from_le_bytes)),
        "i32" => Array::I32(le::<4, _>(bytes, i32::from_le_bytes)),
        "f32" => Array::F32(le::<4, _>(bytes, f32::from_le_bytes)),
        "f64" => Array::F64(le::<8, _>(bytes, f64::from_le_bytes)),
        "ci16" => Array::CI16(pairs(&le::<2, _>(bytes, i16::from_le_bytes), |re, im| {
            CQ15 {
                re: Q15(re),
                im: Q15(im),
            }
        })),
        "cf32" => Array::CF32(pairs(&le::<4, _>(bytes, f32::from_le_bytes), C32::new)),
        _ => Array::CF64(pairs(
            &le::<8, _>(bytes, f64::from_le_bytes),
            Complex64::new,
        )),
    })
}

fn le<const N: usize, T>(bytes: &[u8], f: impl Fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .filter_map(|c| <[u8; N]>::try_from(c).ok())
        .map(f)
        .collect()
}

fn pairs<A: Copy, T>(v: &[A], f: impl Fn(A, A) -> T) -> Vec<T> {
    v.chunks_exact(2)
        .filter_map(|c| match *c {
            [a, b] => Some(f(a, b)),
            _ => None,
        })
        .collect()
}

/// Real or complex view of data as f64, used by the metrics.
enum View {
    Real(Vec<f64>),
    Complex(Vec<Complex64>),
}

fn q15_f64(q: Q15) -> f64 {
    f64::from(q.0) / 32768.0
}

fn cq15_c64(c: CQ15) -> Complex64 {
    Complex64::new(q15_f64(c.re), q15_f64(c.im))
}

/// The expected array as an f64 view (floats and Q15 raw values).
fn expected_view(array: &Array) -> Result<View> {
    Ok(match array {
        Array::F64(v) => View::Real(v.clone()),
        Array::F32(v) => View::Real(v.iter().map(|&x| f64::from(x)).collect()),
        Array::I16(v) => View::Real(v.iter().map(|&r| q15_f64(Q15(r))).collect()),
        Array::CF64(v) => View::Complex(v.clone()),
        Array::CF32(v) => View::Complex(
            v.iter()
                .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
                .collect(),
        ),
        Array::CI16(v) => View::Complex(v.iter().map(|&c| cq15_c64(c)).collect()),
        other => {
            return Err(GoldenError::Validation(format!(
                "dtype {} cannot be compared as a signal",
                other.dtype()
            )));
        }
    })
}

/// The actual samples as an f64 view.
fn actual_view<T: Sample>(actual: &[T]) -> Result<View> {
    let internal = || GoldenError::Validation("sample type does not match its tag".to_string());
    Ok(match T::TYPE {
        SampleType::F32 => View::Real(
            bytemuck::try_cast_slice::<T, f32>(actual)
                .map_err(|_| internal())?
                .iter()
                .map(|&x| f64::from(x))
                .collect(),
        ),
        SampleType::Q15 => View::Real(
            bytemuck::try_cast_slice::<T, Q15>(actual)
                .map_err(|_| internal())?
                .iter()
                .map(|&q| q15_f64(q))
                .collect(),
        ),
        SampleType::C32 => View::Complex(
            bytemuck::try_cast_slice::<T, C32>(actual)
                .map_err(|_| internal())?
                .iter()
                .map(|c| Complex64::new(f64::from(c.re), f64::from(c.im)))
                .collect(),
        ),
        SampleType::CQ15 => View::Complex(
            bytemuck::try_cast_slice::<T, CQ15>(actual)
                .map_err(|_| internal())?
                .iter()
                .map(|&c| cq15_c64(c))
                .collect(),
        ),
        _ => {
            return Err(GoldenError::Validation(
                "unsupported sample type".to_string(),
            ));
        }
    })
}

/// Raw little-endian bytes of the array elements, for bit-exact comparison.
fn expected_bytes(array: &Array) -> Vec<u8> {
    match array {
        Array::U8(v) => v.clone(),
        Array::I8(v) => bytemuck::cast_slice::<i8, u8>(v).to_vec(),
        Array::I16(v) => bytemuck::cast_slice::<i16, u8>(v).to_vec(),
        Array::I32(v) => bytemuck::cast_slice::<i32, u8>(v).to_vec(),
        Array::F32(v) => bytemuck::cast_slice::<f32, u8>(v).to_vec(),
        Array::F64(v) => bytemuck::cast_slice::<f64, u8>(v).to_vec(),
        Array::CI16(v) => bytemuck::cast_slice::<CQ15, u8>(v).to_vec(),
        Array::CF32(v) => bytemuck::cast_slice::<C32, u8>(v).to_vec(),
        Array::CF64(v) => bytemuck::cast_slice::<Complex64, u8>(v).to_vec(),
    }
}

/// The sample type that matches an array for bit-exact comparison.
fn array_sample_type(array: &Array) -> Option<SampleType> {
    match array {
        Array::I16(_) => Some(SampleType::Q15),
        Array::CI16(_) => Some(SampleType::CQ15),
        Array::F32(_) => Some(SampleType::F32),
        Array::CF32(_) => Some(SampleType::C32),
        _ => None,
    }
}

/// Compares `actual` with `expected` under the tolerance of `case`.
///
/// - `BitExact`: the raw bits must be equal; `expected` must be `i16`, `ci16`,
///   `f32` or `cf32` matching `T`.
/// - `Snr`: SNR over `skip..` (and the largest absolute error if given).
/// - `MaxAbsErr`: largest absolute error over the selected samples.
///
/// Returns the [`Comparison`] on success and [`GoldenError::Mismatch`]
/// (with case name, achieved SNR and first mismatch index) on failure.
pub fn assert_matches<T: Sample>(
    case: &GoldenCase,
    expected: &Array,
    actual: &[T],
) -> Result<Comparison> {
    let name = format!("{}/{}", case.kernel, case.case);
    let fail = |c: &Comparison, detail: String| GoldenError::Mismatch {
        case: name.clone(),
        snr_db: c.snr_db,
        first_mismatch: c.first_mismatch,
        detail,
    };
    if expected.len() != actual.len() {
        return Err(GoldenError::Mismatch {
            case: name,
            snr_db: f64::NEG_INFINITY,
            first_mismatch: Some(expected.len().min(actual.len())),
            detail: format!("length {} instead of {}", actual.len(), expected.len()),
        });
    }
    match case.tolerance {
        Tolerance::BitExact => {
            if array_sample_type(expected) != Some(T::TYPE) {
                return Err(GoldenError::Validation(format!(
                    "{name}: bit-exact comparison of {} with {:?} is not defined",
                    expected.dtype(),
                    T::TYPE
                )));
            }
            let want = expected_bytes(expected);
            let got: &[u8] = bytemuck::cast_slice(actual);
            let size = core::mem::size_of::<T>();
            let first = want
                .chunks_exact(size)
                .zip(got.chunks_exact(size))
                .position(|(w, g)| w != g);
            let metrics = compare_views(&expected_view(expected)?, &actual_view(actual)?, 0);
            match first {
                None => Ok(Comparison {
                    first_mismatch: None,
                    ..metrics
                }),
                Some(i) => Err(fail(
                    &Comparison {
                        first_mismatch: Some(i),
                        ..metrics
                    },
                    "raw bits differ".to_string(),
                )),
            }
        }
        Tolerance::Snr {
            min_db,
            skip,
            max_abs_err,
        } => {
            let c = compare_views(&expected_view(expected)?, &actual_view(actual)?, skip);
            if c.snr_db.is_nan() || c.snr_db < min_db {
                return Err(fail(&c, format!("snr below the minimum of {min_db} db")));
            }
            if let Some(limit) = max_abs_err
                && (c.max_abs_err.is_nan() || c.max_abs_err > limit)
            {
                return Err(fail(
                    &c,
                    format!("largest error {} exceeds {limit}", c.max_abs_err),
                ));
            }
            Ok(c)
        }
        Tolerance::MaxAbsErr { max, skip, min_ref } => max_abs_err_check(
            &name,
            &expected_view(expected)?,
            &actual_view(actual)?,
            max,
            skip,
            min_ref,
        ),
    }
}

fn compare_views(expected: &View, actual: &View, skip: usize) -> Comparison {
    match (expected, actual) {
        (View::Real(e), View::Real(a)) => compare_real(e, a.iter().copied(), skip),
        (View::Complex(e), View::Complex(a)) => compare_complex(e, a.iter().copied(), skip),
        // A real expected value against complex data (or the reverse) is compared on
        // the real axis; callers get a meaningful SNR only for matching kinds.
        (View::Real(e), View::Complex(a)) => compare_real(e, a.iter().map(|c| c.re), skip),
        (View::Complex(e), View::Real(a)) => {
            let e: Vec<f64> = e.iter().map(|c| c.re).collect();
            compare_real(&e, a.iter().copied(), skip)
        }
    }
}

fn max_abs_err_check(
    name: &str,
    expected: &View,
    actual: &View,
    max: f64,
    skip: usize,
    min_ref: Option<(f64, RefUnit)>,
) -> Result<Comparison> {
    // (reference value, reference magnitude, absolute error) per sample; for complex
    // data the "value" is the magnitude.
    let samples: Vec<(f64, f64, f64)> = match (expected, actual) {
        (View::Real(e), View::Real(a)) => e
            .iter()
            .zip(a)
            .map(|(e, a)| (*e, libm::fabs(*e), libm::fabs(e - a)))
            .collect(),
        (View::Complex(e), View::Complex(a)) => e
            .iter()
            .zip(a)
            .map(|(e, a)| {
                let mag = libm::hypot(e.re, e.im);
                (mag, mag, libm::hypot(e.re - a.re, e.im - a.im))
            })
            .collect(),
        _ => {
            return Err(GoldenError::Validation(format!(
                "{name}: expected and actual differ in real/complex kind"
            )));
        }
    };
    let mut worst = 0.0f64;
    let mut first_bad = None;
    let mut compared = 0usize;
    let mut sig = 0.0;
    let mut err = 0.0;
    for (i, (value, mag, abs_err)) in samples.iter().copied().enumerate().skip(skip) {
        let selected = match min_ref {
            None => true,
            Some((level, RefUnit::Db)) => value >= level,
            Some((level, RefUnit::Linear)) => 20.0 * libm::log10(mag) >= level,
        };
        if !selected {
            continue;
        }
        compared += 1;
        sig += mag * mag;
        err += abs_err * abs_err;
        if abs_err.is_nan() {
            worst = f64::NAN;
        } else if abs_err > worst {
            worst = abs_err;
        }
        if (abs_err.is_nan() || abs_err > max) && first_bad.is_none() {
            first_bad = Some(i);
        }
    }
    let snr_db = if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * libm::log10(sig / err)
    };
    let c = Comparison {
        snr_db,
        max_abs_err: worst,
        rms_err: if compared == 0 {
            0.0
        } else {
            libm::sqrt(err / compared as f64)
        },
        first_mismatch: first_bad,
        compared,
    };
    if first_bad.is_some() {
        return Err(GoldenError::Mismatch {
            case: name.to_string(),
            snr_db,
            first_mismatch: first_bad,
            detail: format!("largest error {worst} exceeds {max}"),
        });
    }
    Ok(c)
}
