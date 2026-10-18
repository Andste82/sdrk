# M04 — Golden-vector infrastructure

> Plan: [PLAN.md › Testing and performance](../PLAN.md#testing-and-performance) · Depends on: [M03](M03-testkit-foundations.md) · Next: [M05](M05-first-kernels.md)

## Goal

Create the reproducible pipeline that produces golden samples with NumPy/SciPy, the on-disk format, the Rust loader in the testkit, the Python Q15 integer model, and the first cases (Q15 arithmetic cross-check and `scale`). Every later kernel milestone adds cases with the recipe defined here.

## Why

Our reference implementations must be validated by an *independent* implementation. NumPy/SciPy are the de-facto standard; the Python Q15 model is a second, independent implementation of R7 so that "bit-exact" means bit-exact against something we did not write in Rust.

## Ground rules (identical in every milestone — read them completely before you start)

These rules apply to every line of code in this repository. They are repeated in every milestone file so that each milestone is self-contained. If a rule and a milestone-specific instruction ever conflict, stop and ask; do not guess.

### R1 — Work order and scope

1. Milestones are implemented strictly in numeric order. A milestone may only use what earlier milestones delivered. Never implement items of a later milestone "while you are at it".
2. Every milestone must leave the repository in a state where **all** of these commands succeed from a clean checkout:
   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   - `cargo test --workspace --all-features`
   - `cargo xtask check` (exists from M00 on; runs the architecture checks in R2–R6)
   - `cargo xtask bare-metal` (exists from M00 on; builds every heap-free crate for `thumbv7em-none-eabihf`)
   - plus every additional command listed in the milestone's *Acceptance* section.
3. **Host platforms:** everything (build, tests, `xtask`, golden generator) must work on Linux x86_64, Linux arm64 (Raspberry Pi, 64-bit OS) and Windows x86_64. No shell scripts or Makefiles; tools are Rust (`xtask`) or the Python golden generator; paths are built with `Path::join`; line endings are fixed by `.gitattributes`. CI runs every tier-1 job on all three hosts.
4. When something is ambiguous, choose the simplest solution that satisfies the written tests, and record the decision in `docs/decisions-log.md` (one line: date, milestone, decision, reason). Never silently change a public API defined by an earlier milestone; if a change is unavoidable, update every caller and the affected docs in the same milestone and log it.
5. Naming: write **SDRk** in prose, docs and messages; `sdrk` only where lowercase is required (crate, package, module, file and path names).
6. Plan (living document): https://claude.ai/code/artifact/694f271a-ba38-4b7c-b2fa-fb2e919d35c1 — snapshot in [`docs/PLAN.md`](../PLAN.md). The plan explains *why*; the milestones say *what exactly* to build.

### R2 — Layers and dependencies

The folder of a crate is its layer. A crate may depend only on crates in **lower** layers (or, for the testkit, on any library layer). `cargo xtask check` enforces this by reading `cargo metadata` and the crate paths.

| Layer (folder) | Crates | May depend on (workspace crates) | `no_std`, no heap |
| --- | --- | --- | --- |
| `crates/core` (lowest) | `sdrk-device` | — | yes |
| `crates/core` | `sdrk-core` | `sdrk-device` | yes |
| `crates/backends` | `sdrk-backend-*` | `sdrk-core`, `sdrk-device` | yes |
| `crates/facade` | `sdrk` | core, backends | yes |
| `crates/blocks` | `sdrk-blocks` | facade, core | yes |
| `crates/pipeline` | `sdrk-pipeline` | blocks, facade, core | no (std/alloc allowed) |
| `crates/io` | `sdrk-io` | core (types only) | no |
| `crates/testing` | `sdrk-testkit` | any library crate above | `no_std` by default; feature `std` adds file I/O and proptest helpers |
| `bindings/*`, `examples/*`, `targets/*`, `benches/*`, `xtask` | — | anything | no |

Additional rules:
- Within `crates/core` the rank is by crate name: `sdrk-device` (rank 0) < `sdrk-core` (rank 1); `sdrk-core → sdrk-device` is allowed, the reverse is not. All other layers are one rank per folder.
- Backends never depend on each other. Only the facade `sdrk` knows the backends.
- Only `sdrk-testkit` and test code use the `reference` feature of `sdrk-core`.
- External crates allowed in heap-free crates: `num-complex` (default-features = false, features = ["bytemuck"]), `libm`, `bytemuck`, `heapless`, `fearless_simd` (backends only). Anything else needs a line in `docs/decisions-log.md` and must pass `cargo deny`.

### R3 — `no_std` and no heap (applies to every crate marked "yes" above)

1. The crate root starts with `#![no_std]`. The crate never writes `extern crate alloc;` and never uses `Vec`, `Box`, `String`, `Rc`, `Arc`, `BTreeMap` or anything else that allocates. `cargo xtask bare-metal` builds these crates for `thumbv7em-none-eabihf`, which has no global allocator, so any accidental allocation fails the build.
2. Memory is provided by the caller. State types are generic over their storage: a field that holds samples has a type parameter `S: AsRef<[T]> + AsMut<[T]>` (read-only data: `S: AsRef<[T]>`). On a microcontroller the caller passes `[T; N]` or `&mut [T]`; in upper layers the caller may pass `Vec<T>`. Constructors validate lengths and return `Err`, never panic.
3. Fixed-capacity containers come from `heapless` when needed.
4. No function may keep more than 4 KiB on its stack (WASM and MCU stacks are small). Large buffers belong in caller-provided storage. `clippy.toml` sets `stack-size-threshold = 4096` and the workspace enables `clippy::large_stack_frames` and `clippy::large_stack_arrays` as errors.
5. Math functions come from `libm` (e.g. `libm::sinf`, `libm::atan2f`, `libm::log10f`), never from `f32::sin()` & co. This keeps code `no_std` **and** makes results identical on every platform because `libm` is pure Rust.
6. Upper layers (`pipeline`, `io`, `bindings`, `examples`, `xtask`, testkit feature `std`) may use `std` and allocate. The only `std` use allowed below the heap boundary besides the testkit's `std` feature is runtime CPU-feature detection in backend crates behind their `std` feature (M13); it never allocates. The pipeline allocates only while *building*; after `build()` returns, processing must not allocate (tested with a counting global allocator in M10).

### R4 — Clean interfaces

1. Each crate exposes a small, documented public API through `lib.rs` re-exports; everything else is `pub(crate)`. Module layout inside a crate is an implementation detail.
2. Public enums that may grow are `#[non_exhaustive]`. Public structs with invariants have private fields, a validating constructor (`new(...) -> Result<Self>`) and getters.
3. Kernels follow the **one trait per kernel** pattern defined in M02: the trait is generic over the sample type, `Fallback` (in `sdrk-core`) implements it for every supported sample type, `Reference` (feature `reference`) implements it as the f64/integer specification, backends implement only the variants they accelerate, and the facade type `sdrk::Sdrk` implements all of them by delegating to the chosen implementation.
4. Kernel calls never allocate, never block, never read a clock, never use randomness, and never print.
5. Every public item has a rustdoc comment; every public function with non-trivial behaviour has a doc example that compiles (`#![deny(missing_docs)]` in every library crate; `cargo test --doc` runs the examples).
6. All public types are `Send`; state types and blocks are also `Sync` unless they need interior mutability (they should not). Add compile-time assertions in each crate's tests: `fn _assert_send<T: Send>() {}` called for every public type.

### R5 — One error-handling scheme for the whole workspace

1. The lowest crate `sdrk-device` defines the shared classification:
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
   #[non_exhaustive]
   pub enum ErrorKind {
       InvalidArgument, LengthMismatch, BufferTooSmall, OutOfRange, Unsupported,
       CapacityExceeded, InvalidState, ResourceLimit, Parse, Validation,
       Io, Device, Timeout, Disconnected, Internal,
   }
   pub trait SdrkError: core::error::Error {
       fn kind(&self) -> ErrorKind;
   }
   ```
2. Every library crate defines **its own** `pub enum Error` (`#[non_exhaustive]`, `Debug`, `Clone`, `PartialEq`) and `pub type Result<T, E = Error> = core::result::Result<T, E>;`. The enum implements `core::fmt::Display` (lowercase message, no trailing period), `core::error::Error` and `SdrkError`. Heap-free crates keep their error variants allocation-free (`&'static str` and integers only). Upper crates may carry `String` paths (e.g. `nodes.ddc.params.frequency_hz`).
3. Wrapping: an upper crate's error has a variant like `Core(sdrk_core::Error)` with `From` impls; `kind()` forwards to the inner error. `source()` returns the wrapped error.
4. **No panics in library code for any input that can reach a public API.** Every library crate root contains:
   `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::todo, clippy::unimplemented, clippy::indexing_slicing)]`
   Tests may opt out with `#[cfg_attr(test, allow(...))]` on the test module only. Validate lengths and parameters once at the public entry point; inner loops then iterate with `zip`, `chunks_exact`, `iter_mut` instead of indexing. Where indexing is unavoidable, use `.get()` and return `Error::Internal` if it ever fails.
5. Integer arithmetic that may overflow is written explicitly as `wrapping_*`, `saturating_*` or `checked_*`. The test profile keeps `overflow-checks = true`.

### R6 — Determinism and global state

1. No global mutable state: no `static mut`, no `static` with interior mutability (`Cell`, `RefCell`, `Mutex`, atomics, `OnceCell`, `LazyLock`), no `lazy_static`. Lookup tables are `const`/`static` immutable data, generated by `build.rs` where needed. `cargo xtask check` greps the source of library crates for these patterns and fails.
2. Same backend + same input = bit-identical output, independent of how the input is split into blocks. Every stateful kernel and block gets a block-size-invariance test (testkit helper from M03).
3. No reliance on CPU floating-point modes (flush-to-zero, rounding modes). Feedback states are flushed explicitly: `sdrk_core::num::flush_denormal(x)` returns 0.0 when `|x| < 1e-30`.
4. No hash-map iteration order in any output; use `BTreeMap` or ordered `Vec`s in upper layers.
5. Fused multiply-add is used only where the numeric contract allows it; the reference and the fallback never use FMA.
6. Little-endian only: `sdrk-device` contains `#[cfg(target_endian = "big")] compile_error!("SDRk supports little-endian targets only");`.

### R7 — Numeric conventions (from the plan's Decisions table)

| Topic | Rule |
| --- | --- |
| Q15 format | `Q15(i16)`, value = raw / 32768, range −1 … 1 − 2⁻¹⁵ |
| Rounding | round half up: add ½ LSB, then arithmetic right shift (`(x + (1 << (s - 1))) >> s` on a signed integer) |
| Overflow | always saturate to `[-32768, 32767]`, never wrap |
| Multiply | `sat16((a as i32 * b as i32 + (1 << 14)) >> 15)`; therefore `-1 × -1 = 32767` |
| Accumulation | reference and fallback accumulate Q30 products in `i64`, round and saturate once at the end; accelerated variants may use narrower accumulators if their headroom is documented |
| f32 → Q15 | computed in **f64**: `v = (x as f64) * 32768.0`; NaN → 0; otherwise `floor(v + 0.5)` clamped to `[-32768, 32767]` (so +1.0 → 32767, ±∞ → ±limit). Doing this in f32 would round twice and differ from the Python model |
| Q15 → f32 | `raw as f32 / 32768.0` (exact) |
| Complex Q15 | `CQ15 { re: Q15, im: Q15 }`, `#[repr(C)]`, I before Q — byte-identical to interleaved little-endian `cs16` |
| Denormals | flush feedback state to 0.0 below 1e-30 (R6.3) |
| Kernel classes | *bit-exact* kernels: fallback output equals the reference output bit for bit (Q15 and f32). *tolerance* kernels: output must reach the minimum SNR stated in the kernel's numeric contract (`docs/kernels.md`) against the f64 reference |

### R8 — Test vectors and golden samples (full specification in M04)

1. Golden samples live in `tests/golden/vectors/<kernel>/<case>.json` plus binary arrays `<case>.<array>.bin`. They are produced **only** by `tests/golden/generate.py` (Python 3.12, NumPy and SciPy with versions pinned exactly in `tests/golden/requirements.txt`). Never edit them by hand; never generate them with code from this repository.
2. Every case is deterministic: fixed seed via `numpy.random.default_rng(seed)`, fixed sizes, parameters stored in the JSON. Regenerating must produce byte-identical files (`python tests/golden/generate.py --check`).
3. Expected values are computed in float64 (`cf64`/`f64`). For bit-exact Q15 cases the generator computes the expected result with the integer model in `tests/golden/sdrk_golden/q15.py`, which implements R7 in NumPy `int64`.
4. Rust tests load cases with `sdrk_testkit::golden::load("<kernel>", "<case>")` (feature `std`). Tolerances live in the case JSON, not in Rust code.
5. Signal metric used everywhere: `SNR_dB = 10·log10( Σ|ref|² / Σ|ref − out|² )` over samples `skip..len` (`skip` stored per case; `+∞` when the error is zero). Bit-exact comparisons use `==` on raw bits. Tolerance classes (M04): `bit_exact`, `tolerance` (minimum SNR), `max_abs_err` (largest absolute error, optionally only where the reference is above a level).
6. Each case (JSON + all its `.bin` arrays) stays at or below 512 KiB; all vectors together at or below 16 MiB. Prefer N ≤ 4096 samples; use longer inputs only when the kernel needs them (long filters, decimation).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.


## Scope

**In:** `tests/golden/` Python package and CLI, `requirements.txt`, file format, Q15 model, loader `sdrk_testkit::golden`, cases `q15_ops/*` and `scale/*`, CI job, documentation of the recipe.

**Out:** cases for later kernels (each kernel milestone adds its own), GNU Radio vectors (optional, see §9).

## Specification

### 1. Directory layout

```text
tests/golden/
├── README.md                 # how to regenerate, how to add a case
├── requirements.txt          # exact pins: numpy==X.Y.Z, scipy==X.Y.Z (newest stable at implementation time)
├── generate.py               # CLI entry point
├── sdrk_golden/
│   ├── __init__.py
│   ├── registry.py           # @case decorator, CASES list
│   ├── fileio.py             # writer + reader for the on-disk format
│   ├── q15.py                # R7 integer model in numpy int64
│   └── cases/
│       ├── __init__.py       # imports every case module so they register
│       ├── q15_ops.py
│       └── scale.py
└── vectors/                  # generated output, committed to git
    └── <kernel>/<case>.json, <case>.<array>.bin
```

### 2. CLI

`python tests/golden/generate.py [--all | --kernel NAME] [--check] [--list] [--out DIR]`
- `--all` (default): generate every registered case into `tests/golden/vectors`.
- `--kernel NAME`: only cases of that kernel.
- `--check`: generate into a temporary directory and compare byte-for-byte with the committed files; print every differing or missing file; exit 1 on any difference, also when a committed file has no registered case (stale file).
- `--list`: print `kernel/case` lines.
Python 3.12, standard library + numpy + scipy only. Script must be runnable from any working directory (resolve paths relative to the script).

### 3. Case definition API

```python
from sdrk_golden.registry import case, Array

@case(kernel="scale", name="c32_random_4096", seed=1)
def scale_c32(rng):                       # rng = numpy.random.default_rng(seed), created by the registry
    x = ((rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) * 0.25).astype(np.complex64)
    gain = 0.75
    y = x.astype(np.complex128) * gain     # float64 reference computed from the quantized input
    return dict(
        params={"gain": gain},
        arrays={"input": Array("cf32", x), "expected": Array("cf64", y)},
        tolerance={"class": "tolerance", "snr_db_min": 140.0, "skip": 0},
        notes="output[n] = input[n] * gain",
    )
```
Rules:
- The `rng` passed in is the **only** randomness source. Draw order is part of the case; changing it changes the file (that is fine, `--check` will show it).
- Inputs are stored in the dtype the Rust test feeds to the kernel (`cf32` for C32 kernels, `ci16` for CQ15 kernels). The generator must quantize inputs itself (e.g. `x.astype(np.complex64)`, or `q15.from_float`), and compute `expected` **from the quantized input**, so that only kernel error is measured.
- `expected` is `f64`/`cf64` for tolerance cases, and the exact output dtype for bit-exact cases (`i16`/`ci16` for Q15 outputs, `f32`/`cf32` for bit-exact float outputs such as `convert` to C32).
- `tolerance` is one of
  - `{"class": "bit_exact"}` — raw bits equal;
  - `{"class": "tolerance", "snr_db_min": float, "skip": int}` with optional `"max_abs_err": float` (both must hold);
  - `{"class": "max_abs_err", "max_abs_err": float, "skip": int}` with optional `"min_ref_db": float` — only samples whose reference value is at or above `min_ref_db` (interpreted as dB for outputs that are already in dB, e.g. spectra; as `20·log10|ref|` otherwise) are checked. Used for windows, `power_db`, filter design and spectra.
- Keep each case at or below 512 KiB on disk and all vectors at or below 16 MiB (R8.6; assert both in the registry and fail generation otherwise).

### 4. On-disk format (`fileio.py`)

`<case>.json` (UTF-8, `json.dumps(obj, sort_keys=True, indent=2) + "\n"`, written with `open(path, "w", encoding="utf-8", newline="\n")` so that Windows hosts produce identical bytes; binary arrays are written with `open(path, "wb")`):
```json
{
  "format_version": 1,
  "kernel": "scale",
  "case": "c32_random_4096",
  "seed": 1,
  "params": {"gain": 0.75},
  "arrays": {
    "expected": {"file": "c32_random_4096.expected.bin", "dtype": "cf64", "len": 4096},
    "input":    {"file": "c32_random_4096.input.bin",    "dtype": "cf32", "len": 4096}
  },
  "tolerance": {"class": "tolerance", "snr_db_min": 140.0, "skip": 0},
  "notes": "output[n] = input[n] * gain",
  "provenance": {"generator": "tests/golden/generate.py", "python": "3.12.x", "numpy": "X.Y.Z", "scipy": "X.Y.Z"}
}
```
Binary arrays: raw little-endian, no header. dtypes: `u8`, `i8`, `i16`, `i32`, `f32`, `f64`, `ci16` (interleaved re, im as i16), `cf32` (interleaved f32), `cf64` (interleaved f64). `len` counts *elements* (complex samples count once). Float parameters in JSON are written with Python's shortest round-trip repr. Use `np.ascontiguousarray(arr, dtype=...).tobytes()` with explicit little-endian dtypes (`'<f8'`, `'<i2'`, …).

### 5. Q15 integer model (`q15.py`)

Implement R7 on `numpy.int64` arrays, mirroring the Rust function names:
```python
def sat16(x): return np.clip(x, -32768, 32767).astype(np.int64)
def round_shift(x, s): return x if s == 0 else (x + (1 << (s - 1))) >> s   # numpy >> on int64 is arithmetic
def from_float(x):            # float64 array -> int64 Q15 raw
    v = np.floor(np.where(np.isnan(x), 0.0, x) * 32768.0 + 0.5)
    v = np.where(np.isposinf(x), 32767, np.where(np.isneginf(x), -32768, v))
    return sat16(v.astype(np.int64))
def mul(a, b): return sat16(round_shift(a * b, 15))
def cmul(ar, ai, br, bi): return sat16(round_shift(ar*br - ai*bi, 15)), sat16(round_shift(ar*bi + ai*br, 15))
def to_float(q): return q.astype(np.float64) / 32768.0
```
Add a self-test function run by `generate.py` before generating: the R7 spot values (−32768·−32768 → 32767, 1.0 → 32767, NaN → 0, `1/65536` → 1, `−1/65536` → 0).

### 6. Rust loader (`sdrk_testkit::golden`, feature `std`)

Adds dependencies under the `std` feature: `serde` (derive), `serde_json`.

```rust
pub struct GoldenCase { pub kernel: String, pub case: String, pub seed: u64, pub params: serde_json::Value, pub tolerance: Tolerance, arrays: BTreeMap<String, Array> }
pub enum Tolerance { BitExact, Snr { min_db: f64, skip: usize, max_abs_err: Option<f64> }, MaxAbsErr { max: f64, skip: usize, min_ref_db: Option<f64> } }
pub enum Array { U8(Vec<u8>), I8(Vec<i8>), I16(Vec<i16>), I32(Vec<i32>), F32(Vec<f32>), F64(Vec<f64>), CI16(Vec<CQ15>), CF32(Vec<C32>), CF64(Vec<Complex64>) }
pub fn load(kernel: &str, case: &str) -> Result<GoldenCase, GoldenError>;
pub fn list(kernel: &str) -> Result<Vec<String>, GoldenError>;  // sorted case names
impl GoldenCase { pub fn array(&self, name: &str) -> Result<&Array, GoldenError>; pub fn param_f64(&self, name: &str) -> Result<f64, GoldenError>; pub fn param_u64(...); pub fn param_str(...); }
pub fn assert_matches<T: Sample>(case: &GoldenCase, expected: &Array, actual: &[T]) -> Result<Comparison, GoldenError>;
```
- Directory: env var `SDRK_GOLDEN_DIR` if set, else `<workspace root>/tests/golden/vectors` (find the workspace root from `CARGO_MANIFEST_DIR` by walking up to the directory containing `tests/golden`).
- Reading validates `format_version == 1`, file sizes against `len × dtype size`, and rejects unknown dtypes.
- `assert_matches` applies the case tolerance: `BitExact` → raw bit equality (`bit_exact`), `Snr` → `compare_complex`/`compare_real` with `skip` and `min_db` (and `max_abs_err` if present), `MaxAbsErr` → largest absolute error over the selected samples. Bit-exact float cases compare the f32 bits of `actual` with the stored `f32`/`cf32` expected. Returns a `GoldenError::Mismatch` with case name, achieved SNR and first mismatch index.
- `GoldenError` follows R5 (kinds `Io`, `Parse`, `Validation`).

### 7. Initial cases

`q15_ops` (bit-exact cross-check of `sdrk_core::num` against `q15.py`):
- `from_float_4096`: seed 2; inputs: 4000 values `rng.uniform(-1.2, 1.2)` + the 96 specials `[0, -0, 1, -1, 0.5, ±1/65536, ±3/65536, nan, inf, -inf, ±(1 - 2**-15), ±(2**-16 ± 2**-40), …]` (list them explicitly in the file), all converted with `x32 = x.astype(np.float32)` and stored as `f32` (the Rust function under test takes `f32`); expected `i16` = `from_float(x32.astype(np.float64))`. This also checks that the Rust side rounds in f64 (R7).
- `mul_pairs_4096`: seed 3; `a`, `b` = `rng.integers(-32768, 32768, 4096)` with the first 8 pairs forced to edge cases (−32768·−32768, −32768·32767, 32767·32767, 1·1, −1·1, 16384·16384, 0·x, −32768·1); expected `mul(a, b)`.
- `cmul_pairs_4096`: seed 4; complex pairs stored as `ci16`; expected `cmul`.

`scale`:
- `c32_random_4096` (as in §3), tolerance SNR ≥ 140 dB.
- `cq15_random_4096`: seed 5; input `ci16` from `from_float(0.9 · uniform(-1,1))`; gain Q15 raw 24576 (0.75) stored as param `gain_raw`; expected `ci16` via `mul` on re and im; bit-exact.

Rust tests in `crates/testing/sdrk-testkit/tests/golden_basic.rs` (feature `std`): load every case above, run the corresponding `sdrk_core::num` function or `Sdrk.scale(...)`, and call `assert_matches`.

### 8. CI

Add job `golden-check` (on changes under `tests/golden/**` and nightly): `actions/setup-python` 3.12, `pip install -r tests/golden/requirements.txt`, `python tests/golden/generate.py --check`. The normal `tier1` job does not need Python: it only reads committed vectors.

### 9. Optional: GNU Radio vectors

Not required for any milestone. If added later, they go to `tests/golden/gnuradio/` with their own script that runs inside the official GNU Radio Docker image (pinned tag), and produce the same file format. GNU Radio is GPL: only generated *data* enters the repo, never GNU Radio code.

### 10. Documentation

`tests/golden/README.md`: setup (`python3.12 -m venv .venv && . .venv/bin/activate && pip install -r tests/golden/requirements.txt`), regenerate, check, how to add a case (copy the template in §3, choose a new seed, quantize inputs, compute expected from quantized inputs, pick the tolerance class from `docs/kernels.md`, run `--kernel`, commit vectors together with code). Add a "Golden vectors" section to `docs/kernels.md` pointing there.

## Test vectors

Exactly the cases in §7.

## Tests

1. `python tests/golden/generate.py --check` passes on a clean checkout (run once locally, then in CI).
2. `cargo test -p sdrk-testkit --features std --test golden_basic` passes.
3. Loader unit tests: corrupt length → `Validation` error; unknown dtype → `Parse` error; missing case → `Io` error; `SDRK_GOLDEN_DIR` override works.
4. A deliberately wrong expected value (test-only fixture written to a temp dir by the test) produces `Mismatch` with the right index.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; the `golden-check` CI job is green.

## Deliverables checklist

- [ ] Python package, CLI, pinned requirements, Q15 model with self-test
- [ ] Vectors for `q15_ops/*`, `scale/*` committed
- [ ] Loader + `assert_matches` in the testkit
- [ ] Rust golden tests
- [ ] CI job, `tests/golden/README.md`, `docs/kernels.md` section
