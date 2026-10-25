# M02 — `sdrk-core` foundations and the facade skeleton

> Plan: [PLAN.md › Phase 1](../PLAN.md#phase-1--core-foundations-medium) · Depends on: [M01](M01-device-foundations.md) · Next: [M03](M03-testkit-foundations.md)

## Status

Fully completed: all items of the deliverables checklist below (types, num, error, timing, `Scale`, fallback, reference, `map_kernel!`, facade `Sdrk`/`delegate!`/`backend_for`, tests 1-8, `docs/kernels.md`, CHANGELOG, decisions log). All R1.2 commands (fmt, clippy, workspace tests, `cargo xtask check`, `cargo xtask bare-metal`) and `cargo test -p sdrk-core --features reference` pass.

Partial/open: nothing. No entries in `docs/open-items.md` for this milestone; judgement calls are recorded in `docs/decisions-log.md`.

## Goal

Establish the types and patterns every kernel will use: the four sample types, Q15 arithmetic, the crate error type, timing declarations (rate ratio, group delay), the **one-trait-per-kernel** pattern with `Fallback`, `Reference` and the facade `Sdrk`, and one complete example kernel (`scale`) that later milestones copy as a template.

## Why

The plan's decision "traits per sample type, composed per kernel in the facade" must be fixed in code before any real kernel exists; a generic method with per-type overrides is impossible in stable Rust (no specialization), so the trait itself is generic over the sample type.

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
3. **Host platforms:** everything (build, tests, `xtask`, golden generator) must work on Linux x86_64, Linux arm64 (Raspberry Pi, 64-bit OS) and Windows x86_64. No shell scripts or Makefiles; tools are Rust (`xtask`) or the Python golden generator; paths are built with `Path::join`; line endings are fixed by `.gitattributes`. CI builds the workspace and runs the whole test suite, `xtask` and the golden generator on all three hosts; the variants that depend only on the CPU architecture (SIMD levels fixed at compile time, builds without the static tables, the fallback-only build) run on one host per architecture (x86-64 on Linux, AArch64 on Linux arm64).
4. When something is ambiguous, choose the simplest solution that satisfies the written tests, and record the decision in `docs/decisions-log.md` (one line: date, milestone, decision, reason). Never silently change a public API defined by an earlier milestone; if a change is unavoidable, update every caller and the affected docs in the same milestone and log it.
5. Naming: write **SDRk** only in prose, code comments (incl. rustdoc) and documentation; everything else in code is lowercase `sdrk` (identifiers, crate/package/module/file names, string literals such as error messages). Exception: fixed upper-case machine tokens — the `SDRK_` environment-variable prefix, the emulator protocol markers `SDRK-SUITE`/`SDRK-CASE`/`SDRK-PANIC` and the wire magic `b"SDRK"`.
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
6. Little-endian only: `sdrk-device` contains `#[cfg(target_endian = "big")] compile_error!("sdrk supports little-endian targets only");`.

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
7. Tests that use `sdrk-testkit` live in integration tests (`<crate>/tests/*.rs`) or in `crates/testing/sdrk-testkit/tests/`, never in `#[cfg(test)]` unit modules of a crate the testkit itself depends on (a dev-dependency cycle would compile two copies of that crate and produce confusing trait errors).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.


## Scope

**In:** `sdrk-core` modules `types`, `num` (Q15 arithmetic, denormal flush), `error`, `timing`, `kernels` (trait pattern + `Scale`), `fallback`, `reference` (feature), the facade crate `sdrk` with `Sdrk` and backend introspection, `docs/kernels.md` (structure + Scale contract).

**Out:** real DSP kernels (M05+), testkit (M03), golden infrastructure (M04).

## Specification

### 1. Dependencies

`sdrk-core`: `sdrk-device`, `num-complex` (default-features = false, features = ["bytemuck"]), `bytemuck` (derive), `libm`. Feature `reference = []` (off by default).
`sdrk` (facade): `sdrk-core`, `sdrk-device`.

### 2. `types.rs`

```rust
/// Complex f32 sample; layout = interleaved little-endian cf32.
pub type C32 = num_complex::Complex32;

/// Q1.15 fixed-point real sample. value = raw / 32768.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct Q15(pub i16);

/// Complex Q15 sample; byte-identical to interleaved little-endian cs16 (I first).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CQ15 { pub re: Q15, pub im: Q15 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleType { C32, CQ15, F32, Q15 }
impl SampleType { pub const fn of<T: Sample>() -> Self { T::TYPE } }
```

Traits (sealed — implemented only for the four types):

```rust
pub trait Sample: Copy + Default + PartialEq + Send + Sync + bytemuck::Pod + core::fmt::Debug + 'static + sealed::Sealed {
    const TYPE: SampleType;
    const ZERO: Self;
}
pub trait RealSample: Sample { fn to_f64(self) -> f64; fn from_f64_round(x: f64) -> Self; }
pub trait ComplexSample: Sample {
    type Real: RealSample;
    fn re(self) -> Self::Real; fn im(self) -> Self::Real;
    fn new(re: Self::Real, im: Self::Real) -> Self;
    fn to_c64(self) -> num_complex::Complex64;
    fn from_c64_round(x: num_complex::Complex64) -> Self;
}
```
- `f32: RealSample` (`from_f64_round` = `x as f32`), `Q15: RealSample` (`to_f64` = raw/32768; `from_f64_round` = R7 conversion applied to the f64 value: NaN → 0, `floor(x·32768 + 0.5)` clamped).
- `C32: ComplexSample<Real = f32>`, `CQ15: ComplexSample<Real = Q15>`.
- Convenience: `Q15::from_f32(f32)`, `Q15::to_f32(self)`, `Q15::MAX = Q15(32767)`, `Q15::MIN = Q15(-32768)`, `CQ15::from_c32`, `CQ15::to_c32`.

### 3. `num.rs` — Q15 arithmetic (R7, exact)

All functions `#[inline]`, `const fn` where possible, documented with the formula:

| Function | Definition |
| --- | --- |
| `sat16(x: i64) -> i16` | clamp to `[-32768, 32767]` |
| `round_shift(x: i64, s: u32) -> i64` | `s == 0 → x`, else `(x + (1 << (s-1))) >> s` (arithmetic shift; round half up) |
| `q15_mul(a: Q15, b: Q15) -> Q15` | `sat16(round_shift(a·b, 15))` with `i64` intermediates |
| `q15_add_sat(a, b)`, `q15_sub_sat(a, b)` | saturating `i16` add/sub |
| `q15_from_f32(x: f32) -> Q15` | R7 |
| `cq15_mul(a: CQ15, b: CQ15) -> CQ15` | `re = sat16(round_shift(ar·br − ai·bi, 15))`, `im = sat16(round_shift(ar·bi + ai·br, 15))`, each computed in `i64` |
| `cq15_mul_conj(a, b)` | `a · conj(b)` analogously |
| `Acc64` | newtype over `i64` with `mac(a: Q15, b: Q15)` (adds `a·b`) and `finish(self) -> Q15` = `sat16(round_shift(acc, 15))` |
| `flush_denormal(x: f32) -> f32` | `if libm::fabsf(x) < 1e-30 { 0.0 } else { x }` (also for `f64`: `flush_denormal_f64`) |

### 4. `error.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    LengthMismatch { expected: usize, got: usize },   // kind LengthMismatch
    BufferTooSmall { needed: usize, got: usize },     // kind BufferTooSmall
    InvalidParameter(&'static str),                   // kind InvalidArgument
    Unsupported(&'static str),                        // kind Unsupported
    Device(sdrk_device::Error),                       // kind forwarded
    Internal(&'static str),                           // kind Internal
}
pub type Result<T, E = Error> = core::result::Result<T, E>;
```
`From<sdrk_device::Error>`; Display/Error/SdrkError per R5.

### 5. `timing.rs` — what every kernel/block declares

```rust
pub use sdrk_device::{Ratio, Rate, SampleIndex};

/// Group delay in *input* samples as an exact fraction (e.g. 31/1 or 63/2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Delay { num: u64, den: u64 }   // new(num, den) -> Result, ZERO, as_f64(), add(), reduced

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RatioKind { Constant, Variable }

/// Implemented by every kernel *state* and every block.
pub trait Timing {
    /// output_rate / input_rate
    fn ratio(&self) -> Ratio;
    fn ratio_kind(&self) -> RatioKind { RatioKind::Constant }
    /// Delay between an input sample and its effect on the output, in input samples.
    fn group_delay(&self) -> Delay;
}

/// Result of a rate-changing kernel call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Progress { pub consumed: usize, pub produced: usize }
```

### 6. The kernel pattern (normative for every later kernel)

For each kernel `foo` there is:
1. **A trait** in `sdrk_core::kernels::<category>`:
   ```rust
   pub trait Foo<T: Sample> {
       fn foo(&self, /* state, */ input: &[T], output: &mut [T] /* , params */) -> Result<()>;
   }
   ```
   - Same-rate kernels require `output.len() == input.len()` else `Error::LengthMismatch { expected: input.len(), got: output.len() }`.
   - Rate-changing kernels return `Result<Progress>`; they consume as much input as possible and produce at most `output.len()` samples; leftover input is *not* consumed (the caller resubmits it).
   - Kernels with state receive `state: &mut FooState<...>` as the first argument after `&self`.
2. **`Fallback`** (`pub struct Fallback;` in `sdrk_core::fallback`) implements the trait for every supported sample type. It is the portable, optimised-scalar version.
3. **`Reference`** (`pub struct Reference;`, `#[cfg(feature = "reference")]`) implements the trait as the executable specification: f64 computation for f32 kernels, the exact R7 integer model for Q15 kernels. It favours clarity over speed.
4. **The facade** `sdrk::Sdrk` (`pub struct Sdrk;`, `Copy`, `Default`) implements the trait for every sample type by delegation. Delegation is written with the macro `sdrk::delegate!` (defined in the facade crate):
   ```rust
   delegate!(Scale<C32>::scale(&self, input: &[C32], output: &mut [C32], gain: f32) -> Result<()> => Fallback);
   ```
   which expands to an `impl Scale<C32> for Sdrk` forwarding to `<Fallback as Scale<C32>>::scale(&Fallback, …)`. Later backends change only the `=> Backend` target, guarded by `#[cfg(...)]`.
5. **Backend introspection** in the facade: `pub fn backend_for(kernel: &'static str, ty: SampleType) -> &'static str` backed by a `const` table generated by the same `delegate!` invocations (the macro also emits a table entry; keep a single source of truth). Unknown → `"none"`.
6. **Registration in `docs/kernels.md`** with its numeric contract (class bit-exact or tolerance + min SNR).

### 7. Example kernel `scale` (complete template)

`kernels::basic::Scale<T: ComplexSample>`:
```rust
pub trait Scale<T: ComplexSample> {
    /// output[n] = input[n] · gain (real gain)
    fn scale(&self, input: &[T], output: &mut [T], gain: T::Real) -> Result<()>;
}
```
- C32: plain f32 multiply per component (no FMA).
- CQ15: `q15_mul` per component (bit-exact).
- Reference C32: compute in f64 and round to f32 (`as f32`). Note: for `scale` the f64 path and the f32 path can differ in the last bit; the contract is therefore *tolerance* for C32 (SNR ≥ 140 dB) and *bit-exact* for CQ15. Write this in `docs/kernels.md`.
- Facade delegations for C32 and CQ15 to `Fallback`.

### 8. Per-sample helper macro

`sdrk_core::map_kernel!` generates a fallback body for same-rate, stateless, per-sample kernels: it validates lengths, then `for (o, &i) in output.iter_mut().zip(input) { *o = $expr(i) }`. Use it for `Scale`. Document it as the starting point for later SIMD "process one" helpers.

### 9. `lib.rs`

Re-export: `types::*`, `num` (as module), `error::{Error, Result}`, `timing::*`, `kernels`, `fallback::Fallback`, `#[cfg(feature="reference")] reference::Reference`.

## Test vectors

No golden files yet (they start in M04, which will add cross-checks of `num` against the Python Q15 model). Use the explicit values below.

## Tests

1. `round_shift`: (3, 1) → 2, (2, 1) → 1, (−3, 1) → −1, (−2, 1) → −1, (−1, 1) → 0, (x, 0) → x.
2. `q15_mul`: (−32768, −32768) → 32767; (16384, 16384) → 8192; (−16384, 16384) → −8192; (1, 1) → 0; (32767, 32767) → 32766; plus 1 000 000 pairs from a fixed-seed xorshift64 generator compared with the `i64` formula written directly in the test.
3. `q15_from_f32`: 1.0 → 32767, −1.0 → −32768, 0.5 → 16384, NaN → 0, +∞ → 32767, −∞ → −32768, `1.0/65536.0` → 1 (half rounds up), `-1.0/65536.0` → 0; exhaustive round trip: for all 65 536 raw values `q15_from_f32(Q15(r).to_f32()) == Q15(r)`.
4. `cq15_mul` identities: multiply by (32767, 0) changes values by at most 1 LSB; multiply by (0, 32767) rotates by 90° (re' = −im, im' = re within 1 LSB); bit-exact against the i64 formula for 100 000 random pairs.
5. `Acc64`: sum of 1000 products of full-scale values does not overflow and `finish` saturates.
6. `flush_denormal`: 1e-31 → 0, −1e-31 → 0, 1e-29 unchanged, NaN unchanged.
7. `Scale`: Fallback vs Reference on 4096 random samples (fixed seed): CQ15 bit-exact; C32 SNR ≥ 140 dB (compute SNR in the test with f64 sums). Length mismatch → `LengthMismatch`. Facade `Sdrk` gives identical results to `Fallback`. `backend_for("scale", SampleType::C32) == "fallback"`.
8. `Delay`, `Progress`, `SampleType::of` sanity tests; `Send + Sync` assertions for all public types.

## Acceptance

R1.2 commands; `cargo test -p sdrk-core --features reference`; `cargo xtask bare-metal` builds `sdrk-core` and `sdrk`.

## Deliverables checklist

- [x] `types`, `num`, `error`, `timing`, `kernels::basic::Scale`, `fallback`, `reference`, `map_kernel!`
- [x] Facade `Sdrk`, `delegate!`, `backend_for`
- [x] Tests 1–8
- [x] `docs/kernels.md`: kernel pattern, contract table with `scale`
- [x] CHANGELOG, decisions log

Status: all deliverables are complete. `cargo test -p sdrk-core --features reference` and `cargo xtask bare-metal` (builds `sdrk-core` and `sdrk`) pass.
