# M09 — `sdrk-blocks`: filter design, storage, block interface, first blocks

> Plan: [PLAN.md › Phase 3](../PLAN.md#phase-3--blocks-pipeline-files-large) · Depends on: [M08](M08-transmit-and-fused.md) · Next: [M10](M10-pipeline-core.md)

## Status

Fully completed: filter design checked against SciPy golden vectors, the storage abstraction (`Vec` and byte arena), the block interface with mandatory machine-readable descriptions, all listed blocks (`Convert`, `ConvertOut`, `Scale`, `DcBlock`, `Fir`, `Ddc`, `Duc`, `FmDemod`, `FmMod`, `Resampler`, `Spectrum`), the dot kernels with reference and fallback implementations, golden and end-to-end tests, `docs/blocks.md`, and the bare-metal build. All R1.2 commands pass on Linux x86_64.

Open: CI confirmation on Windows x86_64 and Linux arm64 (and the first green CI run of `golden-check`, `bare-metal`, `test`) cannot be done locally; tracked in `docs/open-items.md`. Decisions taken are in `docs/decisions-log.md`.

## Goal

Create the heap-free crate `crates/blocks/sdrk-blocks`: FIR filter design checked against SciPy, a storage abstraction that works with `Vec` (pipeline) and with a byte arena (microcontroller), the uniform block interface with **mandatory machine-readable descriptions**, and the blocks needed for FM receive/transmit and spectra: `Convert`, `ConvertOut`, `Scale`, `DcBlock`, `Fir`, `Ddc`, `Duc`, `FmDemod`, `FmMod`, `Resampler` (rational and ppm-tuned), `Spectrum`.

## Why

Kernels are numeric building blocks with caller-designed taps. Users (and the pipeline in M10, and AI agents in the follow-up projects) think in parameters with units: "DDC to 600 kHz, decimate by 10, cutoff 100 kHz". Blocks translate those parameters into kernel states, and their descriptions make the library self-documenting.

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

**In:** crate `sdrk-blocks` (layer `crates/blocks`, `no_std`, no heap), modules `design`, `storage`, `describe`, `block`, `blocks::*`; new kernel `Dot` in `sdrk-core` (needed by the ppm resampler); golden cases; suite cases; `docs/blocks.md`.

**Out:** anything pipeline-related (JSON, registry, scheduling, events — M10/M11); preemphasis, AGC, squelch, AM/SSB (backlog).

## Specification

### 1. Crate setup

`crates/blocks/sdrk-blocks`, `#![no_std]`, dependencies `sdrk` (facade), `sdrk-core`, `sdrk-device`, `libm`, `bytemuck`, `heapless`. Register the layer in `xtask` (path prefix `crates/blocks`, heap-free), so `check heap` and `bare-metal` cover it automatically. Own `Error` per R5 with variants `Core(sdrk_core::Error)`, `InvalidParameter { name: &'static str, reason: &'static str }` (kind `InvalidArgument`), `OutOfRange { name: &'static str, min: f64, max: f64, got: f64 }` (kind `OutOfRange`), `Storage { needed: usize, available: usize }` (kind `CapacityExceeded`), `UnknownParameter` (unit variant — runtime names are borrowed `&str` and cannot be stored; kind `InvalidArgument`), `ReadOnlyParameter` (kind `InvalidState`), `WrongParamType` (kind `InvalidArgument`).

All blocks use the facade `sdrk::Sdrk` for every kernel call (never `Fallback` directly), so backends speed up blocks automatically.

### 2. Storage abstraction (`storage.rs`)

```rust
/// Source of buffers for block construction. Blocks never allocate themselves.
pub trait Storage {
    type Buf<T: bytemuck::Pod + Default + Send>: AsRef<[T]> + AsMut<[T]> + Send;
    /// A zero-initialised buffer of exactly `len` elements.
    fn alloc<T: bytemuck::Pod + Default + Send>(&mut self, len: usize) -> Result<Self::Buf<T>>;
}

/// Heap-free storage carved from one caller-provided byte slice (bump allocation, never freed).
pub struct ArenaStorage<'a> { rest: &'a mut [u8], used: usize }
impl<'a> ArenaStorage<'a> { pub fn new(bytes: &'a mut [u8]) -> Self; pub fn used(&self) -> usize; pub fn remaining(&self) -> usize; }
impl<'a> Storage for ArenaStorage<'a> { type Buf<T: ...> = &'a mut [T]; /* ... */ }
```
Sizing is done per block with `storage_bytes(config, rate)` (see §4), not through the storage trait.
`ArenaStorage::alloc`: take the remaining slice with `core::mem::take`, compute the padding with `rest.as_ptr().align_offset(align_of::<T>())`, `split_at_mut` the padding and `len · size_of::<T>()` bytes, convert with `bytemuck::try_cast_slice_mut` (no `unsafe`), zero-fill, update `used`. Not enough space → `Error::Storage { needed, available }`. `sdrk-pipeline` (M10) implements `Storage` with `Buf<T> = Vec<T>`; the testkit adds the same `VecStorage` behind its `std` feature for tests in this milestone.

Requirement for C32/CQ15/f32/Q15: they implement `bytemuck::Pod` and `Default` (from M02; add the missing derives there if necessary and log it).

### 3. Filter design (`design.rs`, computed in f64, written to caller slices)

All functions mirror SciPy so they can be checked against golden vectors:

```rust
#[non_exhaustive] pub enum DesignWindow { Hann, Hamming, Blackman, Kaiser { beta: f64 } }
/// = scipy.signal.firwin(num_taps, cutoff_hz, window=…, pass_zero=True, scale=True, fs=fs_hz)
pub fn firwin_lowpass(cutoff_hz: f64, fs_hz: f64, window: DesignWindow, taps: &mut [f64]) -> Result<()>;
/// = scipy.signal.kaiserord(ripple_db, width = 2·transition_hz / fs_hz); returns (num_taps, beta) with num_taps forced odd (+1 if even)
pub fn kaiserord(ripple_db: f64, transition_hz: f64, fs_hz: f64) -> Result<(usize, f64)>;
/// = scipy.signal.kaiser_beta(a)
pub fn kaiser_beta(attenuation_db: f64) -> f64;
pub fn to_f32(src: &[f64], dst: &mut [f32]) -> Result<()>;
/// R7 rounding; error if any |tap| > 1 − 2^-15 by more than 1 % (tap would saturate badly)
pub fn to_q15(src: &[f64], dst: &mut [Q15]) -> Result<()>;
```
Formulas (normative, copy into the rustdoc):
- `firwin_lowpass`: `fc = cutoff_hz / fs_hz` (0 < fc < 0.5 else `OutOfRange`), `m = (N − 1)/2`, `h[n] = 2·fc · sinc(2·fc·(n − m)) · w[n]` with `sinc(x) = sin(πx)/(πx)`, `sinc(0) = 1`; symmetric window `w` (Hann `0.5 − 0.5 cos(2πn/(N−1))`, Hamming `0.54 − 0.46 cos(…)`, Blackman `0.42 − 0.5 cos(2πn/(N−1)) + 0.08 cos(4πn/(N−1))`, Kaiser via `bessel_i0` from M06 with `I0(β·sqrt(1 − (2n/(N−1) − 1)²)) / I0(β)`); then scale so that `Σ h = 1`. `N ≥ 1`; `N = 1` → `[1.0]`.
- `kaiserord`: `A = |ripple_db|`, `width = 2·transition_hz/fs_hz`; `A < 8` → `InvalidParameter`; `N = ceil((A − 7.95) / (2.285·π·width) + 1)`; `beta = kaiser_beta(A)`.
- `kaiser_beta`: `A > 50` → `0.1102·(A − 8.7)`; `21 ≤ A ≤ 50` → `0.5842·(A − 21)^0.4 + 0.07886·(A − 21)`; else `0`.

`DesignSpec { cutoff_hz, transition_hz, attenuation_db }` is the common parameter set used by blocks. A block designs its taps with `kaiserord` + `firwin_lowpass(Kaiser)`; the tap count is capped by the block's `max_taps` parameter (default 1023) → `OutOfRange` if exceeded.

### 4. Block interface (`block.rs`)

```rust
/// One streaming input, one streaming output (multi-port routing is the pipeline's job).
pub trait Block: Send {
    type In: Copy + Send;
    type Out: Copy + Send;
    fn description(&self) -> &'static BlockDescription;
    /// Consume as much input as possible, produce at most output.len(); same contract as rate-changing kernels.
    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> Result<Progress>;
    /// Exact ratio out/in and group delay (in input samples).
    fn timing(&self) -> BlockTiming;               // { ratio: Ratio, kind: RatioKind, delay: Delay }
    /// Clear all history (used by the pipeline on gaps); parameters stay.
    fn reset(&mut self);
    /// Change a runtime parameter. The new value applies to the first sample of the next `process` call.
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;
    /// Current value of a parameter (for lossless config readout).
    fn param(&self, name: &str) -> Result<ParamValue>;
    /// Smallest input count that is guaranteed to produce at least one output (used for buffer sizing).
    fn min_input(&self) -> usize;
}

/// Blocks that output frames (spectrum, measurements) instead of a stream.
pub trait FrameBlock: Send {
    type In: Copy + Send;
    fn description(&self) -> &'static BlockDescription;
    fn frame_len(&self) -> usize;
    /// Consume input; when a frame completes, write it into `frame` and return its metadata. At most one frame per call:
    /// the call stops consuming right after completing a frame.
    fn push(&mut self, input: &[Self::In], frame: &mut [f32]) -> Result<(usize, Option<FrameMeta>)>;
    fn reset(&mut self);
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;
    fn param(&self, name: &str) -> Result<ParamValue>;
}
pub struct FrameMeta { pub first_sample: u64 /* input index of the first sample in the frame's window, counted since construction/reset */, pub averaged: u32, pub exponent: i32 }

#[derive(Clone, Copy, Debug, PartialEq)] #[non_exhaustive]
pub enum ParamValue { F64(f64), I64(i64), Bool(bool), Choice(u16) /* index into ParamKind::Choice */, Format(SampleFormat) }
```
Construction pattern for every block `Foo` (normative):
```rust
pub struct FooConfig { /* pub fields with units in names: frequency_hz, cutoff_hz, decimation … */ }
impl FooConfig { pub fn validate(&self, rate_in: Rate) -> Result<()>; }
impl Default for FooConfig { /* the "typical" values from the description */ }
impl<T, S: Storage> Foo<T, S> {
    pub fn new(config: &FooConfig, rate_in: Rate, storage: &mut S) -> Result<Self>;
    pub fn storage_bytes(config: &FooConfig, rate_in: Rate) -> Result<usize>; // exact bytes for an arena starting at a 16-byte aligned address
    pub fn rate_out(config: &FooConfig, rate_in: Rate) -> Result<Rate>;
}
```
`storage_bytes` is **exact** for an `ArenaStorage` whose byte slice starts at a 16-byte aligned address: buffers are allocated in a fixed order and each one starts at the next multiple of its element alignment, so the padding is deterministic. (Callers with unaligned slices may need up to 15 more bytes; document this.) Provide `pub const ARENA_ALIGN: usize = 16;` and a helper type `#[repr(C, align(16))] pub struct AlignedBytes<const N: usize>(pub [u8; N]);` for tests and MCU users.

Every config additionally implements the name-based access used by the pipeline (M10), driven by the same `ParamDesc` table:
```rust
impl FooConfig {
    /// Set a parameter by its description name (build *and* runtime parameters); checks kind and range.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()>;
    pub fn get(&self, name: &str) -> Result<ParamValue>;
}
```

### 5. Descriptions (`describe.rs`) — mandatory

```rust
pub struct BlockDescription {
    pub kind: &'static str,            // snake_case, unique, e.g. "ddc"
    pub summary: &'static str,         // one sentence
    pub details: &'static str,         // a paragraph: what it does, typical use, pitfalls
    pub inputs: &'static [PortDesc], pub outputs: &'static [PortDesc],
    pub params: &'static [ParamDesc],
    pub required: &'static [&'static str],   // parameter names that must be given (no default at all)
}
pub struct PortDesc { pub name: &'static str, pub types: &'static [PortType], pub description: &'static str }
#[non_exhaustive] pub enum PortType { C32, CQ15, F32, Q15, Raw /* bytes in a SampleFormat */, Frame { len_param: &'static str } }
pub struct ParamDesc {
    pub name: &'static str, pub unit: Unit, pub kind: ParamKind,
    pub typical: &'static str,         // human text, e.g. "100e3 for FM broadcast"
    pub mutability: Mutability,        // Build | Runtime
    pub description: &'static str,
}
#[non_exhaustive] pub enum Unit { None, Hz, Seconds, Microseconds, Decibel, Ppm, Samples, Ratio }
#[non_exhaustive] pub enum ParamKind {
    F64 { min: f64, max: f64, default: Option<f64> },   // None = required, or the default depends on the rate (see below)
    I64 { min: i64, max: i64, default: Option<i64> },
    Bool { default: bool },
    Choice { options: &'static [&'static str], default: u16 },
    Format,                            // a SampleFormat given by name (e.g. "cu8"); default none
}
```
`min`/`max` are the absolute limits that the schema can check; limits and defaults that depend on the input rate (e.g. `frequency_hz` within ±fs/2, `cutoff_hz` default `0.4·fs/decimation`) are applied by `FooConfig::validate`/`new`, described in `typical`/`description`, and their `default` is `None` with the rate-dependent rule documented (the schema then omits `default`; the pipeline does not report such a parameter as missing — only parameters listed in the description's `required` field, e.g. `resampler.rate_out_hz`, are required).

Rules (tested): every text non-empty and without trailing whitespace; `summary` ≤ 120 characters and ends with a period; parameter names snake_case and unique per block; numeric parameters carry a unit (or `Unit::None` explicitly for dimensionless counts); defaults lie inside the range; every runtime parameter is accepted by `set_param` and readable by `param`; `pub fn all_descriptions() -> &'static [&'static BlockDescription]` lists every block.

### 6. Blocks (sample types: complex blocks for C32 and CQ15; audio blocks for f32 and Q15)

| Kind | Struct | In → Out | Parameters (unit, mutability, default) | Implementation |
| --- | --- | --- | --- | --- |
| `convert` | `ConvertBlock<T>` | Raw → C32/CQ15 | `format` (Format, build) | `convert`; `In = u8`; consumes whole packing groups only; `Progress` counts bytes in, samples out |
| `convert_out` | `ConvertOutBlock<T>` | C32/CQ15 → Raw | `format` (Format, build) | `convert_out`; `Out = u8`, produces whole groups |
| `scale` | `ScaleBlock<T>` | T → T | `gain_db` (dB, runtime, 0; −120…+40 for C32, −120…0 for CQ15) | `scale` |
| `dc_block` | `DcBlockBlock<T>` | T → T | `cutoff_hz` (Hz, build, 10) → `r = exp(−2π·cutoff/fs)` | `dc_block` |
| `fir` | `FirBlock<T>` | T → T | `cutoff_hz` (build), `transition_hz` (build), `attenuation_db` (build, 60), `max_taps` (build, 1023) | design + `fir` |
| `ddc` | `DdcBlock<T>` | T → T | `frequency_hz` (Hz, **runtime**, 0; ±fs/2), `decimation` (I64, build, 1…1024), `cutoff_hz` (build, default 0.4·fs/decimation), `transition_hz` (build, default 0.1·fs/decimation), `attenuation_db` (build, 60), `max_taps` | NCO at `−frequency_hz` (shift the signal at +f to 0 Hz) + designed taps + fused `ddc` with scratch sized to `max_chunk` (build param, default 4096) |
| `duc` | `DucBlock<T>` | T → T | `frequency_hz` (runtime), `interpolation` (build), `cutoff_hz`, `transition_hz`, `attenuation_db`, `max_taps` | taps · L, fused `duc` |
| `fm_demod` | `FmDemodBlock<T>` | C32→f32, CQ15→Q15 | `deviation_hz` (build, 75e3), `deemphasis_us` (Microseconds, build, 50; 0 = off), `audio_cutoff_hz` (build, 15e3), `audio_decimation` (build, 1…64, default 1) | `quad_demod` (gain per M08: f32 `fs/(2π·dev)`; Q15 gain raw 32767, as in M08 E2E-RX) + `deemphasis` + `resample_real` (L=1, M=audio_decimation) |
| `fm_mod` | `FmModBlock<T>` | f32→C32, Q15→CQ15 | `deviation_hz` (runtime, 75e3) | `fm_mod` |
| `resampler` | `ResamplerBlock<T>` | T → T (complex) and `ResamplerRealBlock<T>` (real) | `rate_out_hz` (build, required; together with the input rate defines L/M = reduce(rate_out/rate_in), L, M ≤ 1024), `cutoff_hz` (build, default 0.45·min(rate_in, rate_out)), `transition_hz`, `attenuation_db`, `ppm` (Ppm, **runtime**, 0, ±1000) | see §7 |
| `spectrum` | `SpectrumBlock<T>` (FrameBlock) | C32/CQ15 → Frame | `fft_size` (build, 1024, power of two 16…65536), `window` (Choice hann/blackman_harris/kaiser, build), `kaiser_beta` (build, 8.6), `averaging` (I64, runtime, 1…1024, default 1), `interval_samples` (Samples, runtime, default = fft_size; ≥ fft_size·averaging not required — overlapping windows allowed when smaller) | see §8 |

`rate_in` for `ConvertBlock` is the sample rate; byte counts derive from `format.bytes_for_samples`.

### 7. Resampler

- **Rational mode** (`ppm == 0` at construction and never changed — if `ppm` is a runtime parameter the block always uses the tuned mode; therefore add build parameter `tunable` (Bool, default false): false → rational, `set_param("ppm")` returns `ReadOnlyParameter`).
  Taps designed at `L·rate_in`, multiplied by `L`; uses `resample` / `resample_real`. Ratio kind `Constant`.
- **Tuned mode** (`tunable = true`): polyphase bank with `P = 128` phases. Design one prototype low-pass of `P·K` taps (`K` = taps per phase from `kaiserord` at the input rate, rounded up) at `P·rate_in` with gain `P`; phase `p` uses taps `h[p + i·P]`, `i = 0…K−1`. Position accumulator `pos: u64` in 32.32 fixed point (input samples). Step `= round(2^32 · rate_in / rate_out / (1 + ppm·1e-6))`, recomputed on `set_param("ppm")` (applies from the next output). For each output: `n = pos >> 32` (needs input samples up to `n`), `frac = pos & 0xFFFF_FFFF`, `p = frac >> 25` (top 7 bits), `mu = (frac >> 9) & 0xFFFF` (next 16 bits as Q16), `y = dot(x[n−K+1 ..= n] reversed, bank[p]) · (1 − mu) + dot(…, bank[p+1]) · mu` where `bank[P]` = bank[0] applied one sample later (`x[n−K+2 ..= n+1]`, so the output waits for sample `n+1`). Then `pos += step`. Keep `pos` relative to the oldest input sample still held in the history (subtract `consumed << 32` whenever whole input samples are released), so it never overflows however long the stream runs. C32 interpolation in f32 (`mu` as `f32 = mu / 65536`); CQ15: `round_shift(a·(65536 − mu) + b·mu, 16)` in i64, saturate. Ratio kind `Variable`; `timing().ratio` returns the nominal L/M.
- New kernel in `sdrk-core` (follow the M02 kernel pattern completely: Fallback, Reference, facade, golden, suite): `pub trait Dot<T: ComplexSample> { fn dot(&self, x: &[T], taps: &[T::Real]) -> Result<T>; }` and `pub trait DotReal<T: RealSample> { fn dot_real(&self, x: &[T], taps: &[T]) -> Result<T>; }` (the real variant is needed by `ResamplerRealBlock`), both `= Σ x[i]·taps[i]` with `x.len() == taps.len()` (else `LengthMismatch`), accumulation rules as `fir` (f32 ascending without FMA; Q15/CQ15 `Acc64`, one `round_shift(…, 15)` and saturation at the end). C32/f32 tolerance ≥ 120 dB; CQ15/Q15 bit-exact.

### 8. Spectrum

For every frame: take `fft_size` consecutive input samples starting at the frame position, multiply by the window (`window` kernel; window generated with `Symmetry::Periodic`), FFT forward (C32: `Scaling::Default`; CQ15: `Scaling::BlockFloat`), power `P[k]`: C32 via `mag_sq`; CQ15 **not** via Q15 `mag_sq` (it saturates for |X| ≥ 1 and rounds weak bins to 0) but in f32 from the CQ15 bins: `P[k] = ((re as f32)² + (im as f32)²) · 2^(2·exponent − 30)` (M07 exponent rule), accumulate over `averaging` frames (arithmetic mean of power), then `power_db` with reference `(fft_size · coherent_gain)²` so that a full-scale complex tone at a bin centre reads 0 dBFS, and finally **fftshift** (bin order −fs/2 … +fs/2, DC at index `fft_size/2`). Consecutive FFT windows start `interval_samples` apart (overlap if smaller than `fft_size`; samples are skipped if larger). `FrameMeta.first_sample` = index of the first sample of the **first** averaged window; `exponent` = 0 (already applied). Storage: input ring of `fft_size`, FFT buffer, twiddles, window, accumulator.

### 9. Facade/core additions and documentation

- `sdrk-core`: kernels `Dot` and `DotReal` (§7).
- `docs/blocks.md`: block interface, construction pattern, storage model (Vec vs arena), description rules, table of blocks generated by a test: `cargo test -p sdrk-blocks --test describe -- --ignored print` writes the Markdown table to stdout; the committed table must match (test compares).
- `docs/timing.md`: timing of each block (ratio, delay formula).

## Test vectors (`tests/golden/sdrk_golden/cases/design.py`, `blocks.py`, `dot.py`)

| Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- |
| `design/firwin_{hann,hamming,blackman}_{31,128}` | — (no randomness; seed 0) | params `cutoff_hz=0.1·fs`, `fs=1` (and `cutoff 0.23`) | `scipy.signal.firwin(N, c, window=…, fs=1)` as `f64` | max_abs_err ≤ 1e-12 |
| `design/firwin_kaiser_{beta}` | 0 | β ∈ {0.5, 5.0, 8.6}, N 101 | `firwin(N, c, window=('kaiser', β), fs=1)` | max_abs_err ≤ 1e-12 |
| `design/kaiserord_table` | 0 | 20 pairs (ripple 20…100 dB, width 0.01…0.2) stored as params arrays | `kaiserord` (num_taps forced odd) and `beta` | num_taps exact, beta ≤ 1e-12 |
| `dot/c32_{7,64,1023}` | 300–302 | noise σ 0.3 complex64, taps uniform(−0.5, 0.5) float32 | f64 dot | SNR ≥ 120 |
| `dot/cq15_{7,64,1023}` | 303–305 | quantized | integer model | bit-exact |
| `dot/f32_{7,1023}` | 306–307 | real noise σ 0.3 float32, taps as above | f64 dot | SNR ≥ 120 |
| `dot/q15_{7,1023}` | 308–309 | quantized | integer model | bit-exact |
| `blocks/ddc_c32_fs2m4_f600k_d10` | 310 | 20 000 samples: tone at +600 kHz + 50 kHz offset (amp 0.5) + noise σ 0.05, complex64 | Python chain: taps via `kaiserord`+`firwin(kaiser)` exactly as the block, rotate by `exp(−j2π·600e3·n/fs)` in f64, `upfirdn(h, y, 1, 10)` restricted as in M06 | SNR ≥ 100 |
| `blocks/fm_demod_c32` | 311 | FM tone from the Python analytic FM formula (M08 §5) at 240 kHz | numpy chain: `angle(x·conj(x[−1]))·gain`, deemphasis IIR via `scipy.signal.lfilter([alpha],[1, alpha−1])`, `upfirdn` | SNR ≥ 90 |
| `blocks/resampler_rational_c32_48k_to_44k1` | 312 | noise σ 0.2 low-passed by `firwin(63, 0.3)` | `upfirdn` with block taps | SNR ≥ 100 |
| `blocks/resampler_tuned_c32_ppm` | 313 | same input, rate 48 kHz → 48 kHz, ppm +125 | Python implementation of §7 tuned mode in f64 (same bank, same fixed-point positions) | SNR ≥ 100 |
| `blocks/resampler_tuned_cq15_ppm` | 314 | same, quantized | Python integer model of §7 | bit-exact |
| `blocks/spectrum_c32_1024_hann_avg4` | 315 | tone at bin 100.0 (amp 1 − 2⁻¹⁵) + noise σ 0.01 | numpy: window periodic Hann, `fft`, `\|X\|²`, mean over 4 windows, `10·log10(P / (N·cg)²)`, `fftshift` | class `max_abs_err`: ≤ 0.01 dB, `min_ref_db` −100, `ref_unit` `db` |
| `blocks/spectrum_cq15_1024_bfp` | 316 | same signal quantized | same numpy result | class `max_abs_err`: ≤ 0.2 dB, `min_ref_db` −60, `ref_unit` `db` |

The Python block cases use `tests/golden/sdrk_golden/design.py`, a port of §3 that **calls SciPy** (`firwin`, `kaiserord`) — it is the reference; the Rust design must equal it.

## Tests

1. Golden tests for all cases above (blocks driven with random chunk sizes; output equal for any chunking).
2. Descriptions: all rules of §5 for every entry of `all_descriptions()`; `docs/blocks.md` table up to date.
3. Construction: every block × every sample type constructs inside an `ArenaStorage` over a 16-byte aligned slice of exactly `storage_bytes` bytes and fails with a `Storage` error with one byte less (where the requirement is > 0).
4. `set_param`: every runtime parameter changes behaviour from the next call (e.g. `ddc.frequency_hz` change → output tone moves); unknown name → `UnknownParameter`; build parameter → `ReadOnlyParameter`; out-of-range → `OutOfRange`; `param` returns what was set.
5. `reset` makes the output after reset equal to a freshly constructed block.
6. Timing: `BlockTiming` matches the measured delay of an impulse through the block (±0.5 sample) for `fir`, `ddc`, `duc`, `resampler` (rational).
7. Ppm: with `ppm = +100`, 1 000 000 input samples at 48 kHz → output count = `round(1e6 · (1 + 1e-4))` ± 1.
8. `cargo xtask bare-metal` builds `sdrk-blocks`; a `#[test]` asserts no block struct is larger than 512 bytes (all big data lives in storage).
9. Suite cases (no_std, in the testkit, using `ArenaStorage` on the suite arena): `block_ddc_vs_chain`, `block_fm_demod_sinad` (E2E-RX from M08 rebuilt with blocks; same thresholds), `block_spectrum_tone_level` (full-scale tone reads 0 dBFS ± 0.05 dB in f32, ± 0.3 dB in Q15), `block_resampler_ppm_count`.

## Acceptance

R1.2 commands; golden check; `cargo test -p sdrk-blocks --all-features`.

## Deliverables checklist

- [x] Crate, layer registered in `xtask`
- [x] `design`, `storage` (`ArenaStorage`, testkit `VecStorage`), `describe`, `block`
- [x] 11 blocks with descriptions, both sample types
- [x] Kernels `Dot`, `DotReal` (full kernel pattern)
- [x] Golden cases (design, dot, blocks), suite cases
- [x] `docs/blocks.md`, `docs/timing.md` update, CHANGELOG, decisions log

## Status

Completely done: everything in the Specification (sections 1 to 9), all golden cases of the table (27 cases: `design` 10, `dot` 10, `blocks` 7), the tests 1 to 9 of the Tests section (golden, descriptions with the generated table of `docs/blocks.md`, construction in exact arenas, `set_param` and `param`, `reset`, impulse timing, ppm count, `bare-metal` build with the 512-byte size test, the four suite cases `block_ddc_vs_chain`, `block_fm_demod_sinad`, `block_spectrum_tone_level`, `block_resampler_ppm_count`), the R1.2 commands and the golden check (run locally on Linux x86_64). Judgement calls are in [`docs/decisions-log.md`](../decisions-log.md) (M09). Open: confirmation of the CI run on the other two hosts (see [`docs/open-items.md`](../open-items.md)).
