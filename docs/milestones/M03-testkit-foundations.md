# M03 — `sdrk-testkit` foundations

> Plan: [PLAN.md › Testing and performance](../PLAN.md#testing-and-performance) · Depends on: [M02](M02-core-foundations.md) · Next: [M04](M04-golden-vectors.md)

## Status

**Fully completed:** all modules (`rng`, `gen`, `metrics`, `invariance`, `arena`, `encode`, `virtual_rx`, `suite`), `StdReporter` and the host suite test, the five registered suite cases, tests 1-8, and the "How kernels are tested" section in `docs/kernels.md`. All R1.2 commands, `cargo test -p sdrk-testkit --features std` and the `sdrk-testkit` bare-metal build (no `std`) pass.

**Partial / open:** nothing. Judgement calls are recorded in `docs/decisions-log.md` (M03 entries); `docs/open-items.md` has no entry for this milestone.

## Goal

Build the heap-free test toolkit that every later milestone uses on the PC **and** inside emulators: a deterministic PRNG, signal generators, comparison metrics, a block-size-invariance harness, a bump arena for test buffers, a format encoder (the independent oracle for `convert`), a minimal virtual RX device, and the suite runner that executes registered test cases on any target.

## Why

The plan's chain of trust ends in "the same suite runs everywhere". On microcontrollers there is no `cargo test` and no heap, so the suite must be a plain function over a caller-provided memory arena.

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

**In:** `rng`, `gen`, `metrics`, `invariance`, `arena`, `encode`, `virtual_rx`, `suite` modules; std-only `StdReporter`; host integration test that runs the suite.

**Out:** golden-file loader (M04), FM generator and SINAD/tone meters (M08), loopback channel and TX virtual device (M12), conformance suite (M12).

## Specification

### 1. Crate setup

`sdrk-testkit` is `#![no_std]`; feature `std` enables `extern crate std` items. Dependencies: `sdrk` (facade), `sdrk-core` with `features = ["reference"]`, `sdrk-device`, `num-complex`, `libm`, `bytemuck`, `heapless`. Dev-dependency: `proptest` (host only).

### 2. `rng.rs` — PCG32 (PCG-XSH-RR 64/32), exact algorithm

```rust
pub struct Pcg32 { state: u64, inc: u64 }
impl Pcg32 {
    pub fn new(seed: u64, stream: u64) -> Self;  // state = 0; inc = (stream << 1) | 1; step(); state = state.wrapping_add(seed); step();
    fn step(&mut self);                           // state = state * 6364136223846793005 + inc (wrapping)
    pub fn next_u32(&mut self) -> u32;            // old = state; step(); xorshifted = (((old >> 18) ^ old) >> 27) as u32; rot = (old >> 59) as u32; xorshifted.rotate_right(rot)
    pub fn next_u64(&mut self) -> u64;            // (hi << 32) | lo, hi drawn first
    pub fn next_f64(&mut self) -> f64;            // (next_u64() >> 11) as f64 * 2^-53, in [0, 1)
    pub fn below(&mut self, n: u32) -> u32;       // Lemire's unbiased method or rejection; document which
    pub fn normal(&mut self) -> f64;              // Box–Muller: u1 = 1 - next_f64() (in (0,1]), u2 = next_f64(); sqrt(-2 ln u1) * cos(2π u2); uses libm
}
```

### 3. `gen.rs` — generators (write into caller slices, depend only on the absolute sample index)

All generators compute in f64 and write `Complex64` (complex) or `f64` (real) into a caller slice; helpers `write_as::<T: ComplexSample>(src: &[Complex64], dst: &mut [T])` convert with `from_c64_round`. Every generator takes `start: u64` (absolute index of the first sample) so that generating in chunks gives exactly the same samples as generating at once.

| Function | Definition (n = start + i) |
| --- | --- |
| `tone(out, freq: f64, amp: f64, phase: f64, start)` | `amp · e^{j(2π·frac(freq·n) + phase)}`; `freq` in cycles/sample (may be negative); `frac(x) = x − floor(x)` |
| `real_tone(out, freq, amp, phase, start)` | `amp · cos(2π·frac(freq·n) + phase)` |
| `multitone(out, tones: &[(f64,f64,f64)], start)` | sum of `tone`s |
| `noise(out, rng, power)` | complex Gaussian, `E\|x\|² = power`: `re, im = sqrt(power/2) · rng.normal()` (re drawn first) |
| `real_noise(out, rng, power)` | `sqrt(power) · normal()` |
| `chirp(out, f0, f1, len, amp, start)` | linear chirp, instantaneous frequency `f0 + (f1−f0)·n/len`, phase `2π(f0 n + (f1−f0) n²/(2 len))` |
| `impulse(out, at: u64, amp, start)` | `amp` at index `at`, else 0 |
| `step(out, at, amp, start)` | 0 before `at`, `amp` from `at` |
| `extreme::silence`, `extreme::full_scale_alternating` (±(1 − 2⁻¹⁵) per sample, re and im), `extreme::f32_specials` (fills with 0.0, −0.0, NaN, +∞, −∞, f32::MIN_POSITIVE/2 (denormal), f32::MAX, cycling) | for robustness tests |

### 4. `metrics.rs`

```rust
pub struct Comparison { pub snr_db: f64, pub max_abs_err: f64, pub rms_err: f64, pub first_mismatch: Option<usize>, pub compared: usize }
pub fn compare_complex(expected: &[Complex64], actual: impl Iterator<Item = Complex64>, skip: usize) -> Comparison;
pub fn compare_real(expected: &[f64], actual: impl Iterator<Item = f64>, skip: usize) -> Comparison;
pub fn bit_exact<T: Sample>(a: &[T], b: &[T]) -> Option<usize>; // index of the first raw-bytes difference
```
SNR exactly as in R8.5 (`f64::INFINITY` when the error energy is 0; `-f64::INFINITY` if signal energy is 0 but error is not). `first_mismatch` = first index where `|e − a| > 0`.

### 5. `invariance.rs` — block-size invariance harness

```rust
pub struct ChunkPlan { seed: u64, max: usize }  // yields chunk sizes uniformly in 1..=max, deterministic
pub fn check_block_invariance<I: Copy, O: Sample, S>(
    input: &[I],
    out_whole: &mut [O],
    out_chunked: &mut [O],
    plan: ChunkPlan,
    mut make_state: impl FnMut() -> S,
    mut step: impl FnMut(&mut S, &[I], &mut [O]) -> Result<Progress, sdrk_core::Error>,
) -> Result<InvarianceReport, Failure>;
```
Algorithm: run once over the whole input with a fresh state, collecting outputs (call `step` repeatedly until all input is consumed, handling partial consumption); run again with a fresh state feeding chunks from `plan`; both runs must produce the same number of outputs and bit-identical samples (`bit_exact`). Report the first differing index. Use it for every stateful kernel, block and later the pipeline.

### 6. `arena.rs` — bump allocator over a caller buffer (no heap)

```rust
pub struct Arena<'a> { buf: &'a mut [u8], used: usize }
impl<'a> Arena<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self;
    pub fn take<T: bytemuck::Pod>(&mut self, n: usize) -> Result<&'a mut [T], Failure>; // aligns to align_of::<T>(), zero-filled
    pub fn reset(&mut self);
    pub fn remaining(&self) -> usize;
}
```
Implementation note: hand out non-overlapping sub-slices via `split_at_mut` and `bytemuck::try_cast_slice_mut`; keep it `unsafe`-free (the testkit is under `forbid(unsafe_code)`). If lifetimes make that impossible without `unsafe`, return the slices from a single `take_many` call or restructure; do **not** add `unsafe`.

### 7. `encode.rs` — raw-format encoder (independent oracle for M05 `convert`)

`pub fn encode(format: &SampleFormat, samples: &[Complex64], out: &mut [u8]) -> Result<usize, Failure>` implements the *inverse* of the value mapping documented in M01 for every container except `Bits1`:
- `raw = round_half_up(x · full_scale + offset)` per component (offset 127.5 for U8), saturated to the representable range of `bits` (e.g. 12-bit signed: −2048…2047).
- `Msb` alignment: shift left by `container_bits − bits`.
- `swap_iq`, `Planar` layout, `Endian::Big` byte order, Packed10/Packed12 LSB-first packing exactly as in `docs/device.md`.
- `Real` layout uses only `re`.
Write it from the specification, **not** by calling any `sdrk` code; it must stay an independent implementation.

Also `pub fn decode_reference(format, bytes, out: &mut [Complex64]) -> Result<usize, Failure>` (the f64 forward mapping), again written independently.

### 8. `virtual_rx.rs` — minimal virtual receiver

```rust
pub trait SignalSource { fn fill(&mut self, start: u64, out: &mut [Complex64]); }
pub struct VirtualRx<'a, Src: SignalSource> { /* format, rate, source, scratch: &'a mut [Complex64], plan: ChunkPlan, gaps: &'a [(u64, u64)], next_index: u64, anchor: Option<TimeAnchor> */ }
impl RxStream for VirtualRx<'_, Src> { /* Error = sdrk_core::Error */ }
```
- `read` produces the next chunk (size from the plan, limited by `buf` and `scratch`), encodes it with `encode`, and returns `BlockMeta { start, flags, anchor }`.
- `gaps: &[(at_index, missing)]`: when the next index reaches `at_index`, skip `missing` samples (advance the index without emitting them) and set `GAP_BEFORE` on the following block.
- The first block carries `anchor` if configured.
- Sources provided: `ToneSource`, `NoiseSource`, `SumSource<A, B>`.

### 9. `suite.rs` — runner that works on every target

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)] pub enum Tier { Quick, Full }
pub struct Failure { pub case: &'static str, pub msg: &'static str, pub detail: heapless::String<160> }
pub struct TestCase { pub name: &'static str, pub tier: Tier, pub run: fn(&mut Arena<'_>) -> Result<(), Failure> }
pub trait Reporter { fn started(&mut self, name: &str); fn passed(&mut self, name: &str); fn failed(&mut self, f: &Failure); fn finished(&mut self, s: &Summary); }
pub struct Summary { pub passed: u32, pub failed: u32, pub skipped: u32 }
pub fn run_suite(cases: &[TestCase], tier: Tier, arena: &mut Arena<'_>, reporter: &mut impl Reporter) -> Summary;
pub fn all_cases() -> &'static [TestCase];  // returns a `static` immutable slice listing every case
```
- `Tier::Quick` runs quick cases; `Tier::Full` runs all.
- The arena is reset before each case.
- Output protocol (used by emulator runners in M15): reporters print lines `SDRK-SUITE START <n>`, `SDRK-CASE PASS <name>`, `SDRK-CASE FAIL <name> <msg> <detail>`, and finally `SDRK-SUITE END passed=<p> failed=<f>`.
- `#[cfg(feature = "std")] pub struct StdReporter` prints that protocol to stdout. (The testkit is allowed to print under `std`; add `#[allow(clippy::print_stdout)]` locally.)
- Required arena size: every case must work with **256 KiB**; quick cases with **64 KiB**. Document per case how much it takes.

M03 registers these cases in `all_cases()`: `rng_known_answer`, `gen_tone_frequency`, `encode_decode_roundtrip_all_formats`, `scale_fallback_vs_reference`, `scale_block_invariance` (Scale is stateless; the harness must still pass).

### 10. Host integration test

`crates/testing/sdrk-testkit/tests/suite.rs` (requires `std`): allocates a 256 KiB `Vec<u8>`, runs `run_suite(all_cases(), Tier::Full, …)` with `StdReporter`, asserts `failed == 0`.

## Test vectors

1. PCG32 known-answer values (from the PCG reference C implementation's `pcg32-demo`, seed 42, stream 54): first six `next_u32()` outputs are `0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e`. If your implementation differs, re-check the algorithm against the reference C code (`pcg_basic.c`) before changing these numbers, and record the source in the decisions log.
2. Encoder spot values: `CU8` amplitude 0.0 → raw 127.5 → rounds half up to 128; amplitude −1.0 → −0.5 → 0; amplitude +1.0 → 255.5 → saturates to 255. `CS16` +1.0 → 32768 → saturates to 32767. `CS12_LSB16` +0.5 → 1024. `SC16_Q11` −1.0 → −2048. `PACKED12` values (I=1, Q=−1) → bytes `01 F0 FF`. `PACKED10` values (1, −1, 2, −2) → bytes computed by hand in the test with the LSB-first rule (show the derivation in a comment).

## Tests

1. PCG known answer; `next_f64` in [0,1); `below(n)` never ≥ n over 10⁶ draws; normal mean/variance within 1 % over 10⁶ draws.
2. `tone` generated in one piece equals `tone` generated in 7 random chunks bit for bit; a DFT (computed in the test) of a 1024-sample tone at bin 37 has its maximum at bin 37 with ≥ 100 dB above every other bin.
3. `noise` power within 1 % of the request over 10⁵ samples.
4. `encode` → `decode_reference` round trip for every format constant in `sdrk_device::formats` (except Bits1) with random samples in (−1, 1): error ≤ ½ LSB of the format; spot values above.
5. `check_block_invariance` passes for an identity kernel and **fails** (reports the right index) for a deliberately broken kernel that resets its state at every call.
6. `Arena`: alignment of `take::<C32>` after `take::<u8>(3)`; exhaustion returns `Failure`; reset frees everything.
7. `VirtualRx`: chunk sizes follow the plan; gap handling sets `GAP_BEFORE` and skips exactly `missing` indices; first block carries the anchor.
8. Host suite integration test passes.

## Acceptance

R1.2 commands; `cargo test -p sdrk-testkit --features std`; `cargo xtask bare-metal` builds `sdrk-testkit` without `std`.

## Deliverables checklist

- [x] `rng`, `gen`, `metrics`, `invariance`, `arena`, `encode`, `virtual_rx`, `suite`
- [x] `StdReporter` and host suite test
- [x] Five registered suite cases
- [x] Tests 1–8
- [x] `docs/kernels.md` section "How kernels are tested" (suite, invariance, extremes)
