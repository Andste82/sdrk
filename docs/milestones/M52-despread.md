# M52 — Kernel `despread`: integrate-and-dump correlation

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M43](M43-equalizer.md), [M51](M51-spread.md) · Next: [M53](M53-code-search.md)

> **Status: planned** (written 2026-11-08). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

## Goal

A kernel `despread` that correlates a chip stream against a code and dumps one symbol per `sf` chips: `symbol[m] = sum_{j} input[m·sf + j] · conj(code[c])`. It is the matched receiver for `spread` (M51): with the same code, chip alignment and `sf`, despreading the output of `spread` recovers the symbols with the full processing gain. The summation order is the fixed order of `dot_complex`, so the kernel is bit-exact and so that it is bit-identical to the prompt branch of the tracking loop `code_track` (M54) when that loop's bandwidth is zero.

## Why

Despreading collapses `sf` chips back to one symbol and concentrates the signal power while the noise averages down, the receive half of the processing gain. It is the inner correlation of every DSSS receiver (and of each early/prompt/late finger of the tracking loop). It needs its own kernel rather than a generic sum so the order is fixed (reproducible on every backend) and so the chip phase and code indexing match `spread` exactly.

## Order

M52 depends on M51 (the chip indexing and code model) and on M43 (the `dot_complex` summation order it reuses). It can run in parallel with every milestone after M51.

## Ground rules (identical in every milestone — read them completely before you start)

These rules apply to every line of code in this repository. They are repeated in every milestone file so that each milestone is self-contained. If a rule and a milestone-specific instruction ever conflict, stop and ask; do not guess.

### R1 — Work order and scope

1. A milestone starts when every milestone named in its *Depends on* line is delivered (merged on `main`); up to M25.1 the order is strictly numeric. Milestones that do not depend on each other may be implemented in parallel, each on its own branch; the second one to merge rebases onto the first. A milestone may only use what its dependencies (and their dependencies) delivered. Never implement items of a later milestone "while you are at it".
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
| `crates/proto` | `sdrk-proto` | `sdrk-device` | yes |
| `crates/blocks` | `sdrk-blocks` | facade, core, proto | yes |
| `crates/pipeline` | `sdrk-pipeline` | blocks, proto, facade, core | no (std/alloc allowed) |
| `crates/io` | `sdrk-io` | core (types only) | no |
| `crates/testing` | `sdrk-testkit` | any library crate above | `no_std` by default; feature `std` adds file I/O and proptest helpers |
| `bindings/*`, `examples/*`, `targets/*`, `benches/*`, `xtask` | — | anything | no |

Additional rules:
- Within `crates/core` the rank is by crate name: `sdrk-device` (rank 0) < `sdrk-core` (rank 1); `sdrk-core → sdrk-device` is allowed, the reverse is not. All other layers are one rank per folder.
- Backends never depend on each other. Only the facade `sdrk` knows the backends.
- `sdrk-proto` (from M37 on) holds protocol logic without signal processing (bit and byte framing, checksums, packet types); it depends on `sdrk-device` only, and nothing below `crates/blocks` depends on it.
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
6. Each case (JSON + all its `.bin` arrays) stays at or below 512 KiB; all vectors together at or below 32 MiB. Prefer N ≤ 4096 samples; use longer inputs only when the kernel needs them (long filters, decimation).
7. Tests that use `sdrk-testkit` live in integration tests (`<crate>/tests/*.rs`) or in `crates/testing/sdrk-testkit/tests/`, never in `#[cfg(test)]` unit modules of a crate the testkit itself depends on (a dev-dependency cycle would compile two copies of that crate and produce confusing trait errors).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.




## Scope

**In:** kernel `despread` (`C32`, `CQ15`) in `sdrk_core::kernels::spread` (the module of M51) with `DespreadState`, Fallback, Reference, facade delegation, contract, `docs/kernels.md`; golden cases; unit, invariance, extreme tests; suite cases; `cargo xtask count`; CHANGELOG, decisions log; the block `DespreadBlock<T>` and the pipeline kind `despread` (§6) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the code-phase search (`code_search`, M53) and the tracking loop (`code_track`, M54); the composite `dsss_demod` (later milestone; name reserved); a SIMD path (allowed later; the `±1` case is add/subtract, which maps to a `dot`-style vector sum in the fixed order).

## Specification

General rules of M19 apply.

### 1. Trait, state, indexing

```rust
pub struct DespreadState { /* chip_index: u64, sf: u32, code_len: u32, long: bool */ }
impl DespreadState {
    pub fn new(sf: u32, code_len: u32, long: bool) -> Result<Self>; // same rules as SpreadState
    pub fn chip_index(&self) -> u64;
    pub fn seek(&mut self, chip: u64);
    pub fn reset(&mut self);
}
pub trait Despread<T> {
    /// Consumes whole groups of `sf` chips; output length ≤ input.len()/sf.
    fn despread(&self, state: &mut DespreadState, input: &[T], code: &[T], output: &mut [T]) -> Result<Progress>;
}
```

- **Output.** For each complete group of `sf` input chips, `output[m] = sum_{j=0}^{sf-1} input[m·sf + j] · conj(code[c])` with the code position `c` indexed exactly as `spread` (short: `c = j`; long: `c = (chip_index0 + m·sf + j) mod code_len`). The conjugation makes it a correlation (it undoes the phasor of a complex code; for a real `±1` code the conjugate is the code itself).
- **Summation order (bit-exact).** The sum over the `sf` products uses the `dot_complex` order of M43: eight partial sums per component, product `i` added to partial sum `i mod 8`, then the fixed tree `s[k] = acc[k] + acc[k+4]`, `t[k] = s[k] + s[k+2]`, result `t[0] + t[1]`. The product is `x · conj(code) = (xr·cr + xi·ci, xi·cr − xr·ci)` in f32 (each product rounded before the sum/difference). This is `dot_complex(input_window, conj(code_window))`, so `despread` reuses `dot_complex` directly for `C32`. For `CQ15` the products accumulate in `i64` (Q30) and are rounded and saturated once per component at the dump (the `dot` `CQ15` rule).
- **Consumption.** A call consumes whole groups of `sf` chips and keeps the remainder (fewer than `sf` chips) in a small history of up to `sf − 1` chips, so a symbol split across two calls still dumps once, in the call that completes it (chunk invariance). `Progress` counts chips consumed and symbols produced; `min_input` `sf`. `chip_index += (symbols produced)·sf`.
- Timing: ratio `1/sf` (constant), delay 0: declared at the first chip of the integration window, as `spread` declares its delay at the first chip (M51 §1). The centre convention of timing.md would give `−(sf − 1)/2` here (the window lies after the output position `m · sf`), which `Delay` (unsigned) cannot express; with 0 on both sides `spread` → `despread` reports exactly the delay its samples show. Logged. Storage: the chip history (`sf − 1` complex) and the small state.

### 2. Relation to `code_track` and `spread`

- **Zero-bandwidth tracking.** `code_track` (M54) with loop bandwidth 0 and on-time alignment produces exactly this prompt correlation; its prompt output must equal `despread` bit for bit under that setting. This milestone states the invariant; M54 tests it.
- **Round-trip with `spread`.** Over a noiseless channel, `despread(spread(symbols, code), code)` returns `sf · symbols` for a `±1` code (`sum conj(c)·c = sf` per symbol) within the float accumulation error; the test checks the recovered symbols after dividing by `sf`.

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `despread` | `kernels::spread::Despread` | `C32` | bit-exact | `dot_complex` of the chip window with `conj(code)`, the eight-partial-sum order of M43; fallback equals reference |
| `despread` | `kernels::spread::Despread` | `CQ15` | bit-exact | `Acc64` per component over the products with `conj(code)`, one round/saturate per symbol (the `dot` `CQ15` rule) |

`Reference` follows the `dot_complex` order step by step (the bit-exact rule of the kernels that reuse `dot_complex`). `backend_for("despread", ..)` reports `fallback` (the SIMD `dot_complex` path may be wired later; it already gives the fallback bits).

### 4. Golden vectors

New cases in `tests/golden/sdrk_golden/cases/despread.py` (the shared code helper of M51).

| Case | Seed | Type | `sf` | code | Chips | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `despread/short_bpsk_c32_sf31` | 720 | C32 | 31 | short ±1 m-seq 31 | the output of `spread/short_bpsk_c32_sf31` plus noise |
| `despread/long_bpsk_c32_sf1023` | 721 | C32 | 1023 | long ±1 m-seq 1023 | peak recovery at −10 dB chip SNR |
| `despread/short_qpsk_cq15_sf15` | 722 | CQ15 | 15 | short complex code | QPSK |

- Expected values from the `dot_complex` f32 model (and the `Acc64` model for `CQ15`), the per-symbol products stored. Tolerance: bit-exact (`==` on raw bits) against the model; an f64 regression guard like `dot_complex` (the exactly rounded f64 sum) with `snr_db_guard`.

### 5. Cost

`cargo xtask count` entries `despread_c32`, `despread_cq15` (per input chip; one complex MAC into the partial sums). Record in `docs/performance.md`.

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`). A library reference enters only where it does the same work (D32); otherwise the rows are information. The D321 gate applies once a SIMD path exists.

### 6. Block and kind `despread`

`DespreadBlock<T>` (`C32`/`CQ15` → same, one output symbol per `sf` input chips) around the kernel `despread`, with the code of `spread` (§6 of M51: the same parameters, the same PRBS mapping, `DespreadBlock::with_code` for a caller code). Chip 0 of the code meets the first input sample after construction or `reset`; the block does not search (that is `code_search`, M53, and `code_track`, M54).

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `despread` | `DespreadBlock<T>` | C32/CQ15 → same | `sf`, `code`, `code_seed`, `long` as `spread` | `despread`; a call consumes whole groups of `sf` chips limited by the output room, the rest stays in the kernel history |

- `DespreadConfig`. Storage: the code and the chip history (`sf − 1`); exact. Timing ratio `1/sf`. **Delay** 0 (§1: the first chip of the window, as `spread`).
- Pipeline kind `despread` (`c32` or `cq15`), cost 4 per input chip (one complex MAC), a row in `docs/pipeline.md` and `docs/timing.md`.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 720 … 729.

1. Golden cases of §4 through facade, `Fallback`, `Reference` (`sdrk-testkit/tests/golden_m52.rs`).
2. Unit tests (`sdrk-core/tests/kernels_m52.rs`): `new` rules; the `dot_complex` equality (`despread` of one window equals `dot_complex(window, conj(code))`); the round-trip with `spread` (recovers `sf·symbols` for a `±1` code, exact ratio after dividing out `sf`); a symbol split across two calls dumps once with the same value; `seek`, `reset`; `CQ15` saturation on a full-scale window.
3. Block-size invariance: 20 chunk plans at the chip level (seed 723), both types.
4. Extremes: fewer than `sf` chips (no output); `sf = 2`, `sf = 65536`; full-scale `CQ15`; NaN/∞ `C32` in one chip (that symbol NaN, the next recovers).
5. Suite cases `despread_c32`, `despread_cq15`, `extremes_m52`.
6. **Block and kind** (`sdrk-blocks/tests/despread_blocks.rs`, `sdrk-pipeline/tests/despread_kinds.rs`; seeds 724, 725): the block equals the kernel bit for bit on whole buffers and in 20 chunk plans, both types; `spread` → `despread` blocks return `sf` times the symbols for every PRBS choice; the description rules of M09, exact storage, struct at most 512 bytes, validation errors as `spread`; a pipeline flow `spread` → `despread` equals the blocks and its reported delay matches the samples; `cargo xtask schema --check` and the WASM kind list include `despread`.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the entries of §5 and `despread_block`); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [ ] Kernel `despread` (`C32`, `CQ15`): `DespreadState`, trait, Fallback (reusing `dot_complex`), Reference, facade delegation, `backend_for`
- [ ] Contract rows and section in `docs/kernels.md` (the `dot_complex` order; the `spread` round-trip and the `code_track` zero-bandwidth invariant)
- [ ] Golden cases (seeds 720–722) and the Python model
- [ ] Tests 1–6 (seeds 720–725); suite cases
- [ ] Count and cycle entries (`cargo xtask count`, `cargo measure`), `docs/performance.md`
- [ ] Block `DespreadBlock<T>` and kind `despread` (§6): `docs/blocks.md` (entry; reserved-name row "delivered"), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `despread_block`; the delay decision
- [ ] CHANGELOG, decisions log (reuse of `dot_complex`, the history length)

## Status

Planned. Nothing is implemented.
