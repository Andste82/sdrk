# M27 — `Chain`: blocks in a row without the pipeline

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M26](M26-block-ports.md) · Next: [M28](M28-clock-drift.md)

## Goal

`Chain<A, B>`, a block made of two blocks in a row, with one intermediate buffer from caller storage, exact timing and parameter access by stage, so that a microcontroller program (or any Rust program) can build a receive or transmit chain in plain Rust, without JSON, without the pipeline and without a heap. A `chain!` macro writes longer chains. A target test runs a chain bare-metal.

## Why

The pipeline needs `std` and builds its graph at run time. On a small device the fixed chain of a receiver (`convert → ddc → fm_demod`) is known at compile time, and the plan has always promised "pure Rust without JSON overhead" for it (backlog item "Compile-time MCU pipelines"). With the single `Block` trait of M26 a chain is itself a `Block`, so chains nest, and a composite block (M31) is a generalisation of the same idea. `Chain` is also the simplest place to settle the rules for the intermediate buffer, the timing of a sequence and the parameter names of inner blocks, which M31 reuses.

## Order

M27 depends on M26. It can run in parallel with M28, M29, M33 and M34.

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

**In:** `Chain<A, B, S>` in `sdrk-blocks` (module `chain.rs`); the macro `chain!`; the description `CHAIN_DESCRIPTION`; storage sizing; tests; a bare-metal target test in `targets/` that runs `convert → ddc → fm_demod` as a chain on the Cortex-M4 emulator; `docs/blocks.md`, `docs/embedded.md`, `CHANGELOG.md`.

**Out:** fan-out, two-input blocks inside a chain and automatic delay alignment (`Fork` of M31 and the pipeline of M34), a pipeline kind for chains (a chain is a Rust construct only), changing any existing block.

## Specification

### 1. Type and construction

```rust
/// Two blocks in a row: the output of `A` feeds `B`. A chain is a `Block` itself.
pub struct Chain<A, B, S: Storage>
where A: Block, B: Block<In = A::Out>, A::Out: StreamElem { /* a, b, buffer, fill, start */ }

impl<A, B, S: Storage> Chain<A, B, S> where /* as above */ {
    /// `chunk`: capacity of the intermediate buffer in samples of `A::Out` (at least
    /// `b.min_input()` and at least 1); the buffer comes from `storage`.
    pub fn new(a: A, b: B, chunk: usize, storage: &mut S) -> Result<Self>;
    /// Bytes `new` takes from the storage for this chunk (for sizing an arena).
    pub fn storage_bytes(chunk: usize) -> usize;
    pub fn first(&self) -> &A;       pub fn first_mut(&mut self) -> &mut A;
    pub fn second(&self) -> &B;      pub fn second_mut(&mut self) -> &mut B;
}
impl<A, B, S> Block for Chain<A, B, S> { type In = A::In; type Out = B::Out; /* … */ }
```

`A::Out` must be a stream element (a chain cannot pass `Two` or `Frame` between its stages); `B::Out` may be any port type, so a chain may end in a frame block (`convert → spectrum`). `A::In` may be any port type.

### 2. `process`

The chain holds the samples that `A` produced and `B` has not yet consumed in its buffer (`buf[start..fill]`). One call:

1. Loop: if the buffer holds at least `b.min_input()` samples, or the input is exhausted and the buffer is not empty, call `B` with `buf[start..fill]` and the remaining output; advance `start` by `consumed`, the output by `produced`. Stop the loop when `B` produced nothing and consumed nothing, or the output is full.
2. Move the rest of the buffer to its front when `start > 0` (`copy_within`).
3. If there is room in the buffer and input left, call `A` with the remaining input and `buf[fill..]`; advance the input and `fill`. If `A` consumed and produced nothing, stop.
4. Repeat 1–3 until the output is full or neither stage makes progress.

`Progress.consumed` is what `A` consumed, `Progress.produced` what `B` produced in this call. The result does not depend on how the caller cuts its input and output (R6.2), and equals running `A` over the whole input and then `B` over the whole result (test §Tests 1). The loop calls each stage at most `2 + input.len() / chunk + output.len()` times per call, never per sample.

### 3. Timing, reset, `min_input`, `map_index`

- `timing().ratio = ratio_a · ratio_b` (reduced). `kind` is `Variable` if either is `Variable`, else `Constant`.
- `timing().delay = delay_a + delay_b / ratio_a` in input samples of `A` (exact `Delay` arithmetic; `ratio_a` as a fraction). Add `Delay::add` and `Delay::div_ratio` to `sdrk-core` if they do not exist (log it).
- `min_input()`: the smallest `n` with `map_index(n) ≥ 1`, computed as `ceil(b.min_input() / ratio_a)` but at least `a.min_input()`.
- `map_index(n) = b.map_index(a.map_index(n))`.
- `reset()` resets both stages and empties the buffer.
- `frame_len()` is `b.frame_len()`.

### 4. Parameters and description

- `description()` returns `&CHAIN_DESCRIPTION`: kind `chain`, summary "Two blocks in a row (Rust only).", inputs and outputs copied from the stages is not possible statically, so they are empty; the kind is not registered in the pipeline and not in `all_descriptions()`; the name check of M24.1 lists `chain` as an allowed exception (log it).
- `set_param(name, value)` and `param(name)` take names of the form `"<stage>.<param>"` with `<stage>` `0` for `A` and `1` for `B` (`"1.frequency_hz"`); nested chains add more prefixes (`"1.0.cutoff_hz"`). A name without a prefix is `UnknownParameter`. The stage's own error is returned unchanged.

### 5. `chain!`

`chain!(storage, chunk; a, b, c, d)` builds `Chain::new(a, Chain::new(b, Chain::new(c, d, chunk, storage)?, chunk, storage)?, chunk, storage)?` (right-nested, so stage indices read `0`, `1.0`, `1.1.0`, `1.1.1`). It is a `macro_rules!` macro exported from `sdrk-blocks`; at least two and at most eight blocks. The doc example builds `convert → ddc → fm_demod` in an `ArenaStorage`.

### 6. Bare-metal target test

`targets/` gets a test `chain_fm_receiver` (built for every target of M15 that runs the quick suite): an `AlignedBytes` arena, `chain!(arena, 256; ConvertBlock<CQ15>, DdcBlock<CQ15>, FmDemodBlock<CQ15>)` with the parameters of the `fm_receiver` flow, 0.05 s of the M08 FM test signal in `cu8`; the output must equal the pipeline run of the same flow on the host bit for bit (the expected output is computed on the host by the testkit and compared through the M15 suite protocol as a digest). The test must fit the quick timeout of `qemu-cortex-m4`.

## Tests

1. **Equivalence:** for the pairs `fir → resample`, `ddc → fm_demod`, `convert → spectrum` (f32 and Q15 variants where they exist), the chain output equals the two blocks run one after the other on whole buffers, bit for bit, for chunk sizes 1, `b.min_input()`, 7, 256 and 4096 and 20 random input/output cut plans (testkit `ChunkPlan`).
2. **Timing:** `timing()`, `min_input()` and `map_index()` of the chains of test 1 equal the values computed by hand from the stages (table in the test); a chain of three resamplers gives the product ratio.
3. **Parameters:** `set_param("1.frequency_hz", …)` reaches the `ddc` of `fm_demod`-chain stage 1; an unprefixed name and a wrong stage are `UnknownParameter`; nested names work three levels deep.
4. **Storage:** `storage_bytes(chunk)` equals what `new` takes from an `ArenaStorage`; too small an arena is `Error::Storage`.
5. **No allocation and no panic:** `process` never allocates (counting allocator of M10) and returns errors for empty or mismatched calls.
6. The target test of §6 passes in the emulator.

## Acceptance

R1.2 commands; `cargo test -p sdrk-blocks --all-features`; `cargo xtask emu --target qemu-cortex-m4` (quick profile within its timeout).

## Deliverables checklist

- [ ] `Chain`, `storage_bytes`, stage accessors, `CHAIN_DESCRIPTION`; `Delay` helpers if needed
- [ ] `chain!` with doc example
- [ ] Tests 1–5; the target test `chain_fm_receiver`
- [ ] `docs/blocks.md` (section "Chains"), `docs/embedded.md` (how to size the arena), CHANGELOG, decisions log
