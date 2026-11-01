# M26 — Block ports: one `Block` trait for streams, pairs and frames

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M25.1](M25.1-release-0.1.md) · Next: [M27](M27-chain.md)

## Goal

One block trait for every kind of port: the input and the output of `Block` become *port types* (`Ports`) instead of sample types, with three port types in this milestone: a stream of one sample type (`C32`, `CQ15`, `f32`, `Q15`, `u8`), a pair of streams (`Two<A, B>`) and frames (`Frame`). `FrameBlock` is merged into `Block` (`SpectrumBlock` becomes `Block<Out = Frame>`), and the pipeline's node wrappers follow. No block changes its output.

## Why

Every later milestone needs blocks with two inputs or two outputs: the building blocks of composite blocks (`stereo_pll` takes the multiplex signal and the pilot, `make_complex`, `add` and `mul` take two streams, M30–M33), multi-input nodes in the pipeline (M35) and packet ports (M38). Today `Block` has exactly one input slice and one output slice, and frames need a second trait (`FrameBlock`) with its own node wrapper. A generic port type keeps one trait, one way to chain blocks (M27) and one node wrapper, and costs nothing at run time: the port types are resolved at compile time and a stream block keeps the signature it has today. This is the only breaking change of the block interface planned after 0.1.0, which is why it comes first.

## Order

M26 depends on M25.1 (release 0.1.0 is made before the interface changes). M27, M29, M30, M33 and M34 can start after it; M28 does not need it and can run in parallel with it.

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

**In:** the trait `Ports` and its implementations for stream elements, `Two<A, B>` and `Frame` in `sdrk-blocks`; the changed `Block` trait; the removal of `FrameBlock`; `SpectrumBlock` on the new trait; the node wrappers in `sdrk-pipeline` (`StreamNode`, `FrameNode`) on the new trait; the migration of every block, test, bench and binding that uses `Block` or `FrameBlock`; `docs/blocks.md`, `docs/architecture.md`, `CHANGELOG.md` with a migration note, `docs/decisions-log.md`.

**Out:** any block with two inputs or outputs (they come in M30 and M33), chaining (M27), multi-input nodes in the pipeline (M35), packet ports (M38), changes of block behaviour, parameters or timing.

## Specification

### 1. Port types (`sdrk-blocks`, new module `ports.rs`)

```rust
/// The element of a stream port. Sealed: implemented for C32, CQ15, f32, Q15 and u8 only.
pub trait StreamElem: Copy + Send + 'static + sealed::Sealed {
    /// The port type of the description (`PortType::C32` …; `PortType::Raw` for u8).
    const PORT_TYPE: PortType;
}

/// What a block reads and writes in one `process` call.
pub trait Ports: 'static {
    /// Read-only view of the input (or output) data of one call.
    type Ref<'a>;
    /// Writable view of the output data of one call.
    type Mut<'a>;
    /// Number of ports (1 for a stream or a frame, 2 for `Two`).
    const COUNT: usize;
}

impl<T: StreamElem> Ports for T {
    type Ref<'a> = &'a [T];
    type Mut<'a> = &'a mut [T];
    const COUNT: usize = 1;
}

/// Two streams that advance together (one sample of each per step).
pub struct Two<A: StreamElem, B: StreamElem>(core::marker::PhantomData<(A, B)>);
impl<A: StreamElem, B: StreamElem> Ports for Two<A, B> {
    type Ref<'a> = (&'a [A], &'a [B]);
    type Mut<'a> = (&'a mut [A], &'a mut [B]);
    const COUNT: usize = 2;
}

/// Frames of `f32` values (spectra, measurements). Output only.
pub struct Frame;
pub struct FrameOut<'a> {
    /// Exactly `Block::frame_len()` values; written when a frame completes.
    pub values: &'a mut [f32],
    /// Set to `Some` when the call completed a frame, left `None` otherwise.
    pub meta: &'a mut Option<FrameMeta>,
}
impl Ports for Frame {
    type Ref<'a> = &'a [f32];
    type Mut<'a> = FrameOut<'a>;
    const COUNT: usize = 1;
}
```

`Two` and `Frame` do not implement `StreamElem`, so the blanket implementation does not conflict (the trait is local and sealed). `Frame` as an *input* port is not used in this milestone; a block with `In = Frame` is rejected by the pipeline (§4).

**Lockstep rule for `Two`:** both input slices of a call have the same length, and the block consumes the same number of samples from each (`Progress.consumed` counts pairs). Both output slices have the same length and the block writes the same number of samples to each (`Progress.produced` counts pairs). Unequal input or output lengths are `LengthMismatch` and change nothing. Record the rule in `docs/blocks.md`.

### 2. The `Block` trait

```rust
pub trait Block: Send {
    type In: Ports;
    type Out: Ports;
    fn description(&self) -> &'static BlockDescription;
    fn process(&mut self, input: <Self::In as Ports>::Ref<'_>, output: <Self::Out as Ports>::Mut<'_>) -> Result<Progress>;
    fn timing(&self) -> BlockTiming;
    fn reset(&mut self);
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;
    fn param(&self, name: &str) -> Result<ParamValue>;
    fn min_input(&self) -> usize;
    fn map_index(&self, rel_in: u64) -> u64 { /* default unchanged */ }
    /// Number of `f32` values of one frame for `Out = Frame`; 0 for every other block.
    fn frame_len(&self) -> usize { 0 }
}
```

- The documentation of every method stays as it is; `process` gets the lockstep rule of §1 and the frame rule of §3.
- A stream block keeps its signature: `type In = C32;` and `fn process(&mut self, input: &[C32], output: &mut [C32])` are accepted because the projections normalize to the concrete types. Write a compile test first (`tests/ports.rs`: one stream block, one `Two` block and one `Frame` block defined in the test). If the compiler of the MSRV rejects the normalized form in impls, write the projections in the impls instead and log it.
- `Block::In` and `Block::Out` were `Copy + Send` sample types before; code that relied on that (generic helpers in tests and benches) uses `StreamElem` now.

### 3. Frames on `Block` (`SpectrumBlock`)

`FrameBlock` is removed. `SpectrumBlock<T, S>` implements `Block<In = T, Out = Frame>`:

- `process(input, FrameOut { values, meta })` does what `FrameBlock::push` did: it consumes input, stops right after completing a frame, writes the frame into `values`, sets `*meta = Some(..)` and returns `Progress { consumed, produced: 1 }`; without a completed frame `produced` is 0 and `*meta` is left unchanged (the caller sets it to `None` before the call). `values.len() != frame_len()` is `LengthMismatch`.
- `frame_len()` returns the frame length (as before).
- `timing()` returns the nominal ratio of frames per input sample, `1 / interval_samples` with `RatioKind::Variable` (the interval is a runtime parameter), and the delay `fft_size − 1` input samples. The pipeline does not use the timing of a frame block for rates (§4); the value is for applications.
- `min_input()` returns 1, `map_index` the identity (as the frame node did).

The outputs, the frame metadata and the golden and block tests of the spectrum stay bit-identical.

### 4. Pipeline nodes (`sdrk-pipeline`, `node.rs`)

- `NodeElem` is replaced by `StreamElem` (re-exported from `sdrk-blocks`); `NodeInput` and `NodeOutput` keep their variants.
- `StreamNode<B>` requires `B::In: StreamElem` and `B::Out: StreamElem` (unchanged behaviour).
- `FrameNode<B>` requires `B: Block<Out = Frame>` and `B::In: StreamElem`; it builds the `FrameOut` from `NodeOutput::Frame` and the `meta` argument and returns `produced` as before.
- Blocks with `Two` ports have no node wrapper yet (M35); no kind uses them in this milestone.
- `PortDesc` and `PortType` are unchanged; a block with `Two` input lists two entries in `inputs` (names chosen by the block, e.g. `mpx` and `pilot`).

### 5. Migration of callers

Every `impl Block for` in the workspace is checked (most need no change); every use of `FrameBlock` (`spectrum.rs`, the tests in `crates/blocks/sdrk-blocks/tests`, `crates/pipeline/sdrk-pipeline/tests/e2e.rs`, the pipeline kinds, benches, the WebAssembly bindings) moves to `Block<Out = Frame>`. The separate repository `sdrk-hardware` is not changed here; `CHANGELOG.md` gets a migration note with a before/after example for a stream block (no change) and for a frame block (`push` → `process` with `FrameOut`).

### 6. Documentation

`docs/blocks.md`: section "The block interface" rewritten for `Ports` (with the examples of §1 and the lockstep rule); `docs/architecture.md`: one paragraph on port types and what later milestones add (`Packets` in M38); `CHANGELOG.md` (Unreleased, *Changed*, with the migration note); `docs/decisions-log.md`.

## Tests

1. `tests/ports.rs` (compile and behaviour): a test-only `Two<f32, f32>` → `C32` block that interleaves its inputs; unequal input lengths give `LengthMismatch`; `Progress` counts pairs; a test-only frame block; the blanket implementation covers all five stream elements (`<T as Ports>::COUNT == 1`).
2. Every existing test of `sdrk-blocks`, `sdrk-pipeline`, `sdrk-testkit` and the bindings passes unchanged except for the changed calls of §5; the spectrum outputs and frame metadata are bit-identical (existing golden and pipeline tests).
3. `cargo xtask bare-metal` builds `sdrk-blocks` with the new module (no heap, no `std`).
4. A doc test on `Ports`, `Two`, `Frame` and `FrameOut` (R9).

## Acceptance

R1.2 commands; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline`; `cargo test -p sdrk-testkit --features std`; the WebAssembly tests of M14 (`cargo xtask wasm-test` or the command named in `docs/wasm.md`); `cargo xtask emu --target qemu-cortex-m4` (quick profile).

## Deliverables checklist

- [ ] `ports.rs` with `StreamElem`, `Ports`, `Two`, `Frame`, `FrameOut`, doc tests
- [ ] `Block` on port types, `frame_len`; `FrameBlock` removed; `SpectrumBlock` on `Block<Out = Frame>`
- [ ] Node wrappers on the new trait; every caller migrated; outputs bit-identical
- [ ] `docs/blocks.md`, `docs/architecture.md`, CHANGELOG with migration note, decisions log
