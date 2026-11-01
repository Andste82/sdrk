# M31 — Composite tools: fork, join and delay alignment

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M27](M27-chain.md), [M29](M29-primitive-blocks.md), [M30](M30-two-port-blocks.md) · Next: [M32](M32-composite-plans.md)

## Goal

Generic Rust building blocks that turn a graph of blocks with fan-out and two-input joins into one `Block`: `Chain` (M27) extended to pass a pair of streams between its stages, `Fork` (one input, two branches, a `Two` output) with automatic delay alignment, and the rules for parameters, timing and storage of such a composition. With them a composite block is written as one expression of existing blocks, without its own buffers or loops.

## Why

The plan of the architecture discussion (2026-10-28) is that the truth lies in graphs of primitive blocks and that a specific block (a *composite block*) is a Rust implementation that computes exactly what its graph computes. The stereo decoder, for example, sends the multiplex signal straight on and through a 19 kHz band-pass and joins both in the pilot loop; `Chain` alone cannot express that. A procedural macro was considered and rejected for this milestone: generic combinators need no new crate, are ordinary typed Rust that the compiler checks, and are easier to test; a macro can later generate the same combinators.

## Order

M31 depends on M27, M29 and M30. It can run in parallel with M28, M33 and M34.

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

**In:** `Chain` with a `Two` intermediate (`chain.rs`); `Fork<A, B, S>` (`fork.rs`); delay alignment; the parameter naming of compositions; storage sizing; tests; `docs/blocks.md` (section "Composite blocks"), `docs/architecture.md`, `CHANGELOG.md`.

**Out:** the plans of the existing blocks (M32), a procedural macro (backlog), a pipeline representation of compositions (`expand_composites`, backlog), forks with more than two branches (nest two forks), joins of streams with different rates.

## Specification

### 1. `Chain` with a pair between its stages

M27 restricted the intermediate of `Chain<A, B>` to a stream element. It now also accepts `A::Out = Two<X, Y>` (and therefore `B::In = Two<X, Y>`): the chain holds two buffers of `chunk` elements, written and consumed in lockstep (M26 §1). `storage_bytes` counts both buffers. Everything else of M27 §2–§4 holds unchanged.

### 2. `Fork`

```rust
/// One input, two branches. Out = Two<A::Out, B::Out>.
pub struct Fork<A, B, S: Storage> where A: Block, B: Block<In = A::In>, A::In: StreamElem, A::Out: StreamElem, B::Out: StreamElem { /* … */ }
impl<A, B, S: Storage> Fork<A, B, S> where /* as above */ {
    /// `chunk`: capacity of each branch's output buffer (and of the alignment delay line).
    /// Fails with `InvalidParameter` if the two branches have different ratios, or a ratio of kind
    /// `Variable`, or a delay difference that is not a whole number of output samples (§3).
    pub fn new(a: A, b: B, chunk: usize, storage: &mut S) -> Result<Self>;
    pub fn storage_bytes(a: &A, b: &B, chunk: usize) -> Result<usize>;
    pub fn first(&self) -> &A; pub fn first_mut(&mut self) -> &mut A;
    pub fn second(&self) -> &B; pub fn second_mut(&mut self) -> &mut B;
}
```

`process(input, (out_a, out_b))`: the fork copies input into its own input buffer (`chunk` samples) and remembers how far each branch has read from it (`read_a`, `read_b`). Each branch is called on its unread part with its own output buffer; samples are dropped from the input buffer once both branches have read them. The fork reports as consumed what it copied into its buffer. Pairs are moved to `(out_a, out_b)` while both branch buffers hold samples. The loop stops when the output is full or no branch made progress. The result does not depend on the cut of input and output (R6.2).

Timing: `ratio` of either branch; `delay` the larger of the two delays (after alignment both are equal); `min_input` the larger; `map_index` that of the branch with the larger delay; `reset` resets both branches, the buffers and the delay line.

### 3. Delay alignment

With the common ratio `r` (output per input), the delays in output samples are `da = delay_a · r` and `db = delay_b · r` (exact `Delay` arithmetic). If `da == db` nothing is inserted. If `da − db` is a non-negative integer `k`, the fork delays the output of branch `b` by `k` samples with an internal ring buffer of `k` elements (initial content zero), and the other way round for `db > da`. A non-integer difference is `InvalidParameter("fork branches are misaligned by a fraction of a sample")`; record in `docs/blocks.md` that such branches need an explicit `delay` or a different design. IIR stages report a delay of 0 (M09), so they never cause an alignment.

### 4. Parameters

Compositions name inner parameters by path, as `Chain` does (M27 §4): `Fork` uses the prefixes `0.` for `a` and `1.` for `b`. A composite block that wants public, flat parameter names (like `fm_stereo_demod`'s `audio_cutoff_hz`) keeps its own `set_param` that maps the flat name to a path; a helper `ParamMap` (a `&'static [(&'static str, &'static str)]` of flat name → path) does the lookup, and `description()` stays the composite's own.

### 5. Example in the documentation

`docs/blocks.md` shows the stereo decoder of M24 as a composition (the expression only; M32 makes it a test):

```rust
let decoder = chain!(st, 256;
    QuadDemodBlock::new(&qd, fs, st)?,
    Fork::new(DelayRealBlock::new(&zero, fs, st)?, BiquadRealBlock::new(&pilot_bp, fs, st)?, 256, st)?,
    StereoPllBlock::new(&pll, fs, st)?,
    DecimateBlock::new(&audio_lp, fs, st)?,
    StereoMatrixBlock::new(fs, st)?,
    BiquadBlock::new(&deemph, fs_audio, st)?);
```

## Tests

1. **Fork equals separate runs:** for branch pairs (`delay(0)`, `biquad_real`), (`decimate` ×4, `decimate` ×4 with a different filter), (`rotate`, `rotate`), the fork output equals running each branch on the whole input, bit for bit, for 20 random cut plans (seed 469).
2. **Alignment:** branches `fir` with 31 taps (delay 15) and `delay(0)`: the fork inserts 15 samples on the short branch, and the outputs are aligned (a test impulse appears at the same index in both); with `decimate` ×2 and 32 taps (delay 15.5 input samples, 7.75 output samples) against `delay(0)` → `InvalidParameter`.
3. **Rate mismatch:** branches with ratios 1/2 and 1/4 → `InvalidParameter`.
4. **Chain with a pair:** `fork → make_complex` and `fork → stereo_pll` chains equal the step-by-step computation; storage sizes are exact.
5. **Parameters:** paths through chain and fork (`"1.0.frequency_hz"`) and a `ParamMap` of flat names reach the right inner block.
6. No allocation in `process`; `cargo xtask bare-metal` builds the new modules.

## Acceptance

R1.2 commands; `cargo test -p sdrk-blocks --all-features`.

## Deliverables checklist

- [ ] `Chain` with `Two` intermediates; `Fork` with alignment; `ParamMap`
- [ ] Tests 1–6
- [ ] `docs/blocks.md` (section "Composite blocks" with the rules of §2–§4 and the example), `docs/architecture.md`, CHANGELOG, decisions log
