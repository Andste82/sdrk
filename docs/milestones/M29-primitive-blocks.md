# M29 — Primitive blocks around existing kernels

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M26](M26-block-ports.md) · Next: [M30](M30-two-port-blocks.md)

## Goal

Seven single-input blocks that each wrap one existing kernel or a plain delay, with descriptions, pipeline kinds and the regenerated schema: `quad_demod`, `deemphasis`, `rotate`, `decimate`, `interpolate`, `mag` and `delay`. Together with M30 they are the building blocks from which the composite blocks (M31, M32) and the new transmit and digital blocks are assembled.

## Why

The existing receiver blocks (`fm_demod`, `ddc`, `duc`, `am_demod`, `ssb_demod`) call several kernels inside one Rust function. To describe them as a plan of smaller blocks (M32), and to build new features from parts instead of writing another special block each time, every step of these chains must exist as a block on its own. All the kernels exist since M05–M20, so each block is a thin wrapper: a constructor that validates and designs, a `process` that calls exactly one kernel (or, for `delay`, copies), and the usual description, timing and parameters.

## Order

M29 depends on M26. It can run in parallel with M27, M28, M30 and M33.

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

**In:** the seven blocks of §2 in `sdrk-blocks` (one module each, named by M24.1), descriptions, storage sizing, pipeline kinds with `plan` figures and costs, the regenerated schema, the WebAssembly kind list (if it lists kinds), tests, `docs/blocks.md` (generated table and one paragraph per block), `docs/pipeline.md` (cost table), `docs/timing.md` (delay table), `CHANGELOG.md`; the reserved names of these kinds move out of the reserved-names table of `docs/blocks.md`.

**Out:** new kernels, two-input blocks (M30), any change of the existing blocks, Q15 or SIMD work beyond what the kernels already have.

## Specification

General rules: the construction pattern of M09 §4–§5 (`Config`, `validate(rate_in)`, `rate_out`, `storage_bytes`, `new(config, rate_in, storage)`), descriptions and parameter checks apply unchanged. Every block works through its input in one kernel call per `process` (no per-sample calls), limited by the output space. The kernels are called through the facade (`Sdrk`), so every backend is used.

### 1. Common rules

- Real-sample variants follow rule 9 of M24.1: `<Kind>Block<T>` for `C32`/`CQ15`, `<Kind>RealBlock<T>` for `f32`/`Q15`. The pipeline kind chooses the variant by the port type of its input, as `convert` and `biquad` do.
- Filter design reuses the code of the `fir` and `duc` blocks (`DesignSpec`, `kaiserord`, `firwin_lowpass`, the quantization of `fir`); no new design function.
- `timing().delay` is the delay of the kernel state (`Timing::group_delay`) in input samples; IIR stages report 0 (as `deemphasis` and `biquad` do).

### 2. The blocks

| Kind | Structs | In → Out | Parameters (unit, mutability, default) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `quad_demod` | `QuadDemodBlock<T>` | C32 → f32, CQ15 → Q15 | `deviation_hz` (Hz, build, 75000; > 0; C32 only, as `fm_demod`) | `quad_demod` with the gain of `fm_demod` (C32 `fs / (2π·deviation_hz)`, CQ15 raw 32767); one output per input |
| `deemphasis` | `DeemphasisRealBlock<T>` | f32/Q15 → same | `tau_us` (us, build, 50; 1 … 10000) | `deemphasis` with `DeemphasisState::new(tau_us·1e-6, fs)`; no complex variant (stereo de-emphasis is a `biquad`, M24) |
| `rotate` | `RotateBlock<T>` | C32/CQ15 → same | `frequency_hz` (Hz, **runtime**, required; `|f| < fs/2`) | `rotate` with `Nco::new(frequency_hz, fs)`; a change keeps the oscillator phase (as `ddc`); positive values shift up |
| `decimate` | `DecimateBlock<T>`, `DecimateRealBlock<T>` | C32/CQ15/f32/Q15 → same | `factor` (I64, build, required; 1 … 1024), `cutoff_hz` (Hz, build, default `0.4·fs/factor`), `transition_hz` (Hz, build, default `0.1·fs/factor`), `attenuation_db` (dB, build, 60), `max_taps` (I64, build, 1023) | complex: `fir_decimate`; real: `resample_real` with `L = 1`, `M = factor` (the kernel `fm_demod` and `am_demod` use for their audio stage, so a plan of those blocks (M32) stays bit-identical); taps designed at the input rate as in `fir` |
| `interpolate` | `InterpolateBlock<T>`, `InterpolateRealBlock<T>` | C32/CQ15/f32/Q15 → same | `factor` (I64, build, required; 1 … 1024), `cutoff_hz` (default `0.4·fs`), `transition_hz` (default `0.1·fs`), `attenuation_db` (60), `max_taps` (1023), all relative to the *input* rate | `fir_interpolate` / `fir_interpolate_real`; taps designed at the output rate and multiplied by `factor`, exactly as in `duc` |
| `mag` | `MagBlock<T>` | C32 → f32, CQ15 → Q15 | none | `mag` (the kernel of `am_demod`) |
| `delay` | `DelayBlock<T>`, `DelayRealBlock<T>` | C32/CQ15/f32/Q15 → same | `samples` (I64, build, required; 0 … 65536) | a ring buffer of `samples` elements from the storage; output `n` is input `n − samples`, the first `samples` outputs are zero; timing ratio 1, delay `samples` |

Validation as in the existing blocks: `decimate` rejects a stop band above the output Nyquist frequency (`cutoff_hz + transition_hz/2 > fs/(2·factor)`) with `OutOfRange`; `interpolate` rejects `cutoff_hz + transition_hz/2 > fs/2`; `rotate` rejects `|frequency_hz| ≥ fs/2`. The pipeline reports these as `rate_unsupported` like the other blocks.

### 3. Pipeline kinds and costs

One kind per block, registered like the existing ones; `kernels()` names the kernel(s) of §2 (`delay`: none). Cost per input sample, added to the cost table of `docs/pipeline.md`: `quad_demod` 12, `deemphasis` 3, `rotate` 6, `decimate` `taps / factor`, `interpolate` `taps`, `mag` 4, `delay` 1. `cargo xtask schema` regenerates the schema (version unchanged).

### 4. Descriptions

Each description has a summary and, in `details`, what the block is for and where it sits in a chain (for example: `decimate` "a low-pass and decimation by an integer factor; for a factor of 2 with a cheap filter use `halfband_decimate`"; `delay` "aligns two paths before a block with two inputs; `Fork` (M31) and the pipeline (M34) insert it automatically"). The established abbreviations are written out once (rule 6 of M24.1).

## Tests

1. **Equals the kernel:** for every block and every sample type it supports, the output equals the direct kernel call on the same input with the same state (designed taps included) bit for bit; whole buffers and 20 chunk plans (testkit `ChunkPlan`, seeds 455–461, one per block).
2. **Equals the existing blocks where they overlap:** `decimate` complex with the defaults of `ddc` and `rotate` → `decimate` equals `ddc` (same frequency, decimation and filter) bit for bit on the fallback; `interpolate` → `rotate` equals `duc` on the fallback; `mag` equals the first stage of `am_demod` with the DC blocker off and `audio_decimation` 1. These tests are the first evidence for the plans of M32; a difference is recorded in the decisions log with the measured SNR and an open item (do not change either block).
3. **Description rules of M09** for all seven blocks; storage in an exact arena; no struct above 512 bytes; validation errors of §2.
4. **Runtime parameter:** `rotate` `frequency_hz` changes from the next sample with continuous phase.
5. **Pipeline:** a flow `rf (cu8) → rotate → decimate → quad_demod → deemphasis → decimate (real) → audio` builds, runs in push mode, and its output equals the `fm_receiver` flow with the same parameters on the fallback (or is recorded as in test 2).
6. Schema check; WebAssembly kind list updated.

## Acceptance

R1.2 commands; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline`; `cargo xtask schema --check`; `cargo xtask check` (name check of M24.1).

## Deliverables checklist

- [ ] Seven blocks with descriptions, storage sizing and doc examples
- [ ] Seven pipeline kinds, costs, regenerated schema, WASM list
- [ ] Tests 1–6; differences of test 2 logged if any
- [ ] `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, reserved-names table, CHANGELOG, decisions log
