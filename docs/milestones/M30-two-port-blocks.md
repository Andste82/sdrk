# M30 — Two-port blocks and the basic math kernels

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M26](M26-block-ports.md) · Next: [M31](M31-composite-tools.md)

## Goal

Four new kernels (`add`, `mul`, `oscillate`, `interleave` with its inverse `deinterleave`), each in both precisions with reference, fallback, facade delegation, golden cases and the SIMD decision, and the blocks built on them and on the stereo kernels of M24: `add`, `mul`, `make_complex`, `to_complex`, `real_part`, `imag_part`, `nco`, `stereo_pll` and `stereo_matrix`. Blocks with one input get pipeline kinds now; blocks with two inputs (`Two` ports) get theirs with multi-input nodes in M34.

## Why

The composite blocks of M31–M32 and the transmitters of M35–M36 need to combine two signals (sum, product, real and imaginary part into one complex stream), to split a complex stream, to turn a real signal into a complex one, and to generate a carrier. Without these, every such feature would again be a special block with its own loop. The oscillator takes its clock from an input stream (it outputs one sample per input sample and ignores the values), so the pipeline needs no source nodes; its `harmonic` parameter gives phase-locked multiples of one oscillator (19 kHz pilot and 38 kHz subcarrier, M36).

## Order

M30 depends on M26. It can run in parallel with M27, M28, M29 and M33.

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

**In:** kernels `add`, `add_real`, `mul`, `mul_real`, `oscillate`, `interleave`, `deinterleave` in `sdrk-core` (Fallback, Reference, facade delegation, `docs/kernels.md` contracts); golden cases; suite cases; the SIMD paths and counts of §6; the nine blocks of §3 with descriptions; pipeline kinds for the five single-input blocks (§4); the regenerated schema; `docs/kernels.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, `CHANGELOG.md`.

**Out:** pipeline kinds for two-input blocks (M34), composition (`Chain` M27, `Fork` M31), an adder with more than two inputs (backlog), changes to `fm_stereo_demod` (it keeps its own loop; M32 gives it a plan).

## Specification

General rules from M02/M05/M06 apply (see M19). All f32 arithmetic is written without FMA in exactly the order given; Q15 arithmetic is in `i64` with `round_shift` and `sat16` (R7). Thresholds are not relaxed (the rule of M24 §Specification applies).

### 1. Kernels (all stateless except `oscillate`; unequal lengths are `LengthMismatch` and change nothing)

```rust
pub trait Add<T: ComplexSample>     { fn add(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>; }
pub trait AddReal<T: RealSample>    { fn add_real(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>; }
pub trait Mul<T: ComplexSample>     { fn mul(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>; }
pub trait MulReal<T: RealSample>    { fn mul_real(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<()>; }
pub trait Oscillate<T: ComplexSample> {
    /// out[n] = exp(j·2π·(k·phase_n)/2^32) with phase_n the NCO phase before sample n, k = harmonic;
    /// the NCO advances by out.len() steps.
    fn oscillate(&self, nco: &mut Nco, harmonic: u32, out: &mut [T]) -> Result<()>;
}
pub trait Interleave<T: ComplexSample> {
    /// out[n] = (re[n], im[n]); `im = None` gives im = 0.
    fn interleave(&self, re: &[T::Real], im: Option<&[T::Real]>, out: &mut [T]) -> Result<()>;
    /// re[n] = input[n].re, im[n] = input[n].im; either output may be None.
    fn deinterleave(&self, input: &[T], re: Option<&mut [T::Real]>, im: Option<&mut [T::Real]>) -> Result<()>;
}
```

| Kernel | C32 / f32 | CQ15 / Q15 | Class |
| --- | --- | --- | --- |
| `add`, `add_real` | `a + b` per component | `sat16(a + b)` per component in `i32` | bit-exact |
| `mul` | `re = ar·br − ai·bi`, `im = ar·bi + ai·br`, products first, then the difference/sum | `re = sat16(round_shift(ar·br − ai·bi, 15))`, `im = sat16(round_shift(ar·bi + ai·br, 15))` in `i64` (so `(-1)·(-1) = 32767` as R7) | bit-exact |
| `mul_real` | `a·b` | `sat16(round_shift(a·b, 15))` (the R7 multiply) | bit-exact |
| `oscillate` | `cos` and `sin` from `COS_F32` with the interpolation of the C32 `rotate` fallback at the phase `k·phase` (wrapping `u32` multiply) | `COS_Q15` with the interpolation of the CQ15 `rotate` at `k·phase` | C32 tolerance (SNR ≥ 120 dB against f64 `exp`, like `rotate`); CQ15 bit-exact against the integer table model |
| `interleave`, `deinterleave` | copies | copies | bit-exact |

`harmonic` is 1 … 8, else `InvalidParameter`. The phase model and the NCO are those of `rotate` (`docs/timing.md`); `oscillate` with harmonic 1 equals `rotate` applied to the constant input `(1, 0)` for C32 bit for bit (test), so the oscillator and the mixer agree.

### 2. Golden cases (`tests/golden/sdrk_golden/cases/basic_math.py`, `oscillate.py`)

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| add | `add_c32_4096`, `add_cq15_4096` | 463 | two noise vectors σ 0.4 (CQ15: quantized, plus 64 pairs at ±full scale for saturation) | sum (integer model for Q15) | bit-exact |
| add_real | `add_f32_4096`, `add_q15_4096` | 464 | as above, real | sum | bit-exact |
| mul | `mul_c32_4096`, `mul_cq15_4096` | 465 | noise σ 0.4 and 64 corner pairs (`-1·-1`, `-1·(−1 + j·−1)`) | float32 model / integer model | bit-exact |
| mul_real | `mul_f32_4096`, `mul_q15_4096` | 466 | as above, real | float32 / integer model | bit-exact |
| oscillate | `osc_c32_k1_8192`, `osc_c32_k2_8192`, `osc_cq15_k1_8192`, `osc_cq15_k3_8192` | 467 | 19 000 Hz at 192 kHz, start phase 0x12345678; k 1, 2, 3 | f64 `exp` (C32); table model (CQ15) | SNR ≥ 120 dB; bit-exact |
| interleave | `interleave_c32_1024`, `interleave_cq15_1024` (with and without `im`) | 468 | noise | copies | bit-exact |

All stateful cases also run in random chunks (ChunkPlan seed = case seed).

### 3. Blocks

The construction pattern, descriptions, storage and parameter rules of M09 §4–§5 apply. Two-input blocks use `Two<A, B>` (M26 §1, lockstep). Real variants follow rule 9 of M24.1.

| Kind | Structs | In → Out | Parameters | Kernel and notes |
| --- | --- | --- | --- | --- |
| `add` | `AddBlock<T>`, `AddRealBlock<T>` | `Two<T, T>` → T | none | `add` / `add_real` |
| `mul` | `MulBlock<T>`, `MulRealBlock<T>` | `Two<T, T>` → T | none | `mul` / `mul_real` (complex product for complex streams) |
| `make_complex` | `MakeComplexBlock<T>` | `Two<R, R>` → C (f32 → C32, Q15 → CQ15) | none | `interleave(re = a, im = Some(b))` |
| `to_complex` | `ToComplexBlock<T>` | R → C | none | `interleave(re, None)` |
| `real_part`, `imag_part` | `RealPartBlock<T>`, `ImagPartBlock<T>` | C → R | none | `deinterleave` with one output |
| `nco` | `NcoBlock<T>` | any stream of the precision (C32/f32 → C32, CQ15/Q15 → CQ15) | `frequency_hz` (Hz, **runtime**, required, `|f| < fs/2`), `harmonic` (I64, build, 1; 1 … 8), `phase_turns` (None, build, 0; 0 … 1, start phase as a fraction of a turn) | `oscillate`; one output per input sample, the input values are ignored (the input is the clock); a frequency change keeps the phase |
| `stereo_pll` | `StereoPllBlock<T>` | `Two<R, R>` (mpx, pilot) → C | `pilot_hz` (Hz, build, 19000), `natural_hz` (Hz, build, 20; 1 … 190), `pilot_level` (None, build, 0.1), `stereo` (Choice `auto`/`mono`, **runtime**, `auto`) | `stereo_pll`; `pilot_locked()` as in `fm_stereo_demod` |
| `stereo_matrix` | `StereoMatrixBlock<T>` | C → C | none | `stereo_matrix` |

All have ratio 1 and delay 0. `nco` reports `min_input` 1.

### 4. Pipeline kinds now

`to_complex`, `real_part`, `imag_part`, `nco`, `stereo_matrix` (one input each); cost per input sample 1, 1, 1, 6, 2. `add`, `mul`, `make_complex` and `stereo_pll` are not registered yet: the registry would reject a two-input description (record in `docs/pipeline.md` that M34 adds them). `cargo xtask schema` regenerates the schema.

### 5. Suite cases

Fallback vs Reference and invariance with 20 chunk plans: `add_c32`, `add_cq15`, `mul_c32`, `mul_cq15`, `mul_real_q15`, `oscillate_c32`, `oscillate_cq15`, `interleave_c32`, `extremes_m30` (silence, full scale, NaN/∞ for f32 inputs, Q15 corners); block cases `block_nco_phase_lock` (two `nco` blocks at 19 kHz harmonic 1 and harmonic 2 on the same clock: the second equals the square of the first within 1e-6 for C32, every sample, over 10^6 samples) and `block_make_complex_roundtrip` (`make_complex` → `real_part`/`imag_part` returns the inputs bit for bit).

### 6. SIMD backend and instruction counts

The rules of M24 §8 apply unchanged.

| Kernel | Accelerated | Reason | Count |
| --- | --- | --- | --- |
| `add`, `add_real`, `mul`, `mul_real` (all types) | yes | element-wise | against VOLK (`volk_32fc_x2_add_32fc`, `volk_32fc_x2_multiply_32fc`, `volk_32f_x2_add_32f`, `volk_32f_x2_multiply_32f`, `volk_16ic_x2_multiply_16ic`) where one exists, else against the fallback (`dev`, goal ≤ 0.5×) |
| `oscillate` | yes (C32: the table lookups are gathers; accelerate only if it beats the fallback, else list under "Not accelerated") | – | against the fallback |
| `interleave`, `deinterleave` | yes | shuffles | against VOLK `volk_32f_x2_interleave_32fc` / `volk_32fc_deinterleave_32f_x2` |

## Tests

1. Golden tests for every case of §2, whole and chunked; the property tests of `add`/`mul` against the reference with random inputs.
2. `oscillate` harmonic 1 equals `rotate` of `(1, 0)` for C32 bit for bit; harmonic k equals the phase multiplied (test on 10^5 samples).
3. Blocks: description rules of M09, exact arena, lockstep `LengthMismatch` for `Two` inputs, `nco` frequency change keeps the phase, `stereo_pll` block equals the kernel and `fm_stereo_demod`'s internal loop on the same input bit for bit.
4. Pipeline: `nco` driven by an input of 1 kHz `f32` audio gives a complex tone; `real_part` of `to_complex` returns the input; schema check.
5. The suite passes on the host and in the emulators (quick profile).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32`; `cargo xtask emu --target qemu-cortex-m4` (quick profile).

## Deliverables checklist

- [ ] Kernels `add`, `add_real`, `mul`, `mul_real`, `oscillate`, `interleave`, `deinterleave` with reference, fallback, delegation, contracts
- [ ] Golden cases (seeds 463–468), suite cases, extremes
- [ ] SIMD paths, counts and results
- [ ] Nine blocks; five pipeline kinds; regenerated schema
- [ ] Docs (`kernels.md`, `backends.md`, `performance.md`, `blocks.md`, `pipeline.md`, `timing.md`), reserved-names table, CHANGELOG, decisions log
