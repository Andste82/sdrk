# M13 — SIMD backend, SIMD FFT, instruction counting and comparative benchmarks

> Plan: [PLAN.md › Phase 4](../PLAN.md#phase-4--simd-backend-and-comparative-benchmarks-large) and [Testing and performance](../PLAN.md#testing-and-performance) · Depends on: [M12](M12-io-device-loopback-examples.md) · Next: [M14](M14-wasm-bindings.md)

## Status

Fully completed: Scope items 1 to 5 of the Specification except the performance limits. The crate `sdrk-backend-simd` (x86-64 SSE2/AVX2/AVX-512, AArch64 NEON, wasm32 SIMD128) with `convert`, `convert_out`, `rotate`, `fir`, `fir_decimate`, `fir_interpolate` (and `resample`), `mag_sq`, `quad_demod`, `fm_mod` and the in-place FFT for `C32` and `CQ15`, run-time or compile-time level selection, the facade feature `simd` and the flag `sdrk_fallback_only`; the suite cases `backend_vs_fallback_*` and the tests of the crate; `benches/` (VOLK declarations, rustfft comparison, `limits.json`, results for four platforms and the pipeline benchmark); `cargo xtask count` for callgrind, Intel SDE, QEMU with the `insn` plugin and wasmtime fuel; the CI jobs; `docs/backends.md` and `docs/performance.md`. Verified here: the whole workspace (default build, `-C target-cpu=x86-64-v3`, `--cfg sdrk_fallback_only`, without `--all-features`), the backend tests at SSE2, AVX2, AVX-512 (SDE), NEON (QEMU) and wasm32 (wasmtime, with and without `simd128`), and `cargo xtask count` for all four platforms.

Differences from the text, all in `docs/decisions-log.md` (M13): `rotate` `C32` uses polynomials instead of the table lookup (the table needs more instructions than VOLK allows); the FFT is a fused radix-2^2 transform on the bit-reversed order, not a radix-4 transform (the plan has no per-stage twiddles and the call no scratch); `iai-callgrind` is replaced by reading the `callgrind` total directly; the AVX-512 level needs the Ice Lake feature set (`fearless_simd`); `limits.json` has the optional key `waiver`; small kernels run more than `K = 16` calls; the facade keeps the fallback on targets without SIMD.

Not met (open items M13-4, M13-8 and M13-9 in `docs/open-items.md`): after the optimization (decisions D1 to D14) 156 of the 168 gated entries are `release` and pass, 143 of them at or below the goal ratio 1.00 against their fair reference; the release limit is 1.10. Two entries are `dev`: `convert_out` CU8 at `sse2` (1.35, M13-8) and `fft` `C32` at N = 64 with run-time detection (1.07 of the same kernel at compile-time `avx2`, limit 1.02, M13-9). Ten entries are `info` because the reference has no SIMD code at that level (VOLK's `rotate` and `quad_demod` at `sse2`/`sse4.2`, and `rotate`, the FIR family and `quad_demod` on wasm32, decision D12). `cargo xtask count --strict` therefore fails on x86_64 and wasm32, and the release 0.1 (version bump and tag `v0.1.0`) is not made.

## Goal

Create `sdrk-backend-simd` (x86-64 SSE2/AVX2/AVX-512, AArch64 NEON, wasm32 SIMD128) for `convert`, `rotate`, `fir`, `fir_decimate`, `fir_interpolate`, `mag_sq`, `quad_demod`, `fm_mod`, and a radix-4 SIMD FFT; wire it into the facade; build the instruction-counting infrastructure and comparative benchmarks against VOLK and rustfft; extend CI to AVX-512 (Intel SDE), AArch64 (QEMU), and wasm32 (wasmtime, with and without SIMD128). Release 0.1 (git tag) when every kernel meets the 1.15× hard limit.

## Why

Speed is the reason this library exists. The fallback defines correctness; backends must match it (bit-exact for simple kernels, within the numeric contract otherwise) while needing no more instructions than the established libraries.

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

**In:** backend crate, facade wiring (`simd` feature, default on), suite coverage of the backend, `benches/` (separate Cargo workspace), `cargo xtask count`, CI jobs, `docs/performance.md` results, `docs/backends.md`.

**Out:** embedded backends (M16 and backlog), relaxed SIMD (M14, opt-in), GPU.

## Specification

### 1. Crate and unsafe policy

- `crates/backends/sdrk-backend-simd`, `#![no_std]`, depends on `sdrk-core` and `fearless_simd` (pin the newest release; read its documentation for level/dispatch APIs). If `fearless_simd` cannot express an operation efficiently, use `core::arch` intrinsics in a separate module per architecture; such modules override `unsafe_code` to `deny` → allowed locally with `#[allow(unsafe_code)]`, and **every** `unsafe` block carries a `// SAFETY:` comment (enforced by `#![deny(clippy::undocumented_unsafe_blocks)]` in the crate's `lib.rs` — per-crate keys cannot be mixed with `[lints] workspace = true`). Log every such use in `docs/decisions-log.md`.
- `pub struct Simd;` implements the kernel traits it accelerates; for everything else the facade keeps `Fallback`.
- **Dispatch:** without `std`, the SIMD level is chosen at compile time from `cfg(target_feature = …)` (e.g. build with `-C target-cpu=x86-64-v3` for AVX2). Feature `std` on the backend (enabled by the facade's `std` feature) adds runtime detection (`is_x86_feature_detected!` / `std::arch::is_aarch64_feature_detected!`, or fearless_simd's level detection) at each kernel entry. These macros cache inside `std`, which is not *our* global state, so R6 is respected; the per-call cost is one load and branch (verify in the instruction counts and note it in `docs/backends.md`). Levels: x86-64 `sse2` (baseline) / `avx2+fma` / `avx512f+bw`; aarch64 `neon`; wasm32 `simd128` (compile-time only).
- Facade: feature `simd` (default) adds `delegate!(... => Simd)` for accelerated kernels under `#[cfg(feature = "simd")]`; `backend_for` reports `"simd"`. The configuration flag `--cfg sdrk_fallback_only` (set via `RUSTFLAGS`, **not** a cargo feature — a feature would be switched on by `cargo test --all-features` in R1.2 and silently disable the backend) forces the fallback everywhere for comparison and debugging. Declare it in the workspace lints: `unexpected_cfgs = { level = "warn", check-cfg = ['cfg(sdrk_fallback_only)'] }`.

### 2. Kernel requirements

| Kernel | Types | Contract for the backend | Notes |
| --- | --- | --- | --- |
| `convert` | all formats → C32/CQ15 | **bit-exact** vs fallback | vectorize U8/I8/I16 interleaved; packed formats may stay scalar |
| `convert_out` | C32/CQ15 → U8/I8/I16 | **bit-exact** | rounding `floor(v + 0.5)` must match exactly (no `round-half-even` instructions unless proven equal) |
| `rotate` | C32, CQ15 | C32 ≥ 120 dB, CQ15 ≥ 80 dB (same as fallback contract) | phase stays the u32 accumulator; table lookup + lerp vectorized (gather or per-lane loads) |
| `fir`, `fir_decimate`, `fir_interpolate` | C32, CQ15 | C32 ≥ 110 dB; **CQ15 bit-exact** | CQ15: products are exact in i32 (≤ 2^30); accumulate in i64 lanes; `madd` pair sums (i32) are allowed only when the state constructor proved that no tap equals −32768 (then a pair sum cannot overflow) and are widened to i64 before accumulation; pure i32 accumulation only if the constructor proved `Σ|h[k]| · 32768 < 2^31`. Final `round_shift(·,15)` + saturation as in the fallback |
| `mag_sq` | C32, CQ15 | C32 ≥ 140 dB; CQ15 bit-exact | |
| `quad_demod` | C32, CQ15 | C32 ≥ 100 dB; CQ15 ≥ 70 dB | vectorized `atan2` approximation allowed only if it meets the contract (polynomial of sufficient degree); document max error |
| `fm_mod` | C32, CQ15 | as fallback contract | phase accumulation must stay exact (integer) |
| `fft` | C32, CQ15 | C32 ≥ 110 dB; CQ15 fixed/BFP: M07 table | radix-4 (with a final radix-2 stage for odd log2 N), vectorized butterflies, structure modelled on rustfft's SIMD radix-4; the BFP exponent may differ from the fallback — compare after applying `2^exponent` |

FMA is allowed in backends for tolerance kernels (state it in `docs/backends.md`); never in bit-exact kernels.

### 3. Correctness coverage

- Suite (testkit): add `backend_vs_fallback_*` cases for every accelerated kernel and type, run through the facade with lengths 0, 1, 3, 15, 16, 17, 255, 4096 and random chunk plans (tail handling), plus extreme inputs (full scale, −32768, NaN in f32 where defined). Bit-exact kernels compare bits; tolerance kernels compare with the contract against the **reference**.
- All golden tests already run through the facade, so they now exercise the backend automatically; additionally run the whole test suite with `RUSTFLAGS="--cfg sdrk_fallback_only"` in CI so the fallback stays covered.

### 4. Benchmarks and instruction counting (`benches/`, its own Cargo workspace, not a member)

- `benches/Cargo.toml` (workspace with `publish = false` crates): `kernels` (our kernels via the facade, rustfft as dependency), `volk-sys` (hand-written `extern "C"` declarations for the used VOLK functions; links the system VOLK found by `pkg-config`; VOLK is LGPL-3.0 — it is never a dependency of a library crate, which is why `benches/` is a separate workspace outside `cargo deny`'s scope; document this).
- Measured pairs (same input size N = 4096 unless stated, same build flags, `-C target-cpu` per platform level):

| Our kernel | Compared against |
| --- | --- |
| `convert` CS8→C32, CS16→C32 | `volk_8i_s32f_convert_32f`, `volk_16i_s32f_convert_32f` on 2N values |
| `convert_out` C32→CS16 | `volk_32f_s32f_convert_16i` on 2N values |
| `rotate` C32 | `volk_32fc_s32fc_x2_rotator2_32fc` (or `_rotator_32fc` if the installed VOLK lacks it; log) |
| `fir` C32, 64 taps | `volk_32fc_32f_dot_prod_32fc` called per output (as GNU Radio's `fir_filter_ccf`) |
| `fir_decimate` C32, 127 taps, M = 10 | same, per output |
| `mag_sq` C32 | `volk_32fc_magnitude_squared_32f` |
| `quad_demod` C32 | `volk_32fc_x2_multiply_conjugate_32fc` + `volk_32fc_s32f_atan2_32f` |
| `fm_mod` C32 | own fallback (no VOLK equivalent) |
| `fft` C32 N = 64, 1024, 4096 | `rustfft` (`FftPlanner`, forward, in place) |
| `fft` CQ15 | own fallback |

- **Counting method (differential):** each measurement binary runs `warmup` then the kernel `k` times on the same buffers; it is executed twice (k = 0 and k = K) under the counter; `instructions per sample = (count(K) − count(0)) / (K · N)`. This removes startup, allocation and setup. `K = 16`.
- Counters per platform, all driven by `cargo xtask count --platform <p>`:
  - `x86_64` (`sse2`, `avx2`): Valgrind `callgrind` via `iai-callgrind` (Linux x86_64 and Linux arm64 hosts; not available on Windows — the counting job runs on Linux only; log).
  - `x86_64-avx512`: Intel SDE (`sde64 -spr -mix -- <bin>`; parse the total instruction count of the mix report).
  - `aarch64`: QEMU user mode `qemu-aarch64 -plugin <libinsn.so> -d plugin`; `xtask` builds the `insn` plugin (source `tests/tcg/plugins/insn.c` in the QEMU tree; QEMU configured with `--enable-plugins`; verify the location for the pinned QEMU version, it has moved between releases) into `target/tools/` if missing (document the needed packages).
  - `wasm32` (`simd128` vs scalar fallback): `wasmtime` with fuel metering (`--fuel`, read consumed fuel) — compare our SIMD128 build against our fallback build.
- `benches/limits.json`: per kernel/platform `{ "status": "release" | "dev" }`. `xtask count` writes `benches/results/<platform>.json` (counts, ratio, SNR of ours and of the comparison against the f64 reference) and fails if a `release` entry exceeds **1.15×** or a `dev` entry exceeds **1.50×**; between 1.10× and 1.15× it prints a warning. Accuracy must be equal or better: if our SNR is lower than the comparison's by more than 1 dB, fail.
- **Pipeline benchmark:** `fm_receiver.json` at 2.4 MHz: instructions per input sample (callgrind, x86_64) reported together with the latency from `Pipeline::latency` (samples and µs) in `benches/results/pipeline.json` and `docs/performance.md`.

### 5. CI additions (`.github/workflows/ci.yml`)

| Job | Host | What |
| --- | --- | --- |
| `tier1` (existing, three hosts) | as M00 | now with `simd` default; plus a second test run with `-C target-cpu=x86-64-v3` on x86_64 hosts; `RUSTFLAGS="--cfg sdrk_fallback_only"` run |
| `avx512-sde` | ubuntu-24.04 | download pinned Intel SDE (cache it), run `cargo test --workspace` with `RUSTFLAGS="-C target-cpu=sapphirerapids"` and runner `sde64 -spr --` |
| `aarch64-qemu` | ubuntu-24.04 | cross-compile for `aarch64-unknown-linux-gnu`, run tests under `qemu-aarch64` (complements the native arm64 host) |
| `wasm-wasmtime` | ubuntu-24.04 | `wasm32-wasip1` tests under `wasmtime` with and without `-C target-feature=+simd128`, with wasmtime's **default** stack size (catches oversized stack frames) |
| `count` | ubuntu-24.04 (+ nightly arm64 QEMU) | `cargo xtask count --platform x86_64` on every PR touching kernels/backends; all platforms nightly |

### 6. Release 0.1

When every kernel in §4 has status `release` and passes: update CHANGELOG, bump workspace version to `0.1.0`, create git tag `v0.1.0`. **Do not publish to crates.io** — publishing needs an explicit maintainer decision (log it).

## Test vectors

No new golden files. Benchmarks use PCG32 noise (seed 1000+) and must report the error against the f64 reference.

## Tests

1. All existing tests pass with the backend active, on all three hosts and in SDE/QEMU/wasmtime jobs.
2. `backend_vs_fallback_*` suite cases.
3. `backend_for` returns `"simd"` for every accelerated kernel when the feature is on and `"fallback"` under `--cfg sdrk_fallback_only` (test written with `#[cfg(sdrk_fallback_only)]` / `#[cfg(not(sdrk_fallback_only))]`).
4. Miri (`cargo +nightly miri test -p sdrk-backend-simd` on the scalar/portable paths) in the nightly CI job if any `unsafe` exists.
5. `cargo xtask count` self-test: a deliberately slow dummy kernel (status `release`) makes the command fail (unit test of the limit logic with fabricated numbers).

## Acceptance

R1.2 commands; all CI jobs in §5 green; `cargo xtask count --platform x86_64` passes; nightly counts for `x86_64-avx512`, `aarch64`, `wasm32` pass; tag `v0.1.0` exists.

## Deliverables checklist

- [x] `sdrk-backend-simd` with all kernels in §2, dispatch, unsafe policy (every kernel of §2 for `C32` and `CQ15`; no `unsafe` code, so no Miri job; the contract per kernel is in [backends.md](../backends.md))
- [x] Facade feature `simd`, cfg flag `sdrk_fallback_only` (and `sdrk::SIMD_BACKEND`)
- [x] Suite backend cases (`backend_vs_fallback_*`, ten cases) and the tests of the backend crate
- [x] `benches/` workspace, VOLK FFI, rustfft comparison, `limits.json`, results (`benches/results/{x86_64,x86_64-avx512,aarch64,wasm32,pipeline}.json`)
- [x] `cargo xtask count` (callgrind, SDE, QEMU plugin, wasmtime fuel), with unit tests of the limit logic and the parsers; run for all four platforms
- [x] `docs/performance.md`, `docs/backends.md`
- [x] CI jobs written (`tier1` extra runs, `avx512-sde`, `aarch64-qemu`, `wasm-wasmtime`, `count`, `count-nightly`); the jobs ran on the pull request of the SIMD optimization and passed except `wasm-bindings` (Chrome and chromedriver of the runner differ, open item M13-5)
- [ ] Every kernel within the limit and the tag `v0.1.0`: **not met**, see the status above. The old waivers for `convert_out`, the `C32` FFT and `fft` at N = 64 on wasm32 are gone: those entries pass the release limit of 1.10 now. What remains are M13-8, M13-9 and the `info` entries (decision pending in M13-4). The workspace version stays 0.0.1 and no tag is created.

### Status of the acceptance criteria

| Criterion | State |
| --- | --- |
| R1.2 commands (fmt, clippy, tests, `xtask check`, `xtask bare-metal`) | pass locally on Linux x86_64 (also with `-C target-cpu=x86-64-v3` and `--cfg sdrk_fallback_only`) |
| backend tests under SDE (AVX-512), QEMU (NEON), wasmtime (with and without `simd128`) | pass locally |
| `cargo xtask count --platform x86_64` | passes; `--strict` fails on the `dev` and `info` entries listed in the status |
| nightly counts `x86_64-avx512`, `aarch64`, `wasm32` | pass locally; `--strict` passes at AVX-512 and AArch64, fails on wasm32 (`info` entries) |
| all CI jobs of §5 green, tag `v0.1.0` | all green on the pull request except `wasm-bindings` (M13-5); the nightly jobs have not run on the branch; the tag is not made (M13-4) |
