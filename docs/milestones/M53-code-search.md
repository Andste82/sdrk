# M53 — Kernel `code_search`: parallel code-phase acquisition

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M07](M07-fft.md), [M50](M50-xcorr-fft.md) · Next: [M54](M54-code-track.md)

> **Status: planned** (written 2026-11-08). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

## Goal

A kernel `code_search` that acquires a direct-sequence signal: over one or more dwells of `K` samples (one code period at `samples_per_chip` samples per chip), it searches every code phase and a grid of Doppler frequencies at once with the FFT, and returns the strongest cell (code phase, Doppler, peak magnitude) together with a CFAR test statistic. It is the parallel-code-phase-search (PCPS) acquisition of a GNSS receiver (`GPS_L1_CA_PCPS_Acquisition` in GNSS-SDR), the first stage a DSSS receiver runs before it can track.

## Why

Before a receiver can despread, it must find where in the code it is (the code phase, up to `K` chips) and how far the carrier has drifted (the Doppler), with the chip SNR deep below 0 dB. The FFT does every code phase in one transform (`IFFT{ FFT{x} · conj(FFT{code}) }` is the circular correlation at all lags); a loop over Doppler bins covers the frequency axis. This is the only way to acquire a long code at a tractable cost, and it is the standard method of every software GNSS receiver.

## Order

M53 depends on M07 (the `fft` kernel) and reuses the idea of M50 (`xcorr_fft`, but circular here, one code period), and on M51/M52 only for the code model (not a hard dependency). It can run in parallel with every milestone after M07; the DSSS receiver milestone that uses it also needs M54.

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

**In:** kernel `code_search` (`C32`) in `sdrk_core::kernels::acquire` with `CodeSearchState`, Fallback, Reference, facade delegation, contract, `docs/kernels.md`; golden cases; unit, statistic and extreme tests; suite cases; `cargo xtask count`; CHANGELOG, decisions log; the block `CodeSearchBlock` and the pipeline kind `code_search` (§6) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the block and kind `code_acquire` with the threshold policy (from a false-alarm probability) and the hand-off to `code_track` (later milestone; the name `code_acquire` is reserved; the primitive block `code_search` of §6 reports every search and decides nothing); a bit-synchronisation search over data-bit boundaries (backlog); fine Doppler (`make_two_steps`; backlog); `CQ15` (backlog). Non-coherent integration across dwells is in (it is cheap and essential below 0 dB); coherent integration longer than one code period is a flow-level choice (longer `K`), not a kernel feature.

## Specification

General rules of M19 apply. The kernel works one dwell at a time and accumulates the search grid in caller storage, so a flow can run `max_dwells` dwells and stop early.

### 1. Method (PCPS, after GNSS-SDR)

`K` samples per dwell, `D` Doppler bins. The local code is supplied once as its length-`K` forward FFT conjugated, `Dconj[k] = conj(FFT_K{ code_resampled }[k])` (the block resamples the code to `samples_per_chip` and FFTs it; the kernel takes `Dconj` ready). Per dwell, for each Doppler bin `d` with frequency `f_d`:

1. **Carrier wipe-off:** `x_d[n] = x[n] · exp(−j 2π f_d n / fs)`, `n = 0…K−1`, the phase from the `rotate`/`oscillate` tables (`Nco` at `−f_d`), with the NCO phase referenced to the dwell's absolute start sample so dwells are consistent.
2. `X_d = FFT_K{ x_d }`.
3. `Y_d[k] = X_d[k] · Dconj[k]` (element-wise; `Dconj` already holds the conjugate of the code FFT).
4. `R_d = IFFT_K{ Y_d }` (the circular correlation at every code phase τ = 0…K−1; the normalised inverse of M07, so `R_d[τ] = (1/K) Σ ...`).
5. **Grid:** `grid[d][τ] += |R_d[τ]|²` (accumulated over the dwells; one dwell = overwrite on the first, add after).

After the requested dwells:
- **Peak:** `{S_max, d*, τ*} = max over d, τ of grid[d][τ]`.
- **Input power:** `P_in = (1/K) Σ_n |x[n]|²` over the dwells (the mean per dwell, averaged).
- **Test statistics** (both returned; the block picks one):
  - GLRT / peak-to-input: `Γ = 2·K·S_max / P_in` (the GNSS-SDR `Gamma_GLRT`; CFAR, independent of the input level).
  - peak-to-grid-floor: `Γ_floor = S_max / mean(grid over all cells except a guard window around the peak)` (a classic peak-to-average; more robust when a strong cross-correlation sits in one Doppler bin).
- Output: `CodeSearchResult { code_phase: τ*, doppler_bin: d*, doppler_hz: f_{d*}, peak: S_max, glrt: Γ, peak_to_floor: Γ_floor, input_power: P_in }`. The kernel declares nothing acquired/not; the block compares `Γ` to a threshold (from a false-alarm probability) at a later milestone.

**Doppler grid.** `f_d = f_center + (d − floor((D−1)/2))·f_step`, `d = 0…D−1`, the symmetric placement of GNSS-SDR (so `D = ceil(2·doppler_max/doppler_step)` bins span ±doppler_max around `f_center`). `f_center`, `f_step`, `D` and `fs` are state parameters.

### 2. State and trait

```rust
pub struct CodeSearchState<..> { /* K, D, fs, f_center, f_step, samples_per_chip, dwell_count, grid: &mut [f32] (D*K), scratch FFT buffers, FftPlan<K> */ }
impl CodeSearchState<..> {
    pub fn new(code_fft_conj: &[C32], doppler_bins: u32, f_center: f64, f_step: f64, fs: f64,
               grid: GridStore, scratch: ScratchStore, fft: FftPlan<..>) -> Result<Self>;
    pub fn reset(&mut self);   // clears the grid and the dwell count
    pub fn dwells(&self) -> u32;
    pub fn code_len(&self) -> usize;  // K
}
pub trait CodeSearch {
    /// One dwell of exactly K samples. Accumulates into the grid; returns the current best.
    fn code_search(&self, state: &mut CodeSearchState<..>, dwell: &[C32]) -> Result<CodeSearchResult>;
}
```

- `dwell.len()` must equal `K` (`LengthMismatch`). `code_fft_conj.len()` must equal `K`; `K` is the FFT size (a power of two 16…65536). `grid` must hold `D·K` `f32`. A non-power-of-two `K`, `D = 0`, a non-finite or non-positive `fs`/`f_step`, or `f_center` not finite give `InvalidParameter`.
- Each call runs one dwell (steps 1–5 for every Doppler bin) and returns the running peak and statistics; calling it `n` times does `n` non-coherent dwells. `reset` starts a fresh search (e.g. for the next code or after a failed acquisition).
- Timing: the kernel is a block-wise analysis step, not a streaming rate-changer; the block of the later milestone owns the stream and feeds dwells. No `Progress`. Group delay is not meaningful (the result reports an absolute code phase).
- Storage: the grid (`D·K` f32), two `K`-sample complex scratch buffers (wipe-off/FFT in place and the IFFT result), the FFT plan twiddles. `storage_bytes` exact.

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `code_search` | `kernels::acquire::CodeSearch` | `C32` | tolerance | SNR ≥ 90 dB on the accumulated grid vs. an f64 model of steps 1–5 (two FFTs and a wipe-off per bin; the `fft` tolerance dominates); the peak index, the Doppler bin and both statistics match the f64 model (the index exactly, the statistics within the grid SNR); the fallback and the facade agree with the model within that |

`Reference` computes the grid directly in f64 (the circular correlation per bin by the naive DFT of M07's reference for small `K`, or by the f64 FFT path the reference exposes; the wipe-off exact in f64), and the statistics from the f64 grid. Because the index of the maximum could in principle flip between two near-equal cells under rounding, the golden cases are built so the true peak exceeds the runner-up by a clear margin (≥ 3 dB), and the test asserts the index exactly and the statistics within tolerance. `backend_for("code_search", C32)` reports `fallback` (the backend FFT is used inside; its result stays within the tolerance).

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/acquire.py`.

| Case | Seed | `K` | `samples_per_chip` | `D` | dwells | Scenario |
| --- | --- | --- | --- | --- | --- | --- |
| `code_search/clean_1023_1x` | 730 | 1024 | 1 (code length 1023 zero-padded to 1024) | 21 | 1 | code at a known phase and 0 Hz, high SNR; peak at the phase, bin = center |
| `code_search/doppler_511_2x` | 731 | 2048 | 2 (511-chip code) | 41 | 1 | a nonzero Doppler and a nonzero code phase; the statistics locate both |
| `code_search/weak_1023_8dwell` | 732 | 1024 | 1 | 21 | 8 | −12 dB chip SNR; the peak only clears the floor after 8 non-coherent dwells (the grid after 1 and after 8 dwells both stored) |

- The code, its FFT-conjugate, the input (code · carrier + noise), `f_center`, `f_step`, `fs` are stored. Expected: the f64 grid, the peak index, the Doppler bin, `Γ`, `Γ_floor`, `P_in`. Tolerance: grid `tolerance` SNR 90; indices exact; statistics `max_abs_err` scaled.
- The `Γ = 2·K·S_max/P_in` formula and the symmetric Doppler-bin placement are cross-checked against the GNSS-SDR documentation (cited in the decisions log; the page's "conjugate of code FFT" wording vs. its `D[k]` formula is noted — we conjugate, which is the correct correlation).

### 5. Cost

`cargo xtask count` entry `code_search_1024_21bins` (per dwell; `D` wipe-offs + `D` forward FFTs + `D` IFFTs + the grid update, over `K` samples — report per input sample = cost/(K)). Record in `docs/performance.md`; note that acquisition runs once per lock, not per sample, so throughput is secondary.

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`). A library reference enters only where it does the same work (D32); otherwise the rows are information. The D321 gate applies once a SIMD path exists.

**Reference (D32).** No library row: GNSS-SDR's PCPS is the same method, but it is built from pieces that already have reference rows (the carrier wipe-off as `rotate` against GNU Radio's rotator, the FFT of M07, the spectral product as `mul`), and a transcription would only repeat them. The rows are information; the speed baseline is the vectorised scalar fallback (and the D321 gate once a SIMD path exists).

### 6. Block and kind `code_search`

`CodeSearchBlock` (`C32` → packets) around the kernel `code_search`. The block cuts its input into back-to-back dwells of `K` samples from the first sample after construction or `reset`; after `dwells` dwells it writes one packet with the result and starts the next search (`CodeSearchState::reset`). The code is a PRBS (`sdrk_proto::prbs`, chip `b` → `1 − 2b`), every chip repeated `samples_per_chip` times, zero-padded to `K` = the smallest power of two ≥ `period · samples_per_chip` (at most 65536); the block computes `Dconj` once.

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `code_search` | `CodeSearchBlock` | C32 → packets | `code`, `code_seed` (as `spread`), `samples_per_chip` (I64, build, 1; 1 … 16), `doppler_max_hz` (Hz, build, required; > 0, below `rate / 2`), `doppler_step_hz` (Hz, build, required; > 0, at most `doppler_max_hz`), `doppler_center_hz` (Hz, build, 0), `dwells` (I64, build, 1; 1 … 1024) | `code_search`; `D = ceil(2 doppler_max_hz / doppler_step_hz)` bins (§1); more than 2^22 grid cells `D · K` is `invalid_value` at `params.doppler_step_hz` |

- **Packet:** 36 bytes little-endian — `code_phase` (u32, samples), `doppler_bin` (u32), `doppler_hz` (f64), `peak`, `glrt`, `peak_to_floor`, `input_power` (f32 each), `dwells` (u32); `index` = the input index of the first sample of the search; flags 0. `CodeSearchReport::from_packet` reads it back.
- `CodeSearchConfig`. Storage: the grid (`D · K` f32), `Dconj` and two scratch buffers (`K` C32 each), the twiddles; exact. Timing `Packets { max_per_input: 1/(K · dwells) }`, delay 0.
- Pipeline kind `code_search` (`c32` only, a `PacketOutNode`), cost `D · (5 log2(K) + 8)` per input sample (per bin the wipe-off, two `K`-point FFTs, the spectral product and the grid update; the implementer checks it against the count and logs it), a row in `docs/pipeline.md` and `docs/timing.md`.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 730 … 739.

1. Golden cases of §4 through facade, `Fallback`, `Reference` (`sdrk-testkit/tests/golden_m53.rs`): grid within 90 dB, exact indices, statistics within tolerance.
2. Unit tests (`sdrk-core/tests/kernels_m53.rs`): `new` rules (non-power-of-two `K`, `D = 0`, bad `fs`, wrong storage); a clean code at phase 0 and 0 Hz peaks at `(center bin, τ = 0)`; a shift of the input by `s` chips moves `τ*` by `s` (circular); a Doppler of one bin moves `d*` by one; `reset` clears the grid; two dwells accumulate (grid doubles for a static signal); the `LengthMismatch` on a wrong dwell length.
3. Statistic behaviour (`sdrk-testkit/tests/code_search_m53.rs`, seed 733): over 200 noise-only dwells (no signal) the GLRT stays below a value whose implied false-alarm rate matches the theory within a factor of 2; with a signal at −10 dB chip SNR and enough dwells the GLRT exceeds it — a small ROC sanity check (full ROC tooling is a later testkit milestone).
4. Extremes: `D = 1` (a single Doppler bin); the largest `K` (65536) with `D = 3`; an all-zero dwell (grid 0, `P_in` 0, the statistics defined as 0 with a guard against division by zero — documented); NaN/∞ in one input sample (that dwell's grid NaN; `reset` recovers).
5. Suite cases `code_search_c32` and `extremes_m53`.
6. **Block and kind** (`sdrk-blocks/tests/code_search_blocks.rs`, `sdrk-pipeline/tests/code_search_kinds.rs`; seeds 734, 735): the block gives the kernel results (the packet fields bit for bit) on whole buffers and in 20 chunk plans; a PRBS10 code at a known code phase and Doppler is found in one packet per search with `index` at the search start; the description rules of M09, exact storage, struct at most 512 bytes, validation errors at their parameter; a pipeline flow with a packets output equals the block; `cargo xtask schema --check` and the WASM kind list include `code_search`.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the entry of §5 and `code_search_block`); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [ ] Kernel `code_search` (`C32`): `CodeSearchState`, `CodeSearchResult`, trait, Fallback, Reference, facade delegation, `backend_for`
- [ ] Contract row and section in `docs/kernels.md` (PCPS steps, both statistics, the Doppler grid, the GNSS-SDR cross-check)
- [ ] Golden cases (seeds 730–732) and the Python model
- [ ] Tests 1–6 (seeds 730–735); suite cases
- [ ] Count and cycle entry (`cargo xtask count`, `cargo measure`), `docs/performance.md`
- [ ] Block `CodeSearchBlock`, `CodeSearchReport` and kind `code_search` (§6): `docs/blocks.md` (entry and reserved-name row), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `code_search_block`
- [ ] CHANGELOG, decisions log (the conjugation, the two statistics, the zero-division guard, the GNSS-SDR citation)

## Status

Planned. Nothing is implemented.
