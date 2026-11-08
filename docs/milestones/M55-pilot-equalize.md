# M55 — Kernel `pilot_equalize`: OFDM pilot-based channel equalization

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M07](M07-fft.md), [M45](M45-channelizer.md) (lane ports; through it M19 and M38) · Next: [M56](M56-fwht.md)

> **Status: planned** (written 2026-11-08). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone, and the last of the DSSS/OFDM kernels together with `fwht` (M56).

## Goal

A kernel `pilot_equalize` that equalizes one OFDM symbol in the frequency domain from its pilot subcarriers: a least-squares channel estimate at the pilot positions (`H = Y / P`), linear interpolation across frequency to the data subcarriers, a one-pole smoothing across OFDM symbols, a one-tap equalization of every data subcarrier (`X̂ = Y / H`), and a common-phase-error correction. It is the receive-side channel correction for the OFDM demodulator, following the forms of GNU Radio's `ofdm_chanest_vcvc` (LS estimate `H = Y/X` on a reference symbol) and `ofdm_equalizer_simpledfe` (per-carrier update with a forgetting factor α).

## Why

An OFDM receiver turns each symbol into `N` subcarriers with the FFT, but the channel multiplies every subcarrier by an unknown complex gain. Pilots — known symbols on known subcarriers — let the receiver measure that gain and divide it out, which is the whole point of OFDM (a flat one-tap equalizer per subcarrier instead of a long time-domain equalizer). For the LPI work the pilots may hop from symbol to symbol (keyed positions), so the estimate must interpolate from whatever positions are given, not a fixed comb.

## Order

M55 depends on M07 only in spirit (it processes the FFT output; it does not call `fft`). It is independent of the other kernels of the series and can run in parallel with every milestone after M07. The OFDM demodulator milestone that uses it also needs `fft`/`ifft` on lanes and the lane plumbing of M45.

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

**In:** kernel `pilot_equalize` (`C32`) in `sdrk_core::kernels::ofdm_eq` with `PilotEqualizeState`, Fallback, Reference, facade delegation, contract, `docs/kernels.md`; golden cases; unit, invariance, extreme tests; suite cases; `cargo xtask count`; CHANGELOG, decisions log; the block `PilotEqualizeBlock` and the pipeline kind `pilot_equalize` (§7) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the composite `ofdm_demod` and the lane kinds `fft`, `to_lanes`, `subcarrier_demap` (later milestone; names reserved); the FFT and guard removal (`to_lanes`, `fft` on lanes — later milestone); decision-feedback updating from data decisions (the `simpledfe` data path; backlog, because it needs the constellation and couples to the slicer); superimposed-pilot estimation (backlog); `CQ15` (backlog, OFDM dynamic range). Carried in: the LS pilot estimate, frequency interpolation, time smoothing, one-tap equalization, common-phase correction — the pieces a basic OFDM receiver needs.

## Specification

General rules of M19 apply. The kernel processes one OFDM symbol per call step but keeps a channel estimate across symbols (the time smoothing is a one-pole feedback per subcarrier), so as for the other feedback kernels the arithmetic is fixed and `Reference` follows it step by step (bit-exact, float32-model rule, D111); an f64 run is a regression guard.

### 1. Layout

One OFDM symbol is `N` subcarriers (`N` a power of two 16…65536), given as the FFT output in natural order (subcarrier 0 = DC). The subcarrier roles are fixed at construction by index lists:
- `pilots: &[u32]` — the pilot subcarrier indices for this symbol, with their known complex values `pilot_values: &[C32]` (same length). (Keyed/hopping pilots: the block passes a different `pilots`/`pilot_values` per symbol; the kernel takes them as inputs.)
- `data: &[u32]` — the data subcarrier indices to equalize and output.
- subcarriers in neither list (DC, guard/null carriers) are passed through unchanged or zeroed per a flag.

### 2. Per symbol

1. **LS estimate at pilots:** for each pilot index `p`, `Hp = Y[p] / Pval` (complex division `num/den = num·conj(den)/|den|²`, f32, no FMA; `|den|²` from the `mag_sq` rule). This is the `ofdm_chanest_vcvc` estimate `H = Y/X`.
2. **Interpolation across frequency:** the channel at every subcarrier `k` is the piecewise-linear interpolation of `Hp` over the pilot indices (real and imaginary parts interpolated separately, f32): between two pilots `p0 < p1`, `H[k] = Hp0 + (Hp1 − Hp0)·((k − p0)/(p1 − p0))`; below the first pilot and above the last, hold the nearest pilot (GNU Radio's `ofdm_equalizer_static` holds; we interpolate between and hold at the edges — logged, with the reason that interpolation beats the nearest-pilot hold of `simpledfe` for a frequency-selective channel). With a single pilot, `H[k]` is that pilot's value everywhere (flat-channel / common-phase only).
3. **Time smoothing:** `Hs[k] = α·Hs_prev[k] + (1 − α)·H[k]` per subcarrier, `α` the forgetting factor (default 0.5; GNU Radio `simpledfe` default α = 0.1 weights the *old* estimate — note the convention: there `H ← α·H_old + (1−α)·H_new`, so our α has the same meaning; logged). On the first symbol after `reset`, `Hs = H` (no old estimate). An optional `initial_taps` seeds `Hs_prev` (as `simpledfe`'s `initial_taps`).
4. **Equalization:** for each data index `k`, `X̂[k] = Y[k] / Hs[k]` (the same complex division). Output the equalized data subcarriers in the order of `data`.
5. **Common-phase-error (CPE):** optional (`cpe: bool`). The residual common phase is `φ = arg( Σ_p Y[p]·conj(Hp·Pval) )` estimated from the pilots (the sum of pilot residuals), and every equalized output is derotated by `exp(−jφ)` using the `rotate` tables. With `cpe = false` step 5 is skipped. (CPE tracking is standard in OFDM where the pilots are too sparse to catch a per-symbol phase; logged as an addition over the GNU Radio equalizers, which fold phase into the per-carrier H.)

### 3. State and trait

```rust
pub struct PilotEqualizeState<..> { /* N, alpha: f32, cpe: bool, pass_nulls: bool, hs: Store (N complex), first: bool */ }
impl PilotEqualizeState<..> {
    pub fn new(fft_len: usize, alpha: f32, cpe: bool, pass_nulls: bool, hs_store: Store) -> Result<Self>;
    pub fn alpha(&self) -> f32;  pub fn set_alpha(&mut self, a: f32) -> Result<()>;
    pub fn seed_taps(&mut self, taps: &[C32]) -> Result<()>;  // initial_taps
    pub fn reset(&mut self);   // first = true, hs cleared
}
pub trait PilotEqualize {
    fn pilot_equalize(&self, state: &mut PilotEqualizeState<..>,
        symbol: &[C32], pilots: &[u32], pilot_values: &[C32], data: &[u32],
        output: &mut [C32]) -> Result<()>;
}
```

- `symbol.len()` must equal `N`; `output.len()` must equal `data.len()`; `pilots.len()` must equal `pilot_values.len()` and be ≥ 1; every index in `pilots` and `data` must be `< N` and `pilots` strictly increasing; `alpha` finite in `[0, 1)`. Violations → `InvalidParameter`/`LengthMismatch`, nothing written.
- A pilot value of 0, or a smoothed `Hs[k]` of 0 magnitude, would divide by zero: the kernel outputs 0 for that subcarrier and sets a `degenerate` count (readable), rather than NaN — documented, because a null channel tap is a real case.
- Timing: one symbol in, `data.len()` subcarriers out; the kernel is a per-symbol transform with state. On lanes (later milestone) it is one lane step per call step. Group delay 0 (frequency domain).
- Storage: `Hs` (`N` complex) and the small state. `storage_bytes` exact.

### 4. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `pilot_equalize` | `kernels::ofdm_eq::PilotEqualize` | `C32` | bit-exact | the float32 model of §2 (complex division `num·conj(den)/mag_sq(den)`, linear frequency interpolation, one-pole time smoothing, optional CPE derotation with the `rotate` tables), each operation f32 without FMA; `Reference` follows it step by step; golden from the exact float32 model with an f64 guard; not accelerated (`backend_for` reports `fallback`) |

### 5. Golden vectors

New file `tests/golden/sdrk_golden/cases/ofdm_eq.py`, the float32 model.

| Case | Seed | `N` | pilots | scenario |
| --- | --- | --- | --- | --- |
| `pilot_equalize/flat_64` | 750 | 64 | 4 evenly spaced | a flat channel (one gain); the estimate ≈ the gain, data recovered |
| `pilot_equalize/selective_256_smooth` | 751 | 256 | 16 comb | a frequency-selective channel over 8 symbols; time smoothing α = 0.5; interpolation across the comb |
| `pilot_equalize/keyed_pilots_128_cpe` | 752 | 128 | 8 hopping per symbol, CPE on | pilots at different indices each symbol, a per-symbol common phase; CPE removes it |

- Stored: the symbols (FFT output), the pilot index/value lists per symbol, the data indices, α, flags; expected equalized data per symbol and the final `Hs`. `snr_db_guard` from the f64 model. Bit-exact `==` against the float32 model.
- Cross-check noted in the comment: GNU Radio `ofdm_chanest_vcvc` uses `H = Y/X` (our step 1) and `ofdm_equalizer_simpledfe` uses `H ← α·H + (1−α)·Y/P` for pilots and `Y/H` for data (our steps 3–4); we add frequency interpolation and CPE, which those blocks leave to neighbours.

### 6. Cost

`cargo xtask count` entry `pilot_equalize_256` (per subcarrier; a complex division plus the interpolation and smoothing). Record in `docs/performance.md`.

**Measurement (D32, D33, D68).** The entries above and the block entry of §7 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`). A library reference enters only where it does the same work (D32); otherwise the rows are information. The D321 gate applies once a SIMD path exists.

### 7. Block and kind `pilot_equalize`

`PilotEqualizeBlock` (`Lanes<C32>` of width `N` → `Lanes<C32>` of width `data`, one lane step per OFDM symbol) around the kernel `pilot_equalize`. The layout comes from scalar parameters: carrier `k` is a null when `dc_null` and `k = 0`, or when it lies among the `guard` carriers centred on `N/2` (the band edge in natural order); the pilots are the non-null carriers with `k mod pilot_spacing = pilot_offset`, all with the value `1 + 0j`; the data carriers are the others, in index order. `PilotEqualizeBlock::with_layout` takes any index lists and pilot values of the caller (Rust API only).

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `pilot_equalize` | `PilotEqualizeBlock` | lanes<c32> (`N`) → lanes<c32> (`data`) | `fft_size` (I64, build, required; a power of two 16 … 65536, equal to the input width), `pilot_spacing` (I64, build, required; 2 … `N`), `pilot_offset` (I64, build, 0; 0 … `pilot_spacing − 1`), `guard` (I64, build, 0; 0 … `N − 2`), `dc_null` (Bool, build, true), `alpha` (linear, **runtime**, 0.5; 0 ≤ α < 1), `cpe` (Bool, build, false) | `pilot_equalize`; a layout with no pilot or no data carrier is `invalid_value` at `params.pilot_spacing`; the output width is `data` |

- Reading `degenerate` (the kernel's count). `PilotEqualizeConfig`. Storage: `Hs` (`N`) and the index lists; exact. Timing ratio 1 (steps), delay 0. A change of `alpha` takes effect with the next symbol (`set_alpha`).
- Pipeline kind `pilot_equalize` (a `LanesNode`; the width of the input through `Kind::input_width`, a width other than `fft_size` is `type_mismatch`), cost 12 per input value (a complex division per data carrier, the interpolation and the smoothing; checked against the count, logged), a row in `docs/pipeline.md` and `docs/timing.md`.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 750 … 759.

1. Golden cases of §5 through facade, `Fallback`, `Reference` (`sdrk-testkit/tests/golden_m55.rs`), with the f64 guard.
2. Unit tests (`sdrk-core/tests/kernels_m55.rs`): construction/range rules (bad `N`, α ≥ 1, empty pilots, index ≥ N, non-increasing pilots, length mismatches); a flat channel estimate equals the gain; the interpolation is exact on a linear channel; single-pilot gives a constant `H`; time smoothing converges (a static channel reaches the true H within tolerance after several symbols); CPE removes a known common phase; the zero-division path (a zero pilot value / null tap outputs 0 and counts, no NaN); `reset`, `seed_taps`.
3. Block-size invariance / state continuity: processing a stream of symbols one at a time vs. the same symbols is identical (the kernel is per-symbol; the "chunking" here is symbol granularity) — 20 orderings not applicable; instead a continuity test that `Hs` after `n` symbols is independent of any grouping (seed 753).
4. Extremes: all-zero symbol (outputs 0, degenerate count = data len); full scale; NaN/∞ in one subcarrier (that output 0 or NaN per the zero-division rule — documented); the largest `N` (65536) with a sparse pilot set; `cpe` on with pilots summing to 0 residual (φ = 0).
5. Suite cases `pilot_equalize_c32` and `extremes_m55`.
6. **Block and kind** (`sdrk-blocks/tests/pilot_equalize_blocks.rs`, `sdrk-pipeline/tests/pilot_equalize_kinds.rs`; seeds 754, 755): the block equals the kernel bit for bit over a sequence of symbols, in 20 cuts of the lane steps into calls; the layout from the parameters for `N` = 64 and 256; the description rules of M09, exact storage, struct at most 512 bytes, validation errors at their parameter; a pipeline flow `channelize` (as a lane source of width `N`) → `pilot_equalize` → `channel_select` equals the blocks wired by hand; `cargo xtask schema --check` and the WASM kind list include `pilot_equalize`.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the entry of §6 and `pilot_equalize_block`); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [ ] Kernel `pilot_equalize` (`C32`): `PilotEqualizeState`, trait, Fallback, Reference, facade delegation, `backend_for` (fallback)
- [ ] Contract row and section in `docs/kernels.md` (LS estimate, interpolation, smoothing, CPE; the GNU Radio cross-check)
- [ ] Golden cases (seeds 750–752) and the float32 Python model with the f64 guard
- [ ] Tests 1–6 (seeds 750–755); suite cases
- [ ] Count and cycle entry (`cargo xtask count`, `cargo measure`), `docs/performance.md`
- [ ] Block `PilotEqualizeBlock` and kind `pilot_equalize` (§7): `docs/blocks.md` (entry; reserved-name row "delivered"), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `pilot_equalize_block`
- [ ] CHANGELOG, decisions log (the α convention, edge-hold interpolation, CPE, the zero-division rule)

## Status

Planned. Nothing is implemented.
