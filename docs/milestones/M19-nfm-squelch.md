# M19 — NFM receive with squelch

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M15](M15-emulator-targets.md) (uses only what M00–M15 delivered; see *Order*) · Next: [M20](M20-am-receive.md)

## Goal

A complete narrowband FM receiver (two-way radio, amateur 2 m/70 cm, marine VHF, PMR446): the power squelch `squelch` as kernel and block, its pipeline kind, the example flow `nfm_receiver` and an end-to-end test that checks both the audio quality and the gate.

## Why

The FM demodulator of M08/M09 already handles narrowband FM through its parameters (`deviation_hz` 2.5–5 kHz, `deemphasis_us` 0 or 750). What makes it usable for channelised radio is the squelch: without it the speaker plays loud noise between transmissions. The squelch is also needed by AM (M20) and SSB (M21), so building it first, with the smallest feature around it, keeps the following milestones small. It also introduces the exact float32 golden model that kernels with discrete decisions need.

## Order

M19 to M23 form one series of receive features: M19 NFM, [M20](M20-am-receive.md) AM, [M21](M21-ssb-receive.md) SSB, [M22](M22-audio-filters.md) audio filters, [M23](M23-real-sampling-input.md) real-sampling devices. Each milestone delivers one feature completely (kernels, blocks, pipeline kinds, example flow, end-to-end test) and reuses what the earlier ones delivered. The series uses nothing from M16–M18; it is numbered after them because it was written later. Implementing it before M16 is allowed when the maintainer decides so; that decision goes into `docs/decisions-log.md`, and R1.1 then reads "in the order M00–M15, M19–M23, M16, …".

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

**In:** kernel `squelch` in `sdrk-core` (state, `Fallback`, `Reference`, facade delegation); block `squelch` with description; its pipeline kind and the regenerated schema; the example flow `nfm_receiver` (JSON and YAML); golden cases; suite cases; the end-to-end case `e2e_nfm_rx_*`; the SIMD decision and instruction counts of §6; the exact float32 model in the golden generator; `docs/kernels.md`, `docs/timing.md`, `docs/blocks.md`, `docs/pipeline.md`.

**Out:** noise-based squelch (high-pass noise detector), CTCSS/DCS decoding, CTCSS removal (M22), squelch state as pipeline events, AGC (M20).

## Specification

General rules from M02/M05/M06 apply: one trait per kernel, `Fallback` + `Reference` + facade delegation, no allocation, validation at entry, `Timing` implemented by every state. All f32 arithmetic is written without FMA in exactly the order given; Q15 arithmetic is in `i64` with `round_shift` and `sat16` from `sdrk_core::num` (R7).

**Thresholds and limits are not relaxed.** If a threshold, tolerance or count limit of this milestone is not met, do not change it in code, case files or `benches/limits.json`. Measure it, record the measured value and the reason in `docs/decisions-log.md` and an open item in `docs/open-items.md`, and report it in the pull request; a changed threshold needs the maintainer's approval. The only pre-approved changes are the ones stated in this file.

### 1. `squelch` — power gate (bit-exact in both types)

```rust
pub struct SquelchState<T: ComplexSample> { /* alpha, average power p, open, thresholds */ }
impl<T: ComplexSample> SquelchState<T> {
    /// threshold_db: open level in dBFS (−150 … 0); hysteresis_db ≥ 0 (close level = threshold − hysteresis); tau_s > 0
    pub fn new(threshold_db: f64, hysteresis_db: f64, tau_s: f64, rate: Rate) -> Result<Self>;
    pub fn set_thresholds(&mut self, threshold_db: f64, hysteresis_db: f64) -> Result<()>;
    pub fn is_open(&self) -> bool;
    pub fn reset(&mut self);   // p = 0, closed
}
// Timing: ratio 1/1, group delay 0
pub trait Squelch<T: ComplexSample> { fn squelch(&self, s: &mut SquelchState<T>, input: &[T], output: &mut [T]) -> Result<()>; }
```

`α = 1 − exp(−1 / (fs·tau_s))` in f64 (`OutOfRange` unless `0 < α ≤ 0.5`). Per sample: update the average power, then the state (open when `p ≥ P_open`, closed when `p < P_close`, otherwise unchanged), then output `x` when open and zero when closed. The squelch starts closed with `p = 0`.

- **C32:** `α` as f32; `P_open = 10^(threshold/10)`, `P_close = 10^((threshold − hysteresis)/10)` computed in f64 and cast to f32; `q = re·re + im·im`; `p = flush_denormal(p + α·(q − p))`. A non-finite `q` (NaN or ±∞ from a non-finite input) leaves `p` and the state unchanged; the sample itself is passed or zeroed by the current state. Closed output is `(+0.0, +0.0)`.
- **CQ15:** `α_raw = q15_from_f64(α)`; thresholds `floor(10^(db/10) · 2^30 + 0.5)` as `i64`; `q = re² + im²` of the raw values (Q30); `p = p + round_shift(α_raw · (q − p), 15)` in `i64`. Closed output is `CQ15` zero.
- **Reference:** for both types the same arithmetic (f32 for C32). The gate decision is discrete, so a reference in f64 could switch one sample earlier or later than the fallback; the specification therefore fixes the arithmetic, and the golden cases are computed by an exact float32 model (record this in the decisions log as an exception to the f64 reference rule, like the integer Q15 models).
- `set_thresholds` applies from the next sample; `p` and the state are kept.

### 2. Exact float32 model in the golden generator

`tests/golden/sdrk_golden/f32model.py`: helpers that compute with NumPy `float32` scalars one operation at a time (`f32(a) + f32(b)` etc.; NumPy rounds each scalar operation to float32 and never fuses), plus `flush_denormal`. Cases that use it are declared with class `bit_exact` and state `model: "float32"` in their JSON. Write a self-test of the helpers (against hand-computed values) like the Q15 model self-test of M04.

### 3. Block `squelch`

The construction pattern, descriptions, storage and parameter rules of M09 §4–§5 apply unchanged.

| Kind | Struct | In → Out | Parameters (unit, mutability, default) | Implementation |
| --- | --- | --- | --- | --- |
| `squelch` | `SquelchBlock<T>` | C32/CQ15 → same | `threshold_db` (dB, runtime, −40; −150 … 0), `hysteresis_db` (dB, runtime, 3; 0 … 40), `tau_s` (Seconds, build, 0.01) | `squelch`; `set_param` of a threshold calls `set_thresholds` |

`docs/blocks.md`: the squelch works on the channel signal, so it belongs **after** the channel filter (`ddc`) and **before** the demodulator and any gain control (after an AGC every signal, noise included, has the same level and the gate would always open).

### 4. Pipeline kind and example flow

- Kind `squelch` in `sdrk-pipeline` (`kinds.rs`: `Kind` with `kernels()` = `["squelch"]`). `plan` returns `Figures` exactly like `FmDemodKind::plan` (`crates/pipeline/sdrk-pipeline/src/kinds.rs`): `rate_out` and `memory_bytes` from the block's `rate_out` and `storage_bytes`, `delay` from its timing, `min_input` from `min_input()`, and `cost` per input sample by this formula, which is also added to the cost table of `docs/pipeline.md`: `squelch` **3**.
- `cargo xtask schema` regenerates `schema/sdrk-pipeline-v1.json`; adding a kind keeps the schema version (record in the decisions log).
- `examples/flows/nfm_receiver.json` with a YAML twin (M11 test 8 covers it): input `cu8` at 2.4 MS/s → `ddc` (decimation 50 → 48 kHz, `cutoff_hz` 6000, `transition_hz` 3000) → `squelch` (defaults) → `fm_demod` (`deviation_hz` 5000, `deemphasis_us` 0, `audio_cutoff_hz` 4000) → output `f32`. The YAML comments explain the parameters for 12.5 kHz (`deviation_hz` 2500) and 25 kHz channels.
- If the WASM bindings list block kinds anywhere (`index.d.ts`, `FlowBuilder`), add the new kind there.

### 5. Testkit

- Suite cases (Fallback vs Reference, invariance with 20 chunk plans): `squelch_c32`, `squelch_cq15`, `extremes_m19` (silence, full scale, NaN/±∞ for C32, Q15 corners); block case `block_squelch_gate` (opens and closes at the documented sample).
- **E2E-NFM (`e2e_nfm_rx_{f32,q15}`):** at 2.4 MS/s, `fm_baseband` (M08 §5; offset +300 kHz, deviation 5 kHz, audio 1 kHz, amplitude 1.0) scaled to 0.5 and present only in the middle half of the run (the transmission), plus complex noise σ 0.003 over the whole run, encoded to `CU8` with the testkit encoder → `convert` → `ddc` (frequency 300 kHz, decimation 50, taps as in the flow) → `squelch` (−40 dB, 3 dB, 0.01 s) → `fm_demod` (as in the flow). Checks:
  1. SINAD at 1 kHz over the middle 80 % of the transmission: **f32 ≥ 30 dB, Q15 ≥ 25 dB** (initial thresholds, see the M07 rule on thresholds).
  2. The gate opens at most 1 ms after the transmission starts and closes at most 0.1 s after it ends (closing takes `tau · ln(P_signal / P_close)`).
  3. Before the transmission and more than 0.1 s after it the audio is exactly zero (zeros through `quad_demod` give `arg(0) = 0`).
  Quick variant 0.4 s, full variant 2.0 s.
- The Cortex-M4 quick profile already takes about 280 of its 300 s: the new quick cases must stay cheap; if the quick profile of a target no longer fits its timeout, move cases to the full profile or raise that target's timeout, and log it.

### 6. SIMD backend and instruction counts

The rules of M13 and of the SIMD optimization apply to every new kernel unchanged (decisions D1–D14; `docs/backends.md`, `docs/performance.md`, `docs/decisions-log.md`). A new kernel gets level paths in `sdrk-backend-simd` for every level the backend supports (x86-64 SSE2, SSE4.2 where it differs, AVX2, AVX-512, AArch64 NEON, wasm32 SIMD128; run-time detection with the feature `std`; a generic `fearless_simd` path as the fallback of every level-specific path, D1) whenever vectorization pays off. A kernel whose work is a per-sample recursion cannot be vectorized without changing its result; it stays with the fallback, is added to the "Not accelerated" list of `docs/backends.md` with the reason, and is counted as `info` against the fallback. For every accelerated kernel:

1. **Contract:** the class of the fallback. Bit-exact kernels are bit-exact with `Fallback` on every level; tolerance kernels meet the SNR of their contract; FMA only where the contract allows it (D5).
2. **Verification:** a `backend_vs_fallback_*` suite case and a property test against the fallback at every level (the CI jobs `tier1` with the static levels, `avx512-sde`, `aarch64-qemu` and `wasm-wasmtime` run them); a level-specific rounding or saturation path also gets an exhaustive or boundary sweep like `convert_out_sweep`, run by the nightly job `exhaustive`.
3. **Counting:** the kernel in `benches/kernels` (`count run`, `count accuracy`), one entry per level in `benches/limits.json` against the fair reference named below, `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32` pass, the files in `benches/results/` are committed and `docs/performance.md` gets the rows. Release limit 1.10×, goal ≤ 1.00× (D14); the accuracy rule applies. An entry that cannot meet its limit yet carries a waiver that names an open item.
4. **Facade:** `backend_for` reports `simd` for the kernel; `RUSTFLAGS="--cfg sdrk_fallback_only"` still restores the fallback.
5. **Embedded backends:** M16–M18 accelerate the kernels listed in their own milestones. If one of them is implemented after this milestone, it decides there whether to include the new kernels.

| Kernel | Accelerated | Reason | Count |
| --- | --- | --- | --- |
| `squelch` (C32, CQ15) | no | the average power is a first-order recursion (one multiply-add per sample on the previous value) and the gate decision depends on it; only the cheap `\|x\|²` could be vectorized | `info` against the fallback at every level; listed under "Not accelerated" in `docs/backends.md` |

### 7. Documentation

`docs/kernels.md` (contract of `squelch`, the float32-model rule), `docs/timing.md` (ratio and delay), `docs/backends.md` (not accelerated, with the reason), `docs/performance.md` (the `info` rows), `docs/blocks.md` (generated table, chain position), `docs/pipeline.md` (kinds), `examples/README.md` (the flow), `tests/golden/README.md` (the float32 model), `CHANGELOG.md`, `docs/decisions-log.md`.

## Test vectors (`tests/golden/sdrk_golden/cases/squelch.py`)

Common inputs and quantization as in M06 (`noise(seed, n, sigma)`, complex64 for C32, `q15.from_float` for CQ15).

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| squelch | `bursts_c32_24000` | 410 | 48 kHz; noise σ 0.001 (−60 dBFS) with complex tone bursts at −20 dBFS (samples 4000–9999 and 15000–17999) and at −41 dBFS (20000–21999, between the close and the open level); threshold −40 dB, hysteresis 3 dB, τ 0.002 s | float32 model of §1 | bit-exact |
| squelch | `bursts_cq15_24000` | 411 | same signal quantized | integer model of §1 | bit-exact |
| squelch | `retune_c32_8192` | 412 | noise σ 0.01 (−40 dBFS); threshold −30 dB for samples 0–4095 and −45 dB from sample 4096 (the case lists the change; the Rust test calls `set_thresholds` at that sample) | float32 model | bit-exact |

All Rust golden tests run the input once in one piece and once in random chunks (ChunkPlan seed = case seed) and compare both against the file.

## Tests

1. Golden tests for every case (host), whole and chunked.
2. Invariance test for `squelch` (20 plans per type).
3. The gate opens on the first sample where `p ≥ P_open`, does not close while `P_close ≤ p < P_open`, and the closed output is exactly zero.
4. `set_thresholds` applies from the next sample.
5. Extremes: no panic on silence, full scale, NaN and ±∞ (C32), and the Q15 corners; NaN and ±∞ leave `p` unchanged.
6. Block: description rules of M09; construction in an exact arena; `threshold_db` and `hysteresis_db` change behaviour from the next call; no block struct larger than 512 bytes.
7. Pipeline: `nfm_receiver` validates, builds and runs on the E2E-NFM signal with the same thresholds; schema check.
8. Counting: `squelch` appears as `info` in the results of all four platforms.
9. E2E-NFM meets its checks; the suite passes on the host and in the emulators of M15 (`cargo xtask emu --all`, quick profile).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32`; `cargo xtask emu --target qemu-cortex-m4` (quick profile within its timeout).

## Deliverables checklist

- [ ] Kernel `squelch` (C32, CQ15) with state, fallback, reference, delegation
- [ ] Float32 model with self-test in the golden generator
- [ ] Block `squelch`, pipeline kind, regenerated schema
- [ ] Flow `nfm_receiver` with YAML twin
- [ ] SIMD decision documented, `info` entries in `benches/limits.json` and the results
- [ ] Golden cases (seeds 410–412), suite, invariance, extremes, E2E-NFM
- [ ] `docs/kernels.md`, `docs/timing.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, `tests/golden/README.md`, CHANGELOG, decisions log
