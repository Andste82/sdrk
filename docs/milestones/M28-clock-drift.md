# M28 — Clock drift between devices

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M25.1](M25.1-release-0.1.md) · Next: [M29](M29-real-decimation.md)

## Goal

A controller that keeps the buffer between two clock domains (an SDR and a sound card, a file player and a transmitter) at a set fill level by steering the `ppm` parameter of the tunable `resample` block: `DriftController` in `sdrk-blocks` (heap-free, usable with a `Chain` on a microcontroller), a helper for the pipeline, a simulation in the testkit and a long-run test with ±100 ppm drift that never overruns or underruns.

## Why

Every live receiver has two quartz clocks: the SDR delivers samples at its rate, the sound card consumes audio at its own. 50 ppm of difference is 2.4 samples per second at 48 kHz, so a buffer of 100 ms runs over or dry within about half an hour, and a smaller one within minutes. The tunable `resample` block of M09 already accepts `ppm` at run time (plus or minus 1000 ppm, continuous phase); what is missing is the loop that measures the buffer and sets `ppm`, with defaults that do not make the pitch audibly wobble. The same loop serves the transmit side (audio from a sound card into the HackRF clock).

## Order

M28 depends on M25.1 and uses only the existing `resample` block and the pipeline API (`available`, `set_param`). It can run in parallel with M26, M27, M29 and later milestones.

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

**In:** `DriftController` and `DriftConfig` in `sdrk-blocks` (module `drift.rs`); `sdrk_pipeline::drift::PipelineDrift` (a helper that reads `available` of an output and sets `ppm` of a node); the testkit simulation `drift::TwoClockSim`; tests; the YAML comment of `examples/flows/resample_audio.yaml`; `docs/blocks.md` (section "Clock drift"), `docs/pipeline.md`, `CHANGELOG.md`.

**Out:** changes to the `resample` block or its kernels, a new pipeline kind, automatic insertion of the controller (the application decides where the buffer is), drift between two inputs of one pipeline (multi-input with two clock domains is in the backlog).

## Specification

### 1. Model

The buffer sits between a *producer* (the `resample` node, at its output rate `fs` corrected by `ppm`) and a *consumer* (the sound card, at its own clock). The application calls `update` at roughly regular moments (for example in every audio callback) with the current fill level in samples. The error is measured in seconds, `e = (fill − target) / fs`. A positive `ppm` makes the resampler produce more output per input (M09 §Resampler: `step = round(2^32 · rate_in / rate_out / (1 + ppm·1e-6))`), so a buffer that fills needs a *lower* `ppm`.

### 2. `DriftController` (`sdrk-blocks`, no heap, f64 arithmetic)

```rust
pub struct DriftConfig {
    pub rate_hz: f64,          // output rate of the resample node (fs), > 0
    pub target: f64,           // target fill level in samples, > 0
    pub time_constant_s: f64,  // loop time constant τ, default 30.0, 1 … 600
    pub max_ppm: f64,          // limit of the output, default 500.0, ≤ 1000 (the range of `resample`)
    pub slew_ppm_per_s: f64,   // largest change of the output per second, default 20.0, > 0
    pub smoothing_s: f64,      // time constant of the fill measurement filter, default 1.0, > 0
}
pub struct DriftController { /* config, smoothed error, integrator, output ppm, last time, started */ }
impl DriftController {
    pub fn new(config: DriftConfig) -> Result<Self>;          // InvalidParameter outside the ranges
    /// `now_s`: monotonic time of the observation in seconds; `fill`: samples in the buffer.
    /// Returns the ppm value to set. The first call only initialises the state and returns `initial_ppm`.
    pub fn update(&mut self, now_s: f64, fill: f64) -> f64;
    pub fn ppm(&self) -> f64;
    pub fn set_initial_ppm(&mut self, ppm: f64);              // a known offset (e.g. a measured device error), clamped to max_ppm
    pub fn reset(&mut self);                                  // forget the history; keeps config and initial ppm
}
```

Per call, with `dt = now_s − last` (calls with `dt ≤ 0` change nothing and return the current output):

1. `e = (fill − target) / rate_hz`.
2. Smoothing: `a = 1 − exp(−dt / smoothing_s)`, `es = es + a·(e − es)`.
3. Gains (critically damped second-order loop, `ωn = 1 / time_constant_s`; the plant is `de/dt = 1e-6·ppm + drift`): `kp = 2·ωn / 1e-6`, `ki = ωn² / 1e-6`.
4. Integrator: `integ = integ + ki·es·dt`, clamped to `±max_ppm` (anti-windup).
5. Target value: `u = initial_ppm − (kp·es + integ)`, clamped to `±max_ppm`.
6. Slew limit: the output moves towards `u` by at most `slew_ppm_per_s·dt`.
7. Store and return the output.

The arithmetic is f64 throughout (`libm::exp`), so the controller gives the same results on every platform. Document the formulas in `docs/blocks.md`.

### 3. Pipeline helper (`sdrk-pipeline`, module `drift.rs`)

```rust
pub struct PipelineDrift { /* controller, output name, node name */ }
impl PipelineDrift {
    /// `output`: the stream output whose queue is the buffer; `node`: a `resample` node with
    /// `tunable = true` on the path to that output. Checks both (InvalidValue otherwise).
    pub fn new(pipeline: &Pipeline, output: &str, node: &str, config: DriftConfig) -> Result<Self>;
    /// Reads `available(output)` plus `extra_fill` (samples the application holds outside the
    /// pipeline, e.g. in the sound card buffer), runs the controller and sets `ppm` on the node.
    pub fn update(&mut self, pipeline: &mut Pipeline, now_s: f64, extra_fill: usize) -> Result<f64>;
}
```

`update` does not allocate (the parameter value is set through the non-allocating path of `set_param`; add one if `set_param` with a `serde_json::Value` allocates, and log it).

### 4. Testkit simulation (`sdrk-testkit`, module `drift.rs`, feature `std` not required)

`TwoClockSim::new(rate_hz, drift_ppm, buffer_capacity, consumer_block, jitter_seed)` simulates a producer whose output rate is `rate_hz · (1 + (drift_ppm + ppm_set)·1e-6)` and a consumer that takes `consumer_block` samples every `consumer_block / rate_hz` seconds with a uniformly distributed jitter of ±20 % of that period (PCG32 with `jitter_seed`). `step()` advances to the next consumer call and returns `(now_s, fill)`; `set_ppm(p)` applies the controller output. It reports `overruns` and `underruns` (fill above capacity or below 0). It works on fill counts only, no samples, so two hours simulate in well under a second.

### 5. Example

The YAML twin of `examples/flows/resample_audio.yaml` gets a comment block on clock drift (set `tunable: true`, then drive `ppm` with `PipelineDrift`), and `examples/README.md` one paragraph with a code snippet.

## Tests

1. **Convergence (TwoClockSim, 48 kHz, capacity 9600, target 4800, block 480, defaults):** for drifts −100, −20, 0, +20 and +100 ppm (seed 462): no overrun or underrun over 2 h of simulated time; after `5·τ` the fill stays within `target ± 0.002·fs` (2 ms) and the output stays within `−drift ± 1` ppm; the largest change of the output between two calls is at most `slew·dt` + 1e-9.
2. **Step:** a drift that jumps from 0 to +100 ppm after 10 min: no overrun, re-settled within `5·τ`.
3. **Limits:** a drift of 2000 ppm (beyond `max_ppm`) saturates at `−max_ppm` and the integrator does not wind up: when the drift returns to 0 the output leaves the limit within `2·τ`.
4. **Edge cases:** `dt ≤ 0` changes nothing; invalid configurations are `InvalidParameter`; `reset` restarts from `initial_ppm`.
5. **Pipeline:** the flow `resample_audio` with `tunable = true`, input pushed in blocks of 480 at `48000·(1 + 50e-6)` per second of simulated time, output read in blocks of 441 at 44.1 kHz, `PipelineDrift` every read: no output overflow, no read finds the queue empty after the first second, over 10 min of simulated time (the audio itself is not checked); `update` does not allocate (counting allocator).
6. `cargo xtask bare-metal` builds `sdrk-blocks` with `drift.rs`.

## Acceptance

R1.2 commands; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline`; `cargo test -p sdrk-testkit`.

## Deliverables checklist

- [ ] `DriftController`, `DriftConfig` with doc example; formulas in `docs/blocks.md`
- [ ] `PipelineDrift`, non-allocating `update`
- [ ] `TwoClockSim` in the testkit; tests 1–6
- [ ] Example comment and README paragraph; `docs/pipeline.md`, CHANGELOG, decisions log
