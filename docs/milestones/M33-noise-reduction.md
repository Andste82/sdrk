# M33 — Noise reduction for HF: noise blanker and automatic notch

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M26](M26-block-ports.md) · Next: [M34](M34-multi-input.md)

## Goal

Two kernels with blocks, pipeline kinds and an example flow: `noise_blank` removes short impulses (ignition noise, power-line spikes) from the complex signal before the channel filter, and `auto_notch` removes steady tones (carriers, heterodynes) from the audio with an adaptive filter. A new example flow `hf_am_receiver` uses both, and end-to-end tests show the improvement on synthetic interference.

## Why

AM and SSB reception on short wave (M20, M21) is often limited not by noise but by interference: impulses that the channel filter stretches into clicks, and whistles from nearby carriers. A noise blanker must act on the wide-band signal before the filter, where an impulse is short and strong; a notch must follow the audio tone, whose frequency is unknown and can drift. Both are standard features of every HF receiver and were missing from the plan.

## Order

M33 depends on M26 (single-input blocks only). It can run in parallel with M27–M32.

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

**In:** kernels `noise_blank` (C32, CQ15; bit-exact against the float32 and integer models) and `auto_notch` (f32; bit-exact against the float32 model) in `sdrk-core` with states, Fallback, Reference, facade delegation; golden cases; suite cases; blocks `noise_blank` and `auto_notch` with descriptions; pipeline kinds; schema; the flow `examples/flows/hf_am_receiver.json` with a YAML twin; end-to-end tests; `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/backends.md` ("Not accelerated"), `examples/README.md`, `CHANGELOG.md`.

**Out:** spectral noise reduction of audio (backlog), a Q15 `auto_notch` (backlog), changes to the existing flows `am_receiver` and `ssb_receiver` (their outputs stay as they are).

## Specification

General rules of M19 apply: exact arithmetic order, no FMA, Q15 in `i64`, denormal flush (R6.3). The float32-model rule of `squelch` (M19) applies to both kernels: the decisions are discrete or the loop feeds back, so the golden cases are computed by the exact float32 model and the integer model; the f64 reference is a regression guard as in M24 (D111).

### 1. `noise_blank`

```rust
pub struct NoiseBlankState<T: ComplexSample, Hist: AsRef<[T]> + AsMut<[T]>> { /* ring of D samples, power average, counter, coefficients */ }
impl<..> NoiseBlankState<..> {
    /// threshold_db: an impulse is a sample whose power exceeds the average by this much (6 … 40);
    /// tau_s: time constant of the power average (0.001 … 1); hold_s: blanking time after the last
    /// impulse sample (0 … 0.01); history: exactly D = round(lookahead_s·fs) elements (lookahead 0 … 0.001 s).
    pub fn new(threshold_db: f64, tau_s: f64, hold_s: f64, history: Hist, rate: Rate) -> Result<Self>;
    pub fn reset(&mut self);
    pub fn blanked(&self) -> u64;   // count of blanked output samples since construction or reset
}
// Timing: ratio 1, group delay D
pub trait NoiseBlank<T: ComplexSample> { fn noise_blank<H>(&self, s: &mut NoiseBlankState<T, H>, input: &[T], output: &mut [T]) -> Result<()>; }
```

Coefficients (in `new`, f64): `α = 1 − exp(−1 / (fs·tau_s))`, `k = 10^(threshold_db/10)`, `H = round(hold_s·fs)`.

Per sample (f32; `x` the input, `p` the average power, `c` the counter):

1. `q = x.re·x.re + x.im·x.im`.
2. If `q > k·p` and `p > 0`: `c = D + H` (the impulse is blanked from the ring's oldest sample on, so `lookahead` covers its rising edge). Otherwise: `p = flush_denormal(p + α·(q − p))` (the average is not updated by impulse samples). Non-finite `q` counts as an impulse.
3. `y` = the oldest sample of the ring (input `n − D`; zero during the first `D` samples); the ring takes `x`.
4. If `c > 0`: output 0 and `c = c − 1`, else output `y`.

Q15: `q` in Q30 (`i64`), `p` in Q61 with `α` in Q31 as the CQ15 `squelch`; the comparison `q·2^31 > k_q31·p_q30` with `k` as a Q31.16 fixed-point number (`k_q = floor(k·2^16 + 0.5)`, compare `q·2^16 > k_q·(p >> 31)` in `i128`). Record the exact formula in `docs/kernels.md`.

A signal without impulses passes unchanged, only delayed by `D` (bit-exact, tested).

### 2. `auto_notch` (f32)

```rust
pub struct AutoNotchState<W: AsRef<[f32]> + AsMut<[f32]>, X: AsRef<[f32]> + AsMut<[f32]>> { /* weights (L), history (L + Δ), mu */ }
impl<..> AutoNotchState<..> {
    /// taps L (8 … 256), delay Δ (1 … 64), mu (0.0001 … 0.1): normalized LMS line enhancer.
    pub fn new(weights: W, history: X, delay: usize, mu: f32) -> Result<Self>;
    pub fn set_mu(&mut self, mu: f32) -> Result<()>;
    pub fn reset(&mut self);
}
// Timing: ratio 1, group delay 0
pub trait AutoNotch { fn auto_notch<W, X>(&self, s: &mut AutoNotchState<W, X>, input: &[f32], output: &mut [f32]) -> Result<()>; }
```

Per sample: the reference vector `u = (x[n−Δ], x[n−Δ−1], …, x[n−Δ−L+1])`; prediction `ŷ = Σ w[i]·u[i]` (ascending `i`, no FMA); error `e = x[n] − ŷ`; output `e` (the predictable tones are removed, noise and speech pass); power `P = Σ u[i]·u[i]` (ascending); update `w[i] = flush_denormal(w[i] + (mu·e / (P + 1e-6))·u[i])` with the factor computed once. Non-finite input is treated as 0 and gives output 0.

### 3. Golden cases (`noise_blank.py`, `auto_notch.py`)

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| noise_blank | `nb_c32_16384` | 471 | 192 kHz, complex noise σ 0.05 plus a 10 kHz tone 0.3, impulses of amplitude 3 and 4 samples at 50 random positions; threshold 15 dB, tau 0.01, hold 0.0002, lookahead 0.00005 | float32 model | bit-exact |
| noise_blank | `nb_cq15_16384` | 472 | the input above quantized | integer model | bit-exact |
| noise_blank | `nb_clean_c32_4096` | 473 | the noise and tone without impulses | the input delayed by D | bit-exact |
| auto_notch | `an_tone_f32_16384` | 474 | 8 kHz, white noise σ 0.1 plus tones 1000 Hz 0.5 and 1700 Hz 0.3; L 32, Δ 8, mu 0.01 | float32 model | bit-exact |
| auto_notch | `an_sweep_f32_16384` | 475 | as above with the 1000 Hz tone sweeping slowly to 1100 Hz | float32 model | bit-exact |

The f64 regression guard: the generator also runs both loops in f64 and stores the SNR of the model output against it as `snr_db_guard` (lowest value over the builds minus 1 dB).

### 4. Blocks and kinds

| Kind | Struct | In → Out | Parameters (unit, mutability, default) |
| --- | --- | --- | --- |
| `noise_blank` | `NoiseBlankBlock<T>` | C32/CQ15 → same | `threshold_db` (dB, **runtime**, 15; 6 … 40), `tau_s` (s, build, 0.01), `hold_s` (s, build, 0.0002), `lookahead_s` (s, build, 0.00005) |
| `auto_notch` | `AutoNotchRealBlock` | f32 → f32 | `taps` (I64, build, 32), `delay` (I64, build, 8), `mu` (None, **runtime**, 0.01) |

Kinds with costs 8 (`noise_blank`) and `2·taps + 4` (`auto_notch`); `auto_notch` rejects Q15 input with `type_mismatch` (only f32). `docs/blocks.md`: **chain position** — `noise_blank` directly after `convert` (or after a wide first decimation), before the `ddc`; `auto_notch` after the demodulator and before the AGC of the audio, if any; the notch also removes a wanted steady tone (CW, test tones), so it is off unless placed in the flow.

### 5. Flow and end-to-end tests

- `examples/flows/hf_am_receiver.json` (+ YAML): `rf` `cs16` at 192 kHz → `nb` = `noise_blank` → `tune` = `ddc` (decimation 4, cutoff 4500) → `squelch` → `agc` (AM time constants) → `demod` = `am_demod` → `notch` = `auto_notch` → `audio`. The comments explain both blocks and when to remove them.
- **E2E-NB (`e2e_noise_blank_{f32,q15}`):** the AM test signal of M20 at 192 kHz with impulses (amplitude 20 times the carrier, 3 samples, 100 per second, seed 476): the SINAD of the 1 kHz audio with `noise_blank` is at least **10 dB** better than without (record both values); with the impulses removed from the input, the flow with and without the blanker gives the same audio apart from the delay (bit-exact after alignment).
- **E2E-NOTCH (`e2e_auto_notch_f32`):** the AM signal with an interfering carrier 1.5 kHz off the station (amplitude 0.3): the 1.5 kHz whistle in the audio is attenuated by at least **30 dB** after 0.5 s, and the modulation passes: for this test the station is modulated with speech-like band-limited noise (300–3000 Hz, seed 477) instead of the 1 kHz tone, and the audio level in 300–3000 Hz changes by at most **2 dB** with the notch against without it.

## Tests

1. Golden tests whole and chunked; invariance with 20 chunk plans; the f64 guard.
2. Extremes (`extremes_m33`): silence, full scale, NaN/∞, Q15 corners; `noise_blank` with lookahead 0 and hold 0.
3. Blocks: description rules, exact arena, runtime parameters apply from the next sample.
4. The end-to-end tests of §5; flow test for `hf_am_receiver` (validates, builds, runs in push mode).
5. SIMD: both kernels are per-sample recursions and stay with the fallback; listed under "Not accelerated" in `docs/backends.md`; `info` counts against the fallback.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (info entries); `cargo xtask emu --target qemu-cortex-m4` (quick profile).

## Deliverables checklist

- [ ] Kernels `noise_blank` (C32, CQ15) and `auto_notch` (f32) with states, fallback, reference, delegation, contracts
- [ ] Golden cases (seeds 471–475) with guards; suite cases; extremes
- [ ] Blocks, kinds, schema; flow `hf_am_receiver` with YAML twin
- [ ] E2E-NB and E2E-NOTCH (seeds 476, 477)
- [ ] `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/backends.md`, `examples/README.md`, CHANGELOG, decisions log
