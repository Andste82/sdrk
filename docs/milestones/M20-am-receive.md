# M20 — AM receive

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M19](M19-nfm-squelch.md) · Next: [M21](M21-ssb-receive.md)

## Goal

A complete AM receiver (medium and short wave broadcast, airband): the kernels `mag` (envelope), `dc_block_real` and `agc`, the blocks `agc` and `am_demod`, their pipeline kinds, the example flows `am_receiver` (broadcast) and `airband_receiver` (with the squelch of M19), the testkit generator `am_tone` and an end-to-end AM test in f32 and Q15.

## Why

AM is the second most common analog mode and the first one whose audio level follows the signal strength, so it needs an automatic gain control. The AGC is the hardest part of the receive series (a feedback loop with a fixed-point gain in Q15); putting it into the feature whose other parts are simple (envelope and DC removal) keeps the risk in one place. SSB (M21) then reuses the AGC and needs no new kernel.

## Order

Part of the receive series M19–M23 (see the *Order* section of [M19](M19-nfm-squelch.md)): uses M00–M15 and M19.

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

**In:** kernels `mag`, `dc_block_real` and `agc` in `sdrk-core` (states, `Fallback`, `Reference`, facade delegation) and `sdrk_core::num::isqrt_u64`; blocks `agc` and `am_demod` with descriptions; their pipeline kinds and the regenerated schema; the example flows `am_receiver` and `airband_receiver` (JSON and YAML); the generator `am_tone`; golden cases; suite cases; the end-to-end case `e2e_am_rx_*`; the SIMD paths and instruction counts of §7; `docs/kernels.md`, `docs/timing.md`, `docs/blocks.md`, `docs/pipeline.md`.

**Out:** synchronous AM, AM modulation (transmit), noise blanker, audio shaping filters (M22).

## Specification

General rules from M02/M05/M06 apply (see M19). All f32 arithmetic is written without FMA in exactly the order given; Q15 arithmetic is in `i64` (and `i128` where stated) with `round_shift` and `sat16` (R7).

**Thresholds and limits are not relaxed.** If a threshold, tolerance or count limit of this milestone is not met, do not change it in code, case files or `benches/limits.json`. Measure it, record the measured value and the reason in `docs/decisions-log.md` and an open item in `docs/open-items.md`, and report it in the pull request; a changed threshold needs the maintainer's approval. The only pre-approved changes are the ones stated in this file.

### 1. `mag` — envelope (C32 tolerance ≥ 130 dB, CQ15 bit-exact)

```rust
pub trait Mag<T: ComplexSample> { fn mag(&self, input: &[T], output: &mut [T::Real]) -> Result<()>; }
```

- C32: `libm::sqrtf(re·re + im·im)` (no FMA). Inputs above about 1.8e19 overflow to `+∞`; NaN gives NaN (document).
- CQ15: `s = re² + im²` of the raw values in `i64` (≤ 2^31), `r = isqrt_u64(s)` (exact floor of the square root), `y = r + (s − r² > r ? 1 : 0)` (round to nearest; a tie is impossible for integers), output `sat16(y)`. Full-scale corners (`|x| > 1`) saturate, as in `mag_sq`.
- Reference: C32 `sqrt` in f64 of the f64 components; CQ15 the integer rule (Python: `math.isqrt`).

`pub fn isqrt_u64(x: u64) -> u64` in `sdrk_core::num`: the exact floor of the square root (any correct integer method; tested against `u128` squaring at the boundaries `r² − 1`, `r²`, `(r + 1)² − 1` and over random values).

### 2. `dc_block_real` (f32 tolerance ≥ 100 dB, Q15 bit-exact)

The real variant of `dc_block` (M06 §5) with identical rules: `DcBlockRealState<T: RealSample>`, `new(r)` with `0 < r < 1`, `y[n] = x[n] − x[n−1] + r·y[n−1]`; f32 with `flush_denormal` of `y_prev`; Q15 `v = (x − x_prev) as i64 + round_shift(r · y_prev, 15)`, `y = sat16(v)`. Timing ratio 1/1, delay 0. The Q15 dead band of M06 test 6 applies and is documented.

### 3. `agc` — automatic gain control (C32 tolerance, CQ15 bit-exact)

```rust
pub struct AgcState<T: ComplexSample> { /* gain, limits, reference, attack and release constants */ }
impl<T: ComplexSample> AgcState<T> {
    /// reference: output amplitude (linear, 2^-5 ≤ reference ≤ 1); attack_s, release_s: time constants;
    /// gains in dB: −40 ≤ min ≤ initial ≤ max ≤ 90
    pub fn new(reference: f64, attack_s: f64, release_s: f64, min_gain_db: f64, max_gain_db: f64, initial_gain_db: f64, rate: Rate) -> Result<Self>;
    /// Same parameters except the initial gain; the current gain is kept and clamped to the new range.
    pub fn set_params(&mut self, reference: f64, attack_s: f64, release_s: f64, min_gain_db: f64, max_gain_db: f64, rate: Rate) -> Result<()>;
    pub fn gain_db(&self) -> f64;
    pub fn reset(&mut self);   // gain back to the initial gain
}
// Timing: ratio 1/1, group delay 0
pub trait Agc<T: ComplexSample> { fn agc(&self, s: &mut AgcState<T>, input: &[T], output: &mut [T]) -> Result<()>; }
```

Rates per sample (f64): `r_a = 1 − exp(−1 / (fs·attack_s))`, `r_r = 1 − exp(−1 / (fs·release_s))`; each must be in `(0, 0.5]` (a time constant of at least `1 / (fs·ln 2)`), else `OutOfRange`. The gain update is multiplicative, `g ← g·(1 + (r / reference)·e)` with the error `e = reference − |y|`; near the set point this is a first-order loop with the time constant `1 / r` samples, independent of the input level (attack when the output is too loud, release when it is too quiet).

- **C32**, constants as f32 (cast from f64): `ref`, `k_a = r_a / ref`, `k_r = r_r / ref`, `g`, `g_min = 10^(min/20)`, `g_max = 10^(max/20)`. Per sample, in this order: `y = (g·x.re, g·x.im)` (written to the output); `m = sqrtf(y.re·y.re + y.im·y.im)`; if `m` is NaN the gain is not updated; otherwise `e = ref − m`, `k = if e < 0 { k_a } else { k_r }`, `g = g · (1 + k·e)`, then `g = g.max(g_min).min(g_max)` (an infinite input drives `g` to `g_min`).
- **CQ15**, gain `g: i64` in Q.32 (value = raw / 2^32; `g_max` ≤ 31 623 · 2^32 < 2^47): `ref_raw = floor(reference · 32768 + 0.5)` (≥ 1024), `k_raw = floor(r · 2^48 / ref_raw + 0.5)` (≤ 2^37), limits `floor(10^(db/20) · 2^32 + 0.5)`. Per sample: `y.c = sat16(round_shift(x.c · g, 32))` per component; `s = y.re² + y.im²`; `m` = the rounded square root of §1 **without** saturation (≤ 46 341); `e = ref_raw − m`; `k_raw = e < 0 ? k_a : k_r`; `delta = (g · e · k_raw + 2^47) >> 48` computed in `i128` (bounded by 2^100); `g = clamp(g + delta, g_min, g_max)`.
- **Reference:** C32 the same algorithm in f64 with the f64 values of the f32 constants; CQ15 the integer model.
- **Pre-approved fallback for the C32 class:** in a feedback loop the f32 and f64 gains may drift apart. If the C32 golden case `steps_c32_24000` stays below 80 dB although the fallback follows §3 exactly, make the C32 `agc` bit-exact like `squelch` (M19): the `Reference` uses the same f32 arithmetic, the golden case uses the float32 model of M19 with class `bit_exact`, and a separate Rust test keeps the f64 model as a check with the SNR that was measured (recorded in the decisions log).

### 4. Blocks

The construction pattern, descriptions, storage and parameter rules of M09 §4–§5 apply unchanged.

| Kind | Struct | In → Out | Parameters (unit, mutability, default) | Implementation |
| --- | --- | --- | --- | --- |
| `agc` | `AgcBlock<T>` | C32/CQ15 → same | `reference_db` (dB, runtime, −6; −30 … 0), `attack_s` (Seconds, runtime, 0.005), `release_s` (Seconds, runtime, 0.2), `min_gain_db` (dB, runtime, −20), `max_gain_db` (dB, runtime, 60), `initial_gain_db` (dB, build, 0) | `agc`; a runtime change calls `set_params` (the current gain is kept) |
| `am_demod` | `AmDemodBlock<T>` | C32 → f32, CQ15 → Q15 | `dc_cutoff_hz` (Hz, build, 30; 0 = off), `audio_cutoff_hz` (Hz, build, 5000), `audio_decimation` (I64, build, 1 … 64, default 1) | `mag` → `dc_block_real` with `r = exp(−2π·dc_cutoff/fs)` → `resample_real` (L = 1, M = `audio_decimation`; audio low-pass designed exactly as in `fm_demod`, M09 §6) |

`docs/blocks.md`, chain order: `ddc → squelch → agc → am_demod`. AM needs a slow AGC (time constants well above the audio period, e.g. attack 0.1 s, release 0.5 s), otherwise the AGC follows the modulation and distorts it; the defaults suit SSB (M21).

### 5. Pipeline kinds and example flows

- Kinds `agc` (`kernels()` = `["agc"]`) and `am_demod` (`["mag", "dc_block_real", "resample_real"]`). `plan` returns `Figures` exactly like `FmDemodKind::plan` (`crates/pipeline/sdrk-pipeline/src/kinds.rs`): `rate_out` and `memory_bytes` from the block's `rate_out` and `storage_bytes`, `delay` from its timing, `min_input` from `min_input()`, and `cost` per input sample by this formula, which is also added to the cost table of `docs/pipeline.md`: `agc` **6**, `am_demod` **`4 + taps / audio_decimation`** (`taps` of the audio filter, 0 without it).
- `cargo xtask schema` regenerates the schema (version unchanged). Add the kinds to the WASM declarations if they list kinds.
- `examples/flows/am_receiver.json` + YAML: input `cu8` at 2.4 MS/s → `ddc` (decimation 50 → 48 kHz, `cutoff_hz` 5000, `transition_hz` 2000) → `agc` (attack 0.1 s, release 0.5 s) → `am_demod` (defaults) → output `f32`.
- `examples/flows/airband_receiver.json` + YAML: the same with `squelch` (M19, −40 dB) between `ddc` and `agc`; the comments explain the 8.33 kHz and 25 kHz channel spacing.

### 6. Testkit

- Generator (module `r#gen`, no allocation): `am_tone(out: &mut [C32], rate: Rate, audio_hz: f64, index: f64, carrier: f64, phase: f64, start: u64)` = `carrier·(1 + index·cos(2π·audio_hz·n/fs))·e^{j·phase}` for `n = start + i`, computed in f64 and rounded once to f32.
- Suite cases (Fallback vs Reference, invariance for the stateful kernels): `mag_c32`, `mag_cq15`, `dc_block_real_*`, `agc_c32`, `agc_cq15`, `extremes_m20` (full); block case `block_agc_level`.
- **E2E-AM (`e2e_am_rx_{f32,q15}`):** at 2.4 MS/s, `am_tone` (1 kHz, index 0.5, carrier 0.5) shifted to +300 kHz, plus complex noise σ 0.01, encoded to `CU8` with the testkit encoder (as E2E-RX of M08) → `convert` → `ddc` (frequency 300 kHz, decimation 50, taps as in the flow) → `agc` (attack 0.1 s, release 0.5 s) → `am_demod` (defaults). SINAD at 1 kHz over the last 50 % of the audio: **f32 ≥ 35 dB, Q15 ≥ 30 dB** (initial thresholds, M07 rule). The audio level after the AGC is within 1 dB of `reference · index` (the modulation depth at the AGC reference). Quick variant 0.2 s, full variant 1.0 s.
- Keep the quick profile of every emulator target within its timeout (see M19 §5).

### 7. SIMD backend and instruction counts

The rules of M13 and of the SIMD optimization apply to every new kernel unchanged (decisions D1–D14; `docs/backends.md`, `docs/performance.md`, `docs/decisions-log.md`). A new kernel gets level paths in `sdrk-backend-simd` for every level the backend supports (x86-64 SSE2, SSE4.2 where it differs, AVX2, AVX-512, AArch64 NEON, wasm32 SIMD128; run-time detection with the feature `std`; a generic `fearless_simd` path as the fallback of every level-specific path, D1) whenever vectorization pays off. A kernel whose work is a per-sample recursion cannot be vectorized without changing its result; it stays with the fallback, is added to the "Not accelerated" list of `docs/backends.md` with the reason, and is counted as `info` against the fallback. For every accelerated kernel:

1. **Contract:** the class of the fallback. Bit-exact kernels are bit-exact with `Fallback` on every level; tolerance kernels meet the SNR of their contract; FMA only where the contract allows it (D5).
2. **Verification:** a `backend_vs_fallback_*` suite case and a property test against the fallback at every level (the CI jobs `test_x64_linux_fixed_arch`, `test_x64_linux_sde_*`, `test_arm64_linux*` and `test_wasm32` run them); a level-specific rounding or saturation path also gets an exhaustive or boundary sweep like `convert_out_sweep`, run by the nightly jobs `sweep_*`.
3. **Counting:** the kernel in `benches/kernels` (`count run`, `count accuracy`), one entry per level in `benches/limits.json` against the fair reference named below, `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32` pass, the files in `benches/results/` are committed and `docs/performance.md` gets the rows. Release limit 1.10×, goal ≤ 1.00× (D14); the accuracy rule applies. An entry that cannot meet its limit yet carries a waiver that names an open item.
4. **Facade:** `backend_for` reports `simd` for the kernel; `RUSTFLAGS="--cfg sdrk_fallback_only"` still restores the fallback.
5. **Embedded backends:** M16–M18 accelerate the kernels listed in their own milestones. If one of them is implemented after this milestone, it decides there whether to include the new kernels.

| Kernel | Accelerated | Reason | Count |
| --- | --- | --- | --- |
| `mag` C32 | yes | independent per sample; `sqrt` exists on every level | against `volk_32fc_magnitude_32f`, `release` at every level; at `simd128` like `mag_sq` (D12; extending D12 to `mag` goes into the decisions log) |
| `mag` CQ15 | yes | independent per sample; the rounded integer square root on vectors (for example an f64 or f32 square root of the exact sum followed by an integer correction step) must equal §1 for every input: exhaustive sweep over all 2^32 raw pairs in the nightly jobs `sweep_*` | against `volk_16ic_magnitude_16i`, first `dev` (VOLK rounds differently; the accuracy rule compares the SNR of both) |
| `agc` (C32, CQ15) | no | per-sample feedback: the gain of every sample depends on the output of the previous one | `info` against the fallback |
| `dc_block_real` (f32, Q15) | no | first-order recursion, as `dc_block` (M13) | `info` against the fallback |

`am_demod` profits from the accelerated `mag`; `resample_real` stays with the fallback as in M13.

### 8. Documentation

`docs/kernels.md` (contracts of `mag`, `dc_block_real`, `agc`, the Q15 gain format), `docs/timing.md`, `docs/backends.md` (`mag` paths; `agc` and `dc_block_real` not accelerated), `docs/performance.md`, `docs/blocks.md` (table, chain order, AGC time constants for AM), `docs/pipeline.md`, `examples/README.md`, `CHANGELOG.md`, `docs/decisions-log.md`.

## Test vectors (`tests/golden/sdrk_golden/cases/mag.py`, `dc_block.py`, `agc.py`, `blocks.py`)

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| mag | `c32_4096` | 413 | noise σ 0.5 plus the listed values 0, ±1, ±(1 ± 1j) | `np.abs` in f64 | SNR ≥ 130 |
| mag | `cq15_4096` | 414 | quantized, including the four full-scale corners | integer rule of §1 (`math.isqrt`) | bit-exact |
| dc_block_real | `r0p999_f32_8192` | 415 | real noise σ 0.2 + DC 0.3 | `lfilter([1, −1], [1, −r], x)`, r as f32 | SNR ≥ 100 |
| dc_block_real | `r0p999_q15_8192` | 416 | quantized | Python loop, Q15 rules | bit-exact |
| agc | `steps_c32_24000` | 417 | 48 kHz; complex tone 1 kHz with amplitude 0.01, 0.3, 0.003 (8000 samples each) plus noise σ 1e-4; reference 0.5, attack 0.005 s, release 0.05 s, gains −20 / 60 / initial 0 dB | f64 model of §3 with the f64 values of the f32 constants | SNR ≥ 80 |
| agc | `steps_cq15_24000` | 418 | quantized, same parameters | integer model of §3 (Python `int`; assert the `i64`/`i128` bounds) | bit-exact |
| blocks | `am_demod_c32` | 419 | 48 kHz: `am_tone` (1 kHz, index 0.5, carrier 0.5) + noise σ 1e-3, 16 000 samples; `audio_decimation` 1 | NumPy chain: `abs`, `lfilter` DC blocker, `upfirdn` with the block taps | SNR ≥ 90 |
| blocks | `am_demod_cq15` | 420 | same signal quantized | integer chain with the models of §1, §2 and M06 `resample_real` | bit-exact |

All stateful Rust golden tests run whole and in random chunks (ChunkPlan seed = case seed).

## Tests

1. Golden tests for every case, whole and chunked.
2. Invariance tests for `dc_block_real`, `agc` and both blocks (20 plans each).
3. `isqrt_u64` at the boundaries and over random values; `mag` CQ15 equals `round(sqrt(s))` for all `s` of a sweep over the raw corners.
4. AGC: an input step from −40 dBFS to −10 dBFS (tone) settles to within 1 dB of the reference after `5·attack_s`, a step down within 1 dB after `5·release_s`; the gain never leaves `[min, max]`; a NaN sample (C32) leaves the gain unchanged; `set_params` keeps the gain.
5. Extremes (`extremes_m20`): no panic on silence, full scale, NaN and ±∞ (f32), and the Q15 corners.
6. Blocks: description rules; construction in an exact arena; every runtime parameter of `agc` changes behaviour from the next call; no block struct larger than 512 bytes.
7. Pipeline: both flows validate, build and run on the E2E-AM signal (the airband flow with a burst as in E2E-NFM: audio exactly zero outside the transmission); schema check.
8. SIMD: `backend_vs_fallback_mag` passes at every level; the `mag` CQ15 sweep finds no difference; the counts pass their limits.
9. E2E-AM meets its thresholds; the suite passes on the host and in the emulators (quick profile).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32`; the `mag` sweep (`--ignored`); `cargo xtask emu --target qemu-cortex-m4` (quick profile within its timeout).

## Deliverables checklist

- [ ] Kernels `mag`, `dc_block_real`, `agc` with states, fallback, reference, delegation; `isqrt_u64`
- [ ] `mag` paths in `sdrk-backend-simd` for every level, sweep, counts and limits; `info` entries for `agc` and `dc_block_real`
- [ ] Blocks `agc`, `am_demod`, pipeline kinds, regenerated schema
- [ ] Flows `am_receiver`, `airband_receiver` with YAML twins
- [ ] Generator `am_tone`; golden cases (seeds 413–420), suite, invariance, extremes, E2E-AM
- [ ] `docs/kernels.md`, `docs/timing.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, CHANGELOG, decisions log
