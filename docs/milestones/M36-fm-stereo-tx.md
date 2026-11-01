# M36 — FM stereo transmit: pre-emphasis with limiter and stereo multiplex

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M30](M30-two-port-blocks.md) · Next: [M37](M37-packet-ports.md)

## Goal

A broadcast FM stereo transmitter as a flow: the stereo variant of `preemph_limit` (M25's pre-emphasis and peak limiter, with one gain linked for both channels) and a kernel and block `fm_stereo_mod` (stereo audio → multiplex signal with 19 kHz pilot and 38 kHz DSB subcarrier). With the existing `resample`, `fm_mod` and `duc` this gives the flow `fm_stereo_transmitter`, and a loop test proves that the receiver `fm_stereo_demod` (M24) decodes it with the expected channel separation.

## Why

The receive side of FM stereo exists since M24; the transmit side was missing. The multiplex signal is also the carrier for RDS (M42), which adds its 57 kHz subcarrier to it.

## Order

M36 depends on M30 (the oscillator tables of `oscillate`, used for the equality test of §2). It can run in parallel with M31–M35.
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

**In:** the stereo variant of the kernel `preemph_limit` of M25 (C32, linked gain) and the kernel `stereo_encode` (C32 → f32) in `sdrk-core` with Fallback, Reference, facade delegation, contracts; golden cases; the C32 variant of the block `preemph_limit` (the kind chooses it by the input type); block `FmStereoModBlock` and kind `fm_stereo_mod`; schema; the flow `examples/flows/fm_stereo_transmitter.json` with a YAML twin; loop tests; `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/backends.md`, `examples/README.md`, `CHANGELOG.md`.

**Out:** CQ15 variants of both kernels (backlog: the multiplex needs headroom that Q15 makes awkward), an RDS input (M42 adds a second port to `fm_stereo_mod`), a multiband audio processor, a composite clipper (backlog).

## Specification

General rules of M19 apply (exact order, no FMA). Both kernels are bit-exact against a float32 model; `preemph_limit` has a feedback (the limiter gain) and follows the float32-model rule of `squelch` with an f64 regression guard.

### 1. `preemph_limit` for stereo (C32)

M25 §2 specifies `preemph_limit` for one real channel (`PreemphLimitState<T: RealSample>`). A stereo signal needs the **same gain on both channels**: two independent limiters would reduce the louder channel alone and move the stereo image. The stereo variant reuses everything of M25 §2 and changes only the envelope:

```rust
pub struct PreemphLimitStereoState { /* coefficients, x1 and v1 per channel, envelope, hold counter, limit, release, hold */ }
impl PreemphLimitStereoState {
    pub fn new(coefs: [f64; 3], limit: f64, hold_s: f64, release_s: f64, rate: Rate) -> Result<Self>;  // as M25
    pub fn set_limit(&mut self, limit: f64) -> Result<()>;
    pub fn gain(&self) -> f64;
    pub fn reset(&mut self);
}
pub trait PreemphLimitStereo { fn preemph_limit_stereo(&self, s: &mut PreemphLimitStereoState, input: &[C32], output: &mut [C32]) -> Result<()>; }
```

Validation and constants as M25 §2. **Per sample** (f32, no FMA):

1. Non-finite components become `0.0` (each channel separately).
2. Pre-emphasis per channel exactly as M25 §2 step 2, with its own `x1`, `v1` (`re` first, then `im`).
3. Envelope: `m = max(|v.re|, |v.im|)`, then hold and release as M25 §2 step 3 (one envelope, one hold counter).
4. Gain: if `env > T` both channels are multiplied by the same factor `T / env` (computed once), then each is clamped to `[−T, T]`; output `(y.re, y.im)`.

With identical channels (`re == im`) the output equals the mono kernel of M25 applied to one channel, bit for bit (test). Timing ratio 1, delay 0.

### 2. `stereo_encode`

```rust
pub struct StereoEncodeState { nco: Nco /* 19 kHz pilot phase */, main: f32, pilot: f32 }
impl StereoEncodeState {
    /// pilot_hz 18000 … 20000 (default 19000), main 0 … 1 (default 0.9), pilot 0 … 0.2 (default 0.09).
    pub fn new(pilot_hz: f64, main: f32, pilot: f32, rate: Rate) -> Result<Self>;   // fs ≥ 120 kHz
    pub fn set_pilot(&mut self, pilot: f32) -> Result<()>;   // 0 turns the pilot off (mono receivers ignore the subcarrier)
    pub fn reset(&mut self);
}
pub trait StereoEncode { fn stereo_encode(&self, s: &mut StereoEncodeState, input: &[C32], output: &mut [f32]) -> Result<()>; }
```

Per sample, with `φ` the NCO phase before the sample, `s1 = sin(φ)` and `s2 = sin(2φ)` from the tables of `rotate` with its interpolation (`s2` at the phase `φ.wrapping_mul(2)`, exactly as `stereo_pll` does): `m = (L + R)·0.5`, `d = (L − R)·0.5`, `mpx = main·m + main·d·s2 + pilot·s1` (products first, then the sums from left to right). The NCO advances one step per sample. At `|L|, |R| ≤ 1` the peak is at most `main + pilot ≤ 1` at the defaults (0.99). Timing ratio 1, delay 0.

The phase convention matches the receiver: `stereo_pll` locks to the pilot `sin(φ)` and demodulates with `sin(2φ)`, so `L` comes out left (tested in §5). Equality check: `stereo_encode` equals the composition `nco` (harmonic 1 and 2, `imag_part`) · `mul` · `add` of the M30 blocks on the same input within 1e-6 absolute (the composition is not part of the deliverable, only of the test).

### 3. Blocks and kinds

| Kind | Structs | In → Out | Parameters (unit, mutability, default) | Cost |
| --- | --- | --- | --- | --- |
| `preemph_limit` | `PreemphLimitBlock<T>` of M25, now also for C32 (stereo) | C32 → C32 (new), f32/Q15 → same (M25) | as M25: `preemphasis_us` (50), `limit` (**runtime**, 1.0), `hold_s` (0.02), `release_s` (0.2) | 12 for C32 (6 for real, M25) |
| `fm_stereo_mod` | `FmStereoModBlock` | C32 → f32 | `pilot_hz` (Hz, build, 19000), `main` (None, build, 0.9), `pilot` (None, **runtime**, 0.09) | 10 |

`fm_stereo_mod` rejects Q15 and CQ15 input with `type_mismatch`; `preemph_limit` rejects CQ15 (backlog). `fm_stereo_mod` below 120 kHz is `rate_unsupported` (the subcarrier reaches 53 kHz). `docs/blocks.md` chain position: `preemph_limit` at the audio rate before the up-sampling `resample`, `fm_stereo_mod` at 192 kHz or more, followed by `fm_mod` with `deviation_hz` 75000.

### 4. Golden cases (`preemph_limit.py`, `stereo_encode.py`)

| Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- |
| `preemph_limit/stereo_limit_c32_16384` | 485 | 48 kHz, τ 50 µs, `limit` 0.9; L: 1 kHz 0.5 plus bursts of 10 kHz 0.9, R: noise σ 0.3 (the limiter attacks on either channel) | float32 model | bit-exact |
| `preemph_limit/stereo_equal_c32_8192` | 486 | L = R = the input of M25's `limit_f32_8192` | the M25 expected output in both channels | bit-exact |
| `stereo_encode/c32_16384` | 487 | 192 kHz, L 1 kHz 0.6, R 3 kHz 0.4, start phase 0x40000000 | float32 model | bit-exact |

The limiter rule of M25 (exact float32 model, log the exception) applies to the stereo case.

### 5. Loop test and flow

- `examples/flows/fm_stereo_transmitter.json` (+ YAML): `audio` C32 at 48 kHz (re left, im right) → `pre` = `preemph_limit` → `up` = `resample` (to 192000) → `mpx` = `fm_stereo_mod` → `mod` = `fm_mod` (75000) → `duc` (interpolation 10, frequency 600000) → `rf` `cs8`, pull mode, `max_chunk` 4800.
- **E2E-STEREO-TX (`e2e_fm_stereo_loop_f32`):** the flow without `duc` (output C32 at 192 kHz) feeds the kind `fm_stereo_demod` (de-emphasis 50 µs) at 192 kHz. Left 1 kHz 0.5, right 3 kHz 0.3 (seed 488 adds noise σ 0.001 to both): the pilot is locked after 0.5 s; in the last second the left output has the 1 kHz tone within 1 dB of the input level and the 3 kHz tone at least **30 dB** below it, the right output the opposite. With `pilot` 0 at runtime, the receiver falls back to mono within 0.5 s.
- Limiter: a full-scale 15 kHz burst in both channels (seed 489) keeps the `fm_stereo_mod` output within ±0.99 and the `fm_mod` frequency deviation within 75 kHz (measured by `quad_demod`).

## Tests

1. Golden tests whole and chunked (20 chunk plans each).
2. Extremes (`extremes_m36`): silence, full scale, NaN/∞; `preemphasis_us` 0 is the identity below the limit; a `limit` change at runtime applies from the next sample; random stereo input up to 20 times full scale never exceeds `limit` in either channel.
3. Linked gain: a loud burst on the left channel only lowers the right channel by the same factor (the ratio `out.im / in_preemphasized.im` equals `out.re / in_preemphasized.re` for every sample while limiting).
4. `stereo_encode` equals the M30 composition within 1e-6 (§2).
5. The loop tests of §5; the flow test (validates, builds, pull output equals push output).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask emu --target qemu-cortex-m4` (quick profile).

## Deliverables checklist

- [ ] Kernels `preemph_limit_stereo` (C32) and `stereo_encode` (C32 → f32) with reference, fallback, delegation, contracts
- [ ] Blocks, kinds, costs, schema
- [ ] Golden cases (seeds 485–487), extremes, equality with M25's mono kernel and with the M30 composition
- [ ] E2E-STEREO-TX and limiter test (seeds 488, 489); flow `fm_stereo_transmitter` with YAML twin
- [ ] `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/backends.md` ("Not accelerated": both are recursions or table lookups), `examples/README.md`, CHANGELOG, decisions log
