# M24 — FM stereo

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M23](M23-real-sampling-input.md) (uses the `biquad` kernels of [M22](M22-audio-filters.md)) · Next: [M25](M25-fm-transmit-audio.md)

## Goal

Stereo reception of FM broadcast: the kernels `stereo_pll` (pilot phase-locked loop and 38 kHz demodulation) and `stereo_matrix` (sum and difference to left and right), the block `fm_stereo`, its pipeline kind, the example flow `fm_stereo_receiver`, stereo WAV output in the `fm-receiver` example, and an end-to-end test that checks channel separation, audio quality, pilot residue and the switching between stereo and mono.

## Why

The `fm_receiver` flow of M08/M09 plays only the sum signal L+R. Almost every broadcast station transmits stereo: the multiplex signal (MPX) at the output of the FM discriminator carries L+R from 30 Hz to 15 kHz, a pilot tone at 19 kHz and the difference signal L−R as a double-sideband signal with suppressed carrier around 38 kHz (23 to 53 kHz). A receiver regenerates the 38 kHz carrier from the pilot with a phase-locked loop, demodulates L−R with it, and forms L and R. Stereo is the feature users notice first when they listen to a station; it also prepares the 57 kHz RDS decoder (backlog), which uses the same multiplex signal and the same pilot phase.

## Order

M24 follows the receive series M19–M23 (see the *Order* section of [M19](M19-nfm-squelch.md)) and uses M00–M15 and M19–M23, in particular the kernels `biquad` and `biquad_real` and the design function `biquad_coefs` of M22. Like the series it uses nothing from M16–M18.

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

**In:** kernels `stereo_pll` and `stereo_matrix` in `sdrk-core` (states, `Fallback`, `Reference`, facade delegation); the block `fm_stereo` with description; its pipeline kind and the regenerated schema; the convention for stereo audio in the pipeline (§1); the example flow `fm_stereo_receiver` (JSON and YAML); stereo WAV output in the `fm-receiver` example; the testkit generator `fm_multitone_baseband` and the helper `stereo_mpx_tones`; golden cases; suite cases; the end-to-end case `e2e_fm_stereo_*`; the SIMD decision and counts of §8; `docs/kernels.md`, `docs/timing.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`.

**Out:** RDS (57 kHz), soft blending between stereo and mono as a function of the signal-to-noise ratio (§5 switches hard, with hysteresis), the stereo indicator as a pipeline event, noise reduction of the difference signal, a stereo port type in the pipeline (§1 explains why none is needed), FM stereo transmission (stereo encoder), changes to `fm_demod` and to the `fm_receiver` flow. Stereo output in applications outside this repository (the live receiver `sdrk-live` of the separate `sdrk-rtlsdr` repository) is a follow-up there.

## Specification

General rules from M02/M05/M06 apply (see M19). All f32 arithmetic is written without FMA in exactly the order given; Q15 arithmetic is in `i64` with `round_shift` and `sat16` (R7).

**Thresholds and limits are not relaxed.** If a threshold, tolerance or count limit of this milestone is not met, do not change it in code, case files or `benches/limits.json`. Measure it, record the measured value and the reason in `docs/decisions-log.md` and an open item in `docs/open-items.md`, and report it in the pull request; a changed threshold needs the maintainer's approval. The only pre-approved changes are the ones stated in this file.

### 1. Stereo audio in the pipeline

A stereo audio stream is a stream of the complex sample type of the precision: **`re` is the left channel, `im` the right channel** (`C32` for f32, `CQ15` for Q15). No new port type is introduced:

- `C32` and `CQ15` are `#[repr(C)]` with `re` first, so a stream of them is exactly the interleaved two-channel layout (L, R, L, R, …) that sound cards and WAV files use; a reader copies the bytes without reordering.
- Every block that works on complex streams works on stereo audio unchanged and treats both channels alike: the `resampler` (also tunable, for the clock drift of a sound card), `scale`, `fir`, `biquad` (M22). `ddc`, `squelch`, `fm_demod` and `spectrum` accept the type too but make no sense on audio; the pipeline does not prevent such a flow.
- `sdrk-io` already writes `CF32` and `CS16` as two-channel WAV files with I in the left channel (M12), which is the same convention.
- A port type `stereo` would need changes in the pipeline model, the schema, the wire format, the WASM bindings and the I/O crate, for no new capability. Record this choice in `docs/decisions-log.md` and describe the convention in `docs/pipeline.md` and in the description of `fm_stereo`.

### 2. The multiplex signal (reference for tests and docs)

ITU-R BS.450 (pilot-tone system). With the instantaneous frequency deviation `Δf(t) = deviation · m(t)` and left and right audio `L(t)`, `R(t)` (each at most 1):

`m(t) = 0.45·(L + R) + 0.45·(L − R)·sin(2·ω_p·t) + a_p·sin(ω_p·t)`, `ω_p = 2π·19 kHz`, pilot amplitude `a_p` = 0.08 … 0.10 (nominal **0.1** in this milestone).

The 38 kHz subcarrier and the pilot cross zero together with positive slope, so the receiver's subcarrier is `sin(2θ)` when the pilot is `a_p·sin(θ)`. After low-pass filtering to 15 kHz, `S = LPF(m) = 0.45·(L + R)` and `D = LPF(m·sin(2θ)) = 0.225·(L − R)`, hence `L_out = S + 2D = 0.9·L` and `R_out = S − 2D = 0.9·R`: the stereo output has the same level as the mono output of `fm_demod` for the same L+R. A mono transmission has no pilot and `m = 0.9·M` for a mono signal M; the receiver then outputs `L_out = R_out = S`.

### 3. `stereo_pll` — pilot PLL and 38 kHz demodulation (bit-exact in both types)

```rust
pub struct StereoPllState<T: RealSample> { /* phase (u32), integrator, lock level, stereo flag, enabled, coefficients */ }
impl<T: RealSample> StereoPllState<T> {
    /// pilot_hz: nominal pilot frequency (19 000); natural_hz: natural frequency of the loop;
    /// pilot_level: nominal pilot amplitude in units of the input signal (f32: 0.1; Q15: 0.1 times the discriminator scale, §5)
    pub fn new(pilot_hz: f64, natural_hz: f64, pilot_level: f64, rate: Rate) -> Result<Self>;
    pub fn set_enabled(&mut self, enabled: bool);   // false: the difference output is zero; the loop keeps running
    pub fn is_stereo(&self) -> bool;                // the lock decision of the last sample (independent of enabled)
    pub fn phase(&self) -> u32;
    pub fn reset(&mut self);                        // phase 0, integrator 0, level 0, not stereo; enabled is kept
}
// Timing: ratio 1/1, group delay 0
pub trait StereoPll<T: RealSample> {
    /// mpx: the multiplex signal; pilot: the same signal through the pilot band-pass (§5);
    /// output: re = mpx, im = mpx · sin(2θ) while stereo and enabled, else 0. All three slices have equal length.
    fn stereo_pll(&self, s: &mut StereoPllState<T>, mpx: &[T], pilot: &[T], output: &mut [T::Complex]) -> Result<()>;
}
```

`T::Complex` is the complex type of the real sample type (`C32` for `f32`, `CQ15` for `Q15`); add the associated type to `RealSample` if it does not exist yet (log it).

**Coefficients** (computed in `new`, in f64; `fs` the rate):

- `0 < pilot_hz < fs / 4` (so that `2·pilot_hz` stays below Nyquist), `0 < natural_hz ≤ pilot_hz / 100`, `0 < pilot_level ≤ 1`, else `InvalidParameter`.
- Phase-detector gain `kd = pilot_level / 2` (the detector output for a small phase error `φ` is `kd·sin φ`). Damping `ζ = 1/√2`, `ωn = 2π·natural_hz / fs` (radians per sample). Gains in **turns per sample** per unit of detector output: `kp = 2·ζ·ωn / (kd·2π)`, `ki = ωn² / (kd·2π)`.
- `step0 = floor(pilot_hz / fs · 2^32 + 0.5)` as `u32`; integrator limit `imax = 50 / fs` turns per sample (±50 Hz; BS.450 allows ±2 Hz).
- Lock detector: `β = 1 − exp(−1 / (fs · 0.05))` (time constant 50 ms); stereo on at `level ≥ on = 0.4·kd`, off at `level < off = 0.3·kd` (a pilot of 4 % and 3 % of the full deviation at the nominal level 0.1).

**Sine and cosine** come from the cosine tables of `rotate` with its interpolation (`docs/kernels.md`, `rotate`; the C32 fallback rule for f32, the CQ15 rule for Q15): `c = cos(phase)`, `s = sin(phase)` (cosine at `phase − 2^30`), `s2 = sin(2·phase)` with `2·phase` as `phase.wrapping_mul(2)`.

**f32, per sample** (`x = mpx[n]`, `p = pilot[n]`, values from the state; no FMA):

1. `c`, `s`, `s2` at the current phase.
2. If `p` is finite: `e = p·c`, `q = p·s`; otherwise `e = 0`, `q = 0` and the level is not updated in step 3.
3. `level = flush_denormal(level + β·(q − level))`; `stereo = true` if `level ≥ on`, `false` if `level < off`, otherwise unchanged.
4. `output[n] = (x, if stereo && enabled { x·s2 } else { +0.0 })`.
5. `integ = clamp(flush_denormal(integ + ki·e), −imax, imax)`; `f = kp·e + integ` (`kp`, `ki`, `imax`, `β`, `on`, `off` as f32).
6. `phase = phase + step0 + (floor(f as f64 · 2^32 + 0.5) as i64 as u32)`, all additions wrapping.

**Q15, per sample** (raw `i16` values `x`, `p`; tables and interpolation of the CQ15 `rotate`):

1. `c`, `s`, `s2` as Q15 at the current phase.
2. `e = p·c`, `q = p·s` in `i64` (Q30).
3. Level in Q61 with a Q31 coefficient, as the CQ15 `squelch` of M19: `β_q31 = floor(β·2^31 + 0.5)`, `level = level + β_q31·(q − round_shift(level, 31))`; `on_q61 = floor(on·2^61 + 0.5)`, `off_q61` likewise; the comparisons as in f32.
4. `output[n] = (x, if stereo && enabled { sat16(round_shift(x·s2, 15)) } else { 0 })`.
5. Gains in units of `2^−64` turn per sample per Q30 unit: `kp_q = floor(kp·2^34 + 0.5)`, `ki_q = floor(ki·2^34 + 0.5)`, `imax_q = floor(imax·2^64 + 0.5)` (`i64`). `integ = clamp(integ + ki_q·e, −imax_q, imax_q)` (`i64`, units `2^−64` turn), `f = kp_q·e + integ`.
6. `phase = phase + step0 + (round_shift(f, 32) as u32)`, wrapping.
7. `new` returns `InvalidParameter` when `kp_q·2^30 + imax_q` or `ki_q·2^30 + imax_q` does not fit in `i64`, or when `ki_q` or `β_q31` is zero.

**Reference:** the same arithmetic for both types (f32 for `f32`). The lock decision is discrete and the loop feeds back on itself, so an f64 reference would drift away from the fallback; as with `squelch` (M19), the specification fixes the arithmetic and the golden cases are computed by the exact float32 model and the integer model (record this in the decisions log as an exception to the f64 reference rule).

### 4. `stereo_matrix` — sum and difference to left and right (bit-exact in both types)

```rust
pub trait StereoMatrix<T: ComplexSample> {
    /// input: re = S (sum), im = D (difference after demodulation); output: re = S + 2D (left), im = S − 2D (right). Equal lengths.
    fn stereo_matrix(&self, input: &[T], output: &mut [T]) -> Result<()>;
}
```

Stateless (no `Timing`). **C32:** `d2 = d + d`, `l = s + d2`, `r = s − d2` in f32 (exact order). **CQ15:** `d2 = 2·d` in `i64`, `l = sat16(s + d2)`, `r = sat16(s − d2)`. The reference is the same arithmetic. Unequal lengths give `LengthMismatch` and change nothing.

### 5. Block `fm_stereo`

The construction pattern, descriptions, storage and parameter rules of M09 §4–§5 apply unchanged.

| Kind | Struct | In → Out | Parameters (unit, mutability, default) | Implementation |
| --- | --- | --- | --- | --- |
| `fm_stereo` | `FmStereoBlock<T>` | C32/CQ15 → same type, stereo (§1) | `deviation_hz` (Hz, build, 75000; C32 only, as `fm_demod`), `deemphasis_us` (us, build, 50; 0 switches it off), `audio_cutoff_hz` (Hz, build, 15000; 1 … 16000), `audio_decimation` (I64, build, 10; 1 … 64), `stereo` (Choice `auto`/`mono`, **runtime**, `auto`), `pll_natural_hz` (Hz, build, 20; 1 … 190) | see below |

Processing order, in pieces of at most 256 input samples like `fm_demod` (the result must not depend on the piece size; R6.2):

1. `quad_demod` with the gain of `fm_demod` (C32 `fs / (2π·deviation_hz)`, so full deviation reads 1.0; CQ15 raw 32767, unit phase/π) → MPX `m` (real).
2. `biquad_real` band-pass (`biquad_coefs(Bandpass, 19 000, 5, fs)`, `to_f32_sos` or `to_q14`; one section) of `m` → `p`. The RBJ band-pass has zero phase at its centre, so the loop locks to the pilot phase itself. Measured while planning (NumPy model of this chain, see *Notes*): without the band-pass the audio in the detector input leaves a static phase error of about 3° that limits the separation to 46 dB; with it the error is below 0.05° and the separation of the bare decoder is 63 dB.
3. `stereo_pll(m, p)` with `pilot_hz` 19 000, `natural_hz` = `pll_natural_hz`, `pilot_level` 0.1 (C32) or `0.1 · 2·deviation_hz / fs` (CQ15: the discriminator output of a deviation `Δf` is `2·Δf / fs` in units of π); `set_enabled(stereo == auto)`.
4. `fir_decimate` (complex, factor `audio_decimation`) with the low-pass designed like the audio filter of `fm_demod` (`DesignSpec { cutoff_hz: audio_cutoff_hz, transition_hz: audio_cutoff_hz / 4, attenuation_db: 60 }`, at most 1023 taps) → `(S, D)` at the audio rate. Both components pass the same filter, so sum and difference keep the same delay and gain (the separation depends on it). With `audio_decimation` 1 the filter still runs (it removes the pilot and the subcarrier products).
5. `stereo_matrix` → `(L, R)`.
6. De-emphasis of both channels with `biquad` (complex, one section, the one-pole of `deemphasis` as a biquad: `α = 1 − exp(−1 / (fs_audio · τ))`, coefficients `[α, 0, 0, −(1 − α), 0]`); skipped when `deemphasis_us` is 0. The de-emphasis acts on L and R after the matrix; applying it to the MPX before the decoder (as `fm_demod` does) would attenuate the subcarrier and destroy the separation.

Validation (`validate`, `rate_out`): the input rate must be at least 120 kHz (the MPX reaches 53 kHz; the products of step 3 reach 91 kHz and their aliases must stay above the audio band), else `OutOfRange` (the pipeline reports `rate_unsupported`); the output rate `fs / audio_decimation` must exceed `2·audio_cutoff_hz + audio_cutoff_hz / 4` (the rule of `fm_demod`).

`pilot_locked()` (a method of the block, not a parameter) returns `is_stereo()` of the PLL state, for applications that show a stereo indicator. `reset()` resets every state. Timing: ratio `1/audio_decimation`; delay = delay of `quad_demod` + delay of the FIR (the IIR stages count 0, as in `fm_demod`).

The description says, in `details`: the output is stereo audio as complex samples (`re` left, `im` right); the input rate must be 120 kHz or more, and a channel filter narrower than the FM signal (about 2 × (75 + 53) kHz) costs separation (§6, §7); `stereo` `mono` forces mono output, for weak stations whose difference signal is noisy.

### 6. Pipeline kind and flow

- Kind `fm_stereo` (complex input; `kernels()` = `["quad_demod", "biquad_real", "stereo_pll", "fir_decimate", "stereo_matrix", "biquad"]`). `plan` returns `Figures` like `FmDemodKind::plan`; cost per input sample, also added to the cost table of `docs/pipeline.md`: **`25 + (2 * taps + 12) / audio_decimation`** (`taps` of the audio filter).
- `cargo xtask schema` regenerates the schema (version unchanged). Add the kind to the WASM declarations if they list kinds.
- `examples/flows/fm_stereo_receiver.json` with a YAML twin: input `rf` `cu8` at 2.4 MS/s → `tune` = `ddc` (`frequency_hz` 600000, **decimation 5** → 480 kHz, `cutoff_hz` 130000, `transition_hz` 30000) → `demod` = `fm_stereo` (defaults; `audio_decimation` 10 → 48 kHz) → stream output `audio` (`C32`, stereo), plus the `spectrum` branch of `fm_receiver`. The YAML comments explain the stereo convention of §1, the `stereo` parameter, and why the channel is wider than in `fm_receiver`: measured while planning with this chain and the E2E signal of §7, a channel of 240 kHz (decimation 10, cutoff 100 kHz, the `fm_receiver` values) limits the separation to about 29 dB, because the FM signal of a stereo station is about 256 kHz wide and the filter and the aliasing at 240 kHz distort its outer sidebands; 480 kHz with a cutoff of 130 kHz gives 43 dB.
- The `fm-receiver` example writes a two-channel WAV file when the audio output carries complex samples (`CF32`, stereo by §1) and a one-channel file as before for `f32`; `--flow examples/flows/fm_stereo_receiver.json` then produces a stereo file. `examples/README.md` shows the command.

### 7. Testkit

- `fm::fm_multitone_baseband(out: &mut [Complex64], rate_hz, carrier_offset_hz, deviation_hz, tones: &[Tone], start)` with `Tone { amplitude, freq_hz, phase_rad }`: instantaneous frequency `offset + deviation · Σ amplitude·cos(2π·freq·t + phase)`, the phase integrated analytically (`Σ deviation·amplitude / freq · sin(2π·freq·t + phase)`, terms with `freq` 0 are rejected), so chunked generation is exact like `fm_baseband` (M08).
- `fm::stereo_mpx_tones(left: (f64, f64), right: (f64, f64), pilot: f64) -> [Tone; 9]` (`(amplitude, freq_hz)` of a sine tone per channel; amplitude 0 allowed): the MPX of §2 for `L = a_l·sin(2π f_l t)`, `R = a_r·sin(2π f_r t)`, written as cosines (sum terms with phase −π/2, the products `sin·sin` as `½·(cos(Δ) − cos(Σ))` around 38 kHz, the pilot with phase −π/2). A unit test compares the sum of the tones with the direct formula of §2 to 1e-12.
- Suite cases (Fallback vs Reference, invariance with 20 chunk plans): `stereo_pll_f32`, `stereo_pll_q15`, `stereo_matrix_c32`, `stereo_matrix_cq15`, `extremes_m24`; block case `block_fm_stereo_separation` (quick: 0.2 s of the E2E signal at 480 kHz without the RF part, separation ≥ 40 dB).
- **E2E-STEREO (`e2e_fm_stereo_{f32,q15}`):** at 2.4 MS/s, `fm_multitone_baseband` (offset +600 kHz, deviation 75 kHz, `stereo_mpx_tones((1.0, 1000), (1.0, 3000), 0.1)`) scaled to 0.9, plus complex noise σ 0.003, encoded to `CU8` with the testkit encoder → `convert` → the blocks `ddc` (as in the flow) → `fm_stereo` (defaults). Measured over the audio from 0.25 s on (after lock and filter start-up), with `ToneMeter`:
  1. **Separation:** the 1 kHz component in L over the one in R, and the 3 kHz component in R over the one in L: **f32 ≥ 40 dB, Q15 ≥ 35 dB** (initial thresholds, see the M07 rule on thresholds; f32 measured 43.3 and 45.2 dB while planning).
  2. **SINAD per channel**, with both tones removed by a joint least-squares fit: **f32 ≥ 55 dB, Q15 ≥ 45 dB** (f32 measured 67 and 65 dB).
  3. **Level:** the 1 kHz amplitude in L is `0.9 · |H_de(1 kHz)|` within ±0.5 dB (`H_de` the one-pole de-emphasis at 48 kHz); the same for 3 kHz in R.
  4. **Pilot residue:** the 19 kHz component of L and R is at most −70 dBFS (f32) / −60 dBFS (Q15) (f32 measured below −110 dBFS).
  5. **Lock:** `pilot_locked()` becomes true within 0.1 s of the start (measured 28 ms) and stays true.
  Quick variant 0.5 s, full variant 2.0 s.
- **E2E-STEREO switching (`e2e_fm_stereo_switch_f32`, full profile):** the same signal with the pilot present only from 0.3 s to 0.7 s of 1.0 s: stereo within 0.1 s after the pilot appears, mono within 0.2 s after it disappears (measured 44 ms and 60 ms); while mono, L and R are bit-identical.
- **E2E-STEREO mono transmissions (quick):** a mono signal (`L = R` = 1 kHz, no pilot) gives bit-identical L and R and never reports stereo; complex noise σ 0.3 without a carrier for 1.0 s never reports stereo.
- The Cortex-M4 quick profile is close to its timeout (M19 §5): `block_fm_stereo_separation` and the quick E2E cases must fit or move to the full profile (log it).

### 8. SIMD backend and instruction counts

The rules of M13 and of the SIMD optimization apply to every new kernel unchanged (decisions D1–D14; `docs/backends.md`, `docs/performance.md`, `docs/decisions-log.md`). A new kernel gets level paths in `sdrk-backend-simd` for every level the backend supports (x86-64 SSE2, SSE4.2 where it differs, AVX2, AVX-512, AArch64 NEON, wasm32 SIMD128; run-time detection with the feature `std`; a generic `fearless_simd` path as the fallback of every level-specific path, D1) whenever vectorization pays off. A kernel whose work is a per-sample recursion cannot be vectorized without changing its result; it stays with the fallback, is added to the "Not accelerated" list of `docs/backends.md` with the reason, and is counted as `info` against the fallback. For every accelerated kernel:

1. **Contract:** the class of the fallback. Bit-exact kernels are bit-exact with `Fallback` on every level; tolerance kernels meet the SNR of their contract; FMA only where the contract allows it (D5).
2. **Verification:** a `backend_vs_fallback_*` suite case and a property test against the fallback at every level (the CI jobs `test_x64_linux_fixed_arch`, `test_x64_linux_sde_*`, `test_arm64_linux*` and `test_wasm32` run them); a level-specific rounding or saturation path also gets an exhaustive or boundary sweep like `convert_out_sweep`, run by the nightly jobs `sweep_*`.
3. **Counting:** the kernel in `benches/kernels` (`count run`, `count accuracy`), one entry per level in `benches/limits.json` against the fair reference named below, `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32` pass, the files in `benches/results/` are committed and `docs/performance.md` gets the rows. Release limit 1.10×, goal ≤ 1.00× (D14); the accuracy rule applies. An entry that cannot meet its limit yet carries a waiver that names an open item.
4. **Facade:** `backend_for` reports `simd` for the kernel; `RUSTFLAGS="--cfg sdrk_fallback_only"` still restores the fallback.
5. **Embedded backends:** M16–M18 accelerate the kernels listed in their own milestones. If one of them is implemented after this milestone, it decides there whether to include the new kernels.

| Kernel | Accelerated | Reason | Count |
| --- | --- | --- | --- |
| `stereo_pll` (f32, Q15) | no | the loop is a per-sample recursion: the phase of every sample depends on the detector output of the previous one | `info` against the fallback at every level; listed under "Not accelerated" in `docs/backends.md` |
| `stereo_matrix` (C32, CQ15) | yes | element-wise additions (and saturation) on interleaved pairs | against the fallback at every level (`dev`, goal ≤ 0.5× of the fallback); the CQ15 saturation path gets a boundary sweep |

The block's other kernels (`quad_demod`, `biquad_real`, `fir_decimate`, `biquad`) keep the decisions of their milestones. Add an `info` row for the whole `fm_stereo` block at 480 kHz to `docs/performance.md` (instructions per input sample, fallback and best level), next to the `fm_demod` row.

### 9. Documentation

`docs/kernels.md` (contracts of `stereo_pll` and `stereo_matrix`, the multiplex formula of §2, the float32-model exception), `docs/timing.md` (ratios and delays), `docs/backends.md`, `docs/performance.md`, `docs/blocks.md` (generated table, the processing order of §5, the chain position after a wide `ddc`), `docs/pipeline.md` (the stereo convention of §1, the kind, the cost formula), `examples/README.md` (the flow and stereo WAV), `CHANGELOG.md`, `docs/decisions-log.md`.

## Test vectors (`tests/golden/sdrk_golden/cases/stereo.py`)

The generator builds the cosine tables of `rotate` itself (`COS_F32[k] = float32(cos(2πk/4096))` with `scalar_math.cos`, `COS_Q15` by the Q15 rule from the same f64 values) and stores the SHA-256 of both tables in every `stereo_pll` case; a Rust test compares them with the tables of `sdrk-core`, so a table difference fails with a clear message instead of a diverging loop.

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| stereo_pll | `lock_f32_16384` | 436 | 192 kHz, `natural_hz` 50, `pilot_level` 0.1; `mpx` = the MPX of §2 (L 1 kHz, R 3 kHz, amplitude 1, pilot 0.1 at 19 000.5 Hz with start phase 1.0 rad) plus real noise σ 0.01; `pilot` = the pilot term alone plus independent noise σ 0.005 (no band-pass in the case) | float32 model of §3 | bit-exact |
| stereo_pll | `lock_q15_16384` | 437 | same signals scaled by 0.625 and quantized; `pilot_level` 0.0625 | integer model of §3 | bit-exact |
| stereo_pll | `fade_f32_16384` | 438 | as `lock_f32`, the pilot amplitude falls linearly from 0.1 to 0 over samples 8192–12287 and stays 0 | float32 model; the case also stores the stereo flag per sample (`u8`) | bit-exact |
| stereo_pll | `disable_f32_8192` | 439 | as `lock_f32`; `set_enabled(false)` at sample 4096 (the case lists the change) | float32 model | bit-exact |
| stereo_matrix | `c32_4096` | 440 | uniform noise in ±0.6 per component | `s + (d + d)`, `s − (d + d)` in float32 | bit-exact |
| stereo_matrix | `cq15_4096` | 441 | uniform raw values over the whole range (saturates often) | integer model of §4 | bit-exact |

The cases of 16384 samples are about 256 KiB (two real inputs and one complex output); keep each at or below 512 KiB (R8.6). All stateful Rust golden tests run whole and in random chunks (ChunkPlan seed = case seed); `fade_f32` also compares `is_stereo()` after every chunk with the stored flag.

## Tests

1. Golden tests for every case, whole and chunked; the table hash check; invariance tests for both kernels and the block (20 plans each).
2. PLL behaviour (f32 and Q15): with a clean pilot at 19 000 Hz the phase error `(2π·19 000·n/fs − 2π·phase/2^32)` settles below 0.5° within 0.1 s at 480 kHz; with the pilot at 19 002 Hz it settles too (the integrator takes the offset); with no pilot and noise σ 0.3 the integrator stays within `±imax` and `is_stereo()` stays false.
3. `set_enabled(false)` zeroes the difference output from the next sample and keeps the loop running: `set_enabled(true)` after 0.1 s restores stereo output from the next sample without a new lock time.
4. Validation of `new` (pilot at or above `fs/4`, `natural_hz` 0 or above `pilot_hz / 100`, `pilot_level` 0, NaN, a Q15 gain that does not fit) and of the block (input rate below 120 kHz, `audio_cutoff_hz` above 16 000, an output rate too low for the cutoff).
5. Block: the `stereo` parameter switches between `auto` and `mono` from the next call; in `mono` L and R are bit-identical; `pilot_locked()` follows the PLL; description rules of M09; construction in an exact arena; no block struct larger than 512 bytes.
6. Extremes (`extremes_m24`): no panic on silence, full scale, NaN and ±∞ (f32 MPX and pilot inputs; a non-finite pilot sample leaves the loop and the level unchanged), and the Q15 corners of both kernels.
7. SIMD: `backend_vs_fallback_stereo_matrix` is bit-exact at every level; the counts pass their limits.
8. Pipeline: `fm_stereo_receiver` validates, builds and runs on the E2E-STEREO signal with the same thresholds (pipeline output equals the hand-wired blocks bit for bit); a `resampler` (tunable) after `fm_stereo` works on the stereo stream; schema check; the `fm-receiver` example writes a two-channel WAV file for the stereo flow (system test with a short generated recording) and still a one-channel file for `fm_receiver`.
9. E2E-STEREO and its switching and mono checks pass; the suite passes on the host and in the emulators (quick profile).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32`; `cargo xtask emu --target qemu-cortex-m4` (quick profile within its timeout).

## Deliverables checklist

- [ ] Kernels `stereo_pll` (f32, Q15) and `stereo_matrix` (C32, CQ15) with states, fallback, reference, delegation
- [ ] Block `fm_stereo` with `pilot_locked()`, pipeline kind, regenerated schema
- [ ] Stereo convention documented; flow `fm_stereo_receiver` with YAML twin; stereo WAV in the `fm-receiver` example
- [ ] Testkit `fm_multitone_baseband`, `stereo_mpx_tones`
- [ ] SIMD path for `stereo_matrix`, `info` for `stereo_pll`, counts and limits, block row in `docs/performance.md`
- [ ] Golden cases (seeds 436–441) with table hashes, suite, invariance, extremes, E2E-STEREO (separation, SINAD, level, pilot residue, lock, switching, mono)
- [ ] `docs/kernels.md`, `docs/timing.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, CHANGELOG, decisions log

## Notes

The figures marked "measured while planning" come from a NumPy model of the chain (2.4 MS/s CU8 input, the Kaiser `ddc` filter, `quad_demod`, the band-pass, the PLL of §3 in f64, the complex audio low-pass with decimation, the matrix and the de-emphasis), not from SDRk code. They justify the thresholds and the channel width of the flow; the implementation measures its own values and reports them in the pull request.

| Chain | Separation L→R / R→L | SINAD L / R (σ 0.003) |
| --- | --- | --- |
| Decoder alone at 240 or 480 kHz, without the pilot band-pass | 46 / 66 dB | — |
| Decoder alone, with the band-pass (Q 5) | 63 / 62 dB | — |
| `ddc` 240 kHz, cutoff 100 kHz (the `fm_receiver` values) | 29 / 30 dB | 47 / 45 dB |
| `ddc` 240 kHz, cutoff 110 kHz | 31 / 31 dB | 52 / 49 dB |
| `ddc` 480 kHz, cutoff 130 kHz (the flow of §6) | 43 / 45 dB | 67 / 65 dB |
| `ddc` 480 kHz, cutoff 150 kHz | 44 / 47 dB | 69 / 67 dB |
