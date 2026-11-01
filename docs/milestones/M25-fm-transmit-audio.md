# M25 — FM transmit audio (pre-emphasis and limiter)

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M24.1](M24.1-block-naming.md) (block names; uses the `biquad` block of [M22](M22-audio-filters.md)) · Next: [BACKLOG](BACKLOG.md)

## Goal

Transmit audio that a standard FM receiver reproduces correctly and that never exceeds the allowed deviation: the kernel `preemph_limit` (pre-emphasis and peak limiter in one pass), the block `preemph_limit` (it wraps the kernel and takes its name, rule 8 of the naming scheme of M24.1), its pipeline kind, the updated flow `fm_transmitter`, the new flow `nfm_transmitter`, de-emphasis in the `nfm_receiver` flow, and loop tests transmitter flow → receiver flow that check the frequency response and the peak deviation.

## Why

The transmit chain of M08 (`fm_mod`, `duc`, `convert_out`, flow `fm_transmitter`) modulates the audio as it is. Every FM receiver applies a de-emphasis (50 µs in Europe, 75 µs in North America for broadcast; about 6 dB per octave from 300 Hz for two-way radio), so audio sent without the matching pre-emphasis arrives dull: at 10 kHz it is about 10 dB too low with 50 µs. And nothing limits the deviation: loud audio, or the pre-emphasis boosting the treble, pushes the deviation past 75 kHz (5 kHz for NFM), which widens the signal beyond its channel. A transmitter needs both before it goes on the air. The first user is a HackRF One (driver in the separate repository [`sdrk-hardware`](https://github.com/Andste82/sdrk-hardware)).

## Order

M25 follows M24.1 and uses M00–M15, M19–M24 and M24.1 (the `biquad` block of M22 for the voice band of the NFM transmitter; the kind names, the naming scheme and the reserved names of M24.1: the block of this milestone is `preemph_limit` and the resampler kind is `resample`). Like the receive series it uses nothing from M16–M18.

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

**In:** kernel `preemph_limit` in `sdrk-core` (state, `Fallback`, `Reference`, facade delegation); the design function `preemphasis_coefs` in `sdrk-blocks`; the block `preemph_limit` with description; its pipeline kind and the regenerated schema; the node `shape` in `fm_transmitter`; the flow `nfm_transmitter`; de-emphasis in `nfm_receiver`; golden cases; suite cases; the loop tests of §6; `docs/kernels.md`, `docs/timing.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`.

**Out:** loudness processing (compressor, multiband processing, AGC on the transmit side), look-ahead limiting, the FM stereo encoder `fm_stereo_mod` (pilot and 38 kHz subcarrier; planned after M25), RDS, CTCSS tone generation, drivers for transmitters and the live audio input (both in `sdrk-hardware`).

## Specification

General rules from M02/M05/M06 apply (see M19). All f32 arithmetic is written without FMA in exactly the order given; Q15 arithmetic is in `i64` (and `i128` where stated) with `round_shift` and `sat16` (R7).

**Thresholds and limits are not relaxed.** If a threshold, tolerance or count limit of this milestone is not met, do not change it in code, case files or `benches/limits.json`. Measure it, record the measured value and the reason in `docs/decisions-log.md` and an open item in `docs/open-items.md`, and report it in the pull request; a changed threshold needs the maintainer's approval. The only pre-approved changes are the ones stated in this file.

### 1. Pre-emphasis design (`sdrk-blocks`, `design.rs`, in f64)

```rust
/// First-order pre-emphasis for the de-emphasis time constant tau_s: [b0, b1, a1] of
/// H(z) = (b0 + b1 z⁻¹) / (1 + a1 z⁻¹), unity gain at DC.
pub fn preemphasis_coefs(tau_s: f64, fs_hz: f64) -> Result<[f64; 3]>;
```

The ideal pre-emphasis `1 + sτ` has no upper limit; the design adds a pole at `f_p = fs / 2.4` (20 kHz at 48 kHz) and maps `H(s) = (1 + sτ) / (1 + sτ_p)`, `τ_p = 1 / (2π f_p)`, by the bilinear transform without prewarping: with `K = 2·fs`, `b0 = (1 + Kτ) / (1 + Kτ_p)`, `b1 = (1 − Kτ) / (1 + Kτ_p)`, `a1 = (1 − Kτ_p) / (1 + Kτ_p)`. `tau_s` in `(0, 1e-3]` and `fs_hz ≥ 40 000`, else `OutOfRange`.

Measured while planning (SciPy): the magnitude of this design differs from `|1 + j2πfτ|` by at most 0.2 dB from 30 Hz to `min(15 kHz, 0.45·fs)` for `fs` 44.1, 48, 96 and 240 kHz and τ 50 and 75 µs; at 48 kHz with τ 50 µs the product with the de-emphasis of the `fm_receiver` flow (`deemphasis` at 240 kHz) is flat within ±0.05 dB up to 15 kHz. At 32 kHz the error reaches 1 dB, hence the lower limit. The gain at Nyquist is `τ / τ_p` (16 dB for 50 µs at 48 kHz, 36 dB for 530 µs at 48 kHz, 56 dB for 1000 µs at 240 kHz); the kernel therefore keeps the pre-emphasized value in a wide format until the limiter (§2).

### 2. `preemph_limit` — pre-emphasis and peak limiter (bit-exact in both types)

```rust
pub struct PreemphLimitState<T: RealSample> { /* coefficients, x1, v1, envelope, hold counter, limit, release, hold */ }
impl<T: RealSample> PreemphLimitState<T> {
    /// coefs: from preemphasis_coefs, or [1, 0, 0] for no pre-emphasis; limit: output peak in (0, 1];
    /// hold_s ≥ 0; release_s > 0
    pub fn new(coefs: [f64; 3], limit: f64, hold_s: f64, release_s: f64, rate: Rate) -> Result<Self>;
    pub fn set_limit(&mut self, limit: f64) -> Result<()>;   // from the next sample; envelope and hold are kept
    pub fn gain(&self) -> f64;                              // current gain reduction (1.0 when the limiter is idle), for meters
    pub fn reset(&mut self);                                // x1 = v1 = 0, envelope 0, hold 0
}
// Timing: ratio 1/1, group delay 0 (IIR and a gain without look-ahead)
pub trait PreemphLimit<T: RealSample> {
    fn preemph_limit(&self, s: &mut PreemphLimitState<T>, input: &[T], output: &mut [T]) -> Result<()>;
}
```

`H = round(hold_s · fs)` samples, `r = exp(−1 / (fs · release_s))` (f64), `T` = `limit`. Validation: `limit` in `(0, 1]`, `release_s > 0`, `hold_s ≥ 0`, `H ≤ 2^31`, every coefficient finite with `|c| < 2^14`, else `InvalidParameter`.

**f32, per sample** (`b0`, `b1`, `a1`, `r`, `T` as f32; no FMA):

1. `x` = the input sample, or `0.0` when it is NaN or infinite (a transmitter never passes a non-finite value to the modulator).
2. Pre-emphasis: `v = (b0·x + b1·x1) − a1·v1`; then `x1 = x`, `v1 = flush_denormal(v)`.
3. Envelope with hold and release: `m = |v|`; if `m ≥ env` then `env = m`, `hold = H`; else if `hold > 0` then `hold −= 1`; else `env = flush_denormal(env · r)`.
4. Gain: `y = if env > T { v · (T / env) } else { v }`; then `y = clamp(y, −T, T)`; output `y`.

The attack is instant (the envelope jumps to every new peak), so no sample exceeds `T`; the hold keeps the gain constant over a period of the lowest audio frequencies (20 ms covers 50 Hz) and avoids distortion from a gain that follows the waveform; the release recovers the gain after a loud passage.

**Q15, per sample** (raw `i16` `x`):

1. Coefficients in Q24: `c_q = floor(c · 2^24 + 0.5)` (`i64`). The pre-emphasized value `v` is kept in Q30 in `i64` (it may exceed 1).
2. `v = round_shift((b0_q·x + b1_q·x1) · 2^15 − a1_q·v1, 24)` with every product and sum in `i128` (with the largest allowed coefficients the sum exceeds `i64`), the result as `i64`; then `x1 = x`, `v1 = v`.
3. Envelope in Q30: `m = |v|`; hold as in f32; the release `env = round_shift(env · r_q31, 31)` with `r_q31 = floor(r · 2^31 + 0.5)`.
4. `T_q30 = floor(T · 2^30 + 0.5)`. If `env > T_q30`: `y = div_round(v · T_q30, env)` with the product in `i128` and `div_round(n, d) = sign(n) · floor((|n| + floor(d / 2)) / d)` for `d > 0`; else `y = v`. Output `sat16(round_shift(y, 15))` clamped to `±T_q15`, `T_q15 = min(floor(T · 32768 + 0.5), 32767)`.

**Reference:** the same arithmetic for both types. The limiter compares and switches, so as for `squelch` (M19) the golden cases are computed by the exact float32 model and the integer model (log it as an exception to the f64 reference rule).

### 3. Block `preemph_limit`

The construction pattern, descriptions, storage and parameter rules of M09 §4–§5 apply unchanged. Names follow the scheme of M24.1 (`docs/blocks.md`, section "Names"): module `blocks/preemph_limit.rs`, block `PreemphLimitBlock<T>`, pipeline kind type `PreemphLimitKind`; the name check of `cargo xtask check` covers it; remove its line from the reserved-names table.

| Kind | Struct | In → Out | Parameters (unit, mutability, default) | Implementation |
| --- | --- | --- | --- | --- |
| `preemph_limit` | `PreemphLimitBlock<T>` | f32/Q15 → same | `preemphasis_us` (us, build, 50; 0 … 1000, 0 switches it off), `limit` (None, **runtime**, 1.0; 0.1 … 1.0), `hold_s` (Seconds, build, 0.02; 0 … 1), `release_s` (Seconds, build, 0.2; 0.001 … 10) | `preemphasis_coefs` (or `[1, 0, 0]`), `preemph_limit`; `set_param("limit")` calls `set_limit` |

`limit` is the output peak as a fraction of full scale, which `fm_mod` maps to its `deviation_hz`: `limit` 1.0 keeps the peak deviation at `deviation_hz`. `rate_out` equals `rate_in`; a pre-emphasis at an input rate below 40 kHz is `OutOfRange`.

`docs/blocks.md`: the block belongs **directly before `fm_mod`**, at the rate of the modulator. Any filter or resampler between the limiter and the modulator can raise the peaks again: measured while planning, limiting at 48 kHz and then resampling to 240 kHz (the `fm_transmitter` chain) let overdriven noise reach 1.41 times the limit, while limiting at 240 kHz kept it at exactly 1.0 with the distortion of an overdriven 1 kHz tone at −52 dB (100 Hz: −46 dB). When a channel filter must follow the limiter (the voice band of NFM, §4), set `limit` below 1 to leave room for its overshoot.

### 4. Pipeline kind and flows

- Kind `preemph_limit` (real inputs; `kernels()` = `["preemph_limit"]`). `plan` returns `Figures` like `FmDemodKind::plan`; cost per input sample, also added to the cost table of `docs/pipeline.md`: **6**.
- `cargo xtask schema` regenerates the schema (version unchanged). Add the kind to the WASM declarations if they list kinds.
- **`fm_transmitter`** (JSON and YAML): a node `shape` = `preemph_limit` (`preemphasis_us` 50, `limit` 1.0) between `up` (the `resample` node to 240 kHz) and `mod`. The output bytes of the flow change; update the tests and committed files that pin them and log it. The YAML comments explain pre-emphasis (75 µs outside Europe) and the limit.
- **`nfm_transmitter`** (new, JSON and YAML; narrow-band FM for two-way and amateur radio): input `audio` `f32` at 48 kHz → `voice` = `biquad` (lowpass, order 4, `cutoff_hz` 3000) → `shape` = `preemph_limit` (`preemphasis_us` 530, `limit` 0.85) → `splatter` = `biquad` (lowpass, order 4, `cutoff_hz` 3000) → `mod` = `fm_mod` (`deviation_hz` 5000) → `up` = `resample` (complex, `rate_out_hz` 240000) → `duc` (`frequency_hz` 200000, `interpolation` 10) → output `rf` `cs8` (`underrun` `zeros`). 530 µs is the 6 dB per octave pre-emphasis from 300 Hz of ETSI EN 300 086 (1 / (2π · 300 Hz)); TIA-603 radios use 750 µs. The `splatter` filter removes the harmonics the limiter makes above the voice band and overshoots: measured while planning with overdriven noise through this chain, `limit` 0.85 kept the peak below 0.88 (0.9 reached 0.97). The comments explain 12.5 kHz channels (`deviation_hz` 2500) and the offset of the `duc` (it keeps the signal away from the carrier leakage of the transmitter at the centre frequency).
- **`nfm_receiver`** (amends M19): `fm_demod` gets `deemphasis_us` 530 (was 0), so that it reproduces radios with pre-emphasis and the `nfm_transmitter`. The f32 output still becomes exactly zero after the squelch closes (the de-emphasis state decays below the denormal flush within about 40 ms). The Q15 `deemphasis` has a dead band (`docs/kernels.md`), so its output can stay at a small constant: amend the silence check of M19/M22 for Q15 to allow the measured dead-band level (assert it at most 16 LSB) and record it in `docs/kernels.md`. The E2E-NFM kernel chain of M19 keeps its own parameters and checks. Log the change.

### 5. Testkit

- Suite cases (Fallback vs Reference, invariance with 20 chunk plans): `preemph_limit_f32`, `preemph_limit_q15`, `extremes_m25`; block case `block_preemph_limit_peak` (random input up to 20 times full scale: no output sample exceeds `limit`).

### 6. Loop tests (transmitter flow → receiver flow)

All run the transmitter flow in pull mode, feed its `cs8` output (as bytes, without a channel) to the receiver flow whose input format is set to `cs8`, and measure the audio with `ToneMeter`. Run them in f32 and Q15 precision.

- **TX-FLAT (`loop_fm_flat_{f32,q15}`):** tones at 100 Hz, 1, 5, 10 and 12 kHz, one after the other (0.3 s each, amplitude 0.1 so the limiter stays idle) through `fm_transmitter` → `fm_receiver` (`tune.frequency_hz` 600000): every level within **±0.5 dB** of the 1 kHz level (f32; Q15 ±1.0 dB). Before this milestone 10 kHz arrived about 10 dB low; the test fails without the `shape` node.
- **TX-PEAK (`loop_fm_peak_{f32,q15}`):** band-limited noise (fourth-order Butterworth to 15 kHz at 48 kHz, σ 3, seed 450) for 1 s through `fm_transmitter`; the instantaneous frequency of the transmitted signal, measured by a `ddc` to 240 kHz and `quad_demod` with gain `fs / (2π · 75 kHz)` and without de-emphasis, never exceeds **1.02** (75 kHz plus 2 % for the filters and the `cs8` quantization; initial threshold, see the M07 rule on thresholds). The same input without `shape` exceeds 1.5 (record the value).
- **NFM (`loop_nfm_{f32,q15}`):** a 1 kHz tone at amplitude 0.3 through `nfm_transmitter` → `nfm_receiver` (`tune.frequency_hz` 200000): SINAD **f32 ≥ 30 dB, Q15 ≥ 25 dB**; with the noise of TX-PEAK as input the peak deviation stays below **1.0 × 5 kHz**.

### 7. SIMD backend and instruction counts

The rules of M13 and of the SIMD optimization apply to every new kernel unchanged (decisions D1–D14; `docs/backends.md`, `docs/performance.md`, `docs/decisions-log.md`). A new kernel gets level paths in `sdrk-backend-simd` for every level the backend supports (x86-64 SSE2, SSE4.2 where it differs, AVX2, AVX-512, AArch64 NEON, wasm32 SIMD128; run-time detection with the feature `std`; a generic `fearless_simd` path as the fallback of every level-specific path, D1) whenever vectorization pays off. A kernel whose work is a per-sample recursion cannot be vectorized without changing its result; it stays with the fallback, is added to the "Not accelerated" list of `docs/backends.md` with the reason, and is counted as `info` against the fallback. For every accelerated kernel:

1. **Contract:** the class of the fallback. Bit-exact kernels are bit-exact with `Fallback` on every level; tolerance kernels meet the SNR of their contract; FMA only where the contract allows it (D5).
2. **Verification:** a `backend_vs_fallback_*` suite case and a property test against the fallback at every level (the CI jobs `test_x64_linux_fixed_arch`, `test_x64_linux_sde_*`, `test_arm64_linux*` and `test_wasm32` run them); a level-specific rounding or saturation path also gets an exhaustive or boundary sweep like `convert_out_sweep`, run by the nightly jobs `sweep_*`.
3. **Counting:** the kernel in `benches/kernels` (`count run`, `count accuracy`), one entry per level in `benches/limits.json` against the fair reference named below, `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32` pass, the files in `benches/results/` are committed and `docs/performance.md` gets the rows. Release limit 1.10×, goal ≤ 1.00× (D14); the accuracy rule applies. An entry that cannot meet its limit yet carries a waiver that names an open item.
4. **Facade:** `backend_for` reports `simd` for the kernel; `RUSTFLAGS="--cfg sdrk_fallback_only"` still restores the fallback.
5. **Embedded backends:** M16–M18 accelerate the kernels listed in their own milestones. If one of them is implemented after this milestone, it decides there whether to include the new kernels.

| Kernel | Accelerated | Reason | Count |
| --- | --- | --- | --- |
| `preemph_limit` (f32, Q15) | no | both the pre-emphasis (one pole) and the envelope are per-sample recursions, and the gain of a sample depends on the envelope | `info` against the fallback at every level; listed under "Not accelerated" in `docs/backends.md` |

### 8. Documentation

`docs/kernels.md` (contract of `preemph_limit`, the design of §1, the Q15 dead band of `deemphasis` in the NFM receiver), `docs/timing.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md` (generated table, the position before `fm_mod`), `docs/pipeline.md` (kind, cost), `examples/README.md` (the flows; `fm-transmitter --flow examples/flows/nfm_transmitter.json`), `CHANGELOG.md`, `docs/decisions-log.md`.

## Test vectors (`tests/golden/sdrk_golden/cases/preemph_limit.py`, `design.py`)

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| design | `preemphasis_{50us_48k,75us_48k,530us_48k,50us_240k}` | 0 | formulas of §1; the generator also asserts that the result equals `scipy.signal.bilinear([τ, 1], [τ_p, 1], fs)` within 1e-12 | formulas in f64 | max_abs_err ≤ 1e-14 |
| preemph_limit | `preemph_f32_8192` | 444 | 48 kHz, τ 50 µs, `limit` 1.0, hold 0.02 s, release 0.2 s; noise σ 0.05 (limiter idle) | float32 model of §2 | bit-exact |
| preemph_limit | `limit_f32_8192` | 445 | as above with `limit` 0.5; tone bursts at 1 and 8 kHz with amplitude 1.5 over noise σ 0.05 (the limiter attacks, holds and releases) | float32 model | bit-exact |
| preemph_limit | `limit_q15_8192` | 446 | the input of `limit_f32` quantized | integer model of §2 | bit-exact |
| preemph_limit | `preemph_q15_8192` | 447 | 48 kHz, τ 530 µs (gain up to 36 dB inside the kernel), `limit` 1.0; noise σ 0.2 | integer model | bit-exact |
| preemph_limit | `nonfinite_f32_4096` | 448 | the input of `preemph_f32` with NaN, +∞ and −∞ at 20 positions | float32 model (non-finite counts as 0) | bit-exact |
| preemph_limit | `retune_f32_8192` | 449 | as `limit_f32`; `limit` 1.0 for samples 0–4095 and 0.5 from 4096 (the case lists the change) | float32 model | bit-exact |

Seed 450 is the noise of the loop tests (§6). All stateful Rust golden tests run whole and in random chunks (ChunkPlan seed = case seed).

## Tests

1. Golden tests for every case, whole and chunked; invariance tests for the kernel and the block (20 plans each).
2. Design: the magnitude of `preemphasis_coefs` differs from `|1 + j2πfτ|` by at most 0.2 dB from 30 Hz to `min(15 kHz, 0.45·fs)` for `fs` 44.1, 48, 96 and 240 kHz and τ 50, 75 and 530 µs; DC gain 1 within 1e-12; the rejections of §1.
3. Limiter: no output sample exceeds `limit` (property test with random input up to 20 times full scale, both types); a tone below the limit passes unchanged except for the pre-emphasis; after a burst the gain stays constant for `hold_s` and then recovers with `release_s` (time constant within one sample); `set_limit` applies from the next sample.
4. Extremes (`extremes_m25`): no panic on silence, full scale, NaN and ±∞ (f32), the Q15 corners, and the largest pre-emphasis gain (1000 µs at 240 kHz) with a full-scale square wave in Q15 (the `i128` product must not overflow).
5. Block: description rules of M09; construction in an exact arena; `limit` changes behaviour from the next call; a pre-emphasis below 40 kHz is rejected; no block struct larger than 512 bytes.
6. Pipeline: the flows `fm_transmitter` and `nfm_transmitter` validate, build and run in push and pull mode; the amended `nfm_receiver` passes the M19 flow test with the amended Q15 silence check; schema check.
7. The loop tests of §6 pass; the suite passes on the host and in the emulators (quick profile).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32`; `cargo xtask emu --target qemu-cortex-m4` (quick profile within its timeout).

## Deliverables checklist

- [ ] Kernel `preemph_limit` (f32, Q15) with state, fallback, reference, delegation; design function `preemphasis_coefs`
- [ ] Block `preemph_limit`, pipeline kind, regenerated schema
- [ ] `fm_transmitter` with the `shape` node, new `nfm_transmitter`, `nfm_receiver` with de-emphasis (all with YAML twins)
- [ ] SIMD decision (not accelerated), `info` entries and results
- [ ] Golden cases (seeds 444–449 and the design cases), suite, invariance, extremes, loop tests TX-FLAT, TX-PEAK and NFM
- [ ] `docs/kernels.md`, `docs/timing.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, CHANGELOG, decisions log
