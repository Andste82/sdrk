# M06 — Receive kernels

> Plan: [PLAN.md › Phase 2](../PLAN.md#phase-2--mvp-kernels-both-directions-large) · Depends on: [M05](M05-first-kernels.md) · Next: [M07](M07-fft.md)

## Status

**Fully completed:** `fir_decimate`, `resample`, `resample_real`, `quad_demod`, `deemphasis`, `dc_block`, `window`, `mag_sq` and `power_db` with traits, states, `Fallback`, `Reference` and facade delegation for `C32`/`CQ15` (and `f32`/`Q15` where real); `sdrk_core::window` (`hann`, `blackman_harris`, `kaiser`, `bessel_i0`, `window_gain`); all 23 golden cases of the milestone table with Rust golden tests (whole and chunked runs); 21 suite cases; invariance tests (20 chunk plans per stateful kernel), Progress tests, extreme-input tests and Fallback-vs-Reference property tests; `docs/kernels.md`, `docs/timing.md` and the changelog. All R1.2 commands, `python tests/golden/generate.py --check`, `cargo deny check` and `cargo test -p sdrk-testkit --features std` pass.

**Partial / open:** nothing in the scope of this milestone. Spec readings and deviations are recorded in [`decisions-log.md`](../decisions-log.md) (M06 entries).

## Goal

Implement the remaining receive-side kernels of the MVP: decimating FIR, rational resampler, quadrature (FM) demodulator, de-emphasis, DC blocker, windowing with window generators, magnitude squared and power in dB — each with fallback, reference, facade delegation, golden cases, suite cases, invariance and extreme-input tests.

## Why

Together with M05 these kernels form a complete FM receive chain: convert → rotate → fir_decimate → quad_demod → deemphasis → resample, plus the building blocks of a spectrum display (window, FFT in M07, mag_sq, power_db).

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

**In:** kernels `fir_decimate`, `resample`, `quad_demod`, `deemphasis`, `dc_block`, `window`, `mag_sq`, `power_db`; module `sdrk_core::window` with window generators; states; golden cases; suite cases; `docs/kernels.md` entries.

**Out:** FFT (M07), transmit kernels and fused kernels (M08), filter *design* (M09 — kernels here receive taps from the caller).

## Specification

General rules from M02/M05 apply: one trait per kernel, `Fallback` + `Reference` + facade delegation, no allocation, validation at entry, `Timing` implemented by every state. Rate-changing kernels return `Progress` and consume input only as far as output space allows; unconsumed input must be resubmitted by the caller.

### 1. `fir_decimate` (C32 tolerance ≥ 110 dB, CQ15 bit-exact)

```rust
pub struct FirDecimState<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> { /* taps, hist (len N−1), factor M ≥ 1, countdown */ }
// new(taps, history, factor) -> Result; reset(); factor()
// Timing: ratio 1/M, group delay Delay::new(N−1, 2) (input samples)
pub trait FirDecimate<T: ComplexSample> {
    fn fir_decimate<Taps, Hist>(&self, s: &mut FirDecimState<T, Taps, Hist>, input: &[T], output: &mut [T]) -> Result<Progress>;
}
```
Definition: with `y_full[n] = Σ h[k]·x[n−k]` over the whole stream (zero history at start/reset), the outputs are `y[m] = y_full[m·M]` — i.e. the **first** input sample after a reset produces the first output. Accumulation rules as in `fir` (f32 ascending, no FMA; CQ15 `Acc64`). The fallback computes only the outputs it emits (polyphase not required, but no computation of discarded outputs).

### 2. `resample` — rational L/M polyphase (C32 tolerance ≥ 110 dB, CQ15 bit-exact)

```rust
pub struct ResampleState<T, Taps, Hist> { /* taps h (designed at the upsampled rate, gain L included by the caller), up L ≥ 1, down M ≥ 1, history of ceil(N/L) input samples, output counter j */ }
// new(taps, history, up, down) -> Result (history.len() must equal ceil(N / L)); reset()
// Timing: ratio L/M, group delay Delay::new(N − 1, 2·L)
pub trait Resample<T: ComplexSample> {
    fn resample<Taps, Hist>(&self, s: &mut ResampleState<T, Taps, Hist>, input: &[T], output: &mut [T]) -> Result<Progress>;
}
```
Definition = `scipy.signal.upfirdn(h, x, up=L, down=M)` restricted to outputs that depend only on received input: output `y[j] = Σ_k h[k] · u[j·M − k]` with `u[i] = x[i / L]` if `i mod L == 0` else 0. Output `j` is produced as soon as input sample `floor(j·M / L)` has been received. Polyphase evaluation: `t = j·M`, `p = t mod L`, `b = t div L`, `y[j] = Σ_{i ≥ 0, p + i·L < N} h[p + i·L] · x[b − i]` (history supplies negative indices; zeros before the first sample).
Also add the real-valued variant `ResampleReal<T: RealSample>` with the same definition (needed for audio in M08/M09).

### 3. `quad_demod` — FM discriminator (tolerance: f32 ≥ 100 dB, Q15 ≥ 70 dB)

```rust
pub struct QuadDemodState<T: ComplexSample> { prev: T }   // reset → prev = 0
// Timing: ratio 1/1, group delay Delay::new(1, 2)
pub trait QuadDemod<T: ComplexSample> {
    /// output[n] = gain · arg(x[n] · conj(x[n−1]))   (f32 output in radians · gain)
    /// Q15 output: gain_q15 · arg(…)/π              (Q15 unit = phase / π)
    fn quad_demod(&self, s: &mut QuadDemodState<T>, input: &[T], output: &mut [T::Real], gain: T::Real) -> Result<()>;
}
```
- `arg(0) = 0` (`atan2(0, 0) = 0`).
- C32 fallback: product in f32 (no FMA), `libm::atan2f(im, re) · gain`.
- CQ15 fallback: exact product in i64 (`re = ar·br + ai·bi`, `im = ai·br − ar·bi`), `phi = libm::atan2(im as f64, re as f64) / π`, `q = from_f64_round(phi)` (R7 from f64), then `q15_mul(q, gain)`. Q15 fallbacks may use f64 libm internally for tolerance kernels (allowed, deterministic); accelerated variants may use CORDIC later.
- Reference: everything in f64.
- Typical gain for FM: `fs / (2π · deviation)` (f32) so that full deviation maps to ±1.

### 4. `deemphasis` — single-pole low-pass on real audio (f32 tolerance ≥ 100 dB, Q15 bit-exact)

```rust
pub struct DeemphasisState<T: RealSample> { alpha: T, y: T }
// DeemphasisState::new(tau_s: f64, rate: Rate) -> Result: alpha = 1 − exp(−1 / (fs · tau)) computed in f64, then cast (f32) or R7-rounded (Q15); tau ≤ 0 → InvalidParameter
// Timing: ratio 1/1, group delay 0 (IIR; document that its delay is frequency-dependent and not part of exact event timing)
pub trait Deemphasis<T: RealSample> {
    /// y[n] = y[n−1] + α · (x[n] − y[n−1])
    fn deemphasis(&self, s: &mut DeemphasisState<T>, input: &[T], output: &mut [T]) -> Result<()>;
}
```
- f32: compute exactly as written (no FMA), then `y = flush_denormal(y)`.
- Q15: `d = x − y` in i64; `y = sat16(y + round_shift(α · d, 15))`.

### 5. `dc_block` (f32 tolerance ≥ 100 dB, Q15 bit-exact), complex

```rust
pub struct DcBlockState<T: ComplexSample> { r: T::Real, x_prev: T, y_prev: T }  // new(r) with 0 < r < 1
// Timing: ratio 1/1, group delay 0
pub trait DcBlock<T: ComplexSample> {
    /// y[n] = x[n] − x[n−1] + r · y[n−1]   (per component)
    fn dc_block(&self, s: &mut DcBlockState<T>, input: &[T], output: &mut [T]) -> Result<()>;
}
```
f32: as written, flush `y_prev`. Q15 per component: `v = (x − x_prev) as i64 + round_shift(r · y_prev, 15)`; `y = sat16(v)`.

### 6. Window generators and `window` kernel (bit-exact)

`sdrk_core::window` (f64 math via `libm`, results written into caller slices of `f32` or `Q15`):
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Symmetry { Periodic, Symmetric }
pub fn hann<R: RealSample>(out: &mut [R], sym: Symmetry) -> Result<()>;
pub fn blackman_harris<R: RealSample>(out: &mut [R], sym: Symmetry) -> Result<()>;   // 4-term: a0=0.35875 a1=0.48829 a2=0.14128 a3=0.01168
pub fn kaiser<R: RealSample>(out: &mut [R], beta: f64, sym: Symmetry) -> Result<()>;
pub fn bessel_i0(x: f64) -> f64;   // power series Σ ((x/2)^k / k!)^2, stop when the term < 1e-17 · sum, max 500 terms
pub fn window_gain(w: &[f32]) -> (f64, f64); // (coherent gain Σw/N, noise-equivalent bandwidth N Σw² / (Σw)²)
```
Definitions: Symmetric uses denominator `N − 1`; Periodic computes the symmetric window of length `N + 1` and drops the last value (matches `scipy.signal.get_window(..., fftbins=True)`). Hann `0.5 − 0.5 cos(2πn/D)`. Blackman-Harris `a0 − a1 cos(2πn/D) + a2 cos(4πn/D) − a3 cos(6πn/D)`. Kaiser `I0(β·sqrt(1 − (2n/D − 1)²)) / I0(β)`.

```rust
pub trait Window<T: ComplexSample> {
    /// output[n] = input[n] · w[n]; w.len() must equal input.len()
    fn window(&self, input: &[T], output: &mut [T], w: &[T::Real]) -> Result<()>;
}
```
C32: f32 multiply per component (bit-exact with the reference, which computes in f64 and casts: the product of two f32 values rounded once is identical). CQ15: `q15_mul`.

### 7. `mag_sq` (f32 tolerance ≥ 140 dB, Q15 bit-exact)

```rust
pub trait MagSq<T: ComplexSample> { fn mag_sq(&self, input: &[T], output: &mut [T::Real]) -> Result<()>; }
```
f32: `re·re + im·im` (no FMA). Q15: `sat16(round_shift(re² + im², 15))` in i64 (note: full-scale values saturate; documented).

### 8. `power_db` (tolerance: max_abs_err ≤ 1e-4 dB)

```rust
pub trait PowerDb<T: RealSample> {
    /// output[n] = 10·log10(max(p, 1e-20)) + offset_db, p = input as linear power (Q15: raw/32768)
    fn power_db(&self, input: &[T], output: &mut [f32], offset_db: f32) -> Result<()>;
}
```
Uses `libm::log10f` on f32 power (reference: `libm::log10` on f64). Negative or NaN input → treated as 1e-20 (document; never panic).

### 9. Facade, suite, docs

`delegate!` entries for all new traits (C32/CQ15 and f32/Q15 variants). Suite cases (Fallback vs Reference with testkit generators, plus invariance for stateful kernels): `fir_decimate_*`, `resample_*` (including L/M = 1/5, 3/2, 147/160), `quad_demod_*`, `deemphasis_*`, `dc_block_*`, `window_*`, `mag_sq_*`, `power_db_*`, `extremes_rx_kernels` (full). Contracts in `docs/kernels.md`; `docs/timing.md`: ratio and delay of each kernel (table).

## Test vectors (Python, add `cases/*.py` per kernel)

Common inputs: `noise(seed, n, sigma)` = complex Gaussian `σ·(randn + j randn)/√2`; quantize to complex64 for C32 cases and with `q15.from_float` for CQ15 cases; compute expected from the quantized input in float64 (or with the integer model for bit-exact cases).

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| fir_decimate | `m10_n63_c32_20000` | 40 | `h = firwin(63, 0.08)`→f32; x noise σ 0.3 | `lfilter(h, 1, x)[::10]` | SNR ≥ 110 |
| fir_decimate | `m10_n63_cq15_20000` | 41 | taps/x via `q15.from_float` | integer model: `np.convolve` int64 per component, `[:len][::10]`, round_shift 15, sat16 | bit-exact |
| fir_decimate | `m3_n31_c32_9000` | 42 | `firwin(31, 0.25)` | as above | SNR ≥ 110 |
| resample | `l1m5_n121_c32_10000` | 43 | `h = 1·firwin(121, 0.18)` (gain L = 1) | `upfirdn(h, x, 1, 5)[:K]`, K = number of outputs computable from the input (formula below) | SNR ≥ 110 |
| resample | `l3m2_n96_c32_6000` | 44 | `h = 3·firwin(96, 1/3·0.9)` | `upfirdn(h, x, 3, 2)[:K]` | SNR ≥ 110 |
| resample | `l147m160_n1470_c32_16000` | 45 | `h = 147·firwin(1470, 0.9/160)` | `upfirdn(h, x, 147, 160)[:K]` | SNR ≥ 100 |
| resample | `l3m2_n96_cq15_6000` | 46 | quantized taps (gain in taps saturating at Q15: scale taps by 1/3 and document the gain loss) | integer polyphase model in Python (loop over outputs, int64) | bit-exact |
| resample_real | `l1m5_n121_f32_10000` | 47 | real noise | `upfirdn` real | SNR ≥ 110 |
| quad_demod | `fm_tone_c32_20000` | 48 | x = exp(j·phase), phase = cumsum(2π·0.05·sin(2π·0.002·n)) + small noise σ 1e-3; complex64; gain 1.0 | `np.angle(x[n]·conj(x[n−1]))` with x[−1] = 0 | SNR ≥ 100 |
| quad_demod | `fm_tone_cq15_20000` | 49 | same signal at amplitude 0.9 quantized; gain_raw 32767 | `np.angle(...)/π · (32767/32768)` from the quantized input | SNR ≥ 70 |
| deemphasis | `tau50us_fs48k_f32_8192` | 50 | real noise σ 0.3; α = 1 − exp(−1/(48000·50e−6)) cast to f32 | `lfilter([α], [1, −(1−α)], x)` with α as f64 of the f32 value | SNR ≥ 100 |
| deemphasis | `tau50us_fs48k_q15_8192` | 51 | quantized x, α_raw = from_float(α) | Python loop of the exact Q15 recursion | bit-exact |
| dc_block | `r0p999_c32_8192` | 52 | noise σ 0.2 + DC offset (0.3 + 0.2j) | `lfilter([1, −1], [1, −r], x)` per component, r as f32 | SNR ≥ 100 |
| dc_block | `r0p999_cq15_8192` | 53 | quantized | Python loop, Q15 rules | bit-exact |
| window | `hann_periodic_1024`, `hann_symmetric_1023`, `blackmanharris_periodic_4096`, `kaiser8_symmetric_255` | 54–57 | — | `scipy.signal.get_window(...)` f64 | max_abs_err ≤ 2e-7 (f32 window generator) |
| window | `apply_hann_c32_1024` / `apply_hann_cq15_1024` | 58/59 | noise × w (w f32 / Q15 from the generator rules) | f64 product / integer `mul` | bit-exact |
| mag_sq | `c32_4096` / `cq15_4096` | 60/61 | noise σ 0.5 (cq15 incl. full-scale values) | `re²+im²` f64 / integer model | SNR ≥ 140 / bit-exact |
| power_db | `f32_4096` | 62 | powers 10^uniform(−25, 1) incl. 0, negative, NaN entries (listed) | `10·log10(max(p, 1e−20))` | max_abs_err ≤ 1e-4 |

Resampler output count `K` for an input of length `n`: number of `j ≥ 0` with `floor(j·M / L) ≤ n − 1`, i.e. `K = floor((n·L − 1) / M) + 1` (check: L = 3, M = 2, n = 1 gives K = 2). Write this formula in the Python case and in `docs/kernels.md`.

All stateful Rust golden tests run the input once in one piece and once in random chunks (ChunkPlan seed = case seed) and compare both against the file.

## Tests

1. Golden tests for every case (host).
2. Invariance tests for every stateful kernel (20 plans each).
3. `fir_decimate` with M = 1 equals `fir` bit for bit; `resample` with L = M = 1 equals `fir`.
4. `quad_demod` of a pure tone `e^{j2πfn}` gives the constant `2πf·gain` (f32 within 1e-6).
5. `deemphasis` step response reaches 63.2 % after τ·fs samples (±1 sample).
6. `dc_block` removes a constant input (0.5 + 0.25j) after 10 000 samples (r = 0.999): f32 output magnitude < 1e-3; Q15: each component settles within ±496 LSB (magnitude < 0.022) — the round-half-up feedback has a dead band of ⌊0.5/(1 − r_q15)⌋ = 496 LSB per component with `r_q15 = 32735`; document this limit in `docs/kernels.md`.
7. Window generators: symmetric windows are symmetric bit for bit; `window_gain` of Hann periodic = (0.5, 1.5) within 1e-9 for N = 4096.
8. `power_db` never panics on NaN/negative/zero.
9. Progress semantics: with an output buffer smaller than needed, `consumed` stops at the right sample and resubmitting the rest yields identical output.
10. Suite cases pass on the host.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`.

## Deliverables checklist

- [x] 8 kernels (+ `resample_real`) with states, fallback, reference, delegation
- [x] `sdrk_core::window`
- [x] Golden cases and Rust golden tests
- [x] Suite, invariance, extremes
- [x] `docs/kernels.md`, `docs/timing.md` updated
