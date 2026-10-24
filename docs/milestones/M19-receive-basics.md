# M19 — Receive basics: biquad, halfband, AGC, squelch, AM and SSB

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M15](M15-emulator-targets.md) (uses only what M00–M15 delivered; see *Order*) · Next: see [BACKLOG.md](BACKLOG.md)

## Goal

Add the receive building blocks that every analog receiver beyond FM needs: the IIR filter `biquad` (cascades, Butterworth and RBJ designs), the efficient decimator `halfband_decimate`, automatic gain control `agc`, a power `squelch`, the envelope `mag`, a real DC blocker `dc_block_real`, and the blocks `biquad`, `halfband`, `agc`, `squelch`, `am_demod` and `ssb_demod` with pipeline kinds, example flows and end-to-end AM and SSB receive tests in f32 and Q15.

## Why

With M00–M15 SDRk can receive FM only. AM (broadcast, airband) and SSB (amateur radio, HF utility) are the next most common analog modes, and both need two parts that FM does not: a gain control (the AM and SSB audio level follows the signal strength) and a squelch (to mute noise between transmissions). The biquad and the halfband decimator are the cheapest filters for audio shaping and for decimation by two; they are needed by these chains and by most later backlog items (CIC compensation, `real_to_complex`, FM stereo). Everything is built from the kernel pattern of M02, so the SIMD and embedded backends can accelerate it later without API changes.

## Order

This milestone uses nothing from M16–M18. It is numbered after them because it was written later. Implementing it before M16 is allowed when the maintainer decides so; that decision goes into `docs/decisions-log.md`, and R1.1 then reads "in the order M00–M15, M19, M16, …".

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

**In:** kernels `biquad` and `biquad_real`, `halfband_decimate` and `halfband_decimate_real`, `mag`, `agc`, `squelch` and `dc_block_real` in `sdrk-core` (states, `Fallback`, `Reference`, facade delegation); the design functions `biquad_coefs`, `butterworth_sos`, `to_q14`, `halfband_len` and `halfband_taps` in `sdrk-blocks`; the blocks `biquad`, `halfband`, `agc`, `squelch`, `am_demod` and `ssb_demod` with descriptions; their pipeline kinds and the regenerated schema; the example flows `am_receiver` and `ssb_receiver` (JSON and YAML); the testkit generators `am_tone` and `ssb_tone`; golden cases, suite cases and the end-to-end cases `e2e_am_rx_*` and `e2e_ssb_rx_*`; `docs/kernels.md`, `docs/timing.md`, `docs/blocks.md` and `docs/pipeline.md`.

**Out:** accelerated paths in `sdrk-backend-simd` or the embedded backends (the facade delegates the new kernels to `Fallback`; a backend may add them later under its own milestone), halfband interpolation, CIC, FM preemphasis, FM stereo and RDS, synchronous AM, AM and SSB modulators, noise blanker, squelch state as pipeline events, `iq_balance`, `real_to_complex` (all backlog).

## Specification

General rules from M02/M05/M06 apply: one trait per kernel, `Fallback` + `Reference` + facade delegation, no allocation, validation at entry, `Timing` implemented by every state. Rate-changing kernels return `Progress` and consume input only as far as output space allows. All f32 arithmetic is written without FMA in exactly the order given; `flush_denormal` is applied where stated. Q15 arithmetic is in `i64` (and `i128` where stated) with `round_shift` and `sat16` from `sdrk_core::num` (R7).

### 1. `biquad` and `biquad_real` — cascaded second-order sections (f32 tolerance, Q15 bit-exact)

Coefficients per section are `[b0, b1, b2, a1, a2]` with `a0 = 1`, i.e. one row of a `scipy.signal` SOS matrix `[b0, b1, b2, 1, a1, a2]` without the 1:

`H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)`, sections in order (the output of section `k` is the input of section `k + 1`).

```rust
/// Coefficient type of a sample type: f32 for C32/f32, i16 for CQ15/Q15 (Q2.14: value = raw / 2^14, range −2 … 2 − 2^-14).
pub trait BiquadSample { type Coef: Copy + Send + Sync; }   // implemented for C32, CQ15, f32 and Q15
pub struct BiquadState<T: ComplexSample + BiquadSample, Coefs: AsRef<[[T::Coef; 5]]>, Hist: AsRef<[T]> + AsMut<[T]>> { /* coefs, hist (len 4·sections) */ }
pub struct BiquadRealState<T: RealSample + BiquadSample, Coefs: AsRef<[[T::Coef; 5]]>, Hist: AsRef<[T]> + AsMut<[T]>> { /* same */ }
// new(coefs, history) -> Result: sections ≥ 1 and ≤ 32, history.len() == 4·sections, else InvalidParameter / LengthMismatch
// set_coefs(coefs) -> Result: same section count (else LengthMismatch); history is kept (used for runtime retuning)
// reset() clears the history; sections()
// Timing: ratio 1/1, group delay 0 (IIR; the delay is frequency-dependent and not part of exact event timing, as `deemphasis`)
pub trait Biquad<T: ComplexSample> {
    fn biquad<C, H>(&self, s: &mut BiquadState<T, C, H>, input: &[T], output: &mut [T]) -> Result<()>;
}
pub trait BiquadReal<T: RealSample> {
    fn biquad_real<C, H>(&self, s: &mut BiquadRealState<T, C, H>, input: &[T], output: &mut [T]) -> Result<()>;
}
```

Complex variants filter `re` and `im` independently with the same coefficients.

- **f32 (transposed direct form II), per section and component:** `y = b0·x + s1`; `s1 = (b1·x − a1·y) + s2`; `s2 = b2·x − a2·y`; then `s1 = flush_denormal(s1)`, `s2 = flush_denormal(s2)`. The history holds `s1, s2` in its first `2·sections` entries (the rest is unused).
- **Q15 (direct form I), per section and component:** `acc = b0·x + b1·x1 + b2·x2 − a1·y1 − a2·y2` in `i64` (each product `i16 × i16`), `y = sat16(round_shift(acc, 14))`, then `x2 = x1`, `x1 = x`, `y2 = y1`, `y1 = y`. Each section saturates its own output.
- **Reference:** C32/f32 in f64 (direct form I) with the coefficient values of the f32 coefficients converted to f64; CQ15/Q15 the integer model above.
- **Q15 limits (document in `docs/kernels.md`):** the rounding error of the feedback is amplified by `1 / |1 + a1 + a2|` at DC; for a low-pass or high-pass with a cutoff below about `fs / 500` this exceeds 50 dB and the output is dominated by rounding noise. The `biquad` block rejects such designs for Q15 (§7). DF1 in Q15 can show a small limit cycle at silence; test 7 measures it.

`sdrk_core::num::q14_from_f64(x: f64) -> Option<i16>`: `floor(x · 16384 + 0.5)`, `None` if the result is outside `−32768 ..= 32767` (so `−2.0` is exact and `+2.0` is rejected).

### 2. Design functions (`sdrk-blocks`, `design.rs`, in f64)

```rust
#[non_exhaustive] pub enum BiquadResponse { Lowpass, Highpass, Bandpass, Notch }
/// RBJ Audio EQ Cookbook (R. Bristow-Johnson), normalized by a0
pub fn biquad_coefs(response: BiquadResponse, f0_hz: f64, q: f64, fs_hz: f64) -> Result<[f64; 5]>;
/// Butterworth of even order 2 … 8 as order/2 sections, low-pass or high-pass (Bandpass/Notch → InvalidParameter)
pub fn butterworth_sos(response: BiquadResponse, order: usize, cutoff_hz: f64, fs_hz: f64, sos: &mut [[f64; 5]]) -> Result<()>;
/// R7 rounding at scale 2^14 (q14_from_f64); error if a coefficient does not fit
pub fn to_q14(src: &[[f64; 5]], dst: &mut [[i16; 5]]) -> Result<()>;
pub fn to_f32_sos(src: &[[f64; 5]], dst: &mut [[f32; 5]]) -> Result<()>;
/// Tap count of a halfband filter: kaiserord(attenuation_db, transition_hz, fs_hz), rounded up to the next N with N mod 4 == 3
pub fn halfband_len(transition_hz: f64, attenuation_db: f64, fs_hz: f64) -> Result<usize>;
/// Halfband taps of length taps.len() (N mod 4 == 3, N ≥ 3) with a Kaiser window of the given beta
pub fn halfband_taps(beta: f64, taps: &mut [f64]) -> Result<()>;
```

Formulas (normative, copy into the rustdoc):

- `biquad_coefs`: `0 < f0_hz < fs_hz / 2` and `q > 0` (else `OutOfRange`); `w0 = 2π·f0/fs`, `c = cos w0`, `α = sin(w0) / (2q)`. Numerator and denominator before normalization: Lowpass `b = [(1 − c)/2, 1 − c, (1 − c)/2]`; Highpass `b = [(1 + c)/2, −(1 + c), (1 + c)/2]`; Bandpass (0 dB peak gain) `b = [α, 0, −α]`; Notch `b = [1, −2c, 1]`; all with `a = [1 + α, −2c, 1 − α]`. Result `[b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]`.
- `butterworth_sos`: `order` even in `2 ..= 8`, `sos.len() == order / 2`; section `k = 1 … order/2` is `biquad_coefs(response, cutoff_hz, Q_k, fs_hz)` with `Q_k = 1 / (2·cos((2k − 1)·π / (2·order)))` (lowest Q first). Both the cookbook and `scipy.signal.butter` use the bilinear transform prewarped at the cutoff, so the cascade has the response of `scipy.signal.butter(order, cutoff_hz, btype, fs=fs_hz, output='sos')` (checked by the generator, see *Test vectors*).
- `halfband_taps`: `N = taps.len()`, `m = (N − 1)/2` (odd). `h = firwin_lowpass(fs/4, fs, Kaiser { beta })` for `fs = 1` (M09 §3); then set `h[m ± 2i] = 0.0` exactly for `i ≥ 1` (they are about 1e-17 from `sin(kπ)`), and scale so that `Σ h = 1`. The taps are symmetric, the outermost taps are non-zero, and about half of the taps are zero.

### 3. `halfband_decimate` and `halfband_decimate_real` (C32/f32 tolerance ≥ 110 dB, CQ15/Q15 bit-exact)

```rust
pub struct HalfbandState<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> { /* taps, hist (len N − 1), phase */ }
pub struct HalfbandRealState<T: RealSample, Taps: AsRef<[T]>, Hist: AsRef<[T]> + AsMut<[T]>> { /* same */ }
// new(taps, history) -> Result: N ≥ 3, N mod 4 == 3, h[m ± 2i] exactly zero for i ≥ 1 (else InvalidParameter), history.len() == N − 1
// Timing: ratio 1/2, group delay Delay::new(N − 1, 2)
pub trait HalfbandDecimate<T: ComplexSample> {
    fn halfband_decimate<Taps, Hist>(&self, s: &mut HalfbandState<T, Taps, Hist>, input: &[T], output: &mut [T]) -> Result<Progress>;
}
pub trait HalfbandDecimateReal<T: RealSample> { /* halfband_decimate_real, same signature with the real state */ }
```

Definition: identical to `fir_decimate` (M06 §1) with factor 2 and the same taps — the first input sample after a reset produces the first output. The fallback evaluates only the non-zero taps (the centre tap and the odd offsets), which halves the multiplications; that is the purpose of the kernel. C32/f32: the products of the non-zero taps are summed in ascending tap index without FMA. CQ15/Q15: `Acc64` over the non-zero taps, one `round_shift(…, 15)` and `sat16` — bit-exact with `fir_decimate` with factor 2, because the omitted terms are exact zeros.

### 4. `mag` — envelope (C32 tolerance ≥ 130 dB, CQ15 bit-exact)

```rust
pub trait Mag<T: ComplexSample> { fn mag(&self, input: &[T], output: &mut [T::Real]) -> Result<()>; }
```

- C32: `libm::sqrtf(re·re + im·im)` (no FMA). Inputs above about 1.8e19 overflow to `+∞`; NaN gives NaN (document).
- CQ15: `s = re² + im²` of the raw values in `i64` (≤ 2^31), `r = isqrt(s)` (exact floor of the square root, `sdrk_core::num::isqrt_u64`), `y = r + (s − r² > r ? 1 : 0)` (round to nearest; a tie is impossible), output `sat16(y)`. Full-scale corners (`|x| > 1`) saturate, as in `mag_sq`.
- Reference: C32 `sqrt` in f64 of the f64 components; CQ15 the integer rule (Python: `math.isqrt`).

### 5. `agc` — automatic gain control (C32 tolerance, CQ15 bit-exact)

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

Rates per sample (f64): `r_a = 1 − exp(−1 / (fs·attack_s))`, `r_r = 1 − exp(−1 / (fs·release_s))`; each must be in `(0, 0.5]` (i.e. a time constant of at least `1 / (fs·ln 2)`), else `OutOfRange`. The gain update is multiplicative, `g ← g·(1 + (r / reference)·e)` with the error `e = reference − |y|`; near the set point this is a first-order loop with the time constant `1 / r` samples, independent of the input level (attack when the output is too loud, release when it is too quiet).

- **C32**, constants as f32 (cast from f64): `ref`, `k_a = r_a / ref`, `k_r = r_r / ref`, `g`, `g_min = 10^(min/20)`, `g_max = 10^(max/20)`. Per sample, in this order: `y = (g·x.re, g·x.im)` (written to the output); `m = sqrtf(y.re·y.re + y.im·y.im)`; if `m` is NaN the gain is not updated; otherwise `e = ref − m`, `k = if e < 0 { k_a } else { k_r }`, `g = g · (1 + k·e)`, then `g = g.max(g_min).min(g_max)` (an infinite input drives `g` to `g_min`).
- **CQ15**, gain `g: i64` in Q.32 (value = raw / 2^32; `g_max` ≤ 31 623 · 2^32 < 2^47): `ref_raw = floor(reference · 32768 + 0.5)` (≥ 1024), `k_raw = floor(r · 2^48 / ref_raw + 0.5)` (≤ 2^37), limits `floor(10^(db/20) · 2^32 + 0.5)`. Per sample: `y.c = sat16(round_shift(x.c · g, 32))` per component; `s = y.re² + y.im²`; `m` = the rounded square root of §4 **without** saturation (≤ 46 341); `e = ref_raw − m`; `k_raw = e < 0 ? k_a : k_r`; `delta = (g · e · k_raw + 2^47) >> 48` computed in `i128` (bounded by 2^100); `g = clamp(g + delta, g_min, g_max)`.
- **Reference:** C32 the same algorithm in f64 with the f64 values of the f32 constants; CQ15 the integer model.

### 6. `squelch` — power gate (bit-exact in both types)

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

`α = 1 − exp(−1 / (fs·tau_s))` in f64. Per sample: update the average power, then the state (open when `p ≥ P_open`, closed when `p < P_close`, otherwise unchanged), then output `x` when open and zero when closed. The squelch starts closed with `p = 0`.

- **C32:** `α` as f32; `P_open = 10^(threshold/10)`, `P_close = 10^((threshold − hysteresis)/10)` as f32; `q = re·re + im·im`; `p = flush_denormal(p + α·(q − p))`. A NaN `q` leaves `p` unchanged. Closed output is `(+0.0, +0.0)`.
- **CQ15:** `α_raw = q15_from_f64(α)`; thresholds `floor(10^(db/10) · 2^30 + 0.5)` as `i64`; `q = re² + im²` (Q30); `p = p + round_shift(α_raw · (q − p), 15)` in `i64`. Closed output is `CQ15` zero.
- **Reference:** for both types the same arithmetic (f32 for C32): the gate decision is discrete, so the specification fixes the arithmetic, and the golden cases are computed by an exact float32 model (record this in the decisions log as an exception to the f64 reference rule, as for the integer Q15 models).

### 7. `dc_block_real` (f32 tolerance ≥ 100 dB, Q15 bit-exact)

The real variant of `dc_block` (M06 §5) with identical rules: `DcBlockRealState<T: RealSample>`, `new(r)` with `0 < r < 1`, `y[n] = x[n] − x[n−1] + r·y[n−1]`, f32 with `flush_denormal`, Q15 `v = (x − x_prev) as i64 + round_shift(r · y_prev, 15)`, `y = sat16(v)`; the Q15 dead band of M06 test 6 applies and is documented. Needed by `am_demod`.

### 8. Blocks (complex blocks for C32 and CQ15; real blocks for f32 and Q15)

The construction pattern, descriptions, storage and parameter rules of M09 §4–§5 apply unchanged.

| Kind | Struct | In → Out | Parameters (unit, mutability, default) | Implementation |
| --- | --- | --- | --- | --- |
| `biquad` | `BiquadBlock<T>` (complex) and `BiquadRealBlock<T>` (real) | T → T | `response` (Choice `lowpass`/`highpass`/`bandpass`/`notch`, build, `lowpass`), `cutoff_hz` (Hz, **runtime**, required; centre frequency for bandpass and notch), `order` (I64, build, 2 … 8 even, default 2; lowpass and highpass only, must be 2 otherwise), `q` (None, **runtime**, 0.7071; bandpass and notch only) | `butterworth_sos` or `biquad_coefs`, `to_f32_sos`/`to_q14`, `biquad`/`biquad_real`; a runtime change recomputes the coefficients (`set_coefs`, history kept). For CQ15/Q15, `cutoff_hz < fs / 500` for lowpass and highpass is `OutOfRange` (§1) |
| `halfband` | `HalfbandBlock<T>` (complex) and `HalfbandRealBlock<T>` (real) | T → T, rate / 2 | `transition_hz` (Hz, build, default 0.1·fs), `attenuation_db` (dB, build, 60), `max_taps` (I64, build, 1023) | `halfband_len` + `halfband_taps` (β = `kaiser_beta(attenuation_db)`), quantized like the `fir` block, `halfband_decimate`; usable output band `±(fs/4 − transition_hz/2)` |
| `agc` | `AgcBlock<T>` | C32/CQ15 → same | `reference_db` (dB, runtime, −6; −30 … 0), `attack_s` (Seconds, runtime, 0.005), `release_s` (Seconds, runtime, 0.2), `min_gain_db` (dB, runtime, −20), `max_gain_db` (dB, runtime, 60), `initial_gain_db` (dB, build, 0) | `agc`; a runtime change calls `set_params` (the current gain is kept) |
| `squelch` | `SquelchBlock<T>` | C32/CQ15 → same | `threshold_db` (dB, runtime, −40; −150 … 0), `hysteresis_db` (dB, runtime, 3; 0 … 40), `tau_s` (Seconds, build, 0.01) | `squelch` |
| `am_demod` | `AmDemodBlock<T>` | C32 → f32, CQ15 → Q15 | `dc_cutoff_hz` (Hz, build, 30; 0 = off), `audio_cutoff_hz` (Hz, build, 5000), `audio_decimation` (I64, build, 1 … 64, default 1) | `mag` → `dc_block_real` with `r = exp(−2π·dc_cutoff/fs)` → `resample_real` (L = 1, M = `audio_decimation`; audio low-pass designed exactly as in `fm_demod`, M09 §6) |
| `ssb_demod` | `SsbDemodBlock<T>` | C32 → f32, CQ15 → Q15 | `sideband` (Choice `usb`/`lsb`, **runtime**, `usb`), `low_hz` (Hz, build, 300), `high_hz` (Hz, build, 3000), `transition_hz` (Hz, build, 200), `attenuation_db` (dB, build, 60), `max_taps` (I64, build, 1023), `audio_decimation` (I64, build, 1 … 64, default 1) | see below |

**`ssb_demod`:** `c = (low_hz + high_hz)/2`, `b = (high_hz − low_hz)/2`, `s = +1` for USB and `−1` for LSB. The input is shifted by `−s·c` (`rotate`, NCO at the input rate), low-pass filtered and decimated (`fir_decimate` with factor `audio_decimation`; taps `kaiserord` + `firwin_lowpass(Kaiser)` with cutoff `b`, `transition_hz` and `attenuation_db` at the input rate, as the `fir` block), shifted back by `+s·c` (second NCO at the output rate), and the output is the real part. The result contains the sideband `[low_hz, high_hz]` above (USB) or below (LSB) the carrier at 0 Hz, mapped to audio `low_hz … high_hz`. Validation: `0 ≤ low_hz < high_hz`, `high_hz + transition_hz/2 ≤ fs_out/2`. Changing `sideband` at runtime flips the sign of both NCO increments from the next call; both phases continue. Both NCOs start at phase 0 after construction and `reset`.

**Chain order (document in `docs/blocks.md`):** `squelch` must come **before** `agc` (after the AGC every signal, noise included, has the reference level, so the squelch would always open); a typical AM or SSB chain is `ddc → squelch → agc → am_demod | ssb_demod`. AM needs a slow AGC (time constants well above the audio period, e.g. attack 0.1 s, release 0.5 s), otherwise the AGC follows the modulation; SSB uses the defaults.

### 9. Pipeline, schema, examples

- One kind per block in `sdrk-pipeline` (`kinds.rs`, the `Kind` trait with `kernels()`, `plan`, cost and memory figures), so `validate`, `build` and the report know them; `biquad` and `halfband` accept complex and real inputs (two block structs), the other four complex inputs.
- `cargo xtask schema` regenerates `schema/sdrk-pipeline-v1.json`; adding kinds keeps the schema version (record in the decisions log).
- Example flows with YAML twins (M11 test 8 covers them automatically): `examples/flows/am_receiver.json` (input `cu8` at 2.4 MS/s → `ddc` (decimation 50 → 48 kHz) → `squelch` → `agc` (attack 0.1 s, release 0.5 s) → `am_demod` → output `f32`) and `examples/flows/ssb_receiver.json` (same front end, `ssb_demod` USB with defaults). If the WASM bindings list block kinds anywhere (`index.d.ts`, `FlowBuilder`), add the new kinds there.

### 10. Testkit

- Generators (module `r#gen`, no allocation): `am_tone(out: &mut [C32], rate: Rate, audio_hz: f64, index: f64, carrier: f64, phase: f64)` = `carrier·(1 + index·cos(2π·audio_hz·n/fs))·e^{j·phase}` computed in f64 and rounded once to f32; `ssb_tone(out: &mut [C32], rate: Rate, audio_hz: f64, sideband: Sideband, amplitude: f64)` = `amplitude·e^{±j2π·audio_hz·n/fs}` (`+` for USB).
- Suite cases (Fallback vs Reference with testkit generators, invariance for every stateful kernel): `biquad_*`, `biquad_real_*`, `halfband_*`, `mag_*`, `agc_*`, `squelch_*`, `dc_block_real_*`, `extremes_m19` (full); block cases `block_biquad_response`, `block_halfband_alias`, `block_agc_level`, `block_squelch_gate`.
- **E2E-AM (`e2e_am_rx_{f32,q15}`):** at 2.4 MS/s: `am_tone` (1 kHz, index 0.5, carrier 0.5) shifted to +300 kHz, plus complex noise σ 0.01, encoded to `CU8` with the testkit encoder (as E2E-RX of M08) → `convert` → `ddc` (frequency 300 kHz, decimation 50) → `squelch` (−40 dB) → `agc` (attack 0.1 s, release 0.5 s) → `am_demod` (defaults). SINAD at 1 kHz over the last 50 % of the audio: **f32 ≥ 35 dB, Q15 ≥ 30 dB** (initial thresholds, see the M07 rule on thresholds).
- **E2E-SSB (`e2e_ssb_rx_{f32,q15}`):** same front end; the signal is a USB tone at carrier + 1 kHz and an equally strong LSB tone at carrier − 1.5 kHz → `ssb_demod` (USB, defaults). SINAD at 1 kHz **f32 ≥ 35 dB, Q15 ≥ 30 dB**; the 1.5 kHz component (measured with `ToneMeter`) is at least **50 dB (f32) / 40 dB (Q15)** below the 1 kHz tone. The same chain with `sideband` = LSB swaps the roles (test only, same thresholds).
- Both E2E cases have a quick (0.1 s) and a full (1.0 s) variant like the FM cases. The Cortex-M4 quick profile already takes about 280 of its 300 s: new quick cases must stay cheap; if the quick profile of a target no longer fits its timeout, move cases to the full profile or raise that target's timeout, and log it.

### 11. Documentation

`docs/kernels.md` (contracts of the eight kernels with the Q15 limits of §1 and §7), `docs/timing.md` (ratios and delays), `docs/blocks.md` (the generated table, the chain order of §8), `docs/pipeline.md` (kinds), `examples/README.md` (the two flows), `CHANGELOG.md`, `docs/decisions-log.md`.

## Test vectors (`tests/golden/sdrk_golden/cases/biquad.py`, `halfband.py`, `mag.py`, `agc.py`, `squelch.py`, `dc_block.py`, `design.py`, `blocks.py`)

Common inputs and quantization as in M06 (`noise(seed, n, sigma)`, complex64 for C32, `q15.from_float` for CQ15). Seeds 410–429.

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| design | `biquad_{lowpass,highpass,bandpass,notch}` | 0 | `f0 = 0.1`, `fs = 1`, `q` 0.7071 / 0.7071 / 2 / 10 | formulas of §2 in f64 | max_abs_err ≤ 1e-14 |
| design | `butterworth_{lp2,lp4,lp8,hp6}` | 0 | cutoff 0.05 (lp) / 0.2 (hp), `fs = 1` | formulas of §2; the generator also asserts that `scipy.signal.sosfreqz` of the result equals that of `scipy.signal.butter(..., output='sos')` within 1e-9 at 512 frequencies | max_abs_err ≤ 1e-14 |
| design | `halfband_n{11,23,63}` | 0 | β = `kaiser_beta(60)` | `firwin(N, 0.25, window=('kaiser', β), fs=1)`, even offsets set to 0, scaled to Σ = 1 | max_abs_err ≤ 1e-12 |
| biquad | `butter4_lp_c32_8192` | 410 | `butterworth_sos(lp, 4, 0.1)` → f32; noise σ 0.3 | `scipy.signal.sosfilt` (f64) with the f32 coefficients, per component | SNR ≥ 100 |
| biquad | `butter4_lp_cq15_8192` | 411 | coefficients `to_q14`, quantized noise | Python DF1 integer loop of §1 | bit-exact |
| biquad_real | `notch_q10_f32_8192` | 412 | `biquad_coefs(notch, 0.05, 10)`; real noise σ 0.3 | `sosfilt` | SNR ≥ 90 |
| biquad_real | `bandpass_q2_f32_8192` | 413 | `biquad_coefs(bandpass, 0.1, 2)` | `sosfilt` | SNR ≥ 100 |
| biquad_real | `butter2_hp_q15_8192` | 414 | `butterworth_sos(hp, 2, 0.01)` → `to_q14` | integer loop | bit-exact |
| halfband_decimate | `n23_c32_8192` | 415 | `halfband_taps` N 23, β = `kaiser_beta(60)` → f32; noise σ 0.3 | `lfilter(h, 1, x)[::2]` | SNR ≥ 110 |
| halfband_decimate | `n23_cq15_8192` | 416 | taps and input quantized (the zero taps stay 0) | integer model of `fir_decimate` with factor 2 | bit-exact |
| halfband_decimate_real | `n23_f32_8192` / `n23_q15_8192` | 417 / 418 | real noise | as above | SNR ≥ 110 / bit-exact |
| mag | `c32_4096` | 419 | noise σ 0.5 plus the listed values 0, ±1, ±(1 ± 1j) | `np.abs` in f64 | SNR ≥ 130 |
| mag | `cq15_4096` | 420 | quantized, including the four full-scale corners | integer rule of §4 (`math.isqrt`) | bit-exact |
| agc | `steps_c32_24000` | 421 | 48 kHz; complex tone 1 kHz with amplitude 0.01, 0.3, 0.003 (8000 samples each) plus noise σ 1e-4; reference 0.5, attack 0.005 s, release 0.05 s, gains −20 / 60 / initial 0 dB | f64 model of §5 with the f64 values of the f32 constants | SNR ≥ 80 |
| agc | `steps_cq15_24000` | 422 | quantized, same parameters | integer model of §5 (Python `int`; assert the `i64`/`i128` bounds) | bit-exact |
| squelch | `bursts_c32_24000` | 423 | 48 kHz; noise σ 0.001 (−60 dBFS) with tone bursts at −20 dBFS (samples 4000–9999 and 15000–17999) and at −41 dBFS (20000–21999); threshold −40 dB, hysteresis 3 dB, τ 0.002 s | exact float32 model of §6 (NumPy `float32` scalars in a loop) | bit-exact |
| squelch | `bursts_cq15_24000` | 424 | quantized | integer model | bit-exact |
| dc_block_real | `r0p999_f32_8192` | 425 | real noise σ 0.2 + DC 0.3 | `lfilter([1, −1], [1, −r], x)`, r as f32 | SNR ≥ 100 |
| dc_block_real | `r0p999_q15_8192` | 426 | quantized | Python loop, Q15 rules | bit-exact |
| blocks | `am_demod_c32` | 427 | 48 kHz: `am_tone` (1 kHz, index 0.5, carrier 0.5) + noise σ 1e-3, 16 000 samples; `audio_decimation` 1 | NumPy chain: `abs`, `lfilter` DC blocker, `upfirdn` with the block taps | SNR ≥ 90 |
| blocks | `ssb_demod_usb_c32` | 428 | 48 kHz: USB tone 1 kHz + LSB tone 1.5 kHz (each amplitude 0.3) + noise σ 1e-3, 16 000 samples | NumPy chain: rotation in f64, `upfirdn(h, ·, 1, M)` restricted as in M06, rotation back, real part | SNR ≥ 90 |
| blocks | `ssb_demod_lsb_cq15` | 429 | same signal with the tones mirrored, quantized | integer chain with the Q15 models of `rotate` (M05) and `fir_decimate` (M06) | bit-exact |

All stateful Rust golden tests run the input once in one piece and once in random chunks (ChunkPlan seed = case seed) and compare both against the file.

## Tests

1. Golden tests for every case (host).
2. Invariance tests for every stateful kernel (20 plans each) and for the six blocks.
3. `halfband_decimate` with CQ15/Q15 equals `fir_decimate` with factor 2 and the same taps bit for bit; with C32/f32 equal for finite inputs as well (a zero result may differ in sign).
4. Butterworth: the magnitude response of `butterworth_sos` (f64, evaluated in Rust) is −3.0103 dB ± 0.001 dB at the cutoff for every order; a 1 kHz tone through the `biquad` block (lowpass, order 4, cutoff 2 kHz at 48 kHz) is attenuated by less than 0.1 dB, a 8 kHz tone by at least 48 dB.
5. AGC: an input step from −40 dBFS to −10 dBFS (tone) settles to within 1 dB of the reference after `5·attack_s`, a step down within 1 dB after `5·release_s`, the gain never leaves `[min, max]`, and a NaN sample (C32) leaves the gain unchanged.
6. Squelch: opens on the first sample where `p ≥ P_open`, does not close between `P_close` and `P_open`, closed output is exactly zero; `set_thresholds` applies from the next sample.
7. Q15 `biquad`: after the input becomes silent, the output of the `butter4_lp` filter settles to a limit cycle or zero; record the largest amplitude in `docs/kernels.md` and assert it is at most 16 LSB.
8. `mag`, `agc`, `squelch`, `biquad` never panic on silence, full scale, NaN and ±∞ (f32), and on the Q15 corners (`extremes_m19`).
9. Blocks: descriptions pass the M09 rules; construction in an exact arena; every runtime parameter changes behaviour from the next call (`biquad.cutoff_hz` moves the −3 dB point, `ssb_demod.sideband` swaps the sidebands, `agc.reference_db` changes the level, `squelch.threshold_db` the gate); timing matches an impulse for `halfband` (±0.5 sample); no block struct is larger than 512 bytes.
10. Pipeline: the two example flows validate, build and run on the E2E signals with the same SINAD thresholds as the testkit cases; the schema check passes.
11. E2E-AM and E2E-SSB meet their thresholds; the suite passes on the host and in the emulators of M15 (`cargo xtask emu --all`, quick profile).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask emu --target qemu-cortex-m4` (quick profile within its timeout).

## Deliverables checklist

- [ ] Kernels `biquad`, `biquad_real`, `halfband_decimate`, `halfband_decimate_real`, `mag`, `agc`, `squelch`, `dc_block_real` with states, fallback, reference, delegation; `q14_from_f64`, `isqrt_u64`
- [ ] Design functions `biquad_coefs`, `butterworth_sos`, `to_q14`, `to_f32_sos`, `halfband_len`, `halfband_taps`
- [ ] Blocks `biquad`, `halfband`, `agc`, `squelch`, `am_demod`, `ssb_demod` with descriptions, both sample types
- [ ] Pipeline kinds, regenerated schema, `am_receiver` and `ssb_receiver` flows with YAML twins
- [ ] Generators `am_tone`, `ssb_tone`; golden cases (seeds 410–429 and the design cases); suite, invariance, extremes, E2E-AM, E2E-SSB
- [ ] `docs/kernels.md`, `docs/timing.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, CHANGELOG, decisions log
