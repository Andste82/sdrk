# M22 — Audio filters (biquad)

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M21](M21-ssb-receive.md) · Next: [M23](M23-real-sampling-input.md)

## Goal

IIR filtering for the receivers of M19–M21: the kernels `biquad` (complex) and `biquad_real` (cascaded second-order sections), the design functions for RBJ cookbook sections and Butterworth cascades, the block `biquad`, its pipeline kind, and two applications in the example flows: removal of the CTCSS sub-tone in the NFM receiver and a selectable audio band for AM and SSB.

## Why

Audio shaping in a receiver is done with short IIR filters: a 300 Hz high-pass removes the CTCSS tone of two-way radio, a notch removes a carrier whistle, a low-pass limits hiss. FIR filters of the same steepness at audio rates need hundreds of taps. The biquad is also the building block of later backlog items (CIC compensation, FM stereo pilot filter, tone controls). It comes after the three receive modes because none of them needs it to work; here it improves them.

## Order

Part of the receive series M19–M23 (see the *Order* section of [M19](M19-nfm-squelch.md)): uses M00–M15 and M19–M21.

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

**In:** kernels `biquad` and `biquad_real` in `sdrk-core` (states, `Fallback`, `Reference`, facade delegation) and `sdrk_core::num::q14_from_f64`; the design functions `biquad_coefs`, `butterworth_sos`, `to_q14` and `to_f32_sos` in `sdrk-blocks`; the block `biquad` with description; its pipeline kind and the regenerated schema; the CTCSS high-pass in `nfm_receiver` and an audio low-pass option in `am_receiver` and `ssb_receiver`; golden cases; suite cases; the end-to-end checks of §6; the SIMD decision and counts of §7; `docs/kernels.md`, `docs/timing.md`, `docs/blocks.md`, `docs/pipeline.md`.

**Out:** filter designs other than Butterworth and the four cookbook sections (Chebyshev, elliptic, shelving and peaking EQ), error-feedback (noise-shaped) Q15 biquads, lattice structures, CIC, FM stereo.

## Specification

General rules from M02/M05/M06 apply (see M19). All f32 arithmetic is written without FMA in exactly the order given; Q15 arithmetic is in `i64` with `round_shift` and `sat16` (R7).

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
pub trait Biquad<T: ComplexSample + BiquadSample> {
    fn biquad<C, H>(&self, s: &mut BiquadState<T, C, H>, input: &[T], output: &mut [T]) -> Result<()>;
}
pub trait BiquadReal<T: RealSample + BiquadSample> {
    fn biquad_real<C, H>(&self, s: &mut BiquadRealState<T, C, H>, input: &[T], output: &mut [T]) -> Result<()>;
}
```

Complex variants filter `re` and `im` independently with the same coefficients.

- **f32 (transposed direct form II), per section and component:** `y = b0·x + s1`; `s1 = (b1·x − a1·y) + s2`; `s2 = b2·x − a2·y`; then `s1 = flush_denormal(s1)`, `s2 = flush_denormal(s2)`. The history holds `s1, s2` in its first `2·sections` entries (the rest is unused).
- **Q15 (direct form I), per section and component:** `acc = b0·x + b1·x1 + b2·x2 − a1·y1 − a2·y2` in `i64` (each product `i16 × i16`), `y = sat16(round_shift(acc, 14))`, then `x2 = x1`, `x1 = x`, `y2 = y1`, `y1 = y`. Each section saturates its own output.
- **Reference:** C32/f32 in f64 (direct form I) with the values of the f32 coefficients converted to f64; CQ15/Q15 the integer model above.
- **Q15 limits (document in `docs/kernels.md`):** the rounding error of the feedback is amplified by `1 / |1 + a1 + a2|` at DC; for a low-pass or high-pass with a cutoff below about `fs / 500` this exceeds 50 dB and the output is dominated by rounding noise. The `biquad` block rejects such designs for Q15 (§3). DF1 in Q15 can show a small limit cycle at silence; test 6 measures it.

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
```

Formulas (normative, copy into the rustdoc):

- `biquad_coefs`: `0 < f0_hz < fs_hz / 2` and `q > 0` (else `OutOfRange`); `w0 = 2π·f0/fs`, `c = cos w0`, `α = sin(w0) / (2q)`. Numerator before normalization: Lowpass `b = [(1 − c)/2, 1 − c, (1 − c)/2]`; Highpass `b = [(1 + c)/2, −(1 + c), (1 + c)/2]`; Bandpass (0 dB peak gain) `b = [α, 0, −α]`; Notch `b = [1, −2c, 1]`; all with `a = [1 + α, −2c, 1 − α]`. Result `[b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]`.
- `butterworth_sos`: `order` even in `2 ..= 8`, `sos.len() == order / 2`; section `k = 1 … order/2` is `biquad_coefs(response, cutoff_hz, Q_k, fs_hz)` with `Q_k = 1 / (2·cos((2k − 1)·π / (2·order)))` (lowest Q first). Both the cookbook and `scipy.signal.butter` use the bilinear transform prewarped at the cutoff, so the cascade has the response of `scipy.signal.butter(order, cutoff_hz, btype, fs=fs_hz, output='sos')` (checked by the generator, see *Test vectors*).

### 3. Block `biquad`

The construction pattern, descriptions, storage and parameter rules of M09 §4–§5 apply unchanged.

| Kind | Struct | In → Out | Parameters (unit, mutability, default) | Implementation |
| --- | --- | --- | --- | --- |
| `biquad` | `BiquadBlock<T>` (complex) and `BiquadRealBlock<T>` (real) | T → T | `response` (Choice `lowpass`/`highpass`/`bandpass`/`notch`, build, `lowpass`), `cutoff_hz` (Hz, **runtime**, required; centre frequency for bandpass and notch), `order` (I64, build, 2 … 8 even, default 2; lowpass and highpass only, must be 2 otherwise), `q` (None, **runtime**, 0.7071; bandpass and notch only) | `butterworth_sos` or `biquad_coefs`, `to_f32_sos`/`to_q14`, `biquad`/`biquad_real`; a runtime change recomputes the coefficients (`set_coefs`, history kept). For CQ15/Q15, `cutoff_hz < fs / 500` for lowpass and highpass is `OutOfRange` (§1) |

### 4. Pipeline kind and flows

- Kind `biquad` (complex and real inputs; `kernels()` = `["biquad", "biquad_real"]`); `cargo xtask schema` regenerates the schema (version unchanged). Add the kind to the WASM declarations if they list kinds.
- `examples/flows/nfm_receiver.json` (+ YAML): a node `ctcss` = `biquad` (highpass, order 4, `cutoff_hz` 300) after `fm_demod`; the comments explain CTCSS (67–254 Hz).
- `am_receiver` and `ssb_receiver`: a node `audio` = `biquad` (lowpass, order 4, `cutoff_hz` 4000 for AM, 2800 for SSB) after the demodulator, with a comment that `cutoff_hz` can be changed at runtime.

### 5. Testkit

- Suite cases (Fallback vs Reference, invariance with 20 chunk plans): `biquad_c32`, `biquad_cq15`, `biquad_real_f32`, `biquad_real_q15`, `extremes_m22`; block case `block_biquad_response`.

### 6. End-to-end checks

- **E2E-NFM with CTCSS (`e2e_nfm_ctcss_{f32,q15}`):** the E2E-NFM signal of M19 with an added 88.5 Hz sub-tone at 10 % of the deviation, through the updated `nfm_receiver` chain: the 88.5 Hz component is at least **50 dB (f32) / 40 dB (Q15)** below the 1 kHz tone, and the SINAD at 1 kHz stays within 1 dB of E2E-NFM.
- E2E-AM and E2E-SSB with the new audio low-pass keep their thresholds.

### 7. SIMD backend and instruction counts

The rules of M13 and of the SIMD optimization apply to every new kernel unchanged (decisions D1–D14; `docs/backends.md`, `docs/performance.md`, `docs/decisions-log.md`). A new kernel gets level paths in `sdrk-backend-simd` for every level the backend supports (x86-64 SSE2, SSE4.2 where it differs, AVX2, AVX-512, AArch64 NEON, wasm32 SIMD128; run-time detection with the feature `std`; a generic `fearless_simd` path as the fallback of every level-specific path, D1) whenever vectorization pays off. A kernel whose work is a per-sample recursion cannot be vectorized without changing its result; it stays with the fallback, is added to the "Not accelerated" list of `docs/backends.md` with the reason, and is counted as `info` against the fallback. For every accelerated kernel:

1. **Contract:** the class of the fallback. Bit-exact kernels are bit-exact with `Fallback` on every level; tolerance kernels meet the SNR of their contract; FMA only where the contract allows it (D5).
2. **Verification:** a `backend_vs_fallback_*` suite case and a property test against the fallback at every level (the CI jobs `tier1` with the static levels, `avx512-sde`, `aarch64-qemu` and `wasm-wasmtime` run them); a level-specific rounding or saturation path also gets an exhaustive or boundary sweep like `convert_out_sweep`, run by the nightly job `exhaustive`.
3. **Counting:** the kernel in `benches/kernels` (`count run`, `count accuracy`), one entry per level in `benches/limits.json` against the fair reference named below, `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32` pass, the files in `benches/results/` are committed and `docs/performance.md` gets the rows. Release limit 1.10×, goal ≤ 1.00× (D14); the accuracy rule applies. An entry that cannot meet its limit yet carries a waiver that names an open item.
4. **Facade:** `backend_for` reports `simd` for the kernel; `RUSTFLAGS="--cfg sdrk_fallback_only"` still restores the fallback.
5. **Embedded backends:** M16–M18 accelerate the kernels listed in their own milestones. If one of them is implemented after this milestone, it decides there whether to include the new kernels.
| Kernel | Accelerated | Reason | Count |
| --- | --- | --- | --- |
| `biquad_real` (f32, Q15) | no | every output depends on the previous output of the same section (a recursion); sections are in series | `info` against the fallback |
| `biquad` (C32, CQ15) | conditional | `re` and `im` are two independent recursions with the same coefficients and can run in two lanes of one vector with the same operation order per lane (bit-exact with the fallback in f32 and Q15). Implement this path where it saves at least 25 % of the instructions against the fallback at that level; otherwise keep the fallback there and record the measured numbers | against the fallback at every level: `dev` where the two-lane path exists (goal: ≤ 0.75× of the fallback), `info` elsewhere |

### 8. Documentation

`docs/kernels.md` (contracts, Q15 limits, measured limit cycle), `docs/timing.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md` (table, design formulas, the Q15 cutoff limit), `docs/pipeline.md`, `examples/README.md`, `CHANGELOG.md`, `docs/decisions-log.md`.

## Test vectors (`tests/golden/sdrk_golden/cases/biquad.py`, `design.py`)

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| design | `biquad_{lowpass,highpass,bandpass,notch}` | 0 | `f0 = 0.1`, `fs = 1`, `q` 0.7071 / 0.7071 / 2 / 10 | formulas of §2 in f64 | max_abs_err ≤ 1e-14 |
| design | `butterworth_{lp2,lp4,lp8,hp6}` | 0 | cutoff 0.05 (lp) / 0.2 (hp), `fs = 1` | formulas of §2; the generator also asserts that `scipy.signal.sosfreqz` of the result equals that of `scipy.signal.butter(..., output='sos')` within 1e-9 at 512 frequencies | max_abs_err ≤ 1e-14 |
| biquad | `butter4_lp_c32_8192` | 424 | `butterworth_sos(lp, 4, 0.1)` → f32; noise σ 0.3 | `scipy.signal.sosfilt` (f64) with the f32 coefficients, per component | SNR ≥ 100 |
| biquad | `butter4_lp_cq15_8192` | 425 | coefficients `to_q14`, quantized noise | Python DF1 integer loop of §1 | bit-exact |
| biquad_real | `notch_q10_f32_8192` | 426 | `biquad_coefs(notch, 0.05, 10)`; real noise σ 0.3 | `sosfilt` | SNR ≥ 90 |
| biquad_real | `bandpass_q2_f32_8192` | 427 | `biquad_coefs(bandpass, 0.1, 2)` | `sosfilt` | SNR ≥ 100 |
| biquad_real | `butter2_hp_q15_8192` | 428 | `butterworth_sos(hp, 2, 0.01)` → `to_q14` | integer loop | bit-exact |
| biquad_real | `retune_f32_8192` | 429 | lowpass order 2, cutoff 0.05 for samples 0–4095 and 0.15 from 4096 (`set_coefs` at that sample, history kept) | the DF2T recursion of §1 in f64 in Python, switching the coefficients at sample 4096 and keeping `s1`, `s2` | SNR ≥ 90 |

All stateful Rust golden tests run whole and in random chunks (ChunkPlan seed = case seed).

## Tests

1. Golden tests for every case, whole and chunked; invariance tests for both kernels and the block (20 plans each).
2. Butterworth: the magnitude response of `butterworth_sos` (f64, evaluated in Rust) is −3.0103 dB ± 0.001 dB at the cutoff for every order.
3. Block: a 1 kHz tone through `biquad` (lowpass, order 4, cutoff 2 kHz at 48 kHz) is attenuated by less than 0.1 dB, an 8 kHz tone by at least 48 dB; `notch` at 1 kHz with q 10 attenuates 1 kHz by at least 40 dB.
4. Runtime: `cutoff_hz` and `q` changes take effect from the next call; the output stays finite.
5. Validation: Q15 cutoffs below `fs/500`, odd orders, `order ≠ 2` for bandpass/notch and coefficients that do not fit Q2.14 give errors.
6. Q15 limit cycle: after the input becomes silent, the output of the `butter4_lp` filter settles to a limit cycle or zero; record the largest amplitude in `docs/kernels.md` and assert it is at most 16 LSB.
7. Extremes (`extremes_m22`): no panic on silence, full scale, NaN and ±∞ (f32), and the Q15 corners.
8. SIMD: where the two-lane path exists, `backend_vs_fallback_biquad` is bit-exact at every level; the counts pass their limits.
9. Block rules: description, exact arena, no block struct larger than 512 bytes.
10. Pipeline: the three updated flows validate, build and run; E2E checks of §6; schema check; the suite passes on the host and in the emulators (quick profile).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32`; `cargo xtask emu --target qemu-cortex-m4` (quick profile within its timeout).

## Deliverables checklist

- [ ] Kernels `biquad`, `biquad_real` with states, fallback, reference, delegation; `q14_from_f64`
- [ ] Design functions `biquad_coefs`, `butterworth_sos`, `to_q14`, `to_f32_sos`
- [ ] Block `biquad`, pipeline kind, regenerated schema; updated flows with YAML twins
- [ ] SIMD decision for `biquad` (two-lane path where it pays off), counts and limits
- [ ] Golden cases (seeds 424–429 and the design cases), suite, invariance, extremes, E2E checks
- [ ] `docs/kernels.md`, `docs/timing.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, CHANGELOG, decisions log
