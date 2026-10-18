# M07 — FFT (f32 and Q15, fixed and block-floating-point scaling)

> Plan: [PLAN.md › Phase 2](../PLAN.md#phase-2--mvp-kernels-both-directions-large) · Depends on: [M06](M06-receive-kernels.md) · Next: [M08](M08-transmit-and-fused.md)

## Goal

Implement our own heap-free FFT as an ordinary kernel: power-of-two sizes 16…65 536, in-place radix-2, complex forward/inverse in `C32` and `CQ15`, Q15 with two scaling modes (fixed 1/N and block floating point), and a real-input FFT for `f32` and its inverse. The fallback is a clear radix-2 implementation; speed comes later from backends (M13, M16).

## Why

The plan decided against external FFT crates in the library (heap, no Q15, no backend integration). Spectra, channelizers and OFDM all build on this kernel.

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

**In:** `FftPlan`, `RealFftPlan`, traits `Fft` and `RealFft`, twiddle generation, bit reversal, fixed and BFP Q15 scaling, reference DFT, golden cases, suite cases, docs.

**Out:** radix-4/SIMD versions (M13), Q15 real FFT and its inverse (backlog), FFT-based convolution.

## Specification

### 1. Types

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Direction { Forward, Inverse }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] #[non_exhaustive] pub enum Scaling { Default, BlockFloat }
/// One rule for every type and direction: the unnormalized sum `Σ x[n] e^{∓j2πkn/N}` equals `output · 2^exponent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)] pub struct FftScale { pub exponent: i32 }

pub struct FftPlan<T: ComplexSample, Tw: AsRef<[T]>> { n: usize, log2n: u32, twiddles: Tw, tw_stride: usize /* 1 for new/from_twiddles, 4096/n for from_static_4096 */ }
impl<T: ComplexSample, Tw: AsRef<[T]> + AsMut<[T]>> FftPlan<T, Tw> {
    /// n power of two, 16 ≤ n ≤ 65536; twiddles.len() == n/2. Fills tw[k] = e^{−j2πk/n}
    /// computed in f64 with libm::cos/sin, then cast to f32 (C32) or R7-rounded (CQ15).
    pub fn new(n: usize, twiddles: Tw) -> Result<Self>;
}
impl<T, Tw: AsRef<[T]>> FftPlan<T, Tw> {
    /// Use precomputed twiddles (e.g. a `static` table); validates length only.
    pub fn from_twiddles(n: usize, twiddles: Tw) -> Result<Self>;
    pub fn len(&self) -> usize;
}

pub trait Fft<T: ComplexSample> {
    fn fft<Tw: AsRef<[T]>>(&self, plan: &FftPlan<T, Tw>, data: &mut [T], dir: Direction, scaling: Scaling) -> Result<FftScale>;
}
```
`data.len()` must equal `plan.len()`. `Scaling::BlockFloat` with `C32` → `Unsupported`.

Optional feature `static-twiddles` on `sdrk-core`: `build.rs` additionally emits `TWIDDLES_C32_4096: [C32; 2048]` and `TWIDDLES_CQ15_4096: [CQ15; 2048]`; smaller power-of-two plans use every `4096/n`-th entry through a helper `FftPlan::from_static_4096(n)` (n ≤ 4096), which stores the full static table with `tw_stride = 4096/n`. Document the flash cost (16 KiB / 8 KiB).

### 2. Conventions (normative)

| Type | Direction | Output | `exponent` |
| --- | --- | --- | --- |
| C32 | Forward | `X[k] = Σ x[n] e^{−j2πkn/N}` (unnormalized, like `numpy.fft.fft`) | 0 |
| C32 | Inverse | `x[n] = (1/N) Σ X[k] e^{+j2πkn/N}` (like `numpy.fft.ifft`) | `log2 N` |
| CQ15 | Forward, Default (fixed) | `X[k] / N` | `log2 N` |
| CQ15 | Inverse, Default (fixed) | `(1/N) Σ X[k] e^{+j2πkn/N}` (= `numpy.fft.ifft`; halves after every stage like CMSIS-DSP, so it cannot overflow for inputs with |x| ≤ 1; forward-then-inverse returns `x/N` — use BlockFloat when precision matters) | `log2 N` (relative to the unnormalized sum) |
| CQ15 | Forward/Inverse, BlockFloat | transform divided by `2^e`, where `e` = sum of the per-stage shifts (0, 1 or 2 per stage) | `e` |

For displaying spectra from a Q15 FFT: `power_db(|X_out|²) + 6.0206 · exponent` gives the level relative to the unnormalized transform (each halving = ×4 in power = +6.02 dB). Document this in `docs/kernels.md`.

### 3. Algorithm (fallback, normative order of operations)

1. Bit-reversal permutation in place (swap `i` and `rev(i)` when `i < rev(i)`).
2. For each stage `s = 1 … log2 N`: `half = 2^(s−1)`, `stride = N / 2^s`. For each block start `b` in steps of `2·half`, for `k = 0 … half−1`: `w = tw[k·stride·tw_stride]` (Inverse: `conj(w)`), `a = data[b+k]`, `c = data[b+k+half]`, `t = w·c`, then `data[b+k] = a + t`, `data[b+k+half] = a − t`.
3. f32: complex multiply `(wr·cr − wi·ci, wr·ci + wi·cr)`, no FMA. Inverse: multiply every output by `1.0 / N as f32` (exact power of two) at the end.
4. Q15 multiply **without** intermediate saturation (unlike `cq15_mul`): `tr = round_shift(wr·cr − wi·ci, 15)`, `ti = round_shift(wr·ci + wi·cr, 15)` in i64; `t` may exceed the i16 range (up to ≈ 46341) and is only saturated after the butterfly.
   - **Fixed:** in both directions every stage stores `round_shift(a ± t, 1)` per component (i64 arithmetic, then `sat16`).
   - **BlockFloat:** before each stage compute `m = max |component|` over the whole block. A radix-2 butterfly output can grow to `(1 + √2)·m`. Choose the shift `s` of this stage: `m > 27145` (= ⌊2·32767/(1 + √2)⌋) → `s = 2`; else `m > 13572` (= ⌊32767/(1 + √2)⌋) → `s = 1`; else `s = 0`. The stage stores `sat16(round_shift(a ± t, s))` per component (i64 arithmetic) and adds `s` to `e`. With this rule no butterfly can saturate for inputs whose complex magnitude is ≤ 1 (|x| ≤ 1); corner inputs such as (32767, 32767) have magnitude √2 and may still saturate in the first stage — document this.
5. Index arithmetic must not use unchecked slice indexing (R5.4); use `split_at_mut` per block to obtain `a`/`c` halves and iterate with `zip`.

### 4. Real FFT, forward and inverse (f32 only in this milestone)

```rust
pub struct RealFftPlan<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>> { inner: FftPlan<C32, Tw>, post: Tw2 }  // inner size N/2; post[k] = e^{−j2πk/N}, k < N/2
// new(n, inner_twiddles (len n/4), post_twiddles (len n/2)) -> Result; n ≥ 32
pub trait RealFft {
    /// Forward. input: N real samples; output: N/2 + 1 complex bins (= numpy.fft.rfft); scratch: N/2 complex. Exponent 0.
    fn rfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(&self, plan: &RealFftPlan<Tw, Tw2>, input: &[f32], output: &mut [C32], scratch: &mut [C32]) -> Result<FftScale>;
    /// Inverse. input: N/2 + 1 complex bins; output: N real samples (= numpy.fft.irfft(X, n=N)); scratch: N/2 complex.
    /// The imaginary parts of bin 0 and bin N/2 are ignored (as in NumPy). Exponent log2 N (same rule as §2).
    fn irfft<Tw: AsRef<[C32]>, Tw2: AsRef<[C32]>>(&self, plan: &RealFftPlan<Tw, Tw2>, input: &[C32], output: &mut [f32], scratch: &mut [C32]) -> Result<FftScale>;
}
```
Both directions use the same plan, the same scratch size and the same return type; lengths are checked identically (`LengthMismatch`).
Algorithm: `z[n] = x[2n] + j·x[2n+1]` (n < N/2) into `scratch`; `Z = FFT_{N/2}(z)`; with `Z[N/2] := Z[0]`, for `k = 0 … N/2`: `E = (Z[k] + conj(Z[N/2−k])) / 2`, `O = (Z[k] − conj(Z[N/2−k])) / 2`, `X[k] = E − j · post[k] · O` (for `k = N/2` use `post = −1`, i.e. `e^{−jπ}`).

Inverse algorithm (exact mirror of the forward one): for `k = 0 … N/2 − 1`: `E = (X[k] + conj(X[N/2−k])) / 2`, `D = (X[k] − conj(X[N/2−k])) / 2 · conj(post[k])`, `Z[k] = E + j·D` into `scratch` (use `Re` only of `X[0]` and `X[N/2]`); `z = IFFT_{N/2}(Z)` with the complex `Fft` in direction `Inverse` (which divides by N/2); then `x[2n] = Re z[n]`, `x[2n+1] = Im z[n]`.

### 5. Reference

`Reference` implements `Fft` and `RealFft` as a naive O(N²) DFT in f64 (`libm::cos/sin` of `2π·((k·n) mod N)/N` — reduce the product modulo N in integers first to keep precision), then applies the conventions in §2 (for Q15 BlockFloat the reference returns `round(X/N)` (R7) with exponent `log2 N`, which the one exponent rule allows; comparisons always multiply both sides by `2^exponent` before computing the SNR). Only used up to N = 4096 in tests.

### 6. Facade, suite, docs

Delegations for `Fft<C32>`, `Fft<CQ15>`, `RealFft`. Suite cases: `fft_c32_vs_reference_{64,1024}`, `fft_cq15_fixed_vs_reference_1024`, `fft_cq15_bfp_vs_reference_1024`, `fft_roundtrip_c32`, `fft_roundtrip_cq15_bfp`, `rfft_vs_reference_1024`, `irfft_vs_reference_1024`, `rfft_roundtrip_f32`, `fft_weak_tone_bfp` (full), `fft_parseval_c32`. `docs/kernels.md`: conventions table, scaling modes, dB offset rule, numeric contracts below.

## Numeric contracts (initial thresholds)

| Case | Threshold |
| --- | --- |
| C32 forward/inverse, N ≤ 4096, noise σ = 0.3 | SNR ≥ 110 dB |
| C32 rfft and irfft | SNR ≥ 110 dB |
| CQ15 fixed forward, full-scale tone | SNR ≥ 55 dB |
| CQ15 fixed forward, noise σ = 0.3 | SNR ≥ 40 dB |
| CQ15 BFP forward, full-scale tone | SNR ≥ 60 dB |
| CQ15 BFP forward, noise σ = 0.3 | SNR ≥ 55 dB |
| CQ15 BFP, tone at −60 dBFS, no noise, N = 1024 | tone bin ≥ 30 dB above the median of all other bins |

These are first estimates. If an implementation that follows §3 exactly misses one, analyse why (wrong rounding, overflow, twiddle error) and record the analysis and the corrected threshold in `docs/decisions-log.md`; never lower a threshold silently.

## Test vectors (`cases/fft.py`)

All expected values from `numpy.fft` in complex128, computed from the **quantized** input.

| Case | Seed | Input | Expected | Tolerance |
| --- | --- | --- | --- | --- |
| `c32_fwd_{16,64,1024,4096}` | 70–73 | noise σ 0.3, complex64 | `np.fft.fft(x)` | SNR ≥ 110 |
| `c32_inv_1024` | 74 | X = fft of noise, complex64 | `np.fft.ifft(X)` | SNR ≥ 110 |
| `rfft_f32_{32,1024,4096}` | 75–77 | real noise σ 0.3, float32 | `np.fft.rfft(x)` | SNR ≥ 110 |
| `irfft_f32_{32,1024,4096}` | 85–87 | X = `np.fft.rfft(noise σ 0.3)` cast to complex64, then `X[0].imag` and `X[N/2].imag` set to 0.5 (must be ignored) | `np.fft.irfft(X, n=N)` | SNR ≥ 110 |
| `cq15_fixed_tone_1024` | 78 | full-scale tone at bin 37 (amplitude 1 − 2⁻¹⁵) quantized | `np.fft.fft(x)/1024` | SNR ≥ 55 |
| `cq15_fixed_noise_{1024,4096}` | 79/80 | noise σ 0.3 quantized | `np.fft.fft(x)/N` | SNR ≥ 40 |
| `cq15_fixed_inv_1024` | 81 | X = complex noise σ 0.3, quantized (a white spectrum) | `np.fft.ifft(X)` | SNR ≥ 40 |
| `cq15_bfp_tone_1024` | 82 | as fixed tone | `np.fft.fft(x)` (unnormalized); the Rust test multiplies the output by `2^exponent` before comparing | SNR ≥ 60 |
| `cq15_bfp_noise_1024` | 83 | noise σ 0.3 | as above | SNR ≥ 55 |
| `cq15_bfp_weak_tone_1024` | 84 | tone at bin 100, amplitude 10^(−60/20), quantized, no noise | `np.fft.fft(x)` | criterion from the table (computed in Rust) |

## Tests

1. Golden tests for all cases.
2. Round trip: C32 `inverse(forward(x)) ≈ x` and f32 `irfft(rfft(x)) ≈ x` (SNR ≥ 110 dB). CQ15 with BlockFloat in both directions: `out · 2^(e_fwd + e_inv) / N ≈ x` (SNR ≥ 40 dB for σ 0.3 noise). CQ15 fixed: `inverse(forward(x))` equals `x/N` within the fixed contract (documents the precision loss).
3. Parseval (C32): `Σ|x|² · N ≈ Σ|X|²` within 1e-5 relative.
4. Linearity and time-shift property on random inputs (C32, tolerance 1e-5 relative).
5. Single impulse at 0 → flat spectrum; tone at bin k → energy concentrated in bin k (≥ 100 dB above others for C32).
6. Plan validation: non-power-of-two, N < 16, N > 65536, wrong twiddle length → errors; `data.len() != N` → `LengthMismatch`.
7. `Scaling::BlockFloat` with C32 → `Unsupported`.
8. BFP never overflows: inputs of full-scale alternating values **and** random values with complex magnitude ≤ 1 (random phase, magnitude uniform in [0, 1 − 2⁻¹⁵], 20 seeds) of every size 16…4096 produce no saturation. Implement a test-only counter (`#[cfg(test)]` field or a debug variant of the stage function returning the number of clamped values) and assert it is 0; the counter also counts any intermediate |t| that would exceed 32767 *after* the butterfly sum.
9. `static-twiddles` feature: plans from the static table equal plans from `new` bit for bit (N = 16…4096).
10. Suite cases pass on the host.

## Acceptance

R1.2 commands; `cargo test -p sdrk-core --features reference,static-twiddles`; golden check.

## Deliverables checklist

- [ ] `FftPlan`, `RealFftPlan`, `Fft`, `RealFft` (forward and inverse), scaling modes
- [ ] Reference DFT
- [ ] Golden cases, Rust golden tests, property tests
- [ ] Suite cases
- [ ] `docs/kernels.md` FFT section
