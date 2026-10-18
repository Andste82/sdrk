# M05 — First kernels: `convert`, `rotate`, `fir`

> Plan: [PLAN.md › Phase 1/2](../PLAN.md#phase-1--core-foundations-medium) · Depends on: [M04](M04-golden-vectors.md) · Next: [M06](M06-receive-kernels.md)

## Goal

Implement the first three real kernels end to end (trait, `Fallback`, `Reference`, facade delegation, golden cases, suite cases, invariance and extreme-input tests): raw-format conversion into the internal types, the NCO mixer, and the complex FIR filter with real taps, in both `C32` and `CQ15`.

## Why

These three kernels prove the full pattern on a format kernel, a phase-continuous stateful kernel and a classic filter. Every later kernel copies their structure.

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
5. Naming: write **SDRk** only in prose, code comments (incl. rustdoc) and documentation; everything else in code is lowercase `sdrk` (identifiers, crate/package/module/file names, string literals such as error messages).
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

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.


## Scope

**In:** `kernels::convert::{Convert, ConvertReal}`, zero-copy views, `kernels::mix::Rotate` with state `Nco`, `kernels::filter::Fir` with state `FirState`, build-time cosine tables, golden cases, suite cases, docs.

**Out:** `convert_out` (M08), decimating/interpolating FIR (M06/M08), any SIMD (M13).

## Specification

### 1. `convert` (bit-exact kernel)

```rust
pub trait Convert<T: ComplexSample> {
    /// Decode whole samples from `input` (layout Interleaved or Planar) into `output`.
    /// Returns the number of samples written.
    fn convert(&self, format: &SampleFormat, input: &[u8], output: &mut [T]) -> Result<usize>;
}
pub trait ConvertReal<T: RealSample> {
    /// Same for Layout::Real.
    fn convert_real(&self, format: &SampleFormat, input: &[u8], output: &mut [T]) -> Result<usize>;
}
pub fn view_cq15(format: &SampleFormat, bytes: &[u8]) -> Option<&[CQ15]>; // zero-copy if format is CS16-compatible (I16, 16 bits, Interleaved, Little, no swap, PowerOfTwo) and alignment allows
pub fn view_c32(format: &SampleFormat, bytes: &[u8]) -> Option<&[C32]>;   // zero-copy for CF32, Little, no swap
```

Rules (all exact; reference and fallback must agree bit for bit):
- Input length must contain whole samples (and whole packing groups): otherwise `InvalidParameter("partial sample")`. Output must hold `format.samples_in_bytes(input.len())` samples: otherwise `BufferTooSmall`. `Bits1` → `Unsupported`. Wrong layout for the trait (Real passed to `Convert`, or I/Q to `ConvertReal`) → `InvalidParameter`.
- Effective integer value `e` per component as documented in `docs/device.md` (endian, alignment, sign extension, packing, swap, planar).
- **To C32/f32:** `(e as f32 − offset) / full_scale` where `offset = 127.5` for U8 else 0, `full_scale` = `2^(bits−1)` or the explicit value (as f32). All these operations are exact in f32 for ≤ 24 effective bits; for `I32` with more than 24 bits compute in f64 and round to f32 (document). `F32` container: copy the value (byte-swapped if big-endian), NaN/∞ pass through unchanged.
- **To CQ15/Q15:**
  - `U8`: `q = (2·u − 255) << 7` (exactly (u − 127.5) · 256).
  - signed, `PowerOfTwo`, bits ≤ 16: `q = e << (16 − bits)`.
  - signed, `PowerOfTwo`, bits > 16: `q = sat16(round_shift(e, bits − 16))`.
  - `Explicit(n)`: `q = sat16(floor((e · 65536 + n) / (2n)))` in i64 (round half up of `e·32768/n`).
  - `F32`: `q15_from_f32` (R7).

### 2. `rotate` (tolerance kernel)

```rust
/// Numerically controlled oscillator state: phase accumulator in units of 2π/2^32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Nco { phase: u32, inc: u32 }
impl Nco {
    pub fn new(freq_hz: f64, rate: Rate) -> Result<Nco>;   // inc = wrapping round(frac(freq_hz / rate) · 2^32); NaN/∞ → InvalidParameter
    pub fn from_increment(inc: u32) -> Nco;
    pub fn set_frequency(&mut self, freq_hz: f64, rate: Rate) -> Result<()>; // keeps the phase (phase-continuous retune)
    pub fn phase(&self) -> u32; pub fn set_phase(&mut self, p: u32); pub fn increment(&self) -> u32;
    pub fn reset(&mut self);  // phase = 0, increment kept
}
impl Timing for Nco { ratio 1/1, group delay 0 }

pub trait Rotate<T: ComplexSample> {
    /// output[n] = input[n] · e^{j·2π·phase_n / 2^32}; phase_{n+1} = phase_n + inc (wrapping).
    /// Positive frequency moves the spectrum up; to bring a signal at +f to 0 Hz use −f.
    fn rotate(&self, nco: &mut Nco, input: &[T], output: &mut [T]) -> Result<()>;
}
```
`frac(x) = x − floor(x)`; negative frequencies therefore map to `inc > 2^31`.

**Cosine tables** generated by `crates/core/sdrk-core/build.rs` (build-dependency `libm`), written to `OUT_DIR/tables.rs` and included with `include!`:
- `COS_F32: [f32; 4097]` with `COS_F32[k] = libm::cos(2π·k/4096) as f32` (f64 computation, then cast).
- `COS_Q15: [i16; 4097]` with R7 `from_float` of the same f64 value.
Sine is cosine at `phase − 2^30` (wrapping).

**Fallback algorithm** (normative; other implementations are judged against the reference):
- `idx = (phase >> 20) as usize` (0…4095), C32: `f = (phase & 0xFFFFF) as f32 · (1.0 / 1048576.0)`, `c = a + (b − a) · f` with `a = COS_F32[idx]`, `b = COS_F32[idx+1]` (no FMA); same for sine. Output `(re·c − im·s, re·s + im·c)` in f32 without FMA.
- CQ15: `f15 = ((phase >> 5) & 0x7FFF) as i64`, `c = sat16(a + round_shift((b − a) · f15, 15))` with i64 arithmetic; then `cq15_mul(x, CQ15{c, s})`.
**Reference:** `θ = 2π · phase / 2^32` in f64, `libm::cos`, `libm::sin`, complex multiply in f64, then `from_c64_round`.
Numeric contract: C32 SNR ≥ 120 dB, CQ15 SNR ≥ 80 dB versus the golden f64 expected value.

### 3. `fir` (C32: tolerance, CQ15: bit-exact)

```rust
/// FIR state with real taps and a history of the last taps.len() − 1 input samples.
pub struct FirState<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> { taps: Taps, hist: Hist, _t: PhantomData<T> }
impl<...> FirState<...> {
    pub fn new(taps: Taps, history: Hist) -> Result<Self>; // taps.len() ≥ 1, history.len() == taps.len() − 1, history zeroed
    pub fn reset(&mut self);                               // history = 0
    pub fn taps(&self) -> &[T::Real];
}
impl Timing for FirState { ratio 1/1, group delay (N−1)/2 as Delay::new(N − 1, 2) }

pub trait Fir<T: ComplexSample> {
    /// y[n] = Σ_{k=0}^{N−1} h[k] · x[n−k], where x[n−k] for n−k < 0 comes from the history.
    fn fir<Taps, Hist>(&self, state: &mut FirState<T, Taps, Hist>, input: &[T], output: &mut [T]) -> Result<()>
    where Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>;
}
```
- After the call the history holds the last N−1 samples of (old history ++ input).
- **C32 fallback:** accumulate `re` and `im` separately in f32, `k` ascending from 0, accumulator starts at 0.0, no FMA.
- **CQ15 fallback and reference:** `Acc64` per component: Σ `h[k]·x[n−k]` (Q30 in i64), then `sat16(round_shift(acc, 15))`. Bit-exact.
- **C32 reference:** f64 accumulation, cast to f32 at the end.
Contract: C32 SNR ≥ 110 dB; CQ15 bit-exact.

### 4. Facade and suite

Add `delegate!` entries for `Convert<C32>`, `Convert<CQ15>`, `ConvertReal<f32>`, `ConvertReal<Q15>`, `Rotate<C32>`, `Rotate<CQ15>`, `Fir<C32>`, `Fir<CQ15>` → `Fallback`. Register suite cases (quick unless noted): `convert_all_formats_vs_encoder`, `rotate_c32_vs_reference`, `rotate_cq15_vs_reference`, `rotate_block_invariance`, `fir_c32_vs_reference`, `fir_cq15_vs_reference`, `fir_block_invariance`, `fir_impulse_response`, `extremes_convert_rotate_fir` (full). Suite cases compare Fallback against `Reference` (no files on MCUs); golden files are used by host tests only.

## Test vectors (add to `tests/golden/sdrk_golden/cases/`)

`convert.py` — kernel `convert`:
- For each integer format constant in `sdrk_device::formats` (all of them except `CF32` and `RF32`; `PACKED10` and `PACKED12` included): case `<format_lowercase>_256`, seed `100 + position in the M01 constant table` (0-based). For `CF32`/`RF32` add `cf32_256`/`rf32_256` with random floats including ±0.0, NaN and ±∞ (expected = the same values; CQ15 expected via `q15.from_float`). Generate 256 random **raw** values uniformly over the full integer range of the format (for U8: 0…255; 12-bit: −2048…2047; I32/24 MSB: −2^23…2^23−1, then shifted left by 8), pack them exactly as `docs/device.md` describes (implement the packing in Python independently), store `input` as `u8` bytes, expected `cf32` = mapping to amplitude computed in f64 and then cast with `.astype(np.complex64)`, and expected `ci16` = the CQ15 rules above (integer arithmetic in Python). Both bit-exact (`bit_exact` class).
- Extra variants not in the constant table: `cu8_swapiq_256` (seed 120, `CU8.with_swap_iq(true)`) and `cs16_bigendian_256` (seed 121, CS16 with `Endian::Big`). (`cs16_planar_256` already comes from the `CS16_PLANAR` constant.)

`rotate.py` — kernel `rotate`:
- Phase model in Python: `inc = int(round(frac(f) * 2**32)) % 2**32`; `phase[n] = (phase0 + n*inc) % 2**32` using Python ints or `np.uint64` arithmetic with explicit modulo; `theta = 2*np.pi*phase/2**32`; `expected = x * np.exp(1j*theta)` in complex128.
- `c32_noise_f0p1234_8192`: seed 20, x = complex noise σ = 0.3 cast to complex64, f = 0.1234 cycles/sample, phase0 = 0, params `inc`, `phase0`; tolerance SNR ≥ 120 dB.
- `c32_negfreq_8192`: seed 21, f = −0.3, phase0 = 123456789.
- `c32_wrap_4096`: seed 22, f = 0.4999, phase0 = 2^32 − 1000 (phase wraps immediately).
- `cq15_noise_f0p05_8192`: seed 23, x = from_float(noise σ = 0.25) stored `ci16`, expected `cf64` computed from the quantized input; tolerance SNR ≥ 80 dB.
- `cq15_fullscale_tone_4096`: seed 24, x = full-scale tone (amplitude 1 − 2⁻¹⁵), checks saturation behaviour; tolerance SNR ≥ 75 dB.

`fir.py` — kernel `fir`:
- `lowpass63_c32_8192`: seed 30, `h = scipy.signal.firwin(63, 0.2)` cast to float32 (param array `taps` stored as `f32`), x complex noise σ = 0.3 as complex64, expected = `scipy.signal.lfilter(h_f32.astype(np.float64), [1.0], x.astype(np.complex128))`; SNR ≥ 110 dB.
- `lowpass63_cq15_8192`: seed 31, taps `i16` = `q15.from_float(firwin(63, 0.2))`, x `ci16` = from_float(noise σ = 0.25); expected `ci16` via the integer model: `acc = np.convolve(x_re, h)[:len]` (int64) → `sat16(round_shift(acc, 15))`, same for im; bit-exact.
- `long255_c32_16384`: seed 32, `firwin(255, 0.05, window=('kaiser', 8.0))`; SNR ≥ 110 dB.
- `single_tap_c32_1024`: seed 33, h = [0.5]; SNR = ∞ expected (bit-exact class allowed).
- `gain_overflow_cq15_2048`: seed 34, taps all 0.25 (8 taps, sum 2.0), x = constant DC `(32767, −32768)` for all samples (after 8 samples the accumulator is 2·x → output saturates to `(32767, −32768)`), with the first 64 samples replaced by `rng` noise so the ramp-in is also checked; bit-exact.

Every Rust golden test feeds the input in chunks of a random size (ChunkPlan) **and** in one piece, so golden tests also verify block invariance.

## Tests

1. Golden tests for all cases above via `Sdrk` (facade) — host only.
2. `convert` versus the independent testkit `encode`/`decode_reference` round trip for every format (proptest: random samples, random formats).
3. `view_cq15`/`view_c32` return `Some` only for compatible formats and aligned buffers; the viewed slice equals `convert` output bit for bit.
4. `Nco::new` with f = 0.25 gives inc = 2^30; with f = −0.25 gives 3·2^30; phase continuity across `set_frequency`.
5. `rotate` with f = 0 is the identity (bit-exact for both types).
6. `fir` impulse response equals the taps (C32 exact; CQ15 equals taps for in-range values); `FirState::new` rejects wrong history lengths; `reset` clears history.
7. Block invariance (testkit harness) for `rotate` and `fir`, 20 random chunk plans each.
8. Extremes: f32 NaN/∞ inputs never panic (NaN propagates); Q15 full-scale inputs saturate correctly; empty input slices return `Ok` and produce nothing.
9. Suite cases registered and passing on the host.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`.

## Deliverables checklist

- [ ] `convert`, `rotate`, `fir` traits, states, fallback, reference, facade delegation
- [ ] `build.rs` cosine tables
- [ ] Golden cases + Rust golden tests
- [ ] Suite cases, invariance, extremes
- [ ] `docs/kernels.md` contracts for the three kernels; `docs/timing.md` part 1 (Timing trait, ratio, group delay, Nco phase model)
