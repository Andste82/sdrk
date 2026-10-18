# M08 — Transmit kernels, fused kernels, measurement meters, FM end-to-end chain

> Plan: [PLAN.md › Phase 2](../PLAN.md#phase-2--mvp-kernels-both-directions-large) · Depends on: [M07](M07-fft.md) · Next: [M09](M09-blocks.md)

## Goal

Complete phase 2: the transmit kernels (`convert_out`, `fir_interpolate`, `fm_mod`), the fused kernels (`ddc`, `duc`, `fm_demod`) with a chained default, the testkit meters (tone, SINAD, spectral mask) and FM generator, and two end-to-end kernel-chain tests (FM receive from CU8, and FM transmit-then-receive) in f32 and Q15.

## Why

The plan's phase-2 exit criterion: "a synthetic FM broadcast reaches the SINAD threshold in f32 and Q15". Transmit must exist from the start (receive and transmit are equal citizens).

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
5. Plan (living document): https://claude.ai/code/artifact/694f271a-ba38-4b7c-b2fa-fb2e919d35c1 — snapshot in [`docs/PLAN.md`](../PLAN.md). The plan explains *why*; the milestones say *what exactly* to build.

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

**In:** kernels `ConvertOut`, `ConvertOutReal`, `FirInterpolate`, `FmMod`; fused `Ddc`, `Duc`, `FmDemod` with chained default functions; testkit `fm`, `meters`, `design_helpers`; golden cases; suite cases; E2E kernel tests.

**Out:** blocks and filter design (M09), pipeline (M10), channel simulation (M12).

## Specification

### 1. `convert_out` (bit-exact)

```rust
pub trait ConvertOut<T: ComplexSample> {
    /// Encode samples into raw bytes of `format` (Interleaved/Planar); returns bytes written.
    fn convert_out(&self, format: &SampleFormat, input: &[T], output: &mut [u8]) -> Result<usize>;
}
pub trait ConvertOutReal<T: RealSample> { fn convert_out_real(&self, format: &SampleFormat, input: &[T], output: &mut [u8]) -> Result<usize>; }
```
Inverse of the `convert` mapping (M05), with rounding and saturation:
- From C32/f32: `v = x as f64 · full_scale + offset`; NaN → amplitude 0; `raw = floor(v + 0.5)` saturated to the format's range (`U8`: 0…255; signed b bits: −2^(b−1)…2^(b−1)−1); then alignment (`Msb`: shift left by `container_bits − bits`), packing, swap, planar, endian exactly as in `docs/device.md`. `F32` container: copy.
- From CQ15/Q15 (integer only): `U8`: `raw = clamp(round_shift(q + 32640, 8), 0, 255)`; signed PowerOfTwo bits ≤ 16: `raw = sat_b(round_shift(q, 16 − bits))`; bits > 16: `q << (bits − 16)`; `Explicit(n)`: `raw = sat_b(floor((q·2n + 32768) / 65536))`; `F32`: `to_f32`.
- Output buffer too small → `BufferTooSmall`; `Bits1` → `Unsupported`; packing groups must be complete (`InvalidParameter`).

### 2. `fir_interpolate` (C32 tolerance ≥ 110 dB, CQ15 bit-exact)

State `FirInterpState<T, Taps, Hist>` with factor `L ≥ 1`, taps designed at the output rate with gain `L` included by the caller, history `ceil(N/L)` input samples. Definition: identical to `resample` with `up = L`, `down = 1` (= `upfirdn(h, x, L, 1)`), producing outputs only for received input (each input sample yields exactly `L` outputs). Timing: ratio `L/1`, group delay `Delay::new(N − 1, 2·L)` input samples. Trait `FirInterpolate<T>` returning `Progress`. Also `FirInterpolateReal<T: RealSample>`.

### 3. `fm_mod` (C32 tolerance ≥ 120 dB, CQ15 ≥ 80 dB)

```rust
pub struct FmModState { phase: u32, k: f64 }   // k = deviation_hz / sample_rate · 2^32 (phase units per unit amplitude)
// FmModState::new(deviation_hz: f64, rate: Rate) -> Result; reset() (phase = 0); phase(); set_phase()
// Timing: ratio 1/1, group delay 0
pub trait FmMod<T: ComplexSample> {
    /// phase += floor(x[n] · k + 0.5) (as i64, wrapping into u32); output[n] = e^{j·2π·phase/2^32} (amplitude 1 − 2^-15 for CQ15 via the cosine table)
    fn fm_mod(&self, s: &mut FmModState, input: &[T::Real], output: &mut [T]) -> Result<()>;
}
```
Q15 input: `x = q / 32768` in f64 before the formula. Output lookup uses the same cosine tables and interpolation as `rotate` (M05) applied to a unit vector. Reference: exact `cos/sin` in f64.

### 4. Fused kernels with chained defaults

For each fused kernel there is a trait **and** a generic chaining function in `sdrk_core::fused`. The facade implements the trait by calling the chaining function with `self` (so it uses the best single kernels); a backend may later provide a truly fused version, which must be **bit-identical** to the chained result of the same backend's single kernels and is only kept if it needs fewer instructions (decision table).

```rust
pub fn ddc_chained<T, K, Taps, Hist>(k: &K, nco: &mut Nco, fir: &mut FirDecimState<T, Taps, Hist>, input: &[T], output: &mut [T], scratch: &mut [T]) -> Result<Progress>
where T: ComplexSample, K: Rotate<T> + FirDecimate<T>;
```
- Determine `n_in = fir.max_input_for_output(output.len()).min(input.len()).min(scratch.len())` (add this helper to `FirDecimState`: the largest number of input samples whose decimated outputs fit), rotate exactly `input[..n_in]` into `scratch[..n_in]`, then `fir_decimate(scratch[..n_in], output)`; it must consume all `n_in`. This keeps the NCO phase exact when output space is limited.
- `duc_chained(k, fir_interp, nco, input, output, scratch)`: interpolate as many whole input samples as `output` can take (`output.len() / L`), into `scratch`, then rotate `scratch` → `output`.
- `fm_demod_chained(k, quad_state, deemph_state, input, output_audio, scratch_real, gain)`: `quad_demod` into `scratch_real`, then `deemphasis` into `output_audio`.
- Traits `Ddc<T>`, `Duc<T>`, `FmDemod<T>` with the same signatures (minus `k`). Facade: `impl Ddc<T> for Sdrk { fn ddc(...) { ddc_chained(self, ...) } }`.

### 5. Testkit additions

- `fm::fm_baseband(out: &mut [Complex64], rate_hz, carrier_offset_hz, deviation_hz, audio_hz, audio_amp, start)` — analytic FM: instantaneous frequency `offset + deviation·audio_amp·sin(2π·audio_hz·n/rate)`, phase integrated **analytically** (closed form `2π(offset·n/rate) − (deviation·audio_amp/audio_hz)·cos(2π·audio_hz·n/rate) + const`), so chunked generation is exact.
- `meters::ToneMeter::measure(signal: &[f64], rate, freq) -> Tone { amplitude, phase, dc, residual_power }` — least-squares fit of `a·cos + b·sin + dc` at the known frequency.
- `meters::sinad_db(signal: &[f64], rate, freq) -> f64` — `10·log10(total_power / residual_power)` with the residual from `ToneMeter` (total power excludes DC).
- `meters::estimate_frequency(signal: &[Complex64], rate) -> f64` — peak of a zero-padded (4×) f64 DFT with parabolic interpolation; independent of the FFT kernel.
- `meters::spectral_mask(signal, rate, mask: &[(f64, f64, f64)] /* (f_lo, f_hi, max_db) */) -> Result<(), MaskViolation>` — Hann-windowed f64 DFT, averaged over non-overlapping segments of 1024.
- `design_helpers::lowpass_hann(cutoff_norm: f64, taps: &mut [f64])` — windowed-sinc low-pass used **only by tests** until M09 provides the real designer. `cutoff_norm` = cutoff / fs in cycles per sample (0 < c < 0.5; *not* the SciPy Nyquist convention). With `N = taps.len()`, `m = (N − 1)/2`: `h[n] = 2c · sinc(2c·(n − m)) · w[n]`, `sinc(x) = sin(πx)/(πx)`, `sinc(0) = 1`, symmetric Hann `w[n] = 0.5 − 0.5·cos(2πn/(N − 1))`, then normalised so that `Σ h = 1`. Tests convert to f32 with `as f32` and to Q15 with R7.

### 6. End-to-end kernel tests (host tests + suite cases)

**E2E-RX (`e2e_fm_rx_{f32,q15}`):**
1. Generate 1.0 s (full tier; quick tier 0.1 s) of `fm_baseband(rate 2.4 MHz, offset +600 kHz, deviation 75 kHz, audio 1 kHz, amp 1.0)`, scale to amplitude 0.9, encode to `CU8` with the testkit encoder.
2. `convert` → `C32` (or `CQ15`).
3. `ddc` with `Nco` at −600 kHz and `fir_decimate` M = 10, taps = `lowpass_hann(100 kHz / 2.4 MHz, 127 taps)` (Q15 taps via R7).
4. `fm_demod` at 240 kHz with deemphasis τ = 50 µs. Gain: f32 `240_000 / (2π · 75_000)` (full deviation → ±1.0). Q15: the output unit is phase/π; full deviation is a phase step of 2π·75k/240k = 0.625π, i.e. output 0.625 at gain 1.0, so use gain raw 32767 and expect audio amplitude ≈ 0.625 (no gain > 1 is possible in Q15; SINAD is amplitude-independent).
5. `resample_real` L = 1, M = 5, taps `lowpass_hann(15 kHz / 240 kHz, 241 taps)`.
6. SINAD at 1 kHz over the last 50 % of the audio: **f32 ≥ 40 dB, Q15 ≥ 35 dB** (initial thresholds, see the M07 rule on thresholds).
Also check the measured audio frequency with `ToneMeter` within ±0.1 Hz.

**E2E-TX→RX (`e2e_fm_txrx_{f32,q15}`):** audio 1 kHz at 48 kHz → `fir_interpolate_real` ×5 (taps `5·lowpass_hann(15k/240k, 241)`) → `fm_mod` (deviation 75 kHz at 240 kHz) → `duc` (interp ×10, taps `10·lowpass_hann(100k/2.4M, 127)`, NCO +600 kHz) → `convert_out` to `CS8` → then E2E-RX steps 2–6. Thresholds: f32 ≥ 35 dB, Q15 ≥ 30 dB (CS8 limits).

### 7. Facade, suite, docs

Delegations for all new traits; fused traits via chained functions. Suite cases: `convert_out_all_formats_roundtrip`, `fir_interpolate_*`, `fm_mod_*`, `ddc_equals_chain`, `duc_equals_chain`, `fm_demod_equals_chain`, `ddc_partial_output_keeps_phase`, `e2e_fm_rx_*`, `e2e_fm_txrx_*` (quick variants with 0.1 s). Docs: `docs/kernels.md` (contracts, fused-kernel rule), `docs/timing.md` (lead time on transmit = summed group delays of interpolation filters).

## Test vectors (Python)

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| convert_out | `<format>_from_c32_256` for every integer format | 200+i | random C32 in (−1.05, 1.05) incl. NaN, ±1 exactly, values at ±½ LSB boundaries | Python implementation of §1 (float path) | bit-exact |
| convert_out | `<format>_from_cq15_256` | 220+i | random raw Q15 incl. −32768, 32767 | Python integer path | bit-exact |
| fir_interpolate | `l5_n241_c32_4000` | 90 | `5·firwin(241, 0.18)` f32, noise σ 0.3 | `upfirdn(h, x, 5, 1)[:5·len]` | SNR ≥ 110 |
| fir_interpolate | `l5_n241_cq15_4000` | 91 | taps divided by 5 so that every polyphase sum is ≤ 1 (document the gain loss of 1/5) | integer model | bit-exact |
| fm_mod | `dev75k_fs240k_c32_8192` | 92 | real noise σ 0.3, float32; k = 75e3/240e3·2^32 | `phase = (cumsum(floor(x·k + 0.5)) mod 2^32)`, `exp(j·2π·phase/2^32)` | SNR ≥ 120 |
| fm_mod | `dev75k_fs240k_cq15_8192` | 93 | Q15 input | same from `x = q/32768` | SNR ≥ 80 |

## Tests

1. Golden tests for all cases.
2. `convert_out` ∘ `convert` round trip: for every format, decoding then encoding raw bytes reproduces them exactly (where the format is not lossy); encoding then decoding stays within ½ LSB.
3. Fused = chained, bit for bit, for random chunk plans and small output buffers (phase continuity).
4. `fm_mod` → `quad_demod` recovers the input × (2π·deviation/fs) (f32 within 1e-4 after the first sample).
5. E2E-RX and E2E-TX→RX meet their thresholds.
6. Meters self-test: a synthetic tone with known SNR (tone + noise of known power) measures within ±0.5 dB.
7. Suite cases pass on the host (quick and full).

## Acceptance

R1.2 commands; golden check; `cargo test -p sdrk-testkit --features std -- --include-ignored` if the long E2E tests are marked `#[ignore]` for the default run (they must not be ignored in CI: add a CI step running them).

## Deliverables checklist

- [ ] `convert_out`, `fir_interpolate` (+ real), `fm_mod`
- [ ] Fused traits, chained functions, `max_input_for_output`
- [ ] Testkit `fm`, `meters`, `design_helpers`
- [ ] E2E tests f32 and Q15
- [ ] Golden cases, suite cases, docs
