# M23 — Real-sampling devices (halfband, real to complex)

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M22](M22-audio-filters.md) · Next: [M24](M24-fm-stereo.md)

## Goal

Receive from devices that deliver real samples (Airspy R2 and Mini, direct-sampling front ends, sound-card IF receivers): the decimator `halfband_decimate` (complex and real), the converter `real_to_complex` (fs/4 shift and halfband decimation in one kernel), real formats in the `convert` block, the blocks `halfband` and `real_to_complex`, their pipeline kinds, the example flow `real_input_fm_receiver` and an end-to-end FM test from a real 16-bit input.

## Why

Real-sampling devices put the whole band from 0 to fs/2 into real samples; a receiver first turns them into complex samples at half the rate. The standard way is a shift by fs/4 (multiplying by `1, −j, −1, j`, which costs no multiplication) followed by a halfband low-pass, whose every other tap is zero. Both together make the halfband decimator the cheapest way to halve a rate, which also helps every other chain (for example a cascade of halfbands in front of a `ddc`).

## Order

Part of the receive series M19–M23 (see the *Order* section of [M19](M19-nfm-squelch.md)): uses M00–M15 and M19–M22.

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

**In:** kernels `halfband_decimate`, `halfband_decimate_real` and `real_to_complex` in `sdrk-core` (states, `Fallback`, `Reference`, facade delegation); the design functions `halfband_len` and `halfband_taps` in `sdrk-blocks`; formats with `Layout::Real` in the `convert` block; the blocks `halfband` and `real_to_complex` with descriptions; their pipeline kinds and the regenerated schema; the example flow `real_input_fm_receiver` (JSON and YAML); golden cases; suite cases; the end-to-end case `e2e_real_fm_rx_*`; the SIMD paths and counts of §7; `docs/kernels.md`, `docs/timing.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/device.md`.

**Out:** halfband interpolation, CIC, device drivers (Airspy and others live in their own repos), packed 12-bit real formats beyond what M01 defines, fs/4 shifts in the other direction (`+fs/4`; a mirrored spectrum is handled by the `ddc` frequency sign).

> **Amendment 2026-10-31 (maintainer decisions, D63 to D65):** external C references for the three kernels are **in** scope: liquid-dsp 1.8.3 (already fetched by `benches/refs/build.sh`) and libairspy v1.0.10 (MIT licence; its `iqconverter_float` and `iqconverter_int16`, conversion only, without `remove_dc`; later libairspy versions are under a restrictive licence and must not be used) are built by `benches/refs/build.sh` under its pinning and licence conventions and linked into the measurement binary. The sentence 'external C references for `real_to_complex` (libairspy; backlog)' is removed from *Out*. Section 7 is amended below.

## Specification

General rules from M02/M05/M06 apply (see M19). Rate-changing kernels return `Progress` and consume input only as far as output space allows.

**Thresholds and limits are not relaxed.** If a threshold, tolerance or count limit of this milestone is not met, do not change it in code, case files or `benches/limits.json`. Measure it, record the measured value and the reason in `docs/decisions-log.md` and an open item in `docs/open-items.md`, and report it in the pull request; a changed threshold needs the maintainer's approval. The only pre-approved changes are the ones stated in this file.

### 1. Design functions (`sdrk-blocks`, `design.rs`, in f64)

```rust
/// Tap count of a halfband filter: kaiserord(attenuation_db, transition_hz, fs_hz), rounded up to the next N with N mod 4 == 3
pub fn halfband_len(transition_hz: f64, attenuation_db: f64, fs_hz: f64) -> Result<usize>;
/// Halfband taps of length taps.len() (N mod 4 == 3, N ≥ 3) with a Kaiser window of the given beta
pub fn halfband_taps(beta: f64, taps: &mut [f64]) -> Result<()>;
```

`halfband_taps` (normative): `N = taps.len()`, `m = (N − 1)/2` (odd). `h = firwin_lowpass(fs/4, fs, Kaiser { beta })` for `fs = 1` (M09 §3); then set `h[m ± 2i] = 0.0` exactly for `i ≥ 1` (they are about 1e-17 from `sin(kπ)`), and scale so that `Σ h = 1`. The taps are symmetric, the outermost taps are non-zero, and about half of the taps are zero. The usable band after decimation is `±(fs/4 − transition_hz/2)` of the input rate.

### 2. `halfband_decimate` and `halfband_decimate_real` (C32/f32 tolerance ≥ 110 dB, CQ15/Q15 bit-exact)

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

### 3. `real_to_complex` (C32 tolerance ≥ 110 dB, CQ15 bit-exact)

```rust
pub struct RealToComplexState<T: ComplexSample, Taps: AsRef<[T::Real]>, Hist: AsRef<[T]> + AsMut<[T]>> { /* halfband taps, hist (len N − 1), n mod 4, phase */ }
// new(taps, history) -> Result: the halfband rules of §2; reset(): n = 0, history cleared
// Timing: ratio 1/2, group delay Delay::new(N − 1, 2)
pub trait RealToComplex<T: ComplexSample> {
    fn real_to_complex<Taps, Hist>(&self, s: &mut RealToComplexState<T, Taps, Hist>, input: &[T::Real], output: &mut [T]) -> Result<Progress>;
}
```

Definition (normative): with `n` counted from construction or reset, `u[n] = x[n]·(−j)^n`, i.e. `(x, 0)`, `(0, −x)`, `(−x, 0)`, `(0, x)` for `n mod 4 = 0, 1, 2, 3` (Q15: the negation saturates, `−(−32768) = 32767`, document); the output is `halfband_decimate(u)` with the same taps. A real input component at frequency `f` (0 … fs/2) appears in the complex output at `f − fs/4` (the output rate is `fs/2`; output 0 Hz is input `fs/4`); the mirror component at `−f` lands outside the pass band and is removed. The fallback may use that every `u[n]` has one zero component (it is the reason for the kernel), but its result must equal the definition: CQ15 bit-exact, C32 summed in ascending tap index without FMA.

### 4. Real formats in the `convert` block

The `convert` block (M09 §6) accepts formats with `Layout::Real` (for example `RS16`, M01) and then outputs `f32`/`Q15` through `convert_real` (M05): struct `ConvertRealBlock<T>`, the same parameter `format`. The pipeline kind `convert` and the automatic conversion of a raw input (M10) choose the real variant by the layout of the format; the port type of the node is then `F32`/`Q15`. `docs/device.md` and `docs/pipeline.md` describe real inputs.

### 5. Blocks

The construction pattern, descriptions, storage and parameter rules of M09 §4–§5 apply unchanged.

| Kind | Struct | In → Out | Parameters (unit, mutability, default) | Implementation |
| --- | --- | --- | --- | --- |
| `halfband` | `HalfbandBlock<T>` (complex) and `HalfbandRealBlock<T>` (real) | T → T, rate / 2 | `transition_hz` (Hz, build, default 0.1·fs), `attenuation_db` (dB, build, 60), `max_taps` (I64, build, 1023) | `halfband_len` + `halfband_taps` (β = `kaiser_beta(attenuation_db)`), quantized like the `fir` block (zeros stay exactly zero), `halfband_decimate`/`halfband_decimate_real` |
| `real_to_complex` | `RealToComplexBlock<T>` | f32 → C32, Q15 → CQ15, rate / 2 | `transition_hz` (Hz, build, default 0.1·fs), `attenuation_db` (dB, build, 60), `max_taps` (I64, build, 1023) | the same design, `real_to_complex` |

`docs/blocks.md`: the output centre of `real_to_complex` is input `fs/4`; a station at real frequency `f` is tuned with a `ddc` frequency of `f − fs/4`. Several `halfband` nodes in a row decimate by powers of two before a `ddc`.

### 6. Pipeline kinds, flow and testkit

- Kinds `halfband` (complex and real inputs; `kernels()` = `["halfband_decimate", "halfband_decimate_real"]`) and `real_to_complex` (`["real_to_complex"]`). `plan` returns `Figures` exactly like `FmDemodKind::plan` (`crates/pipeline/sdrk-pipeline/src/kinds.rs`): `rate_out` and `memory_bytes` from the block's `rate_out` and `storage_bytes`, `delay` from its timing, `min_input` from `min_input()`, and `cost` per input sample by this formula, which is also added to the cost table of `docs/pipeline.md`: `halfband` and `real_to_complex` **`(N + 3) / 4`** (the `(N + 3)/2` non-zero taps once per two input samples).
- the `convert` kind gets the real variant; `cargo xtask schema` regenerates the schema (version unchanged). Add the kinds to the WASM declarations if they list kinds.
- `examples/flows/real_input_fm_receiver.json` + YAML: input `rs16` (real) at 4.8 MS/s → `convert` (real) → `real_to_complex` (→ 2.4 MS/s complex, centre = 1.2 MHz of the input) → `ddc` (frequency `f − 1.2 MHz`, decimation 10) → `fm_demod` (as `fm_receiver`) → output `f32`.
- Suite cases (Fallback vs Reference, invariance with 20 chunk plans): `halfband_c32`, `halfband_cq15`, `halfband_real_f32`, `halfband_real_q15`, `real_to_complex_c32`, `real_to_complex_cq15`, `extremes_m23`; block cases `block_halfband_alias`, `block_real_to_complex_tone`.
- **E2E-REAL (`e2e_real_fm_rx_{f32,q15}`):** the FM signal of E2E-RX (M08: deviation 75 kHz, audio 1 kHz) placed at a real IF of 1.8 MHz at 4.8 MS/s (`Re{fm_baseband(…) · e^{j2π·1.8e6·n/fs}}`, amplitude 0.45), plus real noise σ 0.005, encoded to `RS16` → the flow's chain with `ddc` at +600 kHz. SINAD at 1 kHz over the last 50 %: **f32 ≥ 40 dB, Q15 ≥ 35 dB** (as E2E-RX; initial thresholds, M07 rule). Quick variant 0.1 s, full variant 1.0 s.
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
| `halfband_decimate` (C32, CQ15) | yes | an FIR: the dot products of the non-zero taps vectorize like `fir_decimate` (M13); C32 may use FMA on AVX2, AVX-512 and NEON like the other FIR kernels (D5, extended to the halfband family; record) and is then a tolerance kernel within 110 dB of the reference; CQ15 stays bit-exact | against our `fir_decimate` with factor 2 and the same (expanded) taps at the same level (`Reference::OwnKernel`, see below): `release` with the per-entry limit 0.60 (it does half the multiplications) |
| `halfband_decimate_real` (f32, Q15) | yes | as above for real lanes | against our `halfband_decimate` per value, `info` (no real `fir_decimate` path exists in the backend) |
| `real_to_complex` (C32, CQ15) | yes | the shift is a sign pattern on lanes and the filter is the halfband | against our `halfband_decimate` on the shifted samples at the same level (`Reference::OwnKernel`, `release` with the default limit 1.10) |

**Note 2026-10-28 (D32):** a comparison with one of our own kernels is not an established reference and no fair gate. The maintainer decided on 2026-10-31 (D63): all `Reference::OwnKernel` comparisons are `info` and the gates are external; see the *Amendment 2026-10-31* below. The `Accelerated`/`Count` columns of the table above are replaced by the amendment.

**Comparison with an own kernel (extension of `xtask count`, normative):**

- `xtask/src/count.rs`: a new variant `Reference::OwnKernel { kernel: &'static str }`; `implementation()` returns `"ours"`. The measurement binary (`benches/kernels`) gets the comparison workloads `fir_decimate_x2_halfband_taps` (`fir_decimate` with factor 2 and the expanded halfband taps of the kernel's workload) and `halfband_decimate_shifted` (`halfband_decimate` on the input after the `(−j)^n` shift, the shift itself done outside the counted region), counted with the same differential method and level as the kernel.
- `benches/limits.json`: an entry may carry an optional `"limit"` (number). It replaces `_limits.release` (status `release`) or `_limits.dev` (status `dev`) for that entry; `goal` stays `_limits.goal` unless the entry gives `"goal"` as well. The `halfband_decimate` entries use `"limit": 0.60`.
- `benches/results/<platform>.json`: `compared_with` is `"ours:<kernel>"`; the verdict and the limit used are written as for the other entries.
- Unit tests of `xtask` for the parser (an entry with and without `limit`, a wrong type) and for the verdict with a per-entry limit; `docs/performance.md` documents the new reference kind and the field.

> **Amendment 2026-10-31 (maintainer decisions, D63 to D65; replaces the `Count` column of the table above, the per-entry `limit` 0.60 and the release status of the `Reference::OwnKernel` entries):**
>
> 1. **Count references are external.** `Reference::OwnKernel` comparisons are kept as work-saving evidence only, with status `info` (for example `halfband_decimate` <= 0.60 of `fir_decimate` with factor 2 and the expanded taps; `real_to_complex` against `halfband_decimate` on the shifted samples). They are no gate. The gates are:
>
>    | Kernel (type) | Level | Gate (reference) | Status |
>    | --- | --- | --- | --- |
>    | `halfband_decimate` (C32) | `sse2`, `sse4.2` | liquid-dsp 1.8.3 `resamp2_crcf` (`decim_execute`) with `dotprod_crcf`, SSE | `release` |
>    | | `avx2` | the same with `dotprod_crcf`, AVX | `release` |
>    | | `avx512` | the same, AVX-512F | `release` |
>    | | `neon` | the same, NEON | `release` |
>    | | `simd128` | the same, liquid portable code built with `-msimd128` (precedent: CMSIS-DSP generic C at `simd128`, D15) | `release` as D15, unless the compiler leaves it scalar; then `info` with a decision-log row |
>    | `halfband_decimate` (C32) fallback | all | liquid-dsp portable `resamp2_crcf` (`dotprod.proto.c`), as `Family::FallbackVsLiquid` | `release` |
>    | `halfband_decimate_real` (f32) | `sse4.2` | liquid-dsp `resamp2_rrrf` with `dotprod_rrrf`, SSE (its SSE path needs SSE3 `hadd`, so the reference starts at `sse4.2`) | `release` |
>    | | `sse2` | liquid portable `resamp2_rrrf` (no SSE path without SSE3) | `release` |
>    | | `avx2`, `avx512`, `neon`, `simd128` | `dotprod_rrrf` AVX, AVX-512F, NEON; `simd128` as above | `release` (`simd128` as above) |
>    | `halfband_decimate_real` (f32) fallback | all | liquid portable `resamp2_rrrf` | `release` |
>    | `real_to_complex` (C32) | `sse4.2`, `avx2`, `avx512`, `neon`, `simd128` | liquid-dsp `firhilbf_decim_execute` (delay branch plus one real dot product over (N + 1)/2 taps and the sign toggle) with `dotprod_rrrf` of the level | `release` |
>    | | `sse2` | additionally libairspy v1.0.10 `iqconverter_float` (SSE2) as a second gate; liquid portable `firhilbf` as the first | `release` (both must pass) |
>    | `real_to_complex` (C32) fallback | all | liquid portable `firhilbf`; libairspy generic `iqconverter_float` as information | `release` |
>    | `real_to_complex` (CQ15) | all | libairspy v1.0.10 `iqconverter_int16` (lower accuracy class: truncating `>> 15`, `i32` accumulator, halves the odd samples) | **pending, measured as `info`** (see 4) |
>    | `halfband_decimate` (CQ15), `halfband_decimate_real` (Q15) | all | no library for the SIMD branch: `info` with an efficiency verdict (D33), as the Q15 biquads (D21) | `info` |
>    | the same, fallbacks | all | a CMSIS-DSP 1.18.0 composition in the measurement binary (polyphase split, `arm_fir_q15` with the compacted odd taps, plus the centre tap), `Family::FallbackVsCmsis`-like | `release` (risk: open item I20-1, the CQ15 FIR fallback engine) |
>
>    **Workload:** N = 47 taps (24 non-zero odd taps plus the centre tap; equals liquid `m = 12` and the libairspy 47-tap kernel). Our taps are injected into every reference (liquid's `resamp2` and `firhilb` filter coefficients, libairspy's `HB_KERNEL_FLOAT`/`HB_KERNEL_INT16` tables), so that accuracy (`count accuracy`) is comparable.
> 2. **libairspy v1.0.10 moves from *Out/backlog* into scope as a reference:** conversion only, without `remove_dc`. `benches/refs/build.sh` pins the version and licence like the other references (MIT; the notice goes into the licence list of `docs/performance.md`).
> 3. **Symmetric tap pre-add** (`h[k] * (x[m-k] + x[m+k])`) is allowed in the C32/f32 tolerance SIMD paths (contract >= 110 dB, a regression guard in the golden case, at most 1 dB below the reference). The fallback stays in ascending tap order without FMA. Pre-add in integers (Q15/CQ15) is exact and allowed anyway.
> 4. **Status of the `real_to_complex` CQ15 gate is pending.** The entry is measured against libairspy `iqconverter_int16` and recorded provisionally as `info`. The maintainer decides whether it becomes `release` with an accuracy waiver (like D58) or stays `info` (like D56); the measured numbers go into `docs/open-items.md`.
>
> The mechanism of the original section stays: `Reference::OwnKernel { kernel }` (`implementation()` returns `"ours"`), the optional per-entry `"limit"` in `benches/limits.json`, `compared_with: "ours:<kernel>"`, the comparison workloads `fir_decimate_x2_halfband_taps` and `halfband_decimate_shifted`, the unit tests and the documentation, now for `info` entries (the per-entry `limit` 0.60 is recorded as the efficiency target of the `info` entry, it does not fail the command). New: `Reference::Liquid` entries for the halfband/Hilbert kernels, `Reference::Libairspy`, and a liquid resampler/Hilbert build per level in `benches/refs/build.sh`.

### 8. Documentation

`docs/kernels.md` (contracts of the three kernels, the frequency mapping of `real_to_complex`), `docs/timing.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md` (table, tuning rule, halfband cascades), `docs/pipeline.md` and `docs/device.md` (real inputs), `examples/README.md`, `CHANGELOG.md`, `docs/decisions-log.md`.

## Test vectors (`tests/golden/sdrk_golden/cases/halfband.py`, `real_to_complex.py`, `design.py`)

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| design | `halfband_n{11,23,63}` | 0 | β = `kaiser_beta(60)` | `firwin(N, 0.25, window=('kaiser', β), fs=1)`, even offsets set to 0, scaled to Σ = 1 | max_abs_err ≤ 1e-12 |
| design | `halfband_len_table` | 0 | 10 pairs (attenuation 40 … 100 dB, transition 0.02 … 0.2) | `kaiserord` rounded up to N mod 4 == 3 | exact |
| halfband_decimate | `n23_c32_8192` | 430 | `halfband_taps` N 23, β = `kaiser_beta(60)` → f32; noise σ 0.3 | `lfilter(h, 1, x)[::2]` | SNR ≥ 110 |
| halfband_decimate | `n23_cq15_8192` | 431 | taps and input quantized (the zero taps stay 0) | integer model of `fir_decimate` with factor 2 | bit-exact |
| halfband_decimate_real | `n23_f32_8192` / `n23_q15_8192` | 432 / 433 | real noise | as above | SNR ≥ 110 / bit-exact |
| halfband_decimate / halfband_decimate_real | `n23_cq15_corners_2048` / `n23_q15_corners_2048` | 438 / 439 | added in review (D97): full-scale windows aligned with the tap signs, runs of −32768 and +32767, full-scale alternations | integer model, both saturation limits occur | bit-exact |
| real_to_complex | `n63_c32_16384` | 434 | real noise σ 0.2 plus two real tones at 0.11·fs and 0.37·fs (amplitude 0.3); taps N 63 | f64: `u = x·(−j)^n`, `lfilter(h, 1, u)[::2]` | SNR ≥ 110 |
| real_to_complex | `n63_cq15_16384` | 435 | quantized (input includes −32768 to exercise the saturated negation) | integer model of §3 | bit-exact |

All stateful Rust golden tests run whole and in random chunks (ChunkPlan seed = case seed).

## Tests

1. Golden tests for every case, whole and chunked; invariance for the three kernels and both blocks (20 plans each).
2. `halfband_decimate` with CQ15/Q15 equals `fir_decimate` with factor 2 and the same taps bit for bit; with C32/f32 for finite inputs as well (a zero result may differ in sign).
3. `real_to_complex`: a real tone at `f` appears at `f − fs/4` at the output rate (`estimate_frequency` within 1e-6·fs), its mirror is at least `attenuation_db − 3` dB below; `n mod 4` continues across calls and restarts at `reset`.
4. Validation: taps with `N mod 4 ≠ 3` or a non-zero even offset give `InvalidParameter`.
5. Block `halfband`: a tone above `fs/4 + transition_hz/2` is attenuated by at least `attenuation_db − 3` dB after decimation (alias test); timing matches an impulse (±0.5 sample).
6. `convert` with `RS16` equals `convert_real` (M05); the pipeline inserts the real automatic conversion for a real raw input.
7. Extremes (`extremes_m23`): no panic on silence, full scale, NaN and ±∞ (f32), and the Q15 corners.
8. SIMD: `backend_vs_fallback_halfband` and `backend_vs_fallback_real_to_complex` pass at every level (CQ15 bit-exact, C32 within the contract); the counts pass their limits.
9. Block rules: descriptions, exact arena, no block struct larger than 512 bytes.
10. Pipeline: `real_input_fm_receiver` validates, builds and runs on the E2E-REAL signal with the same thresholds; schema check; E2E-REAL meets its thresholds; the suite passes on the host and in the emulators (quick profile).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64`, `x86_64-avx512`, `aarch64` and `wasm32`; `cargo xtask emu --target qemu-cortex-m4` (quick profile within its timeout).

## Deliverables checklist

- [x] Kernels `halfband_decimate`, `halfband_decimate_real`, `real_to_complex` with states, fallback, reference, delegation
- [x] Design functions `halfband_len`, `halfband_taps`
- [x] Real formats in the `convert` block and the automatic conversion
- [x] Blocks `halfband`, `real_to_complex`, pipeline kinds, regenerated schema; flow `real_input_fm_receiver` with YAML twin
- [x] SIMD paths for the three kernels on every level, `Reference::OwnKernel`, the efficiency target `own_target` (instead of the per-entry `limit`, D96) and the second `sse2` gate `real_to_complex_airspy` (D95) in `xtask count`, counts and limits (all `release` entries pass at all levels; the `CQ15` `real_to_complex` entries are provisionally `info`, I23-1)
- [x] Golden cases (seeds 430–435, 438 and 439 (saturation corners of the `Q15` halfband kernels) and the design cases), suite, invariance, extremes, E2E-REAL
- [x] `docs/kernels.md`, `docs/timing.md`, `docs/backends.md`, `docs/performance.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/device.md`, `examples/README.md`, CHANGELOG, decisions log

## Status (2026-10-31)

Complete: all deliverables above, the tests 1 to 10, and the acceptance commands (fmt, clippy `-D warnings`, `cargo test --workspace --all-features`, `cargo xtask check`, `cargo xtask bare-metal`, the wasm tests, the golden check, `cargo xtask schema --check`, the quick profiles of the Cortex-M4 emulator; the counts at `x86_64`, `x86_64-avx512`, `aarch64` and `wasm32`, with `cargo measure` at `avx2`). Every gated count entry (kernels and fallbacks) is within its limit at every level and every accuracy rule holds (D81 to D94, `docs/performance.md`).

Provisional / open (see `docs/open-items.md`): I23-1 (status of the `CQ15` `real_to_complex` entries against libairspy: `info` until the maintainer decides; the measured numbers are there), I23-2 (confirmation of D88: the `f32` fallback adds the centre tap last), I23-3 (the work-saving evidence of 0.60 is missed on `neon` and `simd128`; no `neon`/`simd128` specific path). Not run in this environment: the ESP32-P4 and ESP32-S3 emulator targets.
