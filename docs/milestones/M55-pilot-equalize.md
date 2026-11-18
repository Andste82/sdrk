# M55 — Kernel `pilot_equalize`: OFDM pilot-based channel equalization

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M07](M07-fft.md), [M45](M45-channelizer.md) (lane ports; through it M19 and M38) · Next: [M56](M56-fwht.md)

> **Status: planned** (written 2026-11-08; revised 2026-11-17 after an algorithm review (maintainer decision)). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone, and the last of the DSSS/OFDM kernels together with `fwht` (M56).

> **Revision 2026-11-17.** The shared conventions §A to §D cited in this file are defined in [CONVENTIONS-M49-M56.md](CONVENTIONS-M49-M56.md). The maintainer accepted every recommendation of the algorithm review of M49 to M56 (decision 6 for this milestone, decision 9 for SIMD; shared conventions §D). What changed:
> 1. **Error corrected, the CPE estimate (§2 step 2).** The old formula `φ = arg(Σ_p Y[p]·conj(Hp·Pval))` with `Hp = Y[p]/Pval` is `arg(Σ_p |Y[p]|²) = 0` for every input, so the step did nothing, and test 2 and the golden case `keyed_pilots_128_cpe` would have frozen a no-op bit for bit. The common phase is now measured against the previous smoothed estimate, `S = Σ_p Y[p]·conj(Hs_prev[p]·Pval)`, and removed from this symbol's pilot estimates before the smoothing, so it is not blended into `Hs`.
> 2. **CPE factor without `arg` and tables (§2 step 2).** `c = conj(S)/|S|` with one `sqrt_f32` and one division per symbol, folded into the pilot estimates and the equalization. No `atan2`, no phase conversion, no `rotate` table.
> 3. **Reciprocals instead of per-carrier divisions (§2.1, step 4).** The interpolation fraction uses `1/(p_b − p_a)` once per pilot segment; every complex division is `conj(d)·(1/mag_sq(d))`, one f32 division per carrier instead of two or three. The contract fixes these bits.
> 4. **Feedback state cleaned (§2 step 4).** `Hs` passes `flush_denormal` after each update (R6.3), and a non-finite component becomes 0.
> 5. **One defined zero threshold (§2.1, §3).** A divisor is degenerate when its `mag_sq` is not in `[1e-30, f32::MAX]` (NaN included); the same rule for pilot values, `Hs` and the CPE sum. A pilot with a non-finite input or a degenerate value is skipped and counted instead of reaching the state.
> 6. **SIMD in scope (§4, §6).** Every lane-parallel part gets a SIMD path on every level in this milestone, bit-identical to the fallback; the contract fixes element-wise formulas and the order of the CPE sum. "Not accelerated" is removed. Fair reference rows are named.
> 7. Tests, golden cases (one new case, seed 756), acceptance and the deliverables checklist follow these changes. `CQ15` stays out of scope (the review agrees), with a concrete backlog note.
> 8. **Test bounds corrected (test 2, §5).** The interpolation tolerance is absolute (a few ulp of the larger pilot estimate per component) instead of "1 ulp per component", which cancellation at a zero crossing breaks; the `cdiv` bound is stated for quotients in the normal range; the CPE assertion of `keyed_pilots_128_cpe` compares the factor with the phase relative to the first symbol, modulo 2π, on a channel smooth enough for the 0.01 rad bound.
> 9. **Cross-milestone consistency (M49–M56 pass).** The contract row and test 7 compare an output that is NaN by class (D296), as every `C32` kernel of the series.

## Goal

A kernel `pilot_equalize` that equalizes one OFDM symbol in the frequency domain from its pilot subcarriers: a least-squares channel estimate at the pilot positions (`H = Y / P`), linear interpolation across frequency to the data subcarriers, a one-pole smoothing across OFDM symbols, a one-tap equalization of every data subcarrier (`X̂ = Y / H`), and a common-phase-error (CPE) correction measured against the channel estimate of the previous symbols. It is the receive-side channel correction for the OFDM demodulator, following the forms of GNU Radio's `ofdm_chanest_vcvc` (LS estimate `H = Y/X` on a reference symbol) and `ofdm_equalizer_simpledfe` (per-carrier update with a forgetting factor α). Each complex division costs one f32 division (a reciprocal), and fallback, reference and the SIMD paths of every level give the same bits.

## Why

An OFDM receiver turns each symbol into `N` subcarriers with the FFT, but the channel multiplies every subcarrier by an unknown complex gain. Pilots — known symbols on known subcarriers — let the receiver measure that gain and divide it out, which is the whole point of OFDM (a flat one-tap equalizer per subcarrier instead of a long time-domain equalizer). For the LPI work the pilots may hop from symbol to symbol (keyed positions), so the estimate must interpolate from whatever positions are given, not a fixed comb.

## Order

M55 depends on M07 only in spirit (it processes the FFT output; it does not call `fft`). It is independent of the other kernels of the series and can run in parallel with every milestone after M07. The OFDM demodulator milestone that uses it also needs `fft`/`ifft` on lanes and the lane plumbing of M45. It follows the shared conventions §D of the series (SIMD rule, reduction order); it needs no new dependency.

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
| `crates/proto` | `sdrk-proto` | `sdrk-device` | yes |
| `crates/blocks` | `sdrk-blocks` | facade, core, proto | yes |
| `crates/pipeline` | `sdrk-pipeline` | blocks, proto, facade, core | no (std/alloc allowed) |
| `crates/io` | `sdrk-io` | core (types only) | no |
| `crates/testing` | `sdrk-testkit` | any library crate above | `no_std` by default; feature `std` adds file I/O and proptest helpers |
| `bindings/*`, `examples/*`, `targets/*`, `benches/*`, `xtask` | — | anything | no |

Additional rules:
- Within `crates/core` the rank is by crate name: `sdrk-device` (rank 0) < `sdrk-core` (rank 1); `sdrk-core → sdrk-device` is allowed, the reverse is not. All other layers are one rank per folder.
- Backends never depend on each other. Only the facade `sdrk` knows the backends.
- `sdrk-proto` (from M37 on) holds protocol logic without signal processing (bit and byte framing, checksums, packet types); it depends on `sdrk-device` only, and nothing below `crates/blocks` depends on it.
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
6. Each case (JSON + all its `.bin` arrays) stays at or below 512 KiB; all vectors together at or below 32 MiB. Prefer N ≤ 4096 samples; use longer inputs only when the kernel needs them (long filters, decimation).
7. Tests that use `sdrk-testkit` live in integration tests (`<crate>/tests/*.rs`) or in `crates/testing/sdrk-testkit/tests/`, never in `#[cfg(test)]` unit modules of a crate the testkit itself depends on (a dev-dependency cycle would compile two copies of that crate and produce confusing trait errors).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.




## Scope

**In:** kernel `pilot_equalize` (`C32`) in `sdrk_core::kernels::ofdm_eq` with `PilotEqualizeState`, Fallback, Reference, facade delegation, contract, `docs/kernels.md`; SIMD paths in `sdrk-backend-simd` on every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`), bit-identical to the fallback (shared conventions §D, §6), with `docs/backends.md`; golden cases; unit, invariance, extreme and SIMD tests; suite cases; `cargo xtask count` and `cargo measure` entries with the reference rows of §6; CHANGELOG, decisions log, `docs/performance.md`; the block `PilotEqualizeBlock` and the pipeline kind `pilot_equalize` (§7) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the composite `ofdm_demod` and the lane kinds `fft`, `to_lanes`, `subcarrier_demap` (later milestone; names reserved); the FFT and guard removal (`to_lanes`, `fft` on lanes — later milestone); decision-feedback updating from data decisions (the `simpledfe` data path; backlog, because it needs the constellation and couples to the slicer); superimposed-pilot estimation (backlog); `CQ15` (backlog, OFDM dynamic range: a per-carrier division in `Q15` needs a block exponent or a normalized reciprocal; until then a `CQ15` FFT output is converted with `convert`, and a later `CQ15` variant would carry a per-symbol block exponent). Carried in: the LS pilot estimate, frequency interpolation, time smoothing, one-tap equalization, common-phase correction against the previous estimate — the pieces a basic OFDM receiver needs.

## Specification

General rules of M19 apply. The kernel processes one OFDM symbol per call step but keeps a channel estimate across symbols (the time smoothing is a one-pole feedback per subcarrier), so as for the other feedback kernels the arithmetic is fixed and `Reference` follows it step by step (bit-exact, float32-model rule, D111); an f64 run is a regression guard. The SIMD paths of every level give the fallback bits (§4).

### 1. Layout

One OFDM symbol is `N` subcarriers (`N` a power of two 16…65536), given as the FFT output in natural order (subcarrier 0 = DC). The subcarrier roles are given with each call by index lists:
- `pilots: &[u32]` — the pilot subcarrier indices for this symbol, with their known complex values `pilot_values: &[C32]` (same length). (Keyed/hopping pilots: the block passes a different `pilots`/`pilot_values` per symbol; the kernel takes them as inputs.)
- `data: &[u32]` — the data subcarrier indices to equalize and output, in any order.
- subcarriers in neither list (DC, guard/null carriers) are not output: the output holds the equalized data carriers only, in the order of `data` (a flag to pass the nulls would have no effect on that output, so there is none). Their `Hs` is still updated (step 4), because hopping pilots may land on any carrier in a later symbol.

### 2. Per symbol

All arithmetic is f32 without FMA (R6.5); each product is rounded before the sum or difference that uses it. `T = 1e-30` (the R6.3 value) is the one zero threshold of the kernel.

#### 2.1 Building blocks (fixed formulas)

- `cmul(a, b) = (a.re·b.re − a.im·b.im, a.re·b.im + a.im·b.re)`, both products before the difference or sum (the product of `dot_complex` and of the `mul` fallback). The kernel writes it itself; it does not call the `mul` backend entry, which fuses products at some levels (D326; shared conventions §D.2).
- `conj(a) = (a.re, −a.im)` (an IEEE sign flip, exact).
- `mag_sq(d) = d.re·d.re + d.im·d.im` (the `mag_sq` rule).
- `valid(m)`: `T ≤ m` and `m ≤ f32::MAX`. It is false for NaN, for `+∞` (overflow of `mag_sq` for `|d|` above about `1.8e19`) and for everything below `T` (`|d|` below `1e-15`). Two ordered comparisons, one lane mask in SIMD.
- **Division with an optional phase factor** `cdiv(y, d, c)`: `m = mag_sq(d)`; when `!valid(m)` the division is degenerate and the caller's rule applies. Otherwise `r = 1.0 / m` (the one f32 division), `q = conj(d)` when no CPE factor is active and `q = cmul(c, conj(d))` when one is, `w = (q.re·r, q.im·r)`, result `cmul(y, w)`. This is `y·c/d` up to rounding.
  - The reciprocal is applied to `q` before the product with `y`: `|w| = |c|/|d| ≤ 10^15`, so the product overflows only where the quotient itself is above `f32::MAX` (multiplying `y·q` first would overflow already for `|y|·|d|` above `f32::MAX`).
  - An inactive factor is skipped, never applied as a product with `(1, 0)`: that product changes signed zeros (the reasoning of shared conventions §A).
  - Divisions are correctly rounded IEEE divisions on every level (`divps`, `vdivq_f32`, `f32x4.div`); reciprocal estimates (`rcpps`, `vrecpeq_f32`) and Newton steps are not allowed, because they give other bits.
- `clean(x)`: `x` when `T ≤ |x| ≤ f32::MAX`, else `+0.0`. This is `flush_denormal` (R6.3) plus "a non-finite value becomes 0"; a select with one mask in SIMD.

#### 2.2 Steps

Input: the symbol `Y` (`N` carriers), the pilot list `(p_i, P_i)`, `i = 0 … n_p − 1`, and the data list. `Hs_prev` is `Hs` before this call.

1. **Valid pilots.** Pilot `i` is valid when `valid(mag_sq(P_i))` and both components of `Y[p_i]` are finite. An invalid pilot is skipped in steps 2 to 4 and adds 1 to `degenerate`. A zero pilot value or a NaN or ∞ on a pilot carrier therefore never reaches the state.
2. **Common phase (CPE), optional (`cpe: bool`).** Estimated only when a previous estimate exists (not on the first symbol after `reset`, unless `seed_taps` was called) and at least one pilot is valid; otherwise no factor is active (`φ = 0`).
   - Per valid pilot: `g_i = cmul(Hs_prev[p_i], P_i)` (the pilot as the previous estimate predicts it) and the term `e_i = cmul(Y[p_i], conj(g_i))`.
   - `S = Σ e_i` in the reduction order of shared conventions §D.2 (the `dot_complex` order of M43, D801): eight partial sums per component, each starting at `+0.0`; term `i` (its position in the pilot list) goes to partial sum `i mod 8`, and the term of an invalid pilot is omitted (its partial sum is not touched); then the fixed tree `s[k] = acc[k] + acc[k+4]`, `t[k] = s[k] + s[k+2]`, `t[0] + t[1]`. Bit for bit this is `dot_complex` of the `Y[p_i]` with the weights `conj(g_i)` (`a − (−b)` and `a + b` are the same IEEE operation; checked with NumPy float32 on 20000 random pairs, no difference).
   - `m_S = mag_sq(S)`. When `valid(m_S)`: `k = 1.0 / sqrt_f32(m_S)` and the factor `c = (S.re·k, −S.im·k)`, which is `conj(S)/|S| = exp(−jφ)` with `φ = arg S`; the factor is active. When `!valid(m_S)` (`S` zero, tiny or overflowed) no factor is active.
   - This is the usual pilot-based common-phase tracking against the channel estimate of earlier symbols (IEEE 802.11a receivers track the pilot phase against the preamble estimate; Speth, Fechtel, Fock and Meyr, "Optimum receiver design for wireless broad-band systems using OFDM", part I, IEEE Trans. Commun. 1999, and "Optimum receiver design for OFDM-based broadband transmission", part II, 2001). It costs one `sqrt_f32` and one division per symbol and needs no `atan2`, no phase-to-`u32` conversion and no `rotate` table.
   - The factor is applied to this symbol's pilot estimates (step 3) and to its data (step 5). The common phase is thus removed **before** the estimate enters the one-pole average and is not blended into `Hs` with weight `1 − α`; `Hs` keeps the phase reference of the earlier symbols.
   - With `α = 0` each symbol's estimate replaces `Hs`, so the factor cancels between step 3 and step 5 up to rounding: CPE is redundant then. It is allowed and documented.
   - (CPE tracking is standard in OFDM where the pilots are too sparse to catch a per-symbol phase; logged as an addition over the GNU Radio equalizers, which fold phase into the per-carrier H.)
3. **LS estimate at the valid pilots:** `Hp_i = cdiv(Y[p_i], P_i, c)`, the `ofdm_chanest_vcvc` estimate `H = Y/X`, derotated by the CPE factor when it is active. A non-finite `Hp_i` (overflow, `|Y[p_i]/P_i|` above `f32::MAX`) is caught by `clean` in step 4.
4. **Interpolation and smoothing, one pass over all `N` carriers.** `H` is computed per carrier and not stored. With no valid pilot in the symbol this step is skipped: `Hs` keeps its values and the first-symbol flag is unchanged.
   - **Interpolation** (real and imaginary parts separately). Let `p_a < p_b` be adjacent valid pilots (invalid ones skipped).
     - At `k = p_a`: `H[k] = Hp_a` (no arithmetic).
     - For `p_a < k < p_b`: `H[k] = Hp_a + Δ·t` per component, with `Δ = Hp_b − Hp_a` and `inv = 1.0 / ((p_b − p_a) as f32)` computed once per segment, and `t = ((k − p_a) as f32)·inv` per carrier. The integers are exact in f32 (`N ≤ 65536 < 2^24`); for a power-of-two pilot spacing `inv` and `t` are exact. `t` is computed from `k − p_a` for every carrier, never accumulated from carrier to carrier (shared conventions §D.2).
     - Below the first valid pilot and above the last, `H[k]` is the nearest valid pilot's value. GNU Radio's `ofdm_equalizer_static` holds; we interpolate between and hold at the edges — logged, with the reason that interpolation beats the nearest-pilot hold of `simpledfe` for a frequency-selective channel. With a single valid pilot, `H[k]` is that pilot's value everywhere (flat-channel / common-phase only).
   - **Time smoothing** per component: `Hs[k] = clean(α·Hs_prev[k] + β·H[k])`, both products before the sum, `α` the forgetting factor (default 0.5) and `β = 1 − α` rounded once to f32 whenever `α` is set. GNU Radio `simpledfe` (default α = 0.1) weights the *old* estimate the same way, `H ← α·H_old + (1−α)·H_new`, so our α has the same meaning (logged). On the first symbol after `reset` (no old estimate), `Hs[k] = clean(H[k])`. `seed_taps` (as `simpledfe`'s `initial_taps`) provides the old estimate instead.
   - `clean` is the R6.3 flush of the feedback state: a carrier whose input fades decays to exactly 0 instead of running through subnormals (slow on x86), and a non-finite value becomes 0 instead of poisoning `Hs` until `reset`. Every stored component of `Hs` is therefore `+0.0` or in `[T, f32::MAX]`.
5. **Equalization:** for each data index `k`, in the order of `data`, `X̂[k] = cdiv(Y[k], Hs[k], c)` with the `Hs` of step 4. When the division is degenerate (`!valid(mag_sq(Hs[k]))`, for example a null channel tap), the output is `+0` in both components and `degenerate` grows by 1. A non-finite `Y[k]` on a data carrier gives a non-finite output on that carrier only; the state is not touched.

Divisions per symbol: one per valid pilot, one per pilot segment, one per data carrier and one per symbol for the CPE factor, plus one `sqrt_f32`. The first draft needed two or three per data carrier. Checked with a NumPy float32 model of these steps (seed 1, `N = 64`, a comb of 8 pilots, a common phase growing by 0.3 rad per symbol, `α = 0.5`): the factor recovers the applied phases 0.3, 0.6, … 1.5 rad (relative to the first symbol) to four decimals and the output error stays at the interpolation error of the first symbol, while without CPE it grows from symbol to symbol; the old formula gives `φ = 0` for every symbol. This was a check of the formulas, not a golden case.

### 3. State and trait

```rust
pub struct PilotEqualizeState<..> { /* N, alpha: f32, beta: f32, cpe: bool, hs: Store (N complex), first: bool, degenerate: u64 */ }
impl PilotEqualizeState<..> {
    pub fn new(fft_len: usize, alpha: f32, cpe: bool, hs_store: Store) -> Result<Self>;
    pub fn alpha(&self) -> f32;  pub fn set_alpha(&mut self, a: f32) -> Result<()>;  // also sets beta = 1 - a
    pub fn seed_taps(&mut self, taps: &[C32]) -> Result<()>;  // initial_taps: Hs = clean(taps), first = false
    pub fn degenerate(&self) -> u64;  // invalid pilots plus degenerate data carriers since reset, saturating
    pub fn reset(&mut self);   // first = true, Hs = +0, degenerate = 0
}
pub trait PilotEqualize {
    fn pilot_equalize(&self, state: &mut PilotEqualizeState<..>,
        symbol: &[C32], pilots: &[u32], pilot_values: &[C32], data: &[u32],
        output: &mut [C32]) -> Result<()>;
}
```

- `symbol.len()` must equal `N`; `output.len()` must equal `data.len()`; `pilots.len()` must equal `pilot_values.len()` and be ≥ 1; every index in `pilots` and `data` must be `< N` and `pilots` strictly increasing; `alpha` finite in `[0, 1)`; `seed_taps` takes exactly `N` taps. Violations → `InvalidParameter`/`LengthMismatch`, nothing written.
- **Zero-division rule:** a pilot value or a smoothed `Hs[k]` whose `mag_sq` fails `valid` (below `T = 1e-30`, `+∞` or NaN) would divide by zero or overflow. The kernel skips that pilot (step 1) or outputs 0 for that data carrier (step 5) and counts it in `degenerate`, rather than producing NaN — documented, because a null channel tap is a real case.
- Timing: one symbol in, `data.len()` subcarriers out; the kernel is a per-symbol transform with state. On lanes (later milestone) it is one lane step per call step. Group delay 0 (frequency domain).
- Storage: `Hs` (`N` complex) and the small state. `storage_bytes` exact. No scratch buffer: `H` and the pilot estimates are computed during the pass (implementations may keep the two estimates of the current segment in registers).

### 4. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `pilot_equalize` | `kernels::ofdm_eq::PilotEqualize` | `C32` | bit-exact | the float32 model of §2: pilot validity, CPE factor `conj(S)/\|S\|` from `S = Σ Y[p]·conj(Hs_prev[p]·P)` in the `dot_complex` order (D801, one `sqrt_f32` and one division per symbol, none when no previous estimate), division `cmul(y, q·(1/mag_sq(d)))` with the factor folded into `q`, interpolation with `1/(p_b − p_a)` per segment and `t` from `k − p_a`, one-pole smoothing with `β = 1 − α`, `clean` of `Hs` (`flush_denormal`, non-finite → 0), zero threshold `mag_sq ∉ [1e-30, f32::MAX]` → skip or output 0 and count; each operation f32 without FMA; `Reference` follows it step by step; fallback, reference and the SIMD paths of every level equal bit for bit (outputs, `Hs`, `degenerate`; NaN sign and payload of an output excepted, D296: NaN lanes by class); golden from the exact float32 model with an f64 guard |

**Why SIMD gives the fallback bits.** Every per-carrier and per-pilot result is one fixed formula of its own inputs and of per-segment or per-symbol constants (`inv`, `Δ`, `α`, `β`, `c`), with no running increment across carriers; the only reduction over carriers (the CPE sum) has the fixed order of §2 step 2; `valid` and `clean` are compare masks and selects; `degenerate` is an integer count (order free); a gather of `C32` values is a load of their bits. Lane schedules, segment tails and data-list runs therefore cannot change the bits (shared conventions §D.2).

### 5. Golden vectors

New file `tests/golden/sdrk_golden/cases/ofdm_eq.py`, the float32 model.

| Case | Seed | `N` | pilots | scenario |
| --- | --- | --- | --- | --- |
| `pilot_equalize/flat_64` | 750 | 64 | 4 evenly spaced | a flat channel (one gain); the estimate ≈ the gain, data recovered |
| `pilot_equalize/selective_256_smooth` | 751 | 256 | 16 comb | a frequency-selective channel over 8 symbols; time smoothing α = 0.5; interpolation across the comb |
| `pilot_equalize/keyed_pilots_128_cpe` | 752 | 128 | 8 hopping per symbol, CPE on | pilots at different indices each symbol, a smooth static channel (two taps, the second at a delay of one sample with magnitude 0.1, so that `H` changes little between neighbouring pilots) and a common phase that changes from symbol to symbol (a random walk); α = 0.5; the first symbol has no CPE, every later one removes its phase against `Hs_prev` |
| `pilot_equalize/degenerate_64` | 756 | 64 | 8 comb, CPE on | a null channel tap on two data carriers, a zero pilot value in one symbol, NaN and ∞ on a pilot carrier and on a data carrier in other symbols, a fading carrier that decays to 0 over 12 symbols; α = 0.5 |

- Stored: the symbols (FFT output), the pilot index/value lists per symbol, the data indices, α, flags; expected equalized data per symbol, the final `Hs`, the `degenerate` count after each symbol and, for the CPE cases, the factor `c` per symbol. `snr_db_guard` from the f64 model (over the finite outputs). Bit-exact `==` against the float32 model.
- The generator asserts that the CPE factor of `keyed_pilots_128_cpe` is active from the second symbol on and that, for every symbol `n ≥ 1`, `arg(conj(c))` equals `φ_n − φ_0` within 0.01 rad, compared modulo 2π (`φ_n` the phase applied to symbol `n`). `Hs` keeps the phase reference of the first symbol (§2 step 2), so the factor measures the phase relative to symbol 0, not `φ_n` itself; the random walk need not start at 0. The generator also runs the same symbols with the walk set to 0 and asserts that the factor's phase stays below 0.002 rad there, so the bias of the interpolated `Hs_prev` at the hopped pilot positions is small against the 0.01 rad bound. A no-op CPE (the defect of the first draft) therefore cannot be frozen into the case, and neither can a wrong expectation.
- Cross-check noted in the comment: GNU Radio `ofdm_chanest_vcvc` uses `H = Y/X` (our step 3) and `ofdm_equalizer_simpledfe` uses `H ← α·H + (1−α)·Y/P` for pilots and `Y/H` for data (our steps 4–5); we add frequency interpolation and CPE, which those blocks leave to neighbours.

### 6. Cost, SIMD plan and reference rows

`cargo xtask count` and `cargo measure` entries `pilot_equalize_256` (the facade, SIMD at each level) and `pilot_equalize_256_fallback` (the D321 floor), per input subcarrier, on the layout of the block with `N = 256`, a comb of 16 pilots, `cpe` on; and `pilot_equalize_block` (§7). Record in `docs/performance.md`.

**Division count.** Per data carrier one f32 division (the first draft: two or three, the interpolation fraction plus one or two in the complex division). On a Cortex-M4F `VDIV.F32` takes 14 cycles and does not pipeline, so the review estimated about 70 cycles per carrier for the old form and about 40 for this one (estimate, to be confirmed by `cargo xtask count`). On x86 and NEON, vector division throughput is several times lower than multiplication, so the reciprocal form also helps the SIMD paths.

**SIMD plan (`sdrk-backend-simd`; shared conventions §D).** SIMD follows the shared conventions §D (maintainer rule 2026-11-17): every lane-parallel part of the kernel gets a path on every level in this milestone, bit-identical to the fallback through the fixed element-wise formula or reduction order of the contract; D321 is a quality gate (a slower path is improved, not removed; delegation to the fallback only after several measured failed attempts, with an open item); cycles per sample with `cargo measure` against the fallback and the fair reference (D32), unfair rows marked.

| Part | Lanes | Notes |
| --- | --- | --- |
| step 1, pilot validity | pilots | `mag_sq`, `valid` and the finiteness test as masks; the count of invalid pilots by popcount |
| step 2, CPE sum | pilots, the eight partial sums of §2 | the `dot_complex` lane layout of M43 (eight complex partial sums: one 256-bit vector per component on AVX2 and AVX-512, two 128-bit vectors on SSE2, SSE4.2, NEON, `simd128`); invalid pilots masked out of the sum (select, not `+0` padding) |
| step 3, pilot estimates | pilots | `cdiv` with the vector division |
| step 4, interpolation and smoothing | carriers within a pilot segment and within the edge holds | `k − p_a` from a lane-index vector plus a base, converted to f32 (exact); `inv`, `Δ`, `Hp_a` broadcast per segment; smoothing and `clean` in the same pass, contiguous loads and stores of `Hs`; a segment tail shorter than a vector runs masked or scalar with the same formula |
| step 5, equalization | data carriers | `Hs[k]` and `Y[k]` gathered by the data index (AVX2 and AVX-512: a 64-bit gather, one `C32` per element; SSE2, SSE4.2, NEON, `simd128`: scalar loads into lanes); runs of consecutive indices (the block layout) may use plain loads, which give the same bits; `cdiv`, the degenerate select and the count |

- **Scalar part and the reason:** the per-symbol CPE factor (one `sqrt_f32` and one division on the single value `S`) and the walk over the pilot list to the segment boundaries (a few scalar setups per segment). Each is one value per symbol or per segment, so there is no lane-parallel work; the per-segment constants may also be computed in lanes, with the same bits.
- **Tests:** test 7.

**Measurement (D32, D33, D68).** The entries above and the block entry of §7 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`, Cortex-M). Each SIMD entry is compared with its `_fallback` entry (the D321 gate) and with the fair reference rows below.

**Reference rows (D32; to be confirmed in the implementation).** GNU Radio's `ofdm_equalizer_simpledfe` does other work (decision feedback on the data carriers, no interpolation between pilots, no CPE) and `ofdm_equalizer_static` holds the nearest pilot, so neither is a fair reference; both rows are information. Fair rows for the same work class, composed as for `stereo_matrix` (D150):
- x86 levels and NEON: a VOLK composition per symbol — `volk_32fc_x2_divide_32fc` for the pilot estimates and the data carriers, `volk_32fc_x2_multiply_conjugate_32fc` and a sum for the CPE term, `volk_32fc_s32f_multiply_32fc` twice and `volk_32fc_x2_add_32fc` for the smoothing over `N`, and a plain C loop for the interpolation (VOLK has no kernel for it). The composition has no zero threshold and no `clean`; the row is marked "reference does less work" where that difference matters in the numbers.
- Cortex-M (instruction counts): a CMSIS-DSP composition (`arm_cmplx_mag_squared_f32`, a scalar reciprocal loop, `arm_cmplx_conj_f32`, `arm_cmplx_mult_real_f32`, `arm_cmplx_mult_cmplx_f32`, `arm_cmplx_dot_prod_f32` for the CPE sum); CMSIS-DSP has no complex division.
- The exact kernel names are checked against the pinned library versions when the transcriptions are written; a row that turns out to do different work is marked so, as in M48 §5.

### 7. Block and kind `pilot_equalize`

`PilotEqualizeBlock` (`Lanes<C32>` of width `N` → `Lanes<C32>` of width `data`, one lane step per OFDM symbol) around the kernel `pilot_equalize`. The layout comes from scalar parameters: carrier `k` is a null when `dc_null` and `k = 0`, or when it lies among the `guard` carriers centred on `N/2` (the band edge in natural order); the pilots are the non-null carriers with `k mod pilot_spacing = pilot_offset`, all with the value `1 + 0j`; the data carriers are the others, in index order. `PilotEqualizeBlock::with_layout` takes any index lists and pilot values of the caller (Rust API only).

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `pilot_equalize` | `PilotEqualizeBlock` | lanes<c32> (`N`) → lanes<c32> (`data`) | `fft_size` (I64, build, required; a power of two 16 … 65536, equal to the input width), `pilot_spacing` (I64, build, required; 2 … `N`), `pilot_offset` (I64, build, 0; 0 … `pilot_spacing − 1`), `guard` (I64, build, 0; 0 … `N − 2`), `dc_null` (Bool, build, true), `alpha` (linear, **runtime**, 0.5; 0 ≤ α < 1), `cpe` (Bool, build, false) | `pilot_equalize`; a layout with no pilot or no data carrier is `invalid_value` at `params.pilot_spacing`; the output width is `data` |

- Reading `degenerate` (the kernel's count since `reset`). `PilotEqualizeConfig`. Storage: `Hs` (`N`) and the index lists; exact. Timing ratio 1 (steps), delay 0. A change of `alpha` takes effect with the next symbol (`set_alpha`, which also sets `β`).
- Pipeline kind `pilot_equalize` (a `LanesNode`; the width of the input through `Kind::input_width`, a width other than `fft_size` is `type_mismatch`), cost 12 per input value (a complex division per data carrier, the interpolation and the smoothing; to be confirmed against the count of the revised arithmetic, logged), a row in `docs/pipeline.md` and `docs/timing.md`.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 750 … 759.

1. Golden cases of §5 through facade, `Fallback`, `Reference` (`sdrk-testkit/tests/golden_m55.rs`), with the f64 guard; through every SIMD level in test 7.
2. Unit tests (`sdrk-core/tests/kernels_m55.rs`):
   - construction/range rules (bad `N`, α ≥ 1, α NaN, empty pilots, index ≥ N, non-increasing pilots, length mismatches, `seed_taps` with a wrong length);
   - a flat channel estimate equals the gain; the interpolation is exact on a linear channel with a power-of-two pilot spacing and dyadic values; for other spacings and values each component is within an absolute error of 8 ulp of `max(|Hp_a|, |Hp_b|)` (that component) of the exact value `Hp_a + (Hp_b − Hp_a)·(k − p_a)/(p_b − p_a)` (a relative bound per component does not hold: where a component crosses zero inside the segment, `Hp_a + Δ·t` cancels; a NumPy float32 check gave 25206 ulp of the result at spacing 26, `k − p_a = 14`, `Hp_a = 0.17005178`, `Hp_b = −0.1457473`, result `6.109e-06` against the exact `6.121e-06`; a float32 emulation of the formula on 300000 random segments, spacings 2 … 300, stayed below 3.5 ulp of the larger endpoint); single-pilot gives a constant `H`; time smoothing converges (a static channel reaches the true H within tolerance after several symbols);
   - **CPE:** after a static estimate, a symbol rotated by a known `φ` (several values in `(−π, π]`) gives the factor `exp(−jφ)` within 1e-6 and outputs equal to the unrotated symbol's within tolerance (the regression test against the old no-op formula); no CPE on the first symbol after `reset`; CPE active on the first symbol after `seed_taps`; with `α = 0` CPE on equals CPE off within rounding; `S = 0` (a previous estimate of 0 at every pilot) leaves the factor inactive;
   - the reciprocal forms: `cdiv` against the f64 quotient on random inputs over a wide magnitude range, chosen so that `|y/d|` stays in the normal f32 range (an underflowing quotient, small `|y|` over large `|d|`, has no relative bound; the top of the valid range is fine: at `|d|` near 1e19, where `1/mag_sq(d)` is subnormal, the measured error was `6.07·2^-24`), with a norm-wise relative error `|X̂ − y/d| / |y/d|` below `8·2^-24` (a NumPy float32 check of the formula gave at most `3.95·2^-24` on 200000 pairs; per component the relative error is not bounded under cancellation); folding of the factor (`cdiv(y, d, c)` against `cdiv(y, d)·c` within tolerance);
   - the zero-division path: the threshold neighbours (`mag_sq` at the f32 values just below and at `1e-30`, `+∞`, NaN) on a pilot value and on `Hs`; a zero pilot value is skipped and counted; a null tap outputs 0 and counts, no NaN; with no valid pilot `Hs` and the first-symbol flag stay unchanged;
   - `clean`: a carrier with a fading input reaches exactly `+0` and never holds a subnormal; an overflowing estimate stores 0;
   - `reset`, `seed_taps`, `set_alpha` (β follows).
3. Block-size invariance / state continuity: processing a stream of symbols one at a time vs. the same symbols is identical (the kernel is per-symbol; the "chunking" here is symbol granularity) — 20 orderings not applicable; instead a continuity test that `Hs` and `degenerate` after `n` symbols are independent of any grouping (seed 753).
4. Extremes: all-zero symbol (outputs 0, `degenerate` = data len, `Hs` stays 0); full scale; NaN/∞ on a data carrier (that output non-finite, the other outputs and the state unchanged); NaN/∞ on a pilot carrier (the pilot skipped and counted, `Hs` finite); inputs near `f32::MAX` (an overflowing `Hs` component is stored as 0); the largest `N` (65536) with a sparse pilot set; `cpe` on with a zero CPE sum (factor inactive, `φ = 0`).
5. Suite cases `pilot_equalize_c32` and `extremes_m55`.
6. **Block and kind** (`sdrk-blocks/tests/pilot_equalize_blocks.rs`, `sdrk-pipeline/tests/pilot_equalize_kinds.rs`; seeds 754, 755): the block equals the kernel bit for bit over a sequence of symbols, in 20 cuts of the lane steps into calls; the layout from the parameters for `N` = 64 and 256; the description rules of M09, exact storage, struct at most 512 bytes, validation errors at their parameter; a pipeline flow `channelize` (as a lane source of width `N`) → `pilot_equalize` → `channel_select` equals the blocks wired by hand; `cargo xtask schema --check` and the WASM kind list include `pilot_equalize`.
7. **SIMD** (`sdrk-backend-simd/tests/pilot_equalize.rs`; seeds 757, 758): every level equals `Fallback` bit for bit in the outputs, `Hs` and `degenerate` after each call (an output that is NaN is compared by class, D296; `Hs` is always finite through `clean`):
   - data lists of lengths 0 … 67 and 4096 (contiguous runs, random orders, repeated indices) and `N` = 16, 64, 256, 4096, 65536;
   - pilot segment lengths 1 … 67 (vector tails inside a segment, the edge holds), pilot counts 1 … 20 (across the eight partial sums), invalid pilots at every position modulo 8;
   - special values: `±0`, subnormals, the threshold neighbours of `T`, NaN, `±∞`, `f32::MAX`; `cpe` on and off; `α` 0 and 0.5;
   - the golden cases of §5 through each level.

## Acceptance

R1.2 commands (`cargo xtask check` includes `cargo xtask check cycles`, D321); `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-backend-simd --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the entries of §6 and `pilot_equalize_block`); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [ ] Kernel `pilot_equalize` (`C32`): `PilotEqualizeState` (with `degenerate`, `β`), trait, Fallback, Reference, facade delegation, `backend_for` (the SIMD level)
- [ ] SIMD paths of §6 on every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`), bit-identical to the fallback, each passing D321 (or, after several measured failed attempts, delegated with an open item); `docs/backends.md`
- [ ] Contract row and section in `docs/kernels.md` (pilot validity and the zero threshold, CPE against the previous estimate, the reciprocal forms, interpolation, smoothing with `clean`; the GNU Radio cross-check)
- [ ] Golden cases (seeds 750–752, 756) and the float32 Python model with the f64 guard and the CPE assertion
- [ ] Tests 1–7 (seeds 750–758); suite cases
- [ ] Count and cycle entries (`cargo xtask count`, `cargo measure`: `pilot_equalize_256`, `pilot_equalize_256_fallback`), the reference rows of §6 with their marking, `docs/performance.md`
- [ ] Block `PilotEqualizeBlock` and kind `pilot_equalize` (§7): `docs/blocks.md` (entry; reserved-name row "delivered"), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `pilot_equalize_block`
- [ ] CHANGELOG, decisions log (the α convention, edge-hold interpolation, CPE against the previous smoothed estimate and its factor without `arg`, the reciprocal forms, the zero threshold and the skipped pilots, `clean` of `Hs`, the reference rows, the SIMD levels that stay on the fallback with their numbers, if any)

## Status

Planned (revised 2026-11-17). Nothing is implemented.
