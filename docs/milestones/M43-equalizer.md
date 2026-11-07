# M43 — Adaptive equalizer

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M40.1](M40.1-psk-variants.md) · Next: [M44](M44-qam.md)

## Goal

A kernel and block `equalize`: an adaptive FIR equalizer on the symbol stream, blind with the constant-modulus algorithm (CMA) and then decision-directed (LMS), for PSK constellations now and QAM in M44. A multipath channel in the testkit and the option `equalizer_taps` of `psk_demod` complete it.

## Why

Above a few kilobaud, reflections and the filters of cheap front ends cause inter-symbol interference that no timing or carrier loop removes. QAM (M44) does not work at all without an equalizer; fast PSK links profit from it.

## Order

M43 depends on M40.1 (the 8PSK decision rule; through it M40). It can run in parallel with M41 and M42.
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

**In:** kernel `equalize` (C32) in `sdrk-core` with Fallback, Reference, facade delegation, contract; block `EqualizeBlock` and kind `equalize`; option `equalizer_taps` of `psk_demod`; `channel::Multipath` in the testkit; golden cases; tests; SIMD path for the filter part (`dot`); `docs/kernels.md`, `docs/blocks.md`, `docs/backends.md`, `docs/testing.md`, `CHANGELOG.md`.

**Out:** fractionally spaced equalization (T/2; backlog), decision-feedback equalizers (backlog), RLS (backlog), Q15 (backlog).

> **Amendment 2026-11-06 (maintainer decision of 2026-11-06, D830):** the decision-feedback equalizer moves from Out to In: `equalize` has feedback taps (`feedback_taps`, 0 = linear), the block parameter `feedback_taps` and the `psk_demod` option `equalizer_feedback_taps`.

## Specification

### 1. Kernel `equalize`

```rust
pub enum EqMode { Cma, Decision }
pub enum EqConstellation { Bpsk, Qpsk, Psk8 /* M44 adds Qam16, Qam64 */ }
pub struct EqualizeState<W: AsRef<[C32]> + AsMut<[C32]>, H: AsRef<[C32]> + AsMut<[C32]>> { /* weights (N), history (N), mode, counters */ }
impl<..> EqualizeState<..> {
    /// N taps (odd, 3 … 63), centre tap 1, others 0; mu_cma, mu_dd 1e-5 … 0.1; cma_symbols: symbols in CMA
    /// before switching to decision mode (0 = start in decision mode); constellation for decisions and the CMA radius.
    pub fn new(weights: W, history: H, constellation: EqConstellation, mu_cma: f32, mu_dd: f32, cma_symbols: u32) -> Result<Self>;
    pub fn mode(&self) -> EqMode;
    pub fn mse(&self) -> f32;      // average |e|² of the decision error, time constant 100 symbols
    pub fn reset(&mut self);
}
pub trait Equalize { fn equalize<W, H>(&self, s: &mut EqualizeState<W, H>, input: &[C32], output: &mut [C32]) -> Result<()>; }
```

One input sample per symbol. Per symbol, with the history `u` (newest first) and the weights `w`:

1. `y = Σ w[i]·u[i]` (complex, ascending `i`, no FMA; the plain dot product, not conjugated).
2. CMA: `R2 = E|s|⁴ / E|s|²` of the constellation (1 for PSK; the value for QAM in M44), `e = y·(|y|² − R2)`, `w[i] = w[i] − mu_cma·e·conj(u[i])`.
3. Decision: `d` = the nearest constellation point (the same decision rule as `slice`/`costas`), `e = y − d`, `w[i] = w[i] − mu_dd·e·conj(u[i])`; `mse` update with `|e|²`.
4. The switch from CMA to decision happens after `cma_symbols` symbols; it does not switch back (a `reset` returns to CMA).
5. Output `y`. Non-finite input: output 0, weights unchanged.

Timing ratio 1, delay `(N − 1)/2` symbols. The float32 model is exact (bit-exact class) with the f64 guard. The filter part (step 1) uses the `dot` kernel of the backend where it is accelerated; the update loop stays scalar unless the SIMD path is equal bit for bit (it must be: element-wise products and one sum order; if the backend's `dot` sums in another order, the kernel calls the fallback `dot`; record the decision).

> **Amendment 2026-11-06 (maintainer decision of 2026-11-06, D800 to D803):** step 1 does not sum in ascending `i`. The filter part is the new reusable kernel `dot_complex` (`kernels::dot::DotComplex`, `C32`, bit-exact on every backend and level), and `y` is formed in its order: eight partial sums per component, element `i` to partial sum `i mod 8`, the product `(u.re w.re − u.im w.im, u.re w.im + u.im w.re)` without FMA, then the fixed tree `(s0 + s2) + (s1 + s3)` with `s_k = acc_k + acc_{k+4}` (see `docs/kernels.md`). The `dot` kernel is not used: it multiplies by real taps and its `C32` SIMD paths are a tolerance kernel. Because every backend's `dot_complex` gives the bits of the fallback, the SIMD backend uses its own for the filter part and the fallback call of the last sentence above is never needed.

> **Amendment 2026-11-06 (maintainer decision of 2026-11-06 to adapt the solutions of other projects, D826 to D833):** the constructor is `EqualizeState::new(weights, history, params)` with `EqParams { constellation, feedback_taps, mu_cma, mu_dd, cma_symbols, phase_bw }`; the weights and the history hold `N + M` values (`N` forward taps, then `M` feedback taps, D830, D831). Per symbol (the exact order is the contract in `docs/kernels.md`): `y = Σ w[i] h[i]` over the joint vector `h` (the forward symbols, then the fed-back decisions) with one `dot_complex` call; a decision-directed phase tracker turns the output back, `z = y e^(-jθ)`, and the decision `d` is made on `z` (D829); CMA `e = clip(y (|y|² − R2))` with each component clipped to `[-1, 1]` (GNU Radio's `adaptive_algorithm_cma`, D827); decision mode `e = (z − d) e^(jθ)`; both updates are normalized, `w[i] = w[i] − (mu / (Σ|h|² + 1e-6)) e conj(h[i])` (liquid-dsp's `eqlms`, GNU Radio's `adaptive_algorithm_nlms`, D826), so `mu_cma` and `mu_dd` are normalized step sizes from 1e-5 to 1; the tracker is a second-order loop on `Im(z conj(d))` with the gains of `costas` for `phase_bw` (0 = off) and the damping 0.7071, its frequency limited to ±0.125 cycles per symbol, the rotation exactly the identity at phase 0 (D829, D833); the decisions `d e^(jθ)` are fed back in decision mode only (D830); the output is `z`; `mse` is the decision error power of `z` (D831).

### 2. Block and kind `equalize`

`EqualizeBlock`: C32 → C32 (symbols). Parameters: `taps` (I64, build, 11), `constellation` (Choice `bpsk`/`qpsk`/`8psk`, build, `qpsk`), `mu_cma` (None, **runtime**, 0.001), `mu_dd` (None, **runtime**, 0.0005), `cma_symbols` (I64, build, 2000). Cost `8·taps`. **Chain position:** after `symbol_sync` and after `costas`. CMA converges whatever the carrier phase does, and the decision mode needs the phase that `costas` corrected; an equalizer before `costas` would make decisions on rotating symbols.

> **Amendment 2026-11-06 (maintainer decision of 2026-11-06, D825, D828 to D830):** parameters `taps` (I64, build, 11), `feedback_taps` (I64, build, 0, 0 to 63), `constellation`, `mu_cma` (runtime, 0.05, 1e-5 to 1), `mu_dd` (runtime, 0.02, 1e-5 to 1), `cma_symbols` (I64, build, 1000), `phase_bw` (F64, build, 0.02; 0 or 1e-4 to 0.2). Cost `8·(taps + feedback_taps)`. **Chain position:** after `symbol_sync` and before `costas`, the order of GNU Radio's PSK receiver (`pfb_clock_sync` → `linear_equalizer` → `costas_loop_cc`) and of liquid-dsp's blind equalizer example (`eqlms` → PLL): CMA needs no carrier phase, the phase tracker of the kernel gives the decision mode its phase, and `costas` after it removes what remains; a carrier loop before the equalizer runs on symbols whose inter-symbol interference makes it slip (I43-3).

### 3. `psk_demod` option

`equalizer_taps` (I64, build, 0; 0 = off, else odd 3 … 63): inserts `equalize` (constellation from `order`, defaults above) after `costas`. In differential mode the equalizer runs in CMA only (`cma_symbols` = infinite).

> **Amendment 2026-11-06 (D815):** in differential mode there is no `costas`; the equalizer runs on the symbols of `symbol_sync` before the differential product, in CMA only. In `offset` mode it runs after that mode's carrier loop. Whether the equalizer belongs before the carrier loop in coherent mode is open (I43-3).
>
> **Amendment 2026-11-06 (maintainer decision of 2026-11-06, D825, D828 to D830):** in coherent mode `equalize` runs between `symbol_sync` and `costas`, with the defaults of §2 as amended and its phase tracker at `carrier_bw`; in differential mode without the tracker (`phase_bw` 0). The option `equalizer_feedback_taps` (I64, build, 0, 0 to 63, only with `equalizer_taps` above 0) gives the equalizer feedback taps.

### 4. `channel::Multipath`

`Multipath::new(taps: &[(delay_samples: f64, gain: C64)])` with a fractional delay by the windowed-sinc interpolator of M38; presets `Multipath::mild()` (`[(0, 1), (1.3, 0.3·e^{j1}), (2.7, 0.1·e^{−j2})]` at the symbol rate) and `Multipath::severe()` (`[(0, 1), (0.8, 0.6·e^{j0.5}), (2.1, 0.4·e^{−j1.5}), (4.0, 0.2)]`), delays in symbols, scaled by `sps`.

> **Amendment 2026-11-06 (D810):** the presets take the samples per symbol as an argument, `Multipath::mild(sps)` and `Multipath::severe(sps)`, and return `Result`.

## Tests

1. Golden: `equalize/cma_qpsk_c32_16384` (seed 590), `equalize/dd_8psk_c32_16384` (seed 591), float32 model.
2. **QPSK through `severe`:** without equalizer the BER at Eb/N0 14 dB is above 1e-2; with `equalizer_taps` 15 it is within 2 dB of the AWGN theory after 3000 symbols of training (seed 592).
3. **8PSK through `mild`:** `psk_mod` order 8 (M40.1) → `mild` → `psk_demod` order 8 with `equalizer_taps` 11: BER within 2 dB of the AWGN result of M40.1 at Eb/N0 16 dB (seed 593).
4. Convergence: `mse` falls below −20 dB within 2000 symbols for `mild` at Eb/N0 20 dB (seed 594); the CMA phase is arbitrary but constant.
5. SIMD: the filter part on every backend equals the fallback bit for bit; `cargo xtask count` entries.
6. Chunk invariance with 20 plans (seed 595).

> **Amendment 2026-11-06 (D818, D819, D824):** "within 2 dB" in tests 2 and 3 means `BER <= theory(Eb/N0 - 2 dB)` over 10^6 counted bits without slips (D818). Test 4 runs inside `psk_demod` with `equalizer_taps` 11; the constant CMA phase means a spread of the window phases of at most 0.05 rad (D819). The part of test 2 without the equalizer needs at least 500 000 bits pushed after the training and a synchronized meter (D824). The criteria with the equalizer in tests 2, 3 and 4 are not met and wait for maintainer decisions (I43-1 to I43-3).
>
> **Amendment 2026-11-06 (maintainer decision of 2026-11-06, D834 to D836):** test 2 with the equalizer runs `equalizer_taps` 15 with `equalizer_feedback_taps` 5 (a decision feedback equalizer, the answer to the spectral null of `severe`); "within 2 dB" stays D818. Test 4: `mse` is the decision error of the phase-corrected output (D831), and the constant CMA phase is the phase of the centre weight read every 50 symbols from symbol 250 to the end of the CMA mode (1000), within 0.05 rad (D835). All criteria are met; no test is ignored.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64`.

## Deliverables checklist

- [x] Kernel `equalize` with reference, fallback, delegation, contract (D803 to D809); the new reusable kernel `dot_complex` for its filter part (D800 to D802)
- [x] Block and kind `equalize`; `psk_demod` option `equalizer_taps` (D814 to D817)
- [x] `channel::Multipath` with presets (D810)
- [x] Tests (seeds 590–595, and 596/597 for `dot_complex`) (D811 to D813, D818, D819, D824, D832, D834 to D836); every criterion of tests 1 to 6 is met; the count entries of test 5 are counted (see Status)
- [x] The adaptation of the solutions of other projects (maintainer decision of 2026-11-06): the order before `costas`, NLMS, the clipped CMA error, the phase tracker, the decision feedback and the GNU Radio reference of `equalize_fallback` (D825 to D839)
- [x] `docs/kernels.md`, `docs/blocks.md`, `docs/backends.md`, `docs/testing.md`, CHANGELOG, decisions log (D800 to D838); count entries counted and measured (D821 to D823, D838, D840 to D845; see Status)

## Status

Done: the reusable bit-exact kernel `dot_complex` (`kernels::dot::DotComplex`, `C32`, not conjugated, eight partial sums per component; Reference, Fallback, facade delegation, the SIMD paths of `sdrk-backend-simd`, contract in `docs/kernels.md`, rows in `docs/backends.md`; D800 to D802); the kernel `equalize` (CMA then decision-directed, both normalized by the window power (NLMS), the CMA error clipped, a decision-directed phase tracker on the output, optional decision feedback; its filter part over forward and feedback taps on `dot_complex`, the `mse`, the mode switch and the rules for non-finite values and silence; D803 to D809, D826 to D831, D833, D839); `channel::Multipath` with the presets (D810); the golden cases `equalize/cma_qpsk_c32_16384` (590), `equalize/dd_8psk_c32_16384` (591), `dot_complex/c32_13` (596) and `dot_complex/c32_1021` (597) (D811, D812, D832); the suite cases `dot_complex_c32` and `equalize_c32` (D813, D832); the block and kind `equalize` with `feedback_taps` and `phase_bw` (D814, D828 to D830); the `psk_demod` options `equalizer_taps` and `equalizer_feedback_taps`, the equalizer between `symbol_sync` and `costas` (D815 to D817, D825, D830); tests 2 to 6, all run by default (D818, D819, D824, D834 to D836); the count entries `dot_complex`, `dot_complex_fallback`, `equalize` and `equalize_fallback` with their references, `equalize_fallback` against a C translation of GNU Radio's `linear_equalizer` with `adaptive_algorithm_nlms` (D821 to D823, D838); `docs/kernels.md`, `docs/blocks.md`, `docs/backends.md`, `docs/testing.md`, `docs/pipeline.md`, `docs/performance.md`, the schema, CHANGELOG and the decisions log.

The adaptation of the solutions of other projects (maintainer decision of 2026-11-06), with the sources read:

| Change | Source | Decision |
| --- | --- | --- |
| The equalizer before the carrier loop in `psk_demod` | GNU Radio `gr-digital/examples/mpsk_stage6.grc` (`pfb_clock_sync_ccf` -> `linear_equalizer` with `adaptive_algorithm_cma` -> `costas_loop_cc`); liquid-dsp `examples/eqlms_cccf_blind_example.c` (`eqlms_cccf_step_blind`, then the `nco_crcf` PLL) | D825 |
| Both updates normalized by the window power, `mu / (P + 1e-6)` | liquid-dsp `src/equalization/src/eqlms.proto.c` `EQLMS(_step)` (`mu conj(alpha) r[i] / x2_sum`, skipped at zero power); GNU Radio `adaptive_algorithm_nlms::update_taps` | D826 |
| The CMA error `y (abs(y)^2 - R2)` clipped to `[-1, 1]` per component | GNU Radio `adaptive_algorithm_cma::error` | D827 |
| Normalized defaults and a fixed CMA symbol count | liquid-dsp's range of `mu` (its blind example 0.08); GNU Radio and MATLAB switch after a fixed count | D828 |
| Decision mode before the carrier loop: a phase tracker in the kernel, the decision on `z = y e^(-j theta)`, the error turned back by `e^(j theta)` | own decision (the references have no decision mode before a carrier loop) | D829, D833 |
| Decision feedback: one joint vector, one NLMS error and update, hard decisions in decision mode only | GNU Radio `lib/decision_feedback_equalizer_impl.cc` (`d_filt_buf` of the decision history and the forward window); MATLAB `comm.DecisionFeedbackEqualizer` documentation | D830, D834 |
| The scalar reference of `equalize_fallback` | GNU Radio `lib/linear_equalizer_impl.cc`, `adaptive_algorithm_nlms.h`, `constellation_qpsk`, `fir_filter_ccc` of gr-filter | D838 |

Deviations from the specification, recorded in the decisions log:
- §1 step 1: `y` is summed in the order of the new kernel `dot_complex`, not in ascending `i` (amendment above, maintainer decision of 2026-11-06; D800 to D803); the `dot` kernel is not used.
- §1 leaves open when `mse` is updated, how the mode switch counts and what non-finite values do: `mse` follows the decision error in CMA mode too (D804, of the phase-corrected output since D831), every symbol in CMA mode counts towards `cma_symbols` (D805), a non-finite input enters the history as 0 (D806).
- §1, §2, §3 and the Scope as amended on 2026-11-06 (maintainer decision): NLMS step sizes from 1e-5 to 1 with new defaults, the clipped CMA error, the phase tracker, the decision feedback, the order before `costas` (D825 to D833, D839).
- §4: the presets take the samples per symbol as an argument, `Multipath::mild(sps)` and `Multipath::severe(sps)`, instead of no argument (D810).
- §3: in differential mode the equalizer runs on the symbols of `symbol_sync` before the differential product (there is no `costas`); in `offset` mode after its carrier loop (D815).
- Test 4 runs inside `psk_demod`; the constant CMA phase is read from the centre weight (D819, D835).
- "Within 2 dB" is `BER <= theory(Eb/N0 - 2 dB)` over 10^6 counted bits without slips (D818, D834).

Measured (local runs in the test profile, Linux x86-64):

| Test | Limit | Measured | Result |
| --- | --- | --- | --- |
| 1: golden `equalize` (590, 591) | bit-exact against the float32 model; f64 guard | bit-exact; guard 130.1 and 127.8 dB (D832) | passes |
| 1: golden `dot_complex` (596, 597) | bit-exact against the float32 model; f64 guard | bit-exact; guard 143.0 and 131.7 dB (D812) | passes |
| 2: QPSK through `severe`, 14 dB, without the equalizer (592) | BER above 1e-2 (at least 500 000 bits pushed, D824) | BER 9.3e-2 over 260 349 counted bits, 498 slips | passes |
| 2: with `equalizer_taps` 15 and `equalizer_feedback_taps` 5 | BER at most 9.0e-9 over 10^6 bits, no slips (D818) | 0 errors in 10^6 bits, no slips; mean `mse` -15.34 dB, equivalent Eb/N0 12.33 dB | passes |
| 2: comparison, linear with `equalizer_taps` 15 | information | BER 5.5e-4 in 10^6 bits, no slips; mean `mse` -10.80 dB, equivalent Eb/N0 7.79 dB | (I43-1: a linear equalizer cannot reach it) |
| 3: 8PSK through `mild`, 16 dB, `equalizer_taps` 11 (593) | BER at most 8.7e-7 over 10^6 bits, no slips (D818) | 0 errors in 10^6 bits, no slips; mean `mse` -19.29 dB, equivalent Eb/N0 14.52 dB | passes |
| 4: `mse` on `mild`, 20 dB, 11 taps (594) | below -20 dB within 2000 symbols | first below -20 dB after symbol 1029; smallest within 2000 -21.42 dB; final -21.46 dB | passes |
| 4: CMA phase constant | spread of the centre-weight phase at most 0.05 rad (D835) | 0.0104 rad over 16 readings from symbol 250 to 1000 | passes |
| 5: SIMD filter part | `dot_complex` and `equalize` of every backend bit for bit equal to the fallback | equal (suite case `dot_complex_c32`, `sdrk-backend-simd/tests/equalize.rs`, facade tests) | passes |
| 6: chunk invariance, 20 plans (595) | bit-exact whole and in 20 cut plans | `equalize` (with feedback and tracker) and `psk_demod` with the equalizer bit-exact | passes |

Comparisons behind the decisions (an f64 model of the loop on the `symbol_sync` output of the test signals, D828, D829, D834):

| Arrangement | `severe` 14 dB, 15 taps | `severe` 14 dB, 15 + 5 taps | 8PSK `mild` 16 dB, 11 taps |
| --- | --- | --- | --- |
| Carrier loop before the equalizer (the first order of §2) | BER 9.0e-2, 1572 slips (I43-1) | — | BER 0.28, 3926 slips (I43-3) |
| CMA only before the carrier loop (GNU Radio's tutorial) | 6.6e-4 | not possible (feedback needs decisions) | 1 error in 10^6 bits (above the edge 8.7e-7) |
| CMA, then decisions with the phase tracker (D829) | 5.5e-4 | 0 in 10^6 bits | 0 in 10^6 bits |

| Tracker on `severe`, 15 + 5 taps, 3 10^5 bits | no carrier phase | 0.785 rad | 0.785 rad and 0.0005 cycles per symbol | 0.3 rad and 0.002 cycles per symbol |
| --- | --- | --- | --- | --- |
| Without the tracker (decision mode as in liquid-dsp's example) | 0 | 5 errors | BER 0.12 | BER 2.9e-3 |
| With the tracker, `phase_bw` 0.02 | 0 | 0 | 0 | 0 |

Counts (2026-11-07, after the performance round of D840 to D845): the four entries were counted on all four platforms (`cargo xtask count --platform <p> --kernel dot_complex --kernel dot_complex_fallback --kernel equalize --kernel equalize_fallback --merge`; x86_64 per level `sse2`, `sse4.2`, `avx2`, `avx2-runtime`) and measured with `cargo measure dot_complex` and `cargo measure equalize` on every level with a core model; the rows are in `benches/results/*.json` and `docs/performance.md`. Every judged level is within the release limit and is `release` in `benches/limits.json`; `dot_complex` at `avx2-runtime` (1.00 against the limit 1.02, after the two-block loop of `dot_complex_avx2`) needs no waiver. D321 holds at every level.

Instructions per sample, ours / comparison:

| Entry (comparison) | sse2 | sse4.2 | avx2 | avx2-runtime | avx512 | neon | simd128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dot_complex` (VOLK) | 5.29 / 8.53 info | 5.29 / 6.03 | 2.35 / 3.02 | 2.36 (1.00) | 2.35 / 3.02 | 3.53 / 4.03 | 13.20 / 40.55 info |
| `dot_complex_fallback` (CMSIS-DSP) | 14.41 / 15.02 | 8.66 / 15.02 | 8.41 / 13.02 | - | 6.41 / 13.02 | 9.03 / 11.52 | 13.20 / 35.30 |
| `equalize` (our fallback, info) | 502.98 / 502.98 | 496.45 / 496.45 | 533.11 / 533.11 | 502.21 | 492.09 / 492.09 | 437.31 / 437.31 | 1027.31 / 1027.31 |
| `equalize_fallback` (GNU Radio `linear_equalizer`) | 502.98 / 722.54 | 496.45 / 722.54 | 533.11 / 729.03 | - | 492.09 / 729.03 | 437.31 / 493.03 | 1027.31 / 1900.06 |

Simulated cycles per symbol of `equalize` (whole-loop, reference / counter model; SIMD = fallback since D841):

| Level (models) | SIMD before D840 | Fallback before D840 | Now | GNU Radio | Now / GNU Radio |
| --- | ---: | ---: | ---: | ---: | ---: |
| sse2 (`skylake` / `znver2`) | 434.72 / 400.79 | 391.41 / 350.08 | 157.42 / 164.21 | 168.47 / 201.30 | 0.93 / 0.82 |
| sse4.2 (`skylake` / `znver2`) | 398.8 / 351.6 | 344.9 / 304.3 | 146.43 / 157.81 | 168.47 / 201.30 | 0.87 / 0.78 |
| avx2 (`znver3` / `alderlake`) | 252.1 / 237.9 | 226.8 / 240.2 | 114.56 / 116.18 | 931.44 / 220.19 | 0.12 / 0.53 |
| avx512 (`znver4` / `sapphirerapids`) | 238.7 / 244.0 | 207.3 / 243.6 | 107.27 / 121.97 | 929.60 / 220.19 | 0.12 / 0.55 |
| neon (`cortex-a72` / `cortex-a76`) | 549.3 / 456.5 | 466.5 / 392.6 | 262.11 / 191.23 | 229.47 / 236.11 | 1.14 / 0.81 |

`neon` on `cortex-a72` stays above GNU Radio: the phase tracker and the smoothed `mse` cost about 81 cycles per symbol there (D845). `dot_complex` simulates 0.46 / 0.58 cycles per sample at `avx2` (VOLK 0.78 / 0.78), 1.32 / 1.33 at `sse2`, 1.31 / 1.33 at `sse4.2`, 0.45 / 0.58 at `avx512` and 3.67 / 1.77 at `neon` (VOLK 3.55 / 2.03); 0.24 to 0.86 of its fallback.

Verified locally on Linux x86-64: the M43 test files of `sdrk-core` (`kernels_m43`), `sdrk-backend-simd` (`equalize`), `sdrk` (`facade`), `sdrk-blocks` (`equalize_blocks`, `psk_blocks`, `describe`), `sdrk-pipeline` (`equalize_kinds`, `psk_kinds`) and `sdrk-testkit` (`golden_m43`, `suite_m43`, `channel_multipath_m43`, `psk_ber_m43` with all its tests, the M40 and M40.1 BER tests), `cargo clippy` on the touched crates, `cargo xtask check`, `cargo xtask schema --check` and `python tests/golden/generate.py --check` (Python 3.12, the pinned NumPy and SciPy).

Not done or not run:
- The whole workspace test suite, the emulator, the wasm tests and the other hosts; CI and the final gate run them.

Open items: none of M43; I43-1 to I43-4 are closed (D837), I43-5 (`dot_complex` at `avx2-runtime`) is closed by the two-block loop of `dot_complex_avx2` (CHANGELOG).

Gate and counts: see below.
