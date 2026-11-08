# M48 — Kernel `gauss_noise`: deterministic Gaussian noise

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M38](M38-symbol-sync.md) · Next: [M49](M49-scramble.md)

> **Status: planned** (written 2026-11-08). First of the kernel milestones M48 to M56 for spread-spectrum and multicarrier work (DSSS and OFDM). Each of them delivers one kernel only; the blocks, kinds and flows that use the kernels come in later milestones.

## Goal

A kernel `gauss_noise` that fills a buffer with white Gaussian noise of a given standard deviation, for `C32`, `CQ15`, `f32` and `Q15`, bit-exact between `Fallback` and `Reference`, from a seed, with random access to any sample index. The algorithm is the one the testkit already uses for its channel model (`sdrk_testkit::rng::Pcg32` and `Pcg32::normal`), so a noise source in a flow and the AWGN of the tests produce the same numbers.

## Why

Today noise exists only in the testkit (`channel::Awgn`, feature `std`, `Complex64`). A flow cannot add noise of a known level, a microcontroller cannot generate it, and there is no reference noise source for measurements (noise figure, detector thresholds, BER curves in the browser). Spread-spectrum and OFDM experiments need noise inside the flow: as a calibrated reference, as additive channel noise in loop flows, and as test input of every later kernel of this series.

## Order

M48 depends on M38 (the testkit channel model `Awgn` that test 4 compares against) and through it on M37. It has no dependency on M39 to M47 and can run in parallel with every milestone after M38. The range of decision numbers is fixed with the maintainer before the start, as with every parallel milestone.

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

**In:** kernel `gauss_noise` (`C32`, `CQ15`, `f32`, `Q15`) in a new module `sdrk_core::kernels::noise` with `GaussNoiseState`, Fallback, Reference, facade delegation, contract row and section in `docs/kernels.md`; golden cases; unit, statistics, invariance and extreme tests; suite cases; `cargo xtask count` entries; `CHANGELOG.md`, `docs/decisions-log.md`.

**Out:** a block or kind `noise` (later milestone; the name is reserved by this milestone); coloured noise, phase noise, impulsive noise; a SIMD path (allowed later if it gives the bits of the fallback; not required here); changing `sdrk_testkit::rng::Pcg32` or `channel::Awgn` (they stay as they are and serve as the cross-check).

## Specification

General rules of M19 apply.

### 1. Generator

The generator is PCG32 (PCG-XSH-RR 64/32, `pcg_basic.c` of the PCG reference implementation), exactly as `sdrk_testkit::rng::Pcg32`:

- `MULT = 6364136223846793005`; `new(seed, stream)`: `state = 0`, `inc = (stream << 1) | 1`, one step, `state += seed` (wrapping), one step. A step is `state = state * MULT + inc` (wrapping `u64`).
- `next_u32`: `old = state`, step, `xorshifted = (((old >> 18) ^ old) >> 27) as u32`, `rot = (old >> 59) as u32`, result `xorshifted.rotate_right(rot)`.
- `next_f64 = (next_u64 >> 11) as f64 * 2^-53` with `next_u64 = (next_u32 << 32) | next_u32` (high half drawn first).
- `normal()`: `u1 = 1.0 - next_f64()`, `u2 = next_f64()`, result `libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2π * u2)` in f64 (Box–Muller, cosine branch only; `u1` lies in `(0, 1]`, so the logarithm is finite). One normal value consumes exactly four `next_u32` draws.
- `advance(delta)`: skips `delta` draws in `O(log delta)` (`pcg_advance_lcg_64`, the algorithm of `Pcg32::advance`).

The kernel has its own private copy of these few functions in `sdrk-core` (the core cannot depend on the testkit, R2). R4.4 forbids randomness in kernels; this is a seeded, deterministic sequence held in caller-visible state, not an entropy source. Log this reading of R4.4 as a decision.

### 2. State and trait

```rust
pub struct GaussNoiseState { /* rng, sigma: f32, seed, stream, index: u64 */ }
impl GaussNoiseState {
    /// `sigma` is the standard deviation of each real component (the real value, or the
    /// real and the imaginary part of a complex value); finite and 0 ≤ sigma ≤ 16, else
    /// `InvalidParameter`.
    pub fn new(seed: u64, stream: u64, sigma: f32) -> Result<Self>;
    pub fn sigma(&self) -> f32;
    pub fn set_sigma(&mut self, sigma: f32) -> Result<()>;  // same rule; the position is kept
    pub fn index(&self) -> u64;          // the sample index of the next output sample
    pub fn seek(&mut self, index: u64);  // re-creates the generator and advances it
    pub fn reset(&mut self);             // seek(0)
}
pub trait GaussNoise<T> {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [T]) -> Result<()>;
}
```

- **Draws per sample:** a complex sample (`C32`, `CQ15`) takes two normals, real part first (8 draws), a real sample (`f32`, `Q15`) one normal (4 draws). `seek(n)` = `new(seed, stream)` followed by `advance(n · draws_per_sample)`, so sample `n` depends only on the seed, the stream, the sample type and `n`. The state does not know the sample type in advance: it stores the index and the draws per sample of the last call; a call with another sample type than the previous one is `InvalidState` (the first call fixes it, `reset` and `seek` clear it). Log the choice if a cleaner form is found (for example a type parameter on the state).
- **Values (f64 first, then one rounding):** with `z` the normal value and `s = sigma as f64`: `v = z * s`.
  - `f32`: `v as f32` (round to nearest, one rounding). `C32`: the same per component.
  - `Q15`: the f32 → Q15 rule of R7 applied directly to the f64 value `v` (`floor(v · 32768 + 0.5)` clamped to `[-32768, 32767]`), not to `v as f32` (one rounding, as R7 demands). `CQ15`: per component. Values beyond full scale saturate; the clipping is part of the contract (for `sigma` 0.25 about 6·10⁻⁵ of the samples clip).
- `sigma = 0` gives `+0.0` (respectively 0) in every sample and still advances the generator, so switching the level at run time keeps the sequence aligned to the sample index.
- An empty output is allowed and changes nothing.
- `set_sigma` at run time takes effect with the next call (the blocks of the later milestone apply runtime parameters between calls).

**Power convention.** `sigma` is per real component, as in `channel::Awgn` (`N0 = 2 sigma²` per complex sample). Comparison with established libraries (cross-check of this milestone, sources in the decisions log): GNU Radio's complex `noise_source_c` with amplitude `A` multiplies `gr::random::rayleigh_complex()` (variance 1 per component) by `A / sqrt(2)`, so its total power is `A²` and its `sigma` is `A / sqrt 2`; liquid-dsp's `crandnf` has variance 1 per component and its `cawgn` scales by `nstd / sqrt 2`. The rustdoc of `GaussNoiseState::new` states these conversions. GNU Radio uses the xoroshiro128+ generator and liquid-dsp the sine branch of Box–Muller, so their sequences are not ours; only the statistics are compared (test 3).

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `gauss_noise` | `kernels::noise::GaussNoise` | `C32`, `CQ15`, `f32`, `Q15` | bit-exact | PCG32 and Box–Muller (cosine branch) of §1 in f64 with `libm`, `z · sigma` rounded once to the output type (R7 for Q15); fallback equals reference bit for bit; golden cases within `max_abs_err` (CPython's `math` and Rust `libm` may differ in the last f64 bit) |

`Reference` is the plain loop of §1 and §2, one sample at a time. `Fallback` may draw in batches but gives the same bits. `backend_for("gauss_noise", ..)` reports `fallback`.

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/gauss_noise.py`. The Python model implements PCG32 with Python integers masked to 64 bits and `normal()` with `math.sqrt`, `math.log`, `math.cos` in float64, written independently from the Rust code (R8.1).

| Case | Seed (case) | Generator seed, stream | Type | `sigma` | Length | Start index |
| --- | --- | --- | --- | --- | --- | --- |
| `gauss_noise/c32_unit_4096` | 680 | 680, 1 | C32 | 1.0 | 4096 | 0 |
| `gauss_noise/c32_seek_4096` | 681 | 681, 7 | C32 | 0.3 | 4096 | 1 000 000 007 |
| `gauss_noise/f32_unit_4096` | 682 | 682, 1 | f32 | 1.0 | 4096 | 0 |
| `gauss_noise/cq15_quarter_4096` | 683 | 683, 2 | CQ15 | 0.25 | 4096 | 0 |
| `gauss_noise/q15_clip_4096` | 684 | 684, 3 | Q15 | 0.6 | 4096 | 0 |

- Expected values: the f64 values `v` (`cf64`/`f64` arrays) and, for the Q15 cases, the integer model of R7 (`q15.py`) applied to them.
- Tolerances in the case JSON: `max_abs_err` 2e-7 · max(1, sigma) for f32/C32 against `v`, 1 LSB for Q15/CQ15 (a last-bit difference of the f64 logarithm can move a value across a rounding boundary). In practice the comparisons are expected to be exact; the test prints the number of non-exact values, and more than 4 per case fails.
- The generator seed and stream are stored in the case JSON (`numpy.random.default_rng(seed)` is not used: the sequence is the PCG32 model; the case seed only names the case).

### 5. Cost

`cargo xtask count` entries `gauss_noise_c32` and `gauss_noise_f32` (instructions per output sample at every level; the `libm` calls dominate). No reference library entry: GNU Radio and liquid-dsp produce other sequences, so their counts are not a like-for-like comparison. Record the numbers in `docs/performance.md`.

## Tests

Seeds 680 … 689.

1. Golden cases of §4 through the facade, `Fallback` and `Reference` (`sdrk-testkit/tests/golden_m48.rs`).
2. Unit tests (`sdrk-core/tests/kernels_m48.rs`): construction rules (`sigma` negative, NaN, infinite, above 16), `set_sigma`, `index` after calls, `seek` equals generating and discarding (for indices 0, 1, 4095, 2³² + 3), `reset`, `sigma = 0` gives zeros and advances, an empty output, the `InvalidState` of a type change, and `Fallback` against `Reference` for every type.
3. Statistics (`sdrk-testkit/tests/gauss_noise_m48.rs`, f32 and C32, 10⁶ samples, seed 685): the mean within ±5 σ/√n, the variance within ±1 % of `sigma²`, kurtosis within 3 ± 0.03, the Kolmogorov–Smirnov distance to the normal CDF below 1.63/√n (the 1 % level), the normalized autocorrelation at lags 1 … 16 below 4/√n, the correlation of the real and imaginary parts below 4/√n, and the Q15 clipping rate at `sigma` 0.6 within 10 % of the theoretical `erfc(1/(0.6 √2))`.
4. Cross-check with the testkit: `channel::Awgn::new(sigma, seed)` uses `Pcg32::new(seed, AWGN_STREAM)` and draws real then imaginary part per sample; `GaussNoiseState::new(seed, AWGN_STREAM, sigma)` with `C32` must give, sample for sample, `(v.re as f32, v.im as f32)` of the values `Awgn` adds to a zero input (seed 686, 10 000 samples, bit for bit). This pins the kernel to the noise all existing BER tests use. `AWGN_STREAM` becomes `pub` in the testkit for this test, or the test repeats the constant (implementer's choice, logged).
5. Block-size invariance (testkit helper of M03): 20 chunk plans (seed 687), every type.
6. Extremes: `sigma` 0, 16 (saturates nearly every Q15 value, f32 stays finite), the largest index `u64::MAX / 8` for `seek`.
7. Suite cases `gauss_noise_c32`, `gauss_noise_q15` and `extremes_m48` (`Fallback` against `Reference`, the facade against `Fallback`, chunk plans).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo xtask count --platform x86_64` (the two entries of §5).

## Deliverables checklist

- [ ] Module `kernels::noise`: `GaussNoiseState`, trait `GaussNoise`, Fallback, Reference, facade delegation, `backend_for`
- [ ] Contract row and section `gauss_noise` in `docs/kernels.md` (with the power convention and the GNU Radio and liquid-dsp conversions)
- [ ] Golden cases (seeds 680–684) and the Python model
- [ ] Tests 1–7 (seeds 680–687); suite cases
- [ ] `cargo xtask count` entries, `docs/performance.md`
- [ ] Reserved kind name `noise` in the table of `docs/blocks.md`
- [ ] CHANGELOG, decisions log (the reading of R4.4, the state's sample-type rule, the tolerances)

## Status

Planned. Nothing is implemented.
