# M53 — Kernel `code_search`: parallel code-phase acquisition

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M07](M07-fft.md), [M51](M51-spread.md) (the code type `Code` of the shared conventions §B) · Next: [M54](M54-code-track.md)

> **Status: planned** (written 2026-11-08, revised 2026-11-17 after an algorithm review (maintainer decision)). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

> **Revision 2026-11-17.** The shared conventions §A to §D cited in this file are defined in [CONVENTIONS-M49-M56.md](CONVENTIONS-M49-M56.md). The maintainer accepted every recommendation of the algorithm review of M49 to M56. The shared definitions are in the shared conventions of M49 to M56: §B (code format), §C.3 (`CQ15` backlog plan), §D (SIMD rule) and §D.3 (tolerance compositions). Changes in this milestone:
> 1. **Double block instead of a code zero-padded to one period (error corrected).** The input is periodic with the code length in samples `L = period · samples_per_chip`, not with `K`. A code zero-padded to `K` = the next power of two ≥ `L`, correlated with a dwell of `K` samples, is a correct circular correlation only at small lags. The review measured this with NumPy (`P = 1023`, `K = 1024`): the peak loses 2.5 dB at a quarter of the code phase and 5.5 dB at half of it, and at three quarters the maximum is 1 sample off. Now:
>    - `K` is the smallest power of two ≥ `2·L`, and the dwell is `K` samples.
>    - The code is one period followed by zeros.
>    - Only the lags `0 … L − 1` are searched and stored, and every one of them is exact (§1, §2).
>
>    The GLRT is normalised by the code energy `E_c` (`L` for a binary code); it replaces `2·K·S_max/P_in`, whose scaling did not match the normalised M07 inverse (§1). The golden case `doppler_511_2x` already used `K = 2048`, which contradicted the old block rule. All cases now follow one rule (§4).
> 2. **Doppler by bin shift.** A Doppler of `m·fs/K` is an exact circular shift of `FFT_K{x}` by `m` bins. A dwell therefore needs one forward FFT (after one wipe-off by `f_center`, only when it is not 0). Each bin then needs only the product with a shifted index, one inverse FFT and the grid update: `1 + D` FFTs per dwell instead of `2·D`. The Doppler step is `step_bins · fs/K`, an integer multiple of `fs/K` (§1, §2, §6). The wipe-off phase is no longer referenced to the absolute dwell start: a constant phase per bin and dwell cancels in `|R|²` (§1).
> 3. **SIMD in scope** (shared conventions §D and §D.3, maintainer rule 2026-11-17). `sdrk-backend-simd` gets a path on every level: the fallback's driver compiled at the level, with the backend `fft` and `rotate` of that level, as for `channelize` (D918). The spectral product, the grid update, the row maxima and sums and `P_in` follow fixed formulas and orders, so they give the fallback bits on every level. `backend_for` reports the SIMD level. The old sentence "`backend_for` reports `fallback` (the backend FFT is used inside)" was impossible under R2 (the `Fallback` in `sdrk-core` cannot call a backend) and is removed (§3, §5, test 7).
> 4. **Code format** (shared conventions §B, introduced in M51). The constructor takes a `Code<'_, C32>` and builds the code spectrum once with the `Fallback` FFT. The block generates its PRBS with `Prbs::fill_words`. The Depends line is now M07 and M51. M50 is removed from it: this kernel computes its own circular correlation and does not call `xcorr_fft` (Order, §2, §6).
> 5. **Smaller grid and a grid-less single dwell.** The grid holds `D·L` cells (the valid lags only) instead of `D·K`. Without grid storage, a call is a self-contained single-dwell search that keeps only the running maximum and the sums (§2). The block uses this mode for `dwells = 1`.
> 6. **Corrections and precisions:**
>    - The block's Doppler bin count is `D = 2·ceil(doppler_max_hz / f_step) + 1`: odd, centred, with the outermost bins at or beyond ±`doppler_max_hz`. The old `ceil(2·doppler_max/doppler_step)` left one edge a whole step away from the nearest bin when the count was even (§1, §6).
>    - The guard window of `peak_to_floor`, the tie rule of the peak search, the NaN result and the summation orders are now defined (§1).
>    - The weak golden case runs at about −20 dB chip SNR instead of −12 dB. At −12 dB, one dwell already finds the peak with a wide margin (§4).
>    - The golden peak margin is ≥ 1 dB over every other cell. With the finer Doppler step, the adjacent bins sit only about 3.9 dB below the peak (§3, §4).
>    - The false-alarm test names its noise source (testkit `Awgn`) and checks the per-cell χ² law (test 3).
>    - The pipeline cost formula follows the new method (§6).
>    - `CQ15` is a concrete backlog item with the plan of the shared conventions §C.3 (Scope).
> 7. **Cross-milestone consistency (M49–M56 pass).** The canonical sentences of the shared conventions §B (code) and §D (SIMD) are in §1 and §5; SIMD helper tests compare NaN by class (D296); the GLRT history wording is precise (the old `2·K·S_max/P_in` did not match the M07 inverse); the zero rule covers `F ≤ 0` and a non-finite `F`; the grid-cell limit also applies to `dwells = 1`; the block cost formula is marked to be confirmed.

## Goal

A kernel `code_search` that acquires a direct-sequence signal. It works over one or more dwells of `K` samples, where `K` is the smallest power of two that holds two code periods of `L = period · samples_per_chip` samples. It searches every code phase and a grid of Doppler frequencies at once with the FFT, and returns the strongest cell (code phase, Doppler, peak magnitude) together with a CFAR test statistic. It is the parallel-code-phase-search (PCPS) acquisition of a GNSS receiver (`GPS_L1_CA_PCPS_Acquisition` in GNSS-SDR), the first stage a DSSS receiver runs before it can track.

## Why

Before a receiver can despread, it must find where in the code it is (the code phase, anywhere in one code period) and how far the carrier has drifted (the Doppler), with the chip SNR deep below 0 dB. The FFT handles every code phase in one transform: `IFFT{ FFT{x} · conj(FFT{code}) }` is the correlation at all lags. A circular shift of the input spectrum covers the frequency axis. This is the only way to acquire a long code at a tractable cost, and it is the standard method of every software GNSS receiver.

## Order

M53 depends on:
- M07: the `fft` kernel and `FftPlan`.
- M51: the code type `Code`/`CodeBits` of the shared conventions §B, and `Prbs::fill_words`, which the block uses. Through M51 it also depends on M49 (the PRBS polynomials).

It reuses only the idea of M50 (`xcorr_fft`): it computes its own circular correlation per Doppler bin and does not call that kernel, so M50 is not a dependency (revision 2026-11-17). M53 can run in parallel with M52 and every later milestone. The DSSS receiver milestone that uses it also needs M54.

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

**In:**
- The kernel `code_search` (`C32`) in `sdrk_core::kernels::acquire`, with `CodeSearchState`, Fallback, Reference, the SIMD paths in `sdrk-backend-simd` on every level (shared conventions §D, §D.3), facade delegation, the contract and `docs/kernels.md`.
- Golden cases; unit, statistic, SIMD and extreme tests; suite cases.
- `cargo measure` and, for `simd128`, `cargo xtask count --platform wasm32`; CHANGELOG and decisions log.
- The block `CodeSearchBlock` and the pipeline kind `code_search` (§6), with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

Non-coherent integration across dwells is in: it is cheap and essential below 0 dB.

**Out:**
- The block and kind `code_acquire`, with the threshold policy (from a false-alarm probability) and the hand-off to `code_track`. This comes in a later milestone, and the name `code_acquire` is reserved. The primitive block `code_search` of §6 reports every search and decides nothing.
- A bit-synchronisation search over data-bit boundaries (backlog).
- Fine Doppler (`make_two_steps`; backlog).
- A Doppler step that is not a multiple of `fs/K` (backlog). It would need one wipe-off and one forward FFT per fractional sub-grid.
- A single-block mode `K = L` for code lengths that are themselves powers of two (backlog). It halves the cost, but the smallest Doppler step is then `fs/L`, a whole bin of the coherent time, with up to 3.9 dB scalloping loss (§1).
- Coherent integration across dwells (backlog). It would need the absolute wipe-off phase that §1 drops. Coherent integration over several code periods is the same kernel with a code of several periods (`L` and `K` grow); the block offers one period.
- `CQ15` (backlog, plan of the shared conventions §C.3): `rotate` `CQ15`, the M07 block-floating-point FFT forward and inverse, the code spectrum stored as `CQ15` with its own exponent, and `mag_sq` into a grid with a per-row exponent; tolerance class. Every one of these pieces exists already.

## Specification

General rules of M19 apply. The kernel works one dwell at a time and accumulates the search grid in caller storage (or, without a grid, keeps only the running result of one dwell), so a flow can run `max_dwells` dwells and stop early.

### 1. Method (PCPS with double block and bin shift, after GNSS-SDR)

**Code and sizes.** The code is a `Code<'_, T>` (shared conventions §B, introduced in M51): `Code::Bits` is the bit-packed binary form (chip `i` = bit `i & 31` of word `i >> 5`, chip value `1 − 2b`, chip `i` = the `i`-th `Prbs::next_bit` of a PRBS), applied by the exact sign operations of §A; `Code::Complex` is the general form with the `mul` product. Here `T` is `C32`; in this kernel the code only enters the code spectrum, built once in `new`, so the two forms differ only in the samples `c[n]` below.
- **Code samples:** `c[n]` for `n = 0 … L − 1` is chip `floor(n / samples_per_chip)` of the code, so `L = code length · samples_per_chip`. A `Code::Bits` chip `b` has the value `1 − 2b` (shared conventions §B.1); a `Code::Complex` chip has its complex value.
- **`K`:** the smallest power of two ≥ `max(2·L, 16)`, at most 65536 (so `L ≤ 32768`).
- **Code energy:** `E_c = Σ |c[n]|²`. It is exactly `L` for `Code::Bits`. For `Code::Complex` it is `samples_per_chip · Σ |chip|²`, summed in f64 in chip order once in `new`.
- **Code spectrum:** `C[k] = conj(FFT_K{ c padded with K − L zeros }[k])`. It is computed once in `new` with the `Fallback` forward `fft`, so the stored spectrum has the same bits on every backend.

**Doppler bins.** There are `D` bins. Bin `d` has the shift `m_d = (d − floor((D−1)/2)) · s` bins (`s = step_bins ≥ 1`) and the frequency `f_d = f_center + m_d · fs / K`. Bin `floor((D−1)/2)` is `f_center`, and the step is `f_step = s · fs / K`.

**Per dwell** `x[0 … K−1]`:

1. **Centre wipe-off:**
   - If `f_center ≠ 0`: `x_c[n] = x[n] · e^(−j 2π f_center n / fs)`, computed by the `rotate` kernel (an `Nco` at `−f_center / fs`, phase 0 at the first sample of every dwell).
   - If `f_center = 0`: `x_c = x`, with no product (exact, also for non-finite values and signed zeros).
2. `X = FFT_K{ x_c }`: one forward FFT per dwell.
3. **Bin shift and product.** For each bin `d`: `Y_d[k] = X[(k + m_d) mod K] · C[k]`.
   - `X[(k + m) mod K]` is exactly the spectrum of `x_c[n] · e^(−j 2π m n / K)`, by the shift theorem over the `K`-sample window. This replaces a time-domain wipe-off and a forward FFT per bin.
   - The index is two contiguous runs of `X`, so no per-element modulo is needed.
   - The product uses the fixed formula `re = ar·br − ai·bi`, `im = ar·bi + ai·br`: the four products first, no FMA (the `mul` fallback formula).
4. `R_d = IFFT_K{ Y_d }`, the normalised inverse of M07 (factor `1/K`). In exact arithmetic,
   `R_d[τ] = Σ_{n=0}^{L−1} conj(c[n]) · x_c[n + τ] · e^(−j 2π m_d (n + τ) / K)` for `τ = 0 … L − 1`.
   - Since `n + τ ≤ 2L − 2 < K`, no lag wraps. Every lag is a true correlation with one full code period.
   - Only `R_d[0 … L − 1]` is used.
5. **Grid:** on the first dwell after `new` or `reset`, `g[d][τ] = |R_d[τ]|²`; after that, `g[d][τ] = g[d][τ] + |R_d[τ]|²`.
   - `|R|² = re·re + im·im` in f32 without FMA (the `mag_sq` formula), and the addition is done as written.
   - This covers `τ = 0 … L − 1`.

- **Code phase.** `τ` is the input sample at which a code period starts in the dwell: the code phase in samples, `0 … L − 1`. A delay of the input by `s` samples moves it to `(τ + s) mod L`.
- **Wipe-off phase.** The start phase of a dwell multiplies `R_d` by a constant per bin and dwell, which cancels in `|R|²`. So every dwell starts the wipe-off at phase 0, and the kernel keeps no wipe-off phase across dwells. The absolute phase would matter only for coherent integration across dwells, which is out of scope.
- **Double-block trade-off.** The second part of a dwell only supplies the samples for the later lags, so a dwell of `K ≥ 2L` samples integrates one code period coherently.

**Per call, after the grid update** (`v[d][τ]` is the grid value, or in grid-less mode the dwell's `|R_d[τ]|²`):

- **Peak:** `S_max` is the largest `v`.
  - Ties go to the smallest `d`, then the smallest `τ`. Comparisons are `>`, so a NaN never becomes the maximum.
  - If any `v` is NaN (detected through the row sums), `peak`, `glrt` and `peak_to_floor` are NaN, and `code_phase` and `doppler_bin` are 0.
- **Input power:**
  - Per dwell, `p = (Σ_n |x[n]|²) / K` over the raw dwell, before the wipe-off. `|x|² = re·re + im·im` is computed in f32 and summed in f64.
  - `P_in` is the mean of `p` over the dwells since `new` or `reset`, in f64.
- **GLRT:** `Γ = 2·S_max / (E_c · P_in)`, computed in f64 and rounded once to f32.
  - Under noise only (white complex Gaussian), `2·v / (E_c · P_in)` of each cell follows approximately χ² with `2·dwells` degrees of freedom (mean `2·dwells`), independent of the input level. Γ is therefore a CFAR statistic.
  - A NumPy check in this revision (L = 1023, K = 2048, 200 searches) gave a mean of 2.00 and a variance of 4.0 for one dwell, and 7.99 and 15.9 for four dwells.
  - It plays the role of GNSS-SDR's `Gamma_GLRT`. The normalisation by `E_c` rather than `K` follows from the double block: only `L` samples enter a lag.
- **Peak to floor:** `Γ_floor = S_max / F`, with `F = (T − G) / (D·L − g)`.
  - `T` is the sum of all `v`.
  - `G` is the sum of the guard window: the `g = min(2·samples_per_chip + 1, L)` cells of the peak row `d*` whose circular lag distance (mod `L`) to `τ*` is at most `samples_per_chip`.
  - Only the peak row is guarded. The adjacent Doppler bins add a few cells near the peak, which change the mean of the `D·L − g` other cells negligibly at acquisition grid sizes, and the rule keeps the grid-less mode exact.
- **Zero rules:** `P_in = 0` gives `Γ = 0`, and `F ≤ 0`, a non-finite `F` or `D·L = g` gives `Γ_floor = 0`. An all-zero dwell therefore reports 0 and 0. `F` can come out slightly negative because `T` and `G` are summed in different orders, and a very large finite input can make cells `+∞` without a NaN (then `T − G` is NaN); both cases are covered by this rule, and the result is the same on every level because the formulas are fixed.
- **Summation orders** (fixed, so every level gives the same bits for the same `v` and `x`; shared conventions §D.2):
  - `T`, the per-dwell sum of `p` and every row sum are f64 sums with eight partial sums per row (or per dwell). Term `τ` (or `n`) goes to partial sum `τ mod 8`, and the partial sums are combined by the fixed tree of §D.2. The rows are added in ascending `d`.
  - `G` is summed in ascending circular order from `τ* − samples_per_chip`.
  - Every f32 → f64 conversion is exact.

**Output:** `CodeSearchResult { code_phase: τ*, doppler_bin: d*, doppler_hz: f_{d*} (f64), peak: S_max, glrt: Γ, peak_to_floor: Γ_floor, input_power: P_in, dwells }`. The kernel declares nothing acquired or not acquired. A later milestone's block compares `Γ` with a threshold derived from a false-alarm probability.

**Doppler grid.**
- **Distinct bins:** `(D − 1) · s < K` keeps the shifts distinct.
- **Step size:** the step `fs/K ≈ fs/(2L)` is half the width `1/T` of the Doppler response of one coherent period (`T = L/fs`). The worst loss at half a step is 0.91 dB for `s = 1`, against 3.92 dB for a step of `fs/L` (NumPy, `L = 1023`, computed in this revision).
- **Bin placement:** GNSS-SDR places `ceil(2·doppler_max/doppler_step)` bins from `−doppler_max` upwards. This kernel centres the bins on `f_center`, and the block chooses an odd `D` (§6). The difference is recorded with the cross-check of §4.

### 2. State and trait

```rust
pub struct CodeSearchState<..> { /* L, K, D, step_bins, fs, f_center, E_c, samples_per_chip, dwell_count, P_in sum,
                                    spectrum C (K C32), grid (D*L f32, or empty), scratch (2*K C32), FftPlan<K> */ }
impl CodeSearchState<..> {
    pub fn new(code: Code<'_, C32>, samples_per_chip: u32, doppler_bins: u32, step_bins: u32,
               f_center: f64, fs: f64, spectrum: SpecStore, grid: GridStore, scratch: ScratchStore,
               fft: FftPlan<..>) -> Result<Self>;
    pub fn reset(&mut self);              // clears the grid, the dwell count and P_in
    pub fn dwells(&self) -> u32;
    pub fn code_len(&self) -> usize;      // L = code length · samples_per_chip, the lags searched
    pub fn fft_len(&self) -> usize;       // K, the dwell length
    pub fn doppler_step_hz(&self) -> f64; // step_bins · fs / K
    pub fn is_grid_less(&self) -> bool;
}
pub trait CodeSearch {
    /// One dwell of exactly K samples. Accumulates into the grid; returns the current best.
    fn code_search(&self, state: &mut CodeSearchState<..>, dwell: &[C32]) -> Result<CodeSearchResult>;
}
```

- **Parameter validation** (`InvalidParameter`; errors of the code itself come from `CodeBits::new`, shared conventions §B.4):
  - `samples_per_chip` 0, or `L > 32768`;
  - `D = 0`, `step_bins = 0`, or `(D − 1) · step_bins ≥ K`;
  - `fs` not finite or not positive, or `f_center` not finite;
  - a `Code::Complex` with a non-finite chip or `E_c = 0`.
- **Storage validation** (`LengthMismatch`): `spectrum.len()` must be `K`, `scratch.len()` must be `2·K`, `grid.len()` must be `D·L` or 0 (grid-less), and the FFT plan must have size `K`.
- **Dwell length:** `dwell.len()` must equal `K` (`LengthMismatch`).
- **Grid mode:** each call runs one dwell (steps 1–5 for every bin) and returns the running result over all dwells since `new` or `reset`. Calling it `n` times does `n` non-coherent dwells. `reset` starts a fresh search (e.g. for the next code or after a failed acquisition).
- **Grid-less mode** (empty grid): every call is a self-contained one-dwell search. It computes the result from the rows as they are produced: the running maximum, `T`, and the `G` of the row that holds the running maximum (computed when that row takes the lead, from the same `|R|²` values). The dwell count and `P_in` restart at every call, so `dwells()` is 1 after a call. For one dwell, the result equals that of the grid mode bit for bit (same values, same orders).
- **Timing:** the kernel is a block-wise analysis step, not a streaming rate-changer. The block of the later milestone owns the stream and feeds dwells. There is no `Progress`. Group delay is not meaningful, because the result reports an absolute code phase.
- **Storage:** the spectrum (`K` `C32`), the scratch (`2·K` `C32`: `X`, and the product and inverse-FFT buffer), the grid (`D·L` f32, or none) and the FFT plan twiddles. `storage_bytes` is exact.
  - Example: `L = 1023`, `K = 2048`, `D = 21` gives a grid of 85,932 bytes. That is about the size of the old `D·K` grid at `K = 1024` (86,016 bytes), although `K` doubled.
  - Spectrum and scratch take 48 KiB.

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `code_search` | `kernels::acquire::CodeSearch` | `C32` | tolerance | SNR ≥ 90 dB on the accumulated grid vs. the f64 model of §1 (per bin the error of one forward FFT, a spectral product and one inverse FFT, plus the wipe-off when `f_center ≠ 0`; the `fft` tolerance dominates); the peak index and the Doppler bin exactly, both statistics within the grid SNR, on golden cases whose true peak exceeds every other cell by ≥ 1 dB; `C` is the `Fallback` FFT of the code on every backend; the spectral product, the grid update, the peak search, the sums and `P_in` follow the fixed formulas and orders of §1 on every level, so fallback and SIMD levels differ only through the `fft` and `rotate` of their level (shared conventions §D.3): fallback, facade and every SIMD level each reach the floor against `Reference` and agree with each other within it; per backend and level the result is deterministic, `P_in` and the dwell count equal the fallback bit for bit, and the block's packets are chunk-invariant bit for bit (`==`) |

**Reference.** `Reference` computes the definition of step 4 directly in f64.
- **Method:** for each bin, the sum over `n` with the phasor `e^(−j 2π (f_center/fs + m_d/K) j)` at the input index `j = n + τ`, from `libm::cos`/`libm::sin` in f64. This is `O(D·L²)` at the golden sizes; larger `K` uses the f64 FFT path that the reference exposes. The reference therefore checks the bin shift against an explicit time-domain wipe-off.
- **Statistics:** computed from the f64 grid with the formulas of §1.
- **Index stability:** the index of the maximum could in principle flip between two near-equal cells under rounding. The golden cases are therefore built so that the true peak exceeds every other cell by ≥ 1 dB (the generator checks this and stores the margin). With the step `fs/K`, the adjacent Doppler bin of a signal on a bin centre sits only about 3.9 dB below the peak, so a larger margin cannot be required of every cell. The tests assert the indices exactly and the statistics within tolerance.
- **Backend report:** `backend_for("code_search", C32)` reports the SIMD level where a path exists (every level, §5) and `fallback` in a fallback-only build.

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/acquire.py`.

| Case | Seed | `K` | `L` (code, `samples_per_chip`) | `D`, `step_bins`, `f_center` | dwells | Scenario |
| --- | --- | --- | --- | --- | --- | --- |
| `code_search/clean_1023_1x` | 730 | 2048 | 1023 (PRBS10, 1) | 21, 1, 0 | 1 | code at `τ = 767` (three quarters of the period, where the old single block was 1 sample off) and 0 Hz, high SNR; peak at `τ`, bin = centre |
| `code_search/doppler_511_2x` | 731 | 2048 | 1022 (PRBS9, 2) | 41, 1, 0 | 1 | a nonzero Doppler on a bin centre and a code phase above `L/2`; the statistics locate both |
| `code_search/weak_1023_8dwell` | 732 | 2048 | 1023 (PRBS10, 1) | 21, 1, 0 | 8 | about −20 dB chip SNR (to be confirmed by the generator); the grid after 1 and after 8 dwells is stored, and the indices are asserted only after 8 |
| `code_search/centre_127_5x_s2` | 736 | 2048 | 635 (PRBS7, 5) | 15, 2, `f_center ≠ 0` (not a multiple of `fs/K`) | 2 | the wipe-off path, a step of two bins, `K > 2L`; the Doppler on a bin centre, the code phase near `L − 5` |

- **Weak case level.** A NumPy check in this revision (L = 1023, D = 21, 10 trials per level) found the true cell after one dwell in 10 of 10 trials at −12 dB, in about 2 of 10 at −20 dB, and after eight dwells at −20 dB in 10 of 10, with at least 4.5 dB over the runner-up outside a 3 × 3 window. The generator picks the level and checks that one dwell does not give a 1 dB margin while eight do.
- **Stored inputs:** the code as `code_words` and `code_len` (shared conventions §B.2), `samples_per_chip`, `K`, `D`, `step_bins`, `f_center`, `fs`, and the input (code · carrier + noise from `numpy.random.default_rng`).
- **Expected values:** the grid of the f64 model rounded once to float32 (that rounding is about 150 dB, far above the 90 dB floor; it keeps the weak case, with two grids and eight dwells, near 300 KB, below 512 KiB), the peak index, the Doppler bin, `Γ`, `Γ_floor`, `P_in` and the peak margin.
- **Tolerance:** grid `tolerance` SNR 90; indices exact; statistics `max_abs_err`, scaled to their value.
- **GNSS-SDR cross-check:** the `Γ` normalisation (by `E_c`, the χ² law under noise) and the Doppler-bin placement (a centred odd grid against GNSS-SDR's grid from `−doppler_max`) are cross-checked against the GNSS-SDR documentation and source, cited in the decisions log. The page's "conjugate of code FFT" wording against its `D[k]` formula is noted: we conjugate, which is the correct correlation.

### 5. Cost, SIMD plan and reference rows

**Entries.** `cargo measure` entries:
- `code_search_2048_21bins`: `L = 1023`, `K = 2048`, `D = 21`, `step_bins` 1, `f_center` 0; per dwell, reported per input sample as cost / `K`;
- `code_search_2048_21bins_wipe`: the same with `f_center ≠ 0`;
- the block entry `code_search_block` of §6.

Each SIMD entry is compared with its `_fallback` entry (D321) on every level. Record the numbers in `docs/performance.md`, and note that acquisition runs once per lock, not per sample, so throughput is secondary.

**Estimated cost (to be confirmed by `cargo measure`).**
- **FFT count:** a dwell runs `1 + D` FFTs of size `K`: 22 FFTs of 2048 points for `D = 21`. The old (incorrect) single block ran 42 FFTs of 1024 points.
- **Comparison with the old method:** in the `5·N·log2 N` model, the new dwell is about 1.15× the work of the old one, for a dwell twice as long. The double block without the bin shift would run 42 FFTs of 2048 points, about 2.2× the old work (the review's figure).
- **Per input sample:** about `(D + 1) · 2.5 · log2 K + D · (4 + 2·L/K)` multiply-accumulates, ≈ 710 at `K = 2048`, `D = 21` (the pipeline cost unit, §6).

**SIMD plan (`sdrk-backend-simd`, shared conventions §D.3).** SIMD follows the shared conventions §D (maintainer rule 2026-11-17): every lane-parallel part of the kernel gets a path on every level in this milestone, bit-identical to the fallback through the fixed element-wise formula or reduction order of the contract; D321 is a quality gate (a slower path is improved, not removed; delegation to the fallback only after several measured failed attempts, with an open item); cycles per sample with `cargo measure` against the fallback and the fair reference (D32), unfair rows marked. For this tolerance composition, bit identity with the fallback holds for the fixed-formula parts; the composed `fft` is tolerance at some levels, so the whole kernel is bit-identical per backend and level and within the SNR floor against the reference (§3, shared conventions §D.3). The paths are on every level: `sse2`, `sse4.2`, `avx2`, `avx512`, `neon` and `simd128`.
- **Driver:** the fallback driver (`code_search_with`) compiled at the dispatched level, calling the backend `C32` `fft` (forward and inverse) and the backend `rotate` at that level, as `channelize` does (D918).
- **Shifted spectral product:** the two runs of `X` times `C`, with the fixed formula of step 3 (no FMA on any level).
- **Grid update and row reductions:** one pass over `R_d[0 … L − 1]` computes `re·re + im·im` (no FMA) and the grid update. In the same pass it finds the row maximum, with the index carried per lane and the lanes reduced by value and then lower index (the tie rule of §1), and it forms the row sum in f64 lanes in the eight-partial order (`cvtps2pd`, `vcvt_f64_f32`, `f64x2.promote_low_f32x4`).
- **`P_in`:** the same f64 eight-partial sum over the raw dwell.
- **Scalar parts, with the reason:** combining the `D` row results and computing the two statistics run once per row or once per call. That is `O(D)` work with no lane-parallel part worth a vector. The code spectrum in `new` is the `Fallback` FFT, run once.
- **Fallback:** written so that the compiler can vectorise the element-wise loops.
- **D321:** the gate applies to every level as a quality gate. A path slower than its fallback on either core model is improved, not removed. Only after several measured failed attempts (each recorded in `docs/decisions-log.md`) may a level dispatch to the fallback, with an open item.

**Reference rows (D32).** No library row is transcribed.
- GNSS-SDR's `pcps_acquisition` is the same method family but does different work: a time-domain wipe-off and a forward FFT for every Doppler bin, `2·D` FFTs per dwell against our `1 + D`. If an informational row is added, it is marked **reference does different work**.
- Its pieces already have rows of their own: the wipe-off as `rotate` against GNU Radio's rotator, the FFT of M07, and the spectral product as `mul`.
- The speed baseline is the fallback (D321) and the rows of the composed kernels.

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model). Cycles are the speed measure and the only one on the out-of-order levels (M13.1, D2700); instruction counts (`cargo xtask count --platform wasm32 --merge`) stand only where no core model exists (`simd128`, Cortex-M). A library reference enters only where it does the same work (D32); otherwise the rows are information. The D321 gate applies to every SIMD path.

### 6. Block and kind `code_search`

`CodeSearchBlock` (`C32` → packets) wraps the kernel `code_search`.
- **Code:** a PRBS (`sdrk_proto::prbs`, generated into `u32` words with `Prbs::fill_words`, chip value `1 − 2b`; shared conventions §B), given to the kernel as `Code::Bits` with `samples_per_chip`.
- **Sizes:** `L = period · samples_per_chip`, and `K` = the smallest power of two ≥ `2·L`, at most 65536 (`prbs15` only at `samples_per_chip` 1).
- **Dwells:** the block cuts its input into back-to-back dwells of `K` samples from the first sample after construction or `reset`. After `dwells` dwells it writes one packet with the result and starts the next search (`CodeSearchState::reset`).
- **Spectrum and grid:** the kernel builds the code spectrum once. For `dwells = 1` the block runs the kernel grid-less, with no grid storage.

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `code_search` | `CodeSearchBlock` | C32 → packets | `code`, `seed` (as `spread`), `samples_per_chip` (I64, build, 1; 1 … 16), `doppler_max_hz` (Hz, build, required; > 0, below `rate / 2`), `doppler_step_hz` (Hz, build, required; > 0, at most `doppler_max_hz`), `doppler_center_hz` (Hz, build, 0), `dwells` (I64, build, 1; 1 … 1024) | `code_search`; `step_bins = max(1, floor(doppler_step_hz · K / rate))`, so the actual step `step_bins · rate / K` is at most the requested one (one bin of `rate / K` when the request is smaller); `D = 2 · ceil(doppler_max_hz / (step_bins · rate / K)) + 1` bins, centred on `doppler_center_hz` (§1); `period · samples_per_chip > 32768` is `invalid_value` at `params.samples_per_chip`; `(D − 1) · step_bins ≥ K` is `invalid_value` at `params.doppler_max_hz`; more than 2^22 grid cells `D · L` is `invalid_value` at `params.doppler_step_hz` (also for `dwells = 1`, where no grid is stored, so the valid parameter set does not depend on `dwells`) |

- **Packet:** 36 bytes little-endian: `code_phase` (u32, samples, `0 … L − 1`), `doppler_bin` (u32), `doppler_hz` (f64), `peak`, `glrt`, `peak_to_floor`, `input_power` (f32 each), `dwells` (u32). `index` is the input index of the first sample of the search, and the flags are 0. `CodeSearchReport::from_packet` reads it back.
- **`CodeSearchConfig` and storage:** the code words (`ceil(period / 32)` `u32`), the spectrum (`K` `C32`), the scratch (`2·K` `C32`), the grid (`D · L` f32, only for `dwells > 1`) and the twiddles; exact. The node report shows the actual Doppler step and `D`.
- **Timing:** `Packets { max_per_input: 1/(K · dwells) }`, delay 0.
- **Pipeline kind** `code_search` (`c32` only, a `PacketOutNode`):
  - Cost per input sample: `2.5 · log2(K) · (D + 1) + D · (4 + 2 · L / K)`, plus 4 when `doppler_center_hz ≠ 0` (to be confirmed). This covers one forward FFT per dwell, then per bin the shifted spectral product, one inverse FFT and the grid update over `L` lags, plus the centre wipe-off. The implementer checks it against the count and logs it.
  - A row in `docs/pipeline.md` and one in `docs/timing.md`.
- **Common rules of M29:** the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room. `timing()` takes ratio and delay from the kernel state. `storage_bytes` is exact, and the struct is at most 512 bytes. The description has a summary and details, and every parameter has its unit, mutability, default and range.

## Tests

Seeds 730 … 739.

1. Golden cases of §4 through the facade, `Fallback`, `Reference` and every SIMD level of the build (`sdrk-testkit/tests/golden_m53.rs`): grid within 90 dB, exact indices, statistics within tolerance.
2. Unit tests (`sdrk-core/tests/kernels_m53.rs`):
   - **Construction:** the `new` rules (`samples_per_chip` 0, `L > 32768`, `D = 0`, `step_bins = 0`, `(D − 1)·step_bins ≥ K`, bad `fs`, non-finite `f_center`, a `Code::Complex` with a non-finite chip or zero energy, wrong storage lengths, an FFT plan of another size). `K` is the smallest power of two ≥ `max(2L, 16)` for `L` = 1, 7, 8, 1022, 1023 and 32768.
   - **Peak position:** a clean code at `τ = 0` and 0 Hz peaks at (centre bin, 0) with the value `a² · E_c²` for amplitude `a`. A delay of the input by `s` samples moves `τ*` to `s mod L` with the same peak value, for every `s` in `0 … L − 1` at `L = 127` and for `s` = `L/4`, `L/2`, `3L/4` and `L − 1` at `L = 1023` (the lags where the old single block failed). A Doppler of one step moves `d*` by one.
   - **Bin shift and wipe-off:** the bin shift equals the explicit time-domain wipe-off within 90 dB, for `m_d` of both signs and across the wrap of `X`. With `f_center = 0` no product runs: the spectrum `X` equals the `Fallback` `fft` of the raw dwell bit for bit. With `f_center ≠ 0` the wipe-off runs.
   - **Dwells and modes:** `reset` clears the grid, the dwell count and `P_in`. Two dwells accumulate: the grid doubles for a static signal and `P_in` is averaged. The grid-less mode equals the grid mode bit for bit for one dwell (every result field) and restarts at every call.
   - **Edge rules:** `LengthMismatch` on a wrong dwell length. The tie rule: two equal cells give the lower `d`, then the lower `τ`. A `Code::Complex` of `(±1, 0)` chips gives the binary result within the tolerance for finite input.
3. Statistic behaviour (`sdrk-testkit/tests/code_search_m53.rs`, seed 733; noise from the testkit channel model `Awgn`, used only statistically):
   - Over 200 noise-only searches, the cell statistic `2·v / (E_c · P_in)` exceeds the χ² threshold with `2·dwells` degrees of freedom for a per-cell false-alarm probability of 10⁻² at a rate within a factor of 2 of 10⁻², for 1 and 4 dwells.
   - With a signal at −20 dB chip SNR, Γ after 8 dwells exceeds the threshold of a search false-alarm probability of 10⁻² (the χ² threshold for `D·L` cells treated as independent) in most of 50 searches (the rate to be confirmed in this milestone and recorded).
   - This is a small ROC sanity check; full ROC tooling is a later testkit milestone.
4. Extremes:
   - `D = 1` (a single Doppler bin);
   - the largest `K` (65536, `L = 32767`, PRBS15, `samples_per_chip` 1) with `D = 3`;
   - an all-zero dwell: grid 0, `P_in` 0, both statistics 0 by the zero rules of §1;
   - full-scale input: a finite grid;
   - NaN or ∞ in one input sample: the forward FFT spreads it, so every cell of that dwell is NaN, the result is NaN with indices 0, and `reset` recovers.
5. Suite cases `code_search_c32` and `extremes_m53`.
6. **Block and kind** (`sdrk-blocks/tests/code_search_blocks.rs`, `sdrk-pipeline/tests/code_search_kinds.rs`; seeds 734, 735):
   - **Results:** the block gives the kernel results (the packet fields bit for bit) on whole buffers and in 20 chunk plans. A PRBS10 code at a known code phase above `L/2` and a known Doppler is found, one packet per search, with `index` at the search start.
   - **Parameter mapping:** `doppler_step_hz` maps to `step_bins` (a request below one bin, at one bin, and between two multiples), and `D` is odd and centred. With `dwells = 1` the block runs without a grid, and its storage holds none.
   - **Block rules:** the description rules of M09, exact storage, a struct of at most 512 bytes, and validation errors at their parameter (`samples_per_chip`, `doppler_max_hz`, `doppler_step_hz`).
   - **Pipeline:** a pipeline flow with a packets output equals the block. `cargo xtask schema --check` and the WASM kind list include `code_search`.
7. **SIMD** (`sdrk-backend-simd/tests/code_search.rs`, seed 737; shared conventions §D.1, item 6):
   - **Helpers:** on every level of the build, the element-wise helpers and reductions equal the fallback helpers bit for bit. These are the shifted spectral product (`m_d` negative, 0 and positive, and at the wrap), the grid update with the row maximum and row sum, and the `P_in` sum. They are checked on lengths 0 to 67 and 4096 and on special values (NaN, ±∞, −0, subnormals, equal maxima in different lanes for the tie rule); NaN lanes are compared by class (D296), every other lane by `to_bits()`.
   - **Whole kernel:** against `Fallback` and `Reference` within the 90 dB floor with exact indices, on the golden cases and on random cases (`L` from 1 to 1023, `D` from 1 to 41, `step_bins` 1 to 3, `f_center` 0 and not 0).
   - **Determinism:** two runs give `==` results, and the block's packets give `==` results in 20 chunk plans. `P_in` and the dwell count equal the fallback bit for bit after every call.
   - **Backend report:** `backend_for` reports the level.

## Acceptance

- The R1.2 commands.
- `python tests/golden/generate.py --check`.
- `cargo test -p sdrk-core --all-features`, `cargo test -p sdrk --all-features`, `cargo test -p sdrk-backend-simd --all-features`, `cargo test -p sdrk-testkit --features std`, `cargo test -p sdrk-blocks --all-features` and `cargo test -p sdrk-pipeline --all-features`.
- `cargo xtask schema --check`.
- `cargo xtask count --platform wasm32`: the `simd128` instructions of the entries of §5 and `code_search_block`.
- `cargo measure`: the same entries and their `_fallback` entries, on every level with a core model.
- `cargo xtask check cycles`.

## Deliverables checklist

- [ ] Kernel `code_search` (`C32`): `CodeSearchState` (grid and grid-less mode), `CodeSearchResult`, trait, Fallback (double block, bin shift), Reference (explicit wipe-off), facade delegation, `backend_for`
- [ ] SIMD paths in `sdrk-backend-simd` on every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`): the driver with the backend `fft` and `rotate` of the level, the fixed-formula product, grid update and reductions (shared conventions §D.3)
- [ ] Contract row and section in `docs/kernels.md` (the double block and the valid lags, the bin shift, both statistics and their normalisation, the guard window, the tie and NaN rules, the summation orders, the Doppler grid, the GNSS-SDR cross-check)
- [ ] Golden cases (seeds 730–732, 736) and the Python model
- [ ] Tests 1–7 (seeds 730–737); suite cases
- [ ] Count and cycle entries (`cargo measure`, `cargo xtask count --platform wasm32`, against the fallback entries), `docs/performance.md`
- [ ] Block `CodeSearchBlock`, `CodeSearchReport` and kind `code_search` (§6): `docs/blocks.md` (entry and reserved-name row), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `code_search_block`
- [ ] CHANGELOG, decisions log (the double block and the GLRT normalisation by `E_c`, the bin shift and the step rule, the dropped absolute wipe-off phase, the conjugation, the two statistics and the guard window, the zero-division guard, the GNSS-SDR citation and the bin placement, the SIMD measurements)

## Status

Planned. Nothing is implemented.
