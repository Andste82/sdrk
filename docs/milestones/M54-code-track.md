# M54 — Kernel `code_track`: delay-locked loop

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M38](M38-symbol-sync.md), [M52](M52-despread.md) · Next: [M55](M55-pilot-equalize.md)

> **Status: planned** (written 2026-11-08, revised 2026-11-17 after an algorithm review (maintainer decision)). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

> **Revision 2026-11-17.** The shared conventions §A to §D cited in this file are defined in [CONVENTIONS-M49-M56.md](CONVENTIONS-M49-M56.md). The maintainer accepted all recommendations of the algorithm review of M49 to M56; the shared definitions are the shared conventions §A to §D (written into `docs/kernels.md` by the milestones that introduce them). Changes in this milestone, marked *(rev. 2026-11-17)* where they appear:
> 1. **Correlate, then interpolate once per period** (§1). The old form interpolated every input sample three times with the cubic of `symbol_sync` (about 50 to 70 flops per sample; the review's estimate). Now the arms are formed from despread sums at a few integer lags (5 to 12 lags, sign operations and additions only for a binary code), and the cubic formula of M38 combines the four lag sums of each arm once per period. The code-rate drift inside one period is not followed (documented, as in GNSS receivers).
> 2. **Fixed-point position, integer `samples_per_chip`** (§1, §2). The position is a Q32.32 value (integer index and `u32` fraction), advanced once per period; `samples_per_chip` is a `u32`. No `f64` runs while processing.
> 3. **The bit-packed code** of the shared conventions §B (introduced in M51), chips applied by the exact sign operations of §A (introduced in M49); long codes through the code cursor of §B.3.
> 4. **SIMD in scope** (§6, shared conventions §D): the lag sums and the per-period combination get SIMD paths on every level in this milestone, bit-identical to the fallback through the fixed summation order (NaN sign and payload excepted, D296); only the discriminator, the loop filter and the position update stay scalar, with the reason stated. "Not accelerated" is withdrawn; `backend_for` reports the SIMD level. New test 9 (seed 749).
> 5. **Discriminator corrected** (§2). The GNSS-SDR normalisation factor is `1 − δ`, not a fixed `0.5` (that is its value at `δ` = 0.5 only); with it the slope is 1 per chip for the ideal correlation triangle at every `δ`. At `δ` = 1 the factor is 0, so `early_late_chips` ends at 0.9. The sign is fixed (`e > 0`: the signal is late). The constant left to the implementer is replaced by the parameter `disc_gain` (as `ted_gain` of `symbol_sync`).
> 6. **Smaller errors corrected** (§2, §3, §7): `max_rate` 0.05 chips per period is a relative clock error of `0.05/sf`, not ±5 %; `loop_bw` is normalised to the period (symbol) rate; carrier aiding shortens the period in samples (the old factor `1 + f/f_rf` on the advance had the wrong sign); the length rule of `carrier_doppler_hz` is checkable at the entry (0 or `output.len()`); the step clamp is defined; the GNSS-SDR citation is limited to the discriminator.
> 7. **State as running accumulators** (§1, §3): lag accumulators and a short history (`ceil(2d) + 2·spc + 3` samples, independent of the period length) in caller storage, never a buffered period; `min_input` 1.
> 8. **Cost and references** (§6, §7): entries with their `_fallback` floor; the fair reference row `volk_gnsssdr` resampler plus `dot_prod_xn` (the old "no library row" held only without the interpolation cost); the pipeline cost formula re-derived.
> 9. **`CQ15` as a concrete backlog item** (Scope).
> 10. Golden cases store the code as `code_words`; a new case with carrier aiding and `samples_per_chip` 4 (seed 748).
> 11. Precision fixes found when the revision was checked: (a) **NaN and D296** (§4 *Contract*, §6, test 9): the bit identity of the SIMD levels with the fallback excepts the sign and payload of a NaN, as `despread` (M52) does for the same sums; NaN lanes are compared by class. (b) **Lock level at non-finite input** (§2, test 6): a non-finite lock metric skips the smoothing update, so one NaN/∞ sample can no longer leave the lock state NaN for good. (c) The loop-filter denominator is named `den` (§2), so `d` means only the arm offset in samples. (d) `n_lags` reaches 12 once `2d > 7` (it said `2d > 8`). (e) The `−0` exception of the zero-bandwidth identity is stated by its effect (§3). (f) Test 6 no longer bounds the number of periods that see a non-finite sample by two (with short periods a sample lies in three or more period windows).

## Goal

A kernel `code_track` that keeps a direct-sequence receiver aligned to the chip clock after acquisition: a delay-locked loop with early, prompt and late correlators, a non-coherent code discriminator and a loop filter. It consumes the chip stream and outputs one prompt (despread) symbol per code period. The correlators are despread sums at a few integer sample lags, combined once per period with the cubic interpolation formula of `symbol_sync` at the loop's sub-sample position (*rev. 2026-11-17*: correlate, then interpolate), so the work per input sample is a handful of sign operations and additions, and it runs on SIMD on every level. With the loop bandwidth set to zero and the on-time phase fixed it reduces, on the prompt arm, to `despread` (M52) bit for bit. The discriminator follows the normalised early-minus-late form of GNSS-SDR's DLL (`dll_nc_e_minus_l_normalized`, BPSK); the loop filter is the second-order filter of `symbol_sync` and `costas`.

## Why

Acquisition (`code_search`, M53) gives a coarse code phase and Doppler; from there a tracking loop must follow the chip clock as it drifts (oscillator error, Doppler, ppm clock offset), or the correlation collapses within milliseconds. The early-minus-late discriminator measures the alignment error, and a loop filter steers the sub-sample position of the correlators to the chip timing. This is the receive-side counterpart of `symbol_sync`, but at the chip rate and with a code: without it a long-code DSSS link cannot stay despread.

## Order

M54 depends on M38 (the interpolation formula and loop-filter form it reuses) and M52 (`despread`, which the prompt arm must equal at zero bandwidth). Through M52 it also has M51 (the bit-packed code of the shared conventions §B) and M49 (the exact sign operations of §A), so the *Depends on* line is unchanged (*rev. 2026-11-17*). It can run in parallel with every milestone after M52; the DSSS receiver milestone needs M53 and M54 together.

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

**In:** kernel `code_track` (`C32`) in `sdrk_core::kernels::code_track` with `CodeTrackState`, Fallback, Reference, SIMD paths in `sdrk-backend-simd` on every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`; §6, *rev. 2026-11-17*), facade delegation, `backend_for`, contract, `docs/kernels.md`; golden cases; unit, lock, invariance, extreme and SIMD tests; suite cases; `cargo xtask count` and `cargo measure` entries; CHANGELOG, decisions log; the block `CodeTrackBlock` and the pipeline kind `code_track` (§7) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the carrier loop (that stays in `costas`, M40, placed after `code_track` as for PSK); acquisition (`code_search`, M53); the composite `dsss_demod` and the carrier-aiding input of the block (later milestone; name reserved); a second (data-aided) discriminator (backlog); the replica-resampling correlator form of GNSS-SDR (a nearest-chip replica from a code NCO; it has a dead zone from the code-phase quantisation at `samples_per_chip` 1 and at commensurate rates, so it is not used; logged). Carrier-aiding of the code position (feeding the carrier Doppler into the code rate) is in as an optional kernel input, because without it a long integration slips.

**Backlog item `code_track` `CQ15`** (*rev. 2026-11-17*, a concrete item instead of the bare "backlog"): `CQ15` input, `C32` symbols and loop. The lag sums of §1 become exact integer sums of `±x` (the integer sign of the shared conventions §A.1 inside a wide accumulation, and the width bound of §C.1: an `i32` lane holds 65535 binary terms, longer periods widen to `i64`). Once per period they are converted to `f32` and enter the unchanged per-period part of §1–§2, which is cheap even with soft float because it runs once per period. This needs an exception to D529 ("the digital receivers have no `Q15` variants"), recorded with the item. GNSS-SDR's `16ic` correlators (`volk_gnsssdr_16ic_xn_resampler_16ic_xn`, `volk_gnsssdr_16ic_x2_dot_prod_16ic_xn`) show the use and are the fair reference then.

## Specification

General rules of M19 apply. The loop feeds back once per code period, so as for `symbol_sync` and `costas` the arithmetic is **fixed here** and the kernel is bit-exact to its own float32 model (the float32-model rule, maintainer decision D111); `Reference` follows it step by step and the golden cases come from the exact float32 model with an f64 regression guard. *(rev. 2026-11-17)* Inside a period every position is fixed by the state after the previous period, so the work per input sample (the lag sums of §1) is lane-parallel and gets SIMD paths on every level (§6), bit-identical to the fallback through the fixed summation order (NaN sign and payload excepted, D296). Only the discriminator, the loop filter and the position update of §2 form a serial recursion (once per period).

The code is a `Code<'_, T>` (shared conventions §B, introduced in M51): `Code::Bits` is the bit-packed binary form (chip `i` = bit `i & 31` of word `i >> 5`, chip value `1 − 2b`, chip `i` = the `i`-th `Prbs::next_bit` of a PRBS), applied by the exact sign operations of §A; `Code::Complex` is the general form with the `mul` product. `±1` and `±j` are applied by the exact sign and swap operations of the shared conventions §A (`q15_neg_sat`, `cq15_quadrant`, `c32_quadrant`, introduced in M49), never by a product. In `CQ15` a stored negation saturates (`−(−32768) = 32767`); inside a wide accumulation it is plain integer negation. (`code_track` itself is `C32` only; the `CQ15` part concerns the backlog item.)

### 1. Positions, lag sums and the per-period interpolation *(rev. 2026-11-17)*

The receiver is aligned when the prompt replica sits on the incoming chips. Three correlators per code period, at code offsets `−δ`, `0`, `+δ` chips (`δ` = `early_late_chips`, default 0.5, range 0.1…0.9; GNSS-SDR's `early_late_space_chips` default is 0.25, ours 0.5 as the DSSS default, logged).

- **Samples.** `spc` = `samples_per_chip` (an integer, 1…1024) and `N = sf · spc` input samples per period (at most 2^26). The kernel works on `b`, the `H` history samples followed by the input of the call (`H` below); after construction and `reset` the history is zero.
- **Position.** The start of the prompt replica of the current period is `pos`, a Q32.32 value in samples: the integer index `i` into `b` and the `u32` fraction `frac` (`mu = frac / 2^32`). The early and late arms are at `pos − dfx` and `pos + dfx` with `dfx = round(δ · spc · 2^32)`, computed once in `new` in f64 (`(early_late_chips * spc as f64 * 4294967296.0).round()` as `i64`); `d = dfx / 2^32` is the arm offset in samples. Each arm `A ∈ {E, P, L}` has `i_A = floor(pos_A)` (integer arithmetic on the Q32.32 value), its fraction `frac_A`, and the `f32` fraction `m_A = ((frac_A >> 8) as f32) · 2^−24` (exact). There is no `f64` and no division per sample.
- **Replica.** Term `n = 0 … N − 1` of a period uses chip `c_n` = code chip `(code_pos + floor(n / spc)) mod len`, kept as a chip cursor and a sub-chip counter (shared conventions §B.3: `code_pos` is a `u32` advanced by compare-and-subtract; no modulo and no division per sample). Code length and short/long indexing as in `despread` (M52); `code_pos` advances by `sf` chips per period.
- **Lag sums.** For each arm `A` and `k ∈ {−1, 0, 1, 2}`: `S_A[k] = Σ_n t(b[i_A + n + k], c_n)` over `n = 0 … N − 1`, where `t` is the term of `despread` (M52) for the code form. `Code::Bits` gives `c32_chip(x, b)` (§A.2: `x` or `(−x.re, −x.im)` by IEEE negation, so `±∞` stays `±∞`); `Code::Complex` gives the product `x · conj(c)` written as in the `mul` fallback (two multiplications and one subtraction or addition per part, no FMA). The sum is in the `dot_complex` order of M43 (D801, shared conventions §D.2): eight partial sums per component, each starting at `+0.0`, term `n` into partial sum `n mod 8`, the fixed tree `s[k] = acc[k] + acc[k+4]`, `t[k] = s[k] + s[k+2]`, result `t[0] + t[1]`; `n` counts from the start of the period, never from the start of a call. Two arms whose lags `i_A + k` coincide share the sum (it is the same value). The number of distinct lags is `n_lags = min(12, ceil(2d) + 4)`: 5 at `spc` 1 and `δ` 0.5, 6 at `spc` 2, 8 at `spc` 4, and 12 once `2d > 7`, that is `ceil(2d) ≥ 8` (checked by enumeration over positions).
- **Per-period interpolation.** Each arm is `A = interp(S_A[−1], S_A[0], S_A[1], S_A[2]; m_A)`: the interpolation formula of the M38 contract (`c1`, `c2`, `c3` and the Horner form with `THIRD`, `HALF`, `SIXTH` as `f32`, no FMA) with the four lag sums in place of the four samples. By linearity this equals `Σ_n c_n · interp(b, i_A + n, mu_A)` (the per-sample interpolation of the old form) in exact arithmetic; an f64 check of the two forms agrees to a relative 4e-16. In `f32` the rounding points differ, and this section defines the bits.
- **Accepted approximation.** All `N` samples of a period sit at `i_A + n + mu_A` with one fixed fraction. The loop corrects the position once per period, so the code-rate drift inside a period (at most the period correction of §2, by default within 0.05 chips over the period) is not followed inside it. GNSS tracking loops hold the code rate over an integration in the same way. Logged.
- **Cost per input sample** (`Code::Bits`): `n_lags` sign-adds per component, that is `2 · n_lags` real additions and no multiplication (10 at `spc` 1 and `δ` 0.5, at most 24). The old form needed about 50 to 70 flops for three per-sample cubic interpolations (the review's estimate). The per-period part costs a few hundred operations once per `N` samples (estimate, to be confirmed by the count).
- **Streaming.** The kernel accumulates the lag sums as samples arrive (running partial sums in the state: `8 · n_lags` complex partial sums) and keeps the last `H = ceil(2d) + 2 · spc + 3` input samples. The history is needed because the next period can start before the last sample of the current one: by up to the late-arm lookahead plus the largest period correction of §2 (2 chips). The bound `H` was checked by enumeration. A period is complete when sample `i_L + N + 1` has been accumulated. No period is ever buffered (the state size does not depend on `N`), so `min_input` is 1.
- After construction and `reset`: `pos` at the first input sample (`i = H`, `frac = 0`), `code_pos = 0`, the accumulators at `+0.0`, `rate = 0`, the lock level 0. Samples before the first input read as the zero history.

### 2. Discriminator and loop

- **Magnitudes:** `|A| = sqrt_f32(A.re · A.re + A.im · A.im)` (the rule of the `mag` kernel, `C32`, no FMA).
- **Code discriminator** *(corrected 2026-11-17)*: `e = g_δ · (|L| − |E|) / (|E| + |L|)` in f32, with `g_δ = 1 − δ` computed in f64 and rounded to f32. This is GNSS-SDR's `dll_nc_e_minus_l_normalized` for BPSK (slope 1, y-intercept 1, so its factor `(y_intercept − slope·δ)/slope` is `1 − δ`). The old text fixed the factor at `0.5`, which is its value at `δ` = 0.5 only. The sign is chosen so that `e > 0` means the signal arrives later than the prompt replica (the loop then advances further). For the ideal correlation triangle `R(τ) = 1 − |τ|` the discriminator equals the offset in chips, `e = ε`, for `|ε| ≤ min(δ, 1 − δ)`, so its slope is 1 per chip at every `δ`. At `δ` = 1 the factor is 0 and the normalised form has no linear region, which is why the range ends at 0.9. `e = 0` when `|E| + |L|` is 0 or not finite, or when `e` is not finite (silence, NaN, overflow).
- **Discriminator gain** `disc_gain` (default 1.0, finite and `> 0`; the slope per chip, as `ted_gain` of `symbol_sync`; it replaces the constant the implementer was to fix). With interpolated arms on unfiltered rectangular chips the slope differs from the ideal 1: a NumPy model of §1 gives about 0.96 at `spc` 1, 1.19 at `spc` 2 and 0.95 at `spc` 4 for `δ` 0.5 (computed, to be confirmed by unit test 2), and band-limited front ends move it again. The block default stays 1.0; the measured slopes go into `docs/kernels.md`.
- **Loop filter:** the second-order filter of `symbol_sync`/`costas` in the same form. Parameters: `loop_bw`, normalised to the period rate (one update per code period, which is the symbol rate; *corrected*, the old text said "of the code rate"), 0 or 0.0001…0.1, default 0.005, where 0 is the open loop (`theta = 0`, so `kp = ki = 0` exactly); and `damping` (default 1/√2). Constants in f64: `theta = loop_bw/(damping + 1/(4 damping))`, `den = (1 + 2 damping·theta) + theta²`, `kp = ((4 damping·theta)/den)/disc_gain`, `ki = ((4 theta²)/den)/disc_gain`, rounded to f32 (both finite, else `InvalidParameter`), in chips per period per chip of error. `max_rate` (chips per period, default 0.05, range 0…0.5) as f32, flushed when subnormal. *(corrected)* `rate` is the code-clock offset in chips per period, so `max_rate` is a relative clock error of `max_rate / sf`: the default 0.05 is about 394 ppm at `sf` 127 and 49 ppm at `sf` 1023 (the old "±5 % clock error" was wrong).
- **Update per code period** (f32, no FMA, in this order; *rev. 2026-11-17*):
  1. `rate = clamp(flush_denormal(rate + ki·e), −max_rate, max_rate)`;
  2. `step = clamp(kp·e, −0.5, 0.5)` chips (the clamp keeps positions monotone and the history bound of §1; it is never reached in normal operation);
  3. `aid = clamp(−((sf as f32) · (f_d · k_aid)), −1, 1)` chips, with `k_aid = 1/f_rf` computed in f64 and rounded to f32 (0 when `f_rf = 0` or `carrier_doppler_hz` is empty) and `f_d` the value of `carrier_doppler_hz` for this period (a non-finite value counts as 0);
  4. `corr = (rate + step) + aid` (so `|corr| ≤ 2` chips);
  5. the advance in Q32.32 samples `adv = (N << 32) + (((corr · (spc as f32)) · 4294967296.0) as i64)` (the scaling by `2^32` is exact in f32 and the cast truncates toward zero; `|corr · spc| ≤ 2048`, so it never saturates); `pos = pos + adv` in integer arithmetic on `(i, frac)`; `code_pos` advances by `sf` chips (mod `len`, compare-and-subtract).
- **Carrier aiding (optional)** *(sign corrected)*: the received code rate is the nominal rate times `1 + f_d/f_rf` (GNSS-SDR's form), so a period lasts `N / (1 + f_d/f_rf)` samples. The kernel uses the first-order form `N · (1 − f_d/f_rf)`, which is the `aid` of step 3. The old text scaled the advance by `1 + f/f_rf`, which has the wrong sign. The second-order term `(f_d/f_rf)² · N` is below 0.007 samples per period for `|f_d/f_rf| ≤ 1e-5` and `N ≤ 2^26`, and the rate integrator absorbs it. Aiding keeps the code and carrier Doppler consistent over long integrations (standard in GNSS tracking). Logged as the one addition beyond the GNSS-SDR BPSK DLL.
- **Output:** one prompt symbol `P` per code period (the despread symbol, handed to `costas` and the slicer downstream). `lock_level()` is a smoothed `|P|²/((|E|²+|L|²)/2)`-style metric (one-pole smoothing in f32 with the state flushed by `flush_denormal`, R6.3; the smoothing constant and the threshold of `locked()` are fixed by the implementer as for `costas` and logged). *(rev. 2026-11-17)* When the metric of a period is not finite (a NaN or `±∞` symbol, or a zero denominator as at silence), the smoothing update of that period is skipped and the lock state keeps its value, so the lock state is always finite and its bits are defined for every input; `code_phase()`, `rate()`, `mu()` read the loop.

### 3. State and trait

```rust
pub struct CodeTrackState<'a, S> { /* code: Code<'a, C32>, code_pos, sf, spc, dfx, gains, pos (i, frac), term index n, storage S (history and lag accumulators), rate, lock */ }
impl<'a, S: AsRef<[C32]> + AsMut<[C32]>> CodeTrackState<'a, S> {
    pub fn storage_len(samples_per_chip: u32, early_late_chips: f64) -> Result<usize>; // H + 8·n_lags
    pub fn new(code: Code<'a, C32>, sf: u32, samples_per_chip: u32, early_late_chips: f64,
               loop_bw: f64, damping: f64, max_rate: f64, disc_gain: f64, f_rf: f64,
               storage: S) -> Result<Self>;
    pub fn gains(&self) -> CodeTrackGains;    // kp, ki, max_rate, disc_gain, g_delta, k_aid
    pub fn lag_count(&self) -> u32;           // n_lags of §1
    pub fn code_phase(&self) -> f64;  pub fn rate(&self) -> f64;  pub fn mu(&self) -> f64;
    pub fn lock_level(&self) -> f32;  pub fn locked(&self) -> bool;
    pub fn set_initial(&mut self, code_phase_chips: f64, rate: f64) -> Result<()>;  // from code_search
    pub fn reset(&mut self);
}
pub trait CodeTrack {
    fn code_track<S: AsRef<[C32]> + AsMut<[C32]>>(&self, state: &mut CodeTrackState<'_, S>, input: &[C32],
                  carrier_doppler_hz: &[f32], output: &mut [C32]) -> Result<Progress>;
}
```

- Parameter ranges validated in `new` (as `symbol_sync`/`costas`): `sf` 2…65536, `samples_per_chip` 1…1024 (an integer, *rev. 2026-11-17*), `early_late_chips` 0.1…0.9 (*corrected*, §2), `loop_bw` 0 or 0.0001…0.1, `damping` 0.1…4, `max_rate` 0…0.5, `disc_gain` finite and `> 0`, `f_rf` finite and `> 0`, or 0 to disable aiding; the code as in `despread` (M52); non-finite or out-of-range → `InvalidParameter`. `storage.len()` must equal `storage_len(samples_per_chip, early_late_chips)` (`LengthMismatch`): `H` history samples and `8 · n_lags` partial-sum pairs, from 46 `C32` at `spc` 1 and `δ` 0.5 up to 3991 `C32` (about 31 KiB) at `spc` 1024 and `δ` 0.9. The struct itself holds no sample array.
- `set_initial(code_phase_chips, rate)`: the prompt of the next period starts `code_phase_chips · spc` samples after the first sample of the next call's input (`pos = (H << 32) + round(code_phase_chips · spc · 2^32)`), `rate` is stored as f32; the accumulators and the term index restart, `code_pos` is 0, the history stays. `code_phase_chips` must be finite and in `0 ≤ … < sf`, and `|rate| ≤ max_rate`; otherwise `InvalidParameter` and nothing changes. `code_phase()` is the start of the next period's prompt in chips, counted from the first sample of the next call's input (negative while it lies in the history); `mu()` is `frac / 2^32`; `rate()` the integrator in chips per period.
- `carrier_doppler_hz` *(corrected)* is either empty (aiding off for this call) or has exactly `output.len()` values; value `m` is used for the update after the `m`-th symbol produced in this call. Any other length is `LengthMismatch`, checked at the entry (the old rule "matching the symbols produced" could only be checked after the call).
- **Zero-bandwidth identity:** `loop_bw = 0` opens the loop (`kp = ki = 0`, `step = 0`); with `set_initial(phase, 0)` and aiding off the position advances by exactly `N` samples per period. With `samples_per_chip = 1` and an integer `phase`, `frac` stays 0, `m_P = 0` and `P = ((c3·0 + c2)·0 + c1)·0 + S_P[0]`, where `S_P[0]` is the `despread` sum of the same chips in the same order (§D.2, M52). The prompt output therefore equals `despread` (M52) of the same chips bit for bit, for any `δ` and both code forms, whenever the lag sums are finite, with one exception: a prompt sum of `−0` can come out as `+0` (the sign of the result follows `((c3·0 + c2)·0 + c1)·0 + S_P[0]` under IEEE rules, and a zero from the cubic part can turn `−0 + (+0)` into `+0`). A non-finite sample in a neighbouring lag gives NaN (`∞ · 0` in the cubic) where `despread` may be finite. No internal switch.
- Timing: `RatioKind::Variable`, nominal ratio `1/(sf·samples_per_chip)`, group delay `ceil(d) + 2` input samples (the lookahead of the late arm beyond the prompt window; to be confirmed by the timing test). A call consumes input until the input ends or the output is full (it stops right after the period that fills the output); the state carries partial periods, `min_input` 1; the result is chunk-invariant bit for bit (positions are integers and term indices count from the period start).

### 4. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `code_track` | `kernels::code_track::CodeTrack` | `C32` | bit-exact | the float32 model of §1–§2 (*rev. 2026-11-17*: lag sums of the `despread` terms in the `dot_complex` order of M43, chips of `Code::Bits` by the exact signs of the shared conventions §A, the M38 interpolation formula on the lag sums once per period, the normalised E−L discriminator with `1 − δ`, the second-order loop of `costas`, Q32.32 positions), each f32 operation without FMA; `Reference` follows it step by step; golden cases from the exact float32 model, an f64 model of the same structure as a regression guard (`snr_db_guard`); SIMD paths on every level bit-identical to the fallback, NaN sign and payload excepted (D296; NaN lanes compared by class) (`backend_for` reports the level). A binary chip applied to `±∞` gives `±∞`, never NaN; `Code::Complex` keeps the product behaviour (`∞ · 0 = NaN`) |

### 5. Golden vectors

New file `tests/golden/sdrk_golden/cases/code_track.py`, the float32 model of the loop (as `costas.py`/`symbol_sync.py`), with the lag sums of §1 in the §D.2 order and the integer position arithmetic of §2.

| Case | Seed | `sf` | sps | scenario |
| --- | --- | --- | --- | --- |
| `code_track/lock_sf127_40ppm` | 740 | 127 | 2 | a 127-chip code, 40 ppm clock drift (0.0051 chips per period), Eb/N0 10 dB; the loop follows the drift, prompt symbols recovered |
| `code_track/acq_handoff_sf1023` | 741 | 1023 | 1 | `set_initial` from an acquisition offset of 0.3 chip; the loop pulls in within N periods |
| `code_track/zero_bw_equals_despread` | 742 | 31 | 1 | `loop_bw` 0, integer alignment; the prompt output equals `despread` (stored from `despread/…` for cross-check) |
| `code_track/aided_sf255_spc4` | 748 | 255 | 4 | *(new 2026-11-17)* `δ` 0.25 (`d` = 1 sample, 6 lags), a Doppler ramp with `f_rf` set and the matching `carrier_doppler_hz` array; the aided loop follows with a small rate error |

- Stored: the code as `code_words` (`u32` little-endian) and `code_len` (shared conventions §B.2), the input, the loop parameters, the carrier-aiding array (empty or values); expected prompt symbols, the lock flag per period, the final loop state (rate bits, `i`, `frac`, `code_pos`, the term index, the lock level bits), the loop constants as IEEE bits (`kp`, `ki`, `g_δ`, `k_aid`, `max_rate`) and `dfx`. `snr_db_guard` from the f64 model.

### 6. Cost, SIMD plan and reference rows *(rev. 2026-11-17)*

`cargo xtask count` entries `code_track_sf127` (`spc` 2, per input sample; the lag sums plus the per-period part) and `code_track_sf127_fallback` (the same work on the fallback, the D321 floor). Estimates (to be confirmed): 12 additions per input sample for the lag sums at `spc` 2 and `δ` 0.5 (6 lags), plus the per-period part (three cubic combinations, the trees, two magnitudes, one division, the loop) of a few hundred operations spread over 254 samples. Record in `docs/performance.md`.

**SIMD plan** (shared conventions §D.4, row M54):
- *Lag sums (lane-parallel, every level).* The partial-sum index `n mod 8` is the lane: a vector holds the terms of 8 consecutive `n` (16 f32: one `avx512` register, two `avx2`, four `sse2`/`sse4.2`/`neon`/`simd128`). The sign mask of the 8 chips (one byte of a code word at `spc` 1, each chip repeated `spc` times otherwise) is expanded once and shared by all `n_lags` lags; each lag loads the input at its own offset (unaligned) and adds `x XOR mask` (§A.2). `Code::Complex` uses the product of the contract per lane without FMA (not the `mul` SIMD entry, which uses FMA at some levels, D326).
- *Per-period combination (lane-parallel across lags, arms and components).* The tree reductions of the `n_lags` lag sums and the three cubic combinations (the same formula per lane, each lane with its arm's `m_A`). The §D.4 row lists it with the lane-parallel part (the rule "SIMD wherever lane-parallel").
- *Scalar, with the reason:* the discriminator, the loop filter and the position update form a serial recursion once per period (the positions of the next period depend on this period's `e`), with no lane-parallel part.
- *NaN (D296).* Every level gives the fallback's bits except the sign and payload of a NaN: a NaN input sample passes through the `Code::Complex` products, the lag-sum trees and the cubic, where x86 `a − NaN` and `a + (−NaN)` give NaNs of different sign and `simd128` NaN bits are not deterministic (as `despread`, M52). NaN results are NaN on every level and are compared by class.
- *Seams:* a call can end at any `n`; the next call resumes at lane `n mod 8` (scalar terms up to the lane boundary, or a lane rotation, with the same bits); the code period seam that is not word-aligned (§B.3); the history wrap; the vector tail. The state after a call equals the fallback's.
- SIMD follows the shared conventions §D (maintainer rule 2026-11-17): every lane-parallel part of the kernel gets a path on every level in this milestone, bit-identical to the fallback through the fixed element-wise formula or reduction order of the contract; D321 is a quality gate (a slower path is improved, not removed; delegation to the fallback only after several measured failed attempts, with an open item); cycles per sample with `cargo measure` against the fallback and the fair reference (D32), unfair rows marked.

**Measurement (D32, D33, D68).** The entries above and the block entry of §7 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`, Cortex-M). Each SIMD entry is compared with its `_fallback` entry (D321) and with the fair reference row.

**Reference (D32)** *(rev. 2026-11-17; replaces "no library row", which held only without the interpolation cost)*: GNSS-SDR's multicorrelator, `volk_gnsssdr_32fc_xn_resampler_32fc_xn` plus `volk_gnsssdr_32fc_x2_dot_prod_32fc_xn` with three correlators (E, P, L) over one period, per input sample. It does the same job (three code correlations per period from the same input) with another algorithm (a nearest-chip replica resampled per sample and float multiplications, no sub-sample interpolation of the arms). The row is marked so, and a gap counts. It is measured on the host levels where `volk_gnsssdr` builds (x86-64 and AArch64 NEON); `simd128` and Cortex-M have no row. The `16ic` variants are information until the `CQ15` backlog item.

### 7. Block and kind `code_track`

`CodeTrackBlock` (`C32` → `C32`, one prompt symbol per code period) around the kernel `code_track` with a short PRBS code (`sdrk_proto::prbs` generated into words with `Prbs::fill_words` and wrapped in `CodeBits::new`, shared conventions §B; chip `b` → `1 − 2b`; `sf` is the code period). Carrier aiding is off (`f_rf = 0`): it needs a second input and comes with `dsss_demod`.

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `code_track` | `CodeTrackBlock` | C32 → C32 | `code`, `seed` (as `spread`), `samples_per_chip` (count, build, 1; 1 … 1024, an integer), `early_late_chips` (chips, build, 0.5; 0.1 … 0.9), `loop_bw` (linear, build, 0.005; 0 (open loop) or 0.0001 … 0.1, relative to the period (symbol) rate), `damping` (linear, build, 0.7071; 0.1 … 4), `max_rate` (chips per period, build, 0.05; 0 … 0.5), `disc_gain` (linear, build, 1; > 0), `initial_code_phase` (chips, build, 0; 0 ≤ … < `sf`) | `code_track`; `initial_code_phase` is `set_initial(phase, 0)` at construction and `reset` (the value a `code_search` packet gives, divided by `samples_per_chip`) |

- Getters `locked()`, `code_phase()`, `rate()` (as `costas`). `CodeTrackConfig`. Storage: the code words (`ceil(sf/32)` `u32`) and the kernel storage of `storage_len` (history and lag accumulators); exact. Timing `RatioKind::Variable`, nominal ratio `1/(sf · samples_per_chip)`, delay and `min_input` from the kernel state.
- Pipeline kind `code_track` (`c32` only), cost `2 · n_lags + 400 / (sf · samples_per_chip)` per input sample with `n_lags = min(12, ceil(2 · early_late_chips · samples_per_chip) + 4)` (*rev. 2026-11-17*: the lag sums and the per-period part; the old `12 + 3 · 24 / samples_per_chip` did not match even the per-sample interpolation; the constants are estimates to be confirmed: the implementer checks them against the count and logs them), a row in `docs/pipeline.md` and `docs/timing.md`. Chain position: `code_track` → `costas` → `slice`.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 740 … 749.

1. Golden cases of §5 (seeds 740–742, 748) through facade, `Fallback`, `Reference` and every SIMD level (`sdrk-testkit/tests/golden_m54.rs`), with the f64 guard.
2. Unit tests (`sdrk-core/tests/kernels_m54.rs`): construction and range rules (`samples_per_chip` an integer, `early_late_chips` up to 0.9, `disc_gain`); `storage_len` and `lag_count` (`n_lags = min(12, ceil(2d) + 4)`, `H = ceil(2d) + 2·spc + 3`) on a grid of `spc` and `δ`; the loop constants; the discriminator: exactly `e = ε` on synthetic arm magnitudes of the ideal triangle for several `δ`, and the slope on unfiltered chips at `spc` 1, 2, 4 and `δ` 0.5, measured and logged (the model gives about 0.96, 1.19, 0.95); the lag form against per-sample cubic interpolation in f64 (relative difference at most 1e-12); lock onto a clean code with a static offset; the integrator following a constant clock drift; the carrier-aiding sign and scaling; the clamps (`step` at ±0.5, `aid` at ±1, `|corr| ≤ 2`) and that no position ever leaves the history with `rate` at ±`max_rate` and `step` and `aid` at their clamps; `set_initial` and its ranges; `reset`; the length rule on `carrier_doppler_hz` (0 or `output.len()`).
3. Zero-bandwidth identity: the prompt output equals `despread` bit for bit in the configured case (seed 743), for `Code::Bits` and `Code::Complex` and several `δ`; a constructed case shows the documented `−0` exception.
4. Lock / BER sanity (`sdrk-testkit/tests/code_track_m54.rs`, seed 744): transmit `spread` → channel (`Awgn`, `ClockDrift`, `FrequencyOffset` of M38/M43) → `code_track` → the recovered symbols decode (after a `costas` and slicer) with a BER near the BPSK theory shifted by the processing gain, over enough symbols; a smoke test, not the full BER milestone.
5. Block-size invariance: 20 chunk plans at the chip level (seed 745), bit for bit, including calls that end at every `n mod 8`, an output buffer of 1 and the aiding array sliced per call.
6. Extremes: silence (no lock, no panic); full scale; NaN/∞ in one chip (`e = 0` in the periods whose windows contain it, then recovers; *rev. 2026-11-17*: the lock level stays finite and equals the value with those periods' updates skipped, §2); `samples_per_chip = 1` with `δ` = 0.1; `samples_per_chip = 1024` with `δ` = 0.9 (the largest storage); the largest `sf`; a code length that is not a multiple of 32.
7. Suite cases `code_track_c32` and `extremes_m54`.
8. **Block and kind** (`sdrk-blocks/tests/code_track_blocks.rs`, `sdrk-pipeline/tests/code_track_kinds.rs`; seeds 746, 747): the block equals the kernel bit for bit on whole buffers and in 20 chunk plans; `spread` (block) → channel → `code_track` (block) locks from `initial_code_phase` 0.3 chips off; the description rules of M09, exact storage, struct at most 512 bytes, validation errors at their parameter; a pipeline flow `code_track` → `costas` equals the blocks; `cargo xtask schema --check` and the WASM kind list include `code_track`.
9. **SIMD** *(new 2026-11-17)* (`sdrk-backend-simd/tests/code_track.rs`, seed 749; the pattern of M48 test 9): every level equals `Fallback` bit for bit (NaN sign and payload excepted, D296: NaN lanes compared by class) on input lengths 0 to 67 and 4096 and on the special values (NaN, ±∞, ±0, full scale); the seams: period starts, calls ending at every `n mod 8`, the code period seam with a length that is not a multiple of 32, word boundaries, the history wrap, the vector tail; both code forms; `spc` 1, 2, 4 and 1024; the state after each call; the golden cases through each level.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-backend-simd --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask check cycles`; `cargo xtask count --platform x86_64` (the entries of §6 and `code_track_block`); `cargo measure` (the same entries and the reference row of §6, every level with a core model).

## Deliverables checklist

- [ ] Kernel `code_track` (`C32`): `CodeTrackState` (Q32.32 position, lag accumulators and history in caller storage, `storage_len`, `lag_count`), `CodeTrackGains`, trait, Fallback, Reference, facade delegation, `backend_for` (the SIMD level)
- [ ] SIMD paths in `sdrk-backend-simd` on every level (lag sums, per-period combination), bit-identical to the fallback (NaN sign and payload excepted, D296)
- [ ] Contract row and section in `docs/kernels.md` (correlate-then-interpolate with the lag sums and the accepted intra-period approximation, the E−L discriminator with `1 − δ`, the loop, the reuse of the M38 formula, the `despread` identity and its `−0` exception, the GNSS-SDR citation for the discriminator, the measured discriminator slopes)
- [ ] Golden cases (seeds 740–742, 748) and the float32 Python model with the f64 guard
- [ ] Tests 1–9 (seeds 740–749); suite cases
- [ ] Count and cycle entries (`code_track_sf127`, `code_track_sf127_fallback`; `cargo xtask count`, `cargo measure`) and the reference row (`volk_gnsssdr` resampler + `dot_prod_xn`), `docs/performance.md`
- [ ] Block `CodeTrackBlock` and kind `code_track` (§7): `docs/blocks.md` (entry; reserved-name row "delivered"), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `code_track_block`
- [ ] Backlog entry `code_track` `CQ15` (Scope) in `docs/milestones/BACKLOG.md` (entered with the 2026-11-17 revision) still matches the Scope text
- [ ] CHANGELOG, decisions log (correlate-then-interpolate and the intra-period approximation, the replica-resampling form not used, the discriminator factor `1 − δ` and the `δ` range, `disc_gain` and the measured slopes, the `δ` default, Q32.32 positions and integer `samples_per_chip`, the step and aid clamps and the history bound, carrier aiding and its sign, `loop_bw` 0 as the open loop, the meaning of `max_rate`)

## Status

Planned. Nothing is implemented.
