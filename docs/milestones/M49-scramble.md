# M49 — Kernel `scramble`: keyed phase scrambling

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M38](M38-symbol-sync.md) (`sdrk_proto::prbs`; through it M30) · Next: [M50](M50-xcorr-fft.md)

> **Status: implemented** (2026-11-18, branch `m49-scramble`, decisions D2050 to D2093, see [Status](#status)); no open items (I49-1 and I49-2 closed by the maintainer's decision D2092 of 2026-11-18). Written 2026-11-08, revised 2026-11-17 after an algorithm review (maintainer decision). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

> **Revision 2026-11-17.** The shared conventions §A to §D cited in this file are defined in [CONVENTIONS-M49-M56.md](CONVENTIONS-M49-M56.md). The maintainer accepted every recommendation of the algorithm review of M49 to M56; the shared definitions are in the shared conventions of M49 to M56 (§A exact `±1`/`±j` operations, introduced here; §D SIMD rule). Changes in this milestone:
> 1. **Exact quadrant part** (shared conventions §A, decision 1): the index `k` is split into a quadrant `q` and a residue `r`; `±1` and `±j` are applied by exact sign and swap operations, never by a product with the table. `b = 1` and `b = 2` need no multiplication and are lossless (in `CQ15` except a component −32768, which returns as −32767), and `k = 0` is the identity. The first draft multiplied by the table value `COS_Q15[0] = 32767`, so `k = 0` was not the identity and a `CQ15` round trip lost up to 2 LSB at `b = 1, 2, 3` (measured by the review with NumPy), not the 1 LSB the draft promised. The residue `r ≠ 0` (`b ≥ 3`) keeps the `rotate` product. The kernel no longer equals "one `rotate` step at the phase `k << (32 − b)`" bit for bit; for `r ≠ 0` it equals one `rotate` step at the residual phase followed by the quadrant map, and for `r = 0` the quadrant map of the input, with no `rotate` step (§1, §2, §3, test 2).
> 2. **The `±1`/`±j` helpers are introduced here** (`q15_neg_sat`, `cq15_quadrant`, `c32_quadrant`, `cq15_chip`, `c32_chip` in `sdrk_core::num`; the vector helpers in `sdrk-backend-simd`, module `sign`), with their full text in `docs/kernels.md`, for M51, M52 and M54 to reuse (§1.1, test 2).
> 3. **Direct table lookup** instead of "the interpolation of `rotate`": for `b ≤ 8` the interpolation fraction is identically 0, so the lookup gives the same bits and saves about 6 operations per sample (§1, §2, §5, §6 cost).
> 4. **SIMD in scope** (shared conventions §D, maintainer rule 2026-11-17): paths on every level in this milestone, bit-identical to the fallback; D321 is a quality gate (§5 "SIMD plan", test 7). The first draft had "a SIMD path allowed later".
> 5. **Fair reference rows named** (D32): phasor lookup plus VOLK `volk_32fc_x2_multiply_32fc` (`C32`) and CMSIS-DSP `arm_cmplx_mult_cmplx_q15` (`CQ15`); GNU Radio `dvbs2_physical_cc` for `b = 2` as information (§5).
> 6. **Corrections:** the sentence "Bit-exact class: fallback equals reference" is removed (the kernel is a tolerance kernel; only `b ≤ 2` and `r = 0` are exact against the reference, §2, §3); the round-trip bounds are restated per case (§3); the DVB-S2 pattern is reproduced exactly in both types (§4); the pipeline cost follows the direct form (§6); the *Order* section names M38, which the block needs (§Order); the block may step the PRBS several bits at a time with the same bits (§6); two golden cases for `b = 8` (seeds 698, 699) and a SIMD test are added (§4, tests 1 and 7).
> 7. **Cross-milestone consistency (M49–M56 pass).** NaN of the residual product (`r ≠ 0`, `C32`) follows D296 (NaN by class) in the contract row and in test 7; samples with `r = 0` stay `to_bits()`-exact including NaN. *Order* now names M50 and M51 as dependants.

## Goal

A kernel `scramble` that multiplies a complex stream, element by element, by unit phasors `exp(j 2π p[n] / 2^b)` taken from a caller-supplied index sequence `p` (one index per sample, `b` phase bits), and a descrambler that multiplies by the conjugate. The multiples of `π/2` (`±1`, `±j`) are applied exactly by sign and swap operations; the remaining angle below `π/2` (only for `b ≥ 3`) comes from the cosine/sine tables of `rotate` by a direct lookup, so the kernel gives the same bits on every platform and the same residual phasors as `nco`, `rotate` and `oscillate`. For `b = 1` the phasor is `±1` (sign scrambling); for `b = 2` it is `{1, j, −1, −j}` (the DVB-S2 physical-layer scrambling pattern), both without a multiplication. The index sequence is data to the kernel; a later milestone generates it from a keyed keystream (`sdrk-proto`).

## Why

A "noise-like" transmitter removes the structure a detector keys on: the residual carrier that a power detector (BPSK x², QPSK x⁴) finds, the visible constellation, the repeating pilot pattern. Multiplying every sample (DSSS) or every subcarrier (OFDM) by a pseudo-random phase from a key does that, and the receiver removes it with the conjugate of the same sequence. The operation is element-wise, but it needs its own kernel so that the phasor is exact where it can be (`±1`, `±j` by sign and swap, which `CQ15` cannot represent as a product because `+1` is 32767/32768) and comes from the `rotate` tables elsewhere (the same bits on every platform) rather than from a per-sample `exp`, and so that both DSSS streams and OFDM lane steps use one definition.

## Order

M49 depends on M30 (the `rotate` tables it reuses) and on M38 (`sdrk_proto::prbs`, which the block of §6 uses). It needs none of the other kernels of the series and can run in parallel with every milestone after M38 that does not depend on it; M50 (the PRBS polynomials of §6) and M51 (the polynomials and §A) depend on it. It introduces the exact `±1`/`±j` operations of the shared conventions §A, which M51, M52 and M54 reuse. The decision numbers of this milestone are D2050 to D2099.

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

**In:** kernel `scramble` (`C32`, `CQ15`) in `sdrk_core::kernels::scramble` with the trait `Scramble<T>`, Fallback, Reference, facade delegation, contract row and section in `docs/kernels.md`; the exact `±1`/`±j` operations of the shared conventions §A (§1.1): the scalar helpers in `sdrk_core::num` and the vector helpers in `sdrk-backend-simd` (module `sign`), with the full text of §A as a section of `docs/kernels.md`; SIMD paths of `scramble` in `sdrk-backend-simd` on every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`), bit-identical to the fallback, each passing D321 (§5); golden cases; unit, invariance, extreme and SIMD tests; suite cases; `cargo xtask count` and `cargo measure` entries with the reference rows of §5; CHANGELOG, decisions log, `docs/performance.md`, `docs/backends.md`; the block `ScrambleBlock<T>` and the pipeline kind `scramble` (§6) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** a keyed keystream (ChaCha20 with random access) and key synchronisation between transmitter and receiver (`sdrk-proto`, backlog; the block of §6 uses the public PRBS of `sdrk_proto::prbs`); a bit-packed phase input (one `u8` index per sample keeps the kernel simple and SIMD-friendly; the packed code format of the shared conventions §B comes with M51 and is not used here); `Prbs::fill_words` (M51); real (`f32`/`Q15`) scrambling (there is nothing to rotate on a real stream; `b = 1` sign scrambling of a real stream is a sign flip per sample and needs no kernel).

## Specification

General rules of M19 apply. The kernel is a stateless element-wise operation, like `mul`.

### 1. Trait, signature and index decomposition

```rust
/// `b` phase bits: the index `p[n]` selects the phasor `exp(j 2π p[n] / 2^b)`.
pub trait Scramble<T> {
    fn scramble(&self, input: &[T], phase: &[u8], bits: u32, inverse: bool, output: &mut [T]) -> Result<()>;
}
```

- `bits` (`b`) is 1 to 8, else `InvalidParameter` (nothing written). `input`, `phase` and `output` must have the same length, else `LengthMismatch` (nothing written).
- Each index is taken modulo `2^b`: `k = p[n] & ((1 << b) − 1)`, so an out-of-range byte never indexes out of the table (no error for a large byte; documented). `inverse = true` replaces `k` by `(−k) & ((1 << b) − 1)`, the index of the conjugate phasor, and then applies the same rule as the forward direction.
- **Decomposition** (shared conventions §A.4): `k` selects the angle `2π k / 2^b = q · π/2 + 2π r / 2^b` with
  - `q = k >> (b − 2)` and `r = k & ((1 << (b − 2)) − 1)` for `b ≥ 2`;
  - `q = 2k` and `r = 0` for `b = 1`.

  `q` is the quadrant (0 to 3) and `r` the residue (0 to `2^(b−2) − 1`); the residual angle `2π r / 2^b` lies in `[0, π/2)`. For `b ≤ 2`, `r` is always 0. *(Revised by M49.1 for `b ≥ 3`: the quadrant split applies to `b ≤ 2` only; for `b ≥ 3` the output is the `rotate` product with the table phasor of the whole index, `COS[k << (12 − b)]` and `COS[((k << (12 − b)) + 3072) & 4095]`, with no quadrant map and no `r = 0` selection; see [M49.1](M49.1-scramble-simplify.md) §1.)*
- **Output:** `output[n] = quadrant(y, q)`, where
  - `y = input[n]` exactly for `r = 0` (no product with the table value 32767 of angle 0 or with the tiny `f32` sine of angle 0);
  - for `r ≠ 0`, `y` is the `rotate` product of `input[n]` with the table phasor at the residual phase `r << (32 − b)`: table index `i = r << (12 − b)` (1 to 1023), `c = COS[i]`, `s = COS[i + 3072]` (the cosine at the phase `− 2^30`, as `rotate` reads its sine; `i + 3072 ≤ 4095`). The product is formed exactly as the `rotate` fallback forms it (§2);
  - `quadrant` is `c32_quadrant` or `cq15_quadrant` of §1.1 (multiplication by `j^q` by sign and swap, never a product).

  *(Revised by M49.1: this output rule holds for `b ≤ 2` only; for `b ≥ 3`, `output[n]` is the `rotate` product of `input[n]` with `c = COS[i]`, `s = COS[(i + 3072) & 4095]`, `i = k << (12 − b)`, also for `k = 0`.)*
- **Direct lookup, same bits as the interpolation.** For `b ≤ 8` the residual phase `r << (32 − b)` has zero low 20 bits, so the interpolation fraction of `rotate` is 0. In `CQ15` the interpolation gives `a + round_shift(0, 15) = a`; in `C32` it gives `a + (b − a) · 0 = a`, because no entry of `COS_F32` is `0.0` or `−0.0` (checked in a unit test). The kernel therefore reads the table entries directly and does not interpolate (about 6 operations per sample fewer than the interpolation).
- **Relation to `rotate`.** For `r ≠ 0` and `q = 0` the output equals one `rotate` step at the constant phase `r << (32 − b)` bit for bit. For `q ≠ 0` it equals that step followed by the exact quadrant map. It does **not** in general equal one `rotate` step at the full phase `k << (32 − b)`: that step reads other table entries and, in `CQ15`, multiplies by 32767 where this kernel swaps and negates (a NumPy check during the revision found about 79 % of `CQ15` outputs equal at `b = 3`). The residual phasors are those of `nco`, `rotate` and `oscillate`; the quadrant part is exact. *(Revised by M49.1: for `b ≥ 3` the output now equals one `rotate` step at the full phase `k << (32 − b)` bit for bit for every `k`.)*
- **Infinity and signed zero (`C32`):** a quadrant phasor (`r = 0`) applied to `±∞` gives `±∞` (sign per the table of §1.1), never NaN, and signed zeros follow IEEE negation. For `r ≠ 0` the product follows IEEE arithmetic as in `rotate` (`c, s > 0`, so one infinite component gives infinite components; two infinite components may give NaN through `∞ − ∞`). *(Revised by M49.1: for `b ≥ 3` every sample takes the product, so the exact `±∞` rule holds for `b ≤ 2` only; table entries may be negative, none is `±0.0`.)*

### 1.1 Exact `±1` and `±j` operations (shared conventions §A, introduced here)

`±1` and `±j` are applied by the exact sign and swap operations of the shared conventions §A (`q15_neg_sat`, `cq15_quadrant`, `c32_quadrant`, introduced in M49), never by a product. In `CQ15` a stored negation saturates (`−(−32768) = 32767`); inside a wide accumulation it is plain integer negation.

M49 delivers, in `sdrk_core::num` (`pub`, documented "for backend crates and tests"):

```rust
pub const fn q15_neg_sat(x: Q15) -> Q15;        // i16::saturating_neg: −(−32768) = 32767
pub fn cq15_quadrant(x: CQ15, q: u32) -> CQ15;  // multiply by j^(q & 3), §A.1
pub fn c32_quadrant(x: C32, q: u32) -> C32;     // multiply by j^(q & 3), §A.2
pub fn cq15_chip(x: CQ15, b: u32) -> CQ15;      // chip value 1 − 2(b & 1)
pub fn c32_chip(x: C32, b: u32) -> C32;
```

| `q & 3` | factor | result `(re, im)` |
| --- | --- | --- |
| 0 | `+1` | `(x.re, x.im)` |
| 1 | `+j` | `(−x.im, x.re)` |
| 2 | `−1` | `(−x.re, −x.im)` |
| 3 | `−j` | `(x.im, −x.re)` |

- Every `−` is `q15_neg_sat` in `CQ15` and IEEE negation (`-x`, only the sign bit flips; exact for `±0`, `±∞` and NaN) in `C32`. Not allowed: `x * −1.0`, a complex product with `(−1, 0)` or `(0, ±1)`.
- The inverse of `q` is `(4 − q) & 3`. The round trip (`q`, then `(4 − q) & 3`) returns `x` exactly, except that in `CQ15` a component equal to −32768 that is negated on the way returns as −32767 (1 LSB, only at that value). The negated components are `re` for `q ∈ {2, 3}` and `im` for `q ∈ {1, 2}`; each such component is negated twice (once in each direction), and the first saturating negation already turns −32768 into 32767. For `q = 0` nothing changes; `q = 2` (also the case `b = 1`) affects both components, `q = 1` only `im`, `q = 3` only `re` (checked by brute force of `cq15_quadrant` during the revision). In `C32` it is exact for every value (`to_bits()`, NaN included).
- `cq15_chip`/`c32_chip` give `x` for `b = 0` and the quadrant map with `q = 2` for `b = 1`. M49 does not use them; they are delivered here so that M51, M52 and M54 find one definition.
- The vector helpers go into `sdrk-backend-simd`, module `sign`, written once against `fearless_simd`, with level paths only where the portable operations are not enough (D1):
  - unconditional negation: x86 `psubsw(0, x)`, NEON `vqnegq_s16`, SIMD128 `i16x8.sub_sat_s(0, x)`; conditional negation by a lane mask `m ∈ {0, −1}`: `subs(xor(x, m), m)` (gives 32767 for −32768); **not** `psignw`/`vpsignw` (it wraps −32768 to −32768);
  - `C32`: XOR with `0x8000_0000` (masked per lane for a conditional sign);
  - swap of `re` and `im`: a 16-bit shuffle within each 32-bit pair (`CQ15`), a 32-bit shuffle within each 64-bit pair (`C32`);
  - quadrant by lane masks: swap where `q & 1`, then negate `re` where `(q ^ (q >> 1)) & 1` and `im` where `(q >> 1) & 1`; this equals the table above (checked against it for all four `q` and random inputs during the revision).
- M49 writes the full text of the shared conventions §A (with the canonical sentence and the infinity rule of §A.2) into `docs/kernels.md`, section "Exact `±1` and `±j` operations"; M51, M52 and M54 refer to it.

### 2. Fallback, Reference and SIMD

- **Fallback `C32`:** `k`, `q`, `r` as in §1. For `r = 0`, `y = x`. For `r ≠ 0`, `c = COS_F32[i]`, `s = COS_F32[i + 3072]`, `y = (re·c − im·s, re·s + im·c)` in `f32` without FMA (two products, then the difference or sum), the product of the `rotate` `C32` fallback. Then `c32_quadrant(y, q)`. *(Revised by M49.1 for `b ≥ 3`: the product with the table phasor of `k`, no quadrant map.)*
- **Fallback `CQ15`:** for `r ≠ 0`, `c = COS_Q15[i]`, `s = COS_Q15[i + 3072]` (both in `(0, 32767]` for `r ≠ 0`), then `cq15_mul` per component, computed in `i32` as in the `CQ15` `rotate` fallback (`sat16((xr·c − xi·s + 2^14) >> 15)`, `sat16((xr·s + xi·c + 2^14) >> 15)`; `|xr·c − xi·s| ≤ 2·32768·32767 < 2^31 − 2^14`, so it is exact). Then `cq15_quadrant(y, q)`. *(Revised by M49.1 for `b ≥ 3`: `cq15_mul` with the table phasor of `k`, whose entries may be −32768; exact in `i32` because `|c| + |s| ≤ 46341`.)*
- **Public per-sample helpers:** `pub fn scramble_one_c32(x: C32, k: u32, bits: u32) -> Result<C32>` and `scramble_one_cq15` in `sdrk_core::kernels::scramble` (documented "for backend crates and tests"; `k` is the already inverted index, masked inside; `bits` outside 1 to 8 is `InvalidParameter`). The fallback loop and the vector tails of the SIMD paths call them (shared conventions §D.2). *(Implemented: the vector tails call them; the fallback computes the same formula inline with a per-call residue table, same bits, D2056.)*
- **Reference:** the decomposition of §1 in exact arithmetic: `z = x` as `f64` (`CQ15`: `raw / 32768`); for `r ≠ 0`, `z = z · (cos θ_r, sin θ_r)` with `θ_r = 2π r / 2^b` from `libm::cos/sin` in `f64`; then the quadrant map in `f64` (negation and swap, exact); rounded once (`C32`: `as f32`; `CQ15`: `q15_from_f64`, so `−(−1.0) = 1.0` becomes 32767 like `q15_neg_sat`). The class is **tolerance** (§3). For `r = 0` the reference and the fallback agree bit for bit (`C32`: every non-NaN input; a NaN stays NaN in the same component). *(Revised by M49.1: for `b ≥ 3` the reference is `x · (cos θ, sin θ)`, `θ = 2π k / 2^b`, rounded once.)*
- **SIMD:** paths in `sdrk-backend-simd` on every level (§5), bit-identical to the fallback, including the state-free tail. `backend_for("scramble", ..)` reports the level that runs it; a level reports `fallback` only under the D321 rule of §5.

The kernel reuses the `rotate` tables `COS_F32` and `COS_Q15`; it does not copy them and needs no new table. It does not call the `rotate` helpers (`interp_f32`, `interp_q15`), because the direct lookup replaces the interpolation. Those helpers are private to `sdrk-core` (`pub(super)` in the module `fallback`), so the equality of both forms is checked through the public `Rotate` kernel: a zero-increment `Nco` (`Nco::from_increment(0)`, `set_phase(r << (32 − b))`) rotating the `C32` input `(1, 0)` gives exactly `(c, s)`, and the `CQ15` input `(−32768, 0)` gives exactly `(−c, −s)`, for every residue (test 2).

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `scramble` | `kernels::scramble::Scramble` | `C32` | tolerance | SNR ≥ 120 dB vs. the f64 reference (the `rotate` `C32` contract); `k` split into quadrant `q` and residue `r` (§1); `r = 0` (every index for `b ≤ 2`) exact by `c32_quadrant`, equal to the reference bit for bit, `±∞` stays `±∞`; `r ≠ 0`: `COS_F32[r << (12 − b)]`, `COS_F32[(r << (12 − b)) + 3072]` by direct lookup, `(re c − im s, re s + im c)` in f32 without FMA, then `c32_quadrant`; `inverse` uses `(−k) mod 2^b`; fallback and every SIMD path equal bit for bit (for `r ≠ 0` NaN sign and payload excepted, D296) |
| `scramble` | `kernels::scramble::Scramble` | `CQ15` | tolerance | SNR ≥ 80 dB vs. the f64 reference (the `rotate` `CQ15` contract); `r = 0` exact by `cq15_quadrant` (saturating negation), equal to the reference bit for bit; `r ≠ 0`: `COS_Q15` by direct lookup, `cq15_mul` in `i32`, then `cq15_quadrant`; `inverse` uses `(−k) mod 2^b`; fallback and every SIMD path equal bit for bit |

The thresholds are those of `rotate`, because the residual product is a `rotate` step; the quadrant part adds no error.

*(Revised by M49.1 for `b ≥ 3`: the rows above describe M49 as first merged; the current contract (`b ≤ 2` exact quadrant map, `b ≥ 3` the `rotate` product with the table phasor at `k << (12 − b)`) is in [M49.1](M49.1-scramble-simplify.md) §2 and `docs/kernels.md`.)*

**Round trip** (`scramble(input, phase, bits, false)` followed by `scramble(·, phase, bits, true)`):
- `b ≤ 2`, and every sample with `r = 0` at any `b`: exact in `C32` for every value (`to_bits()`, `±0`, `±∞` and NaN included); exact in `CQ15` except that a component −32768 that is negated on the way returns as −32767 (`re` for `q ∈ {2, 3}`, `im` for `q ∈ {1, 2}`, where `q` is the forward quadrant; each such component is negated twice, §1.1). *(Revised by M49.1: exact for `b ≤ 2` only; at `b ≥ 3` every sample, `r = 0` included, goes through the product.)*
- `b ≥ 3` and `r ≠ 0`: `C32` within the `rotate` round-trip error (SNR ≥ 120 dB of the descrambled stream against the input; a NumPy check during the revision gave about 144 to 147 dB for `b` from 3 to 8, about 147 dB at `b = 3`); `CQ15` within 2 LSB per component for inputs with `|x| ≤ 1` (the same NumPy check, 4·10⁵ random inputs per `b`, found at most 2 LSB; the unit test fixes the bound, to be confirmed). Larger inputs saturate in the product as in `rotate`. *(Revised by M49.1: for `b ≥ 3` this holds for every `k`; the `CQ15` bound measured over every `k` is 2 LSB.)*

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/scramble.py`. The expected values come from the f64 reference of §2 (exact quadrant map, `f64` residual phasor from `numpy.cos/sin`); the cosine-table model of `rotate.py` is not needed. `CQ15` cases with `b ≤ 2` use the integer model (`q15.py` gains `neg_sat`, the saturating negation).

| Case | Seed | Type | `bits` | `inverse` | Phase sequence | Length | Class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `scramble/bpsk_c32_2048` | 690 | C32 | 1 | false | PN (0/1 from `numpy.random.default_rng(seed)`, R8.2) | 2048 | `bit_exact` |
| `scramble/qpsk_c32_2048` | 691 | C32 | 2 | false | 0…3 | 2048 | `bit_exact` |
| `scramble/oct_c32_2048` | 692 | C32 | 3 | false | 0…7 | 2048 | `tolerance` |
| `scramble/qpsk_inv_c32_2048` | 693 | C32 | 2 | true | 0…3 | 2048 | `bit_exact` |
| `scramble/qpsk_cq15_2048` | 694 | CQ15 | 2 | false | 0…3 | 2048 | `bit_exact` |
| `scramble/byte_cq15_2048` | 698 | CQ15 | 8 | false | 0…255 | 2048 | `tolerance` |
| `scramble/byte_inv_c32_2048` | 699 | C32 | 8 | true | 0…255 | 2048 | `tolerance` |

- *(Revised by M49.1: the three `tolerance` cases are regenerated from the reference with the phasor of the whole index, `θ = 2π k / 2^b`.)*
- The `tolerance` cases carry the SNR thresholds of §3 as `snr_db_min` only (no `max_abs_err`; the `rotate` cases have none either). The phase sequence is stored as a `u8` array in the case. The `CQ15` input of `qpsk_cq15_2048` includes components −32768 and 32767 (the saturating negation).
- **DVB-S2 cross-check** (noted in the comment): the standard multiplies payload symbols by `j^{R_n}` with `R_n ∈ {0,1,2,3}`, which is exactly `exp(j 2π R_n / 4)`; `scramble` with `bits = 2` and `p = R_n` reproduces it exactly in both types (the quadrant map, as GNU Radio `dvbs2_physical_cc` does it by swap and negation). The `R_n` generation from the two m-sequences is a `sdrk-proto` concern of a later milestone, not this kernel.

### 5. Cost, SIMD plan and reference rows

SIMD follows the shared conventions §D (maintainer rule 2026-11-17): every lane-parallel part of the kernel gets a path on every level in this milestone, bit-identical to the fallback through the fixed element-wise formula or reduction order of the contract; D321 is a quality gate (a slower path is improved, not removed; delegation to the fallback only after several measured failed attempts, with an open item); cycles per sample with `cargo measure` against the fallback and the fair reference (D32), unfair rows marked. In this kernel every part is lane-parallel (no state, no reduction); there is no scalar part.

**Entries.** `cargo xtask count` and `cargo measure` entries per sample, each with its `_fallback` counterpart (the D321 floor): `scramble_c32_b2` and `scramble_cq15_b2` (`bits = 2`, the quadrant map only), `scramble_c32_b8` and `scramble_cq15_b8` (`bits = 8`, the residual product and the quadrant map), plus `scramble_block` (§6). *(Revised by M49.1: at `bits = 8` the entries run the one table phasor and one product; new entries `scramble_c32_b1` and `scramble_c32_b1_fallback`.)* Record them in `docs/performance.md` next to `rotate`.

**Estimated cost (to be confirmed by `cargo measure` and `cargo xtask count`).**

| Target | `bits ≤ 2` per sample | `bits ≥ 3` per sample | Notes |
| --- | --- | --- | --- |
| x86-64 scalar fallback | 2–4 cycles | 5–8 cycles | branch-free quadrant select; direct lookup saves the interpolation of the first draft |
| AVX2 | 0.3–0.6 cycles | 0.8–1.5 cycles | memory-bound for `bits ≤ 2` |
| Cortex-M4F (estimate from the instruction sequence) | 6–9 cycles | `C32` 12–16, `CQ15` 10–14 cycles | `CQ15`: `SMUSD`/`SMUADX`, rounding add, `SSAT`; no multiplication for `bits ≤ 2` |

**SIMD plan (`sdrk-backend-simd`, module `scramble`, using the module `sign` of §1.1).** All lanes run the element formula of §2, so bit identity with the fallback follows from the arithmetic.
- *Indices:* load 16 or 32 phase bytes, mask with `2^b − 1`, for `inverse` negate (`psubb(0, k)`, `vnegq_s8`, `i8x16.neg`) and mask again, then `q` and `r` by a shift and a mask; widen to one lane per complex sample (`CQ15`: 32-bit lanes holding the `(re, im)` pair; `C32`: two 32-bit lanes).
- *Quadrant (every `b`):* lane masks from `q` (swap where `q & 1`, negate `re` where `(q ^ (q >> 1)) & 1`, `im` where `(q >> 1) & 1`); the swap by a shuffle (`pshuflw`/`pshufhw` or `pshufb`, `vpshufb`, `vrev32q_s16`, `i8x16.shuffle` for `CQ15`; `shufps`/`vpermilps`, `vrev64q_f32`, `i32x4.shuffle` for `C32`) and a blend (`blendvps`/`pblendvb` from `sse4.2`, and/andnot/or at `sse2`, `vbslq`, `v128.bitselect`); the negation by `subs(xor(x, m), m)` (`CQ15`) or XOR with the masked sign bit (`C32`). For `b ≤ 2` this is the whole kernel: no multiplication, no table.
- *Residual phasor (`b ≥ 3`):* `2^(b−2)` residues (2 to 64). The per-sample `(c, s)` come from an in-register permute where the table fits the registers of the level (`vpermps`/`vpermd` up to 8 entries; `vpermi2ps`/`vpermi2d` up to 32, 64 with two permutes and a select; NEON `tbl`/`tbx` up to 64 bytes per instruction; `i8x16.swizzle`), from a gather (`vgatherdps`/`vpgatherdd`) or from scalar lane loads above that; the form per level and `b` is chosen by `cargo measure`, and every form gives the same bits (exact loads of the same entries). The small table of the call (at most 64 entries, 512 bytes for `C32`) may be built once per call from `COS_*`. *(Revised by M49.1: `2^b` phasors (8 to 256) of the whole index, no residues; the forms per level are in `docs/backends.md` and `docs/performance.md`.)*
- *Product:* `C32` two `mulps` and a `subps`/`addps` per part in the order of §2 (the operands paired by a shuffle), **no FMA** at any level; the SIMD `rotate` `C32` path (a block oscillator) is not reused. `CQ15` the form of the `rotate` `CQ15` SIMD path (D45): `pmaddwd` of the `(re, im)` pairs with `(c, −s)` and `(s, c)` (`−s` fits `i16` because `s > 0`; *revised by M49.1: with the phasors of the whole index `s` reaches −32768, so the operand is `(c, !s)` with `!s = −s − 1` and `xi` added back, the form of the `rotate` `CQ15` path*), add `2^14`, arithmetic shift by 15, saturating pack; NEON `smull`/`smlsl`/`smlal` and `sqrshrn #15`; SIMD128 `i32x4.dot_i16x8_s` and the saturating narrow. Not allowed: `pmulhrsw`/`vqrdmulh` per product (it rounds each product separately, not the sum).
- *`r = 0` lanes:* the product is computed for all lanes and the input is **selected** (a blend, not arithmetic) where `r = 0`, then the quadrant map runs on every lane. *(Revised by M49.1: no selection and no quadrant map for `b ≥ 3`.)*
- *Tail:* a remainder shorter than one vector calls `scramble_one_c32`/`scramble_one_cq15` (§2).
- *Planned backends (notes only):* `rvv` maps directly (`vrgather` for the residual table, masked `vneg`/`vssub`, `vsmul` is not used for the same reason as `pmulhrsw`); `arm-dsp` (Cortex-M4) is the scalar loop with `SMUSD`/`SMUADX`, which the fallback already compiles to.
- D321 applies to every level as a quality gate: a path that is slower than its fallback on either model is improved until it passes, not removed; only after several measured failed attempts (each in `docs/decisions-log.md` with its numbers) may a level dispatch to the fallback, with an open item in `docs/open-items.md` and a `cycle_waiver` naming it.

**Reference rows (D32).**

| Our entry | Library row | Marking | Why |
| --- | --- | --- | --- |
| `scramble_c32_b2`, `scramble_c32_b8` (and `_fallback`) | lookup of the phasor from the same table into a buffer + VOLK `volk_32fc_x2_multiply_32fc` (`benches/refs/volk-nofma.c` and `volk-generic.c`, the builds of the other VOLK rows) | fair (same task: one complex product per sample with a table phasor); for `b2` our form does less work by design | the obvious library composition; for `b ≤ 2` the gap shows the gain of the quadrant form |
| `scramble_cq15_b2`, `scramble_cq15_b8` (and `_fallback`) | the same lookup into a buffer + CMSIS-DSP `arm_cmplx_mult_cmplx_q15` (`benches/refs/cmsis-dsp.c`) | fair (same work); output scaling differs (CMSIS writes 3.13 format), marked | the CMSIS complex multiply is the reference on Cortex-M |
| `scramble_cq15_b2`, `scramble_c32_b2` | GNU Radio `dvbs2_physical_cc` PL scrambling loop (new `benches/refs/gr-dvbs2-plscramble.c`, transcribed from the pinned GNU Radio version, only the per-symbol `switch` on `R_n`) | **reference does different work** (information) | it generates its own `R_n` sequence and works on `gr_complex` symbols in a frame layout; it shows a switch-based quadrant loop (implemented for `C32` only, `scramble_c32_b2_dvbs2`: GNU Radio has no 16-bit form, D2083) |

*(Revised by M49.1 §5: D2092 is revised for the `C32` rows at `bits = 8`: `scramble_c32_b8` and `scramble_c32_b8_fallback` are fair and gated again (`release`; `sse2` and `simd128` of the SIMD entry stay information for the reasons of the `b2` rows); the `CQ15` `b8` rows stay "reference does less work"; a fair row `scramble_c32_b1` (and `_fallback`) is added: a `±1.0` sequence from the same index bits plus VOLK `volk_32fc_32f_multiply_32fc`, the form of srsRAN's `srsran_scrambling_c`.)*

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`, Cortex-M). The D321 gate applies to every SIMD path.

### 6. Block and kind `scramble`

`ScrambleBlock<T>` (`C32`/`CQ15` → same) around the kernel `scramble`. The phase indices come from `sdrk_proto::prbs`: sample `n` takes the next `bits` PRBS bits, the first one as the most significant bit of `p[n]`. The PRBS position runs on across calls; `reset` restarts it from `key_seed`. A scrambler and a descrambler (`inverse`) with the same `key` and `key_seed` agree when both start at the same sample; key synchronisation is backlog.

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `scramble` | `ScrambleBlock<T>` | C32/CQ15 → same | `bits` (I64, build, 1; 1 … 8), `inverse` (Bool, build, false), `key` (Choice, build, `prbs15`; `prbs5`, `prbs7`, `prbs9`, `prbs10`, `prbs11`, `prbs15`), `key_seed` (I64, build, 1; 1 … 32767, the low `n` bits are used, zero after masking is `invalid_value`) | `scramble`; a call takes at most `SCRAMBLE_CHUNK` (256) samples (the phase bytes of the call live in the struct); no real variant (§Out) |

- **`sdrk_proto::prbs` gains four polynomials** (`Poly` is `#[non_exhaustive]`; two-tap Fibonacci form of M38 §4): `Prbs5` `x^5 + x^3 + 1` (31), `Prbs7` `x^7 + x^6 + 1` (127), `Prbs10` `x^10 + x^7 + 1` (1023), `Prbs11` `x^11 + x^9 + 1` (2047). `sdrk-proto/tests/prbs_m49.rs` checks that every one has the full period `2^n − 1` (every non-zero state once) and the first 16 bits from seed 1 against the values of the Python model `tests/golden/sdrk_golden/prbs.py` (created by M49, because M38 delivered no Python PRBS model: the Fibonacci LFSR of `prbs.rs` with all six polynomials; the values are stored in the golden case `proto/prbs_first_bits`, the first 64 bits of every polynomial from seed 1 and from one random seed, never as hand-edited vectors). The spreading and search blocks of M50 to M54 take their codes from the same choices.
- **PRBS stepping.** The block draws `bits` PRBS bits per sample (up to 8 LFSR steps at `bits = 8`, which can cost more than the kernel). It may step the register several bits at a time from the two-tap recurrence (a word-parallel step) if the bit sequence is the same as `Prbs::next_bit`; the per-bit loop is the specification and a test compares the two. The LFSR is a serial recursion and is not part of the kernel's SIMD plan; `scramble_block` measures it.
- `ScrambleConfig`; no storage. Timing ratio 1, delay 0. Pipeline kind `scramble` (`c32` or `cq15`), cost 2 per sample for `bits` 1 and 2 (the quadrant map and the PRBS bits) and 5 for `bits` 3 to 8 (the four products of the residual phasor, the quadrant map and the PRBS bits), relative to `rotate` (6) (both values to be confirmed; measured and corrected to `7 + 7·bits` at `bits` 1 and 2 and `17 + 7·bits` at `bits` 3 to 8, which includes the PRBS key, D2090), a row in `docs/pipeline.md` and `docs/timing.md`. The rustdoc states that a public PRBS key gives spectral whitening and separation, not secrecy.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 690 … 699.

1. Golden cases of §4 through the facade, `Fallback` and `Reference` (`sdrk-testkit/tests/golden_m49.rs`); the `bit_exact` cases compare with `==` on raw bits.
2. Unit tests (`sdrk-core/tests/kernels_m49.rs`):
   - parameters: `bits` 0 and 9 are `InvalidParameter`; length mismatches; the index mask (`p = 255`, `bits = 2` equals `p = 3`);
   - the helpers of §1.1: `q15_neg_sat(−32768) = 32767` and `−x` otherwise for every `Q15` value; `cq15_quadrant` and `c32_quadrant` against the table for `q` 0 … 7 (`q & 3`); the quadrant round trip (`q`, then `(4 − q) & 3`) for every `Q15` value in each component and every `q` 0 … 3: exact, except that −32768 returns as −32767 in `re` for `q ∈ {2, 3}` and in `im` for `q ∈ {1, 2}`, and stays −32768 in the other components (`q = 0` changes nothing); `c32_quadrant` on `±0`, `±∞`, subnormals and NaN payloads by `to_bits()`; `cq15_chip`/`c32_chip` for `b` 0 and 1;
   - `k = 0` is the identity at every `bits`, both types, full-scale `CQ15` and the `C32` special values included; *(revised by M49.1: only at `bits ≤ 2`; at `bits ≥ 3` `k = 0` multiplies by the table phasor `(32767, 0)` / `(1.0, s₀)`)*
   - `bits = 1` gives `±x` and `bits = 2` the four quadrant maps, exactly; the forward-then-inverse round trip at `bits` 1 and 2 is exact for every `Q15` value in each component, except −32768 → −32767 in exactly the components the quadrant rule of §1.1 negates (both components at `bits = 1`, `k = 1`);
   - the direct lookup equals the `rotate` interpolation for every residue at every `bits` 3 … 8, checked through the public `Rotate` kernel (the `interp_*` helpers are private): with `Nco::from_increment(0)` at phase `r << (32 − bits)`, `Rotate` on the `C32` input `(1, 0)` gives exactly `(COS_F32[i], COS_F32[i + 3072])`, and on the `CQ15` input `(−32768, 0)` exactly `(−COS_Q15[i], −COS_Q15[i + 3072])`, `i = r << (12 − bits)`; no entry of `sdrk_core::tables::COS_F32` is `±0.0`;
   - for `r ≠ 0` the output equals one `rotate` step at the constant phase `r << (32 − bits)` followed by the quadrant map, bit for bit, for every `k` at `bits` 3 and 8; *(revised by M49.1: for every `k` at `bits` 3 to 8 the output equals one `rotate` step at the phase `k << (32 − bits)`, no quadrant map)*
   - the `inverse` index is `(−k) mod 2^b`; the round trip at `bits` 3 … 8 is within 2 LSB (`CQ15`, `|x| ≤ 1`) and SNR ≥ 120 dB (`C32`);
   - `Fallback` against `Reference` with special inputs (silence, full scale, NaN/∞ for `C32`): bit for bit where `r = 0`, within the contract elsewhere; `scramble_one_*` equals the kernel per sample.
3. Block-size invariance: 20 chunk plans (seed 695), both types, a fixed phase sequence sliced with the input.
4. Extremes: empty input; every index value 0…255 at `bits` 8 (both directions); `CQ15` full-scale input (saturation in the residual product as in `rotate`, the saturating negation in the quadrant map); NaN and ∞ in `C32` (`±∞` stays `±∞` where `r = 0`).
5. Suite cases `scramble_c32`, `scramble_cq15` and `extremes_m49`.
6. **Block and kind** (`sdrk-blocks/tests/scramble_blocks.rs`, `sdrk-pipeline/tests/scramble_kinds.rs`; seeds 696, 697): the block equals the kernel with the PRBS indices bit for bit on whole buffers and in 20 chunk plans, both types; a word-parallel PRBS step, if used, equals `next_bit` over the full period of every polynomial; scramble then descramble with the same key returns the input exactly at `bits` 1 and 2 (`CQ15` except −32768) and within the round-trip bounds of §3 at `bits` 3 to 8; the description rules of M09, struct at most 512 bytes, validation errors at their parameter (`bits` 0 and 9, `key_seed` 0 and a seed that masks to 0); a pipeline flow `scramble` → `scramble` (`inverse`) equals the blocks; `cargo xtask schema --check` and the WASM kind list include `scramble`.
7. **SIMD** (`sdrk-backend-simd/tests/scramble.rs`, `sdrk-backend-simd/tests/sign.rs`; inputs from the golden seeds 690 … 694, 698, 699): every level that has a path equals `Fallback` bit for bit, both types, every `bits` 1 … 8, both directions, lengths 0 … 67 and 4096, unaligned input and output slices, every phase byte 0 … 255; special values (`CQ15` components −32768 and 32767; `C32` `±0`, `±∞`, subnormals, NaN; a sample with `r = 0` is compared by `to_bits()` with NaN payloads included (pure sign and swap operations, §1.1), a sample with `r ≠ 0` by `to_bits()` except that a NaN result is compared by class (the residual product, D296)); the `sign` helpers against the scalar helpers of §1.1 on every `Q15` value; the golden cases of §4 through each level; `backend_for("scramble", ..)` reports the level.

## Acceptance

R1.2 commands (`cargo xtask check` includes `cargo xtask check cycles`, D321); `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-backend-simd --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-proto`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the entries of §5 and `scramble_block`); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [x] Kernel `scramble` (`C32`, `CQ15`): trait, the decomposition into quadrant and residue, direct table lookup, Fallback, Reference, `scramble_one_c32`/`scramble_one_cq15`, facade delegation, `backend_for`
- [x] The exact `±1`/`±j` helpers of §1.1 in `sdrk_core::num` (`q15_neg_sat`, `cq15_quadrant`, `c32_quadrant`, `cq15_chip`, `c32_chip`) and the module `sign` in `sdrk-backend-simd`; the section "Exact `±1` and `±j` operations" (shared conventions §A) in `docs/kernels.md`
- [x] Contract rows and section in `docs/kernels.md` (quadrant map plus residual `rotate` product; the round-trip bounds; the relation to `rotate`; the DVB-S2 note)
- [x] SIMD paths of §5 on every level (each passing D321), bit-identical to the fallback; `docs/backends.md`
- [x] Golden cases (seeds 690–694, 698, 699) and the Python model (`q15.py` `neg_sat`)
- [x] Tests 1–7 (seeds 690–699); suite cases
- [x] Count and cycle entries of §5 (`cargo xtask count`, `cargo measure`) with the reference rows (`benches/refs/gr-dvbs2-plscramble.c` new), `docs/performance.md`
- [x] Block `ScrambleBlock<T>` and kind `scramble` (§6); the polynomials `Prbs5`, `Prbs7`, `Prbs10`, `Prbs11` in `sdrk_proto::prbs` with their tests; `docs/blocks.md` (entry; reserved-name row "delivered"), `docs/pipeline.md` (cost 2 / 5, to be confirmed; measured: `7 + 7·bits` / `17 + 7·bits`, D2090), `docs/timing.md`, schema, WASM kind list; count entry `scramble_block`
- [x] CHANGELOG, decisions log (the quadrant decomposition and the exact `±1`/`±j` operations (maintainer, 2026-11-17); the direct lookup instead of the interpolation; the `inverse` index rule; the reference definition and the tolerance class; the index-mask rule; the SIMD forms per level and any level that stays on the fallback, with its numbers; the PRBS stepping of the block)

## Status

Implemented (2026-11-18, branch `m49-scramble`, decisions D2050 to D2093; the maintainer decisions of 2026-11-17 are D2050 and the corrections of D2054, that of 2026-11-18 is D2092). Every item of the deliverables checklist is done; no open item is left (I49-1 and I49-2 are closed by D2092). Since M50 (maintainer decision 2026-11-23, D2618) the parameter `key_seed` of this spec is named `seed`.

**Done:**
- §1 to §3: `kernels::scramble` with `Scramble<T>`, the quadrant/residue split, the direct lookup from `COS_F32`/`COS_Q15`, `Fallback` (branch-free, a per-call residue table, D2055, D2056), `Reference` (`f64`, `libm`, rounded once, D2053), `scramble_one_c32`/`scramble_one_cq15` and the backend helpers (D2057; total for every `bits`, D2058), facade delegation to the SIMD backend; contract rows, the `scramble` section and the section "Exact `±1` and `±j` operations" (§A) in `docs/kernels.md`. Round trip measured: `CQ15` at most 2 LSB at every `bits` 3 to 8 (the provisional bound is confirmed, D2064), `C32` 143.9 to 147.1 dB.
- §1.1: `q15_neg_sat`, `cq15_quadrant`, `c32_quadrant`, `cq15_chip`, `c32_chip` in `sdrk_core::num`; the module `sign` in `sdrk-backend-simd` with public slice functions (D2065).
- §4: the seven golden cases (seeds 690 to 694, 698, 699; corrections 1 and 2 of D2054 applied), `q15.neg_sat`, and the new model `prbs.py` with the case `proto/prbs_first_bits` (seed 696, correction 3, D2061).
- §5: SIMD paths on every level (SSE2, SSE4.2, AVX2, AVX-512, NEON, `simd128`), both types, every `bits` 1 to 8, both directions, bit-identical to the fallback; no level dispatches to the fallback (D2065 to D2071). D321 holds on every level and both core models: the SIMD paths simulate 0.10 to 0.99 of their fallbacks' cycles, the closest `scramble_c32_b8` at `avx512` (1.99 against 2.01, D2079). Three measured attempts and one dropped attempt are logged (D2085 to D2088). Count entries on all four platforms and cycles on every level with a core model, the reference rows (VOLK compositions, CMSIS-DSP composition marked 3.13, GNU Radio `dvbs2_physical_cc` as information in the new `benches/refs/gr-dvbs2-plscramble.c`; D2080 to D2084), `docs/performance.md`, `docs/backends.md`.
- §6: `Prbs5`, `Prbs7`, `Prbs10`, `Prbs11`, `Poly::second_tap` and the word-parallel `Prbs::next_bits` (D2072, D2074); `ScrambleBlock<T>`, `ScrambleConfig`, `SCRAMBLE_CHUNK` (D2073, D2075 to D2077); the kind `scramble` with the measured cost `7 + 7·bits` / `17 + 7·bits` instead of the spec estimate 2 / 5 (D2090); `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list (81 kinds).
- Tests 1 to 7 (seeds 690 to 699): `golden_m49.rs`, `kernels_m49.rs`, the suite cases `scramble_c32`, `scramble_cq15`, `extremes_m49` (`suite_m49.rs`), `prbs_m49.rs`, `scramble_blocks.rs`, `scramble_kinds.rs`, the SIMD tests `scramble.rs` and `sign.rs` (run on every level: SSE2, SSE4.2, AVX2 natively and by fixed builds, AVX-512 under SDE, NEON under QEMU, `simd128` under wasmtime).

**Reference does less work (information, D2092):**
- The library compositions at `bits = 8` (the phasor lookup plus VOLK's `volk_32fc_x2_multiply_32fc`, its generic protokernel for the fallback, or CMSIS-DSP's `arm_cmplx_mult_cmplx_q15`) only look up the phasor and do one complex product; the contract also does the index split into quadrant and residue, the exact quadrant map and the `r = 0` selection, and in `CQ15` the correctly rounded and saturated product (CMSIS truncates to Q3.13). The maintainer decided on 2026-11-18 (D2092) that these comparisons are not fair: `scramble_c32_b8` (every level), `scramble_c32_b8_fallback`, `scramble_cq15_b8` and `scramble_cq15_b8_fallback` are `info` with the waiver "reference does less work". The former open items I49-1 (the fallbacks, `C32` 1.09 to 1.72, `CQ15` 2.13 to 3.62) and I49-2 (the SIMD `C32` path at `sse4.2` 1.35 / 1.19, `avx2` 1.07 / 1.14, `neon` reference model 1.10) are closed (D2093). The D321 gate against our own fallbacks is unchanged and holds.

**Not done:**
- The Cortex-M4 count of `scramble` (deferred: embedded targets on hold, D2084).
- The AVX2 gather form was dropped on a wall-clock timing only and not measured with `cargo measure` (D2070); this stays a note, not an open item (D2093).
- The whole `cargo test --workspace` and the complete crate suites of the acceptance list were not run here (only the M49 targets and the gate commands); the full Cortex-M4 emulator profile (which runs `extremes_m49` on the target) was not run.
