# M49 — Kernel `scramble`: keyed phase scrambling

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M30](M30-two-port-blocks.md) · Next: [M50](M50-xcorr-fft.md)

> **Status: planned** (written 2026-11-08). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

## Goal

A kernel `scramble` that multiplies a complex stream, element by element, by unit phasors `exp(j 2π p[n] / 2^b)` taken from a caller-supplied index sequence `p` (one index per sample, `b` phase bits), and a descrambler that multiplies by the conjugate. It uses the cosine/sine tables of `rotate`, so it is bit-exact and phase-locked to `nco`, `rotate` and `oscillate`. For `b = 1` the phasor is `±1` (sign scrambling); for `b = 2` it is `{1, j, −1, −j}` (the DVB-S2 physical-layer scrambling pattern). The index sequence is data to the kernel; a later milestone generates it from a keyed keystream (`sdrk-proto`).

## Why

A "noise-like" transmitter removes the structure a detector keys on: the residual carrier that a power detector (BPSK x², QPSK x⁴) finds, the visible constellation, the repeating pilot pattern. Multiplying every sample (DSSS) or every subcarrier (OFDM) by a pseudo-random phase from a key does that, and the receiver removes it with the conjugate of the same sequence. The operation is one element-wise complex multiply, but it needs its own kernel so that the phasor comes from the exact `rotate` tables (bit-exact, same on every platform) rather than a per-sample `exp`, and so that both DSSS streams and OFDM lane steps use one definition.

## Order

M49 depends on M30 (the kernels `mul` and `oscillate` and the `rotate` tables it reuses). It is independent of the other kernels of the series and can run in parallel with every milestone after M30.

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

**In:** kernel `scramble` (`C32`, `CQ15`) in `sdrk_core::kernels::scramble` with the trait `Scramble<T>`, Fallback, Reference, facade delegation, contract row and section in `docs/kernels.md`; golden cases; unit, invariance and extreme tests; suite cases; `cargo xtask count` entries; CHANGELOG, decisions log.

**Out:** the block and kind `scramble`, the keystream generator and the index sequences (`sdrk-proto`, later milestone; the name `scramble` is reserved here); a SIMD path (allowed later if it gives the bits of the fallback); real (`f32`/`Q15`) scrambling (there is nothing to rotate on a real stream; `b = 1` sign scrambling of a real stream is `mul` by `±1` and needs no kernel).

## Specification

General rules of M19 apply. The kernel is a stateless element-wise operation, like `mul`.

### 1. Trait and signature

```rust
/// `b` phase bits: the index `p[n]` selects the phasor `exp(j 2π p[n] / 2^b)`.
pub trait Scramble<T> {
    fn scramble(&self, input: &[T], phase: &[u8], bits: u32, inverse: bool, output: &mut [T]) -> Result<()>;
}
```

- `bits` is 1 to 8, else `InvalidParameter` (nothing written). `input`, `phase` and `output` must have the same length, else `LengthMismatch` (nothing written).
- Each index is taken modulo `2^bits`: the phasor uses `k = (p[n] & ((1 << bits) - 1))`, so an out-of-range byte never indexes out of the table (no error for a large byte; documented).
- The phasor angle is `theta_n = 2π k / 2^bits`, as a `u32` NCO phase `ph = k << (32 - bits)` (exact: `2^bits` divides `2^32` for `bits ≤ 8`). `inverse = true` uses `-ph` (`ph.wrapping_neg()`), the conjugate phasor.
- Output: `output[n] = input[n] · exp(j · 2π · ph / 2^32)`, the complex product of the sample with the phasor. The phasor's cosine and sine come from the `rotate` tables **with the interpolation of `rotate`** (`c = cos(ph)`, `s = sin(ph)` = the cosine at `ph - 2^30`), and the product is formed exactly as `rotate` forms it, so `scramble(x, [k], bits, false, y)` equals one `rotate` step at the constant phase `ph` bit for bit. For `bits ≤ 12` the index `ph >> 20` hits a table entry exactly and the interpolation fraction is 0, so `b = 1` gives the exact `±(32767/32768)` of the table and `b = 2` the exact table entries at 0, π/2, π, 3π/2; the one-LSB table offset at angle 0 (§`rotate`) applies unchanged and the golden cases carry it.

### 2. Fallback and Reference

- **Fallback `C32`:** `c, s` by the `rotate` f32 interpolation of `ph`, then `(re·c − im·s, re·s + im·c)` in f32 without FMA — the `rotate` fallback with a per-sample constant phase instead of an accumulator.
- **Fallback `CQ15`:** `c, s` by the `rotate` `CQ15` `i32` interpolation of `ph`, then `cq15_mul` per component — the `CQ15` `rotate` fallback.
- **Reference:** `theta = 2π ph / 2^32` in f64, `libm::cos/sin`, f64 complex product rounded once (the `rotate` reference). Bit-exact class: fallback equals reference.
- `backend_for("scramble", ..)` reports `fallback`.

Because the per-sample work is exactly a `rotate` step, the kernel reuses the `rotate` helpers (`cos_table_interp_f32`, the `CQ15` interpolation) rather than copying them; the implementer exposes them `pub(crate)` if needed and logs it.

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `scramble` | `kernels::scramble::Scramble` | `C32` | tolerance | SNR ≥ 120 dB vs. the f64 reference (the `rotate` `C32` contract); table interpolation in f32 without FMA; the conjugate phasor for `inverse` |
| `scramble` | `kernels::scramble::Scramble` | `CQ15` | tolerance | SNR ≥ 80 dB vs. the f64 reference (the `rotate` `CQ15` contract); `i32` table interpolation, `cq15_mul` |

The classes and thresholds are those of `rotate`, because the kernel is a `rotate` step. **Round-trip:** for any `input`, `bits` and `phase`, `scramble(input, phase, bits, false)` followed by `scramble(·, phase, bits, true)` returns `input` within the `rotate` round-trip error (exact in `C32` where `re` or `im` is zero after the first step is avoided; within one LSB in `CQ15`). A unit test checks the round-trip SNR (≥ 120 dB `C32`, the descrambled stream against the input).

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/scramble.py`, reusing the cosine-table model of `rotate.py` (the generator builds the same `COS_F32`/`COS_Q15` tables and interpolates).

| Case | Seed | Type | `bits` | `inverse` | Phase sequence | Length |
| --- | --- | --- | --- | --- | --- | --- |
| `scramble/bpsk_c32_2048` | 690 | C32 | 1 | false | PN (Python `random`, 0/1) | 2048 |
| `scramble/qpsk_c32_2048` | 691 | C32 | 2 | false | 0…3 | 2048 |
| `scramble/oct_c32_2048` | 692 | C32 | 3 | false | 0…7 | 2048 |
| `scramble/qpsk_inv_c32_2048` | 693 | C32 | 2 | true | 0…3 | 2048 |
| `scramble/qpsk_cq15_2048` | 694 | CQ15 | 2 | false | 0…3 | 2048 |

- Expected values from the f64 reference (`rotate` model), the same `max_abs_err`/SNR tolerances as the `rotate` cases. The phase sequence is stored as a `u8` array in the case.
- A `b = 2` case against the DVB-S2 physical-layer scrambling pattern `{1, j, −1, −j}` is noted in the comment (cross-check): the standard multiplies payload symbols by `j^{R_n}` with `R_n ∈ {0,1,2,3}`, which is exactly `exp(j 2π R_n / 4)`; `scramble` with `bits = 2` and `p = R_n` reproduces it (the `R_n` generation from the two m-sequences is a `sdrk-proto` concern of a later milestone, not this kernel).

### 5. Cost

`cargo xtask count` entries `scramble_c32` and `scramble_cq15` (per sample; equal to a `rotate` step plus the index mask). Record next to `rotate` in `docs/performance.md`.

## Tests

Seeds 690 … 699.

1. Golden cases of §4 through the facade, `Fallback` and `Reference` (`sdrk-testkit/tests/golden_m49.rs`).
2. Unit tests (`sdrk-core/tests/kernels_m49.rs`): `bits` 0 and 9 are `InvalidParameter`; length mismatches; the index mask (`p = 255`, `bits = 2` equals `p = 3`); `bits = 1` gives `±` the angle-0 and angle-π table values; `bits = 2` the four axis phasors; the round-trip (forward then inverse equals the input within tolerance) for `C32` and `CQ15`; equality with one `rotate` step at a constant phase for 8 constant indices; `Fallback` against `Reference` with special inputs (silence, full scale, NaN/∞ for `C32`).
3. Block-size invariance: 20 chunk plans (seed 695), both types, a fixed phase sequence sliced with the input.
4. Extremes: empty input; every index value 0…255 at `bits` 8; `CQ15` full-scale input (saturation as in `rotate`); NaN and ∞ in `C32`.
5. Suite cases `scramble_c32`, `scramble_cq15` and `extremes_m49`.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo xtask count --platform x86_64` (the entries of §5).

## Deliverables checklist

- [ ] Kernel `scramble` (`C32`, `CQ15`): trait, Fallback, Reference, facade delegation, `backend_for`
- [ ] Contract rows and section in `docs/kernels.md` (as a `rotate` step; the round-trip and the DVB-S2 note)
- [ ] Golden cases (seeds 690–694) and the Python model
- [ ] Tests 1–5 (seeds 690–695); suite cases
- [ ] `cargo xtask count` entries, `docs/performance.md`
- [ ] Reserved kind name `scramble` in `docs/blocks.md`
- [ ] CHANGELOG, decisions log (reuse of the `rotate` helpers, the index-mask rule)

## Status

Planned. Nothing is implemented.
