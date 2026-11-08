# M56 — Kernel `fwht`: fast Walsh–Hadamard transform

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M29](M29-primitive-blocks.md) (the block; through it M19, whose general rules apply) · Next: [BACKLOG](BACKLOG.md)

> **Status: planned** (written 2026-11-08). Last kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

## Goal

A kernel `fwht` that applies the fast Walsh–Hadamard transform to a complex vector of a power-of-two length, in place, in natural (Sylvester/Hadamard) ordering, forward (unnormalised) and inverse (divided by `N`). It is the spreading transform of multicarrier CDMA (MC-CDMA / spread-OFDM): multiplying the data subcarriers by the Hadamard matrix spreads every data symbol across all subcarriers, so the power per subcarrier drops below the noise floor — a frequency-domain route to low probability of detection that complements the time-domain spreading of `spread`.

## Why

Spreading across OFDM subcarriers with a Hadamard matrix is the cheapest way to flatten an OFDM spectrum and hide each symbol's energy: it is `N·log2(N)` additions, no multiplies, and orthogonal, so the receiver inverts it exactly. It is the one transform MC-CDMA needs that the library does not have; the FFT, pilot equalizer and scrambler are already covered by this series. A dedicated kernel fixes the ordering and the summation order so transmitter and receiver agree bit for bit on every platform.

## Order

M56 depends only on M02 (the sample types and kernel pattern). It is independent of every other kernel and can run at any time; it is placed last in the series because MC-CDMA (`freq_spread`) is the last of the OFDM variants.

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

**In:** kernel `fwht` (`C32`, `CQ15`) in `sdrk_core::kernels::fwht` with `Fwht<T>`, Fallback, Reference, facade delegation, contract, `docs/kernels.md`; golden cases; unit, invariance, extreme tests; suite cases; `cargo xtask count`; CHANGELOG, decisions log; the block `FwhtBlock<T>` and the pipeline kind `fwht` (§5) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the MC-CDMA blocks `freq_spread`/`freq_despread` with their `1/√N` scaling on lanes (later milestone; names reserved; the primitive block `fwht` of §5 is the plain transform on a stream); sequency or dyadic ordering (natural ordering only; a permutation to sequency order is a flow-level reorder if ever needed, backlog); non-power-of-two lengths (the transform is defined only for powers of two).

## Specification

General rules of M19 apply.

### 1. Transform

The Walsh–Hadamard transform of a length-`N` vector (`N` a power of two, 2…65536) is `X = H_N · x`, with `H_N` the Sylvester–Hadamard matrix (`H_1 = [1]`, `H_{2N} = [[H_N, H_N], [H_N, −H_N]]`), natural ordering. The fast algorithm is `log2(N)` butterfly stages in place:

```
for len in powers 1, 2, 4, …, N/2:          # stage with half-size `len`
    for each block of 2*len starting at i:
        for j in 0..len:
            a = data[i + j]; b = data[i + j + len]
            data[i + j]       = a + b
            data[i + j + len] = a − b
```

- **Forward** (`inverse = false`): the stages above, **unnormalised**. `X[0]` is the plain sum of the input (matches `scipy.linalg.hadamard` ⋅ x and `sympy.discrete.fwht`, which are unnormalised: `fwht([4,2,2,0,0,2,-2,0]) = [8,0,8,0,8,8,0,0]`, first element the sum).
- **Inverse** (`inverse = true`): the same butterflies, then divide every element by `N` (`H_N⁻¹ = H_N / N`). `C32` multiplies by the exact `1.0/N`; `CQ15` applies the rounding/saturation rule of R7 once per element after the butterflies (the division by a power of two is an arithmetic right shift with round-half-up). This matches `sympy.ifwht` (it divides by `n`: `ifwht([19,-1,11,-9,-7,13,-15,5]) = [2,0,4,0,3,10,0,0]`).
- The transform applies to a complex vector per component (real and imaginary independently); a real transform is the same on each `C32` component. `CQ15` butterfly sums saturate to `[−32768, 32767]` (a full-scale spread can grow by up to `N`; the forward transform of near-full-scale data saturates, documented — MC-CDMA scales the data down by `1/√N` before spreading so the spread stays in range, which the block does, not the kernel).

### 2. State and trait

```rust
pub trait Fwht<T> {
    /// In place; `data.len()` a power of two in 2..=65536.
    fn fwht(&self, data: &mut [T], inverse: bool) -> Result<()>;
}
```

- A length that is not a power of two or outside the range is `InvalidParameter` (nothing changed). The kernel is stateless (no state type).
- **Summation order (bit-exact).** The butterfly order above is fixed (stages of increasing `len`, blocks in index order, `j` ascending), so the result is deterministic and reproducible on every backend. `C32` sums in f32 without FMA. The forward-then-inverse round trip returns the input exactly for `CQ15` only up to the rounding of the final `/N` (the division loses the low `log2(N)` bits; documented — spreading keeps the amplitude, despreading divides, so the data symbols survive but the chip-level values are rounded). For `C32` the round trip returns the input within the f32 accumulation error (a unit test checks the SNR).

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `fwht` | `kernels::fwht::Fwht` | `C32` | bit-exact | the Sylvester butterflies of §1 in the fixed order, f32 without FMA; inverse multiplies by `1.0/N`; fallback equals reference |
| `fwht` | `kernels::fwht::Fwht` | `CQ15` | bit-exact | `i32` butterfly sums saturated to Q15 each stage; inverse divides by `N` with round-half-up and one saturation (R7); fallback equals reference |

`Reference` is the naive `H_N · x` (an O(N²) matrix multiply in f64 for `C32`, the integer model for `CQ15`), independent of the butterfly algorithm, so it validates both the ordering and the fast transform. `backend_for("fwht", ..)` reports `fallback`.

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/fwht.py`, the matrix model (`scipy.linalg.hadamard(N)` ⋅ x for the forward, divided by `N` for the inverse).

| Case | Seed | Type | `N` | direction |
| --- | --- | --- | --- | --- |
| `fwht/fwd_c32_8` | 760 | C32 | 8 | forward (matches the SymPy example vector, cross-check) |
| `fwht/fwd_c32_1024` | 761 | C32 | 1024 | forward, random complex |
| `fwht/inv_c32_1024` | 762 | C32 | 1024 | inverse, random complex |
| `fwht/fwd_cq15_256` | 763 | CQ15 | 256 | forward, scaled so no saturation |
| `fwht/roundtrip_cq15_64` | 764 | CQ15 | 64 | forward then inverse, the rounding loss recorded |

- The `N = 8` case stores the SymPy example input/output (`fwht([4,2,2,0,0,2,-2,0]) = [8,0,8,0,8,8,0,0]`) as a real-valued check that the ordering and normalisation match an independent library.
- Expected from `scipy.linalg.hadamard` (forward) and `/N` (inverse); `CQ15` from the integer model. Bit-exact `==` for `C32` against the f32 model (the matrix model in f32 is itself order-dependent, so the golden uses the same fixed butterfly order as the contract for the f32 expected values, and the f64 matrix product as a guard); `CQ15` bit-exact against the integer model.

### 5. Cost

`cargo xtask count` entries `fwht_c32_1024` and `fwht_cq15_256` (per element; `log2(N)` add/subtract pairs). Record in `docs/performance.md`; note there are no multiplies (except the single `1/N` on the inverse), so it is far cheaper than an FFT of the same length.

**Measurement (D32, D33, D68).** The entries above and the block entry of §5 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`). A library reference enters only where it does the same work (D32); otherwise the rows are information. The D321 gate applies once a SIMD path exists.

### 5. Block and kind `fwht`

`FwhtBlock<T>` (`C32`/`CQ15` → same) around the kernel `fwht`: the stream is cut into frames of `size` samples counted from the first sample after construction or `reset`, and every frame is transformed in place (a copy to the output, then the kernel on the output slice).

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `fwht` | `FwhtBlock<T>` | C32/CQ15 → same | `size` (I64, build, required; a power of two 2 … 65536), `inverse` (Bool, build, false) | `fwht`; a call takes whole frames limited by input and output room, `min_input` `size` |

- `FwhtConfig`; no storage. Timing ratio 1, delay 0 (a frame transform, not a filter; output frame `k` is the transform of input frame `k`; logged). A lane variant for the OFDM chain comes with `freq_spread`.
- Pipeline kind `fwht` (`c32` or `cq15`), cost `log2(size)` per input sample (one add or subtract per stage; the inverse adds the `1/N`), a row in `docs/pipeline.md` and `docs/timing.md`. Rule 6 of the names: `fwht` is an established abbreviation; the summary writes it out once.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 760 … 769.

1. Golden cases of §4 through facade, `Fallback`, `Reference` (`sdrk-testkit/tests/golden_m56.rs`).
2. Unit tests (`sdrk-core/tests/kernels_m56.rs`): non-power-of-two and out-of-range lengths are `InvalidParameter`; `N = 2` is one butterfly `[a+b, a−b]`; `X[0]` is the sum of the input; the transform equals the Sylvester matrix product (`Fallback` against `Reference`) for `N` = 2…1024; orthogonality (forward of a unit vector `e_k` is the `k`-th Walsh function, ±1); the round trip (`C32` within SNR, `CQ15` within the `/N` rounding); the SymPy example vector; `CQ15` saturation on a full-scale forward.
3. Suite cases `fwht_c32`, `fwht_cq15`, `extremes_m56` (silence, full scale, NaN/∞ for `C32`, every power-of-two length to 1024, both directions).
4. (No streaming state, so no chunk-invariance test; the transform is whole-vector.)
5. **Block and kind** (`sdrk-blocks/tests/fwht_blocks.rs`, `sdrk-pipeline/tests/fwht_kinds.rs`; seeds 765, 766): the block equals the kernel frame by frame bit for bit on whole buffers and in 20 chunk plans that cut frames, both types and both directions; forward then inverse blocks return the input (`C32` within SNR, `CQ15` within the `/N` rounding); the description rules of M09, no storage, struct at most 512 bytes, validation errors at `params.size`; a pipeline flow equals the block; `cargo xtask schema --check` and the WASM kind list include `fwht`.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the entries of §5 and `fwht_block`); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [ ] Kernel `fwht` (`C32`, `CQ15`): trait, Fallback, Reference (matrix), facade delegation, `backend_for`
- [ ] Contract rows and section in `docs/kernels.md` (Sylvester ordering, normalisation, the SciPy/SymPy cross-check)
- [ ] Golden cases (seeds 760–764) and the Python model
- [ ] Tests 1–3 and 5 (seeds 760–766); suite cases
- [ ] Count and cycle entries (`cargo xtask count`, `cargo measure`), `docs/performance.md`
- [ ] Block `FwhtBlock<T>` and kind `fwht` (§5): `docs/blocks.md` (entry and a new reserved-name row), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `fwht_block`
- [ ] CHANGELOG, decisions log (the ordering choice, the normalisation convention, the `CQ15` round-trip loss)

## Status

Planned. Nothing is implemented.
