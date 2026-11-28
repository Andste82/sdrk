# M56 — Kernel `fwht`: fast Walsh–Hadamard transform

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M29](M29-primitive-blocks.md) (the block; through it M19, whose general rules apply) · Next: [BACKLOG](BACKLOG.md)

> **Status: planned** (written 2026-11-08; revised 2026-11-17 after an algorithm review (maintainer decision)). Last kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

> **Revision 2026-11-17.** The shared conventions §A to §D cited in this file are defined in [CONVENTIONS-M49-M56.md](CONVENTIONS-M49-M56.md). The maintainer accepted every recommendation of the algorithm review of M49 to M56 (decision of 2026-11-17). The shared definitions are those of the shared conventions for M49 to M56; this milestone introduces §C.2 (per-stage transform shift) and writes its full text into `docs/kernels.md`. Changes in this spec:
> 1. **Error corrected: the `CQ15` inverse clipped.** The old inverse saturated every stage and divided by `N` only at the end, so it clipped as soon as the chips exceeded about `32767/N`. The review measured this in NumPy at `N = 64`: amplitude 511 was exact, amplitude 4095 had errors up to 3643 LSB, and one spread symbol of 4000 came back as 512. The direction flag `inverse: bool` is replaced by a scaling argument with the per-stage rounding shift of the M07 `CQ15` FFT (shared conventions §C.2): `Unscaled` (the forward), `Fixed` (the inverse `H/N`) and `BlockFloat`. The exponent is returned as `FftScale` (§1, §2, §3).
> 2. **Error corrected: class and reference contradicted each other.** An `f64` matrix product cannot equal `f32` butterflies bit for bit, and with saturation the `CQ15` matrix model cannot either. `Reference` now runs the butterfly model, as `squelch`, `stereo_pll` and `costas` do (D111). The matrix product is the tolerance guard (§3, §4, tests 1 and 2).
> 3. **Error corrected: the round trip.** Without saturation the `CQ15` round trip is lossless. The old text claimed it loses the low `log2(N)` bits (§1, §2, decisions log).
> 4. **Improvement: SIMD in scope** (maintainer rule 2026-11-17, shared conventions §D). Both types and every scaling get paths on every level, bit-identical to the fallback. The contract states that any lane schedule, stage fusion or cache blocking gives the same bits (§2, §5, test 6).
> 5. **Improvement: reference row.** FFHT (FALCONN, MIT licence) is the fair reference for `C32` on x86. The old text said that no library has the transform (§5).
> 6. **Smaller fixes.**
>    - The MC-CDMA note on `1/√N` is corrected: it keeps the RMS of the chips, not the peak (§1).
>    - The two sections numbered 5 are now §5 and §6.
>    - The Order section names the same dependencies as the Depends line.
>    - New golden cases (seeds 767, 768), a SIMD test (seed 769) and new cost entries; tests, acceptance and the checklist follow.
> 7. **Cross-milestone consistency (M49–M56 pass).** The canonical sentence of the shared conventions §C is in §1 (*Scaling*); the backlog entries of the checklist were entered in `BACKLOG.md` with this revision.

## Goal

A kernel `fwht` that applies the fast Walsh–Hadamard transform to a complex vector of a power-of-two length, in place, in natural (Sylvester/Hadamard) ordering: unnormalised (the forward), divided by `N` (the inverse), and for `CQ15` also with block floating point. It is the spreading transform of multicarrier CDMA (MC-CDMA / spread-OFDM): multiplying the data subcarriers by the Hadamard matrix spreads every data symbol across all subcarriers, so the power per subcarrier drops below the noise floor — a frequency-domain route to low probability of detection that complements the time-domain spreading of `spread`.

**Generality (noted 2026-11-28).** The kernel is the generic fast Walsh-Hadamard transform; MC-CDMA is its first user, not its only one. The same transform correlates a vector with all `N` Walsh codes at once (the matched filter bank of Walsh-coded systems), decodes a first-order Reed-Muller code by maximum likelihood in one transform of the soft bits, and decodes codes built from such codes as a set of cosets (the backlog item `block_code`, with the shortened (30,14) Reed-Muller code of the TETRA access assignment channel as its first preset). Nothing in the specification below is specific to MC-CDMA: the ordering, the scalings and the `CQ15` rules hold for every use. Real inputs such as soft bits go through the `C32` path with a zero imaginary part; a real `f32` and `Q15` variant is a follow-up in the backlog. The block and the kind are named for the transform, not for the application.

## Why

Spreading across OFDM subcarriers with a Hadamard matrix is the cheapest way to flatten an OFDM spectrum and hide each symbol's energy: it is `N·log2(N)` additions or subtractions per component, no multiplies, and orthogonal, so the receiver inverts it exactly. It is the one transform MC-CDMA needs that the library does not have; the FFT (M07), the pilot equalizer and the scrambler already exist or come in this series. A dedicated kernel fixes the ordering, the operands of every butterfly and the `CQ15` scaling, so transmitter and receiver agree bit for bit on every platform and every SIMD level.

## Order

The kernel needs only the sample types and the kernel pattern of M02 and the type `FftScale` of M07. The block and the kind need M29, which is the Depends line. M56 is independent of every other kernel of the series and can run at any time. It is placed last because MC-CDMA (`freq_spread`) is the last of the OFDM variants.

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
- the kernel `fwht` (`C32`, `CQ15`) in `sdrk_core::kernels::fwht` with `Fwht<T>`, the enum `Scaling`, Fallback, Reference (the butterfly model, D111 rule), facade delegation, contract, `docs/kernels.md` (with the full text of shared conventions §C.2);
- SIMD paths in `sdrk-backend-simd` on every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`) for both types and every scaling, bit-identical to the fallback (maintainer rule 2026-11-17, §5), and `docs/backends.md`;
- golden cases; unit, extreme and SIMD tests; suite cases;
- `cargo measure` entries with the reference rows of §5, including FFHT in `benches/refs`;
- CHANGELOG, decisions log;
- the block `FwhtBlock<T>` and the pipeline kind `fwht` (§6) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:**
- the MC-CDMA blocks `freq_spread`/`freq_despread` with their `1/√N` scaling on lanes (later milestone; names reserved). The primitive block `fwht` of §6 is the plain transform on a stream. The scaling of a `CQ15` spreader (`BlockFloat`, `Fixed`, or an orthonormal mode with a shift in every second stage, which exists only for even `log2(N)`) is decided there;
- sequency or dyadic ordering (natural ordering only; a permutation to sequency order is a flow-level reorder if ever needed, backlog);
- non-power-of-two lengths (the transform is defined only for powers of two);
- paths in the embedded backends of M16 to M18 (Arm DSP extension, RVV, ESP32-P4 PIE). The reason is the dependency, not the algorithm: those backend crates are scheduled after M29 and M56 does not depend on them (R1.1). The `CQ15` path for the Arm DSP extension (`QADD16`/`QSUB16` for `Unscaled`; `Fixed` needs a correction because `SHADD16` truncates, a few instructions more, to be confirmed) is recorded in `BACKLOG.md` for those backends.

## Specification

General rules of M19 apply.

### 1. Transform

The Walsh–Hadamard transform of a length-`N` vector (`N` a power of two, 2…65536) is `X = H_N · x`, with `H_N` the Sylvester–Hadamard matrix (`H_1 = [1]`, `H_{2N} = [[H_N, H_N], [H_N, −H_N]]`; equivalently `H_N[r][c] = (−1)^popcount(r & c)`), natural ordering. The fast algorithm is `log2(N)` butterfly stages in place:

```
for len in powers 1, 2, 4, …, N/2:          # stage with half-size `len`
    for each block of 2*len starting at i:
        for j in 0..len:
            a = data[i + j]; b = data[i + j + len]
            data[i + j]       = a + b       # with the scaling of the stage (below)
            data[i + j + len] = a − b
```

**Scaling** (shared conventions §C.2, introduced in this milestone). The kernel takes a `Scaling` instead of a direction flag. `H_N⁻¹ = H_N / N`, so the inverse is the transform with the scaling `Fixed`, and the default is not hidden in a direction. `CQ15` growth follows the shared conventions §C: sums are exact in `i64` and leave through one rounding shift `round_shift(acc, f + shift)` with one saturation (default `shift = ceil(log2 sf)`, introduced in M52); transforms shift by 0 or 1 per stage with R7 rounding and one saturation, the exponent returned as `FftScale` (the M07 FFT rule, introduced for `fwht` in M56).
- **`Unscaled`** (the forward): the stages above, unnormalised; exponent 0. `X[0]` is the plain sum of the input. This matches `scipy.linalg.hadamard(N) @ x` and `sympy.discrete.fwht`, which are unnormalised: `fwht([4,2,2,0,0,2,-2,0]) = [8,0,8,0,8,8,0,0]`, first element the sum (checked with SciPy in the revision).
- **`Fixed`** (the inverse): the result is `H_N · x / N`, exponent `log2 N`. This matches `sympy.ifwht`, which divides by `n`: `ifwht([19,-1,11,-9,-7,13,-15,5]) = [2,0,4,0,3,10,0,0]` (checked with SciPy in the revision).
  - `C32`: the stages above, then every component multiplied by the exact `1.0/N` (as the `C32` `fft` inverse).
  - `CQ15`: a shift of 1 with rounding in every stage (below).
- **`BlockFloat`** (`CQ15` only; `C32` gives `Unsupported`): a shift of 0 or 1 per stage, chosen from the largest component; the exponent `E` is the sum of the shifts, and `output · 2^E` is `H_N · x` up to the rounding.

**Per component.** The transform applies to a complex vector per component (real and imaginary independently); a real transform is the same on each `C32` component.

**`CQ15` butterfly** (shared conventions §C.2, the rule of the M07 `CQ15` FFT without twiddles). Computed in `i32`:
- `a' = sat16(round_shift(a + b, s))` and `b' = sat16(round_shift(a − b, s))`, with `round_shift(v, 0) = v` and `round_shift(v, 1) = (v + 1) >> 1` (R7, round half up).
- Nothing saturates before the shift.
- The stage shift `s`:
  - `Unscaled`: 0;
  - `Fixed`: 1;
  - `BlockFloat`: before each stage, `m` is the largest absolute component of the whole vector, real and imaginary parts (`|−32768| = 32768`); `s = 1` for `m ≥ 16384`, else 0. A Walsh–Hadamard butterfly at most doubles `m` and has no twiddle, so one bit per stage is enough. The FFT needs 0, 1 or 2.

**Range and accuracy of `CQ15`.** These are bounds, not measurements; the NumPy checks of the revision stayed within them.
- **`Unscaled`:** exact `H·x` as long as no stage result leaves `[−32768, 32767]`. Otherwise that stage saturates (documented). The growth is up to `N`, so near-full-scale data saturates.
- **`Fixed`:** no stage saturates except `a − b = 65535` (`a = 32767`, `b = −32768`), which gives 32767 for 32767.5, a rounding error of −1/2 that the bound below covers.
  - Each stage adds a rounding error of at most 1/2 LSB in magnitude and halves the sum of the two incoming errors, so `|output − H·x/N| ≤ log2(N)/2` LSB.
  - **Lossless round trip:** `Fixed` applied to `y = H·x` computed exactly (an `Unscaled` forward that did not saturate) returns `x` exactly. After `k` stages the partial transform of `H·x` is `2^k` times an integer vector, because the first `k` stages form `I_{N/2^k} ⊗ H_{2^k}` and `H_{2^k}² = 2^k I`. So every rounding shift meets an even sum and rounds nothing (checked in NumPy for every `N` from 2 to 4096).
- **`BlockFloat`:** no stage saturates except the same `a − b = 65535` corner. An unshifted stage doubles the incoming error, and a shifted stage halves it and adds at most 1/2. The error of `output` against `H·x / 2^E` is therefore at most `Σ_i 2^(u_i) / 2` LSB over the shifted stages `i`, with `u_i` the number of unshifted stages after `i`. For a schedule that shifts in every stage this is `log2(N)/2`. The bound held in the NumPy check of the revision, with random input of several amplitudes up to full scale.

**MC-CDMA note (corrected).** Scaling the data by `1/√N` before an unscaled spread keeps the RMS of the chips equal to that of the symbols, but not the peak: a chip can reach `√N` times the symbol amplitude, so the `Unscaled` forward of `1/√N`-scaled full-scale data still saturates a part of the chips. A `CQ15` spreader therefore uses `BlockFloat` or `Fixed`. The choice belongs to `freq_spread`, not to this kernel.

### 2. State and trait

```rust
/// How `fwht` scales (shared conventions §C.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Scaling {
    /// `H·x`, unnormalised (the forward); exponent 0.
    Unscaled,
    /// `H·x / N` (the inverse); exponent `log2 N`.
    Fixed,
    /// `CQ15` only: shift 0 or 1 per stage; exponent the sum of the shifts.
    BlockFloat,
}

pub trait Fwht<T> {
    /// In place; `data.len()` a power of two in 2..=65536. Returns the scale:
    /// `output · 2^exponent` is the unnormalised transform (up to the `CQ15` rounding).
    fn fwht(&self, data: &mut [T], scaling: Scaling) -> Result<FftScale>;
}
```

- `FftScale` is the type of `kernels::fft` (M07), so a caller treats both transforms alike. `kernels::fwht::Scaling` is its own enum because the `Default` of `kernels::fft::Scaling` hides the direction.
- A length that is not a power of two or outside the range is `InvalidParameter`. `BlockFloat` with `C32` is `Unsupported`. In both cases nothing is changed. The kernel is stateless (no state type).
- **Order of operations (bit-exact).** The butterflies of §1 define the operands of every output: in stage `len`, output `i + j` is `a + b` and output `i + j + len` is `a − b`, both computed from that stage's inputs at `i + j` and `i + j + len`. Each output of a stage is one addition or subtraction of two fixed operands: in `C32` one IEEE `f32` operation without FMA, in `CQ15` the formula of §1. Every schedule that computes the same butterflies from the same operands therefore gives the same bits (shared conventions §D.2). The fallback and the SIMD paths may use any such schedule:
  - any order of the blocks and of `j`;
  - lane-parallel stages;
  - fused stages (radix 4 or 8 in registers);
  - cache blocking: all stages with `2·len ≤ B` inside each block of `B` elements first, then the outer stages, as FFHT does.
  - `C32` `Fixed` may fold the `1.0/N` multiply into the pass of the last stage (the same multiply per component).
  - **Exception, `BlockFloat`:** the shift of a stage depends on the whole vector after the previous stage, so stages are neither fused nor blocked. Each stage is one pass over the vector, with the threshold test for the next stage folded into it.
- **Round trip.**
  - `C32`: `Unscaled` then `Fixed` returns the input within the `f32` rounding error (a unit test checks the SNR).
  - `CQ15`: `Unscaled` then `Fixed` returns the input exactly as long as the forward did not saturate (§1). If the forward saturated, the clipped chips are lost.

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `fwht` | `kernels::fwht::Fwht` | `C32` | bit-exact | `float32` model of §1: the Sylvester butterflies, each output one `f32` addition or subtraction of the fixed operands of §2, no FMA; `Fixed` multiplies by `1.0/N`; `BlockFloat` `Unsupported`; fallback, reference and SIMD equal bit for bit (NaN by class, D296); golden cases from the model plus an `f64` matrix guard (D111) |
| `fwht` | `kernels::fwht::Fwht` | `CQ15` | bit-exact | integer model of §1 (shared conventions §C.2): `sat16(round_shift(a ± b, s))` in `i32` per stage, `s` from `Unscaled`/`Fixed`/`BlockFloat`, exponent as `FftScale`; `Fixed` error at most `log2(N)/2` LSB against `H·x/N`, lossless after an unsaturated `Unscaled`; fallback, reference and SIMD equal bit for bit; golden cases from the model plus an integer matrix guard |

**Reference (D111 rule).** `Reference` is the butterfly model of §1 in the fixed order of §1: in `f32` for `C32`, and as the integer stage model in `i64` for `CQ15`. It is written independently of the fallback: a plain triple loop with no fusion, no blocking and no shared helper. This follows the exception already used for `squelch`, `stereo_pll`, `costas` and `equalize` (D111).

**Guard.** The mathematical definition `H_N · x`, independent of the butterfly algorithm, is a tolerance guard. It is the O(N²) matrix product with the entries `(−1)^popcount(r & c)`:
- in `f64` for `C32`, with an SNR floor `snr_db_guard` per case;
- in exact integers for `CQ15`, divided by `2^E`: equal for `Unscaled` without saturation, within the bounds of §1 for `Fixed` and `BlockFloat`.

Together with the SciPy and SymPy vectors, the guard still validates the ordering, the normalisation and the fast transform.

`backend_for("fwht", ..)` reports the SIMD level on every level that has a path (§5). A level reports `fallback` only after the D321 procedure of §5.

**SIMD.** SIMD follows the shared conventions §D (maintainer rule 2026-11-17): every lane-parallel part of the kernel gets a path on every level in this milestone, bit-identical to the fallback through the fixed element-wise formula or reduction order of the contract; D321 is a quality gate (a slower path is improved, not removed; delegation to the fallback only after several measured failed attempts, with an open item); cycles per sample with `cargo measure` against the fallback and the fair reference (D32), unfair rows marked. For `fwht` every stage is lane-parallel, for every scaling, so no part stays scalar.

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/fwht.py`:
- **The model:**
  - the butterfly model in NumPy `float32`: per stage one array addition and one array subtraction with the operands of §2, IEEE round to nearest;
  - the integer stage model in `int64` per §1, with the rounding and saturation helpers of `q15.py`.
- **The guard:** the matrix product `scipy.linalg.hadamard(N) @ x`, in `float64` for `C32` and `int64` for `CQ15`.

| Case | Seed | Type | `N` | Scaling |
| --- | --- | --- | --- | --- |
| `fwht/fwd_c32_8` | 760 | C32 | 8 | `Unscaled` (the SymPy example vector, cross-check) |
| `fwht/fwd_c32_1024` | 761 | C32 | 1024 | `Unscaled`, random complex |
| `fwht/inv_c32_1024` | 762 | C32 | 1024 | `Fixed`, random complex |
| `fwht/fwd_cq15_256` | 763 | CQ15 | 256 | `Unscaled`, scaled so no saturation |
| `fwht/roundtrip_cq15_64` | 764 | CQ15 | 64 | `Unscaled` then `Fixed`; returns the input exactly (loss 0 recorded) |
| `fwht/fixed_cq15_64` | 767 | CQ15 | 64 | `Fixed` on chips of amplitude 4095 (the review's case that the old inverse clipped) |
| `fwht/bfp_cq15_1024` | 768 | CQ15 | 1024 | `BlockFloat`, near-full-scale random complex; the exponent stored |

- The `N = 8` case stores the SymPy example input/output (`fwht([4,2,2,0,0,2,-2,0]) = [8,0,8,0,8,8,0,0]`) as a real-valued check that the ordering and normalisation match an independent library.
- Comparisons: bit-exact `==` for `C32` against the `float32` model and for `CQ15` against the integer model.
- The JSON stores the scaling and the expected exponent, and the guard:
  - `C32`: `snr_db_guard`, the SNR of the `float32` model against the `f64` matrix product minus 1 dB, rounded down to 0.1 dB (as D115);
  - `CQ15`: `max_abs_err_guard` in LSB against the exact matrix product divided by `2^E`: 0 for `fwd_cq15_256` and `roundtrip_cq15_64`, `log2(N)/2` for `fixed_cq15_64`, and the bound of §1 from the stage shifts for `bfp_cq15_1024`.

### 5. Cost, SIMD plan and reference rows

**Entries.** `cargo measure` entries, per element, every level, each with its `<k>_fallback` row, plus `fwht_block` (§6):
- `fwht_c32_1024`;
- `fwht_c32_65536` (new: the cache-blocked path; 512 KiB of data);
- `fwht_cq15_256` (`Unscaled`);
- `fwht_cq15_fixed_256` (new);
- `fwht_cq15_bfp_256` (new).

Numbers go into `docs/performance.md`.

**Cost.**
- Per complex element and stage: one addition or subtraction per component, that is one add/sub pair per butterfly of two elements, and `N·log2(N)` per component in total.
- There are no multiplies, except the `1.0/N` of `C32` `Fixed`.
- `CQ15` `Fixed` adds the rounding. `BlockFloat` adds a threshold test per element and stage, plus one test pass over the input before the first stage.
- The transform is far cheaper than an FFT of the same length; the `fft` of the same length is shown as information.
- Without blocking, a large `C32` transform makes `log2(N)` passes over `8N` bytes and is memory-bound. With blocking in L1 (for example `B` = 2048 elements = 16 KiB), `N = 65536` needs one pass for the 11 inner stages and the 5 outer stages on top (to be confirmed; a second blocking level for L2 is allowed).

**SIMD plan (`sdrk-backend-simd`, new module `fwht`).** The paths are written once against `fearless_simd`, with level paths only where the portable operations are not enough (D1). Bit identity follows from §2: every lane computes the same butterfly from the same operands.
- **Stages with `len` of at least one vector** (`C32`: 2 elements per 128-bit vector, 4 at `avx2`, 8 at `avx512`; `CQ15`: 4, 8, 16): plain lane-wise add and subtract of the vectors at `i + j` and `i + j + len`.
- **Small stages:** in-register shuffles, fused into one radix-4 or radix-8 pass over a group of vectors. A `C32` element is a 64-bit unit (`shufpd`, `vperm2f128`, NEON `zip`/`ext`), a `CQ15` element a 32-bit unit (`pshufd`, `vrev64`, `i32x4.shuffle`).
- **`C32`:** `add`/`sub` of `f32` lanes, never FMA. `Fixed` multiplies by `1.0/N` in the last pass. NaN outputs are compared by class (D296).
- **`CQ15` `Unscaled`:** saturating 16-bit add and subtract (`paddsw`/`psubsw`, `vqaddq_s16`/`vqsubq_s16`, `i16x8.add_sat_s`/`sub_sat_s`). These are exactly `sat16(a ± b)` of `i16` operands.
- **`CQ15` `Fixed`:** the rounding halving add and subtract.
  - NEON: the sum is `vrhaddq_s16(a, b)`, which is `(a + b + 1) >> 1` exactly (it never saturates). The difference is `vqaddq_s16(vhaddq_s16(a, vmvnq_s16(b)), 1)`: `a − b + 1 = a + ~b + 2`, and the saturating `+1` gives the corner 32767.
  - x86 and `simd128`: the sum is `pavgw` / `i16x8.avgr_u` on operands whose sign bit is flipped (`xor 0x8000`), with the result flipped back. The difference is the floor average `(a & c) + ((a ^ c) >> 1)` (arithmetic shift) with `c = ~b`, followed by a saturating `+1`.
  - Both identities were checked in NumPy against the formula of §1, on all edge pairs and 2·10⁶ random pairs.
  - A widening alternative is allowed: add in `i32`, round-shift, then the saturating narrow (`packssdw`, `vqmovn`, `i16x8.narrow_i32x4_s`).
  - `psraw` of a wrapping 16-bit sum is not allowed.
- **`CQ15` `BlockFloat`:** the shift of the next stage needs only the test `m ≥ 16384`, that is, some component `≥ 16384` or `≤ −16384`. Per lane this is an OR of two compares, reduced once per stage. The test for the first stage is one pass over the input; the tests for the later stages are folded into the pass of the previous stage. Stages are not fused or blocked (§2).
- **Cache blocking** for `N` above the block size `B` (per level, chosen by measurement; to be confirmed).
- **D321 quality gate.** Every path must be at least as fast as its fallback in simulated cycles per element, on the reference model and the counter model of its level. A slower path is improved, not removed. Only after several measured failed attempts, each recorded with its numbers in `docs/decisions-log.md`, may that level dispatch to the fallback (`backend_for` reports `fallback` there), with an open item in `docs/open-items.md`.
- The fallback is written so that it vectorises (plain loops over the stage halves), so that the D321 floor is a fair one.
- Instruction estimates per element and stage (to be confirmed): `C32` about 1 vector operation per 2 or 4 elements plus loads and stores; `CQ15` `Unscaled` about the same per 4 to 16 elements; `Fixed` about 2 to 4 operations more per vector pair on x86 and `simd128`, and none more on NEON for the sum.

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model). Cycles are the speed measure and the only one on the out-of-order levels (M13.1, D2700); instruction counts (`cargo xtask count --platform wasm32 --merge`) stand only where no core model exists (`simd128`, Cortex-M). A library reference enters only where it does the same work (D32); otherwise the rows are information. The D321 gate applies to every SIMD path.

**Reference rows (D32).**
- **`C32` on x86: FFHT** (FALCONN project, MIT licence; `fht_float(buf, log_n)`: in place, unnormalised, natural order). It is pinned to a commit in `benches/refs`, logged in the decisions log.
  - FFHT transforms one real array. Our transform per component is FFHT on the planar re and im arrays (two calls of `N` floats): the same butterflies and the same work. The row is marked "layout difference (planar input)", and the deinterleave is not counted.
  - FFHT on the interleaved `2N` floats is not the same transform. `H_{2N}` on the interleaved array equals `(H_N ⊗ I_2)(I_N ⊗ H_2)`: an extra first stage that mixes the re and im of each element, then the per-component transform. It does more work and gives another result, so it is not used.
  - Levels: the SSE build of FFHT at `sse2` and `sse4.2`, the AVX build at `avx2` and `avx512`. At `avx512` the row is marked "reference at a lower level" (to be confirmed: the instruction sets the FFHT builds need).
  - FFHT computes the same pairwise butterflies, so its output is expected to equal ours bit for bit per component. The harness compares the outputs once and logs the result.
  - The fair rows are the SIMD entries `fwht_c32_1024` and `fwht_c32_65536`.
- **No library row elsewhere:** `C32` at `neon` and `simd128`, every `CQ15` entry and every fallback. liquid-dsp, GNU Radio, VOLK and CMSIS-DSP have no Walsh–Hadamard transform. These entries are judged by their efficiency (D33) and against their own fallback (the D321 floor). The `fft` of the same length is information.

### 6. Block and kind `fwht`

`FwhtBlock<T>` (`C32`/`CQ15` → same) around the kernel `fwht`: the stream is cut into frames of `size` samples counted from the first sample after construction or `reset`, and every frame is transformed in place (a copy to the output, then the kernel on the output slice).

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `fwht` | `FwhtBlock<T>` | C32/CQ15 → same | `size` (I64, build, required; a power of two 2 … 65536), `scaling` (Enum, build, `none`; `none` the forward `Unscaled`, `fixed` the inverse `Fixed` divided by `size`) | `fwht`; a call takes whole frames limited by input and output room, `min_input` `size` |

- `FwhtConfig`; no storage. Timing ratio 1, delay 0 (a frame transform, not a filter; output frame `k` is the transform of input frame `k`; logged). A lane variant for the OFDM chain comes with `freq_spread`.
- `BlockFloat` is not offered by the block: the exponent of each frame would have no place in a sample stream. It comes with `freq_spread` on lanes.
- The parameter `scaling` replaces the earlier `inverse` (Bool). Nothing has been built yet, so there is no compatibility alias.
- Pipeline kind `fwht` (`c32` or `cq15`), cost `log2(size)` per input sample (one add or subtract per stage; `fixed` adds the rounding for `CQ15` and the `1/N` for `C32`), a row in `docs/pipeline.md` and `docs/timing.md`. Rule 6 of the names: `fwht` is an established abbreviation; the summary writes it out once.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 760 … 769.

1. **Golden cases** of §4 through the facade, `Fallback` and `Reference` (`sdrk-testkit/tests/golden_m56.rs`): bit-exact against the model, the exponent equal to the stored one, and the guard of each case. For `C32` the SNR against an `f64` matrix product of the stored input must be at least `snr_db_guard` and below it plus 3 dB, so the guard cannot become vacuous (as D115). For `CQ15` the error against the exact integer matrix product divided by `2^E` must be at most `max_abs_err_guard`.
2. **Unit tests** (`sdrk-core/tests/kernels_m56.rs`):
   - errors: non-power-of-two and out-of-range lengths are `InvalidParameter`, and `BlockFloat` with `C32` is `Unsupported`; in each case the data is unchanged;
   - `N = 2` is one butterfly `[a+b, a−b]` (`Fixed` in `CQ15`: `[(a+b+1)>>1, (a−b+1)>>1]` saturated, so `(32767, −32768)` gives `(0, 32767)`);
   - `X[0]` is the sum of the input;
   - `Fallback` equals `Reference` bit for bit for `N` = 2…1024, every scaling and both types, including saturating `CQ15` inputs;
   - the matrix guard for `N` = 2…1024: the `C32` SNR; `CQ15` `Unscaled` exact without saturation, `Fixed` within `log2(N)/2` LSB, `BlockFloat` within the bound of §1;
   - the exponents: 0, `log2 N` and the sum of the shifts;
   - orthogonality: the forward of a unit vector `e_k` is the `k`-th Walsh function (±1);
   - the round trip: `C32` within SNR; `CQ15` `Unscaled` then `Fixed` exact for every input whose forward does not saturate;
   - the review's case: one spread symbol of 4000 at `N = 64` returns 4000 (the old design gave 512);
   - `Fixed` on full-scale random input within the bound;
   - `BlockFloat`: no saturation except the `a − b = 65535` corner; the threshold values 16383 and 16384;
   - the SymPy vectors of §1: `fwht` with `Unscaled` and `ifwht` with `Fixed`, for both types;
   - `CQ15` saturation on a full-scale `Unscaled` forward.
3. **Suite cases** `fwht_c32`, `fwht_cq15`, `extremes_m56`: silence, full scale, NaN/∞ for `C32`, every power-of-two length to 1024, every scaling.
4. (No streaming state, so no chunk-invariance test; the transform is whole-vector.)
5. **Block and kind** (`sdrk-blocks/tests/fwht_blocks.rs`, `sdrk-pipeline/tests/fwht_kinds.rs`; seeds 765, 766):
   - the block equals the kernel frame by frame bit for bit, on whole buffers and in 20 chunk plans that cut frames, for both types and both values of `scaling`;
   - a `none` block followed by a `fixed` block returns the input (`C32` within SNR; `CQ15` exactly for input whose forward does not saturate);
   - the description rules of M09, no storage, the struct at most 512 bytes, validation errors at `params.size` and `params.scaling`;
   - a pipeline flow equals the block;
   - `cargo xtask schema --check` and the WASM kind list include `fwht`.
6. **SIMD** (`sdrk-backend-simd/tests/fwht.rs`, seed 769; pattern of M48 test 9). Every level that has a path equals `Fallback` bit for bit (NaN by class, D296), for both types and every scaling:
   - every power of two from 2 to 65536, so that every small-stage shuffle, every fused group and the cache-blocking seam (`N` below, at and above `B`) are covered;
   - the lengths 0 to 67 and 4095: the valid ones as above, the invalid ones give the same error as the fallback with the data unchanged;
   - special values: `CQ15` full scale, −32768 everywhere, the pairs `(32767, −32768)`, and the `BlockFloat` thresholds ±16383, ±16384; `C32` ±0, ±∞, NaN and subnormals;
   - the returned exponent equals the fallback's;
   - the golden cases of §4 through each level.

## Acceptance

R1.2 commands (`cargo xtask check` includes `cargo xtask check cycles`, D321); `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-backend-simd --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform wasm32` (the `simd128` instructions of the entries of §5 and `fwht_block`); `cargo measure` (the same entries, every level with a core model, with the FFHT rows at the x86 levels).

## Deliverables checklist

- [ ] Kernel `fwht` (`C32`, `CQ15`): trait, `Scaling`, `FftScale` result, Fallback, Reference (butterfly model, D111 rule), facade delegation, `backend_for`
- [ ] Contract rows and section in `docs/kernels.md`: Sylvester ordering, the scalings and exponents, the full text of shared conventions §C.2, the schedule freedom of §2, the bounds of §1, the matrix guard and the SciPy/SymPy cross-check
- [ ] SIMD paths of §5 on every level, both types, every scaling (each passing D321); `docs/backends.md`
- [ ] Golden cases (seeds 760–764, 767, 768) and the Python model with the matrix guard
- [ ] Tests 1–3, 5 and 6 (seeds 760–769); suite cases
- [ ] Count and cycle entries of §5 (`cargo measure`, `cargo xtask count --platform wasm32`) with their fallback rows and the FFHT rows (pinned in `benches/refs`, licence noted); `docs/performance.md`
- [ ] Block `FwhtBlock<T>` and kind `fwht` (§6): `docs/blocks.md` (entry and a new reserved-name row), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `fwht_block`
- [ ] `BACKLOG.md`: the `fwht` paths for the embedded backends of M16 to M18 (§Scope) and the orthonormal mode as an option for `freq_spread` (both entered with the 2026-11-17 revision) still match the Scope text
- [ ] CHANGELOG; decisions log: the ordering choice; the scalings (`Scaling` instead of `inverse: bool`, §C.2 taken over from M07, one bit per stage for `BlockFloat`); the Reference by the D111 rule with the matrix guard; the lossless `CQ15` round trip and the `Fixed` bound; the block's `scaling` parameter without `BlockFloat`; the FFHT row and its marking; the cache block size per level; the SIMD levels that stay on the fallback, with their numbers

## Status

Planned (revised 2026-11-17). Nothing is implemented.
