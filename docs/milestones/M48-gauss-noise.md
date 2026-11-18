# M48 — Kernel `gauss_noise`: deterministic Gaussian noise

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M38](M38-symbol-sync.md) · Next: [M49](M49-scramble.md)

> **Status: implemented** (2026-11-17, branch `m48-gauss-noise`, decisions D2000 to D2043, see [Status](#status)); open items I48-1 and I48-2 wait for the maintainer. Written 2026-11-08, revised and approved by the maintainer 2026-11-17 after the maintainer decision to replace the f64 Box–Muller method by an inverse-CDF method that works in the target data types; decision numbers D2000 to D2049. First of the kernel milestones M48 to M56 for spread-spectrum and multicarrier work (DSSS and OFDM). Each of them delivers one kernel with its primitive block and pipeline kind (as M29 did for the kernels before it); the composite blocks and the flows come in later milestones.

## Goal

A kernel `gauss_noise` that fills a buffer with white Gaussian noise of a given standard deviation, for `C32`, `CQ15`, `f32` and `Q15`, bit-exact between `Fallback`, `Reference` and every SIMD path, from a seed, with random access to any sample index. Each real noise value costs **one** 32-bit PCG32 draw and about 15 integer operations: an inverse normal CDF read from a 4 KiB table of octave segments and interpolated linearly in Q15 arithmetic. There is no `f64`, no transcendental function and no rejection per value, so the kernel is fast on Cortex-M (Cortex-M4F has a single-precision FPU only, Cortex-M0 none), maps onto integer SIMD on every level, and gives the same bits on every platform by construction.

## Why

Today noise exists only in the testkit (`channel::Awgn`, feature `std`, `Complex64`). A flow cannot add noise of a known level, a microcontroller cannot generate it, and there is no reference noise source for measurements (noise figure, detector thresholds, BER curves in the browser). Spread-spectrum and OFDM experiments need noise inside the flow: as a calibrated reference, as additive channel noise in loop flows, and as test input of every later kernel of this series.

The first draft copied the testkit's Box–Muller method (`f64` `log`, `sqrt`, `cos`, four draws per value). That is exact but expensive: on Cortex-M4F every value runs soft-float `f64` `libm` (thousands of cycles), and the SIMD path had to replicate the `libm` polynomials to keep the bits (WIP commit `c3f7836`: about 160 to 195 instructions per value). The inverse CDF reaches the same goal with integer arithmetic: the uniform draw is mapped to `-Φ⁻¹(q)` by a table, which suits `Q15` directly and needs one `i32 → f32` conversion and one `f32` multiply for `f32` output. Ziggurat, the usual fast method, rejects values, so the number of draws per value is not fixed; that breaks `seek` in `O(log n)` and lane-parallel SIMD. The central-limit and Wallace methods have poor tails or no random access. The inverse CDF is the one common method that meets every requirement at once. The price is a bounded table error (5.6·10⁻⁵ in units of sigma) and a tail limit of 6.338 sigma, both part of the contract (§2, §3).

## Order

M48 depends on M38 (the testkit channel model `Awgn` that test 4 compares against statistically) and through it on M37. It has no dependency on M39 to M47 and can run in parallel with every milestone after M38. The decision numbers of this milestone are D2000 to D2049.

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

**In:** kernel `gauss_noise` (`C32`, `CQ15`, `f32`, `Q15`) in a new module `sdrk_core::kernels::noise` with `GaussNoiseState`, Fallback, Reference, facade delegation, contract row and section in `docs/kernels.md`; the inverse-CDF table `GAUSS_ICDF` generated by `sdrk-core/build.rs` (§2); SIMD paths in `sdrk-backend-simd` on every level where SIMD is algorithmically feasible (maintainer rule 2026-11-17), each at least as fast as the fallback in simulated cycles (D321), bit-identical to the fallback (§5); golden cases including the independent table case; unit, statistics, invariance, extreme and SIMD tests; suite cases; `cargo xtask count` and `cargo measure` entries with the reference rows of §5 (including the new transcription `benches/refs/gr-fastnoise.c`); `CHANGELOG.md`, `docs/decisions-log.md`, `docs/performance.md`, `docs/backends.md`; the block `NoiseBlock<T>`, `NoiseRealBlock<T>` and the pipeline kind `noise` (§6) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** flows that add noise to a signal (an `add` of the signal and a `noise` node on the same clock is enough; examples later); coloured noise, phase noise, impulsive noise; a two-draw variant with a longer tail (documented upgrade path, §2.6, not implemented); Ziggurat or other rejection methods; the planned `arm-dsp` and `rvv` backends (§5 only notes how the kernel maps onto them); changing `sdrk_testkit::rng::Pcg32` or `channel::Awgn` (they stay as they are; the existing BER tests keep using `Awgn`, test 4 compares the two statistically).

## Specification

General rules of M19 apply.

### 1. Generator and draws

The generator is PCG32 (PCG-XSH-RR 64/32, `pcg_basic.c` of the PCG reference implementation), exactly as `sdrk_testkit::rng::Pcg32`:

- `MULT = 6364136223846793005`; `new(seed, stream)`: `state = 0`, `inc = (stream << 1) | 1`, one step, `state += seed` (wrapping), one step. A step is `state = state * MULT + inc` (wrapping `u64`).
- `next_u32`: `old = state`, step, `xorshifted = (((old >> 18) ^ old) >> 27) as u32`, `rot = (old >> 59) as u32`, result `xorshifted.rotate_right(rot)`.
- `advance(delta)`: skips `delta` draws in `O(log delta)` (`pcg_advance_lcg_64`, the algorithm of `Pcg32::advance`), wrapping modulo 2^64.
- `next_u64`, `next_f64` and `normal()` of the testkit are **not** part of the kernel.

The kernel has its own copy of these functions in `sdrk-core` (the core cannot depend on the testkit, R2). R4.4 forbids randomness in kernels; this is a seeded, deterministic sequence held in caller-visible state, not an entropy source. Log this reading of R4.4 as a decision.

- **One draw per real value.** Every real noise value consumes exactly one `next_u32` draw `r`; there is no rejection and no data-dependent draw count. A real sample (`f32`, `Q15`) takes 1 draw, a complex sample (`C32`, `CQ15`) takes 2: the real part from draw `2k`, the imaginary part from draw `2k + 1`. In memory a complex buffer of `n` samples therefore holds exactly the `2n` real values in draw order (re, im, re, im, ...), and the implementation may run the real kernel on the `bytemuck` cast `[f32]` / `[i16]` of length `2n` (one inner loop per output format).
- `seek(n)` = `new(seed, stream)` followed by `advance(n · d)` with `d = 1` (real) or `2` (complex), computed wrapping. Sample `n` depends only on the seed, the stream, the sample type and `n`. The period is 2^64 draws, so for complex types the samples `n` and `n + 2^63` coincide (documented in the rustdoc).
- **Interleaved chains (allowed, same bits).** The fallback and the SIMD paths may run `k` independent chains for instruction-level parallelism: chain `i` (`0 ≤ i < k`) starts at the state stepped `i` times and moves with the jump `s' = A_k · s + C_k` (`A_k = MULT^k`, `C_k = inc · (MULT^(k−1) + … + 1)` mod 2^64, computed per call with the `advance` algorithm in `O(log k)`). The values are written in draw order, and at the end of the call the state equals the start state advanced by the number of draws of the call, so a call of any implementation leaves the same state.

### 2. Inverse CDF and its table

#### 2.1 From a draw to an integer normal value

For a draw `r: u32` the kernel computes a signed integer `zq` (the normal value in Q17, `|zq| < 2^20`):

```text
sign  = r >> 31                         // 1 = negative
x     = (r << 1) | 1                    // u32, odd, 1 ≤ x ≤ 2^32 − 1; = 2·(r & 0x7FFF_FFFF) + 1
c     = x.leading_zeros()               // 0 ..= 31
L     = 31 − c                          // octave: x in [2^L, 2^(L+1))
n     = x << c                          // leading one at bit 31
sub   = (n >> 26) & 31                  // 5 bits after the leading one: 32 segments per octave
frac  = (n >> 11) & 0x7FFF              // next 15 bits: position in the segment, Q15 in [0, 1)
e     = GAUSS_ICDF[(L << 5) | sub]      // u32 entry, see 2.2
base  = e >> 12                         // Q17 knot value (20 bits)
slope = e & 0xFFF                       // knot difference to the next knot (12 bits)
mag   = base − ((slope · frac + (1 << 14)) >> 15)    // u32; slope · frac < 2^27
zq    = if sign == 1 { −(mag as i32) } else { mag as i32 }
```

Meaning: `q = x / 2^33` lies in `(0, 1/2)`; it is the midpoint of one of `2^31` equiprobable cells, so no end point (`q = 0` or `q = 1/2`) is ever reached and no branch is needed. The ideal magnitude is `-Φ⁻¹(q)` (`Φ⁻¹` = `scipy.special.ndtri`); `zq · 2^-17` approximates it within the table error of §2.4. The sign bit makes the distribution exactly symmetric: `zq(r ^ 0x8000_0000) = −zq(r)` for every `r`, and `mag = 0` gives the integer 0 for either sign, so no output is ever `−0.0`. Implementations may apply the sign branch-free (`s = (r as i32) >> 31; zq = (mag ^ s) − s`).

The interpolation step is exactly the R7 Q15 multiply of a positive 12-bit slope and a Q15 fraction (round half up; saturation cannot occur). It maps one to one onto `pmulhrsw`/`vpmulhrsw`, NEON `vqrdmulhq_s16`, WASM `i16x8.q15mulr_sat_s`, RVV `vsmul` (`vxrm = rnu`) and Cortex-M4 `SMLABB` + shift.

#### 2.2 Table

```text
K[L][j] = −Φ⁻¹((32 + j) · 2^(L − 38))      for L = 0 ..= 31, j = 0 ..= 32   (q = (1 + j/32) · 2^L / 2^33)
K[31][32] = 0                               (q = 1/2)
B[L][j] = floor(K[L][j] · 2^17 + 0.5)       (Q17, round half up)
GAUSS_ICDF[(L << 5) | j] = (B[L][j] << 12) | (B[L][j] − B[L][j + 1])     for j = 0 ..= 31
```

`B[L][32]` and `B[L + 1][0]` are the same knot, so the interpolant is continuous across segment and octave boundaries and non-increasing in `x`; at `frac = 32767` the rounded product equals the slope (`slope < 2^14`), so the interpolant reaches the next knot, which the next cell also starts at (corrected 2026-11-17, D2019). Measured properties (asserted by the generator, §2.3): `B[0][0] = 830729` (20 bits; `830729 · 2^-17 = 6.337959`), every slope lies in `[323, 3197]` (12 bits), every entry is below 2^32. Size: 1024 entries × 4 B = **4096 B**, one `static`, always present (not behind `static-twiddles`).

#### 2.3 Generation (R6.1) and independent check (R8)

- `sdrk-core/build.rs` gains `emit_gauss_icdf()`, which writes `pub static GAUSS_ICDF: [u32; 1024]` into the generated `tables.rs` next to `COS_F32` and `COS_Q15`. It computes `Φ⁻¹` with Wichura's AS241 (`PPND16`, relative error about 1e-16) in `f64` with `libm::log` and `libm::sqrt` (`libm` is already a build dependency), rounds as in §2.2 and packs the entries.
- The build script asserts and fails the build otherwise: continuity (`B[L][32] = B[L + 1][0]`), `B[0][0] = 830729`, base non-increasing, `1 ≤ slope < 4096`, the last knot is 0, and a **rounding-tie guard**: every `K · 2^17` lies at least 1e-4 LSB away from a `.5` boundary (measured minimum 4.94e-4 LSB, about 3.8e-9 absolute; AS241 and `scipy.special.ndtri` differ by at most 3.6e-15 on the 1056 knots). Any correct `f64` inverse normal therefore gives the same integers.
- R8: the golden generator builds the same table independently from `scipy.special.ndtri` (NumPy `int64`, its own tie guard) and stores it in the case `gauss_noise/icdf_table` (§4). A Rust test compares the static with it bit for bit. This test is mandatory: it is the only check of the AS241 transcription.
- Access: `kernels::noise::gauss_icdf_table() -> &'static [u32; 1024]` and `kernels::noise::normal_q17(r: u32) -> i32` (the computation of §2.1) are public and documented "for backend crates and tests", so `sdrk-backend-simd` uses the same data and the same scalar code for its remainders.

#### 2.4 Accuracy against `Φ⁻¹` (measured with NumPy/SciPy, to be reproduced by the golden generator)

| Quantity | Value |
| --- | --- |
| largest `\|zq · 2^-17 − (−Φ⁻¹(q))\|` (signed magnitude error) | 5.62e-5 (measured 5.6196e-5 near `\|z\|` ≈ 1.141; about 0.05/32² in every octave; corrected from 5.61e-5, D2018) |
| RMS error | 1.91e-5 |
| SNR of the `f32` output against the ideal `sigma · Φ⁻¹` value | 94.4 dB (the f32 optimum of one rounding is about 150 dB) |
| exact variance of the generator (enumerated over all 2^31 cells) | 1.0000279 (bias +2.8e-5, +0.00012 dB) |
| exact kurtosis | 2.999938 |
| largest `\|z\|` | 6.337959 (`−Φ⁻¹(2^-33)` = 6.337958) |
| systematic CDF error (KS distance of the exact distribution to `Φ`) | ≤ φ(z) · 5.6e-5 ≤ 2.3e-5 (to be confirmed by the enumeration) |
| output lattice | 2^-17 sigma (quantization about −113 dB relative to sigma²) |

The variance bias is positive because the chords lie above the convex half inverse CDF. It is documented, not compensated (a per-segment offset would remove it but break continuity; open question in the status section).

#### 2.5 Tail limit

`|z| ≤ 6.337959 sigma`. The truncated two-sided mass is `P(|N(0,1)| > 6.338) = 2.33e-10`. The far tail is discrete: one cell carries 2^-32 of probability, the step between neighbouring values at `|z| = 6` is about 0.04 sigma, and the generated one-sided tail probability `P(Z > t)` differs from `Φ(−t)` by at most `2^-33 ≈ 1.16e-10` (plus a negligible `φ(t) · 5.6e-5`). For a BPSK BER this is up to about 12 % at 1e-9, about 1.2 % at 1e-8 and 0.1 % at 1e-7. The contract, `docs/kernels.md` and the block rustdoc state: **not for BER or false-alarm studies below about 1e-8**.

#### 2.6 Documented upgrade path (not implemented)

A two-draw variant (64-bit quantile, 63 octaves) reaches about 9.1 sigma and keeps BER unbiased to about 1e-17, at twice the generator cost and a larger table. It would be a new kernel or an explicit option with its own golden cases, never a silent change of this contract (every bit and every golden vector would change). Logged as a decision.

### 3. State, trait, output formats and contract

```rust
pub struct GaussNoiseState { /* rng, seed, stream, sigma: f32, index: u64, sample type of the last call,
                                 derived scale: s17: f32, ms: u32, sh: u32 */ }
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

- **Sample type of the state.** The state does not know the sample type in advance: it stores the index and the draws per sample of the last call; a call with another sample type than the previous one is `InvalidState` (the first call fixes it, `reset` and `seek` clear it). Log the choice if a cleaner form is found (for example a type parameter on the state).
- **The exact noise value.** With `zq` from §2.1 and `sigma_e = num::flush_denormal(sigma)` (0 for `sigma < 1e-30`, R6.3; the getter `sigma()` still returns the value that was set): `v = zq · 2^-17 · sigma_e`. `v` is exactly representable in `f64` (20 + 24 significant bits). Each output format rounds `v` **once**:
  - `f32`: `v` rounded to nearest-even `f32`. Implementation: `(zq as f32) * s17` with `s17 = sigma_e · 2^-17` computed once in `new`/`set_sigma`. `zq as f32` is exact (`|zq| < 2^24`) and `s17` is exact (power-of-two scaling of a normal `f32`; the flush keeps `s17` and every product normal, so no flush-to-zero mode can change a bit). The one IEEE multiply is the one rounding; no FMA (R6.5). `C32`: the same per component.
  - `Q15`: R7 applied to `v`: `sat16(floor(v · 32768 + 0.5))`, computed in integers. Write `sigma_e = Ms · 2^Es` with `Ms` the 24-bit significand (implicit bit included) and `Es` the exponent from the `f32` bits; then `v · 32768 = zq · Ms · 2^-sh` with `sh = 2 − Es` (`sh ≥ 21` because `sigma ≤ 16`). With `a = |zq| · Ms` (`< 2^44`, `u64`) and `half = 2^(sh − 1)`: positive `zq` gives `(a + half) >> sh`, negative `zq` gives `−((a + half − 1) >> sh)` (this is `floor(−a/2^sh + 1/2)`), then the result is clamped to `[-32768, 32767]`. For `sh ≥ 45` every output is 0 (`a + half < 2^sh`); `new`/`set_sigma` then store `ms = 0, sh = 21`, so no shift reaches 64. `sigma_e = 0` also gives `ms = 0`. `CQ15`: per component. Values beyond full scale saturate; the clipping is part of the contract (`sigma` 0.25: about 6.3e-5 of the values clip; `sigma` 0.6: 9.56e-2).
  - Consequence for the accuracy against the ideal `sigma · Φ⁻¹(q)`: `f32` within `5.62e-5 · sigma` plus half an ulp; `Q15` within `0.5 + 5.62e-5 · sigma · 32768` LSB, i.e. up to 2 LSB from the ideal rounding for `sigma ≥ 0.6` (measured share of values that differ from the ideal rounding: 16 % at `sigma` 0.25, 29 % at 0.6, 37 % at 1.0). The contract row states both: one rounding of the defined value `v`, and the table error of `v` against `Φ⁻¹`.
- `sigma = 0` (or below 1e-30) gives `+0.0` (respectively 0) in every sample and still advances the generator (the implementation may call `advance(len · d)` instead of drawing), so switching the level at run time keeps the sequence aligned to the sample index.
- An empty output is allowed and changes nothing.
- `set_sigma` at run time takes effect with the next call and recomputes `s17`, `ms`, `sh` (the blocks apply runtime parameters between calls).

**Power convention.** `sigma` is per real component, as in `channel::Awgn` (`N0 = 2 sigma²` per complex sample). Comparison with established libraries (cross-check of this milestone, sources in the decisions log): GNU Radio's complex `noise_source_c` with amplitude `A` multiplies `gr::random::rayleigh_complex()` (variance 1 per component) by `A / sqrt(2)`, so its total power is `A²` and its `sigma` is `A / sqrt 2`; liquid-dsp's `crandnf` has variance 1 per component and its `cawgn` scales by `nstd / sqrt 2`. The rustdoc of `GaussNoiseState::new` states these conversions and the tail limit of §2.5. GNU Radio uses the xoroshiro128+ generator and liquid-dsp the sine branch of Box–Muller, so their sequences are not ours; only the statistics are compared (test 3).

**Contract row** (`docs/kernels.md`):

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `gauss_noise` | `kernels::noise::GaussNoise` | `C32`, `CQ15`, `f32`, `Q15` | bit-exact | PCG32, one draw per real value (complex: re then im); inverse CDF of §2 from the 1024-entry Q17 table `GAUSS_ICDF` (AS241 in `build.rs`, checked against `scipy.special.ndtri`); exact value `v = zq · 2^-17 · sigma` (sigma below 1e-30 counts as 0); `f32` = `v` rounded once to nearest, `Q15` = R7 of `v` (round half up, saturate); max `\|v/sigma − Φ⁻¹(q)\|` 5.62e-5, `\|z\| ≤ 6.338`; not for BER below about 1e-8; fallback, reference and SIMD equal bit for bit; golden cases exact |

`Reference` is the plain loop of §1 to §3, one value at a time, written with `i64` arithmetic; its `Q15` output is `q15_from_f64((zq as f64) * 2^-17 * (sigma_e as f64))` and its `f32` output is `((zq as f64) * 2^-17 * (sigma_e as f64)) as f32`, so it checks the integer `Q15` path and the `f32` scale independently of the fallback. `Fallback` uses the integer path of §3, may interleave chains (§1) and gives the same bits. `backend_for("gauss_noise", ..)` reports the level that runs it (§5).

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/gauss_noise.py`, written independently from the Rust code (R8.1). The Python model implements PCG32 with Python integers masked to 64 bits (`next_u32`, `advance`), builds the table of §2.2 from `scipy.special.ndtri` in NumPy `int64` with its own tie guard, and runs §2.1 and §3 in integers (`clz` via `int.bit_length`). `f32` output: `numpy.float32(zq * 2.0**-17 * float(numpy.float32(sigma)))` (exact `f64` product, one rounding). `Q15` output: the integer formula of §3; the generator asserts that it equals `q15.from_float` of the exact `f64` value `v` for every value. No `math` library call touches a sample value, so every case is `bit_exact`.

| Case | Seed (case) | Generator seed, stream | Type | `sigma` | Length | Start index |
| --- | --- | --- | --- | --- | --- | --- |
| `gauss_noise/c32_unit_4096` | 680 | 680, 1 | C32 | 1.0 | 4096 | 0 |
| `gauss_noise/c32_seek_4096` | 681 | 681, 7 | C32 | 0.3 | 4096 | 1 000 000 007 |
| `gauss_noise/f32_unit_4096` | 682 | 682, 1 | f32 | 1.0 | 4096 | 0 |
| `gauss_noise/cq15_quarter_4096` | 683 | 683, 2 | CQ15 | 0.25 | 4096 | 0 |
| `gauss_noise/q15_clip_4096` | 684 | 684, 3 | Q15 | 0.6 | 4096 | 0 |
| `gauss_noise/icdf_table` | — (no random input) | — | `u32` | — | 1024 | — |

- Each sample case stores the bit-exact `expected` array (`cf32`/`f32`, `cq15`/`q15`) and an `ideal` array: `sigma · sign · (−Φ⁻¹(x / 2^33))` in `f64` (`sigma` = the `f32` value as `f64`), with `max_abs_err_ideal` in the JSON: `5.7e-5 · sigma + 2.5e-7 · sigma` for `f32`/`C32`, `0.5 + 5.7e-5 · sigma · 32768` LSB for `Q15`/`CQ15` (compared only where the ideal value lies inside full scale). The test reports the SNR against `ideal` (step 1 of the accuracy policy) and checks a regression guard `snr_db_guard` = measured value − 1 dB (step 3), both stored in the JSON. Step 2 does not apply: no established library computes the same values from the same uniform draws.
- `gauss_noise/icdf_table` stores `entries` (`u32[1024]`, bit-exact against `gauss_icdf_table()`) and the contract numbers, computed by the generator by exact enumeration over all 2^31 magnitude cells (per segment and `frac`, with the count of cells per value): `max_abs_err_vs_ndtri` (evaluated at the first and the last cell of every `(segment, frac)` group; asserted ≤ 5.7e-5), `zmax` (= 830729 · 2^-17), `variance`, `kurtosis`, `ks_vs_phi`, and the exact one-sided tail probabilities `P(Z > t)` with `Φ(−t)` for `t` = 1, 2, 3, 4, 5, 5.5, 6, 6.3. These are the documented distribution of the contract.
- The generator seed and stream are stored in the case JSON (`numpy.random.default_rng(seed)` is not used: the sequence is the PCG32 model; the case seed only names the case).

### 5. Cost, SIMD plan and reference rows

**Entries.** `cargo xtask count` and `cargo measure` entries `gauss_noise_f32`, `gauss_noise_q15` and `gauss_noise_c32` (per output sample, every level), plus `noise_block` (§6). Cycles are the speed measure on every level with a core model; instruction counts stand where no core model exists (`simd128`, Cortex-M) (D32, D33, D68). Record the numbers in `docs/performance.md`.

**Estimated cost (to be confirmed by `cargo measure` and `cargo xtask count`).**

| Target | `f32` per real value | `Q15` per real value | Notes |
| --- | --- | --- | --- |
| x86-64 scalar fallback | about 31 instructions, 7–9 cycles | about 9–11 cycles | bound by the LCG chain (`imul` + `add`, 4 cycles per draw) without interleaving; 2 chains (§1) break it. The WIP Box–Muller: about 195 instructions per value |
| AVX2 (8 lanes) | 2–3 cycles | 2.5–3.5 cycles | about 62 uops plus one `vpgatherdd` per 8 values |
| AVX-512 (16 lanes) | 1–1.5 cycles | 1.5–2 cycles | `vpmullq`, `vprorvd`, `vplzcntd` |
| SSE2 / SSE4.2 | to be measured | to be measured | implemented; tuned until it passes D321 |
| NEON (Cortex-A72) | 3–4 cycles (scalar AArch64: 6–8) | 3.5–4.5 cycles | no 64-bit lane multiply |
| `simd128` | about 14 instructions (scalar about 28) | about 16 instructions | instruction count only |
| Cortex-M4F (estimate from the instruction sequence) | 30–35 cycles plus flash wait states | 38–45 cycles | `UMULL` + 2 `MLA` + `ADDS`/`ADC` for the LCG, `CLZ`, `UBFX`, `SMLABB`; `VCVT` + `VMUL` for `f32`; `UMULL` + `ADDS`/`ADC` + shift + `SSAT` for `Q15`. The WIP soft-float Box–Muller: thousands of cycles |
| Cortex-M0 (estimate) | 120–140 cycles (soft-float multiply) | 95–115 cycles | no `CLZ`, no `UMULL`: software `clz` and 64-bit products |

A complex sample costs twice a real value. On Cortex-M the random table access can add flash wait states on parts without a flash cache; placing the 4 KiB table in RAM is a linker choice of the application, documented in the rustdoc and `docs/kernels.md`.

**SIMD plan (`sdrk-backend-simd`).** All lanes run the integer formula of §2.1 and the output step of §3, so bit-identity with the fallback follows from the arithmetic; nothing emulates `libm`. The WIP module `crates/backends/sdrk-backend-simd/src/noise.rs` (the `libm` polynomial copies) is deleted and rewritten.

- *Lane-parallel PCG32:* per call `A_W`, `C_W` for the lane count `W` (§1); the lane states are the scalar state stepped `0 … W − 1` times; all lanes step with `s' = A_W · s + C_W`; at the end the state is the start state advanced by the number of draws of the call. A remainder shorter than one vector runs the scalar `normal_q17`. The state stays a plain `Pcg32`; `seek` is untouched. Optional block structure: stage 1 draws 64 values into a 256 B stack buffer, stage 2 transforms them (far below the 4 KiB stack limit).
- *AVX2:* LCG on two `ymm` of 4 `u64` states (even and odd draws), the 64-bit multiply as `vpmuludq` lo·lo + ((hi·lo + lo·hi) << 32); output via `vpsrlq`/`vpxor`, packed to 8 `u32` in draw order with `vpsllq 32` + `vpblendd 0xAA`, rotate with `vpsrlvd`/`vpsllvd(32 − rot)`/`vpor` (a count of 32 gives 0, so `rot = 0` is correct). `clz` without AVX-512CD: two exact conversions, `Lb = max(expfield(cvtdq2ps(x >> 8)) + 8, expfield(cvtdq2ps(x & 0xFF)))` (both inputs below 2^24, so exact; `cvt(0)` has exponent field 0 and `max` handles it). `n = vpsllvd(x, 158 − Lb)`, index from `Lb` and `sub` (the `−127 << 5` folded into the gather base), `vpgatherdd` (scale 4), interpolation `vpsrld 12` / `vpand 0xFFF` / `vpmulhrsw` / `vpsubd`, sign with `vpsignd(mag, r | 1)`. `f32`: `vcvtdq2ps` + `vmulps`. `Q15`: `vpmuludq` on `|zq|` (even and odd lanes), add `half` (`half − 1` for negative values), `vpsrlq` by `sh` (no `vpsraq` on AVX2, hence the magnitude form), clamp, sign, `vpackssdw`. A gather-free variant (8 scalar loads + `vpinsrd`) is measured as the alternative (issue #20); the faster one stays.
- *AVX-512 (F/CD/BW/DQ/VL):* 16 lanes, `vpmullq`, `vprorvd`, `vplzcntd` (exact `clz`), `vpgatherdd zmm`; `vpsraq`/`vpmuldq` for `Q15`.
- *SSE2 / SSE4.2:* rotate via the multiply trick (`rotr(x, r)` = low | high half of `x · 2^((32 − r) & 31)`, `pmuludq`), variable left shift as a multiply by `2^c` built from `f32` exponent bits, `clz` as for AVX2, 4 scalar table loads, sign via `xor`/`sub`. Implemented like the other levels; if it misses D321 it is tuned until it passes (a slower SIMD path is a defect of the SIMD code, not a reason to drop it); only if no form passes after measured attempts does `backend_for` report `fallback` at that level, with an open item and the measured attempts in the decisions log; the decisions log records the measured numbers.
- *NEON:* `vclzq_u32`, `vshlq_u32` for `n` and the rotate, `vqrdmulhq_s16`, `vcvtq_f32_s32` + `vmulq_f32` (not `vfma`), `vshrn_n_u64` for the xorshift narrowing, `vld1q_lane` × 4 for the table. LCG either as `vmull_u32`/`vmlaq_u32` emulation or as the hybrid "scalar `MADD` LCG + NEON transform"; `cargo measure` chooses, both give the same bits.
- *`simd128`:* `i64x2.mul` for the LCG; rotate via `i64x2.extmul_{low,high}_i32x4_u` with `2^((32 − r) & 31)`; `clz` via the two-conversion trick; shift via `i32x4.mul` by `2^c`; `i16x8.q15mulr_sat_s`; `v128.load32_lane` × 4. Judged in instructions.
- *Planned backends (notes only):* `rvv` maps directly (`vmul.vv` e64, `vnsrl`, `vsrl`/`vsll.vv`, `vror`/`vclz` with Zvbb, `vluxei32.v`, `vsmul`); `arm-dsp` (Cortex-M4) is the scalar loop above, which the fallback already compiles to.
- SIMD wherever it is algorithmically feasible (maintainer rule 2026-11-17): every level gets a path. D321 applies to every level as a quality gate: a path that is slower than its fallback on either model is improved until it passes, not removed.

**Reference rows (D32).** No established library does the same work (a seeded, seekable, cross-platform bit-exact Gaussian sequence), so there is no fair comparison; the branches are judged by their efficiency (`port bound / cycles`, [performance.md](../performance.md)) and by D321. The rows below are **information**, not a gap to close:

| Our entry | Library row | Marking | Why |
| --- | --- | --- | --- |
| `gauss_noise_f32`, `gauss_noise_c32` | liquid-dsp `randnf`, `crandnf` (`benches/refs/liquid-randn.c`, the version of the other liquid rows) | **reference does different work** | libc `rand()`, Box–Muller with `f32` `sqrtf`, `logf`, `cosf`; no random access, no bit-exact sequence across platforms. Ours is expected to be faster (to be confirmed) |
| `gauss_noise_f32`, `gauss_noise_c32` | GNU Radio `fastnoise_source_f`/`_c` (new `benches/refs/gr-fastnoise.c`, transcribed from `fastnoise_source_impl.cc` of the pinned GNU Radio version) | **reference does less work** | a xoroshiro128+ draw picks a value from a precomputed pool (default 8192 samples) and copies it: no normal mapping per sample, the output repeats pool values, its tail ends at the largest pool value (about 4 sigma), no random access. The row shows the floor of what a noise sample can cost |

GNU Radio's `noise_source` (Box–Muller via `gr::random`) is not added: the same kind of difference as liquid-dsp. An informational Ziggurat row (`rand_distr::StandardNormal` for `f32`, a dependency of the bench harness only, checked by `cargo deny`) is added, marked **reference does different work** (rejection sampling with a variable number of draws, no random access, no bit-exact contract across platforms; maintainer decision 2026-11-17).

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`). The D321 gate applies to every SIMD path. The committed `benches/limits.json` entries and results of the WIP for `gauss_noise` are regenerated; the old numbers are not comparable.

### 6. Block and kind `noise`

`NoiseBlock<T>` (`C32`/`CQ15` → same) and `NoiseRealBlock<T>` (`f32`/`Q15` → same) around the kernel `gauss_noise`. The input is the clock, as for `nco` (its values are never read, so a flow needs no source node): one noise sample per input sample, from `GaussNoiseState::new(seed, stream, sigma)`, sample `n` of the output stream = sample `n` of the kernel since construction or `reset`.

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `noise` | `NoiseBlock<T>`, `NoiseRealBlock<T>` | C32/CQ15/f32/Q15 → same | `sigma` (linear, **runtime**, 0.1; 0 … 16, per real component), `seed` (I64, build, 0; 0 … 2^53 − 1), `stream` (I64, build, 1; 0 … 2^53 − 1) | `gauss_noise`; a change of `sigma` takes effect with the next call and keeps the position (`set_sigma`); `reset` is `GaussNoiseState::reset` |

- `NoiseConfig` is the config of both. No storage. Timing ratio 1, delay 0. The rustdoc states the power convention of §3 (`N0 = 2 sigma²` per complex sample), the conversions to GNU Radio and liquid-dsp, and the tail limit of §2.5 (`|z| ≤ 6.338 sigma`, not for BER below about 1e-8).
- Pipeline kind `noise` (any of the four stream types; the output type is the input type), cost about 3 per real component and 6 for complex inputs (one PCG multiply, one interpolation multiply, one scale; to be confirmed from the measured cycles relative to `nco`; measured: 12 and 24, four times `nco`, D2041), a row in `docs/pipeline.md` and `docs/timing.md`. Additive channel noise in a flow is `add` of the signal and `noise` on the same clock.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 680 … 689.

1. **Golden cases** of §4 through the facade, `Fallback` and `Reference` (`sdrk-testkit/tests/golden_m48.rs`): every sample case bit for bit, plus the fidelity check against `ideal` (`max_abs_err_ideal`, SNR and `snr_db_guard`); `gauss_icdf_table()` equals `icdf_table.entries` bit for bit, and `zmax` of the case equals `B[0][0] · 2^-17` of the static.
2. **Unit tests** (`sdrk-core/tests/kernels_m48.rs`):
   - construction rules (`sigma` negative, NaN, infinite, above 16), `set_sigma` (position kept, scale recomputed), `index` after calls, `seek` equals generating and discarding (indices 0, 1, 4095, 2³² + 3; draws per sample 1 and 2), `reset`, an empty output, the `InvalidState` of a type change, `Fallback` against `Reference` for every type;
   - the PCG32 known-answer test of `pcg_basic` (`new(42, 54)`) and interleaved chains (`k` = 2, 4) equal the single chain, including the state after the call;
   - table properties: 1024 entries, continuity `base[i] − slope[i] = base[i + 1]` within and across octaves, the last knot 0, base non-increasing, `1 ≤ slope ≤ 4095`, `base[0] = 830729`;
   - `normal_q17` is non-increasing in `x` at `frac` 0 and 32767 of every segment and across every segment boundary; symmetry `normal_q17(r ^ 0x8000_0000) = −normal_q17(r)` for 10⁵ draws and the extremes (`r` = 0, `0x7FFF_FFFF`, `0x8000_0000`, `0xFFFF_FFFF`);
   - output formats: for 10⁵ values at `sigma` ∈ {2^-20, 1e-4, 0.25, 0.6, 1, 4, the `f32` just below 16, 16}: `f32` output = `((zq as f64) · 2^-17 · (sigma as f64)) as f32` and `Q15` output = `q15_from_f64` of the same value; the `sh ≥ 45` path (`sigma` 1e-12 gives all `Q15` zeros); `sigma` 9e-31 (flushed) and 0 give the bit pattern of `+0.0` and Q15 0 and still advance the generator.
3. **Statistics** (`sdrk-testkit/tests/gauss_noise_m48.rs`, 10⁶ samples, seed 685): for `f32` and `C32`: the mean within ±5 σ/√n, the variance within ±1 % of `sigma²`, kurtosis within 3 ± 0.03, the Kolmogorov–Smirnov distance to the normal CDF below 1.63/√n (the 1 % level), a chi-square test on 64 equiprobable bins of `Φ(z/sigma)` below the 1 % critical value (92.0 for 63 degrees of freedom), the count of `|z| > 4 sigma` within `6.334e-5 · n ± 5 √(6.334e-5 · n)`, the counts of positive and negative values within `5 √n` of each other, the normalized autocorrelation at lags 1 … 16 below 4/√n, the correlation of the real and imaginary parts below 4/√n, every output finite and `|out| ≤ 6.33796 · sigma`. For `Q15` and `CQ15`: the mean and the variance at `sigma` 0.1 (within ±5 σ/√n and ±1 %), and the clipping rate at `sigma` 0.6 within 10 % of `erfc(1/(0.6 √2))` = 9.558e-2. Deterministic tail reach (no sampling): the largest magnitude, reached at `r` = 0, equals 830729 · 2^-17 · sigma.
4. **Statistical cross-check with the testkit** (replaces the bit-for-bit comparison of the first draft; maintainer decision 2026-11-17): `channel::Awgn::new(sigma, 686)` on a zero input against `GaussNoiseState::new(686, AWGN_STREAM, sigma)` with `C32` (`sigma` 0.7, 10⁵ samples): the two-sample KS distance per component below `1.63 · √(2/n)`, the variance ratio within ±2 %, the mean difference within `5 sigma √(2/n)`, and `N0 = 2 sigma²` per complex sample within ±2 % for both. BPSK BER with `gauss_noise` (`f32`, `sigma = 1/√(2 Eb/N0)`, 2·10⁶ bits, deterministic through seed 686) at Eb/N0 = 4, 6 and 8 dB within ±4 binomial standard errors of `Q(√(2 Eb/N0))`, with at least 200 errors per point. The test repeats the constant `AWGN_STREAM` (the testkit is not changed). The rustdoc states that the flow noise and the testkit noise are different sequences with the same statistics. (Checked when this spec was revised: no existing test compares `channel::Awgn` with flow noise sample by sample; every BER test uses `Awgn` in a `ChannelChain` and stays unchanged.)
5. **Block-size invariance** (testkit helper of M03): 20 chunk plans (seed 687), every type.
6. **Extremes:** `sigma` 0, 1e-30 (not flushed) and 9e-31 (flushed), 16 (saturates most `Q15` values, `f32` stays finite and within `6.338 · 16`), the largest index `u64::MAX` for `seek` with real and complex types (wrapping; for complex types equal to the index `u64::MAX − 2^63`).
7. **Suite cases** `gauss_noise_c32`, `gauss_noise_q15` and `extremes_m48` (`Fallback` against `Reference`, the facade against `Fallback`, chunk plans).
8. **Block and kind** (`sdrk-blocks/tests/noise_blocks.rs`, `sdrk-pipeline/tests/noise_kinds.rs`; seeds 688, 689): the blocks equal the kernel bit for bit on whole buffers and in 20 chunk plans, every type; a runtime change of `sigma` between two calls equals `set_sigma` at the same sample index; the description rules of M09, exact storage (none), struct at most 512 bytes; validation errors at their parameter (`sigma` negative, NaN, above 16; `seed` and `stream` outside their range); a pipeline flow `nco` clock → `noise` equals the block; `cargo xtask schema --check` and the WASM kind list include `noise`.
9. **SIMD** (`sdrk-backend-simd/tests/noise.rs`): every level that has a path equals `Fallback` bit for bit, every type: lengths 0 … 67 and 4096, start indices 0, 1, 2³² + 3, every `sigma` of test 2, and positions found by search whose draws hit every octave `L` = 0 … 31 and both signs; the state after each call equals the fallback's; the golden cases of §4 through each level.

## Acceptance

R1.2 commands (`cargo xtask check` includes `cargo xtask check cycles`, D321); `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-backend-simd --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the three entries of §5 and `noise_block`); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [x] Module `kernels::noise`: `Pcg32`, `GaussNoiseState`, trait `GaussNoise`, `normal_q17`, `gauss_icdf_table`, Fallback, Reference, facade delegation, `backend_for`
- [x] `build.rs`: `emit_gauss_icdf` (AS241, packing, the asserts and the tie guard of §2.3)
- [x] SIMD paths of §5 on every level (each passing D321); the WIP `libm`-replicating module deleted; `docs/backends.md`
- [x] Contract row and section `gauss_noise` in `docs/kernels.md` (algorithm, table error, tail limit, the power convention and the GNU Radio and liquid-dsp conversions, the RAM placement note for Cortex-M)
- [x] Golden cases (seeds 680–684, `icdf_table`) and the Python model with the independent table and the exact distribution numbers
- [x] Tests 1–9 (seeds 680–689); suite cases
- [x] Count and cycle entries (`cargo xtask count`, `cargo measure`), `docs/performance.md`; the reference rows against liquid-dsp `randnf`/`crandnf` ("reference does different work") and GNU Radio `fastnoise_source` (`benches/refs/gr-fastnoise.c`, "reference does less work"); WIP limits and results regenerated
- [x] Blocks `NoiseBlock<T>`, `NoiseRealBlock<T>` and kind `noise` (§6): `docs/blocks.md` (entry; the reserved-name row becomes "delivered"), `docs/pipeline.md` (cost), `docs/timing.md`, schema, WASM kind list; count entry `noise_block`
- [x] CHANGELOG; decisions log (D2000–D2049): the reading of R4.4; the inverse CDF instead of Box–Muller (maintainer, 2026-11-17); one draw per real value, the midpoint grid and the sign bit; the table (32 octaves × 32 segments, Q17, packing, AS241 in `build.rs`, checked against SciPy, tie guard); the output formats (one rounding of `v`, `Q15` from the significand of `sigma`); `sigma` below 1e-30 counts as 0; the tail limit and the two-draw upgrade path; the documented variance bias; the replacement of test 4; the state's sample-type rule; complex as the real kernel on `2n` values; the pipeline cost; the marking of the reference rows; the SIMD levels that stay on the fallback, with their numbers

## Status

Implemented (2026-11-17, branch `m48-gauss-noise`, decisions D2000 to D2043; maintainer decisions D2000 to D2007). The WIP commit `c3f7836` (the superseded `f64` Box–Muller form) was reworked as planned: `Pcg32`, `GaussNoiseState`, the blocks, the kind, the WASM kind entry, the bench harness and the test skeletons are reused; the value paths, `draws_per_sample` (2/1), the golden values and the contract text are replaced; the SIMD module with its `libm` copies is deleted and rewritten. The final gate passed: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo xtask check` (cycles: 470 pairs, 0 findings), `cargo xtask bare-metal`, `generate.py --check` (950 files), `cargo xtask schema --check`, `cargo deny check`, the M48 test targets in debug and release, the doc tests of the touched crates, the SIMD test 9 at every level (SSE2 and AVX-512 in SDE, NEON in qemu-aarch64, `simd128` and wasm32 scalar in wasmtime, the fallback-only build), and the Cortex-M4 emulator quick suite (153 passed, 0 failed, 676 s).

Maintainer decisions of 2026-11-17 (logged as D2000 to D2007): the inverse CDF instead of Box–Muller (D2000); 1. 32 segments per octave, 4 KiB (D2001); 2. the tail 6.338 sigma, the two-draw variant documented only (D2002); 3. `Q15` keeps the exact R7 rounding, `SMMULR` + `SSAT` re-evaluated after the Cortex-M4 count (D2003; the count is in, open item I48-1); 4. the variance bias documented, not compensated (D2004); 5. the informational Ziggurat row (D2005); the rule "SIMD wherever algorithmically feasible; a slower SIMD path is improved, not removed" (D2006); the statistical test 4 (D2007).

**Done:**
- §1 to §3: `kernels::noise` with `Pcg32` (known-answer test, `advance`, `jump`, interleaved chains up to `MAX_CHAINS` 16), `GaussNoiseState`, `GaussScale`, `normal_q17`, `gauss_icdf_table`, `Fallback` (2 interleaved chains, the complex types as the real kernel on `2n` values), `Reference` (an independent `i64` model), facade delegation to `Fast` for all four types (D2008 to D2017); `build.rs` `emit_gauss_icdf` with AS241, the packing, the asserts and the tie guard (D2010); contract row and section in `docs/kernels.md`.
- §2.4 measured exactly by the golden generator (D2021): largest table error 5.6196e-5 at `z` ≈ 1.141 (5.62e-5 in the contract, D2018), variance 1.0000278503, kurtosis 2.999937965, KS distance 1.704e-5 (bound 2.3e-5), zmax 6.337959; the spec sentence on `frac = 32767` is corrected (D2019).
- §4 golden cases (seeds 680 to 684, `icdf_table`) with the independent Python model (PCG32 in Python integers, the table from `scipy.special.ndtri`); SNR against the ideal 87.99 to 94.52 dB (D2020).
- §5 SIMD on every level (SSE2, SSE4.2, AVX2, AVX-512, NEON, `simd128`), all four types, bit-identical, every level faster than its fallback on both models (D2026 to D2033): `f32` simulated cycles per value, SIMD / fallback, reference model: `sse2` 7.81 / 12.37, `sse4.2` 7.48 / 12.07, `avx2` 1.71 / 9.06, `avx512` 0.84 / 4.94, `neon` 8.04 / 13.78; `simd128` 33.42 / 69.04 instructions. The gather-free AVX2 forms (issue #20) and the NEON hybrid LCG were measured; the gather stays, NEON takes the hybrid LCG for `Q15`/`CQ15` only (D2031, D2032). No level dispatches to the fallback.
- §5 bench entries with the `Q15` pair for the D321 gate (D2034), the reference rows liquid-dsp (different work), GNU Radio `fastnoise_source` (less work, `benches/refs/gr-fastnoise.c`) and Ziggurat (different work) as cycle-simulated entries (D2035 to D2038), counts merged on all four platforms and the `gauss_noise` rows of the Cortex-M4 count (D2039, D2040), `docs/performance.md`, `docs/backends.md`, `docs/embedded.md`.
- §6 blocks `NoiseBlock<T>`, `NoiseRealBlock<T>`, kind `noise` with the measured cost 12 / 24 (D2041 to D2043), `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.
- Tests 1 to 9 (seeds 680 to 689) and the suite cases `gauss_noise_c32`, `gauss_noise_q15`, `extremes_m48` (D2022 to D2025).

**Not met:** nothing of the acceptance criteria that ran. The spec estimate of the scalar fallback (7 to 9 cycles per `f32` value at x86-64) is above it in the measurement (12.37 / 13.88 at `sse2`, 9.06 / 8.45 at `avx2`); the estimate was not a requirement, and the SIMD paths are within or below the estimates.

**Not done:**
- Open items for the maintainer: I48-1 (keep the exact R7 `Q15` or change to `SMMULR` + `SSAT`, from the Cortex-M4 count 72.08 against 40.34 instructions per value) and I48-2 (the stale committed `benches/results/cortex-m4.json`).
- The whole `cargo test --workspace --all-features` and the complete crate suites of the acceptance list were not run in this run (only the M48 targets and the gate above); the full Cortex-M4 emulator profile (which runs `extremes_m48` on the target) was not run.
- The fallback's chain count (2) was not measured against 1 or 4 (D2015); the results files keep their old `references` manifests until a full count (D2039); no `--branches` run and no `--cycle-counter sde` check at `avx2`.
