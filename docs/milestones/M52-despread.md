# M52 — Kernel `despread`: integrate-and-dump correlation

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M43](M43-equalizer.md), [M51](M51-spread.md) · Next: [M53](M53-code-search.md)

> **Status: planned** (written 2026-11-08; revised 2026-11-17 after an algorithm review (maintainer decision)). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

> **Revision 2026-11-17.** The shared conventions §A to §D cited in this file are defined in [CONVENTIONS-M49-M56.md](CONVENTIONS-M49-M56.md). The maintainer accepted all recommendations of the algorithm review of M49 to M56. The definitions shared by these milestones are the *shared conventions* §A to §D; each is written into `docs/kernels.md` by the milestone that introduces it, and this milestone introduces §C.1. Changes in this file:
> 1. **Error corrected: `CQ15` output scaling** (§1, §3). The old rule (`round_shift(acc, 15)` and `sat16` per symbol) clipped at every realistic `sf` (QPSK at amplitude 0.7 with `sf = 15` sums to about 10.5 and was cut to 1.0), and the `CQ15` round trip of the old test 2 could not hold. New: the output shift of §C.1, `sat16(round_shift(acc, f + shift))`, default `shift = ceil(log2 sf)`.
> 2. **Error corrected: state** (§1, §6). The chip history of up to `sf − 1` chips (up to 512 KiB for `C32` at `sf = 65536`, against "small history" in the old text) is replaced by running accumulators (eight partial sums per component for `C32`, two `i64` for `CQ15`) and the chip count in the window. Same bits, `min_input` 1, no history storage.
> 3. **Bit-packed code format** (§1, §4, §6): the code is a `Code<'_, T>` (shared conventions §B, introduced in M51); the blocks store packed bits (`prbs15`: 4 KiB instead of 256 KiB as `C32`). The code position is a running `u32` counter, no `u64` modulo per chip.
> 4. **Exact signs for binary chips** (§1, §3): a binary chip adds `±x` (shared conventions §A, introduced in M49), never a product with `32767`; the old "`±1` case is add/subtract" remark in *Out* is replaced by a contract of its own.
> 5. **SIMD in scope** (§5, test 7): every lane-parallel part gets a path on every level in this milestone, bit-identical to the fallback (shared conventions §D, maintainer rule 2026-11-17); D321 is a quality gate.
> 6. **Fair reference rows named** (§5): VOLK `volk_32fc_x2_conjugate_dot_prod_32fc`, CMSIS-DSP `arm_cmplx_dot_prod_f32`/`_q15`, liquid-dsp `dotprod_cccf`.
> 7. Smaller text fixes: the `code_track` invariant holds for finite input; `chip_index` counts consumed chips; the `CQ15` "saturation on a full-scale window" test now tests the shift; two new golden cases (seeds 726, 727) and the SIMD test (seed 728); tests, acceptance and checklist updated. Estimates are marked "(to be confirmed)".
> 8. Precision fixes found when the revision was checked: (a) **Error corrected: the `CQ15` round trip** (§2, test 2). The claim "exactly `sat16(round_shift(sf · s, shift))`" was false for `s = −32768` at every `sf`: a chip with `b = 1` carries `q15_neg_sat(−32768) = 32767`, so the exact sum is `sf · s + n1` (`n1` chips with `b = 1`); the formula now states this term, and the result for −32768 is within 1 LSB with the default shift (NumPy, the integer model: `sf = 15`, `n1 = 8` gives −30719, not −30720). (b) **Error corrected: the long-code position** (§1). With the global window index `m` the old formula `(chip_index0 + m·sf + j) mod code_len` (M51's per-call form) counted the call start twice and ignored an open window; it is now the stream index of the chip modulo `code_len`, which `code_pos = chip_index mod code_len` implements. (c) **Short code aligned with M51** (§1, tests 2 and 7): `code_pos` is not used and stays 0 for a short code, `c = j`; `seek` sets `chip_index` and starts a window. (d) **NaN in the complex `C32` form** (§1, tests 2 and 7): the equality with `dot_complex(window, conj(code_window))` holds for non-NaN results, NaN lanes are compared by class (D296), as M51 revision item 6a. (e) The golden table of §4 has its missing *Chips* cells; *Acceptance* counts the entries on all four platforms with `--merge`, as §5 *Measurement* says.

## Goal

A kernel `despread` that correlates a chip stream against a code and dumps one symbol per `sf` chips: `symbol[m] = sum_{j} input[m·sf + j] · conj(code[c])`. It is the matched receiver for `spread` (M51): with the same code, chip alignment and `sf`, despreading the output of `spread` recovers the symbols with the full processing gain. A binary chip is applied by an exact sign (no product), a general complex chip by the product with its conjugate. The `C32` summation order is the fixed order of `dot_complex` and the `CQ15` sum is exact in `i64`, so the kernel is bit-exact on every backend and level, and for finite input it is bit-identical to the prompt sum of the tracking loop `code_track` (M54) at zero loop bandwidth.

## Why

Despreading collapses `sf` chips back to one symbol and concentrates the signal power while the noise averages down, the receive half of the processing gain. It is the inner correlation of every DSSS receiver (and of the prompt sum of the tracking loop). It needs its own kernel rather than a generic sum so the order is fixed (reproducible on every backend), so the chip phase and code indexing match `spread` exactly, so binary codes cost a sign instead of a complex product, and so `CQ15` keeps the processing gain instead of clipping it.

## Order

M52 depends on M51 (the chip indexing, the code format `Code` of the shared conventions §B and, through M51, the exact sign operations of M49, §A) and on M43 (the `dot_complex` summation order it reuses). It can run in parallel with every milestone after M51.

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

**In:** kernel `despread` (`C32`, `CQ15`; both code forms of `Code`) in `sdrk_core::kernels::spread` (the module of M51) with `DespreadState`, Fallback, Reference, facade delegation, contract, `docs/kernels.md` (including the full text of the shared conventions §C.1, introduced here); SIMD paths in `sdrk-backend-simd` on every level (§5) with `docs/backends.md`; golden cases; unit, invariance, extreme and SIMD tests; suite cases; `cargo measure` and, for `simd128`, `cargo xtask count --platform wasm32`; CHANGELOG, decisions log; the block `DespreadBlock<T>` and the pipeline kind `despread` (§6) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the code-phase search (`code_search`, M53) and the tracking loop (`code_track`, M54); the composite `dsss_demod` (later milestone; name reserved); a wide `i32` output per component in the style of CMSIS-DSP `arm_cmplx_dot_prod_q15` (the review's alternative to the output shift; not chosen, the shift keeps the `CQ15` stream type).

## Specification

General rules of M19 apply.

### 1. Trait, state, indexing

```rust
pub struct DespreadState {
    /* chip_index: u64, code_pos: u32, sf: u32, code_len: u32, long: bool, shift: u32,
       in_window: u32, the accumulators of the open window (C32: [f32; 8] re and [f32; 8] im;
       CQ15: i64 re and i64 im) and the sample type of the open window */
}
impl DespreadState {
    /// Same rules as `SpreadState::new` (`sf` 2..=65536; a short code needs `code_len == sf`;
    /// `code_len >= 1`); `shift` is `default_shift(sf)`.
    pub fn new(sf: u32, code_len: u32, long: bool) -> Result<Self>;
    /// The `CQ15` output shift, 0..=16 (`InvalidParameter` otherwise); `C32` does not use it.
    pub fn with_shift(self, shift: u32) -> Result<Self>;
    /// `ceil(log2 sf)`: 2 → 1, 15 → 4, 16 → 4, 31 → 5, 1023 → 10, 65536 → 16.
    pub const fn default_shift(sf: u32) -> u32;
    pub fn shift(&self) -> u32;
    /// The `CQ15` gain `sf / 2^shift` (1 for a power-of-two `sf` with the default shift).
    pub fn cq15_gain(&self) -> f64;
    /// Chips consumed since construction, `reset` or `seek` (plus the `seek` position).
    pub fn chip_index(&self) -> u64;
    /// Chips of the open window, 0..sf.
    pub fn chips_in_window(&self) -> u32;
    /// The next consumed chip is chip `chip` of the stream and the first chip of a window.
    pub fn seek(&mut self, chip: u64);
    pub fn reset(&mut self);
}
pub trait Despread<T> {
    /// Consumes chips into the open window and dumps one symbol per completed window;
    /// output length ≤ (chips_in_window + input.len()) / sf.
    fn despread(&self, state: &mut DespreadState, input: &[T], code: Code<'_, T>, output: &mut [T]) -> Result<Progress>;
}
```

- **Code.** The code is a `Code<'_, T>` (shared conventions §B, introduced in M51): `Code::Bits` is the bit-packed binary form (chip `i` = bit `i & 31` of word `i >> 5`, chip value `1 − 2b`, chip `i` = the `i`-th `Prbs::next_bit` of a PRBS), applied by the exact sign operations of §A; `Code::Complex` is the general form with the `mul` product. `code.len()` must equal `code_len` (`LengthMismatch`, nothing consumed).
- **Window and code position.** Window `m` is the `m`-th group of `sf` consumed chips since construction, `reset` or `seek`; `j` (0 to `sf − 1`) is the index of a chip within its window. The code position `c` follows the indexing of `spread` (M51 §1), with the window index `m` counted globally (not per call, because a window may be split across calls):
  - **short code** (`long = false`, `code_len = sf`): `c = j`, the code restarts with every window. `code_pos` is not used for a short code and stays 0 in every state: `new`, `reset`, `seek` (which sets only `chip_index` and starts a window) and every call leave it 0, and the code position never depends on `chip_index` (as M51 §1).
  - **long code** (`long = true`): `c = (index of that chip in the stream) mod code_len = (p0 + m·sf + j) mod code_len`, with `p0` the `seek` position (0 after construction or `reset`) and `m` the global window index above. This is `chip_index mod code_len` at the moment the chip is consumed (`chip_index` counts consumed chips, including those of an open window, plus the `seek` position). The state keeps `code_pos = chip_index mod code_len` as a `u32` and advances it by one per consumed chip with compare-and-subtract at the period end; a `u64` modulo runs only in `new` and `seek`, never per chip (also not in `Fallback`; §B.3). `chip_index` advances with `wrapping_add`; the formula and `code_pos` agree as long as `chip_index` has not wrapped (2⁶⁴ chips, as M51 §1).
- **Terms.** Each chip `x` contributes one term `t_j`:
  - `Code::Bits`, chip bit `b`: `t_j = x` for `b = 0` and `−x` for `b = 1`. `±1` and `±j` are applied by the exact sign and swap operations of the shared conventions §A (`q15_neg_sat`, `cq15_quadrant`, `c32_quadrant`, introduced in M49), never by a product. In `CQ15` a stored negation saturates (`−(−32768) = 32767`); inside a wide accumulation it is plain integer negation. Here the term is never stored: `C32` uses IEEE negation (`c32_chip`; `acc + (−x)` and `acc − x` are the same IEEE operation), `CQ15` the plain integer negation in the wide accumulator (`−(−32768) = +32768`).
  - `Code::Complex`, chip `c`: `t_j = x · conj(c) = (xr·cr + xi·ci, xi·cr − xr·ci)`; in `C32` each of the four products is rounded to f32 before the sum or difference, no FMA (R6.5); in `CQ15` the products are exact `Q30` values in `i64`, as in `dot` `CQ15`.
- **`C32` sum (bit-exact).** The `dot_complex` order of M43 (D801): eight partial sums per component, each starting at `+0.0` at the window start; term `t_j` is added to partial sum `j mod 8`; at the dump the fixed tree `s[k] = acc[k] + acc[k+4]`, `t[k] = s[k] + s[k+2]`, result `t[0] + t[1]`. The index `j` counts from the window start, never from the start of a call, so the call boundaries do not change the bits (shared conventions §D.2). Two equalities follow; both were checked numerically when this spec was revised (20 000 random windows of 1 to 39 chips with `±0`, subnormals and values near `f32::MAX`, no mismatch):
  - for non-NaN results the complex form equals `dot_complex(window, conj(code_window))` bit for bit (`xr·cr − xi·(−ci)` and `xr·cr + xi·ci` are the same IEEE result when no operand is NaN, as are the imaginary forms); NaN lanes are compared by class (D296): on x86 `a − NaN` returns the NaN unchanged and `a + (−NaN)` returns it with the sign flipped, so the two forms may give NaNs of different sign (the correction of M51, revision item 6a);
  - for finite input the binary form equals the complex form with the code `(1 − 2b, 0)` bit for bit: the products differ from the signs only in signed zeros, and a partial sum that starts at `+0.0` never becomes `−0.0` under round-to-nearest, so those zeros do not reach the result. For `±∞` input the forms differ on purpose: a binary chip gives `±∞`, the product gives `∞ · 0 = NaN` (§A.2).
  - A NaN result is NaN on every backend and level; its sign and payload are not part of the contract (D296, as `dot_complex`).
- **`CQ15` sum and output shift (shared conventions §C.1, introduced here).** `CQ15` values that grow by a known factor are scaled by a shift with R7 rounding: `round_shift(v, s) = (v + 2^(s−1)) >> s` (round half up, `sdrk_core::num::round_shift`; `s = 0` is the identity), then saturated **once** with `sat16`. Nothing saturates before the scaling.
  - The sum over the `sf` terms is accumulated **exactly** in `i64`, per component, as the `Acc64` rule of R7: binary terms are `±x` in `Q15` units (the integer sign of §A.1), complex terms are the `Q30` products of `x · conj(c)` per component, formed as in `dot` `CQ15`.
  - Output per component: `sat16(round_shift(acc, f + shift))`, with `f = 0` for binary terms and `f = 15` for `Q30` products. The value is `(Σ terms) / 2^shift`.
  - `shift` is a parameter of the state, 0 to 16 (`InvalidParameter` otherwise), default `ceil(log2 sf)`. With the default, a binary window (any `Q15` input) and a complex window whose input and code chips have complex magnitude at most 1 (per-chip term at most 1 per component by Cauchy–Schwarz) cannot overflow; the only exception is the exact value `+32768` (for example `sf = 2` and two terms `+32768`), which saturates to 32767. With a smaller shift the caller accepts saturation.
  - The state reports the gain `sf / 2^shift` (`cq15_gain`). For a power-of-two `sf` the default gives 1, so `spread` → `despread` returns the symbols; for `sf = 15` it gives `15/16`, for `sf = 1023` it gives `1023/1024`.
  - Integer sums are exact, so every summation order gives the same bits as long as no intermediate overflows. A SIMD path may keep per-lane partial sums in `i32` only where a bound shows they cannot overflow, and widens to `i64` otherwise: binary terms have `|x| ≤ 32768`, so an `i32` lane holds 65535 terms (`65535 · 32768 = 2147450880 < 2^31`); `Q30` products have `|xr·cr + xi·ci| ≤ 2^31` and need `i64` or an exact split (§5). The `i64` accumulator holds every window: `65536 · 2^31 = 2^47`.
  - `C32` has no shift and returns the plain sum (gain `sf`).
- **Consumption (running accumulators).** The state holds the accumulators of the open window and the chip count `chips_in_window`, not a chip history. A call adds its chips to the accumulators one term at a time in window order and dumps when the count reaches `sf`: it writes the symbol, clears the accumulators to `+0.0` (`C32`) or 0 (`CQ15`) and starts the next window. A partial window carries over to the next call, so a symbol split across two or more calls dumps once, in the call that completes it, with the bits of the unsplit window (the term-to-partial assignment depends only on `j`). A chip that would complete a window when `output` is full is not consumed (the rule of `fir_decimate`); every other chip is consumed. `Progress` counts chips consumed and symbols produced; `min_input` is 1; `chip_index += chips consumed`.
- **Sample type.** The accumulators belong to one sample type: a call with the other type while `chips_in_window > 0` is `InvalidState` (nothing consumed); with an empty window a call may switch the type. `reset` and `seek` clear the window. Log the choice (a type parameter on the state is the alternative).
- **Timing.** Ratio `1/sf` (constant), delay 0: declared at the first chip of the integration window, as `spread` declares its delay at the first chip (M51 §1). The centre convention of timing.md would give `−(sf − 1)/2` here (the window lies after the output position `m · sf`), which `Delay` (unsigned) cannot express; with 0 on both sides `spread` → `despread` reports exactly the delay its samples show. Logged.
- **Storage.** None outside the state: the accumulators are 64 bytes plus the count for `C32` and 16 bytes for `CQ15`, inside `DespreadState` (no storage type parameter needed). The code is passed per call.

### 2. Relation to `code_track` and `spread`

- **Zero-bandwidth tracking.** `code_track` (M54, revised 2026-11-17: correlate first, then interpolate once per code period) forms its prompt sum from the lag-0 sum of the binary code in the order of the shared conventions §D.2. At loop bandwidth 0, on-time alignment and fractional position 0 its prompt output must equal `despread` (binary form, the same code and window) bit for bit for finite input; at non-finite input the interpolation weights `0 · S` may add NaN. This milestone states the invariant; M54 defines its exact form and tests it.
- **Round trip with `spread`.** Over a noiseless channel with a binary code, `despread(spread(symbols, code), code)` returns:
  - `C32`: `sf · symbols` up to the float accumulation error, exactly when the products of the sum are representable (for example symbols with components `±1` or `±0.5` and `sf ≤ 2^23`); the chips of `spread` are `±s` exactly (§A), so the terms are `s` exactly.
  - `CQ15`: per component exactly `sat16(round_shift(sf · s + n1 · [s = −32768], shift))`, with `n1` the number of chips with `b = 1` in the window and `[s = −32768]` 1 for that component value and 0 otherwise. The term of a chip with `b = 1` is `−q15_neg_sat(s)`, which equals `s` for every `s ≠ −32768`; for `s = −32768` the stored chip is `q15_neg_sat(−32768) = 32767` (§A.1), so the term is `−32767 = s + 1` and the sum is `sf · s + n1`. Consequences:
    - for `s ≠ −32768` the result is `sat16(round_shift(sf · s, shift))`; with a power-of-two `sf` and the default shift this is `s` itself;
    - for `s = −32768` the result is within 1 LSB of `sat16(round_shift(sf · s, shift))` whenever `n1 ≤ 2^shift`, in particular for every `sf` with the default shift (`n1 ≤ sf ≤ 2^shift`): adding at most `2^shift` before the shift changes the result by at most 1. With a power-of-two `sf` and the default shift it is −32768 or −32767 (−32767 when `n1 ≥ sf/2`). It is not exact for any `sf` once `n1` is large enough: checked numerically when this spec was revised (NumPy, the integer model), `sf = 15`, `n1 = 8`: −30719 against −30720 without the correction; `sf = 31`, `n1 = 16`: −31743 against −31744; `sf = 1023`, `n1 = 512`: −32735 against −32736. The exact formula above holds in every case; it was also checked for the power-of-two `sf` 2, 4, 16, 64 and 1024.

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `despread` | `kernels::spread::Despread` | `C32`, `Code::Bits` | bit-exact | terms `±x` by IEEE negation (§A.2), the eight-partial-sum order and tree of `dot_complex` (M43) over the window index; equals the complex form with code `(1 − 2b, 0)` for finite input; `±∞` input gives `±∞` terms; fallback, reference and every SIMD level give the same bits (NaN sign and payload excepted, D296) |
| `despread` | `kernels::spread::Despread` | `C32`, `Code::Complex` | bit-exact | `dot_complex(window, conj(code_window))`: products without FMA, the order of M43; fallback, reference and every SIMD level give the same bits (D296 for NaN) |
| `despread` | `kernels::spread::Despread` | `CQ15`, `Code::Bits` | bit-exact | exact `i64` sum of `±x` per component (integer sign, no saturation inside the sum), output `sat16(round_shift(acc, shift))`, default `shift = ceil(log2 sf)` (shared conventions §C.1) |
| `despread` | `kernels::spread::Despread` | `CQ15`, `Code::Complex` | bit-exact | exact `i64` sum of the `Q30` products of `x · conj(c)` per component, output `sat16(round_shift(acc, 15 + shift))` (§C.1) |

`Reference` follows the `C32` order step by step (the bit-exact rule of the kernels that reuse `dot_complex`) and the `CQ15` integer model; both keep the running accumulators, so chunked reference calls equal unchunked ones. An f64 regression guard (the exactly rounded sum, for `CQ15` divided by `2^shift`) is checked with `snr_db_guard`. `backend_for("despread", ..)` reports the SIMD level on every level with a path (§5), `fallback` only where a level is delegated under D321 with an open item.

`docs/kernels.md` gets the section `despread` with the full text of the shared conventions §C.1 (the canonical sentence: "`CQ15` growth follows the shared conventions §C: sums are exact in `i64` and leave through one rounding shift `round_shift(acc, f + shift)` with one saturation (default `shift = ceil(log2 sf)`, introduced in M52); transforms shift by 0 or 1 per stage with R7 rounding and one saturation, the exponent returned as `FftScale` (the M07 FFT rule, introduced for `fwht` in M56)."), the two `C32` equalities of §1, the `spread` round trip and the `code_track` invariant.

### 4. Golden vectors

New cases in `tests/golden/sdrk_golden/cases/despread.py` (the shared code helper of M51). Binary codes are stored as the array `code_words` (`u32` little-endian) and the JSON field `code_len` (shared conventions §B.2); complex codes as a `code` array. `CQ15` cases store `shift` and `gain` in the JSON.

| Case | Seed | Type | `sf` | code | Chips | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `despread/short_bpsk_c32_sf31` | 720 | C32 | 31 | short binary m-seq 31 (`Code::Bits`) | 7936 (256 symbols) | the output of `spread/short_bpsk_c32_sf31` plus noise |
| `despread/long_bpsk_c32_sf1023` | 721 | C32 | 1023 | long binary m-seq 1023 (`Code::Bits`) | 16 368 (16 symbols) | peak recovery at −10 dB chip SNR |
| `despread/short_qpsk_cq15_sf15` | 722 | CQ15 | 15 | short complex code (`Code::Complex`) | 3840 (256 symbols) | QPSK; default `shift` 4 (gain 15/16) |
| `despread/long_bpsk_cq15_sf1023` | 726 | CQ15 | 1023 | long binary `prbs11` (2047 chips, `Code::Bits`) | 16 368 (16 symbols) | default `shift` 10; the period seam falls inside windows and off the word boundaries (2047 is not a multiple of 32); some input components at −32768 and +32767 |
| `despread/short_qpsk_c32_sf15` | 727 | C32 | 15 | short complex code (`Code::Complex`) | 3840 (256 symbols) | the complex form in `C32` |

- Noise is added by the Python generator (`numpy.random.default_rng(seed)`), not by an SDRk kernel. Expected values from the `dot_complex` f32 model (`C32`) and the exact integer model of §1 (`CQ15`), the per-symbol outputs stored. Tolerance: bit-exact (`==` on raw bits) against the model; an f64 regression guard with `snr_db_guard` as for `dot_complex`.

### 5. Cost, SIMD plan and reference rows

**Entries.** `cargo measure` entries `despread_c32`, `despread_cq15` (binary code, the block's form) and `despread_c32_complex`, `despread_cq15_complex` (complex code), per input chip, every level, each with its `<k>_fallback` row (the D321 floor), plus `despread_block` (§6). Record the numbers in `docs/performance.md`.

**Estimated cost (to be confirmed by `cargo measure`).** Per chip the binary form is two additions with a sign (no multiplication), the complex form one complex multiply-accumulate (4 multiplications, 2 additions or subtractions, 2 accumulations). Estimates of the review (to be confirmed): Cortex-M4F `C32` complex about 4–6 cycles per chip (the sixteen partial sums fit in the 32 single-precision registers); `CQ15` complex about 2–3 cycles per chip (`SMLALD`/`SMLSLDX`, dual 16 × 16 into 64 bit); Cortex-M0 `CQ15` complex about 15–20 cycles per chip (the `i64` additions). The binary forms need no multiplication and are expected to be cheaper on every target (to be measured). The dump (tree or shift, once per symbol) adds a fixed cost per `sf` chips.

**SIMD plan (`sdrk-backend-simd`, module `despread`).** SIMD follows the shared conventions §D (maintainer rule 2026-11-17): every lane-parallel part of the kernel gets a path on every level in this milestone, bit-identical to the fallback through the fixed element-wise formula or reduction order of the contract; D321 is a quality gate (a slower path is improved, not removed; delegation to the fallback only after several measured failed attempts, with an open item); cycles per sample with `cargo measure` against the fallback and the fair reference (D32), unfair rows marked.

- *Lane-parallel part:* the chip terms (sign by §A or product) into the eight partial sums (`C32`) or exact integer lanes (`CQ15`). Levels: `sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`.
- *`C32` layout:* the layout of the `dot_complex` SIMD backend (kernels.md): eight complex lanes per step, partial sum `k` in lane `k`; AVX2 and AVX-512 one 256-bit vector per component (the in-lane split of `dot_complex`, lanes reordered before the tree), the 128-bit levels two vectors per component. Lane `k` holds the terms with `j mod 8 = k`, which is the contract.
- *Binary signs:* eight code bits per step are expanded to lane masks of `0x8000_0000` and XORed into both components (the conditional IEEE negation of §A.2). The bits are taken from the packed words at the cursor `code_pos >> 5`, `code_pos & 31` with a funnel shift of two words; runs end at the period seam and at window boundaries, and a run crossing the seam continues at chip 0 of word 0 (§B.3). The SIMD helpers of `sdrk-backend-simd` module `sign` (M49) are used.
- *Complex products:* the four products and the sum and difference written in the kernel, without FMA. The `mul` `C32` SIMD entry is not used (it has FMA at `avx2`, `avx512` and `neon`, D326).
- *Window start and tail:* a call that starts with `chips_in_window` not a multiple of 8 runs at most seven terms with the scalar code (the same formula) or rotates the lanes; a window end that is not a multiple of 8 is padded with `+0.0` terms, which leaves every partial sum unchanged (a partial sum is never `−0.0`, §1; `NaN + 0` and `±∞ + 0` keep their class) — checked numerically when this spec was revised. Padding never adds terms of the next window.
- *Small `sf`:* when a window is shorter than a few vectors, several windows may run side by side with one window per lane (a transposed layout); every output still evaluates the same fixed sequence of additions and the same tree, so the bits are the same. `cargo measure` chooses the threshold.
- *`CQ15` binary:* widen `i16` to `i32` (`pmovsxwd` or unpack with a sign mask, `vmovl_s16`, `i32x4.extend_*_i16x8_s`), conditional negation by `xor`/`sub` with the lane mask in `i32` (exact: −(−32768) = +32768, no saturation), add into `i32` lane sums and fold them into the `i64` accumulators at every dump, at the end of the call and at least every 65535 terms per lane (§1). Not allowed: `psignw`/`vpsignw` and saturating `i16` negation before widening (they change −32768 inside the sum).
- *`CQ15` complex:* `Q30` products widened to `i64` lanes, or `pmaddwd` (`i32x4.dot_i16x8_s`, `vmull_s16`/`vmlal_s16`) only in a form that cannot overflow `i32` (`xr·cr + xi·ci` reaches `2^31` at `(−32768, −32768) · (−32768, −32768)`), with the pair sums widened to `i64` before they accumulate. Every exact form gives the same bits.
- *Dump:* once per symbol, the tree of `dot_complex` (`C32`) or `round_shift` and `sat16` (`CQ15`) on the lanes of one window, a horizontal reduction in registers; in the window-per-lane layout it is lane-parallel across windows. It is the only part without lane parallelism across chips, at a cost of a few operations per `sf` chips.
- *State after a call* equals the fallback's (accumulators bit for bit, `chips_in_window`, `code_pos`, `chip_index`).
- *Planned backends (notes only):* `rvv` maps directly (masked `vfsgnjn`/`vxor` for the sign, `vwadd` for the widening sums); `arm-dsp` (Cortex-M4): `SMLALD`/`SMLSLDX` for the complex `CQ15` form, a conditional `RSB` and two 32-bit additions for the binary form.

**Reference rows (D32).** Fair rows for the complex form (the same work: a sum of `x · conj(c)` per symbol over a pre-conjugated or conjugating code; per input chip):

| Our entry | Library row | Platforms | Marking |
| --- | --- | --- | --- |
| `despread_c32_complex` | VOLK `volk_32fc_x2_conjugate_dot_prod_32fc` | x86-64, AArch64 | fair; its summation order differs and its dispatch may fuse (noted as in D326) |
| `despread_c32_complex` | liquid-dsp `dotprod_cccf` (code pre-conjugated once) | x86-64, AArch64 | fair |
| `despread_c32_complex`, `despread_cq15_complex` | CMSIS-DSP `arm_cmplx_dot_prod_f32`, `arm_cmplx_dot_prod_q15` (code pre-conjugated once) | Cortex-M | fair per chip; the `q15` row returns a wide `q31` result without shift or saturation, which is noted |

The binary entries `despread_c32` and `despread_cq15` are compared with the same rows as **information**, marked "reference does different work" (a complex product per chip where the binary form applies a sign). A fair gap above 10 % becomes an open item; the rows are measured, not assumed.

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) (simulated cycles per sample; cycles are the speed measure and the only one on the out-of-order levels (M13.1, D2700); instruction counts (`cargo xtask count --platform wasm32 --merge`) stand only where no core model exists (`simd128`, Cortex-M)). The D321 gate applies to every SIMD path.

### 6. Block and kind `despread`

`DespreadBlock<T>` (`C32`/`CQ15` → same, one output symbol per `sf` input chips) around the kernel `despread`, with the code of `spread` (§6 of M51: the same parameters, the same PRBS mapping). The block generates the code into packed words with `Prbs::fill_words` (`ceil(period / 32)` `u32` words, 4 KiB for `prbs15`) and passes it as `Code::Bits`; `DespreadBlock::with_code` takes a caller code in either form of `Code` (caller-provided storage, R3.2). Chip 0 of the code meets the first input sample after construction or `reset`; the block does not search (that is `code_search`, M53, and `code_track`, M54).

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `despread` | `DespreadBlock<T>` | C32/CQ15 → same | `sf`, `code`, `seed`, `long` as `spread`; `shift` (I64, build, absent = `ceil(log2 sf)`; 0 … 16; `cq15` only, `invalid_value` at `params.shift` in a `c32` flow) | `despread`; a call consumes chips limited by the output room, a partial window stays in the kernel accumulators |

- `DespreadConfig`. Storage: the packed code words; exact. Timing ratio `1/sf`. **Delay** 0 (§1: the first chip of the window, as `spread`). The rustdoc states the `CQ15` gain `sf / 2^shift`.
- Pipeline kind `despread` (`c32` or `cq15`), cost 2 per input chip (two additions with a sign; to be confirmed from the measured cycles), a row in `docs/pipeline.md` and `docs/timing.md`.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 720 … 729.

1. Golden cases of §4 (seeds 720–722, 726, 727) through facade, `Fallback`, `Reference` (`sdrk-testkit/tests/golden_m52.rs`).
2. Unit tests (`sdrk-core/tests/kernels_m52.rs`):
   - `new` rules (as `SpreadState`), `with_shift` (0 and 16 accepted, 17 rejected), `default_shift` for `sf` 2, 3, 15, 16, 31, 1023, 65536, `cq15_gain`; code length checks for both forms (`LengthMismatch`);
   - the `dot_complex` equality (complex form of one window equals `dot_complex(window, conj(code_window))`; non-NaN results by `to_bits()`, NaN lanes by class, D296) and the binary form equal to the complex form with code `(1 − 2b, 0)` for finite `C32` input including `±0` components; `±∞` input gives `±∞` in the binary form and NaN in the complex form;
   - `CQ15`: output equals `sat16(round_shift(Σ terms, f + shift))` of the integer model for both forms and every shift 0 … 16; with the default shift a full-scale binary window does not saturate except the `+32768` case (`sf = 2`, both chips −32768 with chip bit 1); with `shift = 0` a full-scale window saturates;
   - the round trip with `spread` (§2): `C32` `sf · symbols` exact for `±1` symbols; `CQ15` the exact integer formula `sat16(round_shift(sf · s + n1 · [s = −32768], shift))` of §2 for `sf` 2, 3, 4, 15, 16, 31, 64, 1023, 1024 and every shift 0 … 16, with symbols including −32768, −32767 and 32767 and codes with every `n1` from 0 to `sf` where `sf ≤ 16` (otherwise random codes plus the all-zero and all-one code); `s` itself for `s ≠ −32768` at power-of-two `sf` with the default shift; for `s = −32768` within 1 LSB of `sat16(round_shift(sf · s, shift))` with the default shift at every `sf`, including the cases `sf = 15`, `n1 = 8` (−30719), `sf = 31`, `n1 = 16` (−31743) and `sf = 1023`, `n1 = 512` (−32735);
   - a symbol split across two calls at every position 1 … `sf − 1`, and across `sf` calls of one chip, dumps once with the bits of the unsplit window; `chips_in_window` and `chip_index` after each call; a full output buffer leaves the completing chip unconsumed;
   - `seek` equals despreading and discarding the chips before the position: for a long code (positions 0, 1, `sf − 1`, a position near the period seam and 2⁴⁰ + 5; after `seek` `chip_index` equals `chip` and `code_pos` equals `chip mod code_len`) and for a short code (positions 0, 1, `sf + 3` and 2⁴⁰ + 5, also not a multiple of `sf`; `code_pos` stays 0 and the next window uses `c = j` from its first chip); a long code split into calls that end inside a window gives the code positions of the unsplit stream (`c` from the stream index, not from the call start); `reset`, the `InvalidState` of a type change with an open window.
3. Block-size invariance: 20 chunk plans at the chip level including chunks of one chip (seed 723), both types, both code forms.
4. Extremes: fewer than `sf` chips (consumed, no output, kept in the accumulators); `sf = 2`, `sf = 65536` (default shift 16); full-scale `CQ15` with every chip bit pattern; NaN/∞ `C32` in one chip (binary form: that symbol `±∞` or NaN, complex form: NaN; the next symbol recovers because the accumulators restart at `+0.0`).
5. Suite cases `despread_c32`, `despread_cq15`, `despread_c32_complex`, `despread_cq15_complex`, `extremes_m52`.
6. **Block and kind** (`sdrk-blocks/tests/despread_blocks.rs`, `sdrk-pipeline/tests/despread_kinds.rs`; seeds 724, 725): the block equals the kernel bit for bit on whole buffers and in 20 chunk plans, both types; `spread` → `despread` blocks return the symbols times the gain (`sf` for `C32`, `sf / 2^shift` for `CQ15`) for every PRBS choice; the packed code equals `Prbs::next_bit` chip by chip; the description rules of M09, exact storage, struct at most 512 bytes, validation errors as `spread` plus `shift` (17, and any `shift` in a `c32` flow); a pipeline flow `spread` → `despread` equals the blocks and its reported delay matches the samples; `cargo xtask schema --check` and the WASM kind list include `despread`.
7. **SIMD** (`sdrk-backend-simd/tests/despread.rs`, seed 728; pattern of M48 test 9): every level with a path equals `Fallback` bit for bit, both types and both code forms: input lengths 0 … 67 and 4096; `sf` 2 … 17, 31, 1023, 65536; every start offset `chips_in_window` 0 … 7 and calls ending at every position of a window; long codes whose period seam falls inside a window and inside a code word (lengths 31, 33, 2047); the special values (`±0`, subnormals, `±∞`, NaN, `f32::MAX`; `CQ15` −32768 and 32767 in every lane; for `C32` NaN results are compared by class and every other result by `to_bits()`, D296 as in §3); short codes after a `seek` to a chip that is not a multiple of `sf`; the state after each call (`code_pos` 0 for a short code); the golden cases of §4 through each level.

## Acceptance

R1.2 commands (`cargo xtask check` includes `cargo xtask check cycles`, D321); `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-backend-simd --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform wasm32 --kernel <entry>... --merge` for the `simd128` instructions (the entries of §5 and `despread_block`, §5 *Measurement*); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [ ] Kernel `despread` (`C32`, `CQ15`, `Code::Bits` and `Code::Complex`): `DespreadState` with running accumulators and the output shift, trait, Fallback (the `dot_complex` order; exact `i64` sums), Reference, facade delegation, `backend_for`
- [ ] SIMD paths of §5 on every level (each passing D321); `docs/backends.md`
- [ ] Contract rows and section in `docs/kernels.md` (the full text of the shared conventions §C.1; the `dot_complex` order and the two `C32` equalities, the first for non-NaN results with NaN by D296; the code position: stream index modulo `code_len` for a long code, `c = j` and `code_pos` 0 for a short code; the `spread` round trip with the exact `CQ15` term for −32768; the `code_track` zero-bandwidth invariant)
- [ ] Golden cases (seeds 720–722, 726, 727) and the Python model (packed codes, integer model with `shift`)
- [ ] Tests 1–7 (seeds 720–728); suite cases
- [ ] Count and cycle entries (`cargo measure`, `cargo xtask count --platform wasm32`) with their `_fallback` rows, `docs/performance.md`; the reference rows of §5 (VOLK, liquid-dsp, CMSIS-DSP) with their markings
- [ ] Block `DespreadBlock<T>` and kind `despread` (§6): `docs/blocks.md` (entry; reserved-name row "delivered"), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `despread_block`; the delay decision
- [ ] CHANGELOG, decisions log (the output shift and its default; running accumulators instead of a chip history; the two code forms and their contracts; the sample-type rule of the state; `chip_index` counting consumed chips; the long-code position from the stream index and `code_pos` unused and 0 for a short code; the SIMD layouts and any level delegated under D321, with its measured attempts; the reference markings; the pipeline cost)

## Status

Planned (revised 2026-11-17). Nothing is implemented.
