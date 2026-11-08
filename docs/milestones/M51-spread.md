# M51 — Kernel `spread`: direct-sequence spreading

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M30](M30-two-port-blocks.md) · Next: [M52](M52-despread.md)

> **Status: planned** (written 2026-11-08). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

## Goal

A kernel `spread` that turns a stream of complex symbols into a stream of chips at `sf` times the symbol rate: each symbol is multiplied by `sf` consecutive chips of a caller-supplied code. It supports a **short code** (the code repeats every symbol, period `sf`) and a **long code** (the chip index runs on across symbol boundaries, period `≫ sf`), the latter being the LPI-relevant case because it leaves no symbol-rate line. The code chips are data to the kernel (`±1`, or any complex values); a later milestone generates them from an m-sequence, a Gold code or a keyed keystream in `sdrk-proto`.

## Why

Direct-sequence spread spectrum trades bandwidth for a processing gain of `10·log10(sf)` dB, which is what lets a signal sit below the noise floor of a detector (low probability of detection) while a matched receiver still recovers it. Spreading is one multiply per chip, but it needs its own kernel so that the chip phase is tracked across calls (chunk invariance for a long code) and so that the transmit and receive sides (`despread`, M52) share one definition of the chip indexing.

## Order

M51 depends on M30 (the `mul` kernel it generalises). It is independent of the other kernels of the series and can run in parallel with every milestone after M30.

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

**In:** kernel `spread` (`C32`, `CQ15`) in `sdrk_core::kernels::spread` (a new module shared with `despread`, M52) with `SpreadState`, Fallback, Reference, facade delegation, contract, `docs/kernels.md`; golden cases; unit, invariance, extreme tests; suite cases; `cargo xtask count`; CHANGELOG, decisions log.

**Out:** the blocks and kinds `spread`, `dsss_mod` (later milestone; the names stay reserved); the code generators (m-sequence, Gold, Kasami, keyed) in `sdrk-proto`; separate I and Q codes beyond "the code is complex" (a complex code slice already covers I/Q spreading; QPSK with two real codes is `make_complex` of two `spread` calls, documented, no kernel); a SIMD path (allowed later for the `±1` case, which is add/subtract).

## Specification

General rules of M19 apply.

### 1. Trait, state and indexing

```rust
pub struct SpreadState { /* chip_index: u64, sf: u32, code_len: u32, long: bool */ }
impl SpreadState {
    /// `sf` 2..=65536 chips per symbol; `code_len` the length of the code slice passed to
    /// `spread`, ≥ 1; `long` selects long-code indexing. For a short code `code_len` must
    /// equal `sf` (InvalidParameter otherwise).
    pub fn new(sf: u32, code_len: u32, long: bool) -> Result<Self>;
    pub fn chip_index(&self) -> u64;   // the global chip index of the next output chip
    pub fn seek(&mut self, chip: u64); // for a keyed long code at an absolute position
    pub fn reset(&mut self);           // chip_index = 0
}
pub trait Spread<T> {
    /// `code` holds `code_len` chips. Output length must be `input.len() * sf` (checked).
    fn spread(&self, state: &mut SpreadState, input: &[T], code: &[T], output: &mut [T]) -> Result<Progress>;
}
```

- **Output.** For symbol `m` (0-based within the call) and chip `j` in `0..sf`, `output[m·sf + j] = input[m] · code[c]` where the code position `c` is:
  - **short code** (`long = false`): `c = j` (the code restarts every symbol; `code_len = sf`).
  - **long code** (`long = true`): `c = (chip_index0 + m·sf + j) mod code_len`, with `chip_index0` the state's `chip_index` at the start of the call. Here `code_len` may be any length ≥ 1 (typically `≫ sf`, e.g. a 2¹⁵−1 m-sequence); the chip index carries across calls, so the same symbol spread in two chunks gives the same chips.
- The product is the `mul` kernel's product (`C32`: four f32 products then the combinations, no FMA; `CQ15`: `i64`, one round and saturate per component), so `spread` with a `±1` real-valued complex code (`im = 0`) equals `mul` by that code bit for bit. For a real `±1` code the common case is `code[c] = (±1, 0)`, giving `output = ±input`.
- **Progress / lengths.** `output.len()` must equal `input.len()·sf` (`LengthMismatch` otherwise, nothing written — `sf` is fixed, so the output size is determined by the input; the kernel does not emit partial symbols). `code.len()` must equal `code_len` (`LengthMismatch`). After the call, `chip_index += input.len()·sf`.
- Timing: ratio `sf/1` (constant), group delay 0, `min_input` 1 symbol.
- `seek` lets a keyed long code resume at an absolute chip position (same access model as `gauss_noise::seek`, so the chips align to the sample timeline after a gap).

### 2. Fallback and Reference

Bit-exact class. `Reference` is the loop of §1 one symbol at a time, the `mul` reference product. `Fallback` may process chips in runs but gives the same bits; for a `±1` code it may branch to add/subtract (documented as equal to the product because `±1·x` is exact). `backend_for("spread", ..)` reports `fallback`.

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `spread` | `kernels::spread::Spread` | `C32` | bit-exact | `output[m·sf+j] = input[m] · code[c]` with the `mul` `C32` product (four f32 products, no FMA); short/long indexing of §1; fallback equals reference |
| `spread` | `kernels::spread::Spread` | `CQ15` | bit-exact | the `mul` `CQ15` product (`i64`, one round/saturate per component) |

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/spread.py` (shared with `despread.py` of M52; a common code helper).

| Case | Seed | Type | `sf` | code | Symbols | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `spread/short_bpsk_c32_sf31` | 710 | C32 | 31 | short ±1 m-seq length 31 | BPSK symbols |
| `spread/long_bpsk_c32_sf16` | 711 | C32 | 16 | long ±1 code length 127, start chip 50 | long code across symbols |
| `spread/short_qpsk_cq15_sf15` | 712 | CQ15 | 15 | short complex code (±1±j)/… | QPSK, complex code |
| `spread/long_bpsk_c32_sf1023` | 713 | C32 | 1023 | long ±1 m-seq length 1023 | GPS-like SF |

- Expected values from the exact product model (the `mul` f64/Q15 model). The code and its length, `long`, and the start chip are stored in the case.

### 5. Cost

`cargo xtask count` entries `spread_c32` and `spread_cq15` (per output chip; equal to a `mul` step, less for the `±1` fast path). Record in `docs/performance.md`.

## Tests

Seeds 710 … 719.

1. Golden cases of §4 through facade, `Fallback`, `Reference` (`sdrk-testkit/tests/golden_m51.rs`).
2. Unit tests (`sdrk-core/tests/kernels_m51.rs`): `new` rules (`sf` out of range, short code with `code_len ≠ sf`, `code_len = 0`); output-length and code-length checks; a short `±1` code gives `±input` per chip; equality with `mul` for a constant complex code; long-code continuity (spreading `[a,b]` in one call equals two calls `[a]`,`[b]` with the chip index carried); `seek` equals spreading and discarding; `reset`.
3. Round-trip with `despread` (M52) is tested in M52.
4. Block-size invariance: 20 chunk plans at the symbol level (seed 714), both types, short and long code.
5. Extremes: empty input; `sf = 2` and `sf = 65536`; a full-scale `CQ15` symbol and code (saturation as `mul`); NaN/∞ `C32`.
6. Suite cases `spread_c32`, `spread_cq15`, `extremes_m51`.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo xtask count --platform x86_64` (the entries of §5).

## Deliverables checklist

- [ ] Kernel `spread` (`C32`, `CQ15`): `SpreadState`, trait, Fallback, Reference, facade delegation, `backend_for`
- [ ] Contract rows and section in `docs/kernels.md` (short/long indexing; equals `mul` by the code)
- [ ] Golden cases (seeds 710–713) and the Python model (shared code helper with M52)
- [ ] Tests 1–6 (seeds 710–714); suite cases
- [ ] `cargo xtask count` entries, `docs/performance.md`
- [ ] Reserved kind names `spread`, `dsss_mod`, `dsss_demod` in `docs/blocks.md`
- [ ] CHANGELOG, decisions log (short vs long indexing, the `±1` fast path)

## Status

Planned. Nothing is implemented.
