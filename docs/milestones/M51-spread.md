# M51 — Kernel `spread`: direct-sequence spreading

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M38](M38-symbol-sync.md), [M49](M49-scramble.md) (the PRBS codes of the block and the exact sign operations of the shared conventions §A) · Next: [M52](M52-despread.md)

> **Status: planned** (written 2026-11-08; revised 2026-11-17 after an algorithm review (maintainer decision)). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

> **Revision 2026-11-17.** The shared conventions §A to §D cited in this file are defined in [CONVENTIONS-M49-M56.md](CONVENTIONS-M49-M56.md). The maintainer accepted all recommendations of the algorithm review of M49 to M56. Changes in this milestone:
> 1. **Binary chips are exact sign operations, not products** (decision 1; shared conventions §A). The old "`±1` fast path equal to the product because `±1·x` is exact" was wrong for `CQ15`: `+1` is 32767 in `Q15`, and a product with it changes 32768 of the 65536 `Q15` inputs (measured with NumPy in the review). It was also wrong for `C32` with `±∞` input (`∞ · 0 = NaN`) and for signed zeros. The binary form now has its own exact contract; the general complex form keeps the `mul` product (§1, §2, §3).
> 2. **Bit-packed code format** `Code<'_, T>` / `CodeBits` (decision 2; shared conventions §B, **introduced in this milestone**, together with `Prbs::fill_words`). A `prbs15` code takes 4 KiB instead of 256 KiB (`C32`); the block stores bits (§1, §6).
> 3. **No `u64` modulo per chip:** the state carries `code_pos` and wraps it by compare-and-subtract (§1, §2; shared conventions §B.3).
> 4. **SIMD paths in scope on every level** (decision 9, maintainer rule 2026-11-17; shared conventions §D), bit-identical to the fallback; the old "SIMD allowed later" is removed. Fair and informational reference rows are named (§5).
> 5. Text fixes: the *Order* section now matches the *Depends on* line; the `C32` product is the `mul` **fallback** model (the `mul` `C32` SIMD entry is tolerance class, D326, and is not used); the golden table has its missing cells and a concrete complex code; two golden cases are added (seeds 717, 718); the block's `seed` range is stated; the pipeline cost reflects the binary code.
> 6. Precision fixes found when the revision was checked: (a) **NaN in the complex `C32` form** follows D296 (a NaN is NaN on every path, its sign and payload are not part of the contract; NaN lanes are compared by class). The earlier claim that the `sse2` subtraction as an addition of the sign-flipped product "gives the same bits" is false for NaN lanes (on x86 `1 − NaN` returns the NaN unchanged, `1 + (−NaN)` returns it with the sign flipped; checked with NumPy: `∞ · 0 = 0xffc00000`, `1 − NaN = 0xffc00000`, `1 + (−NaN) = 0x7fc00000`). The `to_bits()` comparison of NaN stays for `Code::Bits`, where IEEE negation is exact (§1, §3, §5, tests 2, 5, 9). (b) **Short code and `seek`:** `code_pos` is not used for a short code and stays 0; `seek` then sets only `chip_index` (§1, test 2). (c) **`CodeBits::new` errors** are stated per case: `len = 0` and a set unused bit are `InvalidParameter`, a wrong word count is `LengthMismatch` (§1, test 2).

## Goal

A kernel `spread` that turns a stream of complex symbols into a stream of chips at `sf` times the symbol rate: each symbol is multiplied by `sf` consecutive chips of a caller-supplied code. It supports a **short code** (the code repeats every symbol, period `sf`) and a **long code** (the chip index runs on across symbol boundaries, period `≫ sf`), the latter being the LPI-relevant case because it leaves no symbol-rate line. The code chips are data to the kernel, in one of two forms: a **binary code** packed one bit per chip (chip value `±1`, applied by exact sign operations; the form of every PRBS, m-sequence, Gold or keyed code) or a **complex code** of arbitrary complex values (applied with the `mul` product). A later milestone generates Gold codes or a keyed keystream in `sdrk-proto`; this milestone delivers the packed code type and the PRBS generator into words.

## Why

Direct-sequence spread spectrum trades bandwidth for a processing gain of `10·log10(sf)` dB, which is what lets a signal sit below the noise floor of a detector (low probability of detection) while a matched receiver still recovers it. Spreading is one sign operation (binary code) or one complex multiply (complex code) per chip, but it needs its own kernel so that the chip phase is tracked across calls (chunk invariance for a long code), so that the transmit and receive sides (`despread`, M52) share one definition of the chip indexing and one code format, and so that long binary codes fit microcontroller RAM (one bit per chip instead of 8 bytes).

## Order

M51 depends on M49 (the exact sign and swap operations `q15_neg_sat`, `cq15_chip`, `c32_chip` of the shared conventions §A in `sdrk_core::num`, and the polynomials `Prbs5` … `Prbs11`) and on M38 (`sdrk_proto::prbs`), and through them on M30 (the `mul` product that the complex code form uses). It is independent of M50 and can run in parallel with it once M49 is delivered. M52, M53 and M54 use the code format introduced here.

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

**In:** kernel `spread` (`C32`, `CQ15`) in `sdrk_core::kernels::spread` (a new module shared with `despread`, M52) with `SpreadState`, Fallback, Reference, facade delegation, contract, `docs/kernels.md`; the code format of the shared conventions §B, introduced here: `CodeBits` and `Code` in `sdrk_core::kernels::code` and `Prbs::fill_words` in `sdrk_proto::prbs`, with the full text of §B in `docs/kernels.md`; **SIMD paths on every level** (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`) in `sdrk-backend-simd`, bit-identical to the fallback, for both code forms and both types (§5); golden cases; unit, invariance, extreme and SIMD tests; suite cases; `cargo xtask count` and `cargo measure`; CHANGELOG, decisions log; the block `SpreadBlock<T>` and the pipeline kind `spread` (§6) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the composite `dsss_mod` (later milestone; the name stays reserved); code generators beyond the PRBS of `sdrk_proto::prbs` (Gold, Kasami, keyed; backlog); separate I and Q codes beyond "the code is complex" (a complex code already covers I/Q spreading; QPSK with two real codes is `make_complex` of two `spread` calls, documented, no kernel).

## Specification

General rules of M19 apply.

### 1. Trait, state, code and indexing

**Code format (shared conventions §B, introduced in this milestone).** The code is a `Code<'_, T>` (shared conventions §B, introduced in M51): `Code::Bits` is the bit-packed binary form (chip `i` = bit `i & 31` of word `i >> 5`, chip value `1 − 2b`, chip `i` = the `i`-th `Prbs::next_bit` of a PRBS), applied by the exact sign operations of §A; `Code::Complex` is the general form with the `mul` product. The full definition, written into `docs/kernels.md` (section "Code format") by this milestone:

- **Chip value and order (§B.1).** Chip bit `b ∈ {0, 1}` has the chip value `1 − 2b` (`0 → +1`, `1 → −1`), the usual DSSS/GNSS convention in which the product of two chips is the XOR of their bits. It differs on purpose from the data mapping of `psk_mod` and `symbol_map` (BPSK `2b − 1`), which stays unchanged: a code is not data. Chip `i` (0-based) of a PRBS code is the `i`-th value returned by `Prbs::next_bit` after `Prbs::new(poly, seed)` (register convention of `prbs.rs`: bit `k` of the state is stage `k + 1`, the output becomes stage 1).
- **Packed form (§B.2).** A code of `len` chips is stored in `ceil(len / 32)` words of `u32`; chip `i` is bit `i & 31` (LSB first) of word `i >> 5`: `chip(i) = (words[i >> 5] >> (i & 31)) & 1`. As bytes (little-endian, R6.6) byte `j` holds chips `8j … 8j + 7`, LSB first. The unused bits `len & 31 … 31` of a partial last word must be 0 (`CodeBits::new` checks it, `InvalidParameter`, so the packed form of a code is unique); kernels never use those bits as chips, and a SIMD path that loads whole words masks or stops at `len`. Size: `prbs15` (32767 chips) takes 1024 words = 4 KiB, against 256 KiB as `C32` and 128 KiB as `CQ15`.
- **Length, period and position (§B.3).** `len` is 1 to `u32::MAX` (`len = 0` is `InvalidParameter`); `words.len()` must equal `ceil(len / 32)` (`LengthMismatch`). The code is periodic with period `len`. For a code whose position carries across symbols (the long code of §1) the state keeps `code_pos = chip_index mod len` as a `u32` and advances it with compare-and-subtract at the period end; a `u64` modulo runs only in `new` and `seek`, never per chip (also not in `Fallback`). The word and bit cursor are `code_pos >> 5` and `code_pos & 31`. When `len` is not a multiple of 32 the period seam is not word-aligned: Fallback and SIMD paths process runs that end at the period seam and at symbol boundaries, and a run crossing the seam continues at chip 0 of word 0.
- **Types (§B.4)**, in `sdrk_core::kernels::code`:

```rust
/// A binary code: `len` chips packed LSB first into `u32` words (§B.2), chip value 1 − 2b.
#[derive(Clone, Copy, Debug)]
pub struct CodeBits<'a> { /* words: &'a [u32], len: u32 */ }
impl<'a> CodeBits<'a> {
    pub fn new(words: &'a [u32], len: u32) -> Result<Self>; // len = 0: InvalidParameter; words.len() != ceil(len/32): LengthMismatch; a set unused bit: InvalidParameter
    pub fn len(&self) -> u32;
    pub fn words(&self) -> &'a [u32];
    pub fn chip(&self, i: u32) -> u32;                      // 0 or 1, i < len
}
/// The two forms of a code; every kernel with a code takes this.
#[derive(Clone, Copy, Debug)]
pub enum Code<'a, T> { Bits(CodeBits<'a>), Complex(&'a [T]) }
impl<T> Code<'_, T> { pub fn len(&self) -> u32; }
```

  `Code::Bits` applies each chip by §A (`c32_chip`, `cq15_chip`); it is exact and symmetric and needs no multiplication. `Code::Complex` takes any complex chips (phasor codes, I/Q codes, the QPSK case of §4) and keeps the `mul` product. The two forms are **separate contracts**: a complex code `(±1, 0)` equals the binary form for finite `C32` input apart from signed zeros, and it does not equal it in `CQ15` (see "Product" below). The blocks use `Bits`; `Complex` is for callers of the Rust API. `Code::len` of a `Complex` slice longer than `u32::MAX` is not representable: the kernel checks the slice length against the state's `code_len` first (`LengthMismatch`).
- **Generator (`sdrk-proto`).** `Prbs::fill_words(&mut self, words: &mut [u32], chips: u32) -> Result<()>` writes the next `chips` bits in the order of §B.1 and sets the unused bits to 0; `words.len()` must be `ceil(chips / 32)` (`InvalidArgument` otherwise, nothing written). A word-parallel LFSR is allowed if it gives the same words. `sdrk-proto` cannot depend on `sdrk-core` (R2), so it writes plain `u32` words and the block wraps them in `CodeBits::new`.

**Trait and state.**

```rust
pub struct SpreadState { /* chip_index: u64, code_pos: u32, sf: u32, code_len: u32, long: bool */ }
impl SpreadState {
    /// `sf` 2..=65536 chips per symbol; `code_len` the length of the code passed to
    /// `spread`, ≥ 1; `long` selects long-code indexing. For a short code `code_len` must
    /// equal `sf` (InvalidParameter otherwise).
    pub fn new(sf: u32, code_len: u32, long: bool) -> Result<Self>;
    pub fn chip_index(&self) -> u64;   // the global chip index of the next output chip
    pub fn seek(&mut self, chip: u64); // chip_index = chip; long code: code_pos = chip mod code_len
                                       // (one u64 modulo); short code: code_pos stays 0
    pub fn reset(&mut self);           // chip_index = 0, code_pos = 0
}
pub trait Spread<T> {
    /// `code` holds `code_len` chips in either form. Output length must be `input.len() * sf` (checked).
    fn spread(&self, state: &mut SpreadState, input: &[T], code: Code<'_, T>, output: &mut [T]) -> Result<Progress>;
}
```

- **Output.** For symbol `m` (0-based within the call) and chip `j` in `0..sf`, `output[m·sf + j] = chip(input[m], c)` where the code position `c` is:
  - **short code** (`long = false`): `c = j` (the code restarts every symbol; `code_len = sf`). `code_pos` is not used for a short code and stays 0 in every state: `new`, `reset`, `seek` (which sets only `chip_index`) and every call leave it 0, and the code position never depends on `chip_index`.
  - **long code** (`long = true`): `c = (chip_index0 + m·sf + j) mod code_len`, with `chip_index0` the state's `chip_index` at the start of the call. Here `code_len` may be any length ≥ 1 (typically `≫ sf`, e.g. a 2¹⁵−1 m-sequence); the chip index carries across calls, so the same symbol stream spread in two chunks gives the same chips. The implementation does not evaluate this formula per chip: `code_pos` holds `chip_index mod code_len` and advances by one per chip with compare-and-subtract at `code_len` (§B.3). `chip_index` advances with `wrapping_add`; the formula and `code_pos` agree as long as `chip_index` has not wrapped (2⁶⁴ chips, not reachable in practice), and `code_pos` is the defining position after a wrap.
- **Product** `chip(x, c)`:
  - **`Code::Bits`:** `±1` is applied by the exact sign and swap operations of the shared conventions §A (`q15_neg_sat`, `cq15_quadrant`, `c32_quadrant`, introduced in M49), never by a product. In `CQ15` a stored negation saturates (`−(−32768) = 32767`); inside a wide accumulation it is plain integer negation. Here: bit 0 gives `x` unchanged (both types, bit for bit), bit 1 gives `cq15_chip(x, 1) = (q15_neg_sat(x.re), q15_neg_sat(x.im))` for `CQ15` and `c32_chip(x, 1) = (−x.re, −x.im)` (IEEE negation, only the sign bit flips) for `C32`.
  - **`Code::Complex`:** the `mul` product of the fallback: `C32` the float32 model `re = xr·cr − xi·ci`, `im = xr·ci + xi·cr` (the four products first, then the combinations, no FMA); `CQ15` `num::cq15_mul` (`i64`, one round and saturate per component). `spread` with a complex code equals the `mul` **fallback** by that code bit for bit; it is not tied to the `mul` `C32` SIMD entry, which fuses products (tolerance class, D326).
  - **Why the forms differ** (stated in the contract and the rustdoc). In `CQ15` `+1` is not representable: a complex chip `(32767, 0)` scales by `1 − 2⁻¹⁵` and changes 32768 of the 65536 `Q15` values (every value with `|x| ≥ 16384` moves by 1 LSB toward zero; measured in the review and rechecked exhaustively), while `(−32768, 0)` equals the saturating negation for every input. The binary form is exact and symmetric. In `C32` a binary chip applied to `±∞` gives `±∞` (sign per the chip), never NaN; the complex path keeps the product behaviour, so `(∞, 0) · (1, 0)` has the imaginary part `∞ · 0 + 0 · 1 = NaN`, and `(−0, −0) · (1, 0)` has the real part `−0 − (−0) = +0` where the binary form keeps `−0`.
  - **NaN (`C32`).** `Code::Bits` is exact for NaN too: IEEE negation keeps the payload and flips the sign, and every path gives the same bits (`to_bits()`). `Code::Complex` follows D296: where the fallback gives NaN, every path gives NaN in the same component, but the sign and payload of that NaN are not part of the contract (a NaN arises from a NaN input or from `∞ · 0`, e.g. input `(∞, 0)` with chip `(1, 0)`); non-NaN outputs are bit-exact.
- **Progress / lengths.** `output.len()` must equal `input.len()·sf`, computed with `checked_mul` (`LengthMismatch` otherwise, also on overflow; nothing written — `sf` is fixed, so the output size is determined by the input; the kernel does not emit partial symbols). `code.len()` must equal `code_len` (`LengthMismatch`); a `CodeBits` is already validated by its constructor. After the call, `chip_index += input.len()·sf` (wrapping) and `code_pos` has advanced by the same number of chips modulo `code_len` (long code; for a short code it stays 0).
- Timing: ratio `sf/1` (constant), delay 0, `min_input` 1 symbol. The delay is declared at the first chip of a symbol, not at the centre of its `sf` chips (the `interpolate` rule of timing.md would give `(sf − 1)/(2 sf)`), so that `spread` → `despread`, which maps symbol `m` to output `m`, reports the delay 0 the samples show (`despread` would need the negative centre delay `−(sf − 1)/2`, which `Delay` cannot express). Logged.
- `seek` lets a keyed long code resume at an absolute chip position (same access model as `gauss_noise::seek` of M48, so the chips align to the sample timeline after a gap). It is the only place besides `new` with a `u64` modulo.

### 2. Fallback and Reference

Bit-exact class, both code forms. `Reference` is the loop of §1 one symbol at a time: the code position from the `u64` formula of §1, the chip from `CodeBits::chip` and the §A helpers (`Code::Bits`) or the `mul` reference product (`Code::Complex`). `Fallback` gives the same bits without a per-chip division: it processes **runs** of one symbol broadcast over `min(sf − j, code_len − code_pos)` chips for a long code and over the whole symbol for a short code (a run ends at the symbol boundary and at the period seam, §B.3), reads the binary chips word by word from the cursor `code_pos >> 5`, `code_pos & 31`, and applies `cq15_chip` / `c32_chip` or the product. There is no "fast path equal to the product": the binary form is its own contract, and the complex form is always the product (also for a complex code whose chips happen to be `(±1, 0)`). The run structure is also the structure of the SIMD paths (§5). `backend_for("spread", ..)` reports the SIMD level where a path exists (§5), otherwise `fallback`.

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `spread` | `kernels::spread::Spread` | `C32`, `Code::Bits` | bit-exact | `output[m·sf+j] = c32_chip(input[m], chip(c))` (shared conventions §A.2: IEEE negation, exact for every value; `±∞` stays `±∞`, NaN keeps its payload); short/long indexing of §1; fallback, reference and every SIMD level equal |
| `spread` | `kernels::spread::Spread` | `C32`, `Code::Complex` | bit-exact | `output[m·sf+j] = input[m] · code[c]` with the `mul` fallback float32 model (four f32 products, then the combinations, no FMA; not the `mul` SIMD entry, D326); fallback, reference and every SIMD level equal; a NaN output is NaN on every path, its sign and payload are not part of the contract (D296) |
| `spread` | `kernels::spread::Spread` | `CQ15`, `Code::Bits` | bit-exact | `cq15_chip(input[m], chip(c))`: `x` for bit 0, `(q15_neg_sat(x.re), q15_neg_sat(x.im))` for bit 1 (§A.1; `−(−32768) = 32767`); not equal to `mul` by `(32767, 0)` |
| `spread` | `kernels::spread::Spread` | `CQ15`, `Code::Complex` | bit-exact | `num::cq15_mul(input[m], code[c])` (`i64`, one round/saturate per component) |

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/spread.py` (shared with `despread.py` of M52; a common code helper that packs codes by §B.2 and models §A).

| Case | Seed | Type | `sf` | Code form and code | Symbols | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `spread/short_bpsk_c32_sf31` | 710 | C32 | 31 | `Bits`, short, `prbs5` m-sequence (31 chips, one partial word) | 256 BPSK symbols | short code |
| `spread/long_bpsk_c32_sf16` | 711 | C32 | 16 | `Bits`, long, `prbs7` (127 chips), start chip 50 | 256 BPSK symbols | long code across symbols; the period seam is not word-aligned |
| `spread/short_qpsk_cq15_sf15` | 712 | CQ15 | 15 | `Complex`, short, 15 random chips from `(±1 ± j)/√2` as `Q15` (`±23170`) | 256 QPSK symbols at amplitude 0.7 | complex code, `cq15_mul` |
| `spread/long_bpsk_c32_sf1023` | 713 | C32 | 1023 | `Bits`, long, `prbs10` m-sequence (1023 chips) | 4 BPSK symbols | GPS-like SF |
| `spread/long_qpsk_cq15_sf16` | 717 | CQ15 | 16 | `Bits`, long, `prbs7` (127 chips), start chip 100 | 256 QPSK symbols at full scale, including components −32768 and 32767 | the saturating negation of §A.1 |
| `spread/long_phasor_c32_sf8` | 718 | C32 | 8 | `Complex`, long, 37 unit phasors `exp(j·2πk/8)` with random `k` | 256 QPSK symbols | complex long code, `mul` float32 model |

- The case sizes stay below R8.6 (the largest output, 4092 `C32` chips of case 713, is 32 KiB).
- Expected values from the exact models: `Code::Bits` by the sign model (`C32`: float32 negation in NumPy; `CQ15`: `int64` saturating negation of the `q15.py` model); `Code::Complex` by the `mul` models (`C32`: the float32 model with the four products first; `CQ15`: the `Q15` product of `q15.py`). The code (`code_words` as `u32` little-endian `.bin` and the JSON field `code_len` for `Bits`, the array `code` for `Complex`), the form, `long` and the start chip are stored in the case.

### 5. Cost, SIMD plan and reference rows

**Entries.** `cargo xtask count` and `cargo measure` entries `spread_bits_c32`, `spread_bits_cq15`, `spread_complex_c32` and `spread_complex_cq15` (per output chip, every level), each with its `<k>_fallback` entry, plus `spread_block` (§6). Record the numbers in `docs/performance.md`.

**Estimated cost (to be confirmed by `cargo measure` and `cargo xtask count`).** A binary chip is one conditional sign per component and a bit extraction; a complex chip is a `mul` step. On Cortex-M0 the review estimates 3–5 cycles per binary `CQ15` chip (a saturating negate) against 15–25 cycles for the `cq15_mul` product (no `SMLAD`/`SSAT`) (to be confirmed). On host targets the binary path is bound by the output stores (to be confirmed).

**SIMD plan (`sdrk-backend-simd`).** SIMD follows the shared conventions §D (maintainer rule 2026-11-17): every lane-parallel part of the kernel gets a path on every level in this milestone, bit-identical to the fallback through the fixed element-wise formula or reduction order of the contract; D321 is a quality gate (a slower path is improved, not removed; delegation to the fallback only after several measured failed attempts, with an open item); cycles per sample with `cargo measure` against the fallback and the fair reference (D32), unfair rows marked.

- *Lane-parallel part:* everything. `spread` is element-wise (§D.2): each output chip is one fixed formula of one symbol and one chip, with no reduction and no running increment. A run of §2 is a broadcast of the symbol into a vector and one operation per chip. *Scalar part:* none in the kernel (the PRBS generation of the block is a serial LFSR, but `fill_words` runs once at construction, and a word-parallel form is allowed).
- *Binary `CQ15`:* chip bits become per-lane masks `m ∈ {0, −1}` (one mask per 32-bit `re/im` pair, from the code word shifted by the lane's bit offset), then the conditional saturating negation `subs(xor(x, m), m)` of §A.1 (x86 `psubsw`, NEON `vqsubq_s16`, `simd128` `i16x8.sub_sat_s`). `psignw`/`vpsignw` is not allowed (it wraps −32768). AVX-512 may build the masks from the code word through mask registers (each chip bit duplicated for its two lanes); any mask construction is allowed that gives the lane masks of the code bits.
- *Binary `C32`:* XOR of the sign bit `0x8000_0000` with the same per-chip lane masks (§A.2); compare with `to_bits()`, NaN included.
- *Complex `C32`:* the broadcast symbol times the chips with the four products first and the combinations as separate additions, no FMA (`addsubps` where available is the same IEEE operations; on `sse2`, which has no `addsubps`, the subtraction may be done as an addition of the sign-flipped product: for every non-NaN lane `a + (−b)` and `a − b` are the same rounded operation and give the same bits, while a NaN lane may come out with the opposite sign — on x86 `a − NaN` returns the NaN unchanged and `a + (−NaN)` returns it sign-flipped — which D296 allows, §1 "NaN"). The backend does not call the `mul` `C32` SIMD entry (D326).
- *Complex `CQ15`:* `cq15_mul` per lane. In `i32` each component sum with the rounding constant fits except one case: the imaginary part `xr·ci + xi·cr + 2¹⁴` with all four components −32768 reaches `2³¹ + 2¹⁴` (result 32767 after saturation); every other input stays within `±(2³¹ − 32768) + 2¹⁴` (checked by a bound on the corners of the bilinear form). A path either widens to `i64` or handles that case explicitly (e.g. a `pmaddwd` split as in `dot` `CQ15`); both give the fallback bits.
- *Seams:* a vector that crosses a symbol boundary, the period seam (not word-aligned when `code_len` is not a multiple of 32) or a word boundary is split into runs as in §2 or uses lane-wise chip extraction that gives the same chips; the tail shorter than a vector runs the scalar code. The state after a call (`chip_index`, `code_pos`; `code_pos` 0 for a short code) equals the fallback's.
- *Planned backends (notes only):* `arm-dsp` (Cortex-M4) binary `CQ15` as `QSUB16` from zero with a per-chip select (two chips per word access); `rvv` with `vssub` under a mask.

**Reference rows (D32).** No established library has a spreading kernel that does the same work (symbol broadcast, short/long code indexing with a carried chip phase, bit-packed codes, exact sign operations). The rows below are **information**, not a gap to close (to be confirmed when the bench harness is written):

| Our entry | Library row | Marking | Why |
| --- | --- | --- | --- |
| `spread_complex_c32` | VOLK `volk_32fc_x2_multiply_32fc` on a pre-broadcast symbol buffer and the code as `C32` | **reference does less work** | no broadcast and no code indexing; some VOLK kernels fuse products |
| `spread_complex_cq15` | CMSIS-DSP `arm_cmplx_mult_cmplx_q15` (Cortex-M) on the same buffers | **reference does different work** | output in 3.13 format with truncation, not the R7 rounding of `cq15_mul` |
| `spread_bits_cq15` | CMSIS-DSP `arm_negate_q15` on the chips of `−1` | **reference does less work** | unconditional negation, no code bits; shows the floor of a binary chip on Cortex-M |

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`, Cortex-M). The D321 gate applies to every SIMD path of this milestone.

### 6. Block and kind `spread`

`SpreadBlock<T>` (`C32`/`CQ15` → same, `sf` output chips per input symbol) around the kernel `spread`. The code is a PRBS of `sdrk_proto::prbs`, generated once into packed words with `Prbs::fill_words` (`period` chips, `ceil(period / 32)` words) and used as `Code::Bits` (chip `b` → `1 − 2b`, applied by §A); `SpreadBlock::with_code` takes any code of the caller in either form, packed words or complex chips (Rust API only).

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `spread` | `SpreadBlock<T>` | C32/CQ15 → same | `sf` (I64, build, required; 2 … 65536), `code` (Choice, build, `prbs7`; `prbs5`, `prbs7`, `prbs9`, `prbs10`, `prbs11`, `prbs15`), `seed` (I64, build, 1; 1 … 32767, the low `n` bits are used, zero after masking is `invalid_value`), `long` (Bool, build, false) | `spread` with `Code::Bits`; a short code (`long` false) needs `sf` equal to the code period (31, 127, 511, 1023, 2047 or 32767), else `invalid_value` at `params.sf` naming the period; a long code runs on across symbols and calls |

- `SpreadConfig`. Storage: the packed code, `ceil(period / 32)` words of `u32` (1024 words = 4 KiB for `prbs15`, 4 words = 16 B for `prbs7`), the same for both types; a caller code of `with_code` takes its packed words or `period` values of `T`; exact. Timing ratio `sf/1`, delay as the kernel states it (0); a call takes `min(input, output / sf)` symbols.
- Pipeline kind `spread` (`c32` or `cq15`), cost `sf` per input symbol (one exact sign operation per chip, no multiply; the earlier `4 sf` assumed a complex product per chip), a row in `docs/pipeline.md` and `docs/timing.md`. Chain position: symbols at the symbol rate → `spread` → pulse shaping (`interpolate` or `fir`).
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 710 … 719.

1. Golden cases of §4 (seeds 710–713, 717, 718) through facade, `Fallback`, `Reference` (`sdrk-testkit/tests/golden_m51.rs`).
2. Unit tests (`sdrk-core/tests/kernels_m51.rs`):
   - `new` rules (`sf` out of range, short code with `code_len ≠ sf`, `code_len = 0`); output-length and code-length checks (both forms; `input.len()·sf` overflow is `LengthMismatch`);
   - `CodeBits::new`: wrong word count (`LengthMismatch`), `len = 0` (`InvalidParameter`), a set unused bit (`InvalidParameter`), nothing built in each case; `chip(i)` against the packing rule;
   - binary form: a code bit 0 gives `input` bit for bit and bit 1 gives `cq15_chip`/`c32_chip`, exhaustively over all 65536 `Q15` values per component (−32768 → 32767) and for `C32` on `±0`, `±∞`, NaN (payload kept, sign flipped), subnormals and random values (`to_bits()`);
   - the forms differ as documented: `Code::Complex` with `(32767, 0)` changes exactly 32768 of the 65536 `Q15` values, `(−32768, 0)` equals the binary bit 1 for all of them; in `C32` `(∞, 0)` through `(1, 0)` gives an imaginary NaN in the complex form and `(∞, 0)` in the binary form;
   - complex form: equality with the `mul` fallback for a constant complex code, both types;
   - long-code continuity (spreading `[a,b]` in one call equals two calls `[a]`,`[b]` with the chip index carried), with `code_len` 127 so that runs cross the period seam at a bit offset that is not word-aligned, and with `code_len` 64 (word-aligned seam);
   - `seek` equals spreading and discarding (positions 0, 1, 126, 127, 2⁴⁰ + 5), for a long code and for a short code; after `seek` `chip_index` equals `chip`, and `code_pos` equals `chip mod code_len` for a long code and stays 0 for a short code (also for a `chip` that is not a multiple of `sf`); `reset`.
3. Round-trip with `despread` (M52) is tested in M52.
4. Block-size invariance: 20 chunk plans at the symbol level (seed 714), both types, both code forms, short and long code.
5. Extremes: empty input; `sf = 2` and `sf = 65536`; a full-scale `CQ15` symbol with both forms (saturating negation for `Bits`, saturation as `mul` for `Complex`, including the all-−32768 product whose imaginary part saturates to 32767); NaN/∞ `C32` with both forms (`Code::Bits` compared by `to_bits()`, NaN included; `Code::Complex` bit for bit on non-NaN outputs and NaN lanes by class, D296, including the NaN of input `(∞, 0)` with chip `(1, 0)`).
6. Suite cases `spread_c32`, `spread_cq15`, `extremes_m51` (both code forms).
7. **Block and kind** (`sdrk-blocks/tests/spread_blocks.rs`, `sdrk-pipeline/tests/spread_kinds.rs`; seeds 715, 716): the block equals the kernel with the packed PRBS code bit for bit on whole buffers and in 20 chunk plans, both types, short and long code; the packed code equals the bits of `next_bit` for every `code` choice; the description rules of M09, exact storage, struct at most 512 bytes, validation errors at their parameter (`sf` out of range, a short code with `sf` not the period, `seed` zero after masking); a pipeline flow equals the block; `cargo xtask schema --check` and the WASM kind list include `spread`.
8. **PRBS into words** (`sdrk-proto/tests/prbs_m51.rs`): `fill_words` equals `next_bit` packed by §B.2 for every polynomial and for `chips` 1, 31, 32, 33, the period and two periods; the unused bits are 0; two calls continue the sequence; a wrong `words.len()` is `InvalidArgument` and writes nothing.
9. **SIMD** (`sdrk-backend-simd/tests/spread.rs`, seed 719): every level that has a path equals `Fallback` bit for bit, both types, both code forms, short and long code: symbol counts 0 … 67 and 4096 at `sf` ∈ {2, 3, 8, 15, 16, 31, 33}, `code_len` ∈ {sf, 31, 32, 33, 127, 1023}, start chips just before word boundaries and the period seam; the special values of tests 2 and 5 (−32768, `±∞`, NaN, `±0`; for `C32` `Code::Complex` NaN lanes are compared by class and non-NaN lanes by `to_bits()`, D296; for `Code::Bits` every lane by `to_bits()`); the state after each call equals the fallback's (`code_pos` 0 for a short code); the golden cases of §4 through each level.

## Acceptance

R1.2 commands (`cargo xtask check` includes `cargo xtask check cycles`, D321); `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-proto --all-features`; `cargo test -p sdrk-backend-simd --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the entries of §5 and `spread_block`); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [ ] Code format of the shared conventions §B: `CodeBits`, `Code` in `sdrk_core::kernels::code`; `Prbs::fill_words` in `sdrk_proto::prbs`; section "Code format" in `docs/kernels.md` with the full text of §B
- [ ] Kernel `spread` (`C32`, `CQ15`, both code forms): `SpreadState` (with `code_pos`), trait, Fallback (runs, no per-chip division), Reference, facade delegation, `backend_for`
- [ ] SIMD paths of §5 on every level (each passing D321, or delegated after measured attempts with an open item); `docs/backends.md`
- [ ] Contract rows and section in `docs/kernels.md` (short/long indexing; the binary form by the exact sign operations of §A; the complex form equals the `mul` fallback; why the two forms differ in `CQ15` and for non-finite `C32`; NaN of the complex `C32` form by D296; `code_pos` and `seek` for a short code)
- [ ] Golden cases (seeds 710–713, 717, 718) and the Python model (shared code helper with M52, packing by §B.2)
- [ ] Tests 1–9 (seeds 710–719); suite cases
- [ ] Count and cycle entries (`cargo xtask count`, `cargo measure`), `docs/performance.md`, the information rows of §5
- [ ] Block `SpreadBlock<T>` and kind `spread` (§6): `docs/blocks.md` (entry; reserved-name row "delivered"), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `spread_block`
- [ ] CHANGELOG, decisions log (short vs long indexing; the two code forms as separate contracts instead of a `±1` fast path; the packed code format; `code_pos` instead of a per-chip modulo, unused and 0 for a short code; D296 for NaN of the complex `C32` form; the pipeline cost `sf`; the marking of the reference rows; the SIMD levels that stay on the fallback, with their numbers)

## Status

Planned (revised 2026-11-17). Nothing is implemented.
