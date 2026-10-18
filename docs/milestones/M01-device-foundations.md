# M01 — `sdrk-device`: formats, time types, stream metadata, errors

> Plan: [PLAN.md › Phase 1](../PLAN.md#phase-1--core-foundations-medium) · Depends on: [M00](M00-workspace-foundation.md) · Next: [M02](M02-core-foundations.md)

## Goal

Implement the lowest crate `sdrk-device`: the shared error classification, the description of raw sample formats (`SampleFormat`), the time types (sample index, exact rates, time anchors) and the minimal synchronous RX/TX stream traits with per-block metadata. Everything else in the workspace builds on these types, and external drivers will implement these traits, so they must be small, precise and stable.

## Why

Drivers live in other repositories and must only depend on this crate. Formats and time types are needed by `convert` (M05), the testkit's virtual device (M03) and the pipeline (M10/M11). Control capabilities and async streams follow in M12.

## Ground rules (identical in every milestone — read them completely before you start)

These rules apply to every line of code in this repository. They are repeated in every milestone file so that each milestone is self-contained. If a rule and a milestone-specific instruction ever conflict, stop and ask; do not guess.

### R1 — Work order and scope

1. Milestones are implemented strictly in numeric order. A milestone may only use what earlier milestones delivered. Never implement items of a later milestone "while you are at it".
2. Every milestone must leave the repository in a state where **all** of these commands succeed from a clean checkout:
   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   - `cargo test --workspace --all-features`
   - `cargo xtask check` (exists from M00 on; runs the architecture checks in R2–R6)
   - `cargo xtask bare-metal` (exists from M00 on; builds every heap-free crate for `thumbv7em-none-eabihf`)
   - plus every additional command listed in the milestone's *Acceptance* section.
3. **Host platforms:** everything (build, tests, `xtask`, golden generator) must work on Linux x86_64, Linux arm64 (Raspberry Pi, 64-bit OS) and Windows x86_64. No shell scripts or Makefiles; tools are Rust (`xtask`) or the Python golden generator; paths are built with `Path::join`; line endings are fixed by `.gitattributes`. CI runs every tier-1 job on all three hosts.
4. When something is ambiguous, choose the simplest solution that satisfies the written tests, and record the decision in `docs/decisions-log.md` (one line: date, milestone, decision, reason). Never silently change a public API defined by an earlier milestone; if a change is unavoidable, update every caller and the affected docs in the same milestone and log it.
5. Naming: write **SDRk** in prose, docs and messages; `sdrk` only where lowercase is required (crate, package, module, file and path names).
6. Plan (living document): https://claude.ai/code/artifact/694f271a-ba38-4b7c-b2fa-fb2e919d35c1 — snapshot in [`docs/PLAN.md`](../PLAN.md). The plan explains *why*; the milestones say *what exactly* to build.

### R2 — Layers and dependencies

The folder of a crate is its layer. A crate may depend only on crates in **lower** layers (or, for the testkit, on any library layer). `cargo xtask check` enforces this by reading `cargo metadata` and the crate paths.

| Layer (folder) | Crates | May depend on (workspace crates) | `no_std`, no heap |
| --- | --- | --- | --- |
| `crates/core` (lowest) | `sdrk-device` | — | yes |
| `crates/core` | `sdrk-core` | `sdrk-device` | yes |
| `crates/backends` | `sdrk-backend-*` | `sdrk-core`, `sdrk-device` | yes |
| `crates/facade` | `sdrk` | core, backends | yes |
| `crates/blocks` | `sdrk-blocks` | facade, core | yes |
| `crates/pipeline` | `sdrk-pipeline` | blocks, facade, core | no (std/alloc allowed) |
| `crates/io` | `sdrk-io` | core (types only) | no |
| `crates/testing` | `sdrk-testkit` | any library crate above | `no_std` by default; feature `std` adds file I/O and proptest helpers |
| `bindings/*`, `examples/*`, `targets/*`, `benches/*`, `xtask` | — | anything | no |

Additional rules:
- Within `crates/core` the rank is by crate name: `sdrk-device` (rank 0) < `sdrk-core` (rank 1); `sdrk-core → sdrk-device` is allowed, the reverse is not. All other layers are one rank per folder.
- Backends never depend on each other. Only the facade `sdrk` knows the backends.
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
6. Little-endian only: `sdrk-device` contains `#[cfg(target_endian = "big")] compile_error!("SDRk supports little-endian targets only");`.

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
6. Each case (JSON + all its `.bin` arrays) stays at or below 512 KiB; all vectors together at or below 16 MiB. Prefer N ≤ 4096 samples; use longer inputs only when the kernel needs them (long filters, decimation).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.


## Scope

**In:** `error` (ErrorKind, SdrkError, Error), `format` (SampleFormat and friends, constants for known devices, byte-size helpers), `time` (SampleIndex, Rate, TimeAnchor, TimeSource, Duration helpers), `stream` (BlockFlags, BlockMeta, RxStream, TxStream), little-endian guard.

**Out:** device control capabilities, async traits, conformance suite (M12). Any conversion of sample data (M05).

## Specification

All items are public and re-exported from `lib.rs`. Crate is `#![no_std]`, no dependencies except `core`.

### 1. `error.rs`

- `ErrorKind` and trait `SdrkError` exactly as in R5.1.
- `ErrorKind` implements `Display` with snake_case names (`invalid_argument`, …).
- Crate error:
  ```rust
  #[derive(Debug, Clone, PartialEq, Eq)]
  #[non_exhaustive]
  pub enum Error {
      InvalidFormat(&'static str),        // kind: InvalidArgument
      InvalidRate { num: u64, den: u64 }, // kind: InvalidArgument
      Overflow(&'static str),             // kind: OutOfRange
      BufferTooSmall { needed: usize, got: usize }, // kind: BufferTooSmall
  }
  ```
  with `Display`, `core::error::Error`, `SdrkError`.

### 2. `format.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] #[non_exhaustive]
pub enum Container { U8, I8, I16, I32, F32, Packed10, Packed12, Bits1 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout { Interleaved, Planar, Real }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Align { Lsb, Msb }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endian { Little, Big }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleFormat { /* private fields */ }
```

`SampleFormat` fields: `container`, `bits: u8` (effective bits per component), `signed: bool`, `layout`, `align`, `endian`, `swap_iq: bool`, `full_scale: FullScale`.

```rust
/// What raw value corresponds to amplitude 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FullScale { /// 2^(bits-1), the normal case
    PowerOfTwo, /// an explicit integer, e.g. 2048 for bladeRF SC16_Q11
    Explicit(u32) }
```

Constructor `SampleFormat::new(container, bits, signed, layout, align, endian) -> Result<Self>` validates:
- `U8`/`I8`: bits ≤ 8; `I16`: bits ≤ 16; `I32`: bits ≤ 32; `F32`: bits == 32 and signed; `Packed10`: bits == 10; `Packed12`: bits == 12; `Bits1`: bits == 1.
- `U8` must be `signed == false`; every other integer container must be `signed == true` (unsigned 16/32-bit formats are not supported; return `InvalidFormat`).
- `Planar` is not allowed for `Packed*` and `Bits1`.
Builder methods: `with_swap_iq(bool)`, `with_full_scale(FullScale)` (validates `Explicit(n)` with `n > 0`).
Getters for every field.

Helpers:
- `fn values_per_sample(&self) -> usize` — 2 for Interleaved/Planar, 1 for Real.
- `fn bytes_for_samples(&self, n: usize) -> Result<usize>` — exact byte count for `n` samples (complex samples for I/Q layouts). Packed rules (bit order is **LSB-first, little-endian**, values are written in the order I0, Q0, I1, Q1 …): `Packed12` packs 2 values into 3 bytes; `Packed10` packs 4 values into 5 bytes; `Bits1` packs 8 values per byte (bit 0 first). Return `Err(InvalidFormat)` if `n` does not fill whole groups (Packed12 Real: n even; Packed10 Interleaved: n even; Packed10 Real: n multiple of 4; Bits1 Interleaved: n multiple of 4, Real: multiple of 8). Use `checked_mul` and return `Overflow` on overflow.
- `fn samples_in_bytes(&self, bytes: usize) -> usize` — number of whole samples contained in `bytes`.
- `fn full_scale_value(&self) -> f64` — `2^(bits-1)` for `PowerOfTwo` (128 for U8 with offset), or the explicit value; `1.0` for F32.

Constants in `pub mod formats` (document the device each comes from):

| Constant | Container, bits, signed, layout, align | Notes |
| --- | --- | --- |
| `CU8` | U8, 8, unsigned, Interleaved, Lsb | RTL-SDR; zero at 127.5 |
| `CS8` | I8, 8, signed, Interleaved, Lsb | HackRF |
| `CS16` | I16, 16, signed, Interleaved, Lsb | Airspy HF+, LR2021 IQ16 (I in low half-word) |
| `CS12_LSB16` | I16, 12, signed, Interleaved, Lsb | PlutoSDR/AD9361 (sign-extended 12-bit in 16) |
| `SC16_Q11` | I16, 12, signed, Interleaved, Lsb, FullScale::Explicit(2048) | bladeRF |
| `CS16_PLANAR` | I16, 16, signed, Planar, Lsb | SDRplay API style (all I, then all Q) |
| `PACKED12` | Packed12, 12, signed, Interleaved, Lsb | LimeSDR-style wire packing |
| `PACKED10` | Packed10, 10, signed, Interleaved, Lsb | ESP-SDR CAP20-style packing (exact device bit order to be confirmed by its driver) |
| `I2S32_MSB` | I32, 24, signed, Interleaved, Msb | SX1255 mode B over I2S, 24 valid bits MSB-aligned (example) |
| `CF32` | F32, 32, signed, Interleaved, Lsb | files, SoapySDR CF32 |
| `RF32` | F32, 32, signed, Real, Lsb | real-valued audio / real sampling |
| `RS16` | I16, 16, signed, Real, Lsb | PCM audio |

Constants are built with a `const fn` constructor variant `SampleFormat::new_const(...)` that skips validation (private helper) — every constant is additionally checked by a unit test that calls the validating `new` and expects `Ok` and equality.

Semantics that M05's `convert` will implement (document them on `SampleFormat` now):
- **Value mapping:** amplitude = (raw − offset) / full_scale. `offset` is 127.5 for `U8`, 0 otherwise. For `Msb` alignment the effective value is `raw >> (container_bits − bits)` (arithmetic shift); for `Lsb` alignment the effective value is the low `bits` bits, sign-extended.
- **swap_iq:** if true, the first value of each pair is Q.
- **Planar:** the buffer holds all I values followed by all Q values; length must be even and both halves equal.
- **Endian::Big:** affects only the byte order of multi-byte containers in the *input data*; the host is always little-endian (R6.6).

### 3. `time.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SampleIndex(pub u64);
```
Methods: `checked_add(u64)`, `saturating_add(u64)`, `checked_sub(SampleIndex) -> Option<u64>`.

```rust
/// Exact rational rate in samples per second, always reduced, den > 0, num > 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rate { num: u64, den: u64 }
```
- `Rate::new(num, den) -> Result<Rate>` reduces by gcd; `num == 0 || den == 0` → `InvalidRate`.
- `Rate::hz(u64)` shorthand for `new(hz, 1)`.
- `fn as_f64(&self) -> f64`.
- `fn scale(&self, ratio: Ratio) -> Result<Rate>` — multiply by an exact ratio (used for decimation/interpolation); uses `u128` intermediates, reduces, returns `Overflow` if the reduced result does not fit `u64`.
- `fn samples_in_ns(&self, ns: u64) -> Result<u64>` — `floor(ns · num / (den · 1e9))` with `u128` math.
- `fn ns_for_samples(&self, n: u64) -> Result<u64>` — `floor(n · den · 1e9 / num)`.

```rust
/// Exact ratio out/in of a rate-changing kernel or block, e.g. 1/10 for decimation by 10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ratio { num: u64, den: u64 }
```
- `Ratio::new(num, den) -> Result<Ratio>` (reduced, both > 0), `Ratio::ONE`.
- `fn mul(self, other) -> Result<Ratio>` (u128, reduce, Overflow on overflow).
- `fn map_index(&self, input_index: u64) -> Result<u64>` — `floor(input_index · num / den)` (u128).
- `fn inverse(self) -> Ratio`.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] #[non_exhaustive]
pub enum TimeSource { Device, Host, Gps, Unknown }

/// "Sample `index` was taken at `unix_ns` nanoseconds since 1970-01-01T00:00:00Z."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeAnchor { pub index: SampleIndex, pub unix_ns: i64, pub source: TimeSource }
```
- `fn time_of(&self, index: SampleIndex, rate: Rate) -> Result<i64>` — `unix_ns + (index − self.index) · 1e9 · den / num`, computed in `i128` and rounded toward −∞ (`div_euclid`), works for indices before and after the anchor, `Overflow` if the result leaves `i64`.

Document in rustdoc why floats are never used for time (precision loss after days at MS/s rates).

### 4. `stream.rs`

```rust
/// Bit flags describing a block of samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BlockFlags(u16);
impl BlockFlags {
    pub const GAP_BEFORE: Self;   // samples are missing before this block
    pub const OVERFLOW: Self;     // RX: hardware dropped samples
    pub const UNDERRUN: Self;     // TX: hardware ran out of samples
    pub const BURST_START: Self;  // TX/RX: first sample of a burst
    pub const BURST_END: Self;    // TX/RX: last sample of a burst
    pub const LATE: Self;         // TX: a timed burst arrived too late
    pub const RETUNED: Self;      // frequency/rate/gain changed at the first sample of this block
    // methods: empty(), contains(), union(), insert(), remove(), bits(), from_bits_truncate()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockMeta {
    pub start: SampleIndex,          // index of the first sample in this block
    pub flags: BlockFlags,
    pub anchor: Option<TimeAnchor>,  // optional absolute time for some sample
    pub tx_time_ns: Option<i64>,     // TX: requested start time of a burst (unix ns)
}
```

Synchronous stream traits (drivers implement these; async variants come in M12):

```rust
pub trait RxStream {
    type Error: SdrkError;
    /// Raw format of the bytes delivered by `read`.
    fn format(&self) -> SampleFormat;
    /// Current sample rate.
    fn sample_rate(&self) -> Rate;
    /// Fill `buf` with whole samples; returns (bytes written, metadata of the block).
    /// Must write a whole number of samples (and whole packing groups).
    fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Self::Error>;
}

pub trait TxStream {
    type Error: SdrkError;
    fn format(&self) -> SampleFormat;
    fn sample_rate(&self) -> Rate;
    /// Write whole samples; returns bytes accepted.
    fn write(&mut self, buf: &[u8], meta: &BlockMeta) -> Result<usize, Self::Error>;
    /// Status events reported back by the device since the last call (underrun, late, burst done).
    fn poll_status(&mut self) -> Result<Option<BlockMeta>, Self::Error>;
}
```

### 5. Little-endian guard

`lib.rs`: `#[cfg(target_endian = "big")] compile_error!("SDRk supports little-endian targets only");`

### 6. Docs

`docs/device.md`: purpose of the crate, the value mapping rules, the packing rules with a byte diagram for Packed12 (bytes `b0 b1 b2` → `v0 = b0 | (b1 & 0x0F) << 8`, `v1 = (b1 >> 4) | b2 << 4`, both sign-extended from 12 bits) and Packed10 (40 bits LSB-first → 4 values), the format constant table, the time model (sample index, rate as fraction, anchors), and the stream contract.

## Test vectors

No golden files. Table-driven unit tests with hand-computed values (listed below) are sufficient because everything here is exact integer arithmetic.

## Tests

1. `Rate::new(2_400_000, 1).scale(Ratio::new(1, 10))` = `240_000/1`; `Rate::new(48_000,1)` vs `Rate::new(96_000, 2)` equal after reduction; zero numerator/denominator → `InvalidRate`.
2. `Ratio::new(1,10).mul(Ratio::new(1,5))` = 1/50; `map_index` for 1/10 at indices 0, 9, 10, 2^63 (u128 path, no overflow).
3. `samples_in_ns` / `ns_for_samples` round-trip at 2.4 MS/s for 1 s, 1 h and 10 years (no overflow, exact).
4. `TimeAnchor::time_of`: anchor at index 1_000_000 = 1_700_000_000_000_000_000 ns, rate 2.4 MS/s; index 3_400_000 → +1 s exactly; index 0 → `1_700_000_000_000_000_000 − 416_666_667` ns (rounded toward −∞); index near `u64::MAX` → `Overflow`.
5. `bytes_for_samples`: CU8 1000 samples = 2000 bytes; CS16 = 4000; PACKED12 1000 samples = 3000 bytes; PACKED10 1000 = 2500; PACKED10 999 → error; Bits1 interleaved 4 samples = 1 byte; overflow with `usize::MAX`.
6. Every constant in `formats` equals the value built with `SampleFormat::new(...)` followed by `with_full_scale(...)` / `with_swap_iq(...)` where the table lists one (e.g. `SC16_Q11`), and validates.
7. Invalid combinations rejected: U8 signed, I16 unsigned, F32 with 16 bits, Planar + Packed12, Explicit(0).
8. `BlockFlags` set operations.
9. Compile-time `Send + Sync` assertions for all public types.
10. Error `Display` strings are lowercase without trailing period; `kind()` mapping matches the comments above.

## Acceptance

R1.2 commands, plus `cargo xtask bare-metal` builds `sdrk-device` for `thumbv7em-none-eabihf`.

## Deliverables checklist

- [ ] `error`, `format`, `time`, `stream` modules with docs and examples
- [ ] All tests above
- [ ] `docs/device.md` written
- [ ] CHANGELOG and decisions log updated
