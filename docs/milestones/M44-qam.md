# M44 — QAM receive and transmit (16-QAM, 64-QAM)

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M43](M43-equalizer.md) · Next: [BACKLOG](BACKLOG.md)

## Goal

Quadrature amplitude modulation in both directions: blocks `qam_mod` and `qam_demod` for 16-QAM and 64-QAM with Gray mapping, the constellations `qam16`/`qam64` in `slice` (soft bits) and `equalize` (decisions, CMA radius), a decision-directed carrier loop for QAM, and loop tests through noise and multipath.

## Why

QAM gives the highest data rate per bandwidth of the planned modes and is the last building block before the backlog topics (OFDM, strong FEC). It needs everything before it: timing (M38), carrier (M40), equalizer (M43).

## Order

M44 depends on M43 (through it on M40.1, M40, M38). It can run in parallel with M41 and M42.
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
6. Each case (JSON + all its `.bin` arrays) stays at or below 512 KiB; all vectors together at or below 16 MiB. Prefer N ≤ 4096 samples; use longer inputs only when the kernel needs them (long filters, decimation).
7. Tests that use `sdrk-testkit` live in integration tests (`<crate>/tests/*.rs`) or in `crates/testing/sdrk-testkit/tests/`, never in `#[cfg(test)]` unit modules of a crate the testkit itself depends on (a dev-dependency cycle would compile two copies of that crate and produce confusing trait errors).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.


## Scope

**In:** the constellations `qam16` and `qam64` (mapping table in `sdrk-core::kernels::constellation` shared by modulator, slicer, equalizer and carrier loop); `costas` mode `dd_qam` (decision-directed with the QAM decision); blocks and kinds `qam_mod`, `qam_demod`; `slice` and `equalize` constellations `qam16`, `qam64`; flows `qam16_transmitter` and `qam16_receiver` with YAML twins; tests; `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, `CHANGELOG.md`.

**Out:** 256-QAM and above (backlog), APSK (backlog), FEC (backlog), Q15 (backlog: the dynamic range of 64-QAM needs care in Q15).

## Specification

### 1. Constellations

Square QAM with `M = 16` or `64`, `k = log2 M` bits per symbol, `m = √M` levels per axis. The first `k/2` bits select the in-phase level, the last `k/2` bits the quadrature level, each by the Gray code of the level index: level index `i` (0 … m−1) has the Gray code `i XOR (i >> 1)`, the level value is `2i − (m − 1)`. Bit order: most significant bit first. The points are scaled by `1/√E` with `E = 2·(M − 1)/3` (unit average power). Constants for `equalize`: `R2 = E|s|⁴ / E|s|²` (16-QAM: 1.32, 64-QAM: 1.381; compute exactly in the code and test against these values to 3 digits). Decision: per axis, round `(y·√E + (m−1)) / 2` to the nearest integer and clamp to `0 … m−1`.

### 2. Soft bits (`slice` `qam16`, `qam64`)

Per axis with the max-log rule, simplified to the standard piecewise-linear form for Gray square QAM: for 16-QAM per axis with `v = y·√E` (levels −3, −1, 1, 3): `s_0 = ±scale·v` and `s_1 = ±scale·(2 − |v|)`, where each sign is chosen so that **a soft bit > 0 means 1** for the mapping of §1 (with the Gray code of §1 the most significant bit is 1 on the positive side, so `s_0 = +scale·v`; the second bit is 1 for the inner levels −1 and +1, so `s_1 = +scale·(2 − |v|)`). For 64-QAM the three bits per axis use `v`, `4 − |v|` and `2 − ||v| − 4|`, with the signs chosen by the same rule. The unit test derives the expected signs from the mapping of §1 for every point, so the formulas and the mapping cannot disagree.

### 3. Carrier recovery for QAM

`costas` gets the mode `dd_qam` (with the constellation): `d` = the decision of §1, `e = im(y·conj(d)) / |d|²` (the normalization keeps the loop gain equal for inner and outer points), the loop filter and NCO of M40. Lock detector as in M40.1 with `d`.

### 4. `qam_mod` and `qam_demod`

- `qam_mod`: In soft bits (±1), Out C32: mapping of §1, pulse shaping as `psk_mod` (RRC, `√sps` scaling). Parameters `order` (I64, build, 16; 16 or 64), `samples_per_symbol`, `rolloff`, `span`. Ratio `sps / k`.
- `qam_demod`: In C32 (after `agc`), Out soft bits: RRC matched filter → `symbol_sync` (Gardner) → `costas` (`dd_qam`) → `equalize` (constellation `qam16`/`qam64`, `taps` from `equalizer_taps`, default 11; CMA first) → `slice`. Parameters `order`, `symbol_rate_hz`, `rolloff`, `span`, `equalizer_taps`, `sync_bw`, `carrier_bw` (default 0.005). The 90° ambiguity is resolved by `frame_decode` `polarity` extended to `rotations` (Choice `normal`/`both`/`four`): `four` tests the sync word for the four rotations of the I/Q bit pairs; add this to `frame_decode`.

### 5. Flows

`qam16_transmitter`: `msg` packets (`rate_hz` 100000) → `frame_encode` → `qam_mod` (16, sps 8) → `duc` (interpolation 5) → `rf` `cs16`. `qam16_receiver`: `rf` `cs16` at 1 MHz → `ddc` (decimation 5) → `agc` → `qam_demod` (16, symbol rate 25000) → `frame_decode` (`rotations` four) → `packets`.

## Tests

1. Constellation tables: mapping of every point, unit average power, `R2` values, decision rule at boundaries; soft-bit signs for every point (test derives them from the mapping).
2. Golden: `costas/dd_qam16_c32_8192` (seed 600), `equalize/dd_qam16_c32_16384` (seed 601), float32 model.
3. **16-QAM BER (AWGN):** within 1 dB of the Gray-coded approximation `P_b ≈ (4/k)·(1 − 1/√M)·Q(√(3k/(M−1)·Eb/N0))` at Eb/N0 12 and 14 dB over 10^6 bits (seed 602).
4. **64-QAM BER:** same formula, within 1.5 dB at Eb/N0 18 and 20 dB (seed 603).
5. **Multipath:** 16-QAM through `Multipath::mild()` with `equalizer_taps` 15: within 2.5 dB of the AWGN result (seed 604); without equalizer the BER is above 1e-2.
6. **Offset and drift:** frequency offset 0.001 cycles per symbol and 20 ppm drift, acquisition within 5000 symbols (seed 605).
7. **Flow loop:** `qam16_transmitter` → channel (Eb/N0 16 dB, mild multipath, offset 1 kHz; seed 606) → `qam16_receiver`: at least 990 of 1000 packets `CRC_OK`.
8. Chunk invariance of `qam_mod`, `qam_demod` with 20 plans (seed 607).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask emu --target qemu-cortex-m4` (quick profile).

## Deliverables checklist

- [ ] Shared constellation tables `qam16`, `qam64`
- [ ] `slice`, `equalize` and `costas` (`dd_qam`) extensions
- [ ] Blocks and kinds `qam_mod`, `qam_demod`; `rotations` option of `frame_decode`
- [ ] Flows `qam16_transmitter`, `qam16_receiver` with YAML twins
- [ ] Tests (seeds 600–607)
- [ ] `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, CHANGELOG, decisions log
