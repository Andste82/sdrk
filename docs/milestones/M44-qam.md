# M44 — QAM receive and transmit (16-QAM, 64-QAM)

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M43](M43-equalizer.md) · Next: [BACKLOG](BACKLOG.md)

> **Amended on 2026-11-08** (before implementation) to match the delivered M40–M43 and the maintainer's decisions: the order of `qam_demod` follows D825, the carrier loop of QAM uses GNU Radio's decision-directed phase detector, `costas` gets the parameter `constellation`, `frame_decode` the parameter `rotations` with an exact bit map for square QAM, and acceptance and deliverables include the count entries and the cycles per sample. The choices against GNU Radio's own QAM receiver are stated where they differ.

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
6. Each case (JSON + all its `.bin` arrays) stays at or below 512 KiB; all vectors together at or below 32 MiB. Prefer N ≤ 4096 samples; use longer inputs only when the kernel needs them (long filters, decimation).
7. Tests that use `sdrk-testkit` live in integration tests (`<crate>/tests/*.rs`) or in `crates/testing/sdrk-testkit/tests/`, never in `#[cfg(test)]` unit modules of a crate the testkit itself depends on (a dev-dependency cycle would compile two copies of that crate and produce confusing trait errors).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.


## Scope

**In:** the constellations `qam16` and `qam64` (mapping table in `sdrk-core::kernels::constellation` shared by modulator, slicer, equalizer and carrier loop); `costas` parameter `constellation` with the QAM constellations (decision-directed with the QAM decision, replacing `order`); blocks and kinds `qam_mod`, `qam_demod`; `frame_decode` parameters `rotations` (replacing `polarity`) and `bits_per_symbol`; count entries and cycles per sample of the new paths and blocks; `slice` and `equalize` constellations `qam16`, `qam64`; flows `qam16_transmitter` and `qam16_receiver` with YAML twins; tests; `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/performance.md`, `examples/README.md`, `CHANGELOG.md`.

**Out:** 256-QAM and above (backlog), APSK (backlog), FEC (backlog), Q15 (backlog: the dynamic range of 64-QAM needs care in Q15).

## Specification

### 1. Constellations

Square QAM with `M = 16` or `64`, `k = log2 M` bits per symbol, `m = √M` levels per axis. The first `k/2` bits select the in-phase level, the last `k/2` bits the quadrature level, each by the Gray code of the level index: level index `i` (0 … m−1) has the Gray code `i XOR (i >> 1)`, the level value is `2i − (m − 1)`. Bit order: most significant bit first. The points are scaled by `1/√E` with `E = 2·(M − 1)/3` (unit average power). Constants for `equalize`: `R2 = E|s|⁴ / E|s|²` (16-QAM: 1.32, 64-QAM: 1.381; compute exactly in the code and test against these values to 3 digits). Decision: per axis, round `(y·√E + (m−1)) / 2` to the nearest integer and clamp to `0 … m−1` (a grid slicer, as GNU Radio's `constellation_rect`).

Rotation by 90°: `j·y` maps the in-phase level index `i` to the quadrature axis and the quadrature level index `q` to `m − 1 − q` on the in-phase axis; with the reflected Gray code of this section, `m − 1 − i` has the Gray code of `i` with only its most significant bit inverted. So a rotation by +90° maps the bit groups `(I bits, Q bits)` of a symbol to `(Q bits with the first bit inverted, I bits)`; two and three rotations follow by repetition. For QPSK (one bit per axis) this is the known map of the four QPSK rotations. The unit test checks the map against the points for every symbol and rotation.

### 2. Soft bits (`slice` `qam16`, `qam64`)

Per axis with the max-log rule, simplified to the standard piecewise-linear form for Gray square QAM (GNU Radio's `calc_soft_dec` computes the exact log-sum over all points instead; the piecewise-linear form is the usual cheaper approximation and is kept): for 16-QAM per axis with `v = y·√E` (levels −3, −1, 1, 3): `s_0 = ±scale·v` and `s_1 = ±scale·(2 − |v|)`, where each sign is chosen so that **a soft bit > 0 means 1** for the mapping of §1 (with the Gray code of §1 the most significant bit is 1 on the positive side, so `s_0 = +scale·v`; the second bit is 1 for the inner levels −1 and +1, so `s_1 = +scale·(2 − |v|)`). For 64-QAM the three bits per axis use `v`, `4 − |v|` and `2 − ||v| − 4|`, with the signs chosen by the same rule. The unit test derives the expected signs from the mapping of §1 for every point, so the formulas and the mapping cannot disagree.

### 3. Carrier recovery for QAM

`costas` gets the parameter `constellation` (Choice `bpsk`/`qpsk`/`8psk`/`qam16`/`qam64`, build, default `bpsk`) in place of `order` (a hard break before 1.0: every caller and test is changed, no alias; `bpsk`, `qpsk` and `8psk` keep the phase detectors of M40/M40.1 unchanged). For `qam16` and `qam64` the loop is decision-directed: `d` = the decision of §1 on the derotated sample `y`, the phase error `e = arg(y·conj(d))` (the sign convention of the M40 loop), the exact phase as GNU Radio's `constellation_receiver_cb` computes it (`decision_maker_pe`), without a normalization by `|d|²`; the loop filter and NCO of M40. The `atan2` may use the kernel's existing fast arctangent if its contract allows it (log the choice). Lock detector as in M40.1 with `d`. The golden case of test 2 is the float32 model of this definition.

### 4. `qam_mod` and `qam_demod`

- `qam_mod`: In soft bits (±1), Out C32: mapping of §1, pulse shaping as `psk_mod` (RRC, `√sps` scaling). Parameters `order` (I64, build, 16; 16 or 64), `samples_per_symbol`, `rolloff`, `span`. Ratio `sps / k`.
- `qam_demod`: In C32 (after `agc`), Out soft bits: RRC matched filter → `symbol_sync` (Gardner) → `equalize` (constellation `qam16`/`qam64`, `taps` from `equalizer_taps`, default 11; CMA first, then decision-directed with the kernel's phase tracker of D829) → `costas` (constellation `qam16`/`qam64`) → `slice`. This is the order of `psk_demod` (D825, after GNU Radio's PSK tutorial and liquid-dsp's blind equalizer example), not the order of the first draft (carrier loop before the equalizer). Parameters `order`, `symbol_rate_hz`, `rolloff`, `span`, `equalizer_taps`, `sync_bw`, `carrier_bw` (default 0.005). The 90° ambiguity is resolved at the sync word: `frame_decode`'s parameter `polarity` becomes `rotations` (Choice `normal`/`both`/`four`, build, default `normal`; a hard break: `qpsk_receiver.json`/`.yaml` and every test that sets `polarity` are changed, no alias), and `frame_decode` gets `bits_per_symbol` (I64, build, 1 … 6, default 2; even for `four`). `normal` and `both` behave as `polarity` did. `four` tests the sync word in the four rotations of §1 applied to every group of `bits_per_symbol` soft bits (swap of the I and Q halves and inversion of the first bit of a half, a soft bit inverted by its sign), decodes the frame under the matching inverse rotation and reports the rotation in the packet flags (bits of the flags word to be chosen and logged). GNU Radio resolves the ambiguity differently, by differential coding of the whole symbol modulo M with a quadrant layout and no Gray code (`qam.py`, `diff_encoder_bb`); that roughly doubles the bit errors and would miss the coherent Gray bound of tests 3 and 4, so the coherent Gray mapping with the sync-word rotation stays.

### 5. Flows

`qam16_transmitter`: `msg` packets (`rate_hz` 100000) → `frame_encode` → `qam_mod` (16, sps 8) → `duc` (interpolation 5) → `rf` `cs16`. `qam16_receiver`: `rf` `cs16` at 1 MHz → `ddc` (decimation 5) → `agc` → `qam_demod` (16, symbol rate 25000) → `frame_decode` (`rotations` four) → `packets`.

## Tests

1. Constellation tables: mapping of every point, unit average power, `R2` values, decision rule at boundaries; soft-bit signs for every point (test derives them from the mapping); the rotation map of §1 for every symbol and rotation (16- and 64-QAM, and QPSK); `frame_decode` `four` finds the sync word and decodes the payload under each of the four rotations for `bits_per_symbol` 2, 4 and 6; `costas` `constellation` `bpsk`/`qpsk`/`8psk` give bit-identical output to the former `order` 2/4/8.
2. Golden: `costas/dd_qam16_c32_8192` (seed 600), `equalize/dd_qam16_c32_16384` (seed 601), float32 model.
3. **16-QAM BER (AWGN):** within 1 dB of the Gray-coded approximation `P_b ≈ (4/k)·(1 − 1/√M)·Q(√(3k/(M−1)·Eb/N0))` at Eb/N0 12 and 14 dB over 10^6 bits (seed 602).
4. **64-QAM BER:** same formula, within 1.5 dB at Eb/N0 18 and 20 dB (seed 603).
5. **Multipath:** 16-QAM through `Multipath::mild()` with `equalizer_taps` 15: within 2.5 dB of the AWGN result (seed 604); without equalizer the BER is above 1e-2.
6. **Offset and drift:** frequency offset 0.001 cycles per symbol and 20 ppm drift, acquisition within 5000 symbols (seed 605).
7. **Flow loop:** `qam16_transmitter` → channel (Eb/N0 16 dB, mild multipath, offset 1 kHz; seed 606) → `qam16_receiver`: at least 990 of 1000 packets `CRC_OK`.
8. Chunk invariance of `qam_mod`, `qam_demod` with 20 plans (seed 607).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; the new tests of this milestone (`sdrk-testkit`, `sdrk-blocks`, `sdrk-pipeline`; the whole suite and the emulators run in CI); `cargo xtask schema --check`; `cargo xtask count` entries for the QAM paths of `costas` and `equalize` and the blocks `qam_mod`, `qam_demod` on every platform the existing PSK entries are counted on, with their simulated cycles per sample (`cargo measure`, reference and counter model on every level with a core model; instructions only where no cycles exist) recorded in `docs/performance.md`, against a like-for-like reference where one exists (for example a C translation of GNU Radio's `constellation_receiver_cb` for the carrier loop, as `benches/refs/gr-costas.c` is for `costas`, or liquid-dsp's `modemcf` for the mapping and soft bits; the choice and reason are logged).

## Deliverables checklist

- [x] Shared constellation tables `qam16`, `qam64`
- [x] `slice`, `equalize` and `costas` (`constellation`, QAM decision-directed) extensions
- [x] Blocks and kinds `qam_mod`, `qam_demod`; `rotations` and `bits_per_symbol` of `frame_decode` (and the renamed callers)
- [x] Count entries and cycles per sample of the QAM paths and blocks, with the reference
- [x] Flows `qam16_transmitter`, `qam16_receiver` with YAML twins
- [x] Tests (seeds 600–607)
- [x] `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/performance.md`, `examples/README.md`, CHANGELOG, decisions log

## Status

Delivered (2026-11-09, branch `m44`; decisions D1200 to D1220, numbered D980 to D1000 on the branch; D1221 records the renumbering when M44 was rebased onto main). Every section (1 to 5) and every test (1 to 8, seeds 600 to 607) is implemented; no part of the specification is left out. The renames of §3 and §4 are hard breaks without aliases: `costas` `order` is `constellation`, `frame_decode` `polarity` is `rotations`, with every caller, test, flow and doc changed.

Cycles per sample (simulated by llvm-mca with the corrections of D942, whole-loop mode, reference / counter model, measured on main on 2026-11-09, D1221; the unit is per input symbol for `slice` and the two kernels, per output sample for `qam_mod`, per input sample for `qam_demod`, per soft bit for `frame_decode` `four`; `simd128` has no core model, so its column is instructions):

| entry | `sse2` | `sse4.2` | `avx2` | `avx512` | `neon` | `simd128` (instructions) |
| --- | --- | --- | --- | --- | --- | --- |
| `slice_qam16_block` | 1.73 / 2.05 | 2.71 / 2.92 | 0.75 / 1.02 | 0.53 / 0.51 | 3.78 / 2.53 | 18.02 |
| `slice_qam64_block` | 4.57 / 4.64 | 4.07 / 4.31 | 2.70 / 2.19 | 1.07 / 1.10 | 6.06 / 3.56 | 51.02 |
| `costas_qam16_fallback` | 173.21 / 148.76 | 174.73 / 149.35 | 141.04 / 163.07 | 146.80 / 177.29 | 187.82 / 178.73 | 445.38 |
| GNU Radio `constellation_receiver_cb` | 206.53 / 181.45 | 206.53 / 181.45 | 166.48 / 172.91 | 158.60 / 172.91 | 194.17 / 175.12 | 471.28 |
| `equalize_qam16_fallback` | 172.34 / 178.76 | 170.35 / 177.08 | 135.09 / 131.78 | 131.57 / 137.19 | 273.82 / 196.27 | 1090.47 |
| GNU Radio `linear_equalizer` (`constellation_rect`) | 191.52 / 213.78 | 191.52 / 213.78 | 944.25 / 277.26 | 938.01 / 277.22 | 244.41 / 243.20 | 1952.06 |
| `qam_mod_block` (16-QAM, 8 sps) | 14.04 / 16.36 | 14.04 / 16.36 | 7.99 / 7.38 | 6.63 / 6.10 | 17.40 / 16.67 | 108.55 |
| `qam_demod_block` (block defaults) | 116.95 / 106.84 | 117.18 / 106.04 | 67.94 / 74.05 | 66.71 / 71.76 | 136.37 / 100.98 | 571.70 |
| `frame_decode_four_block` | 34.77 / 33.53 | 34.82 / 33.27 | 23.43 / 20.77 | 22.08 / 21.17 | 47.75 / 45.47 | 166.41 |

The facade entries `costas_qam16` and `equalize_qam16` equal their fallbacks (both kernels are recursions without a SIMD branch, information only). The `slice` and block entries are our own code as information (no library computes the same work, D1201, D1216). The like-for-like references of the two kernels are the GNU Radio C translations of D1210; `costas_qam16_fallback` is 3 to 16 % faster on the reference models and at most 3 % slower on the counter-models (since M44 is on main with the shorter chain of D980; on the branch 3 to 17 % slower on the reference models), `equalize_qam16_fallback` is faster except at `neon` on the reference model (1.12, open item I44-7). `cargo xtask check cycles` passes: no SIMD entry is slower than its own fallback. The PSK paths keep their cycles (equalizer within 3.6 %, D1211; `frame_decode` `normal`/`both` within 0.02 cycles per bit, D1220). Accuracy: the golden cases `costas/dd_qam16_c32_8192` and `equalize/dd_qam16_c32_16384` are bit-exact against the `f32` model of the Python generator, with the `f64` guards 132.0 and 126.4 dB; the soft bits of `slice` are 149.2 (`qam16`) and 146.0 dB (`qam64`) from the `f64` definition on every level.

Measured values of the thresholds of tests 3 to 7:

| test | threshold | measured |
| --- | --- | --- |
| 3: 16-QAM BER at Eb/N0 12 dB | 5.65e-4 (1 dB edge) | 1.73e-4 |
| 3: 16-QAM BER at Eb/N0 14 dB | 2.42e-5 (1 dB edge) | 3.0e-6 |
| 4: 64-QAM BER at Eb/N0 18 dB | 1.03e-4 (1.5 dB edge) | 1.1e-5 |
| 4: 64-QAM BER at Eb/N0 20 dB | 2.0e-6 (1.5 dB edge) | 0 errors in 10^6 bits |
| 5: `mild`, 15 equalizer taps, 14 dB | 2.91e-4 (2.5 dB edge) | 4.0e-5 |
| 5: `mild` without the equalizer | above 1e-2 | 0.44 |
| 6: 0.001 cycles per symbol, 20 ppm | lock within 5000 symbols, then the AWGN edge 2.42e-5 | locked at symbol 5000, 1.0e-5 |
| 7: flow loop, seed 606 | at least 990 of 1000 packets `CRC_OK` | 992 at each of the four input rotations (open item I44-4) |

Deviations and judgement calls of the specification, all in the decisions log: the exact `libm::atan2f` in the QAM detector (no fast arctangent exists, D1207); the QAM lock metric `cos(8 err)` (D1208); the boundary and NaN rule of the grid decision (D1202); the slicer without a SIMD kernel, compiler-vectorized (D1204, open item I44-1); no `max_offset` and no feedback taps in `qam_demod` and its `sync_bw` default 0.005 (D1214, D1215); the flag bits 16 and 17 and the separate `RotatingDecoder` of `frame_decode` `four` (D1213); random bits instead of PRBS-15 and the reading of "within x dB" in tests 3 to 6 (D1217); the extra `scale` node of `qam16_transmitter` and the filler packets of test 7 (D1218); the receiver settings `carrier_bw` 0.1, `equalizer_taps` 15, `sync_bw` 0.002 of `qam16_receiver` (D1219); count entries for the `qam16` workload only (the `qam64` paths share the code).

Progress by step:

- **Done (step 1, sections 1 and 2):** the shared constellations `sdrk_core::kernels::constellation` (`Qam::Qam16`, `Qam::Qam64`: Gray mapping with the in-phase bits first, unit power, the level and point tables, `R2` computed exactly as 33/25 and 29/21, the grid decision, `rotate_symbol` for the 90-degree rotation map; D1202, D1203) and the `slice` constellations `qam16` and `qam64` (the piecewise-linear max-log soft bits over the components in memory order, kind cost 10 and 18; D1200, D1204). Test 1 parts without `costas`, `equalize` and `frame_decode`: the unit tests of `kernels::constellation` (mapping of every point, unit power, `R2` to 3 digits, the decision at the boundaries and for the extremes, the rotation map for every symbol and rotation of 16-QAM, 64-QAM and QPSK) and `sdrk-blocks/tests/qam_blocks.rs` (the soft-bit signs of every point derived from the mapping, the `f32` definition bit for bit, the signs against the grid decision, 20 cut plans of seed 607, the extremes). Count entries `slice_qam16_block` and `slice_qam64_block` on all four platforms, information only (no library reference, D1201); simulated cycles per input symbol, reference / counter model: `qam16` `sse2` 1.73 / 2.05, `sse4.2` 2.71 / 2.92, `avx2` 0.75 / 1.02, `avx512` 0.53 / 0.51, `neon` 3.78 / 2.53; `qam64` 4.57 / 4.64, 4.07 / 4.31, 2.70 / 2.19, 1.07 / 1.10, 6.06 / 3.56 (`simd128`: 18.02 and 51.02 instructions per symbol). Open: I44-1 (whether the slicer gets a kernel with SIMD paths).
- **Done (step 2, section 3 and the `equalize` part):** `costas` takes `CostasConstellation` (`bpsk`, `qpsk`, `8psk`, `qam16`, `qam64`) instead of `order` in the kernel, the block and the kind, a hard break without an alias; every caller, test and doc is changed (D1205); the golden cases of M40 and M40.1 pass unchanged, so `bpsk`, `qpsk` and `8psk` give the bits of the former orders 2, 4 and 8 (test 1). The QAM carrier loop is decision-directed with the exact phase `e = libm::atan2f(im(y conj d), re(y conj d))` of GNU Radio's `constellation_receiver_cb` against the grid decision (no fast arctangent exists; the generator transcribes msun's `atan2f`, D1207) and the lock metric `cos(8 err)` with the QAM decision (D1208); kind cost 45 (D1206). `equalize` has the constellations `qam16` and `qam64` (the grid decision, the CMA radius `33 / 25` and `29 / 21` as `f32`, D1209). Contracts in `docs/kernels.md`. Golden cases `costas/dd_qam16_c32_8192` (seed 600, `snr_db_guard` 132.0 dB) and `equalize/dd_qam16_c32_16384` (seed 601, 126.4 dB), bit-exact through facade, `Fallback` and `Reference`, whole and in 20 chunk plans (test 2). Unit tests `sdrk-core/tests/kernels_m44.rs`, block tests in `qam_blocks.rs` (cut plans of seed 607), kind tests, the testkit suite (`costas_c32`, `extremes_m40` with QAM), the SIMD backend's equalizer tests with QAM. Count entries `costas_qam16`, `costas_qam16_fallback`, `equalize_qam16`, `equalize_qam16_fallback` on all four platforms against C translations of GNU Radio's `constellation_receiver_cb` and `linear_equalizer` on a `constellation_rect` (`benches/refs/gr-constellation.c`, D1210). Simulated cycles per symbol, ours / GNU Radio, reference model, on main (D1221; the table above): `costas_qam16_fallback` `sse2` 173.21 / 206.53, `sse4.2` 174.73 / 206.53, `avx2` 141.04 / 166.48, `avx512` 146.80 / 158.60, `neon` 187.82 / 194.17 (counter-model 148.76 / 181.45, 149.35 / 181.45, 163.07 / 172.91, 177.29 / 172.91, 178.73 / 175.12); `equalize_qam16_fallback` 172.34 / 191.52, 170.35 / 191.52, 135.09 / 944.25, 131.57 / 938.01, 273.82 / 244.41 (counter-model 178.76 / 213.78, 177.08 / 213.78, 131.78 / 277.26, 137.19 / 277.22, 196.27 / 243.20); `simd128` instructions 445.38 / 471.28 and 1090.47 / 1952.06. On the branch, before the rebase, `costas_qam16_fallback` simulated 215.22 / 206.52, 216.73 / 206.52, 161.05 / 137.55, 170.79 / 158.58 and 249.70 / 237.53 on the reference models. The PSK paths keep their cycles (D1211). Open: I44-2 (the 64-QAM lock flag on noise).
- **Done (step 3, section 4):** the blocks and kinds `qam_mod` (`QamModBlock`: the Gray points of `kernels::constellation`, root-raised-cosine shaped as `psk_mod`; `order` 16 or 64, `samples_per_symbol` required, `rolloff`, `span`; cost `4 span`) and `qam_demod` (`QamDemodBlock`: matched filter, `symbol_sync`, `equalize` with `qam16`/`qam64` and its phase tracker at `carrier_bw`, `costas` with `qam16`/`qam64`, the soft bits of `slice`, the order of `psk_demod`, D825; `equalizer_taps` default 11 (0 = off), `carrier_bw` default 0.005, `sync_bw` default 0.005 (D1215), no `max_offset` and no feedback taps (D1214); cost `4 span + (55 + 10 or 18) / sps + 8 equalizer_taps / sps`, D1212), registered, described, in the schema, `docs/blocks.md` and `docs/pipeline.md`. `frame_decode` takes `rotations` (`normal`, `both`, `four`; a hard break of `polarity`, `qpsk_receiver.json`/`.yaml` and every test changed, no alias) and `bits_per_symbol` (1 to 6, default 2, even for `four`); `four` tests the sync word under the four inverse rotations of §1 of every symbol, reads the frame under the matching one and reports the quarter turns in bits 16 and 17 of the packet flags (`sdrk_proto::frame::RotatingDecoder`, `rotation_of`; kind cost 4; D1213); `normal` and `both` keep their bits and their cost (`frame_decode` 26.15 and `frame_decode_block` 24.15 instructions per bit at `avx2`, base 26.12 and 24.12). Tests: the `frame_decode` part of test 1 (`sdrk-blocks/tests/qam_blocks.rs`: every rotation for 2, 4 and 6 bits per symbol and four start positions, five frames of different rotations in 20 cut plans, the parameter checks, `rotate_group` equal to `rotate_symbol`, a QAM link turned by 0 to 270 degrees with rotations stepping with the turn; `sdrk-proto/tests/frame.rs`), test 8 (`qam_mod` and `qam_demod` equal their kernels whole and in 20 cut plans of seed 607, also short outputs, `reset`, timing and parameters; `sdrk-pipeline/tests/qam_kinds.rs` push mode equal to the blocks), tests 3 to 6 (`sdrk-testkit/tests/qam_ber_m44.rs`, D1217; measured: 16-QAM 1.73e-4 at 12 dB and 3.0e-6 at 14 dB against the 1 dB edges 5.65e-4 and 2.42e-5; 64-QAM 1.1e-5 at 18 dB and 0 errors at 20 dB against the 1.5 dB edges 1.03e-4 and 2.0e-6; `mild` with 15 taps 4.0e-5 against the 2.5 dB edge 2.91e-4 and 0.44 without the equalizer; offset 0.001 cycles per symbol with 20 ppm drift locked at symbol 5000 and 1.0e-5 from there against 2.42e-5). Count entries `qam_mod_block`, `qam_demod_block` and `frame_decode_four_block` on all four platforms, our own code as information (D1216). Simulated cycles per sample, reference / counter model: `qam_mod_block` (per output sample) `sse2` 14.04 / 16.36, `sse4.2` 14.04 / 16.36, `avx2` 7.99 / 7.38, `avx512` 6.63 / 6.10, `neon` 17.40 / 16.67; `qam_demod_block` (per input sample) 116.95 / 106.84, 117.18 / 106.04, 67.94 / 74.05, 66.71 / 71.76, 136.37 / 100.98 (on main, D1221; on the branch 122.28 / 110.32, 122.48 / 109.58, 70.70 / 78.50, 67.95 / 77.58, 149.75 / 101.38); `frame_decode_four_block` (per soft bit) 34.77 / 33.53, 34.82 / 33.27, 23.43 / 20.77, 22.08 / 21.17, 47.75 / 45.47; `simd128` instructions 108.55, 571.70 and 166.41. Open: I44-3 (64-QAM pulls in only small carrier offsets from a blind start).
- **Done (step 4, section 5 and test 7):** the flows `qam16_transmitter` (`msg` packets at 100000 bit/s -> `frame_encode`, idle `preamble` -> `qam_mod` 16, 8 samples per symbol -> `level` = `scale` -6 dB, not in §5, so that the pulse peaks of about 1.5 fit into `cs16`, measured peak 0.751 -> `duc` by 5 -> `rf` `cs16`; D1218) and `qam16_receiver` (`rf` `cs16` at 1 MHz -> `ddc` by 5 -> `agc`, `reference_db` 0, attack and release 0.02 s -> `qam_demod` 16 at 25000 Bd with `carrier_bw` 0.1, `equalizer_taps` 15, `sync_bw` 0.002 -> `frame_decode` `rotations` four, `bits_per_symbol` 4 -> `packets`; D1219), with YAML twins, the canonical round trip (`model`, `yaml`, `schema` tests), the WASM validation list and `examples/README.md`. Test 7 (`sdrk-pipeline/tests/qam_loop.rs`, seed 606; filler packets around the counted ones, D1218): 992 of 1000 packets with `CRC_OK` and the payload unchanged at each of the four 90 degree rotations of the input (threshold 990). Measured numbers of every threshold: test 3, 16-QAM BER 1.73e-4 at 12 dB and 3.0e-6 at 14 dB against the 1 dB edges 5.65e-4 and 2.42e-5; test 4, 64-QAM 1.1e-5 at 18 dB and 0 errors at 20 dB against the 1.5 dB edges 1.03e-4 and 2.0e-6; test 5, `mild` with 15 taps 4.0e-5 against the 2.5 dB edge 2.91e-4, 0.44 without the equalizer (above 1e-2); test 6, locked at symbol 5000 and 1.0e-5 from there against 2.42e-5; test 7, 992 of 1000 against 990. No new kernel path or block in this step: the cycles of every M44 count entry are those of steps 1 to 3 above. Open: I44-4 (the small margin of test 7: the 0.04 cycles per symbol of the test need `carrier_bw` 0.1, whose phase jitter costs about 1 % of the packets; other seeds 993, 992 and 986) and I44-5 (the idle signals of `frame_encode` are one constant QAM symbol).
- **Done (review findings):** the cycles of `frame_decode` and `frame_decode_block` (`normal`/`both`) measured again after the change of D1213 on every level with a core model, within 0.02 cycles per bit of the base commit simulated on the same host (`avx2` 5.91 / 6.18 and 6.92 / 5.80, D1220); the regression of the PSK equalizer (+0.1 to +3.6 % in cycles) named and accepted in D1211; the leftover `costas` `order` wording replaced in code comments, the `describe` text of `rds_demod` and the docs; Send/Sync assertions for `CostasConstellation`, `Qam` and `RotatingDecoder`. Open: I44-6 (`frame_decode_block` +7.1 % instructions at `simd128`, no core model).
- **Done (on main, 2026-11-09):** the 7 commits of the branch rebased onto main (M45.1 and the `costas` chain of D980); the decisions renumbered from D980 to D1000 to D1200 to D1220 (D1221). The `costas` fallback keeps both changes: the constellation dispatch of M44 (`run::<CODE>`, the QAM loops in `run_qam`) and the forms of D980 (`detector4` for QPSK, `integrate`, `phase_step_f32` for every constellation). Every M44 count entry and the PSK `costas`, `equalize` and `frame_decode` entries counted again on all four platforms and simulated again on every level with a core model with the corrected `llvm-mca` of D942 (the table above; the branch had simulated with the uncorrected tool of the development container). The PSK paths against the committed rows of main, reference / counter model: `costas` equal at every level (`neon` 86.13 / 54.09 against 86.13 / 54.10), `equalize` between 3.1 % faster and 1.0 % slower (D1211), `frame_decode` and `frame_decode_block` within +0.03 cycles per bit. `costas_qam16_fallback` gains the shorter chain of D980: 0.84 to 0.97 of GNU Radio's cycles on the reference models (on the branch 1.04 to 1.17).

Verified locally on Linux x86-64 (2026-11-09, final gate on the branch head): `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo xtask check` (including `check cycles`), `cargo xtask bare-metal`, `cargo xtask schema --check`, `python tests/golden/generate.py --check` (911 files), the `sdrk-pipeline` tests built with `--no-default-features`, `benches/test_cycle_tables.py`, and the M44 test files and those touched by the renames: `sdrk-core` `kernels_m44`, `kernels_m40`, `kernels_m43` and the `constellation` unit tests; `sdrk-blocks` `qam_blocks`, `psk_blocks`, `rds_blocks`, `equalize_blocks`, `symbol_blocks`, `describe`, `construction`, `packet_ports`; `sdrk-proto` `frame`, `packet`; `sdrk-pipeline` `qam_kinds`, `qam_loop`, `psk_kinds`, `qpsk_loop`, `equalize_kinds`, `rds_kinds`, `symbol_kinds`, `schema`, `yaml`, `model`, `packets`; `sdrk-testkit` (`--features std`) `qam_ber_m44`, `golden_m40`, `golden_m40_1`, `golden_m43`, `suite_m40`, `costas_lock_m40`; `sdrk-backend-simd` `equalize`; `sdrk` `facade`; the `xtask` unit tests.

Not done or not run:
- The whole workspace test suite, the emulators, the wasm tests (`bindings/wasm/tests/wasm.rs` validates the two new flows; compiled by clippy, not run) and the other hosts; CI runs them.
- No counted flow chain of the QAM flows in `benches/results/pipeline.json` (the QPSK flows have none either).
- `cargo clippy` on the separate `benches` workspace still fails on files M44 did not touch (`cref-sys` `lib.rs`, `liquid_nco.rs`, `halfband.rs`); CI runs `cargo check` there, which passes.

Open items for the maintainer (docs/open-items.md): I44-1 (no SIMD branch for the QAM soft bits of `slice`), I44-2 (the `costas` `qam64` lock flag does not separate 64-QAM from noise), I44-3 (64-QAM pulls in only small carrier offsets from a blind start), I44-4 (the small margin of test 7, 992 against 990), I44-5 (the idle signals of `frame_encode` are one constant QAM symbol), I44-6 (`frame_decode_block` +7.1 % instructions at `simd128`), I44-7 (`equalize_qam16_fallback` 12 % slower than GNU Radio at `neon` on `cortex-a72`; `costas_qam16_fallback`, 3 to 17 % slower on the branch, is faster on main with D980).
