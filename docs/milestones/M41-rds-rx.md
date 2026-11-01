# M41 — RDS reception

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M40](M40-psk.md) · Next: [M42](M42-rds-tx.md)

## Goal

Decode the Radio Data System of FM broadcast: a block `rds_demod` (57 kHz subcarrier → soft data bits: mixing, filtering, chip timing, BPSK carrier loop, biphase and differential decoding), a block `rds_decode` (soft bits → RDS groups as packets: block synchronization with offset words and syndromes), `sdrk_proto::rds` (group parsing, programme service name and radiotext assembly), the flow `fm_rds_receiver` and the end-to-end test E2E-RDS against a reference signal from the Python generator.

## Why

RDS is the most common digital signal a broadcast receiver meets: station name, radiotext, programme type. It is BPSK at a low rate and uses the parts of M38 and M40 almost unchanged, which makes it the first real-world test of the digital chain.

## Order

M41 depends on M40 (`costas`, `psk` conventions; through it on M38 and M37). It can run in parallel with M39.1, M40.1, M43 and M44.
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

**In:** `sdrk_proto::rds` (§1); blocks and kinds `rds_demod` and `rds_decode`; the flow `examples/flows/fm_rds_receiver.json` with a YAML twin; the generator `tests/golden/sdrk_golden/cases/rds.py` (an independent RDS modulator in Python/NumPy) and its reference files; tests; `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, `CHANGELOG.md`, decisions log.

**Out:** error correction of RDS blocks (backlog; blocks with a wrong syndrome are marked invalid), RDS2 and the extra subcarriers, ODA/TMC decoding, clock-time and alternative-frequency interpretation beyond the raw group (backlog; `sdrk_proto::rds` parses group types 0A, 0B, 2A, 2B only), using the 19 kHz pilot as the phase reference (the carrier loop is enough).

## Specification

### 1. `sdrk_proto::rds`

- Constants: generator `g(x) = x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1` (`0x5B9`); offset words A `0x0FC`, B `0x198`, C `0x168`, C' `0x350`, D `0x1B4`.
- `checkword(info: u16, offset: Offset) -> u16` and `syndrome(block26: u32) -> u16` (the remainder of the 26-bit block; a block is valid for offset `o` when the syndrome equals the syndrome of `o`'s offset word: A `0x3D8`, B `0x3D4`, C `0x25C`, C' `0x3CC`, D `0x258`; test all five).
- `Group { blocks: [u16; 4], valid: [bool; 4] }`, `Group::pi()`, `group_type() -> (u8, Version)`, `tp()`, `pty()`.
- `RdsState::update(&Group)`: assembles PS (8 characters from the 2-character segments of 0A/0B, address in the low 2 bits of block B) and RT (64 characters from the 4-character segments of 2A, 2-character segments of 2B; the A/B flag clears the text); `ps() -> &str`, `rt() -> &str` once all segments arrived. Characters 0x20–0x7E map to ASCII, others to `?` (the full RDS character table is backlog).

### 2. `rds_demod`

In `f32` (the FM multiplex, the output of `quad_demod`, at `fs ≥ 128 kHz`; `rate_unsupported` below), Out soft data bits `f32` at 1187.5 bit/s. Composition, each step a kernel call per piece:

1. `interleave` (to complex), `rotate` by −57000 Hz.
2. `decimate` complex by `d = floor(fs / 19000)` with cutoff 2400 Hz (Kaiser design, 60 dB, transition 1500 Hz).
3. `symbol_sync` (Gardner, complex) at the chip rate 2375 Hz.
4. `costas` order 2 (`carrier_bw` 0.01): the BPSK chips are on the real axis.
5. Biphase: data bit `k` from the chips `c[2k]` and `c[2k+1]`: `u_k = (c[2k].re − c[2k+1].re) / 2`. The chip pairing is unknown: the block runs both pairings (even and odd) and keeps averages of `|u|` with time constant 32 bits; it outputs the pairing with the larger average and switches only when the other is larger by a factor 1.2 (hysteresis).
6. Differential decoding: `soft_k = −u_k·u_{k−1}` (a data 1 is a change of the transmitted bit; the 180° ambiguity of the carrier loop cancels).

Parameters: `sync_bw` (None, build, 0.01), `carrier_bw` (None, build, 0.01). Ratio `1187.5 / fs`, `RatioKind::Variable`. Cost `6 + 60 / d`.

### 3. `rds_decode`

In soft bits `f32`, Out `Packets` (8 bytes: the four 16-bit information words A, B, C, D, big-endian). A shift register of the last 26 hard bits:

- **Search:** at every bit, compute the syndrome; on a valid block for any offset, remember its position and offset; synchronized when a second valid block follows exactly 26 bits later with the next offset in the order A, B, C/C', D, A.
- **Synchronized:** read 26-bit blocks at the expected positions; a block is valid if its syndrome matches the expected offset (C or C' for the third block). After block D emit a packet: `CRC_OK` if all four blocks are valid, else `CRC_FAIL` (emitted only with `emit_bad`); flag bits 16–19 = validity of blocks A–D, bit 20 = C' seen (version B). Lose sync after more than `max_bad` (I64, build, 10) invalid blocks among the last 50, then search again.

`index` = input index of the first bit of block A. Parameters `emit_bad` (Bool, build, false), `max_bad`. Cost 2. `max_per_input` = 1/104.

### 4. Flow `fm_rds_receiver`

`rf` `cs16` at 480 kHz → `tune` = `ddc` (decimation 2, cutoff 100000) → two branches: `stereo` = `fm_stereo_demod` → `audio` (as `fm_stereo_receiver`), and `quad` = `quad_demod` (deviation 75000) → `rds` = `rds_demod` → `groups` = `rds_decode` → `rds` packets output. The example program `fm-receiver` (or `examples/README.md` with a short Rust snippet) prints PS and RT using `RdsState`.

### 5. Reference signal (`rds.py`)

An independent NumPy implementation of the RDS modulator after IEC 62106: groups → checkwords with offsets → differential encoding → biphase symbols shaped with the data-shaping filter `cos(π·f·Tb/4)` for `|f| ≤ 2/Tb` → 57 kHz carrier locked to three times the pilot phase, level 4 % of the multiplex; plus stereo audio (1 kHz left, 2 kHz right) and pilot as in M24; FM modulated at 240 kHz and written as `cs16` at 480 kHz with a carrier offset of 0. Groups: PI `0xD123`, PTY 10, PS `SDRK TST`, RT `Hello from sdrk!` (2A), alternating 0A and 2A. Seed 570 for the noise. The files are generated by `generate.py` (not committed if larger than the case limit; the test generates them on the fly with the same code via `--emit rds_reference`).

## Tests

1. `sdrk-proto`: checkwords and syndromes of all offsets; a known group (from the reference) parses to PI/PTY/PS segment; `RdsState` assembles PS and RT, A/B flag clears RT.
2. **E2E-RDS (`e2e_rds_reference`):** the reference signal of §5 at an SNR of 30 dB (seed 571) through `fm_rds_receiver`: PS `SDRK TST` complete within 2 s, RT complete within 6 s, at least 99 % of the groups after the first second `CRC_OK`; the stereo audio is unaffected (the same as without RDS in the signal within the noise).
3. Robustness: carrier offset of 2 kHz in the RF (fixed by `ddc` tuning in the test), 100 ppm clock drift (seed 572), 15 dB SNR with at least 90 % valid groups (seed 573).
4. `rds_decode` with injected bit errors (seed 574): no packet with `CRC_OK` contains a wrong block; resynchronization after a 1 s dropout within 0.2 s.
5. Chunk invariance of `rds_demod` and `rds_decode` with 20 plans (seed 575).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-proto`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`.

## Deliverables checklist

- [ ] `sdrk_proto::rds` (checkwords, syndromes, groups, PS/RT state)
- [ ] Blocks and kinds `rds_demod`, `rds_decode`
- [ ] Reference generator `rds.py`
- [ ] Flow `fm_rds_receiver` with YAML twin; PS/RT display example
- [ ] Tests (seeds 570–575)
- [ ] `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, CHANGELOG, decisions log
