# M57 — TETRA downlink synchronization: kernel `viterbi` and the BSCH

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M13](M13-simd-backend.md), [M37](M37-packet-ports.md), [M38](M38-symbol-sync.md), [M39.1](M39.1-afsk-ook-adsb.md) (JSON text packets as in `adsb_parse`), [M40.1](M40.1-psk-variants.md) (the pi/4-DQPSK mapping), [M44.1.1](M44.1.1-parameter-names.md) (`pulse_match`) · Next: [BACKLOG](BACKLOG.md) (TETRA: AACH, SYSINFO and the upper MAC)

> **Status: planned** (written 2026-11-28 after the analysis of a TETRA recording: [signal-analysis-notes.md](../signal-analysis-notes.md), [tetra-decoder-gaps.md](../tetra-decoder-gaps.md)). First milestone of TETRA and first user of the *Generic building blocks for framed digital modes* of the backlog: it builds the generic kernel `viterbi` (with `conv_encode`) and the generic bit helpers that the broadcast synchronization channel needs, each defined by its algorithm and parameters, with TETRA as the first preset. The decisions to confirm before the work starts are listed in §12.

## Goal

A receiver for the **broadcast synchronization channel (BSCH)** of a TETRA downlink: complex samples of one 25 kHz carrier in, the 60-bit SYNC PDU of every synchronization burst out, checked by its CRC and parsed into JSON (colour code, timeslot, frame and multiframe number, MCC, MNC and the rest of the SYNC PDU and of the D-MLE-SYNC it carries), together with the carrier error measured on the frequency correction field of the burst. On the way it delivers a generic soft-decision **Viterbi decoder** for convolutional codes of constraint length 3 to 9 with 2 to 4 outputs per input bit, its encoder, and generic bit-level helpers (a Fibonacci LFSR with an additive scrambler, the multiplicative block interleaver, the index puncturing rule, a CRC of any width over any number of bits), all with TETRA as the first preset.

## Why

The analysis of a recording at 427.977 MHz found three TETRA carriers: the frequency correction field and the synchronization training sequence of EN 300 392-2 match in 301 slots per carrier, with 4 to 5 wrong bits of 38 where 19 are expected at random. SDRk can already produce the symbols of such a signal (`pulse_match`, `symbol_sync`, the pi/4-DQPSK mapping of `psk_demod`, whose table equals table 5.1 of the standard), but nothing after them: no convolutional decoder, no interleaver, no additive scrambler, no CRC over a bit string that is not whole bytes, no burst synchronization on phase steps. The BSCH is the smallest complete target that needs all of these: it is scrambled with a fixed sequence (no colour code needed), it is never fragmented, and its CRC proves the whole chain with one number. Its content is the broadcast that every receiver of the network is meant to read.

The Viterbi decoder is the largest single piece and the one the DAB and DVB roadmap of the backlog needs as its first step. Built here as a generic kernel (constraint length, generators and termination as parameters, integer metrics, SIMD across the states), it serves TETRA (K = 5, four outputs), DAB (K = 7, four outputs), DVB-S/T and IEEE 802.11 (K = 7, two outputs) and GSM (K = 5, two outputs) with presets only.

## Order

M57 needs the SIMD backend (M13), the packet ports and `sdrk-proto` with its CRCs (M37), `symbol_sync` (M38), the JSON text packets of `adsb_parse` (M39.1) as the pattern of `tetra_parse`, the pi/4-DQPSK mapping of M40.1 and `pulse_match` (M44.1.1). It does not depend on the DSSS/OFDM series (M48 to M56) and can run in parallel with every milestone after M44.1.1.

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
- the kernel `viterbi` in `sdrk_core::kernels::conv` (§2): soft-decision decoding of a zero-tail block of a convolutional code with K = 3 to 9 and n = 2 to 4 outputs per input bit, `i8` soft input with erasures, `i16` path metrics in modulo arithmetic, a fixed tie rule, packed decisions in caller storage; the soft quantizer `viterbi_quantize` (`f32` to `i8`); Fallback, Reference, facade delegation, contract row and section in `docs/kernels.md`, the SIMD path on every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`; §10);
- the encoder `conv_encode` in the same module (Fallback and facade; SIMD in the backlog, §11);
- generic bit helpers in `sdrk-proto` (§3): `sdrk_proto::lfsr` (Fibonacci LFSR up to degree 32, the additive scrambler of bits and the sign flip of soft bits), `sdrk_proto::fec` (the multiplicative block interleaver, the index puncturing rule), `sdrk_proto::crc::crc_bits` (a CRC of width 1 to 32 over any number of bits, with the existing byte CRCs as presets that keep their tables);
- `sdrk_proto::tetra` (§1, §4): the facts of EN 300 392-2 needed here as constants with their clause, the BSCH chain as bit functions, the SYNC PDU and D-MLE-SYNC parser with JSON;
- the composite block `TetraBschDecodeBlock` and the kind `tetra_bsch_decode` (§5), the block `TetraParseBlock` and the kind `tetra_parse` (§6), with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list;
- the flow `tetra_bsch_receiver` (§7);
- the TETRA synchronization burst transmitter of the testkit (§8), golden cases (§9), tests 1 to 10, `cargo measure` entries (cycles, M13.1) with the reference rows of §10, CHANGELOG, decisions log.

**Out:** tail-biting and streaming termination of `viterbi` (backlog, needed by LTE and DAB/DVB streams); the periodic puncturing patterns of DVB, 802.11 and DAB; the public kind `sync_detect` and its other domains (the phase-step template of §5 is a crate-internal building block here; the name stays reserved); `frame_clock`; Reed-Solomon, `block_code` and the Reed-Muller code of the AACH; the colour-code scrambling, the normal bursts, the AACH, the BNCH/SYSINFO, the upper MAC, LLC and MLE (the next TETRA milestone, backlog); `CQ15`/`Q15` variants of the composite (the digital blocks are `C32`/`f32` only, D529); speech (EN 300 395-2), encryption, uplink bursts, pi/8-D8PSK and TEDS.

## Specification

General rules of M19 apply. Source of every TETRA fact: ETSI EN 300 392-2 V3.8.1 (2016-08), named by clause; the implementation is written from the standard only (§12, item 1).

### 1. The TETRA facts (`sdrk_proto::tetra`, constants)

All bit sequences are bits of the modulation B(m) (clause 5.4), 0 or 1, the first transmitted bit first. Each constant carries its clause in its documentation.

| Constant | Value | Clause |
| --- | --- | --- |
| `SYMBOL_RATE_HZ` | 18 000 (36 kbit/s, two bits per symbol) | 5.3 |
| `SLOT_SYMBOLS`, `SLOT_BITS` | 255, 510 (14.167 ms) | 4.5.2, 9.4.4.2 |
| `SLOTS_PER_FRAME`, `FRAMES_PER_MULTIFRAME`, `MULTIFRAMES_PER_HYPERFRAME` | 4, 18, 60 | 4.5.2 |
| `PI4_STEP` | table 5.1: B(2k-1) B(2k) = 11 -> -3pi/4, 01 -> +3pi/4, 00 -> +pi/4, 10 -> -pi/4 | 5.4 |
| `FREQUENCY_CORRECTION` | f1..f80: 8 ones, 64 zeros, 8 ones (a tone 2.25 kHz above the nominal carrier, with 4 symbols at 6.75 kHz below it at each end) | 9.4.4.3.1 |
| `TRAINING_N`, `TRAINING_P`, `TRAINING_Q` | the three 22-bit normal training sequences, formulas 9.4 to 9.6 | 9.4.4.3.2 |
| `TRAINING_Y` | the 38-bit synchronization training sequence, formula 9.11 | 9.4.4.3.4 |
| `SYNC_BURST` | the field table of the synchronization continuous downlink burst (table 9.9): 1 to 12 q11..q22, 13 to 14 phase adjustment hc, 15 to 94 frequency correction, 95 to 214 sb(1..120), 215 to 252 y, 253 to 282 bb(1..30), 283 to 498 bkn2(1..216), 499 to 500 phase adjustment hd, 501 to 510 q1..q10 | 9.4.4.2.6 |
| `NORMAL_BURST` | the field table of the normal continuous downlink burst (table 9.7), for the next milestone and for the testkit | 9.4.4.2.5 |
| `PHASE_ADJUST_SB` | hc: n1 = 8, n2 = 108; hd: n1 = 109, n2 = 249 (table 9.15) | 9.4.4.3.6 |
| `SCRAMBLER_TAPS` | c(x) with c_i = 1 for i = 0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 16, 22, 23, 26, 32 | 8.2.5.2 |
| `BSCH` | K1 = 60, (76,60) CRC, 4 zero tail bits, RCPC 2/3 (t = 3, P = 1, 2, 5), interleaver (120, 11), scrambled with e(1..30) = 0 | 8.3.1.2, 8.2.5.2 |
| `MOTHER_CODE` | K = 5, n = 4, G1 = 1 + D + D^4, G2 = 1 + D^2 + D^3 + D^4, G3 = 1 + D + D^2 + D^4, G4 = 1 + D + D^3 + D^4 | 8.2.3.1.1 |

The field tables are data (`&'static [Field]` with offset, length, name), so the next milestone adds burst types without code.

### 2. Kernel `viterbi` and encoder `conv_encode` (`sdrk_core::kernels::conv`)

**Code.** `ConvCode::new(k: u32, generators: &[u16], inverted: u8) -> Result<ConvCode>`: constraint length `k` 3 to 9, 2 to 4 generators, each with bit `j` the coefficient of `D^j` (bit 0 is the current input bit), every generator with bit 0 and bit `k - 1` set (`InvalidParameter` otherwise), `inverted` a bit mask of the outputs that are sent inverted (CCSDS; 0 for TETRA). Presets as associated constants with their source: `ConvCode::TETRA` (K = 5, generators `0b10011`, `0b11101`, `0b10111`, `0b11011` for G1 to G4 of §1), and, as checked parameter sets for the tests and for the DAB/DVB roadmap, `K7_R2` (generators 171 and 133 octal) and `K7_R4` (133, 171, 145, 133 octal); the values of the last two are checked against their standards when those milestones are written.

**State and output convention.** The encoder register holds the last `k` input bits, bit 0 the current one. The state is the `k - 1` previous bits: after input `u` in state `s` the next state is `((s << 1) | u) & (2^(k-1) - 1)`. Output `i` of a step is the parity of `generator_i & ((s << 1) | u)`, XOR bit `i` of `inverted`; the outputs of a step are sent in the order of the generators (TETRA: V(4(k-1) + i), i = 1 to 4, clause 8.2.3.1.1). The encoder starts in state 0.

**Encoder.** `conv_encode(code, bits: &[u8], out: &mut [u8])`: one byte per bit (0 or 1) in and out, `out.len() == n · bits.len()`, the caller appends the tail bits (k - 1 zeros for zero tail).

**Soft input.** The decoder takes one `i8` per received code bit: positive means the bit is 1 with that confidence (the soft-bit convention of M38), negative means 0, **0 is an erasure** (no information). The value -128 is not used (inputs are clamped to -127 by the quantizer). `viterbi_quantize(x: &[f32], scale: f32, out: &mut [i8])` maps `round_half_even(x · scale)` clamped to -127..=127; NaN gives 0.

**Branch metric.** For a branch with outputs `c_1 .. c_n` (0 or 1, after `inverted`): `bm = Σ_i (2 c_i - 1) · q_i` (a correlation, larger is better), at most `n · 127` in magnitude.

**Path metrics in modulo arithmetic.** The path metrics are `i16` with wrapping addition. A comparison of two metrics `a` and `b` is the sign of the wrapping difference `a.wrapping_sub(b)` as `i16`. This equals the comparison of the exact integer metrics as long as their spread stays below 2^15 (Hekstra's modulo normalization); the spread of a code with constraint length `k` and `n` outputs is at most `2 · (k - 1) · n · 127`, at most 8128 for K = 9, n = 4, so no renormalization step is needed and every implementation gives the same decisions. The start metrics are 0 for state 0 and -B for every other state with B = 2 · (k - 1) · n · 127 + 1, more than any path can gain before it merges with the path from state 0, so the decoder is a true zero-start decoder; the spread then stays below 2 · B ≤ 16 258 < 2^15.

**Add-compare-select and the tie rule.** The two predecessors of a state `s'` are `p0 = s' >> 1` and `p1 = (s' >> 1) | 2^(k-2)`, both with the input `u = s' & 1`. `m(s') = max(m(p0) + bm(p0 -> s'), m(p1) + bm(p1 -> s'))` with the comparison above; the decision bit is 1 if the path through `p1` is **strictly** larger, so a tie selects `p0`. The decisions of a step are packed: state `s` is bit `s & 31` of word `s >> 5` of the step's words (`ceil(2^(k-1) / 32)` words per step).

**Traceback.** Zero tail: after the last step the traceback starts in state 0. The decoded input bit of a step is bit 0 of the state the traceback is in after that step; the previous state is `p0` or `p1` by the decision bit. Output: one byte per decoded bit, the tail included (the caller drops it).

**API.**

```rust
pub struct ViterbiInfo {
    /// Metric of state 0 at the end minus the best other end metric (the margin of the decision; i32).
    pub margin: i32,
}

pub trait Viterbi {
    /// Decodes `steps` input bits from `n · steps` soft values, zero start and zero tail.
    /// `decisions` holds at least `steps · ceil(2^(k-1) / 32)` words; `out.len() == steps`.
    fn viterbi_zero_tail(
        &self,
        code: &ConvCode,
        soft: &[i8],
        decisions: &mut [u32],
        out: &mut [u8],
    ) -> Result<ViterbiInfo>;
}
```

Errors: `BufferTooSmall` for short `decisions` or `out`, `InvalidParameter` for `soft.len() != n · out.len()` or `steps < k - 1`. The exact generic form follows the kernel pattern of M02 (trait in the module, `Fallback`, `Reference`, the facade `Sdrk` through `delegate!`, `backend_for("viterbi", ..)`).

**Reference.** The textbook decoder with `i64` metrics, the same start metrics and the same tie rule, no modulo arithmetic; by the bound above its decisions are those of the fallback. The contract is **bit-exact**: decisions, output and `margin` equal on every backend and level and to the reference.

### 3. Generic bit helpers (`sdrk-proto`, `no_std`, no heap)

- **`sdrk_proto::lfsr::Lfsr`:** a Fibonacci register of degree 1 to 32 given by the tap mask of its connection polynomial (bit `i` for `c_i`, `c_0` implied) and by its initial history of `degree` bits. `next_bit()` gives p(k) = Σ_{i=1..degree} c_i p(k - i) mod 2 (the form of EN 300 392-2 formula 8.41). The two-tap polynomials `Prbs5` to `Prbs11` of `sdrk_proto::prbs` are **not** changed in this milestone; that they are presets of `Lfsr` with the same bit order (shared conventions §B.1) is shown by a test, and their merge is left to the backlog.
- **`additive_scramble_bits(lfsr, bits: &mut [u8])`** XORs bits with the sequence; **`additive_descramble_soft(lfsr, soft: &mut [f32])`** flips the sign of every soft value whose sequence bit is 1 (an erasure stays 0). The caller makes a new `Lfsr` per block (TETRA restarts the sequence for every block, 8.2.5).
- **`sdrk_proto::fec::MultiplicativeInterleaver { k, a }`:** the (K, a) block interleaver b4(k') = b3(i), k' = 1 + ((a · i) mod K), i = 1 to K (8.2.4.1); `new` checks that `gcd(a, K) = 1` (the rule is a permutation only then); `interleave` and `deinterleave` over any `T: Copy`, without a table (one multiply and one modulo per element).
- **`sdrk_proto::fec::IndexPuncture { t, coefficients, group, extra_every }`:** b3(j) = V(k), k = group · ((i - 1) div t) + P(i - t ((i - 1) div t)) with i = j, or i = j + (j - 1) div `extra_every` (8.2.3.1.2 to 8.2.3.1.6; TETRA rate 2/3: t = 3, P = (1, 2, 5), group 8, no extra). `puncture` selects the kept mother bits; `depuncture` writes the received values to their mother positions and 0 (erasure) everywhere else.
- **`sdrk_proto::crc::crc_bits(spec: &CrcSpec, bits: &[u8]) -> u32`** with `CrcSpec { width, poly, init, reflect_in, reflect_out, xor_out }`, bit-serial over any number of bits (one byte per bit). The existing byte CRCs (`crc16_x25`, `crc16_ccitt_false`, `crc32`, `crc24_adsb`) get their `CrcSpec` constants and a test that `crc_bits` over their bits equals them; their byte tables stay. **`CrcSpec::TETRA`** is the (K1 + 16, K1) code of 8.2.3.3: width 16, polynomial 0x1021, initial value 0xFFFF, no reflection, final XOR 0xFFFF, the parity bits sent highest coefficient first (b2(K1 + 1) = f(15)). This reading of formulas 8.14 to 8.18 is checked by a Python model that evaluates the formulas literally (§9) and, in the end, by the CRC of the recording (test 10).

### 4. The BSCH chain (`sdrk_proto::tetra` bit functions, the decoder in `sdrk-blocks`)

Transmit side (`sdrk_proto::tetra::bsch_encode(pdu: &[u8; 60]) -> [u8; 120]`, used by the testkit): CRC (76 bits), 4 zero tail bits (80 type-2 bits), `conv_encode` with `ConvCode::TETRA` (320 mother bits), `IndexPuncture` rate 2/3 (120 type-3 bits), `MultiplicativeInterleaver { 120, 11 }`, additive scrambling with e(1..30) = 0 (p(-31) = p(-30) = 1, p(-29..0) = 0). Since `sdrk-proto` cannot call the kernel (R2), `bsch_encode` takes the encoder as a function argument; the testkit passes the facade's `conv_encode`.

Receive side (`sdrk_blocks::tetra::BschDecoder`, crate-internal): 120 soft values of sb(1..120) -> `additive_descramble_soft` -> `deinterleave` -> `depuncture` (320 values, 200 erasures) -> `viterbi_quantize` with `soft_scale` -> `viterbi_zero_tail` (80 steps) -> drop the tail -> `crc_bits` over the 76 bits -> on a match the 60 bits. The number of corrected bits is counted by re-encoding the decoded 80 bits and comparing with the hard decisions of the received 120 (an erasure is not counted).

### 5. Composite `tetra_bsch_decode` (`TetraBschDecodeBlock`, `C32` to packets)

Input: `C32` samples of one carrier, centred within a few kHz, at a rate within 1 % of an integer multiple (2 to 1024) of `symbol_rate_hz` (the rule of `pulse_match`). Stages (building blocks; only the first two are existing kinds):

1. `pulse_match` (`rolloff` 0.35, `span` 10) and `symbol_sync` (Gardner, `loop_bw` = `sync_bw`) to one sample per symbol, with the input sample index of every symbol.
2. **Phase steps** z_k = s_k · conj(s_(k-1)).
3. **Frequency correction field search** (the phase-step domain of the reserved `sync_detect`): over every window of 40 phase steps the template t = -135 degrees (4), +45 degrees (32), -135 degrees (4), S = Σ z_(j+i) · exp(-j t_i), score = |S|^2 / (40 · Σ |z_(j+i)|^2) (Cauchy-Schwarz: 0 to 1, no square root or arc tangent per symbol). A candidate is a local maximum over ±20 symbols with score ≥ `min_score`. θ = arg(S) is the residual rotation per symbol: the carrier error is θ / (2π) · `symbol_rate_hz` (the template already contains the +2.25 kHz of the tone, so θ refers to the nominal carrier).
4. **Confirmation by y:** the 19 phase steps at +100 to +118 symbols from the first field symbol, rotated by -θ, sliced by table 5.1 (soft bits b(2k-1) = -Im, b(2k) = -Re of the rotated step, the mapping of `psk_demod` `variant: pi4`); at most `max_y_errors` of 38 bits differ from `TRAINING_Y`.
5. **sb(1..120):** the phase steps at +40 to +99 symbols, rotated by -θ, as 120 soft bits (the same formula), divided by the RMS of |z| over the 255 symbols of the burst; then the BSCH chain of §4.

Parameters (build): `symbol_rate_hz` (default 18000), `rolloff` (0.35), `span` (10), `sync_bw` (0.005), `min_score` (default fixed in this milestone from tests 6 and 7, recorded in the decisions log), `max_y_errors` (8), `soft_scale` (32), `deliver_failed` (false: only CRC OK).

Output packet per synchronization burst (only CRC OK unless `deliver_failed`): `index` = the input sample index of the first symbol of the frequency correction field; `flags` = `CRC_OK` (or `CRC_FAIL`), plus `CORRECTED` when the re-encoding found corrected bits; `data` (12 bytes): bytes 0 to 7 the 60 type-1 bits, b1(1) in bit 7 of byte 0, bits 60 to 63 zero; bytes 8 and 9 the carrier error in Hz as `i16` little-endian (saturated); byte 10 the score times 255; byte 11 the corrected bits (saturated at 255). Delay and chunk rules per `docs/timing.md`: the packet comes out when the last symbol of sb(120) (+99) is in, independent of the chunking (R6.2).

### 6. Kind `tetra_parse` (`TetraParseBlock`, packets to packets)

One JSON text packet per input packet with `CRC_OK` (others are dropped and counted): `{"pdu":"sync","sys":S,"cc":C,"tn":T,"fn":F,"mn":M,"share":H,"tsr":R,"dtx":D,"f18":E,"mcc":MCC,"mnc":MNC,"ncb":N,"load":L,"late":A,"df":HZ}` with the fields of the SYNC PDU (table 21.76: system code 4, colour code 6, timeslot number 2, frame number 5, multiframe number 6, sharing mode 2, TS reserved frames 3, U-plane DTX 1, frame 18 extension 1, reserved 1) and of the D-MLE-SYNC in its TM-SDU (table 18.17: MCC 10, MNC 14, neighbour cell broadcast 2, cell load CA 2, late entry supported 1), every value as the plain number of its bits (the timeslot number 0 to 3 stands for timeslot 1 to 4; frame and multiframe numbers are coded from 1) and `df` the carrier error. At most 160 bytes, well inside a packet. Reserved code values are passed through, not rejected.

### 7. Flow `tetra_bsch_receiver`

`examples/flows/tetra_bsch_receiver.json` and `.yaml`: input `cu8` at 1 152 000 S/s (16 times 72 kS/s, an RTL-SDR rate), `ddc` (`decimation` 16, `frequency_hz` the carrier offset, cutoff 13 kHz, transition 6 kHz) to 72 kS/s (4 samples per symbol), `tetra_bsch_decode`, `tetra_parse`, a packet output `sync`. Validates without warnings.

### 8. Testkit: synchronization burst transmitter (`sdrk_testkit::tetra`)

`SyncBurstStream` produces a continuous downlink in which every slot is a synchronization burst: given SYNC PDUs (or random ones from a seed), random bits for bb(1..30) and bkn2(1..216), the field table of §1, the phase adjustment bits hc and hd by formula 9.14 with table 9.15, the pi/4-DQPSK mapping of table 5.1 continued across slots (S(0) = 1, formula 5.1), the RRC pulse 0.35 through `psk_mod` (`variant` `pi4`, `differential`) at a chosen number of samples per symbol, then the channel of `sdrk_testkit::channel` (carrier offset, AWGN by Es/N0, optionally `Multipath::mild`). It returns the samples and, per burst, the PDU and the sample index of the first field symbol.

### 9. Golden vectors

New files `tests/golden/sdrk_golden/cases/conv.py` and `tetra.py`. The Python model of `conv.py` implements the encoder and a decoder with exact integer metrics, the start metrics and the tie rule of §2; `tetra.py` evaluates formulas 8.14 to 8.18 (CRC), 8.34 (interleaver), 8.7 and 8.8 (puncturing) and 8.39 to 8.42 (scrambler) literally, without the shortcuts of the Rust code.

| Case | Seed | Code | Input | Note |
| --- | --- | --- | --- | --- |
| `viterbi/tetra_clean` | 910 | `TETRA` | 80 steps, random bits, soft ±100, no noise | output = input |
| `viterbi/tetra_noisy` | 911 | `TETRA` | 80 steps, ±32 plus Gaussian noise (σ 40), clamped | decisions and output of the model |
| `viterbi/tetra_punctured` | 912 | `TETRA` | the depunctured 2/3 pattern (200 of 320 erased) at σ 30 | erasures |
| `viterbi/k7_r2` | 913 | `K7_R2` | 512 steps, σ 50 | the 64-state path |
| `viterbi/k7_r4` | 914 | `K7_R4` | 256 steps, σ 60 | n = 4 with 64 states |
| `viterbi/k3_r2`, `viterbi/k9_r4` | 915 | K = 3 (generators 7, 5 octal), K = 9 (a 4-output code chosen in the case file) | 128 steps | the range ends; K = 9 at the spread bound |
| `tetra/bsch_chain` | 917 | — | 20 random SYNC PDUs | the 120 transmitted bits per PDU and every intermediate (CRC, type-2, mother, type-3, type-4) |
| `tetra/scrambler_bsch` | — | — | e = 0 | the first 432 bits of p(k) |

Tolerance: `exact` (integers) for every case. The Rust golden tests also run each `viterbi` case in 20 orders of independent blocks (the kernel has no state between calls; the decisions buffer is reused) and compare with `==`.

### 10. Cost, SIMD plan and reference rows

**Entries (M13.1: cycles are the speed measure).** `cargo measure` entries `viterbi_k5n4` (TETRA, 80 and 288 steps), `viterbi_k7n2` and `viterbi_k7n4` (512 steps), each with its `_fallback` (the D321 floor), per decoded bit; `viterbi_quantize` per value; `tetra_bsch_decode_block` per input sample at 4 samples per symbol (stages 1 to 5, a synchronization burst in every slot).

**SIMD plan (`sdrk-backend-simd`, shared conventions §D).** The add-compare-select of all states of a step is lane-parallel: `i16` lanes, 8 per 128-bit vector (`sse2`, `sse4.2`, `neon`, `simd128`), 16 per 256-bit (`avx2`), 32 per 512-bit (`avx512`); K = 5 (16 states) is two 128-bit vectors or one 256-bit vector, K = 7 (64 states) eight or four. The butterfly form (states `2j` and `2j + 1` from predecessors `j` and `j + 2^(k-2)`) needs the even/odd deinterleave of the new metrics (a shuffle per level). The branch metrics of a step take 2^n values; they are computed once per step and distributed to the states by a per-code table (a byte shuffle, `pshufb` / `vqtbl1q_u8` / `i8x16.swizzle`, on the `i16` pairs). Comparisons by the sign of the wrapping difference; decisions from the compare masks (`movemask`, a narrowing on `neon`, `i8x16.bitmask`), written as the packed words of §2. Lanes beyond the state count (K = 3: 4 states) are masked. The traceback is serial (one table lookup per bit) and stays scalar. `viterbi_quantize` is lane-parallel (`cvt` with round-to-nearest-even, saturating narrow). Every level is bit-identical to the fallback (integer arithmetic, a fixed order); D321 is a quality gate: a slower path is improved, not removed.

**Reference rows (D32).** For `viterbi_k7n2`: VOLK `volk_8u_x4_conv_k7_r2_8u` (the Spiral decoder, K = 7, rate 1/2, `u8` metrics with renormalization; generic, SSE and NEON variants; already in the reference build) as the reference of the SIMD levels with the same work per bit (64 states, 2 outputs), and its generic variant as the reference of `viterbi_k7n2_fallback`. For `viterbi_k5n4` and `viterbi_k7n4`: GNU Radio `gr::fec::code::cc_decoder` (configurable constraint length, rate and generators; to be checked against its source when the translation is made; GPL, a pinned C translation `benches/refs/gr-cc-decoder.c` as the other GNU Radio references) as the scalar reference. The rows are measured in simulated cycles per decoded bit on the reference and counter model of every level (M13.1); instruction counts stand only for `simd128`. Gaps above 10 % are recorded as open items, there is no acceptance gate against a library.

### 11. Algorithm review against the goals of SDRk

- **Integer kernel.** `viterbi` uses `i8` inputs and `i16` metrics only: no floating point, bit-exact on every level and target, usable without an FPU. Only `viterbi_quantize` takes `f32`; a `Q15` quantizer is one shift (backlog with the `Q15` digital blocks).
- **Memory.** Decisions take `2^(k-1)` bits per step from caller storage: 160 bytes for the BSCH (K = 5, 80 steps), 576 bytes for a full slot of 288 steps; K = 7 takes 8 bytes per step. No heap; the composite takes them from its storage plan.
- **SIMD where the algorithm allows.** The add-compare-select and the quantizer are lane-parallel and get paths on every level here. The traceback is a serial dependence. `conv_encode` is serial per bit (word-parallel over packed bits is possible); it is a transmit and test function here and gets its SIMD path in the backlog with the first high-rate transmitter (DVB, DAB).
- **Burst-level helpers.** Descrambling, deinterleaving, depuncturing and the CRC handle at most 432 values per 14.2 ms slot, far below the cost of stage 1. They are protocol functions in `sdrk-proto` without SIMD; the generic SIMD kernel `additive_scramble` of the backlog comes with the first high-rate user (DVB energy dispersal). Decision to confirm: §12 item 3.
- **No per-symbol transcendental.** The template search of §5 uses products and sums only (the score without a square root); one `atan2` per accepted candidate.
- **Targets.** The kernel is `no_std`; the embedded runs wait with the other embedded targets. The composite is `C32`/`f32` like the blocks it is made of (D529).

### 12. Decisions to confirm before the work starts

1. **Sources.** The implementation is written from EN 300 392-2 only. Open implementations of TETRA (osmo-tetra, tetra-kit; GPL) are not read and not copied; their output may be compared with ours as black boxes (the rule of the DAB/DVB roadmap and of M46). The only GPL code in the repository stays the pinned bench references of `benches/refs`.
2. **The recording.** The recording of 2026-11-28 is foreign data and is not committed. Test 10 runs only when `SDRK_TETRA_RECORDING` names the file and is `#[ignore]` otherwise; its numbers go into the decisions log.
3. **Burst-level helpers without SIMD** (§11).
4. **The packet layout of §5** (12 bytes, the carrier error and the quality inside the packet).
5. **Content.** The milestone decodes the synchronization broadcast only. The next TETRA milestone (AACH, SYSINFO, colour code) stays with broadcast information too; content addressed to other parties is out of scope.

## Tests

Seeds 910 … 929.

1. **Golden cases** of §9 through the facade, `Fallback` and `Reference` (`sdrk-testkit/tests/golden_m57.rs`), `==`, in 20 block orders.
2. **Unit tests of `viterbi`** (`sdrk-core/tests/kernels_m57.rs`, seed 916): `ConvCode::new` rejects K outside 3 to 9, fewer than 2 or more than 4 generators, a generator without bit 0 or bit k - 1; buffer and length errors; for every preset and 1000 random blocks without noise the output equals the input; an all-erasure input decodes to all zeros (the tie rule) with `margin` 0; the start metric -B of §2 makes the decoder equal to an exhaustive zero-start maximum-likelihood search on 1000 short random blocks (12 steps, all 2^12 inputs tried); the free distance `d_free` of every preset is computed in the test by a search, and every pattern of up to `floor((d_free - 1) / 2)` hard errors (soft ±127 flipped) at random positions in a block of 80 steps is corrected (1000 patterns per preset); the modulo comparison equals the `i64` comparison on 10^6 random metric pairs within the spread bound; `viterbi_quantize` rounds half to even, clamps to ±127 and maps NaN to 0.
3. **SIMD** (`sdrk-backend-simd/tests/conv.rs`, seed 918): every level equals the fallback bit for bit (decisions, output, `margin`) for every preset and K = 3 to 9 on random soft input, including all-erasure and ±127 inputs; `viterbi_quantize` on every level equals the fallback including ±∞ and NaN.
4. **`sdrk-proto` helpers** (`sdrk-proto/tests/fec.rs`, `lfsr.rs`, `crc_bits.rs`, `tetra.rs`; seed 917): `crc_bits` with the presets of the byte CRCs equals them on random messages; `CrcSpec::TETRA` equals the literal model of formulas 8.14 to 8.18; `MultiplicativeInterleaver { 120, 11 }`, `{ 216, 101 }` and `{ 432, 103 }` are permutations, `deinterleave(interleave(x)) == x`, and `new` rejects `gcd(a, K) != 1`; `depuncture(puncture(v))` restores every kept value and erases the others; `Lfsr` with the TETRA taps and e = 0 equals the golden 432 bits; `Lfsr` with the taps of `Prbs5` to `Prbs11` and their seeds gives the bits of `Prbs::next_bit`; `bsch_encode` equals the golden chain; the field tables of §1 cover 510 bits without gaps or overlaps and match tables 9.7 and 9.9 (asserted positions).
5. **BSCH round trip without noise** (`sdrk-blocks/tests/tetra_blocks.rs`, seed 919): `SyncBurstStream` with 1000 random PDUs at 4 samples per symbol, carrier offsets 0, +1.5 and -2.5 kHz, a symbol timing offset of a quarter symbol: every PDU comes out with `CRC_OK`, unchanged, at the right sample index (±1 symbol), with no extra packet; the carrier error within 30 Hz of the offset.
6. **BSCH in noise** (seeds 920 to 922): Es/N0 8, 10 and 12 dB (AWGN), offset +1 kHz, 1000 bursts each: the CRC OK rate is recorded; at 12 dB at least 99 %. With `Multipath::mild` at 12 dB the rate is recorded (no limit).
7. **False alarms** (seed 923): 60 s of complex Gaussian noise and 60 s of a random pi/4-DQPSK signal without synchronization bursts (normal bursts only): no packet with `CRC_OK`; the number of candidates that pass `min_score` and `max_y_errors` is recorded. `min_score` is fixed from tests 6 and 7 together.
8. **Chunk invariance** (seed 924): the packets of test 5 in 20 chunk plans equal the whole run bit for bit (R6.2).
9. **Kinds and flow** (`sdrk-pipeline/tests/tetra_kinds.rs`, seeds 925, 926): both kinds validate, reject wrong input types and rates outside the rule of `pulse_match`; `tetra_parse` gives the JSON of §6 for golden PDUs (every field at its minimum, maximum and a reserved value); `tetra_bsch_receiver` validates without warnings and decodes a `cu8` rendering of test 5 at 1.152 MS/s; schema and descriptions are in sync.
10. **The recording** (`sdrk-testkit/tests/tetra_recording.rs`, `#[ignore]`, §12 item 2): `ddc` and `resample` from 1.024 MS/s to 72 kS/s for each of the three carriers of the analysis; recorded: synchronization bursts found, CRC OK count, carrier error. Pass: at least one SYNC PDU with `CRC_OK` per carrier; every `CRC_OK` PDU of a carrier has the same MCC, MNC and colour code; timeslot, frame and multiframe numbers advance consistently with the sample distance of the bursts (one slot per 255 symbols, modulo 4, 18, 60). Recorded, not a condition: whether the three carriers carry the same MCC, MNC and colour code (one cell) or not.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-proto --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-backend-simd --all-features` (test 3 on the levels of the build host; CI runs every level); `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask check cycles` (D321) and `cargo measure` for the entries of §10; `cargo xtask count --platform wasm32` for `simd128`, which has no core model. Test 10 is run by hand with the recording and its numbers are recorded; it is not part of CI.

## Deliverables checklist

- [ ] Kernel `viterbi` with `ConvCode` (presets `TETRA`, `K7_R2`, `K7_R4`), `viterbi_quantize`, trait, Fallback (modulo metrics, the tie rule), Reference (`i64`), facade delegation, `backend_for`
- [ ] `conv_encode` (Fallback, facade)
- [ ] SIMD paths of `viterbi` and `viterbi_quantize` on every level (§10); `backend_for` reports the level
- [ ] Contract row and section in `docs/kernels.md` (state and output convention, the soft-bit and erasure convention, modulo metrics with the spread bound, the tie rule, packed decisions, bit-exactness)
- [ ] `sdrk_proto::lfsr`, `sdrk_proto::fec` (`MultiplicativeInterleaver`, `IndexPuncture`), `sdrk_proto::crc::crc_bits` with `CrcSpec` and the presets
- [ ] `sdrk_proto::tetra`: the constants of §1 with their clauses, `bsch_encode`, the SYNC PDU and D-MLE-SYNC parser with JSON
- [ ] Composite `TetraBschDecodeBlock` and kind `tetra_bsch_decode`; `TetraParseBlock` and kind `tetra_parse`; `docs/blocks.md` (entries; `sync_detect` stays a reserved name), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list
- [ ] Flow `tetra_bsch_receiver` (JSON and YAML)
- [ ] `sdrk_testkit::tetra::SyncBurstStream`
- [ ] Golden cases (seeds 910 to 915, 917) and the Python models `conv.py`, `tetra.py`
- [ ] Tests 1 to 9 (seeds 910 to 926); test 10 run by hand with its numbers in the decisions log
- [ ] Cycle entries (`cargo measure`; `cargo xtask count --platform wasm32` for `simd128`) with their `_fallback` entries, `docs/performance.md`; the reference rows against VOLK `volk_8u_x4_conv_k7_r2_8u` and GNU Radio `cc_decoder` (`benches/refs/gr-cc-decoder.c`)
- [ ] `docs/milestones/BACKLOG.md`: the generic building blocks entry names M57 for `viterbi`, `conv_encode`, `IndexPuncture`, `MultiplicativeInterleaver`, `Lfsr` and `crc_bits`; the next TETRA milestone (AACH, SYSINFO, colour code, upper MAC) as a backlog item
- [ ] `docs/milestones/README.md`: milestone row, dependency row, seed range 910–929
- [ ] CHANGELOG, decisions log (the reading of formulas 8.14 to 8.18, the start metrics, `min_score`, the packet layout, the numbers of tests 6, 7 and 10)

## Status

Planned (written 2026-11-28). Nothing is implemented.
