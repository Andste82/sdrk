# M47 — Morse (CW): sender and receiver

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M39.1](M39.1-afsk-ook-adsb.md) · Next: [BACKLOG](BACKLOG.md)

## Goal

Send and receive Morse code (CW, continuous wave telegraphy): the Morse table of ITU-R M.1677-1 with prosigns, an encoder that turns text into a keying stream at a chosen speed, a decoder that turns the envelope of a keyed carrier or tone back into text with its own estimate of the speed (5 to 40 words per minute, drifting by up to ±20 %) and an adaptive threshold, and the flows `cw_transmitter` and `cw_receiver` with a loop test transmitter → noisy channel → receiver.

## Why

CW is the oldest digital mode and still in daily use on the amateur bands. The pieces exist (`ook_mod`, `ook_demod`, `ddc`, `mag`, `biquad`), but nothing turns text into keying or timing back into text. The decoder is the interesting part: dots and dashes have a fixed ratio but a free speed, so it has to measure before it can decide.

## Order

M47 depends on M39.1 (`ook_mod` and the OOK flows) and through it on M37 (packet ports, `sdrk-proto`). It is independent of M42 to M46 and can run in parallel with them. The range of decision numbers is fixed with the maintainer before the start (D750 to D799 and D800 to D849 are taken by M42 and M43), as with every parallel milestone.

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

**In:** `sdrk_proto::morse` (the table, `Encoder` with Farnsworth timing, `Decoder` that works on a key state per sample, heap-free); blocks `MorseEncodeBlock` and `MorseDecodeBlock` and kinds `morse_encode` and `morse_decode`; flows `cw_transmitter`, `cw_receiver`, `cw_tone_transmitter` and `cw_tone_receiver` with YAML twins; an independent Python reference (own table, unit expansion and Farnsworth rounding) and golden cases in the golden generator; the loop tests; count entries `morse_encode_block` and `morse_decode_block`; `docs/blocks.md`, `docs/pipeline.md`, `docs/testing.md`, `examples/README.md`, `CHANGELOG.md`.

**Out:** CW on a single sideband transmitter chain beyond what the flow `cw_tone_transmitter` builds from existing blocks, key shaping other than the raised cosine of `rise_ms` (backlog: shaped keying), decoding of several CW signals at once in one band (backlog: a channelizer in front of several `morse_decode` blocks, after M45), machine-readable abbreviations (Q codes, callsign extraction), Farnsworth timing on the decode side beyond what the speed estimate tolerates (the encoder supports it, §1), non-Latin Morse tables, and a decoder that holds the first characters back until the speed is known (the start is decoded with the speed hint, §2).

## Specification

### 0. Licence and sources (mandatory)

SDRk is MIT/Apache-2.0. The Morse table and the timing rules come from the public standard ITU-R M.1677-1 (International Morse code). **No code, table or text of GPL projects (fldigi, cwdecoder and similar) is copied, translated or used as a template.** The decoder is designed from the timing rules; the decision log records the method (how the unit is estimated, how the threshold adapts) and what it was checked against.

### 1. The table and the encoder (`sdrk_proto::morse`)

- Characters: `A`–`Z`, `0`–`9`, and the punctuation of M.1677-1 (`. , : ? ' - / ( ) " = + @`); lower case is folded to upper case. The prosigns `AR`, `BT` and `KN` have the codes of `+`, `=` and `(`, so those three characters stand for them (sending and receiving); `SK` has no printable character and is the byte `0x04` (`SK`, ASCII end of transmission). A space, tab, carriage return or line feed is a word gap. A byte without a code is skipped (`Push::Skipped`).
- Timing in units: dot 1, dash 3, gap inside a character 1, gap between characters 3, gap between words 7. The unit time in seconds is `1.2 / wpm` (the PARIS standard: the word PARIS plus its word gap is 50 units). Farnsworth: `char_wpm` (the speed of the characters, at least `wpm`) and `wpm` (the effective speed) stretch only the gaps between characters and between words to `max(3, round(g))` and `max(7, round(7 g / 3))` units with `g = (1800 r - 1116) / 228`, `r = char_wpm / wpm` (the ARRL formula: the gaps share the `ta = (60 c - 37.2 s) / (s c)` seconds of a word as 3, 3, 3, 3 and 7 parts of 19); `char_wpm = wpm` is plain timing.
- `Encoder`: `new(wpm, char_wpm) -> Result` (`1 <= wpm <= char_wpm <= 100`); `push(c) -> Result<Push>` (one byte at a time, only while the units of the previous character are drained) and `next_unit() -> Option<bool>` (one key state per unit); `word_break()`, `settle()` (the caller has sent the gap itself: the next character follows without one), `reset()`, `is_idle()`, `char_gap()`, `word_gap()`. The gap in front of a character is sent in front of it, so the stream ends with the last element. `units(text, char_gap, word_gap)` counts the units of a text. No heap.

### 2. The decoder (`sdrk_proto::morse::Decoder`)

`Decoder::new(hint_samples)` takes the dot length of the speed hint in samples; `push(on, index) -> Option<Emit>` takes one key state per sample (and the index of that sample) and returns a character or a space when the sample finishes one (`Emit { byte, start }`, `start` the index of the first sample of the first key-down run of the character); `flush()`, `reset()`, `dot_samples()`, `dash_samples()`, and the counters `unknown()`, `overflow()`, `uncertain()`. The decoder measures the runs itself:

- **Elements.** It keeps the last 16 key-down runs. If the 80th percentile of them is at least 1.8 times the 20th, the runs form two clusters (dots and dashes) split half way between those two values; the dot length `m_dot` and the dash length `m_dash` are the means of the clusters. A window of nearly equal runs follows the previous estimate when its mean is within 0.6 to 1.6 times the previous dot or dash length (a drift), and otherwise is taken as dots, or as dashes when the mean is at least twice the hint unit. A run of a finished character is a dash when it is at least `(m_dot + m_dash) / 2`; a character with a run within 20 % of that split is counted as `uncertain`.
- **Gaps.** A key-up run of `2 m_dot` ends the character, of `5 m_dot` the word (a space is returned once after a character), and of `7 * 20 m_dot` (20 word gaps) the transmission: the estimate is forgotten, so a sender at another speed is picked up. Farnsworth gaps are stretched and use the same limits on the dot of the characters.
- **Characters.** The elements of a character are looked up in the table (at most 6 elements); a sequence without a code is returned as `?` and counted in `unknown`; a character with more than 8 key-down runs is discarded and counted in `overflow`.
- No heap: the table is a `const` array, the state is a fixed struct. The first characters of a transmission are decoded with the estimate of the runs seen so far; a character with a dot and a dash gives the estimate at once, a start with equal elements (`E`, `T`, `5`) relies on the hint, so a preamble such as `VVV` settles it (D903).

### 3. Speed and range

5 to 40 words per minute (the blocks accept 5 to 60 for the hint and the encoder and up to 100 for `wpm_max`). The decoder follows a drift of ±20 % over a sentence and a step of ±30 % (slower and faster) after an idle gap. A unit at `wpm_max` (decoder) or at the character speed (encoder) must have at least 8 samples: the blocks give `OutOfRange` of `wpm_max`, `wpm` or `char_wpm` otherwise, which the pipeline reports as `rate_unsupported`.

### 4. Blocks and kinds

`MorseEncodeBlock` (kind `morse_encode`): `Packets` (text) → `f32`, the key envelope **at the sample rate** of the block (1.0 key down, 0.0 key up), the rate of the output being the `rate_hz` of the packet input of the pipeline (`RatioKind::Source`, like `hdlc_encode`). Unit `k` ends at the sample `round((k + 1) * samples_per_unit)` with `samples_per_unit = rate * 1.2 / char_wpm`, so the speed is exact at any rate. Parameters (all build): `wpm` (5 to 60, default 20), `char_wpm` (0 or `wpm` to 100, default 0: same as `wpm`), `rise_ms` (0 to 50, default 5): every edge between two units of different key state is a raised cosine of `rise_ms` (at most half a unit), so the keyed carrier makes no key clicks. **Every packet ends with one word gap of key up** (the gap separates it from the next packet and lets a decoder finish the last word); a packet is consumed when its gap is written, and a packet that does not fit the output is continued by the next call, which must start with the same packet. With no packet the key stays up (pull mode). The output does not depend on how the packets are cut into calls. Cost: 20 per output sample (D909).

`MorseDecodeBlock` (kind `morse_decode`): `f32` (the envelope at the sample rate, for example the output of `mag`) → `Packets` (text, one per word, up to 256 characters, without the space; `index` the input sample index of the first key-down sample of the word). Parameters (all build): `wpm` (the speed hint, 5 to 60, default 20), `wpm_max` (`wpm` to 100, default 40), `threshold` (0: adaptive, default; otherwise a fixed level of the smoothed envelope). The block turns the envelope into key-down and key-up runs:
- A one-pole smoother with a time constant of 1/8 unit at the hint speed (D904).
- Adaptive threshold: a peak-hold signal level (decaying towards the noise level) and a minimum-hold noise level (rising while the key is up), both with a time constant of 20 units; threshold = noise + 0.5 × (signal − noise), hysteresis ±5 % of that span; and at least the noise mean plus four standard deviations of the smoothed envelope, learned in blocks of four smoothing time constants in which the key stayed up and away from an edge (a block counts only with the next block quiet too), so that noise alone gives no key down (D904).
- The first block after the start or a reset only learns the noise (no key down); the stream must begin with noise or silence of about a unit.
- A run shorter than a third of a unit at `wpm_max` (at least 2 samples) is merged into the surrounding run (D905).
- A value that is not finite counts as 0.
- Accessors: `wpm()` (the estimated speed), `unknown()`, `overflow()`, `uncertain()`.
Cost: 69 per input sample (D909). `cargo xtask count` entries `morse_encode_block` (per output sample) and `morse_decode_block` (per input sample), both "our code, no library reference": reported as information (D910).

### 5. Flows

- `cw_transmitter` (+ YAML): `msg` (packets at 8000 Hz) → `keys` (`morse_encode`, 20 wpm) → `cplx` (`to_complex`) → `up` (`duc` by 30 to 240 kS/s, carrier at +10 kHz, cutoff 500 Hz, transition 600 Hz, `max_taps` 2048) → `rf` (`cs8`, pull mode, carrier off when no text waits).
- `cw_receiver` (+ YAML): `rf` (`cu8` at 240 kS/s) → `tune` (`ddc` by 30, carrier at +10 kHz, cutoff 500 Hz, transition 600 Hz, `max_taps` 2048) → `env` (`mag`) → `dec` (`morse_decode`, 20 wpm hint, `wpm_max` 40) → `words` (packets).
- `cw_tone_transmitter` (+ YAML): `msg` (packets at 8000 Hz) → `keys` → `cplx` (`to_complex`) and `tone` (`nco`, 700 Hz, clocked by the envelope) → `mix` (`mul`) → `re` (`real_part`) → `audio` (`f32`, pull mode): the keyed tone for an SSB transmitter or a sound card.
- `cw_tone_receiver` (+ YAML): `audio` (`f32` at 8000 Hz) → `tone` (`biquad` band-pass at 700 Hz, Q 8) → `cplx` (`to_complex`) → `abs` (`mag`, the rectifier) → `env` (`biquad` low-pass 150 Hz) → `dec` (`morse_decode`) → `words`.
- `examples/README.md`: running the receiver with an RTL-SDR (`cu8`), the transmitter into a loop and into a file, and reading the text.

## Tests

1. **Table and timing (golden):** `sdrk-proto/tests/morse.rs` checks the table against a separate copy of ITU-R M.1677-1 (49 characters and `SK`, every code once), PARIS plus its word gap as 50 units, the exact units of SOS, the word gaps, the prosign codes, the skipped bytes and Farnsworth 10/18 (gaps 9 and 22); `sdrk-proto/tests/morse_golden.rs` compares the encoder with the golden cases `morse/encode_units` (seed 650: 40 texts) and `morse/encode_units_farnsworth` (seed 651: six speed pairs) of the Python reference in `tests/golden/sdrk_golden/cases/morse.py`, which has its own table, unit expansion and gap rounding.
2. **Round trip, clean:** `sdrk-blocks/tests/morse_blocks.rs` sends texts at 5, 10, 20, 30 and 40 wpm and at 2000 to 16000 Hz through the blocks and gets the words; `morse_golden.rs` compares the envelope of `morse_encode` (seed 652, 17 wpm at 1000 Hz, hard keying) bit for bit with the Python reference and decodes three transmissions at 6, 12 and 25 wpm in Gaussian noise (seed 653); `sdrk-pipeline/tests/morse_flows.rs` closes the loops `cw_transmitter` → `cw_receiver` (IQ) and `cw_tone_transmitter` → `cw_tone_receiver`; `morse_kinds.rs` runs the loop of both kinds in push mode and compares it with the blocks.
3. **Noise:** `morse_flows.rs` sends about 150 random characters (words of 2 to 6, after the preamble `VVV`) through the tone loop with white Gaussian noise (seeds 652 to 654): the character error rate is at most 1 % at 12 dB and above and at most 5 % at 6 and 9 dB (SNR as the power of the keyed tone over the noise power in 500 Hz). Measured: 2.12 %, 0 % and 0 % at 6 dB, 0 % at 9 dB and above (D911).
4. **Speed drift and steps:** `sdrk-proto/tests/morse.rs` and `morse_blocks.rs`: the unit swings by ±20 % over a text (with noise of 0.1 of the key down level in the block test, seed 653): no error; a second sender 30 % slower or 30 % faster after 200 units of silence (seeds 654 and 657, with noise): every word of both.
5. **Frequency offset:** `morse_flows.rs`: the transmitter at 650, 700 and 750 Hz against a receiver band-pass at 700 Hz at 15 dB (seed 655): at most 1 % (measured 0.53 %, 0 % and 0.53 %).
6. **Key timing faults:** sloppy keying (every key-down unit ±15 %, every key-up unit ±25 %, fixed generator seeds) without noise (`sdrk-proto`) and with noise of 0.1 of the key down level (blocks): at most 2 % (met; see D911).
7. **Prosigns, punctuation, unknown bytes:** `sdrk-proto/tests/morse.rs` and `morse_blocks.rs`: every character of the table and the byte `0x04` round trip; a byte without a code is skipped by the encoder and takes no time; 7 and 9 dots give `?` (counted in `unknown`) and a discarded character (`overflow`), in `sdrk-proto` and in the block.
8. **Chunk invariance:** `morse_blocks.rs`: the decoder in 20 cut plans of seed 651 and the encoder with packets cut into calls (seed 650) and output rooms (seed 652) gives the same bits; `morse_flows.rs` pushes the receivers in pieces of 997 samples; `morse_kinds.rs` pushes 1, 2, 3 or all packets at a time.
9. **Tone variants:** `morse_flows.rs`: `cw_tone_transmitter` → `cw_tone_receiver` at 700 Hz and 450 Hz (`frequency_hz` of `tone`, `cutoff_hz` of the band-pass, both with `set_param`): the words of three packets exactly.
10. **Count entries** `morse_encode_block` and `morse_decode_block` on all four platforms (`cargo xtask count --platform <p> --kernel morse_encode_block --kernel morse_decode_block --merge`) and cycles with `cargo measure`; "our code" rows, information only (D910).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-proto`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (with the two block entries).

## Deliverables checklist

- [x] `sdrk_proto::morse` (table, `Encoder`, `Decoder`), with the source named in the decision log (D900 to D903)
- [x] Blocks and kinds `morse_encode` and `morse_decode`; schema
- [x] Python reference encoder and the golden cases (seeds 650 to 653)
- [x] Flows `cw_transmitter`, `cw_receiver`, `cw_tone_transmitter`, `cw_tone_receiver` with YAML twins
- [x] Tests (seeds 650 to 656)
- [x] Count entries `morse_encode_block`, `morse_decode_block` (counts on all four platforms and cycles per sample on every level with a core model, D909, D910)
- [x] `docs/blocks.md`, `docs/pipeline.md`, `docs/testing.md`, `examples/README.md`, CHANGELOG, decision log

## Status

Done: `sdrk_proto::morse` (the table of ITU-R M.1677-1 with the prosign conventions of D901, `Encoder` with Farnsworth timing, `Decoder` with the speed estimate of D903); the blocks and kinds `morse_encode` and `morse_decode` (D900, D904 to D907), `rate_unsupported` below 8 samples per unit, 63 kinds in the schema; the four flows with YAML twins (D908); the golden cases of the independent Python reference (`morse/encode_units`, `encode_units_farnsworth`, `encode_envelope`, `morse_decode/words`; seeds 650 to 653); the tests 1 to 9 (seeds 650 to 656) with the acceptance values of D911; `docs/blocks.md`, `docs/pipeline.md`, `docs/testing.md`, `examples/README.md`, CHANGELOG and the decision log (D900 to D911, the range D900 to D929 is a proposal of the implementer, to be confirmed with the maintainer); the count entries in `xtask/src/count.rs`, `benches/kernels` and `benches/limits.json` with the costs 20 and 69 (D909, D910), counted and measured (below).

Counts and cycles (2026-11-07; information, no library reference, D910): `cargo xtask count` on all four platforms and `cargo measure` on every level with a core model, rows in `benches/results/*.json` and `docs/performance.md`. Cycles per sample, whole-loop, reference / counter model: the encoder 6.60 / 7.58 (`sse2`, `sse4.2`), 4.44 / 4.50 (`avx2`), 4.20 / 4.64 (`avx512`), 8.56 / 9.45 (`neon`); the decoder 16.81 / 20.53 (`sse2`), 16.62 / 20.31 (`sse4.2`), 13.06 / 16.08 (`avx2`), 13.19 / 17.03 (`avx512`), 22.26 / 28.43 (`neon`). Instructions per sample (encoder, decoder): `sse2` 20.17 and 73.68, `sse4.2` 20.17 and 72.93, `avx2` 19.97 and 69.08, `avx2-runtime` 20.17 and 73.68, `avx512` 18.37 and 68.17, `neon` 19.46 and 62.86, `simd128` 49.57 and 149.15. A speed round (D909) took the decoder from 147.92 to 69.08 instructions per sample at `avx2`.

Verified locally on Linux x86-64: `cargo test` of the M47 files (`sdrk-proto`: `morse`, `morse_golden`; `sdrk-blocks`: `morse_blocks`, `morse_golden`, `describe`, `construction`; `sdrk-pipeline`: `morse_kinds`, `morse_flows`, `yaml`), `cargo clippy -D warnings` on the touched crates, `cargo fmt`, `cargo xtask check`, `cargo xtask schema --check` and `python tests/golden/generate.py --check`.

Not done or not run:
- The whole workspace test suite, the emulator, the wasm tests and the other hosts; CI and the final gate run them.

Open items: none that need a decision of the maintainer except the range of decision numbers (D900 to D929).
