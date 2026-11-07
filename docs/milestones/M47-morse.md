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

**In:** `sdrk_proto::morse` (the table, `Encoder`, timing, the `Decoder` that works on pulse and gap durations, heap-free); blocks `MorseEncodeBlock` and `MorseDecodeBlock` and kinds `morse_encode` and `morse_decode`; flows `cw_transmitter` and `cw_receiver` with YAML twins; a Python reference encoder and golden cases in the golden generator; the loop tests; count entries `morse_encode_block` and `morse_decode_block`; `docs/blocks.md`, `docs/pipeline.md`, `docs/testing.md`, `examples/README.md`, `CHANGELOG.md`.

**Out:** CW on a single sideband transmitter chain beyond what the flow `cw_transmitter` builds from existing blocks, transmit key shaping beyond the interpolation filter of `ook_mod` (backlog: shaped keying), automatic gain on the key signal, decoding of several CW signals at once in one band (backlog: a channelizer in front of several `morse_decode` blocks, after M45), machine-readable abbreviations (Q codes, callsign extraction), Farnsworth timing on the decode side beyond what the speed estimate tolerates (the encoder supports it, §1), and non-Latin Morse tables.

## Specification

### 0. Licence and sources (mandatory)

SDRk is MIT/Apache-2.0. The Morse table and the timing rules come from the public standard ITU-R M.1677-1 (International Morse code). **No code, table or text of GPL projects (fldigi, cwdecoder and similar) is copied, translated or used as a template.** The decoder is designed from the timing rules; the decision log records the method (how the unit is estimated, how the threshold adapts) and what it was checked against.

### 1. The table and the encoder (`sdrk_proto::morse`)

- Characters: `A`–`Z`, `0`–`9`, and the punctuation of M.1677-1 (`. , : ? ' - / ( ) " = + @`); lower case is folded to upper case; the prosigns `<AR>`, `<SK>`, `<BT>`, `<KN>` as the byte sequences `0x01`–`0x04` of the text input (documented; the block docs name them), sent as one character without the pause between its letters. A byte without a code is skipped and counted (`unknown`).
- Timing in units: dot 1, dash 3, gap inside a character 1, gap between characters 3, gap between words 7. The unit time in seconds is `1.2 / wpm` (the PARIS standard: the word PARIS is 50 units). Farnsworth: `char_wpm` (the speed of the characters) and `wpm` (the effective speed, at most `char_wpm`) stretch only the gaps between characters and between words, by the usual formula; `char_wpm = wpm` (default) is plain timing.
- `Encoder`: `new(wpm, char_wpm) -> Result` (`wpm` in 5…60); `push_text(&mut self, text: &[u8])` and `next_unit(&mut self) -> Option<bool>` (one key state per unit; the state machine holds the character in progress, no heap; the caller drains it); `flush`. Farnsworth gaps are whole units; `wpm`/`char_wpm` are rounded to the nearest whole number of units per gap and the decision log records the rounding.

### 2. The decoder (`sdrk_proto::morse::Decoder`)

Input: events `(on: bool, samples: u32)` in time order (a run of key-down or key-up samples), produced by the block of §4 from the envelope. The decoder:

- **Unit estimate.** It keeps a running estimate of the dot length from the key-down runs: two clusters (dots and dashes) are separated by a threshold between them, initially at `2 × shortest run seen so far` and then re-fitted as the mean of the two cluster means; the dot length is the mean of the lower cluster, the dash length must be within 2.0 … 4.5 × the dot length or the run is classified by nearest cluster and counted as `uncertain`. Before the first dash, a run is a dot if it is shorter than `2.5 ×` the shortest run and a dash otherwise. The estimate follows the signal with a time constant of about 6 runs (a drift of ±20 % over a sentence is followed), and `wpm()` returns `1.2 / unit` in words per minute.
- **Gaps.** A key-up run shorter than `2 × dot` is a gap inside the character, shorter than `5 × dot` a gap between characters, longer a gap between words (Farnsworth gaps use the same limits on the stretched gaps: the limits are `2 ×` and `5 ×` the *character* dot). A key-up run longer than `idle_runs × word gap` (default 20 words) is an idle gap: the decoder flushes and forgets the unit estimate, so a new sender at another speed is picked up.
- **Characters.** A sequence of dots and dashes of at most 6 elements is looked up in the table; a sequence without a code is emitted as `?` and counted (`unknown`); more than 8 elements are discarded and counted (`overflow`). A word gap after a character emits a space (once; a second word gap emits nothing).
- No heap: the table is a `const` array, the state is a fixed struct.

### 3. Speed and range

5 to 40 words per minute. The decoder accepts a drift of ±20 % over a sentence and a step of ±30 % at an idle gap. Characters per second are limited only by `samples_per_unit ≥ 8` (the block rejects a configuration where the shortest run at `wpm_max` is shorter than 8 samples).

### 4. Blocks and kinds

`MorseEncodeBlock` (kind `morse_encode`): `Packets` (text, `payload` = bytes) → `f32` (keying stream, 1.0 for key down and 0.0 for key up, **one value per unit**). Parameters: `wpm` (Float, build, 20), `char_wpm` (Float, build, 0 = same as `wpm`). Cost: 4 per output value (record it). The stream feeds `ook_mod`, whose `samples_per_symbol` is the unit duration in samples (`sample_rate × 1.2 / wpm`); if `ook_mod` rejects that value, the decision log records the limit and the encoder gets the parameter `samples_per_unit` with its own interpolation, as `ook_mod` does.

`MorseDecodeBlock` (kind `morse_decode`): `f32` (the envelope at the sample rate; for example the output of `mag`) → `Packets` (text, one packet per word; a word is complete at a word gap, at an idle gap, or at `flush`). Parameters: `sample_rate` (Float, build, needed for the lower limit of §3), `wpm_min` (5), `wpm_max` (40), `threshold` (`adaptive` default, or a fixed level 0…1 for tests). The block turns the envelope into key-down and key-up runs:
- Adaptive threshold: two trackers, the signal level (fast attack, slow decay) and the noise level (slow attack, fast decay), threshold = noise + 0.4 × (signal − noise), hysteresis ±10 % of that span; with a span below 6 dB the output is key-up (no signal). The trackers start at the first 200 ms of input.
- A run shorter than `1 / (8 × wpm_max)` seconds is dropped as a glitch and merged into the surrounding run.
- Outputs: a report of the speed estimate (`wpm`) in the way `afc` reports its ppm, the counters `unknown`, `overflow` and `uncertain`.
Cost: record both blocks (`cargo xtask count` entries `morse_encode_block` per output value and `morse_decode_block` per input sample, both "our code, no library reference": reported as information).

### 5. Flows

- `cw_transmitter` (+ YAML): `text` source → `morse_encode` → `ook_mod` → `duc` → IQ out (the keyed carrier at a chosen offset); an audio variant `cw_tone_transmitter` mixes the keyed carrier with an `nco` tone (default 700 Hz) and outputs `f32` audio for an SSB transmitter chain (M35).
- `cw_receiver` (+ YAML): IQ in → `ddc` (centred on the carrier, 250 Hz cut-off) → `mag` → `morse_decode` → text packets. The audio variant `cw_tone_receiver` takes audio (`f32`), builds the analytic signal with `real_to_iq`, then `ddc` (shift by the tone, narrow cut-off) → `mag` → `morse_decode`.
- `examples/README.md`: running the receiver with an RTL-SDR (`cu8`), the transmitter into a loop and into a file, and reading the text.

## Tests

1. **Table and timing (golden):** the Python reference encoder in `tests/golden/sdrk_golden` (written independently from M.1677-1: its own table and unit rules) produces the key states for 30 texts (all characters, the prosigns, mixed words, seed 650) at 5, 12, 20 and 40 wpm and with Farnsworth 18/10; the Rust encoder returns exactly the same units. `PARIS ` is 50 units.
2. **Round trip, clean:** random texts (seed 651, 40 texts of 8…60 characters, all characters and prosigns) at 5, 10, 20, 30 and 40 wpm through `morse_encode` → `ook_mod` → `ddc` → `mag` → `morse_decode`: exactly the text that was sent, with no `unknown` or `uncertain`.
3. **Noise:** a carrier keyed with 200 random characters at 20 wpm (seed 652) in white noise: the character error rate (insertions, deletions and substitutions after alignment) is at most 1 % at an SNR of 12 dB in the 500 Hz detection bandwidth and at most 5 % at 6 dB; the measured values at 6, 9, 12 and 15 dB go into the decision log. A target that cannot be reached is reported with the measured value; it is not relaxed in the test.
4. **Speed drift and steps:** a text keyed with the speed rising and falling by ±20 % over the text (seed 653): no error; a step of ±30 % at an idle gap of 10 word gaps: the first characters after the step may be wrong, the rest must be right and `wpm()` follows within 10 characters.
5. **Frequency offset:** the carrier at the receiver centre ±50 Hz (±100 Hz for the 250 Hz cut-off of the flow): decoded without error at 15 dB.
6. **Key timing faults:** sloppy keying (dot and dash lengths with ±20 % random error, gaps ±30 %, seed 654): the character error rate stays at most 2 % at 15 dB; this is the handsender case.
7. **Prosigns, punctuation, unknown bytes:** each is sent and received as written; a byte without a code is skipped by the encoder and counted.
8. **Chunk invariance** of `morse_decode` and of both flows with 20 random plans each (seed 655), and of the encoder with texts split at every position.
9. **Tone variants:** `cw_tone_transmitter` → `cw_tone_receiver` round trip at 700 Hz and 450 Hz (seed 656): exact text at 20 dB.
10. **Count entries** `morse_encode_block` and `morse_decode_block` on all four platforms (`cargo xtask count --platform <p> --kernel morse_encode_block --kernel morse_decode_block --merge`) and cycles with `cargo measure`; "our code" rows, information only.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-proto`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (with the two block entries).

## Deliverables checklist

- [ ] `sdrk_proto::morse` (table, `Encoder`, `Decoder`), with the source named in the decisions log
- [ ] Blocks and kinds `morse_encode` and `morse_decode`; schema
- [ ] Python reference encoder and the golden cases
- [ ] Flows `cw_transmitter`, `cw_receiver`, `cw_tone_transmitter`, `cw_tone_receiver` with YAML twins
- [ ] Tests (seeds 650–659)
- [ ] Count entries `morse_encode_block`, `morse_decode_block` (counts and cycles)
- [ ] `docs/blocks.md`, `docs/pipeline.md`, `docs/testing.md`, `examples/README.md`, CHANGELOG, decisions log
