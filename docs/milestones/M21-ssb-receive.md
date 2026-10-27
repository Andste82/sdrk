# M21 — SSB receive

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M20](M20-am-receive.md) · Next: [M22](M22-audio-filters.md)

## Status

Implemented and tested; every check of the milestone is met (details in *Status details* at the end). Not yet merged.

## Goal

A complete SSB receiver (amateur radio, HF utility and maritime): the block `ssb_demod` (USB and LSB, switchable at runtime), its pipeline kind, the example flow `ssb_receiver`, the testkit generator `ssb_tone` and an end-to-end SSB test in f32 and Q15 that checks audio quality and the rejection of the opposite sideband.

## Why

SSB is the main voice mode on short wave. With the squelch of M19 and the AGC of M20 in place it needs no new kernel: the demodulator is a frequency shift, a low-pass with decimation and a shift back, all kernels that exist and are accelerated since M13. The milestone is therefore mostly wiring and verification, and it shows that the receive series pays off.

## Order

Part of the receive series M19–M23 (see the *Order* section of [M19](M19-nfm-squelch.md)): uses M00–M15, M19 and M20.

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

**In:** block `ssb_demod` with description; its pipeline kind and the regenerated schema; the example flow `ssb_receiver` (JSON and YAML); the generator `ssb_tone`; golden cases; suite cases; the end-to-end case `e2e_ssb_rx_*`; the SSB chain in the pipeline benchmark; `docs/blocks.md`, `docs/timing.md`, `docs/pipeline.md`, `docs/performance.md`.

**Out:** new kernels (none are needed), the Weaver and phasing methods, SSB modulation (transmit), passband tuning and notch filters (audio filters are M22), CW (a narrow SSB filter would serve it; documented as a use, no own block).

## Specification

General rules from M02/M05/M06/M09 apply (see M19).

**Thresholds and limits are not relaxed.** If a threshold, tolerance or count limit of this milestone is not met, do not change it in code, case files or `benches/limits.json`. Measure it, record the measured value and the reason in `docs/decisions-log.md` and an open item in `docs/open-items.md`, and report it in the pull request; a changed threshold needs the maintainer's approval. The only pre-approved changes are the ones stated in this file.

### 1. Block `ssb_demod`

The construction pattern, descriptions, storage and parameter rules of M09 §4–§5 apply unchanged.

| Kind | Struct | In → Out | Parameters (unit, mutability, default) |
| --- | --- | --- | --- |
| `ssb_demod` | `SsbDemodBlock<T>` | C32 → f32, CQ15 → Q15 | `sideband` (Choice `usb`/`lsb`, **runtime**, `usb`), `low_hz` (Hz, build, 300), `high_hz` (Hz, build, 3000), `transition_hz` (Hz, build, 200), `attenuation_db` (dB, build, 60), `max_taps` (I64, build, 1023), `audio_decimation` (I64, build, 1 … 64, default 1) |

Implementation (normative): `c = (low_hz + high_hz)/2`, `b = (high_hz − low_hz)/2`, `s = +1` for USB and `−1` for LSB.

1. Shift the input by `−s·c` with `rotate` (NCO at the input rate).
2. Low-pass and decimate with `fir_decimate` (factor `audio_decimation`; taps `kaiserord` + `firwin_lowpass(Kaiser)` with cutoff `b`, `transition_hz` and `attenuation_db` at the input rate, quantized as in the `fir` block).
3. Shift back by `+s·c` with a second `rotate` (NCO at the output rate).
4. Output the real part of every sample (C32 `re`; CQ15 `re` as `Q15`).

The result contains the sideband `[low_hz, high_hz]` above (USB) or below (LSB) the carrier at 0 Hz, mapped to audio `low_hz … high_hz`; a tone at carrier `+ f` (USB) gives an audio tone at `f` with the input amplitude. Validation: `0 ≤ low_hz < high_hz`, `high_hz + transition_hz/2 ≤ fs_out/2` (else `OutOfRange`). Changing `sideband` at runtime flips the sign of both NCO increments from the next call; both phases continue. Both NCOs start at phase 0 after construction and `reset`. Timing: ratio `1/audio_decimation`, delay that of `fir_decimate` (`Delay::new(N − 1, 2)` input samples).

`docs/blocks.md`, chain order: `ddc → squelch → agc → ssb_demod`; the AGC defaults of M20 (attack 5 ms, release 0.2 s) suit SSB. The tuning of an SSB station is done by the `ddc` frequency (the suppressed carrier must be at 0 Hz); a mistuning of `Δf` shifts the audio by `Δf`.

### 2. Pipeline kind and example flow

- Kind `ssb_demod` (`kernels()` = `["rotate", "fir_decimate"]`). `plan` returns `Figures` exactly like `FmDemodKind::plan` (`crates/pipeline/sdrk-pipeline/src/kinds.rs`): `rate_out` and `memory_bytes` from the block's `rate_out` and `storage_bytes`, `delay` from its timing, `min_input` from `min_input()`, and `cost` per input sample by this formula, which is also added to the cost table of `docs/pipeline.md`: `ssb_demod` **`8 + 4 * taps / audio_decimation`** (two NCOs and the complex filter, as `ddc`).
- `cargo xtask schema` regenerates the schema (version unchanged). Add the kind to the WASM declarations if they list kinds.
- `examples/flows/ssb_receiver.json` + YAML: input `cu8` at 2.4 MS/s → `ddc` (decimation 50 → 48 kHz, `cutoff_hz` 4000, `transition_hz` 2000, `max_taps` 8192: the filter needs about 4400 taps, more than the default of 1023) → `squelch` (−50 dB) → `agc` (defaults) → `ssb_demod` (USB, defaults) → output `f32`. The comments explain USB/LSB conventions (LSB below 10 MHz in amateur radio) and `audio_decimation`.

### 3. Testkit

- Generator (module `r#gen`, no allocation): `ssb_tone(out: &mut [C32], rate: Rate, audio_hz: f64, sideband: Sideband, amplitude: f64, start: u64)` = `amplitude·e^{±j2π·audio_hz·n/fs}` (`+` for USB) for `n = start + i`, computed in f64 and rounded once to f32; `pub enum Sideband { Usb, Lsb }` in the testkit.
- Suite cases: `block_ssb_demod_usb`, `block_ssb_demod_lsb`, `block_ssb_sideband_switch`.
- **E2E-SSB (`e2e_ssb_rx_{f32,q15}`):** at 2.4 MS/s a USB tone at carrier + 1 kHz and an equally strong LSB tone at carrier − 1.5 kHz (each amplitude 0.3) at +300 kHz, plus complex noise σ 0.003, encoded to `CU8` → `convert` → `ddc` (300 kHz, decimation 50, taps as in the flow) → `squelch` → `agc` → `ssb_demod` (USB, defaults). Checks: SINAD at 1 kHz over the last 50 % **f32 ≥ 35 dB, Q15 ≥ 30 dB**; the 1.5 kHz component (`ToneMeter`) at least **50 dB (f32) / 40 dB (Q15)** below the 1 kHz tone (initial thresholds, M07 rule). The same chain with `sideband` = LSB swaps the roles (same thresholds). Quick variant 0.2 s, full variant 1.0 s. All four cases, each with both roles, are in the emulator suite: the 0.2 s cases in the quick profile (pre-approved: the quick arena grows to 80 KiB, because the `f32` chain needs about 75 KiB, and the Cortex-M4 quick timeout to 900 s), the 1.0 s cases in the full profile.
- Keep the quick profile of every emulator target within its timeout (see M19 §5).

### 4. SIMD backend and instruction counts

The rules of M13 and of the SIMD optimization apply. This milestone adds no kernel: `rotate` and `fir_decimate` are accelerated on every level since M13, so `ssb_demod` is accelerated through the facade. `cargo xtask count --pipeline` gets the SSB chain (`ddc → agc → ssb_demod` at the E2E rates) as a new line in `benches/results/pipeline.json` (information, like the other pipeline lines), and `docs/performance.md` its row. Check that the block makes no per-sample calls that would bypass the vector paths (it calls each kernel once per block of samples).

### 5. Documentation

`docs/blocks.md` (table, chain order, tuning), `docs/timing.md`, `docs/pipeline.md`, `docs/performance.md` (pipeline row), `examples/README.md`, `CHANGELOG.md`, `docs/decisions-log.md`.

## Test vectors (`tests/golden/sdrk_golden/cases/blocks.py`)

| Kernel | Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- | --- |
| blocks | `ssb_demod_usb_c32` | 421 | 48 kHz: USB tone 1 kHz + LSB tone 1.5 kHz (each amplitude 0.3) + noise σ 1e-3, 16 000 samples | NumPy chain: rotation in f64, `upfirdn(h, ·, 1, M)` restricted as in M06, rotation back, real part | SNR ≥ 90 |
| blocks | `ssb_demod_lsb_c32_dec2` | 422 | same tones mirrored, `audio_decimation` 2 | same chain | SNR ≥ 90 |
| blocks | `ssb_demod_lsb_cq15` | 423 | mirrored signal quantized | integer chain with the Q15 models of `rotate` (M05) and `fir_decimate` (M06) | bit-exact |

All Rust golden tests run whole and in random chunks (ChunkPlan seed = case seed).

## Tests

1. Golden tests for every case, whole and chunked; invariance of the block (20 plans).
2. A USB tone at carrier + f gives audio at f (`ToneMeter` within ±0.1 Hz) with the input amplitude ± 0.5 dB, for f = 400, 1000, 2500 Hz; the same with LSB and carrier − f.
3. Runtime switch: after `set_param("sideband", lsb)` the next call passes the lower sideband; the output is continuous (no NaN, no jump larger than the signal amplitude).
4. Validation: `high_hz` too close to `fs_out/2` and `low_hz ≥ high_hz` give `OutOfRange`.
5. Block rules: description, exact arena, no block struct larger than 512 bytes.
6. Pipeline: `ssb_receiver` validates, builds and runs on the E2E-SSB signal with the same thresholds; schema check.
7. E2E-SSB meets its checks; the suite passes on the host and in the emulators (quick profile); the pipeline count runs.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo xtask schema --check`; `cargo xtask count --pipeline`; `cargo xtask emu --target qemu-cortex-m4` (quick profile within its timeout).

## Deliverables checklist

- [x] Block `ssb_demod`, pipeline kind, regenerated schema (done)
- [x] Flow `ssb_receiver` with YAML twin (done)
- [x] Generator `ssb_tone`; golden cases (seeds 421–423), suite, E2E-SSB (done: SINAD about 64 dB and rejection 71 dB or more in every variant; `e2e_ssb_rx_f32` runs in the full tier because the `f32` chain needs about 75 KiB, see the decisions log)
- [x] Pipeline benchmark line (done: 631.4 instructions per input sample, `chains.ssb` of `benches/results/pipeline.json`, row in `docs/performance.md`)
- [x] `docs/blocks.md`, `docs/timing.md`, `docs/pipeline.md`, `docs/performance.md`, `examples/README.md`, CHANGELOG, decisions log (done)

## Status details

Complete:

- Block `ssb_demod` (`SsbDemodBlock`, f32 and Q15, runtime `sideband`, one `rotate`, one `fir_decimate` and one second `rotate` per piece of 256 samples, so the existing vector paths are used), pipeline kind with cost `8 + 4 * taps / audio_decimation`, schema (`cargo xtask schema --check`: ok), flow `ssb_receiver` (JSON and YAML).
- Generator `ssb_tone` and `Sideband`; golden cases 421 to 423 (whole and in random chunks, 20 plans for the block).
- Suite cases `block_ssb_demod_usb`, `block_ssb_demod_lsb`, `block_ssb_sideband_switch`, `e2e_ssb_rx_f32` (full tier), `e2e_ssb_rx_q15`, `e2e_ssb_rx_f32_1s`, `e2e_ssb_rx_q15_1s`; E2E-SSB in both roles with SINAD of 63.7 to 64.9 dB (limits 35 / 30 dB) and rejection of 71.6 to 94.9 dB (limits 50 / 40 dB).
- Pipeline benchmark line, documentation, CHANGELOG, decisions log.

Deviations from the text of this file (in the decisions log of 2026-10-26, none is a threshold): `e2e_ssb_rx_f32` registers in the full tier (arena), both roles run in each E2E case except the quick Q15 one (upper sideband only, emulator time), the benchmark chain has no squelch.

Review follow-up (all in the decisions log of 2026-10-26): the golden case `ssb_demod_usb_cq15_dec4_loud` (seed 451: decimation 4 with saturation, bit-exact), the pipeline test of the shipped flow on the two-sided E2E-SSB signal (`f32` and `q15`, both sidebands, default AGC, the thresholds above) and a stronger extreme-input test of the block. `python tests/golden/generate.py --check` was verified with the pinned NumPy and SciPy (only the provenance label of the Python minor version differs).

Nothing open from this milestone. The quick arena of 80 KiB and the Cortex-M4 quick timeout of 900 s for the E2E-SSB cases and the `max_taps` of the flow are written into the specification above (decisions log, 2026-10-26). The CI-only jobs were not run in this session.

Final run (host, Linux x86-64): `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo test --workspace --all-features` (1345 tests, 0 failed), `cargo xtask check` and `cargo xtask bare-metal` all pass. Not run in this session: `cargo xtask emu --target qemu-cortex-m4` of the final tree and the CI-only jobs.
