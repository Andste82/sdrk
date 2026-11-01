# M39 — FSK and GFSK receive and transmit, automatic frequency control

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M38](M38-symbol-sync.md) · Next: [M39.1](M39.1-afsk-ook-adsb.md)

## Goal

The first complete digital mode in both directions: blocks `fsk_demod` and `fsk_mod` for 2-FSK and 4-FSK, with a Gaussian pulse filter for GFSK and GMSK (`bt`); a block `afc` that measures and removes a carrier frequency offset and reports it in ppm; PN9 whitening as an option of the test framing; example flows for a GFSK packet link (receiver and transmitter) and loop tests TX → channel → RX with BER curves.

## Why

FSK is the simplest digital modulation to receive (it builds on `quad_demod`) and the one with the most users (sensors at 433/868 MHz, POCSAG, M17, packet radios). Cheap transmitters and receivers are often tens of ppm off, so a receiver needs AFC to keep the signal inside its filter; the measured offset also calibrates the device (M28 measures drift between clocks, AFC the offset of the RF oscillator).

## Order

M39 depends on M38 (`symbol_sync`, `slice`, testkit channel and BER). It can run in parallel with M40.
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

**In:** design function `design::gaussian(bt, sps, span_symbols)`; blocks `FskDemodBlock`, `FskModBlock`, `AfcBlock` (composed from existing kernels, no new kernel); kinds `fsk_demod`, `fsk_mod`, `afc`; the option `whitening` of `frame_encode`/`frame_decode` and `sdrk_proto::whiten` (PN9); flows `examples/flows/gfsk_receiver.json` and `gfsk_transmitter.json` with YAML twins; loop tests; `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, `CHANGELOG.md`, decisions log.

**Out:** AFSK, OOK, ADS-B (M39.1), protocol stacks such as POCSAG, M17 or DMR framing (backlog), coherent FSK detection, Q15 (backlog).

## Specification

### 1. `design::gaussian`

`gaussian(bt, sps, span) -> taps` (f64, length `span·sps + 1`, `sps` an integer ≥ 2, `span` 2 … 8, `bt` 0.2 … 1): `h[n] = exp(−(2π²·bt²/ln 2)·(t/T)²)` with `t/T = (n − (len−1)/2) / sps`, normalized to sum 1. Golden case `design/gaussian_bt0p5_sps8_span4` (`max_abs_err` 1e-12 against the Python formula).

### 2. `fsk_mod`

`FskModBlock`: input soft bits `f32` (±1 from `frame_encode`, one sample per bit), output C32 at `samples_per_symbol` per symbol. Per piece of input:

1. Map bits to levels: `levels` 2: `v = +1` for bit 1, `−1` for bit 0 (one bit per symbol). `levels` 4: two bits `(b0, b1)` per symbol, first bit is `b0`: `01 → +3`, `00 → +1`, `10 → −1`, `11 → −3`, divided by 3 (the M17/DMR convention).
2. Pulse shaping with `fir_interpolate_real` (factor `sps`): taps = the boxcar of `sps` ones (`bt` 0) or the convolution of that boxcar with `gaussian(bt, sps, span)` (`bt` > 0), scaled to sum `sps`, so a constant level `v` gives the output `v`.
3. `fm_mod` with `deviation_hz`: level ±1 gives `±deviation_hz` (for 4-FSK the outer symbols).

Parameters: `levels` (I64, build, 2; 2 or 4), `samples_per_symbol` (I64, build, required; 2 … 64), `deviation_hz` (Hz, **runtime**, required), `bt` (None, build, 0; 0 or 0.2 … 1; 0.5 with `deviation_hz = symbol_rate / 4` is GMSK), `span` (I64, build, 4). Ratio `sps / bits_per_symbol`, delay of the interpolation filter. Kind cost `8 + span·sps`.

### 3. `fsk_demod`

`FskDemodBlock`: input C32 (the channel already filtered by `ddc`), output soft bits `f32`. A chain of existing parts:

1. `quad_demod` with `deviation_hz` (so the outer levels give ±1).
2. Matched filter: `fir` real with `round(sps)` equal taps `1/round(sps)` (the boxcar; for `bt` > 0 the same boxcar, the Gaussian is not inverted).
3. `symbol_sync` (Gardner, real) with `symbol_rate_hz` and the loop parameters of M38 (`loop_bw` default 0.01).
4. `slice`: `levels` 2: `real` (soft bit = level). `levels` 4: two soft bits per symbol `s0 = −v` (bit 0 is 1 for negative levels), `s1 = |v| − 2/3` (bit 1 is 1 for the outer levels); add this as constellation `fsk4` to the `slice` table of M38.

Parameters: `levels`, `symbol_rate_hz` (Hz, build, required), `deviation_hz` (Hz, build, required), `loop_bw` (None, build, 0.01). `rate_unsupported` if `fs / symbol_rate_hz < 4` (FSK needs at least 4 samples per symbol here). Ratio `bits_per_symbol / sps`, `RatioKind::Variable`. Cost `12 + round(sps) + 24/sps`.

### 4. `afc`

`AfcBlock`: C32 → C32, ratio 1, closed loop per piece of `P = 256` samples:

1. Rotate the piece by `−f_est` with `rotate` (phase continuous across pieces).
2. `quad_demod` of the rotated piece gives the residual instantaneous frequency (Hz per sample, with deviation set so the output is in Hz); `r` = its mean over the piece (f64 sum).
3. `f_est = clamp(f_est + k·r, −max_offset_hz, +max_offset_hz)` with `k = 1 − exp(−P / (fs·time_constant_s))`, unless `hold` is set.

Parameters: `time_constant_s` (s, build, 0.5), `max_offset_hz` (Hz, build, required, `< fs/2`), `hold` (Bool, **runtime**, false), `carrier_hz` (Hz, build, 0; the RF frequency, only for the ppm report). Readings (as block statistics in the pipeline report and `AfcBlock::offset_hz()`, `offset_ppm()`): `offset_ppm = f_est / carrier_hz · 1e6` (NaN if `carrier_hz` is 0). Works for signals whose mean instantaneous frequency is the carrier (FSK with balanced data, FM, AM); document that it does not work for SSB. Cost 20.

### 5. Whitening

`sdrk_proto::whiten::Pn9` (`x^9 + x^5 + 1`, seed `0x1FF`, the byte-wise sequence used by many sub-GHz transceivers; the first 8 bytes `FF E1 1D 9A ED 85 33 24` in the test): XOR over length, payload and CRC. `frame_encode` and `frame_decode` get `whitening` (Choice `none`/`pn9`, build, `none`).

### 6. Flows

- `gfsk_receiver` (+ YAML): `rf` `cs16` at 1 MHz → `tune` = `ddc` (decimation 5, cutoff 60000) → `afc` (max 20000) → `demod` = `fsk_demod` (symbol rate 50000, deviation 25000) → `dec` = `frame_decode` (`whitening` pn9) → `packets`.
- `gfsk_transmitter` (+ YAML): `msg` packets (`rate_hz` 50000) → `enc` = `frame_encode` (`whitening` pn9) → `mod` = `fsk_mod` (sps 4, deviation 25000, bt 0.5) → `duc` (interpolation 5) → `rf` `cs8`, pull mode.

## Tests

1. Golden: `design/gaussian_*`; `blocks/fsk_mod_gfsk_c32` (seed 530, f64 chain, SNR ≥ 90 dB); `blocks/fsk_demod_2fsk_f32` (seed 531, float32 model, bit-exact).
2. **Loop 2-FSK (`fsk_loop_ber`):** `fsk_mod` (sps 8, h = 1) → AWGN → `fsk_demod` → BER over 10^6 bits within a factor 2 of `0.5·exp(−Eb/(2·N0))` (non-coherent 2-FSK) at Eb/N0 10 and 12 dB (seed 532). GFSK with bt 0.5 at most 1.5 dB worse (seed 533).
3. **Loop 4-FSK:** 10^5 symbols at Eb/N0 14 dB, symbol error rate below 1e-3 (seed 534).
4. **AFC:** an offset of 5 kHz and 20 ppm drift over 10 s at 868 MHz carrier: `afc` converges to within 50 Hz in 1 s, reports the ppm within 0.1 ppm; packets decode with PER < 1 % at Eb/N0 12 dB with AFC and fail without (seed 535). `hold` freezes the estimate.
5. **Flow loop:** `gfsk_transmitter` → testkit channel (Eb/N0 14 dB, offset 3 kHz, 30 ppm clock drift; seed 536) → `gfsk_receiver`: 1000 packets, at least 990 with `CRC_OK` and identical payload; whitening on both sides.
6. Chunk invariance of `fsk_demod`, `fsk_mod`, `afc` with 20 plans (seed 537); extremes (silence, full scale, NaN) do not panic and recover.
7. `cargo xtask schema --check`.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-proto`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask emu --target qemu-cortex-m4` (quick profile).

## Deliverables checklist

- [ ] `design::gaussian` with golden case
- [ ] Blocks and kinds `fsk_mod`, `fsk_demod`, `afc`; constellation `fsk4` in `slice`
- [ ] PN9 whitening in `sdrk-proto` and the framing option
- [ ] Flows `gfsk_receiver`, `gfsk_transmitter` with YAML twins
- [ ] Tests (seeds 530–537)
- [ ] `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, CHANGELOG, decisions log
