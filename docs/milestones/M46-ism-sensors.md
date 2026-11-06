# M46 — Sensor decoders for the 433 MHz and 868 MHz bands, sensor flows

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M45](M45-ism-band-receiver.md) · Next: [BACKLOG](BACKLOG.md)

## Goal

Turn the pulse trains of M45 into readings: generic slicers (pulse-width, pulse-position, Manchester and NRZ for FSK, with timing tolerance), a set of device decoders for common weather stations and thermometers in the 433 MHz band (on-off keying) and the 868 MHz band (FSK), merging of the repeated copies that sensors send, and readings as JSON Lines (`{"time":…,"model":…,"id":…,"temperature_C":…,…}`); flows `ism433_sensors` and `ism868_sensors` that decode every supported sensor anywhere in the band.

## Why

M45 finds every transmission in the band; users want the temperature, humidity, wind and rain values, not pulse widths. A small, well-tested set of decoders on a clean slicer interface shows the whole chain end to end and makes adding further devices a matter of one decoder each.

## Order

M46 depends on M45 (pulse trains, the `ism` generator) and through it on M39.1 and M37. It can run in parallel with M42 to M44.
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

**In:** `sdrk_proto::slicers` (PWM, PPM, Manchester, NRZ) with tolerances; `sdrk_proto::sensors` with the `SensorDecoder` trait, the `Reading` type and its JSON form, and the device decoders chosen in §3; block `SensorDecodeBlock` and kind `sensor_decode` (pulse packets → reading packets) with repeat merging; flows `ism433_sensors` and `ism868_sensors` with YAML twins; Python encoders of the chosen protocols in the golden generator; tests; `docs/blocks.md`, `docs/pipeline.md`, `docs/testing.md`, `examples/README.md` (how to run the flows with an RTL-SDR or a HackRF and read the JSON), `CHANGELOG.md`.

**Out:** devices beyond the set of §3 (backlog: one decoder each, on the same interface), transmitting sensor messages, MQTT or database output (a follow-up project reads the JSON Lines), decoders that need a key or a pairing, frequency hopping devices.

## Specification

### 0. Licence and sources (mandatory)

SDRk is MIT/Apache-2.0. **No code, table or text of GPL projects (rtl_433 and its device files in particular) is copied, translated or used as a template.** Each protocol is implemented from a public description (manufacturer documents, published protocol write-ups, own analysis of recordings); the decision log names the source of every decoder. rtl_433 may be run outside the repository on recordings or on files of the generator to cross-check readings (the result is recorded as a comparison, not as test data derived from its code).

### 1. Slicers (`sdrk_proto::slicers`)

Input: a pulse train of M45 (`sdrk_proto::pulses`). Each slicer has a configuration in µs with a tolerance (default ±20 %, minimum ±2 samples) and returns rows of bits (a gap longer than `reset_us` starts a new row; rows are how repeats appear):

- `pwm { short_us, long_us, gap_us, reset_us }`: a pulse near `short` is 1 or 0 as configured (`short_is_one`), near `long` the other; a pulse outside both is an error that ends the row.
- `ppm { short_gap_us, long_gap_us, reset_us }`: the gap after a pulse carries the bit.
- `manchester { half_bit_us, reset_us }`: pulses and gaps are cut into half-bit periods (1 or 2), the half bits decoded as IEEE 802.3 (falling edge 1) or G.E. Thomas, configurable; a violation ends the row.
- `nrz { bit_us, reset_us }` for FSK: each pulse and gap is `round(width / bit)` bits of 1 and 0.

Every slicer reports per row the number of bits, the measured average of its unit and the worst relative timing error, so a decoder can reject bad rows.

### 2. Readings and decoders (`sdrk_proto::sensors`)

```rust
pub trait SensorDecoder { fn name(&self) -> &'static str; fn modulation(&self) -> Modulation; fn decode(&self, train: &PulseTrain, out: &mut ReadingSink) -> DecodeResult; }
```

A decoder configures its slicer, finds its sync pattern and length, checks the integrity field (checksum or CRC; a decoder of a device without one requires two equal repeats and records `"mic":"repeat"`), and fills a `Reading`: `model`, `id`, optional `channel`, `battery_ok`, and the measured values with fixed field names and units (`temperature_C`, `humidity`, `wind_avg_km_h`, `wind_max_km_h`, `wind_dir_deg`, `rain_mm`, `pressure_hPa`; others only when a device has them, named in the same style), plus `mic`, `freq_MHz`, `rssi_dB` from the pulse train. `Reading::to_json` writes one JSON object with the fields in a fixed order; units are SI/metric, temperatures in °C with the device's resolution. Out-of-range values (for example a temperature beyond the device's range) reject the message.

### 3. The set of devices

At least **four decoders for the 433 MHz band** and **two for the 868 MHz band**, chosen by the implementer from common devices with a public protocol description and logged with the source (§0). The set must cover the four slicers: at least one PWM, one PPM and one Manchester device at 433 MHz, and FSK NRZ at 868 MHz; at least one device with a CRC and one with a simple checksum. Candidates to check (not a commitment; the decision log records what was chosen and why): LaCrosse TX141 family, Nexus/Prologue-style thermometers, Oregon Scientific v2.1/v3 sensors, Fine Offset/Ecowitt WH2 and WH1080 (433 MHz, OOK), LaCrosse "IT+" TX29/TX35 and Fine Offset/Ecowitt WH24/WH65 or WH31 (868 MHz, FSK). For each chosen device, the decision row states the modulation, coding, timing, message layout, integrity check and the public source.

### 4. Block and kind `sensor_decode`

`SensorDecodeBlock`: `Packets` (pulse trains) → `Packets` (readings, payload = the JSON text, `index` from the pulse train). Parameters: `decoders` (a list of decoder names, build, all), `merge_s` (s, build, 2: copies of the same reading — same model, id and values — within this time are merged into one, with `"repeats": n`), `emit_unknown` (Bool, build, false: with it, pulse trains no decoder accepts are emitted as `{"model":"unknown",…}` with their widths, for analysing new devices). Cost: 50 per pulse train (record it). No heap in the decoders (fixed row buffers; the JSON is written into a fixed buffer of the packet payload).

### 5. Flows

- `ism433_sensors` (+ YAML): the chain of `ism433_pulses` (M45) followed by `dec` = `sensor_decode` → `readings` (JSON Lines).
- `ism868_sensors` (+ YAML): the chain of `ism868_pulses` followed by `sensor_decode` → `readings`.
- `examples/README.md`: running both with an RTL-SDR (`cu8`) or a HackRF (`cs8`), two receivers for both bands at once, and the JSON output.

## Tests

1. Slicers: every slicer on generated trains with ±15 % timing error and random glitches (seed 636): exact bits; ±25 % gives an error row, not wrong bits.
2. **Golden protocol cases:** for every chosen decoder, a Python encoder in `tests/golden/sdrk_golden` (written from the same public description, independent of the Rust decoder) produces 20 messages with random fields (seed 637 + an offset per device, recorded in the case) as pulse trains and as `cu8` bursts; the Rust decoder returns exactly the encoded readings from the pulse trains.
3. **The whole chain, every frequency:** for each band a generated scene (M45 `ism::Scene` with the Python-encoded bursts; 433 MHz: seed 640, 868 MHz: seed 641) with 200 transmissions of the chosen devices (each with its usual repeats) at uniform frequencies across the band, including channel boundaries and band edges, levels 12 … 40 dB, timing errors up to ±10 %, overlapping transmissions on different frequencies: every transmission is reported once (repeats merged) with the encoded values; no reading that was not sent.
4. Integrity: a flipped bit in a CRC or checksum device gives no reading; a device without integrity check gives a reading only with two equal repeats.
5. Out-of-range values and truncated rows give no reading; `emit_unknown` emits unknown trains (seed 642).
6. Chunk invariance of `sensor_decode` (it is per packet) and of the two flows with 20 plans (seed 643).
7. Optional, when the maintainer supplies recordings of real devices (with permission to commit them, each within the vector limits): the flows decode them, and the decision log compares the readings with the display of the device or with rtl_433 run outside the repository.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-proto`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`.

## Deliverables checklist

- [ ] `sdrk_proto::slicers` (PWM, PPM, Manchester, NRZ)
- [ ] `sdrk_proto::sensors` with `SensorDecoder`, `Reading` and JSON; at least four 433 MHz and two 868 MHz decoders with their sources in the decisions log
- [ ] Block and kind `sensor_decode` with repeat merging; schema
- [ ] Python encoders and golden protocol cases
- [ ] Flows `ism433_sensors`, `ism868_sensors` with YAML twins
- [ ] Tests (seeds 636–649)
- [ ] `docs/blocks.md`, `docs/pipeline.md`, `docs/testing.md`, `examples/README.md`, CHANGELOG, decisions log
