# M12 — `sdrk-io`, device capabilities, loopback channel simulator, CLI examples

> Plan: [PLAN.md › Phase 3](../PLAN.md#phase-3--blocks-pipeline-files-large) · Depends on: [M11](M11-pipeline-time-events-tx.md) · Next: [M13](M13-simd-backend.md)

## Goal

Finish phase 3: file I/O (SigMF with annotations, WAV, raw) as `RxStream`/`TxStream` implementations, the device control capabilities and async stream traits in `sdrk-device` (checked against SoapySDR), a first driver conformance suite, the loopback channel simulator, the `fm-receiver` and `fm-transmitter` CLI examples, and the system tests that close phase 3 (FM over loopback, burst timing ±1 sample, two stations, long runs, exact event propagation).

## Why

Phase 3 is done when recordings are demodulated via a JSON flow and all loopback tests pass. Drivers live in other repos; the device traits and the conformance suite are their contract.

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

**In:** crate `crates/io/sdrk-io`; `sdrk-device` modules `control`, `asyncio`, `capabilities`; testkit `VirtualTx`, `conformance`, `channel::Loopback`; `examples/fm-receiver`, `examples/fm-transmitter`; system tests; docs.

**Out:** real drivers (other repos), USB transports, SIMD (M13).

## Specification

### 1. Device control capabilities (`sdrk-device`, still `no_std`, no heap)

```rust
pub struct Range { pub min: f64, pub max: f64, pub step: f64 /* 0 = continuous */ }
#[derive(Clone, Copy)] pub enum Direction { Rx, Tx }
pub struct Capabilities {                       // returned by reference, built from `static` data by drivers
    pub rx_channels: u8, pub tx_channels: u8, pub full_duplex: bool,
    pub formats: &'static [SampleFormat], pub hw_timestamps: bool, pub timed_commands: bool,
}
pub trait Device { type Error: SdrkError; fn capabilities(&self) -> &Capabilities; fn driver(&self) -> &str; fn serial(&self) -> Option<&str>; }
pub trait Tuner { type Error: SdrkError;
    fn set_frequency(&mut self, dir: Direction, ch: u8, hz: f64) -> Result<f64, Self::Error>;  // returns the actual frequency
    fn frequency(&self, dir: Direction, ch: u8) -> Result<f64, Self::Error>;
    fn frequency_ranges(&self, dir: Direction, ch: u8) -> &[Range]; }
pub trait RateControl { type Error: SdrkError;
    fn set_sample_rate(&mut self, dir: Direction, ch: u8, rate: Rate) -> Result<Rate, Self::Error>;
    fn sample_rates(&self, dir: Direction, ch: u8) -> &[Range]; }
pub trait GainControl { type Error: SdrkError;
    fn gain_stages(&self, dir: Direction, ch: u8) -> &[(&'static str, Range)];
    fn set_gain(&mut self, dir: Direction, ch: u8, stage: Option<&str>, db: f64) -> Result<f64, Self::Error>; // None = overall
    fn set_agc(&mut self, dir: Direction, ch: u8, on: bool) -> Result<(), Self::Error>; }
pub trait BandwidthControl { type Error: SdrkError;
    fn set_bandwidth(&mut self, dir: Direction, ch: u8, hz: f64) -> Result<f64, Self::Error>; fn bandwidths(&self, dir: Direction, ch: u8) -> &[Range]; }
pub trait HardwareTime { type Error: SdrkError;
    fn hardware_time(&self) -> Result<TimeAnchor, Self::Error>;         // anchor: current sample index ↔ time
    fn set_hardware_time(&mut self, unix_ns: i64) -> Result<(), Self::Error>; }
#[non_exhaustive] pub enum Command { Frequency { dir: Direction, ch: u8, hz: f64 }, Gain { dir: Direction, ch: u8, db: f64 } }
pub trait TimedCommands { type Error: SdrkError;
    fn schedule(&mut self, at: CommandTime, cmd: Command) -> Result<(), Self::Error>; }  // CommandTime { Index(SampleIndex), UnixNs(i64) }
```
All optional: a driver implements only what its hardware can do. `Unsupported`, `Device`, `Timeout`, `Disconnected` kinds cover the error cases.

**Async streams** (`asyncio.rs`, for WebUSB/Web Serial and async USB): `pub trait AsyncRxStream { type Error: SdrkError; fn format(&self) -> SampleFormat; fn sample_rate(&self) -> Rate; async fn read(&mut self, buf: &mut [u8]) -> Result<(usize, BlockMeta), Self::Error>; }` and `AsyncTxStream` analogous (async fn in trait, stable since Rust 1.75; deliberately **no `Send` bound** on the futures because browser futures are not `Send`; annotate both traits with `#[allow(async_fn_in_trait)]` — otherwise rustc's lint fails `clippy -D warnings` — and document why). Adapter `BlockingRx<S>` is **not** provided (no executor in the library).

**SoapySDR check:** `docs/device.md` gets a mapping table from every SoapySDR device API group (identification, channels, stream API incl. flags `SOAPY_SDR_HAS_TIME`, `END_BURST`, `ONE_PACKET`, `END_ABRUPT`, frequency incl. components, gain incl. elements, bandwidth, sample rate, clocking/time, sensors, settings) to our traits, marking gaps (sensors, arbitrary settings, frequency components, clock source → backlog with a reason). Verify the SoapySDR API names against its current header documentation when writing the table.

### 2. `sdrk-io` (layer `crates/io`, std; depends on `sdrk-device` only)

- **Raw:** `RawReader<R: Read>` (format, rate, start index, chunk size) implements `RxStream`; `RawWriter<W: Write>` implements `TxStream` (ignores timing, records flags in a counter).
- **SigMF** (spec 1.x, `core` namespace): `SigmfReader` (opens `<base>.sigmf-meta` + `<base>.sigmf-data`) implements `RxStream`; `SigmfWriter` implements `TxStream`-like `write(&[u8], &BlockMeta)` plus `annotate(Annotation)` and `finish()` (writes meta). Datatype mapping: `cu8`↔CU8, `ci8`↔CS8, `ci16_le`↔CS16, `cf32_le`↔CF32, `ri16_le`↔RS16, `rf32_le`↔RF32; others → `Unsupported`. `core:sample_rate` → `Rate` exactly when it is an integer or an exact decimal with ≤ 6 fractional digits (`num/10^k`, reduced), else `InvalidArgument`. Captures: each capture's `core:sample_start` and `core:global_index` → `BlockMeta.start = global_index`; a jump in `global_index` relative to the data position → `GAP_BEFORE`; `core:datetime` (ISO 8601 UTC `YYYY-MM-DDTHH:MM:SS[.fraction]Z`, own parser + days-from-civil, no date crate) → `TimeAnchor { index: global_index, unix_ns, source: TimeSource::Host }`; `core:frequency` → `RETUNED` flag on change. Writer creates a new capture on every gap/anchor/retune. Annotations: `Annotation { sample_start, sample_count, freq_lower_edge: Option<f64>, freq_upper_edge: Option<f64>, label: Option<String>, comment: Option<String> }`; reader exposes `annotations()`. JSON written via `serde_json` with sorted keys, LF.
- **WAV:** `WavReader` / `WavWriter`: PCM 16-bit and IEEE float 32-bit, 1 or 2 channels; `WAVE_FORMAT_EXTENSIBLE` read support; mono → real format (RS16/RF32), stereo with `iq: true` → CS16/CF32 (I = left). Reader implements `RxStream`; writer patches sizes on `finish()`. Rate must be an integer (WAV limitation).
- Error enum per R5 with `Io(String)` (kind `Io`), `Parse { file: String, reason: String }`, `Unsupported(&'static str)`.

### 3. Testkit additions (`std` feature unless stated)

- `VirtualTx` (no_std): records written bytes into caller storage and `BlockMeta`s into a `heapless::Vec<_, 64>`; can inject `UNDERRUN`/`LATE` statuses at configured indices for `poll_status`.
- `conformance::check_rx(stream, n_blocks, buf_size) -> Result<Report, Failure>` (no_std): whole samples per read, `start` strictly increasing by the previous count unless `GAP_BEFORE`, anchors consistent (`time_of` monotone), format/rate stable unless `RETUNED`. `check_tx` analogous. Run against `VirtualRx`, `VirtualTx`, `RawReader`, `SigmfReader`, `WavReader`. Drivers in other repos call these functions in their tests.
- `channel::Loopback` (std): one TX side (`TxStream`) and one RX side (`RxStream`) sharing a timeline: RX index 0 ↔ `t0_ns`. Parameters: `rate`, `format` (both sides), `delay_samples: f64` (integer part as index shift, fractional part with a 63-tap Kaiser(β 8)-windowed-sinc fractional-delay filter computed in f64), `attenuation_db`, `freq_offset_hz` (applied in f64 by exact phase `2π·f·n/fs` from the absolute index), `noise_sigma` (Gaussian from PCG32 via Box–Muller, seed parameter, generated by absolute index so chunking does not matter), `seed`. Bursts written with `tx_time_ns` are placed at RX index `round((tx_time − t0)·rate/1e9)` + delay; continuous TX starts at index 0. Silence (zeros + noise) elsewhere. RX blocks carry an anchor on the first block.

### 4. CLI examples (`examples/fm-receiver`, `examples/fm-transmitter`; argument parsing with `std::env::args` only)

- `fm-receiver --input rec.sigmf-meta | --input rec.cu8 --format cu8 --rate 2400000 [--flow examples/flows/fm_receiver.json] [--set tune.frequency_hz=600000 …] --output audio.wav`: builds the flow, pushes the file in `max_chunk` pieces with metadata, writes the audio output as 48 kHz WAV (f32), prints the report summary and events to stderr.
- `fm-transmitter --input audio.wav [--flow examples/flows/fm_transmitter.json] [--set …] --output out.cs8 | out.sigmf-meta`: pull mode with a `PullSource` over the WAV reader.
- The example binaries and `sdrk-system-tests` may print: add `#![allow(clippy::print_stdout, clippy::print_stderr)]` to their crate roots (the workspace denies printing for library code).
- Both return exit code 1 with the diagnostics JSON on stderr on invalid flows. Smoke tests in `examples/*/tests/` run the binaries via `CARGO_BIN_EXE_*` on generated files in a temp dir.

### 5. System tests (`tests/system/` as a workspace member `sdrk-system-tests`, `publish = false`; register the `tests/system` path in `xtask` as a top layer like `examples`)

| Test | Setup | Pass criterion |
| --- | --- | --- |
| `fm_over_loopback_{f32,q15}` | `fm_transmitter.json` (1 kHz tone) → `Loopback` (CS8, delay 1234.37, attenuation 6 dB, freq offset +1.5 kHz, noise for 30 dB CNR in 200 kHz) → `fm_receiver.json` with `inputs.rf.format` overridden to `cs8` (and `settings.precision` to `q15` for the Q15 variant; overrides applied to the parsed `PipelineSpec` in the test) | SINAD ≥ 30 dB (f32), ≥ 25 dB (Q15) over the last 50 % |
| `burst_timing_loopback` | TX: bursts of a constant carrier (100 ms) at `t0 + 1 s`, `t0 + 1.5 s`; RX: `ddc` stream output (the test computes |x|² from the read samples); burst start = first output sample whose power exceeds 50 % of the plateau, converted to time with the delay-compensated anchor | abs(measured − (tx_time + delay/rate)) ≤ 1 output sample period, both bursts |
| `two_stations_no_crosstalk` | two FM stations at −400 kHz (1 kHz tone) and +400 kHz (1.7 kHz tone), equal level, one flow with fan-out to two `ddc`→`fm_demod` branches | each output: SINAD of its tone ≥ 35 dB, the other tone ≤ −50 dB relative (ToneMeter) |
| `long_run_no_drift` | start index `2^50`, 20 s at 240 kHz through `resampler` (tuned, ppm +37.5) and `fm_demod` | output count equals the exact formula; anchor time of the last output within ±1 ns of the analytic value; `#[ignore]`d 10-minute variant run nightly |
| `event_propagation_exact` | events (gap, anchor, retune, overflow) at 50 random indices through `fm_receiver.json` | every event appears on `audio` at exactly the index given by the M11 mapping rule |
| `sigmf_roundtrip_pipeline` | generate E2E-RX signal → `SigmfWriter` (with a gap and annotations) → `SigmfReader` → flow | audio identical to direct push; annotations preserved |

## Test vectors

- `tests/golden/sdrk_golden/cases/io.py`: reference SigMF metadata and WAV headers produced with Python's `json` and `wave`/`struct` modules (no SciPy needed) for the formats in §2: `io/sigmf_cu8_meta` (the expected `.sigmf-meta` JSON bytes for a given capture/annotation list), `io/wav_{pcm16_mono,float32_stereo}` (header + 16 samples). Rust writers must produce identical bytes; readers must parse them.
- ISO 8601 parser: 30 fixed date strings with expected unix ns computed by Python `datetime` (case `io/iso8601_table`), including leap years, 1970-01-01, 2038+, fractional seconds with 1–9 digits.

## Tests

1. Device: `Capabilities`/traits compile in `no_std` (`bare-metal`), mock driver in tests implementing all traits; `Send` assertions; async traits usable with a trivial hand-written executor in tests (no tokio).
2. I/O golden tests (§ Test vectors); round trips raw/SigMF/WAV for every supported format; conformance checks pass for all readers.
3. Loopback: delay accuracy (impulse → peak position ±0.01 sample via parabolic interpolation), noise level within ±0.2 dB of the requested σ, chunking invariance.
4. System tests of §5.
5. Examples smoke tests.

## Acceptance

R1.2 commands; `cargo test -p sdrk-system-tests` (excluding `#[ignore]`); CI nightly runs the ignored long test on Linux x86_64. **Phase 3 is complete.**

## Deliverables checklist

- [ ] `sdrk-device` control, capabilities, async traits; SoapySDR mapping in `docs/device.md`
- [ ] `sdrk-io` raw, SigMF (captures, annotations), WAV
- [ ] Testkit `VirtualTx`, `conformance`, `Loopback`
- [ ] `fm-receiver`, `fm-transmitter`, example flows
- [ ] System tests, golden I/O vectors
- [ ] Docs, CHANGELOG, decisions log
