# M45 — Channelizer: polyphase filter bank, lane streams, `channelize` and `channel_select`

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M34](M34-multi-input.md) · Next: [M45.1](M45.1-ism-band-receiver.md)

> **Status: draft.** Written on 2026-11-07 from a design discussion with the maintainer (the kernel `channelize` and its prototype design moved here from the ISM band receiver, which became M45.1); not yet reviewed. The thresholds marked *(to be confirmed)* are first estimates and are fixed by the implementation through the decisions log.

## Goal

A polyphase channelizer that any flow can use, for any band: the search grid of the ISM band receiver (M45.1: overlapping strips for transmitters of unknown frequency) as well as fixed channel plans such as PMR446 (16 channels of 12.5 kHz). A new kernel `channelize` splits a band into M channels at once; its prototype filter takes the passband and stopband edges as parameters, so the same kernel serves both uses. A new port type `Lanes<T>` carries the M channels side by side through one output, so one channelizer node feeds any number of readers; the kind `channel_select` turns one lane into an ordinary stream. The flow `pmr446_monitor` demodulates eight PMR446 channels at the same time and sums their audio.

## Why

A flow that listens to several channels of a band today needs one `ddc` per channel, each filtering the full input rate; a polyphase filter bank computes all M channels for `L + log2(M)·5/2` per input sample, independent of the number of channels used. But a node has exactly one output that carries one stream, and a channelizer produces M streams at once. Without a way to pass them on, every application would hide its own channelizer inside a special block. A lane stream keeps the edge model of the pipeline unchanged (one output per node, fan-out to any number of readers sharing one buffer) and lets the type check reject a lane stream wired into a stream block. One channelizer then serves pulse detection (M45.1), monitoring, recording and audio channels from the same filter bank.

## Order

M45 depends on M34 (the pipeline kinds `add` and the multi-input pipeline used by `pmr446_monitor`) and through it on M26, M29 and M30; the kernel uses the FFT of M07 and the squelch of M19. It can run in parallel with M35 to M44 and M47. M45.1 (ISM band receiver) builds on it.
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

**In:** kernel `channelize` (C32) in `sdrk-core` with Fallback, Reference, facade delegation, SIMD path, contract; the prototype filter design `design::channelizer_prototype` with edge parameters; the port type `Lanes<T>` in `sdrk-blocks` (C32 only in this milestone); lane streams in the pipeline (port kind, node views, node wrapper, runner buffers, rate, delay and type checks, report, schema); the block and kind `channelize` (`ChannelizeBlock`, C32 → `Lanes<C32>`); the block and kind `channel_select` (`ChannelSelectBlock`, `Lanes<C32>` → C32); the flow `pmr446_monitor` (+ YAML); a testkit generator of NFM channel scenes; golden cases; tests; `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/backends.md`, `docs/testing.md`, `docs/performance.md`, `examples/README.md`, `CHANGELOG.md`.

**Out:** the ISM band receiver and `band_pulse_detect` (M45.1); lane streams of other element types (`CQ15`, `f32`; backlog); blocks that process all lanes at once (lane-wise `quad_demod`, `squelch`, `fm_demod`, a `lane_sum`; backlog: they would put the channels into SIMD lanes and shorten flows such as `pmr446_monitor` to a few nodes); a `lanes` pipeline output type (writing all lanes to a file; backlog); a synthesis filter bank (many channels into one transmit stream; backlog); output ports on nodes (`from_port`); templates or `for_each` in flow descriptions (backlog: the repetition in `pmr446_monitor` is the motivating case); retuning while the flow runs.

## Specification

General rules of M19 apply. Rule of thumb for the whole milestone: a lane stream behaves exactly like a stream whose element is one *time step* of M values; every rate, delay, buffer size and count of a lane port is in time steps.

### 1. Kernel `channelize`

```rust
pub struct ChannelizeState<T, S> { /* prototype taps (M·L), history, phase, FFT plan of size M */ }
impl ChannelizeState<..> {
    /// M channels (power of two, 4 … 256), prototype taps M·L (L taps per branch, 4 … 32), outputs
    /// oversampled by 2: one output vector of M channel samples per M/2 input samples.
    pub fn new(taps: S, channels: usize, history: H, fft: FftState<..>) -> Result<Self>;
    pub fn reset(&mut self);
}
pub trait Channelize<T> { fn channelize(&self, s: &mut ChannelizeState<..>, input: &[T], output: &mut [T]) -> Result<Progress>; }
```

- The polyphase analysis filter bank with 2× oversampling (the "M/2 hop" form): per M/2 new input samples, the M branch filters of L taps each produce M values, which are circularly shifted according to the output index parity (so that every channel keeps its phase) and transformed by an M-point FFT (the `fft` kernel of the backend). Channel `k` (0 … M−1, FFT order: 0 is the band centre, `k > M/2` are the negative frequencies) is centred at `k·fs/M` and has a complex output at `2·fs/M`.
- `output` is filled in groups of M values (channel 0 … M−1 of one time step); `Progress` counts input samples consumed and output values produced; a call consumes whole hops only and keeps the rest in the history.
- Class *tolerance*: SNR ≥ 100 dB against an f64 model of the same filter bank (the FFT is a tolerance kernel); the fallback, the facade and the SIMD path agree within that; chunk invariance is exact (the same hop boundaries in every chunking).

`design::channelizer_prototype(channels, taps_per_branch, passband, stopband, attenuation_db) -> taps` (f64, length M·L): a Kaiser-windowed low-pass with the passband edge at `passband·fs/M` and the stopband from `stopband·fs/M` (both fractions of the channel spacing `fs/M`, `0 < passband < stopband ≤ 1.5`; 1.5 is the largest edge that does not alias into the passband at the output rate `2·fs/M`) with `attenuation_db` (default 60), normalized to unit gain at DC; other edges are `InvalidParameter`. The edges are parameters so that the same kernel serves the search grid of the ISM band receiver (M45.1) and fixed channel plans (§6: PMR446 with 0.4 / 0.6, sharp neighbour rejection). The search grid uses `passband` 0.5, `stopband` 1.0: each channel passes ±half a spacing flat, the stopband edge stays free of aliasing at 2× oversampling, and a narrowband signal anywhere in the band lies inside the flat passband of at least one channel (the channels overlap by half a spacing on each side). Golden cases for the design with 0.5 / 1.0 and 0.4 / 0.6 (`max_abs_err` 1e-12).

### 2. Port type `Lanes<T>` (`sdrk-blocks`, `ports.rs`)

```rust
/// M streams of `T` side by side: time step after time step, each step M values (lane 0 … M−1).
pub struct Lanes<T: StreamElem>(PhantomData<T>);

#[derive(Clone, Copy)]
pub struct LanesRef<'a, T> { /* data: &'a [T], width: usize */ }
pub struct LanesMut<'a, T> { /* data: &'a mut [T], width: usize */ }

impl<'a, T> LanesRef<'a, T> {
    pub fn new(data: &'a [T], width: usize) -> Result<Self>; // width ≥ 1, data.len() % width == 0
    pub fn width(&self) -> usize;
    pub fn steps(&self) -> usize;                 // data.len() / width
    pub fn data(&self) -> &'a [T];               // all values, step-major
    pub fn step(&self, i: usize) -> Option<&'a [T]>;  // the M values of step i
}
// LanesMut: the same plus data_mut(), step_mut(i).

impl<T: StreamElem> Ports for Lanes<T> {
    type Ref<'a> = LanesRef<'a, T>;
    type Mut<'a> = LanesMut<'a, T>;
    const COUNT: usize = 1;
    // ref_len / mut_len count time steps; ref_skip / mut_skip skip whole steps.
}
```

- The width is a run-time value (it comes from the flow), not a type parameter. `LanesRef::new` with a length that is not a multiple of the width, or with width 0, is `LengthMismatch` / `InvalidParameter`.
- `Progress` of a block with a lane port counts time steps on that port.
- `describe::PortType` gets `Lanes { elem: LaneElem, width_param: &'static str }` (like `Frame { len_param }`): the description names the element type and the parameter that sets the width (`"channels"` for `channelize`). A reading block (`channel_select`, `band_pulse_detect` of M45.1) declares `width_param: ""`: it takes the width of its input.
- `Lanes<T>` is **not** a `ChainItem` in this milestone: `ChainBufs::bytes(len)` has no width. A Rust user wires `ChannelizeBlock` and `ChannelSelectBlock` by hand (an example in the rustdoc of `ChannelSelectBlock` shows it with two buffers). If the implementer finds a clean way to carry the width through `Chain`, it may be added and logged.

### 3. Lane streams in the pipeline (`sdrk-pipeline`)

- `PortKind::Lanes { elem: PortType, width: usize }` next to `Stream` and `Frames`; `Checked::output` of `channelize` is `Lanes { elem: C32, width: channels }`.
- `NodeInput::Lanes { data: &[C32], width }` and `NodeOutput::Lanes { data: &mut [C32], width }`; a node wrapper `LanesNode<B>` for blocks with a lane port on either side (or `StreamNode` extended; implementation choice, logged). A variant mismatch inside `process` stays an `Internal` error that cannot happen after a successful build.
- **Runner buffers:** a lane output is one buffer of `capacity_steps · width` values; all readers of the node (fan-out) share it without a copy, exactly like a stream buffer. Capacity, `min_input` and progress are in steps. The rule "no allocation after build" holds.
- **Rate:** the rate of a lane stream is the number of time steps per second (`2·fs/M` after `channelize`). `channel_select` outputs one sample per step, so its stream has the same rate.
- **Delay:** group delays along a lane edge are in steps; a node with two inputs never reads a lane stream in this milestone (no two-input kind accepts lanes).
- **Type checks (stage 2):** a lane source into a stream port, a stream into a lane port, a lane source into any output: `type_mismatch` at `edges[i].to` (outputs: `outputs.<name>.type`), the message names "lanes of c32, width 16". A lane port that requires a width (none in this milestone) would compare it.
- **Report:** the node entry of a lane-producing node carries `"lanes": <width>`; its `rate_out` is the step rate.
- **Clock domain:** a lane stream has the clock domain of its source, like a stream.
- **Schema:** `cargo xtask schema` regenerates `schema/sdrk-pipeline-v1.json`; the port tables of `docs/blocks.md` write the type as `lanes<c32>`.

### 4. Kind `channelize`

`ChannelizeBlock`: `C32` → `Lanes<C32>`, a thin block over the kernel `channelize` and `design::channelizer_prototype`.

| Parameter | Unit | Kind | Default | Range |
| --- | --- | --- | --- | --- |
| `channels` | none | I64, build | 16 | power of two, 4 … 256 (the kernel's range) |
| `taps_per_branch` | none | I64, build | 8 | 4 … 32 |
| `passband` | none (fraction of the channel spacing `fs/M`) | F64, build | 0.5 | (0, `stopband`) |
| `stopband` | none (fraction of the channel spacing) | F64, build | 1.0 | (`passband`, 1.5] |
| `attenuation_db` | dB | F64, build | 60 | 20 … 120 |

- The defaults are the ISM search grid of M45.1. The upper limit 1.5 of `stopband` is the largest edge that does not alias into the passband at the oversampled output rate `2·fs/M`.
- If the Kaiser estimate for the requested edges and attenuation needs more than `channels · taps_per_branch` taps, the check fails with `rate_unsupported` at `nodes.<name>.params.taps_per_branch`, and the message names the required `taps_per_branch` (so a user who asks for a sharp filter learns the number). Examples with 16 channels and 60 dB: 0.5 / 1.0 needs about 117 taps (`taps_per_branch` ≥ 8, the default fits), 0.4 / 0.6 about 291 taps (`taps_per_branch` ≥ 19).
- Lane `k` is the channel at `k·fs/M` from the input centre in FFT order (lane 0 is the centre, lanes above M/2 are the negative frequencies), as the kernel defines it.
- Timing: ratio `2/M` (one step per M/2 input samples, `Constant`), delay `(M·L − 1)/2` input samples, `min_input` M/2. Cost: the kernel's `L + log2(M)·5/2` per input sample (record the formula).
- Storage: taps, history and FFT plan of the kernel; `storage_bytes` exact as for every block.
- C32 only; any other input type is `type_mismatch`.

### 5. Kind `channel_select`

`ChannelSelectBlock`: `Lanes<C32>` → `C32`; copies one lane into a stream.

| Parameter | Unit | Kind | Default | Range |
| --- | --- | --- | --- | --- |
| `channel` | none | I64, build | required | −M/2 … M/2 − 1 |

- `channel` is the **signed** channel number: 0 is the centre channel, −1 the one below, +1 the one above (the user thinks in offsets; the block maps it to the FFT lane `channel mod M`). Outside the range for the width of its input: `rate_unsupported` at `nodes.<name>.params.channel`, the message names the width and the valid range. (A new error code instead is a judgement call; log it.)
- Timing: ratio 1 (one sample per step), delay 0, `min_input` 1 step. Cost 1 per step.
- Several `channel_select` nodes may read the same `channelize` node, also with the same `channel`.
- The block is a strided copy; no kernel. If a SIMD gather of every M-th value is faster, the implementer may add it as a kernel (`backend.require` then names it).

### 6. Flow `pmr446_monitor` (+ YAML)

PMR446 analogue channels 1 … 16 are at `446.00625 + 0.0125·(n − 1)` MHz. With the centre on channel 9 (446.10625 MHz) and 16 channels of 12.5 kHz, channel `n` is lane `n − 9` (−8 … +7).

- `rf` `cu8` (RTL-SDR) at 2.4 MS/s centred at 446.10625 MHz → `dec` (`ddc`, `decimation` 12, cutoff about 96 kHz) → 200 kS/s → `chan` (`channelize`, `channels` 16, `passband` 0.4, `stopband` 0.6, `taps_per_branch` 24) → 25 kS/s per channel.
- Channels 1 … 8: `selN` (`channel_select`, `channel` N − 9) → `gateN` (`squelch`) → `demodN` (`fm_demod`, `deviation_hz` 2500, `audio_cutoff_hz` 3000) → the seven `add` nodes `mix1 … mix7` as a tree (8 → 4 → 2 → 1) → `out` (`resample` to 48 kHz) → `audio` (`f32` stream).
- The squelch before the sum is essential: without it the noise of eight channels adds up. The sum is not normalized: with the squelch, usually one channel is open; the YAML comment says so and shows where to add a `scale`.
- The YAML comment explains the channel numbering, shows channels 9 … 16 as the obvious extension (another eight chains) and names the repetition as the motivating case of the `for_each` backlog item.
- Exact parameters of `ddc`, `squelch`, `fm_demod` and `resample` are chosen by the implementer from the existing NFM flow (`nfm_receiver`) and logged.

### 7. Testkit: NFM channel scenes

`nfm::ChannelScene { spacing_hz, channels: [(offset_index, audio: Tone | Noise | Silence, deviation_hz, level_db)], fs, noise_db }` renders an NFM baseband with continuous phase per channel, deterministic with a seed. Used by the tests below.

## Tests

Seeds 660 … 679.

1. `channelize` kernel: golden cases (seeds 660, 661: 16 and 64 channels, a tone per channel and noise, edges 0.5 / 1.0), the f64 model within 100 dB, chunk invariance with 20 plans (seed 662), the facade and the SIMD path against the fallback; a tone swept across the band in 200 steps appears with at most 0.1 dB ripple in the strongest channel (flat passband) and at least 60 dB down in channels two spacings away (0.5 / 1.0); with 0.4 / 0.6, 60 dB, M 16, L 24: a tone swept over ±0.4 spacing of a channel stays within 0.1 dB, and a tone at 0.6 spacing or more from the channel centre is at least 60 dB down *(to be confirmed)*.
2. `channelizer_prototype`: golden cases with 0.5 / 1.0 and 0.4 / 0.6 (`max_abs_err` 1e-12); the tap estimate of §4 against SciPy `kaiserord` for 20 configurations (golden case, seed 663; equal tap counts); invalid edges are `InvalidParameter`.
3. `Lanes`: `<Lanes<C32> as Ports>::COUNT == 1`; a test-only block `Lanes<C32>` → `C32` that sums the lanes; a lane view is rejected where a `StreamElem` is required (`compile_fail`); `LanesRef::new` errors; `steps`, `step(i)`; skip counts whole steps.
4. `channelize` block: equals the kernel output value for value (C32, the same plan); chunk invariance with 20 plans (seed 664); `rate_unsupported` names the required `taps_per_branch` (3 cases).
5. `channel_select`: for every lane of M 4, 16 and 256 the output equals `step(i)[lane]` of the input; signed mapping (−M/2, −1, 0, +1, M/2 − 1); out-of-range is `rate_unsupported` with the range in the message; chunk invariance (seed 665).
6. Pipeline: a `channelize` node with four `channel_select` readers builds, the readers share one buffer (the report shows one lane buffer), and their outputs equal a reference computed outside the pipeline; `chan → quad_demod`, `ddc → channel_select` and `chan → <stream output>` are `type_mismatch` with the lane type in the message; the report entry of `chan` has `"lanes": 16` and the step rate; no allocation after build (counting allocator, as in M10); chunk invariance of the whole graph with 20 plans (seed 666).
7. PMR adjacent-channel rejection: a scene with a 1 kHz tone at full deviation (2.5 kHz) on channel 5 and noise-modulated NFM 10 dB stronger on channels 4 and 6 (seed 667): the demodulated audio of channel 5 has a SINAD of at least 20 dB *(to be confirmed)*; with channels 4 and 6 at the same level as 5, at least 30 dB *(to be confirmed)*.
8. PMR eight channels: eight channels with eight different tones (300 … 2400 Hz), one at a time and all at once (seed 668): every channel's tone appears in the audio output of the flow; with all squelches closed (noise only, seed 669) the output is silent (no squelch opens in 60 s of noise *(to be confirmed)*).
9. Cost: one `channelize` node with 8 `channel_select` readers against eight `ddc` nodes for the same eight channels, both from 200 kS/s: the report cost and `cargo xtask count` (cycles per sample at avx2 by llvm-mca, instructions where no cycles exist) of both are recorded in `docs/performance.md`; no threshold, the numbers are the deliverable.
10. Flow tests: `pmr446_monitor` validates, builds and runs on a scene file (seed 670); the YAML twin parses to the same spec; chunk invariance with 20 plans (seed 671).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the `channelize` kernel and the `channelize` and `channel_select` blocks).

## Deliverables checklist

- [x] Kernel `channelize` with reference, fallback, delegation, SIMD path, contract; `design::channelizer_prototype` with edges
- [ ] `Lanes<T>`, `LanesRef`, `LanesMut`, `PortType::Lanes`; compile tests
- [ ] Lane streams in the pipeline: `PortKind::Lanes`, node views, node wrapper, runner buffers, checks, report; schema
- [ ] Block and kind `channelize`
- [ ] Block and kind `channel_select`
- [ ] Flow `pmr446_monitor` (+ YAML)
- [ ] Testkit `nfm::ChannelScene`
- [ ] Tests (seeds 660–671); cost comparison in `docs/performance.md`
- [ ] `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/backends.md`, `docs/testing.md`, `examples/README.md`, CHANGELOG, decisions log

## Status

In progress (2026-11-07).

- **Done (step 1, section 1 and test 2):** the kernel `channelize` (`sdrk_core::kernels::channelize`) with `Reference` (f64 model), `Fallback`, the facade delegation (to the fallback until the SIMD path exists) and the numeric contract in `docs/kernels.md`; the design `sdrk_blocks::design::channelizer_prototype` with the edge parameters and the Kaiser tap estimate of section 4 (`channelizer_taps`, `channelizer_taps_per_branch`); the golden cases of seeds 660, 661 (kernel), the two prototype cases and the `kaiserord` table of seed 663; test 1 for the fallback and the facade (golden cases, the f64 model within 100 dB, chunk invariance with 20 plans of seed 662, the tone sweeps of 0.5 / 1.0 and of 0.4 / 0.6 with L 24: 0.1 dB and 60 dB are met, the second confirmed at 65.3 dB), test 2, the extreme inputs of R9 and the suite cases `channelize_c32` and `extremes_m45`. Decisions D912 to D917.
- **Done (step 2, the SIMD path):** `channelize` in `sdrk-backend-simd` (generic 128-bit code, one AVX2 path also run at AVX-512, the backend FFT at the dispatched level) with the facade delegating to it; the SIMD part of test 1 (`sdrk-backend-simd/tests/channelize.rs`: against the fallback and the f64 model within 100 dB for 4 to 256 channels, chunk invariance with the plans of seed 662; the golden cases through the facade at 141.8 and 140.3 dB); the count entries `channelize` and `channelize_fallback` on all four platforms against liquid-dsp's `firpfbch2_crcf` (D919), `release` on every level; cycles per sample on every level with a core model, reference / counter model: SIMD 25.93 / 27.72 (`sse2`), 21.95 / 28.21 (`sse4.2`), 12.08 / 12.75 (`avx2`), 12.68 / 16.01 (`avx512`), 42.56 / 32.70 (`neon`), the fallback 53.40 / 58.43, 53.83 / 58.12, 34.92 / 32.41, 38.10 / 32.11, 98.73 / 81.02, ours at 0.10 to 0.19 (SIMD) and 0.23 to 0.49 (fallback) of the cycles of liquid; `simd128` 270.93 instructions per sample (fallback 436.07). Decisions D918 to D922.
- **Open:** sections 2 to 7 (`Lanes<T>`, lane streams in the pipeline, the kinds `channelize` and `channel_select`, the flow `pmr446_monitor`, `nfm::ChannelScene`) and tests 3 to 10.
