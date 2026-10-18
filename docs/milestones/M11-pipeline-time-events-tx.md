# M11 — Pipeline: time, events, timed parameters, transmit (pull mode), groups, YAML, wire format, two-thread probe

> Plan: [PLAN.md › Phase 3](../PLAN.md#phase-3--blocks-pipeline-files-large), [Core concepts › Time](../PLAN.md#time), [Transmit](../PLAN.md#transmit), [Extensibility](../PLAN.md#extensibility) · Depends on: [M10](M10-pipeline-core.md) · Next: [M12](M12-io-device-loopback-examples.md)

## Goal

Complete the pipeline: sample-index time base per stream, gaps/overflow/anchors from device metadata, events moved through the graph, parameter changes at an exact sample index with acknowledgement, joining mid-stream, latency per output, pull mode for transmit with bursts and start times, node groups and backend preferences in the model, YAML input, the versioned wire format for split pipelines, and the two-thread architecture probe proving that partitioning does not change results.

## Why

Recording = live and RX/TX timing to ±1 sample are core promises of the plan. Groups and the wire format are the single extension point for threads, MCU cores, GPUs and network splits later; they must be specified and tested now, while nothing depends on them yet.

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
5. Naming: write **SDRk** only in prose, code comments (incl. rustdoc) and documentation; everything else in code is lowercase `sdrk` (identifiers, crate/package/module/file names, string literals such as error messages).
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

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.


## Scope

**In:** index domains, `push_with_meta`, events and event queue, `set_param_at`, latency, pull mode, bursts, groups/backends in the model, `split_by_groups`, YAML (feature `yaml`), wire format (`wire` module + `docs/wire-format.md`), probe test, `architecture.md` paper check.

**Out:** I/O formats and device capabilities (M12), the pipeline instruction-count benchmark (M13), a real multi-threaded runner (later).

## Specification

### 1. Index domains and mapping

- Every stream (pipeline input, each node output, each pipeline output) has its own **sample index** (`u64`) at its own rate. An input's index starts at the `start` of the first `BlockMeta` pushed (joining mid-stream), or 0 with plain `push`.
- **Mapping rule (normative):** for a node with exact ratio `r = L/M` (from `timing()`), input index `n` maps to output index `m = n0_out + ceil((n − n0_in) · L / M)`, where `(n0_in, n0_out)` is the origin pair set at build/reset/gap (initially `n0_in = start`, `n0_out = ceil(start · L / M)`). This marks the first output sample whose computation includes input `n`. Mapping through a chain applies the rule node by node. Variable-ratio nodes (tuned resampler) map exactly with their position accumulator, always **relative to the origin** so nothing overflows at large absolute indices: expose `fn map_index(&self, rel_in: u64) -> u64` on `Block` (input samples since the node's last origin → output samples since the origin; add it to the M09 trait with a default implementation using the nominal ratio `ceil(rel_in·L/M)`; the tuned resampler overrides it: the number of outputs `j` whose position `pos_j < rel_in·2^32`, computed in `u128`; log the API addition). Then `m = n0_out + map_index(n − n0_in)`.
- **Anchors are delay-compensated:** an anchor `(n, t)` on an input becomes, on a downstream stream, `(m, t')` with `m` from the mapping rule and `t' = t + ((m − n0_out)·M/L − (n − n0_in) − D) / rate_in · 1e9` ns, `D` = cumulative group delay (input samples, exact `Delay`), computed in `i128` with round-half-up. Thus `time_of(m)` on an output is the time of the *signal* in that sample, not of its arrival.

### 2. Metadata input and events

```rust
impl Pipeline {
    pub fn push_with_meta(&mut self, input: &str, data: InputData<'_>, meta: &BlockMeta) -> Result<()>;
    pub fn poll_event(&mut self) -> Option<Event>;
    pub fn node_name(&self, id: NodeId) -> &str;
    pub fn output_name(&self, id: OutputId) -> &str;
}
pub struct Event { pub output: OutputId, pub index: u64, pub kind: EventKind }
#[non_exhaustive] pub enum EventKind {
    Gap { missing: u64 }, Overflow, Underrun, Anchor(TimeAnchor), Retuned,
    BurstStart { tx_time_ns: Option<i64> }, BurstEnd, Late,
    ParamApplied { node: NodeId, param: u16 /* index in the description */, value: ParamValue, late: bool },
    EventsLost { count: u64 },
}
```
- `meta.start > expected` → `Gap { missing }`; all nodes reachable from that input are **reset** at the gap and their origins move (`n0_in = meta.start`, `n0_out = previous mapped end + ceil(missing · ratio)`), so output indices also jump by the mapped gap. Result: processing with a gap is bit-identical to processing the two segments with separately reset pipelines (tested). `meta.start < expected` → `InvalidState` error (time must not go backwards).
- Flags `OVERFLOW`, `UNDERRUN`, `RETUNED`, `LATE`, `BURST_*` and `meta.anchor` become events at the mapped positions on **every output reachable** from that input (one event per output).
- Events are stored in a queue with capacity `settings.event_queue` (default 256), allocated at build; overflow drops the newest and later emits `EventsLost`. Events are delivered in order of `(output, index)` insertion; `poll_event` never allocates.

### 3. Timed parameters

```rust
pub enum ParamTime { Immediate, InputIndex { input: String, index: u64 }, UnixNs(i64) }
impl Pipeline { pub fn set_param_at(&mut self, node: &str, name: &str, value: serde_json::Value, at: ParamTime) -> Result<()>; }
```
- The target index is mapped to the node's input index domain with §1 (`UnixNs` needs an anchor on that input: `n = anchor.index + ceil((t − anchor.unix_ns) · rate / 1e9)`; without anchor → `InvalidState`).
- The runner splits the node's next `process` call exactly at that sample, applies `set_param`, continues. The change becomes visible at the first output sample computed after it (no retroactive changes).
- Pending changes per node: capacity `settings.pending_params` (default 16) → `CapacityExceeded` when full; target in the past → applied immediately with `late: true`.
- Acknowledgement: `ParamApplied` event on every output reachable from the node, at the mapped index of the effective sample.

### 4. Latency

`pipeline.latency(output) -> Latency { group_delay: Delay /* in output samples */, max_buffered: u64 /* worst-case samples held in buffers along the path, in output samples */ }`; also in `Report` per output. `docs/timing.md` explains how both add up.

### 5. Pull mode (transmit)

```rust
pub trait PullSource { fn provide(&mut self, input: &str, max_samples: usize) -> Result<InputData<'_>, PullError>; }
impl Pipeline {
    /// Produce exactly out.len() samples (or bytes for raw outputs, whole samples only) unless the source is exhausted.
    pub fn pull(&mut self, output: &str, out: &mut [u8], source: &mut dyn PullSource) -> Result<(usize, BlockMeta)>;
    pub fn pull_samples<T: Sample>(&mut self, output: &str, out: &mut [T], source: &mut dyn PullSource) -> Result<(usize, BlockMeta)>;
    pub fn begin_burst(&mut self, output: &str, tx_time_ns: Option<i64>) -> Result<()>;
    pub fn end_burst(&mut self, output: &str) -> Result<()>;
    pub fn lead_time(&self, output: &str) -> Result<Delay>;   // cumulative group delay in output samples
    pub fn device_status(&mut self, output: &str, meta: &BlockMeta) -> Result<()>; // forward TxStream::poll_status → events
}
```
- Loop: while the output buffer holds fewer than requested samples, for every input feeding this output ask `provide(input, ceil(missing / R) + min_input)` (capped at `max_chunk`), push it, run. Surplus production stays buffered for the next pull (deterministic).
- Source exhausted (empty data): output setting `"underrun": "zeros"` (default for raw outputs) fills the rest with zeros and emits `Underrun`; `"short"` returns fewer samples.
- The returned `BlockMeta.start` is the output index of the first sample; flags carry `BURST_START` (with `tx_time_ns`) on the block that contains the first sample after `begin_burst` — `pull` stops before that sample if earlier samples are pending, so a block never spans a burst boundary.
- `end_burst`: feeds `ceil(D_in)` zero samples to every input of the path, `D_in` = cumulative group delay of the path in **input** samples (flushing the filter tails; the produced tail is `ceil(lead_time)` output samples ± 1), marks the last produced sample, the block ending there carries `BURST_END`, then resets the path's nodes. Samples are not produced between bursts until `begin_burst` (continuous mode = never calling `begin_burst`).
- Output index continues across bursts (no reset of the index).

### 6. Groups and backend preferences (model only)

```json
"groups": { "front": { "description": "DDC on core 0" }, "audio": {} },
"nodes": { "tune": { "kind": "ddc", "group": "front", "backend": { "require": "simd" } } }
```
- New `DiagCode` variants `UnknownGroup` and `BackendUnavailable` (add invalid-flow files for both). Validation: referenced groups exist (`UnknownGroup`); nodes without group belong to the implicit group `"main"`; `Report.group_edges` lists every edge whose endpoints are in different groups (these are the interfaces).
- `backend`: `{"prefer": name}` or `{"require": name}`; the factory exposes `fn kernels(&self) -> &'static [&'static str]` (kernels the block calls); `require` fails with `BackendUnavailable` if `sdrk::backend_for(kernel, type)` differs for any of them; `prefer` gives a warning if unavailable. Groups and preferences have no runtime effect yet; document that.
- `pub fn split_by_groups(spec: &PipelineSpec, registry: &Registry) -> Result<Vec<(String, PipelineSpec)>, Report>`: one sub-spec per group; a cross-group edge `a → b` becomes output `__edge_a_b` (stream, type of `a`) in the producer group and input `__edge_a_b` (`type` + exact `rate_hz` from the report) in the consumer group.

### 7. YAML (feature `yaml`, default on; the WASM binding disables it)

`PipelineSpec::from_yaml(&str)` using `serde-saphyr` (pinned version, log it). Same model, same `deny_unknown_fields`; comments allowed. Errors carry line/column. Diagnostic paths identical to JSON. No YAML writer (JSON is canonical).

### 8. Wire format (versioned, `wire` module, spec in `docs/wire-format.md`)

Message = 32-byte header + payload, all little-endian:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `b"SDRK"` |
| 4 | 1 | version = 1 |
| 5 | 1 | message type: 1 samples, 2 frame, 3 event, 4 hello (stream description), 5 param_set, 6 param_ack |
| 6 | 2 | flags (`BlockFlags` bits) |
| 8 | 2 | stream id |
| 10 | 1 | payload type: 0 raw (format in hello), 1 c32, 2 cq15, 3 f32, 4 q15, 5 UTF-8 JSON |
| 11 | 1 | reserved = 0 |
| 12 | 8 | start index (samples: first sample; frame: `first_sample`) |
| 20 | 4 | count (elements) |
| 24 | 4 | payload length in bytes |
| 28 | 4 | frames: `averaged`; otherwise reserved = 0 |

Hello payload (JSON): `{ "stream": id, "name", "type" | "format", "rate_hz": "num/den", "anchor": {…} | null }`. Event payload (JSON): the event serialized through pipeline-side serde mirror types (`WireEvent`, `WireAnchor`, `WireParamValue` with `From` conversions; `sdrk-device` and `sdrk-blocks` stay serde-free), snake_case tag. API:
```rust
pub fn encode(msg: &Message, out: &mut Vec<u8>);
pub enum Decoded { Message(Message, usize /* bytes consumed */), NeedMore(usize /* at least this many more bytes */), Skipped(usize /* unknown message type, bytes consumed */) }
pub fn decode(buf: &[u8]) -> Result<Decoded>;           // invalid magic/version/lengths → Parse error
/// Decodes all complete messages in `buf`, skipping unknown types (forward compatibility); returns messages and bytes consumed.
pub fn decode_stream(buf: &[u8]) -> Result<(Vec<Message>, usize)>;
```

### 9. Two-thread architecture probe and paper check

- Integration test `tests/probe_two_threads.rs`: split `fm_receiver_grouped.json` (new example: `tune` in group `front`, the rest in `audio`) with `split_by_groups`; run group `front` in thread A and `audio` in thread B (`std::thread::scope`), connected by `std::sync::mpsc` carrying **wire-encoded** messages (samples and events). Feed the same CU8 input in random chunks. Assert: audio output and events are bit-identical to the single-threaded pipeline.
- `docs/architecture.md` new section "Partitioning" (paper check): how a GPU group (one group mapped to a device queue, edges = upload/download buffers, wire hello describes the stream) and a network split (wire format over TCP/WebSocket, anchors and events in-band) would be built without changing kernels, blocks or the model. No code.

### 10. Example flow `examples/flows/fm_transmitter.json` (new)

Input `audio`: `type` `f32`, `rate_hz` 48000. Nodes: `up` = `resampler` (`rate_out_hz` 240000, i.e. L/M = 5/1; real block), `mod` = `fm_mod` (`deviation_hz` 75000; at 240 kHz the maximum phase step is 2π·75/240 < π), `duc` = `duc` (`interpolation` 10, `frequency_hz` 600000, defaults otherwise). Output `rf`: stream, `format` `cs8` (auto `convert_out`), `underrun` `zeros`. Settings: `precision` `f32`, `max_chunk` 4800. Also add its YAML twin (§7).

## Test vectors

No new Python vectors.

## Tests

1. Gap equivalence: input with a gap of 12 345 samples gives output identical to two separately reset runs, with output indices offset by the mapped gap; `Gap` event at the right index on every output.
2. Anchors: anchor through the FM chain → `time_of(m)` on the audio output differs from the analytic tone time by less than half an audio sample (use the M08 FM generator with known phase).
3. Joining mid-stream: start index 10⁹ — results identical to start 0 apart from indices.
4. `set_param_at` with `InputIndex`: the `ddc` frequency change takes effect exactly at the mapped sample (check the demodulated tone phase discontinuity location ±0 samples for decimation 1 and ±1 output sample for decimation 10); ack event index correct; past target → `late: true`; queue full → error.
5. Events never allocate (extend the M10 `no_alloc` test with metadata pushes and `poll_event`).
6. Pull mode: `fm_transmitter.json` (§10): pulls of random sizes produce bytes identical to a push-mode run; underrun policies; burst: `BURST_START`/`BURST_END` placement, tail length = `ceil(lead_time)` output samples ± 1, index continues.
7. Groups: unknown group error, `group_edges` content, `require` of an unavailable backend fails, `split_by_groups` output validates.
8. YAML: every example flow has a `.yaml` twin with comments; parsed spec equals the JSON spec; a YAML syntax error reports line and column.
9. Wire: round trip for every message type; truncated buffers → `NeedMore`; corrupted magic → `Parse`; property test (random messages) round-trips; golden bytes for one message of each type committed in `crates/pipeline/sdrk-pipeline/tests/wire/*.bin` (spec examples).
10. Probe: bit-identical as described in §9 (10 random chunk plans).

## Acceptance

R1.2 commands; `cargo test -p sdrk-pipeline --all-features`; `cargo test -p sdrk-pipeline --no-default-features` (without YAML) also passes.

## Deliverables checklist

- [ ] Index domains, mapping, `map_index` on `Block`
- [ ] `push_with_meta`, gap reset, events + queue
- [ ] `set_param_at`, acknowledgements, latency
- [ ] Pull mode, bursts, lead time, device status, `fm_transmitter.json` (+ YAML twin)
- [ ] Groups, backend preferences, `split_by_groups`
- [ ] YAML feature, wire format + spec, probe test, architecture paper check
- [ ] Docs (`pipeline.md`, `timing.md`, `wire-format.md`, `architecture.md`), CHANGELOG, decisions log
