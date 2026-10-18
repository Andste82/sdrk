# M10 — `sdrk-pipeline` core: model, JSON, schema, validation, build, push runner

> Plan: [PLAN.md › Phase 3](../PLAN.md#phase-3--blocks-pipeline-files-large) and [Core concepts › Pipeline](../PLAN.md#pipeline) · Depends on: [M09](M09-blocks.md) · Next: [M11](M11-pipeline-time-events-tx.md)

## Goal

Create `crates/pipeline/sdrk-pipeline`: the serde data model with JSON as canonical format, the block registry that turns `sdrk-blocks` descriptions into a versioned JSON Schema, two-stage validation with machine-readable diagnostics, the build step (types, rates, automatic conversions, buffer sizing, resource limits, cost estimate), and the single-threaded push-mode runner with zero-copy fan-out, stream and frame outputs, runtime parameters (immediate), config readout and statistics. **No allocation after `build()`.**

## Why

Host languages and AI agents describe processing as data, not code. The pipeline is the only place with heap; it allocates once at build time and then runs the heap-free blocks deterministically.

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

**In:** model, JSON I/O, format names, registry, schema generation (`cargo xtask schema`), validation, build, runner (push mode), outputs (stream + frame, drop policy), immediate parameter changes, introspection, statistics, example flows, fuzz/property tests, no-alloc test.

**Out (M11):** events, gaps/overflow handling, time anchors, parameters at a sample index with acknowledgement, pull mode/transmit, bursts, latency per output, joining mid-stream, node groups and backend preferences in the model, wire format, YAML, two-thread probe.

## Specification

### 1. Crate setup

`crates/pipeline/sdrk-pipeline` (std allowed). Dependencies: `sdrk-blocks`, `sdrk`, `sdrk-core`, `sdrk-device`, `serde` (derive), `serde_json`, `schemars` (pin the newest stable minor at implementation time; log it). Register layer `crates/pipeline` (heap allowed) in `xtask`. Define `pub struct VecStorage;` implementing `sdrk_blocks::Storage` with `Buf<T> = Vec<T>` (zero-filled `vec![T::default(); len]`, `try_reserve_exact` so that out-of-memory becomes an error instead of an abort). Library code still follows R4/R5 (no panics, `missing_docs`). Error type: see §6.

### 2. Data model (JSON, `format_version` 1)

```json
{
  "sdrk": 1,
  "name": "fm-receiver",
  "description": "Receive one FM broadcast station and output 48 kHz audio.",
  "settings": { "precision": "f32", "max_chunk": 4096 },
  "inputs":  { "rf": { "format": "cu8", "rate_hz": 2400000, "description": "RTL-SDR IQ" } },
  "nodes": {
    "tune":  { "kind": "ddc", "params": { "frequency_hz": 600000, "decimation": 10, "cutoff_hz": 100000 } },
    "demod": { "kind": "fm_demod", "params": { "deviation_hz": 75000, "deemphasis_us": 50, "audio_decimation": 5 } },
    "fft":   { "kind": "spectrum", "params": { "fft_size": 1024, "averaging": 4 } }
  },
  "edges": [
    { "from": "rf", "to": "tune" },
    { "from": "rf", "to": "fft" },
    { "from": "tune", "to": "demod" },
    { "from": "demod", "to": "audio" },
    { "from": "fft", "to": "spectrum" }
  ],
  "outputs": {
    "audio":    { "type": "stream", "capacity": 16384, "drop": "error" },
    "spectrum": { "type": "frames", "queue": 4, "drop": "drop_oldest" }
  }
}
```
Rules:
- `sdrk` = format version (must be 1). Every object may carry an optional `description` (string ≤ 2000 chars); it is preserved.
- Names of inputs, nodes and outputs share one namespace: `^[a-z][a-z0-9_]{0,63}$`, unique.
- `rate_hz`: positive integer, or a string `"num/den"` (exact fraction, reduced on load).
- `format`: a name from the table below or an object `{ "container": "i16", "bits": 12, "align": "lsb", "layout": "interleaved", "endian": "le", "full_scale": "power_of_two" | {"explicit": 2048} }` mapped to `SampleFormat` (validated by `sdrk-device`). Names: `cu8, cs8, cs16, cs12_lsb16, sc16_q11, cs16_planar, packed12, packed10, i2s32_msb, cf32, rf32, rs16` → the M01 constants. An input may instead declare `"type": "c32" | "cq15" | "f32" | "q15"` (already converted samples).
- `settings.precision`: `"f32"` (complex C32, real f32) or `"q15"` (CQ15, Q15). One precision per pipeline.
- `settings.max_chunk`: largest number of samples accepted per `push` and input (1…1 048 576, default 4096).
- `params` values: JSON number, bool, or string (for `Choice` by option name and `Format` by name). Missing params take the description default or the block's rate-dependent default (see M09 §5); parameters listed in the description's `required` (e.g. `resampler.rate_out_hz`) must be given.
- Each node has exactly one incoming edge. Fan-out (several edges from one source) is allowed. No cycles. Every node output must be connected (to a node or an output). Outputs have exactly one incoming edge.
- Output `type`: `"stream"` (from a `Block`) or `"frames"` (from a `FrameBlock`); optional `format` on stream outputs inserts an automatic `convert_out` (complex) or real converter; `capacity` (samples, default `4 · max_chunk`), `queue` (frames, default 4), `drop`: `"error"` (default; `push` fails with `OutputFull` before consuming anything) | `"drop_oldest"` | `"drop_newest"`.

Rust types (`model.rs`): `PipelineSpec`, `Settings`, `InputSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, `Precision`, `DropPolicy`, all `serde` with `#[serde(deny_unknown_fields)]`, plus `PipelineSpec::from_json(&str) -> Result<Self>`, `to_json(&self) -> String` (pretty, keys in a fixed order: the struct order above; maps sorted by key via `BTreeMap`). Unknown fields are errors (typos must not be ignored).

### 3. Registry and JSON Schema

```rust
pub struct Registry { /* kind -> Box<dyn BlockFactory> (BTreeMap) */ }
impl Registry {
    pub fn standard() -> Self;                   // every block of sdrk-blocks
    pub fn descriptions(&self) -> impl Iterator<Item = &'static BlockDescription>;
    pub fn json_schema(&self) -> serde_json::Value;
    pub fn validate(&self, spec: &PipelineSpec, opts: &BuildOptions) -> Report;
    pub fn build(&self, spec: &PipelineSpec, opts: &BuildOptions) -> Result<Pipeline, Report>;
}
pub trait BlockFactory: Send + Sync {
    fn description(&self) -> &'static BlockDescription;
    /// Parse params (already JSON-checked) into a config and validate it for this input rate and precision.
    /// `input` is the port type arriving at this node (lets e.g. `resampler` choose the complex or the real block).
    fn check(&self, params: &serde_json::Map<String, serde_json::Value>, input: PortType, rate_in: Rate, precision: Precision) -> Result<Checked, Vec<Diagnostic>>;
    /// Kernels this block calls (used by backend preferences in M11).
    fn kernels(&self) -> &'static [&'static str];
    fn build(&self, checked: &Checked, storage: &mut VecStorage) -> Result<Box<dyn Node>, Diagnostic>;
}
pub struct Checked { pub rate_out: Rate, pub memory_bytes: usize, pub cost: f64 /* MACs per input sample */, pub output: PortKind, config: Box<dyn core::any::Any + Send + Sync> }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortKind { Stream(PortType), Frames { len: usize } }

/// Typed, borrowed views used by the type-erased runner. The variant always matches the node's port types (checked at build).
pub enum NodeInput<'a> { Raw(&'a [u8]), C32(&'a [C32]), CQ15(&'a [CQ15]), F32(&'a [f32]), Q15(&'a [Q15]) }
pub enum NodeOutput<'a> { Raw(&'a mut [u8]), C32(&'a mut [C32]), CQ15(&'a mut [CQ15]), F32(&'a mut [f32]), Q15(&'a mut [Q15]), Frame(&'a mut [f32]) }

/// Object-safe wrapper around a `Block` or `FrameBlock`, implemented generically (`StreamNode<B: Block>`, `FrameNode<B: FrameBlock>`).
pub trait Node: Send {
    /// Stream nodes return produced samples in `Progress`; frame nodes return `produced = 1` and fill `meta` when a frame completed.
    fn process(&mut self, input: NodeInput<'_>, output: NodeOutput<'_>, meta: &mut Option<FrameMeta>) -> Result<Progress, sdrk_blocks::Error>;
    fn reset(&mut self);
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<(), sdrk_blocks::Error>;
    fn param(&self, name: &str) -> Result<ParamValue, sdrk_blocks::Error>;
    fn timing(&self) -> Option<BlockTiming>;   // None for frame nodes
    fn min_input(&self) -> usize;
    fn description(&self) -> &'static BlockDescription;
}
```
A variant mismatch inside `process` returns `sdrk_blocks::Error` with kind `Internal` (cannot happen after a successful build).
One factory per block, written with a small macro so the factories stay trivial: JSON value → `ParamValue` according to `ParamKind` (integers must be exact for `I64`; strings looked up in `Choice` options / format names), then `FooConfig::set` for each given param, `validate`, `rate_out`, `storage_bytes`.

Schema (`schema.rs`): draft 2020-12. `schemars` derives the static part (top-level model); the `nodes` part is generated from descriptions: `nodes.additionalProperties` = `oneOf` over kinds, each `{ "properties": { "kind": {"const": "ddc"}, "params": { "type": "object", "additionalProperties": false, "properties": { "<param>": { "type": "number"|"integer"|"boolean"|"string", "minimum", "maximum", "enum", "default", "description": "<description> Unit: <unit>. Typical: <typical>. <Build-time|Runtime>." } }, "required": [...] }, "description": "<summary>", ... } }`. `$id`: `https://github.com/Andste82/sdrk/schema/sdrk-pipeline-v1.json`.
`cargo xtask schema` writes `schema/sdrk-pipeline-v1.json` (pretty, sorted keys, LF). A test regenerates it in memory and fails if the committed file differs ("run `cargo xtask schema`"). Any change to a description changes the schema; a **breaking** change (removed param, narrowed range, changed default) requires a new schema version file `-v2.json` — record the rule in `docs/pipeline.md`.

### 4. Validation (two stages, all problems at once)

```rust
pub struct Report { pub errors: Vec<Diagnostic>, pub warnings: Vec<Diagnostic>, pub memory_bytes: usize, pub cost_per_input_sample: f64, pub nodes: Vec<NodeReport> }
pub struct Diagnostic { pub path: String, pub code: DiagCode, pub message: String }
pub struct NodeReport { pub name: String, pub kind: String, pub rate_in: Rate, pub rate_out: Rate, pub delay_samples_in: Delay, pub auto: bool }
#[non_exhaustive] pub enum DiagCode { Syntax, SchemaVersion, UnknownField, InvalidName, DuplicateName, UnknownKind, UnknownParam, WrongType, OutOfRange, MissingParam, InvalidValue, UnknownEndpoint, MultipleInputs, MissingInput, DanglingOutput, Cycle, TypeMismatch, RateUnsupported, LimitExceeded }
```
- `path` uses dots and the JSON names: `nodes.tune.params.frequency_hz`, `edges[2].to`, `inputs.rf.format`. `DiagCode` serializes as snake_case; `Report` serializes to JSON (`Report::to_json`) — this is the machine-readable error format for agents.
- Stage 1 (structure): syntax (with line/column in the message), unknown fields, names, kinds, param names/types/ranges (from descriptions).
- Stage 2 (graph): endpoints exist, one input per node/output, no cycles (Kahn's algorithm; report one node of each cycle), no dangling outputs, port types (`PortType` of the source vs. the block's input types for the chosen precision; stream vs. frames outputs), then rate propagation in topological order with exact `Rate` arithmetic (`rate_out` from the factory); errors from a block's `validate` become diagnostics with the node path.
- `validate` never builds anything and allocates only for the report. `build` runs `validate` first and returns the report as error if it has errors.

### 5. Build

- **Automatic conversions:** an input with a raw `format` gets an auto node `<input>__convert` (kind `convert`) and every edge from that input reads the converted stream, a stream output with `format` gets `<output>__convert_out`. Auto nodes appear in `NodeReport` with `auto: true` and in stats, but **not** in the config readout (it reproduces the user's spec).
- **Order:** topological order with ties broken by the order of first appearance in `edges` (deterministic).
- **Buffers:** one linear buffer per node output (and per input), shared by all readers (zero-copy fan-out: each reader has its own read position). Capacity of the buffer after node `k`: `ceil(max_chunk · R_k) + min_input(consumer) + 64`, where `R_k` is the product of the rate ratios from the input to node `k` (exact fraction, rounded up). Writers compact the buffer (move unread data to the front with `copy_within`) when the free tail is smaller than what the producer could write; this keeps every kernel input contiguous.
- **Memory:** total = buffers + block storage (`storage_bytes`); compared with `BuildOptions.limits.max_memory_bytes`.
- **Cost estimate** (MACs per pipeline input sample, documented as an estimate): `fir`: taps; `ddc`: `4 + 4·taps/decimation` … (each factory documents its formula); scaled by `R_k`. Compared with `limits.max_cost_per_input_sample`.
- **Limits** (`BuildOptions.limits`): `max_nodes` (default 256), `max_memory_bytes` (default 256 MiB), `max_cost_per_input_sample` (default ∞), `max_taps_per_node` (default 16 383). Violations → `LimitExceeded` with the node path.
- After `build()` returns, the pipeline owns all memory; processing must not allocate.

### 6. Runner (push mode) and API

```rust
pub struct Pipeline { /* nodes, buffers, order, outputs, stats */ }
pub enum InputData<'a> { Raw(&'a [u8]), C32(&'a [C32]), CQ15(&'a [CQ15]), F32(&'a [f32]), Q15(&'a [Q15]) }
impl Pipeline {
    /// Feed up to max_chunk samples to one input and run the graph until no node can make progress.
    /// All of `data` is consumed or an error is returned before anything is consumed.
    pub fn push(&mut self, input: &str, data: InputData<'_>) -> Result<()>;
    pub fn read_stream<T: Sample>(&mut self, output: &str, out: &mut [T]) -> Result<usize>;  // T must match the output type, else WrongType
    pub fn read_raw(&mut self, output: &str, out: &mut [u8]) -> Result<usize>;
    pub fn read_frame(&mut self, output: &str, frame: &mut [f32]) -> Result<Option<FrameMeta>>;
    pub fn available(&self, output: &str) -> Result<usize>;
    pub fn set_param(&mut self, node: &str, name: &str, value: serde_json::Value) -> Result<()>; // immediate: applies from the next sample the node processes; build params → ReadOnlyParameter
    pub fn config(&self) -> PipelineSpec;   // lossless: user spec + current runtime parameter values; to_json(config) of an unchanged pipeline == to_json(from_json(input))
    pub fn stats(&self) -> &[NodeStats];    // { name, calls, samples_in, samples_out, frames }
    pub fn reset(&mut self);                // all blocks reset, buffers emptied, stats kept
    pub fn report(&self) -> &Report;
}
```
- Borrowed input: when the first consumer(s) of an input can read directly from the caller's slice (always for raw inputs → `convert` node), the runner passes the slice without copying; leftover samples that a block did not consume (history-limited `min_input`) are copied into the input buffer once.
- Run loop: repeat passes over the topological order; a node runs while its input has data and its output buffer has space; stop when a full pass makes no progress. Output buffers full with `drop: "error"` → `push` checks *before* consuming that the worst-case production (`ceil(len · R)`) fits; otherwise returns `OutputFull` and consumes nothing. `drop_oldest`/`drop_newest` drop and count in stats.
- Errors (`Error`, R5): `UnknownInput`, `UnknownOutput`, `UnknownNode`, `WrongType`, `ChunkTooLarge`, `OutputFull`, `Block { node: String, source: sdrk_blocks::Error }`, `Build(Report)`, `Json(String)`; `kind()` mapping documented.
- `Pipeline: Send` (assert). Strings in errors are allowed here (upper layer), but the hot path must not create them: error construction is only on the error path — the no-alloc test runs only the success path.

### 7. Examples and documentation

- `examples/flows/fm_receiver.json` (as §2), `examples/flows/spectrum_only.json`, `examples/flows/resample_audio.json` (real input `f32` 48 kHz → 44.1 kHz).
- `docs/pipeline.md`: model reference (every field), naming rules, auto conversions, buffer sizing, diagnostic codes table, cost formulas, schema versioning rule, examples.

## Test vectors

No new Python vectors. Pipeline tests compare against blocks driven by hand (bit-exact) and reuse the M08/M09 signal generators.

## Tests

1. JSON round trip: for every example flow, `to_json(from_json(x))` parses back to an equal `PipelineSpec`; for a canonical file the bytes are identical.
2. Schema: committed schema up to date; every example flow validates against it (dev-dependency `jsonschema`); hand-written invalid flows, one per `DiagCode` (19 in this milestone; M11 adds files for its new codes), are rejected by `validate` with the expected `code` and `path` (table-driven test, files in `crates/pipeline/sdrk-pipeline/tests/invalid/*.json` with the expected diagnostic in a sibling `.expected.json`).
3. All-errors-at-once: a flow with three independent mistakes reports three diagnostics.
4. **Pipeline = hand-wired chain:** `fm_receiver.json` run on a CU8 recording generated with the testkit (E2E-RX signal, 0.5 s) gives output bit-identical to the same blocks constructed and chained manually.
5. **Block-size invariance:** pushing the same input in random chunk sizes (ChunkPlan, 10 seeds) gives bit-identical stream outputs and identical frames.
6. **Fan-out:** `rf → tune` and `rf → fft` read the same buffer; spectrum frames equal those of a standalone spectrum block.
7. **No allocation after build:** test binary `tests/no_alloc.rs` with a counting `#[global_allocator]`; after `build`, 1000 iterations of `push` + `read_stream` + `read_frame` + `set_param(frequency_hz)` (with a pre-built `serde_json::Value::Number` — note: `set_param` must not allocate for numbers) → allocation count unchanged.
8. Runtime parameter: changing `tune.frequency_hz` moves the demodulated tone exactly from the next pushed chunk.
9. Limits: each limit triggers `LimitExceeded`; cost and memory in the report are positive and monotonic in `max_chunk`/taps.
10. Drop policies behave as specified; `OutputFull` consumes nothing.
11. Property tests (proptest, 256 cases): random small graphs from the registry (random kinds, random valid params, random edges) — `validate` never panics, and if it reports no errors then `build` succeeds and a push of noise runs without error. Random byte strings and random JSON values into `from_json`/`validate` never panic.
12. Fuzz target (`fuzz/` directory, excluded from the workspace, `cargo fuzz` on Linux nightly): `from_json` + `validate` + `build`. CI: nightly job runs it for 5 minutes on Linux x86_64 only (fuzzing is not required on Windows/arm64; log this).

## Acceptance

R1.2 commands; `cargo xtask schema` leaves the tree clean; `cargo test -p sdrk-pipeline --all-features` including `no_alloc`.

## Deliverables checklist

- [ ] Crate, layer registered
- [ ] Model + JSON + format names
- [ ] Registry, factories, schema + `cargo xtask schema`
- [ ] Validation with diagnostics, build, limits, cost, buffers
- [ ] Push runner, outputs, drop policies, `set_param`, `config`, `stats`
- [ ] Example flows, invalid-flow corpus, property tests, fuzz target
- [ ] `docs/pipeline.md`, CHANGELOG, decisions log
