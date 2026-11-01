# M34 — Multi-input nodes in the pipeline

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M26](M26-block-ports.md), [M30](M30-two-port-blocks.md) · Next: [M35](M35-am-ssb-tx.md)

## Goal

A pipeline node can have two inputs. An edge names the input **port** it feeds (`"to": "mix", "port": "b"`), the runner processes a two-input node in lockstep, the validator checks that both inputs have the same rate and come from the same clock, and it inserts a delay where the two paths have different group delays. The two-input blocks of M30 (`add`, `mul`, `make_complex`, `stereo_pll`) become pipeline kinds. A proof flow `fm_stereo_receiver_primitives` builds the stereo receiver from primitives and gives the same audio as the kind `fm_stereo_demod`, bit for bit.

## Why

Until now every node has exactly one incoming edge (docs/pipeline.md › Edges), so a flow can only be a tree of single-input blocks; the primitives of M29/M30 can only be combined in Rust (`Chain`, `Fork`). Mixing two signals, building a carrier from a pilot, or feeding a modulator with a data clock (FM stereo TX in M36, RDS TX in M42) needs a node with two inputs in the flow itself.

## Order

M34 depends on M26 (port types) and M30 (the two-input blocks). It can run in parallel with M27, M28, M29, M31–M33.
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

**In:** the edge field `port`; the port model of the registry (factories declare their input ports); the lockstep runner for two-input nodes; the diagnostics `unknown_port`, `rate_mismatch`, `clock_domain`, `misaligned_inputs`; automatic delay nodes; the pipeline kinds `add`, `mul`, `make_complex`, `stereo_pll` and the kind `delay` with input alignment; schema, WebAssembly bindings, fuzz target; the flow `examples/flows/fm_stereo_receiver_primitives.json` with a YAML twin; the invalid-flow corpus; `docs/pipeline.md`, `docs/blocks.md`, `docs/timing.md`, `examples/README.md`, `CHANGELOG.md`, `docs/decisions-log.md`.

**Out:** more than two inputs per node (backlog: an adder for N inputs), two different clocks in one pipeline (backlog; the `clock_domain` check rejects it), packet ports (M37), nodes with more than one output (a fork is already possible: a source may have any number of outgoing edges), changes to existing kinds.

## Specification

### 1. Model

An edge gets an optional field `port`:

```json
{ "from": "pilot", "to": "pll", "port": "b" }
```

| Field | Type | Meaning |
| --- | --- | --- |
| `port` | string, optional | The input port of the target node: `"a"` or `"b"`. Omitted means `"a"`. Not allowed when the target is an output (`unknown_port`). |

The short form `"to": "pll.b"` is **not** supported: names may contain no dots, and one way is enough (record in D166). A node of a two-input kind needs exactly one edge per port; a single-input node accepts only port `a`. The rule "every node and every output has exactly one incoming edge" becomes "every input port of a node and every output has exactly one incoming edge"; `multiple_inputs` and `missing_input` now report per port (path `edges[i].port` for a duplicate, `nodes.<name>.ports.b` for a missing port).

Rust: `EdgeSpec { from, to, port: Option<PortName> }`, `PortName` an enum `A`, `B` serialized as `"a"`/`"b"`. `to_json(config())` writes `port` only when it is `b` (so existing flows round-trip unchanged).

### 2. Registry

`KindInfo` (the factory's static description) gets `inputs: &'static [PortInfo]`, `PortInfo { name: PortName, accepts: &'static [PortType] }`. Every existing kind has one port `a` with its current accepted types; nothing else changes for them. The port types of a two-input node are checked per edge (`type_mismatch` at `edges[i].to` as today).

New kinds (blocks of M30; the factory picks the struct by the input type, as `biquad` does):

| Kind | Port a | Port b | Output | Cost per input sample of port a |
| --- | --- | --- | --- | --- |
| `add` | C32/CQ15/f32/Q15 | the same type | the same type | 1 |
| `mul` | C32/CQ15/f32/Q15 | the same type | the same type | 4 (complex), 1 (real) |
| `make_complex` | f32/Q15 | the same type | C32/CQ15 | 1 |
| `stereo_pll` | f32/Q15 (mpx) | f32/Q15 (pilot) | C32/CQ15 | 12 |

Port b must have the same type as port a (`type_mismatch` at the edge of port b otherwise). `delay` (M29) gets no second port.

### 3. Rates and clocks

In topological order, a two-input node has two input rates. Both must be equal as exact `Rate` values: otherwise `rate_mismatch` at `edges[i].to` of the port-b edge, the message names both rates. Equal rates are not enough: two pipeline inputs with the same nominal rate come from different devices and drift (M28). Every stream carries a **clock domain**, the name of the pipeline input it derives from; a node's output inherits the domain of its input(s). Port a and port b must have the same domain, else `clock_domain` at `edges[i].to` of the port-b edge ("inputs `rf` and `audio` are different clocks; resample one with a tunable `resample` node driven by a drift controller, or use one input"). Packet ports (M37) carry no clock domain and are exempt from this check; record that here so that M37 only has to apply it.

The report gets `clock_domain` per node (`nodes[i].clock_domain`, the input name).

### 4. Alignment

The two paths into a node can have different group delays (in samples of the node's input). Let `D_a` and `D_b` be the cumulative group delays of the two paths from their common source (the latest node or input that both paths pass through; with different inputs of the same domain, from the inputs) to the node, as exact `Rate` fractions in samples at the node's input rate.

- `D_a == D_b`: nothing to do.
- The difference `|D_a − D_b|` is a whole number `k`: the build inserts an automatic node `<node>__align` of kind `delay` (`samples = k`) on the edge of the faster path, directly before the node. It appears in the report with `auto: true`. The flow field `"align": "none"` on the node turns this off (then nothing is inserted and a warning `misaligned_inputs` is reported with the difference).
- The difference is not a whole number: error `misaligned_inputs` at `nodes.<name>` with both delays in the message; the user must change a filter so that the delays fit (a fractional delay is in the backlog).

`align` is a new optional node field: `"auto"` (default) or `"none"`. Every node accepts it; on a single-input node it has no effect. The diagnostic table of `docs/pipeline.md` gets the four new codes; the invalid-flow corpus gets one file per new code.

### 5. Runner

A two-input node reads from two buffers. In push mode the runner calls the node when **both** buffers hold at least `min_input` samples; it passes `n = min(available_a, available_b, capacity)` samples of each (lockstep, M26 §3) and advances both read positions by the consumed count. The buffer capacity rule (docs/pipeline.md › Build) uses the larger of the two `R_k` of the paths. Since both paths come from the same input, each `push` gives both paths the same number of samples after their decimations, so no buffer grows without bound; add an assertion in debug builds that the difference of the two buffer fills stays below `max_chunk · R + min_input + 64`.

Pull mode: a two-input node asks its source for samples as before; the source of both paths is the same input, so `provide` is called once per round as today (test). Bursts (`end_burst`) feed zeros into the input and flush both paths.

The node wrapper (`crates/pipeline/sdrk-pipeline/src/node.rs`) gets a variant `NodeInput::Two { a: NodeElem, b: NodeElem }` with the dispatch to `Block<In = Two<A, B>>`.

### 6. Schema, WebAssembly, fuzz

`cargo xtask schema` regenerates the schema with `port` and `align`. The WebAssembly binding passes them through (it uses the same JSON). The fuzz target of the pipeline gets random `port` fields and two-input kinds in its corpus. The JSON Schema test of every example flow stays green.

### 7. Proof flow

`examples/flows/fm_stereo_receiver_primitives.json` (+ YAML with comments): `rf` `cs16` at 480 kHz → `tune` = `ddc` (decimation 2, cutoff 100000) → `quad` = `quad_demod` (deviation 75000) → two edges: to `pll` port a (mpx) and to `pilot` = `biquad` band-pass 19 kHz (the parameters `fm_stereo_demod` uses) → `pll` port b → `pll` = `stereo_pll` → `lpf` = `decimate` (complex, the audio filter of `fm_stereo_demod`) → `matrix` = `stereo_matrix` → `deemph` = `biquad` (the de-emphasis of `fm_stereo_demod`) → `audio`. The alignment is automatic: the band-pass is a biquad with delay 0 in the delay model (check; if the model gives it a non-zero delay, the build inserts `pll__align` and the test records it).

**Test `fm_stereo_primitives_equal`:** the stereo test signal of M24 through this flow and through the existing `fm_stereo_receiver.json` (same `ddc`, kind `fm_stereo_demod` with the same parameters) gives the same audio bit for bit on the Fallback backend, whole and with 20 chunk plans (seed 478). If they differ, the flow is wrong, not the kind.

## Tests

1. Model: `port` round-trip; existing flows unchanged (`to_json(from_json(x)) == x` for every example).
2. Validation: every new code with its path (corpus); `rate_mismatch` with a decimator on one path; `clock_domain` with two inputs; `misaligned_inputs` with a non-integer difference (a decimation by 3 on one path and a filter with delay 1 on the other); automatic alignment with `delay` (a FIR with 31 taps on one path gives `__align` with 15 samples).
3. Runner: `add` of two paths from one input equals the kernel `add` on the same samples; lockstep with chunk sizes 1, 7, 4096 gives the same output; pull mode with a two-input node gives the same samples as push mode; memory bound (debug assertion) over 10^7 samples.
4. Proof flow test of §7 (seed 478).
5. Fuzz target runs 60 s in CI without crash.

## Acceptance

R1.2 commands; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo test -p sdrk-wasm` (if the binding has tests); `cargo +nightly fuzz run pipeline_json -- -max_total_time=60` (or the existing fuzz command of the repo).

## Deliverables checklist

- [ ] `port` and `align` in the model, the schema and the WebAssembly binding
- [ ] Port model of the registry; four two-input kinds
- [ ] Diagnostics `unknown_port`, `rate_mismatch`, `clock_domain`, `misaligned_inputs` with corpus files
- [ ] Automatic alignment with `delay`; lockstep runner in push and pull mode
- [ ] Proof flow `fm_stereo_receiver_primitives` with YAML twin and equality test (seed 478)
- [ ] `docs/pipeline.md`, `docs/blocks.md`, `docs/timing.md`, `examples/README.md`, CHANGELOG, decisions log (D166)
