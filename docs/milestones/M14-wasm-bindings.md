# M14 — WASM bindings and npm package

> Plan: [PLAN.md › Phase 5](../PLAN.md#phase-5--wasm-bindings-small-to-medium) · Depends on: [M13](M13-simd-backend.md) · Next: [M15](M15-emulator-targets.md)

## Status

Fully completed: the `sdrk-wasm` crate (cdylib + rlib) with JS loader, builder and TypeScript declarations; the two WASM builds (with and without SIMD128) selected at load time and the opt-in relaxed-SIMD build; `cargo xtask wasm-pkg` (including `--relaxed`) and the packed tarball; tests in Node, headless Chrome and headless Firefox plus `node --test` on the packed tarball; `docs/wasm.md`. All R1.2 commands pass locally.

Partial / open (tracked in [`docs/open-items.md`](../open-items.md)): M14-1 (npm package name and publishing are a maintainer decision). The CI jobs `wasm-bindings` and `wasm-package` run on the repository and `wasm-bindings` is green. Judgement calls are in [`docs/decisions-log.md`](../decisions-log.md).

## Goal

Expose the pipeline API to JavaScript: build from JSON or a JS builder, set parameters (immediate and timed), push and pull data, read stream/frame outputs and events, introspect. Ship two WASM builds (with and without SIMD128) selected automatically at load time, an opt-in relaxed-SIMD build, an npm package, and tests in Node and headless Chrome and Firefox.

## Why

WASM is the most important binding (browser SDRs via WebUSB, the follow-up web UI). The bindings only *configure*; all processing stays in Rust, and data crosses the boundary once per block.

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

**In:** crate `bindings/wasm` (`sdrk-wasm`, `cdylib` + `rlib`), JS loader and builder, TypeScript declarations, `cargo xtask wasm-pkg`, tests, `docs/wasm.md`.

**Out:** demos and UIs (other repos), C/Python bindings (backlog), WebUSB drivers (other repos).

## Specification

### 1. Crate

- `bindings/wasm/Cargo.toml`: `crate-type = ["cdylib", "rlib"]`, dependencies `sdrk-pipeline` with `default-features = false` (no YAML), `wasm-bindgen` (pinned exact version `=x.y.z`; the `wasm-bindgen-cli` used by `xtask` must have the identical version — `xtask` checks this), `js-sys`, `serde-wasm-bindgen` only if needed for events; dev: `wasm-bindgen-test`.
- Features: default none; `relaxed-simd` (opt-in). This milestone adds the feature `relaxed-simd` to `sdrk-backend-simd` and `sdrk` as well: on `wasm32` with target feature `+relaxed-simd` it lets the C32 variants of `rotate`, `fir`/`fir_decimate`/`fir_interpolate`, `mag_sq` and `fft` use relaxed SIMD instructions (e.g. relaxed madd); on other targets it is a no-op. Those kernels must still meet the M13 contracts (test them with the relaxed build in the wasmtime CI job); documented as *not* bit-exact.
- `sdrk-wasm` does **not** add `#![forbid(unsafe_code)]`: wasm-bindgen's macro expansion may need `unsafe`; it relies on the workspace `deny` and allows only what the expansion needs (log it).
- No `wasm-pack` (its upstream organization was archived): `xtask` drives `cargo build --target wasm32-unknown-unknown --release` and `wasm-bindgen --target web` directly; optionally `wasm-opt -O3` if present in PATH (skip otherwise, log size).

### 2. JS API (wasm-bindgen exports, camelCase)

```ts
export function version(): string;
export function schema(): string;                        // JSON Schema text
export function validate(flowJson: string): Report;      // uses Registry::validate_json; never throws on invalid flows; returns diagnostics incl. syntax errors
export function backends(): Record<string, string>;      // kernel → backend for this build
export class Pipeline {
  static fromJson(flowJson: string): Pipeline;          // throws SdrkError with diagnostics
  push(input: string, data: Uint8Array | Int16Array | Float32Array, meta?: BlockMeta): void;
  pull(output: string, out: Uint8Array | Int16Array | Float32Array, source: (input: string, max: number) => Uint8Array | Int16Array | Float32Array | null): PullResult;
  readStream(output: string, out: Float32Array | Int16Array | Uint8Array): number;   // complex = interleaved re,im; returns samples
  readFrame(output: string, frame: Float32Array): FrameMeta | null;
  available(output: string): number;
  setParam(node: string, name: string, value: number | boolean | string, at?: ParamTime): void;
  pollEvent(): SdrkEvent | null;
  beginBurst(output: string, txTimeNs?: bigint): void;  endBurst(output: string): void;
  latency(output: string): { groupDelay: number; maxBuffered: number };
  config(): string;                                      // lossless JSON
  report(): Report;  stats(): NodeStats[];  reset(): void;  free(): void;
}
```
- Typed-array dtype must match the stream type (C32/f32 → `Float32Array`, CQ15/Q15 → `Int16Array`, raw → `Uint8Array`); mismatch throws `SdrkError { kind: "InvalidArgument" }`.
- Data path: `push` copies the JS typed array into a reusable WASM buffer once (allocated at build to `max_chunk` size — no allocation per call) and calls `Pipeline::push`; `readStream` copies out into the caller's array. 64-bit values (`index`, `txTimeNs`, `unixNs`) are `bigint`.
- Errors: a JS `Error` subclass `SdrkError` with fields `kind` (ErrorKind name), `code` (DiagCode for build errors), `path`, `diagnostics` (array). Rust panics must not be possible; still install `console_error_panic_hook` in debug builds only.
- `Pipeline` is not shared between workers; document the Worker pattern: one pipeline per worker, data via transferable `ArrayBuffer`s, or `SharedArrayBuffer` ring buffers when cross-origin isolated (COOP `same-origin` + COEP `require-corp`).

### 3. Package layout (`bindings/wasm/pkg-template/` → generated `target/wasm-pkg/`)

```text
package.json          # name "sdrk-wasm" (unscoped; check availability on npm before the first publish — log), type "module", exports "./index.js", types "./index.d.ts", files list
index.js              # loader: detects SIMD128 (WebAssembly.validate of a 1-function module using v128 — inline bytes, no dependency), loads sdrk_simd.wasm or sdrk_scalar.wasm; works in browsers (fetch + instantiateStreaming) and Node ≥ 20 (fs.readFile)
builder.js            # FlowBuilder: .input(name, spec) .node(name, kind, params, opts) .edge(from, to) .output(name, spec) .settings(obj) .toJson() → same model as JSON files
index.d.ts            # hand-written types for everything above (incl. Report, Diagnostic, SdrkEvent union, BlockMeta, FrameMeta)
sdrk_simd.js/.wasm    # wasm-bindgen output, built with +simd128
sdrk_scalar.js/.wasm  # without SIMD, built with RUSTFLAGS="--cfg sdrk_fallback_only" (backends() reports only "fallback")
README.md             # usage, worker pattern, cross-origin isolation, relaxed-SIMD note
```
`cargo xtask wasm-pkg [--relaxed]` builds both variants (and `sdrk_relaxed.wasm` with `--relaxed`, selectable only explicitly via `init({ variant: "relaxed" })`), assembles the package, runs `npm pack` if npm exists. Report the `.wasm` sizes in `docs/wasm.md`; soft limit 1.5 MiB per variant after `wasm-opt` (warn above).

### 4. Tests

- Rust-side `wasm-bindgen-test` (in `bindings/wasm/tests/`), run in Node and headless Chrome and Firefox (`wasm-bindgen-test-runner`; CI installs chromedriver/geckodriver via the browser-actions setup actions):
  1. `version`, `schema` parse as JSON; `validate` on each example flow is clean and on an invalid flow returns the expected diagnostic code/path.
  2. FM receive: generate the E2E-RX CU8 signal in Rust test code, push through `Pipeline.fromJson(fm_receiver.json)`, read audio; SINAD ≥ 40 dB (f32 flow) — compute with the testkit meter.
  3. Reproducibility: a native test run with `RUSTFLAGS="--cfg sdrk_fallback_only"` writes a checksum (SHA-256 of the audio bytes) and the audio SINAD into `bindings/wasm/tests/fixtures/fm_receiver.json` (committed; the native fixture test is `#[cfg(sdrk_fallback_only)]` and runs only in the CI job that sets that flag, so the normal SIMD test run is unaffected). The **scalar** WASM build must match the checksum bit for bit (fallback everywhere, so results are identical). The SIMD128 build is a different backend and is only bound to the numeric contracts: its SINAD must be ≥ fixture SINAD − 0.5 dB. The relaxed build is checked the same way as SIMD128.
  4. Pull mode with a JS callback source, bursts, events, `setParam` with `at`, wrong dtype → `SdrkError`.
- JS-side test (`bindings/wasm/js-tests/*.mjs`, run with `node --test`): install the packed tarball into a temp directory (`npm install <tgz>`), import it, run the FM flow, check SINAD with a small JS meter, check that the builder produces JSON equal to `examples/flows/fm_receiver.json` (deep equality).
- Loader test: forcing `variant: "scalar"` reproduces the fixture checksum; `"simd"` meets the SINAD rule of test 3; automatic selection picks `simd` where SIMD128 is available.

### 5. CI

Job `wasm-bindings` (ubuntu-24.04): build package, `wasm-bindgen-test` in Node, Chrome, Firefox; `node --test`. Windows/arm64 hosts: build the package only (no browsers) to satisfy R1.3.

### 6. Documentation

`docs/wasm.md`: API reference, data layout (interleaved complex), worker pattern, cross-origin isolation, variants and selection, relaxed SIMD caveat, sizes, bigint fields, error handling.

## Test vectors

No new Python vectors; a committed checksum fixture as described in test 3.

## Acceptance

R1.2 commands; `cargo xtask wasm-pkg`; Node, Chrome and Firefox tests pass; the packed tarball installs and runs in a fresh Node project.

## Deliverables checklist

- [x] `sdrk-wasm` crate with the API above (every function and method of §2, plus `chunkSize`, `dropped` and `BlockFlags`; `SdrkError` with `kind`, `code`, `path`, `diagnostics`; the feature `relaxed-simd` in `sdrk-backend-simd`, `sdrk` and `sdrk-wasm`; no `unsafe`; engine/wasm split so that the data path is tested on the host, see [wasm.md](../wasm.md))
- [x] Loader with SIMD detection, builder, TypeScript declarations, README (`bindings/wasm/pkg-template`: `index.js`, `builder.js`, `index.d.ts`, `builder.d.ts`, `README.md`; the Worker pattern and cross-origin isolation are in the README and in `docs/wasm.md`)
- [x] `cargo xtask wasm-pkg` (+ relaxed variant): checks the `wasm-bindgen-cli` version against the pinned one, builds the variants with their flags, `wasm-opt -O3` when present, assembles `target/wasm-pkg`, `npm pack`; sizes 0.95 MiB per variant (soft limit 1.5 MiB); additionally `cargo xtask wasm-test`
- [x] wasm-bindgen tests (Node/Chrome/Firefox), JS tests, fixtures (13 `wasm-bindgen-test` tests pass for `simd`, `scalar` and `relaxed` in Node 24.22, headless Chrome 155 and headless Firefox 158; 24 JS tests on the installed tarball; 15 engine tests and 4 receive tests on the host; the scalar build reproduces the fixture checksum bit for bit in all three engines and through the JS loader)
- [x] CI job, `docs/wasm.md`, CHANGELOG, decisions log (jobs `wasm-bindings`, `wasm-package`, a relaxed entry of `wasm-wasmtime`; the job `wasm-bindings` is green)
- [ ] npm publication and the reservation of the name: **not done** (open item M14-1; `sdrk-wasm` was free on 2026-10-19)

### Status of the acceptance criteria

| Criterion | State |
| --- | --- |
| R1.2 commands (fmt, clippy, tests, `xtask check`, `xtask bare-metal`), `cargo deny check`, `cargo xtask schema --check` | pass locally on Linux x86_64 |
| `cargo xtask wasm-pkg` | passes (`--relaxed`; tarball 1.1 MB) |
| Node, Chrome and Firefox tests pass | pass locally for all three variants and in the CI job `wasm-bindings` |
| the packed tarball installs and runs in a fresh Node project | passes (`js-tests` install it with `npm install <tgz>` into a temporary project) |
| backend, facade and block tests with the relaxed build under wasmtime | pass locally (wasmtime 49.0.3, `+simd128,+relaxed-simd`) |
