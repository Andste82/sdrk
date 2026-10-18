# M00 — Workspace foundation and guardrails

> Plan: [PLAN.md › Phase 0](../PLAN.md#phase-0--foundation-small) · Depends on: nothing · Next: [M01](M01-device-foundations.md)

## Goal

Create the empty but fully guarded Cargo workspace: folder structure, four placeholder crates, the `xtask` tool with all architecture checks, lint configuration, `cargo deny`, CI, and the project documents. After this milestone every later milestone only adds code inside an already-checked frame.

## Why

The architecture rules (layers, no heap below the pipeline, no global state, no panics) are cheap to enforce from day one and expensive to retrofit. This milestone makes them automatic.

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
5. Plan (living document): https://claude.ai/code/artifact/694f271a-ba38-4b7c-b2fa-fb2e919d35c1 — snapshot in [`docs/PLAN.md`](../PLAN.md). The plan explains *why*; the milestones say *what exactly* to build.

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

**In:** workspace layout, placeholder crates `sdrk-device`, `sdrk-core`, `sdrk`, `sdrk-testkit`, `xtask` with `check` / `bare-metal` / `test-all` subcommands, lint and format config, `deny.toml`, GitHub Actions CI, docs skeleton.

**Out:** any real types or kernels (M01+), backends, pipeline, bindings, emulator targets (M15).

## Specification

### 1. Repository layout (create exactly this)

```text
sdrk/
├── Cargo.toml                 # workspace
├── rust-toolchain.toml        # channel = "stable", components = ["rustfmt","clippy"], targets below
├── clippy.toml
├── deny.toml
├── rustfmt.toml               # edition = "2024"
├── .gitattributes             # see §6
├── .cargo/config.toml         # alias xtask = "run --package xtask --"
├── .github/workflows/ci.yml
├── README.md  CHANGELOG.md  CONTRIBUTING.md  LICENSE-MIT  LICENSE-APACHE
├── crates/core/sdrk-device/   (lib, #![no_std])
├── crates/core/sdrk-core/     (lib, #![no_std])
├── crates/facade/sdrk/        (lib, #![no_std])
├── crates/testing/sdrk-testkit/ (lib, #![no_std], feature "std")
├── xtask/                     (bin)
├── tests/golden/              (empty, .gitkeep)
└── docs/
    ├── PLAN.md                (already exists)
    ├── milestones/            (already exists)
    ├── architecture.md  kernels.md  backends.md  device.md
    ├── pipeline.md  timing.md  performance.md  decisions-log.md
```

`rust-toolchain.toml` targets: `thumbv7em-none-eabihf`, `wasm32-unknown-unknown`, `wasm32-wasip1`.

### 2. Workspace `Cargo.toml`

- `resolver = "3"`, members = all crates above (use explicit paths, no globs).
- `[workspace.package]`: `edition = "2024"`, `rust-version = "1.85"`, `license = "MIT OR Apache-2.0"`, `repository = "https://github.com/Andste82/sdrk"`, `version = "0.0.1"`.
- `[workspace.dependencies]` (pin minor versions; use the newest release at the time of implementation and record it in `decisions-log.md`): `num-complex` (default-features = false, features = ["bytemuck"]), `libm`, `bytemuck` (features = ["derive"]), `heapless`. Each crate refers to them with `workspace = true`.
- `[workspace.lints.rust]`: `unsafe_code = "deny"`, `missing_docs = "deny"`, `unsafe_op_in_unsafe_fn = "deny"`. (`deny`, not `forbid`: a workspace-level `forbid` cannot be relaxed by a single crate, and backend crates later need local `#[allow(unsafe_code)]`.) Every library crate that is **not** a backend adds `#![forbid(unsafe_code)]` to its `lib.rs`.
- `[workspace.lints.clippy]`: `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `indexing_slicing`, `large_stack_frames`, `large_stack_arrays`, `dbg_macro`, `print_stdout`, `print_stderr` = "deny"; `pedantic` = { level = "warn", priority = -1 }. Every crate has `[lints] workspace = true`. Backend crates (later) allow `unsafe` locally with `#[allow(unsafe_code)]` and must document every `unsafe` block with a `// SAFETY:` comment. `xtask/src/main.rs` adds `#![allow(clippy::print_stdout, clippy::print_stderr)]` (it reports to the console).
- `[profile.test]` and `[profile.dev]`: `overflow-checks = true`. `[profile.release]`: `overflow-checks = false`, `lto = "thin"`, `codegen-units = 1`.

`clippy.toml`: `stack-size-threshold = 4096`, `array-size-threshold = 4096`.

### 3. Placeholder crates

Each crate's `lib.rs` contains the crate-level doc comment (one paragraph: purpose and layer), `#![no_std]` (all four; the testkit adds `#[cfg(feature = "std")] extern crate std;`), the lint attributes from R5.4, `#![forbid(unsafe_code)]`, and nothing else. `sdrk-testkit/Cargo.toml` declares `[features] std = []`. Dependencies between them follow R2 even though nothing is used yet: `sdrk-core → sdrk-device`, `sdrk → sdrk-core`, `sdrk-testkit → sdrk`.

### 4. `xtask` (binary crate, std, may use `anyhow`, `cargo_metadata`, `walkdir`)

Subcommands (exit code 0 = success, 1 = violation, messages on stderr, one line per finding with file:line where applicable):

| Command | What it does |
| --- | --- |
| `cargo xtask check` | runs all checks below |
| `check layers` | Reads `cargo metadata`. Maps each workspace crate to its layer by path prefix using the table in R2; inside `crates/core` the rank comes from the crate name (`sdrk-device` 0, `sdrk-core` 1). Fails if a crate depends (normal or build dependency) on a workspace crate of the same or a higher layer, if a backend depends on another backend, or if a non-testkit/non-test crate enables `sdrk-core/reference`. Dev-dependencies are exempt. |
| `check heap` | For every crate in a heap-free layer: fails if its sources contain `extern crate alloc`, `alloc::`, `std::` (outside `#[cfg(test)]` modules and outside `#[cfg(feature = "std")]` items in the testkit, and outside `#[cfg(feature = "std")]` items in backend crates, which may only use `std::arch` / `is_*_feature_detected!` there), `Vec<`, `Box<`, `String`, `vec!`, `format!`. Implement as a line scanner that skips comments and tracks `#[cfg(test)] mod` blocks by brace counting. |
| `check global-state` | For every library crate: fails on `static mut`, `thread_local!`, `lazy_static!`, `OnceCell`, `OnceLock`, `LazyLock`, `LazyCell`, and on `static` items whose type contains `Cell`, `RefCell`, `Mutex`, `RwLock` or `Atomic`. |
| `check docs` | Runs `cargo doc --workspace --no-deps` with `RUSTDOCFLAGS="-D warnings"`. |
| `cargo xtask bare-metal` | `cargo build --target thumbv7em-none-eabihf -p <each heap-free crate>` with `--no-default-features`. The list of heap-free crates is derived from the layer table, so new crates are covered automatically. |
| `cargo xtask test-all` | Placeholder in M00: runs `fmt --check`, clippy, `test --workspace --all-features`, `check`, `bare-metal`. M14/M15 extend it with WASM and emulator runs. |

Write unit tests for the layer mapping and for the scanners (feed small strings with and without violations; include a `#[cfg(test)] mod` block containing `Vec` that must be ignored).

### 5. `deny.toml`

- `[advisories]`: deny vulnerabilities and unmaintained crates (`unmaintained = "workspace"`), `yanked = "deny"`.
- `[licenses]`: allow `MIT`, `Apache-2.0`, `Apache-2.0 WITH LLVM-exception`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `Unicode-3.0`, `Zlib`; everything else (in particular GPL/LGPL/AGPL) is denied.
- `[bans]`: `multiple-versions = "warn"`.

### 6. Build hosts and CI (`.github/workflows/ci.yml`)

The project must build and test on three **host** platforms: Linux x86_64, Linux arm64 (e.g. Raspberry Pi 4/5 with 64-bit OS) and Windows x86_64. Consequences:
- `xtask` and every tool in the repo are written in Rust (or the Python golden generator); no shell scripts, no Makefiles. Use `std::path::Path`/`PathBuf` joins, never string concatenation with `/`; spawn processes with `std::process::Command` and the `cargo` binary from the `CARGO` environment variable.
- `.gitattributes`: `* text=auto eol=lf`, `*.bin binary`, `*.json text eol=lf` — golden files must stay byte-identical on Windows checkouts.
- The golden generator (M04) writes files with `newline="\n"` and binary mode, so `--check` passes on all hosts.

One job `tier1` with a host matrix, triggered on push and pull request:

| Matrix entry | Runner |
| --- | --- |
| linux-x86_64 | `ubuntu-24.04` |
| linux-arm64 | `ubuntu-24.04-arm` (GitHub-hosted Arm runner; if the repository plan does not provide it, record that in the decisions log and run this entry on a self-hosted Raspberry Pi runner with the label `linux-arm64`) |
| windows-x86_64 | `windows-latest` |

Each entry runs the steps below (on Windows the bare-metal step needs the `thumbv7em-none-eabihf` target, which `rust-toolchain.toml` installs):
1. checkout, install toolchain from `rust-toolchain.toml`, cache cargo.
2. `cargo fmt --all -- --check`
3. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
4. `cargo test --workspace --all-features`
5. `cargo xtask check`
6. `cargo xtask bare-metal`
7. `cargo deny check` (install with `taiki-e/install-action@cargo-deny`).

### 7. Documents

- `README.md`: one-paragraph project description (copy from PLAN.md › Project), pronunciation note ("sdrk, pronounced *shark*"), scope rule, link to `docs/PLAN.md` and `docs/milestones/README.md`, license section.
- `CONTRIBUTING.md`: the ground rules R1–R9 in short form, semver policy (`sdrk-device` aims at 1.0 early; all other crates 0.x until phase 9; breaking changes bump the minor version while < 1.0), changelog format (Keep a Changelog), crate publishing order (`sdrk-device`, `sdrk-core`, backends, `sdrk`, `sdrk-blocks`, `sdrk-io`, `sdrk-pipeline`, `sdrk-testkit`).
- `docs/architecture.md`: layer table (R2), the principles from PLAN.md, the extensibility rules (node groups as the single extension point; no global state; everything `Send`; edges as interfaces; determinism independent of partitioning).
- `docs/performance.md`: the instruction-count method and the 1.00× / 1.10× / 1.15× / 1.50× table from PLAN.md.
- `docs/kernels.md`, `docs/backends.md`, `docs/device.md`, `docs/pipeline.md`, `docs/timing.md`: title + one sentence "filled in by Mxx" each.
- `docs/decisions-log.md`: header line and the table header `| Date | Milestone | Decision | Reason |`.
- `CHANGELOG.md`: Keep-a-Changelog skeleton with an "Unreleased" section.

## Test vectors

None in this milestone.

## Tests

- `xtask` unit tests described in section 4.
- A **negative test** for every check, run by `cargo test -p xtask`: create a temporary copy of a tiny fake workspace (use `tempfile`) that violates exactly one rule and assert that the corresponding check reports it:
  1. `sdrk-device` depending on `sdrk-core` → layers fails.
  2. heap-free crate containing `let v: Vec<u8> = Vec::new();` → heap fails.
  3. library crate containing `static COUNTER: AtomicU32 = AtomicU32::new(0);` → global-state fails.
- Each placeholder crate has a doc test that compiles (e.g. the crate-level example `use sdrk_device as _;`).

## Acceptance

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo xtask check
cargo xtask bare-metal
cargo deny check
```
All succeed on Linux x86_64, Linux arm64 and Windows x86_64; the CI workflow passes on GitHub for all three matrix entries.

## Deliverables checklist

- [ ] Layout, workspace manifest, toolchain, lint, format and deny configuration
- [ ] Four placeholder crates following R2/R3
- [ ] `xtask` with `check`, `bare-metal`, `test-all` and negative tests
- [ ] CI workflow
- [ ] README, CONTRIBUTING, CHANGELOG, licenses, docs skeleton, decisions log
