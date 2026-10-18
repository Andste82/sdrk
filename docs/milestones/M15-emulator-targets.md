# M15 — Embedded test targets in emulators (Cortex-M4, ESP32-P4, ESP32-S3; tier-2 RISC-V and Cortex-M55)

> Plan: [PLAN.md › Phase 6](../PLAN.md#phase-6--embedded-in-emulators-medium) · Depends on: [M14](M14-wasm-bindings.md) · Next: [M16](M16-esp32-p4-pie-backend.md)

## Goal

Run the testkit suite (kernels and blocks, receive and transmit) bare-metal inside emulators: QEMU Cortex-M4 on every commit, ESP32-P4 (esp-emulator), ESP32-S3 (Espressif QEMU fork), RISC-V and Cortex-M55 (QEMU) nightly, with quick and full profiles, all driven by `cargo xtask emu` and `cargo xtask test-all`.

## Why

The heap-free promise is only real if the code runs on microcontrollers. Emulators make this reproducible in CI without hardware; the same `targets/` will later run on real boards via probe-rs (backlog).

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

**In:** `targets/` crates, reporters, linker scripts, `xtask emu` (build, run, parse, timeout), quick/full profiles, CI jobs, `docs/embedded.md`, fallback-only instruction counts for Cortex-M4 (reported, no limit).

**Out:** embedded backends (M16, backlog), real boards (backlog).

## Specification

### 1. Common structure

- `targets/` is **excluded** from the main workspace (`exclude = ["targets/*", "benches", "fuzz"]` in the root `Cargo.toml`); each target is its own small Cargo project with `.cargo/config.toml` (target triple, linker args) and `rust-toolchain.toml` if it needs a special toolchain. Path dependencies point to `crates/testing/sdrk-testkit` (default features, no `std`) and `sdrk-blocks`.
- Each target's `main` does: set up the platform minimally (no heap, no allocator), place a 256 KiB `static` arena buffer **in the target crate** (allowed: targets are applications, not libraries; R6 applies to library crates), run `sdrk_testkit::suite::run_suite(all_cases(), tier, &mut arena, &mut reporter)`, print the protocol lines, then signal the exit status.
- Tier selection at build time: feature `full` (default quick).
- Build profile `emu` in each target: `inherits = "release"`, `overflow-checks = true`, `debug-assertions = true`, `opt-level = "s"` for MCUs with little flash, `lto = "fat"`, `codegen-units = 1`.
- Panic handler (targets only): print `SDRK-PANIC <message>` via the reporter channel, then exit with failure. Any panic is a bug (library code must not panic).
- Protocol (from M03): `SDRK-SUITE START <n>`, `SDRK-CASE PASS|FAIL <name> [detail]`, `SDRK-SUITE END passed=<p> failed=<f>`. The host decides pass/fail **only** from these lines: `END` present, `failed=0`, no `PANIC`, no `FAIL`.

### 2. Targets

| Target dir | Tier | Rust target | Emulator and machine | Output channel | Exit |
| --- | --- | --- | --- | --- | --- |
| `qemu-cortex-m4` | 1 (every commit, quick; full nightly) | `thumbv7em-none-eabihf` | `qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -nographic -semihosting-config enable=on,target=native -kernel <elf>` | semihosting (`cortex-m-semihosting` `hprintln!`) | semihosting `debug::exit(EXIT_SUCCESS/FAILURE)` |
| `qemu-cortex-m55` | 2 | `thumbv8m.main-none-eabihf` | `qemu-system-arm -machine mps3-an547 -nographic -semihosting-config enable=on,target=native -kernel <elf>` | semihosting | semihosting exit |
| `qemu-riscv32` | 2 | `riscv32imac-unknown-none-elf` (scalar fallback) | `qemu-system-riscv32 -machine virt -nographic -bios none -semihosting-config enable=on,target=native -kernel <elf>` | semihosting (`riscv-semihosting`) | semihosting exit |
| `esp32p4` | 2 | `riscv32imafc-unknown-none-elf` | `esp-emu --chip esp32p4 --firmware <merged.bin> --exit-on "SDRK-SUITE END" --timeout <t>` | UART0 → stdout | `--exit-on`; the host parses the summary (esp-emu returns 0 on match regardless of failures) |
| `esp32s3` | 2 | `xtensa-esp32s3-none-elf` (Espressif Rust toolchain via `espup`) | Espressif QEMU fork: `qemu-system-xtensa -machine esp32s3 -nographic -drive file=<flash.bin>,if=mtd,format=raw` | UART0 → stdout | host kills QEMU after `SDRK-SUITE END` or timeout |

Notes:
- **Memory maps:** take `memory.x` for mps2-an386/mps3-an547 from the QEMU machine documentation (verify addresses and sizes when implementing; document the source in `docs/embedded.md`). Use `cortex-m-rt` (and `riscv-rt` for RISC-V virt).
- **ESP32 builds:** use `esp-hal` (pinned) with `esp-println` for UART and `espflash save-image --chip <chip> --merge` to create the flash image. If `esp-hal` does not support the ESP32-P4 at implementation time, fall back to an ESP-IDF C application (`targets/esp32p4/idf/`) that links the suite as a Rust `staticlib` built for `riscv32imafc-unknown-none-elf` and prints through a C callback (`extern "C" fn sdrk_suite_run(write: extern "C" fn(*const u8, usize)) -> u32`); build it with the official ESP-IDF Docker image (pinned tag) in CI. Record the chosen path in the decisions log. The same fallback applies to the S3 if needed.
- **Tools:** QEMU from the distribution packages (Ubuntu 24.04) for Arm/RISC-V; esp-emulator via its release tarball (pinned version, checksum verified by `xtask`, not via `curl | sh`); Espressif QEMU fork release binaries (pinned tag). `xtask emu --install` downloads into `target/tools/`.

### 3. `cargo xtask emu`

`cargo xtask emu [--target <name>|--tier 1|--tier 2|--all] [--full] [--features <comma-separated target features>] [--timeout <s>]`: builds the target with profile `emu`, creates the image, runs the emulator with a timeout (quick: 300 s, full: 3600 s, per target overridable), streams output to the console and a log file `target/emu/<name>.log`, parses the protocol, prints a summary table, exits 1 on any failure or missing tool (with an install hint; `--skip-missing` turns missing tools into a skip — used on Windows where esp-emulator is unavailable).
`cargo xtask test-all` now runs: R1.2 commands, `wasm-pkg` tests if Node is present, `emu --tier 1`.

### 4. Suite content on MCUs

All `TestCase`s from the testkit run (they are `no_std`): kernels both directions, fused kernels, blocks with `ArenaStorage`, quick E2E chains (`e2e_fm_rx_*`, `e2e_fm_txrx_*` quick variants). Cases that need more than the 256 KiB arena must fail with `CapacityExceeded` reported as `FAIL` — so size them to fit; add a test in the testkit that runs the full suite on the host with a 256 KiB arena (catches oversizing before the emulator). The C32 cases run on Cortex-M4F/M55/P4 (hardware FPU) and in soft-float on RISC-V `imac` (slower; allow a longer timeout).

### 5. Instruction counts (report only)

`cargo xtask count --platform cortex-m4` runs selected kernels (same differential method as M13: K = 0 and K = 16 runs) under `qemu-system-arm` with the instruction-count plugin (plugin build as in M13, `-plugin libinsn.so -d plugin`); writes `benches/results/cortex-m4.json` with instructions/sample of the fallback. No limits yet: CMSIS-DSP comparisons come with the `arm-dsp` backend (backlog).

### 6. CI

| Job | Trigger | What |
| --- | --- | --- |
| `emu-tier1` | every push/PR (ubuntu-24.04) | `xtask emu --target qemu-cortex-m4` (quick) |
| `emu-tier2` | nightly | `xtask emu --tier 2` quick + `xtask emu --all --full` |
| `count-embedded` | nightly | `xtask count --platform cortex-m4` (report) |

## Test vectors

None new; the suite already contains its inputs (generators with fixed seeds).

## Tests

1. Each target passes quick (and full, nightly) in its emulator.
2. Negative test (host, unit test of `xtask` parser): logs with a `FAIL`, with a missing `END`, with `PANIC`, and with `failed=1` are all detected as failures; a log with only PASS lines and `END failed=0` passes.
3. A deliberately failing case makes `xtask emu` exit 1: the target crate `targets/qemu-cortex-m4` (outside the workspace, so `cargo test --all-features` never sees it) has a feature `inject-failure` that appends one always-failing `TestCase` defined in the target itself; a CI self-check step runs `xtask emu --target qemu-cortex-m4 --features inject-failure` and expects exit code 1.
4. Host test: full suite fits into a 256 KiB arena.

## Acceptance

R1.2 commands; `cargo xtask emu --target qemu-cortex-m4` and `--full`; `cargo xtask emu --tier 2` passes for `esp32p4`, `esp32s3`, `qemu-riscv32`, `qemu-cortex-m55` (on Linux x86_64). **Phase 6 is complete.**

## Deliverables checklist

- [ ] Five targets with reporters, linker scripts, panic handlers
- [ ] `xtask emu` (+ `--install`, `--skip-missing`), `test-all` extension, parser tests
- [ ] Cortex-M4 instruction-count report
- [ ] CI jobs, `docs/embedded.md`, CHANGELOG, decisions log
