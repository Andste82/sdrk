# M16 — ESP32-P4 PIE backend (first embedded backend) and static instruction counting

> Plan: [PLAN.md › Phase 7](../PLAN.md#phase-7--embedded-backends-large-incremental) · Depends on: [M15](M15-emulator-targets.md) · Next: see [BACKLOG.md](BACKLOG.md)

## Goal

Create `sdrk-backend-esp-pie` for the ESP32-P4 (RISC-V with Espressif's PIE SIMD extension) with the most expensive Q15 kernels — `fir`, `fir_decimate`, `dot`, `mag_sq`, `convert` (CU8/CS8/CS16 → CQ15) and the CQ15 FFT (fixed and block floating point) — verified in esp-emulator by the suite, and compared with ESP-DSP by static instruction counts of the inner loops.

## Why

The plan's first embedded backend proves the architecture (one backend crate, facade selection, suite in the emulator, instruction-count target) on the platform with the most interesting SIMD unit. The other embedded backends follow the same recipe (backlog).

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

**In:** backend crate, instruction-encoding table generator, facade wiring (feature `esp-pie`), suite run in esp-emulator with the backend, ESP-DSP comparison build, static counting tool, `docs/backends.md` section.

**Out:** ESP32-S3 PIE (different instruction set, backlog), C32 kernels on the P4 (the FPU fallback stays), real boards.

## Specification

### 1. Instruction encodings without a special toolchain

- The PIE instructions (`esp.*` mnemonics) are not known to LLVM. Normal builds must not need the Espressif toolchain. Therefore:
  - `xtask pie-encode` (run manually by a developer who has the Espressif GCC toolchain `riscv32-esp-elf-as`/`objdump` installed, or inside the pinned ESP-IDF Docker image) assembles a list of mnemonic templates from `crates/backends/sdrk-backend-esp-pie/pie/instructions.txt` with every register combination used, and writes `src/encodings.rs`: a `macro_rules!` set that emits `.insn 4, 0x…` (or `.word 0x…`) lines for `core::arch::asm!`/`global_asm!`.
  - `src/encodings.rs` is committed; a CI nightly job regenerates it in the ESP-IDF image and fails on differences.
- Source of truth for semantics: the ESP32-P4 Technical Reference Manual (PIE chapter) and ESP-DSP's P4 assembly sources (`*_arp4.S`); cite the exact documents and versions in `docs/backends.md`. Where documentation is missing, determine behaviour by experiment in esp-emulator with a small test program and record the finding in the decisions log.
- `unsafe` policy as in M13 (`#[allow(unsafe_code)]` locally, `// SAFETY:` comments, `undocumented_unsafe_blocks` denied). Every `asm!` block documents clobbered registers and PIE state (vector registers, accumulators) and leaves the PIE state as the ABI expects (verify whether PIE registers are caller- or callee-saved under the ESP-IDF ABI; bare-metal code here has no other users, but document it).
- **Silicon revision (spike, section 6).** The encodings depend on the silicon revision: `riscv32-esp-elf-as -march=rv32imafc_zicsr_zifencei_xespv2p1` and `…_xespv2p2` emit different words for the same mnemonic (`esp.zero.xacc` is `0x0000005b` for v2p1 and `0x0000001b` for v2p2), and the v2p1 words trap as illegal instructions on the v3.1 chip that esp-emu emulates. The generator targets **`xespv2p2` (silicon v3) only** (plus `xesploop2p1` for the `esp.lp.*` hardware-loop instructions); it records the assembler version and the extension names in the header of `encodings.rs`, and `xtask pie-encode --check` fails if a word of another revision appears. Boards below v3 are out of scope; say so in `docs/backends.md` and log it.
- Some mnemonics carry mode fields that only the disassembly shows (`esp.vmulas.s16.xacc q0,q1,sat`, `esp.srcmb.s16.qacc q2,a2,trunc,dyn`). `instructions.txt` names the exact form, and the generator compares the `objdump` output with the template.

### 2. Crate and facade

- `crates/backends/sdrk-backend-esp-pie`, `#![no_std]`, compiled only for `target_arch = "riscv32"` (on other targets the crate is empty so that `cargo build --workspace` on hosts still works; `cfg(all(target_arch = "riscv32", feature = "esp32p4"))` guards the code).
- `pub struct EspPie;` implements: `Fir<CQ15>`, `FirDecimate<CQ15>`, `Dot<CQ15>`, `MagSq<CQ15>`, `Convert<CQ15>` (CU8, CS8, CS16 interleaved), `Fft<CQ15>`.
- Facade feature `esp-pie` → `delegate!(… => EspPie)` for those kernels under the cfg above; `backend_for` returns `"esp-pie"`.
- Alignment: PIE 128-bit loads need 16-byte alignment; kernels handle unaligned heads/tails with the fallback for the remainder, or require aligned storage through state constructors (document per kernel; `ArenaStorage` can align to 16 when requested — add `alloc_aligned::<T>(len, align)` to the `Storage` trait with a default implementation; log the API addition). `storage_bytes` of every block that uses `alloc_aligned` includes that padding (deterministic for a 16-byte aligned arena), so M09 test 3 still passes.
- **The PIE unit is off after reset.** The first PIE instruction raises "Illegal instruction" until the CSR `CSR_PIE_STATE_REG` (`0x7f2`; low bits 00 off, 01 initial, 10 clean, 11 dirty) is set: `csrwi 0x7f2, 1`. The backend crate exposes `pub unsafe fn enable()` that does this; it is a documented precondition of every `EspPie` kernel, and `targets/esp32p4` calls it before the suite. (ESP-IDF's FreeRTOS enables the unit lazily per task; bare-metal code has no such handler.) Log the API addition.

### 3. Kernel contracts (unchanged from the fallback)

| Kernel | Contract | Implementation notes |
| --- | --- | --- |
| `convert` CU8/CS8/CS16 → CQ15 | bit-exact | vector widen + shift |
| `mag_sq` CQ15 | bit-exact | |
| `dot`, `fir`, `fir_decimate` CQ15 | **bit-exact** | MAC into the PIE accumulators; analyse the accumulator width: if a lane accumulator could overflow for the tap count, split the taps into chunks that cannot overflow and add the partial sums in scalar `i64`; extract the full-precision sum and apply R7 rounding (`round_shift(·, 15)` + saturation) in scalar unless a PIE shift/round instruction is proven to round half up identically |
| `fft` CQ15 fixed and BFP | M07 contract table | radix-2 or radix-4 butterflies with PIE complex multiply; BFP: block maximum via vector max/abs; exponent semantics as in M07 |

**Accumulator facts (measured in esp-emu, section 6).** `esp.vmulas.s16.xacc` assembles as the saturating form, and the accumulator XACC is 40 bits wide: 512 products of −32768 × −32768 (2³⁹) give 2³⁹ − 1. One instruction adds at most 8 × 2³⁰ = 2³³, so a chunk holds at most **63 vectors (504 elements)**; then it is read out (`esp.movx.r.xacc.l` and `.h`, the high word carries bits 32 to 39, sign-extend from 8 bits), cleared and added in `i64`. The `len % 8` tail is scalar code.

### 4. Verification

- Suite in esp-emulator with the backend: `targets/esp32p4` gets feature `esp-pie`; `xtask emu --target esp32p4 --features esp-pie` runs quick and full (M15 §3 defines `--features <list>`, passed through to the target build). All `backend_vs_fallback_*` cases (M13) cover the new kernels automatically because they go through the facade.
- esp-emu version: the complete PIE instruction set (checked against silicon by the emulator's authors) exists from v0.49.0, so the M15 pin must be at least that. The emulated chip is revision v3.1; esp-emu v0.50.0 refuses images built for silicon below v3.
- Extra stress cases (testkit, quick tier): lengths 0…40 and 4093…4100, unaligned slices (offset by 1 sample), taps 1…300 (crosses the chunk split), −32768 everywhere. Accumulator edge: −32768 × −32768 and 32767 × −32768 at lengths 496, 504, 512, 520 and 4096 (the spike saturated from 512 elements on).

### 5. Comparison with ESP-DSP (static instruction count)

- `benches/esp32p4/`: an ESP-IDF project (pinned ESP-IDF Docker image) that compiles ESP-DSP (pinned version) for the P4 and links the comparison functions: `dsps_fird_s16` / `dsps_fir_s16` (P4-optimized `_arp4` variant if present), `dsps_dotprod_s16` (`_arp4` if present), `dsps_fft2r_sc16` (`_arp4` if present, else `_ansi`); record which variant was used.
- Our side: the same kernels from `sdrk-backend-esp-pie`, compiled in release.
- **Method (static, as decided in the plan, because esp-emulator only samples PCs):** `cargo xtask count --platform esp32p4` disassembles both binaries (`riscv32-esp-elf-objdump` in the Docker image, or `llvm-objdump` with `--mattr` where it suffices — raw encodings print as `.insn`, which is fine for counting) and evaluates `benches/esp32p4/loops.json`: per kernel and implementation, the function symbol, the address ranges of setup, outer loop body and inner loop body (identified by our local labels `.Lsdrk_<kernel>_inner` etc.; for ESP-DSP by symbol + loop structure, written down once by hand and checked by a test that the instruction at each boundary matches the recorded mnemonic), and the trip-count formulas in `N` and `taps`. Hardware loops (`esp.lp.setup`) count the body per iteration and the setup once. Result: instructions per output sample at N = 4096 (taps 64 for FIR), written to `benches/results/esp32p4.json`, limits from `benches/limits.json` (release ≤ 1.15×, dev ≤ 1.50×).
- Report that static counting ignores stalls and memory waits (P4 is in-order, so it is a reasonable proxy), and that both sides are measured identically.
- **What the comparison means (spike, section 6).** ESP-DSP and our kernels do not solve the same task. `dsps_dotprod_s16_arp4(src1, src2, int16_t *dest, len, shift)` stores a 16-bit result of `(sum + (0x7fff >> shift)) >> (15 − shift)` taken with `esp.srs.s.xacc` (this rounds up, not half up as R7 does), takes the vector path only for `len % 8 == 0` and otherwise jumps to its C version `dsps_dotprod_s16_ansi`, sets bit 1 of the PIE configuration register for unaligned access, and works on real `s16` samples where ours are `CQ15`. Therefore: (a) ESP-DSP is never a correctness reference, only a density reference for the inner loop; (b) the limits apply **per multiply-accumulate of the inner loop**, and the extra work of our contract (XACC read-out, `i64` sum, R7 rounding and saturation, a chunk every 63 vectors, the scalar tail) is reported separately as contract overhead per call and per output sample; (c) for `fir` and `fir_decimate` with 64 taps the overhead is a visible share of every output, so after the first measurement the numbers and the decision on those limits go into the decisions log instead of assuming 1.15× for the whole kernel.
- **Inner-loop baseline.** The spike's naive `dot` loop (`vld`, `vld`, `vmulas.s16.xacc`, `addi`, `bnez`) costs 5 instructions per 8 MACs; ESP-DSP's costs 4 (the second load fused into `esp.vmulas.s16.xacc.ld.ip`); a hardware loop (`esp.lp.setup`) reaches 2. The kernels use the fused forms and hardware loops from the start.
- The ESP-DSP build must target silicon v3: ESP-IDF v5.4 and v5.5 build for pre-v3 silicon by default, and esp-emu refuses those images. Pin an ESP-IDF version and the sdkconfig option for it when implementing and record both.

### 6. Spike results (2026-10-24)

A throw-away program (esp-hal 1.2.2, `espflash save-image --merge`, esp-emu v0.50.0, `riscv32-esp-elf` 16.1.0_20260609, assembler 2.46) ran a `dot` kernel in PIE assembler (five instructions per vector, encodings as `.insn` words from the assembler) against scalar Rust on random data and extremes. Result: 15 of 18 cases bit-exact; the three failures are the XACC saturation above. What it established:

- esp-emu runs PIE as specified here, on a v3.1 chip, with the real ROM and the ESP-IDF second stage bootloader.
- The encodings are revision dependent (section 1), the unit must be enabled (section 2), XACC saturates at ±2³⁹ (section 3), and ESP-DSP computes a different contract (section 5).
- The `esp.lp.*` instructions need the extension `xesploop2p1`; the PIE instructions need `xespv2p2`. The extension name `xesppie` is not accepted.
- Not covered: rounding behaviour of `srcmb` and the shifts, hardware loops, `fir` and `fft`, behaviour on real silicon (the emulator's authors report a bit-exact comparison with boards; not verified here).

## Test vectors

Golden files run on the host only, so add **embedded golden checks**. New Python cases in `cases/fft.py`: `fft/cq15_fixed_noise_256` (seed 400, noise σ 0.3 quantized, expected `np.fft.fft(x)/256`, SNR ≥ 40) and `fft/cq15_bfp_noise_256` (seed 401, expected unnormalized `np.fft.fft(x)`, SNR ≥ 55 after applying `2^exponent`). `build.rs` of `targets/esp32p4` embeds exactly these cases via `include_bytes!` into a `static` table: `fir/gain_overflow_cq15_2048`, `dot/cq15_64`, `mag_sq/cq15_4096`, `convert/cu8_256`, `convert/cs8_256`, `convert/cs16_256`, `fft/cq15_fixed_noise_256`, `fft/cq15_bfp_noise_256` (each ≤ 64 KiB), and a suite extension runs them on the target (target crate only; the testkit gets a `no_std` golden view type `GoldenRef { name: &'static str, arrays: &'static [(&'static str, Dtype, &'static [u8])], params: &'static [(&'static str, f64)], tolerance: Tolerance }`; `include_bytes!` data is only 1-byte aligned, so arrays are copied into the arena before casting (or read with `bytemuck::pod_read_unaligned`); `build.rs` converts the JSON params of each case into the `params` table).

## Tests

1. Suite (quick and full) passes in esp-emulator with `esp-pie` enabled; embedded golden checks pass.
2. `xtask pie-encode --check` passes in the nightly ESP-IDF job.
3. `xtask count --platform esp32p4` meets the limits for every kernel marked `release`.
4. Host builds (`cargo build --workspace`) are unaffected (crate empty on non-riscv32).
5. The accumulator-edge stress cases of section 4 pass; without the chunking they fail (the spike's saturation reproduces), so the test guards the chunk size.

## Acceptance

R1.2 commands; `cargo xtask emu --target esp32p4 --features esp-pie --full`; `cargo xtask count --platform esp32p4`. **First embedded backend complete; phase 7 continues from the backlog.**

## Deliverables checklist

- [ ] `pie-encode` tool, committed encodings, nightly check
- [ ] `sdrk-backend-esp-pie` with six kernels, facade feature
- [ ] `enable()` for the PIE unit, `xespv2p2`-only encodings with the revision recorded in the header, chunking at 63 vectors
- [ ] `Storage::alloc_aligned`
- [ ] Embedded golden checks, stress cases
- [ ] ESP-DSP comparison project (built for silicon v3), static counting tool, `loops.json`, results per MAC with the contract overhead reported separately
- [ ] `docs/backends.md` (P4 section with sources), CHANGELOG, decisions log
