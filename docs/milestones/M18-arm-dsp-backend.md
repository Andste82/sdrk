# M18 — Arm DSP-extension backend (Cortex-M4/M7/M33) with CMSIS-DSP comparison

> Plan: [PLAN.md › Phase 7](../PLAN.md#phase-7--embedded-backends-large-incremental) · Depends on: [M15](M15-emulator-targets.md) (and the backend pattern of [M13](M13-simd-backend.md), [M16](M16-esp32-p4-pie-backend.md) and [M17](M17-rvv-backend.md)) · Next: [M19](M19-receive-basics.md)

## Goal

Create `sdrk-backend-arm-dsp` for Arm Cortex-M cores with the DSP extension (Cortex-M4, M7 and M33, e.g. STM32F4/F7/H7, nRF52, SAMD51, Teensy, RP2350) with the most expensive `CQ15` kernels — `fir`, `fir_decimate`, `dot`, `mag_sq`, `convert` (CU8/CS8/CS16 → CQ15) and the `CQ15` FFT (fixed and block floating point) — verified by the suite in QEMU on all three cores, and compared with **CMSIS-DSP** by instruction counts (QEMU instruction-count plug-in).

## Why

The Cortex-M4/M7/M33 family is the most widespread class of microcontrollers with DSP capability. The backend uses the 32-bit "SIMD within a register" instructions (`SMLALD`, `SMLSLD`, `SMUAD`, `PKHBT`, `QADD16`, …), which make bit-exact Q15 arithmetic with 64-bit accumulators cheap — so unlike on other platforms no accumulator-headroom analysis is needed. CMSIS-DSP is the established reference, the emulator is the same QEMU already used in M15, and the instruction-count method is the plan's standard.

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

**In:** backend crate `sdrk-backend-arm-dsp` (compile-time selected, no run-time detection), facade feature `arm-dsp`, targets `targets/qemu-cortex-m7` and `targets/qemu-cortex-m33` (the M4 target exists since M15), `xtask count --platform cortex-m4` extended with the CMSIS-DSP comparison, a `benches/cortex-m/` C project, CI jobs, `docs/backends.md` section, a documentation paragraph on choosing sample types for chips without FPU.

**Out:** Cortex-M0/M0+/M23 (no DSP extension: the fallback runs there), Cortex-M55/M85 Helium (backlog), `C32` kernels (the single-precision FPU of M4F/M7/M33 is used by the fallback as is), `fir_interpolate`, `quad_demod`, `fm_mod`, `rotate`, `convert_out` (backlog), real boards (the RP2350 is covered by its Cortex-M33 core; QEMU has no model of the chip itself, to the best of our knowledge — verify, and test on the generic M33 machine), the RP2350's RISC-V cores (no vector unit).

## Specification

### 0. Smoke test first

Before writing kernels, prove the toolchain and the emulator with a 20-line target (in `targets/qemu-cortex-m4`, feature `smoke`, removed again or kept behind the feature): a `global_asm!` function that executes `smlald`, `smlsld`, `pkhbt`, `sxtb16`, `qadd16`, `ssat` on known operands and returns the results; the target prints them through the reporter and the host compares them with the values computed in Python. Run it on the three machines of §5. If an instruction is not emulated or the assembler rejects it (for the M33: the DSP extension must be enabled, e.g. with `.arch_extension dsp`; verify), stop and resolve it explicitly (decisions log) before continuing.

### 1. Crate, toolchain and unsafe policy

- `crates/backends/sdrk-backend-arm-dsp`, `#![no_std]`, depends on `sdrk-core`. Compiled only when the DSP instructions exist: `#[cfg(all(target_arch = "arm", <dsp condition>))]`; on every other target the crate is empty (hosts, M0, RISC-V, … keep building). `<dsp condition>`: use `target_feature = "dsp"` if the pinned stable rustc exposes it for `thumbv7em-none-eabihf` and `thumbv8m.main-none-eabihf` (verify); otherwise the `RUSTFLAGS` cfg `sdrk_arm_dsp` (a cfg, **not** a cargo feature, for the reason given in M13 and M17), set by the `.cargo/config.toml` of the M4/M7/M33 targets and by `xtask bare-metal` for this crate, so that the assembly is also assembled in the bare-metal CI step. Declare the cfg in the workspace lints (`unexpected_cfgs`).
- `core::arch::arm` DSP intrinsics are not stable (verify and log; if they are, they may replace assembly where they generate the same instructions). Every kernel loop is therefore a **whole function in `global_asm!`** with the AAPCS (`.syntax unified`, `.thumb`, `push {r4-r11, lr}` / `pop`, stack kept 8-byte aligned; the compiler reserves `r7`/`r9` for itself in `asm!` operands, which whole functions avoid), called from a safe wrapper that validates lengths and arguments. Inner-loop labels: `.Lsdrk_armdsp_<kernel>_inner`.
- `unsafe` policy as in M13 (`#[allow(unsafe_code)]` only in the module declaring and calling the asm functions, `// SAFETY:` on every block, `#![deny(clippy::undocumented_unsafe_blocks)]`). The wrapper guarantees valid pointers/lengths and non-aliasing input/output (`&`/`&mut`).
- **Alignment:** `CQ15` slices are only 2-byte aligned. Use single-word `LDR`/`STR` (allowed unaligned on M4/M7/M33 unless the core traps unaligned access, which this crate requires to be off — set `UNALIGN_TRP = 0`, check in the M7/M33 start-up code); `LDM`/`LDRD`/`STM`/`STRD` fault on unaligned addresses regardless, so they are used only after the wrapper has proven 4-byte alignment (otherwise a single-word loop). The stress cases of §4 include slices offset by one sample.
- The `Q` and `GE` status flags are set by the DSP instructions; no code reads them, and the AAPCS does not require them to be preserved.
- No stack use beyond the register saves (R3.4).

### 2. Kernel contracts (all bit-exact except the FFT)

| Kernel | Contract | Implementation notes |
| --- | --- | --- |
| `convert` CU8/CS8/CS16 → CQ15 | bit-exact | CS8: byte unpack and shift with `SXTB16`/mask-and-shift, repack with `PKHBT`/`PKHTB`; CU8: `UXTB16`, subtract 128 per halfword (`USUB16`), shift; CS16 is a plain copy of the byte-identical layout (check M05). Two complex samples per word-pair iteration |
| `mag_sq` CQ15 | bit-exact | `SMLALD` of a sample with itself (accumulator zero) gives `re² + im²` exactly in 64 bits — `SMUAD` alone would wrap for `−32768 + −32768 j` (2³¹), so it is not used; then the rounding and saturation of M06 |
| `dot` CQ15 | bit-exact | complex product with `SMLSLD` (real part: `re·re − im·im`) and `SMLALDX` (imaginary part: `re·im + im·re`), both into 64-bit accumulators; one rounding at the end (R7) |
| `fir`, `fir_decimate` CQ15 | bit-exact | pack two taps per word and the matching real (`PKHBT`) and imaginary (`PKHTB`) halves of two consecutive samples, then two `SMLALD` per tap pair; **initialise the 64-bit accumulator with `1 << 14`**: the rounding of R7 is then free; the final `>> 15` and saturation with `LSRS`/`ORR` and `SSAT #16`. The 64-bit → 32-bit extraction is exact while `taps · 2³⁰ < 2⁴⁶`; the wrapper calls the fallback beyond that (document the limit) |
| `fft` CQ15 fixed and BFP | M07 contract table (as M16) | radix-2 or radix-4 butterflies; complex multiply by the twiddle with `SMUSD`/`SMUADX` (or `SMLSLD`/`SMLALDX` where a 64-bit sum is needed for the contract), `QADD16`/`QSUB16`/`SHADD16` for the butterfly adds with the per-stage scaling of M07; BFP block maximum with `SSAT16`/`QADD16` tricks or scalar `SXTH`/`CMP` in a separate pass; exponent semantics as in M07 |

`C32` kernels stay on `Fallback` (hardware single-precision FPU). A kernel entry delegates to the fallback for unsupported parameters (never panics, R5).

### 3. Facade

Feature `arm-dsp` → `delegate!(… => ArmDsp)` for the kernels above under the cfg of §1; `backend_for` returns `"arm-dsp"`. No run-time detection: the cfg is the decision. `sdrk_fallback_only` (M13) forces the fallback.

### 4. Verification

- All `backend_vs_fallback_*` testkit cases (M13) cover the kernels automatically through the facade. Extra stress cases (quick tier, PCG32 seeds **520–529**): lengths 0…40 and 4093…4100, slices offset by one sample, taps 1…300, −32768 everywhere, alternating ±32767/−32768 patterns that maximise the accumulator, decimation factors 1…16.
- Embedded golden checks: the cases of M16 (`fir/gain_overflow_cq15_2048`, `dot/cq15_64`, `mag_sq/cq15_4096`, `convert/cu8_256`, `convert/cs8_256`, `convert/cs16_256`, `fft/cq15_fixed_noise_256`, `fft/cq15_bfp_noise_256`) are embedded in the Cortex-M targets by the same mechanism (reuse the `GoldenRef` of M16; if M16 is not yet implemented in your tree, this milestone adds it — log it).
- The quick suite passes on `qemu-cortex-m4`, `qemu-cortex-m7`, `qemu-cortex-m33` with `--features arm-dsp`; with `sdrk_fallback_only` everything still passes.

### 5. Targets (extension of M15 §2)

| Target dir | Tier | Rust target | Machine / CPU |
| --- | --- | --- | --- |
| `qemu-cortex-m4` (existing) | 1 | `thumbv7em-none-eabihf` | `mps2-an386`, `cortex-m4` |
| `qemu-cortex-m7` (new) | 2 | `thumbv7em-none-eabihf` | `mps2-an500`, `cortex-m7` |
| `qemu-cortex-m33` (new) | 2 | `thumbv8m.main-none-eabihf` with the DSP feature (`-C target-feature=+dsp` in the target's `.cargo/config.toml`, verify) | `mps2-an505`, `cortex-m33` |

Machine and CPU names are the expected ones for QEMU 8.2; check them with `qemu-system-arm -machine help` and `-cpu help`, and record the QEMU version in `docs/backends.md`. If QEMU's `cortex-m33` has no FPU, use `thumbv8m.main-none-eabi` (soft float) for that target and log it. `memory.x` and start-up as in M15 (`cortex-m-rt`); the start-up code of M7/M33 also clears `UNALIGN_TRP` and enables the FPU. `xtask emu` accepts the three target names and passes `--features` through (M15 §3).

### 6. Comparison with CMSIS-DSP

- `benches/cortex-m/`: a small C project (no heap, semihosting exit, `arm-none-eabi-gcc` with newlib-nano and `--specs=rdimon.specs`, `-O3 -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16`) that compiles the needed sources of a **pinned CMSIS-DSP release** (Apache-2.0; downloaded by `xtask count --install` into `target/tools/` with a checksum, not vendored) and calls: `arm_fir_q15` (taps 64, once per real/imaginary channel on separate arrays — CMSIS users keep split data, so the de-interleave is not charged to CMSIS), `arm_fir_decimate_q15` (127 taps, M = 10; the accurate 64-bit variant, not `_fast`), `arm_cmplx_dot_prod_q15`, `arm_cmplx_mag_squared_q15`, `arm_q7_to_q15` on 2N values (for CS8), `arm_cfft_q15` (N = 64, 1024, 4096 where supported). CU8 and BFP FFT have no CMSIS counterpart: compare with the own fallback.
- **Accuracy must be equal:** record each CMSIS function's conventions (CMSIS `arm_fir_q15` truncates instead of rounding; some outputs use other Q formats such as 3.13 for the magnitude) in `docs/backends.md`, and give both sides the same error measurement against the f64 reference. Where the conventions force a different result by definition, count our kernel and CMSIS as they are, mark the entry `report` in `benches/limits.json` (no limit) and explain it; entries without such a difference are `release` or `dev`.
- **Method:** as M13 §4 and M15 §5: differential counting (K = 0 and K = 16, N = 4096), `qemu-system-arm -machine mps2-an386 -cpu cortex-m4 -plugin <libinsn.so> -d plugin`, both binaries exit by semihosting; `K` is a build-time constant (`SDRK_BENCH_K` via `option_env!` in Rust, `-DK=` in C). `cargo xtask count --platform cortex-m4` writes `benches/results/cortex-m4.json` (fallback, arm-dsp, CMSIS-DSP, ratios, SNR of each against the f64 reference). Limits: `release` ≤ **1.15×** CMSIS-DSP, `dev` ≤ 1.50×; between 1.10× and 1.15× a warning; accuracy at least equal (our SNR lower by more than 1 dB fails).
- Report that QEMU has no cycle model: the counts measure instruction economy. The Cortex-M4 is in-order, which makes the count a reasonable proxy, but cache, flash wait states and the dual issue of the M7 are not modelled.

### 7. CI additions

| Job | Trigger | What |
| --- | --- | --- |
| `emu-tier1` (M15) | every push/PR | additionally `xtask emu --target qemu-cortex-m4 --features arm-dsp` (quick) |
| `emu-tier2` (M15) | nightly | additionally M7 and M33 targets with `arm-dsp` (quick) and the M4 target with `--full` |
| `count-arm-dsp` | PRs touching `crates/backends/sdrk-backend-arm-dsp` or `sdrk-core`; nightly always | `cargo xtask count --platform cortex-m4` with the CMSIS comparison |

Install in CI and devcontainer: `gcc-arm-none-eabi`, `libnewlib-arm-none-eabi` (the devcontainer already has the compiler), the QEMU packages of M15. `xtask bare-metal` now also builds this crate for `thumbv7em-none-eabihf` (it is a heap-free crate; with the cfg of §1 the assembly is assembled).

### 8. Documentation

`docs/backends.md` gets the Arm DSP section: instruction table with the exact meaning of each used instruction (cite the Armv7-M and Armv8-M Architecture Reference Manuals by document number and version), the alignment and `UNALIGN_TRP` requirement, the CMSIS-DSP version and conventions, the QEMU version and machine names. Add a paragraph **"Choosing sample types by chip"** (also in `docs/embedded.md`): Cortex-M0/M0+/M23, ESP32-C3/C6/H2 and other cores without FPU compute `C32`/`f32` in software and slowly — use `CQ15`/`Q15` there (M15 shows the soft-float cost in `qemu-riscv32`); the ESP32 classic and S2 have no SIMD, only the scalar fallback.

## Test vectors

No new golden files (R8); the embedded golden checks reuse the cases of M16 (seeds 400–401 and the existing ones). Stress inputs: PCG32 seeds **520–529**.

## Tests

1. The quick suite passes on the three targets with `--features arm-dsp`; the embedded golden checks pass.
2. The smoke test of §0 passes on the three machines.
3. `xtask count --platform cortex-m4` meets the limits for every kernel marked `release` and produces the `report` entries with explanations.
4. `xtask bare-metal` builds the crate for `thumbv7em-none-eabihf` in CI; host builds (`cargo build --workspace`) on x86_64, arm64 and Windows are unaffected (crate empty).
5. A deliberately wrong accumulator (temporary change, e.g. `SMLAD` instead of `SMLALD`) makes at least one stress case fail (check once by hand and note it in the decisions log; do not commit the change).

## Acceptance

R1.2 commands; `cargo xtask emu --target qemu-cortex-m4 --features arm-dsp --full`; `cargo xtask emu --tier 2 --features arm-dsp`; `cargo xtask count --platform cortex-m4`. **Arm DSP backend complete.**

## Deliverables checklist

- [ ] Smoke test passed on M4, M7, M33 and logged
- [ ] `sdrk-backend-arm-dsp` with six kernels (asm functions, safe wrappers), facade feature `arm-dsp`
- [ ] `targets/qemu-cortex-m7`, `targets/qemu-cortex-m33`; embedded golden checks on all three
- [ ] `benches/cortex-m/` CMSIS-DSP project, `xtask count --install`, `limits.json`, results
- [ ] CI jobs; `xtask bare-metal` covers the crate
- [ ] `docs/backends.md` (Arm DSP section, CMSIS conventions), `docs/embedded.md` ("Choosing sample types by chip"), CHANGELOG, decisions log
